"""Flow Prefect 3 de réentraînement — churn SaaS B2B.

Architecture du flow
---------------------
Même orchestrateur que le scoring nocturne (``flows/scoring_batch.py``) : chaque étape est
une ``@task`` Prefect, l'enchaînement un ``@flow``. Sans serveur, Prefect 3 s'exécute en
local runner (``make flow``) ; en production, le déploiement ``sur-decision-comite`` de
``prefect.yaml`` le rend déclenchable depuis l'interface ou par
``prefect deployment run reentrainement-churn/sur-decision-comite``.

Prefect apporte les relances avec attente exponentielle (paramètre ``retries`` des tasks),
l'état de chaque étape (réussie, relancée, échouée) et l'historique des exécutions, consultables
dans l'interface Prefect : la trace dont le comité a besoin pour auditer un réentraînement.

Déclencheurs prévus (règle de §2.6 : aucun signal ne lance le flow seul)
-----------------------------------------------------------------------
Le flow est toujours lancé à la main (déploiement sans planification, ou ``make flow``) sur
décision d'un comité ; la promotion, elle, ne dépend que de la gate (``stage_gate``), que
personne ne peut contourner.

* **Calendaire trimestriel** — lancé à l'issue du comité trimestriel de revue (§2.6).
* **Dérive des entrées** — PSI ≥ 0.20 sur au moins une variable surveillée, détecté
  par le rapport de dérive hebdomadaire (déploiement ``derive-hebdomadaire``) → comité ad hoc qui décide
  sous 72 h de lancer ou non le flow.
* **Dégradation de performance** — PR-AUC mesurée sur une cohorte de renouvellements
  (étiquettes observables 1 à 12 mois après la prédiction) sous
  ``config.CIBLES_PERFORMANCE["pr_auc_min"]`` → même comité ad hoc ; sous
  ``pr_auc_critique``, le comité compare aussi le champion précédent sur la même cohorte.

Stages
------
1. ``ingest``   — vérification intégrité des sources brutes
2. ``features`` — construction du gold dataset
3. ``train``    — entraînement du challenger (famille et hyperparamètres du champion)
4. ``evaluate`` — métriques sur split test 20 %
5. ``gate``     — bloque si PR-AUC < seuil ou régression vs champion
6. ``promote``  — rotation des artefacts + enregistrement MLflow Registry

Idempotence
-----------
Chaque stage écrit un marqueur ``reports/tables/flow_cache/<date>_<stage>.done``.
Si le marqueur existe et ``forcer=False``, l'étape est sautée.
Ce mécanisme permet de relancer le flow après un échec partiel sans retraiter
les étapes déjà terminées avec succès. Il remplace le cache de tasks de Prefect
(``NO_CACHE``) : un pipeline scikit-learn en entrée ne se hache pas de façon fiable.
"""

from __future__ import annotations

import datetime
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from loguru import logger
from prefect import flow, task
from prefect.cache_policies import NO_CACHE
from prefect.tasks import exponential_backoff

# Racine du projet : flows/retraining.py → parents[1] = racine du dépôt
_RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_RACINE / "src"))

from churn_saas import config  # noqa: E402  (import après sys.path)

# ---------------------------------------------------------------------------
# Utilitaires internes
# ---------------------------------------------------------------------------


def _marqueur(etape: str) -> Path:
    """Chemin du fichier de marqueur d'idempotence pour une étape du flow."""
    cache_dir = config.TABLES / "flow_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.date.today().isoformat()
    return cache_dir / f"{date_str}_{etape}.done"


# ---------------------------------------------------------------------------
# Stage 1 — Ingest
# ---------------------------------------------------------------------------


@task(
    name="ingest",
    retries=2,
    retry_delay_seconds=exponential_backoff(backoff_factor=5),
    cache_policy=NO_CACHE,
)
def stage_ingest(forcer: bool = False) -> Path:
    """Vérifie l'existence et la lisibilité des sources brutes.

    Idempotent : si le marqueur du jour existe et ``forcer=False``, passe l'étape.

    Returns
    -------
    Chemin vers ``data/raw/churn_saas_complet.csv``.

    Raises
    ------
    FileNotFoundError si une source brute est absente.
    """
    marqueur = _marqueur("ingest")
    if marqueur.exists() and not forcer:
        logger.info("Stage ingest déjà complété aujourd'hui ({}). Saut.", marqueur.name)
        return config.DONNEES_BRUTES / "churn_saas_complet.csv"

    fichiers_requis = [
        config.DONNEES_BRUTES / "churn_saas_complet.csv",
        config.DONNEES_BRUTES / "catalogue_plans.csv",
    ]
    for f in fichiers_requis:
        if not f.exists():
            raise FileNotFoundError(f"Source brute absente : {f}")
        logger.info("Source brute OK : {} ({} Ko)", f.name, f.stat().st_size // 1024)

    marqueur.write_text(datetime.datetime.now().isoformat(), encoding="utf-8")
    logger.info("Stage ingest terminé.")
    return fichiers_requis[0]


# ---------------------------------------------------------------------------
# Stage 2 — Features
# ---------------------------------------------------------------------------


@task(
    name="features",
    retries=1,
    retry_delay_seconds=exponential_backoff(backoff_factor=5),
    cache_policy=NO_CACHE,
)
def stage_features(forcer: bool = False) -> Path:
    """Construit le gold dataset (idempotent).

    Délègue à ``churn_saas.cli.construire_gold_dataset`` qui enchaîne les
    9 étapes de préparation (nettoyage, features, catalogue, enrichissement).

    Returns
    -------
    Chemin vers ``data/gold/gold_dataset.parquet``.
    """
    from churn_saas.cli import construire_gold_dataset  # import tardif

    marqueur = _marqueur("features")
    chemin_gold = config.DONNEES_GOLD / "gold_dataset.parquet"

    if marqueur.exists() and not forcer and chemin_gold.exists():
        logger.info("Stage features déjà complété aujourd'hui. Saut.")
        return chemin_gold

    chemin = construire_gold_dataset(forcer=forcer)
    marqueur.write_text(datetime.datetime.now().isoformat(), encoding="utf-8")
    logger.info("Stage features terminé → {}", chemin)
    return chemin


# ---------------------------------------------------------------------------
# Stage 3 — Train
# ---------------------------------------------------------------------------


@task(
    name="train",
    retries=1,
    retry_delay_seconds=exponential_backoff(backoff_factor=5),
    cache_policy=NO_CACHE,
)
def stage_train(forcer: bool = False) -> tuple[Any, dict[str, float]]:
    """Entraîne le challenger et retourne (pipeline, métriques).

    Utilise la famille et les hyperparamètres du champion actuel (``best_model_meta.json``,
    à défaut l'étude Optuna de §9.7). Ce mécanisme correspond au transfert de connaissances
    décrit en §9.11 : réutilisation du savoir acquis sur le modèle comme point de départ,
    sans transfer learning au sens strict. Toute famille de ``construire_modele_optimise``
    convient : régression logistique recalibrée (champion actuel), forêt ou LightGBM.

    Le seuil de vigilance (premier niveau de la règle de décision : recall cible
    ``config.RECALL_CIBLE_VIGILANCE``) est recalculé pour le challenger sur ses prédictions
    out-of-fold du train, jamais sur le test : un modèle réentraîné n'a pas la même échelle
    de probabilités, le seuil du notebook ne lui est pas transposable.

    Returns
    -------
    Tuple (pipeline sklearn, dict métriques test).
    """
    import joblib
    import pandas as pd
    from sklearn.base import clone
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

    from churn_saas.features.build import ajouter_features_metier
    from churn_saas.models.economics import seuil_pour_recall
    from churn_saas.models.train import (
        construire_modele_optimise,
        famille_et_hyperparametres_champion,
    )

    chemin_challenger = config.ARTIFACTS / "models" / "challenger.pkl"
    chemin_meta_challenger = config.ARTIFACTS / "models" / "challenger_meta.json"
    marqueur = _marqueur("train")

    if marqueur.exists() and not forcer and chemin_challenger.exists():
        logger.info("Stage train déjà complété aujourd'hui. Chargement du challenger.")
        pipeline_cache = joblib.load(chemin_challenger)
        with chemin_meta_challenger.open(encoding="utf-8") as fic:
            metriques_cache: dict[str, float] = json.load(fic).get("metriques", {})
        return pipeline_cache, metriques_cache

    # Chargement du gold
    gold = pd.read_parquet(config.DONNEES_GOLD / "gold_dataset.parquet")
    y = gold["churn"].astype(int)
    # Même périmètre que le modèle déployé (§10) : sans les leurres, dont l'inutilité est
    # prouvée en §12.9 et que l'API ne reçoit pas — sinon le challenger serait inutilisable
    colonnes_a_supprimer = (
        list(config.COLONNES_INTERDITES)
        + list(config.COLONNES_LEURRES_SUSPECTES)
        + ["date_souscription", "jour_souscription"]
    )
    X_brut = gold.drop(columns=colonnes_a_supprimer, errors="ignore")
    X = ajouter_features_metier(X_brut)
    X = X.drop(columns=["churn"], errors="ignore")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.PART_TEST, stratify=y, random_state=config.RANDOM_SEED
    )
    logger.info(
        "Split — train : {} lignes, test : {} lignes (stratifié, seed={}).",
        len(X_train),
        len(X_test),
        config.RANDOM_SEED,
    )

    # Transfert de connaissances (§9.11) : même famille, mêmes hyperparamètres que le champion
    modele_nom, hyperparametres = famille_et_hyperparametres_champion()
    pipeline = construire_modele_optimise(modele_nom, hyperparametres, df_ref=X_train)

    # Seuil de vigilance du challenger : prédictions out-of-fold sur le train uniquement
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
    probas_oof = cross_val_predict(clone(pipeline), X_train, y_train, cv=cv, method="predict_proba")
    vigilance = seuil_pour_recall(y_train, probas_oof[:, 1])
    logger.info(
        "Seuil de vigilance du challenger (OOF, recall cible {:.2f}) : {:.4f}",
        config.RECALL_CIBLE_VIGILANCE,
        vigilance["seuil"],
    )

    logger.info("Entraînement du challenger {} sur {} observations …", modele_nom, len(X_train))
    pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    metriques: dict[str, float] = {
        "pr_auc_test": round(float(average_precision_score(y_test, y_proba)), 4),
        "roc_auc_test": round(float(roc_auc_score(y_test, y_proba)), 4),
        "n_train": float(len(X_train)),
        "n_test": float(len(X_test)),
    }
    logger.info(
        "Challenger — PR-AUC = {:.4f}, ROC-AUC = {:.4f}",
        metriques["pr_auc_test"],
        metriques["roc_auc_test"],
    )

    chemin_challenger.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, chemin_challenger)

    meta_out: dict[str, Any] = {
        "nom": f"Challenger {modele_nom}",
        "modele_nom": modele_nom,
        "date_entrainement": datetime.datetime.now().isoformat(timespec="seconds"),
        "metriques": metriques,
        "hyperparametres": hyperparametres,
        "seuil_vigilance": vigilance["seuil"],
        "recall_vigilance_oof": vigilance["recall"],
    }
    with chemin_meta_challenger.open("w", encoding="utf-8") as fic:
        json.dump(meta_out, fic, ensure_ascii=False, indent=2)

    marqueur.write_text(datetime.datetime.now().isoformat(), encoding="utf-8")
    return pipeline, metriques


# ---------------------------------------------------------------------------
# Stage 4 — Evaluate
# ---------------------------------------------------------------------------


@task(name="evaluate", cache_policy=NO_CACHE)
def stage_evaluate(pipeline: Any, metriques: dict[str, float]) -> dict[str, float]:
    """Logue les métriques du challenger — renvoie le dictionnaire inchangé.

    Étape distincte de ``train`` pour permettre une injection future de métriques
    complémentaires (équité par cohorte, calibration, latence batch).
    """
    logger.info(
        "Évaluation — PR-AUC = {:.4f} | ROC-AUC = {:.4f} | n_test = {}",
        metriques.get("pr_auc_test", 0.0),
        metriques.get("roc_auc_test", 0.0),
        int(metriques.get("n_test", 0)),
    )
    return metriques


# ---------------------------------------------------------------------------
# Stage 5 — Gate
# ---------------------------------------------------------------------------


@task(name="gate", cache_policy=NO_CACHE)
def stage_gate(metriques: dict[str, float]) -> bool:
    """Gate de promotion binaire — règle unique ``decision_promotion`` (§10.4, §13.8).

    1. ``pr_auc_test >= config.CIBLES_PERFORMANCE["pr_auc_min"]`` ;
    2. ``pr_auc_test >= pr_auc_champion − tolerance_regression_promotion``.

    Returns
    -------
    True si le challenger peut être promu, False sinon.
    """
    from churn_saas.models.train import decision_promotion

    pr_auc = metriques.get("pr_auc_test", 0.0)
    pr_auc_champion: float | None = None
    chemin_meta_champion = config.ARTIFACTS / "models" / "best_model_meta.json"
    if chemin_meta_champion.exists():
        meta_champion = json.loads(chemin_meta_champion.read_text(encoding="utf-8"))
        if meta_champion.get("pr_auc_test") is not None:
            pr_auc_champion = float(meta_champion["pr_auc_test"])

    promouvoir, motif = decision_promotion(pr_auc, pr_auc_champion)
    if promouvoir:
        logger.info("Gate PASSÉE — {}.", motif)
    else:
        logger.error("Gate ÉCHOUÉE — {}. Challenger non promu.", motif)
    return promouvoir


# ---------------------------------------------------------------------------
# Stage 6 — Promote
# ---------------------------------------------------------------------------


@task(name="promote", cache_policy=NO_CACHE)
def stage_promote(pipeline: Any, metriques: dict[str, float]) -> None:
    """Promeut le challenger : rotation des artefacts + enregistrement MLflow.

    Procédure de rollback : les archives horodatées dans ``artifacts/models/`` permettent
    de restaurer n'importe quel champion précédent en copiant le fichier archive
    vers ``best_model.pkl`` (opération manuelle documentée dans ``docs/RUNBOOK.md``).
    """
    from churn_saas import economie
    from churn_saas.models.train import enregistrer_au_registre, journaliser_mlflow

    chemin_challenger = config.ARTIFACTS / "models" / "challenger.pkl"
    chemin_champion = config.ARTIFACTS / "models" / "best_model.pkl"
    chemin_meta = config.ARTIFACTS / "models" / "best_model_meta.json"

    # Archivage du champion précédent (base du rollback)
    if chemin_champion.exists():
        horodatage = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        archive = config.ARTIFACTS / "models" / f"archive_{horodatage}_best_model.pkl"
        shutil.copy2(chemin_champion, archive)
        logger.info("Champion précédent archivé → {}", archive.name)

    # Rotation artefact
    shutil.copy2(chemin_challenger, chemin_champion)

    # Famille et hyperparamètres transmis au champion : base du prochain challenger (§9.11)
    chemin_meta_challenger = config.ARTIFACTS / "models" / "challenger_meta.json"
    meta_challenger: dict[str, Any] = {}
    if chemin_meta_challenger.exists():
        meta_challenger = json.loads(chemin_meta_challenger.read_text(encoding="utf-8"))
    modele_nom = meta_challenger.get("modele_nom", "inconnu")

    meta_out: dict[str, Any] = {
        "nom": f"{modele_nom} (champion promu par flow)",
        "modele_nom": modele_nom,
        "hyperparametres": meta_challenger.get("hyperparametres", {}),
        "pr_auc_test": metriques.get("pr_auc_test"),
        "roc_auc_test": metriques.get("roc_auc_test"),
        # Seuil servi par l'API et le batch (ModelStore.seuil) : celui calculé pour ce modèle
        "seuil_economique": meta_challenger.get(
            "seuil_vigilance", economie.seuil_surveillance_par_defaut()
        ),
        "date_entrainement": datetime.datetime.now().isoformat(timespec="seconds"),
        "n_train": metriques.get("n_train"),
        "n_test": metriques.get("n_test"),
        "source": "flows/retraining.py",
    }
    with chemin_meta.open("w", encoding="utf-8") as fic:
        json.dump(meta_out, fic, ensure_ascii=False, indent=2, default=str)

    logger.info("Artefact champion mis à jour → {}", chemin_champion.name)

    # Journalisation MLflow — même backend et même expérience que le notebook (§9.5, §10.4)
    try:
        run_id = journaliser_mlflow(
            f"flow_retrain_{datetime.date.today().isoformat()}",
            pipeline,
            {k: v for k, v in metriques.items() if isinstance(v, (int, float))},
            {"modele": modele_nom, **meta_out["hyperparametres"]},
            tags={"stage": "champion", "source": "retrain_flow"},
        )
        # L'alias @production suit le modèle effectivement servi par l'API
        enregistrer_au_registre(run_id, alias="production")
    except Exception as exc:
        logger.warning("MLflow non disponible — journalisation ignorée : {}", exc)


# ---------------------------------------------------------------------------
# Orchestration principale
# ---------------------------------------------------------------------------


@flow(
    name="reentrainement-churn",
    description=(
        "Réentraîne le challenger avec la famille et les hyperparamètres du champion, "
        "puis le promeut si la gate passe : ingest → features → train → evaluate → gate → "
        "promote. Lancé à la main, sur décision du comité de revue."
    ),
)
def flow_retrainement(forcer: bool = False) -> int:
    """Exécute le flow complet : ingest → features → train → evaluate → gate → promote.

    Une étape qui échoue après ses relances lève son exception : Prefect marque alors
    l'exécution en échec, ce qui la rend visible dans l'historique. Une gate non passée
    n'est pas une erreur : le champion reste en place et le flow se termine normalement.

    Parameters
    ----------
    forcer:
        Si True, ignore les marqueurs d'idempotence et recalcule toutes les étapes.

    Returns
    -------
    Code de retour shell : 0 = challenger promu, 1 = gate non passée.
    """
    debut = datetime.datetime.now()
    logger.info("Flow de réentraînement — début : {}", debut.isoformat(timespec="seconds"))

    stage_ingest(forcer)
    stage_features(forcer)
    pipeline, metriques = stage_train(forcer)
    metriques = stage_evaluate(pipeline, metriques)
    if not stage_gate(metriques):
        logger.error("Flow terminé — gate non passée. Modèle NON promu (motif ci-dessus).")
        return 1

    stage_promote(pipeline, metriques)
    duree = (datetime.datetime.now() - debut).total_seconds()
    logger.info("Flow terminé avec succès en {:.1f}s.", duree)
    return 0


def executer(forcer: bool = False) -> int:
    """Lance le flow et traduit son issue en code de retour shell (``make flow``).

    Returns
    -------
    0 = challenger promu, 1 = gate non passée, 2 = étape en échec après ses relances.
    """
    try:
        return flow_retrainement(forcer=forcer)
    except Exception as exc:
        logger.exception("Flow interrompu par une erreur non récupérable : {}", exc)
        return 2


# ---------------------------------------------------------------------------
# Point d'entrée CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Flow de réentraînement churn SaaS.")
    parser.add_argument(
        "--forcer",
        action="store_true",
        help="Ignore les marqueurs d'idempotence et recalcule toutes les étapes.",
    )
    args = parser.parse_args()
    sys.exit(executer(forcer=args.forcer))
