"""Flow de réentraînement — churn SaaS B2B (CISIA C9).

Architecture du flow
---------------------
Prefect 3 est l'orchestrateur **cible** en production (déploiement sur Prefect Cloud
ou auto-hébergé). Ce fichier implémente le même flow en **Python pur** (stdlib seule,
sans dépendance d'orchestration) pour fonctionner sans infrastructure supplémentaire
lors de la certification CISIA.

Migration vers Prefect : ajouter ``@flow``/``@task`` et remplacer les appels directs
par des task-futures pour bénéficier du DAG visuel, des retries gérés par la
plateforme et des déploiements planifiés via ``prefect deploy``.

Déclencheurs prévus (trois mécanismes complémentaires)
-------------------------------------------------------
* **Calendaire trimestriel** — ``make flow`` ou cron ``0 2 1 */3 *``
  (décidé au cadrage §2 ; comité trimestriel de revue).
* **Dérive des entrées** — PSI > 0.20 sur ≥ 2 features principales détecté
  par le script de monitoring quotidien (``monitoring/drift_report.py``).
* **Dégradation de performance** — PR-AUC descend sous
  ``config.CIBLES_PERFORMANCE["pr_auc_min"]`` dès que les étiquettes réelles
  sont disponibles (renouvellements contractuels — délai 1-12 mois).

Stages
------
1. ``ingest``   — vérification intégrité des sources brutes
2. ``features`` — construction du gold dataset
3. ``train``    — entraînement du challenger (HistGradientBoosting)
4. ``evaluate`` — métriques sur split test 20 %
5. ``gate``     — bloque si PR-AUC < seuil ou régression vs champion
6. ``promote``  — rotation des artefacts + enregistrement MLflow Registry

Idempotence
-----------
Chaque stage écrit un marqueur ``reports/tables/flow_cache/<date>_<stage>.done``.
Si le marqueur existe et ``forcer=False``, l'étape est sautée.
Ce mécanisme permet de relancer le flow après un échec partiel sans retraiter
les étapes déjà terminées avec succès.
"""

from __future__ import annotations

import datetime
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from loguru import logger

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


def _avec_retries(
    fn: Any,
    *args: Any,
    n_tentatives: int = 3,
    delai_initial_s: float = 5.0,
    **kwargs: Any,
) -> Any:
    """Exécute ``fn(*args, **kwargs)`` avec retries exponentiels.

    À chaque échec, attend ``delai_initial_s × 2^(tentative-1)`` secondes avant
    de réessayer. Lève l'exception de la dernière tentative si toutes échouent.
    """
    for tentative in range(1, n_tentatives + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if tentative == n_tentatives:
                raise
            attente = delai_initial_s * (2 ** (tentative - 1))
            logger.warning(
                "Tentative {}/{} échouée : {}. Nouvelle tentative dans {:.0f}s.",
                tentative,
                n_tentatives,
                exc,
                attente,
            )
            time.sleep(attente)


# ---------------------------------------------------------------------------
# Stage 1 — Ingest
# ---------------------------------------------------------------------------


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


def stage_train(forcer: bool = False) -> tuple[Any, dict[str, float]]:
    """Entraîne le challenger et retourne (pipeline, métriques).

    Utilise les mêmes hyperparamètres que le champion actuel (``best_model_meta.json``)
    ou les valeurs par défaut si aucun champion n'existe. Ce mécanisme correspond au
    ``warm_start`` conceptuel décrit en §9 : réutilisation des hyperparamètres du
    champion comme point de départ, sans transfer learning au sens strict.

    Returns
    -------
    Tuple (pipeline sklearn, dict métriques test).
    """
    import joblib
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline

    from churn_saas.features.build import ajouter_features_metier, construire_preprocesseur

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
    colonnes_a_supprimer = list(config.COLONNES_INTERDITES) + [
        "date_souscription",
        "jour_souscription",
    ]
    X_brut = gold.drop(columns=colonnes_a_supprimer, errors="ignore")
    X = ajouter_features_metier(X_brut)
    X = X.drop(columns=["churn"], errors="ignore")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=config.RANDOM_SEED
    )
    logger.info(
        "Split — train : {} lignes, test : {} lignes (stratifié, seed={}).",
        len(X_train),
        len(X_test),
        config.RANDOM_SEED,
    )

    # Récupération des hyperparamètres du champion courant (warm_start conceptuel)
    hyperparametres: dict[str, Any] = {
        "max_iter": 300,
        "learning_rate": 0.10,
        "max_depth": 5,
        "l2_regularization": 0.1,
        "min_samples_leaf": 20,
    }
    chemin_meta_champion = config.ARTIFACTS / "models" / "best_model_meta.json"
    if chemin_meta_champion.exists():
        with chemin_meta_champion.open(encoding="utf-8") as fic:
            meta_champ = json.load(fic)
        if "hyperparametres" in meta_champ:
            hyperparametres = meta_champ["hyperparametres"]
            logger.info("Hyperparamètres récupérés depuis le champion courant.")

    pre = construire_preprocesseur(X_train)
    clf = HistGradientBoostingClassifier(
        random_state=config.RANDOM_SEED,
        **{k: v for k, v in hyperparametres.items()},
    )
    pipeline = Pipeline([("pre", pre), ("clf", clf)])
    logger.info("Entraînement du challenger sur {} observations …", len(X_train))
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
        "nom": "Challenger HistGradientBoosting",
        "date_entrainement": datetime.datetime.now().isoformat(timespec="seconds"),
        "metriques": metriques,
        "hyperparametres": hyperparametres,
    }
    with chemin_meta_challenger.open("w", encoding="utf-8") as fic:
        json.dump(meta_out, fic, ensure_ascii=False, indent=2)

    marqueur.write_text(datetime.datetime.now().isoformat(), encoding="utf-8")
    return pipeline, metriques


# ---------------------------------------------------------------------------
# Stage 4 — Evaluate
# ---------------------------------------------------------------------------


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


def stage_gate(metriques: dict[str, float]) -> bool:
    """Gate de promotion binaire.

    Conditions de blocage (logique ET — les deux doivent passer) :
    1. ``pr_auc_test >= config.CIBLES_PERFORMANCE["pr_auc_min"]``
    2. ``pr_auc_test >= pr_auc_champion - 0.01`` (tolérance de régression 1 pt)

    Returns
    -------
    True si le challenger peut être promu, False sinon.
    """
    seuil_absolu = float(config.CIBLES_PERFORMANCE["pr_auc_min"])
    pr_auc = metriques.get("pr_auc_test", 0.0)

    # Condition 1 — seuil absolu a priori
    if pr_auc < seuil_absolu:
        logger.error(
            "Gate ÉCHOUÉE — PR-AUC challenger = {:.4f} < seuil = {:.2f}. "
            "Challenger non promu. Vérifier les features et les données.",
            pr_auc,
            seuil_absolu,
        )
        return False

    # Condition 2 — régression vs champion
    chemin_meta_champion = config.ARTIFACTS / "models" / "best_model_meta.json"
    if chemin_meta_champion.exists():
        with chemin_meta_champion.open(encoding="utf-8") as fic:
            meta_champion = json.load(fic)
        pr_auc_champion = float(meta_champion.get("pr_auc_test", 0.0))
        tolerance = 0.01
        if pr_auc < pr_auc_champion - tolerance:
            logger.error(
                "Gate ÉCHOUÉE — challenger ({:.4f}) en régression vs champion "
                "({:.4f}, tolérance {:.2f}). Challenger non promu.",
                pr_auc,
                pr_auc_champion,
                tolerance,
            )
            return False
        logger.info(
            "Challenger ({:.4f}) ≥ champion ({:.4f}) − {:.2f} → condition 2 passée.",
            pr_auc,
            pr_auc_champion,
            tolerance,
        )

    logger.info("Gate PASSÉE — PR-AUC = {:.4f} ≥ seuil {:.2f}.", pr_auc, seuil_absolu)
    return True


# ---------------------------------------------------------------------------
# Stage 6 — Promote
# ---------------------------------------------------------------------------


def stage_promote(pipeline: Any, metriques: dict[str, float]) -> None:
    """Promeut le challenger : rotation des artefacts + enregistrement MLflow.

    Procédure de rollback : les archives horodatées dans ``artifacts/models/`` permettent
    de restaurer n'importe quel champion précédent en copiant le fichier archive
    vers ``best_model.pkl`` (opération manuelle documentée dans ``docs/RUNBOOK.md``).
    """
    import mlflow
    import mlflow.sklearn

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

    meta_out: dict[str, Any] = {
        "nom": "HistGradientBoosting (champion promu par flow)",
        "pr_auc_test": metriques.get("pr_auc_test"),
        "roc_auc_test": metriques.get("roc_auc_test"),
        "seuil_economique": 0.40,
        "date_entrainement": datetime.datetime.now().isoformat(timespec="seconds"),
        "n_train": metriques.get("n_train"),
        "n_test": metriques.get("n_test"),
        "source": "flows/retraining.py",
    }
    with chemin_meta.open("w", encoding="utf-8") as fic:
        json.dump(meta_out, fic, ensure_ascii=False, indent=2, default=str)

    logger.info("Artefact champion mis à jour → {}", chemin_champion.name)

    # Journalisation MLflow Registry
    try:
        mlflow.set_tracking_uri(str(config.MLRUNS))
        with mlflow.start_run(run_name=f"flow_retrain_{datetime.date.today().isoformat()}"):
            mlflow.log_metrics({k: v for k, v in metriques.items() if isinstance(v, (int, float))})
            mlflow.sklearn.log_model(pipeline, artifact_path="model")
            mlflow.set_tag("stage", "champion")
            mlflow.set_tag("source", "retrain_flow")
            logger.info("Modèle enregistré dans MLflow Registry.")
    except Exception as exc:
        logger.warning("MLflow non disponible — journalisation ignorée : {}", exc)


# ---------------------------------------------------------------------------
# Orchestration principale
# ---------------------------------------------------------------------------


def flow_retrainement(forcer: bool = False) -> int:
    """Exécute le flow complet : ingest → features → train → evaluate → gate → promote.

    Parameters
    ----------
    forcer:
        Si True, ignore les marqueurs d'idempotence et recalcule toutes les étapes.

    Returns
    -------
    Code de retour shell : 0 = succès, 1 = gate non passée, 2 = erreur non récupérable.
    """
    debut = datetime.datetime.now()
    logger.info("=" * 60)
    logger.info("Flow de réentraînement — début : {}", debut.isoformat(timespec="seconds"))
    logger.info("=" * 60)

    try:
        _avec_retries(stage_ingest, forcer, n_tentatives=3)
        _avec_retries(stage_features, forcer, n_tentatives=2)
        pipeline, metriques = _avec_retries(stage_train, forcer, n_tentatives=2)
        metriques = stage_evaluate(pipeline, metriques)
        gate_ok = stage_gate(metriques)

        if not gate_ok:
            logger.error(
                "Flow terminé — gate non passée. Modèle NON promu. "
                "Voir logs ci-dessus pour la raison du blocage."
            )
            return 1

        stage_promote(pipeline, metriques)

    except Exception as exc:
        logger.exception("Flow interrompu par une erreur non récupérable : {}", exc)
        return 2

    duree = (datetime.datetime.now() - debut).total_seconds()
    logger.info("Flow terminé avec succès en {:.1f}s.", duree)
    return 0


# ---------------------------------------------------------------------------
# Point d'entrée CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Flow de réentraînement churn SaaS — C9 CISIA.")
    parser.add_argument(
        "--forcer",
        action="store_true",
        help="Ignore les marqueurs d'idempotence et recalcule toutes les étapes.",
    )
    args = parser.parse_args()
    sys.exit(flow_retrainement(forcer=args.forcer))
