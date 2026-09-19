"""Interface en ligne de commande — churn_saas.

Commandes disponibles :
    check-data    Vérifie la présence et l'intégrité des fichiers de données.
    build-gold    Construit data/gold/gold_dataset.parquet à partir des sources brutes.
    train         Entraîne le modèle champion et sauvegarde l'artefact.
    evaluate      Évalue le modèle sauvegardé sur le split de test.
    predict       Applique le modèle à un fichier de données et écrit les prédictions.
    retrain       Réentraîne sur l'ensemble des données (pré-déploiement).
"""

from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
from typing import Annotated

import joblib
import numpy as np
import pandas as pd
import typer
from loguru import logger
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SklearnPipeline

from churn_saas import config
from churn_saas.data.loaders import charger_brut
from churn_saas.data.quality import (
    analyser_doublons,
    coercer_numeriques,
    parser_dates,
)
from churn_saas.features.build import (
    ajouter_features_metier,
    construire_preprocesseur,
    joindre_catalogue,
)
from churn_saas.features.enrichissement import enrichir_par_pays, enrichir_par_secteur
from churn_saas.models.train import journaliser_mlflow

app = typer.Typer(name="churn-saas", help="Outils CLI du projet churn SaaS CISIA.")


@app.callback()
def _callback() -> None:
    """Outils CLI du projet churn SaaS CISIA."""


# ---------------------------------------------------------------------------
# Colonnes numériques à coercer
# ---------------------------------------------------------------------------

_COLS_NUMERIQUES = [
    "anciennete_mois",
    "sieges_souscrits",
    "utilisateurs_actifs",
    "taux_adoption_pct",
    "connexions_30j",
    "heures_usage_30j",
    "fonctionnalites_total",
    "fonctionnalites_utilisees",
    "nb_integrations",
    "derniere_connexion_jours",
    "tickets_support_90j",
    "delai_reponse_support_h",
    "csat",
    "retards_paiement_12m",
    "revenu_mensuel_recurrent_eur",
    "valeur_vie_client_eur",
    "churn",
]

_COLS_DATES = ["date_souscription"]

# ---------------------------------------------------------------------------
# Marqueurs textuels de valeur manquante (cohérents avec §5)
# ---------------------------------------------------------------------------

_MARQUEURS_NA = ["", "n/a", "na", "nan", "null", "none", "#n/a", "-", "nd", "nr", "inconnu"]

# ---------------------------------------------------------------------------
# Colonnes présentes dans le gold mais non disponibles au moment de l'inférence API.
# Elles sont exclues avant l'entraînement CLI afin que le schéma du modèle coïncide
# exactement avec ce que l'API peut fournir (raw + features de ajouter_features_metier).
# ---------------------------------------------------------------------------

_COLONNES_NON_INFERABLES: list[str] = [
    # Date brute (anciennete_mois capture déjà l'information temporelle)
    "date_souscription",
    # Numéro du jour de souscription — dérivé de date_souscription, non fourni par l'API
    "jour_souscription",
    # Jointure catalogue des plans (nécessite un lookup externe non disponible à l'inférence)
    "prix_mensuel_par_siege_eur",
    "remise_consentie",
    "adequation_plan",
    # Enrichissement externe simulé (référentiels secteur/pays non disponibles en prod)
    "taux_churn_median_saas_pct",
    "dynamique_croissance",
    "ecart_adoption_secteur",
    "zone_reglementaire",
    "langue_support_fr",
    "decalage_horaire_paris_h",
    # Leurres prouvés inutiles (permutation importance ≈ 0, confirmé §6 EDA)
    "couleur_theme_interface",
    "code_datacenter",
    "groupe_experimentation",
    "commentaire_csm",
    # Agrégat appris sur le train uniquement (ne peut pas être reconstruit à l'inférence)
    "ecart_csat_secteur",
]

_CHEMIN_MODELE: Path = config.ARTIFACTS / "models" / "best_model.pkl"
_CHEMIN_META: Path = config.ARTIFACTS / "models" / "best_model_meta.json"
_CHEMIN_MODELE_FINAL: Path = config.ARTIFACTS / "models" / "best_model_final.pkl"

# ---------------------------------------------------------------------------
# Logique de construction du dataset gold
# ---------------------------------------------------------------------------


def construire_gold_dataset(forcer: bool = False) -> Path:
    """Chaîne complète : brut → nettoyage → features → catalogue → enrichissement → gold.

    Étapes :
    1. Chargement des sources brutes (dtype=str, intégrité vérifiée).
    2. Normalisation des marqueurs de valeur manquante.
    3. Déduplication (doublons exacts).
    4. Coercition des numériques stockés en texte.
    5. Parsing des dates multi-formats.
    6. Correction des valeurs métier impossibles (clip).
    7. Feature engineering métier (~15 features dérivées).
    8. Jointure catalogue des plans tarifaires.
    9. Enrichissement externe simulé (secteur + pays).
    10. Écriture du Parquet gold + fichier de métadonnées JSON.

    Parameters
    ----------
    forcer:
        Si True, recompose le gold même si ``data/gold/gold_dataset.parquet`` existe déjà.

    Returns
    -------
    Path vers ``data/gold/gold_dataset.parquet``.
    """
    chemin_gold = config.DONNEES_GOLD / "gold_dataset.parquet"
    chemin_meta = config.DONNEES_GOLD / "gold_metadata.json"

    if chemin_gold.exists() and not forcer:
        logger.info(
            "Dataset gold déjà présent ({}). Utilisez --forcer pour reconstruire.", chemin_gold
        )
        return chemin_gold

    config.DONNEES_GOLD.mkdir(parents=True, exist_ok=True)

    # --- 1. Chargement des sources brutes ---
    logger.info("Étape 1/9 — Chargement des sources brutes …")
    df = charger_brut("churn_saas_complet")
    catalogue = charger_brut("catalogue_plans")

    hash_principal = _sha256(config.DONNEES_BRUTES / "churn_saas_complet.csv")
    hash_catalogue = _sha256(config.DONNEES_BRUTES / "catalogue_plans.csv")

    # --- 2. Normalisation des marqueurs NA ---
    logger.info("Étape 2/9 — Normalisation des marqueurs de valeur manquante …")
    df = df.replace({m: np.nan for m in _MARQUEURS_NA})

    # --- 3. Déduplication ---
    logger.info("Étape 3/9 — Déduplication …")
    analyser_doublons(df, cle_metier="client_id")
    n_avant = len(df)
    df = df.drop_duplicates(keep="first")
    n_apres = len(df)
    logger.info(
        "Déduplication : {} → {} lignes ({} doublons exacts supprimés)",
        n_avant,
        n_apres,
        n_avant - n_apres,
    )

    # --- 4. Coercition des numériques ---
    logger.info("Étape 4/9 — Coercition des numériques …")
    cols_presents = [c for c in _COLS_NUMERIQUES if c in df.columns]
    df, _ = coercer_numeriques(df, cols_presents)

    # --- 5. Parsing des dates ---
    logger.info("Étape 5/9 — Parsing des dates …")
    df, _ = parser_dates(df, [c for c in _COLS_DATES if c in df.columns])

    # --- 6. Correction des valeurs impossibles (clip) ---
    logger.info("Étape 6/9 — Correction des valeurs impossibles …")
    if "taux_adoption_pct" in df.columns:
        df["taux_adoption_pct"] = pd.to_numeric(df["taux_adoption_pct"], errors="coerce").clip(
            0, 100
        )
    if "utilisateurs_actifs" in df.columns and "sieges_souscrits" in df.columns:
        df["utilisateurs_actifs"] = pd.to_numeric(df["utilisateurs_actifs"], errors="coerce").clip(
            lower=0, upper=pd.to_numeric(df["sieges_souscrits"], errors="coerce")
        )

    # --- 7. Feature engineering métier ---
    logger.info("Étape 7/9 — Feature engineering métier …")
    # Note : ecart_csat_secteur ne peut pas être calculé ici car les médianes CSAT par secteur
    # doivent être apprises sur le train uniquement (AgregatParGroupe dans le Pipeline sklearn).
    # Le gold dataset contient donc les features ligne-par-ligne uniquement.
    df = ajouter_features_metier(df, csat_median_par_secteur=None)

    # --- 8. Jointure catalogue ---
    logger.info("Étape 8/9 — Jointure catalogue des plans …")
    catalogue_num = catalogue.copy()
    cols_num_cat = ["prix_mensuel_par_siege_eur"]
    cols_num_cat_presents = [c for c in cols_num_cat if c in catalogue_num.columns]
    if cols_num_cat_presents:
        catalogue_num, _ = coercer_numeriques(catalogue_num, cols_num_cat_presents)
    df = joindre_catalogue(df, catalogue_num)

    # --- 9. Enrichissement externe simulé ---
    logger.info("Étape 9/9 — Enrichissement externe (simulé) …")
    df = enrichir_par_secteur(df)
    df = enrichir_par_pays(df)

    # Marquage des colonnes issues de l'enrichissement simulé dans les métadonnées
    _COLS_SIMULEES = [
        "taux_churn_median_saas_pct",
        "dynamique_croissance",
        "ecart_adoption_secteur",
        "zone_reglementaire",
        "langue_support_fr",
        "decalage_horaire_paris_h",
    ]
    cols_simulees_presentes = [c for c in _COLS_SIMULEES if c in df.columns]

    # --- Écriture du gold ---
    df.to_parquet(chemin_gold, index=False)
    logger.info(
        "Gold dataset écrit → {} ({} lignes × {} colonnes)",
        chemin_gold.relative_to(config.RACINE),
        len(df),
        len(df.columns),
    )

    # --- Métadonnées ---
    meta: dict[str, object] = {
        "date_construction": datetime.datetime.now().isoformat(timespec="seconds"),
        "version_code": _version_code(),
        "sources": {
            "churn_saas_complet.csv": hash_principal,
            "catalogue_plans.csv": hash_catalogue,
        },
        "nb_lignes": len(df),
        "nb_colonnes": len(df.columns),
        "colonnes": list(df.columns),
        "colonnes_enrichissement_simule": cols_simulees_presentes,
        "nb_doublons_exacts_supprimes": n_avant - n_apres,
        "colonnes_interdites_exclues_du_modele": config.COLONNES_INTERDITES,
        "note_gouvernance": (
            "Les colonnes listées dans 'colonnes_enrichissement_simule' proviennent "
            "de référentiels simulés (ordres de grandeur Gainsight/OpenView/Zuora 2023). "
            "En production, elles seraient remplacées par des données réelles "
            "selon les plans B documentés dans churn_saas.features.enrichissement.FICHES_SOURCES."
        ),
    }

    with chemin_meta.open("w", encoding="utf-8") as fic:
        json.dump(meta, fic, ensure_ascii=False, indent=2, default=str)
    logger.info("Métadonnées écrites → {}", chemin_meta.relative_to(config.RACINE))

    return chemin_gold


def _sha256(chemin: Path) -> str:
    """Empreinte SHA-256 du fichier, ou chaîne d'erreur si le fichier est absent."""
    if not chemin.exists():
        return f"ABSENT:{chemin.name}"
    return hashlib.sha256(chemin.read_bytes()).hexdigest()


def _version_code() -> str:
    """Retourne le hash court du commit git courant, ou 'non-versionné' si hors dépôt."""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=config.RACINE,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "non-versionne"
    except Exception:
        return "non-versionne"


def _charger_gold_pour_modele() -> tuple[pd.DataFrame, pd.Series]:
    """Charge le gold, épure les colonnes non-inférables, retourne (X, y)."""
    chemin_gold = config.DONNEES_GOLD / "gold_dataset.parquet"
    if not chemin_gold.exists():
        raise FileNotFoundError(
            f"Gold dataset introuvable : {chemin_gold}. Exécutez `churn-saas build-gold` d'abord."
        )

    df = pd.read_parquet(chemin_gold)

    if "churn" not in df.columns:
        raise ValueError("Colonne cible 'churn' absente du gold dataset.")

    y = df["churn"].astype(int)

    # Suppression de tout ce qui ne doit pas atteindre le modèle
    colonnes_a_supprimer = list(config.COLONNES_INTERDITES) + _COLONNES_NON_INFERABLES
    X = df.drop(columns=colonnes_a_supprimer, errors="ignore")

    logger.info(
        "Gold chargé — {} lignes × {} features (après exclusions)",
        len(X),
        len(X.columns),
    )
    return X, y


# ---------------------------------------------------------------------------
# Commandes CLI
# ---------------------------------------------------------------------------


@app.command("check-data")
def check_data() -> None:
    """Vérifie la présence et l'intégrité des fichiers de données.

    Contrôles : existence, empreinte SHA-256, volumétrie du gold.
    """
    ok = True

    # Fichiers sources bruts
    fichiers_bruts = [
        config.DONNEES_BRUTES / "churn_saas_complet.csv",
        config.DONNEES_BRUTES / "catalogue_plans.csv",
    ]
    for f in fichiers_bruts:
        if f.exists():
            empreinte = _sha256(f)
            taille_ko = f.stat().st_size // 1024
            typer.echo(f"  ✓ {f.name}  ({taille_ko} Ko)  sha256={empreinte[:12]}…")
        else:
            typer.echo(f"  ✗ ABSENT : {f}")
            ok = False

    # Gold dataset
    chemin_gold = config.DONNEES_GOLD / "gold_dataset.parquet"
    if chemin_gold.exists():
        empreinte = _sha256(chemin_gold)
        taille_ko = chemin_gold.stat().st_size // 1024
        typer.echo(f"  ✓ gold_dataset.parquet  ({taille_ko} Ko)  sha256={empreinte[:12]}…")

        # Vérification de l'absence des colonnes interdites dans le gold
        df_gold = pd.read_parquet(chemin_gold, columns=None)
        fuites = [c for c in config.COLONNES_INTERDITES if c in df_gold.columns and c != "churn"]
        if fuites:
            typer.echo(f"  ⚠ Colonnes interdites présentes dans le gold : {fuites}")
            ok = False
        else:
            typer.echo(f"  ✓ Gold OK — {len(df_gold)} lignes × {len(df_gold.columns)} colonnes")
    else:
        typer.echo(f"  ✗ ABSENT : {chemin_gold} — exécutez `churn-saas build-gold`")

    # Modèle artefact
    if _CHEMIN_MODELE.exists():
        taille_ko = _CHEMIN_MODELE.stat().st_size // 1024
        typer.echo(f"  ✓ best_model.pkl  ({taille_ko} Ko)")
    else:
        typer.echo("  ✗ ABSENT : best_model.pkl — exécutez `churn-saas train`")

    if ok:
        typer.echo("\nTous les contrôles ont réussi.")
    else:
        typer.echo("\nDes fichiers sont manquants. Consultez les messages ci-dessus.")
        raise typer.Exit(code=1)


@app.command("build-gold")
def build_gold(
    forcer: bool = typer.Option(
        False,
        "--forcer",
        help="Recalcule le gold même si data/gold/gold_dataset.parquet existe déjà.",
    ),
) -> None:
    """Construit data/gold/gold_dataset.parquet et son fichier de métadonnées JSON.

    Enchaîne : nettoyage → features métier → jointure catalogue → enrichissement externe.
    """
    chemin = construire_gold_dataset(forcer=forcer)
    typer.echo(f"Gold dataset disponible : {chemin}")


@app.command("train")
def train_cmd(
    forcer_gold: bool = typer.Option(
        False,
        "--forcer-gold",
        help="Reconstruit le gold avant l'entraînement.",
    ),
    n_iter: int = typer.Option(
        300,
        "--n-iter",
        help="Nombre d'itérations HistGradientBoosting.",
    ),
) -> None:
    """Entraîne le modèle champion (HistGradientBoosting) sur le gold dataset.

    Protocole : split stratifié 80/20 (seed unique = config.RANDOM_SEED).
    Évalue sur le split test et sauvegarde l'artefact dans artifacts/models/.
    """
    if forcer_gold:
        construire_gold_dataset(forcer=True)

    X, y = _charger_gold_pour_modele()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=config.RANDOM_SEED,
    )
    logger.info(
        "Split — train : {} lignes, test : {} lignes (stratifié, seed={})",
        len(X_train),
        len(X_test),
        config.RANDOM_SEED,
    )

    pre = construire_preprocesseur(X_train)
    clf = HistGradientBoostingClassifier(
        max_iter=n_iter,
        learning_rate=0.10,
        max_depth=5,
        l2_regularization=0.1,
        min_samples_leaf=20,
        random_state=config.RANDOM_SEED,
    )
    pipeline = SklearnPipeline([("pre", pre), ("clf", clf)])
    logger.info("Entraînement du pipeline sur {} observations …", len(X_train))
    pipeline.fit(X_train, y_train)

    y_proba_test = pipeline.predict_proba(X_test)[:, 1]
    pr_auc = float(average_precision_score(y_test, y_proba_test))
    roc_auc_val = float(roc_auc_score(y_test, y_proba_test))
    logger.info("Évaluation test — PR-AUC = {:.4f}, ROC-AUC = {:.4f}", pr_auc, roc_auc_val)

    # Sauvegarde de l'artefact
    _CHEMIN_MODELE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, _CHEMIN_MODELE)
    logger.info("Modèle sauvegardé → {}", _CHEMIN_MODELE)

    meta: dict[str, object] = {
        "nom": "HistGradientBoosting (champion CLI)",
        "pr_auc_test": round(pr_auc, 4),
        "roc_auc_test": round(roc_auc_val, 4),
        "seuil_economique": 0.40,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(len(X_train.columns)),
        "hyperparametres": {
            "max_iter": n_iter,
            "learning_rate": 0.10,
            "max_depth": 5,
            "l2_regularization": 0.1,
            "min_samples_leaf": 20,
        },
        "date_entrainement": datetime.datetime.now().isoformat(timespec="seconds"),
        "version_code": _version_code(),
        "colonnes_features": list(X_train.columns),
    }
    with _CHEMIN_META.open("w", encoding="utf-8") as fic:
        json.dump(meta, fic, ensure_ascii=False, indent=2, default=str)

    # Journalisation MLflow (optionnelle — ne bloque pas si MLflow échoue)
    try:
        journaliser_mlflow(
            "HistGradientBoosting CLI",
            pipeline,
            {"pr_auc_test": pr_auc, "roc_auc_test": roc_auc_val},
            meta.get("hyperparametres", {}),  # type: ignore[arg-type]
        )
    except Exception as exc:
        logger.warning("MLflow non disponible — journalisation ignorée : {}", exc)

    typer.echo(f"Modèle entraîné — PR-AUC test = {pr_auc:.4f} | artefact : {_CHEMIN_MODELE}")


@app.command("evaluate")
def evaluate_cmd(
    fichier: Annotated[
        Path | None,
        typer.Option(
            "--fichier", help="Fichier parquet à évaluer (défaut : split test du gold, même seed)."
        ),
    ] = None,
) -> None:
    """Évalue le modèle sauvegardé et affiche les métriques de classification.

    Sans --fichier, recrée le split test 20 % avec la même graine (reproductible).
    """
    if not _CHEMIN_MODELE.exists():
        typer.echo(f"Modèle introuvable : {_CHEMIN_MODELE}. Exécutez `churn-saas train`.")
        raise typer.Exit(code=1)

    pipeline = joblib.load(_CHEMIN_MODELE)
    logger.info("Modèle chargé depuis {}", _CHEMIN_MODELE)

    if fichier is not None:
        df_eval = (
            pd.read_parquet(fichier) if str(fichier).endswith(".parquet") else pd.read_csv(fichier)
        )
        if "churn" not in df_eval.columns:
            typer.echo("Le fichier fourni doit contenir la colonne 'churn'.")
            raise typer.Exit(code=1)
        y_eval = df_eval["churn"].astype(int)
        X_eval = df_eval.drop(
            columns=list(config.COLONNES_INTERDITES) + _COLONNES_NON_INFERABLES,
            errors="ignore",
        )
    else:
        X, y = _charger_gold_pour_modele()
        _, X_eval, _, y_eval = train_test_split(
            X,
            y,
            test_size=0.20,
            stratify=y,
            random_state=config.RANDOM_SEED,
        )

    y_proba = pipeline.predict_proba(X_eval)[:, 1]
    pr_auc = float(average_precision_score(y_eval, y_proba))
    roc_auc_val = float(roc_auc_score(y_eval, y_proba))

    typer.echo(f"\n{'─' * 40}")
    typer.echo(f"  Évaluation — {len(X_eval)} observations")
    typer.echo(f"{'─' * 40}")
    typer.echo(f"  PR-AUC   : {pr_auc:.4f}  (cible ≥ {config.CIBLES_PERFORMANCE['pr_auc_min']})")
    typer.echo(f"  ROC-AUC  : {roc_auc_val:.4f}")
    gate = pr_auc >= config.CIBLES_PERFORMANCE["pr_auc_min"]
    typer.echo(f"  Gate PR-AUC : {'✓ PASSÉE' if gate else '✗ ÉCHOUÉE'}")
    typer.echo(f"{'─' * 40}\n")

    if not gate:
        raise typer.Exit(code=1)


@app.command("predict")
def predict_cmd(
    fichier_entree: Annotated[
        Path,
        typer.Argument(
            help="Fichier parquet ou CSV contenant les features clients (sans colonnes interdites)."
        ),
    ],
    fichier_sortie: Annotated[
        Path | None,
        typer.Option(
            "--sortie", help="Fichier CSV de sortie (défaut : <fichier_entree>_predictions.csv)."
        ),
    ] = None,
) -> None:
    """Applique le modèle à un fichier de données et écrit les prédictions.

    Le fichier d'entrée doit avoir le même schéma de features que le gold
    (colonnes interdites absentes ou silencieusement ignorées).
    """
    if not _CHEMIN_MODELE.exists():
        typer.echo(f"Modèle introuvable : {_CHEMIN_MODELE}. Exécutez `churn-saas train`.")
        raise typer.Exit(code=1)

    if not fichier_entree.exists():
        typer.echo(f"Fichier d'entrée introuvable : {fichier_entree}")
        raise typer.Exit(code=1)

    pipeline = joblib.load(_CHEMIN_MODELE)

    if str(fichier_entree).endswith(".parquet"):
        df = pd.read_parquet(fichier_entree)
    else:
        df = pd.read_csv(fichier_entree)

    # Conservation de l'identifiant pour la sortie
    id_col = "client_id" if "client_id" in df.columns else None
    ids = df[id_col].copy() if id_col else None

    X = df.drop(
        columns=list(config.COLONNES_INTERDITES) + _COLONNES_NON_INFERABLES,
        errors="ignore",
    )
    # Application du même feature engineering que lors de l'entraînement CLI
    X = ajouter_features_metier(X, csat_median_par_secteur=None)

    probas = pipeline.predict_proba(X)[:, 1]
    seuil = 0.40  # par défaut ; peut être lu depuis _CHEMIN_META
    if _CHEMIN_META.exists():
        with _CHEMIN_META.open() as fic:
            seuil = float(json.load(fic).get("seuil_economique", 0.40))

    df_sortie = pd.DataFrame(
        {
            "probabilite_churn": probas.round(4),
            "decision": np.where(
                probas >= max(seuil + 0.20, 0.60),
                "ALERTE_ROUGE",
                np.where(probas >= seuil, "SURVEILLANCE", "OK"),
            ),
            "seuil_applique": seuil,
        }
    )
    if ids is not None:
        df_sortie.insert(0, id_col, ids.values)

    sortie = fichier_sortie or fichier_entree.with_name(fichier_entree.stem + "_predictions.csv")
    df_sortie.to_csv(sortie, index=False)
    logger.info("Prédictions écrites → {} ({} lignes)", sortie, len(df_sortie))
    typer.echo(f"Prédictions disponibles : {sortie}")


@app.command("retrain")
def retrain_cmd() -> None:
    """Réentraîne sur l'ensemble des données (train + val) avant gel de déploiement.

    Utilisé en phase de pré-déploiement : après sélection du modèle champion
    (via `train`), on réentraîne sur 100 % des données pour maximiser la couverture.
    L'artefact produit est best_model_final.pkl (distinct du best_model.pkl de validation).
    """
    X, y = _charger_gold_pour_modele()

    # Charger les hyperparamètres du champion si disponibles
    hyperparametres: dict[str, object] = {
        "max_iter": 300,
        "learning_rate": 0.10,
        "max_depth": 5,
        "l2_regularization": 0.1,
        "min_samples_leaf": 20,
    }
    if _CHEMIN_META.exists():
        with _CHEMIN_META.open() as fic:
            meta_champ = json.load(fic)
        hyperparametres = meta_champ.get("hyperparametres", hyperparametres)  # type: ignore[assignment]

    pre = construire_preprocesseur(X)
    clf = HistGradientBoostingClassifier(
        random_state=config.RANDOM_SEED,
        **{k: v for k, v in hyperparametres.items()},
    )
    pipeline = SklearnPipeline([("pre", pre), ("clf", clf)])
    logger.info("Réentraînement sur {} observations (100 %% des données) …", len(X))
    pipeline.fit(X, y)

    _CHEMIN_MODELE_FINAL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, _CHEMIN_MODELE_FINAL)

    meta_final: dict[str, object] = {
        "nom": "HistGradientBoosting (final — 100 %% données)",
        "seuil_economique": 0.40,
        "n_observations": int(len(X)),
        "hyperparametres": hyperparametres,
        "date_entrainement": datetime.datetime.now().isoformat(timespec="seconds"),
        "version_code": _version_code(),
        "note": (
            "Modèle entraîné sur l'ensemble des données disponibles avant déploiement. "
            "Les métriques de performance proviennent du best_model.pkl (split test 20 %)."
        ),
    }
    chemin_meta_final = _CHEMIN_MODELE_FINAL.with_suffix(".json")
    with chemin_meta_final.open("w", encoding="utf-8") as fic:
        json.dump(meta_final, fic, ensure_ascii=False, indent=2, default=str)

    logger.info("Modèle final sauvegardé → {}", _CHEMIN_MODELE_FINAL)
    typer.echo(f"Réentraînement terminé — artefact : {_CHEMIN_MODELE_FINAL}")


if __name__ == "__main__":
    app()
