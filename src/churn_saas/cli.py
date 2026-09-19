"""Interface en ligne de commande — churn_saas.

Commandes disponibles :
    build-gold    Construit data/gold/gold_dataset.parquet à partir des sources brutes.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from loguru import logger

from churn_saas import config
from churn_saas.data.loaders import charger_brut
from churn_saas.data.quality import (
    analyser_doublons,
    coercer_numeriques,
    parser_dates,
)
from churn_saas.features.build import ajouter_features_metier, joindre_catalogue
from churn_saas.features.enrichissement import enrichir_par_pays, enrichir_par_secteur

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


# ---------------------------------------------------------------------------
# Commandes CLI
# ---------------------------------------------------------------------------


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


if __name__ == "__main__":
    app()
