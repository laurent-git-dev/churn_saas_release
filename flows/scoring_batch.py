"""Flow Prefect 3 — batch nocturne de scoring churn (C6).

Architecture : chargement du dataset gold → scoring par le pipeline sklearn →
publication des scores dans reports/tables/. Conçu pour s'exécuter toutes les nuits
(cron 0 2 * * *) sans serveur Prefect grâce au local runner de Prefect 3.

Migration depuis le mode local : ajouter `prefect deploy --all` et un worker Prefect
pour activer l'orchestration distante (aucune modification de code nécessaire).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Literal

import pandas as pd
from loguru import logger
from prefect import flow, task

from churn_saas import config
from churn_saas.api.model_store import ModelStore

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

DecisionChurn = Literal["ALERTE_ROUGE", "SURVEILLANCE", "OK"]

# ---------------------------------------------------------------------------
# Helpers métier (répliqués depuis api/main.py — évite d'importer FastAPI)
# ---------------------------------------------------------------------------

_SEUIL_DEFAUT: float = 0.40


def _decision(probabilite: float, seuil: float) -> DecisionChurn:
    if probabilite >= max(seuil + 0.20, 0.60):
        return "ALERTE_ROUGE"
    if probabilite >= seuil:
        return "SURVEILLANCE"
    return "OK"


def _valeur_a_risque(mrr_eur: float, probabilite: float) -> float:
    horizon: int = int(config.HYPOTHESES_ECONOMIQUES.get("horizon_mois", 12))
    return round(probabilite * mrr_eur * horizon, 2)


# ---------------------------------------------------------------------------
# Tasks Prefect
# ---------------------------------------------------------------------------


@task(
    name="ingest-gold",
    retries=3,
    retry_delay_seconds=30,
    description="Charge le dataset gold et retire les colonnes interdites.",
)
def ingest_gold() -> pd.DataFrame:
    """Charge data/gold/gold_dataset.parquet et retire les COLONNES_INTERDITES."""
    chemin = config.DONNEES_GOLD / "gold_dataset.parquet"
    if not chemin.exists():
        raise FileNotFoundError(
            f"Dataset gold introuvable : {chemin}. "
            "Exécutez `make notebook` ou `churn-saas construire-gold` pour le générer."
        )
    df = pd.read_parquet(chemin)
    logger.info("Gold chargé : {} lignes, {} colonnes", len(df), df.shape[1])

    # On conserve client_id et MRR pour l'output avant de les retirer du DataFrame d'inférence
    colonnes_a_retirer = [c for c in config.COLONNES_INTERDITES if c in df.columns]
    df_propre = df.drop(columns=colonnes_a_retirer)

    # client_id et MRR sont réintégrés via un index commun à la sortie de scorer_comptes
    df_propre["_client_id"] = df["client_id"] if "client_id" in df.columns else df.index
    df_propre["_mrr_eur"] = (
        df["revenu_mensuel_recurrent_eur"] if "revenu_mensuel_recurrent_eur" in df.columns else 0.0
    )
    logger.info(
        "Colonnes interdites retirées : {}",
        colonnes_a_retirer,
    )
    return df_propre


@task(
    name="scorer-comptes",
    description="Charge best_model.pkl et produit les probabilités de churn.",
)
def scorer_comptes(df: pd.DataFrame) -> pd.DataFrame:
    """Charge le pipeline sklearn et score tous les comptes du dataset gold."""
    chemin_modele = config.ARTIFACTS / "models" / "best_model.pkl"
    if not chemin_modele.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {chemin_modele}. "
            "Exécutez `churn-saas train` ou `make notebook` pour entraîner."
        )

    store = ModelStore()
    store.charger(chemin_modele=chemin_modele)
    seuil: float = store.seuil

    # Colonnes de service extraites avant inférence
    client_ids = df["_client_id"].copy()
    mrr_eur = df["_mrr_eur"].copy()
    df_inference = df.drop(columns=["_client_id", "_mrr_eur"])

    logger.info("Scoring de {} comptes (seuil = {:.2f})…", len(df_inference), seuil)
    probas = store.predire(df_inference)[:, 1]  # P(churn)

    decisions: list[DecisionChurn] = [_decision(float(p), seuil) for p in probas]
    valeurs = [
        _valeur_a_risque(float(mrr), float(p)) for mrr, p in zip(mrr_eur, probas, strict=True)
    ]

    resultats = pd.DataFrame(
        {
            "client_id": client_ids.values,
            "probabilite_churn": probas.round(4),
            "decision": decisions,
            "valeur_a_risque_eur": valeurs,
            "seuil_applique": seuil,
        }
    )
    logger.info(
        "Scoring terminé — ALERTE_ROUGE: {}, SURVEILLANCE: {}, OK: {}",
        (resultats["decision"] == "ALERTE_ROUGE").sum(),
        (resultats["decision"] == "SURVEILLANCE").sum(),
        (resultats["decision"] == "OK").sum(),
    )
    return resultats


@task(
    name="publier-scores",
    description="Écrit les scores en parquet et le résumé JSON (idempotent).",
)
def publier_scores(resultats: pd.DataFrame, date_ref: str) -> Path:
    """Persiste les scores dans reports/tables/ avec idempotence sur la date.

    Si le fichier du jour existe déjà, la tâche loggue et retourne le chemin existant
    sans écraser — garantit la réentrée du flow en cas de relance accidentelle.
    """
    chemin_parquet = config.TABLES / f"scores_batch_{date_ref}.parquet"
    chemin_json = config.TABLES / f"scores_batch_{date_ref}_synthese.json"

    if chemin_parquet.exists():
        logger.info("Scores du {} déjà publiés — skip (idempotence).", date_ref)
        return chemin_parquet

    config.TABLES.mkdir(parents=True, exist_ok=True)
    resultats.to_parquet(chemin_parquet, index=False)

    synthese = {
        "date": date_ref,
        "n_comptes": len(resultats),
        "nb_alertes_rouges": int((resultats["decision"] == "ALERTE_ROUGE").sum()),
        "nb_surveillances": int((resultats["decision"] == "SURVEILLANCE").sum()),
        "nb_ok": int((resultats["decision"] == "OK").sum()),
        "valeur_totale_a_risque_eur": float(resultats["valeur_a_risque_eur"].sum()),
    }
    chemin_json.write_text(json.dumps(synthese, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(
        "Scores publiés : {} ({} comptes, valeur à risque totale : {:.0f} €)",
        chemin_parquet,
        synthese["n_comptes"],
        synthese["valeur_totale_a_risque_eur"],
    )
    return chemin_parquet


# ---------------------------------------------------------------------------
# Flow principal
# ---------------------------------------------------------------------------


@flow(
    name="batch-scoring-nocturne",
    description=(
        "Orchestre le scoring nocturne de tous les comptes actifs : "
        "ingest → score → publication. "
        "S'exécute en mode local runner (sans serveur Prefect) ou via `prefect deploy`."
    ),
    log_prints=True,
)
def batch_scoring_nocturne(date_ref: str | None = None) -> dict:
    """Flow de scoring batch nocturne — point d'entrée principal.

    Parameters
    ----------
    date_ref:
        Date de référence au format YYYY-MM-DD (défaut : aujourd'hui).
        Permet le replay d'une journée passée sans modifier les scores du jour.

    Returns
    -------
    dict
        Synthèse du run : n_comptes, répartition des décisions, chemin de sortie.
    """
    if date_ref is None:
        date_ref = date.today().isoformat()

    logger.info("=== Batch scoring nocturne — date de référence : {} ===", date_ref)

    df = ingest_gold()
    resultats = scorer_comptes(df)
    chemin = publier_scores(resultats, date_ref)

    synthese = {
        "date": date_ref,
        "n_comptes": len(resultats),
        "nb_alertes_rouges": int((resultats["decision"] == "ALERTE_ROUGE").sum()),
        "nb_surveillances": int((resultats["decision"] == "SURVEILLANCE").sum()),
        "nb_ok": int((resultats["decision"] == "OK").sum()),
        "valeur_totale_a_risque_eur": float(resultats["valeur_a_risque_eur"].sum()),
        "chemin_sortie": str(chemin),
    }
    logger.info("=== Batch terminé : {} ===", synthese)
    return synthese


# ---------------------------------------------------------------------------
# Point d'entrée CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Lance le batch de scoring nocturne.")
    parser.add_argument(
        "--date-ref",
        default=None,
        help="Date de référence YYYY-MM-DD (défaut : aujourd'hui).",
    )
    args = parser.parse_args()
    batch_scoring_nocturne(date_ref=args.date_ref)
