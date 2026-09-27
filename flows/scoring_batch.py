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

from churn_saas import config, economie
from churn_saas.api.model_store import ModelStore

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

DecisionChurn = Literal["ALERTE_ROUGE", "SURVEILLANCE", "OK"]

# ---------------------------------------------------------------------------
# Helpers métier
#
# La règle de décision est répliquée depuis api/main.py (évite d'importer FastAPI dans
# le worker), mais la valeur à risque NE L'EST PAS : elle vient de `churn_saas.economie`,
# autorité unique partagée avec l'API et la §12.11. Les copies précédentes avaient
# divergé — la marge brute manquait ici et dans l'API, surestimant l'exposition de +39 %.
# ---------------------------------------------------------------------------

_SEUIL_DEFAUT: float = 0.40


def _decision(probabilite: float, seuil: float) -> DecisionChurn:
    if probabilite >= max(seuil + 0.20, 0.60):
        return "ALERTE_ROUGE"
    if probabilite >= seuil:
        return "SURVEILLANCE"
    return "OK"


def _valeur_a_risque(mrr_eur: float, probabilite: float) -> float:
    """Valeur à risque d'un compte, ou NaN si son MRR est inconnu.

    Politique de manquance du batch : on écrit `NaN` dans le Parquet (convention
    colonnaire correcte, là où l'API renvoie `null` + motif en JSON) et on déclare le
    périmètre dans la colonne `mrr_disponible` et dans la synthèse — plutôt que de
    laisser un `NaN` muet que le CRM ne saurait pas distinguer d'une erreur de scoring.
    """
    if pd.isna(mrr_eur):
        return float("nan")
    return round(float(economie.valeur_a_risque(probabilite, float(mrr_eur))), 2)


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
    valeurs = [_valeur_a_risque(mrr, float(p)) for mrr, p in zip(mrr_eur, probas, strict=True)]

    resultats = pd.DataFrame(
        {
            "client_id": client_ids.values,
            "probabilite_churn": probas.round(4),
            "decision": decisions,
            "valeur_a_risque_eur": valeurs,
            # Le CRM doit pouvoir filtrer les comptes non valorisables sans deviner
            # ce que signifie un NaN : ici, MRR inconnu — pas une erreur de scoring.
            "mrr_disponible": [not bool(pd.isna(m)) for m in mrr_eur],
            "seuil_applique": seuil,
        }
    )
    logger.info(
        "Scoring terminé — ALERTE_ROUGE: {}, SURVEILLANCE: {}, OK: {}",
        (resultats["decision"] == "ALERTE_ROUGE").sum(),
        (resultats["decision"] == "SURVEILLANCE").sum(),
        (resultats["decision"] == "OK").sum(),
    )
    nb_sans_mrr = int((~resultats["mrr_disponible"]).sum())
    if nb_sans_mrr:
        logger.warning(
            "{} compte(s) sans MRR — scorés mais non valorisés "
            "(valeur_a_risque_eur = NaN, mrr_disponible = False).",
            nb_sans_mrr,
        )
    return resultats


def _synthese(resultats: pd.DataFrame, date_ref: str) -> dict:
    """Synthèse d'un run — le périmètre de valorisation y est déclaré, jamais implicite.

    `Series.sum()` ignore les NaN par défaut : sans les deux compteurs ci-dessous, le
    total porterait silencieusement sur un sous-ensemble des comptes.
    """
    valorisables = resultats["mrr_disponible"]
    return {
        "date": date_ref,
        "n_comptes": len(resultats),
        "nb_alertes_rouges": int((resultats["decision"] == "ALERTE_ROUGE").sum()),
        "nb_surveillances": int((resultats["decision"] == "SURVEILLANCE").sum()),
        "nb_ok": int((resultats["decision"] == "OK").sum()),
        "valeur_totale_a_risque_eur": float(
            resultats.loc[valorisables, "valeur_a_risque_eur"].sum()
        ),
        "nb_comptes_valorises": int(valorisables.sum()),
        "nb_comptes_sans_mrr": int((~valorisables).sum()),
    }


@task(
    name="publier-scores",
    description="Écrit les scores en parquet et le résumé JSON (idempotent).",
)
def publier_scores(resultats: pd.DataFrame, date_ref: str, forcer: bool = False) -> Path:
    """Persiste les scores dans reports/tables/ avec idempotence sur la date.

    Si le fichier du jour existe déjà, la tâche loggue et retourne le chemin existant
    sans écraser — garantit la réentrée du flow en cas de relance accidentelle.

    ``forcer=True`` republie malgré tout. Nécessaire quand le contenu — et non la date —
    a changé : correction de la formule économique, nouvelle colonne, réentraînement du
    modèle. Sans cette porte de sortie, des scores publiés avec une formule erronée
    resteraient figés jusqu'au lendemain.
    """
    chemin_parquet = config.TABLES / f"scores_batch_{date_ref}.parquet"
    chemin_json = config.TABLES / f"scores_batch_{date_ref}_synthese.json"

    if chemin_parquet.exists() and not forcer:
        logger.info(
            "Scores du {} déjà publiés — skip (idempotence ; --forcer pour republier).",
            date_ref,
        )
        return chemin_parquet

    config.TABLES.mkdir(parents=True, exist_ok=True)
    resultats.to_parquet(chemin_parquet, index=False)

    synthese = _synthese(resultats, date_ref)
    chemin_json.write_text(json.dumps(synthese, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(
        "Scores publiés : {} ({} comptes, valeur à risque totale : {:.0f} € "
        "sur {} comptes valorisés)",
        chemin_parquet,
        synthese["n_comptes"],
        synthese["valeur_totale_a_risque_eur"],
        synthese["nb_comptes_valorises"],
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
def batch_scoring_nocturne(date_ref: str | None = None, forcer: bool = False) -> dict:
    """Flow de scoring batch nocturne — point d'entrée principal.

    Parameters
    ----------
    date_ref:
        Date de référence au format YYYY-MM-DD (défaut : aujourd'hui).
        Permet le replay d'une journée passée sans modifier les scores du jour.
    forcer:
        Republie les scores même si ceux de ``date_ref`` existent déjà — à utiliser
        quand le contenu a changé (formule, colonnes, modèle) et non la date.

    Returns
    -------
    dict
        Synthèse du run : n_comptes, répartition des décisions, périmètre valorisé,
        chemin de sortie.
    """
    if date_ref is None:
        date_ref = date.today().isoformat()

    logger.info("=== Batch scoring nocturne — date de référence : {} ===", date_ref)

    df = ingest_gold()
    resultats = scorer_comptes(df)
    chemin = publier_scores(resultats, date_ref, forcer=forcer)

    synthese = _synthese(resultats, date_ref) | {"chemin_sortie": str(chemin)}
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
    parser.add_argument(
        "--forcer",
        action="store_true",
        help="Republie les scores de la date même s'ils existent déjà (formule modifiée…).",
    )
    args = parser.parse_args()
    batch_scoring_nocturne(date_ref=args.date_ref, forcer=args.forcer)
