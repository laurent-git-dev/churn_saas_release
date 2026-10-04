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
from typing import Any, Literal

import numpy as np
import pandas as pd
from loguru import logger
from prefect import flow, task

from churn_saas import config, economie
from churn_saas.api.model_store import ModelStore

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

DecisionChurn = Literal["ALERTE_ROUGE", "SURVEILLANCE", "OK"]
# Raisons publiées par compte, comme `facteurs_shap` dans la réponse de l'API
NB_FACTEURS_SHAP = 3
ActionCS = Literal["Contacter", "Action automatisée", "Veille"]

# ---------------------------------------------------------------------------
# Helpers métier
#
# Ni la règle de décision ni la valeur à risque ne sont répliquées : toutes deux viennent de
# `churn_saas.economie`, autorité unique partagée avec l'API et la §12.11. Les copies précédentes avaient
# divergé — la marge brute manquait ici et dans l'API, surestimant l'exposition de +39 %.
# ---------------------------------------------------------------------------


def _prioriser(
    probas: np.ndarray, mrr_eur: np.ndarray
) -> tuple[list[int | None], list[ActionCS], np.ndarray]:
    """Rang de priorité et action CS de chaque compte (règle de §12.6).

    Les ``capacite_gestes_mois`` comptes de plus forte valeur attendue positive sont à
    contacter (rang 1 à N) ; les autres comptes à valeur attendue positive reçoivent une
    action automatisée ; les autres restent en veille. MRR inconnu → veille, sans rang.
    """
    valeurs = np.asarray(economie.valeur_attendue_intervention(probas, mrr_eur), dtype=float)
    selection = economie.selection_sous_capacite(valeurs)
    rangs: list[int | None] = [None] * len(valeurs)
    for rang, idx in enumerate(selection, start=1):
        rangs[int(idx)] = rang
    actions: list[ActionCS] = [
        "Contacter" if r is not None else ("Action automatisée" if v > 0 else "Veille")
        for r, v in zip(rangs, np.nan_to_num(valeurs, nan=0.0), strict=True)
    ]
    return rangs, actions, valeurs


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
    """Charge le pipeline sklearn, score tous les comptes du gold et publie leurs raisons.

    Chaque compte reçoit ses ``NB_FACTEURS_SHAP`` variables les plus contributives au risque
    (`facteur_shap_1` à `_3`, libellés métier), calculées par la même fonction que l'API
    (`ModelStore.predire_et_expliquer`) : la notification et la fiche disent la même chose.

    Agnostique de la famille du champion (régression logistique recalibrée, LightGBM…) : le
    pipeline sélectionne lui-même ses colonnes, dont les features dérivées déjà présentes dans
    le gold (`intensite_support`…). Le seuil de surveillance est celui du ``ModelStore`` :
    métadonnées du modèle promu, sinon seuil de vigilance de la règle à deux niveaux.
    """
    chemin_modele = config.ARTIFACTS / "models" / "best_model.pkl"
    if not chemin_modele.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {chemin_modele}. "
            "Exécutez `churn-saas train` ou `make notebook` pour entraîner."
        )

    store = ModelStore()
    # Métadonnées du même dossier que le modèle : elles portent son seuil et sa famille
    store.charger(
        chemin_modele=chemin_modele, chemin_meta=chemin_modele.with_name("best_model_meta.json")
    )
    seuil: float = store.seuil

    # Colonnes de service extraites avant inférence
    client_ids = df["_client_id"].copy()
    mrr_eur = df["_mrr_eur"].copy()
    df_inference = df.drop(columns=["_client_id", "_mrr_eur"])

    logger.info(
        "Scoring de {} comptes — modèle {}, seuil de surveillance = {:.4f}…",
        len(df_inference),
        store.nom_modele,
        seuil,
    )
    # Probabilités et raisons SHAP en un seul passage du prétraitement, comme l'API
    probas_2d, facteurs = store.predire_et_expliquer(df_inference, NB_FACTEURS_SHAP)
    probas = probas_2d[:, 1]  # P(churn)

    decisions: list[DecisionChurn] = [economie.niveau_risque(float(p), seuil) for p in probas]
    valeurs = [_valeur_a_risque(mrr, float(p)) for mrr, p in zip(mrr_eur, probas, strict=True)]
    rangs, actions, valeurs_attendues = _prioriser(probas, mrr_eur.to_numpy(dtype=float))

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
            # Règle de priorisation sous capacité (§12.6) : c'est elle qui décide qui le CSM
            # appelle ; `decision` (seuil) reste l'indicateur de risque compte par compte
            "valeur_attendue_geste_eur": np.round(valeurs_attendues, 2),
            "rang_priorite": rangs,
            "action_cs": actions,
            # Raisons du score (§2, CU2 : notification CRM avec les 3 variables SHAP), une
            # colonne par rang : le CRM les affiche sans décoder de structure imbriquée
            **{
                f"facteur_shap_{rang}": [
                    liste[rang - 1]["libelle"] if len(liste) >= rang else None for liste in facteurs
                ]
                for rang in range(1, NB_FACTEURS_SHAP + 1)
            },
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


def _synthese(resultats: pd.DataFrame, date_ref: str) -> dict[str, Any]:
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
        "nb_a_contacter": int((resultats["action_cs"] == "Contacter").sum()),
        "nb_actions_automatisees": int((resultats["action_cs"] == "Action automatisée").sum()),
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
def batch_scoring_nocturne(date_ref: str | None = None, forcer: bool = False) -> dict[str, Any]:
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
