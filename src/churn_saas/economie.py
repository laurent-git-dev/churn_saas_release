"""Formule unique de la valeur à risque et des paliers de risque — autorité partagée API /
batch / notebook.

Ce module est **la seule** implémentation de la valeur à risque du projet. Il existe
parce que la formule avait été recopiée à trois endroits (API, flow de scoring batch,
module de régression) et que les copies avaient divergé : deux d'entre elles omettaient
la marge brute, surestimant l'exposition de +38,9 % (= 1 / 0,72) par rapport à la formule
défendue en §12.11 du notebook.

Dépendances volontairement minimales (`config`, `cache` + `loguru`) : le module est importé
par le conteneur d'inférence, qui n'a rien à faire de matplotlib — d'où le choix de ne pas
héberger cette fonction dans `models.regression` ni dans `models.economics`.

Politique de manquance : ce module calcule, il ne décide pas. Un MRR absent produit une
valeur absente (`None` / `NaN`) ; c'est à l'appelant de dire comment il l'expose —
`null` + motif pour l'API, colonne `mrr_disponible` pour le batch.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from loguru import logger

from churn_saas import cache, config

# Scalaire ou vecteur — la formule est purement élément par élément
_NumType = float | np.ndarray


def valeur_a_risque(
    proba_churn: _NumType,
    mrr_mensuel: _NumType,
    horizon_mois: int | None = None,
    marge_brute: float | None = None,
) -> _NumType:
    """Valeur future à risque en euros — formule de référence (point de vigilance n°3).

    Formule retenue :

    .. code-block:: text

        valeur_à_risque = P(churn) × MRR_mensuel × horizon_mois × marge_brute

    ⚠️  Ne multiplie **pas** P(churn) par ``valeur_vie_client_eur`` — ce serait compter deux
    fois le même risque. La CLV est prospective (§6.7) : MRR × une durée de vie estimée, plus
    courte chez les churners, donc qui intègre déjà le risque de départ, sur un horizon non
    documenté. La formule serait la même pour une CLV réalisée, puisqu'on ne perd pas une
    valeur déjà encaissée.

    La marge brute est indispensable : un euro de MRR perdu ne coûte pas un euro de
    résultat, mais sa marge. L'omettre gonfle mécaniquement l'exposition de 1 / marge.

    Parameters
    ----------
    proba_churn:
        Probabilité de churn prédite ∈ [0, 1] — scalaire ou ndarray.
    mrr_mensuel:
        Revenu mensuel récurrent en euros — scalaire ou ndarray.
    horizon_mois:
        Horizon en mois. Défaut : ``config.HYPOTHESES_ECONOMIQUES["horizon_mois"]``.
    marge_brute:
        Marge brute ∈ ]0, 1]. Défaut : ``config.HYPOTHESES_ECONOMIQUES["marge_brute_pct"]``.

    Returns
    -------
    Valeur à risque en euros, même forme que les entrées (scalaire ou ndarray).
    """
    if horizon_mois is None:
        horizon_mois = int(config.HYPOTHESES_ECONOMIQUES["horizon_mois"])
    if marge_brute is None:
        marge_brute = float(config.HYPOTHESES_ECONOMIQUES["marge_brute_pct"])

    valeur = proba_churn * mrr_mensuel * horizon_mois * marge_brute

    if isinstance(proba_churn, np.ndarray) or isinstance(mrr_mensuel, np.ndarray):
        arr = np.asarray(valeur, dtype=float)
        logger.debug(
            "valeur_a_risque — médiane={:.0f} €  max={:.0f} €  (horizon={}m, marge={:.0%})",
            float(np.nanmedian(arr)),
            float(np.nanmax(arr)),
            horizon_mois,
            marge_brute,
        )
        return arr

    v = float(valeur)
    logger.debug(
        "valeur_a_risque — {:.0f} €  (P(churn)={:.2%}, MRR={:.0f} €, horizon={}m, marge={:.0%})",
        v,
        float(proba_churn),
        float(mrr_mensuel),
        horizon_mois,
        marge_brute,
    )
    return v


def cout_intervention() -> float:
    """Coût d'un geste de rétention en euros : coût horaire CSM × durée du geste."""
    h = config.HYPOTHESES_ECONOMIQUES
    return float(h["cout_horaire_csm_eur"]) * float(h["duree_geste_retention_h"])


def valeur_attendue_intervention(
    proba_churn: _NumType,
    mrr_mensuel: _NumType,
    taux_succes: float | None = None,
    cout: float | None = None,
) -> _NumType:
    """Gain espéré en euros d'un geste de rétention sur un compte, net de son coût.

    .. code-block:: text

        valeur_attendue = valeur_à_risque × taux_succès − coût_intervention
                        = P(churn) × MRR × horizon × marge × taux_succès − coût

    C'est le critère de priorisation sous contrainte de capacité (§12.6) : contacter en
    priorité les comptes où un geste rapporte le plus en espérance, et non ceux dont la
    probabilité de churn est la plus haute (souvent de petits comptes). Suppose des
    probabilités **calibrées** (`models.calibration`). Un MRR absent donne une valeur absente.
    """
    if taux_succes is None:
        taux_succes = float(config.HYPOTHESES_ECONOMIQUES["taux_succes_retention"])
    if cout is None:
        cout = cout_intervention()
    return valeur_a_risque(proba_churn, mrr_mensuel) * taux_succes - cout


def selection_sous_capacite(valeur_attendue: np.ndarray, capacite: int | None = None) -> np.ndarray:
    """Indices des comptes à contacter : les ``capacite`` meilleures valeurs attendues positives.

    Triés par valeur attendue décroissante. Un compte à valeur attendue négative ou nulle
    n'est jamais retenu, même s'il reste de la capacité : le geste coûterait plus qu'il ne
    rapporte en espérance. Les valeurs absentes (MRR inconnu) sont écartées.
    """
    if capacite is None:
        capacite = int(config.HYPOTHESES_ECONOMIQUES["capacite_gestes_mois"])
    valeurs = np.asarray(valeur_attendue, dtype=float)
    candidats = np.flatnonzero(np.nan_to_num(valeurs, nan=-np.inf) > 0)
    ordre = candidats[np.argsort(-valeurs[candidats], kind="stable")]
    return ordre[: max(capacite, 0)]


NiveauRisque = Literal["ALERTE_ROUGE", "SURVEILLANCE", "OK"]


# Artefact du seuil de vigilance (niveau 1 de la règle de décision), écrit par le notebook à
# partir de `models.economics.seuil_pour_recall` sur les prédictions out-of-fold
ARTEFACT_SEUIL_VIGILANCE: str = "seuil_vigilance.json"


def seuil_surveillance_par_defaut() -> float:
    """Seuil de vigilance appris (``reports/tables``), sinon repli sur ``config.SEUILS_RISQUE``.

    Seuil servi par l'API et le batch quand le modèle n'embarque pas le sien (métadonnées).
    """
    artefact = cache.charger(ARTEFACT_SEUIL_VIGILANCE)
    if isinstance(artefact, dict) and "seuil" in artefact:
        return float(artefact["seuil"])
    return float(config.SEUILS_RISQUE["surveillance"])


def niveau_risque(probabilite: float, seuil_surveillance: float | None = None) -> NiveauRisque:
    """Palier de risque d'un score — règle unique du projet.

    Le seuil de surveillance est, par ordre de priorité : ``seuil_surveillance`` (un modèle
    servi embarque son propre seuil, métadonnées du registre — API et batch inchangés), le
    seuil de vigilance de ``reports/tables/seuil_vigilance.json``, puis
    ``config.SEUILS_RISQUE["surveillance"]``. L'alerte rouge n'est jamais sous ce seuil.
    """
    surveillance = (
        seuil_surveillance_par_defaut() if seuil_surveillance is None else seuil_surveillance
    )
    if probabilite >= max(config.SEUILS_RISQUE["alerte_rouge"], surveillance):
        return "ALERTE_ROUGE"
    if probabilite >= surveillance:
        return "SURVEILLANCE"
    return "OK"
