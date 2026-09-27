"""Formule unique de la valeur à risque — autorité partagée API / batch / notebook.

Ce module est **la seule** implémentation de la valeur à risque du projet. Il existe
parce que la formule avait été recopiée à trois endroits (API, flow de scoring batch,
module de régression) et que les copies avaient divergé : deux d'entre elles omettaient
la marge brute, surestimant l'exposition de +38,9 % (= 1 / 0,72) par rapport à la formule
défendue en §12.11 et dans `docs/POINTS_DE_VIGILANCE.md` (point n°3).

Dépendances volontairement minimales (`config` + `loguru`) : le module est importé par
le conteneur d'inférence, qui n'a rien à faire de matplotlib — d'où le choix de ne pas
héberger cette fonction dans `models.regression` ni dans `models.economics`.

Politique de manquance : ce module calcule, il ne décide pas. Un MRR absent produit une
valeur absente (`None` / `NaN`) ; c'est à l'appelant de dire comment il l'expose —
`null` + motif pour l'API, colonne `mrr_disponible` pour le batch.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from churn_saas import config

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

    ⚠️  Ne multiplie **pas** P(churn) par ``valeur_vie_client_eur`` (CLV historique) —
    ce serait compter deux fois le même euro. La CLV est historique (r > 0,70 en §6.7) ;
    elle contient de la valeur passée déjà encaissée et irrécouvrable.

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
