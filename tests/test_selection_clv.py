"""Tests de la règle de sélection du champion CLV — seule autorité pour §12.10 et §14.

Contexte : le champion `ridge` (R² maximal) prédisait une CLV négative pour une part des
comptes de test. La CLV est strictement positive (§8.1) : un modèle qui sort de ce domaine
est mal spécifié, quel que soit son score. Ces tests verrouillent :

1. le contrôle de domaine écarte un modèle qui prédit une valeur ≤ 0, même s'il a le
   meilleur R² ;
2. les cibles §8.2 (R² minimal, gain de MAE, alerte fuite) restent toutes obligatoires ;
3. aucun champion n'est désigné si aucun modèle n'est conforme.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from churn_saas import config
from churn_saas.models import regression as reg

_MAE_BASELINE = 100.0


def _resultat(r2: float, mae: float, y_pred: list[float]) -> dict[str, Any]:
    return {
        "y_pred": np.asarray(y_pred),
        "metriques": {"rmse": 1.0, "mae": mae, "r2": r2},
    }


def _resultats(**modeles: dict[str, Any]) -> dict[str, dict[str, Any]]:
    baseline = _resultat(-0.1, _MAE_BASELINE, [10.0, 10.0])
    return {"baseline": baseline, **modeles}


def test_prediction_negative_ecarte_le_meilleur_r2() -> None:
    resultats = _resultats(
        lineaire=_resultat(0.80, 50.0, [-5.0, 20.0]),
        arbre=_resultat(0.70, 40.0, [5.0, 20.0]),
    )
    tableau, champion = reg.selectionner_champion_clv(resultats)

    assert champion == "arbre"
    assert not tableau.loc["lineaire", "ok_domaine"]
    assert tableau.loc["lineaire", "n_pred_non_positives"] == 1
    assert "baseline" not in tableau.index


def test_prediction_nulle_hors_domaine() -> None:
    resultats = _resultats(modele=_resultat(0.80, 50.0, [0.0, 20.0]))
    tableau, champion = reg.selectionner_champion_clv(resultats)

    assert champion is None
    assert not tableau.loc["modele", "conforme"]


def test_cibles_performance_toujours_exigees() -> None:
    r2_min = config.CIBLES_PERFORMANCE["clv_r2_min"]
    alerte = config.CIBLES_PERFORMANCE["clv_r2_alerte_fuite"]
    mae_insuffisante = _MAE_BASELINE * (1 - config.CIBLES_PERFORMANCE["clv_gain_mae_min"]) + 1
    resultats = _resultats(
        r2_faible=_resultat(r2_min - 0.01, 10.0, [5.0]),
        mae_faible=_resultat(0.80, mae_insuffisante, [5.0]),
        fuite=_resultat(min(alerte + 0.001, 1.0), 10.0, [5.0]),
    )
    tableau, champion = reg.selectionner_champion_clv(resultats)

    assert champion is None
    assert not tableau.loc["r2_faible", "ok_r2"]
    assert not tableau.loc["mae_faible", "ok_mae"]
    assert not tableau.loc["fuite", "ok_fuite"]
