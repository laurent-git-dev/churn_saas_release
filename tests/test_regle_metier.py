"""Tests de la baseline non-ML B1 (models.train.regle_metier)."""

from __future__ import annotations

import pandas as pd

from churn_saas.models.train import SEUIL_CSAT, regle_metier


def test_seuil_csat_dans_echelle_1_5() -> None:
    """Régression : un seuil ≥ 5 signalait tout client noté (échelle 1-5), baseline ≈ hasard."""
    assert 1 <= SEUIL_CSAT < 5


def test_csat_insatisfait_signale_satisfait_non() -> None:
    df = pd.DataFrame({"derniere_connexion_jours": [5] * 5, "csat": [1, 2, 3, 4, 5]})
    assert list(regle_metier(df)) == [1, 1, 0, 0, 0]


def test_inactivite_signalee_quel_que_soit_csat() -> None:
    df = pd.DataFrame({"derniere_connexion_jours": [30, 31], "csat": [5, 5]})
    assert list(regle_metier(df)) == [0, 1]


def test_csat_manquant_ne_declenche_pas() -> None:
    df = pd.DataFrame({"derniere_connexion_jours": [5], "csat": [None]})
    assert list(regle_metier(df)) == [0]
