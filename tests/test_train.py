"""Tests unitaires rapides de `churn_saas.models.train` (hors entraînement en cross-validation)."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from churn_saas import config
from churn_saas.models.train import journaliser_mlflow, mesurer_empreinte


def test_mesurer_empreinte_sans_parametre_codecarbon_deprecie() -> None:
    with warnings.catch_warnings(record=True) as captures:
        warnings.simplefilter("always")
        resultat, rapport = mesurer_empreinte(lambda: 42)
    messages = [str(w.message) for w in captures if issubclass(w.category, DeprecationWarning)]
    assert not any("save_to_" in m for m in messages)
    assert resultat == 42
    assert rapport["emissions_kg_co2"] >= 0.0
    assert (config.TABLES / "codecarbon_emissions.csv").exists()


def test_journaliser_mlflow_sans_bruit_info(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Les INFO de MLflow (détection uv, export des dépendances) sont tus, le niveau restauré."""
    monkeypatch.setattr(config, "MLRUNS", tmp_path)
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    rng = np.random.default_rng(config.RANDOM_SEED)
    modele = LogisticRegression().fit(rng.normal(size=(40, 2)), np.tile([0, 1], 20))
    niveau_avant = logging.getLogger("mlflow").level

    with caplog.at_level(logging.INFO):
        run_id = journaliser_mlflow("essai", modele, {"auc": 0.5}, {"C": 1.0})

    assert run_id
    bruit = [
        r for r in caplog.records if r.name.startswith("mlflow") and r.levelno < logging.WARNING
    ]
    assert not bruit, [r.getMessage() for r in bruit]
    assert logging.getLogger("mlflow").level == niveau_avant
