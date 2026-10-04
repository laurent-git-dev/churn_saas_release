"""Tests de la régression logistique recalibrée (models.calibration)."""

from __future__ import annotations

import pickle

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from churn_saas import config
from churn_saas.models.calibration import RegressionLogistiqueRecalibree


@pytest.fixture(scope="module")
def donnees() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(config.RANDOM_SEED)
    X = rng.normal(size=(4_000, 3))
    logit = X @ np.array([1.0, -1.0, 0.5]) - 1.2
    y = (rng.random(4_000) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y


def test_probabilite_moyenne_egale_prevalence(donnees: tuple) -> None:
    # La pondération seule surestime le risque ; la correction ramène la moyenne à la prévalence
    X, y = donnees
    brut = LogisticRegression(class_weight="balanced").fit(X, y).predict_proba(X)[:, 1]
    corrige = RegressionLogistiqueRecalibree(class_weight="balanced").fit(X, y)
    assert brut.mean() > y.mean() + 0.05
    assert corrige.predict_proba(X)[:, 1].mean() == pytest.approx(y.mean(), abs=0.01)


def test_seul_l_intercept_change(donnees: tuple) -> None:
    X, y = donnees
    brut = LogisticRegression(class_weight="balanced").fit(X, y)
    corrige = RegressionLogistiqueRecalibree(class_weight="balanced").fit(X, y)
    np.testing.assert_allclose(corrige.coef_, brut.coef_)
    pi = y.mean()
    assert corrige.correction_intercept_ == pytest.approx(np.log((1 - pi) / pi))
    np.testing.assert_allclose(corrige.intercept_, brut.intercept_ - np.log((1 - pi) / pi))


def test_sans_ponderation_aucune_correction(donnees: tuple) -> None:
    X, y = donnees
    modele = RegressionLogistiqueRecalibree().fit(X, y)
    assert modele.correction_intercept_ == 0.0


def test_clone_et_serialisation(donnees: tuple) -> None:
    X, y = donnees
    modele = RegressionLogistiqueRecalibree(class_weight="balanced", C=0.5)
    copie = clone(modele)
    assert isinstance(copie, RegressionLogistiqueRecalibree) and copie.C == 0.5
    relu = pickle.loads(pickle.dumps(modele.fit(X, y)))
    np.testing.assert_allclose(relu.predict_proba(X), modele.predict_proba(X))


def test_multiclasse_refusee() -> None:
    X = np.random.default_rng(0).normal(size=(30, 2))
    with pytest.raises(ValueError, match="binaire"):
        RegressionLogistiqueRecalibree().fit(X, np.arange(30) % 3)
