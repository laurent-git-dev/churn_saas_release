"""Explication locale servie par l'API — SHAP exact sans la librairie `shap`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import shap
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_saas import config
from churn_saas.models.calibration import RegressionLogistiqueRecalibree
from churn_saas.models.explication_locale import (
    contributions_locales,
    facteurs_explicatifs,
    scorer_et_expliquer,
)


@pytest.fixture
def comptes() -> tuple[pd.DataFrame, pd.Series]:
    """Comptes synthétiques : l'inactivité et le plan Starter portent le risque."""
    rng = np.random.default_rng(config.RANDOM_SEED)
    n = 400
    X = pd.DataFrame(
        {
            "derniere_connexion_jours": rng.integers(0, 90, n).astype(float),
            "csat": rng.integers(1, 6, n).astype(float),
            "plan": rng.choice(["Starter", "Pro", "Business"], n),
        }
    )
    logit = 0.05 * X["derniere_connexion_jours"] - 0.4 * X["csat"] + (X["plan"] == "Starter")
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int))
    return X, y


def _pipeline(estimateur: object) -> Pipeline:
    pretraitement = ColumnTransformer(
        [
            ("numerique", StandardScaler(), ["derniere_connexion_jours", "csat"]),
            ("categorielle", OneHotEncoder(handle_unknown="ignore"), ["plan"]),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline([("pre", pretraitement), ("clf", estimateur)])


def _logit(p: np.ndarray) -> np.ndarray:
    return np.log(p / (1 - p))


def test_lineaire_identique_a_shap_linear_explainer(comptes) -> None:
    X, y = comptes
    modele = _pipeline(RegressionLogistiqueRecalibree(class_weight="balanced")).fit(X, y)
    contributions, noms = contributions_locales(modele, X)

    X_trans = modele[:-1].transform(X)
    # Référence = tout le jeu d'apprentissage (par défaut, shap en tire 100 lignes)
    reference = shap.maskers.Independent(X_trans, max_samples=len(X_trans))
    attendu = shap.LinearExplainer(modele[-1], reference).shap_values(X_trans)
    assert noms == list(modele[:-1].get_feature_names_out())
    np.testing.assert_allclose(contributions, attendu, atol=1e-10)


def test_lineaire_additivite_vers_le_logit(comptes) -> None:
    """Valeur de base + somme des contributions = logit servi (intercept corrigé compris)."""
    X, y = comptes
    modele = _pipeline(RegressionLogistiqueRecalibree(class_weight="balanced")).fit(X, y)
    contributions, _ = contributions_locales(modele, X)
    clf = modele[-1]
    base = float(clf.intercept_[0] + clf.coef_[0] @ clf.moyenne_entree_)
    np.testing.assert_allclose(
        base + contributions.sum(axis=1), _logit(modele.predict_proba(X)[:, 1]), atol=1e-8
    )


def test_lightgbm_additivite_vers_le_logit(comptes) -> None:
    X, y = comptes
    modele = _pipeline(
        LGBMClassifier(n_estimators=30, random_state=config.RANDOM_SEED, verbose=-1)
    ).fit(X, y)
    contributions, _ = contributions_locales(modele, X)
    brut = modele[-1].predict(modele[:-1].transform(X), pred_contrib=True)
    np.testing.assert_allclose(
        brut[:, -1] + contributions.sum(axis=1),
        _logit(modele.predict_proba(X)[:, 1]),
        atol=1e-6,
    )


def test_indicatrices_regroupees_et_libellees(comptes) -> None:
    """Les colonnes one-hot du plan forment un seul facteur, libellé avec la valeur du compte."""
    X, y = comptes
    modele = _pipeline(RegressionLogistiqueRecalibree(class_weight="balanced")).fit(X, y)
    facteurs = facteurs_explicatifs(modele, X)

    variables = {f["variable"] for liste in facteurs for f in liste}
    assert variables <= {"derniere_connexion_jours", "csat", "plan"}
    starter = next(
        liste
        for i, liste in enumerate(facteurs)
        if X.loc[i, "plan"] == "Starter" and any(f["variable"] == "plan" for f in liste)
    )
    assert next(f for f in starter if f["variable"] == "plan")["libelle"] == "Plan : Starter"


def test_facteurs_positifs_tries_et_bornes(comptes) -> None:
    X, y = comptes
    modele = _pipeline(RegressionLogistiqueRecalibree(class_weight="balanced")).fit(X, y)
    for liste in facteurs_explicatifs(modele, X, nb_facteurs=2):
        contributions = [f["contribution"] for f in liste]
        assert len(liste) <= 2
        assert all(c > 0 for c in contributions)
        assert contributions == sorted(contributions, reverse=True)


def test_compte_inactif_explique_par_l_inactivite(comptes) -> None:
    X, y = comptes
    modele = _pipeline(RegressionLogistiqueRecalibree(class_weight="balanced")).fit(X, y)
    inactif = pd.DataFrame({"derniere_connexion_jours": [89.0], "csat": [5.0], "plan": ["Pro"]})
    facteur = facteurs_explicatifs(modele, inactif)[0][0]
    assert facteur["variable"] == "derniere_connexion_jours"
    assert facteur["libelle"] == "Jours depuis la dernière connexion : 89"


def test_sans_reference_d_entrainement_aucun_facteur(comptes) -> None:
    """Un linéaire sans moyenne mémorisée n'est pas expliqué plutôt que mal expliqué."""
    X, y = comptes
    modele = _pipeline(LogisticRegression()).fit(X, y)
    assert contributions_locales(modele, X) is None
    assert facteurs_explicatifs(modele, X.head(3)) == [[], [], []]


def test_scorer_et_expliquer_identique_au_pipeline(comptes) -> None:
    """Un seul passage du prétraitement, mêmes probabilités que `Pipeline.predict_proba`."""
    X, y = comptes
    modele = _pipeline(RegressionLogistiqueRecalibree(class_weight="balanced")).fit(X, y)
    probas, facteurs = scorer_et_expliquer(modele, X)
    np.testing.assert_allclose(probas, modele.predict_proba(X))
    assert facteurs == facteurs_explicatifs(modele, X)
