"""Tests pour churn_saas.models.evaluate — sorties obligatoires de classification."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # backend non-interactif — doit précéder tout import matplotlib

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from sklearn.datasets import make_classification

from churn_saas.models import evaluate

# ---------------------------------------------------------------------------
# Jeu de données synthétique partagé
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def donnees_test() -> tuple[pd.Series, np.ndarray, pd.DataFrame]:
    """Jeu synthétique minimal : 200 observations, ~20 % de churn, modèle imparfait."""
    rng = np.random.default_rng(42)
    n = 200
    X_arr, y_arr = make_classification(
        n_samples=n,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        weights=[0.8, 0.2],
        random_state=42,
    )
    # Probabilités simulant un modèle imparfait (bruitées autour du vrai label)
    proba = np.clip(y_arr * 0.6 + rng.uniform(-0.3, 0.3, n), 0.01, 0.99)
    df = pd.DataFrame(X_arr, columns=[f"feat_{i}" for i in range(5)])
    df["categorie"] = rng.choice(["A", "B", "C"], n)
    return pd.Series(y_arr, name="churn"), proba, df


# ---------------------------------------------------------------------------
# courbe_roc
# ---------------------------------------------------------------------------


class TestCourbeROC:
    def test_retourne_figure_et_auc(self, donnees_test: tuple) -> None:
        y, proba, _ = donnees_test
        fig, auc = evaluate.courbe_roc(y, proba)
        assert isinstance(fig, Figure)
        assert 0.5 <= auc <= 1.0

    def test_accepte_numpy_array(self, donnees_test: tuple) -> None:
        y, proba, _ = donnees_test
        fig, auc = evaluate.courbe_roc(np.asarray(y), proba)
        assert isinstance(fig, Figure)
        assert isinstance(auc, float)

    def test_auc_aleatoire_proche_de_0_5(self) -> None:
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 400)
        proba = rng.uniform(0, 1, 400)
        _, auc = evaluate.courbe_roc(y, proba)
        assert abs(auc - 0.5) < 0.15

    def test_auc_parfaite_vaut_1(self) -> None:
        y = np.array([0, 0, 1, 1])
        proba = np.array([0.1, 0.2, 0.8, 0.9])
        _, auc = evaluate.courbe_roc(y, proba)
        assert auc == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# courbe_precision_rappel
# ---------------------------------------------------------------------------


class TestCourbePrecisionRappel:
    def test_retourne_figure_et_pr_auc(self, donnees_test: tuple) -> None:
        y, proba, _ = donnees_test
        fig, pr_auc = evaluate.courbe_precision_rappel(y, proba)
        assert isinstance(fig, Figure)
        assert 0.0 < pr_auc <= 1.0

    def test_pr_auc_superieure_a_la_prevalence(self, donnees_test: tuple) -> None:
        y, proba, _ = donnees_test
        prevalence = float(np.asarray(y).mean())
        _, pr_auc = evaluate.courbe_precision_rappel(y, proba)
        # Un modèle informatif doit dépasser la ligne de base aléatoire
        assert pr_auc >= prevalence

    def test_nom_modele_personnalisable(self, donnees_test: tuple) -> None:
        y, proba, _ = donnees_test
        fig, _ = evaluate.courbe_precision_rappel(y, proba, nom_modele="Forêt aléatoire")
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# matrice_confusion
# ---------------------------------------------------------------------------


class TestMatriceConfusion:
    def test_retourne_figure_et_matrice_2x2(self, donnees_test: tuple) -> None:
        y, proba, _ = donnees_test
        y_pred = (proba >= 0.5).astype(int)
        fig, cm = evaluate.matrice_confusion(y, y_pred, seuil=0.5)
        assert isinstance(fig, Figure)
        assert cm.shape == (2, 2)

    def test_somme_egale_n_observations(self, donnees_test: tuple) -> None:
        y, proba, _ = donnees_test
        y_pred = (proba >= 0.5).astype(int)
        _, cm = evaluate.matrice_confusion(y, y_pred, seuil=0.5)
        assert cm.sum() == len(y)

    def test_valeurs_non_negatives(self, donnees_test: tuple) -> None:
        y, proba, _ = donnees_test
        y_pred = (proba >= 0.3).astype(int)
        _, cm = evaluate.matrice_confusion(y, y_pred, seuil=0.3)
        assert (cm >= 0).all()

    def test_seuil_strict_predit_tout_positif(self) -> None:
        y = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 1, 1])
        _, cm = evaluate.matrice_confusion(y, y_pred, seuil=0.0)
        # Ligne 0 : [VN=0, FP=2] ; ligne 1 : [FN=0, VP=2]
        assert cm[0, 0] == 0 and cm[0, 1] == 2
        assert cm[1, 0] == 0 and cm[1, 1] == 2


# ---------------------------------------------------------------------------
# courbe_calibration
# ---------------------------------------------------------------------------


class TestCourbeCalibration:
    def test_retourne_figure_et_brier(self, donnees_test: tuple) -> None:
        y, proba, _ = donnees_test
        fig, brier = evaluate.courbe_calibration(y, proba)
        assert isinstance(fig, Figure)
        assert 0.0 <= brier <= 0.25

    def test_brier_parfait_vaut_zero(self) -> None:
        y = np.array([1, 1, 0, 0] * 50)
        proba = np.array([1.0, 1.0, 0.0, 0.0] * 50)
        _, brier = evaluate.courbe_calibration(y, proba)
        assert brier == pytest.approx(0.0, abs=1e-9)

    def test_brier_aleatoire_proche_de_0_25(self) -> None:
        y = np.tile([0, 1], 500)
        proba = np.full(1000, 0.5)
        _, brier = evaluate.courbe_calibration(y, proba)
        assert abs(brier - 0.25) < 0.01

    def test_n_bins_respecte(self, donnees_test: tuple) -> None:
        y, proba, _ = donnees_test
        fig, brier = evaluate.courbe_calibration(y, proba, n_bins=5)
        assert isinstance(brier, float)


# ---------------------------------------------------------------------------
# tableau_metriques
# ---------------------------------------------------------------------------


class TestTableauMetriques:
    def test_retourne_dataframe_avec_colonnes_attendues(self, donnees_test: tuple) -> None:
        y, proba, _ = donnees_test
        df = evaluate.tableau_metriques(y, proba, seuil=0.5)
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == {"métrique", "valeur", "description"}

    def test_contient_pr_auc_et_roc_auc(self, donnees_test: tuple) -> None:
        y, proba, _ = donnees_test
        df = evaluate.tableau_metriques(y, proba, seuil=0.5)
        metriques = df["métrique"].tolist()
        assert any("PR-AUC" in m for m in metriques)
        assert any("ROC-AUC" in m for m in metriques)

    def test_sept_metriques_calculees(self, donnees_test: tuple) -> None:
        y, proba, _ = donnees_test
        df = evaluate.tableau_metriques(y, proba, seuil=0.5)
        assert len(df) == 7

    def test_valeurs_dans_intervalle_0_1(self, donnees_test: tuple) -> None:
        y, proba, _ = donnees_test
        df = evaluate.tableau_metriques(y, proba, seuil=0.5)
        assert df["valeur"].between(0.0, 1.0).all()

    def test_seuil_influence_precision_rappel(self, donnees_test: tuple) -> None:
        y, proba, _ = donnees_test
        df_bas = evaluate.tableau_metriques(y, proba, seuil=0.2)
        df_haut = evaluate.tableau_metriques(y, proba, seuil=0.8)
        # Seuil bas → rappel élevé ; seuil haut → rappel faible
        rappel_bas = df_bas.loc[df_bas["métrique"].str.contains("Rappel"), "valeur"].item()
        rappel_haut = df_haut.loc[df_haut["métrique"].str.contains("Rappel"), "valeur"].item()
        assert rappel_bas > rappel_haut


# ---------------------------------------------------------------------------
# analyse_erreurs
# ---------------------------------------------------------------------------


class TestAnalyseErreurs:
    def test_retourne_dataframe_avec_quatre_segments(self, donnees_test: tuple) -> None:
        y, proba, df = donnees_test
        résumé = evaluate.analyse_erreurs(df, y, proba, seuil=0.5)
        assert isinstance(résumé, pd.DataFrame)
        assert len(résumé) == 4

    def test_index_contient_fn_fp_vp_vn(self, donnees_test: tuple) -> None:
        y, proba, df = donnees_test
        résumé = evaluate.analyse_erreurs(df, y, proba, seuil=0.5)
        index = list(résumé.index)
        assert any("FN" in s for s in index)
        assert any("FP" in s for s in index)
        assert any("VP" in s for s in index)
        assert any("VN" in s for s in index)

    def test_n_observations_somme_a_n_total(self, donnees_test: tuple) -> None:
        y, proba, df = donnees_test
        résumé = evaluate.analyse_erreurs(df, y, proba, seuil=0.5)
        assert résumé["n_observations"].sum() == len(y)

    def test_colonnes_numeriques_presentes(self, donnees_test: tuple) -> None:
        y, proba, df = donnees_test
        résumé = evaluate.analyse_erreurs(df, y, proba, seuil=0.5)
        assert "feat_0" in résumé.columns

    def test_proba_moyenne_fn_inferieure_a_0_5(self, donnees_test: tuple) -> None:
        y, proba, df = donnees_test
        résumé = evaluate.analyse_erreurs(df, y, proba, seuil=0.5)
        # Les FN ont une proba < seuil par construction
        proba_fn = résumé.loc[résumé.index.str.contains("FN"), "proba_pred_moyenne"]
        if len(proba_fn) > 0 and not proba_fn.isna().item():
            assert float(proba_fn.item()) < 0.5

    def test_seuil_extreme_1_0_creer_uniquement_fn_vn(self) -> None:
        y = np.array([0, 0, 1, 1])
        proba = np.array([0.1, 0.2, 0.4, 0.6])
        df = pd.DataFrame({"val": [1.0, 2.0, 3.0, 4.0]})
        résumé = evaluate.analyse_erreurs(df, y, proba, seuil=0.99)
        # Tout prédit négatif → VP et FP ont n_observations=0
        n_fp = résumé.loc[résumé.index.str.contains("FP"), "n_observations"].item()
        n_vp = résumé.loc[résumé.index.str.contains("VP"), "n_observations"].item()
        assert n_fp == 0
        assert n_vp == 0
