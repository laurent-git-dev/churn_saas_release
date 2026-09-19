"""Tests pour churn_saas.models.explain — importances, SHAP et fiche compte."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from churn_saas.models.explain import (
    _action_preventive,
    fiche_compte,
    importance_drop_column,
    importance_impurete,
    importance_permutation,
    verdict_leurres,
)

# ---------------------------------------------------------------------------
# Jeux de données synthétiques
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def donnees_simples() -> tuple[pd.DataFrame, pd.Series]:
    """Dataset binaire simple : 3 features utiles, 1 leurre."""
    rng = np.random.default_rng(42)
    n = 300
    # Feature 1 et 2 : signal réel
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    # Feature 3 : bruit aléatoire (leurre)
    x3 = rng.normal(0, 1, n)
    # Feature 4 : aussi bruit (second leurre)
    x4 = rng.normal(0, 1, n)

    logit = 0.8 * x1 - 0.6 * x2
    proba = 1 / (1 + np.exp(-logit))
    y_arr = rng.binomial(1, proba).astype(int)

    X = pd.DataFrame(
        {"feature_signal_1": x1, "feature_signal_2": x2, "leurre_A": x3, "leurre_B": x4}
    )
    y = pd.Series(y_arr, name="churn")
    return X, y


@pytest.fixture(scope="module")
def modele_rf_entraine(donnees_simples: tuple[pd.DataFrame, pd.Series]) -> RandomForestClassifier:
    X, y = donnees_simples
    clf = RandomForestClassifier(n_estimators=30, random_state=42)
    clf.fit(X, y)
    return clf


@pytest.fixture(scope="module")
def pipeline_entraine(donnees_simples: tuple[pd.DataFrame, pd.Series]) -> Pipeline:
    X, y = donnees_simples
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=20, random_state=42)),
        ]
    )
    pipe.fit(X, y)
    return pipe


@pytest.fixture(scope="module")
def modele_lr_entraine(donnees_simples: tuple[pd.DataFrame, pd.Series]) -> LogisticRegression:
    X, y = donnees_simples
    clf = LogisticRegression(random_state=42, max_iter=500)
    clf.fit(X, y)
    return clf


# ---------------------------------------------------------------------------
# importance_impurete
# ---------------------------------------------------------------------------


class TestImportanceImpurete:
    def test_retourne_dataframe(
        self,
        modele_rf_entraine: RandomForestClassifier,
        donnees_simples: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, _ = donnees_simples
        df = importance_impurete(modele_rf_entraine)
        assert isinstance(df, pd.DataFrame)
        assert "feature" in df.columns
        assert "importance" in df.columns
        assert "rang" in df.columns

    def test_trie_par_importance_decroissante(
        self, modele_rf_entraine: RandomForestClassifier
    ) -> None:
        df = importance_impurete(modele_rf_entraine)
        assert df["importance"].is_monotonic_decreasing

    def test_somme_importances_proche_de_un(
        self, modele_rf_entraine: RandomForestClassifier
    ) -> None:
        df = importance_impurete(modele_rf_entraine)
        assert abs(df["importance"].sum() - 1.0) < 1e-6

    def test_mise_en_garde_dans_attrs(self, modele_rf_entraine: RandomForestClassifier) -> None:
        df = importance_impurete(modele_rf_entraine)
        assert "mise_en_garde" in df.attrs
        assert "cardinalité" in df.attrs["mise_en_garde"]

    def test_leve_si_pas_de_feature_importances(
        self, modele_lr_entraine: LogisticRegression
    ) -> None:
        with pytest.raises(AttributeError, match="feature_importances_"):
            importance_impurete(modele_lr_entraine)

    def test_fonctionne_avec_pipeline(self, pipeline_entraine: Pipeline) -> None:
        df = importance_impurete(pipeline_entraine)
        assert len(df) > 0

    def test_rang_commence_a_un(self, modele_rf_entraine: RandomForestClassifier) -> None:
        df = importance_impurete(modele_rf_entraine)
        assert df["rang"].iloc[0] == 1


# ---------------------------------------------------------------------------
# importance_permutation
# ---------------------------------------------------------------------------


class TestImportancePermutation:
    def test_retourne_dataframe(
        self,
        modele_rf_entraine: RandomForestClassifier,
        donnees_simples: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = donnees_simples
        df = importance_permutation(modele_rf_entraine, X, y, n_repeats=5, n_jobs=1)
        assert isinstance(df, pd.DataFrame)
        assert {
            "feature",
            "importance_moyenne",
            "std",
            "ic95_bas",
            "ic95_haut",
            "significatif",
        }.issubset(df.columns)

    def test_nombre_features_correct(
        self,
        modele_rf_entraine: RandomForestClassifier,
        donnees_simples: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = donnees_simples
        df = importance_permutation(modele_rf_entraine, X, y, n_repeats=5, n_jobs=1)
        assert len(df) == X.shape[1]

    def test_ic95_coherent(
        self,
        modele_rf_entraine: RandomForestClassifier,
        donnees_simples: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = donnees_simples
        df = importance_permutation(modele_rf_entraine, X, y, n_repeats=5, n_jobs=1)
        assert (df["ic95_bas"] <= df["ic95_haut"]).all()

    def test_signaux_significatifs(
        self,
        modele_rf_entraine: RandomForestClassifier,
        donnees_simples: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = donnees_simples
        df = importance_permutation(modele_rf_entraine, X, y, n_repeats=20, n_jobs=1)
        df_idx = df.set_index("feature")
        # Les features signal doivent être significatives
        assert (
            df_idx.loc["feature_signal_1", "significatif"]
            or df_idx.loc["feature_signal_2", "significatif"]
        )

    def test_trie_par_importance_decroissante(
        self,
        modele_rf_entraine: RandomForestClassifier,
        donnees_simples: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = donnees_simples
        df = importance_permutation(modele_rf_entraine, X, y, n_repeats=5, n_jobs=1)
        assert df["importance_moyenne"].is_monotonic_decreasing


# ---------------------------------------------------------------------------
# importance_drop_column
# ---------------------------------------------------------------------------


class TestImportanceDropColumn:
    def test_retourne_dataframe(
        self,
        pipeline_entraine: Pipeline,
        donnees_simples: tuple[pd.DataFrame, pd.Series],
        tmp_path: pytest.FixtureRequest,
    ) -> None:
        X, y = donnees_simples
        df = importance_drop_column(pipeline_entraine, X, y, colonnes=["leurre_A", "leurre_B"])
        assert isinstance(df, pd.DataFrame)
        assert {"pr_auc_sans", "pr_auc_baseline", "delta_pr_auc", "negligeable"}.issubset(
            df.columns
        )

    def test_index_par_colonne(
        self,
        pipeline_entraine: Pipeline,
        donnees_simples: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = donnees_simples
        df = importance_drop_column(pipeline_entraine, X, y, colonnes=["leurre_A"])
        assert "leurre_A" in df.index

    def test_leurres_negligeables(
        self,
        pipeline_entraine: Pipeline,
        donnees_simples: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = donnees_simples
        df = importance_drop_column(
            pipeline_entraine, X, y, colonnes=["leurre_A", "leurre_B"], forcer=True
        )
        # Pour les vrais leurres (bruit pur), la perte doit être négligeable
        assert df.loc["leurre_A", "negligeable"] or df.loc["leurre_B", "negligeable"]

    def test_aucune_colonne_retourne_df_vide(
        self,
        pipeline_entraine: Pipeline,
        donnees_simples: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = donnees_simples
        df = importance_drop_column(pipeline_entraine, X, y, colonnes=["colonne_inexistante"])
        assert len(df) == 0

    def test_delta_non_negatif_pour_signaux(
        self,
        pipeline_entraine: Pipeline,
        donnees_simples: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = donnees_simples
        df = importance_drop_column(
            pipeline_entraine, X, y, colonnes=["feature_signal_1"], forcer=True
        )
        # Retirer un signal doit augmenter la perte (Δ ≥ 0 en général)
        assert not df.loc["feature_signal_1", "negligeable"]


# ---------------------------------------------------------------------------
# fiche_compte
# ---------------------------------------------------------------------------


class TestFicheCompte:
    def test_retourne_cles_attendues(
        self,
        modele_rf_entraine: RandomForestClassifier,
        donnees_simples: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = donnees_simples
        # SHAP synthétique (pas de vraie décomposition)
        rng = np.random.default_rng(42)
        sv = rng.normal(0, 0.1, size=(len(X), X.shape[1]))
        client = X.iloc[0]
        fiche = fiche_compte(client, modele_rf_entraine, X, sv)
        attendues = {
            "proba_churn",
            "niveau_risque",
            "valeur_risque_eur",
            "facteurs_churn",
            "facteurs_protection",
            "action_preventive",
        }
        assert attendues.issubset(fiche.keys())

    def test_proba_dans_zero_un(
        self,
        modele_rf_entraine: RandomForestClassifier,
        donnees_simples: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, _ = donnees_simples
        sv = np.zeros((len(X), X.shape[1]))
        client = X.iloc[5]
        fiche = fiche_compte(client, modele_rf_entraine, X, sv)
        assert 0.0 <= fiche["proba_churn"] <= 1.0

    def test_niveau_risque_valide(
        self,
        modele_rf_entraine: RandomForestClassifier,
        donnees_simples: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, _ = donnees_simples
        sv = np.zeros((len(X), X.shape[1]))
        client = X.iloc[0]
        fiche = fiche_compte(client, modele_rf_entraine, X, sv)
        assert fiche["niveau_risque"] in {"FAIBLE", "MODÉRÉ", "ÉLEVÉ"}

    def test_leve_si_client_absent(
        self,
        modele_rf_entraine: RandomForestClassifier,
        donnees_simples: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, _ = donnees_simples
        sv = np.zeros((len(X), X.shape[1]))
        client = pd.Series({"feature_signal_1": 0.5}, name=99999)
        with pytest.raises(KeyError):
            fiche_compte(client, modele_rf_entraine, X, sv)

    def test_facteurs_churn_positifs(
        self,
        modele_rf_entraine: RandomForestClassifier,
        donnees_simples: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, _ = donnees_simples
        # SHAP avec valeurs positives pour les 2 premières features
        sv = np.zeros((len(X), X.shape[1]))
        sv[:, 0] = 0.5
        sv[:, 1] = 0.3
        client = X.iloc[0]
        fiche = fiche_compte(client, modele_rf_entraine, X, sv)
        for f in fiche["facteurs_churn"]:
            assert f["contribution_shap"] > 0

    def test_facteurs_protection_negatifs(
        self,
        modele_rf_entraine: RandomForestClassifier,
        donnees_simples: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, _ = donnees_simples
        sv = np.zeros((len(X), X.shape[1]))
        sv[:, 2] = -0.4
        sv[:, 3] = -0.2
        client = X.iloc[0]
        fiche = fiche_compte(client, modele_rf_entraine, X, sv)
        for f in fiche["facteurs_protection"]:
            assert f["contribution_shap"] < 0


# ---------------------------------------------------------------------------
# _action_preventive
# ---------------------------------------------------------------------------


class TestActionPreventive:
    def test_reconnait_csat(self) -> None:
        action = _action_preventive("csat_score")
        assert (
            "satisfaction" in action.lower()
            or "csat" in action.lower()
            or "satisfaction" in action.lower()
        )

    def test_reconnait_connexion(self) -> None:
        action = _action_preventive("derniere_connexion_jours")
        assert (
            "usage" in action.lower() or "connexion" in action.lower() or "appel" in action.lower()
        )

    def test_fallback_feature_inconnue(self) -> None:
        action = _action_preventive("feature_xyz_inconnue")
        assert "csm" in action.lower() or "feature_xyz_inconnue" in action.lower()


# ---------------------------------------------------------------------------
# verdict_leurres
# ---------------------------------------------------------------------------


class TestVerdictLeurres:
    @pytest.fixture
    def donnees_verdict(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Tableau de criblage et importances synthétiques pour 4 colonnes."""
        colonnes = ["leurre_pur", "redondant", "utile", "ambigu"]

        criblage = pd.DataFrame(
            {
                "type": ["catégorielle", "numérique", "numérique", "numérique"],
                "association_marginale": [0.05, 0.10, 0.55, 0.08],
                "p_value_BH": [0.30, 0.20, 0.001, 0.25],
                "significatif_BH": [False, False, True, False],
                "max_redondance": [0.10, 0.85, 0.20, 0.15],
                "variable_jumelle": ["", "autre_feature", "", ""],
                "permutation_imp": [float("nan")] * 4,
                "drop_column_imp": [float("nan")] * 4,
                "verdict_provisoire": ["leurre probable"] * 4,
            },
            index=pd.Index(colonnes, name="colonne"),
        )

        perm = pd.DataFrame(
            {
                "feature": colonnes,
                "importance_moyenne": [-0.001, -0.002, 0.050, -0.001],
                "std": [0.002, 0.002, 0.010, 0.002],
                "ic95_bas": [-0.005, -0.006, 0.030, -0.005],
                "ic95_haut": [-0.001, -0.001, 0.070, 0.003],  # ambigu: IC > 0 → pas nulle
                "significatif": [False, False, True, True],
            }
        )

        drop = pd.DataFrame(
            {
                "pr_auc_sans": [0.720, 0.721, 0.680, 0.710],
                "pr_auc_baseline": [0.721, 0.721, 0.721, 0.721],
                "delta_pr_auc": [0.001, 0.000, 0.041, 0.011],
                "negligeable": [True, True, False, False],
            },
            index=pd.Index(colonnes, name="colonne"),
        )

        return criblage, perm, drop

    def test_retourne_dataframe(
        self, donnees_verdict: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    ) -> None:
        criblage, perm, drop = donnees_verdict
        df = verdict_leurres(criblage, perm, drop)
        assert isinstance(df, pd.DataFrame)

    def test_colonnes_attendues(
        self, donnees_verdict: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    ) -> None:
        criblage, perm, drop = donnees_verdict
        df = verdict_leurres(criblage, perm, drop)
        attendues = {
            "association_significative",
            "permutation_nulle",
            "drop_negligeable",
            "verdict",
            "raisonnement",
        }
        assert attendues.issubset(df.columns)

    def test_leurre_pur_confirme(
        self, donnees_verdict: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    ) -> None:
        criblage, perm, drop = donnees_verdict
        df = verdict_leurres(criblage, perm, drop)
        assert "LEURRE CONFIRMÉ" in df.loc["leurre_pur", "verdict"]

    def test_redondant_detecte(
        self, donnees_verdict: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    ) -> None:
        criblage, perm, drop = donnees_verdict
        df = verdict_leurres(criblage, perm, drop)
        assert "REDONDANT" in df.loc["redondant", "verdict"]

    def test_utile_conserve(
        self, donnees_verdict: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    ) -> None:
        criblage, perm, drop = donnees_verdict
        df = verdict_leurres(criblage, perm, drop)
        assert "UTILE" in df.loc["utile", "verdict"]

    def test_raisonnement_non_vide(
        self, donnees_verdict: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    ) -> None:
        criblage, perm, drop = donnees_verdict
        df = verdict_leurres(criblage, perm, drop)
        assert df["raisonnement"].str.len().gt(10).all()

    def test_index_par_colonne(
        self, donnees_verdict: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    ) -> None:
        criblage, perm, drop = donnees_verdict
        df = verdict_leurres(criblage, perm, drop)
        assert list(df.index) == ["leurre_pur", "redondant", "utile", "ambigu"]
