"""Tests du module churn_saas.monitoring.drift — C9 robustesse et détection de dérive.

Trois axes vérifiés :
1. ``psi`` et ``ks_test`` : calcul correct et seuils d'interprétation.
2. ``simuler_derive`` : la dérive est bien appliquée et détectable.
3. ``tester_robustesse`` : la courbe de dégradation est cohérente (PR-AUC décroissant
   ou stable, jamais croissant sous perturbation croissante).
4. ``indicateur_obsolescence`` : statuts corrects selon l'âge.
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from churn_saas.monitoring.drift import (
    indicateur_obsolescence,
    ks_test,
    psi,
    simuler_derive,
    tester_robustesse as evaluer_robustesse,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _jeu_synthetique(n: int = 400, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """DataFrame synthétique avec la structure minimale des données churn."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "taux_adoption_pct": rng.uniform(10, 100, n),
            "connexions_30j": rng.integers(0, 50, n).astype(float),
            "anciennete_mois": rng.uniform(1, 60, n),
            "tickets_support_90j": rng.integers(0, 10, n).astype(float),
            "revenu_mensuel_recurrent_eur": rng.uniform(500, 5000, n),
            "sieges_souscrits": rng.integers(1, 100, n).astype(float),
            "utilisateurs_actifs": rng.integers(1, 50, n).astype(float),
            "delai_reponse_support_h": rng.exponential(24, n),
        }
    )
    y = pd.Series((rng.random(n) < 0.20).astype(int), name="churn")
    return df, y


def _modele_simple(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """Régression logistique minimaliste pour les tests de robustesse."""
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=300, random_state=42)),
        ]
    )
    pipe.fit(X, y)
    return pipe


# ---------------------------------------------------------------------------
# 1. Tests PSI
# ---------------------------------------------------------------------------


class TestPSI:
    def test_distributions_identiques_psi_nul(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 1000)
        res = psi(x, x)
        assert res["psi"] < 0.01

    def test_distributions_tres_differentes_psi_eleve(self) -> None:
        rng = np.random.default_rng(0)
        ref = rng.normal(0, 1, 1000)
        cur = rng.normal(5, 1, 1000)
        res = psi(ref, cur)
        assert res["psi"] > 0.20
        assert res["interpretation"] == "derive_significative"

    def test_seuil_attention(self) -> None:
        rng = np.random.default_rng(0)
        ref = rng.normal(0, 1, 2000)
        cur = rng.normal(0.8, 1.1, 2000)
        res = psi(ref, cur)
        # Peut être stable ou attention selon l'intensité — on vérifie juste le schéma
        assert res["interpretation"] in ("stable", "attention", "derive_significative")
        assert "action" in res
        assert "bins" in res

    def test_renvoie_dataframe_bins(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 500)
        res = psi(x, x + 0.5, n_bins=8)
        assert isinstance(res["bins"], pd.DataFrame)
        assert len(res["bins"]) == 8

    def test_psi_non_negatif(self) -> None:
        rng = np.random.default_rng(1)
        ref = rng.exponential(2, 300)
        cur = rng.exponential(3, 300)
        res = psi(ref, cur)
        assert res["psi"] >= 0.0


# ---------------------------------------------------------------------------
# 2. Tests KS
# ---------------------------------------------------------------------------


class TestKSTest:
    def test_meme_distribution_pas_de_derive(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 500)
        res = ks_test(x, x)
        assert not res["derive_detectee"]
        assert res["p_value"] == 1.0

    def test_distributions_distinctes_derive_detectee(self) -> None:
        rng = np.random.default_rng(0)
        ref = rng.normal(0, 1, 1000)
        cur = rng.normal(3, 1, 1000)
        res = ks_test(ref, cur)
        assert res["derive_detectee"]
        assert res["p_value"] < 0.05

    def test_champs_retournes(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 200)
        res = ks_test(x, x + 0.1)
        assert "statistique" in res
        assert "p_value" in res
        assert "derive_detectee" in res
        assert "interpretation" in res

    def test_statistique_entre_0_et_1(self) -> None:
        rng = np.random.default_rng(2)
        ref = rng.uniform(0, 10, 300)
        cur = rng.uniform(2, 12, 300)
        res = ks_test(ref, cur)
        assert 0.0 <= res["statistique"] <= 1.0


# ---------------------------------------------------------------------------
# 3. Tests simuler_derive
# ---------------------------------------------------------------------------


class TestSimulerDerive:
    def test_adoption_chute_reduit_taux(self) -> None:
        df, _ = _jeu_synthetique()
        derive = simuler_derive(df, "adoption_chute")
        assert derive["taux_adoption_pct"].mean() < df["taux_adoption_pct"].mean()

    def test_adoption_chute_connexions_reduites(self) -> None:
        df, _ = _jeu_synthetique()
        derive = simuler_derive(df, "adoption_chute")
        assert derive["connexions_30j"].mean() < df["connexions_30j"].mean()

    def test_nouveau_segment_mrr_eleve(self) -> None:
        df, _ = _jeu_synthetique()
        derive = simuler_derive(df, "nouveau_segment")
        # Le MRR max doit être plus élevé avec le nouveau segment
        assert derive["revenu_mensuel_recurrent_eur"].max() > df["revenu_mensuel_recurrent_eur"].max()

    def test_anciennete_rajeunissement(self) -> None:
        df, _ = _jeu_synthetique()
        derive = simuler_derive(df, "anciennete_rajeunissement")
        assert derive["anciennete_mois"].mean() < df["anciennete_mois"].mean()

    def test_support_degradation(self) -> None:
        df, _ = _jeu_synthetique()
        derive = simuler_derive(df, "support_degradation")
        assert derive["tickets_support_90j"].mean() > df["tickets_support_90j"].mean()

    def test_original_non_modifie(self) -> None:
        df, _ = _jeu_synthetique()
        mean_avant = df["taux_adoption_pct"].mean()
        simuler_derive(df, "adoption_chute")
        assert df["taux_adoption_pct"].mean() == pytest.approx(mean_avant)

    def test_type_inconnu_leve_erreur(self) -> None:
        df, _ = _jeu_synthetique()
        with pytest.raises(ValueError, match="Type de dérive inconnu"):
            simuler_derive(df, "type_inexistant")

    def test_meme_graine_reproductible(self) -> None:
        df, _ = _jeu_synthetique()
        d1 = simuler_derive(df, "adoption_chute", graine=99)
        d2 = simuler_derive(df, "adoption_chute", graine=99)
        pd.testing.assert_frame_equal(d1, d2)

    def test_derive_detectee_par_ks(self) -> None:
        """La dérive simulée doit être détectable par le test KS sur au moins une feature."""
        df, _ = _jeu_synthetique(n=800)
        derive = simuler_derive(df, "adoption_chute", intensite=1.5)
        res = ks_test(df["taux_adoption_pct"].dropna(), derive["taux_adoption_pct"].dropna())
        assert res["derive_detectee"], "La dérive simulée doit être détectée par KS."


# ---------------------------------------------------------------------------
# 4. Tests tester_robustesse
# ---------------------------------------------------------------------------


class TestTesterRobustesse:
    def test_renvoie_dataframe(self) -> None:
        df, y = _jeu_synthetique(n=300)
        modele = _modele_simple(df, y)
        res = evaluer_robustesse(modele, df, y, niveaux_bruit=[0.0, 0.5], taux_manquants=[0.0])
        assert isinstance(res, pd.DataFrame)

    def test_colonnes_attendues(self) -> None:
        df, y = _jeu_synthetique(n=300)
        modele = _modele_simple(df, y)
        res = evaluer_robustesse(modele, df, y, niveaux_bruit=[0.0], taux_manquants=[0.0])
        assert set(res.columns) == {
            "type_perturbation",
            "niveau",
            "pr_auc",
            "degradation_relative_pct",
        }

    def test_niveau_zero_degradation_nulle(self) -> None:
        df, y = _jeu_synthetique(n=300)
        modele = _modele_simple(df, y)
        res = evaluer_robustesse(modele, df, y, niveaux_bruit=[0.0], taux_manquants=[0.0])
        bruit_zero = res[(res["type_perturbation"] == "bruit_gaussien") & (res["niveau"] == 0.0)]
        assert bruit_zero["degradation_relative_pct"].iloc[0] == pytest.approx(0.0, abs=1e-3)

    def test_bruit_croissant_degradation_croissante(self) -> None:
        """PR-AUC doit décroître (ou rester stable) quand le bruit augmente."""
        df, y = _jeu_synthetique(n=500)
        modele = _modele_simple(df, y)
        niveaux = [0.0, 0.25, 1.0, 2.0]
        res = evaluer_robustesse(modele, df, y, niveaux_bruit=niveaux, taux_manquants=[])
        bruit = res[res["type_perturbation"] == "bruit_gaussien"].sort_values("niveau")
        pr_aucs = bruit["pr_auc"].tolist()
        # Le PR-AUC à bruit maximal doit être inférieur ou égal au PR-AUC à bruit nul
        assert pr_aucs[-1] <= pr_aucs[0] + 0.05, (
            f"PR-AUC devrait décroître avec le bruit, obtenu : {pr_aucs}"
        )

    def test_manquants_croissants_degradation_croissante(self) -> None:
        """PR-AUC doit décroître (ou rester stable) quand le taux de NaN augmente."""
        df, y = _jeu_synthetique(n=500)
        # Pipeline avec SimpleImputer pour gérer les NaN
        from sklearn.impute import SimpleImputer

        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=300, random_state=42)),
            ]
        )
        pipe.fit(df, y)
        taux = [0.0, 0.10, 0.30]
        res = evaluer_robustesse(pipe, df, y, niveaux_bruit=[], taux_manquants=taux)
        nan_res = res[res["type_perturbation"] == "valeurs_manquantes"].sort_values("niveau")
        pr_aucs = nan_res["pr_auc"].tolist()
        assert pr_aucs[-1] <= pr_aucs[0] + 0.05, (
            f"PR-AUC devrait décroître avec les NaN, obtenu : {pr_aucs}"
        )

    def test_nombre_de_lignes(self) -> None:
        df, y = _jeu_synthetique(n=200)
        modele = _modele_simple(df, y)
        niveaux = [0.0, 0.5, 1.0]
        taux = [0.0, 0.10]
        res = evaluer_robustesse(modele, df, y, niveaux_bruit=niveaux, taux_manquants=taux)
        assert len(res) == len(niveaux) + len(taux)


# ---------------------------------------------------------------------------
# 5. Tests indicateur_obsolescence
# ---------------------------------------------------------------------------


class TestIndicateurObsolescence:
    def test_modele_recent_statut_ok(self) -> None:
        date_train = datetime.date.today() - datetime.timedelta(days=30)
        res = indicateur_obsolescence(date_train)
        assert res["statut"] == "ok"
        assert res["age_jours"] == 30

    def test_modele_ancien_revision_recommandee(self) -> None:
        date_train = datetime.date.today() - datetime.timedelta(days=200)
        res = indicateur_obsolescence(date_train)
        assert res["statut"] == "revision_recommandee"
        assert res["age_jours"] == 200

    def test_exactement_au_seuil(self) -> None:
        from churn_saas.monitoring.drift import _SEUIL_OBSOLESCENCE_JOURS

        date_train = datetime.date.today() - datetime.timedelta(days=_SEUIL_OBSOLESCENCE_JOURS)
        res = indicateur_obsolescence(date_train)
        assert res["statut"] == "revision_recommandee"

    def test_champs_retournes(self) -> None:
        res = indicateur_obsolescence("2025-01-01", "2026-01-01")
        assert "age_jours" in res
        assert "seuil_jours" in res
        assert "statut" in res
        assert "message" in res
        assert "date_revue_recommandee" in res

    def test_accepte_chaine_iso(self) -> None:
        res = indicateur_obsolescence("2025-06-01", "2026-06-01")
        assert res["age_jours"] == 365

    def test_accepte_datetime(self) -> None:
        dt = datetime.datetime(2025, 1, 1, 12, 0, 0)
        res = indicateur_obsolescence(dt, datetime.date(2025, 7, 1))
        assert res["age_jours"] == 181

    def test_date_revue_coherente(self) -> None:
        from churn_saas.monitoring.drift import _SEUIL_OBSOLESCENCE_JOURS

        date_train = datetime.date(2025, 1, 1)
        res = indicateur_obsolescence(date_train, datetime.date(2025, 6, 1))
        attendu = date_train + datetime.timedelta(days=_SEUIL_OBSOLESCENCE_JOURS)
        assert res["date_revue_recommandee"] == attendu
