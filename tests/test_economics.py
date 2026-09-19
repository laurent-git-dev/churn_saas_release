"""Tests pour churn_saas.models.economics — justification économique du seuil."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # backend non-interactif — doit précéder tout import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from churn_saas import config
from churn_saas.models import economics


@pytest.fixture(autouse=True)
def fermer_figures() -> None:  # type: ignore[return]
    """Ferme toutes les figures matplotlib après chaque test pour éviter les fuites mémoire."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Jeu de données synthétique partagé
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def donnees_eco() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Portefeuille synthétique de 300 comptes avec signal clair et MRR variable.

    Probabilités corrélées avec y pour que le seuil optimal ≠ 0,5.
    MRR variable : certains comptes sont très rentables (forte pénalité FN).
    """
    rng = np.random.default_rng(42)
    n = 300
    n_churn = 60  # 20 % de churn

    # Churners ont des probas hautes, non-churners des probas basses
    proba_churn = rng.beta(5, 2, n_churn)  # distribution concentrée vers 1
    proba_ok = rng.beta(2, 6, n - n_churn)  # distribution concentrée vers 0
    proba = np.concatenate([proba_churn, proba_ok])

    y = np.array([1] * n_churn + [0] * (n - n_churn), dtype=float)

    # MRR hétérogène : quelques gros comptes (distribution log-normale)
    mrr = rng.lognormal(mean=7.0, sigma=0.8, size=n)  # median ≈ 1 100 €

    # Mélanger aléatoirement
    perm = rng.permutation(n)
    return y[perm], proba[perm], mrr[perm]


# ---------------------------------------------------------------------------
# matrice_couts
# ---------------------------------------------------------------------------


class TestMatriceCouts:
    def test_retourne_dict_avec_cles_attendues(self) -> None:
        mc = economics.matrice_couts()
        assert "cout_fp_eur" in mc
        assert "mult_mrr_fn" in mc
        assert "mult_mrr_tp" in mc
        assert "seuil_mrr_rentable_eur" in mc
        assert "hypotheses" in mc

    def test_cout_intervention_couvre_csm(self) -> None:
        mc = economics.matrice_couts()
        h = config.HYPOTHESES_ECONOMIQUES
        attendu = float(h["cout_horaire_csm_eur"]) * float(h["duree_geste_retention_h"])
        assert mc["cout_intervention_eur"] == pytest.approx(attendu)

    def test_mult_mrr_fn_coherent(self) -> None:
        mc = economics.matrice_couts()
        h = config.HYPOTHESES_ECONOMIQUES
        attendu = float(h["horizon_mois"]) * float(h["marge_brute_pct"])
        assert mc["mult_mrr_fn"] == pytest.approx(attendu)

    def test_mult_mrr_tp_inclut_taux_succes(self) -> None:
        mc = economics.matrice_couts()
        h = config.HYPOTHESES_ECONOMIQUES
        attendu = (
            float(h["horizon_mois"])
            * float(h["marge_brute_pct"])
            * float(h["taux_succes_retention"])
        )
        assert mc["mult_mrr_tp"] == pytest.approx(attendu)

    def test_seuil_rentabilite_positif(self) -> None:
        mc = economics.matrice_couts()
        assert mc["seuil_mrr_rentable_eur"] > 0

    def test_cout_tn_est_zero(self) -> None:
        mc = economics.matrice_couts()
        assert mc["cout_tn_eur"] == 0.0

    def test_hypotheses_contient_config(self) -> None:
        mc = economics.matrice_couts()
        for cle in config.HYPOTHESES_ECONOMIQUES:
            assert cle in mc["hypotheses"]


# ---------------------------------------------------------------------------
# gain_par_seuil
# ---------------------------------------------------------------------------


class TestGainParSeuil:
    def test_retourne_figure_dataframe_float(self, donnees_eco: tuple) -> None:
        y, proba, mrr = donnees_eco
        fig, courbe, seuil_opt = economics.gain_par_seuil(y, proba, mrr)
        assert isinstance(fig, Figure)
        assert isinstance(courbe, pd.DataFrame)
        assert isinstance(seuil_opt, float)

    def test_seuil_optimal_different_de_0_5(self, donnees_eco: tuple) -> None:
        """Exigence centrale : le seuil économique ≠ 0,5."""
        y, proba, mrr = donnees_eco
        _, _, seuil_opt = economics.gain_par_seuil(y, proba, mrr)
        assert abs(seuil_opt - 0.5) > 0.05, (
            f"Seuil optimal = {seuil_opt:.3f} trop proche de 0,5 — "
            "vérifier que le signal et les coûts sont asymétriques."
        )

    def test_seuil_dans_intervalle_0_1(self, donnees_eco: tuple) -> None:
        y, proba, mrr = donnees_eco
        _, _, seuil_opt = economics.gain_par_seuil(y, proba, mrr)
        assert 0.0 <= seuil_opt <= 1.0

    def test_courbe_contient_colonnes_attendues(self, donnees_eco: tuple) -> None:
        y, proba, mrr = donnees_eco
        _, courbe, _ = economics.gain_par_seuil(y, proba, mrr)
        assert set(courbe.columns) >= {"seuil", "gain_net_eur", "n_alertes", "precision", "rappel"}

    def test_gain_optimal_superieur_au_gain_a_0_5(self, donnees_eco: tuple) -> None:
        y, proba, mrr = donnees_eco
        _, courbe, seuil_opt = economics.gain_par_seuil(y, proba, mrr)
        gain_opt = courbe["gain_net_eur"].max()
        # Trouver le gain à 0,5 dans la courbe
        idx_05 = (courbe["seuil"] - 0.5).abs().idxmin()
        gain_a_05 = float(courbe.loc[idx_05, "gain_net_eur"])
        assert (
            gain_opt >= gain_a_05
        ), f"Gain optimal ({gain_opt:.0f} €) devrait être ≥ gain à 0,5 ({gain_a_05:.0f} €)."

    def test_n_alertes_decroit_avec_seuil(self, donnees_eco: tuple) -> None:
        y, proba, mrr = donnees_eco
        _, courbe, _ = economics.gain_par_seuil(y, proba, mrr, n_seuils=50)
        # Le nombre d'alertes doit être globalement décroissant quand le seuil monte
        alertes = courbe["n_alertes"].values
        assert alertes[0] >= alertes[-1], "Plus le seuil est haut, moins d'alertes déclenchées."

    def test_seuil_descend_quand_cout_fn_augmente(self) -> None:
        """Preuve de monotonie : FN plus coûteux → seuil optimal plus bas.

        Si le MRR est élevé, manquer un churner coûte très cher → le modèle
        doit abaisser le seuil pour prédire positif plus souvent.
        """
        rng = np.random.default_rng(0)
        n = 200
        y = np.array([1] * 40 + [0] * 160, dtype=float)
        proba = np.concatenate([rng.beta(4, 2, 40), rng.beta(2, 5, 160)])
        perm = rng.permutation(n)
        y, proba = y[perm], proba[perm]

        # Scénario A : petits comptes (FN peu coûteux)
        mrr_petit = np.full(n, 200.0)
        # Scénario B : gros comptes (FN très coûteux)
        mrr_gros = np.full(n, 5000.0)

        _, _, seuil_petit = economics.gain_par_seuil(y, proba, mrr_petit, n_seuils=100)
        _, _, seuil_gros = economics.gain_par_seuil(y, proba, mrr_gros, n_seuils=100)

        assert seuil_gros <= seuil_petit, (
            f"Seuil gros comptes ({seuil_gros:.3f}) devrait être ≤ seuil petits comptes "
            f"({seuil_petit:.3f}) : des FN plus coûteux imposent une intervention plus agressive."
        )

    def test_taux_succes_override(self, donnees_eco: tuple) -> None:
        y, proba, mrr = donnees_eco
        _, _, seuil_ts_bas = economics.gain_par_seuil(y, proba, mrr, taux_succes_override=0.10)
        _, _, seuil_ts_haut = economics.gain_par_seuil(y, proba, mrr, taux_succes_override=0.60)
        # Taux de succès plus élevé → intervention plus rentable → seuil plus bas
        assert seuil_ts_haut <= seuil_ts_bas, (
            f"Seuil ts=0.60 ({seuil_ts_haut:.3f}) devrait être ≤ seuil ts=0.10 "
            f"({seuil_ts_bas:.3f})."
        )


# ---------------------------------------------------------------------------
# courbe_lift
# ---------------------------------------------------------------------------


class TestCourbeLift:
    def test_retourne_figure_et_dataframe(self, donnees_eco: tuple) -> None:
        y, proba, _ = donnees_eco
        fig, courbe = economics.courbe_lift(y, proba)
        assert isinstance(fig, Figure)
        assert isinstance(courbe, pd.DataFrame)

    def test_courbe_contient_colonnes_attendues(self, donnees_eco: tuple) -> None:
        y, proba, _ = donnees_eco
        _, courbe = economics.courbe_lift(y, proba)
        assert "pct_contactes" in courbe.columns
        assert "pct_churners_captures" in courbe.columns
        assert "lift" in courbe.columns

    def test_longueur_egal_n(self, donnees_eco: tuple) -> None:
        y, proba, _ = donnees_eco
        _, courbe = economics.courbe_lift(y, proba)
        assert len(courbe) == len(y)

    def test_gain_cumulatif_monotone(self, donnees_eco: tuple) -> None:
        y, proba, _ = donnees_eco
        _, courbe = economics.courbe_lift(y, proba)
        captures = courbe["pct_churners_captures"].values
        assert (np.diff(captures) >= -1e-9).all(), "Les gains cumulatifs doivent être croissants."

    def test_captures_max_est_100_pct(self, donnees_eco: tuple) -> None:
        y, proba, _ = donnees_eco
        _, courbe = economics.courbe_lift(y, proba)
        assert courbe["pct_churners_captures"].iloc[-1] == pytest.approx(100.0, abs=0.1)

    def test_lift_superieur_a_1_au_debut(self, donnees_eco: tuple) -> None:
        y, proba, _ = donnees_eco
        _, courbe = economics.courbe_lift(y, proba)
        # Le top 10 % doit produire un lift > 1 si le modèle est informatif
        idx_10pct = max(1, len(courbe) // 10)
        lift_top = float(courbe["lift"].iloc[idx_10pct - 1])
        assert lift_top > 1.0, f"Lift au top 10 % = {lift_top:.2f} — modèle non informatif."


# ---------------------------------------------------------------------------
# precision_at_k
# ---------------------------------------------------------------------------


class TestPrecisionAtK:
    def test_retourne_float_dans_0_1(self, donnees_eco: tuple) -> None:
        y, proba, _ = donnees_eco
        p = economics.precision_at_k(y, proba, k=20)
        assert 0.0 <= p <= 1.0

    def test_k_egal_n_est_la_prevalence(self, donnees_eco: tuple) -> None:
        y, proba, _ = donnees_eco
        p = economics.precision_at_k(y, proba, k=len(y))
        prevalence = float(np.asarray(y).mean())
        assert p == pytest.approx(prevalence, abs=1e-9)

    def test_k_zero_retourne_zero(self, donnees_eco: tuple) -> None:
        y, proba, _ = donnees_eco
        assert economics.precision_at_k(y, proba, k=0) == 0.0

    def test_precision_top10_superieure_a_prevalence(self, donnees_eco: tuple) -> None:
        y, proba, _ = donnees_eco
        prevalence = float(np.asarray(y).mean())
        p_top10 = economics.precision_at_k(y, proba, k=10)
        assert (
            p_top10 >= prevalence
        ), f"Précision@10 = {p_top10:.3f} < prévalence {prevalence:.3f} — modèle non informatif."

    def test_precision_monotone_decroissante_avec_k(self, donnees_eco: tuple) -> None:
        """Plus k est grand, plus on inclut des non-churners, donc précision baisse."""
        y, proba, _ = donnees_eco
        p5 = economics.precision_at_k(y, proba, k=5)
        p50 = economics.precision_at_k(y, proba, k=50)
        p150 = economics.precision_at_k(y, proba, k=150)
        assert p5 >= p50 >= p150 - 0.05  # tolérance légère pour données synthétiques


# ---------------------------------------------------------------------------
# sensibilite_seuil
# ---------------------------------------------------------------------------


class TestSensibiliteSeuil:
    def test_retourne_figure_et_dataframe(self, donnees_eco: tuple) -> None:
        y, proba, mrr = donnees_eco
        fig, tableau = economics.sensibilite_seuil(y, proba, mrr, [0.20, 0.30, 0.40])
        assert isinstance(fig, Figure)
        assert isinstance(tableau, pd.DataFrame)

    def test_tableau_contient_colonnes_attendues(self, donnees_eco: tuple) -> None:
        y, proba, mrr = donnees_eco
        _, tableau = economics.sensibilite_seuil(y, proba, mrr, [0.20, 0.30])
        assert "taux_succes_retention" in tableau.columns
        assert "seuil_optimal" in tableau.columns
        assert "gain_max_eur" in tableau.columns

    def test_nombre_lignes_egal_plage(self, donnees_eco: tuple) -> None:
        y, proba, mrr = donnees_eco
        plage = [0.20, 0.25, 0.30, 0.35, 0.40]
        _, tableau = economics.sensibilite_seuil(y, proba, mrr, plage)
        assert len(tableau) == len(plage)

    def test_seuil_optimal_descend_quand_taux_succes_augmente(self, donnees_eco: tuple) -> None:
        """Monotonie : taux de succès élevé → intervention plus rentable → seuil plus bas."""
        y, proba, mrr = donnees_eco
        _, tableau = economics.sensibilite_seuil(y, proba, mrr, [0.10, 0.20, 0.40, 0.60])
        seuils = tableau["seuil_optimal"].values
        # Le seuil doit être globalement décroissant (ou stable) quand ts augmente
        assert (
            seuils[0] >= seuils[-1]
        ), f"Seuil à ts=10 % ({seuils[0]:.3f}) devrait être ≥ seuil à ts=60 % ({seuils[-1]:.3f})."

    def test_gain_augmente_avec_taux_succes(self, donnees_eco: tuple) -> None:
        y, proba, mrr = donnees_eco
        _, tableau = economics.sensibilite_seuil(y, proba, mrr, [0.10, 0.50])
        gain_bas = tableau.loc[tableau["taux_succes_retention"] == 0.10, "gain_max_eur"].item()
        gain_haut = tableau.loc[tableau["taux_succes_retention"] == 0.50, "gain_max_eur"].item()
        assert gain_haut >= gain_bas, "Un meilleur taux de succès doit améliorer le gain maximal."


# ---------------------------------------------------------------------------
# table_de_decision
# ---------------------------------------------------------------------------


class TestTableDeDecision:
    def test_retourne_dataframe(self) -> None:
        df = economics.table_de_decision()
        assert isinstance(df, pd.DataFrame)

    def test_contient_trois_zones(self) -> None:
        df = economics.table_de_decision()
        assert len(df) == 3

    def test_colonnes_attendues(self) -> None:
        df = economics.table_de_decision()
        assert "action" in df.columns
        assert "score_min" in df.columns
        assert "score_max" in df.columns
        assert "responsable" in df.columns

    def test_trois_actions_distinctes(self) -> None:
        df = economics.table_de_decision()
        actions = set(df["action"])
        assert "Surveiller" in actions
        assert "Contacter" in actions
        assert "Escalader" in actions

    def test_zones_couvrent_intervalle_0_1(self) -> None:
        df = economics.table_de_decision()
        assert float(df["score_min"].min()) == pytest.approx(0.0)
        assert float(df["score_max"].max()) == pytest.approx(1.0)

    def test_zones_contiguës_sans_trou(self) -> None:
        df = economics.table_de_decision().sort_values("score_min").reset_index(drop=True)
        for i in range(len(df) - 1):
            assert float(df.loc[i, "score_max"]) == pytest.approx(float(df.loc[i + 1, "score_min"]))

    def test_seuils_personnalisables(self) -> None:
        df = economics.table_de_decision(seuil_contact=0.30, seuil_escalade=0.70)
        assert float(df["score_max"].iloc[0]) == pytest.approx(0.30)
        assert float(df["score_min"].iloc[2]) == pytest.approx(0.70)
