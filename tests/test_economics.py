"""Tests pour churn_saas.models.economics — justification économique du seuil."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # backend non-interactif — doit précéder tout import matplotlib

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from churn_saas import config, economie
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
    MRR variable : certains comptes rapportent beaucoup s'ils sont sauvés.
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
    def test_retourne_dataframe_et_float_sans_figure(self, donnees_eco: tuple) -> None:
        import matplotlib.pyplot as plt

        y, proba, mrr = donnees_eco
        plt.close("all")
        courbe, seuil_opt = economics.gain_par_seuil(y, proba, mrr)
        assert isinstance(courbe, pd.DataFrame)
        assert isinstance(seuil_opt, float)
        # Calcul seul : aucune figure (donc aucun numéro de figure) consommée
        assert plt.get_fignums() == []

    def test_tracer_gain_par_seuil(self, donnees_eco: tuple) -> None:
        y, proba, mrr = donnees_eco
        courbe, seuil_opt = economics.gain_par_seuil(y, proba, mrr)
        assert isinstance(economics.tracer_gain_par_seuil(courbe, seuil_opt), Figure)

    def test_seuil_optimal_different_de_0_5(self, donnees_eco: tuple) -> None:
        """Exigence centrale : le seuil économique ≠ 0,5."""
        y, proba, mrr = donnees_eco
        _, seuil_opt = economics.gain_par_seuil(y, proba, mrr)
        assert abs(seuil_opt - 0.5) > 0.05, (
            f"Seuil optimal = {seuil_opt:.3f} trop proche de 0,5 — "
            "vérifier que le signal et les coûts sont asymétriques."
        )

    def test_seuil_dans_intervalle_0_1(self, donnees_eco: tuple) -> None:
        y, proba, mrr = donnees_eco
        _, seuil_opt = economics.gain_par_seuil(y, proba, mrr)
        assert 0.0 <= seuil_opt <= 1.0

    def test_courbe_contient_colonnes_attendues(self, donnees_eco: tuple) -> None:
        y, proba, mrr = donnees_eco
        courbe, _ = economics.gain_par_seuil(y, proba, mrr)
        assert set(courbe.columns) >= {"seuil", "gain_net_eur", "n_alertes", "precision", "rappel"}

    def test_gain_optimal_superieur_au_gain_a_0_5(self, donnees_eco: tuple) -> None:
        y, proba, mrr = donnees_eco
        courbe, seuil_opt = economics.gain_par_seuil(y, proba, mrr)
        gain_opt = courbe["gain_net_eur"].max()
        # Trouver le gain à 0,5 dans la courbe
        idx_05 = (courbe["seuil"] - 0.5).abs().idxmin()
        gain_a_05 = float(courbe.loc[idx_05, "gain_net_eur"])
        assert (
            gain_opt >= gain_a_05
        ), f"Gain optimal ({gain_opt:.0f} €) devrait être ≥ gain à 0,5 ({gain_a_05:.0f} €)."

    def test_n_alertes_decroit_avec_seuil(self, donnees_eco: tuple) -> None:
        y, proba, mrr = donnees_eco
        courbe, _ = economics.gain_par_seuil(y, proba, mrr, n_seuils=50)
        # Le nombre d'alertes doit être globalement décroissant quand le seuil monte
        alertes = courbe["n_alertes"].values
        assert alertes[0] >= alertes[-1], "Plus le seuil est haut, moins d'alertes déclenchées."

    def test_seuil_descend_quand_le_mrr_augmente(self) -> None:
        """Preuve de monotonie : comptes plus gros → seuil de rentabilité plus bas.

        Si le MRR est élevé, sauver un churner rapporte beaucoup face au coût fixe du geste
        → il devient rentable de contacter des comptes moins risqués.
        """
        rng = np.random.default_rng(0)
        n = 200
        y = np.array([1] * 40 + [0] * 160, dtype=float)
        proba = np.concatenate([rng.beta(4, 2, 40), rng.beta(2, 5, 160)])
        perm = rng.permutation(n)
        y, proba = y[perm], proba[perm]

        # Scénario A : petits comptes (un sauvetage rapporte peu)
        mrr_petit = np.full(n, 200.0)
        # Scénario B : gros comptes (un sauvetage rapporte beaucoup)
        mrr_gros = np.full(n, 5000.0)

        _, seuil_petit = economics.gain_par_seuil(y, proba, mrr_petit, n_seuils=100)
        _, seuil_gros = economics.gain_par_seuil(y, proba, mrr_gros, n_seuils=100)

        assert seuil_gros <= seuil_petit, (
            f"Seuil gros comptes ({seuil_gros:.3f}) devrait être ≤ seuil petits comptes "
            f"({seuil_petit:.3f}) : un sauvetage plus rentable justifie de contacter plus large."
        )

    def test_taux_succes_override(self, donnees_eco: tuple) -> None:
        y, proba, mrr = donnees_eco
        _, seuil_ts_bas = economics.gain_par_seuil(y, proba, mrr, taux_succes_override=0.10)
        _, seuil_ts_haut = economics.gain_par_seuil(y, proba, mrr, taux_succes_override=0.60)
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
# Gain par rapport à « ne rien faire »
# ---------------------------------------------------------------------------


class TestReferenceNeRienFaire:
    def test_gain_nul_quand_personne_n_est_contacte(self, donnees_eco: tuple) -> None:
        # τ = 1 : aucun contact → situation identique à « ne rien faire » → gain 0
        y, proba, mrr = donnees_eco
        courbe, _ = economics.gain_par_seuil(y, proba, mrr)
        assert courbe["gain_net_eur"].iloc[-1] == pytest.approx(0.0)

    def test_churner_manque_ne_coute_rien_de_plus(self) -> None:
        # Un churner non contacté ne pénalise pas le gain (perte subie avec ou sans modèle)
        y = np.array([1.0, 0.0])
        proba = np.array([0.1, 0.9])  # seul le fidèle est au-dessus de 0,5
        courbe, _ = economics.gain_par_seuil(y, proba, np.array([1_000.0, 1_000.0]), n_seuils=3)
        cout, _, _ = economics._params_cout()
        assert courbe.loc[courbe["seuil"] == 0.5, "gain_net_eur"].item() == pytest.approx(-cout)


# ---------------------------------------------------------------------------
# gain_sous_capacite
# ---------------------------------------------------------------------------


class TestGainSousCapacite:
    def test_respecte_la_capacite(self, donnees_eco: tuple) -> None:
        y, proba, mrr = donnees_eco
        bilan = economics.gain_sous_capacite(y, proba, mrr, capacite=20)
        assert bilan["n_contactes"] <= 20

    def test_valeur_attendue_bat_la_proba_quand_les_mrr_different(self) -> None:
        # Deux comptes, capacité 1 : petit compte très risqué contre gros compte moyennement
        # risqué. La valeur attendue choisit le gros compte, qui rapporte davantage.
        y = np.array([1.0, 1.0])
        proba = np.array([0.9, 0.5])
        mrr = np.array([50.0, 5_000.0])
        par_valeur = economics.gain_sous_capacite(y, proba, mrr, capacite=1)
        par_proba = economics.gain_sous_capacite(y, proba, mrr, capacite=1, classement="proba")
        assert par_valeur["gain_net"] > par_proba["gain_net"]

    def test_bilan_coherent(self, donnees_eco: tuple) -> None:
        y, proba, mrr = donnees_eco
        b = economics.gain_sous_capacite(y, proba, mrr, capacite=30)
        assert b["gain_net"] == pytest.approx(b["valeur_sauvee"] - b["cout_gestes"])
        assert 0.0 <= b["precision"] <= 1.0

    def test_classement_mrr_retient_les_plus_gros_comptes(self) -> None:
        # Référence sans modèle : la probabilité est ignorée, seuls les plus gros MRR comptent
        y = np.array([1.0, 0.0, 1.0])
        proba = np.array([0.9, 0.1, 0.5])
        mrr = np.array([100.0, 9_000.0, 5_000.0])
        b = economics.gain_sous_capacite(y, proba, mrr, capacite=2, classement="mrr")
        assert b["n_churners"] == 1.0
        assert b["mrr_churners_couverts"] == pytest.approx(5_000.0)

    def test_classement_inconnu_leve(self, donnees_eco: tuple) -> None:
        y, proba, mrr = donnees_eco
        with pytest.raises(ValueError, match="Classement"):
            economics.gain_sous_capacite(y, proba, mrr, classement="hasard")


class TestGainAttribuableAuModele:
    def test_ecart_entre_regle_retenue_et_reference(self, donnees_eco: tuple) -> None:
        y, proba, mrr = donnees_eco
        r = economics.gain_attribuable_au_modele(y, proba, mrr, capacite=30)
        assert r["gain_attribuable"] == pytest.approx(
            r["modele"]["gain_net"] - r["reference"]["gain_net"]
        )
        # Même budget de gestes : seuls les comptes choisis diffèrent
        assert r["modele"]["cout_gestes"] == pytest.approx(r["reference"]["cout_gestes"])

    def test_plage_encadre_la_valeur_centrale(self, donnees_eco: tuple) -> None:
        y, proba, mrr = donnees_eco
        r = economics.gain_attribuable_au_modele(y, proba, mrr, capacite=30)
        assert r["gain_attribuable_min"] <= r["gain_attribuable"] <= r["gain_attribuable_max"]


# ---------------------------------------------------------------------------
# sensibilite_capacite
# ---------------------------------------------------------------------------


class TestSensibiliteCapacite:
    def test_figure_et_grille_complete(self, donnees_eco: tuple) -> None:
        y, proba, mrr = donnees_eco
        fig, tableau = economics.sensibilite_capacite(y, proba, mrr, [0.2, 0.4], [10, 20, 40])
        assert isinstance(fig, Figure)
        assert len(tableau) == 6

    def test_gain_croit_avec_le_taux_de_succes(self, donnees_eco: tuple) -> None:
        y, proba, mrr = donnees_eco
        _, t = economics.sensibilite_capacite(y, proba, mrr, [0.1, 0.5], [30])
        assert t.loc[t["taux_succes_retention"] == 0.5, "gain_net_eur"].item() > (
            t.loc[t["taux_succes_retention"] == 0.1, "gain_net_eur"].item()
        )


# ---------------------------------------------------------------------------
# table_de_decision
# ---------------------------------------------------------------------------


class TestTableDeDecision:
    def test_trois_zones_trois_actions(self) -> None:
        df = economics.table_de_decision(capacite=45, seuil_rentabilite=0.05)
        assert len(df) == 3
        assert list(df["action"]) == ["Contacter", "Action automatisée", "Veille"]

    def test_capacite_citee(self) -> None:
        df = economics.table_de_decision(capacite=45, seuil_rentabilite=0.05)
        assert "45" in df.loc[0, "zone"]


# ---------------------------------------------------------------------------
# gain_sur_annee / gain_annuel_attribuable / concentration_du_gain
# ---------------------------------------------------------------------------


class TestGainSurAnnee:
    def test_premier_mois_egal_au_bilan_mensuel(self, donnees_eco: tuple) -> None:
        """Le premier mois de la simulation est exactement le mois chiffré par §12.6."""
        y, proba, mrr = donnees_eco
        mensuel = economics.gain_sur_annee(y, proba, mrr, capacite=10)
        attribution = economics.gain_attribuable_au_modele(y, proba, mrr, capacite=10)
        assert mensuel.loc[1, "gain_net_modele"] == pytest.approx(attribution["modele"]["gain_net"])
        assert mensuel.loc[1, "gain_attribuable"] == pytest.approx(attribution["gain_attribuable"])

    def test_aucun_compte_contacte_deux_fois(self, donnees_eco: tuple) -> None:
        """Sur un horizon qui couvre tout le portefeuille, la référence contacte chaque compte
        une seule fois : elle retrouve exactement tous les churners."""
        y, proba, mrr = donnees_eco
        mensuel = economics.gain_sur_annee(y, proba, mrr, capacite=50, n_mois=6)
        assert mensuel["n_churners_reference"].sum() == int(y.sum())
        assert mensuel["n_churners_modele"].sum() <= int(y.sum())

    def test_cumul_coherent(self, donnees_eco: tuple) -> None:
        y, proba, mrr = donnees_eco
        mensuel = economics.gain_sur_annee(y, proba, mrr, capacite=10)
        assert len(mensuel) == economics.MOIS_PAR_AN
        assert mensuel["gain_attribuable_cumule"].iloc[-1] == pytest.approx(
            mensuel["gain_attribuable"].sum()
        )

    def test_annuel_inferieur_a_douze_premiers_mois(self, donnees_eco: tuple) -> None:
        """Les meilleurs comptes ne sont disponibles qu'une fois : l'année vaut moins que
        douze fois le premier mois pour la règle retenue."""
        y, proba, mrr = donnees_eco
        mensuel = economics.gain_sur_annee(y, proba, mrr, capacite=10)
        assert mensuel["gain_net_modele"].sum() < 12 * mensuel.loc[1, "gain_net_modele"]

    def test_fourchette_encadre_le_central(self, donnees_eco: tuple) -> None:
        y, proba, mrr = donnees_eco
        resultat = economics.gain_annuel_attribuable(y, proba, mrr, capacite=10)
        assert resultat["annuel_min"] <= resultat["annuel"] <= resultat["annuel_max"]
        assert resultat["premier_mois"] == pytest.approx(
            resultat["mensuel"]["gain_attribuable"].iloc[0]
        )

    def test_tracer_gain_sur_annee(self, donnees_eco: tuple) -> None:
        y, proba, mrr = donnees_eco
        fig = economics.tracer_gain_sur_annee(economics.gain_sur_annee(y, proba, mrr, capacite=10))
        assert isinstance(fig, Figure)


class TestBilanProjet:
    """Capacité réduite : à 45 gestes, le portefeuille synthétique s'épuise avant un an."""

    def test_couts_lus_dans_config(self, donnees_eco: tuple) -> None:
        y, proba, mrr = donnees_eco
        bilan = economics.bilan_projet(
            economics.gain_annuel_attribuable(y, proba, mrr, capacite=10)
        )
        c = config.COUTS_PROJET
        jours_build = c["jours_build_data_scientist"] + c["jours_build_dsi"] + c["jours_build_cs"]
        assert bilan["cout_build"] == pytest.approx(jours_build * c["cout_journalier_eur"])
        assert bilan["cout_annuel"] == pytest.approx(
            bilan["cout_build"] / c["duree_amortissement_ans"] + bilan["cout_run_annuel"]
        )

    def test_roi_sur_gain_attribuable(self, donnees_eco: tuple) -> None:
        y, proba, mrr = donnees_eco
        gain = economics.gain_annuel_attribuable(y, proba, mrr, capacite=10)
        bilan = economics.bilan_projet(gain)
        assert bilan["roi_projet"] == pytest.approx(
            (gain["annuel"] - bilan["cout_annuel"]) / bilan["cout_annuel"]
        )

    def test_taux_succes_equilibre_annule_le_gain_net(self, donnees_eco: tuple) -> None:
        """Au taux d'équilibre, le gain attribuable sur un an couvre tout juste le coût annuel."""
        y, proba, mrr = donnees_eco
        bilan = economics.bilan_projet(
            economics.gain_annuel_attribuable(y, proba, mrr, capacite=10)
        )
        mensuel = economics.gain_sur_annee(
            y, proba, mrr, capacite=10, taux_succes=bilan["taux_succes_equilibre"]
        )
        assert mensuel["gain_attribuable"].sum() == pytest.approx(bilan["cout_annuel"])

    def test_delai_de_retour(self, donnees_eco: tuple) -> None:
        """Premier mois où le cumul attribuable couvre le build et le run déjà engagés."""
        y, proba, mrr = donnees_eco
        gain = economics.gain_annuel_attribuable(y, proba, mrr, capacite=10)
        bilan = economics.bilan_projet(gain)
        cumul = gain["mensuel"]["gain_attribuable_cumule"]
        engage = bilan["cout_build"] + bilan["cout_run_annuel"] / 12 * cumul.index.to_numpy()
        couverts = cumul.index[cumul.to_numpy() >= engage]
        attendu = int(couverts[0]) if len(couverts) else None
        assert bilan["mois_retour"] == attendu


class TestConcentrationDuGain:
    def test_sans_exclusion_gain_inchange(self, donnees_eco: tuple) -> None:
        """Au quantile 1, aucun compte n'est exclu : on retrouve le gain attribuable."""
        y, proba, mrr = donnees_eco
        resultat = economics.concentration_du_gain(y, proba, mrr, capacite=10, quantile_mrr=1.0)
        attendu = economics.gain_attribuable_au_modele(y, proba, mrr, capacite=10)
        assert resultat["n_exclus"] == 0
        assert resultat["gain_attribuable_sans_gros"] == pytest.approx(attendu["gain_attribuable"])

    def test_part_top_dans_bornes(self, donnees_eco: tuple) -> None:
        y, proba, mrr = donnees_eco
        resultat = economics.concentration_du_gain(y, proba, mrr, capacite=10, n_top=10)
        # Les 10 comptes retenus apportent tout le gain
        assert resultat["part_top"] == pytest.approx(1.0)


class TestDepartsEtales:
    def test_premier_mois_inchange(self, donnees_eco: tuple) -> None:
        """Au premier mois, tous les partants sont encore là : les deux scénarios coïncident."""
        y, proba, mrr = donnees_eco
        fige = economics.gain_sur_annee(y, proba, mrr, capacite=10)
        etale = economics.gain_sur_annee(y, proba, mrr, capacite=10, departs_etales=True)
        assert etale.loc[1, "gain_net_modele"] == pytest.approx(fige.loc[1, "gain_net_modele"])

    def test_gains_bruts_reduits(self, donnees_eco: tuple) -> None:
        """Des partants partis avant l'appel ne sont plus retenus : chaque mois rapporte moins."""
        y, proba, mrr = donnees_eco
        fige = economics.gain_sur_annee(y, proba, mrr, capacite=10)
        etale = economics.gain_sur_annee(y, proba, mrr, capacite=10, departs_etales=True)
        assert (etale["gain_net_modele"] <= fige["gain_net_modele"] + 1e-9).all()
        assert (etale["gain_net_reference"] <= fige["gain_net_reference"] + 1e-9).all()

    def test_resultat_expose_le_scenario(self, donnees_eco: tuple) -> None:
        y, proba, mrr = donnees_eco
        resultat = economics.gain_annuel_attribuable(y, proba, mrr, capacite=10)
        assert resultat["annuel_departs_etales"] == pytest.approx(
            resultat["mensuel_departs_etales"]["gain_attribuable"].sum()
        )


# ---------------------------------------------------------------------------
# Règle de décision à deux niveaux : vigilance (recall) puis appels sous capacité
# ---------------------------------------------------------------------------


class TestSeuilPourRecall:
    def test_atteint_le_recall_cible(self, donnees_eco: tuple) -> None:
        y, proba, _ = donnees_eco
        resultat = economics.seuil_pour_recall(y, proba, recall_cible=0.80)
        assert resultat["recall"] >= 0.80
        signales = proba >= resultat["seuil"]
        assert resultat["recall"] == pytest.approx(y[signales].sum() / y.sum())
        assert resultat["precision"] == pytest.approx(y[signales].mean())
        assert resultat["part_signalee"] == pytest.approx(signales.mean())
        assert resultat["n_faux_negatifs"] == int(y[~signales].sum())

    def test_seuil_le_plus_haut_possible(self, donnees_eco: tuple) -> None:
        """Le seuil est le plus exigeant qui tient la cible : un cran au-dessus, on la rate."""
        y, proba, _ = donnees_eco
        seuil = economics.seuil_pour_recall(y, proba, recall_cible=0.80)["seuil"]
        plus_haut = proba[proba > seuil].min()
        assert y[proba >= plus_haut].sum() / y.sum() < 0.80

    def test_cible_par_defaut_lue_dans_config(self, donnees_eco: tuple) -> None:
        y, proba, _ = donnees_eco
        assert economics.seuil_pour_recall(y, proba) == economics.seuil_pour_recall(
            y, proba, recall_cible=config.RECALL_CIBLE_VIGILANCE
        )

    def test_exemple_a_la_main(self) -> None:
        y = np.array([1, 0, 1, 0, 1, 0])
        proba = np.array([0.9, 0.8, 0.7, 0.4, 0.3, 0.1])
        resultat = economics.seuil_pour_recall(y, proba, recall_cible=0.6)
        # 2 churners sur 3 dès 0,7 : recall 2/3 ≥ 0,6
        assert resultat["seuil"] == pytest.approx(0.7)
        assert resultat["precision"] == pytest.approx(2 / 3)
        assert resultat["part_signalee"] == pytest.approx(0.5)
        assert resultat["n_faux_negatifs"] == 1

    def test_sans_churner_leve_une_erreur(self) -> None:
        with pytest.raises(ValueError):
            economics.seuil_pour_recall(np.zeros(5), np.linspace(0, 1, 5))


class TestTableDeuxNiveaux:
    def test_structure(self, donnees_eco: tuple) -> None:
        y, proba, mrr = donnees_eco
        table = economics.table_deux_niveaux(y, proba, mrr, capacite=10)
        assert isinstance(table, pd.DataFrame)
        assert len(table) == 2
        for col in (
            "niveau",
            "seuil",
            "n_comptes",
            "n_churners",
            "recall",
            "precision",
            "n_faux_negatifs",
            "valeur_faux_negatifs_eur",
            "gain_net_eur",
        ):
            assert col in table.columns

    def test_entonnoir_coherent(self, donnees_eco: tuple) -> None:
        """Le niveau 2 appelle au plus ``capacite`` comptes, tous pris dans le niveau 1."""
        y, proba, mrr = donnees_eco
        table = economics.table_deux_niveaux(y, proba, mrr, capacite=10)
        n1, n2 = table.iloc[0], table.iloc[1]
        assert n2["n_comptes"] <= min(10, n1["n_comptes"])
        assert n2["n_churners"] <= n1["n_churners"]
        assert n2["n_faux_negatifs"] >= n1["n_faux_negatifs"]
        assert n2["valeur_faux_negatifs_eur"] >= n1["valeur_faux_negatifs_eur"]
        assert n1["n_churners"] + n1["n_faux_negatifs"] == y.sum()

    def test_niveau_1_reprend_seuil_pour_recall(self, donnees_eco: tuple) -> None:
        y, proba, mrr = donnees_eco
        table = economics.table_deux_niveaux(y, proba, mrr, capacite=10)
        attendu = economics.seuil_pour_recall(y, proba)
        assert table.iloc[0]["seuil"] == pytest.approx(attendu["seuil"])
        assert table.iloc[0]["recall"] == pytest.approx(attendu["recall"])

    def test_valeur_faux_negatifs_formule_de_reference(self, donnees_eco: tuple) -> None:
        """Valeur perdue d'un churner manqué : MRR × horizon × marge (point de vigilance n°3)."""
        y, proba, mrr = donnees_eco
        table = economics.table_deux_niveaux(y, proba, mrr, capacite=10)
        seuil = table.iloc[0]["seuil"]
        manques = (y == 1) & (proba < seuil)
        h = config.HYPOTHESES_ECONOMIQUES
        attendu = mrr[manques].sum() * h["horizon_mois"] * h["marge_brute_pct"]
        assert table.iloc[0]["valeur_faux_negatifs_eur"] == pytest.approx(attendu)

    def test_gain_niveau_2_egal_a_gain_sous_capacite(self, donnees_eco: tuple) -> None:
        y, proba, mrr = donnees_eco
        table = economics.table_deux_niveaux(y, proba, mrr, capacite=10)
        signales = proba >= table.iloc[0]["seuil"]
        attendu = economics.gain_sous_capacite(
            y[signales], proba[signales], mrr[signales], capacite=10
        )
        assert table.iloc[1]["gain_net_eur"] == pytest.approx(attendu["gain_net"])


class TestNiveauRisqueSeuilVigilance:
    def test_repli_sur_config_sans_artefact(self) -> None:
        surveillance = config.SEUILS_RISQUE["surveillance"]
        assert economie.niveau_risque(surveillance) == "SURVEILLANCE"
        assert economie.niveau_risque(surveillance - 0.01) == "OK"

    def test_seuil_lu_dans_artefact(self) -> None:
        config.TABLES.mkdir(parents=True, exist_ok=True)
        (config.TABLES / "seuil_vigilance.json").write_text(
            json.dumps({"seuil": 0.25}), encoding="utf-8"
        )
        assert economie.niveau_risque(0.30) == "SURVEILLANCE"
        assert economie.niveau_risque(0.20) == "OK"

    def test_seuil_explicite_prioritaire(self) -> None:
        """L'API et le batch passent leur propre seuil : paliers inchangés pour eux."""
        config.TABLES.mkdir(parents=True, exist_ok=True)
        (config.TABLES / "seuil_vigilance.json").write_text(
            json.dumps({"seuil": 0.25}), encoding="utf-8"
        )
        assert economie.niveau_risque(0.30, 0.40) == "OK"
        assert economie.niveau_risque(0.70, 0.40) == "ALERTE_ROUGE"
