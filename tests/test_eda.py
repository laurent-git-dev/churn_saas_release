"""Tests des fonctions d'analyse exploratoire (eda.py).

Couvre : distribution_cible, univarie_numeriques, univarie_categorielles,
repartition_modalites, taux_churn_par_modalite, association_categorielles, pouvoir_discriminant,
matrice_correlation, tester_hypotheses.
"""

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from matplotlib.figure import Figure  # noqa: E402

from churn_saas.eda import (  # noqa: E402
    association_categorielles,
    distribution_cible,
    matrice_correlation,
    pouvoir_discriminant,
    repartition_modalites,
    taux_churn_par_modalite,
    univarie_categorielles,
    univarie_numeriques,
)
from churn_saas.eda import (  # noqa: E402
    tester_hypotheses as _tester_hypotheses,  # alias : le préfixe « test » serait collecté
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)


@pytest.fixture(autouse=True)
def fermer_figures():
    import matplotlib.pyplot as plt

    yield
    plt.close("all")


def _df_test(n: int = 300) -> pd.DataFrame:
    churn = _RNG.integers(0, 2, size=n)
    return pd.DataFrame(
        {
            "churn": churn,
            "anciennete_mois": _RNG.integers(1, 37, size=n),
            "connexions_30j": _RNG.integers(0, 50, size=n),
            "mrr": _RNG.uniform(100.0, 5000.0, size=n),
            "secteur": _RNG.choice(["Tech", "Finance", "Santé", "Commerce"], size=n),
            "plan": _RNG.choice(["Starter", "Pro", "Business"], size=n),
        }
    )


# ---------------------------------------------------------------------------
# distribution_cible
# ---------------------------------------------------------------------------


class TestDistributionCible:
    def test_retourne_figure_et_dataframe(self) -> None:
        fig, tableau = distribution_cible(_df_test())
        assert isinstance(fig, Figure)
        assert isinstance(tableau, pd.DataFrame)

    def test_indices_attendus(self) -> None:
        _, tableau = distribution_cible(_df_test())
        assert "Prévalence churn" in tableau.index
        assert "Métrique recommandée" in tableau.index
        assert "Effectif total" in tableau.index

    def test_prevalence_coherente(self) -> None:
        df = pd.DataFrame({"churn": [1, 1, 0, 0, 0]})
        _, tableau = distribution_cible(df)
        # Format français : virgule décimale, espace fine insécable avant « % »
        assert tableau.loc["Prévalence churn", "valeur"] == "40,0\u202f%"

    def test_colonne_absente_leve_valueerror(self) -> None:
        df = pd.DataFrame({"autre": [0, 1, 0]})
        with pytest.raises(ValueError, match="cible"):
            distribution_cible(df)


# ---------------------------------------------------------------------------
# univarie_numeriques
# ---------------------------------------------------------------------------


class TestUnivarieNumeriques:
    def test_retourne_une_grille_unique(self) -> None:
        df = _df_test()
        fig, _ = univarie_numeriques(df, ["anciennete_mois", "connexions_30j", "mrr", "churn"])
        assert isinstance(fig, Figure)
        # 4 colonnes sur une grille 2 × 3 : 4 panneaux visibles, 2 masqués
        assert sum(ax.get_visible() for ax in fig.axes) == 4

    def test_tableau_colonnes_attendues(self) -> None:
        df = _df_test()
        _, tableau = univarie_numeriques(df, ["anciennete_mois"])
        attendues = {
            "asymétrie",
            "kurtosis_excès",
            "n_outliers_IQR",
            "n_distinctes",
            "médiane",
            "moyenne",
        }
        assert attendues.issubset(set(tableau.columns))

    def test_colonne_absente_ignoree_si_autres_presentes(self) -> None:
        df = _df_test()
        _, tableau = univarie_numeriques(df, ["anciennete_mois", "inexistante"])
        assert list(tableau.index) == ["anciennete_mois"]

    def test_taux_cible_outliers(self) -> None:
        # 20 valeurs à 1 (Q1 = Q3 = 1, IQR nul) : seule la valeur 100 est outlier, et elle churne
        df = pd.DataFrame({"x": [1.0] * 20 + [100.0], "churn": [0] * 20 + [1]})
        _, tableau = univarie_numeriques(df, ["x"], cible="churn")
        assert tableau.loc["x", "n_outliers_IQR"] == 1
        assert tableau.loc["x", "taux_cible_outliers"] == 1.0
        assert tableau.loc["x", "taux_cible_hors_outliers"] == 0.0

    def test_toutes_absentes_leve_valueerror(self) -> None:
        df = _df_test()
        with pytest.raises(ValueError):
            univarie_numeriques(df, ["inexistante"])

    def test_n_manquants_compte(self) -> None:
        df = pd.DataFrame({"x": [1.0, 2.0, float("nan"), 4.0], "churn": [0, 1, 0, 1]})
        _, tableau = univarie_numeriques(df, ["x"])
        assert tableau.loc["x", "n_manquants"] == 1
        assert tableau.loc["x", "n"] == 3


# ---------------------------------------------------------------------------
# univarie_categorielles
# ---------------------------------------------------------------------------


class TestUnivarieCategorielles:
    def test_retourne_une_grille_unique(self) -> None:
        df = _df_test()
        fig, _ = univarie_categorielles(df, ["secteur", "plan", "churn"])
        assert isinstance(fig, Figure)
        # 3 colonnes sur une grille 2 × 2 : 3 panneaux visibles, 1 masqué
        assert sum(ax.get_visible() for ax in fig.axes) == 3

    def test_cardinalite_calculee(self) -> None:
        df = pd.DataFrame({"cat": ["A", "B", "A", "C"], "churn": [0, 1, 0, 1]})
        _, tableau = univarie_categorielles(df, ["cat"])
        assert tableau.loc["cat", "cardinalité"] == 3

    def test_incoherence_casse_detectee(self) -> None:
        df = pd.DataFrame(
            {
                "cat": ["Tech", "tech", "TECH", "Finance", "Finance"],
                "churn": [0, 1, 0, 1, 0],
            }
        )
        _, tableau = univarie_categorielles(df, ["cat"])
        # Tech/tech/TECH → 1 groupe normalisé "tech" ; Finance → 1 groupe
        # cardinalité=4, cardinalité_norm=2, variantes=2
        assert tableau.loc["cat", "variantes_typographiques"] == 2
        assert tableau.loc["cat", "exemple_variantes"] == "«TECH» / «Tech» / «tech»"

    def test_espaces_comptes_comme_variantes(self) -> None:
        df = pd.DataFrame({"cat": ["PME", " PME ", "TPE"], "churn": [0, 1, 0]})
        _, tableau = univarie_categorielles(df, ["cat"])
        assert tableau.loc["cat", "variantes_typographiques"] == 1
        assert tableau.loc["cat", "exemple_variantes"] == "«␣PME␣» / «PME»"

    def test_mode_calcule_apres_normalisation(self) -> None:
        # Brut : « B » est le plus fréquent (3) ; normalisé : « a » (4) l'emporte
        df = pd.DataFrame({"cat": ["A", "A", "a", "a", "B", "B", "B"], "churn": [0] * 7})
        _, tableau = univarie_categorielles(df, ["cat"])
        assert tableau.loc["cat", "mode"] == "a"
        assert tableau.loc["cat", "fréquence_mode_pct"] == pytest.approx(400 / 7)

    def test_modalites_rares_comptees(self) -> None:
        # "rare" n'apparaît qu'une fois → signalé avec seuil_rare=5
        serie = ["courant"] * 50 + ["rare"]
        df = pd.DataFrame({"cat": serie, "churn": [0] * 51})
        _, tableau = univarie_categorielles(df, ["cat"], seuil_rare=5)
        assert tableau.loc["cat", "n_modalités_rares"] == 1

    def test_rares_disparaissent_apres_normalisation(self) -> None:
        # « X » et « x » sont rares séparément (3 chacun), pas une fois fusionnés (6)
        serie = ["courant"] * 50 + ["X"] * 3 + ["x"] * 3
        df = pd.DataFrame({"cat": serie, "churn": [0] * 56})
        _, tableau = univarie_categorielles(df, ["cat"], seuil_rare=5)
        assert tableau.loc["cat", "n_modalités_rares"] == 2
        assert tableau.loc["cat", "n_rares_normalisées"] == 0

    def test_toutes_absentes_leve_valueerror(self) -> None:
        df = _df_test()
        with pytest.raises(ValueError):
            univarie_categorielles(df, ["inexistante"])


# ---------------------------------------------------------------------------
# repartition_modalites
# ---------------------------------------------------------------------------


class TestRepartitionModalites:
    def test_repartition_uniforme(self) -> None:
        df = pd.DataFrame({"cat": ["a", "b", "c"] * 100})
        tableau = repartition_modalites(df, ["cat"])
        assert tableau.loc["cat", "n_modalités"] == 3
        assert tableau.loc["cat", "part_min"] == pytest.approx(1 / 3)
        assert tableau.loc["cat", "p_valeur"] == pytest.approx(1.0)

    def test_repartition_desequilibree_rejetee(self) -> None:
        df = pd.DataFrame({"cat": ["a"] * 900 + ["b"] * 100})
        tableau = repartition_modalites(df, ["cat"])
        assert tableau.loc["cat", "part_max"] == pytest.approx(0.9)
        assert tableau.loc["cat", "p_valeur"] < 0.001

    def test_normalise_casse_et_espaces(self) -> None:
        df = pd.DataFrame({"cat": ["A", " a ", "b", "B"]})
        tableau = repartition_modalites(df, ["cat"])
        assert tableau.loc["cat", "n_modalités"] == 2

    def test_toutes_absentes_leve_valueerror(self) -> None:
        with pytest.raises(ValueError):
            repartition_modalites(pd.DataFrame({"x": [1]}), ["inexistante"])


# ---------------------------------------------------------------------------
# taux_churn_par_modalite
# ---------------------------------------------------------------------------


class TestTauxChurnParModalite:
    def test_retourne_figure_et_dataframe(self) -> None:
        df = _df_test()
        fig, tableau = taux_churn_par_modalite(df, "secteur")
        assert isinstance(fig, Figure)
        assert "taux_churn" in tableau.columns
        assert "ic_bas" in tableau.columns
        assert "ic_haut" in tableau.columns
        assert "effectif" in tableau.columns

    def test_ic_dans_bornes_0_1(self) -> None:
        df = _df_test()
        _, tableau = taux_churn_par_modalite(df, "plan")
        assert (tableau["ic_bas"] >= 0).all()
        assert (tableau["ic_haut"] <= 1).all()
        assert (tableau["ic_bas"] <= tableau["taux_churn"]).all()
        assert (tableau["taux_churn"] <= tableau["ic_haut"]).all()

    def test_ic_plus_large_pour_faible_effectif(self) -> None:
        # 3 obs "rare" → IC beaucoup plus large que 50 obs "fréquent"
        df = pd.DataFrame(
            {
                "cat": ["rare"] * 3 + ["fréquent"] * 50,
                "churn": [1, 1, 1] + [0] * 25 + [1] * 25,
            }
        )
        _, tableau = taux_churn_par_modalite(df, "cat")
        largeur_rare = tableau.loc["rare", "ic_haut"] - tableau.loc["rare", "ic_bas"]
        largeur_freq = tableau.loc["fréquent", "ic_haut"] - tableau.loc["fréquent", "ic_bas"]
        assert largeur_rare > largeur_freq

    def test_trie_par_taux_decroissant(self) -> None:
        df = _df_test()
        _, tableau = taux_churn_par_modalite(df, "plan")
        taux = tableau["taux_churn"].tolist()
        assert taux == sorted(taux, reverse=True)

    def test_colonne_absente_leve_valueerror(self) -> None:
        df = _df_test()
        with pytest.raises(ValueError, match="absente"):
            taux_churn_par_modalite(df, "inexistante")


# ---------------------------------------------------------------------------
# association_categorielles
# ---------------------------------------------------------------------------


class TestAssociationCategorielles:
    def test_detecte_association_et_ignore_bruit(self) -> None:
        rng = np.random.default_rng(0)
        n = 2_000
        signal = rng.choice(["a", "b"], size=n)
        churn = np.where(signal == "a", rng.random(n) < 0.5, rng.random(n) < 0.1).astype(int)
        df = pd.DataFrame(
            {"signal": signal, "bruit": rng.choice(["x", "y", "z"], size=n), "churn": churn}
        )
        tableau = association_categorielles(df, ["bruit", "signal"])
        assert tableau.index[0] == "signal"
        assert bool(tableau.loc["signal", "significatif_BH"])
        assert not bool(tableau.loc["bruit", "significatif_BH"])
        assert (tableau["p_value_BH"] >= tableau["p_value_brute"]).all()
        assert tableau.loc["bruit", "n_modalites"] == 3

    def test_colonne_absente_leve_valueerror(self) -> None:
        with pytest.raises(ValueError, match="absente"):
            association_categorielles(_df_test(), ["inexistante"])


# ---------------------------------------------------------------------------
# pouvoir_discriminant
# ---------------------------------------------------------------------------


class TestPouvoirDiscriminant:
    def test_retourne_figure_et_dataframe(self) -> None:
        df = _df_test()
        fig, tableau = pouvoir_discriminant(df)
        assert isinstance(fig, Figure)
        assert "association" in tableau.columns
        assert "p_value_BH" in tableau.columns
        assert "significatif_BH" in tableau.columns

    def test_auc_superieure_ou_egale_0_5(self) -> None:
        df = _df_test()
        _, tableau = pouvoir_discriminant(df)
        num = tableau[tableau["type"] == "numérique"]
        assert (num["association"] >= 0.5).all()

    def test_trie_par_association_decroissante(self) -> None:
        df = _df_test()
        _, tableau = pouvoir_discriminant(df)
        vals = tableau["association"].tolist()
        assert vals == sorted(vals, reverse=True)

    def test_variable_tres_discriminante_significative(self) -> None:
        # Variable numériquefortement corrélée à la cible → significative
        n = 300
        churn = _RNG.integers(0, 2, size=n)
        signal = churn * 10.0 + _RNG.normal(0, 0.5, size=n)
        df = pd.DataFrame({"churn": churn, "signal": signal, "bruit": _RNG.normal(size=n)})
        _, tableau = pouvoir_discriminant(df)
        assert tableau.loc["signal", "significatif_BH"]

    def test_cible_absente_leve_valueerror(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(ValueError, match="cible"):
            pouvoir_discriminant(df)

    def test_types_colonnes_correctement_identifies(self) -> None:
        df = _df_test()
        _, tableau = pouvoir_discriminant(df)
        assert tableau.loc["anciennete_mois", "type"] == "numérique"
        assert tableau.loc["secteur", "type"] == "catégorielle"


# ---------------------------------------------------------------------------
# matrice_correlation
# ---------------------------------------------------------------------------


class TestMatriceCorrelation:
    def test_retourne_figure_et_dataframe(self) -> None:
        df = _df_test()
        fig, paires = matrice_correlation(df)
        assert isinstance(fig, Figure)
        assert isinstance(paires, pd.DataFrame)

    def test_paire_redondante_detectee(self) -> None:
        n = 200
        x = _RNG.uniform(0, 1, size=n)
        df = pd.DataFrame(
            {
                "x": x,
                "y": x + _RNG.uniform(0, 0.001, size=n),
                "churn": _RNG.integers(0, 2, size=n),
            }
        )
        _, paires = matrice_correlation(df, seuil_redondance=0.99)
        assert len(paires) >= 1
        assert "variable_1" in paires.columns
        assert "corrélation" in paires.columns

    def test_aucune_paire_si_independantes(self) -> None:
        df = pd.DataFrame(
            {
                "a": _RNG.uniform(size=200),
                "b": _RNG.uniform(size=200),
                "churn": _RNG.integers(0, 2, size=200),
            }
        )
        _, paires = matrice_correlation(df, seuil_redondance=0.85)
        assert len(paires) == 0

    def test_aucune_colonne_numerique_leve_valueerror(self) -> None:
        df = pd.DataFrame({"cat": ["a", "b", "c"]})
        with pytest.raises(ValueError, match="numérique"):
            matrice_correlation(df)


# ---------------------------------------------------------------------------
# tester_hypotheses
# ---------------------------------------------------------------------------


def _df_hypotheses(n: int = 600) -> pd.DataFrame:
    """Usage plus faible chez les churners (H vraie) ; tickets sans lien avec le churn."""
    rng = np.random.default_rng(7)
    churn = rng.integers(0, 2, size=n)
    return pd.DataFrame(
        {
            "churn": churn,
            "usage": rng.normal(50.0, 10.0, size=n) - 15.0 * churn,
            "tickets": rng.poisson(3.0, size=n),
            "csat": rng.normal(3.5, 0.8, size=n) - 0.8 * churn,
        }
    )


_HYPOTHESES_TEST = [
    ("HA", "usage", "-", "Un faible usage augmente le risque"),
    ("HB", "usage", "+", "Un fort usage augmente le risque"),
    ("HC", "tickets", "+", "De nombreux tickets augmentent le risque"),
]


class TestTesterHypotheses:
    def test_retourne_figure_et_dataframe(self) -> None:
        fig, tableau = _tester_hypotheses(_df_hypotheses(), _HYPOTHESES_TEST)
        assert isinstance(fig, Figure)
        assert list(tableau.index) == ["HA", "HB", "HC"]
        for col in [
            "variable",
            "sens_attendu",
            "mediane_churn",
            "mediane_non_churn",
            "sens_observe",
            "p_value_brute",
            "p_value_holm",
            "auc_orientee",
            "verdict",
        ]:
            assert col in tableau.columns

    def test_hypothese_vraie_confirmee(self) -> None:
        _, tableau = _tester_hypotheses(_df_hypotheses(), _HYPOTHESES_TEST)
        assert tableau.loc["HA", "verdict"] == "confirmée"
        assert tableau.loc["HA", "sens_observe"] == "-"
        assert tableau.loc["HA", "mediane_churn"] < tableau.loc["HA", "mediane_non_churn"]
        assert tableau.loc["HA", "auc_orientee"] > 0.8

    def test_hypothese_contraire_infirmee(self) -> None:
        _, tableau = _tester_hypotheses(_df_hypotheses(), _HYPOTHESES_TEST)
        assert tableau.loc["HB", "verdict"] == "infirmée"
        assert tableau.loc["HB", "auc_orientee"] < 0.2

    def test_hypothese_sans_lien_non_concluante(self) -> None:
        _, tableau = _tester_hypotheses(_df_hypotheses(), _HYPOTHESES_TEST)
        assert tableau.loc["HC", "verdict"] == "non concluante"

    def test_holm_majore_les_p_values(self) -> None:
        _, tableau = _tester_hypotheses(_df_hypotheses(), _HYPOTHESES_TEST)
        assert (tableau["p_value_holm"] >= tableau["p_value_brute"]).all()
        assert (tableau["p_value_holm"] <= 1.0).all()

    def test_haut_de_distribution_teste_l_indicateur(self) -> None:
        hypotheses = [("HH", "csat", "-", "Une satisfaction élevée réduit le risque")]
        _, tableau = _tester_hypotheses(
            _df_hypotheses(), hypotheses, haut_distribution=frozenset({"HH"})
        )
        assert tableau.loc["HH", "mesure"] == "part ≥ quantile"
        # Les churners sont moins souvent dans le haut de la distribution du CSAT
        assert 0.0 <= tableau.loc["HH", "mediane_churn"] < tableau.loc["HH", "mediane_non_churn"]
        assert tableau.loc["HH", "verdict"] == "confirmée"

    def test_variable_absente_ignoree(self) -> None:
        hypotheses = [*_HYPOTHESES_TEST, ("HZ", "inexistante", "+", "Absente")]
        _, tableau = _tester_hypotheses(_df_hypotheses(), hypotheses)
        assert "HZ" not in tableau.index

    def test_aucune_variable_presente_leve_valueerror(self) -> None:
        with pytest.raises(ValueError, match="Aucune hypothèse"):
            _tester_hypotheses(_df_hypotheses(), [("HZ", "inexistante", "+", "Absente")])

    def test_cible_absente_leve_valueerror(self) -> None:
        with pytest.raises(ValueError, match="cible"):
            _tester_hypotheses(_df_hypotheses().drop(columns="churn"), _HYPOTHESES_TEST)
