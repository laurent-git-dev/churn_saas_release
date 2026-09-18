"""Tests des fonctions d'analyse exploratoire (eda.py).

Couvre : distribution_cible, univarie_numeriques, univarie_categorielles,
taux_churn_par_modalite, pouvoir_discriminant, matrice_correlation.
"""

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from matplotlib.figure import Figure  # noqa: E402

from churn_saas.eda import (  # noqa: E402
    distribution_cible,
    matrice_correlation,
    pouvoir_discriminant,
    taux_churn_par_modalite,
    univarie_categorielles,
    univarie_numeriques,
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
        assert tableau.loc["Prévalence churn", "valeur"] == "40.0%"

    def test_colonne_absente_leve_valueerror(self) -> None:
        df = pd.DataFrame({"autre": [0, 1, 0]})
        with pytest.raises(ValueError, match="cible"):
            distribution_cible(df)


# ---------------------------------------------------------------------------
# univarie_numeriques
# ---------------------------------------------------------------------------


class TestUnivarieNumeriques:
    def test_retourne_une_figure_par_colonne(self) -> None:
        df = _df_test()
        figs, _ = univarie_numeriques(df, ["anciennete_mois", "connexions_30j"])
        assert len(figs) == 2
        assert all(isinstance(f, Figure) for f in figs)

    def test_tableau_colonnes_attendues(self) -> None:
        df = _df_test()
        _, tableau = univarie_numeriques(df, ["anciennete_mois"])
        attendues = {"asymétrie", "kurtosis", "n_outliers_IQR", "médiane", "moyenne"}
        assert attendues.issubset(set(tableau.columns))

    def test_colonne_absente_ignoree_si_autres_presentes(self) -> None:
        df = _df_test()
        figs, tableau = univarie_numeriques(df, ["anciennete_mois", "inexistante"])
        assert len(figs) == 1
        assert "anciennete_mois" in tableau.index

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
    def test_retourne_une_figure_par_colonne(self) -> None:
        df = _df_test()
        figs, _ = univarie_categorielles(df, ["secteur", "plan"])
        assert len(figs) == 2

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
        # cardinalité=4, cardinalité_norm=2, incohérences=2
        assert tableau.loc["cat", "incohérences_casse"] == 2

    def test_modalites_rares_comptees(self) -> None:
        # "rare" n'apparaît qu'une fois → signalé avec seuil_rare=5
        serie = ["courant"] * 50 + ["rare"]
        df = pd.DataFrame({"cat": serie, "churn": [0] * 51})
        _, tableau = univarie_categorielles(df, ["cat"], seuil_rare=5)
        assert tableau.loc["cat", "n_modalités_rares"] == 1

    def test_toutes_absentes_leve_valueerror(self) -> None:
        df = _df_test()
        with pytest.raises(ValueError):
            univarie_categorielles(df, ["inexistante"])


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
