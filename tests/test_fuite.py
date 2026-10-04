"""Tests du diagnostic de fuite univarié (fuite.diagnostiquer_fuite)."""

import numpy as np
import pandas as pd
import pytest

from churn_saas.fuite import caracteriser_clv, cribler_leurres, diagnostiquer_fuite

_RNG = np.random.default_rng(42)


def _df_diag(n: int = 400) -> pd.DataFrame:
    churn = _RNG.integers(0, 2, n)
    return pd.DataFrame(
        {
            "churn": churn,
            # Quasi-copie de la cible : fuite évidente, sens ↑
            "fuite_haute": churn + _RNG.normal(0, 0.01, n),
            # Anti-corrélée à la cible : sens ↓
            "protectrice": -churn + _RNG.normal(0, 0.5, n),
            "bruit": _RNG.normal(0, 1, n),
            "cat": np.where(churn == 1, "a", "b"),
            "commentaire_csm": ["texte"] * n,
        }
    )


class TestDiagnostiquerFuite:
    def test_fuite_signalee(self) -> None:
        tableau = diagnostiquer_fuite(_df_diag()).set_index("colonne")
        assert tableau.loc["fuite_haute", "suspecte_fuite"]
        assert not tableau.loc["bruit", "suspecte_fuite"]

    def test_auc_symetrisee_et_sens_conserve(self) -> None:
        tableau = diagnostiquer_fuite(_df_diag()).set_index("colonne")
        # Une variable qui fait baisser le churn a une AUC affichée > 0,5
        assert tableau.loc["protectrice", "auc_univariee"] > 0.5
        assert tableau.loc["protectrice", "sens"] == "↓ churn"
        assert tableau.loc["fuite_haute", "sens"] == "↑ churn"

    def test_categorielle_sans_sens(self) -> None:
        tableau = diagnostiquer_fuite(_df_diag()).set_index("colonne")
        assert tableau.loc["cat", "type"] == "catégorielle"
        assert tableau.loc["cat", "sens"] == "—"

    def test_part_manquants(self) -> None:
        df = _df_diag()
        df.loc[df.index[:100], "bruit"] = np.nan
        tableau = diagnostiquer_fuite(df).set_index("colonne")
        assert tableau.loc["bruit", "part_manquants"] == pytest.approx(0.25)

    def test_exclusions_par_defaut_et_personnalisees(self) -> None:
        df = _df_diag()
        assert "commentaire_csm" not in diagnostiquer_fuite(df)["colonne"].values
        colonnes = diagnostiquer_fuite(df, exclure={"bruit"})["colonne"].values
        assert "bruit" not in colonnes
        assert "cat" in colonnes

    def test_cible_absente_leve_valueerror(self) -> None:
        with pytest.raises(ValueError):
            diagnostiquer_fuite(pd.DataFrame({"x": [1, 2]}))


def _df_clv(prospective: bool, n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    mrr = rng.lognormal(6, 1, n)
    anciennete = rng.integers(1, 60, n)
    if prospective:
        # MRR × durée de vie estimée, indépendante de l'ancienneté
        clv = mrr * rng.uniform(6, 60, n)
    else:
        # Cumul des loyers passés, un peu sous MRR × ancienneté (remises, hausses de prix)
        clv = mrr * anciennete * rng.uniform(0.80, 1.0, n)
    return pd.DataFrame(
        {
            "valeur_vie_client_eur": clv,
            "anciennete_mois": anciennete,
            "revenu_mensuel_recurrent_eur": mrr,
            "churn": rng.integers(0, 2, n),
        }
    )


class TestCaracteriserClv:
    def test_clv_realisee(self) -> None:
        res = caracteriser_clv(_df_clv(prospective=False))
        assert res["nature_clv"] == "réalisée"
        assert res["part_clv_sup_cumul"] == 0
        assert res["elasticite_anciennete"] == pytest.approx(1, abs=0.05)

    def test_clv_prospective(self) -> None:
        res = caracteriser_clv(_df_clv(prospective=True))
        assert res["nature_clv"] == "prospective"
        # L'ancien critère r(CLV, MRR × ancienneté) > 0,70 aurait conclu à tort à une CLV cumulée
        assert res["r_mrr_x_anciennete"] > 0.70

    def test_formule_suit_horizon_config(self) -> None:
        from churn_saas import config

        res = caracteriser_clv(_df_clv(prospective=True))
        assert (
            f"× {config.HYPOTHESES_ECONOMIQUES['horizon_mois']} ×" in res["formule_valeur_risque"]
        )


def _df_leurres(n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    churn = rng.integers(0, 2, n)
    usage = rng.normal(0, 1, n) - churn
    return pd.DataFrame(
        {
            "churn": churn,
            "usage": usage,
            "segment": rng.choice(["s1", "s2", "s3"], n),
            # Pur bruit : ni lié au churn, ni à une autre variable
            "leurre": rng.choice(["x", "y", "z"], n),
            # Catégorielle jumelle d'une numérique : tranches de l'usage
            "tranche_usage": pd.cut(usage, bins=8, labels=list("abcdefgh")).astype(str),
            # Liée au churn sans jumelle
            "utile": np.where(rng.random(n) < 0.8, churn, 1 - churn).astype(str),
        }
    )


class TestCriblerLeurres:
    def test_verdicts(self) -> None:
        tableau = cribler_leurres(
            _df_leurres(), cible="churn", colonnes=["leurre", "tranche_usage", "utile"]
        )
        assert tableau.loc["leurre", "verdict_provisoire"].startswith("leurre probable")
        assert tableau.loc["utile", "verdict_provisoire"].startswith("potentiellement utile")

    def test_redondance_categorielle_numerique_detectee(self) -> None:
        tableau = cribler_leurres(_df_leurres(), cible="churn", colonnes=["tranche_usage"])
        # La jumelle est numérique : seul le rapport de corrélation η peut la voir
        assert tableau.loc["tranche_usage", "variable_jumelle"] == "usage"
        assert tableau.loc["tranche_usage", "mesure_redondance"] == "η"
        assert tableau.loc["tranche_usage", "max_redondance"] > 0.9

    def test_redondance_prime_sur_association(self) -> None:
        # Liée au churn ET redondante : c'est le cas du point de vigilance n°5
        tableau = cribler_leurres(_df_leurres(), cible="churn", colonnes=["tranche_usage"])
        assert tableau.loc["tranche_usage", "significatif_BH"]
        assert tableau.loc["tranche_usage", "verdict_provisoire"].startswith("redondant")
