"""Tests des fonctions de feature engineering métier.

Couvre : ajouter_features_metier, joindre_catalogue, LogAsymetrique, sélection des colonnes
catégorielles (type ``str`` de pandas ≥ 3), NormalisationCategorielle et jointures d'enrichissement.
Invariants vérifiés :
- Division par zéro → NaN (jamais inf ni exception).
- ecart_csat_secteur absente si csat_median_par_secteur est None (anti-fuite).
- Le DataFrame d'entrée n'est jamais modifié en place.
- joindre_catalogue calcule remise_consentie et adequation_plan correctement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churn_saas.features.build import (
    COLONNES_NON_MODELISEES,
    ajouter_features_metier,
    colonnes_features,
    construire_preprocesseur,
    controle_coherence_adoption,
    joindre_catalogue,
    reconstituer_taux_adoption,
)
from churn_saas.features.enrichissement import enrichir_par_pays, enrichir_par_secteur
from churn_saas.features.transformers import LogAsymetrique, NormalisationCategorielle

# ---------------------------------------------------------------------------
# Fixtures partagées
# ---------------------------------------------------------------------------


def _make_df(n: int = 6) -> pd.DataFrame:
    """DataFrame minimal couvrant tous les cas : valeurs normales, zéros et NaN."""
    return pd.DataFrame(
        {
            "sieges_souscrits": [10, 5, 20, 0, 8, 4],
            "utilisateurs_actifs": [8, 5, 4, 3, 0, 4],
            "heures_usage_30j": [40.0, 20.0, float("nan"), 10.0, 0.0, 15.0],
            "connexions_30j": [20, 10, 5, 3, 0, 8],
            "fonctionnalites_utilisees": [6, 8, 12, 4, 3, 6],
            "fonctionnalites_total": [8, 16, 26, 8, 8, 8],
            "revenu_mensuel_recurrent_eur": [100.0, 125.0, 900.0, float("nan"), 80.0, 48.0],
            "derniere_connexion_jours": [5, 45, 90, 10, 35, 20],
            "anciennete_mois": [2, 8, 24, 6, 1, 14],
            "tickets_support_90j": [1, 3, 0, 2, 1, 2],
            "nb_integrations": [3.0, 1.0, float("nan"), 5.0, 0.0, 9.0],
            "csat": [4.0, float("nan"), 3.0, 2.0, 5.0, 4.0],
            "secteur": ["Tech", "Finance", "Tech", "Retail", "Finance", "Tech"],
            "delai_reponse_support_h": [2.0, float("nan"), 4.0, 1.0, 3.0, 2.0],
        }
    )


def _make_catalogue() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "plan": ["Starter", "Pro", "Business", "Enterprise"],
            "prix_mensuel_par_siege_eur": [12, 25, 45, 80],
            "fonctionnalites_incluses": [8, 16, 26, 40],
            "sla_reponse_h": [24, 12, 6, 3],
            "quota_stockage_go": [10, 100, 500, 2000],
            "support_dedie": ["Non", "Non", "Oui", "Oui"],
        }
    )


# ---------------------------------------------------------------------------
# ajouter_features_metier — présence des colonnes
# ---------------------------------------------------------------------------


class TestAjouterFeaturesMetierColonnes:
    _ATTENDUES = [
        "taux_utilisation_sieges",
        "surdimensionnement",
        "intensite_usage_par_utilisateur",
        "connexions_par_utilisateur",
        "taux_couverture_fonctionnelle",
        "arpu_par_siege",
        "recence_normalisee",
        "compte_dormant",
        "pression_support",
        "intensite_support",
        "tranche_anciennete",
        "tranche_integrations",
        "csat_manquant",
        "heures_usage_30j_manquant",
        "delai_reponse_support_h_manquant",
    ]

    def test_toutes_les_colonnes_attendues_sont_presentes(self) -> None:
        result = ajouter_features_metier(_make_df())
        for col in self._ATTENDUES:
            assert col in result.columns, f"Colonne manquante : {col}"

    def test_df_entree_non_modifie(self) -> None:
        """ajouter_features_metier retourne une copie — l'original est intact."""
        df = _make_df()
        cols_avant = set(df.columns)
        ajouter_features_metier(df)
        assert set(df.columns) == cols_avant, "Le DataFrame d'entrée a été modifié en place."


# ---------------------------------------------------------------------------
# ajouter_features_metier — division par zéro → NaN
# ---------------------------------------------------------------------------


class TestDivisionParZero:
    def test_sieges_zero_donne_nan_sur_taux_utilisation(self) -> None:
        """sieges_souscrits = 0 → taux_utilisation_sieges et arpu_par_siege NaN."""
        df = _make_df().iloc[[3]].reset_index(drop=True)  # ligne sieges=0
        result = ajouter_features_metier(df)
        assert pd.isna(
            result["taux_utilisation_sieges"].iloc[0]
        ), "taux_utilisation_sieges doit être NaN quand sieges_souscrits = 0"
        assert pd.isna(
            result["arpu_par_siege"].iloc[0]
        ), "arpu_par_siege doit être NaN quand sieges_souscrits = 0"

    def test_utilisateurs_zero_donne_nan_sur_ratios_par_utilisateur(self) -> None:
        """utilisateurs_actifs = 0 → ratios par utilisateur NaN."""
        df = _make_df().iloc[[4]].reset_index(drop=True)  # ligne utilisateurs=0
        result = ajouter_features_metier(df)
        assert pd.isna(result["intensite_usage_par_utilisateur"].iloc[0])
        assert pd.isna(result["connexions_par_utilisateur"].iloc[0])
        assert pd.isna(result["pression_support"].iloc[0])

    def test_pas_de_inf_dans_le_resultat(self) -> None:
        """Aucune valeur infinie dans les colonnes numériques dérivées."""
        result = ajouter_features_metier(_make_df())
        cols_num = result.select_dtypes("number").columns
        for col in cols_num:
            assert not np.isinf(result[col]).any(), f"Valeur infinie détectée dans : {col}"


# ---------------------------------------------------------------------------
# ajouter_features_metier — anti-fuite ecart_csat_secteur
# ---------------------------------------------------------------------------


class TestEcartCsatSecteurAntiFuite:
    def test_absent_sans_agregat_externe(self) -> None:
        """ecart_csat_secteur n'est PAS calculée si csat_median_par_secteur est None.

        C'est la garantie anti-fuite : sans agrégat externe appris sur le train,
        la feature n'existe pas — on ne peut pas calculer accidentellement la médiane
        sur le jeu complet.
        """
        result = ajouter_features_metier(_make_df())
        assert "ecart_csat_secteur" not in result.columns, (
            "ecart_csat_secteur NE DOIT PAS être calculée sans csat_median_par_secteur "
            "(risque de fuite si la médiane était calculée sur le jeu complet)."
        )

    def test_calcul_correct_avec_agregat_externe(self) -> None:
        """ecart_csat_secteur = csat − médiane_secteur (issue de l'agrégat externe)."""
        medians = {"Tech": 3.5, "Finance": 4.0, "Retail": 3.0}
        result = ajouter_features_metier(_make_df(), csat_median_par_secteur=medians)
        assert "ecart_csat_secteur" in result.columns
        # Ligne 0 : csat=4.0, secteur=Tech, médiane=3.5 → écart=0.5
        assert abs(result["ecart_csat_secteur"].iloc[0] - 0.5) < 1e-9
        # Ligne 1 : csat=NaN → écart NaN
        assert pd.isna(result["ecart_csat_secteur"].iloc[1])

    def test_secteur_inconnu_produit_nan(self) -> None:
        """Secteur absent du dict → ecart_csat_secteur NaN (pas d'exception)."""
        medians = {"Tech": 3.5}  # Finance et Retail absents
        result = ajouter_features_metier(_make_df(), csat_median_par_secteur=medians)
        # Ligne 1 : secteur=Finance, absent du dict → NaN
        assert pd.isna(result["ecart_csat_secteur"].iloc[1])


# ---------------------------------------------------------------------------
# ajouter_features_metier — indicateurs de manquance
# ---------------------------------------------------------------------------


class TestIndicateursManquance:
    def test_csat_manquant_reflète_nan(self) -> None:
        result = ajouter_features_metier(_make_df())
        # Ligne 0 : csat=4.0 → csat_manquant=False ; ligne 1 : csat=NaN → True
        assert not result["csat_manquant"].iloc[0]
        assert result["csat_manquant"].iloc[1]

    def test_heures_usage_manquant_reflète_nan(self) -> None:
        result = ajouter_features_metier(_make_df())
        # Ligne 2 : heures_usage_30j=NaN → manquant=True
        assert result["heures_usage_30j_manquant"].iloc[2]


# ---------------------------------------------------------------------------
# ajouter_features_metier — tranches catégorielles
# ---------------------------------------------------------------------------


class TestTranches:
    def _df_anciennete(self, mois: list[int]) -> pd.DataFrame:
        base = _make_df().iloc[[0]].reset_index(drop=True)
        dfs = []
        for m in mois:
            row = base.copy()
            row["anciennete_mois"] = m
            dfs.append(row)
        return pd.concat(dfs, ignore_index=True)

    def test_tranches_anciennete(self) -> None:
        df = self._df_anciennete([1, 2, 3, 12, 13, 36])
        result = ajouter_features_metier(df)
        attendu = ["onboarding", "onboarding", "installation", "installation", "mature", "mature"]
        assert list(result["tranche_anciennete"]) == attendu

    def test_tranches_integrations(self) -> None:
        df = _make_df()
        result = ajouter_features_metier(df)
        # Ligne 0 : nb_integrations=3.0 → faible ; ligne 4 : 0.0 → aucune ; ligne 5 : 9.0 → forte
        assert result["tranche_integrations"].iloc[0] == "faible"
        assert result["tranche_integrations"].iloc[4] == "aucune"
        assert result["tranche_integrations"].iloc[5] == "forte"


# ---------------------------------------------------------------------------
# joindre_catalogue
# ---------------------------------------------------------------------------


class TestJoindreCatalogue:
    def _df_client(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "plan": ["Starter", "Pro", "Business"],
                "revenu_mensuel_recurrent_eur": [60.0, 100.0, 900.0],
                # Starter : 12 € × 5 = 60 → remise = 0 %
                # Pro     : 25 € × 5 = 125, MRR=100 → remise = 20 %
                # Business: 45 € × 20 = 900, MRR=900 → remise = 0 %
                "sieges_souscrits": [5, 5, 20],
                "utilisateurs_actifs": [4, 5, 18],
            }
        )

    def test_colonnes_derivees_presentes(self) -> None:
        result = joindre_catalogue(self._df_client(), _make_catalogue())
        assert "remise_consentie" in result.columns
        assert "adequation_plan" in result.columns

    def test_colonnes_catalogue_presentes(self) -> None:
        result = joindre_catalogue(self._df_client(), _make_catalogue())
        assert "prix_mensuel_par_siege_eur" in result.columns
        assert "fonctionnalites_incluses" in result.columns

    def test_remise_nulle_au_plein_tarif(self) -> None:
        """MRR = prix_catalogue × sièges → remise_consentie = 0."""
        result = joindre_catalogue(self._df_client(), _make_catalogue())
        assert (
            abs(result["remise_consentie"].iloc[0]) < 1e-9
        )  # Starter : 60 / (12×5) = 1 → remise=0

    def test_remise_positive_avec_rabais(self) -> None:
        """MRR < prix_catalogue × sièges → remise_consentie > 0."""
        result = joindre_catalogue(self._df_client(), _make_catalogue())
        assert result["remise_consentie"].iloc[1] > 0  # Pro : 100 < 125 → remise > 0

    def test_remise_division_par_zero_donne_nan(self) -> None:
        """sieges_souscrits = 0 → remise_consentie NaN."""
        df = pd.DataFrame(
            {
                "plan": ["Starter"],
                "revenu_mensuel_recurrent_eur": [60.0],
                "sieges_souscrits": [0],
                "utilisateurs_actifs": [0],
            }
        )
        result = joindre_catalogue(df, _make_catalogue())
        assert pd.isna(result["remise_consentie"].iloc[0])

    def test_adequation_plan_sur_dimensionne(self) -> None:
        """Faible taux d'utilisation → sur-dimensionné."""
        df = pd.DataFrame(
            {
                "plan": ["Enterprise"],
                "revenu_mensuel_recurrent_eur": [800.0],
                "sieges_souscrits": [10],
                "utilisateurs_actifs": [3],  # taux = 0.3 < 0.5 → sur-dimensionné
            }
        )
        result = joindre_catalogue(df, _make_catalogue())
        assert result["adequation_plan"].iloc[0] == "sur-dimensionne"

    def test_adequation_plan_sous_dimensionne(self) -> None:
        """Taux quasi-saturation → sous-dimensionné."""
        df = pd.DataFrame(
            {
                "plan": ["Starter"],
                "revenu_mensuel_recurrent_eur": [120.0],
                "sieges_souscrits": [10],
                "utilisateurs_actifs": [10],  # taux = 1.0 ≥ 0.9 → sous-dimensionné
            }
        )
        result = joindre_catalogue(df, _make_catalogue())
        assert result["adequation_plan"].iloc[0] == "sous-dimensionne"

    def test_adequation_plan_adapte(self) -> None:
        """Taux intermédiaire → adapté."""
        df = pd.DataFrame(
            {
                "plan": ["Pro"],
                "revenu_mensuel_recurrent_eur": [125.0],
                "sieges_souscrits": [10],
                "utilisateurs_actifs": [7],  # taux = 0.7 → adapté
            }
        )
        result = joindre_catalogue(df, _make_catalogue())
        assert result["adequation_plan"].iloc[0] == "adapte"

    def test_jointure_insensible_a_la_casse(self) -> None:
        """La normalisation titre tolère les variantes de casse dans le champ plan."""
        df = pd.DataFrame(
            {
                "plan": ["starter", "PRO", "BUSINESS"],
                "revenu_mensuel_recurrent_eur": [60.0, 125.0, 900.0],
                "sieges_souscrits": [5, 5, 20],
                "utilisateurs_actifs": [4, 4, 18],
            }
        )
        result = joindre_catalogue(df, _make_catalogue())
        assert result["prix_mensuel_par_siege_eur"].iloc[0] == 12
        assert result["prix_mensuel_par_siege_eur"].iloc[1] == 25

    def test_df_entree_non_modifie(self) -> None:
        """joindre_catalogue retourne une copie — l'original est intact."""
        df = self._df_client()
        cols_avant = set(df.columns)
        joindre_catalogue(df, _make_catalogue())
        assert set(df.columns) == cols_avant


# ---------------------------------------------------------------------------
# reconstituer_taux_adoption
# ---------------------------------------------------------------------------


class TestReconstituerTauxAdoption:
    def _df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "taux_adoption_pct": [50.0, np.nan, np.nan, np.nan],
                "utilisateurs_actifs": [5, 8, 3, np.nan],
                "sieges_souscrits": [10, 10, 0, 10],
            }
        )

    def test_manquant_recalcule_renseigne_intact(self) -> None:
        df, n = reconstituer_taux_adoption(self._df())
        assert n == 1
        assert df["taux_adoption_pct"].iloc[0] == pytest.approx(50.0)
        assert df["taux_adoption_pct"].iloc[1] == pytest.approx(80.0)

    def test_non_calculable_reste_manquant(self) -> None:
        # Sièges à 0 (division impossible) ou utilisateurs inconnus : on n'invente rien
        df, _ = reconstituer_taux_adoption(self._df())
        assert df["taux_adoption_pct"].iloc[2:].isna().all()

    def test_idempotent_et_sans_modification_en_place(self) -> None:
        source = self._df()
        df, _ = reconstituer_taux_adoption(source)
        _, n_second = reconstituer_taux_adoption(df)
        assert n_second == 0
        assert source["taux_adoption_pct"].isna().sum() == 3

    def test_appliquee_par_ajouter_features_metier(self) -> None:
        df = _make_df()
        df["taux_adoption_pct"] = np.nan  # sièges du 1er client : 10, utilisateurs : 8
        result = ajouter_features_metier(df)
        assert result["taux_adoption_pct"].iloc[0] == pytest.approx(80.0)


class TestIntensiteSupport:
    def test_tickets_rapportes_a_l_anciennete(self) -> None:
        result = ajouter_features_metier(_make_df())
        # 1er client : 1 ticket sur 90 j, 2 mois d'ancienneté
        assert result["intensite_support"].iloc[0] == pytest.approx(0.5)
        assert result["intensite_support"].iloc[2] == pytest.approx(0.0)

    def test_anciennete_nulle_ou_negative_donne_nan(self) -> None:
        df = _make_df()
        df.loc[0, "anciennete_mois"] = 0
        df.loc[1, "anciennete_mois"] = -3
        result = ajouter_features_metier(df)
        assert result["intensite_support"].iloc[:2].isna().all()
        assert not np.isinf(result["intensite_support"]).any()


class TestControleCoherenceAdoption:
    def _df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                # Écarts attendus : 0, 0.5, 5, non comparable (sièges à 0), non comparable (NaN)
                "taux_adoption_pct": [50.0, 80.5, 25.0, 40.0, np.nan],
                "utilisateurs_actifs": [5, 8, 3, 2, 4],
                "sieges_souscrits": [10, 10, 15, 0, 10],
            }
        )

    def test_ecarts_et_part_d_incoherences(self) -> None:
        bilan = controle_coherence_adoption(self._df())
        assert len(bilan) == 1
        ligne = bilan.iloc[0]
        assert ligne["n_comparables"] == 3
        assert ligne["n_incoherences"] == 1
        assert ligne["part_incoherences"] == pytest.approx(1 / 3)
        assert ligne["ecart_max_pts"] == pytest.approx(5.0)
        assert ligne["ecart_moyen_pts"] == pytest.approx(5.5 / 3)

    def test_ecart_d_un_point_exactement_reste_coherent(self) -> None:
        df = pd.DataFrame(
            {"taux_adoption_pct": [51.0], "utilisateurs_actifs": [5], "sieges_souscrits": [10]}
        )
        assert controle_coherence_adoption(df).iloc[0]["n_incoherences"] == 0

    def test_colonnes_absentes_bilan_vide(self) -> None:
        bilan = controle_coherence_adoption(pd.DataFrame({"csat": [3.0]}))
        assert bilan.iloc[0]["n_comparables"] == 0
        assert np.isnan(bilan.iloc[0]["part_incoherences"])

    def test_entree_non_modifiee(self) -> None:
        source = self._df()
        colonnes_avant = list(source.columns)
        controle_coherence_adoption(source)
        assert list(source.columns) == colonnes_avant


# ---------------------------------------------------------------------------
# LogAsymetrique
# ---------------------------------------------------------------------------


class TestLogAsymetrique:
    @staticmethod
    def _df() -> pd.DataFrame:
        rng = np.random.default_rng(42)
        return pd.DataFrame(
            {
                "symetrique": rng.normal(0.0, 1.0, 500),
                "queue_droite": rng.lognormal(0.0, 1.5, 500),
                "binaire_rare": (rng.random(500) < 0.05).astype(float),
            }
        )

    def test_seules_les_colonnes_asymetriques_non_binaires_sont_retenues(self) -> None:
        t = LogAsymetrique(seuil_asymetrie=1.0).fit(self._df())
        assert t.colonnes_log_.tolist() == [False, True, False]

    def test_transformation_reduit_l_asymetrie_et_preserve_le_reste(self) -> None:
        df = self._df()
        sortie = LogAsymetrique().fit_transform(df)
        assert pd.Series(sortie[:, 1]).skew() < df["queue_droite"].skew()
        np.testing.assert_allclose(sortie[:, [0, 2]], df[["symetrique", "binaire_rare"]])

    def test_log_signe_defini_pour_une_valeur_negative(self) -> None:
        df = self._df()
        t = LogAsymetrique().fit(df)
        sortie = t.transform(pd.DataFrame([[0.0, -5.0, 1.0]], columns=df.columns))
        assert sortie[0, 1] == pytest.approx(-np.log1p(5.0))

    def test_entree_non_modifiee_et_noms_conserves(self) -> None:
        df = self._df()
        copie = df.copy()
        t = LogAsymetrique().fit(df)
        t.transform(df)
        pd.testing.assert_frame_equal(df, copie)
        assert list(t.get_feature_names_out()) == list(df.columns)


# ---------------------------------------------------------------------------
# Colonnes catégorielles : type str (pandas ≥ 3), normalisation, enrichissement
# ---------------------------------------------------------------------------


class TestColonnesCategorielles:
    """Régression : sous pandas 3, les colonnes texte (type ``str``) étaient ignorées."""

    @pytest.mark.parametrize("dtype", [object, "string", "str"])
    def test_colonne_texte_retenue_quel_que_soit_son_type(self, dtype: object) -> None:
        df = pd.DataFrame(
            {"mrr": [1.0, 2.0], "plan": pd.Series(["pro", "starter"], dtype=dtype)}  # type: ignore[call-overload]
        )
        num, cat = colonnes_features(df)
        assert num == ["mrr"]
        assert cat == ["plan"]

    def test_colonnes_non_modelisees_exclues(self) -> None:
        df = pd.DataFrame(
            {
                "mrr": [1.0, 2.0],
                "jour_souscription": pd.Series(["lundi", "mardi"], dtype="str"),
                "date_souscription": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            }
        )
        num, cat = colonnes_features(df)
        assert not (set(num) | set(cat)) & COLONNES_NON_MODELISEES

    def test_preprocesseur_encode_les_colonnes_texte(self) -> None:
        df = pd.DataFrame(
            {"mrr": [1.0, 2.0, 3.0], "plan": pd.Series(["pro", "starter", "pro"], dtype="str")}
        )
        noms = construire_preprocesseur(df).fit(df).get_feature_names_out()
        assert {"plan_pro", "plan_starter"} <= set(noms)


class TestNormalisationCategorielle:
    def test_variantes_fusionnees_en_une_modalite(self) -> None:
        df = pd.DataFrame(
            {"mrr": [1.0, 2.0, 3.0], "plan": pd.Series(["Pro", " pro ", "PRO"], dtype="str")}
        )
        noms = construire_preprocesseur(df).fit(df).get_feature_names_out()
        assert [n for n in noms if n.startswith("plan_")] == ["plan_pro"]

    def test_variante_inconnue_du_train_reconnue_a_l_inference(self) -> None:
        train = pd.DataFrame(
            {"mrr": [1.0, 2.0], "plan": pd.Series(["pro", "starter"], dtype="str")}
        )
        pre = construire_preprocesseur(train).fit(train)
        test = pd.DataFrame({"mrr": [1.0], "plan": pd.Series([" Pro"], dtype="str")})
        noms = list(pre.get_feature_names_out())
        assert pre.transform(test)[0, noms.index("plan_pro")] == 1.0

    def test_manquant_preserve(self) -> None:
        df = pd.DataFrame({"plan": pd.Series(["Pro", None], dtype="str")})
        resultat = NormalisationCategorielle().fit_transform(df)
        assert resultat.loc[0, "plan"] == "pro"
        assert pd.isna(resultat.loc[1, "plan"])

    def test_noms_de_sortie_identiques(self) -> None:
        df = pd.DataFrame({"plan": ["Pro"], "pays": ["France"]})
        t = NormalisationCategorielle().fit(df)
        assert list(t.get_feature_names_out()) == ["plan", "pays"]


class TestEnrichissementInsensibleALaCasse:
    def test_secteur_variantes_jointes(self) -> None:
        df = pd.DataFrame({"secteur": ["Santé", " SANTÉ ", "santé"]})
        resultat = enrichir_par_secteur(df)
        assert resultat["taux_churn_median_saas_pct"].nunique() == 1
        attendu = enrichir_par_secteur(pd.DataFrame({"secteur": ["Santé"]}))
        assert resultat["dynamique_croissance"].iloc[0] == attendu["dynamique_croissance"].iloc[0]

    def test_alias_secteur_rattache_au_referentiel(self) -> None:
        resultat = enrichir_par_secteur(pd.DataFrame({"secteur": [" Tech", "Technologie"]}))
        assert resultat["taux_churn_median_saas_pct"].nunique() == 1

    def test_pays_variantes_et_repli(self) -> None:
        df = pd.DataFrame({"pays": ["france", " FRANCE", "Atlantide", None]})
        resultat = enrichir_par_pays(df)
        assert list(resultat["zone_reglementaire"]) == ["ue_rgpd", "ue_rgpd", "autre", "autre"]
