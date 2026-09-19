"""Tests des fonctions de feature engineering métier.

Couvre : ajouter_features_metier, joindre_catalogue.
Invariants vérifiés :
- Division par zéro → NaN (jamais inf ni exception).
- ecart_csat_secteur absente si csat_median_par_secteur est None (anti-fuite).
- Le DataFrame d'entrée n'est jamais modifié en place.
- joindre_catalogue calcule remise_consentie et adequation_plan correctement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from churn_saas.features.build import ajouter_features_metier, joindre_catalogue

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
