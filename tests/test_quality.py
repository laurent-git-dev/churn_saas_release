"""Tests des fonctions de contrôle qualité des données.

Couvre : analyser_doublons, coercer_numeriques, parser_dates,
detecter_valeurs_impossibles, proposer_renommage.
"""


import pandas as pd
import pytest

from churn_saas.data.quality import (
    analyser_doublons,
    coercer_numeriques,
    detecter_valeurs_impossibles,
    parser_dates,
    proposer_renommage,
)

# ---------------------------------------------------------------------------
# analyser_doublons
# ---------------------------------------------------------------------------


class TestAnalyserDoublons:
    def test_aucun_doublon(self) -> None:
        df = pd.DataFrame({"client_id": ["A", "B", "C"], "mrr": [100, 200, 300]})
        res = analyser_doublons(df)
        assert res["nb_doublons_exacts"] == 0
        assert res["nb_doublons_cle_metier"] == 0

    def test_doublons_exacts(self) -> None:
        df = pd.DataFrame({"client_id": ["A", "A", "B"], "mrr": [100, 100, 200]})
        res = analyser_doublons(df)
        assert res["nb_doublons_exacts"] == 2  # les 2 lignes "A/100" sont marquées
        assert res["nb_doublons_cle_metier"] == 2

    def test_doublons_cle_seulement(self) -> None:
        # Même client_id, données différentes : doublon clé mais pas exact
        df = pd.DataFrame({"client_id": ["A", "A", "B"], "mrr": [100, 150, 200]})
        res = analyser_doublons(df)
        assert res["nb_doublons_exacts"] == 0
        assert res["nb_doublons_cle_metier"] == 2
        assert res["nb_doublons_cle_metier_non_exacts"] == 2

    def test_doublons_mixtes(self) -> None:
        # "A/100" apparaît deux fois (exact) ; "B" apparaît deux fois avec des valeurs diff (clé)
        df = pd.DataFrame({"client_id": ["A", "A", "B", "B"], "mrr": [100, 100, 200, 250]})
        res = analyser_doublons(df)
        assert res["nb_doublons_exacts"] == 2
        assert res["nb_doublons_cle_metier"] == 4
        assert res["nb_doublons_cle_metier_non_exacts"] == 2

    def test_cle_metier_absente(self) -> None:
        df = pd.DataFrame({"autre_col": ["A", "B"], "mrr": [100, 200]})
        res = analyser_doublons(df, cle_metier="client_id")
        assert res["nb_doublons_cle_metier"] == 0  # ne plante pas

    def test_dataframe_vide(self) -> None:
        df = pd.DataFrame(
            {"client_id": pd.Series([], dtype=str), "mrr": pd.Series([], dtype=float)}
        )
        res = analyser_doublons(df)
        assert res["nb_total_lignes"] == 0
        assert res["nb_doublons_exacts"] == 0

    def test_exemples_sont_des_dataframes(self) -> None:
        df = pd.DataFrame({"client_id": ["A", "A"], "mrr": [100, 100]})
        res = analyser_doublons(df)
        assert isinstance(res["exemples_doublons_exacts"], pd.DataFrame)
        assert isinstance(res["exemples_doublons_cle_metier"], pd.DataFrame)

    def test_traitements_sont_des_chaines(self) -> None:
        df = pd.DataFrame({"client_id": ["A"], "mrr": [100]})
        res = analyser_doublons(df)
        assert isinstance(res["traitement_doublons_exacts"], str)
        assert isinstance(res["traitement_doublons_cle_metier"], str)
        assert len(res["traitement_doublons_exacts"]) > 0


# ---------------------------------------------------------------------------
# coercer_numeriques
# ---------------------------------------------------------------------------


class TestCoercerNumeriques:
    def _coerce(self, valeurs: list, col: str = "x") -> tuple[pd.DataFrame, pd.DataFrame]:
        df = pd.DataFrame({col: valeurs})
        return coercer_numeriques(df, [col])

    def test_entiers_directs(self) -> None:
        df_out, rapport = self._coerce(["1", "2", "3"])
        assert list(df_out["x"]) == [1.0, 2.0, 3.0]
        assert rapport.loc["x", "nb_directes"] == 3
        assert rapport.loc["x", "nb_irrecuperables"] == 0

    def test_decimale_point(self) -> None:
        df_out, _ = self._coerce(["1.5", "2.75"])
        assert df_out["x"].iloc[0] == pytest.approx(1.5)

    def test_decimale_virgule_francaise(self) -> None:
        df_out, rapport = self._coerce(["1,5", "3,14"])
        assert df_out["x"].iloc[0] == pytest.approx(1.5)
        assert df_out["x"].iloc[1] == pytest.approx(3.14)
        assert rapport.loc["x", "nb_reparees"] == 2

    def test_separateur_milliers_espace(self) -> None:
        df_out, rapport = self._coerce(["1 000", "12 345"])
        assert df_out["x"].iloc[0] == pytest.approx(1000.0)
        assert rapport.loc["x", "nb_reparees"] == 2

    def test_separateur_milliers_nbsp(self) -> None:
        # Espace insécable \xa0
        df_out, rapport = self._coerce(["1\xa0000", "2\xa0500"])
        assert df_out["x"].iloc[0] == pytest.approx(1000.0)
        assert rapport.loc["x", "nb_reparees"] == 2

    def test_format_fr_avec_milliers_et_decimale(self) -> None:
        # "1.234,56" → 1234.56
        df_out, rapport = self._coerce(["1.234,56"])
        assert df_out["x"].iloc[0] == pytest.approx(1234.56)
        assert rapport.loc["x", "nb_reparees"] == 1

    def test_format_en_avec_milliers_et_decimale(self) -> None:
        # "1,234.56" → 1234.56
        df_out, rapport = self._coerce(["1,234.56"])
        assert df_out["x"].iloc[0] == pytest.approx(1234.56)
        assert rapport.loc["x", "nb_reparees"] == 1

    def test_symbole_pourcent(self) -> None:
        df_out, rapport = self._coerce(["42%", "100%"])
        assert df_out["x"].iloc[0] == pytest.approx(42.0)
        assert rapport.loc["x", "nb_reparees"] == 2

    def test_symbole_euro(self) -> None:
        df_out, rapport = self._coerce(["1234€", "500€"])
        assert df_out["x"].iloc[0] == pytest.approx(1234.0)
        assert rapport.loc["x", "nb_reparees"] == 2

    def test_marqueurs_manquants(self) -> None:
        valeurs = ["N/A", "n/a", "-", "", "null", "NULL", "NaN", "NA", "nd"]
        df_out, rapport = self._coerce(valeurs)
        assert df_out["x"].isna().all()
        assert rapport.loc["x", "nb_manquants"] == len(valeurs)
        assert rapport.loc["x", "nb_irrecuperables"] == 0

    def test_irrecuperable(self) -> None:
        df_out, rapport = self._coerce(["abc", "xyz", "???"])
        assert df_out["x"].isna().all()
        assert rapport.loc["x", "nb_irrecuperables"] == 3

    def test_nan_original_compté_comme_manquant(self) -> None:
        df = pd.DataFrame({"x": [1.0, float("nan"), 3.0]})
        _, rapport = coercer_numeriques(df, ["x"])
        assert rapport.loc["x", "nb_manquants"] == 1

    def test_colonne_absente_ignoree(self) -> None:
        df = pd.DataFrame({"x": ["1", "2"]})
        df_out, rapport = coercer_numeriques(df, ["x", "inexistante"])
        assert "x" in rapport.index
        assert "inexistante" not in rapport.index

    def test_cas_ambigu_1234_traite_comme_milliers(self) -> None:
        # "1,234" → 1234 (séparateur de milliers EN, heuristique : 3 chiffres après virgule)
        df_out, _ = self._coerce(["1,234"])
        assert df_out["x"].iloc[0] == pytest.approx(1234.0)

    def test_notation_scientifique(self) -> None:
        df_out, rapport = self._coerce(["1e5", "2.5e3"])
        assert df_out["x"].iloc[0] == pytest.approx(100000.0)
        assert rapport.loc["x", "nb_directes"] == 2

    def test_df_non_modifie_en_place(self) -> None:
        df = pd.DataFrame({"x": ["1,5", "2,5"]})
        df_original = df.copy()
        coercer_numeriques(df, ["x"])
        pd.testing.assert_frame_equal(df, df_original)


# ---------------------------------------------------------------------------
# parser_dates
# ---------------------------------------------------------------------------


class TestParserDates:
    def test_format_iso(self) -> None:
        df = pd.DataFrame({"d": ["2021-01-15", "2021-12-31"]})
        df_out, rapport = parser_dates(df, ["d"])
        assert pd.api.types.is_datetime64_any_dtype(df_out["d"])
        assert rapport.loc["d", "nb_parsees"] == 2
        assert rapport.loc["d", "nb_irrecuperables"] == 0

    def test_format_fr_dd_mm_yyyy(self) -> None:
        # "13/02/2021" : premier champ > 12 → DD/MM détecté sans ambiguïté
        df = pd.DataFrame({"d": ["13/02/2021", "15/06/2021"]})
        df_out, rapport = parser_dates(df, ["d"])
        assert rapport.loc["d", "format_detecte"] == "%d/%m/%Y"
        assert df_out["d"].iloc[0] == pd.Timestamp("2021-02-13")

    def test_detection_format_par_valeur_inequivoque(self) -> None:
        # Mélange : "13/02/2021" lève l'ambiguïté → convention DD/MM
        df = pd.DataFrame({"d": ["01/02/2021", "13/02/2021"]})
        df_out, rapport = parser_dates(df, ["d"])
        assert rapport.loc["d", "format_detecte"] == "%d/%m/%Y"
        # "01/02/2021" parsé en JJ/MM → 1er février
        assert df_out["d"].iloc[0] == pd.Timestamp("2021-02-01")

    def test_dates_toutes_ambigues_convention_par_defaut(self) -> None:
        # Toutes les dates ont jour ≤ 12 ET mois ≤ 12 → convention DD/MM par défaut
        df = pd.DataFrame({"d": ["01/02/2021", "03/04/2021", "05/06/2021"]})
        df_out, rapport = parser_dates(df, ["d"])
        assert rapport.loc["d", "nb_ambigues"] == 3
        assert "DD/MM" in rapport.loc["d", "convention_retenue"]
        assert "défaut" in rapport.loc["d", "convention_retenue"]

    def test_irrecuperable_comptabilise(self) -> None:
        df = pd.DataFrame({"d": ["2021-01-15", "pas-une-date", "2021-06-30"]})
        _, rapport = parser_dates(df, ["d"])
        assert rapport.loc["d", "nb_parsees"] == 2
        assert rapport.loc["d", "nb_irrecuperables"] == 1

    def test_colonne_absente_ignoree(self) -> None:
        df = pd.DataFrame({"d": ["2021-01-15"]})
        df_out, rapport = parser_dates(df, ["d", "inexistante"])
        assert "inexistante" not in rapport.index

    def test_df_non_modifie_en_place(self) -> None:
        df = pd.DataFrame({"d": ["2021-01-15"]})
        df_original = df.copy()
        parser_dates(df, ["d"])
        pd.testing.assert_frame_equal(df, df_original)

    def test_serie_vide(self) -> None:
        df = pd.DataFrame({"d": pd.Series([], dtype=str)})
        df_out, rapport = parser_dates(df, ["d"])
        assert rapport.loc["d", "nb_parsees"] == 0


# ---------------------------------------------------------------------------
# detecter_valeurs_impossibles
# ---------------------------------------------------------------------------


class TestDetecterValeursImpossibles:
    def test_dataframe_propre(self) -> None:
        df = pd.DataFrame(
            {
                "utilisateurs_actifs": [5, 10],
                "sieges_souscrits": [10, 20],
                "taux_adoption_pct": [50.0, 80.0],
                "mrr": [1000.0, 2000.0],
            }
        )
        res = detecter_valeurs_impossibles(df)
        assert len(res) == 0

    def test_utilisateurs_superieur_sieges(self) -> None:
        df = pd.DataFrame({"utilisateurs_actifs": [15, 5], "sieges_souscrits": [10, 10]})
        res = detecter_valeurs_impossibles(df)
        regle = res[res["colonne_ou_paire"] == "utilisateurs_actifs / sieges_souscrits"]
        assert len(regle) == 1
        assert regle.iloc[0]["nb_lignes_concernees"] == 1

    def test_taux_hors_bornes(self) -> None:
        df = pd.DataFrame({"taux_adoption_pct": [-5.0, 50.0, 150.0]})
        res = detecter_valeurs_impossibles(df)
        regle = res[res["colonne_ou_paire"] == "taux_adoption_pct"]
        assert regle.iloc[0]["nb_lignes_concernees"] == 2

    def test_mrr_negatif(self) -> None:
        df = pd.DataFrame({"mrr": [1000.0, -500.0, 2000.0]})
        res = detecter_valeurs_impossibles(df)
        assert any("mrr" in str(r) for r in res["colonne_ou_paire"])
        ligne = res[res["colonne_ou_paire"] == "mrr"]
        assert ligne.iloc[0]["nb_lignes_concernees"] == 1

    def test_anciennete_incoherente(self) -> None:
        # date_souscription il y a ~60 mois, anciennete_mois = 6 → anomalie
        df = pd.DataFrame(
            {
                "anciennete_mois": [6],
                "date_souscription": [pd.Timestamp("2021-01-01")],
            }
        )
        res = detecter_valeurs_impossibles(df)
        regle = res[res["regle"].str.contains("anciennete")]
        assert len(regle) == 1

    def test_anciennete_coherente(self) -> None:
        # date_souscription il y a ~2 mois, anciennete_mois = 2 → pas d'anomalie
        df = pd.DataFrame(
            {
                "anciennete_mois": [2],
                "date_souscription": [pd.Timestamp.now() - pd.Timedelta(days=60)],
            }
        )
        res = detecter_valeurs_impossibles(df)
        regle = res[res["regle"].str.contains("anciennete")]
        assert len(regle) == 0

    def test_colonnes_manquantes_silencieuses(self) -> None:
        # Sans les colonnes spéciales, pas de plantage
        df = pd.DataFrame({"autre": [1, 2, 3]})
        res = detecter_valeurs_impossibles(df)
        assert len(res) == 0

    def test_valeurs_nan_ignorees(self) -> None:
        df = pd.DataFrame(
            {
                "utilisateurs_actifs": [float("nan"), 5.0],
                "sieges_souscrits": [float("nan"), 10.0],
            }
        )
        res = detecter_valeurs_impossibles(df)
        assert len(res) == 0

    def test_retour_est_dataframe(self) -> None:
        df = pd.DataFrame({"mrr": [100.0]})
        res = detecter_valeurs_impossibles(df)
        assert isinstance(res, pd.DataFrame)
        assert set(res.columns) >= {"regle", "colonne_ou_paire", "nb_lignes_concernees", "exemple"}


# ---------------------------------------------------------------------------
# proposer_renommage
# ---------------------------------------------------------------------------


class TestProposerRenommage:
    def test_nom_conforme(self) -> None:
        df = pd.DataFrame({"mrr_eur": [1]})
        res = proposer_renommage(df)
        ligne = res[res["nom_original"] == "mrr_eur"].iloc[0]
        assert not ligne["modifie"]
        assert ligne["raison"] == "conforme"

    def test_mise_en_minuscules(self) -> None:
        df = pd.DataFrame({"MRR": [1]})
        res = proposer_renommage(df)
        assert res.iloc[0]["nom_propose"] == "mrr"
        assert res.iloc[0]["modifie"]

    def test_suppression_accent(self) -> None:
        df = pd.DataFrame({"ancienneté_mois": [1]})
        res = proposer_renommage(df)
        assert res.iloc[0]["nom_propose"] == "anciennete_mois"
        assert "accent" in res.iloc[0]["raison"]

    def test_espace_remplace_par_underscore(self) -> None:
        df = pd.DataFrame({"date souscription": [1]})
        res = proposer_renommage(df)
        assert res.iloc[0]["nom_propose"] == "date_souscription"

    def test_tiret_remplace_par_underscore(self) -> None:
        df = pd.DataFrame({"nb-utilisateurs": [1]})
        res = proposer_renommage(df)
        assert res.iloc[0]["nom_propose"] == "nb_utilisateurs"

    def test_ajout_suffixe_eur(self) -> None:
        df = pd.DataFrame({"montant_contrat": [1]})
        res = proposer_renommage(df)
        assert res.iloc[0]["nom_propose"].endswith("_eur")

    def test_pas_de_suffixe_eur_double(self) -> None:
        df = pd.DataFrame({"montant_contrat_eur": [1]})
        res = proposer_renommage(df)
        # Ne doit pas devenir "montant_contrat_eur_eur"
        assert res.iloc[0]["nom_propose"] == "montant_contrat_eur"

    def test_ajout_suffixe_pct(self) -> None:
        df = pd.DataFrame({"taux_adoption": [1]})
        res = proposer_renommage(df)
        assert res.iloc[0]["nom_propose"].endswith("_pct")

    def test_pas_de_suffixe_pct_double(self) -> None:
        df = pd.DataFrame({"taux_adoption_pct": [1]})
        res = proposer_renommage(df)
        assert res.iloc[0]["nom_propose"] == "taux_adoption_pct"

    def test_retour_est_dataframe_avec_bonnes_colonnes(self) -> None:
        df = pd.DataFrame({"a": [1], "b": [2]})
        res = proposer_renommage(df)
        assert set(res.columns) >= {"nom_original", "nom_propose", "modifie", "raison"}
        assert len(res) == 2

    def test_cas_tordu_accents_majuscules_espaces(self) -> None:
        df = pd.DataFrame({"Chiffre d'Affaires (€)": [1]})
        res = proposer_renommage(df)
        nom = res.iloc[0]["nom_propose"]
        assert nom == nom.lower()
        assert " " not in nom
        assert "é" not in nom
