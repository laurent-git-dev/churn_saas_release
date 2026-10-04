"""Tests des fonctions de contrôle qualité des données.

Couvre : analyser_doublons, analyser_manquance, coercer_numeriques, profil_compact,
parser_dates, detecter_valeurs_impossibles, proposer_renommage.
"""

import numpy as np
import pandas as pd
import pytest

from churn_saas.data.quality import (
    analyser_doublons,
    analyser_manquance,
    coercer_numeriques,
    detecter_valeurs_impossibles,
    diagnostiquer_anciennete,
    estimer_date_reference,
    parser_dates,
    profil_compact,
    proposer_renommage,
)

# ---------------------------------------------------------------------------
# analyser_manquance
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)


def _df_mnar_csat(n: int = 300) -> pd.DataFrame:
    """Simule le scénario CSAT MNAR : les clients qui churneront ne répondent pas."""
    churn = np.array([i % 2 for i in range(n)])
    csat = np.where(churn == 1, np.nan, _RNG.uniform(3.0, 5.0, size=n))
    mrr = _RNG.uniform(500, 5000, size=n)
    return pd.DataFrame({"churn": churn, "csat": csat, "mrr": mrr})


def _df_mcar(n: int = 500) -> pd.DataFrame:
    """Simule une manquance aléatoire (MCAR) sans lien ni avec churn ni avec mrr."""
    churn = _RNG.integers(0, 2, size=n)
    mrr = _RNG.uniform(500, 5000, size=n)
    masque = _RNG.random(n) < 0.25
    feature = np.where(masque, np.nan, _RNG.uniform(1.0, 5.0, size=n))
    return pd.DataFrame({"churn": churn, "mrr": mrr, "feature_aleatoire": feature})


def _df_mar(n: int = 500) -> pd.DataFrame:
    """Simule une manquance MAR : le score est absent quand mrr > 700, churn indépendant."""
    mrr = _RNG.uniform(100, 1000, size=n)
    churn = _RNG.integers(0, 2, size=n)
    masque = mrr > 700
    score = np.where(masque, np.nan, _RNG.uniform(1.0, 5.0, size=n))
    return pd.DataFrame({"churn": churn, "mrr": mrr, "score_mar": score})


class TestAnalyserManquance:
    def test_retourne_dataframe_avec_bonnes_colonnes(self) -> None:
        res = analyser_manquance(_df_mnar_csat(), "churn")
        assert isinstance(res, pd.DataFrame)
        attendues = {
            "taux_manquants",
            "p_value_cible",
            "taux_churn_si_manquant",
            "taux_churn_si_renseigne",
            "lien_cible_significatif",
            "top3_associations",
            "mecanisme_propose",
            "confiance",
            "raisonnement",
            "recommandation",
        }
        assert attendues.issubset(set(res.columns))

    def test_cible_exclue_du_resultat(self) -> None:
        res = analyser_manquance(_df_mnar_csat(), "churn")
        assert "churn" not in res.index

    def test_colonnes_sans_manquants_absentes(self) -> None:
        df = pd.DataFrame({"churn": [0, 1, 0, 1], "mrr": [100.0, 200.0, 300.0, 400.0]})
        res = analyser_manquance(df, "churn")
        assert "mrr" not in res.index

    def test_csat_mnar_detecte(self) -> None:
        """Scénario central : CSAT manquant chez les churners → MNAR."""
        res = analyser_manquance(_df_mnar_csat(300), "churn")
        assert "csat" in res.index
        assert res.loc["csat", "mecanisme_propose"] == "MNAR"
        assert res.loc["csat", "lien_cible_significatif"]
        assert res.loc["csat", "p_value_cible"] < 0.05

    def test_csat_mnar_taux_churn_contraste(self) -> None:
        """Le taux de churn doit être significativement plus élevé chez les CSAT manquants."""
        res = analyser_manquance(_df_mnar_csat(300), "churn")
        taux_manquant = res.loc["csat", "taux_churn_si_manquant"]
        taux_renseigne = res.loc["csat", "taux_churn_si_renseigne"]
        assert taux_manquant > taux_renseigne

    def test_csat_mnar_recommandation_indicateur(self) -> None:
        """La recommandation MNAR doit mentionner la création d'un indicateur de manquance."""
        res = analyser_manquance(_df_mnar_csat(300), "churn")
        assert "indicateur" in res.loc["csat", "recommandation"].lower()

    def test_mcar_detecte(self) -> None:
        res = analyser_manquance(_df_mcar(500), "churn")
        assert "feature_aleatoire" in res.index
        assert res.loc["feature_aleatoire", "mecanisme_propose"] == "MCAR"
        assert not res.loc["feature_aleatoire", "lien_cible_significatif"]

    def test_mar_detecte(self) -> None:
        res = analyser_manquance(_df_mar(500), "churn")
        assert "score_mar" in res.index
        assert res.loc["score_mar", "mecanisme_propose"] == "MAR"

    def test_taux_manquants_correct(self) -> None:
        df = pd.DataFrame(
            {
                "churn": [0, 1, 0, 1],
                "csat": [4.0, float("nan"), 3.0, float("nan")],
            }
        )
        res = analyser_manquance(df, "churn")
        assert res.loc["csat", "taux_manquants"] == pytest.approx(0.5)

    def test_df_sans_manquants_retourne_vide(self) -> None:
        df = pd.DataFrame({"churn": [0, 1], "mrr": [100.0, 200.0]})
        res = analyser_manquance(df, "churn")
        assert isinstance(res, pd.DataFrame)
        assert len(res) == 0

    def test_cible_absente_leve_valueerror(self) -> None:
        df = pd.DataFrame({"mrr": [100.0, float("nan")]})
        with pytest.raises(ValueError, match="cible"):
            analyser_manquance(df, "churn")

    def test_valeurs_p_entre_0_et_1(self) -> None:
        res = analyser_manquance(_df_mnar_csat(100), "churn")
        for p in res["p_value_cible"].dropna():
            assert 0.0 <= p <= 1.0

    def test_mecanisme_propose_valeurs_valides(self) -> None:
        res = analyser_manquance(_df_mnar_csat(100), "churn")
        assert set(res["mecanisme_propose"]).issubset({"MCAR", "MAR", "MNAR"})

    def test_confiance_valeurs_valides(self) -> None:
        res = analyser_manquance(_df_mnar_csat(100), "churn")
        assert set(res["confiance"]).issubset({"faible", "modérée", "élevée"})

    def test_df_non_modifie_en_place(self) -> None:
        df = _df_mnar_csat(50)
        original = df.copy()
        analyser_manquance(df, "churn")
        pd.testing.assert_frame_equal(df, original)

    def test_identifiant_et_numerique_texte_ne_creent_pas_de_faux_mar(self) -> None:
        """Régression : sur un DataFrame brut (tout en texte), la manquance aléatoire sortait MAR.

        Le V de Cramér avec un identifiant ou un numérique traité en catégoriel à milliers de
        modalités vaut ≈ 1, quelle que soit la réalité.
        """
        df = _df_mcar(500).astype(str).replace("nan", np.nan)
        df.insert(0, "client_id", [f"CLI-{i:05d}" for i in range(len(df))])
        res = analyser_manquance(df, "churn")
        assert res.loc["feature_aleatoire", "mecanisme_propose"] == "MCAR"
        assert "client_id" not in res.loc["feature_aleatoire", "top3_associations"]
        assert res.attrs["colonnes_exclues_associations"] == ["client_id"]

    def test_mar_detecte_sur_colonnes_texte(self) -> None:
        # Le vrai MAR doit rester détecté quand mrr est stocké en texte (virgule décimale)
        df = _df_mar(500)
        df["mrr"] = df["mrr"].map(lambda v: f"{v:.2f}".replace(".", ","))
        res = analyser_manquance(df, "churn")
        assert res.loc["score_mar", "mecanisme_propose"] == "MAR"


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

    def test_unite_de_duree_suffixee(self) -> None:
        df_out, rapport = self._coerce(["12.5 h", "3h", "30 j", "45 min", "2,5 H"])
        assert list(df_out["x"]) == pytest.approx([12.5, 3.0, 30.0, 45.0, 2.5])
        assert rapport.loc["x", "nb_reparees"] == 5
        assert rapport.loc["x", "nb_irrecuperables"] == 0

    def test_texte_se_terminant_par_h_reste_irrecuperable(self) -> None:
        # L'unité n'est retirée que si elle suit immédiatement un chiffre
        _, rapport = self._coerce(["high", "match"])
        assert rapport.loc["x", "nb_irrecuperables"] == 2

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
# profil_compact
# ---------------------------------------------------------------------------


class TestProfilCompact:
    def test_numerique_sale_detecte(self) -> None:
        """Régression : une colonne majoritairement mal formatée doit être signalée.

        Un pd.to_numeric brut n'en convertit que 1 sur 5 et la laissait passer sous le seuil.
        """
        df = pd.DataFrame({"x": ["12,5", "45.2 €", "67.3%", "3 h", "10"]})
        profil = profil_compact(df)
        assert profil.loc["x", "numerique_en_texte"]
        assert profil.loc["x", "min"] == pytest.approx(3.0)
        assert profil.loc["x", "max"] == pytest.approx(67.3)

    def test_texte_libre_non_signale(self) -> None:
        df = pd.DataFrame({"x": ["abc", "C0001", "2023-01-05", "12"]})
        assert not profil_compact(df).loc["x", "numerique_en_texte"]

    def test_marqueurs_manquants_exclus_du_denominateur(self) -> None:
        df = pd.DataFrame({"x": ["1,5", "n/a", "-", "2"]})
        assert profil_compact(df).loc["x", "numerique_en_texte"]


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

    def test_formats_melanges_tous_parses(self) -> None:
        """Régression : seul le format dominant était appliqué, les autres devenaient NaT."""
        df = pd.DataFrame({"d": ["2021-01-15", "2021-03-20", "13/02/2021", "05 Aug 2021"]})
        df_out, rapport = parser_dates(df, ["d"])
        assert rapport.loc["d", "nb_parsees"] == 4
        assert rapport.loc["d", "nb_irrecuperables"] == 0
        assert rapport.loc["d", "format_detecte"] == "%Y-%m-%d"
        assert list(df_out["d"]) == [
            pd.Timestamp("2021-01-15"),
            pd.Timestamp("2021-03-20"),
            pd.Timestamp("2021-02-13"),
            pd.Timestamp("2021-08-05"),
        ]

    def test_convention_barre_oblique_meme_si_minoritaire(self) -> None:
        # Les dates JJ/MM sont minoritaires : la convention doit quand même être tranchée
        df = pd.DataFrame({"d": ["2021-01-15", "2021-01-16", "2021-01-17", "02/13/2021"]})
        df_out, rapport = parser_dates(df, ["d"])
        assert "MM/DD" in rapport.loc["d", "convention_retenue"]
        assert df_out["d"].iloc[3] == pd.Timestamp("2021-02-13")

    def test_colonne_deja_datetime_inchangee(self) -> None:
        df = pd.DataFrame({"d": pd.to_datetime(["2021-01-15", "2021-06-30"])})
        df_out, rapport = parser_dates(df, ["d"])
        pd.testing.assert_series_equal(df_out["d"], df["d"])
        assert rapport.loc["d", "nb_irrecuperables"] == 0


# ---------------------------------------------------------------------------
# detecter_valeurs_impossibles
# ---------------------------------------------------------------------------


class TestDetecterValeursImpossibles:
    def test_dataframe_propre(self) -> None:
        df = pd.DataFrame(
            {
                "utilisateurs_actifs": [5, 16],
                "sieges_souscrits": [10, 20],
                "taux_adoption_pct": [50.0, 80.0],  # = 100 × utilisateurs / sièges
                "mrr": [1000.0, 2000.0],
            }
        )
        res = detecter_valeurs_impossibles(df)
        assert len(res) == 0
        # Les règles évaluées sont tracées même sans violation
        assert len(res.attrs["regles_controlees"]) > 0

    @pytest.mark.parametrize(
        ("colonnes", "regle_attendue"),
        [
            ({"csat": [3, 5, 6]}, "csat ∉ [1, 5]"),
            ({"churn": [0, 1, 2]}, "churn ∉ {0, 1}"),
            ({"revenu_mensuel_recurrent_eur": [100.0, -5.0]}, "revenu_mensuel_recurrent_eur < 0"),
            ({"connexions_30j": ["3", "-1"]}, "connexions_30j < 0"),
            (
                {"fonctionnalites_utilisees": [5, 12], "fonctionnalites_total": [10, 10]},
                "fonctionnalites_utilisees > fonctionnalites_total",
            ),
            (
                {"connexions_30j": [4, 0], "derniere_connexion_jours": [45, 60]},
                "connexions_30j > 0 alors que derniere_connexion_jours > 30",
            ),
            (
                {
                    "taux_adoption_pct": [50.0, 90.0],
                    "utilisateurs_actifs": [5, 5],
                    "sieges_souscrits": [10, 10],
                },
                "taux_adoption_pct ≠ 100",
            ),
        ],
    )
    def test_nouvelles_regles(self, colonnes: dict, regle_attendue: str) -> None:
        res = detecter_valeurs_impossibles(pd.DataFrame(colonnes))
        regle = res[res["regle"].str.startswith(regle_attendue)]
        assert len(regle) == 1
        assert regle.iloc[0]["nb_lignes_concernees"] == 1

    def test_date_souscription_future(self) -> None:
        df = pd.DataFrame(
            {"anciennete_mois": [2, 1], "date_souscription": ["2025-11-01", "2026-03-01"]}
        )
        res = detecter_valeurs_impossibles(df, date_reference=pd.Timestamp("2026-01-01"))
        regle = res[res["regle"].str.contains("postérieure")]
        assert regle.iloc[0]["nb_lignes_concernees"] == 1

    def test_tolerance_anciennete_parametrable(self) -> None:
        df = pd.DataFrame({"anciennete_mois": [4], "date_souscription": ["2025-11-01"]})
        ref = pd.Timestamp("2026-01-01")  # ancienneté recalculée : 2 mois, écart 2
        assert detecter_valeurs_impossibles(df, ref)["regle"].str.contains("anciennete").sum() == 0
        res = detecter_valeurs_impossibles(df, ref, tolerance_anciennete_mois=1)
        assert res["regle"].str.contains("anciennete").sum() == 1

    def test_diagnostic_anciennete(self) -> None:
        df = pd.DataFrame(
            {"anciennete_mois": ["12", "24", "6"], "date_souscription": ["2024-01-01"] * 3}
        )
        diag = diagnostiquer_anciennete(df, date_reference=pd.Timestamp("2025-01-01"))
        assert diag["repartition_ecarts"] == {-6: 1, 0: 1, 12: 1}
        assert diag["anciennete_entiere"]

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
        # Souscription ~60 mois avant la date de référence, anciennete_mois = 6 → anomalie
        df = pd.DataFrame(
            {
                "anciennete_mois": [6],
                "date_souscription": [pd.Timestamp("2021-01-01")],
            }
        )
        res = detecter_valeurs_impossibles(df, date_reference=pd.Timestamp("2026-01-01"))
        regle = res[res["regle"].str.contains("anciennete")]
        assert len(regle) == 1

    def test_anciennete_coherente(self) -> None:
        # Souscription ~2 mois avant la date de référence, anciennete_mois = 2 → pas d'anomalie
        df = pd.DataFrame(
            {
                "anciennete_mois": [2],
                "date_souscription": [pd.Timestamp("2025-11-01")],
            }
        )
        res = detecter_valeurs_impossibles(df, date_reference=pd.Timestamp("2026-01-01"))
        regle = res[res["regle"].str.contains("anciennete")]
        assert len(regle) == 0

    def test_anciennete_reference_estimee_isole_la_ligne_aberrante(self) -> None:
        """Sans date fournie, la référence est estimée (médiane) : seule l'aberrante ressort.

        Régression : la comparaison à Timestamp.now() marquait toutes les lignes d'un jeu
        extrait dans le passé, et le résultat dépendait du jour de relance.
        """
        df = pd.DataFrame(
            {
                "anciennete_mois": ["12", "24", "6", "3"],
                "date_souscription": ["2024-02-01", "01/02/2023", "01 Aug 2024", "2020-01-01"],
            }
        )
        assert estimer_date_reference(df) is not None
        res = detecter_valeurs_impossibles(df)
        regle = res[res["regle"].str.contains("anciennete")]
        assert regle.iloc[0]["nb_lignes_concernees"] == 1

    def test_valeur_brute_mal_formatee_controlee(self) -> None:
        # « 150,5 % » échappait au contrôle avec un pd.to_numeric brut
        df = pd.DataFrame({"taux_adoption_pct": ["150,5 %", "50.0"]})
        res = detecter_valeurs_impossibles(df)
        regle = res[res["colonne_ou_paire"] == "taux_adoption_pct"]
        assert regle.iloc[0]["nb_lignes_concernees"] == 1

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

    def test_unite_heures_deja_suffixee(self) -> None:
        """Un délai en heures (_h) ne doit pas recevoir le suffixe _jours."""
        res = proposer_renommage(pd.DataFrame({"delai_reponse_support_h": [1]}))
        assert res.iloc[0]["nom_propose"] == "delai_reponse_support_h"
        assert not res.iloc[0]["modifie"]

    def test_jour_de_semaine_sans_suffixe_duree(self) -> None:
        """« jour_souscription » (jour de la semaine) n'est pas une durée en jours."""
        res = proposer_renommage(pd.DataFrame({"jour_souscription": ["lundi"]}))
        assert res.iloc[0]["nom_propose"] == "jour_souscription"

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
