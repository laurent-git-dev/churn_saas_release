"""Tests anti-fuite — le test le plus strict du projet.

Trois invariants vérifiés :
1. ``colonnes_features()`` ne renvoie jamais une colonne de ``COLONNES_INTERDITES``.
2. ``AgregatParGroupe`` fitté sur le train n'utilise pas les données du test
   (cas construit où les statistiques diffèrent entre train et test).
3. Un Pipeline complet fitté sur un pli de validation croisée ne voit aucune
   donnée du pli de validation (les statistiques apprises diffèrent).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from churn_saas import config
from churn_saas.features.build import colonnes_features, construire_preprocesseur
from churn_saas.features.transformers import AgregatParGroupe

# ---------------------------------------------------------------------------
# 1. colonnes_features() ne renvoie jamais une colonne interdite
# ---------------------------------------------------------------------------


class TestColonnesFeaturesSansLeak:
    def test_leve_si_colonne_cible_presente(self) -> None:
        df = pd.DataFrame({"mrr": [100.0, 200.0], "churn": [0, 1]})
        with pytest.raises(ValueError, match="churn"):
            colonnes_features(df)

    def test_leve_si_client_id_present(self) -> None:
        df = pd.DataFrame({"mrr": [100.0, 200.0], "client_id": ["A", "B"]})
        with pytest.raises(ValueError, match="client_id"):
            colonnes_features(df)

    def test_leve_si_plusieurs_colonnes_interdites(self) -> None:
        df = pd.DataFrame(
            {
                "mrr": [100.0, 200.0],
                "churn": [0, 1],
                "client_id": ["A", "B"],
                "valeur_vie_client_eur": [1000.0, 2000.0],
            }
        )
        with pytest.raises(ValueError):
            colonnes_features(df)

    def test_aucune_colonne_interdite_dans_le_resultat(self) -> None:
        df_propre = pd.DataFrame(
            {
                "mrr": [100.0, 200.0, 300.0],
                "plan": ["starter", "pro", "enterprise"],
                "score_sante": [0.8, 0.5, 0.3],
            }
        )
        num, cat = colonnes_features(df_propre)
        colonnes_interdites = set(config.COLONNES_INTERDITES)
        assert not (
            set(num) & colonnes_interdites
        ), f"Colonnes interdites trouvées dans les numériques : {set(num) & colonnes_interdites}"
        assert not (
            set(cat) & colonnes_interdites
        ), f"Colonnes interdites trouvées dans les catégorielles : {set(cat) & colonnes_interdites}"

    def test_toutes_colonnes_interdites_provoquent_une_erreur(self) -> None:
        """Chaque colonne interdite, prise seule, déclenche l'exception."""
        for col in config.COLONNES_INTERDITES:
            df = pd.DataFrame({col: [1.0, 2.0]})
            with pytest.raises(ValueError, match=col):
                colonnes_features(df)


# ---------------------------------------------------------------------------
# 2. AgregatParGroupe ne fuit pas du train vers le test
# ---------------------------------------------------------------------------


class TestAgregatParGroupeAntiLeak:
    """Vérifie que l'agrégat vient exclusivement du train, jamais du test."""

    @staticmethod
    def _construire_train_test_divergents() -> tuple[pd.DataFrame, pd.DataFrame]:
        """Train et test avec des médianes par groupe très différentes."""
        train = pd.DataFrame(
            {
                "segment": ["A", "A", "B", "B"],
                "mrr": [100.0, 200.0, 300.0, 400.0],
            }
        )
        # Mêmes groupes, valeurs ×10 pour rendre toute confusion évidente
        test = pd.DataFrame(
            {
                "segment": ["A", "A", "B", "B"],
                "mrr": [1000.0, 2000.0, 3000.0, 4000.0],
            }
        )
        return train, test

    def test_statistiques_apprises_sur_le_train(self) -> None:
        train, _ = self._construire_train_test_divergents()
        t = AgregatParGroupe("segment", "mrr", "median")
        t.fit(train)
        # Médiane train : A → 150.0, B → 350.0
        assert t._aggregats_par_groupe_["A"] == pytest.approx(150.0)
        assert t._aggregats_par_groupe_["B"] == pytest.approx(350.0)

    def test_transform_test_utilise_statistiques_du_train(self) -> None:
        train, test = self._construire_train_test_divergents()
        t = AgregatParGroupe("segment", "mrr", "median")
        t.fit(train)
        test_transforme = t.transform(test)

        nom = "mrr_median_par_segment"
        assert nom in test_transforme.columns

        # Les lignes A du test doivent afficher 150.0 (médiane du train), PAS 1500.0
        valeurs_a = test_transforme.loc[test_transforme["segment"] == "A", nom]
        assert valeurs_a.tolist() == pytest.approx(
            [150.0, 150.0]
        ), f"Attendu 150.0 (médiane train pour A), obtenu : {valeurs_a.tolist()}"
        valeurs_b = test_transforme.loc[test_transforme["segment"] == "B", nom]
        assert valeurs_b.tolist() == pytest.approx(
            [350.0, 350.0]
        ), f"Attendu 350.0 (médiane train pour B), obtenu : {valeurs_b.tolist()}"

    def test_repli_global_pour_modalite_inconnue(self) -> None:
        train, _ = self._construire_train_test_divergents()
        t = AgregatParGroupe("segment", "mrr", "median")
        t.fit(train)

        # Groupe "C" inconnu au train → repli sur la médiane globale du train
        # median([100, 200, 300, 400]) = 250.0
        test_inconnu = pd.DataFrame({"segment": ["C", "C"], "mrr": [9999.0, 9999.0]})
        transforme = t.transform(test_inconnu)
        assert transforme["mrr_median_par_segment"].tolist() == pytest.approx([250.0, 250.0])

    def test_fit_ne_modifie_pas_le_train(self) -> None:
        """fit() est non-destructif : le DataFrame d'entraînement reste inchangé."""
        train, _ = self._construire_train_test_divergents()
        train_copie = train.copy()
        AgregatParGroupe("segment", "mrr", "median").fit(train)
        pd.testing.assert_frame_equal(train, train_copie)

    def test_cas_construit_preuve_de_non_contamination(self) -> None:
        """Preuve directe : si les données test avaient contaminé le fit,
        les résultats du transform seraient ×10 — ici ils ne le sont pas."""
        rng = np.random.default_rng(42)

        # Train : groupe X → 10, groupe Y → 20
        train = pd.DataFrame(
            {
                "seg": np.repeat(["X", "Y"], 50),
                "val": np.concatenate([rng.normal(10.0, 0.01, 50), rng.normal(20.0, 0.01, 50)]),
            }
        )
        # Test : groupe X → 100, groupe Y → 200 (×10)
        test = pd.DataFrame(
            {
                "seg": np.repeat(["X", "Y"], 25),
                "val": np.concatenate([rng.normal(100.0, 0.01, 25), rng.normal(200.0, 0.01, 25)]),
            }
        )

        t = AgregatParGroupe("seg", "val", "mean")
        t.fit(train)
        resultat = t.transform(test)

        # Attendu depuis le train : ~10 pour X, ~20 pour Y
        for seg, attendu in [("X", 10.0), ("Y", 20.0)]:
            valeurs = resultat.loc[resultat["seg"] == seg, "val_mean_par_seg"]
            assert np.allclose(valeurs.values, attendu, atol=0.5), (
                f"Attendu ≈{attendu} pour le segment {seg} (depuis le train), "
                f"obtenu {valeurs.tolist()} — possible contamination par le test."
            )

    def test_leve_si_statistique_inconnue(self) -> None:
        t = AgregatParGroupe("seg", "val", "median")
        t.statistique = "variance"  # non autorisée
        df = pd.DataFrame({"seg": ["A"], "val": [1.0]})
        with pytest.raises(ValueError, match="variance"):
            t.fit(df)


# ---------------------------------------------------------------------------
# 3. Pipeline complet fitté sur un pli ne voit pas le pli de validation
# ---------------------------------------------------------------------------


class TestPipelineCVIsolation:
    """Vérifie l'isolation entre plis dans la validation croisée."""

    @staticmethod
    def _df_bimodal(n: int = 200) -> tuple[pd.DataFrame, pd.Series]:
        """DataFrame où les deux moitiés ont des distributions très différentes.

        indices 0..n//2-1  : mrr ≈ 100  (premier pli de CV)
        indices n//2..n-1  : mrr ≈ 5000 (second pli de CV)
        """
        mrr = np.concatenate(
            [
                np.full(n // 2, 100.0),  # train si split [0:n//2]
                np.full(n // 2, 5000.0),  # validation si split [0:n//2]
            ]
        )
        plan = np.where(np.arange(n) < n // 2, "starter", "enterprise")
        y = pd.Series(np.tile([0, 1], n // 2), name="churn")
        return pd.DataFrame({"mrr": mrr, "plan": plan}), y

    def test_mediane_apprise_sur_le_train_seulement(self) -> None:
        """La médiane apprise par l'imputer = médiane du train, PAS du jeu complet."""
        df, y = self._df_bimodal()
        n = len(df)

        X_train = df.iloc[: n // 2]  # mrr ≈ 100
        X_val = df.iloc[n // 2 :]  # mrr ≈ 5000
        y_train = y.iloc[: n // 2]

        preprocesseur = construire_preprocesseur(X_train)
        preprocesseur.fit(X_train, y_train)

        imputer = preprocesseur.named_transformers_["numerique"].named_steps["imputation"]
        mediane_mrr_apprise = imputer.statistics_[0]  # mrr est la première colonne numérique

        mediane_mrr_val = X_val["mrr"].median()
        mediane_mrr_train = X_train["mrr"].median()

        assert abs(mediane_mrr_apprise - mediane_mrr_train) < 1.0, (
            f"Médiane apprise ({mediane_mrr_apprise:.1f}) devrait ≈ médiane train "
            f"({mediane_mrr_train:.1f}), pas médiane val ({mediane_mrr_val:.1f})."
        )
        assert abs(mediane_mrr_apprise - mediane_mrr_val) > 1000.0, (
            f"Médiane apprise ({mediane_mrr_apprise:.1f}) trop proche de la médiane val "
            f"({mediane_mrr_val:.1f}) — indice de fuite possible."
        )

    def test_pipeline_dans_cv_reste_isole(self) -> None:
        """Simulation de 3 plis de CV : le preprocesseur de chaque pli est fitté
        uniquement sur le train du pli, jamais sur la validation."""
        rng = np.random.default_rng(config.RANDOM_SEED)
        n = 300
        df = pd.DataFrame(
            {
                "mrr": rng.uniform(100, 5000, n),
                "nb_utilisateurs": rng.integers(1, 50, n).astype(float),
                "plan": rng.choice(["starter", "pro", "enterprise"], n),
            }
        )
        y = pd.Series(rng.integers(0, 2, n), name="churn")

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.RANDOM_SEED)
        for train_idx, val_idx in skf.split(df, y):
            X_train = df.iloc[train_idx].reset_index(drop=True)
            X_val = df.iloc[val_idx].reset_index(drop=True)
            y_train = y.iloc[train_idx]

            pipeline = Pipeline(
                [
                    ("pre", construire_preprocesseur(X_train)),
                    (
                        "clf",
                        LogisticRegression(max_iter=200, random_state=config.RANDOM_SEED),
                    ),
                ]
            )
            pipeline.fit(X_train, y_train)

            # Les médianes apprises doivent correspondre au train, pas au val
            imputer = (
                pipeline.named_steps["pre"]
                .named_transformers_["numerique"]
                .named_steps["imputation"]
            )
            mediane_mrr_apprise = imputer.statistics_[0]

            mediane_mrr_train = X_train["mrr"].median()
            mediane_mrr_val = X_val["mrr"].median()
            mediane_jeu_complet = df["mrr"].median()

            # L'écart attendu : la médiane apprise est plus proche du train que du jeu complet
            ecart_train = abs(mediane_mrr_apprise - mediane_mrr_train)
            ecart_global = abs(mediane_mrr_apprise - mediane_jeu_complet)

            assert ecart_train < ecart_global + 1.0, (
                f"Pli en cours : médiane apprise={mediane_mrr_apprise:.1f}, "
                f"médiane train={mediane_mrr_train:.1f}, "
                f"médiane jeu complet={mediane_mrr_val:.1f}. "
                "Le preprocesseur semble fitté sur plus que le train."
            )

            # La prédiction sur le val ne doit pas lever d'exception
            pipeline.predict_proba(X_val)

    def test_construire_preprocesseur_exclut_colonnes_interdites(self) -> None:
        """construire_preprocesseur() ne laisse passer aucune colonne interdite."""
        df_complet = pd.DataFrame(
            {
                "mrr": [100.0, 200.0, 300.0],
                "plan": ["A", "B", "C"],
                # Colonnes interdites présentes — doivent être exclues
                "churn": [0, 1, 0],
                "client_id": ["x1", "x2", "x3"],
                "valeur_vie_client_eur": [1000.0, 2000.0, 3000.0],
                "sante_compte_fin_periode": [0.8, 0.5, 0.3],
            }
        )
        # construire_preprocesseur doit gérer le df complet sans erreur
        preprocesseur = construire_preprocesseur(df_complet)
        X_train = df_complet.drop(columns=config.COLONNES_INTERDITES + ["churn"], errors="ignore")
        preprocesseur.fit(X_train)
        # Vérification : le transformeur numérique ne connaît que "mrr"
        cols_numeriques = preprocesseur.transformers[0][2]
        for col_interdite in config.COLONNES_INTERDITES:
            assert (
                col_interdite not in cols_numeriques
            ), f"Colonne interdite '{col_interdite}' présente dans le ColumnTransformer numérique."
