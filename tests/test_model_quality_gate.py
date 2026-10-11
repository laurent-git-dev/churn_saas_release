"""Test de non-régression métrique — preuve C9 : le modèle de référence dépasse le seuil fixé a priori.

Ce test est le filet de sécurité quantitatif du projet :
- La cible PR-AUC >= config.CIBLES_PERFORMANCE["pr_auc_min"] est fixée AVANT tout entraînement.
- Un échec ici signifie une régression du modèle de référence, pas un ajustement de seuil.
- Marqué "slow" car il entraîne un modèle en CV, même sur un sous-échantillon.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from churn_saas import config
from churn_saas.features.build import construire_preprocesseur
from churn_saas.models.train import (
    NOM_BASELINE_LR,
    NOM_LIGHTGBM,
    construire_modele_optimise,
    construire_modeles,
    optimiser,
)

# ---------------------------------------------------------------------------
# Jeu de données synthétique minimal — reproduit la structure du gold dataset
# ---------------------------------------------------------------------------


def _jeu_synthetique(
    n: int = 300, seed: int = config.RANDOM_SEED
) -> tuple[pd.DataFrame, pd.Series]:
    """Génère un DataFrame synthétique avec les features clés et ~20 % de churn.

    Paramétrage : les classes restent déséquilibrées (~20 % churn) pour refléter
    les conditions réelles d'entraînement.  N=300 : assez pour 5 plis CV stables,
    assez petit pour que le test s'exécute en < 30 s.
    """
    rng = np.random.default_rng(seed)
    n_churn = int(n * 0.20)
    n_ok = n - n_churn

    def _bloc(taille: int, churn: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "anciennete_mois": rng.integers(1, 60, taille),
                "revenu_mensuel_recurrent_eur": rng.uniform(500, 5000, taille),
                "sieges_souscrits": rng.integers(5, 100, taille),
                "utilisateurs_actifs": rng.integers(2, 80, taille),
                "heures_usage_30j": rng.uniform(0, 200, taille),
                "connexions_30j": rng.integers(0, 100, taille),
                "fonctionnalites_utilisees": rng.integers(1, 20, taille),
                "fonctionnalites_total": rng.integers(10, 30, taille),
                "nb_integrations": rng.integers(0, 10, taille),
                "tickets_support_90j": rng.integers(0, 15, taille),
                # Signal fort pour les churners : inactivité et CSAT bas
                "derniere_connexion_jours": rng.integers(
                    35 if churn else 0, 120 if churn else 29, taille
                ),
                # Échelle 1-5 : churners insatisfaits (1-2), non-churners satisfaits (4-5)
                "csat": rng.integers(1 if churn else 4, 3 if churn else 6, taille).astype(float),
                "plan": rng.choice(["Starter", "Pro", "Enterprise"], taille),
                "secteur": rng.choice(["SaaS", "Industrie", "Finance"], taille),
                "taux_utilisation_sieges": rng.uniform(0.1, 1.0, taille),
                "surdimensionnement": rng.integers(0, 20, taille).astype(float),
                "intensite_usage_par_utilisateur": rng.uniform(0, 10, taille),
                "connexions_par_utilisateur": rng.uniform(0, 5, taille),
                "taux_couverture_fonctionnelle": rng.uniform(0.1, 1.0, taille),
                "arpu_par_siege": rng.uniform(50, 500, taille),
                "recence_normalisee": rng.uniform(0, 1, taille),
                "compte_dormant": rng.choice([True, False], taille),
                "pression_support": rng.uniform(0, 2, taille),
                "tranche_anciennete": rng.choice(["onboarding", "installation", "mature"], taille),
                "tranche_integrations": rng.choice(
                    ["aucune", "faible", "moderee", "forte"], taille
                ),
            }
        )

    X = pd.concat([_bloc(n_ok, churn=0), _bloc(n_churn, churn=1)], ignore_index=True)
    y = pd.Series([0] * n_ok + [1] * n_churn, name="churn")
    return X, y


# ---------------------------------------------------------------------------
# Gate de qualité métrique
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestModelQualityGate:
    """Vérifie que les modèles de référence dépassent le seuil PR-AUC fixé a priori."""

    @pytest.mark.parametrize("nom", [NOM_BASELINE_LR, NOM_LIGHTGBM])
    def test_pr_auc_superieure_au_seuil(self, nom: str) -> None:
        """Chaque modèle doit atteindre PR-AUC >= config.CIBLES_PERFORMANCE['pr_auc_min'].

        Le test entraîne le pipeline de ``construire_modeles`` (baseline régression logistique
        et LightGBM) en validation croisée 5 plis sur un sous-échantillon synthétique. Si la
        PR-AUC OOF chute sous le seuil a priori, cela signale une régression dans les features
        ou le pipeline.
        """
        X, y = _jeu_synthetique(n=300)
        seuil = config.CIBLES_PERFORMANCE["pr_auc_min"]
        modele = construire_modeles(X)[nom]

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
        y_arr = np.asarray(y)
        y_proba_oof = np.zeros(len(y_arr))

        for train_idx, val_idx in cv.split(X, y_arr):
            m = clone(modele)
            m.fit(X.iloc[train_idx], y_arr[train_idx])
            y_proba_oof[val_idx] = m.predict_proba(X.iloc[val_idx])[:, 1]

        pr_auc = average_precision_score(y_arr, y_proba_oof)

        assert pr_auc >= seuil, (
            f"{nom} : PR-AUC OOF = {pr_auc:.4f} < seuil de non-régression {seuil:.2f}. "
            "Vérifier les features (build.py), le préprocesseur ou les données gold."
        )

    def test_latence_unitaire_sous_seuil(self) -> None:
        """La latence unitaire du pipeline entraîné doit rester < config.CIBLES_PERFORMANCE['latence_unitaire_ms']."""
        import time

        X, y = _jeu_synthetique(n=100)
        seuil_ms = config.CIBLES_PERFORMANCE["latence_unitaire_ms"]

        pre = construire_preprocesseur(X)
        modele = Pipeline(
            [
                ("pre", pre),
                (
                    "clf",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=500,
                        random_state=config.RANDOM_SEED,
                    ),
                ),
            ]
        )
        modele.fit(X, y)

        latences_ms = []
        for i in range(min(50, len(X))):
            t0 = time.perf_counter()
            modele.predict_proba(X.iloc[[i]])
            latences_ms.append((time.perf_counter() - t0) * 1000)

        mediane_ms = float(np.median(latences_ms))
        assert mediane_ms < seuil_ms, (
            f"Latence unitaire médiane = {mediane_ms:.2f} ms > seuil {seuil_ms} ms. "
            "Le pipeline est trop lent pour le webhook CRM synchrone."
        )


# ---------------------------------------------------------------------------
# Familles comparées et reconstruction des modèles optimisés
# ---------------------------------------------------------------------------

_ESPACE_LIGHTGBM = {
    "num_leaves",
    "learning_rate",
    "n_estimators",
    "min_child_samples",
    "reg_lambda",
    "subsample",
    "colsample_bytree",
}


def test_familles_comparees() -> None:
    X, _ = _jeu_synthetique(n=50)
    modeles = construire_modeles(X)
    assert list(modeles) == [
        "B0 — Hasard stratifié",
        "B1 — Règle métier",
        NOM_BASELINE_LR,
        "Forêt aléatoire",
        NOM_LIGHTGBM,
    ]
    lgbm = modeles[NOM_LIGHTGBM].named_steps["clf"]
    assert isinstance(lgbm, LGBMClassifier)
    assert lgbm.get_params()["class_weight"] == "balanced"
    assert lgbm.get_params()["random_state"] == config.RANDOM_SEED
    assert lgbm.get_params()["n_jobs"] == 1


def test_modele_optimise_lightgbm_recoit_ses_hyperparametres() -> None:
    X, _ = _jeu_synthetique(n=50)
    params = {"num_leaves": 15, "learning_rate": 0.05, "subsample": 0.8}
    clf = construire_modele_optimise(NOM_LIGHTGBM, params, df_ref=X).named_steps["clf"]
    assert isinstance(clf, LGBMClassifier)
    assert {k: clf.get_params()[k] for k in params} == params
    # Sans fréquence de bagging, LightGBM ignorerait silencieusement `subsample`
    assert clf.get_params()["subsample_freq"] == 1


def test_famille_inconnue_leve_une_erreur() -> None:
    X, _ = _jeu_synthetique(n=50)
    with pytest.raises(ValueError, match="Famille de modèle inconnue"):
        construire_modele_optimise("Gradient boosting", {}, df_ref=X)


@pytest.mark.slow
def test_optuna_explore_l_espace_lightgbm() -> None:
    X, y = _jeu_synthetique(n=200)
    resultat = optimiser(NOM_LIGHTGBM, X, y, n_essais=2)
    assert set(resultat["best_params"]) == _ESPACE_LIGHTGBM
    assert resultat["n_essais_completes"] >= 1
    assert (config.TABLES / "optuna_lightgbm.db").exists()
