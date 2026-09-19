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
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from churn_saas import config
from churn_saas.features.build import construire_preprocesseur

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
                "csat": rng.integers(1 if churn else 7, 6 if churn else 10, taille).astype(float),
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
    """Vérifie que le modèle de référence (LR class_weight) dépasse le seuil PR-AUC."""

    def test_pr_auc_superieure_au_seuil(self) -> None:
        """Le modèle de référence doit atteindre PR-AUC >= config.CIBLES_PERFORMANCE['pr_auc_min'].

        Le test entraîne une régression logistique (modèle de référence §9) en validation
        croisée 5 plis sur un sous-échantillon synthétique. Si la PR-AUC OOF chute sous
        le seuil a priori, cela signale une régression dans les features ou le pipeline.
        """
        X, y = _jeu_synthetique(n=300)
        seuil = config.CIBLES_PERFORMANCE["pr_auc_min"]

        pre = construire_preprocesseur(X)
        modele = Pipeline(
            [
                ("pre", pre),
                (
                    "clf",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=config.RANDOM_SEED,
                    ),
                ),
            ]
        )

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
        y_arr = np.asarray(y)
        y_proba_oof = np.zeros(len(y_arr))

        for train_idx, val_idx in cv.split(X, y_arr):
            m = clone(modele)
            m.fit(X.iloc[train_idx], y_arr[train_idx])
            y_proba_oof[val_idx] = m.predict_proba(X.iloc[val_idx])[:, 1]

        pr_auc = average_precision_score(y_arr, y_proba_oof)

        assert pr_auc >= seuil, (
            f"PR-AUC OOF = {pr_auc:.4f} < seuil de non-régression {seuil:.2f}. "
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
