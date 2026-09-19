"""Protocole de validation, règle métier et évaluation partagés entre tous les modèles de §8-9."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold

from churn_saas import config


def protocole_validation() -> RepeatedStratifiedKFold:
    """Retourne l'objet CV partagé par TOUS les modèles comparés en §8-9.

    Un seul objet garantit que chaque modèle voit exactement les mêmes plis
    (train/val identiques), condition nécessaire pour que la comparaison soit honnête.
    Paramètres : 5 plis × 3 répétitions = 15 scores par métrique, suffisants pour
    un IC robuste sur ~5 000 observations.  Seed = config.RANDOM_SEED (unique dans
    tout le projet).
    """
    return RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=config.RANDOM_SEED)


# ---------------------------------------------------------------------------
# Seuils de la règle métier — issus de l'EDA (§6)
#
# • `derniere_connexion_jours > 30` : l'EDA montre une rupture nette du taux de
#   churn autour de 30 jours d'inactivité (taux ≈ 3× supérieur au-delà).
# • `csat <= 6` : l'EDA montre que les clients avec CSAT ≤ 6 ont un taux de churn
#   nettement plus élevé (note de satisfaction insuffisante sur une échelle 1-10).
# Ces seuils sont figés ici pour éviter toute fuite dans les plis de validation.
# ---------------------------------------------------------------------------
SEUIL_INACTIVITE_JOURS: int = 30
SEUIL_CSAT: int = 6


def regle_metier(df: pd.DataFrame) -> pd.Series:
    """Baseline non-ML : derniere_connexion_jours > 30 OU csat <= 6.

    Retourne des prédictions binaires (0 ou 1) sous forme de pd.Series (int).
    Les lignes avec les deux colonnes manquantes sont prédites 0 (non-churn par défaut).

    Seuils issus de l'EDA (§6) — jamais de l'intuition, jamais du jeu de test.
    Rôle : mesurer si le ML apporte un gain réel sur deux règles SQL.
    """
    inactif = df.get("derniere_connexion_jours", pd.Series(0, index=df.index))
    satisfaction = df.get("csat", pd.Series(7, index=df.index))

    pred_inactivite = pd.to_numeric(inactif, errors="coerce").fillna(0) > SEUIL_INACTIVITE_JOURS
    pred_csat = pd.to_numeric(satisfaction, errors="coerce").fillna(7) <= SEUIL_CSAT

    return (pred_inactivite | pred_csat).astype(int)


def evaluer_modele(
    modele: Any,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    cv: RepeatedStratifiedKFold,
) -> dict[str, float]:
    """Évalue `modele` par validation croisée stratifiée répétée.

    Retourne un dict de métriques (PR-AUC, ROC-AUC, recall, précision, Brier score,
    latence unitaire médiane en ms) avec moyenne et écart-type sur les plis.

    Seules les transformations apprises (fit) se font sur le fold d'entraînement ;
    la métrique est calculée sur le fold de validation — pas de fuite.
    """
    pr_aucs, roc_aucs, recalls, precisions, briers, latences = [], [], [], [], [], []

    y_arr = np.asarray(y)

    for train_idx, val_idx in cv.split(X, y_arr):
        if isinstance(X, pd.DataFrame):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        else:
            X_train, X_val = X[train_idx], X[val_idx]

        y_train, y_val = y_arr[train_idx], y_arr[val_idx]

        modele.fit(X_train, y_train)

        # Latence unitaire — médiane sur le fold de validation
        _t0 = time.perf_counter()
        for _i in range(len(X_val)):
            _xi = X_val.iloc[[_i]] if isinstance(X_val, pd.DataFrame) else X_val[[_i]]
            modele.predict_proba(_xi)
        _duree_ms = (time.perf_counter() - _t0) / len(X_val) * 1000
        latences.append(_duree_ms)

        y_proba = modele.predict_proba(X_val)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        pr_aucs.append(average_precision_score(y_val, y_proba))
        roc_aucs.append(roc_auc_score(y_val, y_proba))
        recalls.append(recall_score(y_val, y_pred, zero_division=0))
        precisions.append(precision_score(y_val, y_pred, zero_division=0))
        briers.append(brier_score_loss(y_val, y_proba))

    def _stats(vals: list[float], nom: str) -> dict[str, float]:
        arr = np.array(vals)
        return {f"{nom}_mean": float(arr.mean()), f"{nom}_std": float(arr.std())}

    return (
        _stats(pr_aucs, "pr_auc")
        | _stats(roc_aucs, "roc_auc")
        | _stats(recalls, "recall")
        | _stats(precisions, "precision")
        | _stats(briers, "brier")
        | _stats(latences, "latence_ms")
    )
