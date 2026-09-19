"""Protocole de validation, règle métier et évaluation partagés entre tous les modèles de §8-9."""

from __future__ import annotations

import time
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from loguru import logger
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline

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

        # Latence unitaire — médiane sur 20 échantillons (statistiquement suffisant,
        # évite O(n_val) prédictions × 15 plis × 5 modèles qui fait exploser le timeout)
        _N_LAT = min(20, len(X_val))
        _t0 = time.perf_counter()
        for _i in range(_N_LAT):
            _xi = X_val.iloc[[_i]] if isinstance(X_val, pd.DataFrame) else X_val[[_i]]
            modele.predict_proba(_xi)
        _duree_ms = (time.perf_counter() - _t0) / _N_LAT * 1000
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


# ---------------------------------------------------------------------------
# Wrapper sklearn pour la règle métier (baseline B1)
# ---------------------------------------------------------------------------


class RegleMetierClassifier(BaseEstimator, ClassifierMixin):  # type: ignore[misc]
    """Baseline B1 : règle SQL encapsulée en estimateur sklearn.

    fit() ne fait rien (pas de paramètre appris).
    predict_proba() renvoie des probabilités dégénérées 0.0 / 1.0.
    Compatible avec evaluer_modele() quand X est un pd.DataFrame.
    """

    def fit(self, X: pd.DataFrame, y: Any = None) -> RegleMetierClassifier:
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "RegleMetierClassifier requiert un DataFrame pandas (colonnes nommées nécessaires)."
            )
        pred = regle_metier(X).values.astype(float)
        return np.column_stack([1.0 - pred, pred])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ---------------------------------------------------------------------------
# Construction des modèles — toutes familles
# ---------------------------------------------------------------------------


def construire_modeles(df: pd.DataFrame) -> dict[str, Any]:
    """Construit un dict {nom → Pipeline complet} pour toutes les familles.

    Famille linéaire : LogisticRegression(class_weight='balanced').
    Famille arbres   : RandomForestClassifier + HistGradientBoostingClassifier.
    Baselines        : B0 (hasard), B1 (règle métier), B2 (log. reg. = référence ML).

    Tous les Pipelines partagent le même schéma de préprocesseur (même colonnes,
    même transformations) mais chacun a son propre clone, fitté indépendamment.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de référence (features + éventuelles colonnes interdites) utilisé
        pour déduire le schéma du préprocesseur. Non fitté ici.
    """
    from churn_saas.features.build import construire_preprocesseur

    pre = construire_preprocesseur(df)

    def _pipe(estimateur: Any) -> Pipeline:
        return Pipeline([("pre", clone(pre)), ("clf", estimateur)])

    return {
        "B0 — Hasard stratifié": _pipe(
            DummyClassifier(strategy="stratified", random_state=config.RANDOM_SEED)
        ),
        "B1 — Règle métier": RegleMetierClassifier(),
        "B2 — Régression logistique": _pipe(
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=config.RANDOM_SEED,
            )
        ),
        "Forêt aléatoire": _pipe(
            RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=config.RANDOM_SEED,
                n_jobs=-1,
            )
        ),
        "Gradient boosting": _pipe(
            HistGradientBoostingClassifier(
                max_iter=300,
                random_state=config.RANDOM_SEED,
            )
        ),
    }


# ---------------------------------------------------------------------------
# Comparaison sur les mêmes plis
# ---------------------------------------------------------------------------


def comparer_modeles(
    modeles: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    """Évalue tous les modèles sur les MÊMES plis de validation croisée.

    Retourne un DataFrame trié par PR-AUC décroissant avec pour chaque modèle :
    PR-AUC (mean ± std), ROC-AUC, recall, précision, Brier score,
    latence unitaire médiane (ms) et durée totale d'évaluation (s).

    Parameters
    ----------
    modeles : dict
        Résultat de ``construire_modeles()``.
    X : pd.DataFrame
        Features (sans colonnes interdites ni cible).
    y : pd.Series
        Cible binaire.
    """
    cv = protocole_validation()
    resultats = []

    for nom, modele in modeles.items():
        logger.info("Évaluation du modèle : {}", nom)
        t0 = time.perf_counter()
        metriques = evaluer_modele(modele, X, y, cv)
        duree_s = round(time.perf_counter() - t0, 1)

        resultats.append({"modele": nom, "duree_eval_s": duree_s, **metriques})

    df = pd.DataFrame(resultats).set_index("modele")
    return df.sort_values("pr_auc_mean", ascending=False)


# ---------------------------------------------------------------------------
# Journalisation MLflow
# ---------------------------------------------------------------------------


def journaliser_mlflow(
    nom: str,
    modele: Any,
    metriques: dict[str, float],
    params: dict[str, Any],
) -> str:
    """Crée un run MLflow pour un modèle, logge params, métriques et artefact.

    Backend fichier local dans ``config.MLRUNS``.
    Expérience unique : « churn_saas_classification ».

    Returns
    -------
    str — identifiant du run MLflow créé.
    """
    config.MLRUNS.mkdir(parents=True, exist_ok=True)
    # SQLite local — compatible avec MLflow >= 2.14 qui a déprécié le backend fichier pur
    mlflow.set_tracking_uri(f"sqlite:///{config.MLRUNS / 'mlflow.db'}")
    mlflow.set_experiment("churn_saas_classification")

    with mlflow.start_run(run_name=nom) as run:
        mlflow.log_params({k: str(v) for k, v in params.items()})
        for cle, val in metriques.items():
            if isinstance(val, (int, float)):
                mlflow.log_metric(cle, float(val))
        try:
            mlflow.sklearn.log_model(modele, artifact_path="model")
        except Exception as exc:
            logger.warning("MLflow : impossible de logguer le modèle '{}' — {}", nom, exc)

        run_id: str = str(run.info.run_id)

    logger.info("MLflow run créé — nom='{}', run_id={}", nom, run_id)
    return run_id


# ---------------------------------------------------------------------------
# Comparaison class_weight vs SMOTE — calibration
# ---------------------------------------------------------------------------


def comparer_desequilibre(
    X: pd.DataFrame,
    y: pd.Series,
    df_ref: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compare class_weight='balanced' vs SMOTE sur régression logistique.

    SMOTE est appliqué DANS le pipeline imblearn — jamais hors CV.

    Mesures retournées :
    - PR-AUC, ROC-AUC, Brier score (OOF sur 5 plis)
    - Courbe de fiabilité (calibration_curve sur les OOF probas)

    Le modèle retenu pour le seuil économique (§10) est celui avec class_weight,
    car ses probabilités restent calibrées (pas de modification de la prévalence).

    Returns
    -------
    (tableau_comparaison, donnees_calibration)
    tableau_comparaison : DataFrame indexé par approche
    donnees_calibration : dict {approche: {prob_true, prob_pred, y_proba_oof}}
    """
    from churn_saas.features.build import construire_preprocesseur

    pre = construire_preprocesseur(df_ref)
    cv_cal = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)

    modeles_deseq: dict[str, Any] = {
        "class_weight='balanced'": Pipeline(
            [
                ("pre", clone(pre)),
                (
                    "clf",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=config.RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "SMOTE": ImbPipeline(
            [
                ("pre", clone(pre)),
                ("smote", SMOTE(random_state=config.RANDOM_SEED)),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=config.RANDOM_SEED,
                    ),
                ),
            ]
        ),
    }

    donnees_calibration: dict[str, Any] = {}
    resultats = []
    y_arr = np.asarray(y)

    for nom, modele in modeles_deseq.items():
        logger.info("Comparaison déséquilibre — approche : {}", nom)
        y_proba_oof = np.zeros(len(y_arr))

        for train_idx, val_idx in cv_cal.split(X, y_arr):
            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_train = y_arr[train_idx]
            modele.fit(X_train, y_train)
            y_proba_oof[val_idx] = modele.predict_proba(X_val)[:, 1]

        prob_true, prob_pred = calibration_curve(y_arr, y_proba_oof, n_bins=10, strategy="quantile")
        brier = float(brier_score_loss(y_arr, y_proba_oof))
        pr_auc = float(average_precision_score(y_arr, y_proba_oof))
        roc_auc = float(roc_auc_score(y_arr, y_proba_oof))

        donnees_calibration[nom] = {
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist(),
            "y_proba_oof": y_proba_oof.tolist(),
        }
        resultats.append(
            {
                "approche": nom,
                "pr_auc": round(pr_auc, 4),
                "roc_auc": round(roc_auc, 4),
                "brier_score": round(brier, 4),
            }
        )

    tableau = pd.DataFrame(resultats).set_index("approche")
    return tableau, donnees_calibration
