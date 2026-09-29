"""Protocole de validation, règle métier et évaluation partagés entre tous les modèles de §8-9."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
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
    Paramètres : 5 plis × 3 répétitions = 15 scores par métrique, conservés pli par
    pli pour le test de Wilcoxon apparié de ``comparer_au_champion()``.
    Seed = config.RANDOM_SEED (unique dans tout le projet).
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
    latence unitaire médiane en ms) avec moyenne et écart-type sur les plis, plus la
    PR-AUC de chaque pli (`pr_auc_pli_00` … `pr_auc_pli_14`), nécessaire au test apparié
    de ``comparer_au_champion()``.

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

    # Scores bruts par pli : l'ordre des plis est celui de `cv.split`, identique pour tous
    # les modèles puisque l'objet CV est partagé — c'est ce qui rend le test apparié valide.
    scores_par_pli = {f"{PREFIXE_PLI}{i:02d}": float(v) for i, v in enumerate(pr_aucs)}

    return (
        _stats(pr_aucs, "pr_auc")
        | _stats(roc_aucs, "roc_auc")
        | _stats(recalls, "recall")
        | _stats(precisions, "precision")
        | _stats(briers, "brier")
        | _stats(latences, "latence_ms")
        | scores_par_pli
    )


# ---------------------------------------------------------------------------
# Test de supériorité du champion — Wilcoxon apparié + correction de Holm (§8.8)
# ---------------------------------------------------------------------------

PREFIXE_PLI = "pr_auc_pli_"
ALPHA_SIGNIFICATIVITE = 0.05


def correction_holm(p_valeurs: list[float]) -> list[float]:
    """Correction de Holm (step-down) : p-valeurs ajustées, risque global ≤ α.

    La k-ième plus petite p-valeur (k = 0…m-1) est multipliée par (m − k), puis on impose
    la monotonie (une p-valeur ajustée ne peut pas être inférieure à la précédente).
    """
    m = len(p_valeurs)
    ordre = np.argsort(p_valeurs)
    ajustees = np.empty(m)
    maximum_courant = 0.0
    for rang, idx in enumerate(ordre):
        maximum_courant = max(maximum_courant, (m - rang) * p_valeurs[idx])
        ajustees[idx] = min(1.0, maximum_courant)
    return [float(p) for p in ajustees]


def comparer_au_champion(
    tableau: pd.DataFrame,
    champion: str,
    alpha: float = ALPHA_SIGNIFICATIVITE,
) -> pd.DataFrame:
    """Compare `champion` à chaque autre modèle du tableau, pli par pli.

    Test de Wilcoxon apparié unilatéral (H1 : le champion a une PR-AUC plus élevée) sur
    les 15 PR-AUC par pli, puis correction de Holm sur l'ensemble des comparaisons.

    Parameters
    ----------
    tableau : pd.DataFrame
        Sortie de ``comparer_modeles()`` (une ligne par modèle, colonnes ``pr_auc_pli_*``).
    champion : str
        Nom du modèle champion (index du tableau).
    alpha : float
        Risque global d'erreur de première espèce.

    Returns
    -------
    pd.DataFrame indexé par concurrent : écart moyen de PR-AUC, nombre de plis gagnés,
    p-valeur brute, p-valeur ajustée (Holm) et verdict de significativité.
    """
    from scipy.stats import wilcoxon

    colonnes_plis = sorted(c for c in tableau.columns if c.startswith(PREFIXE_PLI))
    if not colonnes_plis:
        raise ValueError(
            "Scores par pli absents du tableau : recalculer la comparaison (make notebook-full)."
        )

    scores_champion = tableau.loc[champion, colonnes_plis].to_numpy(dtype=float)
    lignes = []
    for concurrent in tableau.index.drop(champion):
        ecarts = scores_champion - tableau.loc[concurrent, colonnes_plis].to_numpy(dtype=float)
        # Écarts tous nuls : aucune différence à tester, p = 1 par convention
        p_brute = (
            1.0 if np.allclose(ecarts, 0.0) else float(wilcoxon(ecarts, alternative="greater")[1])
        )
        lignes.append(
            {
                "concurrent": concurrent,
                "ecart_pr_auc_moyen": float(ecarts.mean()),
                "plis_gagnes": f"{int((ecarts > 0).sum())}/{len(ecarts)}",
                "p_valeur_brute": p_brute,
            }
        )

    resultat = pd.DataFrame(lignes).set_index("concurrent")
    resultat["p_valeur_holm"] = correction_holm(resultat["p_valeur_brute"].tolist())
    resultat["significatif"] = resultat["p_valeur_holm"] < alpha
    logger.info(
        "Wilcoxon + Holm : {} significativement battu(s) par '{}' sur {} (α = {})",
        int(resultat["significatif"].sum()),
        champion,
        len(resultat),
        alpha,
    )
    return resultat


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


# ---------------------------------------------------------------------------
# Empreinte carbone — CodeCarbon
# ---------------------------------------------------------------------------


def _detecter_mode_mesure() -> tuple[str, bool]:
    """Détecte si CodeCarbon peut mesurer via RAPL ou doit estimer via TDP.

    Sous WSL2, /sys/class/powercap/intel-rapl/ est absent ou non accessible
    → CodeCarbon bascule sur une estimation TDP × mix électrique du pays.

    Returns (mode_label, is_estimation).
    """
    rapl_path = Path("/sys/class/powercap/intel-rapl")
    if rapl_path.exists():
        try:
            if any(rapl_path.iterdir()):
                return "mesure RAPL", False
        except PermissionError:
            pass
    return "estimation TDP", True


def mesurer_empreinte(fonction: Callable[[], Any]) -> tuple[Any, dict[str, Any]]:
    """Encapsule CodeCarbon pour estimer/mesurer l'empreinte carbone d'une fonction.

    Sous WSL2, CodeCarbon n'a généralement pas accès aux compteurs RAPL et bascule
    sur une estimation à partir du TDP et du mix électrique national.
    Cette fonction détecte et expose le mode réellement utilisé.

    Returns
    -------
    (résultat, rapport)
    rapport : dict avec emissions_kg_co2, energy_kwh, duree_s, mode_mesure_carbone,
              facteur_emission_kg_kwh, is_estimation.
    """
    from codecarbon import EmissionsTracker

    mode_mesure, is_estimation = _detecter_mode_mesure()
    logger.info("CodeCarbon — mode détecté : {}", mode_mesure)

    config.TABLES.mkdir(parents=True, exist_ok=True)

    tracker = EmissionsTracker(
        project_name="churn_saas",
        output_dir=str(config.TABLES),
        output_file="codecarbon_emissions.csv",
        log_level="error",
        save_to_file=True,
        save_to_api=False,
        tracking_mode="process",
    )

    tracker.start()
    resultat = fonction()
    emissions_kg: float = tracker.stop() or 0.0

    data = tracker.final_emissions_data
    energy_kwh = float(data.energy_consumed) if data and data.energy_consumed else 0.0
    duree_s = float(data.duration) if data and data.duration else 0.0
    facteur = emissions_kg / energy_kwh if energy_kwh > 0 else 0.0

    rapport: dict[str, Any] = {
        "emissions_kg_co2": float(emissions_kg),
        "energy_kwh": energy_kwh,
        "duree_s": duree_s,
        "mode_mesure_carbone": mode_mesure,
        "facteur_emission_kg_kwh": facteur,
        "is_estimation": is_estimation,
    }

    logger.info(
        "CodeCarbon — {} — {:.6f} kg CO2 ({:.4f} kWh) en {:.1f} s",
        mode_mesure,
        emissions_kg,
        energy_kwh,
        duree_s,
    )

    return resultat, rapport


# ---------------------------------------------------------------------------
# Latence d'inférence
# ---------------------------------------------------------------------------


def mesurer_latence(
    modele: Any,
    X: pd.DataFrame,
    n_unitaire: int = 200,
) -> dict[str, float | int]:
    """Mesure la latence d'inférence unitaire (médiane + p95) et batch.

    Le modèle doit être préalablement fitté.

    Parameters
    ----------
    modele : pipeline scikit-learn fitté.
    X : DataFrame de features (même schéma que l'entraînement).
    n_unitaire : nombre d'appels unitaires — 200 offre une distribution stable.

    Returns
    -------
    dict avec : latence_unitaire_ms_mediane, latence_unitaire_ms_p95,
    latence_batch_5k_s, n_unitaire, n_batch.
    """
    xi = X.iloc[[0]]
    latences_ms: list[float] = []

    for _ in range(n_unitaire):
        t0 = time.perf_counter()
        modele.predict_proba(xi)
        latences_ms.append((time.perf_counter() - t0) * 1000)

    lat_arr = np.array(latences_ms)

    n_batch = min(5_000, len(X))
    X_batch = X.iloc[:n_batch]
    t0 = time.perf_counter()
    modele.predict_proba(X_batch)
    latence_batch_s = time.perf_counter() - t0

    result: dict[str, float | int] = {
        "latence_unitaire_ms_mediane": float(np.median(lat_arr)),
        "latence_unitaire_ms_p95": float(np.percentile(lat_arr, 95)),
        "latence_batch_5k_s": float(latence_batch_s),
        "n_unitaire": n_unitaire,
        "n_batch": n_batch,
    }

    logger.info(
        "Latence — unitaire : médiane={:.2f} ms, p95={:.2f} ms | batch ({} lignes) : {:.3f} s",
        result["latence_unitaire_ms_mediane"],
        result["latence_unitaire_ms_p95"],
        n_batch,
        latence_batch_s,
    )

    return result


# ---------------------------------------------------------------------------
# Optimisation Optuna — TPE + pruning + SQLite persistant
# ---------------------------------------------------------------------------


def _evaluer_params_defaut(modele_nom: str, X: pd.DataFrame, y: pd.Series) -> float:
    """PR-AUC moyen (5-fold CV) avec les hyperparamètres par défaut de construire_modeles."""
    from churn_saas.features.build import construire_preprocesseur

    pre = construire_preprocesseur(X)

    nom = modele_nom.lower()
    clf: Any
    if "aléatoire" in nom or "forêt" in nom:
        clf = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=config.RANDOM_SEED,
            n_jobs=-1,
        )
    elif "logistique" in nom or "régression" in nom:
        clf = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=config.RANDOM_SEED,
        )
    else:
        clf = HistGradientBoostingClassifier(
            max_iter=300,
            random_state=config.RANDOM_SEED,
        )

    pipe = Pipeline([("pre", clone(pre)), ("clf", clf)])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
    pr_aucs: list[float] = []
    y_arr = np.asarray(y)

    for train_idx, val_idx in cv.split(X, y_arr):
        pipe.fit(X.iloc[train_idx], y_arr[train_idx])
        y_proba = pipe.predict_proba(X.iloc[val_idx])[:, 1]
        pr_aucs.append(float(average_precision_score(y_arr[val_idx], y_proba)))

    return float(np.mean(pr_aucs))


def optimiser(
    modele_nom: str,
    X: pd.DataFrame,
    y: pd.Series,
    n_essais: int = 30,
) -> dict[str, Any]:
    """Optimise les hyperparamètres d'un modèle via Optuna (TPE + pruning).

    L'étude est persistée dans un fichier SQLite dans config.TABLES/ :
    si elle existe déjà, Optuna reprend là où il en était (load_if_exists=True).
    L'empreinte carbone du tuning est estimée via CodeCarbon.

    Espace de recherche — Forêt aléatoire :
      n_estimators ∈ {100, 200, 300} ; max_depth ∈ [3, 12] ;
      min_samples_leaf ∈ [1, 10] ; max_features ∈ {sqrt, log2}.

    Espace de recherche — Gradient boosting :
      max_iter ∈ [100, 400, step=50] ; learning_rate ∈ [0.01, 0.3, log] ;
      max_depth ∈ [3, 8] ; l2_regularization ∈ [0.0, 1.0] ;
      min_samples_leaf ∈ [10, 50].

    Returns
    -------
    dict avec : modele_nom, best_params, best_value (PR-AUC), pr_auc_defaut,
    gain_pr_auc, n_essais_demandes, n_essais_completes, emissions_kg_co2,
    energy_kwh, duree_s, mode_mesure_carbone, facteur_emission_kg_kwh, is_estimation.
    """
    import optuna

    safe_name = (
        modele_nom.lower()
        .replace(" ", "_")
        .replace("—", "")
        .replace("é", "e")
        .replace("ê", "e")
        .strip("_")
    )
    storage_path = config.TABLES / f"optuna_{safe_name}.db"
    config.TABLES.mkdir(parents=True, exist_ok=True)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        study_name=f"churn_saas_{safe_name}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.RANDOM_SEED),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True,
    )

    n_existants = len([t for t in study.trials if t.state.name == "COMPLETE"])
    n_restants = max(0, n_essais - n_existants)
    logger.info(
        "Optuna — {} : {} essais complets existants, {} à effectuer",
        modele_nom,
        n_existants,
        n_restants,
    )

    # Évaluation des hyperparamètres par défaut pour quantifier le gain d'Optuna
    pr_auc_defaut = _evaluer_params_defaut(modele_nom, X, y)

    # Objectif local — X, y capturés depuis la portée englobante (usage mono-processus)
    def _objectif(trial: Any) -> float:
        from churn_saas.features.build import construire_preprocesseur

        nom = modele_nom.lower()
        clf_: Any
        if "aléatoire" in nom or "forêt" in nom:
            params = {
                "n_estimators": trial.suggest_categorical("n_estimators", [100, 200, 300]),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            }
            clf_ = RandomForestClassifier(
                class_weight="balanced",
                random_state=config.RANDOM_SEED,
                n_jobs=-1,
                **params,
            )
        elif "logistique" in nom or "régression" in nom:
            params = {
                "C": trial.suggest_float("C", 0.001, 100.0, log=True),
                "max_iter": trial.suggest_categorical("max_iter", [500, 1000, 2000]),
            }
            clf_ = LogisticRegression(
                class_weight="balanced",
                random_state=config.RANDOM_SEED,
                solver="lbfgs",
                **params,
            )
        else:
            params = {
                "max_iter": trial.suggest_int("max_iter", 100, 400, step=50),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 1.0),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 50),
            }
            clf_ = HistGradientBoostingClassifier(
                random_state=config.RANDOM_SEED,
                **params,
            )

        pre = construire_preprocesseur(X)
        pipe = Pipeline([("pre", clone(pre)), ("clf", clf_)])
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
        pr_aucs_: list[float] = []
        y_arr = np.asarray(y)

        for pli, (train_idx, val_idx) in enumerate(cv.split(X, y_arr)):
            pipe.fit(X.iloc[train_idx], y_arr[train_idx])
            y_proba = pipe.predict_proba(X.iloc[val_idx])[:, 1]
            pr_aucs_.append(float(average_precision_score(y_arr[val_idx], y_proba)))
            trial.report(float(np.mean(pr_aucs_)), pli)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return float(np.mean(pr_aucs_))

    def _lancer_optuna() -> None:
        if n_restants > 0:
            study.optimize(_objectif, n_trials=n_restants, show_progress_bar=False)

    _, rapport_carbone = mesurer_empreinte(_lancer_optuna)

    meilleur = study.best_trial
    best_value = float(meilleur.value) if meilleur.value is not None else 0.0
    gain = best_value - pr_auc_defaut

    result: dict[str, Any] = {
        "modele_nom": modele_nom,
        "best_params": meilleur.params,
        "best_value": best_value,
        "pr_auc_defaut": float(pr_auc_defaut),
        "gain_pr_auc": float(gain),
        "n_essais_demandes": n_essais,
        "n_essais_completes": len([t for t in study.trials if t.state.name == "COMPLETE"]),
        **rapport_carbone,
    }

    logger.info(
        "Optuna terminé — best PR-AUC = {:.4f} (défaut = {:.4f}, gain = {:+.4f})",
        meilleur.value,
        pr_auc_defaut,
        gain,
    )

    return result


# ---------------------------------------------------------------------------
# Reconstruction d'un pipeline avec hyperparamètres optimisés
# ---------------------------------------------------------------------------


def construire_modele_optimise(
    modele_nom: str,
    params: dict[str, Any],
    df_ref: pd.DataFrame,
) -> Pipeline:
    """Reconstruit un Pipeline sklearn avec les hyperparamètres optimisés par Optuna.

    Parameters
    ----------
    modele_nom : nom du modèle (doit correspondre à la famille utilisée dans optimiser).
    params : dict de hyperparamètres retournés par optimiser()["best_params"].
    df_ref : DataFrame de référence pour construire le préprocesseur (non fitté).

    Returns
    -------
    Pipeline non fitté, prêt pour .fit(X_train, y_train).
    """
    from churn_saas.features.build import construire_preprocesseur

    pre = construire_preprocesseur(df_ref)

    nom = modele_nom.lower()
    clf: Any
    if "aléatoire" in nom or "forêt" in nom:
        clf = RandomForestClassifier(
            class_weight="balanced",
            random_state=config.RANDOM_SEED,
            n_jobs=-1,
            **params,
        )
    elif "logistique" in nom or "régression" in nom:
        clf = LogisticRegression(
            class_weight="balanced",
            random_state=config.RANDOM_SEED,
            solver="lbfgs",
            **params,
        )
    else:
        clf = HistGradientBoostingClassifier(
            random_state=config.RANDOM_SEED,
            **params,
        )

    return Pipeline([("pre", clone(pre)), ("clf", clf)])
