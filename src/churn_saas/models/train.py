"""Protocole de validation, règle métier et évaluation partagés entre tous les modèles de §8-9."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from loguru import logger
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
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
from churn_saas.cache import recalcul_force
from churn_saas.models.calibration import RegressionLogistiqueRecalibree


def protocole_validation() -> RepeatedStratifiedKFold:
    """Retourne l'objet CV partagé par TOUS les modèles comparés en §8-9.

    Un seul objet garantit que chaque modèle voit exactement les mêmes plis
    (train/val identiques), condition nécessaire pour que la comparaison soit honnête.
    Paramètres : 5 plis × 3 répétitions = 15 scores par métrique, conservés pli par
    pli pour le test de Wilcoxon apparié de ``comparer_au_champion()``.
    Seed = config.RANDOM_SEED (unique dans tout le projet).
    """
    return RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=config.RANDOM_SEED)


# Seuils de la règle métier — fixés a priori, documentés dans `config` (autorité unique)
SEUIL_INACTIVITE_JOURS: int = config.SEUIL_INACTIVITE_JOURS
SEUIL_CSAT: int = config.SEUIL_CSAT


def regle_metier(df: pd.DataFrame) -> pd.Series:
    """Baseline non-ML : derniere_connexion_jours > 30 OU csat <= 2.

    Retourne des prédictions binaires (0 ou 1) sous forme de pd.Series (int).
    Les lignes avec les deux colonnes manquantes sont prédites 0 (non-churn par défaut).

    Seuils fixés a priori (voir SEUIL_INACTIVITE_JOURS, SEUIL_CSAT) — jamais du jeu de test.
    Rôle : mesurer si le ML apporte un gain réel sur une règle simple, applicable sans ML.
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

NOM_BASELINE_LR = "Baseline — régression logistique"
NOM_FORET = "Forêt aléatoire"
NOM_LIGHTGBM = "LightGBM"

# Ordre de simplicité (règle de parcimonie, §8.4) : à performance non départagée, le modèle
# le plus simple est retenu. Les deux ensembles d'arbres sont au même niveau.
ORDRE_SIMPLICITE: dict[str, int] = {
    "B0 — Hasard stratifié": 0,
    "B1 — Règle métier": 1,
    NOM_BASELINE_LR: 2,
    NOM_FORET: 3,
    NOM_LIGHTGBM: 3,
}


def _famille(modele_nom: str) -> str:
    """Famille d'estimateur (``foret``, ``logistique``, ``lightgbm``) déduite du nom du modèle.

    Les noms historiques (« B2 — Régression logistique ») restent reconnus, pour relire la
    méta d'un champion promu avant le renommage.
    """
    nom = modele_nom.lower()
    if "aléatoire" in nom or "forêt" in nom:
        return "foret"
    if "logistique" in nom:
        return "logistique"
    if "lightgbm" in nom:
        return "lightgbm"
    raise ValueError(f"Famille de modèle inconnue : {modele_nom!r}.")


def _classifieur(modele_nom: str, params: dict[str, Any] | None = None) -> Any:
    """Estimateur non fitté de la famille de ``modele_nom``.

    Sans ``params``, ce sont les hyperparamètres par défaut de ``construire_modeles`` ;
    avec, ceux d'un essai ou du meilleur essai Optuna, qui remplacent les défauts.
    """
    params = params or {}
    famille = _famille(modele_nom)
    if famille == "foret":
        return RandomForestClassifier(
            **{
                "n_estimators": 200,
                "class_weight": "balanced",
                "random_state": config.RANDOM_SEED,
                "n_jobs": -1,
                **params,
            }
        )
    if famille == "logistique":
        return RegressionLogistiqueRecalibree(
            **{
                "class_weight": "balanced",
                "max_iter": 1000,
                "random_state": config.RANDOM_SEED,
                "solver": "lbfgs",
                **params,
            }
        )
    # subsample_freq=1 : sans fréquence de bagging, LightGBM ignore `subsample` ; avec
    # subsample=1.0 (défaut), le bagging est sans effet.
    return LGBMClassifier(
        **{
            "class_weight": "balanced",
            "random_state": config.RANDOM_SEED,
            "n_jobs": 1,
            "verbose": -1,
            "subsample_freq": 1,
            **params,
        }
    )


def identifiant_modele(modele_nom: str) -> str:
    """Identifiant ASCII d'un modèle, utilisé dans les noms d'artefacts (étude Optuna, cache).

    Exemple : « Baseline — régression logistique » → ``baseline__regression_logistique``.
    """
    return (
        modele_nom.lower()
        .replace(" ", "_")
        .replace("—", "")
        .replace("é", "e")
        .replace("ê", "e")
        .strip("_")
    )


def construire_modeles(df: pd.DataFrame) -> dict[str, Any]:
    """Construit un dict {nom → Pipeline complet} pour toutes les familles.

    Famille linéaire : RegressionLogistiqueRecalibree(class_weight='balanced') — probabilités
    recalibrées par correction d'intercept (``models.calibration``).
    Famille arbres   : RandomForestClassifier + LGBMClassifier (LightGBM).
    Baselines        : B0 (hasard), B1 (règle métier), baseline régression logistique
    (référence ML).

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
        NOM_BASELINE_LR: _pipe(_classifieur(NOM_BASELINE_LR)),
        NOM_FORET: _pipe(_classifieur(NOM_FORET)),
        NOM_LIGHTGBM: _pipe(_classifieur(NOM_LIGHTGBM)),
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


def _types_a_approuver(modele: Any) -> list[str]:
    """Types que skops refuse par défaut dans un modèle que l'on vient d'entraîner soi-même.

    MLflow sérialise les modèles scikit-learn avec skops, qui n'accepte au chargement que
    les classes de sa liste blanche. Sont refusées les classes du projet
    (``RegressionLogistiqueRecalibree``, transformateurs de ``features``) et certaines
    structures internes de scikit-learn (nœuds d'arbres). Le modèle est produit dans ce même
    processus : c'est le cas « fichier créé soi-même » pour lequel skops prévoit de déclarer
    ces types de confiance. La liste est calculée sur ce modèle précis, jamais générique.
    """
    import skops.io as sio

    return sorted(sio.get_untrusted_types(data=sio.dumps(modele)))


@contextmanager
def _mlflow_silencieux() -> Iterator[None]:
    """Relève le logger ``mlflow`` au niveau WARNING le temps du bloc, puis le restaure.

    MLflow émet une dizaine d'INFO par run (création de la base, détection du projet uv,
    export des dépendances) qui noient le journal sans rien apprendre au lecteur ; les WARNING passent.
    """
    journal = logging.getLogger("mlflow")
    niveau_avant = journal.level
    journal.setLevel(logging.WARNING)
    try:
        yield
    finally:
        journal.setLevel(niveau_avant)


def journaliser_mlflow(
    nom: str,
    modele: Any,
    metriques: dict[str, float],
    params: dict[str, Any],
    tags: dict[str, str] | None = None,
) -> str:
    """Crée un run MLflow pour un modèle, logge params, métriques et artefact.

    Backend SQLite local dans ``config.MLRUNS`` (``mlflow.db``).
    Expérience unique : « churn_saas_classification ».

    Returns
    -------
    str — identifiant du run MLflow créé.
    """
    config.MLRUNS.mkdir(parents=True, exist_ok=True)
    # SQLite local — compatible avec MLflow >= 2.14 qui a déprécié le backend fichier pur
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    with _mlflow_silencieux():
        mlflow.set_experiment("churn_saas_classification")

    with _mlflow_silencieux(), mlflow.start_run(run_name=nom) as run:
        mlflow.log_params({k: str(v) for k, v in params.items()})
        for cle, val in metriques.items():
            if isinstance(val, (int, float)):
                mlflow.log_metric(cle, float(val))
        if tags:
            mlflow.set_tags(tags)
        try:
            mlflow.sklearn.log_model(
                modele, name="model", skops_trusted_types=_types_a_approuver(modele)
            )
        except Exception as exc:
            logger.warning("MLflow : impossible de logguer le modèle '{}' — {}", nom, exc)

        run_id: str = str(run.info.run_id)

    logger.info("MLflow run créé — nom='{}', run_id={}", nom, run_id)
    return run_id


def enregistrer_au_registre(run_id: str, alias: str = "production") -> str:
    """Enregistre le modèle d'un run au registre MLflow et lui attribue un alias.

    Le registre (``config.NOM_REGISTRE_MLFLOW``) versionne le modèle déployé : chaque appel
    crée une version, et l'alias (``production`` par défaut) est déplacé sur elle. L'ancienne
    version reste au registre, ce qui permet le retour arrière.

    Returns
    -------
    str — numéro de la version créée.
    """
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    with _mlflow_silencieux():
        version = mlflow.register_model(f"runs:/{run_id}/model", config.NOM_REGISTRE_MLFLOW)
    mlflow.MlflowClient().set_registered_model_alias(
        config.NOM_REGISTRE_MLFLOW, alias, version.version
    )
    logger.info(
        "Registre MLflow — {} v{} → alias @{}", config.NOM_REGISTRE_MLFLOW, version.version, alias
    )
    return str(version.version)


# ---------------------------------------------------------------------------
# Gate de promotion — règle unique du flow de réentraînement (§10.4, §13.8)
# ---------------------------------------------------------------------------


def decision_promotion(
    pr_auc_challenger: float,
    pr_auc_champion: float | None,
) -> tuple[bool, str]:
    """Décide si un challenger peut remplacer le champion.

    Deux conditions cumulatives :
    1. ``pr_auc_challenger >= config.CIBLES_PERFORMANCE["pr_auc_min"]`` (seuil a priori) ;
    2. ``pr_auc_challenger >= pr_auc_champion − tolerance_regression_promotion``
       (pas de régression au-delà de la tolérance) — sans objet s'il n'y a pas de champion.

    Returns
    -------
    (promouvoir, motif) — le motif explique la décision en une phrase.
    """
    seuil = float(config.CIBLES_PERFORMANCE["pr_auc_min"])
    tolerance = float(config.CIBLES_PERFORMANCE["tolerance_regression_promotion"])
    from churn_saas.format_fr import nombre

    if pr_auc_challenger < seuil:
        return False, (
            f"PR-AUC {nombre(pr_auc_challenger, 4)} sous le seuil a priori {nombre(seuil, 2)}"
        )
    if pr_auc_champion is not None and pr_auc_challenger < pr_auc_champion - tolerance:
        return False, (
            f"régression : PR-AUC {nombre(pr_auc_challenger, 4)} inférieure à celle du champion "
            f"({nombre(pr_auc_champion, 4)}) moins la tolérance ({nombre(tolerance, 2)})"
        )
    return True, (
        f"PR-AUC {nombre(pr_auc_challenger, 4)} au moins égale au seuil {nombre(seuil, 2)}, sans régression "
        f"au-delà de la tolérance {nombre(tolerance, 2)}"
    )


# ---------------------------------------------------------------------------
# Comparaison class_weight vs SMOTE — calibration
# ---------------------------------------------------------------------------


def comparer_desequilibre(
    X: pd.DataFrame,
    y: pd.Series,
    df_ref: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compare trois traitements du déséquilibre sur régression logistique.

    - ``class_weight='balanced'`` brut ;
    - ``class_weight='balanced'`` + correction d'intercept (``RegressionLogistiqueRecalibree``) ;
    - SMOTE, appliqué DANS le pipeline imblearn — jamais hors CV.

    Mesures retournées :
    - PR-AUC, ROC-AUC, Brier score (OOF sur 5 plis)
    - Courbe de fiabilité (calibration_curve sur les OOF probas)

    La pondération comme SMOTE décalent les probabilités vers la classe minoritaire. Seul le
    décalage de la pondération est connu analytiquement (``log(w₁/w₀)`` sur le logit) et donc
    exactement corrigeable : c'est l'approche retenue pour les calculs en euros (§12).

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
        "class_weight='balanced' + correction d'intercept": Pipeline(
            [
                ("pre", clone(pre)),
                (
                    "clf",
                    RegressionLogistiqueRecalibree(
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
    from codecarbon.output_methods.base_output import OutputMethod

    mode_mesure, is_estimation = _detecter_mode_mesure()
    logger.info("CodeCarbon — mode détecté : {}", mode_mesure)

    config.TABLES.mkdir(parents=True, exist_ok=True)

    tracker = EmissionsTracker(
        project_name="churn_saas",
        output_dir=str(config.TABLES),
        output_file="codecarbon_emissions.csv",
        log_level="error",
        output_methods=[OutputMethod.CSV],
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
    clf = _classifieur(modele_nom)

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

    Espace de recherche — Régression logistique :
      C ∈ [0.001, 100, log] ; max_iter ∈ {500, 1000, 2000}.

    Espace de recherche — LightGBM :
      num_leaves ∈ [8, 64, log] ; learning_rate ∈ [0.01, 0.3, log] ;
      n_estimators ∈ [100, 500, step=50] ; min_child_samples ∈ [5, 100, log] ;
      reg_lambda ∈ [0.001, 10, log] ; subsample ∈ [0.6, 1.0] ; colsample_bytree ∈ [0.5, 1.0].

    Pour une comparaison équitable, chaque famille reçoit le même budget ``n_essais`` ; la
    sélection finale compare ensuite les modèles optimisés entre eux, sur les plis du
    protocole (``comparer_modeles_optimises``, ``selectionner_modele_optimise``).

    Returns
    -------
    dict avec : modele_nom, best_params, best_value (PR-AUC), pr_auc_defaut,
    gain_pr_auc, n_essais_demandes, n_essais_completes, emissions_kg_co2,
    energy_kwh, duree_s, mode_mesure_carbone, facteur_emission_kg_kwh, is_estimation.
    """
    import optuna

    safe_name = identifiant_modele(modele_nom)
    storage_path = config.TABLES / f"optuna_{safe_name}.db"
    config.TABLES.mkdir(parents=True, exist_ok=True)
    # Recalcul forcé : l'étude persistée reprendrait ses anciens essais (load_if_exists),
    # obtenus peut-être avec un autre préprocesseur — on repart d'une étude vide
    if recalcul_force() and storage_path.exists():
        storage_path.unlink()
        logger.info("Optuna — recalcul forcé : étude précédente supprimée ({})", storage_path.name)

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

        famille = _famille(modele_nom)
        params: dict[str, Any]
        if famille == "foret":
            params = {
                "n_estimators": trial.suggest_categorical("n_estimators", [100, 200, 300]),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            }
        elif famille == "logistique":
            params = {
                "C": trial.suggest_float("C", 0.001, 100.0, log=True),
                "max_iter": trial.suggest_categorical("max_iter", [500, 1000, 2000]),
            }
        else:
            params = {
                "num_leaves": trial.suggest_int("num_leaves", 8, 64, log=True),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 10.0, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            }
        clf_ = _classifieur(modele_nom, params)

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
# Famille et hyperparamètres du champion courant (CLI et flow de réentraînement)
# ---------------------------------------------------------------------------

# Repli si ni champion promu ni étude Optuna n'existent : le modèle retenu en §9.3.1
FAMILLE_PAR_DEFAUT = NOM_BASELINE_LR
HYPERPARAMETRES_PAR_DEFAUT: dict[str, Any] = {"max_iter": 1000}


def famille_et_hyperparametres_champion() -> tuple[str, dict[str, Any]]:
    """Famille et hyperparamètres du champion courant, base de tout nouvel entraînement.

    Ordre de recherche : méta du champion promu (``best_model_meta.json``, si elle porte sa
    famille), puis la plus récente étude Optuna du notebook (§9.7), sinon le repli.
    C'est le transfert de connaissances décrit en §9.11 : on réutilise le savoir acquis sur
    le modèle, sans relancer de recherche d'hyperparamètres.
    """
    import json

    chemin_meta = config.ARTIFACTS / "models" / "best_model_meta.json"
    if chemin_meta.exists():
        meta = json.loads(chemin_meta.read_text(encoding="utf-8"))
        if "modele_nom" in meta and "hyperparametres" in meta:
            logger.info("Famille et hyperparamètres repris du champion promu.")
            return str(meta["modele_nom"]), dict(meta["hyperparametres"])

    etudes = sorted(
        config.TABLES.glob("optuna_*_meilleurs_params.json"), key=lambda c: c.stat().st_mtime
    )
    if etudes:
        resultat = json.loads(etudes[-1].read_text(encoding="utf-8"))
        logger.info("Famille et hyperparamètres repris de l'étude Optuna {}.", etudes[-1].name)
        return str(resultat["modele_nom"]), dict(resultat["best_params"])

    logger.warning("Aucun champion ni étude Optuna : famille de repli {}.", FAMILLE_PAR_DEFAUT)
    return FAMILLE_PAR_DEFAUT, dict(HYPERPARAMETRES_PAR_DEFAUT)


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

    Raises
    ------
    ValueError
        Si la famille du modèle n'est pas reconnue (forêt, logistique, LightGBM).
    """
    from churn_saas.features.build import construire_preprocesseur

    pre = construire_preprocesseur(df_ref)
    clf = _classifieur(modele_nom, params)
    return Pipeline([("pre", clone(pre)), ("clf", clf)])


# ---------------------------------------------------------------------------
# Sélection finale entre modèles optimisés
# ---------------------------------------------------------------------------


def comparer_modeles_optimises(
    resultats_optuna: list[dict[str, Any]],
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    """Réévalue chaque modèle optimisé sur les plis du protocole (``protocole_validation``).

    La PR-AUC d'Optuna (meilleur de N essais sur 5 plis) est biaisée vers le haut : elle sert
    à choisir les hyperparamètres, pas à départager les familles. Les modèles optimisés sont
    donc reconstruits puis comparés sur les mêmes 15 plis, ce qui fournit les scores par pli
    nécessaires au test de Wilcoxon de ``selectionner_modele_optimise``.

    Parameters
    ----------
    resultats_optuna : list[dict]
        Sorties de ``optimiser()`` (clés ``modele_nom`` et ``best_params``), une par famille.
    X, y :
        Features et cible d'entraînement.

    Returns
    -------
    pd.DataFrame au format de ``comparer_modeles()``, trié par PR-AUC décroissante.
    """
    modeles = {
        r["modele_nom"]: construire_modele_optimise(r["modele_nom"], r["best_params"], df_ref=X)
        for r in resultats_optuna
    }
    return comparer_modeles(modeles, X, y)


def selectionner_modele_optimise(
    tableau: pd.DataFrame,
    gain_min: float | None = None,
    alpha: float = ALPHA_SIGNIFICATIVITE,
) -> tuple[str, pd.DataFrame]:
    """Retient un modèle parmi les modèles optimisés, avec la règle de parcimonie (§8.4).

    Le meilleur en PR-AUC moyenne est le champion provisoire. Tout modèle qu'il ne bat pas
    significativement (Wilcoxon + Holm), ou de moins de ``gain_min``, reste candidat. Parmi
    les candidats, le plus simple (``ORDRE_SIMPLICITE``) l'emporte ; à simplicité égale, la
    meilleure PR-AUC.

    Parameters
    ----------
    tableau : pd.DataFrame
        Sortie de ``comparer_modeles_optimises()`` (colonnes ``pr_auc_mean`` et ``pr_auc_pli_*``).
    gain_min : float, optional
        Gain minimal de PR-AUC exigé d'un modèle plus complexe ; par défaut
        ``config.CIBLES_PERFORMANCE["gain_pr_auc_min_complexite"]``.
    alpha : float
        Risque global d'erreur de première espèce du test de supériorité.

    Returns
    -------
    (nom du modèle retenu, tests de supériorité du champion provisoire)
    """
    if gain_min is None:
        gain_min = float(config.CIBLES_PERFORMANCE["gain_pr_auc_min_complexite"])

    champion = str(tableau["pr_auc_mean"].idxmax())
    tests = comparer_au_champion(tableau, champion, alpha=alpha)
    candidats = [champion] + [
        str(n)
        for n in tests.index
        if not tests.loc[n, "significatif"] or tests.loc[n, "ecart_pr_auc_moyen"] < gain_min
    ]
    niveau_inconnu = max(ORDRE_SIMPLICITE.values()) + 1
    retenu = min(
        candidats,
        key=lambda n: (
            ORDRE_SIMPLICITE.get(n, niveau_inconnu),
            -float(tableau.loc[n, "pr_auc_mean"]),
        ),
    )
    logger.info(
        "Sélection parmi les modèles optimisés : champion provisoire '{}', candidats {}, "
        "retenu '{}'",
        champion,
        candidats,
        retenu,
    )
    return retenu, tests
