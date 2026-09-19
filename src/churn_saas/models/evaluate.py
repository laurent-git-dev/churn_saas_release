"""Évaluation des modèles de classification churn — sorties obligatoires C4/C8.

Toutes les figures sont produites via :mod:`churn_saas.viz` (style unique,
numérotation cohérente, sauvegarde automatique dans ``reports/figures/``).
Les métriques indépendantes du seuil (PR-AUC, ROC-AUC, Brier) sont calculées
sur les probabilités brutes ; les autres sont calculées après seuillage.
"""

from __future__ import annotations

import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib.figure import Figure
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from churn_saas import viz


def courbe_roc(
    y: pd.Series | np.ndarray,
    proba: np.ndarray,
    *,
    nom_modele: str = "modèle",
) -> tuple[Figure, float]:
    """Trace la courbe ROC avec l'AUC annotée — sortie obligatoire C4/C8.

    Parameters
    ----------
    y :
        Vecteur de vérité terrain (0 = fidèle, 1 = résiliation).
    proba :
        Probabilités prédites de la classe positive (1D, même longueur que y).
    nom_modele :
        Étiquette affichée dans la légende.

    Returns
    -------
    (fig, auc_roc)
    """
    y_arr = np.asarray(y)
    auc = float(roc_auc_score(y_arr, proba))
    fpr, tpr, _ = roc_curve(y_arr, proba)

    fig, ax = viz.figure(
        "courbe_roc",
        "Courbe ROC — discrimination du modèle de churn",
        taille=(7.0, 6.0),
    )

    ax.plot(
        fpr,
        tpr,
        color=viz.COULEUR_CHURN,
        linewidth=2.5,
        label=f"{nom_modele} (AUC = {auc:.3f})",
    )
    ax.plot(
        [0, 1],
        [0, 1],
        "--",
        color="#888888",
        linewidth=1.2,
        label="Classificateur aléatoire (AUC = 0,500)",
    )
    ax.fill_between(fpr, tpr, alpha=0.08, color=viz.COULEUR_CHURN)

    ax.set_xlabel("Taux de faux positifs (1 − spécificité)")
    ax.set_ylabel("Taux de vrais positifs (sensibilité / rappel)")
    ax.legend(loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.text(
        0.06,
        0.88,
        f"AUC-ROC = {auc:.3f}",
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        color=viz.COULEUR_CHURN,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
    )

    viz.sauvegarder(fig)
    logger.info("Courbe ROC tracée — AUC-ROC = {:.4f}", auc)
    return fig, auc


def courbe_precision_rappel(
    y: pd.Series | np.ndarray,
    proba: np.ndarray,
    *,
    nom_modele: str = "modèle",
) -> tuple[Figure, float]:
    """Trace la courbe Précision-Rappel avec la PR-AUC annotée — sortie obligatoire C4/C8.

    La ligne de référence aléatoire est tracée à la prévalence du churn.
    La PR-AUC est la métrique principale pour les jeux déséquilibrés.

    Returns
    -------
    (fig, pr_auc)
    """
    y_arr = np.asarray(y)
    prevalence = float(y_arr.mean())
    pr_auc = float(average_precision_score(y_arr, proba))
    precision_vals, rappel_vals, _ = precision_recall_curve(y_arr, proba)

    fig, ax = viz.figure(
        "courbe_precision_rappel",
        "Courbe Précision-Rappel — PR-AUC",
        taille=(7.0, 6.0),
    )

    ax.plot(
        rappel_vals,
        precision_vals,
        color=viz.COULEUR_CHURN,
        linewidth=2.5,
        label=f"{nom_modele} (PR-AUC = {pr_auc:.3f})",
    )
    ax.axhline(
        prevalence,
        linestyle="--",
        color="#888888",
        linewidth=1.2,
        label=f"Aléatoire (prévalence churn ≈ {prevalence:.2f})",
    )
    ax.fill_between(
        rappel_vals,
        precision_vals,
        prevalence,
        where=(precision_vals >= prevalence),
        alpha=0.08,
        color=viz.COULEUR_CHURN,
    )

    ax.set_xlabel("Rappel (proportion de churners détectés)")
    ax.set_ylabel("Précision (fiabilité des alertes churn)")
    ax.legend(loc="upper right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(max(-0.02, prevalence - 0.10), 1.05)
    ax.text(
        0.06,
        0.12,
        f"PR-AUC = {pr_auc:.3f}",
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        color=viz.COULEUR_CHURN,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
    )

    viz.sauvegarder(fig)
    logger.info("Courbe Précision-Rappel tracée — PR-AUC = {:.4f}", pr_auc)
    return fig, pr_auc


def matrice_confusion(
    y: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    seuil: float,
) -> tuple[Figure, np.ndarray]:
    """Trace la matrice de confusion annotée en effectifs ET en pourcentages.

    Le seuil de décision est rappelé dans le titre.

    Returns
    -------
    (fig, cm) — la figure et la matrice numpy (shape 2×2, ordre sklearn : [TN FP / FN TP]).
    """
    y_arr = np.asarray(y)
    cm = confusion_matrix(y_arr, y_pred)
    cm_pct = cm.astype(float) / cm.sum() * 100.0

    # Couleurs sémantiques : vert pour les prédictions correctes, rouge pour les erreurs
    couleurs_fond = [
        [viz.COULEUR_NON_CHURN, viz.COULEUR_CHURN],
        [viz.COULEUR_CHURN, viz.COULEUR_NON_CHURN],
    ]
    etiquettes = [["VN", "FP"], ["FN", "VP"]]

    fig, ax = viz.figure(
        "matrice_confusion",
        f"Matrice de confusion — seuil de décision = {seuil:.2f}",
        taille=(7.0, 5.5),
    )

    for i in range(2):
        for j in range(2):
            rect = mpatches.FancyBboxPatch(
                (j - 0.45, i - 0.45),
                0.90,
                0.90,
                boxstyle="round,pad=0.04",
                facecolor=couleurs_fond[i][j],
                alpha=0.25,
                linewidth=0,
            )
            ax.add_patch(rect)
            ax.text(
                j,
                i + 0.14,
                f"{cm[i, j]:,}",
                ha="center",
                va="center",
                fontsize=18,
                fontweight="bold",
                color="#1a1a1a",
            )
            ax.text(
                j,
                i - 0.12,
                f"({cm_pct[i, j]:.1f} %)",
                ha="center",
                va="center",
                fontsize=11,
                color="#444444",
            )
            ax.text(
                j,
                i - 0.34,
                etiquettes[i][j],
                ha="center",
                va="center",
                fontsize=9,
                color="#666666",
                style="italic",
            )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Prédit Non-Churn (0)", "Prédit Churn (1)"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Réel Non-Churn (0)", "Réel Churn (1)"])
    ax.set_xlabel("Prédiction du modèle")
    ax.set_ylabel("Classe réelle (observation)")
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(-0.6, 1.6)
    ax.grid(False)
    ax.set_facecolor("white")

    viz.sauvegarder(fig)
    logger.info(
        "Matrice de confusion (seuil={}) — VN={}, FP={}, FN={}, VP={}",
        seuil,
        cm[0, 0],
        cm[0, 1],
        cm[1, 0],
        cm[1, 1],
    )
    return fig, cm


def courbe_calibration(
    y: pd.Series | np.ndarray,
    proba: np.ndarray,
    *,
    nom_modele: str = "modèle",
    n_bins: int = 10,
) -> tuple[Figure, float]:
    """Trace la courbe de fiabilité des probabilités et annote le score de Brier.

    Indispensable lorsque les probabilités sont utilisées pour le calcul du ROI
    (valeur sauvée × P(churn) × taux de succès de la rétention).

    Returns
    -------
    (fig, brier_score) — la figure et le score de Brier (0 = parfait, 0,25 = aléatoire).
    """
    y_arr = np.asarray(y)
    brier = float(brier_score_loss(y_arr, proba))
    prob_true, prob_pred = calibration_curve(y_arr, proba, n_bins=n_bins, strategy="quantile")

    fig, ax = viz.figure(
        "courbe_calibration",
        "Courbe de fiabilité des probabilités prédites (calibration)",
        taille=(7.0, 6.0),
    )

    ax.plot([0, 1], [0, 1], "--", color="#888888", linewidth=1.5, label="Calibration parfaite")
    ax.plot(
        prob_pred,
        prob_true,
        "o-",
        color=viz.COULEUR_CHURN,
        linewidth=2.0,
        markersize=8,
        label=f"{nom_modele}",
    )
    ax.fill_between(
        prob_pred,
        prob_true,
        prob_pred,
        alpha=0.10,
        color=viz.COULEUR_CHURN,
        label="Écart de calibration",
    )

    ax.set_xlabel("Probabilité prédite (moyenne par décile)")
    ax.set_ylabel("Proportion observée de churn réel")
    ax.legend(loc="upper left")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.05, 1.10)
    ax.text(
        0.55,
        0.10,
        f"Score de Brier = {brier:.4f}\n(0 = parfait · 0,25 = aléatoire)",
        transform=ax.transAxes,
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.85},
    )

    viz.sauvegarder(fig)
    logger.info("Courbe de calibration tracée — Score de Brier = {:.4f}", brier)
    return fig, brier


def tableau_metriques(
    y: pd.Series | np.ndarray,
    proba: np.ndarray,
    seuil: float,
) -> pd.DataFrame:
    """Calcule un tableau de métriques de classification à un seuil donné.

    Métriques : PR-AUC, ROC-AUC, précision, rappel, F1, spécificité, score de Brier.
    Les métriques indépendantes du seuil (PR-AUC, ROC-AUC, Brier) sont calculées
    sur les probabilités brutes ; les autres sont calculées après seuillage.

    Returns
    -------
    DataFrame avec colonnes [métrique, valeur, description].
    """
    y_arr = np.asarray(y)
    y_pred = (proba >= seuil).astype(int)
    cm = confusion_matrix(y_arr, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificite = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    lignes: list[tuple[str, float, str]] = [
        (
            "PR-AUC",
            float(average_precision_score(y_arr, proba)),
            "Aire Précision-Rappel — métrique principale sur données déséquilibrées",
        ),
        (
            "ROC-AUC",
            float(roc_auc_score(y_arr, proba)),
            "Aire ROC — capacité de discrimination globale",
        ),
        (
            f"Précision  (seuil {seuil:.2f})",
            float(precision_score(y_arr, y_pred, zero_division=0)),
            "TP / (TP + FP) — fiabilité des alertes churn",
        ),
        (
            f"Rappel     (seuil {seuil:.2f})",
            float(recall_score(y_arr, y_pred, zero_division=0)),
            "TP / (TP + FN) — taux de détection des churners",
        ),
        (
            f"F1         (seuil {seuil:.2f})",
            float(f1_score(y_arr, y_pred, zero_division=0)),
            "Moyenne harmonique précision-rappel",
        ),
        (
            f"Spécificité (seuil {seuil:.2f})",
            specificite,
            "TN / (TN + FP) — non-churners correctement identifiés",
        ),
        (
            "Score de Brier",
            float(brier_score_loss(y_arr, proba)),
            "Erreur quadratique des probabilités — qualité de calibration",
        ),
    ]

    df = pd.DataFrame(lignes, columns=["métrique", "valeur", "description"])
    df["valeur"] = df["valeur"].round(4)
    logger.info(
        "Métriques (seuil={:.2f}) — PR-AUC={:.4f}, Rappel={:.4f}, Précision={:.4f}",
        seuil,
        float(average_precision_score(y_arr, proba)),
        float(recall_score(y_arr, y_pred, zero_division=0)),
        float(precision_score(y_arr, y_pred, zero_division=0)),
    )
    return df


def analyse_erreurs(
    df: pd.DataFrame,
    y: pd.Series | np.ndarray,
    proba: np.ndarray,
    seuil: float,
) -> pd.DataFrame:
    """Caractérise les faux négatifs et les faux positifs pour la section « limites ».

    Identifie les segments systématiquement mal prédits : y a-t-il un profil
    de compte que le modèle rate structurellement ?

    Parameters
    ----------
    df :
        DataFrame avec les features originales (sans la cible).
    y :
        Vérité terrain (0 = fidèle, 1 = résiliation).
    proba :
        Probabilités prédites de la classe positive.
    seuil :
        Seuil de classification appliqué à ``proba``.

    Returns
    -------
    DataFrame indexé par segment d'erreur (FN / FP / VP / VN) avec les
    moyennes des features numériques et le nombre d'observations.
    """
    y_arr = np.asarray(y)
    y_pred = (proba >= seuil).astype(int)

    travail = df.copy()
    travail["_proba_pred"] = proba

    masques: dict[str, np.ndarray] = {
        "Faux Négatifs (FN)": (y_arr == 1) & (y_pred == 0),
        "Faux Positifs (FP)": (y_arr == 0) & (y_pred == 1),
        "Vrais Positifs (VP)": (y_arr == 1) & (y_pred == 1),
        "Vrais Négatifs (VN)": (y_arr == 0) & (y_pred == 0),
    }

    cols_num = [c for c in df.select_dtypes(include="number").columns if not c.startswith("_")]
    cols_analyser = cols_num + ["_proba_pred"]

    segments = []
    for nom_seg, masque in masques.items():
        sous = travail.loc[masque, cols_analyser]
        stats: dict[str, object] = {
            "segment": nom_seg,
            "n_observations": int(masque.sum()),
        }
        for col in cols_analyser:
            label = "proba_pred_moyenne" if col == "_proba_pred" else col
            stats[label] = round(float(sous[col].mean()), 3) if len(sous) > 0 else float("nan")
        segments.append(stats)

    résumé = pd.DataFrame(segments).set_index("segment")

    n_fn = int(masques["Faux Négatifs (FN)"].sum())
    n_fp = int(masques["Faux Positifs (FP)"].sum())
    n_vp = int(masques["Vrais Positifs (VP)"].sum())
    n_vn = int(masques["Vrais Négatifs (VN)"].sum())
    logger.info(
        "Analyse des erreurs (seuil={:.2f}) — FN={}, FP={}, VP={}, VN={}",
        seuil,
        n_fn,
        n_fp,
        n_vp,
        n_vn,
    )

    # Distribution catégorielle par segment — journalisée pour la section limites
    cols_cat = [
        c for c in df.select_dtypes(include=["object", "str"]).columns if not c.startswith("_")
    ]
    if cols_cat:
        travail["_segment"] = np.select(
            list(masques.values()), list(masques.keys()), default="Inconnu"
        )
        for col in cols_cat[:3]:
            taux = (
                travail.groupby(["_segment", col])
                .size()
                .unstack(fill_value=0)
                .apply(lambda r: (r / r.sum()).round(3) if r.sum() > 0 else r, axis=1)
            )
            logger.debug("Distribution '{}' par segment d'erreur :\n{}", col, taux.to_string())

    return résumé
