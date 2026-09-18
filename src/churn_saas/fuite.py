"""Détection et caractérisation des fuites de données (data leakage).

Trois fonctions principales :
- :func:`experience_fuite`   : démonstration chiffrée de l'impact d'une colonne suspecte.
- :func:`diagnostiquer_fuite`: AUC univariée de toutes les colonnes — signale les suspects.
- :func:`caracteriser_clv`   : nature historique ou future de ``valeur_vie_client_eur``.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats as scipy_stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from churn_saas import config, viz

# ---------------------------------------------------------------------------
# Types de retour
# ---------------------------------------------------------------------------


class ResultatCondition(NamedTuple):
    """Métriques d'évaluation pour une condition expérimentale."""

    auc_roc: float
    pr_auc: float
    fpr: np.ndarray
    tpr: np.ndarray
    n_features: int


class ResultatExperience(NamedTuple):
    """Résultat complet de l'expérience avec / sans colonne suspecte."""

    avec: ResultatCondition
    sans: ResultatCondition
    delta_auc_roc: float
    delta_pr_auc: float
    fig_roc: Any  # matplotlib.figure.Figure


# ---------------------------------------------------------------------------
# Constantes internes
# ---------------------------------------------------------------------------

# Colonnes non modélisables directement (texte libre, identifiants, dates)
_EXCLURE_AUTO: set[str] = {
    "client_id",
    "commentaire_csm",
    "date_souscription",
    "jour_souscription",
}


# ---------------------------------------------------------------------------
# Helpers privés
# ---------------------------------------------------------------------------


def _preparer_X(df: pd.DataFrame, colonnes: list[str]) -> pd.DataFrame:
    """Convertit les colonnes en float ; ignore celles non convertibles à ≥ 5 %."""
    series = []
    for col in colonnes:
        if col not in df.columns:
            continue
        num = pd.to_numeric(df[col], errors="coerce")
        if num.notna().mean() < 0.05:
            logger.debug("_preparer_X : {} ignorée (< 5 % de valeurs numériques).", col)
            continue
        series.append(num.rename(col))
    if not series:
        raise ValueError("Aucune colonne numérique valide dans le sous-ensemble fourni.")
    return pd.concat(series, axis=1)


def _pipeline_lr() -> Pipeline:
    """Pipeline identique utilisé dans les deux conditions de l'expérience."""
    return Pipeline(
        [
            ("imputation", SimpleImputer(strategy="median")),
            ("standardisation", StandardScaler()),
            (
                "modele",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1_000,
                    random_state=config.RANDOM_SEED,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def _evaluer(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> ResultatCondition:
    y_score = pipe.predict_proba(X_test)[:, 1]
    auc_roc = float(roc_auc_score(y_test, y_score))
    pr_auc = float(average_precision_score(y_test, y_score))
    fpr, tpr, _ = roc_curve(y_test, y_score)
    return ResultatCondition(
        auc_roc=auc_roc,
        pr_auc=pr_auc,
        fpr=fpr,
        tpr=tpr,
        n_features=X_test.shape[1],
    )


# ---------------------------------------------------------------------------
# API publique
# ---------------------------------------------------------------------------


def experience_fuite(
    df: pd.DataFrame,
    cible: str,
    colonne_suspecte: str,
) -> ResultatExperience:
    """Démontre chiffres en main l'impact d'une colonne suspecte.

    Entraîne le même pipeline (régression logistique, split 80/20 stratifié, graine fixée)
    dans deux conditions : avec et sans la colonne suspecte.

    Parameters
    ----------
    df :
        DataFrame source (tolérant les colonnes texte — conversion interne).
    cible :
        Nom de la colonne cible binaire (0/1).
    colonne_suspecte :
        Colonne dont on veut mesurer la nocivité.

    Returns
    -------
    :class:`ResultatExperience`
        AUC-ROC, PR-AUC, courbes ROC et figure pour les deux conditions.
    """
    if cible not in df.columns:
        raise ValueError(f"Colonne cible '{cible}' absente du DataFrame.")
    if colonne_suspecte not in df.columns:
        raise ValueError(f"Colonne suspecte '{colonne_suspecte}' absente du DataFrame.")

    y = pd.to_numeric(df[cible], errors="coerce").dropna().astype(int)

    # Features de base : tout sauf la cible, les interdites et les exclues automatiquement
    colonnes_interdites = set(config.COLONNES_INTERDITES) | _EXCLURE_AUTO | {cible}
    features_base = [c for c in df.columns if c not in colonnes_interdites]

    # Condition "avec" : on réintègre la colonne suspecte même si elle était dans COLONNES_INTERDITES
    features_avec = (
        features_base if colonne_suspecte in features_base else features_base + [colonne_suspecte]
    )
    features_sans = [c for c in features_avec if c != colonne_suspecte]

    X_avec = _preparer_X(df.loc[y.index], features_avec)
    X_sans = _preparer_X(df.loc[y.index], features_sans)

    # Split identique pour les deux conditions (graine fixée → comparaison équitable)
    idx_train, idx_test = train_test_split(
        y.index, test_size=0.2, stratify=y, random_state=config.RANDOM_SEED
    )

    pipe_avec = _pipeline_lr()
    pipe_avec.fit(X_avec.loc[idx_train], y.loc[idx_train])
    res_avec = _evaluer(pipe_avec, X_avec.loc[idx_test], y.loc[idx_test])

    pipe_sans = _pipeline_lr()
    pipe_sans.fit(X_sans.loc[idx_train], y.loc[idx_train])
    res_sans = _evaluer(pipe_sans, X_sans.loc[idx_test], y.loc[idx_test])

    delta_auc = res_avec.auc_roc - res_sans.auc_roc
    delta_pr = res_avec.pr_auc - res_sans.pr_auc

    logger.info(
        "experience_fuite({}) : AUC avec={:.4f} / sans={:.4f}  Δ={:+.4f} | "
        "PR-AUC avec={:.4f} / sans={:.4f}  Δ={:+.4f}",
        colonne_suspecte,
        res_avec.auc_roc,
        res_sans.auc_roc,
        delta_auc,
        res_avec.pr_auc,
        res_sans.pr_auc,
        delta_pr,
    )

    # Figure : deux courbes ROC côte à côte
    fig, axes = viz.figure_grille(
        f"fuite_roc_{colonne_suspecte}",
        f"Démonstration de fuite — impact de « {colonne_suspecte} »",
        nlignes=1,
        ncols=2,
        taille=(14.0, 5.5),
    )

    specs = [
        (axes[0], res_avec, f"Avec {colonne_suspecte}", viz.COULEUR_CHURN),
        (axes[1], res_sans, f"Sans {colonne_suspecte}", viz.couleur(0)),
    ]
    for ax, res, label, couleur_roc in specs:
        ax.plot(
            res.fpr,
            res.tpr,
            color=couleur_roc,
            lw=2.0,
            label=f"AUC = {res.auc_roc:.4f}\nPR-AUC = {res.pr_auc:.4f}",
        )
        ax.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.5, label="Aléatoire (AUC = 0,50)")
        ax.set_xlabel("Taux de faux positifs")
        ax.set_ylabel("Taux de vrais positifs")
        ax.set_title(label, fontsize=12)
        ax.legend(fontsize=9, loc="lower right")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.05)

    fig.tight_layout()
    viz.sauvegarder(fig)

    return ResultatExperience(
        avec=res_avec,
        sans=res_sans,
        delta_auc_roc=delta_auc,
        delta_pr_auc=delta_pr,
        fig_roc=fig,
    )


def diagnostiquer_fuite(
    df: pd.DataFrame,
    cible: str = "churn",
    seuil: float = 0.85,
) -> pd.DataFrame:
    """AUC univariée pour chaque colonne — signale toute valeur au-dessus du seuil.

    - Variables numériques : AUC ROC directe (Mann-Whitney).
    - Variables catégorielles : encodage par taux de churn moyen de la modalité,
      puis AUC ROC sur le score continu ainsi obtenu.

    Le dépassement du seuil est une alerte, pas une preuve : une variable légitime
    très discriminante peut aussi dépasser 0,85. Le raisonnement doit être vérifié
    manuellement (est-elle calculée *après* l'observation de la cible ?).

    Parameters
    ----------
    df :
        DataFrame source (colonnes texte tolérées).
    cible :
        Colonne binaire cible (0/1 ou '0'/'1').
    seuil :
        AUC univariée au-dessus de laquelle une colonne est signalée.

    Returns
    -------
    DataFrame trié par AUC décroissante avec colonnes :
    ``colonne``, ``type``, ``auc_univariee``, ``suspecte_fuite``, ``seuil``.
    """
    if cible not in df.columns:
        raise ValueError(f"Colonne cible '{cible}' absente du DataFrame.")

    y = pd.to_numeric(df[cible], errors="coerce").dropna().astype(int)
    lignes: list[dict[str, Any]] = []

    for col in df.columns:
        if col == cible or col in _EXCLURE_AUTO:
            continue

        serie = df.loc[y.index, col]
        if serie.notna().sum() < 50:
            logger.debug("diagnostiquer_fuite : {} — moins de 50 valeurs, ignorée.", col)
            continue

        num = pd.to_numeric(serie, errors="coerce")
        pct_num = float(num.notna().mean())

        if pct_num >= 0.5:
            mask = num.notna()
            if mask.sum() < 50 or y.loc[mask].nunique() < 2:
                continue
            try:
                score = float(roc_auc_score(y.loc[mask], num.loc[mask]))
                score = max(score, 1.0 - score)
            except Exception:
                continue
            type_var = "numérique"
        else:
            mask = serie.notna()
            if mask.sum() < 50 or y.loc[mask].nunique() < 2:
                continue
            means = y.loc[mask].groupby(serie.loc[mask].astype(str)).mean()
            encoded = serie.loc[mask].astype(str).map(means)
            if encoded.isna().all():
                continue
            try:
                score = float(roc_auc_score(y.loc[mask], encoded))
                score = max(score, 1.0 - score)
            except Exception:
                continue
            type_var = "catégorielle"

        suspecte = score >= seuil
        if suspecte:
            logger.warning(
                "FUITE SUSPECTÉE : {} (AUC={:.4f}) — vérifier manuellement.",
                col,
                score,
            )

        lignes.append(
            {
                "colonne": col,
                "type": type_var,
                "auc_univariee": round(score, 4),
                "suspecte_fuite": suspecte,
                "seuil": seuil,
            }
        )

    tableau = (
        pd.DataFrame(lignes).sort_values("auc_univariee", ascending=False).reset_index(drop=True)
    )
    n_suspects = int(tableau["suspecte_fuite"].sum())
    logger.info(
        "diagnostiquer_fuite : {}/{} colonne(s) suspecte(s) (AUC ≥ {}).",
        n_suspects,
        len(tableau),
        seuil,
    )
    return tableau


def caracteriser_clv(df: pd.DataFrame) -> dict[str, Any]:
    """Teste si ``valeur_vie_client_eur`` est une CLV réalisée ou future.

    Corrélations testées :

    - CLV vs ``anciennete_mois`` : une CLV purement durée-dépendante aurait r ≈ 1.
    - CLV vs ``revenu_mensuel_recurrent_eur`` (MRR) : une CLV proportionnelle au MRR
      courant signale une valeur instantanée ou cumulée sur contrat actif.
    - CLV vs MRR × ancienneté : si r > 0,70, la CLV est vraisemblablement cumulée
      (somme des loyers passés) — donc *historique*, pas *future*.

    Cette conclusion conditionne la formule de la valeur à risque en §12 :
    une CLV historique ne doit pas être multipliée par P(churn) car elle contient
    déjà de la valeur passée irréversible.

    Returns
    -------
    dict avec les corrélations, statistiques par groupe churn, la nature déduite
    (``'historique'`` ou ``'indéterminée'``) et la formule recommandée.
    """
    _COL = {
        "clv": "valeur_vie_client_eur",
        "anc": "anciennete_mois",
        "mrr": "revenu_mensuel_recurrent_eur",
        "churn": "churn",
    }
    for col in _COL.values():
        if col not in df.columns:
            raise ValueError(f"Colonne requise '{col}' absente du DataFrame.")

    sub = df[list(_COL.values())].copy()
    for col in (_COL["clv"], _COL["anc"], _COL["mrr"], _COL["churn"]):
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
    sub = sub.dropna()
    sub[_COL["churn"]] = sub[_COL["churn"]].astype(int)

    r_anc, p_anc = scipy_stats.pearsonr(sub[_COL["clv"]], sub[_COL["anc"]])
    r_mrr, p_mrr = scipy_stats.pearsonr(sub[_COL["clv"]], sub[_COL["mrr"]])
    sub["_mrr_x_anc"] = sub[_COL["mrr"]] * sub[_COL["anc"]]
    r_calc, p_calc = scipy_stats.pearsonr(sub[_COL["clv"]], sub["_mrr_x_anc"])

    stats_par_churn = (
        sub.groupby(_COL["churn"])[_COL["clv"]]
        .agg(["count", "mean", "median"])
        .rename(columns={"count": "n", "mean": "moyenne_€", "median": "médiane_€"})
        .round(0)
    )

    if r_calc > 0.70:
        nature = "historique"
        formule = "P(churn dans les 12 prochains mois) × MRR_mensuel × 12 × marge_brute"
        avertissement = (
            "CLV historique cumulée : contient de la valeur passée déjà encaissée. "
            "Utiliser MRR × horizon × marge comme proxy de la valeur future à risque."
        )
    else:
        nature = "indéterminée"
        formule = "À préciser avec l'équipe Finance avant la section §12."
        avertissement = (
            "Faible corrélation avec MRR×ancienneté — méthodologie de calcul non documentée. "
            "Traiter comme suspecte jusqu'à clarification."
        )

    logger.info(
        "caracteriser_clv : r(CLV,anc)={:.3f}, r(CLV,MRR)={:.3f}, "
        "r(CLV,MRR×anc)={:.3f} → nature={}",
        r_anc,
        r_mrr,
        r_calc,
        nature,
    )

    return {
        "n": len(sub),
        "r_anciennete": float(r_anc),
        "p_anciennete": float(p_anc),
        "r_mrr": float(r_mrr),
        "p_mrr": float(p_mrr),
        "r_mrr_x_anciennete": float(r_calc),
        "p_mrr_x_anciennete": float(p_calc),
        "stats_par_churn": stats_par_churn,
        "nature_clv": nature,
        "formule_valeur_risque": formule,
        "avertissement": avertissement,
    }
