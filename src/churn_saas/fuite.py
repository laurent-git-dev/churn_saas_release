"""Détection et caractérisation des fuites de données (data leakage).

Trois fonctions principales :
- :func:`experience_fuite`   : démonstration chiffrée de l'impact d'une colonne suspecte.
- :func:`diagnostiquer_fuite`: AUC univariée de toutes les colonnes — signale les suspects.
- :func:`caracteriser_clv`   : nature réalisée ou prospective de ``valeur_vie_client_eur``.
"""

from __future__ import annotations

import math
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
from churn_saas.format_fr import nombre

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

# Contrôle de redondance (criblage des leurres) : au-delà de ce nombre de modalités, une
# catégorielle de référence découpe l'échantillon en groupes trop petits et gonfle
# artificiellement le V de Cramér comme le rapport de corrélation η.
_MODALITES_MAX_REDONDANCE = 50
_EFFECTIF_MIN_REDONDANCE = 20


# ---------------------------------------------------------------------------
# Helpers privés
# ---------------------------------------------------------------------------


def _est_num_serie(serie: pd.Series) -> bool:
    """Vrai si ≥ 50 % des valeurs non-nulles sont numériques."""
    if pd.api.types.is_numeric_dtype(serie):
        return True
    n = int(serie.dropna().shape[0])
    if n == 0:
        return False
    return int(pd.to_numeric(serie, errors="coerce").notna().sum()) / n > 0.5


def _v_cramer(a: pd.Series, b: pd.Series) -> float:
    """V de Cramér entre deux séries catégorielles alignées (sans valeurs manquantes)."""
    ct = pd.crosstab(a.astype(str), b.astype(str))
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return 0.0
    chi2_stat = float(scipy_stats.chi2_contingency(ct).statistic)
    n = int(ct.to_numpy().sum())
    return math.sqrt(max(0.0, chi2_stat / (n * (min(ct.shape) - 1))))


def _rapport_correlation(cat: pd.Series, num: pd.Series) -> float:
    """Rapport de corrélation η entre une catégorielle et une numérique alignées.

    η² est la part de la variance de la numérique expliquée par les modalités. η vaut 0 si les
    moyennes par modalité sont identiques, 1 si la modalité détermine entièrement la valeur, et
    coïncide avec |r de Pearson| quand la catégorielle n'a que deux modalités : le même seuil de
    redondance s'applique donc aux trois mesures.
    """
    valeurs = num.astype(float)
    ss_total = float(((valeurs - valeurs.mean()) ** 2).sum())
    if ss_total == 0.0:
        return 0.0
    groupes = valeurs.groupby(cat.astype(str).to_numpy())
    ss_inter = float((groupes.count() * (groupes.mean() - valeurs.mean()) ** 2).sum())
    return math.sqrt(min(1.0, ss_inter / ss_total))


def _redondance_max(
    serie: pd.Series, est_num: bool, references: pd.DataFrame
) -> tuple[float, str, str]:
    """Redondance maximale d'une variable avec les variables de référence, tous types confondus.

    La mesure dépend du couple de types : |r de Pearson| (numérique × numérique), rapport de
    corrélation η (catégorielle × numérique, dans les deux sens), V de Cramér (catégorielle ×
    catégorielle). Les trois sont comprises entre 0 et 1.

    Returns
    -------
    (redondance maximale, variable jumelle, mesure utilisée)
    """
    meilleure: tuple[float, str, str] = (0.0, "", "")
    if est_num:
        serie = pd.to_numeric(serie, errors="coerce")
    for ref in references.columns:
        ref_s = references[ref]
        ref_num = _est_num_serie(ref_s.dropna())
        if ref_num:
            ref_s = pd.to_numeric(ref_s, errors="coerce")
        elif ref_s.nunique() > _MODALITES_MAX_REDONDANCE:
            continue
        masque = serie.notna() & ref_s.notna()
        if masque.sum() < _EFFECTIF_MIN_REDONDANCE:
            continue
        a, b = serie[masque], ref_s[masque]
        if est_num and ref_num:
            if a.nunique() < 2 or b.nunique() < 2:
                continue
            valeur, mesure = abs(float(scipy_stats.pearsonr(a, b).statistic)), "|r| Pearson"
        elif est_num:
            valeur, mesure = _rapport_correlation(b, a), "η"
        elif ref_num:
            valeur, mesure = _rapport_correlation(a, b), "η"
        else:
            valeur, mesure = _v_cramer(a, b), "V de Cramér"
        if not math.isnan(valeur) and valeur > meilleure[0]:
            meilleure = (valeur, ref, mesure)
    return meilleure


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
        ROC-AUC, PR-AUC, courbes ROC et figure pour les deux conditions.
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
        "experience_fuite({}) : ROC-AUC avec={:.4f} / sans={:.4f}  Δ={:+.4f} | "
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
            label=f"ROC-AUC = {nombre(res.auc_roc, 4)}\nPR-AUC = {nombre(res.pr_auc, 4)}",
        )
        ax.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.5, label="Aléatoire (ROC-AUC = 0,50)")
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
    exclure: set[str] | None = None,
) -> pd.DataFrame:
    """ROC-AUC univariée pour chaque colonne — signale toute valeur au-dessus du seuil.

    - Variables numériques : ROC-AUC directe sur la valeur (statistique U de Mann-Whitney
      ramenée entre 0 et 1), sur les lignes renseignées.
    - Variables catégorielles : encodage par taux de churn moyen de la modalité, puis ROC-AUC
      sur le score continu ainsi obtenu. Les taux sont calculés sur les lignes mêmes où l'AUC
      est mesurée : l'AUC catégorielle est légèrement optimiste.

    L'AUC est **symétrisée** (``max(AUC, 1 − AUC)``) : une variable qui fait baisser le churn
    est aussi discriminante qu'une variable qui le fait monter. Le sens est conservé dans la
    colonne ``sens``.

    Le dépassement du seuil est une alerte, pas une preuve : une variable légitime
    très discriminante peut aussi dépasser le seuil. Inversement, une fuite faible passe
    sous le seuil. Le raisonnement doit être vérifié manuellement (la variable est-elle
    calculée *après* l'observation de la cible ?).

    Parameters
    ----------
    df :
        DataFrame source (colonnes texte tolérées).
    cible :
        Colonne binaire cible (0/1 ou '0'/'1').
    seuil :
        AUC univariée au-dessus de laquelle une colonne est signalée.
    exclure :
        Colonnes non évaluées (texte libre, dates…). Par défaut, les colonnes non
        modélisables du module (identifiant, texte libre, dates, jour de souscription).

    Returns
    -------
    DataFrame trié par AUC décroissante avec colonnes :
    ``colonne``, ``type``, ``auc_univariee``, ``sens``, ``part_manquants``,
    ``suspecte_fuite``, ``seuil``. ``sens`` vaut « ↑ churn » quand les valeurs élevées vont
    avec plus de churn, « ↓ churn » dans le cas inverse, et « — » pour une catégorielle
    (modalités sans ordre).
    """
    if cible not in df.columns:
        raise ValueError(f"Colonne cible '{cible}' absente du DataFrame.")

    exclues = _EXCLURE_AUTO if exclure is None else exclure
    y = pd.to_numeric(df[cible], errors="coerce").dropna().astype(int)
    lignes: list[dict[str, Any]] = []

    for col in df.columns:
        if col == cible or col in exclues:
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
                auc_brute = float(roc_auc_score(y.loc[mask], num.loc[mask]))
            except Exception:
                continue
            sens = "↑ churn" if auc_brute >= 0.5 else "↓ churn"
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
                auc_brute = float(roc_auc_score(y.loc[mask], encoded))
            except Exception:
                continue
            sens = "—"
            type_var = "catégorielle"

        score = max(auc_brute, 1.0 - auc_brute)
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
                "sens": sens,
                "part_manquants": float(serie.isna().mean()),
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


# Seuils de décision de caracteriser_clv, fixés a priori :
# - une CLV réalisée ne peut pas dépasser le CA encaissé (MRR × ancienneté), sauf hausses de
#   prix passées ou extensions de sièges : on tolère 10 % de comptes au-dessus ;
# - une CLV réalisée croît proportionnellement à l'ancienneté (élasticité ≈ 1) : on exige 0,70.
#   Une élasticité < 0,50 signifie que l'ancienneté n'explique pas l'essentiel de la CLV.
_CLV_PART_MAX_REALISEE = 0.10
_CLV_PART_MIN_PROSPECTIVE = 0.50
_CLV_ELASTICITE_MIN_REALISEE = 0.70
_CLV_ELASTICITE_MAX_PROSPECTIVE = 0.50


def caracteriser_clv(df: pd.DataFrame) -> dict[str, Any]:
    """Teste si ``valeur_vie_client_eur`` est une CLV réalisée (passée) ou prospective (future).

    Une CLV *réalisée* est le cumul des revenus déjà encaissés, approché par
    ``MRR × ancienneté``. Elle a deux signatures vérifiables :

    1. elle ne dépasse (presque) jamais ``MRR × ancienneté`` ;
    2. elle croît proportionnellement à l'ancienneté : dans la régression
       ``log CLV = a + b·log MRR + c·log ancienneté``, l'élasticité ``c`` vaut ≈ 1.

    Une CLV *prospective* (``MRR × durée de vie estimée``) dépasse souvent le CA encaissé et
    dépend peu de l'ancienneté (``c`` faible, ``b`` ≈ 1). Le ratio ``CLV / MRR`` s'interprète
    alors comme une durée de vie estimée, en mois.

    Les corrélations de Pearson sont conservées à titre descriptif : elles ne tranchent pas,
    car ``r(CLV, MRR × ancienneté)`` est élevé dès que la CLV suit le MRR, quelle que soit
    sa dépendance à l'ancienneté.

    Quelle que soit la nature de la CLV, la valeur à risque du §12 ne la multiplie pas par
    P(churn) : une CLV réalisée contient de la valeur déjà encaissée, une CLV prospective
    intègre déjà une espérance de durée de vie, donc le risque de départ.

    Parameters
    ----------
    df :
        DataFrame aux colonnes numériques **déjà coercées** (``coercer_numeriques``) : un
        ``pd.to_numeric`` brut perdrait les valeurs au format non standard.

    Returns
    -------
    dict avec les corrélations, les critères de décision, les statistiques par groupe churn,
    la nature déduite (``'réalisée'``, ``'prospective'`` ou ``'indéterminée'``) et la formule
    de valeur à risque.
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

    # Critère 1 — la CLV dépasse-t-elle le CA déjà encaissé ?
    ratio_cumul = sub[_COL["clv"]] / sub["_mrr_x_anc"]
    part_sup_cumul = float((ratio_cumul > 1).mean())

    # Critère 2 — élasticités log-log (montants strictement positifs requis)
    pos = sub[(sub[[_COL["clv"], _COL["anc"], _COL["mrr"]]] > 0).all(axis=1)]
    x_log = np.column_stack([np.ones(len(pos)), np.log(pos[_COL["mrr"]]), np.log(pos[_COL["anc"]])])
    _, elast_mrr, elast_anc = np.linalg.lstsq(x_log, np.log(pos[_COL["clv"]]), rcond=None)[0]

    # Lecture prospective : CLV / MRR = durée de vie estimée, en mois. Si elle est plus courte
    # chez les churners, la CLV intègre déjà le risque de départ (fuite de la cible).
    mois_mrr = sub[_COL["clv"]] / sub[_COL["mrr"]]
    rho_mois_churn, p_mois_churn = scipy_stats.spearmanr(mois_mrr, sub[_COL["churn"]])

    stats_par_churn = (
        sub.groupby(_COL["churn"])
        .agg(
            n=(_COL["clv"], "count"),
            **{
                "CLV moyenne_€": (_COL["clv"], "mean"),
                "CLV médiane_€": (_COL["clv"], "median"),
                "MRR médian_€": (_COL["mrr"], "median"),
            },
        )
        .round(0)
    )

    if part_sup_cumul <= _CLV_PART_MAX_REALISEE and elast_anc >= _CLV_ELASTICITE_MIN_REALISEE:
        nature = "réalisée"
        avertissement = (
            "CLV réalisée : elle contient de la valeur passée déjà encaissée, "
            "qu'un départ ne peut plus faire perdre."
        )
    elif (
        part_sup_cumul >= _CLV_PART_MIN_PROSPECTIVE and elast_anc < _CLV_ELASTICITE_MAX_PROSPECTIVE
    ):
        nature = "prospective"
        avertissement = (
            "CLV prospective : elle intègre déjà une durée de vie estimée, donc le risque "
            "de départ ; la multiplier par P(churn) compterait ce risque deux fois."
        )
    else:
        nature = "indéterminée"
        avertissement = (
            "Signatures contradictoires — méthodologie de calcul non documentée ; "
            "à clarifier avec l'équipe Finance."
        )

    horizon = int(config.HYPOTHESES_ECONOMIQUES["horizon_mois"])
    formule = f"P(churn à {horizon} mois) × MRR_mensuel × {horizon} × marge_brute"

    logger.info(
        "caracteriser_clv : part CLV > MRR×anc={:.3f}, élasticité ancienneté={:.3f}, "
        "élasticité MRR={:.3f}, r(CLV,MRR×anc)={:.3f} → nature={}",
        part_sup_cumul,
        elast_anc,
        elast_mrr,
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
        "part_clv_sup_cumul": part_sup_cumul,
        "ratio_cumul_median": float(ratio_cumul.median()),
        "elasticite_mrr": float(elast_mrr),
        "elasticite_anciennete": float(elast_anc),
        "mois_mrr": mois_mrr.quantile([0.05, 0.5, 0.95]),
        "rho_mois_churn": float(rho_mois_churn),
        "p_mois_churn": float(p_mois_churn),
        "seuils": {
            "part_max_realisee": _CLV_PART_MAX_REALISEE,
            "part_min_prospective": _CLV_PART_MIN_PROSPECTIVE,
            "elasticite_min_realisee": _CLV_ELASTICITE_MIN_REALISEE,
            "elasticite_max_prospective": _CLV_ELASTICITE_MAX_PROSPECTIVE,
        },
        "stats_par_churn": stats_par_churn,
        "nature_clv": nature,
        "formule_valeur_risque": formule,
        "avertissement": avertissement,
    }


def cribler_leurres(
    df: pd.DataFrame,
    cible: str,
    colonnes: list[str],
    seuil_redondance: float = config.SEUIL_REDONDANCE,
) -> pd.DataFrame:
    """Criblage des colonnes leurres — verdict provisoire fondé sur preuves convergentes.

    Trois niveaux de preuve (point de vigilance n°5) :

    1. **Association marginale** avec la cible : V de Cramér (catégorielles) ou AUC avec test
       de Mann-Whitney (numériques), p-value corrigée Benjamini-Hochberg.
    2. **Redondance** avec **toutes** les variables de référence, quel que soit leur type :
       |r de Pearson| (numérique × numérique), rapport de corrélation η (catégorielle ×
       numérique), V de Cramér (catégorielle × catégorielle) — pour écarter l'hypothèse de la
       jumelle.
    3. **(§12) Permutation importance + drop-column importance** — colonnes réservées
       ``permutation_imp`` et ``drop_column_imp``, renseignées en section 12.

    Règle de verdict, dans le même ordre que ``models.explain.verdict_leurres`` : la redondance
    est examinée **avant** l'association, car une variable informative mais redondante est
    précisément celle dont l'importance de permutation sera faible au §12 sans qu'elle soit un
    leurre (point de vigilance n°5).

    Parameters
    ----------
    df :
        DataFrame source (colonnes numériques ou texte tolérées).
    cible :
        Colonne binaire cible (0/1).
    colonnes :
        Colonnes à cribler — en général ``config.COLONNES_LEURRES_SUSPECTES``.
    seuil_redondance :
        Seuil de redondance (|r|, η ou V de Cramér, tous entre 0 et 1) au-delà duquel une
        variable est déclarée « jumelle » d'une autre.

    Returns
    -------
    DataFrame indexé par colonne avec colonnes :
    ``type``, ``association_marginale``, ``p_value_BH``, ``significatif_BH``,
    ``max_redondance``, ``variable_jumelle``, ``mesure_redondance``, ``permutation_imp``,
    ``drop_column_imp``, ``verdict_provisoire``.
    """
    if cible not in df.columns:
        raise ValueError(f"Colonne cible '{cible}' absente du DataFrame.")

    presentes = [c for c in colonnes if c in df.columns]
    if not presentes:
        raise ValueError("Aucune des colonnes spécifiées n'est présente dans le DataFrame.")

    y = pd.to_numeric(df[cible], errors="coerce").dropna().astype(int)
    if y.nunique() < 2:
        raise ValueError("La cible doit être binaire (0/1).")

    # Variables de référence pour la redondance (hors colonnes criblées, cible, interdites)
    refs_exclues = set(presentes) | {cible} | set(config.COLONNES_INTERDITES) | _EXCLURE_AUTO
    references = df.loc[y.index, [c for c in df.columns if c not in refs_exclues]]

    lignes: list[dict[str, Any]] = []

    for col in presentes:
        serie = df.loc[y.index, col]
        idx_valides = y.index.intersection(serie.dropna().index)

        if len(idx_valides) < _EFFECTIF_MIN_REDONDANCE:
            logger.warning("cribler_leurres : {} — moins de 20 valeurs valides, ignorée.", col)
            continue

        est_num = _est_num_serie(serie.loc[idx_valides])
        if est_num:
            # --- Variable numérique ---
            num = pd.to_numeric(serie.loc[idx_valides], errors="coerce").dropna()
            y_num = y.loc[num.index]
            if y_num.nunique() < 2:
                continue
            auc = float(roc_auc_score(y_num, num))
            assoc = max(auc, 1.0 - auc)
            p = float(
                scipy_stats.mannwhitneyu(
                    num[y_num == 0], num[y_num == 1], alternative="two-sided"
                ).pvalue
            )
            type_var = "numérique"
        else:
            # --- Variable catégorielle ---
            ct = pd.crosstab(serie.loc[idx_valides].astype(str), y.loc[idx_valides])
            if ct.shape[0] < 2 or ct.shape[1] < 2:
                continue
            p = float(scipy_stats.chi2_contingency(ct).pvalue)
            assoc = _v_cramer(serie.loc[idx_valides], y.loc[idx_valides])
            type_var = "catégorielle"

        redondance_max, jumelle, mesure = _redondance_max(serie, est_num, references)

        lignes.append(
            {
                "colonne": col,
                "type": type_var,
                "association_marginale": round(assoc, 4),
                "p_value_brute": p,
                "max_redondance": round(redondance_max, 4),
                "variable_jumelle": jumelle,
                "mesure_redondance": mesure,
            }
        )

    if not lignes:
        raise ValueError("Aucune colonne valide pour le criblage des leurres.")

    tableau = pd.DataFrame(lignes)

    pvals_corr = scipy_stats.false_discovery_control(
        tableau["p_value_brute"].to_numpy(), method="bh"
    )
    tableau["p_value_BH"] = pvals_corr
    tableau["significatif_BH"] = pvals_corr < 0.05

    # Colonnes réservées pour §12
    tableau["permutation_imp"] = float("nan")
    tableau["drop_column_imp"] = float("nan")

    def _verdict(row: pd.Series) -> str:
        # Redondance d'abord : même ordre que models.explain.verdict_leurres (§12)
        if float(row["max_redondance"]) >= seuil_redondance:
            return "redondant — importance nulle ≠ leurre (point vigilance n°5)"
        if bool(row["significatif_BH"]):
            return "potentiellement utile — à confirmer en §12"
        return "leurre probable — à confirmer en §12"

    tableau["verdict_provisoire"] = tableau.apply(_verdict, axis=1)

    for _, row in tableau.iterrows():
        logger.info(
            "cribler_leurres : {} → {} | assoc={:.4f}, p_BH={:.2e}, redondance={:.4f} ({}) | {}",
            row["colonne"],
            row["type"],
            row["association_marginale"],
            row["p_value_BH"],
            row["max_redondance"],
            row["mesure_redondance"],
            row["verdict_provisoire"],
        )

    return tableau.set_index("colonne")[
        [
            "type",
            "association_marginale",
            "p_value_BH",
            "significatif_BH",
            "max_redondance",
            "variable_jumelle",
            "mesure_redondance",
            "permutation_imp",
            "drop_column_imp",
            "verdict_provisoire",
        ]
    ]
