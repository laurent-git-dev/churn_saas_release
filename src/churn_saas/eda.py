"""Analyse exploratoire des données — module churn_saas.

Toutes les figures sont créées via ``churn_saas.viz.figure()`` et sauvegardées
automatiquement dans ``reports/figures/``.
"""

from __future__ import annotations

import math
from typing import Any

import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib.artist import Artist
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats as scipy_stats
from sklearn.metrics import roc_auc_score

from churn_saas import config, viz
from churn_saas.format_fr import entier, nombre, pourcentage
from churn_saas.models.train import ALPHA_SIGNIFICATIVITE, correction_holm

# ---------------------------------------------------------------------------
# Helpers privés
# ---------------------------------------------------------------------------


def _wilson_ci(k: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Intervalle de confiance de Wilson pour une proportion p = k/n."""
    if n == 0:
        return 0.0, 0.0
    z = float(scipy_stats.norm.ppf(1.0 - alpha / 2.0))
    p_hat = k / n
    denom = 1.0 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    marge = z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - marge), min(1.0, centre + marge)


def _cramers_v(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """V de Cramér et p-value du test chi2 d'indépendance."""
    ct = pd.crosstab(x, y)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return 0.0, 1.0
    result = scipy_stats.chi2_contingency(ct)
    chi2_stat = float(result.statistic)
    p = float(result.pvalue)
    n = int(ct.to_numpy().sum())
    v = math.sqrt(max(0.0, chi2_stat / (n * (min(ct.shape) - 1))))
    return v, p


def _est_numerique(serie: pd.Series, seuil: float = 0.5) -> bool:
    """Vrai si ≥ seuil % des valeurs non-nulles sont convertibles en float."""
    if pd.api.types.is_numeric_dtype(serie):
        return True
    n_non_nul = int(serie.notna().sum())
    if n_non_nul == 0:
        return False
    n_num = int(pd.to_numeric(serie, errors="coerce").notna().sum())
    return (n_num / n_non_nul) > seuil


# ---------------------------------------------------------------------------
# Fonctions publiques
# ---------------------------------------------------------------------------


def distribution_cible(df: pd.DataFrame, cible: str = "churn") -> tuple[Figure, pd.DataFrame]:
    """Prévalence du churn, déséquilibre de classes et métrique recommandée.

    Returns
    -------
    tuple[Figure, pd.DataFrame]
        Figure en barres + tableau récapitulatif des indicateurs clés.
    """
    if cible not in df.columns:
        raise ValueError(f"Colonne cible '{cible}' absente du DataFrame.")

    serie = df[cible].dropna()
    n_total = len(serie)
    n_churn = int(serie.sum())
    n_non_churn = n_total - n_churn
    prevalence = n_churn / n_total
    ratio = n_non_churn / max(n_churn, 1)

    logger.info(
        "distribution_cible : {} positifs / {} total ({:.1%}), ratio déséquilibre 1:{:.1f}",
        n_churn,
        n_total,
        prevalence,
        ratio,
    )

    fig, ax = viz.figure(
        "distribution_cible",
        f"Prévalence du churn (n = {entier(n_total)})",
        taille=(8.0, 5.5),
    )

    barres = ax.bar(
        ["Non-churn (0)", "Churn (1)"],
        [n_non_churn, n_churn],
        color=[viz.COULEUR_NON_CHURN, viz.COULEUR_CHURN],
        width=0.5,
        edgecolor="white",
        linewidth=1.2,
    )
    for barre, eff in zip(barres, [n_non_churn, n_churn], strict=True):
        ax.text(
            barre.get_x() + barre.get_width() / 2,
            barre.get_height() + n_total * 0.01,
            f"{entier(eff)}\n({pourcentage(eff / n_total, 1)})",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_ylabel("Nombre de clients")
    ax.set_xlabel("Classe cible")
    ax.set_ylim(0, max(n_non_churn, n_churn) * 1.18)
    viz.sauvegarder(fig)

    tableau = pd.DataFrame(
        {
            "valeur": [
                f"{entier(n_total)}",
                f"{entier(n_churn)}",
                f"{entier(n_non_churn)}",
                f"{pourcentage(prevalence, 1)}",
                f"1:{nombre(ratio, 1)}",
                "PR-AUC — insensible au déséquilibre de classes",
            ]
        },
        index=pd.Index(
            [
                "Effectif total",
                "Churners (classe 1)",
                "Non-churners (classe 0)",
                "Prévalence churn",
                "Ratio déséquilibre (0:1)",
                "Métrique recommandée",
            ],
            name="indicateur",
        ),
    )
    return fig, tableau


def univarie_numeriques(
    df: pd.DataFrame, colonnes: list[str], cible: str | None = None, ncols: int = 3
) -> tuple[Figure, pd.DataFrame]:
    """Histogrammes, statistiques descriptives, asymétrie et outliers (règle IQR de Tukey).

    Toutes les distributions sont réunies dans **une seule** figure en grille (un panneau par
    colonne) : une seule numérotation, une seule image pour le support de soutenance.

    Le kurtosis est le kurtosis **en excès** de pandas (0 pour une loi normale).

    Parameters
    ----------
    cible :
        Si fournie (cible binaire 0/1), ajoute le taux de cible parmi les outliers IQR et parmi
        les autres lignes : un outlier nettement plus associé à la cible est un signal, pas du
        bruit à écarter.
    ncols :
        Nombre de panneaux par ligne dans la grille.

    Returns
    -------
    tuple[Figure, pd.DataFrame]
        Figure en grille + tableau récapitulatif indexé par colonne.
    """
    presentes = [c for c in colonnes if c in df.columns]
    if not presentes:
        raise ValueError("Aucune des colonnes spécifiées n'est présente dans le DataFrame.")

    y = pd.to_numeric(df[cible], errors="coerce") if cible is not None else None
    nlignes = math.ceil(len(presentes) / ncols)
    fig, axes = viz.figure_grille(
        "univarie_numeriques",
        "Distributions des variables numériques candidates",
        nlignes=nlignes,
        ncols=ncols,
        taille=(5.0 * ncols, 3.4 * nlignes),
    )
    axes_plats = list(np.atleast_1d(axes).ravel())
    lignes: list[dict[str, Any]] = []

    for ax, col in zip(axes_plats, presentes, strict=False):
        serie = pd.to_numeric(df[col], errors="coerce").dropna()
        n = len(serie)

        q1 = float(serie.quantile(0.25))
        q3 = float(serie.quantile(0.75))
        iqr = q3 - q1
        masque_outliers = (serie < q1 - 1.5 * iqr) | (serie > q3 + 1.5 * iqr)
        n_outliers = int(masque_outliers.sum())
        skewness = float(serie.skew())
        kurt = float(serie.kurtosis())

        ligne: dict[str, Any] = {
            "colonne": col,
            "n": n,
            "n_manquants": len(df[col]) - n,
            "n_distinctes": int(serie.nunique()),
            "moyenne": float(serie.mean()),
            "médiane": float(serie.median()),
            "écart_type": float(serie.std()),
            "min": float(serie.min()),
            "max": float(serie.max()),
            "Q1": q1,
            "Q3": q3,
            "asymétrie": skewness,
            "kurtosis_excès": kurt,
            "n_outliers_IQR": n_outliers,
        }
        if y is not None:
            y_col = y.loc[serie.index]
            ligne["taux_cible_outliers"] = (
                float(y_col[masque_outliers].mean()) if n_outliers else float("nan")
            )
            ligne["taux_cible_hors_outliers"] = float(y_col[~masque_outliers].mean())
        lignes.append(ligne)

        n_bins = min(50, max(10, int(math.sqrt(n))))
        ax.hist(
            serie, bins=n_bins, color=viz.couleur(0), alpha=0.80, edgecolor="white", linewidth=0.4
        )
        ax.axvline(serie.mean(), color=viz.COULEUR_CHURN, linestyle="--", linewidth=1.5)
        ax.axvline(serie.median(), color=viz.couleur(2), linestyle="-.", linewidth=1.5)
        ax.set_title(col, fontsize=10)
        ax.set_ylabel("Effectif", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.text(
            0.98,
            0.95,
            f"asym. = {nombre(skewness, 2)}\noutliers : {entier(n_outliers)} "
            f"({pourcentage(n_outliers / n, 1)})",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.5,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "#F0F0F0",
                "edgecolor": "#BBBBBB",
                "alpha": 0.9,
            },
        )
        logger.debug(
            "Univarié num. {}: n={}, asymétrie={:.2f}, outliers={}", col, n, skewness, n_outliers
        )

    for ax in axes_plats[len(presentes) :]:
        ax.set_visible(False)

    fig.legend(
        handles=[
            Line2D([], [], color=viz.COULEUR_CHURN, linestyle="--", label="Moyenne"),
            Line2D([], [], color=viz.couleur(2), linestyle="-.", label="Médiane"),
        ],
        loc="upper right",
        fontsize=9,
    )
    fig.tight_layout()
    viz.sauvegarder(fig)

    tableau = pd.DataFrame(lignes).set_index("colonne")
    return fig, tableau


def univarie_categorielles(
    df: pd.DataFrame, colonnes: list[str], seuil_rare: int = 10, ncols: int = 2
) -> tuple[Figure, pd.DataFrame]:
    """Fréquences, cardinalité, modalités rares et variantes typographiques.

    Toutes les variables sont réunies dans **une seule** figure en grille (un panneau par
    colonne, 15 modalités les plus fréquentes) : une seule numérotation, une seule image pour
    le support de soutenance. La figure montre les modalités **brutes**, pour rendre visibles
    les variantes typographiques.

    La normalisation de référence est celle du §7.3.5 : ``.str.strip().str.lower()``. Une
    variante typographique est une écriture brute en trop pour une même modalité normalisée
    (« PME », « pme », « ␣PME␣ » → 2 variantes en trop). Le mode et sa fréquence sont calculés
    **après** normalisation : sur les valeurs brutes, ils sous-estimeraient la concentration
    réelle, chaque modalité étant éclatée en plusieurs écritures.

    Parameters
    ----------
    seuil_rare :
        Modalités avec un effectif inférieur à ce seuil sont signalées en rouge.
    ncols :
        Nombre de panneaux par ligne dans la grille.

    Returns
    -------
    tuple[Figure, pd.DataFrame]
        Figure en grille + tableau récapitulatif indexé par colonne. ``exemple_variantes``
        liste les écritures brutes de la modalité qui en a le plus (espaces rendus par « ␣ »).
    """
    presentes = [c for c in colonnes if c in df.columns]
    if not presentes:
        raise ValueError("Aucune des colonnes spécifiées n'est présente dans le DataFrame.")

    nlignes = math.ceil(len(presentes) / ncols)
    fig, axes = viz.figure_grille(
        "univarie_categorielles",
        "Fréquences des variables catégorielles (15 modalités les plus fréquentes)",
        nlignes=nlignes,
        ncols=ncols,
        taille=(8.0 * ncols, 4.8 * nlignes),
    )
    axes_plats = list(np.atleast_1d(axes).ravel())
    lignes: list[dict[str, Any]] = []

    for ax, col in zip(axes_plats, presentes, strict=False):
        serie = df[col].dropna().astype(str)
        n = len(serie)
        cardinalite = int(serie.nunique())
        freq = serie.value_counts()
        n_rares = int((freq < seuil_rare).sum())
        serie_norm = serie.str.strip().str.lower()
        freq_norm = serie_norm.value_counts()
        cardinalite_norm = len(freq_norm)
        variantes = cardinalite - cardinalite_norm

        # Écritures brutes de la modalité normalisée la plus éclatée
        ecritures = serie.groupby(serie_norm).unique()
        exemple = ""
        if variantes > 0:
            plus_eclatee = ecritures.map(len).idxmax()
            exemple = " / ".join(
                f"«{v.replace(' ', '␣')}»" for v in sorted(ecritures[plus_eclatee])
            )

        lignes.append(
            {
                "colonne": col,
                "n": n,
                "n_manquants": len(df[col]) - n,
                "cardinalité": cardinalite,
                "cardinalité_normalisée": cardinalite_norm,
                "variantes_typographiques": variantes,
                "exemple_variantes": exemple,
                "n_modalités_rares": n_rares,
                "n_rares_normalisées": int((freq_norm < seuil_rare).sum()),
                "seuil_rare": seuil_rare,
                "mode": str(freq_norm.index[0]) if n > 0 else "",
                "fréquence_mode_pct": float(freq_norm.iloc[0] / n * 100) if n > 0 else 0.0,
            }
        )

        top = freq.head(15)
        couleurs_b = [viz.COULEUR_CHURN if v < seuil_rare else viz.couleur(0) for v in top.values]
        ax.barh(
            range(len(top)),
            top.values[::-1],
            color=couleurs_b[::-1],
            edgecolor="white",
            linewidth=0.5,
        )
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top.index[::-1].tolist(), fontsize=8)
        ax.set_xlabel("Effectif", fontsize=8)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_title(col, fontsize=10)
        # Marge à droite pour les étiquettes de valeur
        ax.set_xlim(0, float(top.max()) * 1.25)

        for i, v in enumerate(top.values[::-1]):
            ax.text(
                v + max(n * 0.003, 1),
                i,
                f"{entier(v)} ({pourcentage(v / n, 1)})",
                va="center",
                fontsize=7,
            )

        ax.text(
            0.98,
            0.02,
            f"cardinalité : {cardinalite}\nvariantes typographiques : {variantes}\n"
            f"rares (< {seuil_rare}) : {n_rares}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.5,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "#F0F0F0",
                "edgecolor": "#BBBBBB",
                "alpha": 0.9,
            },
        )
        logger.debug(
            "Univarié cat. {}: cardinalité={}, variantes={}, rares={}",
            col,
            cardinalite,
            variantes,
            n_rares,
        )

    for ax in axes_plats[len(presentes) :]:
        ax.set_visible(False)

    fig.legend(
        handles=[
            Patch(color=viz.couleur(0), label="Modalité"),
            Patch(color=viz.COULEUR_CHURN, label=f"Modalité rare (< {seuil_rare})"),
        ],
        loc="upper right",
        fontsize=9,
    )
    fig.tight_layout()
    viz.sauvegarder(fig)

    tableau = pd.DataFrame(lignes).set_index("colonne")
    return fig, tableau


def repartition_modalites(df: pd.DataFrame, colonnes: list[str]) -> pd.DataFrame:
    """Écart de la répartition des modalités à une répartition uniforme.

    Une variable affectée au hasard (groupe d'A/B test, datacenter, thème d'interface…) doit
    avoir des modalités d'effectifs voisins. Le test du χ² d'adéquation à l'uniforme le
    vérifie : une p-valeur élevée signifie que l'écart observé est compatible avec le hasard.
    Ce test ne dit **rien** du lien avec la cible : il décrit seulement la variable.

    Les modalités sont normalisées comme au §7.3.5 (``.str.strip().str.lower()``).

    Returns
    -------
    pd.DataFrame
        Indexé par colonne : nombre de modalités, part attendue sous l'uniforme, parts
        minimale et maximale observées, statistique du χ² et p-valeur.
    """
    presentes = [c for c in colonnes if c in df.columns]
    if not presentes:
        raise ValueError("Aucune des colonnes spécifiées n'est présente dans le DataFrame.")

    lignes: list[dict[str, Any]] = []
    for col in presentes:
        freq = df[col].dropna().astype(str).str.strip().str.lower().value_counts()
        n = int(freq.sum())
        k = len(freq)
        # Test sans objet avec moins de deux modalités
        chi2, p_valeur = (
            scipy_stats.chisquare(freq.to_numpy()) if k >= 2 else (float("nan"), float("nan"))
        )
        lignes.append(
            {
                "colonne": col,
                "n_modalités": k,
                "part_uniforme": 1 / k if k else float("nan"),
                "part_min": freq.min() / n if n else float("nan"),
                "part_max": freq.max() / n if n else float("nan"),
                "chi2": float(chi2),
                "p_valeur": float(p_valeur),
            }
        )
    return pd.DataFrame(lignes).set_index("colonne")


def taux_churn_par_modalite(
    df: pd.DataFrame,
    colonne: str,
    cible: str = "churn",
    alpha: float = 0.05,
) -> tuple[Figure, pd.DataFrame]:
    """Taux de churn par modalité avec intervalle de confiance de Wilson et effectif.

    Une modalité à 100 % de churn sur 3 clients n'est pas un signal robuste :
    l'intervalle de Wilson le signale explicitement par un IC très large.

    Returns
    -------
    tuple[Figure, pd.DataFrame]
        Figure en barres horizontales + tableau trié par taux décroissant.
    """
    for col in (colonne, cible):
        if col not in df.columns:
            raise ValueError(f"Colonne '{col}' absente du DataFrame.")

    sous_df = df[[colonne, cible]].dropna()
    grp = sous_df.groupby(colonne, observed=True)[cible]
    tableau = grp.agg(effectif="count", n_churn="sum").reset_index()
    tableau["taux_churn"] = tableau["n_churn"] / tableau["effectif"]

    ics = tableau.apply(
        lambda r: _wilson_ci(float(r["n_churn"]), int(r["effectif"]), alpha),
        axis=1,
    )
    tableau["ic_bas"] = ics.apply(lambda x: x[0])
    tableau["ic_haut"] = ics.apply(lambda x: x[1])
    tableau = tableau.sort_values("taux_churn", ascending=False).reset_index(drop=True)

    taux_global = float(sous_df[cible].mean())
    n_mod = len(tableau)

    hauteur = float(max(4.5, min(0.55 * n_mod + 1.5, 10.0)))
    fig, ax = viz.figure(
        f"churn_par_{colonne}",
        f"Taux de churn par modalité — {colonne}",
        taille=(11.0, hauteur),
    )

    labels = tableau[colonne].astype(str).tolist()
    taux = tableau["taux_churn"].tolist()
    ic_bas_l = tableau["ic_bas"].tolist()
    ic_haut_l = tableau["ic_haut"].tolist()
    effectifs = tableau["effectif"].tolist()

    yerr_bas = [max(0.0, t - b) for t, b in zip(taux, ic_bas_l, strict=True)]
    yerr_haut = [max(0.0, h - t) for h, t in zip(ic_haut_l, taux, strict=True)]
    couleurs_pt = [viz.COULEUR_CHURN if t > taux_global else viz.couleur(0) for t in taux]

    ax.barh(
        range(n_mod),
        taux[::-1],
        xerr=[yerr_bas[::-1], yerr_haut[::-1]],
        color=couleurs_pt[::-1],
        alpha=0.75,
        edgecolor="white",
        error_kw={"elinewidth": 1.5, "capsize": 4, "ecolor": "#555555"},
    )
    ax.axvline(
        taux_global,
        color=viz.couleur(3),
        linestyle="--",
        linewidth=1.5,
        label=f"Taux global : {pourcentage(taux_global, 1)}",
    )
    ax.set_yticks(range(n_mod))
    ax.set_yticklabels(
        [
            f"{lab}  (n={entier(eff)})"
            for lab, eff in zip(labels[::-1], effectifs[::-1], strict=True)
        ],
        fontsize=9,
    )
    ax.set_xlabel("Taux de churn")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(fontsize=9)
    viz.sauvegarder(fig)

    tableau_out = tableau.set_index(colonne)[
        ["effectif", "n_churn", "taux_churn", "ic_bas", "ic_haut"]
    ]
    return fig, tableau_out


def association_categorielles(
    df: pd.DataFrame,
    colonnes: list[str],
    cible: str = "churn",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Association de chaque variable catégorielle avec la cible : V de Cramér et test du chi2.

    Contrairement à une ROC-AUC calculée sur le taux de churn de chaque modalité, le test du chi2
    a une référence exacte sous l'hypothèse d'indépendance, quel que soit le nombre de
    modalités. Les p-values sont corrigées par Benjamini-Hochberg sur l'ensemble des colonnes
    testées (on teste plusieurs variables à la fois).

    Returns
    -------
    pd.DataFrame
        Index = colonne ; colonnes ``n_modalites``, ``v_cramer``, ``p_value_brute``,
        ``p_value_BH``, ``significatif_BH`` ; trié par V de Cramér décroissant.
    """
    for col in (*colonnes, cible):
        if col not in df.columns:
            raise ValueError(f"Colonne '{col}' absente du DataFrame.")

    lignes: list[dict[str, Any]] = []
    for col in colonnes:
        sous_df = df[[col, cible]].dropna()
        v, p = _cramers_v(sous_df[col].astype(str), sous_df[cible])
        lignes.append(
            {
                "colonne": col,
                "n_modalites": int(sous_df[col].nunique()),
                "v_cramer": v,
                "p_value_brute": p,
            }
        )

    tableau = pd.DataFrame(lignes).set_index("colonne")
    tableau["p_value_BH"] = scipy_stats.false_discovery_control(
        tableau["p_value_brute"].to_numpy(), method="bh"
    )
    tableau["significatif_BH"] = tableau["p_value_BH"] < alpha
    return tableau.sort_values("v_cramer", ascending=False)


def pouvoir_discriminant(df: pd.DataFrame, cible: str = "churn") -> tuple[Figure, pd.DataFrame]:
    """Mesure d'association univariée de chaque variable avec la cible.

    - Variables numériques : AUC de Mann-Whitney (0,5 = aléatoire, 1,0 = parfait).
    - Variables catégorielles : V de Cramér (0 = indépendance, 1 = association parfaite).
    - p-values corrigées par la procédure Benjamini-Hochberg (FDR α = 5 %).

    Parameters
    ----------
    cible :
        Colonne binaire (0/1) à prédire.

    Returns
    -------
    tuple[Figure, pd.DataFrame]
        Barres horizontales triées + tableau avec p-values brutes et corrigées.
    """
    if cible not in df.columns:
        raise ValueError(f"Colonne cible '{cible}' absente du DataFrame.")

    y = df[cible].dropna()
    lignes: list[dict[str, Any]] = []

    for col in df.columns:
        if col == cible:
            continue

        serie = df[col]
        idx_valides = y.index.intersection(serie.dropna().index)
        if len(idx_valides) < 20:
            logger.debug("Colonne {} : trop peu de valeurs ({}) — ignorée.", col, len(idx_valides))
            continue

        if _est_numerique(serie.loc[idx_valides]):
            num = pd.to_numeric(serie.loc[idx_valides], errors="coerce").dropna()
            idx_num = y.index.intersection(num.index)
            y_num = y.loc[idx_num].astype(int)
            if len(y_num) < 20 or y_num.nunique() < 2:
                continue
            try:
                auc = float(roc_auc_score(y_num, num.loc[idx_num]))
            except Exception:
                continue
            auc = max(auc, 1.0 - auc)

            g0 = num.loc[idx_num][y_num == 0].dropna()
            g1 = num.loc[idx_num][y_num == 1].dropna()
            if len(g0) > 0 and len(g1) > 0:
                mw = scipy_stats.mannwhitneyu(g0, g1, alternative="two-sided")
                p = float(mw.pvalue)
            else:
                p = 1.0
            lignes.append(
                {"variable": col, "type": "numérique", "association": auc, "p_value_brute": p}
            )
        else:
            v, p = _cramers_v(serie.loc[idx_valides].astype(str), y.loc[idx_valides])
            lignes.append(
                {"variable": col, "type": "catégorielle", "association": v, "p_value_brute": p}
            )

    if not lignes:
        raise ValueError("Aucune variable valide pour le calcul du pouvoir discriminant.")

    tableau = pd.DataFrame(lignes)
    pvals = tableau["p_value_brute"].to_numpy()
    pvals_corr = scipy_stats.false_discovery_control(pvals, method="bh")
    tableau["p_value_BH"] = pvals_corr
    tableau["significatif_BH"] = pvals_corr < 0.05
    tableau = tableau.sort_values("association", ascending=False).reset_index(drop=True)

    n_vars = len(tableau)
    hauteur = float(max(5.0, min(0.4 * n_vars + 2.0, 16.0)))
    fig, ax = viz.figure(
        "pouvoir_discriminant",
        "Pouvoir discriminant univarié par variable",
        taille=(12.0, hauteur),
    )

    couleurs_b = []
    for _, row in tableau.iterrows():
        if not bool(row["significatif_BH"]):
            couleurs_b.append("#BBBBBB")
        elif row["type"] == "numérique":
            couleurs_b.append(viz.couleur(0))
        else:
            couleurs_b.append(viz.couleur(4))

    labels_y = [
        f"{r['variable']}  ({'N' if r['type'] == 'numérique' else 'C'})"
        for _, r in tableau.iterrows()
    ]
    ax.barh(
        range(n_vars),
        tableau["association"].tolist(),
        color=couleurs_b,
        edgecolor="white",
        linewidth=0.5,
        alpha=0.85,
    )
    ax.set_yticks(range(n_vars))
    ax.set_yticklabels(labels_y, fontsize=9)
    ax.set_xlabel("AUC univariée (N) / V de Cramér (C)")

    legende = [
        Patch(facecolor=viz.couleur(0), label="Numérique — significatif (BH)"),
        Patch(facecolor=viz.couleur(4), label="Catégorielle — significatif (BH)"),
        Patch(facecolor="#BBBBBB", label="Non significatif (p_BH > 0,05)"),
    ]
    ax.legend(handles=legende, fontsize=9, loc="lower right")
    ax.axvline(0.5, color=viz.couleur(3), linestyle=":", linewidth=1.2, alpha=0.7)
    viz.sauvegarder(fig)

    tableau_out = tableau.set_index("variable")[
        ["type", "association", "p_value_brute", "p_value_BH", "significatif_BH"]
    ]
    return fig, tableau_out


def matrice_correlation(
    df: pd.DataFrame, seuil_redondance: float = 0.85
) -> tuple[Figure, pd.DataFrame]:
    """Matrice de corrélation de Pearson et paires de variables redondantes.

    Returns
    -------
    tuple[Figure, pd.DataFrame]
        Heatmap + tableau des paires avec |r| ≥ seuil_redondance.
    """
    numeriques = df.select_dtypes(include="number")
    if numeriques.empty:
        raise ValueError("Aucune colonne numérique dans le DataFrame.")

    corr = numeriques.corr(method="pearson")
    n_vars = len(corr)
    taille = float(max(8.0, min(n_vars * 0.75, 18.0)))

    fig, ax = viz.figure(
        "matrice_correlation",
        "Matrice de corrélation de Pearson",
        taille=(taille, taille * 0.85),
    )

    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", norm=norm, aspect="auto")
    fig.colorbar(im, ax=ax, shrink=0.75, label="r de Pearson")

    ax.set_xticks(range(n_vars))
    ax.set_yticks(range(n_vars))
    ax.set_xticklabels(corr.columns.tolist(), rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(corr.index.tolist(), fontsize=9)

    if n_vars <= 20:
        for i in range(n_vars):
            for j in range(n_vars):
                val = float(corr.iloc[i, j])
                couleur_txt = "white" if abs(val) > 0.65 else "black"
                ax.text(
                    j,
                    i,
                    f"{nombre(val, 2)}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=couleur_txt,
                )

    viz.sauvegarder(fig)

    cols = corr.columns.tolist()
    paires = [
        {
            "variable_1": cols[i],
            "variable_2": cols[j],
            "corrélation": round(float(corr.iloc[i, j]), 4),
        }
        for i in range(len(cols))
        for j in range(i + 1, len(cols))
        if abs(float(corr.iloc[i, j])) >= seuil_redondance
    ]

    tableau = (
        pd.DataFrame(paires)
        if paires
        else pd.DataFrame(columns=["variable_1", "variable_2", "corrélation"])
    )
    logger.info("{} paire(s) redondante(s) détectée(s) (|r| ≥ {}).", len(paires), seuil_redondance)
    return fig, tableau


# ---------------------------------------------------------------------------
# Test des hypothèses métier
# ---------------------------------------------------------------------------

_COULEURS_VERDICT: dict[str, str] = {
    "confirmée": viz.couleur(2),
    "infirmée": viz.COULEUR_CHURN,
    "non concluante": "#888888",
}


def _sens(auc: float) -> str:
    """Sens de l'association variable → churn déduit de l'AUC non orientée."""
    if auc > 0.5:
        return "+"
    if auc < 0.5:
        return "-"
    return "="


def tester_hypotheses(
    df: pd.DataFrame,
    hypotheses: list[tuple[str, str, str, str]] = config.HYPOTHESES_METIER,
    cible: str = "churn",
    haut_distribution: frozenset[str] = config.HYPOTHESES_HAUT_DISTRIBUTION,
    alpha: float = ALPHA_SIGNIFICATIVITE,
) -> tuple[Figure, pd.DataFrame]:
    """Confronte chaque hypothèse métier aux données : sens observé, significativité, effet.

    Pour chaque hypothèse ``(id, variable, sens_attendu, libellé)`` :

    - médianes de la variable chez les churners et les non-churners ;
    - sens observé, déduit de l'AUC univariée (« + » si les churners ont des valeurs plus
      élevées, « - » sinon) — plus robuste que l'écart des médianes, souvent nul sur les
      variables de comptage ;
    - test de Mann-Whitney bilatéral, p-values corrigées par Holm sur l'ensemble des
      hypothèses testées (risque global ≤ ``alpha``) ;
    - AUC univariée orientée dans le sens attendu (> 0,5 : le sens attendu est observé) ;
    - verdict : « confirmée » (significative, sens attendu), « infirmée » (significative, sens
      contraire) ou « non concluante » (non significative après correction).

    Les hypothèses de ``haut_distribution`` sont testées sur l'indicateur
    ``variable ≥ quantile config.QUANTILE_HAUT_DISTRIBUTION`` : les colonnes ``mediane_*``
    contiennent alors la part des comptes au-dessus du quantile (``mesure`` le précise).
    Une hypothèse dont la variable est absente de ``df`` est ignorée (journalisée).

    Parameters
    ----------
    df : pd.DataFrame
        Données contenant la cible binaire (0/1) et les variables des hypothèses.
    hypotheses : list[tuple[str, str, str, str]]
        Hypothèses ``(id, variable, sens_attendu ∈ {+, -}, libellé)``.
    cible : str
        Colonne cible binaire.
    haut_distribution : frozenset[str]
        Identifiants des hypothèses portant sur le haut de la distribution.
    alpha : float
        Risque global d'erreur de première espèce.

    Returns
    -------
    tuple[Figure, pd.DataFrame]
        Petits multiples (boîtes churn vs non-churn, une par hypothèse) et tableau indexé par
        identifiant d'hypothèse.
    """
    if cible not in df.columns:
        raise ValueError(f"Colonne cible '{cible}' absente du DataFrame.")

    retenues = [h for h in hypotheses if h[1] in df.columns]
    for hid, variable, _, _ in hypotheses:
        if variable not in df.columns:
            logger.warning("Hypothèse {} ignorée : variable '{}' absente.", hid, variable)
    if not retenues:
        raise ValueError("Aucune hypothèse testable : aucune variable présente dans df.")

    lignes: list[dict[str, Any]] = []
    series: dict[str, tuple[pd.Series, pd.Series, float | None]] = {}
    for hid, variable, sens_attendu, libelle in retenues:
        donnees = df[[variable, cible]].dropna()
        brute = pd.to_numeric(donnees[variable], errors="coerce")
        y = donnees[cible].astype(int)
        seuil: float | None = None
        if hid in haut_distribution:
            seuil = float(brute.quantile(config.QUANTILE_HAUT_DISTRIBUTION))
            x = (brute >= seuil).astype(float)
            mesure = "part ≥ quantile"
            centre = x.groupby(y).mean()
        else:
            x = brute
            mesure = "médiane"
            centre = x.groupby(y).median()
        series[hid] = (brute, y, seuil)

        auc = float(roc_auc_score(y, x))
        p = float(scipy_stats.mannwhitneyu(x[y == 1], x[y == 0], alternative="two-sided").pvalue)
        lignes.append(
            {
                "id": hid,
                "variable": variable,
                "libelle": libelle,
                "sens_attendu": sens_attendu,
                "mesure": mesure,
                "mediane_churn": float(centre.get(1, np.nan)),
                "mediane_non_churn": float(centre.get(0, np.nan)),
                "sens_observe": _sens(auc),
                "p_value_brute": p,
                "auc_orientee": auc if sens_attendu == "+" else 1.0 - auc,
            }
        )

    tableau = pd.DataFrame(lignes).set_index("id")
    tableau["p_value_holm"] = correction_holm(tableau["p_value_brute"].tolist())
    significatif = tableau["p_value_holm"] < alpha
    tableau["verdict"] = np.where(
        ~significatif,
        "non concluante",
        np.where(tableau["sens_observe"] == tableau["sens_attendu"], "confirmée", "infirmée"),
    )
    logger.info(
        "Hypothèses testées : {} — {}.",
        len(tableau),
        ", ".join(f"{v} : {n}" for v, n in tableau["verdict"].value_counts().items()),
    )

    # --- Petits multiples : une boîte churn vs non-churn par hypothèse
    ncols = min(4, len(tableau))
    nlignes = math.ceil(len(tableau) / ncols)
    fig, axes = viz.figure_grille(
        "test_hypotheses",
        "Hypothèses métier : distribution churn vs non-churn",
        nlignes=nlignes,
        ncols=ncols,
        taille=(3.6 * ncols, 3.4 * nlignes),
    )
    liste_axes = list(np.atleast_1d(axes).ravel())
    for ax, (hid, ligne) in zip(liste_axes, tableau.iterrows(), strict=False):
        brute, y, seuil = series[str(hid)]
        boites = ax.boxplot(
            [brute[y == 0].to_numpy(), brute[y == 1].to_numpy()],
            patch_artist=True,
            showfliers=False,
            widths=0.6,
        )
        for boite, teinte in zip(
            boites["boxes"], [viz.COULEUR_NON_CHURN, viz.COULEUR_CHURN], strict=True
        ):
            boite.set_facecolor(teinte)
            boite.set_alpha(0.7)
        if seuil is not None:
            ax.axhline(seuil, color=viz.couleur(3), linestyle="--", linewidth=1.2)
        ax.set_xticks([1, 2], ["Non-churn", "Churn"])
        ax.set_title(
            f"{hid} · {ligne['variable']} ({ligne['sens_attendu']})\n"
            f"{ligne['verdict']} — AUC orientée {nombre(ligne['auc_orientee'])}",
            fontsize=10,
            color=_COULEURS_VERDICT[str(ligne["verdict"])],
        )
    for ax in liste_axes[len(tableau) :]:
        ax.set_visible(False)

    legende: list[Artist] = [
        Patch(facecolor=viz.COULEUR_NON_CHURN, alpha=0.7, label="Non-churn"),
        Patch(facecolor=viz.COULEUR_CHURN, alpha=0.7, label="Churn"),
        *[Patch(facecolor=c, label=f"Titre : {v}") for v, c in _COULEURS_VERDICT.items()],
    ]
    if any(s[2] is not None for s in series.values()):
        legende.append(
            Line2D(
                [],
                [],
                color=viz.couleur(3),
                linestyle="--",
                label=f"Quantile {nombre(config.QUANTILE_HAUT_DISTRIBUTION)} (haut de distribution)",
            )
        )
    # Légende sous la grille : la marge basse réservée évite qu'elle chevauche les axes
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    fig.legend(handles=legende, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0), fontsize=9)
    viz.sauvegarder(fig)

    colonnes = [
        "variable",
        "libelle",
        "sens_attendu",
        "mesure",
        "mediane_churn",
        "mediane_non_churn",
        "sens_observe",
        "p_value_brute",
        "p_value_holm",
        "auc_orientee",
        "verdict",
    ]
    return fig, tableau[colonnes]
