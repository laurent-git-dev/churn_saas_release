"""Analyse exploratoire des données — module churn_saas.

Toutes les figures sont créées via ``churn_saas.viz.figure()`` et sauvegardées
automatiquement dans ``reports/figures/``.
"""

from __future__ import annotations

import math
from typing import Any

import matplotlib.ticker as mticker
import pandas as pd
from loguru import logger
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from scipy import stats as scipy_stats
from sklearn.metrics import roc_auc_score

from churn_saas import viz

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
        f"Prévalence du churn (n = {n_total:,})",
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
            f"{eff:,}\n({eff / n_total:.1%})",
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
                f"{n_total:,}",
                f"{n_churn:,}",
                f"{n_non_churn:,}",
                f"{prevalence:.1%}",
                f"1:{ratio:.1f}",
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


def univarie_numeriques(df: pd.DataFrame, colonnes: list[str]) -> tuple[list[Figure], pd.DataFrame]:
    """Histogramme, statistiques descriptives, asymétrie et outliers (règle IQR).

    Returns
    -------
    tuple[list[Figure], pd.DataFrame]
        Une figure par colonne + tableau récapitulatif des statistiques.
    """
    presentes = [c for c in colonnes if c in df.columns]
    if not presentes:
        raise ValueError("Aucune des colonnes spécifiées n'est présente dans le DataFrame.")

    figs: list[Figure] = []
    lignes: list[dict[str, Any]] = []

    for col in presentes:
        serie = pd.to_numeric(df[col], errors="coerce").dropna()
        n = len(serie)

        q1 = float(serie.quantile(0.25))
        q3 = float(serie.quantile(0.75))
        iqr = q3 - q1
        n_outliers = int(((serie < q1 - 1.5 * iqr) | (serie > q3 + 1.5 * iqr)).sum())
        skewness = float(serie.skew())
        kurt = float(serie.kurtosis())

        lignes.append(
            {
                "colonne": col,
                "n": n,
                "n_manquants": len(df[col]) - n,
                "moyenne": float(serie.mean()),
                "médiane": float(serie.median()),
                "écart_type": float(serie.std()),
                "min": float(serie.min()),
                "max": float(serie.max()),
                "Q1": q1,
                "Q3": q3,
                "asymétrie": skewness,
                "kurtosis": kurt,
                "n_outliers_IQR": n_outliers,
            }
        )

        fig, ax = viz.figure(f"univarie_{col}", f"Distribution — {col}", taille=(10.0, 4.5))
        n_bins = min(50, max(10, int(math.sqrt(n))))
        ax.hist(
            serie, bins=n_bins, color=viz.couleur(0), alpha=0.80, edgecolor="white", linewidth=0.4
        )
        ax.axvline(
            serie.mean(),
            color=viz.COULEUR_CHURN,
            linestyle="--",
            linewidth=1.8,
            label=f"Moyenne : {serie.mean():.2f}",
        )
        ax.axvline(
            serie.median(),
            color=viz.couleur(2),
            linestyle="-.",
            linewidth=1.8,
            label=f"Médiane : {serie.median():.2f}",
        )
        ax.set_xlabel(col)
        ax.set_ylabel("Effectif")
        ax.legend(fontsize=9)
        annotation = (
            f"Asymétrie = {skewness:.2f}  |  Kurtosis = {kurt:.2f}"
            f"  |  Outliers IQR : {n_outliers} ({n_outliers / n:.1%})"
        )
        ax.text(
            0.98,
            0.95,
            annotation,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "#F0F0F0",
                "edgecolor": "#BBBBBB",
                "alpha": 0.9,
            },
        )
        viz.sauvegarder(fig)
        figs.append(fig)
        logger.debug(
            "Univarié num. {}: n={}, asymétrie={:.2f}, outliers={}", col, n, skewness, n_outliers
        )

    tableau = pd.DataFrame(lignes).set_index("colonne")
    return figs, tableau


def univarie_categorielles(
    df: pd.DataFrame, colonnes: list[str], seuil_rare: int = 10
) -> tuple[list[Figure], pd.DataFrame]:
    """Fréquences, cardinalité, modalités rares et incohérences de casse.

    Parameters
    ----------
    seuil_rare :
        Modalités avec un effectif inférieur à ce seuil sont signalées en rouge.

    Returns
    -------
    tuple[list[Figure], pd.DataFrame]
        Une figure par colonne + tableau récapitulatif.
    """
    presentes = [c for c in colonnes if c in df.columns]
    if not presentes:
        raise ValueError("Aucune des colonnes spécifiées n'est présente dans le DataFrame.")

    figs: list[Figure] = []
    lignes: list[dict[str, Any]] = []

    for col in presentes:
        serie = df[col].dropna().astype(str)
        n = len(serie)
        cardinalite = int(serie.nunique())
        freq = serie.value_counts()
        n_rares = int((freq < seuil_rare).sum())
        cardinalite_norm = int(serie.str.strip().str.lower().nunique())
        incoherences = cardinalite - cardinalite_norm

        lignes.append(
            {
                "colonne": col,
                "n": n,
                "n_manquants": len(df[col]) - n,
                "cardinalité": cardinalite,
                "cardinalité_normalisée": cardinalite_norm,
                "incohérences_casse": incoherences,
                "n_modalités_rares": n_rares,
                "seuil_rare": seuil_rare,
                "mode": str(freq.index[0]) if len(freq) > 0 else "",
                "fréquence_mode_pct": float(freq.iloc[0] / n * 100) if len(freq) > 0 else 0.0,
            }
        )

        top = freq.head(15)
        couleurs_b = [viz.COULEUR_CHURN if v < seuil_rare else viz.couleur(0) for v in top.values]
        hauteur = float(max(4.0, min(0.5 * len(top) + 1.5, 9.0)))
        fig, ax = viz.figure(f"categorielles_{col}", f"Fréquences — {col}", taille=(10.0, hauteur))
        ax.barh(
            range(len(top)),
            top.values[::-1],
            color=couleurs_b[::-1],
            edgecolor="white",
            linewidth=0.5,
        )
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top.index[::-1].tolist(), fontsize=9)
        ax.set_xlabel("Effectif")

        for i, v in enumerate(top.values[::-1]):
            ax.text(v + max(n * 0.003, 1), i, f"{v:,} ({v/n:.1%})", va="center", fontsize=8.5)

        annotation = (
            f"Cardinalité : {cardinalite}  |  Incohérences casse : {incoherences}"
            f"  |  Rares (< {seuil_rare}) : {n_rares}"
        )
        ax.text(
            0.98,
            0.02,
            annotation,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.5,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "#F0F0F0",
                "edgecolor": "#BBBBBB",
                "alpha": 0.9,
            },
        )
        viz.sauvegarder(fig)
        figs.append(fig)
        logger.debug(
            "Univarié cat. {}: cardinalité={}, incohérences={}, rares={}",
            col,
            cardinalite,
            incoherences,
            n_rares,
        )

    tableau = pd.DataFrame(lignes).set_index("colonne")
    return figs, tableau


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
        label=f"Taux global : {taux_global:.1%}",
    )
    ax.set_yticks(range(n_mod))
    ax.set_yticklabels(
        [f"{lab}  (n={eff:,})" for lab, eff in zip(labels[::-1], effectifs[::-1], strict=True)],
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
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=couleur_txt)

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
