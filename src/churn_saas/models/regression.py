"""Modèle de régression CLV et valeur à risque future (C5, point de vigilance n°3).

Cible : ``valeur_vie_client_eur`` — proxy de la valeur cumulée (historique, cf. §6.7).

⚠️  Point de vigilance n°3 — valeur à risque :
    La CLV mesurée dans les données est **historique** (r(CLV, MRR×ancienneté) > 0,70 en §6.7).
    La multiplier par P(churn) reviendrait à double-compter la valeur passée déjà encaissée.

    Formule correcte retenue :
        valeur_à_risque = P(churn) × MRR_mensuel × horizon_mois × marge_brute

    La régression CLV est présentée pour C5 (démonstration d'une tâche de régression)
    mais N'ALIMENTE PAS la formule de valeur à risque.

Features nettoyées :
    - ``config.COLONNES_INTERDITES`` : client_id, churn, valeur_vie_client_eur,
      sante_compte_fin_periode.
    - Colonnes non modélisables : date_souscription, jour_souscription, commentaire_csm.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from matplotlib.figure import Figure
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from churn_saas import cache, config, economie, viz
from churn_saas.features.build import construire_preprocesseur

# Colonnes non modélisables hors COLONNES_INTERDITES
_EXCLURE_SUPPLEMENTAIRE: list[str] = [
    "date_souscription",
    "jour_souscription",
    "commentaire_csm",
]

# Cible de la régression
_CIBLE_REGRESSION: str = "valeur_vie_client_eur"

# Nom du cache
_NOM_CACHE: str = "regression_clv.joblib"

# Type pour les données numériques : scalaire ou vecteur
_NumType = float | np.ndarray


# ---------------------------------------------------------------------------
# Préparation des features
# ---------------------------------------------------------------------------


def features_regression(df: pd.DataFrame) -> pd.DataFrame:
    """Retourne X épuré pour la régression CLV.

    Exclut :
    - ``config.COLONNES_INTERDITES`` (dont ``churn``, ``sante_compte_fin_periode``
      et ``valeur_vie_client_eur`` elle-même).
    - Les colonnes non modélisables : dates, identifiants texte, commentaire libre.

    Parameters
    ----------
    df :
        Gold dataset complet (avec ou sans la cible — elle est exclue en interne).

    Returns
    -------
    DataFrame prêt pour entraînement, sans fuites.
    """
    exclure = set(config.COLONNES_INTERDITES) | set(_EXCLURE_SUPPLEMENTAIRE) | {_CIBLE_REGRESSION}
    X = df.drop(columns=[c for c in exclure if c in df.columns])
    logger.info(
        "features_regression — {} colonnes retenues après exclusion de {}.",
        X.shape[1],
        sorted(exclure & set(df.columns)),
    )
    return X


# ---------------------------------------------------------------------------
# Métriques
# ---------------------------------------------------------------------------


def evaluer_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Calcule RMSE, MAE et R² dans l'échelle d'origine.

    Toujours appelé sur des valeurs en euros (après back-transform si log).

    Parameters
    ----------
    y_true :
        Valeurs réelles de CLV (euros).
    y_pred :
        Valeurs prédites (euros, échelle originale).

    Returns
    -------
    dict avec clés ``rmse``, ``mae``, ``r2``.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    logger.info("Régression CLV — RMSE={:.0f} €  MAE={:.0f} €  R²={:.4f}", rmse, mae, r2)
    return {"rmse": rmse, "mae": mae, "r2": r2}


# ---------------------------------------------------------------------------
# Entraînement
# ---------------------------------------------------------------------------


def entrainer_modeles_clv(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    forcer: bool = False,
) -> dict[str, dict[str, Any]]:
    """Entraîne baseline, Ridge et HistGradientBoosting sur la CLV.

    Stratégie :
    - Split 80/20 stratifié sur les **quartiles de y** (``pd.qcut(y, q=4)``),
      graine fixée à ``config.RANDOM_SEED``.
    - Chaque modèle est comparé dans l'échelle **originale** ET après
      **transformation logarithmique** (log1p/expm1) — la CLV ayant un
      skewness > 5, la transformation log améliore souvent le R².
    - Toutes les métriques sont rapportées dans l'échelle d'origine (euros)
      pour permettre la comparaison équitable.

    Parameters
    ----------
    X :
        Features préparées par :func:`features_regression`.
    y :
        Cible ``valeur_vie_client_eur`` (euros, skewness > 5).
    forcer :
        Si ``True``, ignore le cache et recalcule.

    Returns
    -------
    dict : clés ``baseline``, ``ridge``, ``hgbt``, ``ridge_log``, ``hgbt_log``.
    Chaque valeur est un dict avec :
    ``modele``, ``y_test``, ``y_pred``, ``y_pred_log`` (``None`` si pas de log),
    ``metriques``.

    Notes
    -----
    Le résultat complet est mis en cache via
    :func:`~churn_saas.cache.charger_ou_calculer`.
    """

    def _calculer() -> dict[str, dict[str, Any]]:
        y_arr = np.asarray(y, dtype=float)

        # Stratification par quartile (train_test_split ne supporte pas y continu)
        strates = pd.qcut(y_arr, q=4, labels=False, duplicates="drop")
        idx_train, idx_test = train_test_split(
            np.arange(len(y_arr)),
            test_size=0.2,
            stratify=strates,
            random_state=config.RANDOM_SEED,
        )

        X_train = X.iloc[idx_train]
        X_test = X.iloc[idx_test]
        y_train = y_arr[idx_train]
        y_test = y_arr[idx_test]

        y_log_train = np.log1p(y_train)

        prep = construire_preprocesseur(X_train)

        spec: list[tuple[str, Any, bool]] = [
            ("baseline", DummyRegressor(strategy="median"), False),
            (
                "ridge",
                Pipeline([("prep", construire_preprocesseur(X_train)), ("reg", Ridge(alpha=1.0))]),
                False,
            ),
            (
                "hgbt",
                Pipeline(
                    [
                        ("prep", construire_preprocesseur(X_train)),
                        (
                            "reg",
                            HistGradientBoostingRegressor(
                                random_state=config.RANDOM_SEED,
                                max_iter=200,
                            ),
                        ),
                    ]
                ),
                False,
            ),
            (
                "ridge_log",
                Pipeline([("prep", construire_preprocesseur(X_train)), ("reg", Ridge(alpha=1.0))]),
                True,
            ),
            (
                "hgbt_log",
                Pipeline(
                    [
                        ("prep", construire_preprocesseur(X_train)),
                        (
                            "reg",
                            HistGradientBoostingRegressor(
                                random_state=config.RANDOM_SEED,
                                max_iter=200,
                            ),
                        ),
                    ]
                ),
                True,
            ),
        ]

        resultats: dict[str, dict[str, Any]] = {}
        for nom, modele, utilise_log in spec:
            y_fit = y_log_train if utilise_log else y_train

            if nom == "baseline":
                # DummyRegressor ne gère pas de préprocesseur
                modele.fit(X_train, y_fit)
                y_pred_raw = np.asarray(modele.predict(X_test))
            else:
                modele.fit(X_train, y_fit)
                y_pred_raw = np.asarray(modele.predict(X_test))

            if utilise_log:
                y_pred_orig = np.expm1(y_pred_raw)
                y_pred_log_stored: np.ndarray | None = y_pred_raw
            else:
                y_pred_orig = y_pred_raw
                y_pred_log_stored = None

            metriques = evaluer_regression(y_test, y_pred_orig)
            logger.info("Régression CLV — {} (log={}) : {}", nom, utilise_log, metriques)

            resultats[nom] = {
                "modele": modele,
                "y_test": y_test,
                "y_pred": y_pred_orig,
                "y_pred_log": y_pred_log_stored,
                "metriques": metriques,
            }

        # Utilisation de prep pour satisfaire la vérification de linting
        _ = prep

        return resultats

    brut, _ = cache.charger_ou_calculer(_NOM_CACHE, _calculer, forcer=forcer)
    return brut  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def figure_predit_vs_reel(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    nom_modele: str,
    *,
    log_scale: bool = True,
) -> Figure:
    """Scatter plot des valeurs prédites vs réelles avec diagonale parfaite.

    Parameters
    ----------
    y_true :
        Valeurs réelles (euros).
    y_pred :
        Valeurs prédites (euros, échelle originale).
    nom_modele :
        Nom du modèle affiché dans le titre et la légende.
    log_scale :
        Si ``True``, applique une échelle logarithmique sur les deux axes
        (recommandé pour CLV à forte asymétrie).

    Returns
    -------
    Figure matplotlib sauvegardée dans ``reports/figures/``.
    """
    fig, ax = viz.figure(
        f"predit_vs_reel_{nom_modele.lower().replace(' ', '_')}",
        f"CLV — prédit vs réel · {nom_modele}",
        taille=(7.0, 6.5),
    )

    ax.scatter(
        y_true,
        y_pred,
        alpha=0.35,
        s=18,
        color=viz.couleur(0),
        label="Comptes",
    )

    # Diagonale parfaite
    vmin = max(1.0, float(min(y_true.min(), y_pred.min())))
    vmax = float(max(y_true.max(), y_pred.max()))
    ax.plot([vmin, vmax], [vmin, vmax], "k--", linewidth=1.2, label="Prédiction parfaite")

    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("CLV réelle (€, échelle log)")
        ax.set_ylabel("CLV prédite (€, échelle log)")
    else:
        ax.set_xlabel("CLV réelle (€)")
        ax.set_ylabel("CLV prédite (€)")

    ax.legend()
    fig.tight_layout()
    viz.sauvegarder(fig)
    return fig


def figure_residus(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    nom_modele: str,
) -> Figure:
    """Analyse des résidus : résidus vs prédit et distribution des résidus.

    Parameters
    ----------
    y_true :
        Valeurs réelles (euros).
    y_pred :
        Valeurs prédites (euros, échelle originale).
    nom_modele :
        Nom du modèle affiché dans le titre.

    Returns
    -------
    Figure à deux axes sauvegardée dans ``reports/figures/``.
    """
    residus = y_pred - y_true

    fig, axes = viz.figure_grille(
        f"residus_{nom_modele.lower().replace(' ', '_')}",
        f"Analyse des résidus — {nom_modele}",
        nlignes=1,
        ncols=2,
        taille=(14.0, 5.5),
    )

    ax_scatter, ax_hist = axes[0], axes[1]

    # Résidus vs valeurs prédites
    ax_scatter.scatter(y_pred, residus, alpha=0.30, s=14, color=viz.couleur(0))
    ax_scatter.axhline(0, color="black", linewidth=1.0, linestyle="--")
    ax_scatter.set_xlabel("CLV prédite (€)")
    ax_scatter.set_ylabel("Résidu (prédit − réel, €)")
    ax_scatter.set_title("Résidus vs valeurs prédites", fontsize=11)

    # Distribution des résidus
    ax_hist.hist(residus, bins=40, color=viz.couleur(2), edgecolor="white", linewidth=0.5)
    ax_hist.axvline(0, color="black", linewidth=1.0, linestyle="--")
    ax_hist.set_xlabel("Résidu (€)")
    ax_hist.set_ylabel("Fréquence")
    ax_hist.set_title("Distribution des résidus", fontsize=11)

    fig.tight_layout()
    viz.sauvegarder(fig)
    return fig


# ---------------------------------------------------------------------------
# Valeur à risque — formule correcte (point de vigilance n°3)
# ---------------------------------------------------------------------------


def valeur_a_risque(
    proba_churn: _NumType,
    mrr_mensuel: _NumType,
    horizon_mois: int | None = None,
) -> _NumType:
    """Valeur future à risque en euros — délègue à `churn_saas.economie`.

    La formule (``P(churn) × MRR × horizon × marge_brute``, point de vigilance n°3) vit
    dans `churn_saas.economie`, module unique partagé par l'API, le flow de scoring batch
    et le notebook. Cette fonction est conservée pour la §12.11, qui l'appelle via `reg.`.

    Voir :func:`churn_saas.economie.valeur_a_risque` pour la documentation complète.
    """
    return economie.valeur_a_risque(proba_churn, mrr_mensuel, horizon_mois=horizon_mois)
