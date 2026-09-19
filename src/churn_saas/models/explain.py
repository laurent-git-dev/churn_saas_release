"""Explicabilité — importances, SHAP et fiche compte client.

Trois niveaux de preuve pour trancher sur les leurres (point de vigilance n°5) :
1. Importance par impureté  — biaisée (cardinalité), montrée avec mise en garde obligatoire.
2. Importance de permutation sur le jeu de test — méthode correcte, avec IC à 95 %.
3. Importance drop-column — retrait + réentraînement, la preuve propre.

Deux livrables métier :
- :func:`valeurs_shap`    : SHAP global + local, mis en cache.
- :func:`fiche_compte`    : rapport actionnable pour un compte donné.
- :func:`verdict_leurres` : verdict final, 3 preuves convergentes par colonne.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd
import shap
from loguru import logger
from scipy import stats as scipy_stats
from sklearn.base import clone
from sklearn.inspection import permutation_importance as sk_permutation_importance
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from churn_saas import cache, config

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

# Mises en garde selon le type de modèle
_CAVEAT_IMPURETE: str = (
    "⚠️  Biais de cardinalité : l'importance par impureté (MDI) surestime les variables "
    "à forte cardinalité (continus, quasi-identifiants). "
    "À interpréter uniquement en complément de la permutation importance."
)
_CAVEAT_COEF: str = (
    "⚠️  Importance approximée via |coef_| (modèle linéaire). Sensible à l'échelle des "
    "features et à la régularisation — valide uniquement si les features sont standardisées. "
    "À compléter impérativement par la permutation importance."
)

# Seuils pour le verdict final
_SEUIL_PERM_NUL: float = 0.0  # IC₉₅ supérieur de permutation importance ≤ 0 → nul
_SEUIL_DROP_NEG: float = 0.01  # perte de PR-AUC < 1 % → drop négligeable
_SEUIL_REDONDANCE: float = 0.70  # max_redondance ≥ 0,70 → jumelle probable

# Seuil de significativité après correction BH
_ALPHA_BH: float = 0.05

# Verdicts
_LEURRE = "LEURRE CONFIRMÉ"
_REDONDANT = "REDONDANT (jumelle probable)"
_UTILE = "UTILE — à conserver"
_AMBIGU = "AMBIGU — investigation complémentaire"

# ---------------------------------------------------------------------------
# Helpers internes
# ---------------------------------------------------------------------------


def _estimateur_final(modele: Any) -> Any:
    """Extrait l'estimateur final d'un Pipeline, sinon retourne l'objet lui-même."""
    if isinstance(modele, Pipeline):
        return modele[-1]
    return modele


def _transformer_X(modele: Any, X: pd.DataFrame) -> np.ndarray:
    """Transforme X via toutes les étapes d'un Pipeline sauf la dernière (déjà fitté)."""
    if isinstance(modele, Pipeline) and len(modele.steps) > 1:
        sous_pipeline = modele[:-1]
        return np.asarray(sous_pipeline.transform(X))
    return np.asarray(X)


def _noms_features_transformees(modele: Any, X: pd.DataFrame) -> list[str]:
    """Retourne les noms de features après prétraitement (ou les colonnes de X en entrée)."""
    if isinstance(modele, Pipeline) and len(modele.steps) > 1:
        sous_pipeline = modele[:-1]
        try:
            return [str(n) for n in sous_pipeline.get_feature_names_out()]
        except Exception:
            pass
    estimateur = _estimateur_final(modele)
    if hasattr(estimateur, "feature_names_in_"):
        return [str(n) for n in estimateur.feature_names_in_]
    return list(X.columns)


# ---------------------------------------------------------------------------
# 1. Importance par impureté (MDI)
# ---------------------------------------------------------------------------


def importance_impurete(modele: Any) -> pd.DataFrame:
    """Importance native du modèle (MDI) avec mise en garde sur le biais de cardinalité.

    Cette importance est requise par l'énoncé (C5) mais elle est biaisée vers les variables
    à forte cardinalité. Elle doit toujours être accompagnée de sa mise en garde et complétée
    par la permutation importance.

    Parameters
    ----------
    modele :
        Modèle entraîné (ou Pipeline sklearn). Supporte les modèles à arbres
        (``feature_importances_``) et les modèles linéaires (``coef_`` → ``|coef_|``).

    Returns
    -------
    DataFrame trié par importance décroissante avec colonnes :
    ``feature``, ``importance``, ``rang``.
    L'attribut ``mise_en_garde`` est stocké dans ``df.attrs``.

    Raises
    ------
    AttributeError
        Si l'estimateur final ne dispose ni de ``feature_importances_`` ni de ``coef_``.
    """
    estimateur = _estimateur_final(modele)

    if hasattr(estimateur, "feature_importances_"):
        importances: np.ndarray = estimateur.feature_importances_
        caveat = _CAVEAT_IMPURETE
    elif hasattr(estimateur, "coef_"):
        coef = np.asarray(estimateur.coef_)
        importances = np.abs(coef).flatten()
        caveat = _CAVEAT_COEF
    else:
        raise AttributeError(
            f"L'estimateur {type(estimateur).__name__!r} ne fournit ni "
            "feature_importances_ ni coef_. Méthode non applicable."
        )

    # Récupère les noms de features depuis le Pipeline (ColumnTransformer) si possible
    if isinstance(modele, Pipeline) and hasattr(modele[:-1], "get_feature_names_out"):
        try:
            noms: list[str] = [str(n) for n in modele[:-1].get_feature_names_out()]
        except Exception:
            noms = [f"feature_{i}" for i in range(len(importances))]
    elif hasattr(estimateur, "feature_names_in_"):
        noms = [str(n) for n in estimateur.feature_names_in_]
    else:
        noms = [f"feature_{i}" for i in range(len(importances))]

    # Troncature si la longueur ne correspond pas (robustesse)
    n = min(len(noms), len(importances))
    noms = noms[:n]
    importances = importances[:n]

    df = (
        pd.DataFrame({"feature": noms, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    df.insert(0, "rang", range(1, len(df) + 1))
    df["importance"] = df["importance"].round(6)
    df.attrs["mise_en_garde"] = caveat

    logger.warning("importance_impurete — {}", caveat)
    logger.info(
        "importance_impurete — top-5 : {}",
        df.head(5)[["feature", "importance"]].to_dict("records"),
    )
    return df


# ---------------------------------------------------------------------------
# 2. Importance de permutation
# ---------------------------------------------------------------------------


def importance_permutation(
    modele: Any,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    n_repeats: int = 20,
    n_jobs: int = -1,
    scoring: str = "average_precision",
) -> pd.DataFrame:
    """Importance de permutation sur le jeu de TEST, avec intervalles de confiance à 95 %.

    Méthode correcte pour détecter les variables sans apport au modèle, sans le biais
    de cardinalité de l'importance par impureté.

    ⚠️  Une permutation importance nulle ne prouve pas qu'une variable est un leurre :
    une variable *redondante* (colinéaire avec une jumelle) donne aussi une importance
    proche de zéro. Toujours croiser avec drop-column et association marginale.

    Parameters
    ----------
    modele :
        Modèle entraîné (ou Pipeline complet — la permutation opère sur les features d'entrée).
    X :
        Jeu de TEST exclusivement (sur le jeu d'entraînement, les importances seraient gonflées).
    y :
        Vérité terrain correspondant à X.
    n_repeats :
        Nombre de permutations par feature (≥ 20 pour des IC robustes).
    n_jobs :
        Parallélisation (-1 = tous les cœurs disponibles).
    scoring :
        Métrique (``"average_precision"`` = PR-AUC, métrique principale du projet).

    Returns
    -------
    DataFrame trié par importance décroissante avec colonnes :
    ``feature``, ``importance_moyenne``, ``std``, ``ic95_bas``, ``ic95_haut``,
    ``significatif`` (IC₉₅ supérieur > 0).
    """
    y_arr = np.asarray(y)
    result = sk_permutation_importance(
        modele,
        X,
        y_arr,
        n_repeats=n_repeats,
        random_state=config.RANDOM_SEED,
        n_jobs=n_jobs,
        scoring=scoring,
    )

    # IC à 95 % — approximation normale valide pour n_repeats ≥ 20
    z95: float = float(scipy_stats.norm.ppf(0.975))
    marge: np.ndarray = z95 * result.importances_std / np.sqrt(n_repeats)

    df = (
        pd.DataFrame(
            {
                "feature": list(X.columns),
                "importance_moyenne": result.importances_mean.round(6),
                "std": result.importances_std.round(6),
                "ic95_bas": (result.importances_mean - marge).round(6),
                "ic95_haut": (result.importances_mean + marge).round(6),
            }
        )
        .sort_values("importance_moyenne", ascending=False)
        .reset_index(drop=True)
    )
    df["significatif"] = df["ic95_haut"] > _SEUIL_PERM_NUL

    n_nuls = int((~df["significatif"]).sum())
    logger.info(
        "importance_permutation — {}/{} feature(s) avec IC₉₅ entièrement ≤ 0 "
        "(candidats leurres ou redondants).",
        n_nuls,
        len(df),
    )
    return df


# ---------------------------------------------------------------------------
# 3. Importance drop-column
# ---------------------------------------------------------------------------


def importance_drop_column(
    modele: Any,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    colonnes: list[str] | None = None,
    *,
    forcer: bool = False,
    test_size: float = 0.2,
) -> pd.DataFrame:
    """Perte de PR-AUC quand on retire une colonne et réentraîne le modèle.

    Méthode coûteuse mais conclusive (point de vigilance n°5). Le modèle final est
    réentraîné sur les features transformées, sans les features issues de la colonne
    retirée — ce qui évite les conflits de schéma avec les ColumnTransformer.

    Une perte négligeable ET une association marginale non significative prouvent
    que la variable n'apporte rien, ni seule ni en complément d'une jumelle.

    Parameters
    ----------
    modele :
        Modèle (ou Pipeline) entraîné. Le prétraitement (toutes les étapes sauf la
        dernière) est rejoué sur X_train pour chaque itération.
    X :
        Features d'entrée — divisées en 80/20 en interne (``RANDOM_SEED``).
    y :
        Vérité terrain (même index que X).
    colonnes :
        Colonnes originales à tester. Par défaut : ``config.COLONNES_LEURRES_SUSPECTES``
        présentes dans X.
    forcer :
        Si True, ignore le cache et recalcule.
    test_size :
        Proportion du jeu de test pour la mesure interne.

    Returns
    -------
    DataFrame indexé par colonne avec colonnes :
    ``pr_auc_sans``, ``pr_auc_baseline``, ``delta_pr_auc``, ``negligeable``.
    """
    if colonnes is None:
        colonnes = [c for c in config.COLONNES_LEURRES_SUSPECTES if c in X.columns]

    colonnes_presentes = [c for c in colonnes if c in X.columns]
    if not colonnes_presentes:
        logger.warning("importance_drop_column — aucune colonne candidate présente dans X.")
        return pd.DataFrame(
            columns=["pr_auc_sans", "pr_auc_baseline", "delta_pr_auc", "negligeable"]
        )

    cle = hashlib.md5(
        (
            f"{type(_estimateur_final(modele)).__name__}"
            f"_{'_'.join(sorted(colonnes_presentes))}"
            f"_{X.shape[0]}x{X.shape[1]}"
        ).encode()
    ).hexdigest()[:8]
    nom_cache = f"drop_column_{cle}.parquet"

    def _calculer() -> pd.DataFrame:
        y_arr = np.asarray(y)
        idx_train, idx_test = train_test_split(
            X.index,
            test_size=test_size,
            stratify=y_arr,
            random_state=config.RANDOM_SEED,
        )
        X_train = X.loc[idx_train]
        X_test = X.loc[idx_test]
        pos_train = X.index.get_indexer(idx_train)
        pos_test = X.index.get_indexer(idx_test)
        y_train = y_arr[pos_train]
        y_test = y_arr[pos_test]

        # Prétraitement des deux ensembles (étapes sauf la dernière — déjà fitté)
        X_trans_train = _transformer_X(modele, X_train)
        X_trans_test = _transformer_X(modele, X_test)
        noms_trans = _noms_features_transformees(modele, X_train)

        # Baseline : estimateur final cloné, entraîné sur toutes les features transformées
        estimateur = _estimateur_final(modele)
        clf_base = clone(estimateur)
        clf_base.fit(X_trans_train, y_train)
        proba_base = clf_base.predict_proba(X_trans_test)[:, 1]
        pr_auc_base = float(average_precision_score(y_test, proba_base))
        logger.info("drop_column baseline PR-AUC = {:.4f}", pr_auc_base)

        lignes: list[dict[str, object]] = []
        for col in colonnes_presentes:
            # Masque des features transformées *non* issues de cette colonne
            mask_conserver = np.array([col not in n for n in noms_trans])

            if mask_conserver.all():
                # Colonne absente du prétraitement (ex. commentaire_csm exclu) → Δ = 0
                logger.info(
                    "drop_column '{}' — absente du prétraitement, Δ = 0 (leurre non encodé).",
                    col,
                )
                pr_auc_sans = pr_auc_base
                delta: float = 0.0
                negligeable = True
            else:
                X_train_sans = X_trans_train[:, mask_conserver]
                X_test_sans = X_trans_test[:, mask_conserver]
                clf_sans = clone(estimateur)
                try:
                    clf_sans.fit(X_train_sans, y_train)
                    proba_sans = clf_sans.predict_proba(X_test_sans)[:, 1]
                    pr_auc_sans = float(average_precision_score(y_test, proba_sans))
                    delta = pr_auc_base - pr_auc_sans
                    negligeable = delta < _SEUIL_DROP_NEG
                except Exception as exc:
                    logger.warning("drop_column '{}' — erreur lors de l'ajustement : {}", col, exc)
                    pr_auc_sans = float("nan")
                    delta = float("nan")
                    negligeable = False

            logger.info(
                "drop_column '{}' — PR-AUC={:.4f}  Δ={:+.4f}  négligeable={}",
                col,
                pr_auc_sans,
                delta,
                negligeable,
            )
            lignes.append(
                {
                    "colonne": col,
                    "pr_auc_sans": round(float(pr_auc_sans), 6),
                    "pr_auc_baseline": round(pr_auc_base, 6),
                    "delta_pr_auc": round(delta, 6) if not np.isnan(delta) else float("nan"),
                    "negligeable": negligeable,
                }
            )

        return pd.DataFrame(lignes).set_index("colonne")

    resultat, _ = cache.charger_ou_calculer(nom_cache, _calculer, forcer=forcer)
    return pd.DataFrame(resultat) if not isinstance(resultat, pd.DataFrame) else resultat


# ---------------------------------------------------------------------------
# 4. Valeurs SHAP
# ---------------------------------------------------------------------------


def valeurs_shap(
    modele: Any,
    X: pd.DataFrame,
    *,
    nom_cache: str = "shap_values.joblib",
    forcer: bool = False,
    max_sample_bg: int = 100,
) -> tuple[Any, np.ndarray]:
    """Calcule et met en cache les valeurs SHAP (globales + locales).

    Sélectionne automatiquement l'explainer adapté au type de modèle :
    - **TreeExplainer** pour les arbres (RandomForest, HistGradientBoosting…).
    - **LinearExplainer** pour la régression logistique.
    - **KernelExplainer** (lent) en dernier recours.

    Le calcul opère sur les features transformées (sortie du prétraitement),
    ce qui garantit que les valeurs SHAP reflètent ce que le modèle voit réellement.

    Parameters
    ----------
    modele :
        Modèle entraîné (ou Pipeline sklearn).
    X :
        Données sur lesquelles calculer SHAP. Recommandation : jeu de TEST.
    nom_cache :
        Nom du fichier ``.joblib`` dans ``reports/tables/``.
    forcer :
        Si True, ignore le cache et recalcule.
    max_sample_bg :
        Taille du background pour le KernelExplainer (100 = bon compromis).

    Returns
    -------
    (explainer, shap_values)
        ``shap_values`` de forme *(n_échantillons, n_features)*, valeurs pour la
        classe positive (churn = 1).
    """

    def _calculer() -> dict[str, Any]:
        estimateur = _estimateur_final(modele)
        X_trans = _transformer_X(modele, X)
        nom_classe = type(estimateur).__name__

        sv: np.ndarray
        explainer: Any

        if hasattr(estimateur, "feature_importances_"):
            logger.info("SHAP — TreeExplainer pour {}.", nom_classe)
            explainer = shap.TreeExplainer(estimateur)
            exp = explainer(X_trans)
            raw = exp.values
            if isinstance(raw, list):
                sv = np.asarray(raw[1])
            elif raw.ndim == 3:
                # (n_samples, n_features, n_classes)
                sv = raw[:, :, 1]
            else:
                sv = np.asarray(raw)
        elif hasattr(estimateur, "coef_"):
            logger.info("SHAP — LinearExplainer pour {}.", nom_classe)
            explainer = shap.LinearExplainer(estimateur, X_trans)
            raw = explainer.shap_values(X_trans)
            sv = np.asarray(raw[1] if isinstance(raw, list) else raw)
        else:
            logger.warning("SHAP — KernelExplainer (lent) pour {}.", nom_classe)
            bg = shap.sample(X_trans, max_sample_bg)

            def _predict_proba(x: np.ndarray) -> np.ndarray:
                return np.asarray(estimateur.predict_proba(x)[:, 1])

            explainer = shap.KernelExplainer(_predict_proba, bg)
            raw = explainer.shap_values(X_trans, nsamples=200)
            sv = np.asarray(raw[1] if isinstance(raw, list) else raw)

        return {"explainer": explainer, "shap_values": sv}

    resultat, _ = cache.charger_ou_calculer(nom_cache, _calculer, forcer=forcer)
    return resultat["explainer"], np.asarray(resultat["shap_values"])


# ---------------------------------------------------------------------------
# 5. Fiche compte
# ---------------------------------------------------------------------------

# Mapping feature-préfixe → action préventive rédigée pour le CSM
_ACTIONS: dict[str, str] = {
    "derniere_connexion": (
        "Planifier un appel de reprise d'usage sous 5 jours ouvrés ; "
        "partager une synthèse des nouvelles fonctionnalités activées depuis la dernière connexion."
    ),
    "tickets_ouverts": (
        "Escalader les tickets en attente au niveau N+1 sous 48 h ; "
        "affecter un CSM référent dédié pour les 30 prochains jours."
    ),
    "csat": (
        "Déclencher un appel de satisfaction sous 48 h ; "
        "proposer un audit d'usage gratuit avec l'équipe Success."
    ),
    "nb_utilisateurs": (
        "Proposer un plan d'accompagnement à l'adoption : "
        "formation sur site, webinaire dédié ou session Lunch & Learn."
    ),
    "anciennete": (
        "Revoir les conditions de renouvellement lors du prochain QBR ; "
        "proposer un contrat pluriannuel à tarif préférentiel."
    ),
    "revenu_mensuel": (
        "Mettre en avant les fonctionnalités premium non utilisées "
        "lors du prochain point de suivi pour justifier le MRR actuel."
    ),
    "taux_utilisation": (
        "Lancer un programme d'onboarding ciblé pour les utilisateurs inactifs du compte ; "
        "identifier les obstacles à l'adoption via un court questionnaire."
    ),
    "plan": (
        "Proposer une démonstration des fonctionnalités du tier supérieur "
        "lors du prochain appel trimestriel."
    ),
    "nb_integrations": (
        "Organiser un atelier d'intégration API avec l'équipe technique du client "
        "pour débloquer les cas d'usage non couverts."
    ),
}


def _action_preventive(nom_feature: str) -> str:
    """Retourne l'action préventive associée au feature SHAP dominant."""
    nom_lower = nom_feature.lower()
    for prefixe, action in _ACTIONS.items():
        if prefixe in nom_lower:
            return action
    return (
        f"Analyser le profil complet du compte avec le CSM référent "
        f"(facteur dominant identifié : « {nom_feature} »)."
    )


def fiche_compte(
    client: pd.Series,
    modele: Any,
    X: pd.DataFrame,
    shap_values: np.ndarray,
    *,
    mrr_col: str = "revenu_mensuel_recurrent_eur",
) -> dict[str, Any]:
    """Fiche actionnable pour un compte : probabilité, valeur à risque et facteurs SHAP.

    Produit le livrable métier principal destiné au Customer Success Manager :
    un rapport lisible, sans jargon technique, avec une recommandation d'action
    directement dérivée du facteur SHAP dominant.

    La valeur à risque est calculée sur un horizon futur explicite (point de vigilance n°3) :

    .. code-block:: text

        valeur_à_risque = P(churn) × MRR_mensuel × horizon_mois × marge_brute

    Parameters
    ----------
    client :
        Ligne du DataFrame (``pd.Series``). Son ``name`` doit être un index de X.
    modele :
        Modèle de churn entraîné (ou Pipeline sklearn).
    X :
        Dataset sur lequel ``shap_values`` a été calculé.
    shap_values :
        Valeurs SHAP pour la classe positive, forme *(n_échantillons, n_features)*.
    mrr_col :
        Colonne contenant le MRR mensuel en euros (présente dans ``client``).

    Returns
    -------
    dict avec les clés :
    ``proba_churn``, ``niveau_risque``, ``valeur_risque_eur``,
    ``facteurs_churn`` (3 dicts), ``facteurs_protection`` (2 dicts),
    ``action_preventive``.

    Raises
    ------
    KeyError
        Si ``client.name`` n'est pas dans l'index de X.
    """
    if client.name not in X.index:
        raise KeyError(
            f"L'index du client ({client.name!r}) est absent de X. "
            "Vérifier que X est bien le dataset utilisé pour calculer shap_values."
        )

    pos = int(X.index.get_loc(client.name))
    sv_client = shap_values[pos]
    noms_features = list(X.columns)

    proba = float(modele.predict_proba(X.iloc[[pos]])[0, 1])

    if proba >= 0.70:
        niveau = "ÉLEVÉ"
    elif proba >= 0.40:
        niveau = "MODÉRÉ"
    else:
        niveau = "FAIBLE"

    hyp = config.HYPOTHESES_ECONOMIQUES
    mrr = float(pd.to_numeric(client.get(mrr_col, 0), errors="coerce") or 0.0)
    valeur_risque = proba * mrr * float(hyp["horizon_mois"]) * float(hyp["marge_brute_pct"])

    # Tri des valeurs SHAP : positif = facteur de churn, négatif = facteur protecteur
    indices_desc = np.argsort(sv_client)[::-1]
    facteurs_churn = [
        {"feature": noms_features[i], "contribution_shap": round(float(sv_client[i]), 4)}
        for i in indices_desc[:3]
        if float(sv_client[i]) > 0
    ]
    facteurs_protection = [
        {"feature": noms_features[i], "contribution_shap": round(float(sv_client[i]), 4)}
        for i in np.argsort(sv_client)[:2]
        if float(sv_client[i]) < 0
    ]

    facteur_dominant = (
        facteurs_churn[0]["feature"] if facteurs_churn else noms_features[int(indices_desc[0])]
    )
    action = _action_preventive(str(facteur_dominant))

    fiche: dict[str, Any] = {
        "proba_churn": round(proba, 4),
        "niveau_risque": niveau,
        "valeur_risque_eur": round(valeur_risque, 2),
        "facteurs_churn": facteurs_churn,
        "facteurs_protection": facteurs_protection,
        "action_preventive": action,
    }

    logger.info(
        "fiche_compte({}) — proba={:.2%}  risque={:.0f} €  niveau={}  action=«{}»",
        client.name,
        proba,
        valeur_risque,
        niveau,
        action[:70] + "…" if len(action) > 70 else action,
    )
    return fiche


# ---------------------------------------------------------------------------
# 6. Verdict leurres
# ---------------------------------------------------------------------------


def verdict_leurres(
    tableau_criblage: pd.DataFrame,
    permutation_imp: pd.DataFrame,
    drop_col_imp: pd.DataFrame,
) -> pd.DataFrame:
    """Verdict final par colonne suspectée — 3 preuves convergentes.

    Consolide les résultats de :func:`~churn_saas.fuite.cribler_leurres`,
    :func:`importance_permutation` et :func:`importance_drop_column`.

    Règle de décision (point de vigilance n°5) :

    +---------------------+------------------+----------------+-----------------------+
    | Assoc. marginale     | Permutation      | Drop-column    | Verdict               |
    +=====================+==================+================+=======================+
    | non significative   | nulle (IC₉₅ ≤ 0) | négligeable    | LEURRE CONFIRMÉ       |
    +---------------------+------------------+----------------+-----------------------+
    | —                   | nulle             | —              | REDONDANT (si jumelle)|
    +---------------------+------------------+----------------+-----------------------+
    | significative OU    | positive          | —              | UTILE                 |
    +---------------------+------------------+----------------+-----------------------+
    | contradictions      | —                 | —              | AMBIGU                |
    +---------------------+------------------+----------------+-----------------------+

    Parameters
    ----------
    tableau_criblage :
        Résultat de :func:`~churn_saas.fuite.cribler_leurres` (index = colonne).
        Doit contenir ``significatif_BH``, ``max_redondance``, ``variable_jumelle``.
    permutation_imp :
        Résultat de :func:`importance_permutation` (colonne ``feature``).
    drop_col_imp :
        Résultat de :func:`importance_drop_column` (index = colonne).

    Returns
    -------
    DataFrame indexé par colonne avec colonnes :
    ``association_significative``, ``permutation_nulle``, ``drop_negligeable``,
    ``max_redondance``, ``variable_jumelle``, ``verdict``, ``raisonnement``.
    """
    perm_idx = (
        permutation_imp.set_index("feature")
        if "feature" in permutation_imp.columns
        else permutation_imp
    )

    colonnes = list(tableau_criblage.index)
    lignes: list[dict[str, Any]] = []

    for col in colonnes:
        # Preuve 1 : association marginale (Cramér V ou Mann-Whitney, corrigée BH)
        assoc_sig = (
            bool(tableau_criblage.loc[col, "significatif_BH"])
            if col in tableau_criblage.index
            else False
        )  # noqa: E501

        # Preuve 2 : permutation importance (IC₉₅ supérieur)
        if col in perm_idx.index:
            ic_haut = float(perm_idx.loc[col, "ic95_haut"])
            perm_nulle = ic_haut <= _SEUIL_PERM_NUL
        else:
            ic_haut = float("nan")
            perm_nulle = False

        # Preuve 3 : drop-column (perte de PR-AUC)
        if col in drop_col_imp.index:
            delta = float(drop_col_imp.loc[col, "delta_pr_auc"])
            drop_neg = bool(drop_col_imp.loc[col, "negligeable"])
        else:
            delta = float("nan")
            drop_neg = False

        redondance = (
            float(tableau_criblage.loc[col, "max_redondance"])
            if col in tableau_criblage.index
            else 0.0
        )  # noqa: E501
        jumelle = (
            str(tableau_criblage.loc[col, "variable_jumelle"])
            if col in tableau_criblage.index
            else ""
        )  # noqa: E501

        # --- Décision ---
        # La redondance est vérifiée en premier : une variable colinéaire avec une jumelle
        # peut montrer permutation nulle ET drop-column négligeable sans être un leurre —
        # le modèle compense via la jumelle (point de vigilance n°5).
        if perm_nulle and redondance >= _SEUIL_REDONDANCE:
            verdict = _REDONDANT
            raisonnement = (
                f"Permutation nulle (IC₉₅ = {ic_haut:.4f}) MAIS redondance élevée "
                f"avec « {jumelle} » (r={redondance:.3f} ≥ {_SEUIL_REDONDANCE}). "
                "Le modèle compense via la jumelle — importance nulle ≠ leurre "
                "(point de vigilance n°5)."
            )
        elif not assoc_sig and perm_nulle and drop_neg:
            verdict = _LEURRE
            raisonnement = (
                f"Association non significative (p_BH ≥ {_ALPHA_BH}) ; "
                f"IC₉₅ permutation = {ic_haut:.4f} ≤ 0 ; "
                f"drop-column Δ = {delta:+.4f} (< {_SEUIL_DROP_NEG:.0%}). "
                "Les trois preuves convergent : variable inutile au modèle."
            )
        elif assoc_sig or not perm_nulle:
            verdict = _UTILE
            raisonnement = (
                f"Association {'significative' if assoc_sig else 'non significative'} (p_BH) ; "
                f"IC₉₅ permutation = {ic_haut:.4f}."
                " Au moins une preuve plaide pour la conservation."
            )
        else:
            verdict = _AMBIGU
            raisonnement = (
                f"Preuves contradictoires : permutation nulle (IC₉₅ = {ic_haut:.4f}) "
                f"mais drop-column Δ = {delta:+.4f}. "
                "Investiguer une éventuelle colinéarité non détectée."
            )

        lignes.append(
            {
                "colonne": col,
                "association_significative": assoc_sig,
                "permutation_nulle": perm_nulle,
                "drop_negligeable": drop_neg,
                "max_redondance": round(redondance, 4),
                "variable_jumelle": jumelle,
                "verdict": verdict,
                "raisonnement": raisonnement,
            }
        )
        logger.info("verdict_leurres — {} → {}", col, verdict)

    return pd.DataFrame(lignes).set_index("colonne")
