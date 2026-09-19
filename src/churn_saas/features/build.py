"""Construction du préprocesseur sklearn et des features métier pour le pipeline.

Ce module est la SEULE autorité sur quelles colonnes atteignent le modèle.
Toute modification du périmètre des features passe ici — jamais dispersée
dans le notebook ou dans les scripts d'entraînement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_saas import config


def colonnes_features(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Retourne (colonnes_numériques, colonnes_catégorielles) admissibles au modèle.

    C'est la SEULE fonction qui décide quelles colonnes atteignent le modèle.
    Elle lève une exception si une colonne de ``config.COLONNES_INTERDITES`` est présente,
    forçant l'appelant à épurer le DataFrame avant de continuer.

    Parameters
    ----------
    df:
        DataFrame épuré des colonnes interdites. Si elles sont encore présentes,
        la fonction lève plutôt que de les ignorer silencieusement.

    Returns
    -------
    (numeriques, categorielles) — listes de noms de colonnes, sans les interdites.

    Raises
    ------
    ValueError
        Si une colonne de ``config.COLONNES_INTERDITES`` est présente dans ``df``.
    """
    presentes_interdites = set(df.columns) & set(config.COLONNES_INTERDITES)
    if presentes_interdites:
        raise ValueError(
            f"Colonnes interdites détectées dans le DataFrame : {sorted(presentes_interdites)}. "
            "Épurer avec df.drop(columns=config.COLONNES_INTERDITES, errors='ignore') avant "
            "d'appeler colonnes_features()."
        )

    numeriques = [
        col
        for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col])
    ]
    categorielles = [
        col
        for col in df.columns
        if pd.api.types.is_object_dtype(df[col]) or isinstance(df[col].dtype, pd.CategoricalDtype)
    ]
    logger.info(
        "colonnes_features — {} numériques, {} catégorielles sélectionnées",
        len(numeriques),
        len(categorielles),
    )
    return numeriques, categorielles


def construire_preprocesseur(df: pd.DataFrame) -> ColumnTransformer:
    """Construit un ColumnTransformer prêt à s'intégrer dans un Pipeline sklearn.

    Traitements appliqués :
    - **Numériques** : imputation médiane → StandardScaler.
    - **Catégorielles** : imputation par la modalité ``"inconnu"`` → OneHotEncoder
      avec ``handle_unknown="ignore"`` (modalités inconnues au transform → vecteur nul).
    - **Colonnes interdites** : exclues via ``remainder="drop"`` — elles n'atteignent
      jamais le modèle même si le DataFrame d'entrée les contient.

    Le schéma des colonnes est figé à la construction (pas au fit), ce qui garantit
    que les noms de features sont stables entre les plis de validation croisée.

    Parameters
    ----------
    df:
        DataFrame de référence pour déduire le schéma (colonnes + types).
        Peut contenir les colonnes interdites : elles sont retirées en interne
        avant de construire le ColumnTransformer.

    Returns
    -------
    ColumnTransformer configuré, non encore fitté.
    """
    df_features = df.drop(columns=config.COLONNES_INTERDITES, errors="ignore")
    numeriques, categorielles = colonnes_features(df_features)

    pipeline_numerique = Pipeline(
        steps=[
            ("imputation", SimpleImputer(strategy="median")),
            ("standardisation", StandardScaler()),
        ]
    )
    pipeline_categoriel = Pipeline(
        steps=[
            ("imputation", SimpleImputer(strategy="constant", fill_value="inconnu")),
            (
                "encodage",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocesseur = ColumnTransformer(
        transformers=[
            ("numerique", pipeline_numerique, numeriques),
            ("categoriel", pipeline_categoriel, categorielles),
        ],
        # Toutes les colonnes non listées (interdites, dates…) tombent ici et sont supprimées
        remainder="drop",
        verbose_feature_names_out=False,
    )
    logger.info(
        "construire_preprocesseur — schéma : {} numériques, {} catégorielles, reste exclu",
        len(numeriques),
        len(categorielles),
    )
    return preprocesseur


# ---------------------------------------------------------------------------
# Feature engineering métier
# ---------------------------------------------------------------------------


def ajouter_features_metier(
    df: pd.DataFrame,
    csat_median_par_secteur: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Crée ~15 variables dérivées à partir des colonnes brutes.

    Toutes les features sont calculées ligne par ligne — aucune statistique apprise
    sur le jeu d'entraînement n'est nécessaire ici.

    Exception : ``ecart_csat_secteur`` requiert les médianes CSAT par secteur
    apprises sur le train via ``AgregatParGroupe`` — passées via
    ``csat_median_par_secteur``. Si ce paramètre est None, la colonne n'est
    pas créée, garantissant l'absence de fuite.

    Convention division par zéro : NaN (sera imputé dans le pipeline sklearn).

    Parameters
    ----------
    df:
        DataFrame contenant les colonnes brutes source.
    csat_median_par_secteur:
        Dict ``{secteur: médiane_csat}`` appris **uniquement sur le train**
        (via ``AgregatParGroupe``). Si None, ``ecart_csat_secteur`` n'est pas créée.

    Returns
    -------
    DataFrame enrichi des features dérivées (copie — l'original n'est pas modifié).
    """
    df = df.copy()
    _initiales = set(df.columns)

    def _num(col: str) -> pd.Series:
        """Série numérique ou NaN si la colonne est absente."""
        if col not in df.columns:
            logger.warning("ajouter_features_metier — colonne '{}' absente, NaN substitué", col)
            return pd.Series(float("nan"), index=df.index, dtype=float)
        return pd.to_numeric(df[col], errors="coerce")

    sieges = _num("sieges_souscrits")
    utilisateurs = _num("utilisateurs_actifs")
    heures = _num("heures_usage_30j")
    connexions = _num("connexions_30j")
    fonc_util = _num("fonctionnalites_utilisees")
    fonc_total = _num("fonctionnalites_total")
    mrr = _num("revenu_mensuel_recurrent_eur")
    recence = _num("derniere_connexion_jours")
    anciennete = _num("anciennete_mois")
    tickets = _num("tickets_support_90j")
    integrations = _num("nb_integrations")

    # Dénominateurs sécurisés : 0 → NaN pour éviter inf et ZeroDivisionError
    sieges_d = sieges.where(sieges > 0)
    utilisateurs_d = utilisateurs.where(utilisateurs > 0)
    fonc_total_d = fonc_total.where(fonc_total > 0)
    anciennete_jours_d = (anciennete * 30).where(anciennete > 0)

    # 1. Taux d'utilisation des sièges
    # Sous-utilisation → sièges payés non déployés → risque de non-renouvellement
    df["taux_utilisation_sieges"] = utilisateurs / sieges_d

    # 2. Surdimensionnement absolu (sièges vides payés)
    # Nombre visible sur la facture → renforce l'argument de résiliation
    df["surdimensionnement"] = (sieges - utilisateurs).clip(lower=0)

    # 3. Intensité d'usage par utilisateur actif
    # Usage concentré sur peu d'utilisateurs = ancrage précaire dans l'organisation
    df["intensite_usage_par_utilisateur"] = heures / utilisateurs_d

    # 4. Connexions par utilisateur actif
    # Faible ratio = dépendance partielle au produit, usage concentré ou intermittent
    df["connexions_par_utilisateur"] = connexions / utilisateurs_d

    # 5. Taux de couverture fonctionnelle (breadth of use)
    # Peu de fonctionnalités explorées = valeur perçue faible → risque churn faute de ROI
    df["taux_couverture_fonctionnelle"] = fonc_util / fonc_total_d

    # 6. ARPU par siège (Average Revenue Per Unit)
    # ARPU bas = remise commerciale ou plan sous-utilisé → relation contractuelle fragile
    df["arpu_par_siege"] = mrr / sieges_d

    # 7. Récence normalisée (fraction de vie sans connexion)
    # Inactivité longue / ancienneté totale = décrochage progressif, plus fort que les jours bruts
    df["recence_normalisee"] = recence / anciennete_jours_d

    # 8. Compte dormant (inactif depuis > 30 jours)
    # Signal binaire de décrochage — 30 j ≈ un cycle de reporting client
    df["compte_dormant"] = recence > 30

    # 9. Pression support par utilisateur actif
    # Beaucoup de tickets par utilisateur = frictions répétées → prédictif si CSAT faible
    df["pression_support"] = tickets / utilisateurs_d

    # 10. Écart au CSAT médian du secteur (ANTI-FUITE : agrégat externe obligatoire)
    # Un client en dessous de la médiane sectorielle est vulnérable à la concurrence.
    # Les médianes DOIVENT venir d'un AgregatParGroupe fitté sur le train uniquement —
    # jamais calculées sur le jeu complet, ce qui constituerait une fuite de données.
    if csat_median_par_secteur is not None and "csat" in df.columns and "secteur" in df.columns:
        csat_ref = df["secteur"].map(csat_median_par_secteur)
        df["ecart_csat_secteur"] = _num("csat") - csat_ref

    # 11. Tranche d'ancienneté (3 phases du cycle de vie client)
    # Onboarding (<3 mois), consolidation (3-12 mois) et maturité (>12 mois) ont
    # des profils de risque structurellement différents
    df["tranche_anciennete"] = pd.cut(
        anciennete,
        bins=[-0.001, 2, 12, float("inf")],
        labels=["onboarding", "installation", "mature"],
    ).astype(object)

    # 12. Tranche d'intégrations tierces (proxy de stickiness)
    # Chaque intégration augmente le coût de migration et réduit le risque de churn
    df["tranche_integrations"] = pd.cut(
        integrations,
        bins=[-0.001, 0, 3, 8, float("inf")],
        labels=["aucune", "faible", "moderee", "forte"],
    ).astype(object)

    # 13-15. Indicateurs de manquance MNAR (Missing Not At Random)
    # Un CSAT manquant révèle souvent un client insatisfait qui n'a pas répondu à l'enquête —
    # la manquance elle-même est un signal prédictif du churn
    for _col_mnar in ["csat", "heures_usage_30j", "delai_reponse_support_h"]:
        if _col_mnar in df.columns:
            df[f"{_col_mnar}_manquant"] = df[_col_mnar].isna()

    _nouvelles = sorted(set(df.columns) - _initiales)
    logger.info(
        "ajouter_features_metier — {} features ajoutées : {}",
        len(_nouvelles),
        _nouvelles,
    )
    return df


def joindre_catalogue(df: pd.DataFrame, catalogue: pd.DataFrame) -> pd.DataFrame:
    """Joint le catalogue des plans et dérive les features commerciales.

    Dérive :
    - ``remise_consentie`` : écart entre MRR réel et prix catalogue théorique.
      Une remise forte peut signaler une négociation difficile → risque de churn.
    - ``adequation_plan`` : positionnement du client par rapport au plan souscrit.
      Sur-dimensionné (sièges sous-utilisés) = candidat au churn (perçoit un gaspillage) ;
      sous-dimensionné (quasi-saturation) = candidat à l'upsell, pas au churn.

    Convention division par zéro : NaN.

    Parameters
    ----------
    df:
        DataFrame client avec au minimum ``plan``, ``revenu_mensuel_recurrent_eur``,
        ``sieges_souscrits``, ``utilisateurs_actifs``.
    catalogue:
        DataFrame du catalogue (4 lignes, une par plan).
        Colonnes attendues : ``plan``, ``prix_mensuel_par_siege_eur``.

    Returns
    -------
    DataFrame enrichi des colonnes catalogue et des deux features dérivées.
    """
    df = df.copy()

    # Normalisation de la clé de jointure (casse hétérogène dans le jeu réel)
    df["_plan_norm"] = df["plan"].str.strip().str.title() if "plan" in df.columns else None
    cat = catalogue.copy()
    cat["_plan_norm"] = cat["plan"].str.strip().str.title()

    cols_enrichissement = [c for c in cat.columns if c not in {"plan", "_plan_norm"}]
    df = df.merge(
        cat[["_plan_norm"] + cols_enrichissement],
        on="_plan_norm",
        how="left",
    ).drop(columns=["_plan_norm"])

    # Remise consentie = 1 − (MRR réel / prix catalogue × sièges)
    mrr = pd.to_numeric(df.get("revenu_mensuel_recurrent_eur"), errors="coerce")
    prix = pd.to_numeric(df.get("prix_mensuel_par_siege_eur"), errors="coerce")
    sieges = pd.to_numeric(df.get("sieges_souscrits"), errors="coerce")
    prix_catalogue_total = prix * sieges
    # Division par zéro (prix=0 ou sieges=0) → NaN
    df["remise_consentie"] = 1.0 - mrr / prix_catalogue_total.where(prix_catalogue_total > 0)

    # Adéquation au plan : classification du taux d'utilisation des sièges
    utilisateurs = pd.to_numeric(df.get("utilisateurs_actifs"), errors="coerce")
    taux_util = utilisateurs / sieges.where(sieges > 0)
    df["adequation_plan"] = np.select(
        [
            taux_util.isna(),
            taux_util >= 0.9,  # quasi-saturation → sous-dimensionné
            taux_util < 0.5,  # sièges très sous-utilisés → sur-dimensionné
        ],
        [None, "sous-dimensionne", "sur-dimensionne"],
        default="adapte",
    )

    logger.info(
        "joindre_catalogue — {} lignes enrichies, {} colonnes catalogue + 2 features dérivées",
        len(df),
        len(cols_enrichissement),
    )
    return df
