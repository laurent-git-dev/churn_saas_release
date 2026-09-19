"""Construction du préprocesseur sklearn pour le pipeline de modélisation.

Ce module est la SEULE autorité sur quelles colonnes atteignent le modèle.
Toute modification du périmètre des features passe ici — jamais dispersée
dans le notebook ou dans les scripts d'entraînement.
"""

from __future__ import annotations

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
