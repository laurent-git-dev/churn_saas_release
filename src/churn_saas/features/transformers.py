"""Transformers scikit-learn pour le pipeline de prétraitement anti-fuite.

Chaque transformer apprend (si besoin) exclusivement sur le jeu d'entraînement
dans fit(), et applique sans apprentissage supplémentaire dans transform().
Ces classes s'intègrent naturellement dans un Pipeline sklearn et dans toute
stratégie de validation croisée (KFold, StratifiedKFold, etc.).
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin

from churn_saas.data.quality import coercer_numeriques

# Préfixes de colonnes structurellement positives — repris de quality.py pour cohérence
_PREFIXES_POSITIFS = (
    "mrr",
    "arr",
    "nb_",
    "nombre_",
    "sieges",
    "utilisateurs",
    "anciennete",
    "score",
    "valeur",
    "montant",
    "revenue",
    "ca_",
)


class CoercionNumerique(BaseEstimator, TransformerMixin):
    """Répare les numériques stockés en texte (séparateurs, symboles monétaires, espaces).

    Délègue à :func:`churn_saas.data.quality.coercer_numeriques`.
    Cette transformation est une réparation de format : aucune statistique n'est apprise
    dans fit(), donc elle n'introduit aucune fuite même si elle précède le split.

    Parameters
    ----------
    colonnes:
        Colonnes à coercer. Si None, auto-détection des colonnes object convertibles
        à plus de 80 % des valeurs non nulles.
    """

    def __init__(self, colonnes: list[str] | None = None) -> None:
        self.colonnes = colonnes

    def fit(self, X: pd.DataFrame, y=None) -> CoercionNumerique:
        if self.colonnes is not None:
            self._colonnes_ = [c for c in self.colonnes if c in X.columns]
        else:
            self._colonnes_ = [
                col
                for col in X.columns
                if pd.api.types.is_object_dtype(X[col])
                and pd.to_numeric(X[col], errors="coerce").notna().mean() > 0.8
            ]
        logger.debug(
            "CoercionNumerique.fit — {} colonnes détectées : {}",
            len(self._colonnes_),
            self._colonnes_,
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._colonnes_:
            return X.copy()
        resultat, _ = coercer_numeriques(X, self._colonnes_)
        return resultat


class IndicateursManquance(BaseEstimator, TransformerMixin):
    """Ajoute des colonnes booléennes de manquance pour les colonnes MNAR.

    Pour chaque colonne `c` dans `colonnes`, crée `{c}_manquant` (True si NaN).
    Ces indicateurs sont eux-mêmes prédictifs : un `csat` manquant révèle souvent
    un client insatisfait qui n'a pas répondu à l'enquête.

    Ce transformer ne fuit pas : il n'apprend aucune statistique dans fit().

    Parameters
    ----------
    colonnes:
        Colonnes pour lesquelles créer un indicateur de manquance.
    """

    def __init__(self, colonnes: list[str]) -> None:
        self.colonnes = colonnes

    def fit(self, X: pd.DataFrame, y=None) -> IndicateursManquance:
        self._colonnes_presentes_ = [c for c in self.colonnes if c in X.columns]
        absentes = set(self.colonnes) - set(self._colonnes_presentes_)
        if absentes:
            logger.warning("IndicateursManquance — colonnes absentes ignorées : {}", absentes)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self._colonnes_presentes_:
            X[f"{col}_manquant"] = X[col].isna()
        return X


class AgregatParGroupe(BaseEstimator, TransformerMixin):
    """Agrégat conditionnel au groupe, appris UNIQUEMENT sur le jeu d'entraînement.

    C'est le transformer le plus critique vis-à-vis de la fuite : calculer la médiane
    du MRR par segment sur tout le jeu avant la CV est la fuite la plus courante en pratique.

    Ce transformer y remédie : la statistique de groupe est apprise dans ``fit()``
    (train seulement) et appliquée dans ``transform()``. Les modalités inconnues
    au moment du transform reçoivent la statistique globale du train (repli sûr).

    Parameters
    ----------
    colonne_groupe:
        Colonne catégorielle définissant les groupes (ex. ``"plan_tarifaire"``).
    colonne_valeur:
        Colonne numérique dont on calcule l'agrégat (ex. ``"mrr_eur"``).
    statistique:
        Agrégat à calculer : ``"median"``, ``"mean"``, ``"std"``, ``"min"``, ``"max"``.
    suffixe:
        Nom de la feature créée. Défaut : ``"{valeur}_{stat}_par_{groupe}"``.
    """

    _STATS_AUTORISEES: frozenset[str] = frozenset({"median", "mean", "std", "min", "max"})

    def __init__(
        self,
        colonne_groupe: str,
        colonne_valeur: str,
        statistique: Literal["median", "mean", "std", "min", "max"] = "median",
        suffixe: str | None = None,
    ) -> None:
        self.colonne_groupe = colonne_groupe
        self.colonne_valeur = colonne_valeur
        self.statistique = statistique
        self.suffixe = suffixe

    def _nom_feature(self) -> str:
        if self.suffixe is not None:
            return self.suffixe
        return f"{self.colonne_valeur}_{self.statistique}_par_{self.colonne_groupe}"

    def fit(self, X: pd.DataFrame, y=None) -> AgregatParGroupe:
        if self.statistique not in self._STATS_AUTORISEES:
            raise ValueError(
                f"statistique={self.statistique!r} non reconnue. "
                f"Valeurs autorisées : {sorted(self._STATS_AUTORISEES)}"
            )
        valeurs = pd.to_numeric(X[self.colonne_valeur], errors="coerce")

        # Statistique par groupe — apprise sur le train uniquement
        self._aggregats_par_groupe_: dict[object, float] = (
            valeurs.groupby(X[self.colonne_groupe]).agg(self.statistique).to_dict()
        )

        # Repli global : statistique sur tout le train, pour les modalités inconnues au transform
        serie_method = getattr(valeurs, self.statistique)
        self._repli_global_: float = float(serie_method())

        logger.debug(
            "AgregatParGroupe.fit — {} groupes pour '{}', repli={:.4f}",
            len(self._aggregats_par_groupe_),
            self.colonne_groupe,
            self._repli_global_,
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        nom = self._nom_feature()
        X[nom] = X[self.colonne_groupe].map(self._aggregats_par_groupe_).fillna(self._repli_global_)
        logger.debug(
            "AgregatParGroupe.transform — feature '{}' créée ({} replis sur modalité inconnue)",
            nom,
            X[nom].isna().sum(),
        )
        return X


class PlafonnerValeursImpossibles(BaseEstimator, TransformerMixin):
    """Corrige les incohérences métier détectées en section 5 de l'analyse.

    Corrections appliquées (règles fixes — aucune statistique apprise sur le train) :
    - Valeurs négatives dans les colonnes structurellement positives → 0.
    - ``taux_adoption_pct`` hors [0, 100] → clip à [0, 100].
    - ``utilisateurs_actifs > sieges_souscrits`` → cap à ``sieges_souscrits``.

    Ce transformer ne fuit pas car les bornes sont des contraintes métier connues a priori,
    non des statistiques inférées à partir des données d'entraînement.
    """

    def fit(self, X: pd.DataFrame, y=None) -> PlafonnerValeursImpossibles:
        # Détection par nom de colonne (pas par dtype — coercion peut ne pas être appliquée)
        self._cols_positives_: list[str] = [
            col
            for col in X.columns
            if any(col.lower().startswith(pfx) for pfx in _PREFIXES_POSITIFS)
        ]
        self._has_taux_adoption_: bool = "taux_adoption_pct" in X.columns
        self._has_contrainte_sieges_: bool = (
            "utilisateurs_actifs" in X.columns and "sieges_souscrits" in X.columns
        )
        logger.debug(
            "PlafonnerValeursImpossibles.fit — {} colonnes positives, "
            "taux_adoption={}, contrainte_sieges={}",
            len(self._cols_positives_),
            self._has_taux_adoption_,
            self._has_contrainte_sieges_,
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        for col in self._cols_positives_:
            if col in X.columns:
                numerique = pd.to_numeric(X[col], errors="coerce")
                X[col] = numerique.clip(lower=0)

        if self._has_taux_adoption_ and "taux_adoption_pct" in X.columns:
            X["taux_adoption_pct"] = pd.to_numeric(X["taux_adoption_pct"], errors="coerce").clip(
                0, 100
            )

        if (
            self._has_contrainte_sieges_
            and "utilisateurs_actifs" in X.columns
            and "sieges_souscrits" in X.columns
        ):
            u = pd.to_numeric(X["utilisateurs_actifs"], errors="coerce")
            s = pd.to_numeric(X["sieges_souscrits"], errors="coerce")
            # clip upper= accepte une Series : plafond par ligne
            X["utilisateurs_actifs"] = u.clip(upper=s)

        return X
