"""Transformers scikit-learn pour le pipeline de prétraitement anti-fuite.

Chaque transformer apprend (si besoin) exclusivement sur le jeu d'entraînement
dans fit(), et applique sans apprentissage supplémentaire dans transform().
Ces classes s'intègrent naturellement dans un Pipeline sklearn et dans toute
stratégie de validation croisée (KFold, StratifiedKFold, etc.).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger
from numpy.typing import ArrayLike
from scipy import stats as scipy_stats
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data

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


def est_texte(serie: pd.Series) -> bool:
    """Vrai si la série contient du texte, quel que soit son type pandas.

    pandas ≥ 3 range les chaînes dans le type ``str`` (``StringDtype``) et non plus
    ``object`` : un test ``is_object_dtype`` seul ignore alors silencieusement ces colonnes.
    """
    return pd.api.types.is_object_dtype(serie) or isinstance(serie.dtype, pd.StringDtype)


class NormalisationCategorielle(OneToOneFeatureMixin, BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Normalise la casse et les espaces des modalités (``.strip().lower()``, cf. §7.3.5).

    Sans elle, « Pro », « pro » et « Pro  » deviennent trois colonnes one-hot distinctes, et
    une variante absente du train est encodée en vecteur nul à l'inférence. Placée dans le
    Pipeline, la même normalisation s'applique à l'entraînement, au notebook et à l'API.

    Aucune statistique n'est apprise dans fit() : pas de fuite possible. Les valeurs
    manquantes restent manquantes (``np.nan``) pour l'imputation qui suit.
    """

    def fit(self, X: pd.DataFrame, y: object = None) -> NormalisationCategorielle:
        X = pd.DataFrame(X)
        self.n_features_in_ = X.shape[1]
        if all(isinstance(c, str) for c in X.columns):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "n_features_in_")
        X = pd.DataFrame(X)
        normalise = X.apply(lambda s: s.astype("string").str.strip().str.lower())
        return normalise.astype(object).where(normalise.notna(), np.nan)


class CoercionNumerique(BaseEstimator, TransformerMixin):  # type: ignore[misc]
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

    def fit(self, X: pd.DataFrame, y: object = None) -> CoercionNumerique:
        if self.colonnes is not None:
            self._colonnes_ = [c for c in self.colonnes if c in X.columns]
        else:
            self._colonnes_ = [
                col
                for col in X.columns
                if est_texte(X[col]) and pd.to_numeric(X[col], errors="coerce").notna().mean() > 0.8
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


class IndicateursManquance(BaseEstimator, TransformerMixin):  # type: ignore[misc]
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

    def fit(self, X: pd.DataFrame, y: object = None) -> IndicateursManquance:
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


class AgregatParGroupe(BaseEstimator, TransformerMixin):  # type: ignore[misc]
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

    def fit(self, X: pd.DataFrame, y: object = None) -> AgregatParGroupe:
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
        mapped = X[self.colonne_groupe].map(self._aggregats_par_groupe_)
        n_replis = int(mapped.isna().sum())  # compter avant fillna, sinon toujours 0
        X[nom] = mapped.fillna(self._repli_global_)
        logger.debug(
            "AgregatParGroupe.transform — feature '{}' créée ({} replis sur modalité inconnue)",
            nom,
            n_replis,
        )
        return X


class EcartAuGroupe(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Écart d'une valeur à l'agrégat de son groupe, appris sur le train du pli.

    Branche dédiée du ``ColumnTransformer`` : reçoit ``[colonne_groupe, colonne_valeur]`` et
    renvoie une seule colonne ``valeur − agrégat_du_groupe``. L'agrégat vient d'un
    :class:`AgregatParGroupe` fitté dans ``fit()``, donc sur le train du pli uniquement ; un
    groupe inconnu au train (ou manquant) reçoit l'agrégat global du train.

    Les libellés de groupe sont normalisés (``.strip().lower()``) avant l'agrégation, comme
    dans :class:`NormalisationCategorielle` : sans cela, « Santé » et « SANTÉ » formeraient
    deux groupes de médianes distinctes. Une valeur manquante donne un écart manquant, imputé
    par l'étape suivante du Pipeline.

    Parameters
    ----------
    colonne_groupe:
        Colonne catégorielle définissant les groupes (ex. ``"secteur"``).
    colonne_valeur:
        Colonne numérique comparée à l'agrégat de son groupe (ex. ``"csat"``).
    statistique:
        Agrégat appris par groupe (voir :class:`AgregatParGroupe`).
    nom:
        Nom de la feature produite. Défaut : ``"ecart_{valeur}_{groupe}"``.
    """

    def __init__(
        self,
        colonne_groupe: str,
        colonne_valeur: str,
        statistique: Literal["median", "mean", "std", "min", "max"] = "median",
        nom: str | None = None,
    ) -> None:
        self.colonne_groupe = colonne_groupe
        self.colonne_valeur = colonne_valeur
        self.statistique = statistique
        self.nom = nom

    def _nom_feature(self) -> str:
        return self.nom or f"ecart_{self.colonne_valeur}_{self.colonne_groupe}"

    def _preparer(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X)
        groupe = X[self.colonne_groupe].astype("string").str.strip().str.lower()
        return pd.DataFrame(
            {
                self.colonne_groupe: groupe.astype(object).where(groupe.notna(), np.nan),
                self.colonne_valeur: pd.to_numeric(X[self.colonne_valeur], errors="coerce"),
            },
            index=X.index,
        )

    def fit(self, X: pd.DataFrame, y: object = None) -> EcartAuGroupe:
        X = pd.DataFrame(X)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.agregat_ = AgregatParGroupe(
            self.colonne_groupe, self.colonne_valeur, self.statistique, suffixe="_reference"
        ).fit(self._preparer(X))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "agregat_")
        prepare = self.agregat_.transform(self._preparer(X))
        ecart = prepare[self.colonne_valeur] - prepare["_reference"]
        return pd.DataFrame({self._nom_feature(): ecart}, index=prepare.index)

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        return np.asarray([self._nom_feature()], dtype=object)


class PlafonnerValeursImpossibles(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Corrige les incohérences métier détectées en section 5 de l'analyse.

    Corrections appliquées (règles fixes — aucune statistique apprise sur le train) :
    - Valeurs négatives dans les colonnes structurellement positives → 0.
    - ``taux_adoption_pct`` hors [0, 100] → clip à [0, 100].
    - ``utilisateurs_actifs > sieges_souscrits`` → cap à ``sieges_souscrits``.

    Ce transformer ne fuit pas car les bornes sont des contraintes métier connues a priori,
    non des statistiques inférées à partir des données d'entraînement.
    """

    def fit(self, X: pd.DataFrame, y: object = None) -> PlafonnerValeursImpossibles:
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


class LogAsymetrique(OneToOneFeatureMixin, BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Applique un log1p signé aux colonnes numériques fortement asymétriques à droite.

    Motivation (§6.2) : la plupart des comptages, durées et montants ont une longue queue
    droite. Pour un modèle linéaire, quelques valeurs extrêmes pèsent alors sur les
    coefficients ; une standardisation seule, transformation affine, n'y change rien.

    - ``fit()`` mesure l'asymétrie de chaque colonne **sur le train du pli uniquement** et
      retient celles dont l'asymétrie dépasse ``seuil_asymetrie`` (hors colonnes binaires) :
      le choix des colonnes est une statistique apprise, il reste donc dans le Pipeline.
    - ``transform()`` applique ``signe(x) × log(1 + |x|)`` aux colonnes retenues. Cette forme
      est monotone, vaut 0 en 0 et reste définie pour une valeur négative inattendue en
      production — contrairement à ``log1p`` seul, qui renverrait NaN sous −1.

    Monotone, la transformation ne change rien pour les modèles à arbres (mêmes coupures
    possibles). **Testée puis écartée** (§7.8.3) : elle dégrade la régression logistique, les
    valeurs extrêmes portant le signal de churn. Activable via
    ``construire_preprocesseur(..., log_asymetrique=True)``.

    Conçu pour s'insérer après l'imputation (entrée sans NaN, tableau numpy ou DataFrame).
    """

    def __init__(self, seuil_asymetrie: float = 1.0) -> None:
        self.seuil_asymetrie = seuil_asymetrie

    def fit(self, X: ArrayLike, y: object = None) -> LogAsymetrique:
        valeurs: np.ndarray = validate_data(self, X, reset=True, dtype=float)
        non_binaires = np.array(
            [np.unique(valeurs[:, j]).size > 2 for j in range(valeurs.shape[1])], dtype=bool
        )
        # Même estimateur que pandas.Series.skew (Fisher-Pearson ajusté), utilisé en §6.2.
        # Calculé sur les seules colonnes non binaires : une colonne constante n'a pas
        # d'asymétrie définie (et scipy avertirait d'une perte de précision)
        self.asymetrie_ = np.zeros(valeurs.shape[1])
        if non_binaires.any():
            self.asymetrie_[non_binaires] = np.nan_to_num(
                scipy_stats.skew(valeurs[:, non_binaires], axis=0, bias=False)
            )
        self.colonnes_log_ = (self.asymetrie_ > self.seuil_asymetrie) & non_binaires
        logger.debug(
            "LogAsymetrique.fit — {} colonne(s) sur {} transformée(s)",
            int(self.colonnes_log_.sum()),
            valeurs.shape[1],
        )
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        check_is_fitted(self, "colonnes_log_")
        # copy=True : l'entrée n'est jamais modifiée en place
        valeurs: np.ndarray = validate_data(self, X, reset=False, dtype=float, copy=True)
        sel = valeurs[:, self.colonnes_log_]
        valeurs[:, self.colonnes_log_] = np.sign(sel) * np.log1p(np.abs(sel))
        return valeurs
