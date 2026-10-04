"""Chargement et gestion du modèle sklearn en mémoire — singleton partagé par l'API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from churn_saas import config, economie
from churn_saas.models.explication_locale import FacteurLocal, scorer_et_expliquer

_CHEMIN_MODELE: Path = config.ARTIFACTS / "models" / "best_model.pkl"
_CHEMIN_META: Path = config.ARTIFACTS / "models" / "best_model_meta.json"


class ModelStore:
    """Gère le cycle de vie du pipeline sklearn chargé en mémoire.

    Utilisé comme dépendance FastAPI via ``get_model_store()``. Agnostique de la famille :
    le champion courant (régression logistique recalibrée) comme un challenger LightGBM promu
    par le flow de réentraînement exposent le même ``predict_proba`` sur les mêmes colonnes
    brutes, les features dérivées étant recalculées côté serveur.
    """

    def __init__(self) -> None:
        self._modele: Any | None = None
        self._meta: dict[str, Any] = {}

    def charger(
        self,
        chemin_modele: Path = _CHEMIN_MODELE,
        chemin_meta: Path = _CHEMIN_META,
    ) -> None:
        """Charge le pipeline sklearn et ses métadonnées depuis le disque."""
        if not chemin_modele.exists():
            logger.warning(
                "Aucun modèle trouvé à {} — /ready retournera 503. "
                "Exécutez `churn-saas train` pour entraîner.",
                chemin_modele,
            )
            return
        self._modele = joblib.load(chemin_modele)
        if chemin_meta.exists():
            with chemin_meta.open(encoding="utf-8") as fic:
                self._meta = json.load(fic)
        else:
            self._meta = {}
        logger.info(
            "Modèle chargé depuis {} — famille : {}, seuil de surveillance : {:.4f}",
            chemin_modele,
            self.nom_modele,
            self.seuil,
        )

    @property
    def est_pret(self) -> bool:
        return self._modele is not None

    @property
    def seuil(self) -> float:
        """Seuil de surveillance : celui des métadonnées du modèle promu, sinon le seuil de
        vigilance de la règle à deux niveaux (``economie.seuil_surveillance_par_defaut``)."""
        if "seuil_economique" in self._meta:
            return float(self._meta["seuil_economique"])
        return economie.seuil_surveillance_par_defaut()

    @property
    def nom_modele(self) -> str | None:
        """Famille du modèle servi : ``modele_nom`` des métadonnées, sinon la classe de
        l'estimateur final du pipeline ; ``None`` si aucun modèle n'est chargé."""
        if self._modele is None:
            return None
        if "modele_nom" in self._meta:
            return str(self._meta["modele_nom"])
        etapes = getattr(self._modele, "steps", None)
        estimateur = etapes[-1][1] if etapes else self._modele
        return type(estimateur).__name__

    @property
    def meta(self) -> dict[str, Any]:
        return dict(self._meta)

    def predire(self, X: pd.DataFrame) -> np.ndarray:
        """Retourne les probabilités predict_proba (shape N×2, colonne 1 = P(churn))."""
        if self._modele is None:
            raise RuntimeError("Aucun modèle chargé — appelez charger() d'abord.")
        return np.asarray(self._modele.predict_proba(X))

    def predire_et_expliquer(
        self, X: pd.DataFrame, nb_facteurs: int = 3
    ) -> tuple[np.ndarray, list[list[FacteurLocal]]]:
        """Probabilités (comme `predire`) et facteurs SHAP de chaque compte, en un seul passage
        du prétraitement (`models.explication_locale.scorer_et_expliquer`).

        L'explication est secondaire au score : si elle échoue, chaque compte reçoit une liste
        vide de facteurs plutôt que de refuser une prédiction valide.
        """
        if self._modele is None:
            raise RuntimeError("Aucun modèle chargé — appelez charger() d'abord.")
        return scorer_et_expliquer(self._modele, X, nb_facteurs)


# Singleton module-level — chargé dans le lifespan FastAPI
_store = ModelStore()


def get_model_store() -> ModelStore:
    """Dépendance FastAPI retournant le singleton ModelStore."""
    return _store
