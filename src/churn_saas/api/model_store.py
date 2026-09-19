"""Chargement et gestion du modèle sklearn en mémoire — singleton partagé par l'API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from churn_saas import config

_CHEMIN_MODELE: Path = config.ARTIFACTS / "models" / "best_model.pkl"
_CHEMIN_META: Path = config.ARTIFACTS / "models" / "best_model_meta.json"

# Seuil économique par défaut — issu de l'analyse §10 du notebook (contrainte capacité + ROI)
SEUIL_PAR_DEFAUT: float = 0.40


class ModelStore:
    """Gère le cycle de vie du pipeline sklearn chargé en mémoire.

    Utilisé comme dépendance FastAPI via ``get_model_store()``.
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
        logger.info("Modèle chargé depuis {}", chemin_modele)
        if chemin_meta.exists():
            with chemin_meta.open(encoding="utf-8") as fic:
                self._meta = json.load(fic)
        else:
            self._meta = {}

    @property
    def est_pret(self) -> bool:
        return self._modele is not None

    @property
    def seuil(self) -> float:
        return float(self._meta.get("seuil_economique", SEUIL_PAR_DEFAUT))

    @property
    def meta(self) -> dict[str, Any]:
        return dict(self._meta)

    def predire(self, X: pd.DataFrame) -> np.ndarray:
        """Retourne les probabilités predict_proba (shape N×2, colonne 1 = P(churn))."""
        if self._modele is None:
            raise RuntimeError("Aucun modèle chargé — appelez charger() d'abord.")
        return np.asarray(self._modele.predict_proba(X))


# Singleton module-level — chargé dans le lifespan FastAPI
_store = ModelStore()


def get_model_store() -> ModelStore:
    """Dépendance FastAPI retournant le singleton ModelStore."""
    return _store
