"""Chargeurs de données brutes.

Règle : charger_brut() force dtype=str pour préserver les formats d'origine —
aucune conversion implicite de pandas à ce stade.
"""

from pathlib import Path

import pandas as pd
from loguru import logger

from churn_saas import config


def charger_brut(nom: str) -> pd.DataFrame:
    """Charge un CSV depuis data/raw/ sans aucune conversion de type.

    Parameters
    ----------
    nom:
        Nom du fichier (avec ou sans extension .csv).

    Returns
    -------
    DataFrame dont toutes les colonnes sont de type str (object pandas).
    """
    if not nom.endswith(".csv"):
        nom = nom + ".csv"

    chemin = config.DONNEES_BRUTES / nom
    if not chemin.exists():
        raise FileNotFoundError(f"Fichier introuvable dans data/raw/ : {chemin}")

    df = pd.read_csv(chemin, dtype=str, keep_default_na=False)
    logger.info("Chargé {} → {} lignes × {} colonnes (dtype=str)", nom, len(df), len(df.columns))
    return df
