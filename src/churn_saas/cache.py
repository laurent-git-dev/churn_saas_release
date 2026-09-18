import datetime
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from loguru import logger

from churn_saas import config


def charger_ou_calculer(
    nom: str,
    fonction: Callable[[], Any],
    forcer: bool = False,
) -> tuple[Any, datetime.datetime]:
    """Charge depuis reports/tables/<nom> ou calcule et met en cache.

    Formats supportés selon l'extension :
    - ``.parquet`` → ``pandas.DataFrame``
    - ``.json``    → ``dict | list``
    - ``.joblib``  → objet Python quelconque (modèle scikit-learn, etc.)

    Retourne ``(résultat, date_de_production)``.
    Si l'artefact existe et ``forcer=False``, le résultat est chargé depuis le disque.
    Sinon, ``fonction()`` est exécutée, son résultat écrit, puis retourné.
    """
    chemin = config.TABLES / nom

    if not forcer and chemin.exists():
        date_prod = _date_modification(chemin)
        logger.info(
            "Cache trouvé : {} (produit le {})",
            nom,
            date_prod.strftime("%Y-%m-%d %H:%M"),
        )
        return _charger(chemin), date_prod

    logger.info("Calcul de {} …", nom)
    resultat: Any = fonction()
    config.TABLES.mkdir(parents=True, exist_ok=True)
    _sauvegarder(chemin, resultat)
    date_prod = _date_modification(chemin)
    logger.info("Artefact sauvegardé → {}", chemin.relative_to(config.RACINE))
    return resultat, date_prod


# ---------------------------------------------------------------------------
# Fonctions internes
# ---------------------------------------------------------------------------


def _date_modification(chemin: Path) -> datetime.datetime:
    return datetime.datetime.fromtimestamp(chemin.stat().st_mtime)


def _charger(chemin: Path) -> Any:
    suffixe = chemin.suffix.lower()
    if suffixe == ".parquet":
        return pd.read_parquet(chemin)
    if suffixe == ".json":
        with chemin.open(encoding="utf-8") as fic:
            return json.load(fic)
    if suffixe == ".joblib":
        return joblib.load(chemin)
    raise ValueError(
        f"Extension non supportée : '{chemin.suffix}'. Utiliser .parquet, .json ou .joblib."
    )


def _sauvegarder(chemin: Path, objet: Any) -> None:
    suffixe = chemin.suffix.lower()
    if suffixe == ".parquet":
        if not isinstance(objet, pd.DataFrame):
            raise TypeError(
                f"Extension .parquet requiert un DataFrame pandas, reçu : {type(objet).__name__}."
            )
        objet.to_parquet(chemin, index=False)
    elif suffixe == ".json":
        with chemin.open("w", encoding="utf-8") as fic:
            json.dump(objet, fic, ensure_ascii=False, indent=2)
    elif suffixe == ".joblib":
        joblib.dump(objet, chemin)
    else:
        raise ValueError(
            f"Extension non supportée : '{chemin.suffix}'. Utiliser .parquet, .json ou .joblib."
        )
