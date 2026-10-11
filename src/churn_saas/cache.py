import datetime
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from loguru import logger

from churn_saas import config


def recalcul_force() -> bool:
    """Vrai si ``FORCE_RECALC=1`` (positionnée par ``make notebook-full``) : cache ignoré."""
    return os.environ.get("FORCE_RECALC") == "1"


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
    ``FORCE_RECALC=1`` dans l'environnement équivaut à ``forcer=True`` pour tous les appels.
    """
    chemin = config.TABLES / nom
    forcer = forcer or recalcul_force()

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
    # Chemin relatif au dépôt pour un log lisible ; absolu si TABLES pointe ailleurs (tests)
    affiche = chemin.relative_to(config.RACINE) if chemin.is_relative_to(config.RACINE) else chemin
    logger.info("Artefact sauvegardé → {}", affiche)
    return resultat, date_prod


def charger(nom: str, defaut: Any = None) -> Any:
    """Relit un artefact produit par une autre section, sans jamais le calculer ni l'écrire.

    Pour les sections de synthèse (§1, §14) qui réutilisent un résultat de §9 ou §12.
    ``FORCE_RECALC`` est sans effet : forcer un « calcul » factice écraserait l'artefact.
    Renvoie ``defaut`` si l'artefact n'existe pas encore.
    """
    chemin = config.TABLES / nom
    if not chemin.exists():
        logger.warning("Artefact absent : {} — valeur par défaut utilisée", nom)
        return defaut
    return _charger(chemin)


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
        objet.to_parquet(chemin, index=True)
    elif suffixe == ".json":
        with chemin.open("w", encoding="utf-8") as fic:
            json.dump(objet, fic, ensure_ascii=False, indent=2)
    elif suffixe == ".joblib":
        joblib.dump(objet, chemin)
    else:
        raise ValueError(
            f"Extension non supportée : '{chemin.suffix}'. Utiliser .parquet, .json ou .joblib."
        )
