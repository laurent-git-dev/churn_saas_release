"""Contrôle qualité des sources de données brutes.

Usage autorisé : profil_compact() est le SEUL moyen d'inspecter les données
dans le projet. Ne jamais lire data/raw/*.csv directement.
"""

import csv
import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

# Encodages testés dans l'ordre — du plus strict au plus permissif
_ENCODAGES_CANDIDATS = ["utf-8-sig", "utf-8", "latin-1", "cp1252"]
# Taille du bloc d'échantillonnage pour la détection d'encodage et de séparateur
_BLOC_DETECTION = 32_768


def _detecter_encodage(chemin: Path) -> str:
    """Tente de décoder un bloc du fichier avec plusieurs encodages candidats."""
    with chemin.open("rb") as f:
        bloc = f.read(_BLOC_DETECTION)
    for enc in _ENCODAGES_CANDIDATS:
        try:
            bloc.decode(enc)
            return enc
        except (UnicodeDecodeError, LookupError):
            continue
    return "inconnu"


def _detecter_separateur(chemin: Path, encodage: str) -> str:
    """Utilise csv.Sniffer sur les premières lignes pour identifier le séparateur."""
    try:
        with chemin.open(encoding=encodage, errors="replace") as f:
            echantillon = "".join(f.readline() for _ in range(5))
        dialecte = csv.Sniffer().sniff(echantillon, delimiters=",;\t|")
        return repr(dialecte.delimiter)
    except csv.Error:
        return "inconnu"


def _compter_lignes(chemin: Path) -> int:
    """Compte le nombre de lignes (y compris l'en-tête)."""
    with chemin.open("rb") as f:
        return sum(1 for _ in f)


def _compter_colonnes(chemin: Path, encodage: str, separateur: str) -> int:
    """Lit uniquement la première ligne pour dénombrer les colonnes."""
    sep = separateur.strip("'")
    try:
        with chemin.open(encoding=encodage, errors="replace") as f:
            premiere = f.readline()
        return len(premiere.strip().split(sep))
    except Exception:
        return -1


def controler_source(chemin: str | Path) -> dict[str, Any]:
    """Contrôle métadonnées d'un fichier CSV : existence, taille, hash, encodage, etc.

    Parameters
    ----------
    chemin:
        Chemin vers le fichier (absolu ou relatif).

    Returns
    -------
    dict avec les clés : existe, taille_octets, sha256, encodage, separateur,
    nb_lignes (en-tête inclus), nb_colonnes, date_modification.
    """
    chemin = Path(chemin)
    if not chemin.exists():
        logger.warning("Fichier introuvable : {}", chemin)
        return {"existe": False, "chemin": str(chemin)}

    taille = chemin.stat().st_size
    date_modif = chemin.stat().st_mtime

    # Hash SHA-256 pour vérifier l'intégrité à la livraison
    sha = hashlib.sha256(chemin.read_bytes()).hexdigest()

    encodage = _detecter_encodage(chemin)
    separateur = _detecter_separateur(chemin, encodage)
    nb_lignes = _compter_lignes(chemin)
    nb_colonnes = _compter_colonnes(chemin, encodage, separateur)

    import datetime

    date_iso = datetime.datetime.fromtimestamp(date_modif).isoformat(timespec="seconds")

    logger.info(
        "Contrôle {} → {} lignes, {} colonnes, encodage={}, sep={}",
        chemin.name,
        nb_lignes,
        nb_colonnes,
        encodage,
        separateur,
    )

    return {
        "existe": True,
        "chemin": str(chemin),
        "taille_octets": taille,
        "sha256": sha,
        "encodage": encodage,
        "separateur": separateur,
        "nb_lignes": nb_lignes,
        "nb_colonnes": nb_colonnes,
        "date_modification": date_iso,
    }


def profil_compact(df: pd.DataFrame) -> pd.DataFrame:
    """Produit un tableau de synthèse : une ligne par colonne du DataFrame.

    Colonnes produites : dtype, nb_non_nuls, taux_manquants, cardinalite,
    top3_valeurs, min, max, numerique_en_texte.

    Le flag ``numerique_en_texte`` est True quand la colonne est de type object
    mais convertible en numérique à plus de 80 % des valeurs non nulles.
    """
    n = len(df)
    lignes = []

    for col in df.columns:
        serie = df[col]
        nb_non_nuls = int(serie.notna().sum())
        taux_manquants = round(1 - nb_non_nuls / n, 4) if n > 0 else float("nan")
        cardinalite = int(serie.nunique(dropna=True))

        top3 = serie.value_counts(dropna=True).head(3).index.tolist()
        top3_str = " | ".join(str(v) for v in top3)

        col_min = col_max = None
        flag_num_texte = False

        if pd.api.types.is_numeric_dtype(serie):
            col_min = serie.min()
            col_max = serie.max()
        elif pd.api.types.is_object_dtype(serie) or isinstance(serie.dtype, pd.StringDtype):
            non_nuls = serie.dropna()
            if len(non_nuls) > 0:
                converti = pd.to_numeric(non_nuls, errors="coerce")
                taux_conversion = converti.notna().mean()
                if taux_conversion > 0.80:
                    flag_num_texte = True
                    col_min = converti.min()
                    col_max = converti.max()

        lignes.append(
            {
                "colonne": col,
                "dtype": str(serie.dtype),
                "nb_non_nuls": nb_non_nuls,
                "taux_manquants": taux_manquants,
                "cardinalite": cardinalite,
                "top3_valeurs": top3_str,
                "min": col_min,
                "max": col_max,
                "numerique_en_texte": flag_num_texte,
            }
        )

    return pd.DataFrame(lignes).set_index("colonne")
