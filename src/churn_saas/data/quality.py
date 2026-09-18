"""Contrôle qualité des sources de données brutes.

Usage autorisé : profil_compact() est le SEUL moyen d'inspecter les données
dans le projet. Ne jamais lire data/raw/*.csv directement.
"""

import csv
import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

# Encodages testés dans l'ordre — du plus strict au plus permissif
_ENCODAGES_CANDIDATS = ["utf-8-sig", "utf-8", "latin-1", "cp1252"]
# Taille du bloc d'échantillonnage pour la détection d'encodage et de séparateur
_BLOC_DETECTION = 32_768

# Marqueurs textuels de valeur manquante (comparés après strip().lower())
_MARQUEURS_MANQUANTS = frozenset(
    {
        "",
        "n/a",
        "na",
        "nan",
        "null",
        "none",
        "#n/a",
        "-",
        "nd",
        "nr",
        "inconnu",
    }
)


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


# ---------------------------------------------------------------------------
# Analyse des doublons
# ---------------------------------------------------------------------------


def analyser_doublons(df: pd.DataFrame, cle_metier: str = "client_id") -> dict[str, Any]:
    """Distingue les doublons exacts (toutes colonnes) des doublons sur la clé métier.

    Les deux cas ont un traitement différent :
    - Doublons exacts : suppression sécurisée (lignes strictement redondantes).
    - Doublons sur clé métier seulement : investigation obligatoire avant décision.

    Parameters
    ----------
    df:
        DataFrame à analyser.
    cle_metier:
        Nom de la colonne identifiant un client de manière unique.

    Returns
    -------
    dict avec les clés : nb_total_lignes, nb_doublons_exacts, exemples_doublons_exacts,
    traitement_doublons_exacts, nb_doublons_cle_metier, nb_doublons_cle_metier_non_exacts,
    exemples_doublons_cle_metier, traitement_doublons_cle_metier.
    """
    n = len(df)

    masque_exact = df.duplicated(keep=False)
    nb_doublons_exacts = int(masque_exact.sum())
    exemples_exact = df[masque_exact].head(10) if nb_doublons_exacts > 0 else df.iloc[0:0]

    if cle_metier in df.columns:
        masque_cle = df.duplicated(subset=[cle_metier], keep=False)
        nb_doublons_cle = int(masque_cle.sum())
        exemples_cle = df[masque_cle].head(10) if nb_doublons_cle > 0 else df.iloc[0:0]
        nb_cle_non_exact = int((masque_cle & ~masque_exact).sum())
    else:
        logger.warning("Clé métier '{}' absente du DataFrame", cle_metier)
        nb_doublons_cle = 0
        nb_cle_non_exact = 0
        exemples_cle = df.iloc[0:0]

    logger.info(
        "Doublons — exacts : {}, clé métier : {} (dont {} non-exacts)",
        nb_doublons_exacts,
        nb_doublons_cle,
        nb_cle_non_exact,
    )

    return {
        "nb_total_lignes": n,
        "nb_doublons_exacts": nb_doublons_exacts,
        "exemples_doublons_exacts": exemples_exact,
        "traitement_doublons_exacts": (
            "Suppression sécurisée par drop_duplicates(keep='first') : lignes strictement "
            "identiques sur toutes les colonnes, donc redondantes."
        ),
        "nb_doublons_cle_metier": nb_doublons_cle,
        "nb_doublons_cle_metier_non_exacts": nb_cle_non_exact,
        "exemples_doublons_cle_metier": exemples_cle,
        "traitement_doublons_cle_metier": (
            "Investigation métier obligatoire : même client_id avec données différentes peut "
            "indiquer des mises à jour historiques (garder la plus récente), plusieurs contacts "
            "par compte (agréger), ou une erreur de jointure. Ne pas supprimer sans vérification."
        ),
    }


# ---------------------------------------------------------------------------
# Coercition des numériques stockés en texte
# ---------------------------------------------------------------------------


def _coercer_valeur(valeur: str) -> tuple[str | None, str]:
    """Transforme une chaîne en représentation numérique parsable.

    Returns
    -------
    (valeur_nettoyee_ou_None, statut) où statut ∈ {'direct', 'repare', 'manquant', 'irrecuperable'}.
    Retourne (None, 'manquant') pour les marqueurs de valeur manquante.
    Retourne (None, 'irrecuperable') si la valeur ne peut pas être convertie.
    """
    stripped = valeur.strip()

    if stripped.lower() in _MARQUEURS_MANQUANTS:
        return None, "manquant"

    try:
        float(stripped)
        return stripped, "direct"
    except ValueError:
        pass

    v = stripped
    # Espaces insécables (NBSP \xa0, NNBSP  )
    v = v.replace("\xa0", "").replace(" ", "")
    # Symboles monétaires et pourcentage
    v = re.sub(r"[€$£%]", "", v).strip()

    if "," in v and "." in v:
        # Les deux séparateurs présents : le dernier est le décimal
        if v.rfind(".") > v.rfind(","):
            # "1,234.56" → anglais : virgule = milliers
            v = v.replace(",", "")
        else:
            # "1.234,56" → français : point = milliers, virgule = décimale
            v = v.replace(".", "").replace(",", ".")
    elif "," in v:
        parts = v.split(",")
        # Heuristique : "1,234" (une virgule, 3 chiffres après) → séparateur de milliers
        # sinon "1,5" ou "12,50" → décimale française
        if (
            len(parts) == 2
            and re.fullmatch(r"\d{1,3}", parts[0])
            and re.fullmatch(r"\d{3}", parts[1])
        ):
            v = v.replace(",", "")  # milliers EN ("1,234" → 1234)
        else:
            v = v.replace(",", ".")  # décimale FR ("1,5" → 1.5)

    # Espaces résiduels entre chiffres → séparateurs de milliers ("1 234")
    v = re.sub(r"(\d)\s+(\d)", r"\1\2", v)

    try:
        float(v)
        return v, "repare"
    except ValueError:
        return None, "irrecuperable"


def coercer_numeriques(df: pd.DataFrame, colonnes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convertit les colonnes numériques stockées en texte.

    Gère : séparateurs décimaux virgule/point, séparateurs de milliers,
    symboles (%, €), espaces insécables, marqueurs textuels de manquant
    ("N/A", "n/a", "-", "", "null"…).

    Parameters
    ----------
    df:
        DataFrame source (non modifié en place).
    colonnes:
        Liste des colonnes à convertir.

    Returns
    -------
    (df_converti, rapport) où rapport est un DataFrame indexé par colonne avec
    les compteurs : nb_directes, nb_reparees, nb_manquants, nb_irrecuperables.
    """
    df = df.copy()
    lignes_rapport: list[dict[str, Any]] = []

    for col in colonnes:
        if col not in df.columns:
            logger.warning("Colonne absente : {}", col)
            continue

        resultats: list[tuple[float, str]] = []

        for val in df[col]:
            if pd.isna(val):
                resultats.append((float("nan"), "manquant"))
            else:
                val_nettoyee, statut = _coercer_valeur(str(val))
                resultats.append(
                    (float(val_nettoyee) if val_nettoyee is not None else float("nan"), statut)
                )

        df[col] = pd.to_numeric([r[0] for r in resultats], errors="coerce")
        statuts = [r[1] for r in resultats]

        lignes_rapport.append(
            {
                "colonne": col,
                "nb_directes": statuts.count("direct"),
                "nb_reparees": statuts.count("repare"),
                "nb_manquants": statuts.count("manquant"),
                "nb_irrecuperables": statuts.count("irrecuperable"),
            }
        )
        logger.info(
            "Coercition {} — directes:{} réparées:{} manquants:{} irrécupérables:{}",
            col,
            statuts.count("direct"),
            statuts.count("repare"),
            statuts.count("manquant"),
            statuts.count("irrecuperable"),
        )

    rapport = (
        pd.DataFrame(lignes_rapport).set_index("colonne") if lignes_rapport else pd.DataFrame()
    )
    return df, rapport


# ---------------------------------------------------------------------------
# Parsing multi-formats des dates
# ---------------------------------------------------------------------------

_FORMATS_DATE_CASCADE = [
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%d.%m.%Y",
    "%Y%m%d",
]


def _detecter_format_date(serie: pd.Series) -> tuple[str, str, int]:
    """Détecte le format dominant d'une série de chaînes de dates.

    Returns
    -------
    (format_strptime, description_convention, nb_ambigues)
    """
    non_vides = serie.dropna()
    non_vides = non_vides[non_vides.astype(str).str.strip() != ""]
    if len(non_vides) == 0:
        return "%Y-%m-%d", "aucune valeur non nulle", 0

    compteurs = {
        fmt: int(pd.to_datetime(non_vides, format=fmt, errors="coerce").notna().sum())
        for fmt in _FORMATS_DATE_CASCADE
    }
    meilleur_fmt = max(compteurs, key=lambda f: compteurs[f])

    nb_ambigues = 0
    convention = meilleur_fmt

    if meilleur_fmt in ("%d/%m/%Y", "%m/%d/%Y"):
        # Analyser les valeurs DD/MM ou MM/DD pour lever l'ambiguïté
        candidats = non_vides[non_vides.str.match(r"^\d{1,2}/\d{1,2}/\d{4}$")]
        premier_gt12 = deuxieme_gt12 = ambigues = 0
        for v in candidats:
            parties = str(v).split("/")
            try:
                p1, p2 = int(parties[0]), int(parties[1])
            except (ValueError, IndexError):
                continue
            if p1 > 12:
                premier_gt12 += 1
            if p2 > 12:
                deuxieme_gt12 += 1
            if p1 <= 12 and p2 <= 12:
                ambigues += 1

        nb_ambigues = ambigues

        if premier_gt12 > 0 and deuxieme_gt12 == 0:
            meilleur_fmt = "%d/%m/%Y"
            convention = "DD/MM/YYYY (premier champ > 12 observé → non ambigu)"
        elif deuxieme_gt12 > 0 and premier_gt12 == 0:
            meilleur_fmt = "%m/%d/%Y"
            convention = "MM/DD/YYYY (second champ > 12 observé → non ambigu)"
        else:
            # Indécidable sur ce critère → convention française par défaut
            meilleur_fmt = "%d/%m/%Y"
            convention = (
                f"DD/MM/YYYY (convention par défaut — {ambigues} date(s) ambiguë(s), "
                "aucune valeur > 12 ne permet de trancher)"
            )

    return meilleur_fmt, convention, nb_ambigues


def parser_dates(df: pd.DataFrame, colonnes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse les colonnes de date en cascade de formats.

    Détecte et compte les dates ambiguës (ex. 01/02/2021 : JJ/MM ou MM/JJ ?),
    explicite la convention retenue. Ne pose pas dayfirst=True aveuglément.

    Parameters
    ----------
    df:
        DataFrame source (non modifié en place).
    colonnes:
        Noms des colonnes à parser.

    Returns
    -------
    (df_avec_dates_parsées, rapport) indexé par colonne avec :
    format_detecte, convention_retenue, nb_parsees, nb_ambigues, nb_irrecuperables.
    """
    df = df.copy()
    lignes_rapport: list[dict[str, Any]] = []

    for col in colonnes:
        if col not in df.columns:
            logger.warning("Colonne absente : {}", col)
            continue

        fmt, convention, nb_ambigues = _detecter_format_date(df[col])
        parsed = pd.to_datetime(df[col], format=fmt, errors="coerce")
        nb_parsees = int(parsed.notna().sum())
        nb_irrecup = int(df[col].notna().sum()) - nb_parsees

        df[col] = parsed
        lignes_rapport.append(
            {
                "colonne": col,
                "format_detecte": fmt,
                "convention_retenue": convention,
                "nb_parsees": nb_parsees,
                "nb_ambigues": nb_ambigues,
                "nb_irrecuperables": nb_irrecup,
            }
        )
        logger.info(
            "Dates {} — format:{} parsées:{} ambiguës:{} irrécupérables:{}",
            col,
            fmt,
            nb_parsees,
            nb_ambigues,
            nb_irrecup,
        )

    rapport = (
        pd.DataFrame(lignes_rapport).set_index("colonne") if lignes_rapport else pd.DataFrame()
    )
    return df, rapport


# ---------------------------------------------------------------------------
# Détection des valeurs métier impossibles
# ---------------------------------------------------------------------------

# Préfixes de colonnes structurellement positives (nom en minuscules)
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


def detecter_valeurs_impossibles(df: pd.DataFrame) -> pd.DataFrame:
    """Détecte les incohérences métier dans le DataFrame.

    Règles appliquées :
    - utilisateurs_actifs > sieges_souscrits
    - taux_adoption_pct ∉ [0, 100]
    - valeurs négatives dans les colonnes structurellement positives
    - incohérence entre anciennete_mois et date_souscription (écart > 3 mois)

    Parameters
    ----------
    df:
        DataFrame à analyser.

    Returns
    -------
    DataFrame d'anomalies avec les colonnes : regle, colonne_ou_paire,
    nb_lignes_concernees, exemple.
    """
    anomalies: list[dict[str, Any]] = []

    def _ajouter(regle: str, cible: str, masque: pd.Series) -> None:
        nb = int(masque.sum())
        if nb > 0:
            anomalies.append(
                {
                    "regle": regle,
                    "colonne_ou_paire": cible,
                    "nb_lignes_concernees": nb,
                    "exemple": df[masque].head(3).to_dict(orient="records"),
                }
            )

    # Règle 1 : utilisateurs_actifs > sieges_souscrits
    if "utilisateurs_actifs" in df.columns and "sieges_souscrits" in df.columns:
        u = pd.to_numeric(df["utilisateurs_actifs"], errors="coerce")
        s = pd.to_numeric(df["sieges_souscrits"], errors="coerce")
        _ajouter(
            "utilisateurs_actifs > sieges_souscrits",
            "utilisateurs_actifs / sieges_souscrits",
            u.notna() & s.notna() & (u > s),
        )

    # Règle 2 : taux_adoption_pct hors [0, 100]
    if "taux_adoption_pct" in df.columns:
        col = pd.to_numeric(df["taux_adoption_pct"], errors="coerce")
        _ajouter(
            "taux_adoption_pct ∉ [0, 100]",
            "taux_adoption_pct",
            col.notna() & ((col < 0) | (col > 100)),
        )

    # Règle 3 : valeurs négatives absurdes
    for col in df.columns:
        nom = col.lower()
        if any(nom.startswith(pfx) for pfx in _PREFIXES_POSITIFS):
            numerique = pd.to_numeric(df[col], errors="coerce")
            _ajouter(f"{col} < 0 (valeur impossible)", col, numerique.notna() & (numerique < 0))

    # Règle 4 : incohérence anciennete_mois / date_souscription (tolérance ±3 mois)
    if "anciennete_mois" in df.columns and "date_souscription" in df.columns:
        anc = pd.to_numeric(df["anciennete_mois"], errors="coerce")
        dates = pd.to_datetime(df["date_souscription"], errors="coerce")
        anc_calc = ((pd.Timestamp.now() - dates).dt.days / 30.44).round(0)
        _ajouter(
            "anciennete_mois incohérente avec date_souscription (écart > 3 mois)",
            "anciennete_mois / date_souscription",
            anc.notna() & anc_calc.notna() & ((anc - anc_calc).abs() > 3),
        )

    if not anomalies:
        return pd.DataFrame(
            columns=["regle", "colonne_ou_paire", "nb_lignes_concernees", "exemple"]
        )
    return pd.DataFrame(anomalies)


# ---------------------------------------------------------------------------
# Analyse des mécanismes de manquance
# ---------------------------------------------------------------------------


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    """V de Cramér entre deux séries catégorielles (après dropna commun)."""
    mask = x.notna() & y.notna()
    x, y = x[mask].astype(str), y[mask].astype(str)
    if len(x) < 5 or x.nunique() < 2 or y.nunique() < 2:
        return 0.0
    table = pd.crosstab(x, y)
    chi2, _, _, _ = stats.chi2_contingency(table, correction=False)
    n = table.values.sum()
    r, k = table.shape
    denom = n * (min(r, k) - 1)
    return float(np.sqrt(chi2 / denom)) if denom > 0 else 0.0


def _force_association_binaire(masque_manquant: pd.Series, autre: pd.Series) -> float:
    """Association entre un indicateur binaire de manquance et une autre colonne.

    Corrélation point-bisériale pour les numériques, V de Cramér pour les catégorielles.
    Retourne 0.0 si le calcul est impossible (trop peu de valeurs, variance nulle…).
    """
    valide = autre.notna()
    x = masque_manquant[valide].astype(int)
    y = autre[valide]
    if x.nunique() < 2 or len(x) < 10:
        return 0.0
    if pd.api.types.is_numeric_dtype(y):
        try:
            r, _ = stats.pointbiserialr(x, y)
            return float(abs(r)) if not np.isnan(r) else 0.0
        except Exception:
            return 0.0
    try:
        return _cramers_v(masque_manquant[valide].map({True: "manquant", False: "renseigne"}), y)
    except Exception:
        return 0.0


def analyser_manquance(df: pd.DataFrame, cible: str) -> pd.DataFrame:
    """Qualifie le mécanisme de manquance pour chaque colonne présentant des NA.

    Pour chaque colonne avec valeurs manquantes :
    - taux de manquants
    - lien avec la cible (χ² sur table de contingence manquant × cible, avec p-value)
    - lien avec les 3 autres colonnes les plus associées
    - proposition de mécanisme MCAR / MAR / MNAR avec niveau de confiance et raisonnement
    - recommandation de traitement

    Hypothèse testée : un ``csat`` manquant est probablement MNAR (les insatisfaits ne
    répondent pas → l'indicateur de manquance est lui-même une feature prédictive).
    Cette hypothèse est vérifiée par le code, pas affirmée a priori.

    Parameters
    ----------
    df:
        DataFrame source.
    cible:
        Nom de la colonne cible binaire (0/1 ou bool).

    Returns
    -------
    DataFrame indexé par colonne (une ligne par colonne avec manquants) avec les colonnes :
    taux_manquants, p_value_cible, taux_churn_si_manquant, taux_churn_si_renseigne,
    lien_cible_significatif, top3_associations, mecanisme_propose, confiance,
    raisonnement, recommandation.

    Raises
    ------
    ValueError
        Si ``cible`` est absente du DataFrame.
    """
    if cible not in df.columns:
        raise ValueError(f"Colonne cible '{cible}' absente du DataFrame")

    cible_num = pd.to_numeric(df[cible], errors="coerce")
    _SEUIL_ASSOCIATION = 0.15  # V de Cramér ou |r| au-delà duquel on parle de MAR

    lignes: list[dict[str, Any]] = []

    for col in df.columns:
        if col == cible:
            continue

        masque_manquant = df[col].isna()
        taux_manquants = masque_manquant.mean()
        if taux_manquants == 0.0:
            continue

        # --- Lien avec la cible -----------------------------------------------
        table = pd.crosstab(masque_manquant, cible_num.fillna(0).astype(int))
        if table.shape[0] == 2 and table.shape[1] >= 2 and table.values.sum() > 0:
            _, p_cible, _, _ = stats.chi2_contingency(table, correction=True)
        else:
            p_cible = float("nan")

        taux_churn_manquant = cible_num[masque_manquant].mean()
        taux_churn_renseigne = cible_num[~masque_manquant].mean()
        lien_sig = not np.isnan(p_cible) and p_cible < 0.05

        # --- Lien avec les autres colonnes ------------------------------------
        autres_cols = [c for c in df.columns if c != col and c != cible]
        associations: dict[str, float] = {
            c: _force_association_binaire(masque_manquant, df[c]) for c in autres_cols
        }
        top3 = sorted(associations.items(), key=lambda kv: kv[1], reverse=True)[:3]
        top3_str = ", ".join(f"{c}={v:.3f}" for c, v in top3 if v > 0)
        assoc_max = max(associations.values(), default=0.0)

        # --- Classification MCAR / MAR / MNAR ---------------------------------
        if lien_sig:
            mecanisme = "MNAR"
            confiance = "élevée" if p_cible < 0.01 else "modérée"
            raisonnement = (
                f"La manquance est corrélée à la cible (χ² Yates p={p_cible:.4f} < 0.05). "
                f"Taux de churn : {taux_churn_manquant:.1%} (manquant) vs "
                f"{taux_churn_renseigne:.1%} (renseigné). "
                "La valeur manquante n'est pas aléatoire par rapport à la variable d'intérêt — "
                "c'est la signature d'un mécanisme MNAR."
            )
            recommandation = (
                "Créer un indicateur binaire de manquance (feature prédictive à inclure dans le "
                "pipeline), puis imputer par la médiane ou le mode pour les algorithmes "
                "intolérants aux NA."
            )
        elif assoc_max >= _SEUIL_ASSOCIATION:
            mecanisme = "MAR"
            col_max = max(associations, key=associations.get)  # type: ignore[arg-type]
            confiance = (
                "élevée" if assoc_max > 0.3 else ("modérée" if assoc_max > 0.2 else "faible")
            )
            raisonnement = (
                f"La manquance n'est pas corrélée à la cible (p={p_cible:.4f}) mais présente "
                f"une association de {assoc_max:.3f} avec '{col_max}'. "
                "La valeur est manquante de façon conditionnelle à des variables observées — "
                "mécanisme MAR."
            )
            recommandation = (
                "Imputation conditionnelle (MICE ou KNNImputer conditionné aux variables "
                "associées). Ajouter un indicateur de manquance si le taux dépasse 10%."
            )
        else:
            mecanisme = "MCAR"
            confiance = "faible" if taux_manquants > 0.3 else "modérée"
            raisonnement = (
                f"Ni la cible (p={p_cible:.4f}) ni les autres colonnes "
                f"(association max={assoc_max:.3f}) n'expliquent la manquance. "
                "Hypothèse MCAR retenue par défaut — MCAR est difficile à prouver, "
                "seulement à ne pas réfuter."
            )
            recommandation = (
                "Imputation simple : moyenne pour les variables continues, "
                "mode pour les catégorielles. "
                "Supprimer la colonne si taux_manquants > 50%."
            )

        logger.info(
            "Manquance {} — taux:{:.1%} mécanisme:{} confiance:{} (p_cible={:.4f})",
            col,
            taux_manquants,
            mecanisme,
            confiance,
            p_cible if not np.isnan(p_cible) else -1.0,
        )

        lignes.append(
            {
                "colonne": col,
                "taux_manquants": round(float(taux_manquants), 4),
                "p_value_cible": (
                    round(float(p_cible), 4) if not np.isnan(p_cible) else float("nan")
                ),
                "taux_churn_si_manquant": (
                    round(float(taux_churn_manquant), 4)
                    if not np.isnan(taux_churn_manquant)
                    else float("nan")
                ),
                "taux_churn_si_renseigne": (
                    round(float(taux_churn_renseigne), 4)
                    if not np.isnan(taux_churn_renseigne)
                    else float("nan")
                ),
                "lien_cible_significatif": bool(lien_sig),
                "top3_associations": top3_str,
                "mecanisme_propose": mecanisme,
                "confiance": confiance,
                "raisonnement": raisonnement,
                "recommandation": recommandation,
            }
        )

    colonnes_sortie = [
        "colonne",
        "taux_manquants",
        "p_value_cible",
        "taux_churn_si_manquant",
        "taux_churn_si_renseigne",
        "lien_cible_significatif",
        "top3_associations",
        "mecanisme_propose",
        "confiance",
        "raisonnement",
        "recommandation",
    ]
    if not lignes:
        return pd.DataFrame(columns=colonnes_sortie).set_index("colonne")
    return pd.DataFrame(lignes).set_index("colonne")


# ---------------------------------------------------------------------------
# Proposition de nommage normalisé
# ---------------------------------------------------------------------------

# Ensembles de tokens (séparés par _ dans les noms snake_case) → suffixe normalisé.
# On divise le nom par _ puis on vérifie l'intersection avec l'ensemble de mots-clés.
_REGLES_SUFFIXES_UNITES: list[tuple[frozenset[str], str]] = [
    (frozenset({"euro", "euros", "eur", "prix", "montant", "valeur", "ca", "chiffre"}), "_eur"),
    (frozenset({"pourcent", "pct", "percent", "taux"}), "_pct"),
    (frozenset({"mois", "month"}), "_mois"),
    (frozenset({"jour", "jours", "day", "days", "duree", "delai"}), "_jours"),
]


def _normaliser_nom(nom: str) -> str:
    """Convertit un nom de colonne en snake_case sans accent ni caractère spécial."""
    sans_accent = "".join(
        c for c in unicodedata.normalize("NFD", nom) if unicodedata.category(c) != "Mn"
    )
    s = sans_accent.lower()
    s = re.sub(r"[\s\-\.]+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "", s)
    return re.sub(r"_+", "_", s).strip("_")


def proposer_renommage(df: pd.DataFrame) -> pd.DataFrame:
    """Propose une convention de nommage snake_case sans accent avec unités suffixées.

    Critères :
    - Minuscules, suppression des accents, caractères spéciaux → _.
    - Unités suffixées (_eur, _pct, _mois, _jours) si détectées dans le nom original.

    Parameters
    ----------
    df:
        DataFrame dont on examine les noms de colonnes.

    Returns
    -------
    DataFrame avec les colonnes : nom_original, nom_propose, modifie, raison.
    """
    lignes: list[dict[str, Any]] = []

    for col in df.columns:
        nouveau = _normaliser_nom(col)
        raisons: list[str] = []

        if col != col.lower():
            raisons.append("mise en minuscules")
        nfd = unicodedata.normalize("NFD", col)
        if any(unicodedata.category(c) == "Mn" for c in nfd):
            raisons.append("suppression des accents")
        if re.search(r"[\s\-\.]", col):
            raisons.append("remplacement des séparateurs par _")
        if re.search(r"[^a-zA-Z0-9_\s\-\.]", col):
            raisons.append("suppression des caractères spéciaux")

        # Ajout du suffixe d'unité s'il est absent
        # On tokenise le nom normalisé par _ pour éviter les faux positifs en snake_case
        tokens = set(re.split(r"[_\s\-]+", col.lower()))
        for mots_cle, suffixe in _REGLES_SUFFIXES_UNITES:
            if tokens & mots_cle and not nouveau.endswith(suffixe.lstrip("_")):
                nouveau = nouveau + suffixe
                raisons.append(f"ajout suffixe unité {suffixe}")
                break

        lignes.append(
            {
                "nom_original": col,
                "nom_propose": nouveau,
                "modifie": col != nouveau,
                "raison": ", ".join(raisons) if raisons else "conforme",
            }
        )

    return pd.DataFrame(lignes)
