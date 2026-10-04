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

from churn_saas.format_fr import nombre, pourcentage

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
    mais convertible en numérique à plus de 80 % des valeurs renseignées, après la même
    réparation que ``coercer_numeriques()`` (virgule décimale, symboles, unités, espaces).
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
            # Convertibilité mesurée avec la même logique que coercer_numeriques() : un
            # pd.to_numeric brut échoue sur "12,5" ou "45 €" et écarterait justement les
            # colonnes qui ont le plus besoin de réparation
            coercitions = [_coercer_valeur(str(v)) for v in serie.dropna()]
            valeurs = [float(v) for v, _ in coercitions if v is not None]
            nb_renseignees = sum(1 for _, statut in coercitions if statut != "manquant")
            if nb_renseignees > 0 and len(valeurs) / nb_renseignees > 0.80:
                flag_num_texte = True
                col_min = min(valeurs)
                col_max = max(valeurs)

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
    # Unité de durée accolée au nombre ("12.5 h", "30j", "45 min") : l'unité est portée par le
    # nom de colonne (suffixe _h, _jours…), la valeur seule suffit
    v = re.sub(r"(?<=\d)\s*(?:h|j|min)$", "", v, flags=re.IGNORECASE)

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

# Formats essayés valeur par valeur, dans cet ordre. Les deux conventions à barre oblique
# (JJ/MM et MM/JJ) sont exclusives : une seule est retenue, par _convention_barre_oblique().
_FORMATS_DATE_CASCADE = [
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%d.%m.%Y",
    "%d %b %Y",
    "%d %B %Y",
    "%Y%m%d",
]
_FORMATS_BARRE_OBLIQUE = frozenset({"%d/%m/%Y", "%m/%d/%Y"})


def _convention_barre_oblique(non_vides: pd.Series) -> tuple[str, str, int]:
    """Tranche entre JJ/MM/AAAA et MM/JJ/AAAA pour les dates à barre oblique.

    Returns
    -------
    (format_strptime, description_convention, nb_ambigues). La description vaut ""
    quand la série ne contient aucune date à barre oblique.
    """
    candidats = non_vides[non_vides.str.match(r"^\d{1,2}/\d{1,2}/\d{4}$")]
    if len(candidats) == 0:
        return "%d/%m/%Y", "", 0

    parties = candidats.str.split("/", expand=True).iloc[:, :2].astype(int)
    premier_gt12 = int((parties[0] > 12).sum())
    deuxieme_gt12 = int((parties[1] > 12).sum())
    ambigues = int(((parties[0] <= 12) & (parties[1] <= 12)).sum())

    if premier_gt12 > 0 and deuxieme_gt12 == 0:
        return "%d/%m/%Y", "DD/MM/YYYY (premier champ > 12 observé → non ambigu)", ambigues
    if deuxieme_gt12 > 0 and premier_gt12 == 0:
        return "%m/%d/%Y", "MM/DD/YYYY (second champ > 12 observé → non ambigu)", ambigues
    # Indécidable sur ce critère → convention française par défaut
    return (
        "%d/%m/%Y",
        f"DD/MM/YYYY (convention par défaut — {ambigues} date(s) ambiguë(s), "
        "aucune valeur > 12 ne permet de trancher)",
        ambigues,
    )


def _parser_serie_dates(serie: pd.Series) -> tuple[pd.Series, dict[str, int], str, int]:
    """Parse une série de chaînes en essayant chaque format de la cascade valeur par valeur.

    Un format unique « dominant » laisserait en NaT toutes les dates écrites autrement :
    chaque valeur reçoit donc le premier format de la cascade qui la reconnaît.

    Returns
    -------
    (dates, compteurs_par_format, convention_barre_oblique, nb_ambigues)
    """
    texte = serie.dropna().astype(str).str.strip()
    texte = texte[texte != ""]
    fmt_barre, convention, nb_ambigues = _convention_barre_oblique(texte)
    formats = [
        f for f in _FORMATS_DATE_CASCADE if f not in _FORMATS_BARRE_OBLIQUE or f == fmt_barre
    ]

    dates = pd.Series(pd.NaT, index=serie.index, dtype="datetime64[ns]")
    compteurs: dict[str, int] = {}
    reste = texte
    for fmt in formats:
        if reste.empty:
            break
        essai = pd.to_datetime(reste, format=fmt, errors="coerce")
        reconnues = essai.notna()
        if reconnues.any():
            compteurs[fmt] = int(reconnues.sum())
            dates.loc[essai.index[reconnues]] = essai[reconnues]
        reste = reste[~reconnues]
    return dates, compteurs, convention, nb_ambigues


def parser_dates(df: pd.DataFrame, colonnes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse les colonnes de date, chaque valeur selon son propre format.

    Détecte et compte les dates ambiguës (ex. 01/02/2021 : JJ/MM ou MM/JJ ?),
    explicite la convention retenue. Ne pose pas dayfirst=True aveuglément.
    Une colonne déjà typée datetime est laissée telle quelle.

    Parameters
    ----------
    df:
        DataFrame source (non modifié en place).
    colonnes:
        Noms des colonnes à parser.

    Returns
    -------
    (df_avec_dates_parsées, rapport) indexé par colonne avec :
    format_detecte (format le plus fréquent), repartition_formats, convention_retenue,
    nb_parsees, nb_ambigues, nb_irrecuperables.
    """
    df = df.copy()
    lignes_rapport: list[dict[str, Any]] = []

    for col in colonnes:
        if col not in df.columns:
            logger.warning("Colonne absente : {}", col)
            continue

        compteurs: dict[str, int] = {}
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            parsed, convention, nb_ambigues = df[col], "déjà typée datetime", 0
        else:
            parsed, compteurs, convention, nb_ambigues = _parser_serie_dates(df[col])

        fmt_dominant = max(compteurs, key=lambda f: compteurs[f]) if compteurs else "%Y-%m-%d"
        repartition = " | ".join(f"{f} : {n}" for f, n in compteurs.items())
        nb_parsees = int(parsed.notna().sum())
        nb_renseignees = int((df[col].notna() & (df[col].astype(str).str.strip() != "")).sum())

        df[col] = parsed
        lignes_rapport.append(
            {
                "colonne": col,
                "format_detecte": fmt_dominant,
                "repartition_formats": repartition,
                "convention_retenue": convention or fmt_dominant,
                "nb_parsees": nb_parsees,
                "nb_ambigues": nb_ambigues,
                "nb_irrecuperables": nb_renseignees - nb_parsees,
            }
        )
        logger.info(
            "Dates {} — formats:{} parsées:{} ambiguës:{} irrécupérables:{}",
            col,
            repartition,
            nb_parsees,
            nb_ambigues,
            nb_renseignees - nb_parsees,
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

# Colonnes du jeu fourni structurellement positives : la liste explicite complète les préfixes
# génériques, qui en manquaient la plupart (ex. « revenue » ne couvre pas « revenu_… »)
_COLONNES_POSITIVES = frozenset(
    {
        "anciennete_mois",
        "sieges_souscrits",
        "utilisateurs_actifs",
        "connexions_30j",
        "heures_usage_30j",
        "fonctionnalites_total",
        "fonctionnalites_utilisees",
        "nb_integrations",
        "derniere_connexion_jours",
        "tickets_support_90j",
        "delai_reponse_support_h",
        "retards_paiement_12m",
        "revenu_mensuel_recurrent_eur",
        "valeur_vie_client_eur",
    }
)

# Bornes métier connues a priori (échelle de l'indicateur)
_BORNES: dict[str, tuple[float, float]] = {
    "taux_adoption_pct": (0.0, 100.0),
    "csat": (1.0, 5.0),
}

# Écart toléré entre ancienneté déclarée et recalculée. L'ancienneté est arrondie au mois
# entier : un écart de 1 mois est attendu, 3 mois laisse une marge sans masquer une vraie erreur
_TOLERANCE_ANCIENNETE_MOIS = 3


_JOURS_PAR_MOIS = 30.44


def _vers_numerique(serie: pd.Series) -> pd.Series:
    """Conversion tolérante (virgule décimale, symboles, unités) via _coercer_valeur().

    Un pd.to_numeric brut transformerait « 150,5 % » en NaN et la valeur impossible
    échapperait au contrôle.
    """
    if pd.api.types.is_numeric_dtype(serie):
        return serie
    return pd.Series(
        [np.nan if pd.isna(v) else float(_coercer_valeur(str(v))[0] or np.nan) for v in serie],
        index=serie.index,
        dtype=float,
    )


def _vers_dates(serie: pd.Series) -> pd.Series:
    """Dates parsées valeur par valeur (colonne brute) ou laissées telles quelles (datetime)."""
    if pd.api.types.is_datetime64_any_dtype(serie):
        return serie
    return _parser_serie_dates(serie)[0]


def estimer_date_reference(df: pd.DataFrame) -> pd.Timestamp | None:
    """Estime la date d'extraction à laquelle ``anciennete_mois`` a été calculée.

    Chaque ligne fournit une estimation : ``date_souscription + anciennete_mois``. La médiane
    est robuste aux lignes incohérentes, que le contrôle d'ancienneté doit justement isoler.
    Comparer à ``Timestamp.now()`` serait faux (l'ancienneté a été figée à l'extraction) et
    rendrait le résultat dépendant du jour de relance du notebook.

    Returns
    -------
    La date estimée, ou None si les colonnes sont absentes ou vides.
    """
    if "anciennete_mois" not in df.columns or "date_souscription" not in df.columns:
        return None
    anc = _vers_numerique(df["anciennete_mois"])
    dates = _vers_dates(df["date_souscription"])
    estimations = (dates + pd.to_timedelta(anc * _JOURS_PAR_MOIS, unit="D")).dropna()
    if estimations.empty:
        return None
    return pd.Timestamp(estimations.median()).normalize()


def detecter_valeurs_impossibles(
    df: pd.DataFrame,
    date_reference: pd.Timestamp | None = None,
    tolerance_anciennete_mois: int = _TOLERANCE_ANCIENNETE_MOIS,
) -> pd.DataFrame:
    """Détecte les incohérences métier dans le DataFrame.

    Règles appliquées (chacune seulement si ses colonnes sont présentes) :
    - bornes : taux_adoption_pct ∈ [0, 100], csat ∈ [1, 5], churn ∈ {0, 1} ;
    - positivité des colonnes structurellement positives (liste explicite + préfixes) ;
    - relations : utilisateurs_actifs ≤ sieges_souscrits,
      fonctionnalites_utilisees ≤ fonctionnalites_total,
      taux_adoption_pct = 100 × utilisateurs_actifs / sieges_souscrits (±1 point),
      connexions_30j > 0 ⇒ derniere_connexion_jours ≤ 30 ;
    - temporel : date_souscription ≤ date de référence, et anciennete_mois cohérente avec
      date_souscription (écart > ``tolerance_anciennete_mois``).

    Les colonnes brutes (texte) sont converties avec la même tolérance que
    coercer_numeriques() et parser_dates().

    Parameters
    ----------
    df:
        DataFrame à analyser.
    date_reference:
        Date à laquelle l'ancienneté a été calculée. Par défaut, estimée par
        estimer_date_reference().
    tolerance_anciennete_mois:
        Écart toléré entre ancienneté déclarée et ancienneté recalculée.

    Returns
    -------
    DataFrame des règles violées avec les colonnes : regle, colonne_ou_paire,
    nb_lignes_concernees, exemple. ``resultat.attrs["regles_controlees"]`` liste toutes les
    règles évaluées, violées ou non (dictionnaires regle / colonne_ou_paire / nb_lignes).
    """
    anomalies: list[dict[str, Any]] = []
    controlees: list[dict[str, Any]] = []
    num: dict[str, pd.Series] = {}

    def _n(col: str) -> pd.Series:
        if col not in num:
            num[col] = _vers_numerique(df[col])
        return num[col]

    def _presentes(*cols: str) -> bool:
        return set(cols) <= set(df.columns)

    def _ajouter(regle: str, cible: str, masque: pd.Series) -> None:
        nb = int(masque.sum())
        controlees.append({"regle": regle, "colonne_ou_paire": cible, "nb_lignes": nb})
        if nb > 0:
            anomalies.append(
                {
                    "regle": regle,
                    "colonne_ou_paire": cible,
                    "nb_lignes_concernees": nb,
                    "exemple": df[masque].head(3).to_dict(orient="records"),
                }
            )

    # --- Bornes ---------------------------------------------------------------
    for col, (bas, haut) in _BORNES.items():
        if _presentes(col):
            v = _n(col)
            _ajouter(f"{col} ∉ [{bas:g}, {haut:g}]", col, v.notna() & ((v < bas) | (v > haut)))

    if _presentes("churn"):
        v = _n("churn")
        _ajouter("churn ∉ {0, 1}", "churn", v.notna() & ~v.isin([0, 1]))

    # --- Positivité -----------------------------------------------------------
    for col in df.columns:
        nom = col.lower()
        if nom in _COLONNES_POSITIVES or any(nom.startswith(p) for p in _PREFIXES_POSITIFS):
            v = _n(col)
            _ajouter(f"{col} < 0 (valeur impossible)", col, v.notna() & (v < 0))

    # --- Relations entre colonnes ---------------------------------------------
    if _presentes("utilisateurs_actifs", "sieges_souscrits"):
        u, s = _n("utilisateurs_actifs"), _n("sieges_souscrits")
        _ajouter(
            "utilisateurs_actifs > sieges_souscrits",
            "utilisateurs_actifs / sieges_souscrits",
            u.notna() & s.notna() & (u > s),
        )

    if _presentes("fonctionnalites_utilisees", "fonctionnalites_total"):
        fu, ft = _n("fonctionnalites_utilisees"), _n("fonctionnalites_total")
        _ajouter(
            "fonctionnalites_utilisees > fonctionnalites_total",
            "fonctionnalites_utilisees / fonctionnalites_total",
            fu.notna() & ft.notna() & (fu > ft),
        )

    if _presentes("taux_adoption_pct", "utilisateurs_actifs", "sieges_souscrits"):
        s = _n("sieges_souscrits")
        attendu = 100 * _n("utilisateurs_actifs") / s.where(s > 0)
        ecart = (_n("taux_adoption_pct") - attendu).abs()
        _ajouter(
            "taux_adoption_pct ≠ 100 × utilisateurs_actifs / sieges_souscrits (écart > 1 point)",
            "taux_adoption_pct / utilisateurs_actifs / sieges_souscrits",
            ecart.notna() & (ecart > 1),
        )

    if _presentes("connexions_30j", "derniere_connexion_jours"):
        c, d = _n("connexions_30j"), _n("derniere_connexion_jours")
        _ajouter(
            "connexions_30j > 0 alors que derniere_connexion_jours > 30",
            "connexions_30j / derniere_connexion_jours",
            c.notna() & d.notna() & (c > 0) & (d > 30),
        )

    # --- Temporel -------------------------------------------------------------
    if date_reference is None:
        date_reference = estimer_date_reference(df)
    if date_reference is not None and _presentes("anciennete_mois", "date_souscription"):
        dates = _vers_dates(df["date_souscription"])
        _ajouter(
            "date_souscription postérieure à la date de référence",
            "date_souscription",
            dates.notna() & (dates > date_reference),
        )
        anc_calc = ((date_reference - dates).dt.days / _JOURS_PAR_MOIS).round(0)
        anc = _n("anciennete_mois")
        _ajouter(
            f"anciennete_mois incohérente avec date_souscription "
            f"(écart > {tolerance_anciennete_mois} mois)",
            "anciennete_mois / date_souscription",
            anc.notna() & anc_calc.notna() & ((anc - anc_calc).abs() > tolerance_anciennete_mois),
        )

    resultat = (
        pd.DataFrame(anomalies)
        if anomalies
        else pd.DataFrame(columns=["regle", "colonne_ou_paire", "nb_lignes_concernees", "exemple"])
    )
    resultat.attrs["regles_controlees"] = controlees
    return resultat


def diagnostiquer_anciennete(
    df: pd.DataFrame, date_reference: pd.Timestamp | None = None
) -> dict[str, Any]:
    """Expose les éléments qui justifient le contrôle d'ancienneté.

    Chaque ligne fournit une date d'extraction implicite (date_souscription + anciennete_mois).
    Si elles se regroupent dans une fenêtre d'environ un mois — l'effet attendu d'une
    ancienneté arrondie au mois entier —, les deux colonnes décrivent une même date
    d'extraction et la médiane est une référence fiable.

    Returns
    -------
    Dictionnaire : date_reference, implicite_min, implicite_max, etendue_jours, iqr_jours,
    repartition_ecarts (écart en mois → nombre de lignes), anciennete_entiere,
    date_souscription_max, tolerance_mois (tolérance par défaut du contrôle).
    """
    if date_reference is None:
        date_reference = estimer_date_reference(df)
    anc = _vers_numerique(df["anciennete_mois"])
    dates = _vers_dates(df["date_souscription"])
    implicites = (dates + pd.to_timedelta(anc * _JOURS_PAR_MOIS, unit="D")).dropna()
    ecarts = (anc - ((date_reference - dates).dt.days / _JOURS_PAR_MOIS).round(0)).dropna()
    return {
        "date_reference": date_reference,
        "implicite_min": implicites.min(),
        "implicite_max": implicites.max(),
        "etendue_jours": int((implicites.max() - implicites.min()).days),
        "iqr_jours": int((implicites.quantile(0.75) - implicites.quantile(0.25)).days),
        "repartition_ecarts": {
            int(k): int(v) for k, v in ecarts.value_counts().sort_index().items()
        },
        "anciennete_entiere": bool((anc.dropna() % 1 == 0).all()),
        "date_souscription_max": dates.max(),
        "tolerance_mois": _TOLERANCE_ANCIENNETE_MOIS,
    }


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


# Au-delà, une colonne catégorielle est exclue des associations : sur une colonne quasi unique
# (identifiant, texte libre, date brute), le V de Cramér vaut mécaniquement ≈ 1
_CARDINALITE_MAX_ASSOCIATION = 100
_TAUX_UNICITE_MAX_ASSOCIATION = 0.5


def _preparer_pour_associations(df: pd.DataFrame, cible: str) -> tuple[pd.DataFrame, list[str]]:
    """Type les colonnes pour la mesure d'association et écarte les colonnes quasi uniques.

    Une colonne numérique stockée en texte traitée comme catégorielle aurait des milliers de
    modalités : son V de Cramér avec n'importe quel indicateur binaire serait proche de 1 et
    ferait conclure à tort à un mécanisme MAR.

    Returns
    -------
    (df_type, colonnes_exclues)
    """
    colonnes: dict[str, pd.Series] = {}
    exclues: list[str] = []
    for col in df.columns:
        serie = df[col]
        if col == cible or pd.api.types.is_numeric_dtype(serie):
            colonnes[col] = serie
            continue
        non_nuls = serie.dropna()
        numerique = _vers_numerique(non_nuls)
        if len(non_nuls) > 0 and numerique.notna().mean() > 0.80:
            colonnes[col] = _vers_numerique(serie)
            continue
        cardinalite = non_nuls.nunique()
        if cardinalite > _CARDINALITE_MAX_ASSOCIATION or (
            len(non_nuls) > 0 and cardinalite / len(non_nuls) > _TAUX_UNICITE_MAX_ASSOCIATION
        ):
            exclues.append(col)
            continue
        colonnes[col] = serie
    return pd.DataFrame(colonnes, index=df.index), exclues


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

    Les associations sont mesurées sur des colonnes typées (numériques stockées en texte
    converties) ; les colonnes quasi uniques (identifiants, texte libre, dates brutes) en sont
    exclues et listées dans ``rapport.attrs["colonnes_exclues_associations"]``.

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
    df_assoc, exclues = _preparer_pour_associations(df, cible)
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
        autres_cols = [c for c in df_assoc.columns if c != col and c != cible]
        associations: dict[str, float] = {
            c: _force_association_binaire(masque_manquant, df_assoc[c]) for c in autres_cols
        }
        top3 = sorted(associations.items(), key=lambda kv: kv[1], reverse=True)[:3]
        top3_str = ", ".join(f"{c}={nombre(v, 3)}" for c, v in top3 if v > 0)
        assoc_max = max(associations.values(), default=0.0)

        # --- Classification MCAR / MAR / MNAR ---------------------------------
        if lien_sig:
            mecanisme = "MNAR"
            confiance = "élevée" if p_cible < 0.01 else "modérée"
            raisonnement = (
                f"La manquance est corrélée à la cible (χ² Yates p={nombre(p_cible, 4)} < 0,05). "
                f"Taux de churn : {pourcentage(taux_churn_manquant, 1)} (manquant) vs "
                f"{pourcentage(taux_churn_renseigne, 1)} (renseigné). "
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
                f"La manquance n'est pas corrélée à la cible (p={nombre(p_cible, 4)}) mais présente "
                f"une association de {nombre(assoc_max, 3)} avec '{col_max}'. "
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
                f"Ni la cible (p={nombre(p_cible, 4)}) ni les autres colonnes "
                f"(association max={nombre(assoc_max, 3)}) n'expliquent la manquance. "
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
    if exclues:
        logger.info("Manquance — colonnes quasi uniques exclues des associations : {}", exclues)
    rapport = (
        pd.DataFrame(lignes).set_index("colonne")
        if lignes
        else pd.DataFrame(columns=colonnes_sortie).set_index("colonne")
    )
    rapport.attrs["colonnes_exclues_associations"] = exclues
    return rapport


# ---------------------------------------------------------------------------
# Proposition de nommage normalisé
# ---------------------------------------------------------------------------

# Ensembles de tokens (séparés par _ dans les noms snake_case) → suffixe normalisé.
# On divise le nom par _ puis on vérifie l'intersection avec l'ensemble de mots-clés.
_REGLES_SUFFIXES_UNITES: list[tuple[frozenset[str], str]] = [
    (frozenset({"euro", "euros", "eur", "prix", "montant", "valeur", "ca", "chiffre"}), "_eur"),
    (frozenset({"pourcent", "pct", "percent", "taux"}), "_pct"),
    (frozenset({"mois", "month"}), "_mois"),
    # Pluriels seulement : « jour » / « day » au singulier désignent une date ou un jour de la
    # semaine (« jour_souscription »), pas une durée
    (frozenset({"jours", "days", "duree", "delai"}), "_jours"),
]
# Suffixes qui portent déjà l'unité de la colonne (``_h`` : heures)
_SUFFIXES_UNITES_RECONNUS: tuple[str, ...] = ("_eur", "_pct", "_mois", "_jours", "_h")


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
    - Unités suffixées (_eur, _pct, _mois, _jours) si détectées dans le nom original et si
      le nom ne porte pas déjà une unité (_h compris).

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
            # Un nom déjà suffixé par une unité (« delai_…_h ») est conforme : ajouter
            # « _jours » à une durée en heures fausserait son unité
            if nouveau.endswith(_SUFFIXES_UNITES_RECONNUS):
                break
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
