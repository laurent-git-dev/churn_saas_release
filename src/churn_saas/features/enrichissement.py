"""Enrichissement externe simulé — référentiels sectoriels et pays.

⚠️  SIMULATION DOCUMENTÉE
Les données chiffrées de ce module sont des ordres de grandeur plausibles tirés
de rapports publics (Gainsight, OpenView Partners, Zuora, Gartner, BPI France,
Banque mondiale, RGPD, CCPA).  Elles NE PROVIENNENT PAS d'une API en production
et NE DOIVENT PAS être utilisées à des fins de décision commerciale réelle.

En production, chaque table serait remplacée par son équivalent opérationnel décrit
dans FICHES_SOURCES.  Toute variable issue de ce module doit être tracée comme
« enrichissement simulé » dans les métadonnées du dataset gold.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# Fiches de gouvernance des sources externes
# ---------------------------------------------------------------------------

FICHES_SOURCES: dict[str, dict[str, str]] = {
    "referentiel_sectoriel": {
        "nom": "Référentiel de benchmarks SaaS par secteur",
        "ce_qu_il_apporte": (
            "Taux de churn médian du secteur et dynamique de croissance.  "
            "Permet de situer chaque client par rapport à la norme de son marché : "
            "un client dont le taux d'adoption est faible dans un secteur en croissance "
            "présente un risque de churn supérieur au signal brut."
        ),
        "source_production": (
            "Gainsight 'State of Customer Success' (annuel, US/EU) ; "
            "OpenView Partners SaaS Benchmarks (annuel) ; "
            "Zuora Subscription Economy Index (trimestriel) — "
            "fusionnés par code NAF/NACE reclassé en macro-secteur."
        ),
        "licence": (
            "Rapports publics Gainsight et OpenView (accès libre avec inscription) ; "
            "Zuora Index distribué sous embargo commercial — "
            "nécessite un accord de diffusion interne."
        ),
        "fraicheur": "Annuelle — révision recommandée chaque septembre pour l'exercice N+1.",
        "cout": "Accès aux rapports : gratuit (après inscription).  Intégration ETL : ~2 jours-homme.",
        "plan_b": (
            "Si indisponible : utiliser le taux de churn global SaaS B2B médian "
            "(≈ 5–7 % annuel, source OpenView 2023) comme constante homogène pour tous les "
            "secteurs.  La feature ecart_au_churn_sectoriel devient alors moins discriminante "
            "mais reste calculable.  Alternative : données BPI France / FrenchTech pour les "
            "secteurs à dominante française."
        ),
    },
    "referentiel_pays": {
        "nom": "Référentiel réglementaire et support par pays",
        "ce_qu_il_apporte": (
            "Zone réglementaire (RGPD-UE, CCPA-US, autre) et disponibilité du support "
            "en langue locale.  Capte le surcoût de conformité pour les clients UE "
            "(DPO, registre de traitement) et le risque de déperdition lié à une barrière "
            "linguistique au support."
        ),
        "source_production": (
            "Classification UE : liste officielle des États membres EUR-Lex ; "
            "CCPA : liste CPRA (Californie) + États ayant adopté des lois similaires "
            "(Virginie, Colorado…) — mise à jour par l'IAPP (International Association "
            "of Privacy Professionals) ; "
            "langues de support : inventaire interne du centre de support client."
        ),
        "licence": (
            "Classification UE : données publiques EUR-Lex (licence ouverte CC-BY 4.0) ; "
            "IAPP : abonnement professionnel (~1 500 $/an) ou extraction manuelle publique."
        ),
        "fraicheur": (
            "Zones réglementaires : semi-annuelle (nouvelle législation possible).  "
            "Langues de support : trimestrielle (évolution des effectifs CS)."
        ),
        "cout": "EUR-Lex : gratuit.  IAPP : ~1 500 $/an.  Mise en œuvre ETL : ~1 jour-homme.",
        "plan_b": (
            "Si indisponible : imputer 'reglementaire_autre' pour les pays hors UE/US.  "
            "La variable langue_support_disponible peut être dérivée d'un dictionnaire "
            "pays → langue principale (ISO 639-1) croisé avec les langues du support "
            "documentées en interne, sans dépendance externe."
        ),
    },
}

# ---------------------------------------------------------------------------
# Référentiel sectoriel — simulation
# ---------------------------------------------------------------------------
# Secteurs alignés sur la nomenclature probable du jeu de données (colonne `secteur`).
# Taux de churn annuels : ordres de grandeur Gainsight 2023 / OpenView 2023.
# Dynamique de croissance : Gartner Magic Quadrant et BPI France 2023.

_DONNEES_SECTEUR = [
    # secteur, taux_churn_median_annuel_pct, dynamique_croissance
    ("Technologie", 4.5, "forte"),
    ("Finance", 6.0, "moderee"),
    ("Santé", 5.0, "forte"),
    ("Industrie", 7.5, "stable"),
    ("Services", 6.5, "moderee"),
    ("Commerce", 8.0, "moderee"),
    ("Éducation", 5.5, "forte"),
    ("Immobilier", 9.0, "stable"),
    ("Télécommunications", 10.0, "stable"),
    ("Médias", 8.5, "contraction"),
    ("Logistique", 7.0, "moderee"),
    ("Énergie", 6.0, "forte"),
    # Valeur de repli pour tout secteur non listé
    ("_repli_", 7.0, "moderee"),
]

REFERENTIEL_SECTORIEL: pd.DataFrame = pd.DataFrame(
    _DONNEES_SECTEUR,
    columns=["secteur", "taux_churn_median_saas_pct", "dynamique_croissance"],
).set_index("secteur")

# Libellés des données brutes qui désignent un secteur du référentiel sous un autre nom.
# « public » n'a pas d'équivalent dans le référentiel : il reste sur le repli (documenté en §7)
ALIAS_SECTEURS: dict[str, str] = {"tech": "Technologie"}

# ---------------------------------------------------------------------------
# Référentiel pays — simulation
# ---------------------------------------------------------------------------
# zone_reglementaire : 'ue_rgpd' | 'us_ccpa' | 'autre'
# langue_support_fr : True si le support est disponible en français
# decalage_horaire_paris_h : décalage moyen avec Paris (proxy de friction support)

_DONNEES_PAYS = [
    # pays, zone_reglementaire, langue_support_fr, decalage_horaire_paris_h
    ("France", "ue_rgpd", True, 0),
    ("Allemagne", "ue_rgpd", False, 0),
    ("Espagne", "ue_rgpd", False, 0),
    ("Italie", "ue_rgpd", False, 0),
    ("Pays-Bas", "ue_rgpd", False, 0),
    ("Belgique", "ue_rgpd", True, 0),
    ("Portugal", "ue_rgpd", False, 0),
    ("Suède", "ue_rgpd", False, 0),
    ("Pologne", "ue_rgpd", False, 0),
    ("Autriche", "ue_rgpd", False, 0),
    ("Danemark", "ue_rgpd", False, 0),
    ("Finlande", "ue_rgpd", False, 0),
    ("Suisse", "hors_ue", True, 0),
    ("Royaume-Uni", "hors_ue", False, 0),
    ("États-Unis", "us_ccpa", False, -6),
    ("Canada", "hors_ue", True, -6),
    ("Australie", "hors_ue", False, 9),
    ("Japon", "hors_ue", False, 8),
    ("Brésil", "hors_ue", False, -4),
    ("Inde", "hors_ue", False, 5),
    # Repli pour tout pays non listé
    ("_repli_", "autre", False, 0),
]

REFERENTIEL_PAYS: pd.DataFrame = pd.DataFrame(
    _DONNEES_PAYS,
    columns=["pays", "zone_reglementaire", "langue_support_fr", "decalage_horaire_paris_h"],
).set_index("pays")

# ---------------------------------------------------------------------------
# Fonctions d'enrichissement
# ---------------------------------------------------------------------------


def _aligner_sur_referentiel(
    serie: pd.Series, index_ref: pd.Index, alias: dict[str, str] | None = None
) -> pd.Series:
    """Ramène chaque valeur à la clé du référentiel, sans tenir compte de la casse ni des espaces.

    Les données brutes mélangent « Santé », « SANTÉ » et «  santé » : une jointure exacte
    enverrait ces variantes sur le repli. ``alias`` rattache une abréviation à sa clé
    (« tech » → « Technologie »). Les valeurs sans correspondance reçoivent ``_repli_``.
    """
    correspondance = {str(k).strip().casefold(): k for k in index_ref}
    correspondance |= {cle.casefold(): cible for cle, cible in (alias or {}).items()}
    cles = serie.astype("string").str.strip().str.casefold().map(correspondance)
    return cles.astype(object).where(cles.notna(), "_repli_")


def enrichir_par_secteur(df: pd.DataFrame) -> pd.DataFrame:
    """Joint le référentiel sectoriel et ajoute les features dérivées.

    Features créées :
    - ``taux_churn_median_saas_pct``  : taux de churn médian du secteur (simulation).
    - ``dynamique_croissance``        : dynamique de croissance du secteur (catégorielle).
    - ``ecart_adoption_churn_secteur``: taux_adoption_pct moins le taux_adoption moyen
      du secteur simulé (proxy de sous-performance sectorielle).  NaN si taux_adoption_pct
      est absent ou si le secteur est inconnu.

    ⚠️ ANTI-FUITE : ces features sont issues d'un référentiel externe statique, PAS calculées
    à partir des observations du jeu d'entraînement.  Elles peuvent donc être jointes AVANT
    le split train/test sans introduire de fuite temporelle.

    Parameters
    ----------
    df:
        DataFrame contenant au minimum la colonne ``secteur``.

    Returns
    -------
    DataFrame enrichi (copie).
    """
    df = df.copy()

    if "secteur" not in df.columns:
        logger.warning("enrichir_par_secteur — colonne 'secteur' absente, enrichissement ignoré")
        return df

    # Join vectorisé : les secteurs inconnus reçoivent le repli global
    sect_norm = _aligner_sur_referentiel(
        df["secteur"], REFERENTIEL_SECTORIEL.index, alias=ALIAS_SECTEURS
    )
    joined = REFERENTIEL_SECTORIEL.loc[sect_norm.values]

    for col in ["taux_churn_median_saas_pct", "dynamique_croissance"]:
        df[col] = joined[col].values

    # Feature dérivée : écart entre adoption du client et adoption attendue dans son secteur
    # (dans ce référentiel simulé, on utilise le taux_churn comme proxy inverse de l'adoption)
    if "taux_adoption_pct" in df.columns:
        taux_adoption = pd.to_numeric(df["taux_adoption_pct"], errors="coerce")
        # Adoption espérée = 100 - taux_churn_median (approximation grossière mais documentée)
        adoption_ref = 100.0 - pd.to_numeric(df["taux_churn_median_saas_pct"], errors="coerce")
        df["ecart_adoption_secteur"] = taux_adoption - adoption_ref

    n_repli = int((sect_norm == "_repli_").sum())
    logger.info(
        "enrichir_par_secteur — {} lignes enrichies, {} sur repli (_repli_) [simulation]",
        len(df),
        n_repli,
    )
    return df


def enrichir_par_pays(df: pd.DataFrame) -> pd.DataFrame:
    """Joint le référentiel pays et ajoute les features réglementaires.

    Features créées :
    - ``zone_reglementaire``        : 'ue_rgpd' | 'us_ccpa' | 'hors_ue' | 'autre'.
    - ``langue_support_fr``         : booléen — support disponible en français.
    - ``decalage_horaire_paris_h``  : décalage horaire moyen avec Paris (proxy de friction).

    ⚠️ ANTI-FUITE : même raisonnement que ``enrichir_par_secteur``.
    Ces variables proviennent d'un référentiel statique, sans aucune statistique apprise
    sur les observations — elles peuvent précéder le split.

    Parameters
    ----------
    df:
        DataFrame contenant au minimum la colonne ``pays``.

    Returns
    -------
    DataFrame enrichi (copie).
    """
    df = df.copy()

    if "pays" not in df.columns:
        logger.warning("enrichir_par_pays — colonne 'pays' absente, enrichissement ignoré")
        return df

    pays_norm = _aligner_sur_referentiel(df["pays"], REFERENTIEL_PAYS.index)
    joined = REFERENTIEL_PAYS.loc[pays_norm.values]

    for col in ["zone_reglementaire", "langue_support_fr", "decalage_horaire_paris_h"]:
        df[col] = joined[col].values

    n_repli = int((pays_norm == "_repli_").sum())
    logger.info(
        "enrichir_par_pays — {} lignes enrichies, {} sur repli (_repli_) [simulation]",
        len(df),
        n_repli,
    )
    return df
