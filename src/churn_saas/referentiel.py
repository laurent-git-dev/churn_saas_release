"""Référentiel de compétences C1 → C9 et traçabilité item par item dans le notebook.

Les intitulés sont repris mot pour mot du référentiel de compétences de l'énoncé. Le
référentiel ne numérote pas ses items : la numérotation ``Cn.k`` suit l'ordre de ses tableaux.

:data:`CORRESPONDANCE` associe à chaque item son intitulé et le titre (sans numéro) de la
sous-section du notebook qui en apporte la preuve : les sections ne portent aucune balise.
:func:`localiser_items` croise ces titres cibles avec les titres réels des sources jupytext pour
générer les grilles de couverture (§0, §15.4) ; ``tests/test_tracabilite.py`` vérifie qu'aucun
item n'est orphelin (titre cible introuvable).
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from loguru import logger

from churn_saas import config

COMPETENCES: dict[str, str] = {
    "C1": "Identifier un jeu de données répondant aux besoins métiers",
    "C2": "Identifier les risques éthiques et sociétaux",
    "C3": "Préparer les données",
    "C4": "Choisir un modèle IA",
    "C5": "Entraîner le modèle",
    "C6": "Implémenter le modèle",
    "C7": "Architecture cible",
    "C8": "Mesurer la performance et les impacts",
    "C9": "Amélioration continue",
}

CORRESPONDANCE: dict[str, tuple[str, str]] = {
    "C1.1": (
        "Les besoins métiers sont correctement identifiés",
        "Contexte de l'éditeur SaaS B2B",
    ),
    "C1.2": (
        "Les cas d'usage sont correctement décrits",
        "Trois cas d'usage opérationnels",
    ),
    "C1.3": (
        "Les données pertinentes (et nécessaires a minima) sont identifiées",
        "Dictionnaire de données — extrait avec pertinence",
    ),
    "C1.4": (
        "L'existence, la disponibilité et l'accès des données sont vérifiés",
        "Inventaire et contrôle des sources",
    ),
    "C1.5": (
        "Des solutions alternatives sont envisagées en cas d'indisponibilité",
        "Enrichissements souhaités — tableau disponibilité et plan B",
    ),
    "C2.1": (
        "Les chartes éthiques européennes et françaises sont connues et appliquées",
        "Chartes éthiques — volet européen et volet français",
    ),
    "C2.2": (
        "Les impacts éthiques et sociétaux sont connus et leurs conséquences comprises",
        "RGPD — Analyse de conformité",
    ),
    "C2.3": (
        "Les biais potentiels ou existants sont identifiés",
        "Mesure des biais par sous-groupe",
    ),
    "C2.4": (
        "Les dilemmes éthiques sont identifiés",
        "Dilemmes éthiques identifiés et arbitrages",
    ),
    "C2.5": (
        "Les risques sont portés à la connaissance des acteurs concernés",
        "Note de synthèse au commanditaire",
    ),
    "C2.6": (
        (
            "La vérification par les acteurs concernés des problèmes légaux et éthiques du jeu de "
            "données est faite"
        ),
        "Fiche de revue DPO / juriste",
    ),
    "C3.1": (
        "Les données sont correctement nommées ou renommées",
        "Renommage — convention de nommage",
    ),
    "C3.2": (
        "Le format des données est adapté à l'usage",
        "Nettoyage",
    ),
    "C3.3": (
        "Les données altérées, inexactes ou non pertinentes sont corrigées ou supprimées",
        "Nettoyage",
    ),
    "C3.4": (
        "Les traitements effectués sont correctement documentés",
        "Synthèse — tableau de bord qualité",
    ),
    "C3.5": (
        "Le choix du modèle de stockage est adapté",
        "Choix du modèle de stockage — note d'arbitrage",
    ),
    "C3.6": (
        "Le cycle de vie du jeu de données est documenté",
        "Cycle de vie du jeu de données",
    ),
    "C3.7": (
        "Le cycle de vie documenté est soumis aux parties prenantes",
        "Trace de soumission aux parties prenantes",
    ),
    "C4.1": (
        "La pertinence est évaluée grâce aux bons indicateurs (analyse ROC)",
        "Métriques techniques de classification",
    ),
    "C4.2": (
        "Les contraintes opérationnelles sont prises en compte",
        "Exigences d'exploitation : volumétrie, fréquence, intégration, compétences",
    ),
    "C4.3": (
        "Les contraintes d'éco-conception sont portées à la connaissance des acteurs",
        "Leviers de sobriété et note d'arbitrage carbone",
    ),
    "C4.4": (
        "Les grandes familles d'algorithmes sont connues",
        "Familles candidates pour des données tabulaires",
    ),
    "C4.5": (
        "La démarche scientifique est correctement documentée",
        "Protocole de comparaison",
    ),
    "C4.6": (
        (
            "La performance attendue est déterminée (précision, temps de traitement et d'inférence, "
            "énergie)"
        ),
        "Cibles de performance a priori",
    ),
    "C4.7": (
        "Le type de résultat attendu est identifié (probabiliste/déterministe)",
        "Deux cibles, deux tâches : classification probabiliste et régression",
    ),
    "C4.8": (
        "Le contexte des cas d'usage est pris en compte",
        "Adéquation du modèle aux trois cas d'usage",
    ),
    "C4.9": (
        "Le modèle d'apprentissage choisi est cohérent avec les résultats attendus",
        "Deux cibles, deux tâches : classification probabiliste et régression",
    ),
    "C4.10": (
        "La pertinence des solutions sur l'étagère est évaluée",
        "Analyse build vs buy",
    ),
    "C5.1": (
        "Le modèle est optimisé suivant le contexte",
        "Optimisation des hyperparamètres (Optuna)",
    ),
    "C5.2": (
        "Le modèle créé est entraîné",
        "Tableau comparatif des modèles",
    ),
    "C5.3": (
        "Le modèle choisi est réentraîné le cas échéant",
        "Modèle de production — réapprentissage sur toutes les données",
    ),
    "C5.4": (
        "Les connaissances sont transférées d'un modèle à l'autre le cas échéant",
        "Transfert de connaissances",
    ),
    "C5.5": (
        "Les hyperparamètres sont décrits",
        "Optimisation des hyperparamètres (Optuna)",
    ),
    "C5.6": (
        "Le feature engineering est effectué",
        "Feature engineering — ratios métier",
    ),
    "C6.1": (
        "Le processus de livraison et de déploiement continu est mis en œuvre",
        "Intégration et déploiement continus — `.github/workflows/ci.yml`",
    ),
    "C6.2": (
        "Le versioning est implémenté",
        "Versioning sur quatre axes",
    ),
    "C6.3": (
        "Les besoins d'intégration sont documentés",
        "Besoins d'intégration — contrat d'échange avec le CRM",
    ),
    "C7.1": (
        "Les principales architectures et leurs contraintes sont connues",
        "Trois scénarios comparatifs",
    ),
    "C7.2": (
        "Les contraintes économiques des scénarios sont portées à la connaissance des acteurs",
        "Recommandation argumentée",
    ),
    "C7.3": (
        "Les acteurs sont interrogés pour préciser les contraintes de généralisation",
        "Compte-rendu d'entretien avec les acteurs",
    ),
    "C8.1": (
        "Des indicateurs de performance et seuils associés sont définis",
        "Cibles de performance a priori",
    ),
    "C8.2": (
        "La performance est mesurée grâce au suivi des indicateurs",
        "Métriques techniques de classification",
    ),
    "C8.3": (
        "Les résultats sont interprétés et présentés aux interlocuteurs concernés",
        "Note de restitution au commanditaire",
    ),
    "C8.4": (
        "Les actions adaptées sont déclenchées en fonction des indicateurs",
        "Plan de réentraînement — déclencheurs et retour arrière",
    ),
    "C9.1": (
        "Système d'évaluation automatisé et intégré au CI/CD via les pratiques MLOps",
        "Évaluation automatisée dans la chaîne d'intégration continue",
    ),
    "C9.2": (
        (
            "Les métriques sont intégrées (taux de prévision, robustesse, variations de performance, "
            "obsolescence)"
        ),
        "Dérive des entrées — PSI et test de Kolmogorov-Smirnov",
    ),
    "C9.3": (
        (
            "La pertinence des indicateurs est interrogée selon une périodicité définie en phase de "
            "cadrage"
        ),
        "Périodicité de revue — comité trimestriel",
    ),
}

ITEMS: dict[str, str] = {code: intitule for code, (intitule, _) in CORRESPONDANCE.items()}

# Titre de sous-section (niveaux 2 à 4) : « # ### 2.2 Trois cas d'usage »
_MOTIF_TITRE = re.compile(r"^# (#{2,4}) (.+)$")
# Numéro en tête de titre (« 7.3.0 ») et mention de compétence en fin (« (C3) »)
_MOTIF_NUMERO = re.compile(r"^\d+(?:\.\d+)*\.?\s+")
_MOTIF_COMPETENCE = re.compile(r"\s*\(C\d(?:\.\d+)?\)\s*$")


def titre_sans_numero(titre: str) -> str:
    """Retire le numéro de tête et la mention « (Cn) » finale d'un titre de sous-section."""
    return _MOTIF_COMPETENCE.sub("", _MOTIF_NUMERO.sub("", titre.strip())).strip()


def relever_titres(dossier: Path = config.SECTIONS_NOTEBOOK) -> pd.DataFrame:
    """Collecte les titres de sous-section (niveaux 2 à 4) des sections jupytext.

    Returns
    -------
    pd.DataFrame
        Une ligne par titre : ``titre`` (tel qu'écrit, numéro compris), ``cle`` (titre sans
        numéro, clé de jointure avec :data:`CORRESPONDANCE`) et ``fichier``.
    """
    lignes: list[dict[str, str]] = []
    for fichier in sorted(dossier.glob("[0-9][0-9]_*.py")):
        for ligne in fichier.read_text(encoding="utf-8").splitlines():
            if m := _MOTIF_TITRE.match(ligne):
                titre = m.group(2).strip()
                lignes.append(
                    {"titre": titre, "cle": titre_sans_numero(titre), "fichier": fichier.name}
                )
    return pd.DataFrame(lignes, columns=["titre", "cle", "fichier"])


def localiser_items(dossier: Path = config.SECTIONS_NOTEBOOK) -> pd.DataFrame:
    """Croise :data:`CORRESPONDANCE` avec les titres réels des sections.

    Returns
    -------
    pd.DataFrame
        Une ligne par item : ``code``, ``intitule``, ``titre_cible`` (sans numéro), ``titre``
        (titre réel, numéro compris) et ``fichier``. ``titre`` et ``fichier`` sont vides pour un
        item orphelin, dont le titre cible n'existe dans aucune section.
    """
    titres = relever_titres(dossier).drop_duplicates("cle").set_index("cle")
    lignes = [
        {
            "code": code,
            "intitule": intitule,
            "titre_cible": cible,
            "titre": str(titres.at[cible, "titre"]) if cible in titres.index else "",
            "fichier": str(titres.at[cible, "fichier"]) if cible in titres.index else "",
        }
        for code, (intitule, cible) in CORRESPONDANCE.items()
    ]
    return pd.DataFrame(lignes)


def items_orphelins(dossier: Path = config.SECTIONS_NOTEBOOK) -> pd.DataFrame:
    """Items dont le titre cible est introuvable dans les sections (``code``, ``titre_cible``)."""
    localisation = localiser_items(dossier)
    orphelins = localisation.loc[localisation["titre"] == "", ["code", "titre_cible"]]
    if not orphelins.empty:
        logger.warning("{} item(s) orphelin(s) : {}", len(orphelins), list(orphelins["code"]))
    return orphelins.reset_index(drop=True)


def numero_sous_section(titre: str) -> str:
    """Extrait le numéro « §N.M » d'un titre (« 12.2 Métriques… » → « §12.2 »)."""
    m = re.match(r"(\d+(?:\.\d+)*)", titre)
    return f"§{m.group(1)}" if m else titre


def grille_items(localisation: pd.DataFrame | None = None) -> pd.DataFrame:
    """Matrice item × sous-section de preuve, dans l'ordre du référentiel."""
    if localisation is None:
        localisation = localiser_items()
    preuves = localisation.set_index("code")["titre"].map(
        lambda t: numero_sous_section(t) if t else "—"
    )
    return pd.DataFrame(
        {
            "Compétence": [code.split(".")[0] for code in ITEMS],
            "Item": list(ITEMS),
            "Intitulé de l'item": list(ITEMS.values()),
            "Sous-section(s) de preuve": [preuves.get(code, "—") for code in ITEMS],
        }
    )


def grille_competences(localisation: pd.DataFrame | None = None) -> pd.DataFrame:
    """Synthèse par compétence : items, items couverts, items orphelins, sections concernées."""
    if localisation is None:
        localisation = localiser_items()
    localisation = localisation.assign(
        competence=localisation["code"].str.split(".").str[0],
        section=localisation["titre"].map(lambda t: numero_sous_section(t).split(".")[0]),
    )
    lignes = []
    for comp, intitule in COMPETENCES.items():
        sous = localisation[localisation["competence"] == comp]
        couverts = sous[sous["titre"] != ""]
        orphelins = list(sous.loc[sous["titre"] == "", "code"])
        sections = sorted(set(couverts["section"]), key=lambda s: int(s.lstrip("§")))
        lignes.append(
            {
                "Compétence": comp,
                "Intitulé": intitule,
                "Items": len(sous),
                "Items couverts": len(couverts),
                "Items orphelins": ", ".join(orphelins) or "—",
                "Section(s)": ", ".join(sections),
            }
        )
    return pd.DataFrame(lignes)
