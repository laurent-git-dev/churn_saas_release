"""Schémas d'architecture du §11, tracés en matplotlib.

Jupyter n'affiche pas les blocs Mermaid d'une cellule markdown : les schémas sont donc dessinés
comme des figures ordinaires, numérotées et sauvegardées par :mod:`churn_saas.viz` (réutilisées
telles quelles par le support de soutenance).

Un schéma de flux se décrit par des **nœuds** (boîte ou base de données), des **groupes** (cadre
pointillé titré) et des **liens** (flèche, éventuellement coudée par des points de passage).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle

from churn_saas import viz

# Remplissage clair et bordure franche, tirés de la palette du projet
STYLES: dict[str, tuple[str, str]] = {
    "source": ("#E3EAEE", "#264653"),
    "traitement": ("#DCEBF5", "#2E86AB"),
    "donnees": ("#FBE7D5", "#F4A261"),
    "exposition": ("#DDF2E5", "#3BB273"),
    "securite": ("#F6DDE0", "#E84855"),
    "outil": ("#EFDDE8", "#A23B72"),
}
COULEUR_TEXTE = "#1F2933"
COULEUR_LIEN = "#4A5560"


@dataclass(frozen=True)
class Noeud:
    """Boîte centrée en ``(x, y)`` ; ``base=True`` la dessine en cylindre (stockage)."""

    x: float
    y: float
    texte: str
    style: str = "traitement"
    largeur: float = 2.6
    hauteur: float = 1.0
    base: bool = False


@dataclass(frozen=True)
class Groupe:
    """Cadre pointillé titré, de coin bas-gauche ``(x0, y0)`` à coin haut-droit ``(x1, y1)``.

    ``titre_a_droite`` cale le titre à droite du bord supérieur, là où aucun lien n'entre.
    """

    x0: float
    y0: float
    x1: float
    y1: float
    titre: str
    titre_a_droite: bool = False


@dataclass(frozen=True)
class Lien:
    """Flèche de ``depart`` vers ``arrivee`` (clés de nœuds), coudée par ``via`` si fourni."""

    depart: str
    arrivee: str
    libelle: str = ""
    via: tuple[tuple[float, float], ...] = field(default_factory=tuple)
    position_libelle: tuple[float, float] | None = None


def _bord(noeud: Noeud, cible: tuple[float, float]) -> tuple[float, float]:
    """Point du bord de la boîte situé sur la droite qui joint son centre à ``cible``."""
    dx, dy = cible[0] - noeud.x, cible[1] - noeud.y
    if dx == 0 and dy == 0:
        return noeud.x, noeud.y
    # Le premier côté atteint (vertical ou horizontal) fixe la distance au bord
    t = min(
        (noeud.largeur / 2) / abs(dx) if dx else float("inf"),
        (noeud.hauteur / 2) / abs(dy) if dy else float("inf"),
    )
    return noeud.x + t * dx, noeud.y + t * dy


def _dessiner_noeud(ax: Axes, noeud: Noeud) -> None:
    fond, bord = STYLES[noeud.style]
    x0, y0 = noeud.x - noeud.largeur / 2, noeud.y - noeud.hauteur / 2
    if noeud.base:
        # Cylindre : corps rectangulaire, ellipse pleine en bas, ellipse ouverte en haut
        ellipse_h = min(0.3, noeud.hauteur * 0.3)
        ax.add_patch(
            Ellipse((noeud.x, y0), noeud.largeur, ellipse_h, fc=fond, ec=bord, lw=1.4, zorder=2)
        )
        ax.add_patch(
            Rectangle((x0, y0), noeud.largeur, noeud.hauteur, fc=fond, ec="none", zorder=2)
        )
        ax.plot([x0, x0], [y0, y0 + noeud.hauteur], color=bord, lw=1.4, zorder=3)
        ax.plot([x0 + noeud.largeur] * 2, [y0, y0 + noeud.hauteur], color=bord, lw=1.4, zorder=3)
        ax.add_patch(
            Ellipse(
                (noeud.x, y0 + noeud.hauteur),
                noeud.largeur,
                ellipse_h,
                fc=fond,
                ec=bord,
                lw=1.4,
                zorder=3,
            )
        )
        y_texte = noeud.y - ellipse_h / 4
    else:
        ax.add_patch(
            FancyBboxPatch(
                (x0, y0),
                noeud.largeur,
                noeud.hauteur,
                boxstyle="round,pad=0,rounding_size=0.15",
                fc=fond,
                ec=bord,
                lw=1.4,
                zorder=2,
            )
        )
        y_texte = noeud.y
    ax.text(
        noeud.x,
        y_texte,
        noeud.texte,
        ha="center",
        va="center",
        fontsize=9,
        color=COULEUR_TEXTE,
        zorder=4,
    )


def _dessiner_groupe(ax: Axes, groupe: Groupe) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (groupe.x0, groupe.y0),
            groupe.x1 - groupe.x0,
            groupe.y1 - groupe.y0,
            boxstyle="round,pad=0,rounding_size=0.2",
            fc="#FAFBFC",
            ec="#9AA5B1",
            lw=1.1,
            ls="--",
            zorder=0,
        )
    )
    # Titre posé sur le cadre, sur fond blanc pour masquer le pointillé et les liens voisins
    ax.text(
        groupe.x1 - 0.25 if groupe.titre_a_droite else groupe.x0 + 0.25,
        groupe.y1,
        groupe.titre,
        ha="right" if groupe.titre_a_droite else "left",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color="#52606D",
        bbox={"fc": "white", "ec": "none", "pad": 1.5},
        zorder=5,
    )


def _dessiner_lien(ax: Axes, noeuds: dict[str, Noeud], lien: Lien) -> None:
    depart, arrivee = noeuds[lien.depart], noeuds[lien.arrivee]
    premier = lien.via[0] if lien.via else (arrivee.x, arrivee.y)
    dernier = lien.via[-1] if lien.via else (depart.x, depart.y)
    points = [_bord(depart, premier), *lien.via, _bord(arrivee, dernier)]
    for (xa, ya), (xb, yb) in zip(points[:-2], points[1:-1], strict=True):
        ax.plot([xa, xb], [ya, yb], color=COULEUR_LIEN, lw=1.3, zorder=1)
    ax.add_patch(
        FancyArrowPatch(
            points[-2],
            points[-1],
            arrowstyle="-|>",
            mutation_scale=13,
            color=COULEUR_LIEN,
            lw=1.3,
            zorder=1,
        )
    )
    if lien.libelle:
        x, y = lien.position_libelle or (
            (points[0][0] + points[1][0]) / 2,
            (points[0][1] + points[1][1]) / 2,
        )
        ax.text(
            x,
            y,
            lien.libelle,
            ha="center",
            va="center",
            fontsize=8.5,
            fontstyle="italic",
            color=COULEUR_LIEN,
            bbox={"fc": "white", "ec": "none", "pad": 1.0},
            zorder=5,
        )


def schema_flux(
    nom: str,
    titre: str,
    noeuds: dict[str, Noeud],
    liens: list[Lien],
    groupes: list[Groupe],
    etendue: tuple[float, float],
) -> Figure:
    """Trace un schéma de flux dans une figure :func:`churn_saas.viz.figure`.

    Parameters
    ----------
    nom, titre :
        Transmis à :func:`churn_saas.viz.figure`.
    noeuds :
        Nœuds indexés par une clé courte, référencée par les liens.
    liens, groupes :
        Flèches et cadres à tracer.
    etendue :
        Largeur et hauteur du canevas, en unités de coordonnées (1 unité ≈ 0,65 pouce).

    Returns
    -------
    La figure, à sauvegarder par :func:`churn_saas.viz.sauvegarder`.
    """
    largeur, hauteur = etendue
    fig, ax = viz.figure(nom, titre, taille=(largeur * 0.65, hauteur * 0.65))
    ax.set_xlim(0, largeur)
    ax.set_ylim(0, hauteur)
    ax.set_aspect("equal")
    ax.set_axis_off()
    for groupe in groupes:
        _dessiner_groupe(ax, groupe)
    for lien in liens:
        _dessiner_lien(ax, noeuds, lien)
    for noeud in noeuds.values():
        _dessiner_noeud(ax, noeud)
    return fig


def schema_pipeline_donnees() -> Figure:
    """§11.1 — des sources au CRM : ingestion, pipeline ML de nuit, exposition."""
    sources = {
        "crm": "CRM",
        "support": "Outil support",
        "facturation": "ERP facturation",
        "produit": "Events produit",
    }
    noeuds = {
        cle: Noeud(1.55, 6.0 - 1.5 * i, texte, "source", 2.2, 0.8, base=True)
        for i, (cle, texte) in enumerate(sources.items())
    }
    noeuds |= {
        "etl": Noeud(4.7, 3.75, "Ingestion 23 h 00\nETL Python / dbt", largeur=2.4, hauteur=1.2),
        "parquet": Noeud(7.5, 3.75, "Parquet", "donnees", 1.8, 0.8, base=True),
        "prep": Noeud(10.4, 5.7, "Préparation"),
        "modele": Noeud(13.4, 5.7, "Modèle champion\nregistre MLflow", largeur=2.8),
        "scores": Noeud(16.5, 5.7, "Scores", "donnees", 2.0, 0.8, base=True),
        "api": Noeud(10.8, 1.9, "API FastAPI\nPOST /predict", "exposition"),
        "batch": Noeud(16.5, 2.75, "Job batch Prefect", "exposition"),
        "crm_sortie": Noeud(14.3, 1.0, "CRM — champ\nrisque_churn", "exposition", base=True),
        "tdb": Noeud(18.45, 1.0, "Tableau de bord CS", "exposition", 2.8),
    }
    liens = [Lien(cle, "etl") for cle in sources]
    liens += [
        Lien("etl", "parquet"),
        Lien("parquet", "prep"),
        Lien("prep", "modele"),
        Lien("modele", "scores"),
        Lien("scores", "batch"),
        Lien("batch", "crm_sortie"),
        Lien("batch", "tdb"),
        Lien("api", "modele", "même modèle"),
    ]
    groupes = [
        Groupe(0.2, 0.7, 2.9, 6.9, "Sources de données"),
        Groupe(8.8, 4.6, 20.0, 6.9, "Pipeline ML — batch 02 h 00"),
        Groupe(8.8, 0.2, 20.0, 3.7, "Exposition"),
    ]
    return schema_flux(
        "pipeline_donnees",
        "Pipeline de données, des sources au CRM",
        noeuds,
        liens,
        groupes,
        (20.2, 7.4),
    )


def schema_architecture_cible() -> Figure:
    """§11.2 — scénario B : passerelle, conteneurs managés, données managées, CI/CD."""
    noeuds = {
        "navigateur": Noeud(1.9, 8.5, "Navigateur CSM", "exposition"),
        "crm_client": Noeud(14.6, 8.5, "CRM client", "source", 2.0, 0.8, base=True),
        "gw": Noeud(1.9, 6.4, "API Gateway\nHTTPS + JWT", "securite"),
        "waf": Noeud(4.9, 6.4, "WAF / limitation\nde débit", "securite"),
        "cicd": Noeud(8.9, 6.4, "CI/CD GitHub Actions\ntests · gate · image Docker", "outil", 3.4),
        "api": Noeud(2.2, 4.25, "API FastAPI\n2 tâches × 0,5 vCPU", largeur=3.0),
        "worker": Noeud(5.6, 4.25, "Worker Prefect\n2 vCPU", largeur=3.0),
        "surveillance": Noeud(9.6, 4.25, "Surveillance\nEvidently + Prometheus", largeur=3.0),
        "mlflow": Noeud(2.2, 1.5, "Registre MLflow\net artefacts", "outil", 3.0),
        "s3": Noeud(5.6, 1.5, "Stockage objet\nParquet", "donnees", 2.8, 1.0, base=True),
        "rds": Noeud(9.6, 1.5, "PostgreSQL\nscores, audit", "donnees", 2.8, 1.0, base=True),
    }
    liens = [
        Lien("navigateur", "gw", "HTTPS", position_libelle=(2.7, 7.5)),
        Lien("gw", "waf"),
        Lien("waf", "api"),
        Lien("api", "mlflow"),
        Lien("worker", "s3"),
        Lien("worker", "mlflow"),
        Lien("worker", "rds"),
        Lien("surveillance", "rds"),
        Lien("cicd", "api"),
        Lien("cicd", "worker"),
        # Contourne le bloc de données par le bas pour ne croiser aucun autre lien
        Lien(
            "crm_client",
            "s3",
            "export quotidien",
            via=((14.6, 0.1), (5.6, 0.1)),
            position_libelle=(14.6, 6.6),
        ),
    ]
    groupes = [
        Groupe(0.3, 7.75, 15.7, 9.4, "Réseau client"),
        Groupe(0.3, 5.65, 6.5, 7.25, "Passerelle exposée", titre_a_droite=True),
        Groupe(
            0.3, 3.4, 14.0, 5.1, "Conteneurs managés — ECS Fargate / Cloud Run", titre_a_droite=True
        ),
        Groupe(3.9, 0.6, 11.4, 2.75, "Données managées", titre_a_droite=True),
    ]
    return schema_flux(
        "architecture_cible",
        "Architecture cible — scénario B",
        noeuds,
        liens,
        groupes,
        (16.0, 9.8),
    )


def schema_sequence_score() -> Figure:
    """§11.3 — diagramme de séquence du score à la demande (CSM → CRM → API → CRM)."""
    acteurs = ["CSM", "CRM", "API Gateway", "API FastAPI", "Registre MLflow", "Base scores"]
    # (émetteur, destinataire, message, réponse) ; émetteur = destinataire : traitement interne
    messages = [
        (0, 1, "Ouvre la fiche du compte C-1042", False),
        (1, 2, "POST /predict {variables du compte, sans client_id} + JWT", False),
        (2, 3, "Requête validée (JWT, limitation de débit)", False),
        (3, 4, "Modèle champion @production (cache local 1 h)", False),
        (3, 3, "Inférence → probabilité, décision,\n3 facteurs SHAP, signaux métier", False),
        (3, 5, "Journalise horodatage, version du modèle, score", False),
        (3, 1, "200 OK {probabilite_churn, decision, facteurs_shap, facteurs_principaux}", True),
        (1, 0, "Fiche enrichie, rattachée à C-1042 par le CRM", True),
    ]
    ecart, pas = 3.1, 1.0
    largeur = ecart * len(acteurs)
    hauteur = 2.2 + pas * len(messages)
    fig, ax = viz.figure(
        "sequence_score_a_la_demande",
        "Séquence d'un score à la demande",
        taille=(largeur * 0.62, hauteur * 0.62),
    )
    ax.set_xlim(0, largeur)
    ax.set_ylim(0, hauteur)
    ax.set_axis_off()

    xs = [ecart * (i + 0.5) for i in range(len(acteurs))]
    y_tete = hauteur - 0.6
    for i, (x, acteur) in enumerate(zip(xs, acteurs, strict=True)):
        style = "exposition" if i == 0 else "outil" if i >= 4 else "traitement"
        _dessiner_noeud(ax, Noeud(x, y_tete, acteur, style, 2.5, 0.8))
        ax.plot([x, x], [0.2, y_tete - 0.4], color="#9AA5B1", lw=1, ls="--", zorder=0)

    for k, (de, vers, texte, reponse) in enumerate(messages):
        y = y_tete - 1.3 - pas * k
        if de == vers:
            # Traitement interne : boucle à droite de la ligne de vie, libellé à côté
            x = xs[de]
            ax.add_patch(
                FancyArrowPatch(
                    (x, y + 0.2),
                    (x, y - 0.2),
                    connectionstyle="arc3,rad=-1.2",
                    arrowstyle="-|>",
                    mutation_scale=12,
                    color=COULEUR_LIEN,
                    lw=1.3,
                )
            )
            ax.text(x + 0.45, y, texte, ha="left", va="center", fontsize=8.5, color=COULEUR_TEXTE)
            continue
        ax.add_patch(
            FancyArrowPatch(
                (xs[de], y),
                (xs[vers], y),
                arrowstyle="-|>",
                mutation_scale=12,
                color=COULEUR_LIEN,
                lw=1.3,
                ls="--" if reponse else "-",
            )
        )
        ax.text(
            (xs[de] + xs[vers]) / 2,
            y + 0.08,
            texte,
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=COULEUR_TEXTE,
            bbox={"fc": "white", "ec": "none", "pad": 0.5},
        )
    return fig
