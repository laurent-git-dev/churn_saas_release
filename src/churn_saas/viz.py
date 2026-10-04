"""Style matplotlib unique pour le projet churn_saas.

Toutes les figures du notebook passent par :func:`figure` pour garantir :
- une numérotation incrémentale cohérente,
- une sauvegarde automatique dans ``reports/figures/``,
- un style visuel uniforme (palette, grille, typographie).
"""

from __future__ import annotations

import itertools
import re
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from churn_saas import config
from churn_saas.format_fr import ESPACE_FINE

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

# Palette principale — 8 couleurs accessibles (simulée daltonisme CB91 friendly).
PALETTE_PRINCIPALE: list[str] = [
    "#2E86AB",  # bleu primaire  — fidélité, rétention
    "#E84855",  # rouge alerte   — résiliation, churn
    "#3BB273",  # vert succès    — performance positive
    "#F4A261",  # orange accent  — vigilance
    "#A23B72",  # violet         — catégorie supplémentaire
    "#264653",  # bleu nuit      — données historiques
    "#E9C46A",  # or             — valeur économique
    "#457B9D",  # bleu acier     — référence, baseline
]

# Couleurs sémantiques pour les graphiques de classification churn
COULEUR_CHURN: str = "#E84855"  # rouge  → risque de résiliation
COULEUR_NON_CHURN: str = "#2E86AB"  # bleu   → fidélité confirmée
COULEUR_SUCCES: str = "#3BB273"  # vert   → prédiction correcte, performance positive

# Colormap séquentielle divergente : vert (faible risque) → rouge (risque élevé)
PALETTE_RISQUE: str = "RdYlGn_r"

# ---------------------------------------------------------------------------
# Style global (appliqué à l'import du module)
# ---------------------------------------------------------------------------

_STYLE_GLOBAL: dict[str, Any] = {
    "figure.facecolor": "white",
    "axes.facecolor": "#F8F9FA",
    "axes.edgecolor": "#CCCCCC",
    "axes.grid": True,
    "grid.color": "#E0E0E0",
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.framealpha": 0.9,
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
}

mpl.rcParams.update(_STYLE_GLOBAL)  # type: ignore[arg-type]

# ---------------------------------------------------------------------------
# État interne du module
# ---------------------------------------------------------------------------

# Compteur global de figures — réinitialisé à chaque démarrage de session / import
_compteur: itertools.count[int] = itertools.count(start=1)

# Registre fig_id → chemin de sauvegarde (évite les attributs ad hoc sur Figure)
_registre: dict[int, Path] = {}

# ---------------------------------------------------------------------------
# API publique
# ---------------------------------------------------------------------------


def figure(
    nom: str,
    titre: str,
    taille: tuple[float, float] = (10.0, 5.5),
) -> tuple[Figure, Axes]:
    """Crée une figure numérotée et enregistre son chemin de sauvegarde.

    Paramètres
    ----------
    nom :
        Identifiant court sans extension (ex. : ``"roc_gradient_boosting"``).
    titre :
        Titre complet en français, affiché au-dessus de la figure.
    taille :
        Largeur × hauteur en pouces.

    Retourne
    --------
    ``(fig, ax)`` — la figure matplotlib et son axe principal.

    Notes
    -----
    Appeler :func:`sauvegarder` en fin de cellule notebook pour persister la figure.
    La figure est aussi sauvegardée automatiquement si ``plt.close()`` est appelé.
    """
    num = next(_compteur)
    fig, ax = plt.subplots(figsize=taille)
    ax.set_title(f"Fig. {num} — {titre}", pad=12)

    config.FIGURES.mkdir(parents=True, exist_ok=True)
    chemin = config.FIGURES / f"{num:02d}_{nom}.png"
    _registre[id(fig)] = chemin

    # Sauvegarde automatique si la figure est fermée explicitement (plt.close / plt.clf)
    def _on_close(event: object) -> None:
        _ecrire_png(fig, chemin)

    fig.canvas.mpl_connect("close_event", _on_close)
    return fig, ax


def figure_grille(
    nom: str,
    titre: str,
    nlignes: int = 1,
    ncols: int = 2,
    taille: tuple[float, float] = (14.0, 5.5),
) -> tuple[Figure, Any]:
    """Crée une figure multi-axes numérotée.

    Analogue à :func:`figure` mais avec ``plt.subplots(nlignes, ncols)``.
    :func:`sauvegarder` fonctionne de la même façon sur la figure retournée.

    Retourne
    --------
    ``(fig, axes)`` — la figure et le tableau d'axes numpy.
    """
    num = next(_compteur)
    fig, axes = plt.subplots(nlignes, ncols, figsize=taille)
    fig.suptitle(f"Fig. {num} — {titre}", fontsize=13, fontweight="bold", y=1.02)

    config.FIGURES.mkdir(parents=True, exist_ok=True)
    chemin = config.FIGURES / f"{num:02d}_{nom}.png"
    _registre[id(fig)] = chemin

    def _on_close(event: object) -> None:
        _ecrire_png(fig, chemin)

    fig.canvas.mpl_connect("close_event", _on_close)
    return fig, axes


def sauvegarder(fig: Figure) -> Path:
    """Sauvegarde la figure dans ``reports/figures/`` et retourne le chemin PNG.

    À appeler en fin de cellule notebook, après tout le tracé.
    Lève ``ValueError`` si la figure n'a pas été créée via :func:`figure`.
    """
    chemin = _registre.get(id(fig))
    if chemin is None:
        raise ValueError(
            "Figure non créée via churn_saas.viz.figure() — chemin de sauvegarde inconnu."
        )
    _ecrire_png(fig, chemin)
    return chemin


def couleur(index: int) -> str:
    """Retourne la couleur à la position ``index`` dans la palette principale (modulo)."""
    return PALETTE_PRINCIPALE[index % len(PALETTE_PRINCIPALE)]


# ---------------------------------------------------------------------------
# Fonction interne
# ---------------------------------------------------------------------------


class _FormateurFrancais(mticker.Formatter):
    """Enveloppe un formateur numérique matplotlib et francise ses libellés.

    Le formateur d'origine garde toute sa logique (choix des décimales, décalage, échelle) ;
    seul le texte produit est converti : « 0.25 » → « 0,25 », « 12000 » → « 12 000 »,
    « 40% » → « 40 % ».
    """

    def __init__(self, base: mticker.Formatter) -> None:
        self.base = base

    def set_axis(self, axis: Any) -> None:
        super().set_axis(axis)
        self.base.set_axis(axis)

    def set_locs(self, locs: Any) -> None:
        super().set_locs(locs)
        self.base.set_locs(locs)

    def __call__(self, x: float, pos: int | None = None) -> str:
        return _franciser_graduation(self.base(x, pos))

    def get_offset(self) -> str:
        return _franciser_graduation(self.base.get_offset())

    def format_data_short(self, value: float) -> str:
        return _franciser_graduation(self.base.format_data_short(value))


# Formateurs purement numériques : les autres (dates, catégories, log, FuncFormatter maison)
# produisent des libellés qu'il ne faut pas réécrire
_FORMATEURS_NUMERIQUES = (
    mticker.ScalarFormatter,
    mticker.PercentFormatter,
    mticker.StrMethodFormatter,
    mticker.FormatStrFormatter,
)


def _franciser_graduation(texte: str) -> str:
    texte = texte.replace(".", ",")
    # Séparateur de milliers sur la partie entière uniquement (pas après la virgule)
    texte = re.sub(
        r"(?<![,\d])\d{4,}",
        lambda m: f"{int(m.group()):,}".replace(",", ESPACE_FINE),
        texte,
    )
    return re.sub(r"(\d)%", rf"\1{ESPACE_FINE}%", texte)


def _franciser_axes(fig: Figure) -> None:
    """Applique le format français aux graduations numériques de tous les axes de la figure."""
    for ax in fig.axes:
        for axe in (ax.xaxis, ax.yaxis):
            formateur = axe.get_major_formatter()
            if isinstance(formateur, _FORMATEURS_NUMERIQUES):
                axe.set_major_formatter(_FormateurFrancais(formateur))


def _ecrire_png(fig: Figure, chemin: Path) -> None:
    # Juste avant l'écriture plutôt qu'à la création : les formateurs automatiques (dates,
    # catégories) ne sont installés par matplotlib qu'au tracé, sur un axe au formateur par défaut
    _franciser_axes(fig)
    fig.savefig(chemin, dpi=150, bbox_inches="tight")
