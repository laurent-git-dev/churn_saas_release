"""Schémas d'architecture du §11 : tracés sans erreur et sauvegardés comme toute figure."""

from __future__ import annotations

from collections.abc import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from churn_saas import schemas, viz  # noqa: E402
from churn_saas.schemas import Groupe, Lien, Noeud  # noqa: E402


@pytest.mark.parametrize(
    "tracer",
    [
        schemas.schema_pipeline_donnees,
        schemas.schema_architecture_cible,
        schemas.schema_sequence_score,
    ],
)
def test_schema_sauvegarde_en_png(tracer: Callable[[], Figure]) -> None:
    fig = tracer()
    chemin = viz.sauvegarder(fig)
    plt.close(fig)
    assert chemin.exists() and chemin.stat().st_size > 0


def test_lien_vers_noeud_inconnu_echoue() -> None:
    noeuds = {"a": Noeud(1, 1, "A")}
    with pytest.raises(KeyError):
        schemas.schema_flux("essai", "Essai", noeuds, [Lien("a", "b")], [], (4, 3))


def test_lien_coude_et_groupe_titre_a_droite() -> None:
    noeuds = {"a": Noeud(1, 1, "A", largeur=1), "b": Noeud(4, 3, "B", base=True, largeur=1)}
    lien = Lien("a", "b", "coude", via=((4, 1),))
    groupe = Groupe(0, 0, 5, 4, "Cadre", titre_a_droite=True)
    fig = schemas.schema_flux("essai", "Essai", noeuds, [lien], [groupe], (5, 4))
    plt.close(fig)
