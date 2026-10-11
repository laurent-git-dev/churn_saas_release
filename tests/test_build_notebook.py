"""Tests de l'ordre d'exécution du notebook (notebooks/build_notebook.py)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import nbformat

_CHEMIN = Path(__file__).resolve().parents[1] / "notebooks" / "build_notebook.py"
_spec = importlib.util.spec_from_file_location("build_notebook", _CHEMIN)
assert _spec is not None and _spec.loader is not None
build_notebook = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_notebook)


def _cellule(section: str, source: str) -> nbformat.NotebookNode:
    cellule = nbformat.v4.new_code_cell(source)
    cellule.metadata["section"] = section
    return cellule


def test_resume_executif_execute_en_dernier_ordre_relatif_conserve() -> None:
    cellules = [
        _cellule("00_page_de_garde.py", "a"),
        _cellule("01_resume_executif.py", "r1"),
        _cellule("01_resume_executif.py", "r2"),
        _cellule("12_performance_impacts.py", "b"),
    ]
    ordre = build_notebook.ordre_execution(cellules)
    assert [c.source for c in ordre] == ["a", "b", "r1", "r2"]


def test_cellules_identiques_non_confondues() -> None:
    # Deux cellules de même contenu dans deux sections : chacune garde son rang
    cellules = [
        _cellule("01_resume_executif.py", "display(x)"),
        _cellule("09_entrainement_validation.py", "display(x)"),
    ]
    ordre = build_notebook.ordre_execution(cellules)
    assert ordre[0] is cellules[1] and ordre[1] is cellules[0]


def test_toutes_les_sections_reelles_conservees() -> None:
    fichiers = sorted(build_notebook.SECTIONS_DIR.glob("*.py"))
    nb = build_notebook.fusionner_sections(fichiers)
    ordre = build_notebook.ordre_execution(nb.cells)
    assert len(ordre) == len(nb.cells)
    assert ordre[-1].metadata["section"] == "01_resume_executif.py"
