"""Génère et exécute le notebook CISIA depuis les sections jupytext.

Usage :
    uv run python notebooks/build_notebook.py
    uv run python notebooks/build_notebook.py --sans-execution
    uv run python notebooks/build_notebook.py --force
"""

import argparse
import os
import sys
import time
from pathlib import Path

import jupytext
import nbclient
import nbformat

RACINE = Path(__file__).resolve().parent.parent
SECTIONS_DIR = Path(__file__).resolve().parent / "sections"
SORTIE = Path(__file__).resolve().parent / "churn_saas_certification.ipynb"

NOYAU = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}

TIMEOUT_CELLULE = 600  # secondes par cellule (étapes lourdes : Optuna, SHAP, CodeCarbon)


def lire_sections() -> list[Path]:
    """Retourne les fichiers de sections triés par nom (ordre numérique)."""
    fichiers = sorted(SECTIONS_DIR.glob("*.py"))
    if not fichiers:
        print(f"[ERREUR] Aucune section trouvée dans {SECTIONS_DIR}", file=sys.stderr)
        sys.exit(1)
    print(f"[INFO] {len(fichiers)} section(s) détectée(s) :")
    for f in fichiers:
        print(f"       • {f.name}")
    return fichiers


def fusionner_sections(fichiers: list[Path]) -> nbformat.NotebookNode:
    """Lit chaque section au format jupytext percent et fusionne les cellules."""
    notebooks = []
    for f in fichiers:
        nb = jupytext.read(f, fmt="py:percent")
        notebooks.append(nb)

    # Le premier notebook sert de base ; on lui greffe les cellules des suivants
    combined = notebooks[0]
    for nb in notebooks[1:]:
        combined.cells.extend(nb.cells)

    # Métadonnées du noyau — obligatoires pour nbclient
    combined.metadata["kernelspec"] = NOYAU
    combined.metadata.setdefault("language_info", {"name": "python", "version": "3.12"})

    # Formatage nbformat v4
    nbformat.validate(combined)
    return combined


def executer(nb: nbformat.NotebookNode) -> nbformat.NotebookNode:
    """Exécute le notebook avec nbclient et retourne le notebook avec sorties."""
    client = nbclient.NotebookClient(
        nb,
        timeout=TIMEOUT_CELLULE,
        kernel_name="python3",
        resources={"metadata": {"path": str(RACINE)}},
    )
    client.execute()
    return nb


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Génère le notebook CISIA depuis les sections jupytext."
    )
    parser.add_argument(
        "--sans-execution",
        action="store_true",
        help="Convertit les sections en notebook sans l'exécuter.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force le recalcul de toutes les étapes lourdes (ignore le cache).",
    )
    args = parser.parse_args()

    # --force positionne la variable d'environnement lue par churn_saas.cache
    if args.force or os.environ.get("FORCE_RECALC") == "1":
        os.environ["FORCE_RECALC"] = "1"
        print("[INFO] Mode recalcul forcé activé (FORCE_RECALC=1).")

    t_debut = time.perf_counter()

    print("[INFO] Lecture et fusion des sections…")
    fichiers = lire_sections()
    nb = fusionner_sections(fichiers)

    if args.sans_execution:
        print("[INFO] Mode --sans-execution : écriture du notebook sans exécution.")
        nbformat.write(nb, SORTIE)
    else:
        print("[INFO] Exécution du notebook (timeout par cellule :", TIMEOUT_CELLULE, "s)…")
        try:
            nb = executer(nb)
        except nbclient.exceptions.CellExecutionError as exc:
            print(f"\n[ERREUR] Échec lors de l'exécution d'une cellule :\n{exc}", file=sys.stderr)
            # On sauvegarde quand même le notebook partiel pour faciliter le débogage
            nbformat.write(nb, SORTIE)
            print(f"[INFO] Notebook partiel sauvegardé → {SORTIE.relative_to(RACINE)}")
            sys.exit(1)
        nbformat.write(nb, SORTIE)

    t_fin = time.perf_counter()
    duree = t_fin - t_debut
    print(f"\n[OK] Notebook généré → {SORTIE.relative_to(RACINE)}")
    print(f"[OK] Temps total : {duree:.1f} s")


if __name__ == "__main__":
    main()
