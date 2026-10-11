"""Traçabilité du référentiel C1 → C9 : chaque item pointe vers une sous-section réelle."""

from pathlib import Path

from churn_saas.referentiel import (
    COMPETENCES,
    CORRESPONDANCE,
    ITEMS,
    grille_competences,
    items_orphelins,
    localiser_items,
    titre_sans_numero,
)

_LOCALISATION = localiser_items()


def test_aucun_item_orphelin() -> None:
    orphelins = items_orphelins()
    assert orphelins.empty, f"Titres cibles introuvables dans les sections :\n{orphelins}"


def test_chaque_item_rattache_a_une_competence() -> None:
    assert {code.split(".")[0] for code in ITEMS} == set(COMPETENCES)
    assert set(ITEMS) == set(CORRESPONDANCE)


def test_titres_cibles_sans_numero_ni_competence() -> None:
    fautifs = [
        titre
        for _, titre in CORRESPONDANCE.values()
        if titre != titre_sans_numero(titre) or "(C" in titre
    ]
    assert not fautifs, f"Titres cibles à nettoyer : {fautifs}"


def test_localisation_sous_une_sous_section_numerotee() -> None:
    assert _LOCALISATION["titre"].str.match(r"\d+\.\d+").all()


def test_titre_sans_numero_retire_numero_et_competence() -> None:
    assert titre_sans_numero("7.3.0 Application du renommage (C3)") == "Application du renommage"
    assert titre_sans_numero("Valeur attendue chiffrée") == "Valeur attendue chiffrée"


def test_grille_competences_signale_les_orphelins(tmp_path: Path) -> None:
    # Dossier vide : aucun titre réel, donc tous les items sont orphelins.
    grille = grille_competences(localiser_items(tmp_path))
    assert (grille["Items couverts"] == 0).all()
    assert (grille["Items orphelins"] != "—").all()
    assert grille["Items"].sum() == len(ITEMS)
