"""Tests de `churn_saas.cli` sans données brutes : le chargement est intercepté."""

from __future__ import annotations

from pathlib import Path

import pytest

from churn_saas import cli, config


class _ReconstructionLancee(Exception):
    """Levée par le faux chargeur : prouve que la reconstruction du gold a démarré."""


@pytest.fixture
def gold_existant(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(config, "DONNEES_GOLD", tmp_path)
    chemin = tmp_path / "gold_dataset.parquet"
    chemin.write_bytes(b"gold existant")

    def _faux_chargeur(nom: str) -> None:
        raise _ReconstructionLancee(nom)

    monkeypatch.setattr(cli, "charger_brut", _faux_chargeur)
    return chemin


def test_gold_existant_reutilise_sans_recalcul_force(gold_existant: Path) -> None:
    assert cli.construire_gold_dataset() == gold_existant


def test_gold_reconstruit_si_force_recalc(
    gold_existant: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FORCE_RECALC", "1")
    with pytest.raises(_ReconstructionLancee):
        cli.construire_gold_dataset()
