"""Tests du cache des étapes lourdes (charger_ou_calculer).

`config.TABLES` est redirigé vers un dossier temporaire par la fixture `sorties_isolees`.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pytest

from churn_saas.cache import charger, charger_ou_calculer


def _compteur_de_calculs() -> tuple[list[int], Callable[[], pd.DataFrame]]:
    appels: list[int] = []

    def calculer() -> pd.DataFrame:
        appels.append(1)
        return pd.DataFrame({"valeur": [len(appels)]})

    return appels, calculer


def test_second_appel_lit_le_cache() -> None:
    appels, calculer = _compteur_de_calculs()
    charger_ou_calculer("essai.parquet", calculer)
    resultat, _ = charger_ou_calculer("essai.parquet", calculer)
    assert len(appels) == 1
    assert resultat["valeur"].iloc[0] == 1


def test_force_recalc_ignore_le_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    # `make notebook-full` positionne FORCE_RECALC=1 : le cache existant doit être ignoré
    appels, calculer = _compteur_de_calculs()
    charger_ou_calculer("essai.parquet", calculer)
    monkeypatch.setenv("FORCE_RECALC", "1")
    resultat, _ = charger_ou_calculer("essai.parquet", calculer)
    assert len(appels) == 2
    assert resultat["valeur"].iloc[0] == 2


def test_charger_ne_calcule_ni_n_ecrit_jamais(monkeypatch: pytest.MonkeyPatch) -> None:
    # Lecture seule : même en recalcul forcé, un artefact existant est relu, jamais écrasé
    _, calculer = _compteur_de_calculs()
    charger_ou_calculer("essai.parquet", calculer)
    monkeypatch.setenv("FORCE_RECALC", "1")
    assert charger("essai.parquet")["valeur"].iloc[0] == 1
    assert charger("absent.parquet", "défaut") == "défaut"
