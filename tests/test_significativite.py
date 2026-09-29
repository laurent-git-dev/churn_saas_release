"""Tests pour le test de supériorité du champion (Wilcoxon apparié + Holm, §8.8 / §9.3.1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churn_saas import config
from churn_saas.models.train import PREFIXE_PLI, comparer_au_champion, correction_holm

_N_PLIS = 15


def _tableau(scores: dict[str, np.ndarray]) -> pd.DataFrame:
    """Tableau au format de comparer_modeles() : une ligne par modèle, une colonne par pli."""
    lignes = {
        nom: {f"{PREFIXE_PLI}{i:02d}": float(v) for i, v in enumerate(vals)}
        | {"pr_auc_mean": float(np.mean(vals))}
        for nom, vals in scores.items()
    }
    return pd.DataFrame.from_dict(lignes, orient="index")


# ---------------------------------------------------------------------------
# Correction de Holm
# ---------------------------------------------------------------------------


def test_holm_valeurs_de_reference() -> None:
    # Exemple calculé à la main : tri [0.01, 0.02, 0.04] → ×3, ×2, ×1 → [0.03, 0.04, 0.04]
    assert correction_holm([0.04, 0.01, 0.02]) == pytest.approx([0.04, 0.03, 0.04])


def test_holm_plafonne_a_un_et_preserve_la_monotonie() -> None:
    ajustees = correction_holm([0.5, 0.6, 0.001])
    assert max(ajustees) <= 1.0
    ordre = np.argsort([0.5, 0.6, 0.001])
    assert list(np.array(ajustees)[ordre]) == sorted(ajustees)


def test_holm_un_seul_test_inchange() -> None:
    assert correction_holm([0.03]) == pytest.approx([0.03])


# ---------------------------------------------------------------------------
# Test de supériorité
# ---------------------------------------------------------------------------


def test_champion_nettement_meilleur_est_significatif() -> None:
    rng = np.random.default_rng(config.RANDOM_SEED)
    base = rng.uniform(0.70, 0.75, _N_PLIS)
    # Écarts positifs et tous distincts : sans ex-aequo, scipy calcule la loi exacte
    ecarts = np.linspace(0.01, 0.05, _N_PLIS)
    tableau = _tableau({"champion": base + ecarts, "concurrent": base})

    resultat = comparer_au_champion(tableau, "champion")

    assert resultat.loc["concurrent", "plis_gagnes"] == f"{_N_PLIS}/{_N_PLIS}"
    assert resultat.loc["concurrent", "p_valeur_brute"] == pytest.approx(0.5**_N_PLIS)
    assert bool(resultat.loc["concurrent", "significatif"])


def test_ecart_de_hasard_n_est_pas_significatif() -> None:
    rng = np.random.default_rng(config.RANDOM_SEED)
    base = rng.uniform(0.70, 0.75, _N_PLIS)
    bruit = rng.normal(0, 0.01, _N_PLIS)
    tableau = _tableau({"champion": base + bruit, "concurrent": base - bruit})
    # Écarts de signe alterné : ni l'un ni l'autre n'est systématiquement meilleur
    tableau.loc["champion", [f"{PREFIXE_PLI}{i:02d}" for i in range(0, _N_PLIS, 2)]] -= 0.03

    resultat = comparer_au_champion(tableau, "champion")

    assert not bool(resultat.loc["concurrent", "significatif"])


def test_scores_identiques_donnent_p_egal_un() -> None:
    base = np.linspace(0.70, 0.75, _N_PLIS)
    resultat = comparer_au_champion(_tableau({"a": base, "b": base.copy()}), "a")
    assert resultat.loc["b", "p_valeur_brute"] == 1.0
    assert not bool(resultat.loc["b", "significatif"])


def test_holm_applique_sur_toutes_les_comparaisons() -> None:
    base = np.linspace(0.70, 0.75, _N_PLIS)
    tableau = _tableau({"champion": base + 0.05, "c1": base, "c2": base - 0.01, "c3": base - 0.02})

    resultat = comparer_au_champion(tableau, "champion")

    attendu = correction_holm(resultat["p_valeur_brute"].tolist())
    assert resultat["p_valeur_holm"].tolist() == pytest.approx(attendu)
    assert len(resultat) == 3


def test_sans_scores_par_pli_leve_une_erreur_explicite() -> None:
    tableau = pd.DataFrame({"pr_auc_mean": [0.8, 0.7]}, index=["a", "b"])
    with pytest.raises(ValueError, match="Scores par pli absents"):
        comparer_au_champion(tableau, "a")
