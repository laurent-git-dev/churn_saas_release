"""Tests du chargement des chiffres de synthèse (§1, §14)."""

import json

import pytest

from churn_saas import config, synthese


def _ecrire(dossier, nom, params):
    (dossier / nom).write_text(json.dumps({"best_params": params, "pr_auc_mean": 0.5}))


def test_artefact_du_modele_ignore_un_ancien_champion(tmp_path, monkeypatch):
    """Le fichier relu est celui dont les hyperparamètres sont ceux du modèle livré, même s'il
    n'est pas le premier dans l'ordre alphabétique."""
    monkeypatch.setattr(config, "TABLES", tmp_path)
    _ecrire(tmp_path, "evaluation_protocole_a_ancien_optimise.json", {"max_depth": 3})
    _ecrire(tmp_path, "evaluation_protocole_b_livre_optimise.json", {"C": 0.1})

    contenu = synthese.artefact_du_modele(
        "evaluation_protocole_*_optimise.json", {"C": 0.1, "max_iter": 500}
    )

    assert contenu["best_params"] == {"C": 0.1}


def test_artefact_du_modele_absent_leve_une_erreur(tmp_path, monkeypatch):
    """Aucun fichier compatible : on s'arrête plutôt que de relire un artefact étranger."""
    monkeypatch.setattr(config, "TABLES", tmp_path)
    _ecrire(tmp_path, "evaluation_protocole_a_ancien_optimise.json", {"max_depth": 3})

    with pytest.raises(FileNotFoundError):
        synthese.artefact_du_modele("evaluation_protocole_*_optimise.json", {"C": 0.1})
