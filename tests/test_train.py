"""Tests unitaires rapides de `churn_saas.models.train` (hors entraînement en cross-validation)."""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from churn_saas import config
from churn_saas.models import train
from churn_saas.models.train import journaliser_mlflow, mesurer_empreinte


def test_mesurer_empreinte_sans_parametre_codecarbon_deprecie() -> None:
    with warnings.catch_warnings(record=True) as captures:
        warnings.simplefilter("always")
        resultat, rapport = mesurer_empreinte(lambda: 42)
    messages = [str(w.message) for w in captures if issubclass(w.category, DeprecationWarning)]
    assert not any("save_to_" in m for m in messages)
    assert resultat == 42
    assert rapport["emissions_kg_co2"] >= 0.0
    assert (config.TABLES / "codecarbon_emissions.csv").exists()


def test_journaliser_mlflow_sans_bruit_info(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Les INFO de MLflow (détection uv, export des dépendances) sont tus, le niveau restauré."""
    monkeypatch.setattr(config, "MLRUNS", tmp_path)
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    rng = np.random.default_rng(config.RANDOM_SEED)
    modele = LogisticRegression().fit(rng.normal(size=(40, 2)), np.tile([0, 1], 20))
    niveau_avant = logging.getLogger("mlflow").level

    with caplog.at_level(logging.INFO):
        run_id = journaliser_mlflow("essai", modele, {"auc": 0.5}, {"C": 1.0})

    assert run_id
    bruit = [
        r for r in caplog.records if r.name.startswith("mlflow") and r.levelno < logging.WARNING
    ]
    assert not bruit, [r.getMessage() for r in bruit]
    assert logging.getLogger("mlflow").level == niveau_avant


# ---------------------------------------------------------------------------
# Archives du champion et retour arrière : modèle et méta voyagent ensemble
# ---------------------------------------------------------------------------


class TestArchivesChampion:
    @pytest.fixture
    def dossier(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
        monkeypatch.setattr(config, "ARTIFACTS", tmp_path / "artifacts")
        dossier = tmp_path / "artifacts" / "models"
        dossier.mkdir(parents=True)
        return dossier

    @staticmethod
    def _champion(dossier: Path, contenu: bytes, meta: dict | None) -> None:
        (dossier / "best_model.pkl").write_bytes(contenu)
        chemin_meta = dossier / "best_model_meta.json"
        if meta is None:
            chemin_meta.unlink(missing_ok=True)
        else:
            chemin_meta.write_text(json.dumps(meta), encoding="utf-8")

    def test_archive_le_modele_et_sa_meta_sous_le_meme_horodatage(self, dossier: Path) -> None:
        self._champion(dossier, b"modele A", {"seuil_economique": 0.277})
        horodatage = train.archiver_champion()
        assert horodatage is not None
        assert (dossier / f"archive_{horodatage}_best_model.pkl").read_bytes() == b"modele A"
        meta = json.loads((dossier / f"archive_{horodatage}_best_model_meta.json").read_text())
        assert meta["seuil_economique"] == pytest.approx(0.277)

    def test_sans_champion_rien_a_archiver(self, dossier: Path) -> None:
        assert train.archiver_champion() is None
        assert train.lister_archives() == []

    def test_deux_archives_dans_la_meme_seconde_ne_s_ecrasent_pas(self, dossier: Path) -> None:
        self._champion(dossier, b"modele A", None)
        premiere = train.archiver_champion()
        self._champion(dossier, b"modele B", None)
        seconde = train.archiver_champion()
        assert premiere != seconde
        assert train.lister_archives() == sorted([premiere, seconde])

    def test_ecrire_meta_champion(self, dossier: Path) -> None:
        chemin = train.ecrire_meta_champion({"modele_nom": "LR", "hyperparametres": {"C": 0.1}})
        assert chemin == dossier / "best_model_meta.json"
        assert json.loads(chemin.read_text(encoding="utf-8"))["hyperparametres"] == {"C": 0.1}

    def test_restaure_modele_et_seuil_de_l_archive(
        self, dossier: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        alias: list[str] = []
        monkeypatch.setattr(train, "_deplacer_alias_production", alias.append)
        self._champion(dossier, b"modele A", {"seuil_economique": 0.277, "version_registre": "12"})
        archive_a = train.archiver_champion()
        self._champion(dossier, b"modele B", {"seuil_economique": 0.31, "version_registre": "13"})

        bilan = train.restaurer_archive(archive_a)

        assert (dossier / "best_model.pkl").read_bytes() == b"modele A"
        meta = json.loads((dossier / "best_model_meta.json").read_text(encoding="utf-8"))
        assert meta["seuil_economique"] == pytest.approx(0.277)
        assert alias == ["12"] and bilan["alias_deplace"] is True
        # Le champion retiré est archivé à son tour : le retour arrière est réversible
        archive_b = bilan["archive_du_champion_retire"]
        assert (dossier / f"archive_{archive_b}_best_model.pkl").read_bytes() == b"modele B"

    def test_archive_sans_meta_retire_la_meta_courante(
        self, dossier: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Archive du modèle du notebook (antérieure aux métas) : le seuil de vigilance s'applique."""
        monkeypatch.setattr(train, "_deplacer_alias_production", lambda version: None)
        self._champion(dossier, b"modele A", None)
        archive_a = train.archiver_champion()
        self._champion(dossier, b"modele B", {"seuil_economique": 0.31})

        bilan = train.restaurer_archive(archive_a)

        assert (dossier / "best_model.pkl").read_bytes() == b"modele A"
        assert not (dossier / "best_model_meta.json").exists()
        assert bilan["meta_restauree"] is False and bilan["alias_deplace"] is False

    def test_archive_inconnue_leve_une_erreur(self, dossier: Path) -> None:
        with pytest.raises(FileNotFoundError):
            train.restaurer_archive("20990101_000000")

    def test_pr_auc_du_champion_courant(self, dossier: Path) -> None:
        assert train.pr_auc_champion_courant() is None
        self._champion(dossier, b"modele", {"modele_nom": "LR"})  # méta du notebook : pas de note
        assert train.pr_auc_champion_courant() is None
        self._champion(dossier, b"modele", {"pr_auc_test": 0.77})
        assert train.pr_auc_champion_courant() == pytest.approx(0.77)
