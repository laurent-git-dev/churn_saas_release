"""Tests de `churn_saas.cli` sans données brutes : le chargement est intercepté."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import typer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from churn_saas import cli, config
from churn_saas.models import train


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


# ---------------------------------------------------------------------------
# Seuil de surveillance écrit et appliqué par la CLI : même règle que l'API et le batch
# ---------------------------------------------------------------------------


@pytest.fixture
def artefacts_isoles(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Artefacts isolés dans ``tmp_path``, MLflow neutralisé, plancher de la gate à 0.

    Le plancher nul isole la règle testée (non-régression face au champion) de la qualité du
    modèle jouet ; un test dédié remet un plancher inatteignable.
    """
    dossier = tmp_path / "artifacts" / "models"
    dossier.mkdir(parents=True)
    monkeypatch.setattr(config, "ARTIFACTS", tmp_path / "artifacts")
    monkeypatch.setattr(cli, "_CHEMIN_MODELE", dossier / "best_model.pkl")
    monkeypatch.setattr(cli, "_CHEMIN_META", dossier / "best_model_meta.json")
    monkeypatch.setitem(config.CIBLES_PERFORMANCE, "pr_auc_min", 0.0)

    def _sans_mlflow(*_args: object, **_kwargs: object) -> str:
        raise RuntimeError("MLflow neutralisé en test")

    monkeypatch.setattr(train, "journaliser_mlflow", _sans_mlflow)
    return dossier


@pytest.fixture
def gold_synthetique(monkeypatch: pytest.MonkeyPatch) -> tuple[pd.DataFrame, pd.Series]:
    """Gold réduit à trois variables numériques, modèle simple : aucune donnée réelle lue."""
    rng = np.random.default_rng(config.RANDOM_SEED)
    X = pd.DataFrame(rng.normal(size=(200, 3)), columns=["a", "b", "c"])
    y = pd.Series((X["a"] + rng.normal(size=200) > 0.6).astype(int))
    monkeypatch.setattr(cli, "_charger_gold_pour_modele", lambda: (X, y))
    monkeypatch.setattr(
        cli, "famille_et_hyperparametres_champion", lambda: ("regression_logistique", {})
    )
    monkeypatch.setattr(
        cli,
        "construire_modele_optimise",
        lambda nom, params, df_ref: Pipeline(
            [("echelle", StandardScaler()), ("lr", LogisticRegression())]
        ),
    )
    return X, y


class _EspionSeuil:
    """Remplace ``seuil_vigilance_hors_pli`` : mémorise la taille des données reçues."""

    def __init__(self, seuil: float) -> None:
        self.seuil = seuil
        self.n_lignes: list[int] = []

    def __call__(self, pipeline: object, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        self.n_lignes.append(len(X))
        return {"seuil": self.seuil, "recall": 0.8}


def _champion_en_place(dossier: Path, pr_auc: float) -> None:
    (dossier / "best_model.pkl").write_bytes(b"champion")
    meta = {"modele_nom": "LR", "hyperparametres": {}, "pr_auc_test": pr_auc}
    (dossier / "best_model_meta.json").write_text(json.dumps(meta), encoding="utf-8")


def test_train_sans_champion_promeut_avec_le_seuil_hors_pli_du_train(
    artefacts_isoles: Path, gold_synthetique: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    espion = _EspionSeuil(0.321)
    monkeypatch.setattr(cli, "seuil_vigilance_hors_pli", espion)
    cli.train_cmd(forcer_gold=False)
    assert (artefacts_isoles / "best_model.pkl").exists()
    meta = json.loads((artefacts_isoles / "best_model_meta.json").read_text(encoding="utf-8"))
    assert meta["seuil_economique"] == pytest.approx(0.321)
    assert meta["source"] == "cli" and meta["version_registre"] is None
    # Calculé sur le seul train (80 %), jamais sur le split de test
    assert espion.n_lignes == [int(len(gold_synthetique[0]) * (1 - config.PART_TEST))]
    assert train.lister_archives() == []


def test_train_en_regression_laisse_le_champion_en_place(
    artefacts_isoles: Path, gold_synthetique: tuple
) -> None:
    _champion_en_place(artefacts_isoles, pr_auc=0.99)
    with pytest.raises(typer.Exit) as erreur:
        cli.train_cmd(forcer_gold=False)
    assert erreur.value.exit_code == 1
    assert (artefacts_isoles / "best_model.pkl").read_bytes() == b"champion"
    meta = json.loads((artefacts_isoles / "best_model_meta.json").read_text(encoding="utf-8"))
    assert meta["pr_auc_test"] == pytest.approx(0.99)
    assert train.lister_archives() == []


def test_train_accepte_archive_le_champion_et_sa_meta(
    artefacts_isoles: Path, gold_synthetique: tuple
) -> None:
    _champion_en_place(artefacts_isoles, pr_auc=0.0)
    cli.train_cmd(forcer_gold=False)
    (archive,) = train.lister_archives()
    assert (artefacts_isoles / f"archive_{archive}_best_model.pkl").read_bytes() == b"champion"
    assert (artefacts_isoles / f"archive_{archive}_best_model_meta.json").exists()
    assert (artefacts_isoles / "best_model.pkl").read_bytes() != b"champion"


def test_train_sous_le_plancher_ne_promeut_rien(
    artefacts_isoles: Path, gold_synthetique: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(config.CIBLES_PERFORMANCE, "pr_auc_min", 1.01)
    with pytest.raises(typer.Exit) as erreur:
        cli.train_cmd(forcer_gold=False)
    assert erreur.value.exit_code == 1
    assert not (artefacts_isoles / "best_model.pkl").exists()


def test_commande_retrain_supprimee() -> None:
    """``retrain`` écrivait un artefact qu'aucun service ne lisait (best_model_final.pkl)."""
    assert "retrain" not in {c.name for c in cli.app.registered_commands}


def _ecrire_modele_et_entree(dossier: Path, gold: tuple[pd.DataFrame, pd.Series]) -> Path:
    X, y = gold
    modele = Pipeline([("echelle", StandardScaler()), ("lr", LogisticRegression())]).fit(X, y)
    joblib.dump(modele, dossier / "best_model.pkl")
    entree = dossier / "entree.parquet"
    X.assign(client_id=[f"C{i:04d}" for i in range(len(X))]).to_parquet(entree)
    return entree


def test_predict_sans_meta_applique_le_seuil_de_vigilance(
    artefacts_isoles: Path, gold_synthetique: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    entree = _ecrire_modele_et_entree(artefacts_isoles, gold_synthetique)
    monkeypatch.setattr(cli, "_preparer_features", lambda df: df.drop(columns=["client_id"]))
    monkeypatch.setattr(cli.economie, "seuil_surveillance_par_defaut", lambda: 0.2774)
    sortie = artefacts_isoles / "predictions.csv"
    cli.predict_cmd(entree, sortie)
    assert pd.read_csv(sortie)["seuil_applique"].unique().tolist() == [pytest.approx(0.2774)]


def test_predict_avec_meta_applique_le_seuil_du_modele(
    artefacts_isoles: Path, gold_synthetique: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    entree = _ecrire_modele_et_entree(artefacts_isoles, gold_synthetique)
    (artefacts_isoles / "best_model_meta.json").write_text(
        json.dumps({"seuil_economique": 0.31}), encoding="utf-8"
    )
    monkeypatch.setattr(cli, "_preparer_features", lambda df: df.drop(columns=["client_id"]))
    monkeypatch.setattr(cli.economie, "seuil_surveillance_par_defaut", lambda: 0.2774)
    sortie = artefacts_isoles / "predictions.csv"
    cli.predict_cmd(entree, sortie)
    assert pd.read_csv(sortie)["seuil_applique"].unique().tolist() == [pytest.approx(0.31)]


# ---------------------------------------------------------------------------
# Retour arrière en une commande
# ---------------------------------------------------------------------------


def test_rollback_sans_argument_liste_les_archives(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "lister_archives", lambda: ["20261001_020000", "20261010_020000"])
    cli.rollback_cmd(horodatage=None)
    sortie = capsys.readouterr().out
    assert "20261001_020000" in sortie and "20261010_020000" in sortie


def test_rollback_restaure_l_archive_demandee(monkeypatch: pytest.MonkeyPatch) -> None:
    demandes: list[str] = []

    def _restaurer(horodatage: str) -> dict[str, object]:
        demandes.append(horodatage)
        return {
            "horodatage": horodatage,
            "archive_du_champion_retire": "20261011_090000",
            "meta_restauree": True,
            "version_registre": "12",
            "alias_deplace": True,
        }

    monkeypatch.setattr(cli, "restaurer_archive", _restaurer)
    cli.rollback_cmd(horodatage="20261001_020000")
    assert demandes == ["20261001_020000"]


def test_rollback_archive_inconnue_sort_en_erreur(monkeypatch: pytest.MonkeyPatch) -> None:
    def _absente(horodatage: str) -> dict[str, object]:
        raise FileNotFoundError(horodatage)

    monkeypatch.setattr(cli, "restaurer_archive", _absente)
    with pytest.raises(typer.Exit) as erreur:
        cli.rollback_cmd(horodatage="20990101_000000")
    assert erreur.value.exit_code == 1
