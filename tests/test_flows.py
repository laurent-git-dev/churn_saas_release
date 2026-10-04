"""Flow de réentraînement — un champion LightGBM se réentraîne et se promeut comme la LR.

Gold synthétique (jamais `data/`) : colonnes du contrat d'API plus la cible. Le flow est appelé
étape par étape via `.fn` (la fonction sous la task Prefect, sans moteur d'orchestration), sans
MLflow (journalisation neutralisée), artefacts dans `tmp_path`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from flows import retraining
from lightgbm import LGBMClassifier
from prefect.testing.utilities import prefect_test_harness

from churn_saas import config
from churn_saas.features.build import ajouter_features_metier
from churn_saas.models import train

_PARAMS_LIGHTGBM: dict[str, Any] = {"num_leaves": 7, "n_estimators": 30, "learning_rate": 0.1}


def _gold_synthetique(n: int = 400) -> pd.DataFrame:
    """Gold de même schéma que les champs de l'API ; churn porté par l'inactivité."""
    rng = np.random.default_rng(config.RANDOM_SEED)
    sieges = rng.integers(5, 100, n)
    inactivite = rng.integers(0, 90, n)
    df = pd.DataFrame(
        {
            "anciennete_mois": rng.integers(1, 60, n),
            "sieges_souscrits": sieges,
            "utilisateurs_actifs": (sieges * rng.uniform(0.1, 1.0, n)).astype(int),
            "connexions_30j": rng.integers(0, 300, n),
            "heures_usage_30j": rng.uniform(0, 400, n),
            "fonctionnalites_total": 30,
            "fonctionnalites_utilisees": rng.integers(1, 30, n),
            "nb_integrations": rng.integers(0, 10, n),
            "derniere_connexion_jours": inactivite,
            "tickets_support_90j": rng.integers(0, 15, n),
            "delai_reponse_support_h": rng.uniform(1, 72, n),
            "csat": rng.uniform(1, 5, n),
            "retards_paiement_12m": rng.integers(0, 4, n),
            "revenu_mensuel_recurrent_eur": rng.uniform(200, 9_000, n),
            "secteur": rng.choice(["Finance", "Santé", "Industrie"], n),
            "pays": rng.choice(["France", "Belgique"], n),
            "taille_entreprise": rng.choice(["PME", "ETI"], n),
            "plan": rng.choice(["Starter", "Pro", "Business"], n),
        }
    )
    df["churn"] = (inactivite + rng.normal(0, 15, n) > 45).astype(int)
    return df


@pytest.fixture
def flow_lightgbm(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Gold synthétique, artefacts isolés, champion courant de la famille LightGBM."""
    gold = tmp_path / "gold"
    gold.mkdir()
    _gold_synthetique().to_parquet(gold / "gold_dataset.parquet", index=False)
    monkeypatch.setattr(config, "DONNEES_GOLD", gold)
    monkeypatch.setattr(config, "ARTIFACTS", tmp_path / "artifacts")
    monkeypatch.setattr(
        train,
        "famille_et_hyperparametres_champion",
        lambda: (train.NOM_LIGHTGBM, dict(_PARAMS_LIGHTGBM)),
    )

    def _sans_mlflow(*_args: Any, **_kwargs: Any) -> str:
        raise RuntimeError("MLflow neutralisé en test")

    monkeypatch.setattr(train, "journaliser_mlflow", _sans_mlflow)
    return tmp_path / "artifacts" / "models"


def test_challenger_lightgbm_entraine_avec_seuil_de_vigilance(flow_lightgbm: Path) -> None:
    """Le challenger reprend la famille LightGBM et embarque son seuil de vigilance OOF."""
    pipeline, metriques = retraining.stage_train.fn(forcer=True)

    assert isinstance(pipeline[-1], LGBMClassifier)
    assert 0.0 <= metriques["pr_auc_test"] <= 1.0
    meta = json.loads((flow_lightgbm / "challenger_meta.json").read_text(encoding="utf-8"))
    assert meta["modele_nom"] == train.NOM_LIGHTGBM
    assert meta["hyperparametres"] == _PARAMS_LIGHTGBM
    assert 0.0 < meta["seuil_vigilance"] < 1.0
    assert meta["recall_vigilance_oof"] >= config.RECALL_CIBLE_VIGILANCE


def test_promotion_transmet_famille_et_seuil(flow_lightgbm: Path) -> None:
    """Le champion promu porte sa famille et son seuil : l'API et le batch les relisent."""
    pipeline, metriques = retraining.stage_train.fn(forcer=True)
    retraining.stage_promote.fn(pipeline, metriques)

    meta_challenger = json.loads((flow_lightgbm / "challenger_meta.json").read_text("utf-8"))
    meta = json.loads((flow_lightgbm / "best_model_meta.json").read_text(encoding="utf-8"))
    assert (flow_lightgbm / "best_model.pkl").exists()
    assert meta["modele_nom"] == train.NOM_LIGHTGBM
    assert meta["seuil_economique"] == pytest.approx(meta_challenger["seuil_vigilance"])


def test_batch_score_avec_champion_lightgbm_promu(flow_lightgbm: Path) -> None:
    """Après promotion d'un LightGBM, le batch nocturne le sert avec le seuil promu."""
    from flows import scoring_batch

    pipeline, metriques = retraining.stage_train.fn(forcer=True)
    retraining.stage_promote.fn(pipeline, metriques)
    meta = json.loads((flow_lightgbm / "best_model_meta.json").read_text(encoding="utf-8"))

    # Le vrai gold porte déjà les features dérivées : on les ajoute au gold synthétique
    df = ajouter_features_metier(pd.read_parquet(config.DONNEES_GOLD / "gold_dataset.parquet"))
    df_inference = df.drop(columns=["churn"])
    df_inference["_client_id"] = range(len(df))
    df_inference["_mrr_eur"] = df["revenu_mensuel_recurrent_eur"]
    resultats = scoring_batch.scorer_comptes.fn(df_inference)

    assert len(resultats) == len(df)
    assert resultats["seuil_applique"].unique().tolist() == [
        pytest.approx(meta["seuil_economique"])
    ]
    assert resultats["probabilite_churn"].between(0, 1).all()


def test_batch_publie_les_trois_facteurs_shap(flow_lightgbm: Path) -> None:
    """L'alerte du matin (§2, CU2) affiche les 3 variables SHAP : le batch doit les publier."""
    from flows import scoring_batch

    pipeline, metriques = retraining.stage_train.fn(forcer=True)
    retraining.stage_promote.fn(pipeline, metriques)
    df = ajouter_features_metier(pd.read_parquet(config.DONNEES_GOLD / "gold_dataset.parquet"))
    df_inference = df.drop(columns=["churn"])
    df_inference["_client_id"] = range(len(df))
    df_inference["_mrr_eur"] = df["revenu_mensuel_recurrent_eur"]
    resultats = scoring_batch.scorer_comptes.fn(df_inference)

    colonnes = [f"facteur_shap_{i}" for i in (1, 2, 3)]
    assert set(colonnes) <= set(resultats.columns)
    # Un compte à risque a au moins une raison ; un libellé est métier, pas une colonne
    a_risque = resultats["decision"] != "OK"
    assert resultats.loc[a_risque, "facteur_shap_1"].notna().all()
    libelles = resultats[colonnes].stack()
    assert not libelles.str.contains("_").any()


@pytest.mark.slow
def test_flow_prefect_complet_promeut_le_challenger(
    flow_lightgbm: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Le flow tourne sous le moteur Prefect (tasks, relances) et promeut si la gate passe."""
    brutes = tmp_path / "raw"
    brutes.mkdir()
    for nom in ("churn_saas_complet.csv", "catalogue_plans.csv"):
        (brutes / nom).write_text("x\n", encoding="utf-8")
    monkeypatch.setattr(config, "DONNEES_BRUTES", brutes)
    monkeypatch.setattr(config, "TABLES", tmp_path / "tables")
    # Seuil neutralisé : on teste l'orchestration, pas la qualité du jeu synthétique
    monkeypatch.setattr(
        config, "CIBLES_PERFORMANCE", config.CIBLES_PERFORMANCE | {"pr_auc_min": 0.0}
    )

    # Le gold synthétique est déjà en place : l'étape features n'a rien à reconstruire
    @retraining.task(name="features")
    def _features_synthetiques(forcer: bool = False) -> Path:
        return config.DONNEES_GOLD / "gold_dataset.parquet"

    monkeypatch.setattr(retraining, "stage_features", _features_synthetiques)

    # Serveur Prefect temporaire démarré et arrêté dans le test : base isolée, pas de trace
    with prefect_test_harness():
        assert retraining.executer(forcer=True) == 0
    assert (flow_lightgbm / "best_model.pkl").exists()
