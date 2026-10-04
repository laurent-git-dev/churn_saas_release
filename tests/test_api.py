"""Tests de l'API de prédiction de churn — codes 200, 422, 503."""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from churn_saas import config, economie
from churn_saas.api import main as api_main
from churn_saas.api.main import app
from churn_saas.api.model_store import ModelStore, get_model_store
from churn_saas.api.schemas import DemandePrediction

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_EXEMPLE_VALIDE: dict[str, Any] = {
    "anciennete_mois": 18,
    "sieges_souscrits": 25,
    "utilisateurs_actifs": 12,
    "taux_adoption_pct": 48.0,
    "connexions_30j": 45,
    "heures_usage_30j": 120.5,
    "fonctionnalites_total": 30,
    "fonctionnalites_utilisees": 10,
    "nb_integrations": 1,
    "derniere_connexion_jours": 35,
    "tickets_support_90j": 8,
    "delai_reponse_support_h": 24.0,
    "csat": 3.0,
    "retards_paiement_12m": 2,
    "revenu_mensuel_recurrent_eur": 2500.0,
    "secteur": "Finance",
    "pays": "France",
    "taille_entreprise": "PME",
    "plan": "Business",
}


class _MockStoreReady:
    """Stub de ModelStore avec modèle chargé — retourne probabilité fixe 0.72."""

    est_pret = True
    seuil = 0.40
    nom_modele = "LightGBM"

    def predire(self, X):
        n = len(X)
        return np.array([[0.28, 0.72]] * n, dtype=float)

    def predire_et_expliquer(self, X, nb_facteurs=3):
        facteur = {
            "variable": "csat",
            "libelle": "Satisfaction client (CSAT, sur 5) : 3",
            "contribution": 0.42,
        }
        return self.predire(X), [[facteur] for _ in range(len(X))]


class _MockStoreVide:
    """Stub de ModelStore sans modèle — simule l'état avant `churn-saas train`."""

    est_pret = False
    seuil = 0.40
    nom_modele = None

    def predire(self, X):
        raise RuntimeError("Aucun modèle chargé.")

    def predire_et_expliquer(self, X, nb_facteurs=3):
        raise RuntimeError("Aucun modèle chargé.")


@pytest.fixture
def client_pret():
    """TestClient avec modèle chargé (mock)."""
    app.dependency_overrides[get_model_store] = lambda: _MockStoreReady()
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def client_vide():
    """TestClient sans modèle chargé (→ 503 sur /predict et /ready)."""
    app.dependency_overrides[get_model_store] = lambda: _MockStoreVide()
    yield TestClient(app)
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Tests /health
# ---------------------------------------------------------------------------


def test_health_200(client_pret):
    resp = client_pret.get("/health")
    assert resp.status_code == 200
    corps = resp.json()
    assert corps["statut"] == "ok"
    assert "version" in corps


def test_health_sans_modele_200(client_vide):
    """Liveness : /health doit répondre 200 même sans modèle."""
    resp = client_vide.get("/health")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Tests /ready
# ---------------------------------------------------------------------------


def test_ready_200_avec_modele(client_pret):
    resp = client_pret.get("/ready")
    assert resp.status_code == 200
    corps = resp.json()
    assert corps["modele_charge"] is True
    # La famille du modèle servi est exposée : un champion LightGBM est visible sans log
    assert corps["modele"] == "LightGBM"


def test_ready_503_sans_modele(client_vide):
    resp = client_vide.get("/ready")
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# Tests /predict
# ---------------------------------------------------------------------------


def test_predict_200(client_pret):
    resp = client_pret.post("/predict", json=_EXEMPLE_VALIDE)
    assert resp.status_code == 200
    corps = resp.json()
    assert 0.0 <= corps["probabilite_churn"] <= 1.0
    assert corps["decision"] in {"ALERTE_ROUGE", "SURVEILLANCE", "OK"}
    assert isinstance(corps["facteurs_principaux"], list)
    assert corps["valeur_a_risque_eur"] >= 0.0


def test_predict_renvoie_les_facteurs_shap_du_model_store(client_pret):
    corps = client_pret.post("/predict", json=_EXEMPLE_VALIDE).json()
    assert corps["facteurs_shap"] == [
        {
            "variable": "csat",
            "libelle": "Satisfaction client (CSAT, sur 5) : 3",
            "contribution": 0.42,
        }
    ]


def test_predict_batch_un_jeu_de_facteurs_par_compte(client_pret):
    corps = client_pret.post("/predict-batch", json=[_EXEMPLE_VALIDE] * 3).json()
    assert [len(p["facteurs_shap"]) for p in corps["predictions"]] == [1, 1, 1]


def test_predict_decision_alerte_rouge(client_pret):
    """Avec probabilité 0.72 et seuil 0.40, la décision doit être ALERTE_ROUGE."""
    resp = client_pret.post("/predict", json=_EXEMPLE_VALIDE)
    assert resp.status_code == 200
    assert resp.json()["decision"] == "ALERTE_ROUGE"


def test_predict_422_taux_adoption_hors_bornes(client_pret):
    donnees_invalides = {**_EXEMPLE_VALIDE, "taux_adoption_pct": 150.0}
    resp = client_pret.post("/predict", json=donnees_invalides)
    assert resp.status_code == 422


def test_predict_422_anciennete_negative(client_pret):
    donnees_invalides = {**_EXEMPLE_VALIDE, "anciennete_mois": -5}
    resp = client_pret.post("/predict", json=donnees_invalides)
    assert resp.status_code == 422


@pytest.mark.parametrize("csat", [0.0, 6.0, 11.0])
def test_predict_422_csat_hors_echelle(client_pret, csat):
    """Le CSAT est noté de 1 à 5 à l'entraînement : toute autre note serait extrapolée."""
    donnees_invalides = {**_EXEMPLE_VALIDE, "csat": csat}
    resp = client_pret.post("/predict", json=donnees_invalides)
    assert resp.status_code == 422


def test_predict_503_sans_modele(client_vide):
    resp = client_vide.post("/predict", json=_EXEMPLE_VALIDE)
    assert resp.status_code == 503


def test_predict_champs_optionnels_none(client_pret):
    """Les champs optionnels à None doivent être acceptés (imputation pipeline)."""
    donnees = {**_EXEMPLE_VALIDE, "csat": None, "heures_usage_30j": None}
    resp = client_pret.post("/predict", json=donnees)
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Tests /predict-batch
# ---------------------------------------------------------------------------


def test_predict_batch_200(client_pret):
    demandes = [_EXEMPLE_VALIDE, _EXEMPLE_VALIDE]
    resp = client_pret.post("/predict-batch", json=demandes)
    assert resp.status_code == 200
    corps = resp.json()
    assert corps["nb_comptes"] == 2
    assert len(corps["predictions"]) == 2
    assert corps["nb_alertes_rouges"] + corps["nb_surveillances"] + corps["nb_ok"] == 2


def test_predict_batch_liste_vide_422(client_pret):
    resp = client_pret.post("/predict-batch", json=[])
    assert resp.status_code == 422


def test_predict_batch_503_sans_modele(client_vide):
    resp = client_vide.post("/predict-batch", json=[_EXEMPLE_VALIDE])
    assert resp.status_code == 503


def test_predict_batch_un_compte(client_pret):
    resp = client_pret.post("/predict-batch", json=[_EXEMPLE_VALIDE])
    assert resp.status_code == 200
    assert resp.json()["nb_comptes"] == 1


# ---------------------------------------------------------------------------
# Tests données de référence absentes — le service est indisponible, pas dégradé
# ---------------------------------------------------------------------------


def test_ready_503_sans_catalogue(client_pret, monkeypatch, tmp_path):
    """Modèle chargé mais catalogue des plans absent → le service n'est pas « prêt »."""
    monkeypatch.setattr(api_main, "_CHEMIN_CATALOGUE", tmp_path / "catalogue_absent.csv")
    api_main._catalogue.cache_clear()
    resp = client_pret.get("/ready")
    assert resp.status_code == 503
    assert "référence" in resp.json()["detail"]


def test_predict_503_sans_catalogue(client_pret, monkeypatch, tmp_path):
    """Sans catalogue, l'enrichissement serait vide : on refuse au lieu de scorer faux."""
    monkeypatch.setattr(api_main, "_CHEMIN_CATALOGUE", tmp_path / "catalogue_absent.csv")
    api_main._catalogue.cache_clear()
    resp = client_pret.post("/predict", json=_EXEMPLE_VALIDE)
    assert resp.status_code == 503


def test_predict_batch_503_sans_catalogue(client_pret, monkeypatch, tmp_path):
    """Même refus côté batch."""
    monkeypatch.setattr(api_main, "_CHEMIN_CATALOGUE", tmp_path / "catalogue_absent.csv")
    api_main._catalogue.cache_clear()
    resp = client_pret.post("/predict-batch", json=[_EXEMPLE_VALIDE])
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# Features calculées côté serveur — jamais demandées au client
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "feature",
    ["intensite_support", "taux_utilisation_sieges", "arpu_par_siege", "remise_consentie"],
)
def test_feature_derivee_absente_du_contrat_d_entree(feature):
    """Une feature dérivée n'est jamais un champ de la demande : le client ne peut pas la fausser."""
    assert feature not in DemandePrediction.model_fields


def test_intensite_support_calculee_cote_serveur():
    """`intensite_support` = tickets sur 90 j / mois d'ancienneté, recalculée par l'API."""
    df = api_main._demande_vers_dataframe([DemandePrediction(**_EXEMPLE_VALIDE)])
    attendu = _EXEMPLE_VALIDE["tickets_support_90j"] / _EXEMPLE_VALIDE["anciennete_mois"]
    assert df["intensite_support"].iloc[0] == pytest.approx(attendu)


def test_intensite_support_nan_si_anciennete_nulle():
    """Ancienneté nulle : NaN (imputé par le pipeline), jamais une division par zéro."""
    df = api_main._demande_vers_dataframe(
        [DemandePrediction(**{**_EXEMPLE_VALIDE, "anciennete_mois": 0})]
    )
    assert np.isnan(df["intensite_support"].iloc[0])


def test_import_api_sans_dependances_d_entrainement():
    """L'image Docker n'installe que l'extra `api` : l'API ne doit importer ni MLflow ni Optuna.

    Un import indirect de `churn_saas.models.train` ferait planter le conteneur au démarrage,
    alors que tous les tests (environnement complet) resteraient verts.
    """
    code = (
        "import sys, churn_saas.api.main; "
        "interdits = {'mlflow', 'optuna', 'imblearn', 'shap', 'churn_saas.models.train'}; "
        "print(sorted(interdits & set(sys.modules)))"
    )
    resultat = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert resultat.stdout.strip().splitlines()[-1] == "[]"


# ---------------------------------------------------------------------------
# ModelStore — seuil et famille du champion
# ---------------------------------------------------------------------------


@pytest.fixture
def pipeline_lightgbm(tmp_path):
    """Petit pipeline LightGBM sérialisé, sans métadonnées."""
    rng = np.random.default_rng(config.RANDOM_SEED)
    X = pd.DataFrame({"a": rng.normal(size=60), "b": rng.normal(size=60)})
    y = (X["a"] > 0).astype(int)
    pipeline = Pipeline([("pre", StandardScaler()), ("clf", LGBMClassifier(verbose=-1))])
    chemin = tmp_path / "best_model.pkl"
    joblib.dump(pipeline.fit(X, y), chemin)
    return chemin


def test_model_store_seuil_par_defaut_est_le_seuil_de_vigilance(pipeline_lightgbm, tmp_path):
    """Sans métadonnées, le seuil servi est celui de la règle à deux niveaux (recall cible)."""
    config.TABLES.mkdir(parents=True, exist_ok=True)
    (config.TABLES / economie.ARTEFACT_SEUIL_VIGILANCE).write_text(
        json.dumps({"seuil": 0.2773}), encoding="utf-8"
    )
    store = ModelStore()
    store.charger(pipeline_lightgbm, tmp_path / "meta_absente.json")
    assert store.seuil == pytest.approx(0.2773)


def test_model_store_seuil_des_metadonnees_prioritaire(pipeline_lightgbm, tmp_path):
    """Un modèle promu embarque son seuil : il prime sur l'artefact du notebook."""
    meta = tmp_path / "best_model_meta.json"
    meta.write_text(json.dumps({"seuil_economique": 0.31, "modele_nom": "LightGBM"}))
    store = ModelStore()
    store.charger(pipeline_lightgbm, meta)
    assert store.seuil == pytest.approx(0.31)
    assert store.nom_modele == "LightGBM"


def test_model_store_famille_deduite_du_pipeline(pipeline_lightgbm, tmp_path):
    """Sans métadonnées, la famille est lue sur le dernier pas du pipeline."""
    store = ModelStore()
    store.charger(pipeline_lightgbm, tmp_path / "meta_absente.json")
    assert store.nom_modele == "LGBMClassifier"
    assert store.predire(pd.DataFrame({"a": [1.0], "b": [0.0]})).shape == (1, 2)


def test_model_store_sans_modele():
    """Avant chargement : pas prêt, pas de famille."""
    store = ModelStore()
    assert not store.est_pret
    assert store.nom_modele is None
