"""Tests de l'API de prédiction de churn — codes 200, 422, 503."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from churn_saas.api.main import app
from churn_saas.api.model_store import get_model_store

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
    "csat": 5.5,
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

    def predire(self, X):
        n = len(X)
        return np.array([[0.28, 0.72]] * n, dtype=float)


class _MockStoreVide:
    """Stub de ModelStore sans modèle — simule l'état avant `churn-saas train`."""

    est_pret = False
    seuil = 0.40

    def predire(self, X):
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


def test_predict_422_csat_hors_echelle(client_pret):
    donnees_invalides = {**_EXEMPLE_VALIDE, "csat": 11.0}
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
