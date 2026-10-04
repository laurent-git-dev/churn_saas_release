"""Tests de sécurité de l'API — codes 401, 429, 413."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

import churn_saas.api.security as sec
from churn_saas.api.main import app
from churn_saas.api.model_store import get_model_store

# ---------------------------------------------------------------------------
# Stub modèle (nécessaire pour que /predict ne retourne pas 503)
# ---------------------------------------------------------------------------


class _MockStoreReady:
    est_pret = True
    seuil = 0.40

    def predire(self, X):
        n = len(X)
        return np.array([[0.60, 0.40]] * n, dtype=float)

    def predire_et_expliquer(self, X, nb_facteurs=3):
        return self.predire(X), [[] for _ in range(len(X))]


_EXEMPLE: dict[str, Any] = {
    "anciennete_mois": 12,
    "sieges_souscrits": 10,
    "utilisateurs_actifs": 5,
    "taux_adoption_pct": 60.0,
    "connexions_30j": 20,
    "fonctionnalites_total": 20,
    "fonctionnalites_utilisees": 10,
    "nb_integrations": 2,
    "derniere_connexion_jours": 5,
    "tickets_support_90j": 0,
    "retards_paiement_12m": 0,
    "revenu_mensuel_recurrent_eur": 1000.0,
    "secteur": "Technologie",
    "pays": "France",
    "taille_entreprise": "PME",
    "plan": "Starter",
}


@pytest.fixture(autouse=True)
def _reinitialiser_rate_limiter():
    """Remet l'historique de débit à zéro avant chaque test."""
    sec._historique.clear()
    yield
    sec._historique.clear()


@pytest.fixture
def client(monkeypatch):
    """TestClient avec modèle chargé et auth désactivée."""
    monkeypatch.delenv("CHURN_API_KEY", raising=False)
    app.dependency_overrides[get_model_store] = lambda: _MockStoreReady()
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def client_avec_auth(monkeypatch):
    """TestClient avec modèle chargé et clé d'API configurée."""
    monkeypatch.setenv("CHURN_API_KEY", "cle-secrete-test")
    app.dependency_overrides[get_model_store] = lambda: _MockStoreReady()
    yield TestClient(app)
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Tests authentification (401)
# ---------------------------------------------------------------------------


def test_predict_sans_cle_api_401(client_avec_auth):
    """Sans X-API-Key quand la clé est configurée → 401."""
    resp = client_avec_auth.post("/predict", json=_EXEMPLE)
    assert resp.status_code == 401


def test_predict_mauvaise_cle_api_401(client_avec_auth):
    """Avec une mauvaise X-API-Key → 401."""
    resp = client_avec_auth.post(
        "/predict",
        json=_EXEMPLE,
        headers={"X-API-Key": "mauvaise-cle"},
    )
    assert resp.status_code == 401


def test_predict_bonne_cle_api_200(client_avec_auth):
    """Avec la bonne X-API-Key → 200."""
    resp = client_avec_auth.post(
        "/predict",
        json=_EXEMPLE,
        headers={"X-API-Key": "cle-secrete-test"},
    )
    assert resp.status_code == 200


def test_health_sans_auth_200(client_avec_auth):
    """/health ne requiert pas d'authentification."""
    resp = client_avec_auth.get("/health")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Tests limitation de débit (429)
# ---------------------------------------------------------------------------


def test_predict_depasse_rate_limit_429(monkeypatch, client):
    """Dépasser CHURN_RATE_LIMIT requêtes → 429."""
    monkeypatch.setenv("CHURN_RATE_LIMIT", "2")
    # 2 requêtes acceptées
    r1 = client.post("/predict", json=_EXEMPLE)
    r2 = client.post("/predict", json=_EXEMPLE)
    assert r1.status_code == 200
    assert r2.status_code == 200
    # 3e requête → 429
    r3 = client.post("/predict", json=_EXEMPLE)
    assert r3.status_code == 429


def test_429_contient_retry_after(monkeypatch, client):
    """La réponse 429 doit inclure l'en-tête Retry-After."""
    monkeypatch.setenv("CHURN_RATE_LIMIT", "1")
    client.post("/predict", json=_EXEMPLE)
    resp = client.post("/predict", json=_EXEMPLE)
    assert resp.status_code == 429
    assert "retry-after" in {h.lower() for h in resp.headers}


# ---------------------------------------------------------------------------
# Tests taille du corps (413)
# ---------------------------------------------------------------------------


def test_corps_trop_grand_413(monkeypatch, client):
    """Corps dépassant CHURN_MAX_BODY_BYTES → 413."""
    monkeypatch.setenv("CHURN_MAX_BODY_BYTES", "50")
    resp = client.post("/predict", json=_EXEMPLE)
    assert resp.status_code == 413


def test_corps_dans_les_limites_200(monkeypatch, client):
    """Corps dans les limites → 200 (pas de 413)."""
    monkeypatch.setenv("CHURN_MAX_BODY_BYTES", str(1 * 1024 * 1024))
    resp = client.post("/predict", json=_EXEMPLE)
    assert resp.status_code == 200
