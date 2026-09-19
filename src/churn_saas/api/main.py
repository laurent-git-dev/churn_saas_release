"""Application FastAPI — API de prédiction de churn SaaS (C6)."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, status
from prometheus_fastapi_instrumentator import Instrumentator

from churn_saas import config
from churn_saas.api.model_store import ModelStore, get_model_store
from churn_saas.api.schemas import (
    DemandePredicton,
    EtatPret,
    EtatSante,
    ResultatBatch,
    ResultatPrediction,
)
from churn_saas.api.security import (
    LimiteCorpsMiddleware,
    verifier_cle_api,
    verifier_rate_limit,
)
from churn_saas.features.build import ajouter_features_metier

_VERSION = "0.1.0"
_MAX_BATCH = 1_000


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    get_model_store().charger()
    yield


app = FastAPI(
    title="Churn SaaS — API de prédiction",
    description=(
        "Prédit la probabilité de résiliation d'un compte SaaS B2B. "
        "Retourne la probabilité, la valeur à risque (€), la décision recommandée "
        "et les facteurs de risque identifiés. "
        "Authentification via l'en-tête **X-API-Key** (clé env `CHURN_API_KEY`)."
    ),
    version=_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(LimiteCorpsMiddleware)

Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


# ---------------------------------------------------------------------------
# Helpers métier
# ---------------------------------------------------------------------------


def _demande_vers_dataframe(demandes: list[DemandePredicton]) -> pd.DataFrame:
    """Convertit une liste de demandes en DataFrame enrichi par ajouter_features_metier."""
    donnees = [d.model_dump() for d in demandes]
    df = pd.DataFrame(donnees)
    # Enrichissement des features dérivées — même logique que la construction gold
    df = ajouter_features_metier(df, csat_median_par_secteur=None)
    return df


def _extraire_facteurs(demande: DemandePredicton) -> list[str]:
    """Identifie les signaux de risque churn sur la base de règles métier documentées.

    Ces règles proviennent de l'EDA (§6) — jamais inventées a posteriori.
    Elles sont complémentaires aux SHAP values calculées en §11 (notebook).
    """
    facteurs: list[str] = []
    if demande.derniere_connexion_jours > 30:
        facteurs.append(f"inactivité élevée ({demande.derniere_connexion_jours} j sans connexion)")
    if demande.csat is not None and demande.csat <= 6:
        facteurs.append(f"satisfaction faible (CSAT = {demande.csat:.1f}/10)")
    if demande.retards_paiement_12m > 0:
        facteurs.append(f"retards de paiement ({demande.retards_paiement_12m} sur 12 mois)")
    if demande.taux_adoption_pct < 30:
        facteurs.append(f"adoption faible ({demande.taux_adoption_pct:.0f} %)")
    if demande.tickets_support_90j > 5:
        facteurs.append(f"volume élevé de tickets ({demande.tickets_support_90j} sur 90 jours)")
    if demande.fonctionnalites_total > 0:
        frac = demande.fonctionnalites_utilisees / demande.fonctionnalites_total
        if frac < 0.33:
            facteurs.append(f"couverture fonctionnelle faible ({frac:.0%})")
    return facteurs[:3] if facteurs else ["aucun signal négatif identifié"]


def _valeur_a_risque(demande: DemandePredicton, probabilite: float) -> float:
    horizon: int = int(config.HYPOTHESES_ECONOMIQUES.get("horizon_mois", 12))
    return round(probabilite * demande.revenu_mensuel_recurrent_eur * horizon, 2)


def _decision(probabilite: float, seuil: float) -> Literal["ALERTE_ROUGE", "SURVEILLANCE", "OK"]:
    if probabilite >= max(seuil + 0.20, 0.60):
        return "ALERTE_ROUGE"
    if probabilite >= seuil:
        return "SURVEILLANCE"
    return "OK"


def _construire_resultat(
    demande: DemandePredicton,
    probabilite: float,
    seuil: float,
) -> ResultatPrediction:
    return ResultatPrediction(
        probabilite_churn=round(probabilite, 4),
        valeur_a_risque_eur=_valeur_a_risque(demande, probabilite),
        decision=_decision(probabilite, seuil),
        facteurs_principaux=_extraire_facteurs(demande),
        seuil_applique=seuil,
    )


# ---------------------------------------------------------------------------
# Routes monitoring
# ---------------------------------------------------------------------------


@app.get("/health", response_model=EtatSante, tags=["monitoring"])
async def health() -> dict[str, Any]:
    """Sonde liveness — répond toujours 200, même sans modèle chargé."""
    return {"statut": "ok", "version": _VERSION}


@app.get("/ready", response_model=EtatPret, tags=["monitoring"])
async def ready(
    store: Annotated[ModelStore, Depends(get_model_store)],
) -> dict[str, Any]:
    """Sonde readiness — retourne 503 si le modèle n'est pas encore chargé."""
    if not store.est_pret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modèle non chargé. Exécutez `churn-saas train` pour entraîner.",
        )
    return {
        "statut": "pret",
        "modele_charge": True,
        "message": "Modèle opérationnel.",
    }


# ---------------------------------------------------------------------------
# Routes de prédiction
# ---------------------------------------------------------------------------


@app.post(
    "/predict",
    response_model=ResultatPrediction,
    dependencies=[Depends(verifier_cle_api), Depends(verifier_rate_limit)],
    tags=["prédiction"],
    summary="Prédiction unitaire — un compte client",
)
async def predict(
    demande: DemandePredicton,
    store: Annotated[ModelStore, Depends(get_model_store)],
) -> ResultatPrediction:
    """Prédit la probabilité de churn pour **un** compte client.

    - **422** : données d'entrée invalides (bornes, types).
    - **401** : clé d'API manquante ou incorrecte.
    - **429** : quota dépassé (60 req/min par IP).
    - **503** : modèle non disponible (exécuter `churn-saas train`).
    """
    if not store.est_pret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modèle non disponible.",
        )
    X = _demande_vers_dataframe([demande])
    probas = store.predire(X)
    return _construire_resultat(demande, float(probas[0, 1]), store.seuil)


@app.post(
    "/predict-batch",
    response_model=ResultatBatch,
    dependencies=[Depends(verifier_cle_api), Depends(verifier_rate_limit)],
    tags=["prédiction"],
    summary=f"Prédiction batch — jusqu'à {_MAX_BATCH} comptes",
)
async def predict_batch(
    demandes: list[DemandePredicton],
    store: Annotated[ModelStore, Depends(get_model_store)],
) -> ResultatBatch:
    """Prédit le churn pour une liste de comptes en une seule requête (max 1 000).

    Retourne le décompte par catégorie de décision pour faciliter la priorisation CS.
    """
    if not store.est_pret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modèle non disponible.",
        )
    if not demandes:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="La liste de demandes ne peut pas être vide.",
        )
    if len(demandes) > _MAX_BATCH:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Lot trop grand : {_MAX_BATCH} comptes maximum par requête.",
        )

    X = _demande_vers_dataframe(demandes)
    probas = store.predire(X)[:, 1]

    predictions = [
        _construire_resultat(d, float(p), store.seuil)
        for d, p in zip(demandes, probas, strict=False)
    ]
    decisions = [r.decision for r in predictions]
    return ResultatBatch(
        nb_comptes=len(predictions),
        predictions=predictions,
        nb_alertes_rouges=decisions.count("ALERTE_ROUGE"),
        nb_surveillances=decisions.count("SURVEILLANCE"),
        nb_ok=decisions.count("OK"),
    )
