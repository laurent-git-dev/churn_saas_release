"""Application FastAPI — API de prédiction de churn SaaS (C6)."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Annotated, Any, Literal

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, status
from loguru import logger
from prometheus_client import Counter
from prometheus_fastapi_instrumentator import Instrumentator

from churn_saas import config, economie
from churn_saas.api.model_store import ModelStore, get_model_store
from churn_saas.api.schemas import (
    DemandePredicton,
    EtatPret,
    EtatSante,
    ResultatBatch,
    ResultatPrediction,
    champs_nullables,
)
from churn_saas.api.security import (
    LimiteCorpsMiddleware,
    verifier_cle_api,
    verifier_rate_limit,
)
from churn_saas.features.build import ajouter_features_metier, joindre_catalogue
from churn_saas.features.enrichissement import enrichir_par_pays, enrichir_par_secteur

_VERSION = "0.1.0"
_MAX_BATCH = 1_000

_CHEMIN_CATALOGUE = config.DONNEES_BRUTES / "catalogue_plans.csv"


@lru_cache(maxsize=1)
def _catalogue() -> pd.DataFrame:
    """Catalogue des plans — chargé à la première prédiction, jamais à l'import.

    `data/raw/` fait partie des livrables mais n'est pas versionné : lire ce CSV au
    chargement du module rendait `churn_saas.api.main` — et donc tous les tests qui
    l'importent — impossible à importer en CI. Le chargement est différé et mémoïsé ;
    l'absence du fichier devient un 503 explicite au lieu d'un crash à l'import.
    """
    if not _CHEMIN_CATALOGUE.exists():
        raise FileNotFoundError(
            f"Catalogue des plans introuvable : {_CHEMIN_CATALOGUE}. "
            "Les données de référence doivent être déployées avec le service."
        )
    return pd.read_csv(_CHEMIN_CATALOGUE)


def _catalogue_disponible() -> bool:
    """Vrai si les données de référence nécessaires à l'enrichissement sont présentes."""
    return _CHEMIN_CATALOGUE.exists()


def _exiger_catalogue() -> None:
    """Refuse de scorer sans données de référence — 503, jamais un score dégradé.

    Sans le catalogue, `joindre_catalogue` produirait des colonnes vides que le pipeline
    imputerait : le modèle répondrait quand même, sur des features fausses. Mieux vaut
    une indisponibilité franche qu'une prédiction silencieusement dégradée.
    """
    if not _catalogue_disponible():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Données de référence absentes : {_CHEMIN_CATALOGUE.name} introuvable.",
        )


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    get_model_store().charger()
    if not _catalogue_disponible():
        logger.warning(
            "Catalogue des plans absent ({}) — /ready retournera 503 et /predict refusera "
            "de scorer : l'enrichissement commercial serait incomplet.",
            _CHEMIN_CATALOGUE,
        )
    yield


app = FastAPI(
    title="Churn SaaS — API de prédiction",
    description=(
        "Prédit la probabilité de résiliation d'un compte SaaS B2B. "
        "Retourne la probabilité, la valeur à risque (€), la décision recommandée "
        "et les facteurs de risque identifiés. "
        "Les champs porteurs de manquance dans le jeu d'entraînement acceptent `null` : "
        "leur valeur est reconstruite par le pipeline et listée dans `champs_imputes`. "
        "Aucune valeur par défaut n'est fabriquée — un champ obligatoire absent donne un 422. "
        "Authentification via l'en-tête **X-API-Key** (clé env `CHURN_API_KEY`)."
    ),
    version=_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(LimiteCorpsMiddleware)

Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

_PREDICTIONS_COUNTER = Counter(
    "churn_predictions_total",
    "Nombre de prédictions par décision (ALERTE_ROUGE / SURVEILLANCE / OK)",
    ["decision"],
)

# Suivi de la manquance amont : un taux qui grimpe sur un champ signale une rupture
# d'intégration CRM bien avant que la dérive du score ne devienne visible (§13).
_CHAMPS_IMPUTES_COUNTER = Counter(
    "churn_champs_imputes_total",
    "Nombre de demandes reçues avec un champ absent, par champ reconstruit",
    ["champ"],
)


# ---------------------------------------------------------------------------
# Helpers métier
# ---------------------------------------------------------------------------


def _demande_vers_dataframe(demandes: list[DemandePredicton]) -> pd.DataFrame:
    """Convertit une liste de demandes en DataFrame enrichi — même chaîne que le gold dataset."""
    donnees = [d.model_dump() for d in demandes]
    df = pd.DataFrame(donnees)
    df = ajouter_features_metier(df, csat_median_par_secteur=None)
    df = joindre_catalogue(df, _catalogue())
    df = enrichir_par_secteur(df)
    df = enrichir_par_pays(df)
    return df


def _champs_imputes(demande: DemandePredicton) -> list[str]:
    """Champs absents de la demande, dont la valeur sera reconstruite en aval.

    Reconstruction : imputation médiane du `Pipeline` sklearn pour les champs numériques,
    valeur de repli des enrichissements pour `secteur` et `pays`. Le score reste
    exploitable, mais le consommateur doit savoir sur quoi il repose.
    """
    return sorted(nom for nom in champs_nullables() if getattr(demande, nom) is None)


def _extraire_facteurs(demande: DemandePredicton) -> list[str]:
    """Identifie les signaux de risque churn sur la base de règles métier documentées.

    Ces règles proviennent de l'EDA (§6) — jamais inventées a posteriori.
    Elles sont complémentaires aux SHAP values calculées en §11 (notebook).

    Un champ absent ne déclenche aucun facteur : on ne transforme pas une donnée manquante
    en signal de risque. Il apparaît dans `champs_imputes`, ce qui est l'information honnête.
    """
    facteurs: list[str] = []
    if demande.derniere_connexion_jours > 30:
        facteurs.append(f"inactivité élevée ({demande.derniere_connexion_jours} j sans connexion)")
    if demande.csat is not None and demande.csat <= 6:
        facteurs.append(f"satisfaction faible (CSAT = {demande.csat:.1f}/10)")
    if demande.retards_paiement_12m is not None and demande.retards_paiement_12m > 0:
        facteurs.append(f"retards de paiement ({demande.retards_paiement_12m} sur 12 mois)")
    if demande.taux_adoption_pct is not None and demande.taux_adoption_pct < 30:
        facteurs.append(f"adoption faible ({demande.taux_adoption_pct:.0f} %)")
    if demande.tickets_support_90j > 5:
        facteurs.append(f"volume élevé de tickets ({demande.tickets_support_90j} sur 90 jours)")
    if demande.fonctionnalites_total > 0:
        frac = demande.fonctionnalites_utilisees / demande.fonctionnalites_total
        if frac < 0.33:
            facteurs.append(f"couverture fonctionnelle faible ({frac:.0%})")
    return facteurs[:3] if facteurs else ["aucun signal négatif identifié"]


def _valeur_a_risque(
    demande: DemandePredicton, probabilite: float
) -> tuple[float | None, str | None]:
    """Valeur à risque en €, ou (None, motif) si le MRR est absent de la demande.

    Le calcul lui-même vit dans `churn_saas.economie` — autorité unique partagée avec le
    flow de scoring batch et la §12.11 du notebook. Cette fonction ne porte que la
    **politique de manquance de l'API** : le modèle sait scorer un compte sans MRR
    (imputation médiane), mais le montant en € sert à prioriser les gestes de rétention et
    à chiffrer le ROI. Le calculer sur un MRR imputé produirait un euro faux présenté comme
    une mesure : on renvoie `null` et le motif, à charge du consommateur de compléter la
    donnée.
    """
    if demande.revenu_mensuel_recurrent_eur is None:
        return None, (
            "MRR absent de la demande — montant non calculable ; le score de churn, lui, "
            "reste valide (imputation médiane du pipeline)."
        )
    valeur = economie.valeur_a_risque(probabilite, demande.revenu_mensuel_recurrent_eur)
    return round(float(valeur), 2), None


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
    valeur, motif = _valeur_a_risque(demande, probabilite)
    imputes = _champs_imputes(demande)
    for champ in imputes:
        _CHAMPS_IMPUTES_COUNTER.labels(champ=champ).inc()
    return ResultatPrediction(
        probabilite_churn=round(probabilite, 4),
        valeur_a_risque_eur=valeur,
        decision=_decision(probabilite, seuil),
        facteurs_principaux=_extraire_facteurs(demande),
        seuil_applique=seuil,
        champs_imputes=imputes,
        motif_valeur_a_risque=motif,
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
    """Sonde readiness — 503 tant que le modèle *ou* les données de référence manquent.

    Un service capable de répondre mais privé de son catalogue de plans produirait des
    features incomplètes : il n'est pas « prêt », il est dégradé.
    """
    if not store.est_pret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modèle non chargé. Exécutez `churn-saas train` pour entraîner.",
        )
    if not _catalogue_disponible():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Données de référence absentes : {_CHEMIN_CATALOGUE.name} introuvable.",
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

    Un compte incomplet est scoré si la manquance porte sur un champ nullable : les champs
    reconstruits sont listés dans `champs_imputes`, et `valeur_a_risque_eur` passe à `null`
    si le MRR fait partie des manquants.

    - **422** : données d'entrée invalides (bornes, types) ou champ obligatoire absent.
    - **401** : clé d'API manquante ou incorrecte.
    - **429** : quota dépassé (60 req/min par IP).
    - **503** : modèle ou données de référence non disponibles.
    """
    if not store.est_pret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modèle non disponible.",
        )
    _exiger_catalogue()
    X = _demande_vers_dataframe([demande])
    probas = store.predire(X)
    resultat = _construire_resultat(demande, float(probas[0, 1]), store.seuil)
    _PREDICTIONS_COUNTER.labels(decision=resultat.decision).inc()
    return resultat


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

    _exiger_catalogue()
    X = _demande_vers_dataframe(demandes)
    probas = store.predire(X)[:, 1]

    predictions = [
        _construire_resultat(d, float(p), store.seuil)
        for d, p in zip(demandes, probas, strict=False)
    ]
    decisions = [r.decision for r in predictions]
    for dec in decisions:
        _PREDICTIONS_COUNTER.labels(decision=dec).inc()
    return ResultatBatch(
        nb_comptes=len(predictions),
        predictions=predictions,
        nb_alertes_rouges=decisions.count("ALERTE_ROUGE"),
        nb_surveillances=decisions.count("SURVEILLANCE"),
        nb_ok=decisions.count("OK"),
    )
