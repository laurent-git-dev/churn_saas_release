"""Contrats Pydantic pour l'API de prédiction de churn — C6."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class DemandePredicton(BaseModel):
    """Données observables d'un compte client pour la prédiction de churn.

    Les champs optionnels (None) reçoivent une imputation médiane dans le pipeline sklearn.
    Les bornes correspondent aux valeurs métier réalistes du dataset SaaS B2B.
    """

    # Usage produit
    anciennete_mois: Annotated[int, Field(ge=0, le=240)] = 12
    sieges_souscrits: Annotated[int, Field(ge=1, le=10_000)] = 10
    utilisateurs_actifs: Annotated[int, Field(ge=0, le=10_000)] = 5
    taux_adoption_pct: Annotated[float, Field(ge=0.0, le=100.0)] = 50.0
    connexions_30j: Annotated[int, Field(ge=0, le=10_000)] = 20
    heures_usage_30j: Annotated[float | None, Field(ge=0.0, le=10_000.0)] = None
    fonctionnalites_total: Annotated[int, Field(ge=0, le=500)] = 20
    fonctionnalites_utilisees: Annotated[int, Field(ge=0, le=500)] = 10
    nb_integrations: Annotated[int, Field(ge=0, le=100)] = 2
    derniere_connexion_jours: Annotated[int, Field(ge=0, le=3_650)] = 5

    # Support
    tickets_support_90j: Annotated[int, Field(ge=0, le=1_000)] = 0
    delai_reponse_support_h: Annotated[float | None, Field(ge=0.0, le=8_760.0)] = None
    csat: Annotated[float | None, Field(ge=0.0, le=10.0)] = None

    # Facturation
    retards_paiement_12m: Annotated[int, Field(ge=0, le=120)] = 0
    revenu_mensuel_recurrent_eur: Annotated[float, Field(ge=0.0, le=1_000_000.0)] = 1_000.0

    # Attributs client
    secteur: str = "Technologie"
    pays: str = "France"
    taille_entreprise: str = "PME"
    plan: str = "Starter"

    model_config = {
        "json_schema_extra": {
            "example": {
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
                "revenu_mensuel_recurrent_eur": 2_500.0,
                "secteur": "Finance",
                "pays": "France",
                "taille_entreprise": "PME",
                "plan": "Business",
            }
        }
    }


class ResultatPrediction(BaseModel):
    """Résultat de prédiction de churn pour un compte client."""

    probabilite_churn: Annotated[
        float,
        Field(ge=0.0, le=1.0, description="Probabilité de résiliation (0–1)."),
    ]
    valeur_a_risque_eur: Annotated[
        float,
        Field(ge=0.0, description="MRR × probabilité × horizon en mois (en €)."),
    ]
    decision: Literal["ALERTE_ROUGE", "SURVEILLANCE", "OK"]
    facteurs_principaux: list[str]
    seuil_applique: float

    model_config = {
        "json_schema_extra": {
            "example": {
                "probabilite_churn": 0.72,
                "valeur_a_risque_eur": 21_600.0,
                "decision": "ALERTE_ROUGE",
                "facteurs_principaux": [
                    "inactivité > 30 jours",
                    "satisfaction faible (CSAT ≤ 6)",
                    "retards de paiement",
                ],
                "seuil_applique": 0.40,
            }
        }
    }


class ResultatBatch(BaseModel):
    """Résultat d'une prédiction batch (plusieurs comptes)."""

    nb_comptes: int
    predictions: list[ResultatPrediction]
    nb_alertes_rouges: int
    nb_surveillances: int
    nb_ok: int


class EtatSante(BaseModel):
    """Réponse du point /health."""

    statut: Literal["ok"]
    version: str


class EtatPret(BaseModel):
    """Réponse du point /ready."""

    statut: Literal["pret", "non_pret"]
    modele_charge: bool
    message: str
