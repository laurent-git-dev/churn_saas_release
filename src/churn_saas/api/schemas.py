"""Contrats Pydantic pour l'API de prédiction de churn — C6."""

from __future__ import annotations

from typing import Annotated, Any, Literal, get_args

from pydantic import BaseModel, Field

_EXEMPLE_COMPLET: dict[str, Any] = {
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

# Second exemple documenté : compte réel incomplet tel qu'il sort du CRM.
# 23 % des comptes du jeu d'entraînement sont dans ce cas — l'API les score et
# signale les champs reconstruits dans `champs_imputes`.
_EXEMPLE_INCOMPLET: dict[str, Any] = {
    **_EXEMPLE_COMPLET,
    "csat": None,
    "delai_reponse_support_h": None,
    "secteur": None,
}


class DemandePredicton(BaseModel):
    """Données observables d'un compte client pour la prédiction de churn.

    **Règle de conception du contrat d'entrée** — dérivée du jeu d'entraînement,
    pas de l'intuition :

    - un champ **jamais manquant** dans le gold dataset est **obligatoire** : son absence
      est un défaut d'intégration, et un 422 est la réponse honnête ;
    - un champ **porteur de manquance** dans le gold dataset est **nullable** (`None`), et
      sa valeur est reconstruite par le `Pipeline` sklearn (imputation médiane fittée sur
      le train) ou par la valeur de repli des enrichissements pour `secteur` / `pays` ;
    - **aucun champ n'a de valeur par défaut fabriquée.** Un défaut « plausible » (50 %
      d'adoption, 1 000 € de MRR) est traité par le modèle comme une observation réelle :
      la prédiction devient fausse sans qu'aucune trace n'en subsiste. Le contrat rend donc
      la manquance explicite plutôt que de l'inventer.

    Les champs reconstruits sont listés dans `ResultatPrediction.champs_imputes`.
    Les bornes correspondent aux valeurs métier réalistes du dataset SaaS B2B.
    """

    # ── Champs obligatoires : complets à 100 % dans le gold dataset ──────────
    anciennete_mois: Annotated[int, Field(ge=0, le=240)]
    sieges_souscrits: Annotated[int, Field(ge=1, le=10_000)]
    utilisateurs_actifs: Annotated[int, Field(ge=0, le=10_000)]
    connexions_30j: Annotated[int, Field(ge=0, le=10_000)]
    fonctionnalites_total: Annotated[int, Field(ge=0, le=500)]
    fonctionnalites_utilisees: Annotated[int, Field(ge=0, le=500)]
    derniere_connexion_jours: Annotated[int, Field(ge=0, le=3_650)]
    tickets_support_90j: Annotated[int, Field(ge=0, le=1_000)]
    taille_entreprise: str
    plan: str

    # ── Champs nullables : taux de manquance observé dans le gold dataset ────
    # Usage produit
    taux_adoption_pct: Annotated[float | None, Field(ge=0.0, le=100.0)] = None  # 5,0 %
    heures_usage_30j: Annotated[float | None, Field(ge=0.0, le=10_000.0)] = None  # 6,0 %
    nb_integrations: Annotated[int | None, Field(ge=0, le=100)] = None  # 4,0 %
    # Support
    delai_reponse_support_h: Annotated[float | None, Field(ge=0.0, le=8_760.0)] = None  # 21,4 %
    csat: Annotated[float | None, Field(ge=0.0, le=10.0)] = None  # 8,0 %
    # Facturation — un MRR absent n'empêche pas de scorer, mais interdit de chiffrer
    # la valeur à risque (cf. ResultatPrediction.motif_valeur_a_risque)
    retards_paiement_12m: Annotated[int | None, Field(ge=0, le=120)] = None  # 5,0 %
    revenu_mensuel_recurrent_eur: Annotated[float | None, Field(ge=0.0, le=1_000_000.0)] = (
        None  # 3,0 %
    )
    # Attributs client — alimentent les enrichissements sectoriels et pays
    secteur: str | None = None  # 5,0 %
    pays: str | None = None  # 4,0 %

    model_config = {
        "json_schema_extra": {
            "examples": [_EXEMPLE_COMPLET, _EXEMPLE_INCOMPLET],
        }
    }


def champs_nullables() -> frozenset[str]:
    """Champs de `DemandePredicton` acceptant `None` — déduits des annotations.

    Autorité unique côté API : le décompte des champs reconstruits (`champs_imputes`)
    comme le test de contrat `tests/test_api_model_contract.py` en dérivent, pour qu'aucune
    liste codée en dur ne puisse diverger du schéma.
    """
    return frozenset(
        nom
        for nom, champ in DemandePredicton.model_fields.items()
        if type(None) in get_args(champ.annotation)
    )


class ResultatPrediction(BaseModel):
    """Résultat de prédiction de churn pour un compte client."""

    probabilite_churn: Annotated[
        float,
        Field(ge=0.0, le=1.0, description="Probabilité de résiliation (0–1)."),
    ]
    valeur_a_risque_eur: Annotated[
        float | None,
        Field(
            ge=0.0,
            description=(
                "Probabilité × MRR × horizon en mois × marge brute (en €) — formule unique "
                "de `churn_saas.economie`, partagée avec le batch nocturne et la §12.11. "
                "`null` si le MRR est absent de la demande : une probabilité imputée reste "
                "une prédiction, un euro imputé est un chiffre faux présenté à un décideur."
            ),
        ),
    ]
    decision: Literal["ALERTE_ROUGE", "SURVEILLANCE", "OK"]
    facteurs_principaux: list[str]
    seuil_applique: float
    champs_imputes: Annotated[
        list[str],
        Field(
            default_factory=list,
            description=(
                "Champs absents de la demande, reconstruits par le pipeline (imputation "
                "médiane du train) ou par la valeur de repli des enrichissements. Le score "
                "reste exploitable mais repose partiellement sur des valeurs estimées."
            ),
        ),
    ]
    motif_valeur_a_risque: Annotated[
        str | None,
        Field(
            default=None,
            description="Raison pour laquelle `valeur_a_risque_eur` est `null`, le cas échéant.",
        ),
    ]

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
                "champs_imputes": ["csat", "secteur"],
                "motif_valeur_a_risque": None,
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
