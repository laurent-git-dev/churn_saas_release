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
    "csat": 3.0,
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


class DemandePrediction(BaseModel):
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

    **Aucune feature dérivée n'est demandée.** Les ratios (`intensite_support`,
    `taux_utilisation_sieges`, `arpu_par_siege`…), la jointure catalogue et les
    enrichissements sont recalculés côté serveur par la même chaîne que le gold dataset
    (`features.build.ajouter_features_metier`, `joindre_catalogue`, enrichissements) : un
    client ne peut ni les fausser ni les calculer autrement que l'entraînement.

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
    # Échelle 1-5 du jeu d'entraînement : une note hors échelle serait une extrapolation
    csat: Annotated[float | None, Field(ge=1.0, le=5.0)] = None  # 8,0 %
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


# Champs facultatifs dont l'absence n'appelle aucune imputation : leur valeur se recalcule
# exactement à partir de champs obligatoires (features.build.reconstituer_taux_adoption).
# Ils restent nullables — un CRM qui ne les envoie pas n'est pas en défaut —, mais ne sont
# pas signalés dans `champs_imputes`, puisque le score ne repose sur aucune approximation.
CHAMPS_RECALCULES_EXACTEMENT: frozenset[str] = frozenset({"taux_adoption_pct"})


def champs_nullables() -> frozenset[str]:
    """Champs de `DemandePrediction` acceptant `None` — déduits des annotations.

    Autorité unique côté API : le décompte des champs reconstruits (`champs_imputes`)
    comme le test de contrat `tests/test_api_model_contract.py` en dérivent, pour qu'aucune
    liste codée en dur ne puisse diverger du schéma.
    """
    return frozenset(
        nom
        for nom, champ in DemandePrediction.model_fields.items()
        if type(None) in get_args(champ.annotation)
    )


class FacteurExplicatif(BaseModel):
    """Variable qui pousse le score d'un compte vers le churn (valeur SHAP exacte)."""

    variable: Annotated[str, Field(description="Variable d'entrée ou recalculée par le serveur.")]
    libelle: Annotated[str, Field(description="Libellé métier, avec la valeur du compte.")]
    contribution: Annotated[
        float,
        Field(
            gt=0.0,
            description=(
                "Valeur SHAP en log-odds : de combien la variable éloigne le logit du score de "
                "celui d'un compte moyen du jeu d'entraînement. Toujours positive ici."
            ),
        ),
    ]


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
    facteurs_principaux: Annotated[
        list[str],
        Field(
            description=(
                "Signaux métier issus de règles de l'EDA (§6) : décrivent le contexte du compte, "
                "pas la contribution des variables au score (voir `facteurs_shap`)."
            ),
        ),
    ]
    facteurs_shap: Annotated[
        list[FacteurExplicatif],
        Field(
            default_factory=list,
            description=(
                "Les 3 variables qui contribuent le plus au risque selon le modèle (valeurs "
                "SHAP exactes, contributions positives seulement, triées). Liste vide si le "
                "modèle servi n'a pas d'explication exacte disponible."
            ),
        ),
    ]
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
                    "satisfaction faible (CSAT ≤ 2)",
                    "retards de paiement",
                ],
                "facteurs_shap": [
                    {
                        "variable": "derniere_connexion_jours",
                        "libelle": "Jours depuis la dernière connexion : 35",
                        "contribution": 0.81,
                    },
                    {
                        "variable": "retards_paiement_12m",
                        "libelle": "Retards de paiement sur 12 mois : 2",
                        "contribution": 0.37,
                    },
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
    modele: Annotated[
        str | None,
        Field(default=None, description="Famille du modèle servi (ex. LightGBM)."),
    ]
    message: str
