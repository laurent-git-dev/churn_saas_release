"""Test de contrat modèle↔API — charge le vrai `best_model.pkl` sérialisé.

Contexte du bug détecté en préparation de l'IHM de démo : le modèle avait été ré-entraîné
sur le gold dataset (`joindre_catalogue` + `enrichir_par_secteur` + `enrichir_par_pays`)
alors que `_demande_vers_dataframe` n'appliquait que `ajouter_features_metier`. Aucun test
existant ne pouvait le voir :

- `tests/test_api.py` branche un `_MockStoreReady` qui renvoie 0,72 en dur — le pipeline
  sérialisé n'est jamais sollicité, donc le désaccord de colonnes est invisible ;
- la §10 du notebook démontre l'API via `TestClient`, mais elle a été validée avant le
  ré-entraînement ;
- seul `make api` + un vrai appel révélait le `ValueError: columns are missing`, et cette
  vérification de bout en bout n'existait pas dans le workflow quotidien.

Ce module ferme les trois trous, à trois niveaux de profondeur croissante :

1. **Contrat de colonnes** — les colonnes réellement consommées par le `ColumnTransformer`
   fitté sont toutes produites par `_demande_vers_dataframe` (échec explicite, pas un 500).
2. **Contrat de service** — `/predict` et `/predict-batch` répondent 200 avec le vrai modèle.
3. **Contrat numérique** — des lignes du gold dataset rejouées via l'API donnent *exactement*
   la probabilité obtenue en appelant `predict_proba` directement sur la ligne gold. C'est
   le seul test qui détecte un enrichissement présent mais *divergent* (jointure catalogue
   qui retombe en NaN, libellé de plan renommé, ordre des étapes modifié…).

Les tests sont ignorés si les artefacts locaux sont absents (CI sans `churn-saas train`).
"""

from __future__ import annotations

import math
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.compose import ColumnTransformer

from churn_saas import config, economie
from churn_saas.api.main import _demande_vers_dataframe, app
from churn_saas.api.model_store import ModelStore, get_model_store
from churn_saas.api.schemas import (
    CHAMPS_RECALCULES_EXACTEMENT,
    DemandePrediction,
    champs_nullables,
)

_CHEMIN_MODELE = config.ARTIFACTS / "models" / "best_model.pkl"
_CHEMIN_GOLD = config.DONNEES_GOLD / "gold_dataset.parquet"

pytestmark = pytest.mark.skipif(
    not _CHEMIN_MODELE.exists(),
    reason="best_model.pkl absent — exécutez `churn-saas train` pour l'entraîner.",
)

# Champs du schéma tolérant None — déduits des annotations, jamais recopiés à la main.
# Leur valeur est reconstruite en aval : imputation médiane du pipeline pour les numériques,
# valeur de repli des enrichissements pour `secteur` / `pays`.
_CHAMPS_NULLABLES = champs_nullables()

# Colonnes produites par la jointure catalogue puis par les enrichissements externes.
# Un NaN ici signale une jointure silencieusement cassée (libellé de plan, de secteur
# ou de pays renommé en amont) : le modèle prédit quand même, mais sur une valeur imputée.
_COLONNES_CATALOGUE = ["prix_mensuel_par_siege_eur", "remise_consentie"]
_COLONNES_ENRICHISSEMENT = [
    "taux_churn_median_saas_pct",
    "ecart_adoption_secteur",
    "langue_support_fr",
    "decalage_horaire_paris_h",
]

# Profils types (sain / en alerte) — couvrent les deux extrêmes du score de risque
_COMPTE_ALERTE: dict[str, Any] = {
    "anciennete_mois": 18,
    "sieges_souscrits": 25,
    "utilisateurs_actifs": 6,
    "taux_adoption_pct": 24.0,
    "connexions_30j": 8,
    "heures_usage_30j": 15.0,
    "fonctionnalites_total": 30,
    "fonctionnalites_utilisees": 6,
    "nb_integrations": 0,
    "derniere_connexion_jours": 42,
    "tickets_support_90j": 9,
    "delai_reponse_support_h": 48.0,
    "csat": 4.5,
    "retards_paiement_12m": 3,
    "revenu_mensuel_recurrent_eur": 3200.0,
    "secteur": "Finance",
    "pays": "France",
    "taille_entreprise": "PME",
    "plan": "Business",
}

_COMPTE_SAIN: dict[str, Any] = {
    "anciennete_mois": 36,
    "sieges_souscrits": 50,
    "utilisateurs_actifs": 39,
    "taux_adoption_pct": 78.0,
    "connexions_30j": 120,
    "heures_usage_30j": 200.0,
    "fonctionnalites_total": 20,
    "fonctionnalites_utilisees": 18,
    "nb_integrations": 5,
    "derniere_connexion_jours": 2,
    "tickets_support_90j": 1,
    "delai_reponse_support_h": 4.0,
    "csat": 4.0,
    "retards_paiement_12m": 0,
    "revenu_mensuel_recurrent_eur": 5000.0,
    "secteur": "Technologie",
    "pays": "France",
    "taille_entreprise": "PME",
    "plan": "Business",
}

# Compte sain dont *tous* les champs nullables sont absents — cas limite du contrat.
_COMPTE_INCOMPLET: dict[str, Any] = {**_COMPTE_SAIN, **dict.fromkeys(_CHAMPS_NULLABLES)}


# ---------------------------------------------------------------------------
# Fixtures — le vrai artefact, jamais un mock
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def modele() -> Any:
    """Pipeline sklearn désérialisé depuis artifacts/models/best_model.pkl."""
    return joblib.load(_CHEMIN_MODELE)


@pytest.fixture(scope="module")
def preprocesseur(modele: Any) -> ColumnTransformer:
    """ColumnTransformer fitté extrait du pipeline — sans supposer le nom de l'étape."""
    for etape in modele.named_steps.values():
        if isinstance(etape, ColumnTransformer):
            return etape
    pytest.fail("Aucun ColumnTransformer trouvé dans le pipeline sérialisé.")


@pytest.fixture(scope="module")
def colonnes_consommees(preprocesseur: ColumnTransformer) -> dict[str, list[str]]:
    """Colonnes réellement lues par le préprocesseur, par bloc de transformation.

    Le `remainder="drop"` est exclu : ces colonnes sont connues du modèle mais jetées,
    donc leur absence côté API est sans conséquence (cf. `test_colonnes_absentes_sont_ignorees`).
    """
    blocs: dict[str, list[str]] = {}
    for nom, _transformeur, cols in preprocesseur.transformers_:
        if nom == "remainder" or isinstance(cols, str):
            continue
        blocs[nom] = list(cols)
    assert blocs, "Le préprocesseur fitté ne consomme aucune colonne — artefact suspect."
    return blocs


@pytest.fixture(scope="module")
def colonnes_remainder(preprocesseur: ColumnTransformer) -> set[str]:
    """Colonnes vues au fit mais supprimées par `remainder="drop"`."""
    for nom, _transformeur, cols in preprocesseur.transformers_:
        if nom == "remainder" and not isinstance(cols, str):
            return set(cols)
    return set()


@pytest.fixture(scope="module")
def df_api() -> pd.DataFrame:
    """DataFrame produit par la chaîne d'inférence pour deux profils complets contrastés."""
    demandes = [DemandePrediction(**_COMPTE_ALERTE), DemandePrediction(**_COMPTE_SAIN)]
    return _demande_vers_dataframe(demandes)


@pytest.fixture(scope="module")
def df_api_incomplet() -> pd.DataFrame:
    """Chaîne d'inférence sur un compte dont *tous* les champs nullables sont absents.

    Cas limite du contrat d'entrée : la chaîne doit produire le même jeu de colonnes,
    en empruntant le chemin de repli des enrichissements.
    """
    return _demande_vers_dataframe([DemandePrediction(**_COMPTE_INCOMPLET)])


@pytest.fixture(scope="module")
def store_reel() -> ModelStore:
    """ModelStore ayant chargé le vrai artefact — aucun mock."""
    store = ModelStore()
    store.charger()
    assert store.est_pret, "ModelStore.charger() n'a pas chargé le modèle."
    return store


@pytest.fixture(scope="module")
def client_reel(store_reel: ModelStore):
    """TestClient branché sur le vrai best_model.pkl."""
    app.dependency_overrides[get_model_store] = lambda: store_reel
    yield TestClient(app)
    app.dependency_overrides.clear()


def _payload_depuis_ligne(ligne: pd.Series) -> dict[str, Any]:
    """Reconstruit un payload API à partir d'une ligne gold (NaN → None, numpy → natif)."""
    payload: dict[str, Any] = {}
    for champ in DemandePrediction.model_fields:
        valeur = ligne[champ]
        if isinstance(valeur, np.generic):
            valeur = valeur.item()
        if isinstance(valeur, float) and math.isnan(valeur):
            valeur = None
        payload[champ] = valeur
    return payload


@pytest.fixture(scope="module")
def gold() -> pd.DataFrame:
    """Gold dataset complet — référence de la manquance réellement vue à l'entraînement."""
    if not _CHEMIN_GOLD.exists():
        pytest.skip("gold_dataset.parquet absent — exécutez `churn-saas build-features`.")
    return pd.read_parquet(_CHEMIN_GOLD)


@pytest.fixture(scope="module")
def echantillon_gold(gold: pd.DataFrame) -> pd.DataFrame:
    """50 lignes du gold dataset acceptées par le schéma API (champs obligatoires présents)."""
    obligatoires = [c for c in DemandePrediction.model_fields if c not in _CHAMPS_NULLABLES]
    acceptables = gold.dropna(subset=obligatoires)
    assert len(acceptables) >= 50, f"Trop peu de lignes gold exploitables : {len(acceptables)}."
    return acceptables.sample(n=50, random_state=config.RANDOM_SEED)


@pytest.fixture(scope="module")
def echantillon_gold_incomplet(gold: pd.DataFrame) -> pd.DataFrame:
    """50 lignes gold portant au moins un champ nullable manquant.

    C'est la population que l'API rejetait en 422 avant l'alignement du contrat d'entrée
    sur la manquance observée (≈ 23 % du portefeuille).
    """
    obligatoires = [c for c in DemandePrediction.model_fields if c not in _CHAMPS_NULLABLES]
    nullables = sorted(_CHAMPS_NULLABLES)
    trouees = gold[gold[obligatoires].notna().all(axis=1) & gold[nullables].isna().any(axis=1)]
    assert len(trouees) >= 50, f"Trop peu de lignes gold incomplètes : {len(trouees)}."
    return trouees.sample(n=50, random_state=config.RANDOM_SEED)


# ---------------------------------------------------------------------------
# Niveau 0 — contrat d'entrée : la nullabilité reflète la manquance des données
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("champ", sorted(DemandePrediction.model_fields))
def test_nullabilite_reflete_la_manquance_du_gold(champ: str, gold: pd.DataFrame) -> None:
    """Un champ est nullable si et seulement s'il est manquant dans le jeu d'entraînement.

    C'est la règle de conception du contrat d'entrée, vérifiée contre les données plutôt
    que contre une intention : si un futur jeu introduit de la manquance sur un champ
    aujourd'hui obligatoire, ce test échoue et force la mise à jour du schéma — au lieu
    d'un 422 découvert en production sur un compte réel.
    """
    manquants = int(gold[champ].isna().sum())
    est_nullable = champ in _CHAMPS_NULLABLES
    if manquants and not est_nullable:
        pytest.fail(
            f"'{champ}' est manquant sur {manquants} lignes gold ({manquants / len(gold):.1%}) "
            "mais le schéma le refuse : l'API rejetterait en 422 des comptes que le modèle "
            "sait scorer. Rendre le champ nullable (défaut None) dans schemas.py."
        )
    # Un champ recalculé exactement est complet dans le gold par construction, mais peut
    # manquer dans une demande : nullable sans imputation, donc sans contradiction
    if est_nullable and not manquants and champ not in CHAMPS_RECALCULES_EXACTEMENT:
        pytest.fail(
            f"'{champ}' est complet à 100 % dans le gold dataset mais le schéma le déclare "
            "nullable : son absence devrait être un 422 (défaut d'intégration) et non une "
            "imputation silencieuse. Le rendre obligatoire dans schemas.py."
        )


@pytest.mark.parametrize("champ", sorted(DemandePrediction.model_fields))
def test_aucune_valeur_par_defaut_fabriquee(champ: str) -> None:
    """Aucun champ ne reçoit de valeur par défaut « plausible ».

    Un défaut fabriqué (50 % d'adoption, 1 000 € de MRR) est indiscernable d'une
    observation réelle pour le modèle : la prédiction devient fausse sans trace. Un champ
    est donc soit obligatoire, soit nullable avec `None` pour défaut.
    """
    info = DemandePrediction.model_fields[champ]
    if info.is_required():
        return
    assert info.default is None, (
        f"'{champ}' a une valeur par défaut fabriquée ({info.default!r}) : la remplacer par "
        "None (imputation par le pipeline, champ signalé dans champs_imputes) ou rendre le "
        "champ obligatoire."
    )


def test_champ_obligatoire_absent_donne_422(client_reel: TestClient) -> None:
    """L'omission d'un champ obligatoire est un défaut d'intégration → 422, pas un score."""
    donnees = {k: v for k, v in _COMPTE_SAIN.items() if k != "anciennete_mois"}
    resp = client_reel.post("/predict", json=donnees)
    assert resp.status_code == 422, (
        "Un champ obligatoire absent doit être refusé, jamais remplacé par une valeur "
        f"par défaut : obtenu {resp.status_code}."
    )


def test_champ_obligatoire_null_donne_422(client_reel: TestClient) -> None:
    """Même refus si le champ obligatoire est présent mais explicitement `null`."""
    resp = client_reel.post("/predict", json={**_COMPTE_SAIN, "plan": None})
    assert resp.status_code == 422


def test_compte_incomplet_est_score_et_signale(client_reel: TestClient) -> None:
    """Un compte privé de tous ses champs nullables est scoré, et la manquance est tracée."""
    resp = client_reel.post("/predict", json=_COMPTE_INCOMPLET)
    assert resp.status_code == 200, f"Compte incomplet refusé : {resp.text}"
    corps = resp.json()
    assert 0.0 <= corps["probabilite_churn"] <= 1.0
    assert corps["champs_imputes"] == sorted(
        _CHAMPS_NULLABLES - CHAMPS_RECALCULES_EXACTEMENT
    ), "champs_imputes doit lister exactement les champs absents et réellement imputés."
    # MRR absent → aucun montant affiché : un euro imputé serait un chiffre faux
    assert corps["valeur_a_risque_eur"] is None
    assert corps["motif_valeur_a_risque"], "Un montant absent doit être motivé."


def test_taux_adoption_absent_recalcule_et_non_signale(client_reel: TestClient) -> None:
    """Un taux d'adoption absent donne le même score que le taux exact, sans être « imputé »."""
    sieges, utilisateurs = _COMPTE_SAIN["sieges_souscrits"], _COMPTE_SAIN["utilisateurs_actifs"]
    exact = {**_COMPTE_SAIN, "taux_adoption_pct": 100 * utilisateurs / sieges}
    absent = {**_COMPTE_SAIN, "taux_adoption_pct": None}
    p_exact = client_reel.post("/predict", json=exact).json()
    p_absent = client_reel.post("/predict", json=absent).json()
    assert p_absent["probabilite_churn"] == pytest.approx(p_exact["probabilite_churn"], abs=1e-4)
    assert "taux_adoption_pct" not in p_absent["champs_imputes"]


def test_champs_imputes_vide_si_demande_complete(client_reel: TestClient) -> None:
    """Une demande complète ne déclare aucune imputation et chiffre la valeur à risque."""
    corps = client_reel.post("/predict", json=_COMPTE_SAIN).json()
    assert corps["champs_imputes"] == []
    assert corps["motif_valeur_a_risque"] is None
    assert corps["valeur_a_risque_eur"] is not None


def test_mrr_seul_absent_conserve_le_score(client_reel: TestClient) -> None:
    """Le MRR manquant neutralise le montant en € mais pas la prédiction de churn."""
    corps = client_reel.post(
        "/predict", json={**_COMPTE_ALERTE, "revenu_mensuel_recurrent_eur": None}
    ).json()
    assert corps["champs_imputes"] == ["revenu_mensuel_recurrent_eur"]
    assert corps["valeur_a_risque_eur"] is None
    assert corps["decision"] in {"ALERTE_ROUGE", "SURVEILLANCE", "OK"}


# ---------------------------------------------------------------------------
# Niveau 1 — contrat de colonnes (le bug de l'IHM de démo)
# ---------------------------------------------------------------------------


def test_toutes_colonnes_consommees_sont_produites(
    df_api: pd.DataFrame,
    colonnes_consommees: dict[str, list[str]],
) -> None:
    """Chaque colonne lue par le préprocesseur fitté est produite par l'API.

    C'est LE test qui aurait échoué avant le correctif : sans `joindre_catalogue` ni les
    enrichissements, `prix_mensuel_par_siege_eur`, `taux_churn_median_saas_pct`… étaient
    absentes et sklearn levait un `ValueError` transformé en HTTP 500 à la démo.
    """
    attendues = {col for cols in colonnes_consommees.values() for col in cols}
    manquantes = sorted(attendues - set(df_api.columns))
    assert not manquantes, (
        f"Pipeline API désaligné avec best_model.pkl — colonnes manquantes : {manquantes}. "
        "Ajouter l'étape d'enrichissement correspondante dans "
        "churn_saas.api.main._demande_vers_dataframe, puis relancer ce test."
    )


def test_intensite_support_consommee_et_calculee_cote_serveur(
    df_api: pd.DataFrame,
    colonnes_consommees: dict[str, list[str]],
) -> None:
    """Le champion régénéré lit `intensite_support`, que l'API calcule sans la demander."""
    attendues = {col for cols in colonnes_consommees.values() for col in cols}
    assert "intensite_support" in attendues
    assert "intensite_support" not in DemandePrediction.model_fields
    attendu = _COMPTE_ALERTE["tickets_support_90j"] / _COMPTE_ALERTE["anciennete_mois"]
    assert df_api["intensite_support"].iloc[0] == pytest.approx(attendu)


def test_colonnes_consommees_produites_sur_compte_incomplet(
    df_api_incomplet: pd.DataFrame,
    colonnes_consommees: dict[str, list[str]],
) -> None:
    """Même contrat de colonnes quand tous les champs nullables sont absents.

    Les enrichissements doivent emprunter leur chemin de repli sans jamais *omettre* de
    colonne : une colonne absente casse le pipeline, une colonne à NaN est imputée.
    """
    attendues = {col for cols in colonnes_consommees.values() for col in cols}
    manquantes = sorted(attendues - set(df_api_incomplet.columns))
    assert not manquantes, f"Colonnes perdues sur un compte incomplet : {manquantes}."


def test_colonnes_absentes_sont_ignorees_par_le_modele(
    df_api: pd.DataFrame,
    modele: Any,
    colonnes_remainder: set[str],
) -> None:
    """Les colonnes du fit absentes côté API sont toutes jetées par `remainder="drop"`.

    Le schéma API ne demande volontairement ni `date_souscription` (non observable au
    moment de la prédiction) ni les 4 leurres (§11 : importance nulle démontrée). Ce test
    fige cette tolérance : si une future version du modèle se met à *consommer* l'une de
    ces colonnes, l'écart devient un échec ici plutôt qu'un 500 en production.
    """
    vues_au_fit = set(getattr(modele, "feature_names_in_", ()))
    if not vues_au_fit:
        pytest.skip("Le pipeline sérialisé n'expose pas feature_names_in_.")
    absentes = vues_au_fit - set(df_api.columns)
    hors_remainder = sorted(absentes - colonnes_remainder)
    assert (
        not hors_remainder
    ), f"Colonnes attendues par le modèle, absentes de l'API et non ignorées : {hors_remainder}."


def test_colonnes_numeriques_convertibles(
    df_api: pd.DataFrame,
    colonnes_consommees: dict[str, list[str]],
) -> None:
    """Les colonnes du bloc numérique sont convertibles en float sans coercition.

    Un champ nullable entièrement à None arrive en dtype `object` : c'est admis
    (SimpleImputer le convertit), mais une valeur textuelle inattendue doit échouer ici
    et non silencieusement devenir NaN puis la médiane du train.
    """
    for col in colonnes_consommees.get("numerique", []):
        try:
            pd.to_numeric(df_api[col], errors="raise")
        except (ValueError, TypeError) as exc:  # pragma: no cover — filet de sécurité
            pytest.fail(f"Colonne '{col}' non convertible en numérique : {exc}")


def test_aucune_colonne_interdite_en_inference(df_api: pd.DataFrame) -> None:
    """Anti-fuite au moment de l'inférence : aucune colonne de COLONNES_INTERDITES produite.

    Le `remainder="drop"` les écarterait de toute façon, mais leur simple présence dans
    le DataFrame d'inférence signalerait une chaîne de features divergente de la §7.
    """
    interdites = sorted(set(df_api.columns) & set(config.COLONNES_INTERDITES))
    assert not interdites, f"Colonnes interdites produites par l'API : {interdites}."


@pytest.mark.parametrize("plan", ["Starter", "Pro", "Business", "Enterprise"])
def test_jointure_catalogue_renseignee(plan: str) -> None:
    """Pour chaque plan du catalogue, les features commerciales sont renseignées.

    Une jointure qui retombe en NaN (libellé de plan renommé côté catalogue) ne lève
    aucune erreur : le modèle impute et le score dérive sans alerte. D'où ce test.
    """
    df = _demande_vers_dataframe([DemandePrediction(**{**_COMPTE_SAIN, "plan": plan})])
    for col in _COLONNES_CATALOGUE:
        assert col in df.columns, f"Colonne catalogue '{col}' absente."
        assert (
            df[col].notna().all()
        ), f"Jointure catalogue cassée pour le plan '{plan}' : {col} NaN."


def test_enrichissements_externes_renseignes(df_api: pd.DataFrame) -> None:
    """Les enrichissements secteur/pays ne retombent pas sur la valeur de repli.

    Même logique que la jointure catalogue : un secteur ou un pays non reconnu passe en
    `_repli_` sans erreur, et la prédiction devient muettement moins informée.
    """
    for col in _COLONNES_ENRICHISSEMENT:
        assert col in df_api.columns, f"Colonne d'enrichissement '{col}' absente."
        assert df_api[col].notna().all(), f"Enrichissement non résolu pour '{col}'."


# ---------------------------------------------------------------------------
# Niveau 2 — contrat de service : les routes répondent avec le vrai modèle
# ---------------------------------------------------------------------------


def test_predict_200_avec_modele_reel(client_reel: TestClient) -> None:
    """/predict répond 200 avec le pipeline sérialisé — pas un mock."""
    resp = client_reel.post("/predict", json=_COMPTE_ALERTE)
    assert resp.status_code == 200, f"Pipeline API/modèle désaligné : {resp.text}"
    corps = resp.json()
    assert 0.0 <= corps["probabilite_churn"] <= 1.0
    assert corps["decision"] in {"ALERTE_ROUGE", "SURVEILLANCE", "OK"}
    assert corps["valeur_a_risque_eur"] >= 0.0
    assert isinstance(corps["facteurs_principaux"], list)


def test_predict_champs_nullables_imputes(client_reel: TestClient) -> None:
    """Les champs nullables à None sont imputés par le pipeline, pas rejetés."""
    donnees = {**_COMPTE_ALERTE} | dict.fromkeys(_CHAMPS_NULLABLES)
    resp = client_reel.post("/predict", json=donnees)
    assert resp.status_code == 200, f"Imputation échouée : {resp.text}"


def test_predict_batch_200_avec_modele_reel(client_reel: TestClient) -> None:
    """/predict-batch répond 200 et ses compteurs de décision sont cohérents."""
    resp = client_reel.post("/predict-batch", json=[_COMPTE_ALERTE, _COMPTE_SAIN])
    assert resp.status_code == 200, f"Pipeline batch désaligné : {resp.text}"
    corps = resp.json()
    assert corps["nb_comptes"] == 2
    assert len(corps["predictions"]) == 2
    assert corps["nb_alertes_rouges"] + corps["nb_surveillances"] + corps["nb_ok"] == 2


def test_valeur_a_risque_coherente_avec_la_formule_unique(client_reel: TestClient) -> None:
    """Le montant renvoyé est exactement celui de `churn_saas.economie` (marge incluse).

    La formule elle-même et la parité avec le batch nocturne sont testées dans
    `tests/test_valeur_a_risque.py` ; ici on vérifie que l'API n'a pas de calcul à elle.
    """
    corps = client_reel.post("/predict", json=_COMPTE_ALERTE).json()
    attendue = economie.valeur_a_risque(
        corps["probabilite_churn"], _COMPTE_ALERTE["revenu_mensuel_recurrent_eur"]
    )
    # Probabilité renvoyée arrondie à 4 décimales : écart admis = effet de cet arrondi
    euros_par_unite_de_proba = economie.valeur_a_risque(
        1.0, _COMPTE_ALERTE["revenu_mensuel_recurrent_eur"]
    )
    assert abs(corps["valeur_a_risque_eur"] - float(attendue)) <= (
        0.5e-4 * euros_par_unite_de_proba + 0.01
    )


def test_facteurs_shap_du_modele_reel(client_reel: TestClient) -> None:
    """Le compte en alerte reçoit au plus 3 facteurs, positifs, triés, libellés métier."""
    facteurs = client_reel.post("/predict", json=_COMPTE_ALERTE).json()["facteurs_shap"]
    assert 1 <= len(facteurs) <= 3, "Modèle servi sans explication locale : le réentraîner."
    contributions = [f["contribution"] for f in facteurs]
    assert all(c > 0 for c in contributions)
    assert contributions == sorted(contributions, reverse=True)
    # Libellé métier, jamais un nom de colonne transformée (« plan_business »)
    assert all("_" not in f["libelle"] for f in facteurs)


def test_ordre_des_scores_sain_inferieur_alerte(client_reel: TestClient) -> None:
    """Le profil sain score strictement sous le profil en alerte — garde-fou de sens métier."""
    proba_alerte = client_reel.post("/predict", json=_COMPTE_ALERTE).json()["probabilite_churn"]
    proba_sain = client_reel.post("/predict", json=_COMPTE_SAIN).json()["probabilite_churn"]
    assert (
        proba_sain < proba_alerte
    ), f"Ordre des scores incohérent : sain={proba_sain:.3f} ≥ alerte={proba_alerte:.3f}"


# ---------------------------------------------------------------------------
# Niveau 3 — contrat numérique : l'API reproduit la chaîne d'entraînement
# ---------------------------------------------------------------------------


def test_parite_api_vs_pipeline_entrainement(
    client_reel: TestClient,
    modele: Any,
    echantillon_gold: pd.DataFrame,
) -> None:
    """Rejouer des lignes gold via l'API donne la probabilité du `predict_proba` direct.

    Référence : `modele.predict_proba(ligne_gold)`, où la ligne gold porte les features
    telles que construites en §7 pour l'entraînement. Si l'API reconstruit les mêmes
    features à partir des seuls champs observables, l'écart est nul à l'arrondi près
    (l'API arrondit à 4 décimales). Un enrichissement *divergent* — et non simplement
    absent — ne se voit que sur ce test.
    """
    payloads = [_payload_depuis_ligne(ligne) for _, ligne in echantillon_gold.iterrows()]
    resp = client_reel.post("/predict-batch", json=payloads)
    assert resp.status_code == 200, f"Rejeu du gold refusé par l'API : {resp.text[:500]}"

    probas_api = np.array([p["probabilite_churn"] for p in resp.json()["predictions"]])
    probas_ref = modele.predict_proba(echantillon_gold)[:, 1]
    ecarts = np.abs(probas_api - probas_ref)
    pire = int(np.argmax(ecarts))
    assert ecarts.max() <= 1e-4, (
        "Divergence entraînement↔API : la chaîne de features de l'API ne reproduit pas "
        f"celle du gold dataset (écart max {ecarts.max():.6f} sur la ligne {pire} — "
        f"API {probas_api[pire]:.6f} vs référence {probas_ref[pire]:.6f})."
    )


def test_parite_lignes_gold_incompletes(
    client_reel: TestClient,
    modele: Any,
    echantillon_gold_incomplet: pd.DataFrame,
) -> None:
    """La parité tient aussi sur les comptes incomplets — la population la plus risquée.

    Ces lignes portent des `NaN` que le pipeline impute. L'API reçoit des `null` : la
    reconstruction doit produire exactement les mêmes features, sinon les 23 % de comptes
    incomplets seraient scorés selon une chaîne différente de celle de l'entraînement.
    """
    payloads = [_payload_depuis_ligne(ligne) for _, ligne in echantillon_gold_incomplet.iterrows()]
    resp = client_reel.post("/predict-batch", json=payloads)
    assert resp.status_code == 200, f"Comptes incomplets refusés par l'API : {resp.text[:500]}"

    probas_api = np.array([p["probabilite_churn"] for p in resp.json()["predictions"]])
    probas_ref = modele.predict_proba(echantillon_gold_incomplet)[:, 1]
    ecarts = np.abs(probas_api - probas_ref)
    pire = int(np.argmax(ecarts))
    assert ecarts.max() <= 1e-4, (
        "Divergence entraînement↔API sur les comptes incomplets : un `null` d'API ne se "
        f"comporte pas comme un NaN d'entraînement (écart max {ecarts.max():.6f} sur la "
        f"ligne {pire})."
    )


def test_parite_route_unitaire(
    client_reel: TestClient,
    modele: Any,
    echantillon_gold: pd.DataFrame,
) -> None:
    """Même parité sur /predict : la route unitaire ne doit pas dériver de la route batch."""
    ligne = echantillon_gold.iloc[[0]]
    resp = client_reel.post("/predict", json=_payload_depuis_ligne(ligne.iloc[0]))
    assert resp.status_code == 200, resp.text
    proba_ref = float(modele.predict_proba(ligne)[0, 1])
    assert abs(resp.json()["probabilite_churn"] - proba_ref) <= 1e-4


def test_seuil_applique_provient_du_model_store(
    client_reel: TestClient,
    store_reel: ModelStore,
) -> None:
    """Le seuil renvoyé est celui du ModelStore (métadonnées si présentes, défaut sinon)."""
    seuil = client_reel.post("/predict", json=_COMPTE_SAIN).json()["seuil_applique"]
    assert 0.0 < seuil < 1.0, f"Seuil économique hors bornes : {seuil}."
    assert seuil == pytest.approx(store_reel.seuil)
