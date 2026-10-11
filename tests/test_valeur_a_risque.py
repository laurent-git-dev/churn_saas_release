"""Tests de la valeur à risque — formule unique et parité entre ses consommateurs.

Contexte : la formule avait été recopiée à trois endroits (API, flow de scoring batch,
`models.regression`) et les copies avaient divergé — deux d'entre elles omettaient la
marge brute, surestimant l'exposition de +38,9 % (= 1 / 0,72) par rapport à la formule
défendue en §12.11 du notebook. Rien ne le
signalait : chaque implémentation était cohérente avec elle-même.

Ces tests verrouillent trois propriétés :

1. la formule de `churn_saas.economie` est bien celle du dossier (marge brute incluse) ;
2. tous les consommateurs — API, flow batch, `models.regression` — en dérivent, donc
   produisent **le même euro** pour le même compte ;
3. chacun expose la manquance du MRR selon sa convention (`null` + motif en JSON,
   `NaN` + `mrr_disponible` en Parquet), sans jamais imputer un montant.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from flows import scoring_batch

from churn_saas import config, economie
from churn_saas.api.main import app
from churn_saas.api.model_store import ModelStore, get_model_store
from churn_saas.models import regression as reg

_CHEMIN_MODELE = config.ARTIFACTS / "models" / "best_model.pkl"

_COMPTE: dict[str, Any] = {
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


# ---------------------------------------------------------------------------
# 1. La formule de référence
# ---------------------------------------------------------------------------


def test_formule_inclut_la_marge_brute() -> None:
    """valeur_à_risque = P × MRR × horizon × marge — la marge n'est pas optionnelle.

    Un euro de MRR perdu ne coûte pas un euro de résultat mais sa marge : l'omettre
    gonfle l'exposition de 1 / marge, soit +39 % avec les hypothèses du dossier.
    """
    horizon = int(config.HYPOTHESES_ECONOMIQUES["horizon_mois"])
    marge = float(config.HYPOTHESES_ECONOMIQUES["marge_brute_pct"])
    attendu = 0.5 * 1_000.0 * horizon * marge
    assert economie.valeur_a_risque(0.5, 1_000.0) == pytest.approx(attendu)
    # Sans marge, on obtiendrait 1 / 0,72 ≈ 1,389 fois ce montant
    assert economie.valeur_a_risque(0.5, 1_000.0) < 0.5 * 1_000.0 * horizon


def test_formule_lit_les_hypotheses_de_config() -> None:
    """Horizon et marge viennent de `config.HYPOTHESES_ECONOMIQUES`, pas de constantes."""
    assert economie.valeur_a_risque(0.4, 500.0, horizon_mois=6, marge_brute=0.5) == pytest.approx(
        0.4 * 500.0 * 6 * 0.5
    )


def test_formule_vectorisee() -> None:
    """La formule s'applique élément par élément sur un vecteur (usage §12.11)."""
    probas = np.array([0.1, 0.5, 0.9])
    mrr = np.array([1_000.0, 2_000.0, 3_000.0])
    obtenu = economie.valeur_a_risque(probas, mrr)
    attendu = np.array(
        [economie.valeur_a_risque(float(p), float(m)) for p, m in zip(probas, mrr, strict=True)]
    )
    assert isinstance(obtenu, np.ndarray)
    np.testing.assert_allclose(obtenu, attendu)


def test_regression_delegue_a_economie() -> None:
    """`models.regression.valeur_a_risque` (appelée en §12.11) n'est plus une copie."""
    assert reg.valeur_a_risque(0.3, 1_500.0) == pytest.approx(
        economie.valeur_a_risque(0.3, 1_500.0)
    )


# ---------------------------------------------------------------------------
# 2. Parité entre les consommateurs
# ---------------------------------------------------------------------------


def test_flow_batch_utilise_la_formule_commune() -> None:
    """Le flow de scoring batch dérive de `economie`, à l'arrondi de publication près."""
    obtenu = scoring_batch._valeur_a_risque(3_200.0, 0.62)
    assert obtenu == pytest.approx(round(float(economie.valeur_a_risque(0.62, 3_200.0)), 2))


@pytest.mark.skipif(
    not _CHEMIN_MODELE.exists(),
    reason="best_model.pkl absent — exécutez `churn-saas train` pour l'entraîner.",
)
def test_parite_api_flow_batch() -> None:
    """API et batch annoncent le **même euro** pour le même compte et la même probabilité.

    C'est la propriété que la triple copie de la formule ne garantissait plus : le batch
    nocturne publiait au CRM des montants incompatibles avec ceux de la fiche compte.
    """
    store = ModelStore()
    store.charger()
    app.dependency_overrides[get_model_store] = lambda: store
    try:
        corps = TestClient(app).post("/predict", json=_COMPTE).json()
    finally:
        app.dependency_overrides.clear()

    valeur_flow = scoring_batch._valeur_a_risque(
        _COMPTE["revenu_mensuel_recurrent_eur"], corps["probabilite_churn"]
    )
    # L'API calcule la valeur avec la probabilité exacte mais renvoie la probabilité arrondie à
    # 4 décimales, que le batch réutilise ici : l'écart admis est celui de cet arrondi
    # (± 0,00005 × valeur par point de probabilité), pas une tolérance arbitraire en euros
    euros_par_unite_de_proba = economie.valeur_a_risque(
        1.0, _COMPTE["revenu_mensuel_recurrent_eur"]
    )
    tolerance = 0.5e-4 * euros_par_unite_de_proba + 0.01  # + arrondi au centime des montants
    assert corps["valeur_a_risque_eur"] == pytest.approx(valeur_flow, abs=tolerance)


# ---------------------------------------------------------------------------
# 3. Politique de manquance — chacun sa convention, aucun montant imputé
# ---------------------------------------------------------------------------


def test_flow_batch_mrr_absent_donne_nan() -> None:
    """Un MRR inconnu ne produit aucun montant côté batch (NaN, pas 0 €)."""
    assert math.isnan(scoring_batch._valeur_a_risque(float("nan"), 0.8))


def test_synthese_declare_le_perimetre_valorise() -> None:
    """La synthèse JSON déclare le périmètre : total, comptes valorisés, comptes sans MRR.

    `Series.sum()` ignore les NaN silencieusement — sans ces compteurs, le total porterait
    sur un sous-ensemble sans que le CRM puisse le savoir.
    """
    resultats = pd.DataFrame(
        {
            "client_id": ["A", "B", "C"],
            "probabilite_churn": [0.10, 0.50, 0.90],
            "decision": ["OK", "SURVEILLANCE", "ALERTE_ROUGE"],
            "valeur_a_risque_eur": [100.0, float("nan"), 900.0],
            "mrr_disponible": [True, False, True],
            "seuil_applique": 0.40,
            "action_cs": ["Veille", "Veille", "Contacter"],
        }
    )
    synthese = scoring_batch._synthese(resultats, "2026-09-27")
    assert synthese["n_comptes"] == 3
    assert synthese["nb_comptes_valorises"] == 2
    assert synthese["nb_comptes_sans_mrr"] == 1
    assert synthese["valeur_totale_a_risque_eur"] == pytest.approx(1_000.0)
    assert synthese["nb_comptes_valorises"] + synthese["nb_comptes_sans_mrr"] == 3


def test_valeur_attendue_intervention() -> None:
    h = config.HYPOTHESES_ECONOMIQUES
    attendu = (
        0.5 * 1_000.0 * h["horizon_mois"] * h["marge_brute_pct"] * h["taux_succes_retention"]
        - economie.cout_intervention()
    )
    assert economie.valeur_attendue_intervention(0.5, 1_000.0) == pytest.approx(attendu)


def test_selection_sous_capacite_ordre_et_exclusions() -> None:
    # Meilleures valeurs positives d'abord ; négatif et MRR inconnu (NaN) jamais retenus
    valeurs = np.array([5.0, -1.0, float("nan"), 50.0, 10.0])
    assert list(economie.selection_sous_capacite(valeurs, 2)) == [3, 4]
    assert list(economie.selection_sous_capacite(valeurs, 10)) == [3, 4, 0]


def test_batch_priorise_par_valeur_attendue() -> None:
    probas = np.array([0.9, 0.5, 0.01, 0.6])
    mrr = np.array([50.0, 5_000.0, 100.0, float("nan")])
    rangs, actions, _ = scoring_batch._prioriser(probas, mrr)
    # Le gros compte passe devant le petit compte très risqué ; MRR inconnu → veille
    assert rangs[1] == 1 and rangs[0] == 2
    assert actions[3] == "Veille" and rangs[3] is None
    assert actions[2] == "Veille"
