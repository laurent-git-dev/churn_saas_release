"""Chiffres de synthèse partagés par le résumé exécutif (§1) et la conclusion (§14).

Les deux sections résument les mêmes résultats : elles les obtiennent par cette fonction unique,
ce qui garantit des chiffres identiques au début et à la fin du notebook. Tout est relu depuis
les artefacts de §9 et §12 (lecture seule, sans recalcul d'une étape lourde), dont la note du
jeu de test de §9.14, performance de référence du modèle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import average_precision_score, roc_auc_score

from churn_saas import config
from churn_saas.cache import charger
from churn_saas.economie import valeur_attendue_intervention
from churn_saas.models.economics import (
    concentration_du_gain,
    gain_annuel_attribuable,
    gain_attribuable_au_modele,
    gain_par_seuil,
    gain_sous_capacite,
)
from churn_saas.models.evaluate import equite_par_sous_groupe
from churn_saas.models.regression import selectionner_champion_clv

# Attributs audités pour l'équité, comme en §12.14
ATTRIBUTS_EQUITE: tuple[str, ...] = ("pays", "taille_entreprise", "secteur")


@dataclass(frozen=True)
class Synthese:
    """Résultats clés du projet, relus depuis les artefacts de §9 et §12."""

    y: pd.Series
    mrr: pd.Series
    proba_oof: np.ndarray
    nom_modele_final: str
    comparaison_modeles: pd.DataFrame
    protocole: dict[str, Any]
    evaluation_test: dict[str, Any]
    optuna: dict[str, Any]
    pr_auc_oof: float
    roc_auc_oof: float
    seuil_rentabilite: float
    capacite: int
    n_rentables: int
    attribution: dict[str, Any]
    gain_annuel: dict[str, Any]
    concentration: dict[str, float]
    bilan_proba: dict[str, Any]
    latence: dict[str, Any]
    attributs_equite_a_revoir: list[str]
    resultats_clv: dict[str, Any]
    tableau_clv: pd.DataFrame
    champion_clv_conforme: str | None

    @property
    def bilan(self) -> dict[str, Any]:
        """Bilan de la règle retenue (§12.6) : les comptes de plus forte valeur attendue."""
        return dict(self.attribution["modele"])

    @property
    def reference(self) -> dict[str, Any]:
        """Bilan sans modèle, à budget égal : les plus gros comptes par MRR (§12.12)."""
        return dict(self.attribution["reference"])

    @property
    def test(self) -> dict[str, dict[str, float]]:
        """Note du jeu de test (§9.14) : ``{métrique: {valeur, ic_bas, ic_haut}}``."""
        return dict(self.evaluation_test["test"])

    @property
    def champion_clv(self) -> str:
        """Modèle CLV retenu ; à défaut de modèle conforme, celui de meilleur R²."""
        return self.champion_clv_conforme or str(self.tableau_clv["r2"].idxmax())


def _charger_obligatoire(nom: str) -> Any:
    """Relit un artefact ; s'il manque, on s'arrête plutôt que d'afficher des chiffres faux."""
    objet = charger(nom)
    if objet is None:
        raise FileNotFoundError(
            f"Artefact `{nom}` absent de reports/tables/ : exécuter d'abord §9 à §12 "
            "(ou `make notebook`, qui exécute les sections de synthèse en dernier)."
        )
    return objet


def artefact_du_modele(motif: str, params_modele: dict[str, Any]) -> dict[str, Any]:
    """Artefact JSON dont les hyperparamètres (`best_params`) sont ceux du modèle livré : un
    fichier laissé par un ancien champion ne peut pas être relu par erreur."""
    for chemin in sorted(Path(config.TABLES).glob(motif)):
        contenu: dict[str, Any] = json.loads(chemin.read_text(encoding="utf-8"))
        params = contenu.get("best_params", {})
        if params and all(params_modele.get(k) == v for k, v in params.items()):
            return contenu
    raise FileNotFoundError(f"Aucun artefact `{motif}` ne correspond au modèle livré (§9.15).")


def _attributs_equite_a_revoir(gold: pd.DataFrame, y: pd.Series, proba: np.ndarray) -> list[str]:
    """Attributs dont l'écart de rappel ou de calibration dépasse les seuils de §4.4, au même
    seuil qu'en §12.14 (autant de comptes signalés que de churners)."""
    seuils = config.EQUITE
    seuil_decision = float(np.quantile(proba, 1 - y.mean()))
    equite = equite_par_sous_groupe(
        gold[list(ATTRIBUTS_EQUITE)],
        y,
        proba,
        seuil_decision,
        positifs_min=int(seuils["churners_min"]),
    )
    a_revoir = []
    for attribut, bloc in equite[equite["interpretable"]].groupby(level="attribut"):
        ecart_tpr = float(bloc["tpr"].max() - bloc["tpr"].min())
        calibration_max = float(bloc["ecart_calibration"].abs().max())
        if ecart_tpr > seuils["ecart_tpr_max"] or calibration_max > seuils["ecart_calibration_max"]:
            a_revoir.append(str(attribut))
    return a_revoir


def charger_synthese() -> Synthese:
    """Relit les artefacts de §9 et §12 et calcule les chiffres de synthèse."""
    gold = pd.read_parquet(config.DONNEES_GOLD / "gold_dataset.parquet")
    y = gold["churn"].astype(int)
    mrr_brut = pd.to_numeric(gold["revenu_mensuel_recurrent_eur"], errors="coerce")
    # Même imputation que le bilan économique de §12.1
    mrr = mrr_brut.fillna(float(mrr_brut.median()))

    modele_final = _charger_obligatoire("modele_final.joblib")
    clf_final = modele_final[-1]
    nom_classe = type(clf_final).__name__
    nom_modele_final = (
        "régression logistique optimisée et recalibrée"
        if nom_classe == "RegressionLogistiqueRecalibree"
        else nom_classe
    )
    params_final = clf_final.get_params()

    proba_oof = np.asarray(_charger_obligatoire("probas_oof.joblib"), dtype=float)
    capacite = int(config.HYPOTHESES_ECONOMIQUES["capacite_gestes_mois"])
    _, seuil_rentabilite = gain_par_seuil(y, proba_oof, mrr)

    resultats_clv = _charger_obligatoire("regression_clv.joblib")
    tableau_clv, champion_clv_conforme = selectionner_champion_clv(resultats_clv)

    synthese = Synthese(
        y=y,
        mrr=mrr,
        proba_oof=proba_oof,
        nom_modele_final=nom_modele_final,
        comparaison_modeles=_charger_obligatoire("comparaison_modeles.parquet"),
        protocole=artefact_du_modele("evaluation_protocole_*_optimise.json", params_final),
        evaluation_test=_charger_obligatoire("evaluation_test.json"),
        optuna=artefact_du_modele("optuna_*_meilleurs_params.json", params_final),
        pr_auc_oof=float(average_precision_score(y, proba_oof)),
        roc_auc_oof=float(roc_auc_score(y, proba_oof)),
        seuil_rentabilite=float(seuil_rentabilite),
        capacite=capacite,
        n_rentables=int(
            np.sum(np.asarray(valeur_attendue_intervention(proba_oof, mrr.to_numpy())) > 0)
        ),
        attribution=gain_attribuable_au_modele(y, proba_oof, mrr, capacite=capacite),
        gain_annuel=gain_annuel_attribuable(y, proba_oof, mrr, capacite=capacite),
        concentration=concentration_du_gain(y, proba_oof, mrr, capacite=capacite),
        bilan_proba=gain_sous_capacite(y, proba_oof, mrr, capacite=capacite, classement="proba"),
        latence=_charger_obligatoire("latence_modele_final.json"),
        attributs_equite_a_revoir=_attributs_equite_a_revoir(gold, y, proba_oof),
        resultats_clv=resultats_clv,
        tableau_clv=tableau_clv,
        champion_clv_conforme=champion_clv_conforme,
    )
    logger.info(
        "Synthèse chargée — modèle : {}, PR-AUC hors pli : {:.4f}",
        nom_modele_final,
        synthese.pr_auc_oof,
    )
    return synthese
