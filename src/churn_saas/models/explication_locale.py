"""Explication locale d'un score — facteurs SHAP servis par l'API, sans la librairie `shap`.

Pourquoi un module à part ? L'image d'inférence n'installe que l'extra `api` : `shap`,
`scipy.stats` ou les outils d'analyse de `models.explain` n'y sont pas. Or les deux familles
de champion se prêtent à un calcul **exact** et rapide des valeurs SHAP, sans échantillonnage :

- **modèle linéaire** (régression logistique) : avec des variables traitées comme
  indépendantes, la valeur SHAP d'une variable vaut ``coef × (x − moyenne d'entraînement)``,
  en log-odds. C'est exactement ce que calcule ``shap.LinearExplainer`` avec le jeu
  d'entraînement pour référence. La moyenne est mémorisée à l'apprentissage
  (`RegressionLogistiqueRecalibree.moyenne_entree_`) ;
- **LightGBM** : TreeSHAP est intégré à la librairie (``pred_contrib=True``), en log-odds.

Les contributions des colonnes one-hot d'une même variable (``plan_pro``, ``plan_business``…)
sont additionnées : un CSM lit « Plan : Pro », pas quatre indicatrices. L'additivité des
valeurs SHAP rend cette somme exacte.
"""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.pipeline import Pipeline

from churn_saas.format_fr import nombre_tableau


class FacteurLocal(TypedDict):
    """Une variable qui pousse le score d'un compte vers le churn."""

    variable: str
    libelle: str
    contribution: float


# Libellés métier des variables du modèle, pour les fiches destinées aux CSM : un nom de colonne
# (« derniere_connexion_jours ») n'est pas lisible par le métier.
LIBELLES_VARIABLES: dict[str, str] = {
    "anciennete_mois": "Ancienneté (mois)",
    "sieges_souscrits": "Sièges souscrits",
    "utilisateurs_actifs": "Utilisateurs actifs",
    "taux_adoption_pct": "Taux d'adoption des sièges (%)",
    "connexions_30j": "Connexions sur 30 jours",
    "heures_usage_30j": "Heures d'usage sur 30 jours",
    "fonctionnalites_total": "Fonctionnalités disponibles",
    "fonctionnalites_utilisees": "Fonctionnalités utilisées",
    "nb_integrations": "Intégrations actives",
    "derniere_connexion_jours": "Jours depuis la dernière connexion",
    "tickets_support_90j": "Tickets support sur 90 jours",
    "delai_reponse_support_h": "Délai de réponse du support (h)",
    "csat": "Satisfaction client (CSAT, sur 5)",
    "retards_paiement_12m": "Retards de paiement sur 12 mois",
    "revenu_mensuel_recurrent_eur": "Revenu mensuel récurrent (€)",
    "taux_utilisation_sieges": "Taux d'utilisation des sièges",
    "surdimensionnement": "Sièges payés mais inutilisés",
    "intensite_usage_par_utilisateur": "Heures d'usage par utilisateur",
    "connexions_par_utilisateur": "Connexions par utilisateur",
    "taux_couverture_fonctionnelle": "Part des fonctionnalités utilisées",
    "arpu_par_siege": "Revenu par siège (€)",
    "recence_normalisee": "Inactivité rapportée à l'ancienneté",
    "compte_dormant": "Compte dormant",
    "pression_support": "Pression support (tickets par utilisateur)",
    "csat_manquant": "CSAT non renseigné",
    "heures_usage_30j_manquant": "Heures d'usage non renseignées",
    "delai_reponse_support_h_manquant": "Délai support non renseigné",
    "prix_mensuel_par_siege_eur": "Prix catalogue par siège (€)",
    "fonctionnalites_incluses": "Fonctionnalités incluses dans le plan",
    "sla_reponse_h": "Engagement de réponse du plan (h)",
    "quota_stockage_go": "Quota de stockage du plan (Go)",
    "support_dedie": "Support dédié",
    "remise_consentie": "Remise consentie",
    "adequation_plan": "Adéquation du plan",
    "taux_churn_median_saas_pct": "Churn médian du secteur (%)",
    "dynamique_croissance": "Dynamique du secteur",
    "ecart_adoption_secteur": "Adoption comparée au secteur",
    "zone_reglementaire": "Zone réglementaire",
    "langue_support_fr": "Support en français",
    "decalage_horaire_paris_h": "Décalage horaire avec Paris (h)",
    "ecart_csat_secteur": "CSAT comparé au secteur",
    "secteur": "Secteur",
    "pays": "Pays",
    "taille_entreprise": "Taille d'entreprise",
    "plan": "Plan",
    "tranche_anciennete": "Phase de vie du compte",
    "tranche_integrations": "Niveau d'intégration",
    "couleur_theme_interface": "Thème d'interface",
    "code_datacenter": "Datacenter",
    "groupe_experimentation": "Groupe d'expérimentation",
}


def variable_d_origine(nom_transforme: str, colonnes: list[str]) -> tuple[str, str | None]:
    """Rattache une variable transformée à sa colonne d'origine et, si one-hot, à sa modalité.

    ``plan_business`` → ``("plan", "business")`` ; ``csat`` → ``("csat", None)``. La colonne la
    plus longue qui préfixe le nom l'emporte (``taille_entreprise_pme`` n'est pas ``taille``).
    """
    if nom_transforme in colonnes:
        return nom_transforme, None
    candidates = [c for c in colonnes if nom_transforme.startswith(f"{c}_")]
    if not candidates:
        return nom_transforme, None
    colonne = max(candidates, key=len)
    return colonne, nom_transforme[len(colonne) + 1 :]


def libelle_facteur(nom_transforme: str, compte: pd.Series) -> str:
    """Libellé métier d'un facteur SHAP, avec la valeur du compte : « Plan : business »."""
    colonne, modalite = variable_d_origine(nom_transforme, [str(c) for c in compte.index])
    libelle = LIBELLES_VARIABLES.get(colonne, colonne.replace("_", " ").capitalize())
    if colonne not in compte.index:
        # Variable construite dans le pipeline (ex. écart au groupe) : pas de valeur brute
        return libelle
    valeur = compte.get(colonne)
    if modalite is not None:
        # En SHAP linéaire, une modalité absente du compte contribue aussi (indicatrice à 0)
        if valeur is not None and not pd.isna(valeur) and str(valeur).casefold() != modalite:
            return f"{libelle} : {valeur} (pas {modalite})"
        return f"{libelle} : {modalite if valeur is None or pd.isna(valeur) else valeur}"
    if valeur is None or pd.isna(valeur):
        return f"{libelle} : non renseigné"
    if isinstance(valeur, bool | np.bool_):
        return f"{libelle} : {'oui' if valeur else 'non'}"
    if isinstance(valeur, int | float | np.number):
        return f"{libelle} : {nombre_tableau(float(valeur), 0 if abs(float(valeur)) >= 100 else 2)}"
    return f"{libelle} : {valeur}"


def _transformer(modele: Any, X: pd.DataFrame) -> tuple[Any, Any, list[str]]:
    """(estimateur final, X transformé tel que l'estimateur le reçoit, noms des variables)."""
    if isinstance(modele, Pipeline) and len(modele.steps) > 1:
        pretraitement = modele[:-1]
        noms = [str(n) for n in pretraitement.get_feature_names_out()]
        return modele[-1], pretraitement.transform(X), noms
    estimateur = modele[-1] if isinstance(modele, Pipeline) else modele
    return estimateur, X, [str(c) for c in X.columns]


def _shap(estimateur: Any, X_trans: Any) -> np.ndarray | None:
    """Valeurs SHAP exactes en log-odds, ou None si la famille n'en a pas ici."""
    dense = np.asarray(X_trans.toarray() if hasattr(X_trans, "toarray") else X_trans, np.float64)
    if hasattr(estimateur, "booster_"):
        # LightGBM : dernière colonne = valeur de base (espérance du logit), écartée
        return np.asarray(estimateur.predict(dense, pred_contrib=True))[:, :-1]
    if hasattr(estimateur, "coef_") and hasattr(estimateur, "moyenne_entree_"):
        coefficients = np.asarray(estimateur.coef_, dtype=np.float64).ravel()
        reference = np.asarray(estimateur.moyenne_entree_, dtype=np.float64).ravel()
        return np.asarray((dense - reference) * coefficients)
    logger.warning(
        "Explication locale indisponible pour {} (ni LightGBM, ni modèle linéaire avec "
        "moyenne d'entraînement mémorisée) : réentraîner le modèle pour l'obtenir.",
        type(estimateur).__name__,
    )
    return None


def _principaux_facteurs(
    contributions: np.ndarray | None, noms: list[str], X: pd.DataFrame, nb_facteurs: int
) -> list[list[FacteurLocal]]:
    """Regroupe les indicatrices par variable d'origine et garde les plus fortes hausses."""
    if contributions is None:
        return [[] for _ in range(len(X))]
    colonnes = [str(c) for c in X.columns]
    origines = [variable_d_origine(nom, colonnes)[0] for nom in noms]
    # Somme des indicatrices d'une même variable d'origine (additivité de SHAP)
    par_variable = pd.DataFrame(contributions, columns=noms).T.groupby(origines, sort=False).sum().T

    facteurs: list[list[FacteurLocal]] = []
    for position, (_, ligne) in enumerate(par_variable.iterrows()):
        compte = X.iloc[position]
        principales = ligne[ligne > 0].sort_values(ascending=False).head(nb_facteurs)
        facteurs.append(
            [
                FacteurLocal(
                    variable=str(variable),
                    libelle=libelle_facteur(str(variable), compte),
                    contribution=round(float(valeur), 4),
                )
                for variable, valeur in principales.items()
            ]
        )
    return facteurs


def contributions_locales(modele: Any, X: pd.DataFrame) -> tuple[np.ndarray, list[str]] | None:
    """Valeurs SHAP exactes de chaque compte, en log-odds, par variable transformée.

    Parameters
    ----------
    modele :
        Pipeline sklearn fitté (prétraitement puis classifieur) ou classifieur nu.
    X :
        Comptes au format d'entrée du pipeline (features métier déjà recalculées).

    Returns
    -------
    ``(contributions, noms)`` — matrice *(n_comptes, n_variables_transformées)* et noms des
    variables transformées ; ``None`` si la famille du modèle n'a pas d'explication exacte
    disponible ici (le score reste servi, sans facteurs).
    """
    estimateur, X_trans, noms = _transformer(modele, X)
    contributions = _shap(estimateur, X_trans)
    return None if contributions is None else (contributions, noms)


def facteurs_explicatifs(
    modele: Any, X: pd.DataFrame, nb_facteurs: int = 3
) -> list[list[FacteurLocal]]:
    """Les ``nb_facteurs`` variables qui poussent le plus chaque score vers le churn.

    Seules les contributions positives sont retenues : une variable protectrice n'est pas une
    « raison du risque ». Un compte sans contribution positive reçoit une liste vide, de même
    que tous les comptes si le modèle n'est pas explicable (`contributions_locales`).

    Parameters
    ----------
    modele :
        Pipeline sklearn fitté.
    X :
        Comptes au format d'entrée du pipeline ; ses valeurs servent aux libellés.
    nb_facteurs :
        Nombre maximal de facteurs par compte.

    Returns
    -------
    Une liste de `FacteurLocal` par compte, triée par contribution décroissante.
    """
    estimateur, X_trans, noms = _transformer(modele, X)
    return _principaux_facteurs(_shap(estimateur, X_trans), noms, X, nb_facteurs)


def scorer_et_expliquer(
    modele: Any, X: pd.DataFrame, nb_facteurs: int = 3
) -> tuple[np.ndarray, list[list[FacteurLocal]]]:
    """Probabilités et facteurs explicatifs en **un seul** passage du prétraitement.

    Le prétraitement domine le coût d'une prédiction unitaire : l'exécuter deux fois (score,
    puis explication) doublait la latence de `/predict`.

    Returns
    -------
    ``(probas, facteurs)`` — ``predict_proba`` *(n, 2)* identique à celui du pipeline, et
    les facteurs de `facteurs_explicatifs`.
    """
    estimateur, X_trans, noms = _transformer(modele, X)
    probas = np.asarray(estimateur.predict_proba(X_trans))
    try:
        facteurs = _principaux_facteurs(_shap(estimateur, X_trans), noms, X, nb_facteurs)
    except Exception as erreur:  # noqa: BLE001 — l'explication ne doit pas bloquer le score
        logger.exception("Explication locale en échec, score servi sans facteurs : {}", erreur)
        facteurs = [[] for _ in range(len(X))]
    return probas, facteurs
