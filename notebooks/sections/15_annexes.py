# %% [markdown]
# ## 15. Annexes
#
# Cette section rassemble ce qui sert à **reproduire**, **auditer** et **comprendre** le projet,
# sans alourdir le corps du notebook : environnement et bibliothèques, hyperparamètres des
# modèles optimisés, carte du code, traçabilité des compétences du référentiel item par item,
# glossaire et références.

# %%
import importlib
import importlib.metadata
import inspect
import pkgutil
import re
import sys
import warnings

import pandas as pd
from IPython.display import Markdown, display
from packaging.requirements import Requirement

import churn_saas
from churn_saas import config
from churn_saas.cache import charger
from churn_saas.format_fr import entier, nombre, nombre_tableau, styler_fr
from churn_saas.referentiel import (
    COMPETENCES,
    localiser_items,
    numero_sous_section,
    relever_titres,
    titre_sans_numero,
)
from churn_saas.synthese import artefact_du_modele

# %% [markdown]
# ### 15.1 Environnement de reproduction
#
# Pour refaire le calcul à l'identique, il faut les mêmes bibliothèques aux mêmes versions. On
# liste donc **toutes** les dépendances déclarées dans `pyproject.toml` (lues dans les
# métadonnées du paquet installé, pas recopiées), avec la version réellement installée.

# %%
_lignes_versions = []
for _exigence in importlib.metadata.requires("churn-saas") or []:
    _req = Requirement(_exigence)
    _extra = re.search(r"extra == ['\"](\w+)['\"]", str(_req.marker or ""))
    try:
        _installee = importlib.metadata.version(_req.name)
    except importlib.metadata.PackageNotFoundError:
        _installee = "non installée"
    _lignes_versions.append(
        {
            "Bibliothèque": _req.name,
            "Groupe (extra)": _extra.group(1) if _extra else "socle",
            "Contrainte déclarée": str(_req.specifier) or "—",
            "Version installée": _installee,
        }
    )
_df_versions = pd.DataFrame(_lignes_versions).sort_values(["Groupe (extra)", "Bibliothèque"])
_non_installees = _df_versions.loc[_df_versions["Version installée"] == "non installée"]
display(
    Markdown(
        f"**Python** `{sys.version.split()[0]}` sur `{sys.platform}` · graine unique "
        f"`config.RANDOM_SEED` = `{config.RANDOM_SEED}` · {entier(len(_df_versions))} "
        f"dépendances déclarées, {entier(len(_non_installees))} non installée(s)"
    )
)
display(styler_fr(_df_versions).hide(axis="index"))

# %% [markdown]
# **Ce qu'il faut retenir.** Le tableau donne la contrainte de version acceptée par le projet et
# la version utilisée pour produire ce notebook. Les versions exactes de toutes les dépendances,
# y compris indirectes, sont figées dans `uv.lock` : c'est ce fichier qui garantit la
# reproduction. Pour tout relancer :
#
# ```bash
# git clone https://github.com/laurent-git-dev/churn_saas_release churn_saas && cd churn_saas
# uv sync --frozen --all-extras   # Python 3.12 et dépendances aux versions exactes de uv.lock
# # copier les données brutes (exclues de git, fournies dans l'archive) dans data/raw/
# make notebook                   # réutilise les étapes lourdes déjà calculées (reports/tables/)
# make notebook-full              # recalcule tout depuis les données brutes (Optuna, SHAP…)
# make check                      # lint, tests, puis exécution complète du notebook
# ```

# %% [markdown]
# Le tableau précédent dit **quelles versions** ; celui-ci dit **à quoi sert** chacune des
# bibliothèques citées en page de garde, et où le notebook l'emploie. Comme en §15.4, les renvois
# sont retrouvés à partir des titres réels des sous-sections.

# %%
# Numéro de chaque sous-section, retrouvé depuis son titre
_numeros = {titre_sans_numero(t): numero_sous_section(t) for t in relever_titres()["titre"]}
_GOLD = "Jeu de données gold — schéma, volumétrie et versioning"
_OPTUNA = "Optimisation des hyperparamètres (Optuna)"
_MLFLOW = "Journalisation MLflow"
_CONTRAT_API = "Contrat d'API — schémas Pydantic"
# fmt: off
_BIBLIOTHEQUES = [
    ("pandas", "Tableaux de données en mémoire : lecture, filtres, jointures, agrégats", "Toute la chaîne : chargement, nettoyage, tableaux de résultats", ["Nettoyage", _GOLD]),
    ("numpy", "Calcul vectorisé sur des tableaux de nombres", "Socle de pandas, scikit-learn et LightGBM ; calculs de métriques", []),
    ("pyarrow", "Lecture et écriture du format colonnaire Parquet", "Stockage du jeu gold et des tables intermédiaires", ["Choix du modèle de stockage — note d'arbitrage", _GOLD]),
    ("dvc", "Versionnement des gros fichiers de données à côté de git, par empreinte", "Version du jeu gold reliée au code qui l'a produit", ["Versioning DVC", "Versioning sur quatre axes"]),
    ("scipy", "Tests statistiques et lois de probabilité", "Hypothèses métier, comparaison des modèles, dérive", ["Test des hypothèses métier H1 à H14", "L'écart est-il significatif ? — Wilcoxon apparié et correction de Holm", "Dérive des entrées — PSI et test de Kolmogorov-Smirnov"]),
    ("matplotlib", "Tracé de figures", "Toutes les figures, au style unique de `churn_saas.viz`", []),
    ("scikit-learn", "Pipelines de transformation, modèles classiques, validation croisée, métriques", "Préparation anti-fuite, régression logistique, forêt aléatoire, évaluation", ["Stratégie anti-fuite — explication avant le code", "Familles candidates pour des données tabulaires", "Métriques techniques de classification"]),
    ("imbalanced-learn", "Rééchantillonnage des classes rares (dont SMOTE), compatible avec les pipelines", "Variante SMOTE, comparée à la pondération des classes", ["Gestion du déséquilibre — class_weight, correction d'intercept, SMOTE"]),
    ("lightgbm", "Boosting de gradient sur arbres de décision, rapide sur données tabulaires", "Famille d'arbres optimisée, concurrente de la régression logistique", ["Familles candidates pour des données tabulaires", _OPTUNA]),
    ("optuna", "Recherche des meilleurs hyperparamètres, chaque essai guidé par les précédents", "Optimisation de la régression logistique et de LightGBM", [_OPTUNA]),
    ("mlflow", "Journal des expériences (réglages, scores, modèles) et registre de modèles versionné", "Traçabilité des entraînements, alias `@production`, règle de promotion", [_MLFLOW, "MLflow — suivi des exécutions, registre et gate de promotion"]),
    ("skops", "Sauvegarde de modèles scikit-learn sans exécuter de code arbitraire au chargement, contrairement à pickle", "Format du modèle archivé dans MLflow", [_MLFLOW]),
    ("shap", "Valeurs de Shapley : part de chaque variable dans le score d'un compte", "Explication globale et compte par compte", ["SHAP — attributions additives (global et local)"]),
    ("codecarbon", "Estimation de l'énergie consommée et des émissions de CO₂ d'un calcul", "Empreinte du tuning et du modèle livré", ["Empreinte carbone du tuning", "Empreinte carbone (ESTIMATION)"]),
    ("fastapi", "Construction d'API web, avec documentation interactive générée", "API de prédiction appelée par le CRM", [_CONTRAT_API, "Modèle déployé et appels réels à l'API"]),
    ("pydantic", "Validation des données par types déclarés", "Rejet d'une requête mal formée avant qu'elle n'atteigne le modèle", [_CONTRAT_API]),
    ("uvicorn", "Serveur web qui fait tourner l'application FastAPI", "Lancement de l'API (`make api`, conteneur Docker)", ["Docker et Docker Compose"]),
    ("prefect", "Orchestration de tâches planifiées : enchaînement, reprises, historique", "Scoring nocturne et réentraînement", ["Démonstration du batch nocturne Prefect", "Plan de réentraînement — déclencheurs et retour arrière"]),
    ("evidently", "Rapports de dérive des données, variable par variable", "Contrôle de dérive sur toutes les variables", ["Rapport Evidently — dérive sur toutes les variables"]),
    ("prometheus-fastapi-instrumentator", "Exposition des compteurs de l'API (requêtes, erreurs, latence) au format Prometheus", "Surveillance du service en production", ["Surveillance du service — alertes, compteurs et fatigue d'alerte"]),
    ("jupytext", "Conversion entre scripts Python et notebooks Jupyter", "Génération de ce notebook depuis `notebooks/sections/`", []),
]
# fmt: on

_lignes_biblio = [
    f"| `{nom}` {importlib.metadata.version(nom)} | {role} | {usage} | "
    f"{', '.join(_numeros.get(c, '⚠️ introuvable') for c in cles) or '—'} |"
    for nom, role, usage, cles in _BIBLIOTHEQUES
]
display(
    Markdown(
        "| Bibliothèque | Ce qu'elle fait | Usage dans le projet | Voir |\n|---|---|---|---|\n"
        + "\n".join(_lignes_biblio)
        + f"\n\n{entier(len(_lignes_biblio))} bibliothèques · renvois introuvables : "
        f"{entier(sum(ligne.count('⚠️') for ligne in _lignes_biblio))}"
    )
)

# %% [markdown]
# **Ce qu'il faut retenir.** Chaque bibliothèque couvre une étape précise du cycle de vie, et
# aucune n'est là « au cas où ». Toutes sont libres et largement adoptées : aucune ne lie le
# projet à un fournisseur, conformément à la contrainte posée par la DSI (§11.4). À zéro renvoi
# introuvable, chaque usage annoncé pointe vers une sous-section qui existe.

# %% [markdown]
# ### 15.2 Hyperparamètres retenus
#
# Un hyperparamètre est un réglage choisi **avant** l'apprentissage (force de la pénalité,
# nombre d'arbres…), à la différence des coefficients, appris sur les données. Optuna a cherché
# les meilleurs réglages de deux familles (§9.7), la régression logistique et LightGBM : on
# relit leurs résultats, puis les réglages du modèle livré, lus dans `modele_final.joblib`.

# %%
# fmt: off
_ROLES_HP = {
    "C": "Inverse de la force de pénalité : petit C = coefficients ramenés vers zéro",
    "max_iter": "Nombre maximal d'itérations accordées au solveur pour converger",
    "class_weight": "Pondère les erreurs selon la rareté de la classe, pour ne pas négliger les churners",
    "l1_ratio": "Part de pénalité L1 dans la pénalité totale : 0 = pénalité L2 pure",
    "solver": "Algorithme d'optimisation des coefficients",
    "random_state": "Graine unique du projet",
    "num_leaves": "Nombre maximal de feuilles par arbre : borne la complexité de chaque arbre",
    "learning_rate": "Poids de chaque nouvel arbre : petit = apprentissage lent mais prudent",
    "n_estimators": "Nombre d'arbres construits l'un après l'autre",
    "min_child_samples": "Nombre minimal de comptes par feuille : évite d'apprendre des cas isolés",
    "reg_lambda": "Pénalité L2 sur les valeurs des feuilles",
    "subsample": "Part des comptes tirée au hasard pour construire chaque arbre",
    "colsample_bytree": "Part des variables tirée au hasard pour construire chaque arbre",
}
# fmt: on


def _tableau_hp(params: dict, optimises: dict) -> pd.DataFrame:
    """Réglages optimisés puis réglages fixés par conception, avec leur rôle."""
    return pd.DataFrame(
        {
            "Hyperparamètre": cle,
            "Valeur": nombre_tableau(v) if isinstance(v := params[cle], float) else str(v),
            "Origine": "Optuna (§9.7)" if cle in optimises else "fixé",
            "Rôle": _ROLES_HP.get(cle, "—"),
        }
        for cle in dict.fromkeys([*optimises, *_ROLES_HP])
        if cle in params
    )


_runs_optuna = [
    charger(f.name) for f in sorted(config.TABLES.glob("optuna_*_meilleurs_params.json"))
]
_tableau_optuna = pd.DataFrame(
    {
        "Modèle": run["modele_nom"],
        "Essais": entier(run["n_essais_completes"]),
        "Durée (s)": nombre(run.get("duree_s"), 0),
        "PR-AUC par défaut": nombre(run["pr_auc_defaut"], 4),
        "PR-AUC optimisée": nombre(run["best_value"], 4),
        "Gain": nombre(float(run["best_value"]) - float(run["pr_auc_defaut"]), 4, signe=True),
    }
    for run in _runs_optuna
)
display(styler_fr(_tableau_optuna).hide(axis="index"))

# Le modèle livré est reconnu à ses hyperparamètres, ceux d'une des deux recherches Optuna
_clf = charger("modele_final.joblib")[-1]
_nom_livre = artefact_du_modele("optuna_*_meilleurs_params.json", _clf.get_params())["modele_nom"]
for _run in _runs_optuna:
    _livre = _run["modele_nom"] == _nom_livre
    _hp = _tableau_hp(_clf.get_params() if _livre else _run["best_params"], _run["best_params"])
    _statut = "Modèle livré" if _livre else "Concurrent optimisé"
    display(Markdown(f"**{_statut}** : {_run['modele_nom']}"))
    display(styler_fr(_hp).hide(axis="index"))

# %% [markdown]
# **Ce qu'il faut retenir.** Les deux recherches Optuna ont disposé du même nombre d'essais,
# évalués sur les mêmes plis : la comparaison des modèles optimisés est équitable (§9.7). Un
# gain faible signifie que les réglages par défaut étaient déjà proches de l'optimum. Les
# tableaux suivants décrivent le modèle livré (réglages optimisés et fixés par conception) et
# son concurrent ; la correction d'intercept, apprise et non réglée, est détaillée en §9.6.
# Chaque exécution est journalisée dans MLflow (§9.5), où ses réglages et ses scores se relisent.

# %% [markdown]
# ### 15.3 Carte du code — package `churn_saas`
#
# Tout le code réutilisable vit dans `src/churn_saas/` ; le notebook l'appelle et commente les
# résultats. Le rôle de chaque module est rédigé ; ses fonctions et classes publiques sont
# **relevées dans le code** à l'exécution, donc toujours à jour.

# %%
# fmt: off
_ROLES_MODULES = {
    "config": "Chemins, graine unique, hypothèses, colonnes interdites (seule autorité anti-fuite)",
    "cache": "Mise en cache des étapes lourdes dans `reports/tables/`",
    "format_fr": "Affichage des nombres, pourcentages et montants à la française",
    "viz": "Style graphique unique, numérotation et sauvegarde des figures",
    "data.loaders": "Chargement des fichiers bruts",
    "data.quality": "Contrôles qualité : doublons, types, dates, valeurs impossibles, manquance (§5)",
    "eda": "Analyse exploratoire, dont le test des hypothèses H1 à H14 (§6)",
    "fuite": "Démonstration des fuites, audit de la CLV, criblage des leurres (§6.4 à §6.10)",
    "features.transformers": "Transformeurs appris dans chaque pli : la garantie anti-fuite (§7.2)",
    "features.build": "Préprocesseur, ratios métier, contrôle de cohérence de l'adoption (§7)",
    "features.enrichissement": "Enrichissement externe simulé : secteur et pays (§7.7)",
    "models.train": "Validation, comparaison, tests statistiques, Optuna, MLflow, promotion (§8 à §10)",
    "models.calibration": "Régression logistique pondérée à probabilités calibrées (§9.6)",
    "models.evaluate": "Courbes et métriques, analyse d'erreurs, équité par sous-groupe (§12)",
    "models.explain": "Importances, SHAP, verdict des leurres, fiches comptes (§12.8 à §12.11)",
    "models.economics": "Coûts, seuil de rentabilité, règle à deux niveaux, capacité (§12.4 à §12.7)",
    "models.regression": "Régression de la valeur vie client (§12.10)",
    "economie": "Valeur à risque, valeur attendue d'un geste, niveau de risque : formule unique",
    "monitoring.drift": "Dérive (PSI, Kolmogorov-Smirnov, Evidently), robustesse, obsolescence (§13)",
    "synthese": "Relecture des résultats clés pour le résumé exécutif et la conclusion",
    "referentiel": "Relevé des titres de sous-section, correspondance affichée en page de garde",
    "cli": "Commande `churn-saas` : contrôle, jeu gold, entraînement, prédiction, réentraînement",
    "api.main": "Application FastAPI : `/predict`, `/predict-batch`, `/health`, `/ready`, `/metrics`",
    "api.schemas": "Contrats d'entrée et de sortie de l'API (Pydantic)",
    "api.security": "Clé d'API, limitation de débit, taille maximale des requêtes",
    "api.model_store": "Chargement unique du modèle en mémoire pour l'API",
}
# fmt: on

_lignes_modules = []
# Importer tous les modules déclenche des avertissements de bibliothèques tierces (MLflow,
# Evidently) sans rapport avec le projet : on les tait le temps de l'inventaire seulement
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for _info in pkgutil.walk_packages(churn_saas.__path__, "churn_saas."):
        _module = importlib.import_module(_info.name)
        _publics = [
            f"`{nom}`" if inspect.isclass(objet) else f"`{nom}()`"
            for nom, objet in vars(_module).items()
            if not nom.startswith("_")
            and (inspect.isfunction(objet) or inspect.isclass(objet))
            and objet.__module__ == _module.__name__
        ]
        _nom = _info.name.removeprefix("churn_saas.")
        if _info.ispkg and not _publics and _nom not in _ROLES_MODULES:
            continue  # paquet-conteneur sans code propre (api, data, features, models)
        _role = _ROLES_MODULES.get(_nom, "⚠️ rôle à documenter")
        _lignes_modules.append(f"| `{_nom}` | {_role} | {', '.join(_publics) or '—'} |")

_orphelins = set(_ROLES_MODULES) - {ligne.split("`")[1] for ligne in _lignes_modules}
display(
    Markdown(
        "| Module | Rôle | Fonctions et classes publiques |\n|---|---|---|\n"
        + "\n".join(_lignes_modules)
        + f"\n\n{entier(len(_lignes_modules))} modules relevés · rôle documenté pour un "
        f"module disparu : {', '.join(sorted(_orphelins)) or 'aucun'}"
    )
)

# %% [markdown]
# **Ce qu'il faut retenir.** Le code est découpé par étape de la chaîne : données, variables,
# modèles, économie, surveillance, API. `features.transformers` porte la garantie anti-fuite :
# chaque transformation apprise y est réapprise dans chaque pli. `economie` porte l'unique
# formule de la valeur à risque : l'API, le calcul nocturne et le notebook ne divergent pas.

# %% [markdown]
# ### 15.4 Traçabilité des compétences du référentiel, item par item
#
# Pour vérifier que rien n'a été oublié, on relie chaque item Cn.k du référentiel à la
# sous-section qui en apporte la preuve principale ; la page de garde n'en donne que le résumé
# par compétence (C1 à C9). Les numéros ne sont pas recopiés : ils sont **retrouvés à partir des
# titres réels**, si bien qu'un titre renommé ou supprimé apparaît aussitôt comme introuvable. La
# correspondance est tenue à un seul endroit, `referentiel.CORRESPONDANCE`, et un test
# automatique (`tests/test_tracabilite.py`) échoue si un item perd sa sous-section.

# %%
_localisation = localiser_items()
_lignes_items = []
for _code_comp, _intitule_comp in COMPETENCES.items():
    _lignes_items.append(f"| **{_code_comp}** | **{_intitule_comp}** | |")
    _items = _localisation[_localisation["code"].str.split(".").str[0] == _code_comp]
    for _item in _items.itertuples():
        _preuve = (
            f"{numero_sous_section(_item.titre)} {titre_sans_numero(_item.titre)}"
            if _item.titre
            else "⚠️ introuvable"
        )
        _lignes_items.append(f"| {_item.code} | {_item.intitule} | {_preuve} |")
_nb_orphelins = int((_localisation["titre"] == "").sum())
display(
    Markdown(
        "| Item | Intitulé | Preuve principale |\n|---|---|---|\n"
        + "\n".join(_lignes_items)
        + f"\n\n{entier(len(COMPETENCES))} compétences · {entier(len(_localisation))} items · "
        f"items sans preuve : {entier(_nb_orphelins)} · sous-sections distinctes mobilisées : "
        f"{entier(_localisation.loc[_localisation['titre'] != '', 'titre'].nunique())}"
    )
)

# %% [markdown]
# **Ce qu'il faut retenir.** Chaque item renvoie à une sous-section précise, et les preuves se
# répartissent sur l'ensemble du notebook, du cadrage (C1) à l'amélioration continue (C9). La
# preuve indiquée est la principale : elle s'appuie souvent sur des sous-sections voisines (une
# mesure, puis son interprétation). À zéro item sans preuve, la grille est complète.

# %% [markdown]
# ### 15.5 Glossaire
#
# Le premier tableau traduit le jargon anglais du texte ; le second définit les notions du projet.
#
# | Terme EN | Traduction FR | Définition vulgarisée |
# |---|---|---|
# | A/B test | test A/B | Deux groupes tirés au hasard reçoivent deux traitements différents ; l'écart de résultat mesure l'effet du traitement. |
# | accuracy | exactitude | Part des prédictions justes, toutes classes confondues ; trompeuse quand une classe est rare. |
# | API | interface de programmation (abréviation d'*application programming interface*) | Porte d'entrée par laquelle un logiciel, ici le CRM, demande un score au modèle. |
# | AutoML | apprentissage automatisé | Service qui choisit et règle seul le modèle ; pratique, mais opaque et lié à son fournisseur. |
# | baseline | référence de base | Modèle très simple qui fixe le niveau à battre pour qu'un modèle plus riche soit utile. |
# | batch | traitement par lot | Calcul lancé à heure fixe sur tout le portefeuille, plutôt qu'à la demande. |
# | bias | biais | Écart systématique des prédictions, au détriment d'un groupe ou dans une direction. |
# | boosting | boosting (renforcement) | Suite de modèles simples où chacun corrige les erreurs du précédent. |
# | bootstrap | rééchantillonnage | On retire au hasard, avec remise, autant de comptes que le jeu en contient, des centaines de fois : la dispersion de la métrique donne son intervalle de confiance. |
# | build vs buy | développer ou acheter | Arbitrage entre construire la solution en interne et acheter un produit du marché. |
# | calibration | calibration | Accord entre probabilité annoncée et fréquence observée : 30 % annoncé, 30 % constaté. |
# | champion | modèle champion | Modèle en place, que tout nouveau modèle doit battre pour le remplacer. |
# | churn | résiliation | Départ d'un client qui ne renouvelle pas son abonnement. |
# | CI/CD | intégration et déploiement continus | À chaque modification du code, tests et construction automatiques, puis mise en service si tout est vert. |
# | class imbalance | déséquilibre de classes | Une issue (ici la résiliation) est bien plus rare que l'autre. |
# | commit | enregistrement (git) | Version identifiée du code ; son identifiant relie un résultat au code exact qui l'a produit. |
# | cron | planificateur de tâches | Outil système qui lance une commande à heure fixe. |
# | cross-validation | validation croisée | On découpe les données en parts qui servent tour à tour de test, pour une mesure stable. |
# | dashboard | tableau de bord | Page de suivi qui regroupe les indicateurs clés. |
# | data drift | dérive des données | Les données de production ne ressemblent plus à celles de l'apprentissage. |
# | data leakage | fuite de données | Une variable contient une information connue seulement après coup ; le score est alors trop beau. |
# | data owner | propriétaire des données | Responsable métier qui autorise l'usage d'un jeu de données. |
# | drop-column | retrait de colonne | On retire une variable, on réentraîne et on mesure ce que le modèle perd : test direct de son utilité. |
# | early stopping | arrêt anticipé | On arrête l'apprentissage dès que la performance de validation ne progresse plus. |
# | encoding | encodage | Transformation d'une catégorie (« France ») en nombres exploitables par le modèle. |
# | feature | variable explicative | Information sur le client donnée au modèle pour prédire. |
# | feature engineering | ingénierie des variables | Fabrication de variables plus parlantes à partir des données brutes. |
# | feature importance | importance des variables | Mesure de la part de la prédiction qui repose sur chaque variable. |
# | fit | apprentissage (ajustement) | Étape où le modèle ou la transformation apprend ses paramètres sur des données. |
# | flow | flux orchestré | Enchaînement de tâches planifiées, surveillées et relancées en cas d'échec (ici avec Prefect). |
# | fold | pli | Une des parts de la validation croisée. |
# | gate | règle de promotion | Contrôle automatique qui autorise ou bloque la mise en production d'un modèle. |
# | gold | jeu final (couche « or ») | Dernière couche du stockage Bronze → Silver → Gold : données nettoyées, prêtes pour l'apprentissage. |
# | hyperparameter | hyperparamètre | Réglage fixé avant l'apprentissage (profondeur d'arbre, pénalité…), non appris des données. |
# | imputation | imputation | Remplacement d'une valeur manquante par une valeur plausible (médiane, modalité « inconnu »). |
# | intercept | constante (ordonnée à l'origine) | Terme fixe d'un modèle linéaire, qui règle le niveau moyen des probabilités. |
# | KPI | indicateur clé de performance | Chiffre suivi pour piloter l'activité (gain, précision des appels, fatigue d'alerte). |
# | lift | gain de ciblage | Combien de fois plus de churners on trouve qu'en tirant les comptes au hasard. |
# | log loss | perte logarithmique | Pénalité qui punit surtout les probabilités très sûres d'elles et fausses ; 0 = parfait. |
# | log-odds | logarithme de la cote | Échelle native d'une régression logistique : log(p / (1 − p)), où les effets des variables s'additionnent. |
# | logging | journalisation | Enregistrement horodaté de ce que fait le système, pour pouvoir l'auditer. |
# | MLOps | exploitation des modèles | Pratiques qui automatisent l'entraînement, le déploiement et la surveillance des modèles. |
# | monitoring | surveillance | Suivi continu du modèle en production : données, performance, service. |
# | MRR | revenu mensuel récurrent (abréviation de *monthly recurring revenue*) | Ce que chaque compte paie par mois pour son abonnement ; × 12, il donne le revenu annuel récurrent du portefeuille. |
# | one-hot encoding | encodage disjonctif | Une colonne 0/1 par modalité : « France » devient une colonne qui vaut 1 pour les clients français. |
# | OOF | hors pli (abréviation d'out-of-fold) | Voir *out-of-fold*. |
# | out-of-fold | hors pli | Prédiction faite par un modèle qui n'a jamais vu l'observation concernée. |
# | outlier | valeur extrême | Valeur très éloignée des autres, erreur de saisie ou cas réel rare. |
# | overfitting | surapprentissage | Le modèle apprend le bruit de ses données et se trompe sur de nouveaux clients. |
# | p-value | valeur p | Probabilité d'observer un écart au moins aussi grand si seul le hasard jouait ; petite, elle rend le hasard peu crédible. |
# | pipeline | pipeline (chaîne de traitement) | Enchaînement transformations + modèle appris d'un bloc, pour éviter toute fuite. |
# | PR-AUC | aire sous la courbe précision-rappel | Qualité de la liste des comptes signalés, tous seuils confondus ; le hasard vaut la prévalence. |
# | precision | précision | Parmi les comptes signalés, part de vrais churners. |
# | recall | rappel | Parmi les vrais churners, part de ceux que le modèle signale. |
# | registry | registre de modèles | Catalogue versionné des modèles, avec leur statut (en test, en production). |
# | release | version publiée | Version du logiciel étiquetée et livrée ; ici, elle déclenche le déploiement. |
# | ROC-AUC | aire sous la courbe ROC | Probabilité qu'un churner reçoive un score plus haut qu'un client fidèle ; le hasard vaut 0,5. |
# | ROI | retour sur investissement | (Gain − coût) / coût : combien chaque euro dépensé rapporte en plus. |
# | runbook | procédure d'exploitation | Fiche pas à pas pour réagir à un incident en production. |
# | scoring | calcul des scores | Attribution d'un score de risque à chaque compte. |
# | SDK | kit de développement | Bibliothèque fournie par un éditeur pour utiliser son service ; elle lie le projet à ce fournisseur. |
# | SHAP | valeurs de Shapley | Méthode qui répartit le score d'un compte entre ses variables : « pourquoi ce compte est à risque ». |
# | SLO | objectif de niveau de service | Engagement chiffré (disponibilité, latence) suivi en continu. |
# | split | découpage | Séparation des données en une partie pour apprendre et une partie pour évaluer. |
# | target encoding | encodage par la cible | Remplace une modalité par le taux de churn observé pour elle ; fuite assurée s'il est appris hors des plis. |
# | temporal leakage | fuite temporelle | Variable mesurée après le moment de la prédiction : indisponible en production. |
# | threshold | seuil | Score au-delà duquel un compte est signalé comme à risque. |
# | tuning | optimisation des hyperparamètres | Recherche systématique des meilleurs réglages du modèle. |
# | underfitting | sous-apprentissage | Le modèle est trop simple pour capter les signaux utiles. |
# | versioning | versionnement | Conservation de chaque version du code, des données et du modèle, pour comparer ou revenir en arrière. |
# | workflow | flux de travail | Suite d'étapes automatisées, ici le fichier d'intégration continue de GitHub. |
#
# | Notion du projet | Définition |
# |---|---|
# | AIPD | Analyse d'impact relative à la protection des données : étude imposée par le RGPD avant un traitement à risque élevé pour les personnes (§3.6). |
# | CLV | *Customer Lifetime Value*, valeur vie client : cible secondaire de régression, jamais variable du modèle de churn (§6.7, §12.10). |
# | Correction d'intercept | Retrait du décalage log((1 − π)/π) que la pondération des classes introduit (π = prévalence) : classement inchangé, probabilités de nouveau calibrées (§9.6). |
# | CRM | Logiciel de gestion de la relation client : c'est là que les CSM voient le score et la fiche de chaque compte (§10.8). |
# | CSM / Customer Success | Chargé de succès client, et l'équipe qu'il forme : appelle les comptes à risque, dans la limite de sa capacité (§2.1, §12.6). |
# | DPO | Délégué à la protection des données : vérifie la conformité au RGPD et a revu le jeu de données (§4.6.3). |
# | Importance par permutation | On mélange une variable et on mesure la perte de performance : plus elle est forte, plus le modèle s'appuie sur la variable (§12.8.2). |
# | Leurre | Variable sans lien causal avec la cible, présente par construction (ex. : couleur du thème d'interface), écartée sur trois preuves convergentes (§6.10, §12.9). |
# | LightGBM / Random Forest | Deux familles d'ensembles d'arbres de décision : arbres construits l'un après l'autre (boosting) ou en parallèle sur des tirages au hasard (§8.6). |
# | Matrice de confusion | Tableau croisant prédictions et réalité : churners détectés, churners manqués, fausses alertes, fidèles écartés (§12.6). |
# | MCAR / MAR / MNAR | Valeur manquante complètement au hasard, au hasard conditionnellement aux autres colonnes, ou non au hasard : le mécanisme dicte le traitement (§5.9). |
# | Prévalence | Part des churners dans les données (§6.1) ; c'est la PR-AUC et la precision d'un modèle sans information. |
# | PSI / KS | Indice de stabilité de population et test de Kolmogorov-Smirnov : deux mesures de dérive d'une variable (PSI ≥ 0,20 = dérive significative, §13.3). |
# | RGPD | Règlement général sur la protection des données : cadre européen de tout traitement de données personnelles (§4.1). |
# | RMSE / MAE / R² | Erreur quadratique moyenne (pénalise les grosses erreurs), erreur absolue moyenne (en euros), part de variance expliquée : métriques de la régression CLV (§12.10). |
# | SaaS B2B | Logiciel vendu par abonnement en ligne (*software as a service*) à des entreprises (*business to business*) (§2.1). |
# | Score de Brier | Écart quadratique moyen entre probabilité annoncée et issue observée : mesure la calibration, 0 = parfait (§12.2.3). |
# | Seuil de vigilance | Seuil fixé pour atteindre le recall visé : premier niveau de la règle de décision (§12.6). |
# | SMOTE | Création de churners synthétiques par interpolation entre churners voisins ; comparée ici à la pondération des classes (§9.6). |
# | Valeur à risque / valeur attendue | P(churn) × MRR × horizon × marge, puis × taux de succès − coût du geste : critère de priorisation sous capacité (§12.6, §12.11). |
# | V de Cramér | Force du lien entre deux variables catégorielles, de 0 (aucun) à 1 (lien parfait) (§5.9, §6.8). |
# | Wilcoxon / Holm | Test qui compare deux modèles pli par pli, et correction qui tient compte du nombre de comparaisons (§9.3.1). |

# %% [markdown]
# **Ce qu'il faut retenir.** Le premier tableau couvre tout le jargon anglais du texte ; le
# second sert aussi d'index, chaque notion renvoyant à la sous-section où elle est employée.

# %% [markdown]
# ### 15.6 Références
#
# **Hypothèses métier**
# - Gainsight (2023). *Customer Success Report* — taux de succès des actions de rétention (§2.3).
# - OpenView Partners (2023). *SaaS Benchmarks* — marge brute SaaS B2B (§2.3).
# - Glassdoor (2024). Salaires observés des Customer Success Managers en France — coût horaire d'un CSM (§2.3).
#
# **Éthique et réglementation**
# - Règlement (UE) 2016/679 du 27 avril 2016, règlement général sur la protection des données (RGPD) (§4.1).
# - Règlement (UE) 2024/1689 du 13 juin 2024 établissant des règles harmonisées concernant l'intelligence artificielle (AI Act) (§4.2).
# - Groupe d'experts de haut niveau sur l'IA (2019). *Lignes directrices en matière d'éthique pour une IA digne de confiance*. Commission européenne (§4.3.1).
# - Villani, C. (2018). *Donner un sens à l'intelligence artificielle*. Rapport au Premier ministre (§4.3.2).
# - CNIL (2022 et suiv.). Ressources et recommandations sur l'intelligence artificielle, dont les fiches pratiques sur le développement des systèmes d'IA (§4.3.2).
# - Impact AI. Charte d'engagements de la coalition française pour une IA responsable (§4.3.2).
# - Hardt, M., Price, E., & Srebro, N. (2016). *Equality of Opportunity in Supervised Learning*. NeurIPS 2016 (critère d'égalité des chances, §4.4).
#
# **Méthodes statistiques et apprentissage**
# - Rubin, D. B. (1976). *Inference and Missing Data*. Biometrika, 63(3), 581–592 (mécanismes MCAR / MAR / MNAR, §5.9).
# - Kohavi, R. (1995). *A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection*. IJCAI 1995 (§8.8, §9.14).
# - Saito, T., & Rehmsmeier, M. (2015). *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets*. PLoS ONE, 10(3) (§8.8.1).
# - Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32 (§8.6).
# - Ke, G. et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS 2017 (§8.6).
# - King, G., & Zeng, L. (2001). *Logistic Regression in Rare Events Data*. Political Analysis, 9(2), 137–163 (correction d'intercept, §9.6).
# - Platt, J. (1999). *Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods*. Advances in Large Margin Classifiers, MIT Press (§9.6).
# - Chawla, N. V. et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique*. Journal of Artificial Intelligence Research, 16, 321–357 (§9.6).
# - Niculescu-Mizil, A., & Caruana, R. (2005). *Predicting Good Probabilities with Supervised Learning*. ICML 2005 (calibration, §12.2.3).
# - Brier, G. W. (1950). *Verification of Forecasts Expressed in Terms of Probability*. Monthly Weather Review, 78(1), 1–3 (§12.2.3).
# - Wilcoxon, F. (1945). *Individual Comparisons by Ranking Methods*. Biometrics Bulletin, 1(6), 80–83 (§9.3.1).
# - Holm, S. (1979). *A Simple Sequentially Rejective Multiple Test Procedure*. Scandinavian Journal of Statistics, 6(2), 65–70 (§9.3.1).
# - Benjamini, Y., & Hochberg, Y. (1995). *Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing*. Journal of the Royal Statistical Society, Series B, 57(1), 289–300 (§6.8, §6.10).
# - Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS 2017 (SHAP, §12.8.3).
# - Siddiqi, N. (2006). *Credit Risk Scorecards*. Wiley (seuils du PSI, §13.3).
#
# **Outils**
# - McKinney, W. (2010). *Data Structures for Statistical Computing in Python*. Proceedings of the 9th Python in Science Conference, 56–61 (pandas).
# - Harris, C. R. et al. (2020). *Array Programming with NumPy*. Nature, 585, 357–362.
# - Virtanen, P. et al. (2020). *SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python*. Nature Methods, 17, 261–272.
# - Hunter, J. D. (2007). *Matplotlib: A 2D Graphics Environment*. Computing in Science & Engineering, 9(3), 90–95.
# - Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825–2830.
# - Lemaître, G., Nogueira, F., & Aridas, C. K. (2017). *Imbalanced-learn: A Python Toolbox to Tackle the Curse of Imbalanced Datasets in Machine Learning*. Journal of Machine Learning Research, 18(17), 1–5 (§9.6).
# - Zaharia, M. et al. (2018). *Accelerating the Machine Learning Lifecycle with MLflow*. IEEE Data Engineering Bulletin, 41(4), 39–45 (§9.5, §10.4).
# - Akiba, T. et al. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework*. KDD 2019 (§9.7).
# - Bergstra, J., Bardenet, R., Bengio, Y., & Kégl, B. (2011). *Algorithms for Hyper-Parameter Optimization*. NeurIPS 2011 (échantillonneur TPE d'Optuna, §9.7).
# - Lacoste, A. et al. (2019). *Quantifying the Carbon Emissions of Machine Learning*. arXiv:1910.09700 (méthode reprise par CodeCarbon, §9.8, §12.13).
# - CodeCarbon (logiciel libre). <https://github.com/mlco2/codecarbon> (§9.8, §12.13).
# - Les autres bibliothèques du §15.1 (DVC, skops, FastAPI, Pydantic, Uvicorn, Prefect, Evidently, Jupytext…) n'ont pas de publication de référence : leur documentation officielle fait foi, aux versions indiquées.
#
# **Sécurité et exploitation**
# - ISO/IEC 27001:2022. *Systèmes de management de la sécurité de l'information — Exigences* (exigence d'hébergement du RSSI, §11.4).
# - Center for Internet Security. *CIS Docker Benchmark* (règles de durcissement du conteneur, §10.7).
