# %% [markdown]
# ## 15. Annexes
#
# Cette section regroupe les éléments de référence utiles au jury sans alourdir le corps
# principal : environnement de reproduction, hyperparamètres retenus, grille de couverture
# C1→C9 item par item du référentiel, liste des utilitaires du package, et glossaire.

# %%
import importlib.metadata
import json
import subprocess
import sys
import warnings
from pathlib import Path

import pandas as pd
from IPython.display import Markdown, display

from churn_saas import config
from churn_saas.cache import charger_ou_calculer

warnings.filterwarnings("ignore")

# %% [markdown]
# ### 15.1 Environnement de reproduction

# %%
# Python et plateforme
_py_version = sys.version
_py_build = sys.version_info

display(
    Markdown(
        f"**Python** : `{_py_version}`  \n"
        f"**Plateforme** : `{sys.platform}`  \n"
        f"**Encodage** : `{sys.getdefaultencoding()}`"
    )
)

# %% [markdown]
# #### Versions des bibliothèques principales

# %%
# Bibliothèques utilisées dans le projet (capturées, pas recopiées)
_BIBS_CLES = [
    "scikit-learn",
    "xgboost",
    "pandas",
    "numpy",
    "matplotlib",
    "shap",
    "optuna",
    "mlflow",
    "fastapi",
    "evidently",
    "codecarbon",
    "joblib",
    "loguru",
    "pydantic",
    "prefect",
    "imbalanced-learn",
    "uv",
]

_lignes_versions = []
for _bib in _BIBS_CLES:
    try:
        _ver = importlib.metadata.version(_bib)
        _lignes_versions.append({"Bibliothèque": _bib, "Version": _ver})
    except importlib.metadata.PackageNotFoundError:
        _lignes_versions.append({"Bibliothèque": _bib, "Version": "non installée"})

_df_versions = pd.DataFrame(_lignes_versions).set_index("Bibliothèque")
display(_df_versions)

# %% [markdown]
# #### Procédure exacte de reproduction

# %%
display(
    Markdown(
        f"""
**Procédure de reproduction complète** (depuis un dépôt cloné) :

```bash
# 1. Cloner le dépôt (ou décompresser le ZIP de livraison)
git clone <URL_DEPOT>
cd churn_saas

# 2. Installer l'environnement Python 3.12 avec uv (gestionnaire d'environnement)
uv sync --frozen          # reproduit exactement le pyproject.lock

# 3. Restaurer les données (exclues de git — présentes dans le ZIP de livraison)
#    Copier data/ et mlruns/ à la racine du projet

# 4. Régénérer le notebook avec recalcul complet (toutes les étapes lourdes)
make notebook-full        # recalcule Optuna, SHAP, CodeCarbon, Evidently

# 5. Régénérer uniquement (avec cache — < 10 min)
make notebook             # charge les artefacts pré-calculés depuis reports/tables/

# 6. Vérifier la qualité (lint + tests + build)
make check                # ruff + black + mypy + pytest + notebook
```

**Graine unique** : `config.RANDOM_SEED = {config.RANDOM_SEED}` — transmise à sklearn, Optuna,
NumPy et tous les composants stochastiques.

**Durée estimée** :
- `make notebook-full` (premier lancement) : ~25–40 min (Optuna 30 essais, SHAP 1 000 obs.)
- `make notebook` (régénération avec cache) : < 5 min
"""
    )
)

# %% [markdown]
# ### 15.2 Hyperparamètres retenus

# %%
_optuna_files = sorted(Path(config.TABLES).glob("optuna_*_meilleurs_params.json"))
if _optuna_files:
    _optuna = json.loads(_optuna_files[0].read_text(encoding="utf-8"))
    _best_params = _optuna.get("best_params", {})
    _nom_optimise = _optuna.get("nom_modele", "N/A")
    _pr_auc_defaut = float(_optuna.get("pr_auc_defaut", float("nan")))
    _pr_auc_best = float(_optuna.get("best_value", float("nan")))
    _n_essais = int(_optuna.get("n_essais_completes", 0))

    display(
        Markdown(
            f"**Modèle optimisé** : `{_nom_optimise}`  \n"
            f"**Protocole** : Optuna TPE + MedianPruner, {_n_essais} essais,  \n"
            f"StratifiedKFold(n_splits=5, random_state={config.RANDOM_SEED})  \n"
            f"**PR-AUC par défaut** : {_pr_auc_defaut:.4f}  \n"
            f"**PR-AUC optimisé** : {_pr_auc_best:.4f}  \n"
            f"**Gain Optuna** : {_pr_auc_best - _pr_auc_defaut:+.4f}  \n"
        )
    )

    _df_params = pd.DataFrame(
        [{"Hyperparamètre": k, "Valeur retenue": str(v)} for k, v in _best_params.items()]
    ).set_index("Hyperparamètre")
    display(_df_params)
else:
    display(Markdown("⚠️ Fichier `optuna_*_meilleurs_params.json` absent — relancer `make notebook-full`."))

# %% [markdown]
# **Ce qu'il faut retenir.**
# Les hyperparamètres ci-dessus sont ceux du modèle déployé (`reports/tables/modele_final.joblib`).
# Ils sont logués dans MLflow (§9.5) pour chaque run — reproducibles via :
# `mlflow runs get --run-id <RUN_ID> | jq '.run.data.params'`.

# %% [markdown]
# ### 15.3 Liste des fonctions utilitaires du package

# %%
display(
    Markdown(
        f"""
**Package `churn_saas` — modules et rôles** (installé depuis `src/churn_saas/`) :

| Module | Fonctions clés | Rôle |
|---|---|---|
| `config` | constantes globales | Chemins, graine, hypothèses économiques, colonnes interdites |
| `cache` | `charger_ou_calculer()` | Cache joblib/parquet/json des étapes lourdes (Optuna, SHAP, OOF) |
| `data.quality` | `profil_compact()` | Résumé dense d'un DataFrame sans afficher les données brutes |
| `features.build` | `ajouter_features_metier()` | Features dérivées (ratio adoption, ancienneté catégorisée…) |
| `models.train` | `construire_modeles()`, `comparer_modeles()`, `optimiser()`, `mesurer_latence()` | Pipeline CV, comparaison, tuning Optuna, mesure de latence |
| `models.economics` | `matrice_couts()`, `gain_par_seuil()`, `precision_at_k()`, `courbe_lift()`, `sensibilite_seuil()`, `table_de_decision()` | Calculs économiques, seuil optimal, ROI, lift |
| `models.evaluate` | `courbe_roc()`, `courbe_precision_rappel()`, `courbe_calibration()`, `tableau_metriques()`, `matrice_confusion()`, `analyse_erreurs()` | Métriques de classification |
| `models.explain` | `importance_impurete()`, `importance_permutation()`, `valeurs_shap()`, `importance_drop_column()`, `verdict_leurres()` | Explicabilité MDI / permutation / SHAP / drop-column |
| `models.regression` | `features_regression()`, `entrainer_modeles_clv()`, `valeur_a_risque()` | Régression CLV + valeur à risque |
| `fuite` | `cribler_leurres()` | Détection automatisée des leurres (chi-deux, point bisériel) |
| `monitoring` | `psi()`, `ks_test()`, `simuler_derive()`, `tester_robustesse()`, `indicateur_obsolescence()`, `rapport_evidently()` | Monitoring de dérive, robustesse, rapport HTML Evidently |
| `viz` | `figure()`, `sauvegarder()` | Style unique, numérotation, sauvegarde automatique dans `reports/figures/` |
| `api.main` | FastAPI app | Endpoints `/predict`, `/predict-batch`, `/health`, `/ready`, `/metrics` |
| `api.schemas` | Pydantic models | Validation des entrées/sorties de l'API |
| `api.security` | `verifier_cle_api()`, rate limiter | Authentification et limitation de débit |
"""
    )
)

# %% [markdown]
# ### 15.4 Grille de couverture C1 → C9

# %%
# Intitulés et items recopiés mot pour mot du référentiel CISIA (docs/CONTEXTE_EPREUVE.md §5).
_INTITULES = {
    "C1": "C1 — Identifier un jeu de données répondant aux besoins métiers",
    "C2": "C2 — Identifier les risques éthiques et sociétaux",
    "C3": "C3 — Préparer les données",
    "C4": "C4 — Choisir un modèle IA",
    "C5": "C5 — Entraîner le modèle",
    "C6": "C6 — Implémenter le modèle",
    "C7": "C7 — Architecture cible",
    "C8": "C8 — Mesurer la performance et les impacts",
    "C9": "C9 — Amélioration continue",
}

# (compétence, item du référentiel, section(s) du notebook, preuve)
_ITEMS = [
    # C1
    ("C1", "Les besoins métiers sont correctement identifiés", "§2.1, §2.4",
     "Contexte éditeur SaaS, besoins Customer Success, critères de réussite chiffrés"),
    ("C1", "Les cas d'usage sont correctement décrits", "§2.2",
     "3 cas d'usage : revue hebdomadaire, alerte, préparation de renouvellement"),
    ("C1", "Les données pertinentes (et nécessaires a minima) sont identifiées", "§3.2, §5.2",
     "Dictionnaire de données avec colonne « pertinence a priori »"),
    ("C1", "L'existence, la disponibilité et l'accès des données sont vérifiés", "§3.1",
     "Contrôles exécutés : existence, hash SHA-256, volumétrie, fraîcheur, droits d'accès"),
    ("C1", "Des solutions alternatives sont envisagées en cas d'indisponibilité", "§3.4",
     "Tableau enrichissements souhaités × disponibilité × plan B"),
    # C2
    ("C2", "Les chartes éthiques européennes et françaises sont connues et appliquées", "§4.3",
     "7 exigences HLEG (UE) + rapport Villani, recommandations CNIL, charte Impact AI"),
    ("C2", "Les impacts éthiques et sociétaux sont connus et leurs conséquences comprises",
     "§4.1, §4.2, §4.5",
     "RGPD, classification AI Act, prophétie auto-réalisatrice, incitation perverse, boucle "
     "de rétroaction"),
    ("C2", "Les biais potentiels ou existants sont identifiés", "§4.4",
     "Équité mesurée (TPR/FPR) par pays, taille d'entreprise et secteur"),
    ("C2", "Les dilemmes éthiques sont identifiés", "§4.5", "3 dilemmes explicités et arbitrés"),
    ("C2", "Les risques sont portés à la connaissance des acteurs concernés", "§4.6.1, §4.6.2",
     "`docs/RISK_REGISTER.md` + note de synthèse datée au commanditaire"),
    ("C2", "La vérification par les acteurs concernés des problèmes légaux et éthiques du jeu "
     "de données est faite", "§4.6.3", "Fiche de revue DPO / juriste (simulée, assumée)"),
    # C3
    ("C3", "Les données sont correctement nommées ou renommées", "§5.9, §7.3",
     "Tableau de renommage avant/après, convention snake_case en français"),
    ("C3", "Le format des données est adapté à l'usage", "§5.5, §5.6, §7.3",
     "Coercition numérique, dates multi-formats parsées, typage cible"),
    ("C3", "Les données altérées, inexactes ou non pertinentes sont corrigées ou supprimées",
     "§5.4, §5.7, §6.5, §6.10, §7.3",
     "Doublons, valeurs impossibles, fuite `sante_compte_fin_periode`, criblage des leurres"),
    ("C3", "Les traitements effectués sont correctement documentés", "§5.10, §7.8",
     "Tableau de bord qualité, schéma du jeu gold, journaux de bord, `docs/DATASHEET.md`"),
    ("C3", "Le choix du modèle de stockage est adapté", "§3.3",
     "Note d'arbitrage fichier/objet vs relationnel vs documents"),
    ("C3", "Le cycle de vie du jeu de données est documenté", "§3.5",
     "Création, versions, rétention, accès, usages futurs (`docs/DATASHEET.md`)"),
    ("C3", "Le cycle de vie documenté est soumis aux parties prenantes", "§3.6",
     "Trace de soumission datée (DPO, Data Owner, CS Lead) avec retours"),
    # C4
    ("C4", "La pertinence est évaluée grâce aux bons indicateurs (analyse ROC)", "§12.2",
     "Courbe ROC + AUC, PR-AUC comme métrique d'arbitrage justifiée"),
    ("C4", "Les contraintes opérationnelles sont prises en compte", "§8.3, §10.9",
     "Batch nocturne, ~5 000 comptes, intégration CRM, compétences de l'équipe"),
    ("C4", "Les contraintes d'éco-conception sont portées à la connaissance des acteurs",
     "§8.4, §9.8–9.10, §12.13",
     "Empreinte CodeCarbon, note d'arbitrage performance/temps/carbone transmise"),
    ("C4", "Les grandes familles d'algorithmes sont connues", "§8.6",
     "Panorama des familles avec motif de retenue ou d'écartement"),
    ("C4", "La démarche scientifique est correctement documentée", "§8.7, §8.8",
     "Trois baselines et protocole de comparaison écrit avant les résultats"),
    ("C4", "La performance attendue est déterminée (précision, temps de traitement et "
     "d'inférence, énergie)", "§8.2, §9.13, §12.3",
     "Cibles chiffrées a priori, latence unitaire et batch mesurées"),
    ("C4", "Le type de résultat attendu est identifié (probabiliste/déterministe)", "§8.1",
     "Sortie probabiliste assumée, d'où l'exigence de calibration"),
    ("C4", "Le contexte des cas d'usage est pris en compte", "§8.9",
     "Lien explicite avec les 3 cas d'usage de §2"),
    ("C4", "Le modèle d'apprentissage choisi est cohérent avec les résultats attendus", "§8.1",
     "Supervisé : classification binaire + régression"),
    ("C4", "La pertinence des solutions sur l'étagère est évaluée", "§8.5",
     "Build vs buy : Gainsight, ChurnZero, AutoML managé"),
    # C5
    ("C5", "Le modèle est optimisé suivant le contexte", "§9.6, §9.7",
     "Gestion du déséquilibre, Optuna TPE à budget modeste justifié"),
    ("C5", "Le modèle créé est entraîné", "§9.2, §9.3",
     "Stratégie d'entraînement, tableau comparatif des modèles en validation croisée"),
    ("C5", "Le modèle choisi est réentraîné le cas échéant", "§9.12, §13.8",
     "Réentraînement sur train+validation avant gel, plan de réentraînement"),
    ("C5", "Les connaissances sont transférées d'un modèle à l'autre le cas échéant", "§9.11",
     "Transfer learning non applicable ici ; warm_start, réutilisation des hyperparamètres"),
    ("C5", "Les hyperparamètres sont décrits", "§9.7, §15.2",
     "Espace de recherche × valeur retenue × effet observé"),
    ("C5", "Le feature engineering est effectué", "§7.5", "Features métier dérivées"),
    # C6
    ("C6", "Le processus de livraison et de déploiement continu est mis en œuvre",
     "§10.6, §10.7", "CI (lint, types, tests, gate qualité) + job CD, image Docker"),
    ("C6", "Le versioning est implémenté", "§10.4, §10.5, §13.10",
     "Code (git), données (DVC), modèle (MLflow Registry), configuration"),
    ("C6", "Les besoins d'intégration sont documentés", "§10.2, §10.8",
     "Contrat d'API, contrat d'échange CRM : format, fréquence, volumétrie, authentification"),
    # C7
    ("C7", "Les principales architectures et leurs contraintes sont connues", "§11.2, §11.6",
     "3 scénarios : VM + batch, conteneurs managés, cloud managé"),
    ("C7", "Les contraintes économiques des scénarios sont portées à la connaissance des "
     "acteurs", "§11.6, §11.7", "Coût mensuel × complexité × délai × souveraineté, "
     "recommandation"),
    ("C7", "Les acteurs sont interrogés pour préciser les contraintes de généralisation",
     "§11.4, §11.5", "Compte-rendu d'entretien DSI / RSSI / DPO / CS Lead (simulé, assumé)"),
    # C8
    ("C8", "Des indicateurs de performance et seuils associés sont définis", "§8.2, §11.8",
     "Cibles a priori, SLO/SLI et seuils d'alerte"),
    ("C8", "La performance est mesurée grâce au suivi des indicateurs",
     "§12.2, §12.12, §12.13", "Métriques techniques, KPI métier, ROI, carbone"),
    ("C8", "Les résultats sont interprétés et présentés aux interlocuteurs concernés", "§12.15",
     "Note de restitution au commanditaire : ce qui marche, ce qui ne marche pas, décision"),
    ("C8", "Les actions adaptées sont déclenchées en fonction des indicateurs",
     "§12.15, §13.8", "Table de décision seuil → action (réentraîner, alerter, suspendre)"),
    # C9
    ("C9", "Système d'évaluation automatisé et intégré au CI/CD via les pratiques MLOps",
     "§13.2", "`tests/test_model_quality_gate.py` bloquant en CI + gate de promotion MLflow"),
    ("C9", "Les métriques sont intégrées (taux de prévision, robustesse, variations de "
     "performance, obsolescence)", "§13.3–13.6, §13.9",
     "PSI/KS, Evidently, test de robustesse, indicateur d'obsolescence, monitoring"),
    ("C9", "La pertinence des indicateurs est interrogée selon une périodicité définie en "
     "phase de cadrage", "§2.5, §13.11", "Comité trimestriel décidé au cadrage, rituel outillé"),
]

_grille_c = pd.DataFrame(
    _ITEMS, columns=["Compétence", "Item du référentiel", "Section(s)", "Preuve"]
)
_grille_c["Compétence"] = _grille_c["Compétence"].map(_INTITULES)
_grille_c = _grille_c.set_index(["Compétence", "Item du référentiel"])

display(_grille_c.style.set_properties(**{"text-align": "left", "white-space": "pre-wrap"}))
display(
    Markdown(
        f"**{_grille_c.index.get_level_values(0).nunique()} compétences** · "
        f"**{len(_grille_c)} items du référentiel**, chacun relié à au moins une sous-section."
    )
)

# %% [markdown]
# **Ce qu'il faut retenir.**
# Chaque item du référentiel CISIA, repris mot pour mot, est relié aux sous-sections du
# notebook qui en portent la preuve. Le jury peut ainsi aller vérifier chaque point
# directement. La grille de la page de garde en est la synthèse, une ligne par compétence.

# %% [markdown]
# ### 15.5 Glossaire

# %%
display(
    Markdown(
        """
**Glossaire des termes techniques employés dans le notebook :**

| Terme | Définition |
|---|---|
| **AUC / ROC-AUC** | Aire sous la courbe ROC (Receiver Operating Characteristic) — mesure la discrimination globale du modèle sur tous les seuils. Insensible au déséquilibre de classes. |
| **Brier score** | Erreur quadratique moyenne entre probabilités prédites et étiquettes réelles (0 = parfait, 0,25 = aléatoire). Mesure la calibration probabiliste. |
| **Calibration** | Propriété d'un modèle dont les probabilités prédites correspondent aux fréquences observées : P̂(churn) = 0,30 → 30 % de vrais churners dans ce groupe. |
| **Churn** | Résiliation d'un abonnement SaaS par un client. Ici : variable binaire (1 = résiliation dans les 90 jours, 0 = renouvellement). |
| **CLV (Customer Lifetime Value)** | Valeur totale générée par un client sur toute sa durée de vie. Ici : valeur historique cumulée — différente de la valeur future attendue. |
| **class_weight='balanced'** | Paramètre sklearn qui pondère les erreurs inversement à la fréquence de chaque classe — corrige le déséquilibre sans modifier les probabilités. |
| **Drop-column importance** | Méthode d'explicabilité : retirer une feature, réentraîner, mesurer la dégradation. Plus rigoureuse que la permutation (détecte les redondances). |
| **Evidently** | Bibliothèque Python de monitoring ML — génère des rapports HTML de dérive (DataDriftPreset) et de résumé statistique. |
| **Feature engineering** | Construction de nouvelles variables à partir des données brutes (ex. : ratio `utilisateurs_actifs / sieges_souscrits`). |
| **Fuite de données (data leakage)** | Introduction involontaire d'information future dans l'entraînement. Cause principale d'AUC quasi parfaite = fuite présumée. |
| **Gate de promotion** | Règle automatisée bloquant le déploiement d'un modèle challenger si son PR-AUC < PR-AUC champion + marge. |
| **KS test (Kolmogorov-Smirnov)** | Test statistique de comparaison de deux distributions — détecte la dérive d'une feature entre référence et courant. |
| **Lift** | Ratio entre la précision du modèle dans le top-N et la prévalence de base (modèle aléatoire). Lift = 3× → le modèle capture 3 fois plus de churners qu'un tirage aléatoire. |
| **Leurre (feature leurre)** | Variable sans lien causal avec la cible — présente par construction dans le dataset (ex. : couleur du thème d'interface). Identifiée par 3 preuves convergentes. |
| **MLflow** | Plateforme MLOps de suivi d'expériences (logging des métriques, paramètres, artefacts) et de registre de modèles. |
| **MRR (Monthly Recurring Revenue)** | Revenu mensuel récurrent d'un compte — base du calcul de la valeur à risque. |
| **Optuna** | Bibliothèque d'optimisation bayésienne d'hyperparamètres (algorithme TPE + élagage Median). |
| **OOF (Out-of-Fold)** | Prédictions sur des observations n'ayant pas participé à l'entraînement du pli courant — seule estimation non biaisée quand le modèle final est fitté sur toutes les données. |
| **Pipeline sklearn** | Enchaînement de transformations + estimateur — garantit que les transformations apprises (imputation, encodage) ne voient jamais les données de validation. |
| **PR-AUC (Average Precision)** | Aire sous la courbe Précision-Rappel — métrique principale pour les classes déséquilibrées. Pénalise les faux positifs ET les faux négatifs. Valeur de référence = prévalence. |
| **PSI (Population Stability Index)** | Mesure de la dérive d'une distribution entre référence et courant. PSI < 0.10 : stable ; 0.10–0.20 : surveillance ; > 0.20 : dérive significative. |
| **RGPD** | Règlement Général sur la Protection des Données (UE 2016/679) — encadre le traitement des données personnelles. |
| **SHAP** | SHapley Additive exPlanations — attribution additive de la contribution de chaque feature à chaque prédiction, fondée sur la théorie des jeux coopératifs. |
| **SMOTE** | Synthetic Minority Over-sampling Technique — génère des observations synthétiques de la classe minoritaire. Décalibre les probabilités → non retenu en production. |
| **Seuil économique (τ*)** | Seuil de classification choisi par maximisation du gain net espéré (pas 0,5 ni argmax F1). Tient compte de la matrice de coûts métier. |
| **Valeur à risque** | P(churn) × MRR × horizon × marge brute — valeur future espérée perdue si le compte résilie et n'est pas retenu. |
| **Warm start** | Initialisation d'un modèle à partir des arbres du modèle précédent — réduit le temps de réentraînement incrémental. |
"""
    )
)

# %% [markdown]
# ### 15.6 Références bibliographiques

# %%
display(
    Markdown(
        """
**Références :**

- Gainsight (2023). *Customer Success Industry Report*.
- OpenView Partners (2023). *SaaS Benchmarks — Gross Margin & Retention*.
- Villani, C. (2018). *Donner un sens à l'intelligence artificielle*. Rapport au Premier ministre.
- CNIL (2022). *Recommandations sur les systèmes d'IA — Lignes directrices*.
- HLEG AI (2019). *Ethics Guidelines for Trustworthy AI*. Commission européenne.
- Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD 2016.
- Akiba, T. et al. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework*. KDD 2019.
- Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS 2017.
- Siddiqi, N. (2006). *Credit Risk Scorecards*. Wiley (seuils PSI).
- Courty, N. & Domingues, M. (2022). *CodeCarbon: Tracking Carbon Emissions from Machine Learning*. arXiv:2208.02339.
- Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR 12.
- Kohavi, R. (1995). *A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection*. IJCAI 1995.
"""
    )
)

# %% [markdown]
# > ### 📋 Journal de bord — Annexes
# >
# > **Décisions retenues** — Versions capturées par `importlib.metadata` (pas recopiées à la main).
# > Procédure de reproduction en 4 étapes (`uv sync --frozen` → `make notebook-full`).
# > Grille C1→C9 item par item, alignée mot pour mot sur le référentiel CISIA, avec renvoi
# > vers la sous-section qui porte chaque preuve. Pas de colonne « Statut » : c'était une
# > coche saisie à la main, sans valeur de preuve.
# > Glossaire de 25 termes couvrant toute la chaîne (données → ML → déploiement → monitoring).
# >
# > **Alternatives écartées** — `pip freeze` (non reproductible hors venv). Glossaire générique
# > non lié au projet : tous les termes sont définis dans leur contexte churn SaaS.
# >
# > **Difficultés rencontrées** — Certaines bibliothèques (prefect, uv) peuvent ne pas être
# > installées dans tous les environnements ; la cellule gère le `PackageNotFoundError` proprement.
# > La première version de la grille C1→C9 reprenait des intitulés non officiels, décalés de
# > C1 à C7 (ex. C4 « Éco-conception » au lieu de « Choisir un modèle IA ») : réécrite à partir
# > des items du référentiel pour que chaque renvoi pointe vers la bonne section.
# >
# > **Impact sur la suite** — Section terminale : aucune dépendance en aval. Alimente le jury
# > pour la vérification de la couverture C1→C9 et la reproductibilité.
# >
# > **Temps passé** — ~1 h (grille, glossaire, câblage des versions).
