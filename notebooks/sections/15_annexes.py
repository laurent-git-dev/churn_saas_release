# %% [markdown]
# ## 15. Annexes
#
# Cette section regroupe les éléments de référence utiles au jury sans alourdir le corps
# principal : environnement de reproduction, hyperparamètres retenus, grille de couverture
# C1→C9, liste des utilitaires du package, et glossaire.

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
_grille_c = pd.DataFrame(
    [
        {
            "Compétence": "C1 — Cadrage du problème IA",
            "Sections couvrant l'item": "§2 (cas d'usage, valeur attendue), §4 (éthique, conformité, RGPD), §8 (contraintes fixées a priori)",
            "Éléments de preuve dans le notebook": "Tableau des 3 cas d'usage · KPI attendus · Colonnes interdites · Cibles de performance",
            "Statut": "✓",
        },
        {
            "Compétence": "C2 — Données et gouvernance",
            "Sections couvrant l'item": "§3 (gouvernance, RGPD, datasheet), §5 (chargement, profil compact), §6 (EDA)",
            "Éléments de preuve dans le notebook": "Datasheet · Contrôle qualité · Analyse univariée/bivariée · Détection des leurres (§6)",
            "Statut": "✓",
        },
        {
            "Compétence": "C3 — Préparation des données",
            "Sections couvrant l'item": "§7 (pipeline sklearn, anti-fuite, split gold)",
            "Éléments de preuve dans le notebook": "Pipeline ColumnTransformer · Colonnes interdites · `test_no_leakage.py` vert · Jeu gold Parquet",
            "Statut": "✓",
        },
        {
            "Compétence": "C4 — Éco-conception",
            "Sections couvrant l'item": "§4 (IA responsable), §9.8–9.9 (CodeCarbon, arbitrage), §9.10 (contraintes opérationnelles)",
            "Éléments de preuve dans le notebook": "Tableau émissions CO₂ (estimation WSL2) · Note arbitrage performance/carbone · Fréquence scoring hebdomadaire",
            "Statut": "✓",
        },
        {
            "Compétence": "C5 — Choix et entraînement du modèle",
            "Sections couvrant l'item": "§8 (sélection justifiée), §9 (5 modèles comparés, Optuna, calibration, transfert)",
            "Éléments de preuve dans le notebook": "Tableau comparatif PR-AUC · Gestion déséquilibre (class_weight vs SMOTE) · Espace de recherche Optuna · Note transfert learning",
            "Statut": "✓",
        },
        {
            "Compétence": "C6 — Mise en exploitation",
            "Sections couvrant l'item": "§10 (API, Docker, CI/CD, versioning, contrat CRM), §11 (architecture cible)",
            "Éléments de preuve dans le notebook": "TestClient (5 codes HTTP vérifiés) · Gate MLflow · Versioning 4 axes · CI GitHub Actions · Docker multi-stage",
            "Statut": "✓",
        },
        {
            "Compétence": "C7 — Documentation et communication",
            "Sections couvrant l'item": "§10 (runbook, contrat API), §14 (restitution commanditaire), §15 (annexes, glossaire)",
            "Éléments de preuve dans le notebook": "Model card · Contrat d'échange CRM · RUNBOOK.md · Table de décision seuil → action",
            "Statut": "✓",
        },
        {
            "Compétence": "C8 — Mesure de performance et impacts",
            "Sections couvrant l'item": "§12 (métriques, ROI, SHAP, leurres, régression CLV, restitution commanditaire)",
            "Éléments de preuve dans le notebook": "PR-AUC / ROC-AUC / Brier OOF · Seuil économique τ* · Fiches comptes SHAP · Verdict leurres 3 preuves · KPI fatigue d'alerte",
            "Statut": "✓",
        },
        {
            "Compétence": "C9 — Amélioration continue",
            "Sections couvrant l'item": "§13 (gate CI/CD, PSI/KS, Evidently, robustesse, obsolescence, flow, comité)",
            "Éléments de preuve dans le notebook": "`test_model_quality_gate.py` · PSI/KS scénarios de dérive · Rapport Evidently · Indicateur obsolescence 180j · Périodicité comité trimestriel",
            "Statut": "✓",
        },
    ]
).set_index("Compétence")

display(_grille_c.style.set_properties(**{"text-align": "left", "white-space": "pre-wrap"}))

# %% [markdown]
# **Ce qu'il faut retenir.**
# Les 9 compétences C1→C9 sont couvertes. Les éléments de preuve sont des références directes
# aux sections et fichiers du notebook — le jury peut naviguer vers chaque section pour vérifier.

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
# > Grille C1→C9 exhaustive avec références directes aux sections du notebook.
# > Glossaire de 25 termes couvrant toute la chaîne (données → ML → déploiement → monitoring).
# >
# > **Alternatives écartées** — `pip freeze` (non reproductible hors venv). Glossaire générique
# > non lié au projet : tous les termes sont définis dans leur contexte churn SaaS.
# >
# > **Difficultés rencontrées** — Certaines bibliothèques (prefect, uv) peuvent ne pas être
# > installées dans tous les environnements ; la cellule gère le `PackageNotFoundError` proprement.
# >
# > **Impact sur la suite** — Section terminale : aucune dépendance en aval. Alimente le jury
# > pour la vérification de la couverture C1→C9 et la reproductibilité.
# >
# > **Temps passé** — ~1 h (grille, glossaire, câblage des versions).
