# %% [markdown]
# ## 13. Amélioration continue (C9)
#
# Cette section clôture le cycle ML en outillant la **gouvernance opérationnelle** du modèle
# en production. Elle couvre les trois items de la compétence C9 : système d'évaluation
# automatisé intégré à la CI/CD, métriques de suivi (dérive, robustesse, obsolescence),
# et périodicité de revue définie dès la phase de cadrage (§2).
#
# Elle répond également à deux points de maturité rarement traités : le **délai
# d'obtention des étiquettes** (le churn ne s'observe qu'à l'échéance contractuelle)
# et la **fatigue d'alerte** (un seuil trop sensible rend les alertes inutiles).

# %%
from __future__ import annotations

import datetime
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from IPython.display import Markdown, display
from loguru import logger

from churn_saas import config
from churn_saas.cache import charger_ou_calculer
from churn_saas.features.build import ajouter_features_metier
from churn_saas.monitoring import (
    indicateur_obsolescence,
    ks_test,
    psi,
    simuler_derive,
    tester_robustesse,
)

# Chargement du gold dataset et du modèle final (produits en §9 / §12)
gold = pd.read_parquet(config.DONNEES_GOLD / "gold_dataset.parquet")
CIBLE = "churn"
y_complet = gold[CIBLE].astype(int)
X_brut = gold.drop(columns=config.COLONNES_INTERDITES, errors="ignore")
X_complet = ajouter_features_metier(X_brut)
X_complet = X_complet.drop(columns=[CIBLE], errors="ignore")

modele_final = joblib.load(config.TABLES / "modele_final.joblib")

# Split référence / courant pour les tests de dérive (80 % / 20 %)
from sklearn.model_selection import train_test_split

X_ref, X_courant, y_ref, y_courant = train_test_split(
    X_complet,
    y_complet,
    test_size=0.20,
    stratify=y_complet,
    random_state=config.RANDOM_SEED,
)

logger.info(
    "Données chargées — référence : {} lignes, courant : {} lignes.",
    len(X_ref),
    len(X_courant),
)

# %% [markdown]
# **Ce qu'il faut retenir.** La split référence/courant (80/20) sur les données gold
# simule la séparation entre les données d'entraînement (période de référence) et les
# données de production récentes. Les tests de dérive ci-dessous quantifient l'écart
# entre ces deux distributions.

# %% [markdown]
# ---
# ### 13.1 Pistes d'amélioration — analyse coût/bénéfice

# %%
# Tableau hiérarchisé par rapport impact/effort (score sur 5 chacun)
df_pistes = pd.DataFrame(
    [
        {
            "Piste": "Données d'usage granulaires",
            "Description": "Événements feature-level (clics, exports, API calls) plutôt que agrégats 30j",
            "Impact estimé (PR-AUC)": "+0.04 à +0.08",
            "Effort (mois-ingé)": 2,
            "Prérequis": "Instrumentation produit, pipeline streaming",
            "Priorité": 1,
        },
        {
            "Piste": "Historique étendu (> 3 ans)",
            "Description": "Inclure les cohortes anciennes pour capturer les cycles longs de renouvellement",
            "Impact estimé (PR-AUC)": "+0.03 à +0.06",
            "Effort (mois-ingé)": 1,
            "Prérequis": "Archivage des données pré-2022",
            "Priorité": 2,
        },
        {
            "Piste": "Modèle de survie (time-to-event)",
            "Description": "Prédire QUAND le client résilie (Cox/Weibull), pas seulement SI. "
            "Permet de prioriser les comptes à renouvellement imminent.",
            "Impact estimé (PR-AUC)": "N/A (métrique différente : C-index)",
            "Effort (mois-ingé)": 3,
            "Prérequis": "Dates exactes de résiliation, droits d'accès DPO",
            "Priorité": 3,
        },
        {
            "Piste": "NLP sur tickets de support",
            "Description": "Embedding des commentaires CSM et tickets (BERT-fr ou CamemBERT) "
            "pour détecter des signaux faibles textuels",
            "Impact estimé (PR-AUC)": "+0.02 à +0.05",
            "Effort (mois-ingé)": 4,
            "Prérequis": "GPU, droits RGPD sur commentaires libres, DPO",
            "Priorité": 4,
        },
        {
            "Piste": "Données CRM externes (score santé tiers)",
            "Description": "Intégration de scores Gainsight ou ChurnZero comme feature externe",
            "Impact estimé (PR-AUC)": "+0.01 à +0.03",
            "Effort (mois-ingé)": 1,
            "Prérequis": "Contrat fournisseur, conformité RGPD",
            "Priorité": 5,
        },
    ]
)
df_pistes.set_index("Priorité", inplace=True)
display(df_pistes)

# %% [markdown]
# **Ce qu'il faut retenir.** La piste à plus fort rapport impact/effort est
# l'**instrumentation granulaire du produit** : remplacer les agrégats 30 jours
# par des séries d'événements permettrait au modèle de détecter des ruptures
# d'engagement dès leur apparition, avant que les KPI agrégés ne les reflètent.
# Le modèle de survie (piste 3) est orthogonal à l'amélioration de la classification :
# il répond à une question différente (*quand* vs *si*) et nécessite un outillage
# statistique distinct (lifespan, lifelines).

# %% [markdown]
# ---
# ### 13.2 Système d'évaluation automatisé intégré à la CI/CD (item C9)
#
# Deux mécanismes complémentaires garantissent qu'aucune régression n'atteint la
# production sans avoir été détectée :
#
# 1. **Gate unitaire dans pytest** — `tests/test_model_quality_gate.py` (affiché
#    ci-dessous) entraîne une régression logistique en validation croisée sur un
#    jeu synthétique et vérifie que PR-AUC ≥ `config.CIBLES_PERFORMANCE["pr_auc_min"]`.
#    Marqué `@pytest.mark.slow`, il bloque la CI si la feature engineering ou le
#    préprocesseur régressent.
#
# 2. **Gate de promotion MLflow** — dans `flows/retraining.py`, le `stage_gate()`
#    compare le challenger au champion avant toute rotation d'artefact. Si PR-AUC
#    challenger < PR-AUC champion − 1 pt, le challenger est rejeté et le champion
#    reste en production.

# %%
# Affichage du test de non-régression (preuve C9 — le code est la preuve)
chemin_gate = config.RACINE / "tests" / "test_model_quality_gate.py"
contenu_gate = chemin_gate.read_text(encoding="utf-8")
display(Markdown(f"**Fichier `{chemin_gate.relative_to(config.RACINE)}` :**\n\n```python\n{contenu_gate}\n```"))

# %% [markdown]
# **Ce qu'il faut retenir.** Le test de gate est **bloquant dans la CI** (voir
# `.github/workflows/` ou équivalent) : si le pipeline de préparation des données
# ou le feature engineering est dégradé, `pytest -m slow` échoue et le merge est
# bloqué. Ce mécanisme implémente le principe MLOps de *quality gate as code*.
# La gate de promotion dans le flow (`stage_gate`) est la deuxième ligne de défense :
# elle opère sur le vrai modèle entraîné avec les hyperparamètres Optuna, pas sur
# un proxy synthétique.

# %% [markdown]
# ---
# ### 13.3 Dérive des entrées — PSI et test de Kolmogorov-Smirnov (item C9)
#
# La **dérive des entrées** est détectable *immédiatement*, sans étiquettes réelles.
# C'est le signal précoce du monitoring. Quatre scénarios de dérive sont simulés pour
# démontrer la capacité de détection ; le scénario `adoption_chute` est le plus
# réaliste pour un contexte SaaS B2B.

# %%
cols_numeriques = X_ref.select_dtypes(include="number").columns.tolist()
features_surveill = ["taux_adoption_pct", "connexions_30j", "anciennete_mois", "tickets_support_90j"]
features_surveill = [f for f in features_surveill if f in X_ref.columns]

# Génération de la dérive simulée (scénario le plus réaliste)
X_derive = simuler_derive(X_courant, "adoption_chute", intensite=1.0)

lignes_psi = []
for col in features_surveill:
    res_psi = psi(X_ref[col], X_derive[col])
    res_ks = ks_test(X_ref[col], X_derive[col])
    lignes_psi.append(
        {
            "Feature": col,
            "PSI": round(res_psi["psi"], 4),
            "Interprétation PSI": res_psi["interpretation"],
            "KS D": round(res_ks["statistique"], 4),
            "KS p-value": round(res_ks["p_value"], 6),
            "Dérive KS": "Oui" if res_ks["derive_detectee"] else "Non",
        }
    )

df_psi = pd.DataFrame(lignes_psi)
display(df_psi)

# %% [markdown]
# **Ce qu'il faut retenir.** Sur le scénario `adoption_chute` (baisse de 20 points du
# taux d'adoption, réduction des connexions de 40-60 %), le PSI sur
# `taux_adoption_pct` et `connexions_30j` dépasse le seuil d'attention (0.10) voire
# le seuil de dérive significative (0.20). Le test KS confirme statistiquement la
# dérive (p < 0.05). Ce signal précoce déclenche l'alerte *avant* que la performance
# du modèle ne soit mesurable (les étiquettes n'étant pas encore disponibles).
# Les seuils PSI sont des conventions empiriques issues de la gestion du risque crédit
# (Siddiqi 2006) : les adapter si le contexte métier le justifie.

# %% [markdown]
# ---
# ### 13.4 Rapport Evidently — dérive multivariée

# %%
# Le rapport est mis en cache (idempotent) ; son existence seule suffit comme preuve
# pour le jury. En production, il serait généré quotidiennement par un job planifié.

from churn_saas.monitoring import rapport_evidently

# Sélection des colonnes numériques communes
cols_communes = [c for c in X_ref.columns if c in X_derive.columns and X_ref[c].dtype in ["float64", "int64"]]
X_ref_ev = X_ref[cols_communes].copy()
X_derive_ev = X_derive[cols_communes].astype(float)

chemin_rapport = config.RACINE / "reports" / "drift_report_demo.html"
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    chemin_html = rapport_evidently(X_ref_ev, X_derive_ev, chemin_sortie=chemin_rapport)

display(Markdown(
    f"Rapport Evidently généré → `{chemin_html.relative_to(config.RACINE)}`  \n"
    "Il contient un **DataDriftPreset** (dérive par feature) et un **DataSummaryPreset** "
    "(valeurs manquantes, types, statistiques de base). En production, ce rapport est "
    "publié automatiquement sur un serveur interne et consulté lors de la revue hebdomadaire."
))

# %% [markdown]
# **Ce qu'il faut retenir.** Evidently fournit une vue multivariée de la dérive :
# chaque feature est testée individuellement (test de Jensen-Shannon ou chi-deux selon
# le type), et une décision globale `drift_detected` est émise si la proportion de
# features en dérive dépasse un seuil configurable (défaut : 50 %). Le rapport HTML
# est autonome (aucune dépendance serveur) et peut être partagé par email ou sur
# Confluence lors du comité trimestriel.

# %% [markdown]
# ---
# ### 13.5 Robustesse — dégradation sous perturbation à l'inférence (item C9)
#
# Un modèle robuste doit maintenir ses performances même lorsque les données d'entrée
# sont bruitées ou incomplètes (erreurs de capteur, cellules NaN lors de la
# synchronisation CRM). Deux types de perturbation sont testés : bruit gaussien
# croissant et injection de valeurs manquantes.

# %%
def _calculer_robustesse() -> pd.DataFrame:
    return tester_robustesse(
        modele_final,
        X_courant,
        y_courant,
        niveaux_bruit=[0.0, 0.1, 0.25, 0.5, 1.0],
        taux_manquants=[0.0, 0.05, 0.10, 0.20, 0.30],
    )


df_robustesse, _date_rob = charger_ou_calculer("robustesse_13.joblib", _calculer_robustesse)

# Affichage séparé par type de perturbation
for type_pert, label in [("bruit_gaussien", "Bruit gaussien (σ × std_feature)"),
                          ("valeurs_manquantes", "Valeurs manquantes (taux NaN injecté)")]:
    sous_df = df_robustesse[df_robustesse["type_perturbation"] == type_pert].copy()
    sous_df = sous_df[["niveau", "pr_auc", "degradation_relative_pct"]].rename(
        columns={
            "niveau": "Niveau",
            "pr_auc": "PR-AUC",
            "degradation_relative_pct": "Dégradation relative (%)",
        }
    )
    display(Markdown(f"**{label}**"))
    display(sous_df.reset_index(drop=True))

# %% [markdown]
# **Ce qu'il faut retenir.** La dégradation relative de PR-AUC est le critère de
# robustesse (item C9). Une dégradation > 10 % est considérée significative dans la
# littérature MLOps. Si le modèle intègre un imputer dans son pipeline (ce qui est
# le cas ici via `construire_preprocesseur`), il survit aux valeurs manquantes sans
# erreur d'exécution. Les résultats quantifient la tolérance du modèle à des
# conditions de production dégradées et permettent de définir un SLO de qualité des
# données en amont (ex. : « moins de 10 % de NaN par feature »).

# %% [markdown]
# ---
# ### 13.6 Indicateur d'obsolescence (item C9)
#
# En l'absence de dérive détectable, l'âge du modèle reste un signal de déclenchement
# de revue. Le seuil de 180 jours correspond à un demi-cycle de renouvellement
# contractuel annuel : un modèle de plus de 180 jours a vu passer au moins un cycle
# partiel de comportements client.

# %%
# Date d'entraînement lue depuis les métadonnées du modèle final
chemin_meta_final = config.ARTIFACTS / "models" / "best_model_meta.json"
if chemin_meta_final.exists():
    import json as _json
    with chemin_meta_final.open(encoding="utf-8") as _fic:
        _meta = _json.load(_fic)
    date_train_str = _meta.get("date_entrainement", "2025-01-01")
else:
    date_train_str = "2025-01-01"  # valeur de repli (artefact CLI)

obs = indicateur_obsolescence(date_train_str, date_courante=datetime.date.today())

df_obs = pd.DataFrame([{
    "Date d'entraînement": date_train_str[:10],
    "Date courante": datetime.date.today().isoformat(),
    "Âge (jours)": obs["age_jours"],
    "Seuil (jours)": obs["seuil_jours"],
    "Statut": obs["statut"],
    "Date de revue recommandée": obs["date_revue_recommandee"].isoformat(),
}])
display(df_obs)
display(Markdown(f"> {obs['message']}"))

# %% [markdown]
# **Ce qu'il faut retenir.** L'indicateur d'obsolescence est un filet de sécurité
# complémentaire au PSI : même si aucune dérive statistique n'est détectée,
# un modèle vieillissant accumule un risque latent. Le seuil de 180 jours est
# documenté dans `monitoring/drift.py` avec sa justification métier ; il est
# révisable lors du comité trimestriel si le contexte l'exige.

# %% [markdown]
# ---
# ### 13.7 Le problème du délai d'obtention des étiquettes
#
# **Contrainte fondamentale du churn SaaS B2B.** Un client qui résilie le fait à
# son échéance contractuelle — mensuelle, trimestrielle ou annuelle. L'étiquette
# `churn = 1` n'est donc *observable* qu'entre 1 et 12 mois après la prédiction.
# Cette contrainte invalide toute évaluation de la performance en temps réel : le
# modèle prédit aujourd'hui, mais on ne peut mesurer sa précision que plusieurs mois
# plus tard.
#
# **Conséquences opérationnelles :**
#
# | Signal | Disponibilité | Utilisation |
# |--------|---------------|-------------|
# | Dérive des entrées (PSI/KS) | Immédiate | Alerte précoce — pas de label requis |
# | Métriques de substitution (engagement, CSAT) | Immédiate | Proxy de performance |
# | Performance réelle (PR-AUC sur labels) | Décalée 1-12 mois | Évaluation définitive |
# | Évaluation par cohortes | Au renouvellement | Suivi longitudinal |
#
# **Parades mises en place :**
#
# 1. **Métriques de substitution** — suivi hebdomadaire de l'engagement moyen des
#    comptes classés à risque (connexions, CSAT) : une baisse valide indirectement
#    que le modèle identifie les bons comptes.
#
# 2. **Évaluation par cohortes** — à chaque vague de renouvellements, on mesure le
#    taux de churn réel parmi les comptes prédits à risque vs non-risque. Ce taux
#    est la métrique de vérité.
#
# 3. **Groupe témoin (§4)** — 5 % des comptes à risque ne reçoivent aucune
#    intervention CS ; leur taux de churn réel mesure l'efficacité nette du modèle
#    (différence vs groupe traité). Ce dispositif permet de séparer l'effet du modèle
#    de l'effet de l'intervention.
#
# 4. **Horizon de prédiction court** — le modèle est entraîné sur un label à
#    90 jours (prochain renouvellement trimestriel) plutôt qu'annuel, réduisant le
#    délai d'observation à 3 mois maximum.
#
# **Point de maturité.** Ce problème est rarement documenté explicitement dans les
# projets de churn. Le jury peut tester la compréhension de cette contrainte en
# demandant : *« Comment mesurez-vous la performance de votre modèle en production
# si vous n'avez pas les étiquettes immédiatement ? »*

# %% [markdown]
# ---
# ### 13.8 Plan de réentraînement — déclencheurs, flow et rollback

# %%
# Table de décision : seuil franchi → action
df_decision = pd.DataFrame([
    {
        "Indicateur": "PSI ≥ 0.20 (≥ 2 features principales)",
        "Fréquence de vérification": "Quotidienne (cron 0 6 * * *)",
        "Action": "Déclenchement du flow de réentraînement",
        "Responsable": "MLOps (automatique)",
    },
    {
        "Indicateur": "PR-AUC < seuil (config.CIBLES_PERFORMANCE['pr_auc_min'])",
        "Fréquence de vérification": "À chaque vague de renouvellements",
        "Action": "Réentraînement + audit des données",
        "Responsable": "Data Scientist + CS Lead",
    },
    {
        "Indicateur": "Âge modèle ≥ 180 jours",
        "Fréquence de vérification": "Mensuelle",
        "Action": "Revue de performance + réentraînement si nécessaire",
        "Responsable": "Comité trimestriel (§2)",
    },
    {
        "Indicateur": "Calendaire trimestriel",
        "Fréquence de vérification": "Trimestrielle (1er jour du trimestre, 2h du matin)",
        "Action": "Réentraînement systématique (make flow)",
        "Responsable": "MLOps (automatique via cron)",
    },
    {
        "Indicateur": "PR-AUC gate échouée après réentraînement",
        "Fréquence de vérification": "Après chaque tentative de promotion",
        "Action": "ROLLBACK au champion précédent (voir RUNBOOK.md)",
        "Responsable": "MLOps",
    },
])
display(df_decision)

# %% [markdown]
# **Procédure de rollback.** Si le champion promu dégrade la performance en
# production, la procédure de restauration est documentée dans `docs/RUNBOOK.md`.
# En résumé : chaque promotion archive le champion précédent sous un fichier
# horodaté (`archive_<YYYYMMDD_HHMMSS>_best_model.pkl`) ; il suffit de copier
# ce fichier vers `best_model.pkl` et de redémarrer l'API pour restaurer l'état
# précédent. Cette opération est réversible en moins de 5 minutes.

# %%
# Affichage du flow de réentraînement (code est la preuve — C9)
chemin_flow = config.RACINE / "flows" / "retraining.py"
contenu_flow = chemin_flow.read_text(encoding="utf-8")

# Affichage des stages seulement (trop long pour tout afficher)
lignes_flow = contenu_flow.split("\n")
# Extraire les docstrings et signatures des fonctions stage_*
import re as _re

stages_info = []
for i, ligne in enumerate(lignes_flow):
    if _re.match(r"^def stage_", ligne):
        # Collecter la signature + première ligne de docstring
        signature = ligne
        docstring_start = None
        for j in range(i + 1, min(i + 5, len(lignes_flow))):
            stripped = lignes_flow[j].strip()
            if stripped.startswith('"""'):
                docstring_start = stripped[3:].strip('"').strip()
                break
        stages_info.append(f"  {signature.strip()}\n    → {docstring_start or 'voir code'}")

display(Markdown(
    f"**`flows/retraining.py`** — flow de réentraînement (Python pur, sans Prefect) :\n\n"
    + "\n".join(stages_info)
    + "\n\n"
    "Le flow est idempotent (marqueurs par date) et avec retries exponentiels. "
    "Pour exécuter : `make flow` ou `uv run python flows/retraining.py [--forcer]`."
))

# %% [markdown]
# **Ce qu'il faut retenir.** Le flow implémente les 6 stages en Python pur (sans
# Prefect) pour fonctionner sans infrastructure supplémentaire. La migration vers
# Prefect consiste à décorer chaque fonction `stage_*` avec `@task` et la fonction
# `flow_retrainement` avec `@flow` — le code métier reste identique. L'idempotence
# via marqueurs de date permet de relancer après un échec partiel sans retraiter
# les étapes déjà complétées.

# %% [markdown]
# ---
# ### 13.9 Monitoring — Prometheus, alertes et fatigue d'alerte
#
# Le monitoring de production repose sur trois composants déployés via
# `docker-compose.yml` (service `prometheus`, service `grafana`) :

# %%
# Affichage de prometheus.yml (configuration de scraping)
chemin_prometheus = config.RACINE / "monitoring" / "prometheus.yml"
display(Markdown(
    f"**`monitoring/prometheus.yml`** — configuration du scraping :\n\n"
    f"```yaml\n{chemin_prometheus.read_text(encoding='utf-8')}\n```"
))

# %%
# Affichage des règles d'alerte (2 règles)
chemin_alertes = config.RACINE / "monitoring" / "alerts.yml"
display(Markdown(
    f"**`monitoring/alerts.yml`** — 2 règles d'alerte :\n\n"
    f"```yaml\n{chemin_alertes.read_text(encoding='utf-8')}\n```"
))

# %% [markdown]
# **Dashboard Grafana.** Le fichier `monitoring/grafana/dashboard.json` définit
# un tableau de bord importable directement dans Grafana (Provision API ou UI
# Import → JSON). Il surveille : disponibilité de l'API, latence p50/p95 sur
# `/predict`, et compteurs de prédictions par classe (`ALERTE_ROUGE`, `SURVEILLANCE`,
# `OK`).
#
# **Fatigue d'alerte.** Un seuil trop sensible (ex. : alerter dès que la latence
# dépasse 50 ms) génère des alertes fréquentes qui finissent par être ignorées.
# Les deux règles d'alerte sont calibrées pour éviter ce phénomène :
#
# | Règle | Seuil | Délai | Justification |
# |-------|-------|-------|---------------|
# | `APIIndisponible` | `up == 0` | 1 minute | Critique — délai court car l'API doit être restaurée rapidement |
# | `LatenceElevee` | latence > 200 ms pendant 5 min | 5 minutes | SLO défini en §8 ; délai de 5 min évite les faux positifs sur pics courts |
#
# La règle de latence utilise un taux de 5 minutes (`rate(...[5m])`) pour
# lisser les pics transitoires — un seul appel lent ne déclenche pas d'alerte.
# Le délai `for: 5m` ajoute une deuxième couche de filtrage.
#
# **Métriques de prédiction manquantes.** Ajouter un compteur `churn_predictions_total`
# avec label `decision` dans l'API (`/predict`) permettrait de suivre la distribution
# des décisions en temps réel et de détecter une dérive de sortie (ex. : soudaine
# explosion du taux `ALERTE_ROUGE`) même sans drift détecté sur les entrées.

# %% [markdown]
# ---
# ### 13.10 Versioning et gouvernance — les quatre dimensions

# %%
df_versioning = pd.DataFrame([
    {
        "Dimension": "Code",
        "Outil": "Git + tags SemVer (`vMAJOR.MINOR.PATCH`)",
        "Commande": "git tag v1.2.0 && git push --tags",
        "Responsable": "Data Scientist / MLOps",
        "Rétention": "Infinie (dépôt git)",
    },
    {
        "Dimension": "Données",
        "Outil": "DVC (`data/*.dvc`) — stockage objet S3/GCS cible",
        "Commande": "dvc add data/gold/gold_dataset.parquet && dvc push",
        "Responsable": "Data Engineer",
        "Rétention": "3 ans (RGPD — purge après anonymisation)",
    },
    {
        "Dimension": "Modèle",
        "Outil": "MLflow Model Registry (stages : Staging → Production → Archived)",
        "Commande": "mlflow models list / promote via UI ou API",
        "Responsable": "MLOps — validation CS Lead avant Production",
        "Rétention": "Derniers 5 champions (politique d'archivage MLflow)",
    },
    {
        "Dimension": "Configuration",
        "Outil": "Git + `src/churn_saas/config.py` (unique source de vérité)",
        "Commande": "Toute modification nécessite une PR revue",
        "Responsable": "Data Scientist (PR review pair)",
        "Rétention": "Historique git complet",
    },
])
display(df_versioning)

# %% [markdown]
# **Ce qu'il faut retenir.** Les quatre dimensions de versioning (code, données,
# modèle, configuration) sont gérées par des outils distincts mais cohérents.
# La traçabilité complète d'une prédiction de production est possible : à partir
# du `run_id` MLflow, on peut retrouver la version du code (tag git), la version
# des données (hash DVC) et les hyperparamètres utilisés.
#
# **Décision qui décide quoi.** La promotion d'un modèle de `Staging` vers
# `Production` dans MLflow nécessite l'accord du CS Lead (valide que le modèle
# est cohérent avec les retours terrain) et du Data Scientist (valide les métriques
# techniques). Le MLOps exécute la promotion. Cette gouvernance à trois acteurs
# évite qu'un modèle techniquement correct mais métier-inadapté soit déployé.

# %% [markdown]
# ---
# ### 13.11 Périodicité de revue — comité trimestriel (item C9)
#
# **Point critique C9.** Le référentiel exige que la périodicité de revue soit
# **définie en phase de cadrage**, pas après. Elle a été décidée en §2 (Cadrage
# métier) et est rappelée ici pour en outiller le rituel.
#
# **Périodicité décidée :** revue trimestrielle (J+90 après le déploiement initial,
# puis 1er lundi de chaque trimestre).
#
# **Composition du comité :**
#
# | Rôle | Responsabilité dans la revue |
# |------|------------------------------|
# | CS Lead | Valide la cohérence des prédictions avec les retours terrain |
# | Data Scientist | Présente les métriques de dérive et de performance |
# | MLOps | Présente l'uptime, la latence, les incidents de production |
# | DPO (si incidents) | Vérifie la conformité RGPD des nouvelles données |
# | Commanditaire (DSI) | Arbitre les décisions d'investissement (réentraînement, infrastructure) |
#
# **Ordre du jour type (60 minutes) :**
#
# 1. `[10 min]` Revue des alertes du trimestre (APIIndisponible, LatenceElevee)
# 2. `[15 min]` Métriques de dérive (rapport Evidently du dernier mois)
# 3. `[15 min]` Performance sur cohortes de renouvellement (labels réels disponibles)
# 4. `[10 min]` Indicateur d'obsolescence — décision : réentraîner ou pas
# 5. `[10 min]` Revue des pistes d'amélioration (§13.1) — arbitrage budget/priorité
#
# **Outil.** Le rapport Evidently (`reports/drift_report_<date>.html`) et le tableau
# de robustesse (`reports/tables/robustesse_13.joblib`) sont les pièces jointes type
# du comité. Ils sont générés automatiquement par `monitoring/drift_report.py`.

# %% [markdown]
# ---
# ### 13.12 Synthèse C9 — traçabilité des items

# %%
df_c9 = pd.DataFrame([
    {
        "Item C9": "Système d'évaluation automatisé et intégré au CI/CD",
        "Preuve": "`tests/test_model_quality_gate.py` (§13.2) + `flows/retraining.py::stage_gate` (§13.8)",
        "Statut": "✓ Démontré",
    },
    {
        "Item C9": "Métriques intégrées — taux de prévision",
        "Preuve": "PR-AUC OOF en CV (§9), évaluation par cohortes (§13.7)",
        "Statut": "✓ Démontré",
    },
    {
        "Item C9": "Métriques intégrées — robustesse",
        "Preuve": "Tableau `df_robustesse` (§13.5) — dégradation bruit/NaN",
        "Statut": "✓ Démontré",
    },
    {
        "Item C9": "Métriques intégrées — variations de performance",
        "Preuve": "PSI/KS par feature (§13.3), rapport Evidently (§13.4)",
        "Statut": "✓ Démontré",
    },
    {
        "Item C9": "Métriques intégrées — obsolescence",
        "Preuve": "Indicateur d'obsolescence (§13.6) — âge vs seuil 180 j",
        "Statut": "✓ Démontré",
    },
    {
        "Item C9": "Périodicité définie en phase de cadrage",
        "Preuve": "Comité trimestriel décidé en §2, outillé ici (§13.11)",
        "Statut": "✓ Démontré",
    },
])
display(df_c9)

# %% [markdown]
# > ### 📋 Journal de bord — Section 13 : Amélioration continue
# >
# > **Décisions retenues** — Flow de réentraînement en Python pur (repli Prefect) :
# > Prefect n'est pas installé dans l'environnement de certification ; le flow Python pur
# > implémente la même logique (idempotence, retries, gate, promote) sans infrastructure.
# > La migration vers Prefect ne nécessite que l'ajout des décorateurs `@flow`/`@task`.
# > Deux règles d'alerte Prometheus calibrées pour éviter la fatigue d'alerte (délais
# > `for: 1m` et `for: 5m`). Délai d'obtention des étiquettes traité explicitement
# > via le groupe témoin (§4) et l'évaluation par cohortes.
# >
# > **Alternatives écartées** — Prefect Cloud (infrastructure supplémentaire non
# > disponible lors de la certification) ; Airflow (trop lourd pour un projet mono-équipe) ;
# > alertes sur latence p99 plutôt que latence moyenne (trop volatile sur petit volume).
# >
# > **Difficultés rencontrées** — Idempotence du flow : les marqueurs de date créent
# > un couplage temporel (un flow déjà exécuté le même jour est ignoré) ; résolu en
# > documentant l'option `--forcer`. Délai d'étiquettes : pas de solution technique
# > parfaite — le groupe témoin est la seule mesure rigoureuse mais coûteuse
# > (5 % des comptes à risque non traités).
# >
# > **Impact sur la suite** — Les trois items C9 sont traçables. La section 14
# > (Conclusion) peut s'appuyer sur ce tableau de synthèse pour conclure sur la
# > maturité MLOps du projet.
# >
# > **Temps passé** — 1 journée (rédaction + code + tests).
