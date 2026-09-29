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

from churn_saas import config, viz
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

# %%
# Affichage du workflow CI/CD (preuve que le gate est effectivement déclenché sur push/PR)
chemin_ci = config.RACINE / ".github" / "workflows" / "ci.yml"
display(Markdown(
    f"**`.github/workflows/ci.yml`** — pipeline CI/CD complet :\n\n"
    f"```yaml\n{chemin_ci.read_text(encoding='utf-8')}\n```"
))

# %% [markdown]
# **Ce qu'il faut retenir.** Le test de gate est **bloquant dans la CI** : le step
# `Gate qualité modèle (slow)` exécute `pytest -m slow -q` sur tout push vers `main`
# et toute pull request. Si le pipeline de préparation des données ou le feature
# engineering est dégradé, ce step échoue et le merge est bloqué. Ce mécanisme
# implémente le principe MLOps de *quality gate as code*. La gate de promotion dans
# le flow (`stage_gate`) est la deuxième ligne de défense : elle opère sur le vrai
# modèle entraîné avec les hyperparamètres Optuna, pas sur un proxy synthétique.

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
# Import → JSON), organisé en trois rangées : **disponibilité** (statut `up`, débit),
# **latence** (p50/p95/p99 sur `/predict` avec le SLO 200 ms, taux d'erreur 4xx+5xx) et
# **sortie du modèle et qualité des entrées** (prédictions par décision, champs manquants
# reçus). Les deux derniers panels s'appuient sur les compteurs métier de `api/main.py`
# affichés plus bas.
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
# %%
# Extraction des compteurs métier instrumentés dans l'API (preuve C9) — on isole
# chaque déclaration `Counter(...)` complète plutôt que des lignes filtrées par mots-clés
chemin_api = config.RACINE / "src" / "churn_saas" / "api" / "main.py"
lignes_api = chemin_api.read_text(encoding="utf-8").split("\n")

blocs_compteurs: list[str] = []
for _i, _ligne in enumerate(lignes_api):
    if "= Counter(" not in _ligne:
        continue
    _bloc = []
    for _suite in lignes_api[_i:]:
        _bloc.append(_suite)
        if _suite.startswith(")"):
            break
    blocs_compteurs.append("\n".join(_bloc))

# Lignes d'incrémentation — preuve que les compteurs sont réellement alimentés
lignes_inc = [_l.strip() for _l in lignes_api if ".labels(" in _l and ".inc()" in _l]

display(Markdown(
    "**Compteurs Prometheus déclarés dans `api/main.py`** :\n\n"
    f"```python\n{chr(10).join(blocs_compteurs)}\n```\n\n"
    "**Incrémentation** (dans `_construire_resultat` et les routes de prédiction) :\n\n"
    f"```python\n{chr(10).join(lignes_inc)}\n```\n\n"
    f"Nombre de compteurs métier exposés sur `/metrics` : **{len(blocs_compteurs)}**.\n\n"
    "- `churn_predictions_total{decision}` suit le **taux de prévision par classe** "
    "(ALERTE_ROUGE / SURVEILLANCE / OK) : une explosion du taux `ALERTE_ROUGE` révèle une "
    "dérive de sortie, même sans dérive détectée sur les entrées.\n"
    "- `churn_champs_imputes_total{champ}` suit la **manquance des données reçues** : "
    "chaque champ absent d'une demande (donc reconstruit par le pipeline, cf. §10.2) "
    "incrémente son compteur. Une rupture d'intégration CRM — un champ qui cesse "
    "brutalement d'être transmis — se voit ici *avant* que le score ne dérive."
))

# %% [markdown]
# **Ce qu'il faut retenir.** Les deux compteurs métier complètent le dispositif de
# monitoring : ils sont exposés sur `/metrics` (registre par défaut de `prometheus_client`,
# scrapé par Prometheus) et tracés dans le dashboard Grafana. Combinés au PSI/KS sur les
# entrées (§13.3) et à l'indicateur d'obsolescence (§13.6), ils forment un monitoring à
# **quatre niveaux**, du plus précoce au plus tardif : qualité des entrées reçues
# (immédiate — `churn_champs_imputes_total`), dérive statistique d'entrée (immédiate —
# PSI/KS), dérive de sortie (immédiate — `churn_predictions_total`), dégradation de
# performance (décalée, après obtention des étiquettes — §13.7). L'ordre compte : une
# manquance amont fausse les features avant de déplacer la distribution des scores, donc
# elle doit être surveillée en premier.

# %% [markdown]
# ### 13.9.1 — Simulation des données de monitoring (Grafana)
#
# Pour prouver le fonctionnement de la chaîne Prometheus → Grafana sans
# nécessiter une stack live, on génère des métriques synthétiques réalistes
# (intervalle 30 s, conforme au `scrape_interval` de `prometheus.yml`) couvrant
# les six panels de données du dashboard `monitoring/grafana/dashboard.json` :
# statut, débit, latence, taux d'erreur, prédictions par décision et champs manquants.

# %%
# Génération des séries temporelles synthétiques (2 h à 30 s d'intervalle)
_rng = np.random.default_rng(config.RANDOM_SEED)
_n = 240  # 240 × 30 s = 2 heures
_debut = pd.Timestamp("2026-09-27 00:00:00")
_ts = pd.date_range(start=_debut, periods=_n, freq="30s")

# Statut API : 1 partout, sauf index [100, 122] → panne ~11 min (1 alerte valide)
_up = np.ones(_n, dtype=int)
_up[100:122] = 0

# Débit : poisson(12 req/s) × statut (0 pendant la panne)
_req = _rng.poisson(12, _n).astype(float) * _up

# Latence p50 : base 75 ms + bruit ; spike gaussien à t=[80:100] → pic ~380 ms
_base_p50 = 75.0 + _rng.normal(0, 5, _n)
_spike = np.zeros(_n)
_spike[80:100] = _rng.normal(220, 20, 20).clip(min=0)
_p50 = (_base_p50 + _spike).clip(min=20)
_p95 = (_p50 * 1.5 + _rng.normal(0, 8, _n)).clip(min=30)
_p99 = (_p50 * 2.0 + _rng.normal(0, 12, _n)).clip(min=40)
# Forcer à zéro pendant la panne (API muette)
_p50[100:122] = 0
_p95[100:122] = 0
_p99[100:122] = 0

# Taux d'erreur : ~0.5 % nominal, 100 % pendant la panne
_erreurs = 0.5 + _rng.normal(0, 0.1, _n)
_erreurs[100:122] = 100.0
_erreurs = _erreurs.clip(min=0)

# Compteurs cumulatifs par classe. Point de départ = répartition du batch réel
# (reports/tables/scores_batch_*_synthese.json : ~30 % ALERTE_ROUGE, ~14 % SURVEILLANCE),
# puis dérive de ALERTE_ROUGE au-delà de la tolérance de ±10 points du runbook (§4.3).
_TAUX_ALERTE_REF, _TAUX_ALERTE_FIN = 0.30, 0.42
_TOLERANCE_RUNBOOK_PP = 10
_taux_alerte = np.linspace(_TAUX_ALERTE_REF, _TAUX_ALERTE_FIN, _n)
_taux_surv = np.full(_n, 0.14)
_taux_ok = 1.0 - _taux_alerte - _taux_surv
_pred_alerte = np.cumsum((_rng.poisson(12 * _taux_alerte)).astype(int) * _up)
_pred_surv = np.cumsum((_rng.poisson(12 * _taux_surv)).astype(int) * _up)
_pred_ok = np.cumsum((_rng.poisson(12 * _taux_ok)).astype(int) * _up)

# Manquance amont (churn_champs_imputes_total) : `csat` structurellement absent de ~8 %
# des demandes (taux nominal du CRM), tandis que `secteur` cesse d'être transmis à
# t = index 150 — rupture d'intégration simulée, 5 % → 60 % des demandes
_taux_csat_absent = (0.08 + _rng.normal(0, 0.006, _n)).clip(min=0)
_idx_rupture = 150
_taux_secteur_absent = np.concatenate(
    [np.full(_idx_rupture, 0.05), np.full(_n - _idx_rupture, 0.60)]
) + _rng.normal(0, 0.006, _n)
_taux_secteur_absent = _taux_secteur_absent.clip(min=0)
# Conversion en demandes/min concernées (le panel Grafana affiche un rate() équivalent)
_manque_csat = _req * 60 * _taux_csat_absent
_manque_secteur = _req * 60 * _taux_secteur_absent

df_metriques = pd.DataFrame({
    "ts": _ts,
    "up": _up,
    "req_par_s": _req,
    "latence_p50": _p50,
    "latence_p95": _p95,
    "latence_p99": _p99,
    "erreurs_4xx_5xx_pct": _erreurs,
    "predictions_ok": _pred_ok,
    "predictions_surveillance": _pred_surv,
    "predictions_alerte_rouge": _pred_alerte,
    "manque_csat_par_min": _manque_csat,
    "manque_secteur_par_min": _manque_secteur,
})
df_metriques.head(3)

# %%
# Figure 1 — Disponibilité API et taux d'erreur (panels 1 et 4 du dashboard Grafana)
fig, axes = viz.figure_grille(
    "monitoring_statut_api",
    "API — Disponibilité et taux d'erreur",
    nlignes=1,
    ncols=2,
    taille=(14, 4.5),
)
ax_statut, ax_err = axes

# Axe gauche : statut up/down
ax_statut.fill_between(
    df_metriques["ts"],
    df_metriques["up"],
    step="post",
    color="#3BB273",
    alpha=0.7,
    label="UP",
)
_down_mask = df_metriques["up"] == 0
if _down_mask.any():
    ax_statut.fill_between(
        df_metriques["ts"],
        _down_mask.astype(int),
        step="post",
        color="#E84855",
        alpha=0.7,
        label="INDISPONIBLE",
    )
    _idx_panne = df_metriques.index[_down_mask][0]
    ax_statut.annotate(
        "↓ Panne (~11 min)\n→ alerte APIIndisponible",
        xy=(df_metriques.loc[_idx_panne, "ts"], 0.05),
        xytext=(df_metriques.loc[_idx_panne + 25, "ts"], 0.6),
        arrowprops={"arrowstyle": "->", "color": "#E84855"},
        fontsize=9,
        color="#E84855",
    )
ax_statut.set_ylim(0, 1.2)
ax_statut.set_yticks([0, 1])
ax_statut.set_yticklabels(["INDISPONIBLE", "UP"])
ax_statut.set_xlabel("Heure")
ax_statut.set_ylabel("Statut")
ax_statut.legend(loc="lower right", fontsize=9)

# Axe droit : taux d'erreur
ax_err.plot(
    df_metriques["ts"],
    df_metriques["erreurs_4xx_5xx_pct"],
    color=viz.PALETTE_PRINCIPALE[1],
    linewidth=1.2,
    label="Taux erreur 4xx+5xx",
)
ax_err.axhline(1.0, linestyle="--", color="#264653", linewidth=1.0, label="SLO : taux erreur < 1 %")
ax_err.set_ylim(bottom=0)
ax_err.set_ylabel("Taux erreur (%)")
ax_err.set_xlabel("Heure")
ax_err.legend(fontsize=9)

fig.tight_layout()
viz.sauvegarder(fig)

# %%
# Figure 2 — Latence /predict p50/p95/p99 avec seuil SLO et zone d'alerte
fig, ax = viz.figure(
    "monitoring_latence_predict",
    "Latence /predict — p50 / p95 / p99 (SLO 200 ms)",
    taille=(12, 5),
)

for col, lbl, cidx in [
    ("latence_p50", "p50", 0),
    ("latence_p95", "p95", 3),
    ("latence_p99", "p99", 1),
]:
    ax.plot(
        df_metriques["ts"],
        df_metriques[col],
        label=lbl,
        color=viz.PALETTE_PRINCIPALE[cidx],
        linewidth=1.4,
    )

ax.axhline(200, linestyle="--", color="#E84855", linewidth=1.2, label="SLO 200 ms")

# Zone pic latence + panne
_t_debut_zone = df_metriques.loc[80, "ts"]
_t_fin_zone = df_metriques.loc[121, "ts"]
ax.axvspan(_t_debut_zone, _t_fin_zone, alpha=0.12, color="#E84855")
ax.annotate(
    "for:5m → alerte LatenceElevee",
    xy=(_t_debut_zone, 210),
    xytext=(_t_debut_zone, 320),
    arrowprops={"arrowstyle": "->", "color": "#E84855"},
    fontsize=9,
    color="#E84855",
)

ax.set_ylabel("Latence (ms)")
ax.set_xlabel("Heure")
ax.legend(loc="upper left", fontsize=9)
viz.sauvegarder(fig)

# %%
# Figure 3 — Répartition cumulée des prédictions par classe de risque
fig, ax = viz.figure(
    "monitoring_predictions_classes",
    "Répartition des prédictions par classe de risque",
    taille=(12, 5),
)

ax.stackplot(
    df_metriques["ts"],
    df_metriques["predictions_ok"],
    df_metriques["predictions_surveillance"],
    df_metriques["predictions_alerte_rouge"],
    labels=["OK", "SURVEILLANCE", "ALERTE ROUGE"],
    colors=[viz.COULEUR_NON_CHURN, viz.PALETTE_PRINCIPALE[3], viz.COULEUR_CHURN],
    alpha=0.8,
)

# Annotation de la dérive en fin de fenêtre
_idx_fin = _n - 1
_derive_pp = round((_TAUX_ALERTE_FIN - _TAUX_ALERTE_REF) * 100)
# Aires empilées : le haut de la bande ALERTE ROUGE est le total cumulé des trois classes
_sommet_pile = _pred_ok[_idx_fin] + _pred_surv[_idx_fin] + _pred_alerte[_idx_fin]
ax.annotate(
    f"Dérive sortie : ALERTE ROUGE {_TAUX_ALERTE_REF:.0%} → {_TAUX_ALERTE_FIN:.0%}\n"
    f"(+{_derive_pp} pp > tolérance ±{_TOLERANCE_RUNBOOK_PP} pp du runbook)\n"
    "→ déclencheur de rollback, complémentaire au PSI (§13.3)",
    xy=(df_metriques.loc[_idx_fin, "ts"], _sommet_pile),
    xytext=(df_metriques.loc[_n // 5, "ts"], _sommet_pile * 0.80),
    arrowprops={"arrowstyle": "->", "color": viz.COULEUR_CHURN},
    fontsize=9,
    color=viz.COULEUR_CHURN,
)

ax.set_ylabel("Prédictions cumulées")
ax.set_xlabel("Heure")
ax.legend(loc="upper left", fontsize=9)
viz.sauvegarder(fig)

# %%
# Figure 4 — Manquance des données reçues (churn_champs_imputes_total par champ)
fig, ax = viz.figure(
    "monitoring_champs_manquants",
    "Champs absents des demandes reçues — détection d'une rupture d'intégration",
    taille=(12, 5),
)

ax.plot(
    df_metriques["ts"],
    df_metriques["manque_csat_par_min"],
    label="csat absent (~8 % — nominal)",
    color=viz.PALETTE_PRINCIPALE[0],
    linewidth=1.4,
)
ax.plot(
    df_metriques["ts"],
    df_metriques["manque_secteur_par_min"],
    label="secteur absent (rupture CRM)",
    color=viz.COULEUR_CHURN,
    linewidth=1.4,
)

_t_rupture = df_metriques.loc[_idx_rupture, "ts"]
ax.axvline(_t_rupture, linestyle="--", color=viz.COULEUR_CHURN, linewidth=1.0)
ax.annotate(
    "Rupture d'intégration : `secteur` cesse d'être transmis\n"
    "(5 % → 60 % des demandes) — visible avant toute dérive du score",
    xy=(_t_rupture, float(df_metriques["manque_secteur_par_min"].max()) * 0.80),
    xytext=(df_metriques.loc[20, "ts"], float(_manque_secteur.max()) * 0.50),
    arrowprops={"arrowstyle": "->", "color": viz.COULEUR_CHURN},
    fontsize=9,
    color=viz.COULEUR_CHURN,
)

ax.set_ylabel("Demandes/min avec champ absent")
ax.set_xlabel("Heure")
ax.legend(loc="upper left", fontsize=9)
viz.sauvegarder(fig)

# %% [markdown]
# **Ce qu'il faut retenir.** La règle `for:5m` dans `alerts.yml` ajoute une
# deuxième couche de filtrage après la fenêtre `rate(...[5m])` : un pic de latence
# isolé de quelques secondes ne déclenche pas d'alerte, seul un dépassement soutenu
# du SLO 200 ms sur 5 minutes consécutives le fait. À l'inverse, `APIIndisponible`
# se déclenche en 1 minute car l'indisponibilité est immédiatement critique et doit
# mobiliser l'astreinte sans délai. La dérive de la distribution de sortie — ici le taux de
# prédictions `ALERTE_ROUGE` simulé passe de 30 % (niveau du batch réel) à 42 % en 2 heures,
# soit +12 points — dépasse la tolérance de ±10 points fixée comme déclencheur de rollback
# dans `docs/RUNBOOK.md` (§4.3). C'est un signal distinct et
# complémentaire au PSI sur les entrées (§13.3) : une dérive de sortie peut apparaître
# sans dérive d'entrée détectable (changement de comportement client sans changement
# de données). La quatrième figure illustre le signal le plus précoce de la chaîne : la
# manquance des champs reçus. Une rupture d'intégration CRM — ici `secteur` qui passe de
# 5 % à 60 % de demandes incomplètes — n'apparaît ni dans le taux d'erreur (les requêtes
# restent valides : le champ est nullable, la réponse est un 200) ni immédiatement dans la
# distribution des scores, puisque le pipeline impute. Elle se voit en revanche
# instantanément sur `churn_champs_imputes_total`, ce qui laisse le temps de corriger la
# source avant que les décisions de rétention ne reposent sur des valeurs estimées.
# Ces quatre figures sont le miroir des six panels de données de
# `monitoring/grafana/dashboard.json` (mêmes métriques, mêmes seuils SLO),
# importable directement dans Grafana via UI Import ou l'API de provisionnement.

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
# > via le groupe témoin (§4) et l'évaluation par cohortes. Monitoring porté à quatre
# > niveaux en ajoutant la **qualité des entrées reçues** (`churn_champs_imputes_total`,
# > par champ) en amont de la dérive statistique : un champ qui cesse d'être transmis par
# > le CRM ne produit ni erreur HTTP ni dérive immédiate du score, puisque le contrat
# > d'entrée l'accepte et que le pipeline impute (§10.2) — sans ce compteur, la panne
# > d'intégration resterait invisible jusqu'à la dégradation des performances.
# >
# > **Alternatives écartées** — Prefect Cloud (infrastructure supplémentaire non
# > disponible lors de la certification) ; Airflow (trop lourd pour un projet mono-équipe) ;
# > alertes sur latence p99 plutôt que latence moyenne (trop volatile sur petit volume).
# > Règle d'alerte automatique sur la manquance : écartée à ce stade faute de référence de
# > production (le seuil serait arbitraire) — le panel Grafana et la revue hebdomadaire
# > suffisent tant que le trafic réel n'a pas fourni une baseline par champ.
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
