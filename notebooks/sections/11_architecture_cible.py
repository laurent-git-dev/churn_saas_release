# %% [markdown]
# ## 11. Architecture cible (C7)
#
# Cette section répond aux trois items de la compétence C7 : (1) comparaison des principales
# architectures de déploiement avec leurs contraintes, (2) chiffrage économique des scénarios
# porté à la connaissance des acteurs, (3) compte-rendu d'entretien avec DSI, RSSI, DPO et
# responsable Customer Success, simulé et assumé comme tel.

# %% [markdown]
# ### 11.1 Pipeline de données — vue d'ensemble
#
# Le diagramme ci-dessous représente le flux complet depuis les sources jusqu'à l'exposition
# des scores. Il est rendu nativement par JupyterLab et VS Code grâce à l'extension Mermaid.

# %% [markdown]
# ```mermaid
# flowchart LR
#     subgraph Sources["Sources de données"]
#         CRM[(CRM\nSalesforce / HubSpot)]
#         SUPPORT[(Outil support\nZendesk / Freshdesk)]
#         FACTURATION[(ERP facturation\nNetSuite / Stripe)]
#         PRODUIT[(Events produit\nMixpanel / Amplitude)]
#     end
#
#     subgraph Ingestion["Ingestion — quotidienne 23 h 00"]
#         ETL[Script ETL\nPython / dbt]
#         PARQUET[(Parquet\nStockage objet)]
#     end
#
#     subgraph Pipeline["Pipeline ML — 00 h 00"]
#         PREP[Préparation\n& feature engineering]
#         MODELE[Modèle champion\nMLflow Registry]
#         SCORES[(Scores + déciles\nbase cible)]
#     end
#
#     subgraph Exposition["Exposition"]
#         API[API FastAPI\nPOST /predict]
#         BATCH[Job batch\nPrefect]
#         CRM_OUT[(CRM — champ\nrisque_churn)]
#         DASHBOARD[Dashboard CS]
#     end
#
#     CRM --> ETL
#     SUPPORT --> ETL
#     FACTURATION --> ETL
#     PRODUIT --> ETL
#     ETL --> PARQUET
#     PARQUET --> PREP
#     PREP --> MODELE
#     MODELE --> SCORES
#     SCORES --> BATCH
#     BATCH --> CRM_OUT
#     SCORES --> DASHBOARD
#     API --> MODELE
# ```

# %% [markdown]
# **Ce qu'il faut retenir.** Les quatre sources (CRM, support, ERP, events produit) sont
# ingérées chaque nuit vers 23 h. Le pipeline ML s'enchaîne à 00 h pour que les scores soient
# disponibles dans le CRM dès 05 h, avant que les CSM prennent leur poste. L'API à la demande
# partage le même modèle champion, ce qui garantit la cohérence entre les vues batch et temps-réel.

# %% [markdown]
# ### 11.2 Architecture cible — Scénario B retenu

# %% [markdown]
# ```mermaid
# flowchart TB
#     subgraph Client["Réseau client / CSM"]
#         NAVIGATEUR[Navigateur CSM]
#         CRM_CLIENT[(CRM client)]
#     end
#
#     subgraph DMZ["DMZ / API Gateway"]
#         GW[API Gateway\nHTTPS + JWT]
#         WAF[WAF / rate-limiting]
#     end
#
#     subgraph Compute["Cluster conteneurs — ECS Fargate / Cloud Run"]
#         API_CONT[Conteneur API\nFastAPI · 2 vCPU · 2 Gi]
#         BATCH_CONT[Conteneur batch\nPrefect worker · 2 vCPU · 4 Gi]
#         MON_CONT[Conteneur monitoring\nEvidently + Prometheus]
#     end
#
#     subgraph DataLayer["Données managées"]
#         S3[(Stockage objet\nS3 / GCS — Parquet)]
#         RDS[(Base relationnelle\nRDS PostgreSQL / Cloud SQL)]
#     end
#
#     subgraph MLPlatform["Registre de modèles"]
#         MLFLOW[MLflow Tracking\n+ Registry — hébergé]
#         ARTIFACTS_S[(Artefacts modèles\nS3 / GCS)]
#     end
#
#     subgraph CI["CI/CD — GitHub Actions"]
#         PIPELINE_CI[Lint · Types · Tests · Gate qualité]
#         BUILD[Build image Docker\nPublication ECR / Artifact Registry]
#     end
#
#     NAVIGATEUR -->|HTTPS| GW
#     CRM_CLIENT -->|Export CSV quotidien| S3
#     GW --> WAF
#     WAF --> API_CONT
#     API_CONT --> MLFLOW
#     MLFLOW --> ARTIFACTS_S
#     BATCH_CONT --> S3
#     BATCH_CONT --> MLFLOW
#     BATCH_CONT --> RDS
#     MON_CONT --> RDS
#     PIPELINE_CI --> BUILD
#     BUILD --> API_CONT
#     BUILD --> BATCH_CONT
# ```

# %% [markdown]
# ### 11.3 Diagramme de séquence — Score à la demande (CSM → API → CRM)
#
# Cas d'usage 3 de §2 : un CSM ouvre une fiche compte à l'occasion d'un renouvellement
# ou d'une escalade support. Le CRM interroge l'API de façon synchrone et affiche le score
# pendant le chargement de la fiche (p95 < 200 ms).

# %% [markdown]
# ```mermaid
# sequenceDiagram
#     actor CSM as CSM
#     participant CRM as CRM (Salesforce)
#     participant GW as API Gateway
#     participant API as API FastAPI
#     participant REG as MLflow Registry
#     participant DB as Base scores
#
#     CSM->>CRM: Ouvre fiche compte (client_id: C-1042)
#     CRM->>GW: POST /api/v1/predict\n{client_id, features…}\nAuthorization: Bearer JWT
#     GW->>GW: Validation JWT + rate-limit
#     GW->>API: Requête transmise
#
#     API->>REG: Charger modèle champion (cache local 1 h)
#     REG-->>API: Pipeline régression logistique (champion §9) — stade Production
#
#     API->>API: Inférence → score_churn = 0.78 décile = 9
#     API->>API: SHAP local → top 3 facteurs
#
#     API->>DB: Persist résultat (audit trail)
#     DB-->>API: OK
#
#     API-->>GW: 200 OK {score, décile, facteurs, recommandation}
#     GW-->>CRM: Réponse JSON (p95 < 200 ms)
#     CRM->>CRM: Maj champ risque_churn + bandeau alerte rouge
#     CRM-->>CSM: Fiche enrichie — score + top 3 signaux
# ```

# %% [markdown]
# **Ce qu'il faut retenir.** Le champion retenu en §9 étant une régression logistique,
# l'explication SHAP locale y est peu coûteuse : pour un modèle linéaire, la contribution de chaque
# variable se déduit directement de son coefficient (`LinearExplainer`), sans exploration
# d'arbres. Le budget de latence (p95 < 200 ms) est donc surtout consommé par le réseau, la
# passerelle et la persistance du résultat ; la latence d'inférence réelle est mesurée en §9 et
# §12. Le cache modèle (TTL = 1 h) évite les appels répétés au registre sans risquer d'exposer
# un modèle obsolète. Écart assumé entre cible et existant : l'API livrée en §10 renvoie les
# trois facteurs de risque à partir des règles issues de l'EDA (§6) ; le SHAP local en ligne
# est une évolution de cette architecture cible.

# %% [markdown]
# ### 11.4 Compte-rendu d'entretien avec les acteurs
#
# *Note méthodologique : cet entretien est simulé à partir des contraintes typiques d'un éditeur
# SaaS B2B français mid-market. Il est présenté comme un exercice de cadrage ; les citations sont
# reconstituées à partir des préoccupations standard de chaque fonction. Cela est assumé et
# transparent.*

# %%
import pandas as pd
from IPython.display import display

from churn_saas import config

entretiens = [
    {
        "Acteur": "DSI",
        "Contraintes remontées": (
            "Budget infrastructure plafonné à 600 €/mois pour la phase pilote. "
            "L'équipe ops ne maîtrise pas Kubernetes ; préférence pour des services entièrement managés. "
            "La solution doit s'intégrer via API REST standard — pas de SDK propriétaire. "
            "SLA de disponibilité exigé : 99,5 % minimum sur la plage 06 h – 20 h."
        ),
        "Impact sur l'architecture": (
            "Scénario A (VM seule) écarté : trop de maintenance manuelle. "
            "Scénario B (conteneurs managés sans orchestration custom) retenu. "
            "Scénario C (SageMaker/Vertex) écarté : budget dépassé et lock-in fournisseur."
        ),
    },
    {
        "Acteur": "RSSI",
        "Contraintes remontées": (
            "Données clients classifiées 'sensibles B2B' — hébergement obligatoirement en UE "
            "(RGPD + politique interne). Fournisseur cloud certifié ISO 27001 requis. "
            "Chiffrement des données en transit (TLS 1.2+) et au repos (AES-256). "
            "Journalisation de tous les appels API avec rétention 12 mois. "
            "Aucun modèle tiers non audité ne doit traiter les données clients (règle de souveraineté)."
        ),
        "Impact sur l'architecture": (
            "Région cloud restreinte : eu-west-1 (AWS Paris) ou europe-west1 (GCP). "
            "MLflow hébergé en self-managed ou via un fournisseur européen certifié. "
            "API Gateway avec WAF obligatoire. Audit trail complet dans la base PostgreSQL. "
            "Élimine SageMaker US-East-1 par défaut — nécessite configuration région explicite."
        ),
    },
    {
        "Acteur": "DPO",
        "Contraintes remontées": (
            "Le score de churn est une donnée de profilage B2B : analyse d'impact relative "
            "à la vie privée (AIPD) recommandée si des données nominatives transitent. "
            "Durée de conservation des scores limitée à 90 jours (politique interne). "
            "Droit à l'explication applicable aux décisions individuelles ayant un effet "
            "sur la relation commerciale (art. 22 RGPD, interprétation prudente). "
            "Registre des traitements à mettre à jour avant mise en production."
        ),
        "Impact sur l'architecture": (
            "Purge automatique des scores > 90 jours via job cron Prefect. "
            "Endpoint /explain intégré à l'API (SHAP local) pour répondre au droit à l'explication. "
            "Pas de stockage de noms ou emails dans la base scores — uniquement client_id. "
            "Registre des traitements mis à jour dans docs/DATASHEET.md."
        ),
    },
    {
        "Acteur": "Responsable Customer Success",
        "Contraintes remontées": (
            "Les CSM veulent un score visible directement dans Salesforce, pas un outil tiers. "
            "Les 3 signaux explicatifs (top features SHAP) sont indispensables — un chiffre brut "
            "ne suffit pas pour préparer un appel. Le score doit être actualisé chaque matin "
            "avant 07 h. Le modèle doit gérer les nouveaux comptes (< 3 mois d'ancienneté) "
            "avec un indicateur explicite d'incertitude."
        ),
        "Impact sur l'architecture": (
            "Intégration Salesforce via champ custom + Flow Builder appelant POST /predict "
            "au chargement de la fiche client (cas d'usage 3 de §2). "
            "Réponse API obligatoirement enrichie des top 3 features SHAP et d'une recommandation texte. "
            "Job batch Prefect planifié à 00 h 00 pour disponibilité à 05 h. "
            "Champ incertitude_score ajouté (intervalle de confiance à 90 %) pour les comptes jeunes."
        ),
    },
]

df_entretiens = pd.DataFrame(entretiens)
pd.set_option("display.max_colwidth", 120)
display(df_entretiens[["Acteur", "Contraintes remontées"]])

# %% [markdown]
# #### Contraintes de généralisation identifiées lors des entretiens
#
# La **généralisation** désigne ici la capacité du système à fonctionner au-delà du périmètre
# pilote initial (5 000 comptes, un seul produit SaaS, équipe ML de 2 personnes). Trois axes
# ont été explorés avec les acteurs :
#
# | Axe de généralisation | Contrainte remontée | Acteur | Impact architectural |
# |---|---|---|---|
# | **Volume** — passage de 5 k à 50 k comptes | Infrastructure doit scaler sans refonte (ECS auto-scaling) | DSI | Scénario B prévu jusqu'à ~50 k comptes ; au-delà → réévaluation scénario C (Phase 3, §11.7) |
# | **Portabilité** — changement de cloud provider | Pas de lock-in SDK propriétaire ; images Docker standard | RSSI | Conteneurs sans dépendance SageMaker/Vertex ; Parquet + PostgreSQL standards |
# | **Réplicabilité** — extension à d'autres produits / marchés | Séparation config/code ; un seul `config.py` par produit suffit | Responsable CS | Architecture paramétrée via `churn_saas/config.py` ; pipeline réutilisable par simple fork |
# | **Robustesse temporelle** — évolution du comportement des données | Drift détecté avant dégradation modèle | DPO + Équipe ML | PSI > 0,2 → réentraînement automatique (§13) ; SLO PR-AUC ≥ `pr_auc_min` (§11.8) |
# | **Contrainte de compétences** — équipe sans spécialiste MLOps | Services entièrement managés, runbook simple | DSI + Ops | Scénario B exclu Kubernetes ; runbook documenté en §11.9 |

# %% [markdown]
# **Ce qu'il faut retenir.** Les quatre acteurs interrogés font converger les contraintes vers le
# même scénario : conteneurs managés, région UE, services entièrement managés (pas de Kubernetes
# manuel), intégration Salesforce native. Le DPO impose une durée de conservation courte (90 j)
# et un endpoint d'explication, ce qui renforce le choix SHAP déjà retenu en §9.

# %% [markdown]
# ### 11.5 Contraintes techniques, réglementaires et organisationnelles

# %%
contraintes = [
    {
        "Dimension": "Technique",
        "Contrainte": "Latence API p95 < 200 ms",
        "Conséquence architecturale": "Pipeline scikit-learn (régression logistique) chargé en mémoire, cache modèle TTL 1 h, pas de cold-start",
    },
    {
        "Dimension": "Technique",
        "Contrainte": "Fraîcheur des scores < 24 h",
        "Conséquence architecturale": "Job batch quotidien 00 h 00, alerte si exécution > 05 h 00",
    },
    {
        "Dimension": "Technique",
        "Contrainte": "Reproductibilité totale",
        "Conséquence architecturale": "Graine unique (config.RANDOM_SEED), versioning données (DVC) + modèle (MLflow)",
    },
    {
        "Dimension": "Réglementaire",
        "Contrainte": "Hébergement UE (RGPD)",
        "Conséquence architecturale": "Région AWS eu-west-1 (Paris) ou GCP europe-west1 — configuré explicitement",
    },
    {
        "Dimension": "Réglementaire",
        "Contrainte": "Droit à l'explication (art. 22 RGPD)",
        "Conséquence architecturale": "Endpoint GET /explain/{client_id} avec SHAP local obligatoire",
    },
    {
        "Dimension": "Réglementaire",
        "Contrainte": "Rétention des scores 90 jours max",
        "Conséquence architecturale": "Job de purge Prefect hebdomadaire, index sur date_prediction",
    },
    {
        "Dimension": "Réglementaire",
        "Contrainte": "Journalisation des appels 12 mois (RSSI)",
        "Conséquence architecturale": "Audit trail en base + export vers SIEM (bucket S3 chiffré)",
    },
    {
        "Dimension": "Organisationnelle",
        "Contrainte": "Équipe ops sans compétence Kubernetes",
        "Conséquence architecturale": "ECS Fargate ou Cloud Run — orchestration managée, pas d'EKS/GKE",
    },
    {
        "Dimension": "Organisationnelle",
        "Contrainte": "Budget pilote ≤ 600 €/mois",
        "Conséquence architecturale": "Scénario B retenu ; scénario C (≥ 800 €) réservé à la montée en charge",
    },
    {
        "Dimension": "Organisationnelle",
        "Contrainte": "Intégration Salesforce native exigée par les CSM",
        "Conséquence architecturale": "API REST + champ custom Salesforce via Flow Builder (pas d'AppExchange)",
    },
]

df_contraintes = pd.DataFrame(contraintes)
display(df_contraintes)

# %% [markdown]
# ### 11.6 Trois scénarios comparatifs

# %%
scenarios = [
    {
        "Scénario": "A — VM unique + batch cron",
        "Description": "1 VM Linux (t3.medium ou B2ms Azure), scoring via cron, API Flask simple, modèle chargé au démarrage",
        "Coût mensuel estimé (€)": "60 – 120",
        "Complexité mise en œuvre": "Faible",
        "Délai de mise en prod": "2–3 semaines",
        "Compétences requises": "Admin Linux, Python, cron",
        "Souveraineté / hébergement UE": "Oui (région configurable) — contrôle total",
        "Réversibilité": "Maximale — un tar.gz suffit à migrer",
        "Limites critiques": (
            "Scalabilité zéro (une VM = un SPOF). "
            "Pas de HA ni de rollback modèle. "
            "Monitoring manuel. "
            "Déploiements risqués (coupure de service). "
            "Inadapté à > 20 k comptes."
        ),
    },
    {
        "Scénario": "B — Conteneurs managés + base managée + registre modèles ★ RECOMMANDÉ",
        "Description": (
            "ECS Fargate (AWS) ou Cloud Run (GCP). "
            "RDS PostgreSQL managé. MLflow hébergé sur VM légère ou Render. "
            "CI/CD GitHub Actions. Stockage Parquet sur S3/GCS."
        ),
        "Coût mensuel estimé (€)": "280 – 520",
        "Complexité mise en œuvre": "Moyenne",
        "Délai de mise en prod": "4–6 semaines",
        "Compétences requises": "Docker, AWS/GCP basique, Python, CI/CD",
        "Souveraineté / hébergement UE": "Oui — région eu-west-1 / europe-west1 obligatoire",
        "Réversibilité": "Haute — images Docker portables, données Parquet standard",
        "Limites critiques": (
            "MLflow auto-hébergé à maintenir (2–4 h/mois). "
            "Coût RDS non négligeable à faible volumétrie. "
            "Monitoring applicatif à intégrer (Prometheus + Grafana ou CloudWatch)."
        ),
    },
    {
        "Scénario": "C — Plateforme cloud managée (SageMaker / Vertex AI / Azure ML)",
        "Description": (
            "Endpoint d'inférence managé, pipeline builder, feature store, "
            "model registry intégré, A/B testing natif, MLOps clé en main."
        ),
        "Coût mensuel estimé (€)": "800 – 2 500",
        "Complexité mise en œuvre": "Faible à moyenne (vendor abstraction)",
        "Délai de mise en prod": "3–5 semaines (si maîtrise du provider)",
        "Compétences requises": "AWS ML Specialty ou équivalent, SDK SageMaker / Vertex",
        "Souveraineté / hébergement UE": (
            "Partielle — SageMaker EU OK (eu-west-1), mais certains services accessoires "
            "restent hors UE par défaut. Azure ML region France Central : acceptable."
        ),
        "Réversibilité": (
            "Faible à moyenne — fort lock-in SDK (SageMaker Pipelines, Vertex Pipelines). "
            "Migration coûteuse si changement de provider."
        ),
        "Limites critiques": (
            "Budget 3–5× supérieur au scénario B pour la même volumétrie. "
            "Courbe d'apprentissage du SDK propriétaire. "
            "Risque de lock-in fournisseur signalé par le RSSI."
        ),
    },
]

df_scenarios = pd.DataFrame(scenarios).set_index("Scénario")
pd.set_option("display.max_colwidth", 100)
display(df_scenarios)

# %% [markdown]
# **Ce qu'il faut retenir.** Les trois scénarios forment une progression coût / complexité / capacité.
# Le scénario A convient à un POC interne (<1 k comptes, équipe tech réduite) mais n'est pas
# maintenable en production. Le scénario C offre le meilleur MLOps intégré mais dépasse le budget
# pilote et crée un lock-in incompatible avec la politique de réversibilité du RSSI.
# Le **scénario B** offre le meilleur équilibre pour l'éditeur SaaS considéré.

# %% [markdown]
# ### 11.7 Recommandation argumentée

# %% [markdown]
# **Scénario retenu : B — Conteneurs managés (ECS Fargate / Cloud Run) + RDS PostgreSQL + MLflow.**
#
# #### Motifs de sélection
#
# | Critère | Scénario A | **Scénario B** | Scénario C |
# |---|---|---|---|
# | Budget pilote ≤ 600 €/mois | ✅ 60–120 € | ✅ **280–520 €** | ❌ 800–2 500 € |
# | Compétences équipe ops actuelle | ✅ | ✅ | ❌ SDK propriétaire |
# | Hébergement UE garanti | ✅ | ✅ | ⚠️ À configurer |
# | Haute disponibilité + rollback modèle | ❌ | ✅ | ✅ |
# | Réversibilité (pas de lock-in) | ✅ | ✅ | ❌ |
# | CI/CD + gate qualité automatisée | ❌ | ✅ | ✅ |
# | Scalabilité jusqu'à ~50 k comptes | ❌ | ✅ | ✅ |
#
# #### Chemin de migration prévu
#
# - **Phase 0 (maintenant)** : VM légère pour développement et validation (scénario A allégé).
# - **Phase 1 (T+2 mois)** : migration vers ECS Fargate + RDS — batch nocturne opérationnel.
# - **Phase 2 (T+6 mois)** : ajout du monitoring Evidently, alertes automatiques, A/B test champion/challenger.
# - **Phase 3 (T+12 mois)** : réévaluation scénario C si la volumétrie dépasse 20 k comptes
#   ou si le DSI recrute un profil MLOps dédié.
#
# #### Ordre de grandeur des coûts (scénario B, AWS eu-west-1, pilote 5 000 comptes)
#
# | Composant | Détail | Coût mensuel estimé |
# |---|---|---|
# | ECS Fargate — API (2 tâches × 0,5 vCPU × 1 Gi) | Disponibilité 24 h/24 | ≈ 30 € |
# | ECS Fargate — Batch (1 tâche × 2 vCPU × 4 Gi, 5 h/nuit) | 150 h/mois | ≈ 25 € |
# | RDS PostgreSQL db.t3.micro (20 Go SSD) | Multi-AZ désactivé pilote | ≈ 25 € |
# | S3 — données + artefacts (50 Go + transferts) | Standard IA | ≈ 10 € |
# | MLflow sur EC2 t3.nano (auto-hébergé) | Ou Render free tier | ≈ 5–15 € |
# | API Gateway + WAF + CloudWatch | Logs + monitoring de base | ≈ 20–30 € |
# | GitHub Actions CI/CD | 2 000 min/mois inclus | 0 € (plan gratuit) |
# | **Total estimé** | | **≈ 115 – 135 € / mois** |
#
# *Note : les estimations ci-dessus concernent la phase pilote (5 k comptes, trafic faible).
# À 20 k comptes avec multi-AZ et monitoring renforcé, prévoir 300–450 €/mois.
# Ces chiffres sont des estimations basées sur les tarifs publics AWS en vigueur en 2025 ;
# des remises réservées (1 an) peuvent réduire la facture Fargate de 30–40 %.*

# %% [markdown]
# > 📧 **Note d'arbitrage économique — transmise au commanditaire (DSI, direction générale)**
# >
# > *Date simulée : J-30 avant la mise en production pilote.*
# >
# > Le tableau ci-dessus (scénario B, pilote 5 000 comptes) a été présenté aux acteurs concernés
# > lors du comité de cadrage architecture. Points retenus dans le compte-rendu :
# >
# > - **DSI** : budget pilote ≤ 600 €/mois confirmé → scénario B validé (115–135 €/mois) ;
# >   scénario C (800–2 500 €) explicitement écarté pour la phase pilote.
# > - **Direction générale** : l'estimation de ROI (§12) rend le scénario B autofinancé dès
# >   le 3e mois si le taux de rétention progresse de 5 points.
# > - **RSSI** : surcoût de 15–20 €/mois accepté pour activer le chiffrement au repos (AES-256)
# >   et la journalisation 12 mois — non inclus dans l'estimation initiale.
# >
# > *Cette note est simulée à des fins pédagogiques. Elle représente la démarche attendue en
# > contexte réel : chiffrer, arbitrer, tracer la décision avec les parties prenantes.*

# %% [markdown]
# ### 11.8 SLO / SLI et procédures de remédiation
#
# Un SLO (Service Level Objective) définit l'engagement de qualité de service. Le SLI
# (Service Level Indicator) est la mesure qui permet de vérifier que le SLO est tenu.

# %%
# Mêmes seuils de qualité que la table d'actions de §12 : alerte sous la cible a priori,
# critique 5 points en dessous.
_pr_auc_min = config.CIBLES_PERFORMANCE["pr_auc_min"]
_pr_auc_critique = _pr_auc_min - 0.05

slo_sli = [
    {
        "Indicateur (SLI)": "Fraîcheur des scores",
        "SLO cible": "100 % des comptes scorés dans les 24 h précédentes",
        "Mesure": "MAX(NOW() - date_prediction) sur la table scores",
        "Seuil d'alerte": "> 20 h → alerte PagerDuty",
        "Seuil critique": "> 24 h → incident P1, gel des décisions automatiques",
        "Remédiation": (
            "Vérifier les logs Prefect. Si job échoué : relancer manuellement. "
            "Si données sources indisponibles : décision suspendue, CSM informé par email."
        ),
    },
    {
        "Indicateur (SLI)": "Latence API p95",
        "SLO cible": "p95 < 200 ms sur fenêtre glissante 1 h",
        "Mesure": "Percentile 95 du temps de réponse mesuré par CloudWatch / Prometheus",
        "Seuil d'alerte": "> 150 ms → alerte dashboard",
        "Seuil critique": "> 200 ms → alerte ops + investigation",
        "Remédiation": (
            "Vérifier charge CPU du conteneur API. Si cache modèle expiré : chauffer le cache. "
            "Si pic de trafic : scale-out automatique (ECS desired count +1)."
        ),
    },
    {
        "Indicateur (SLI)": "Disponibilité API",
        "SLO cible": "99,5 % sur plage 06 h – 20 h (jours ouvrés)",
        "Mesure": "Taux de succès HTTP 2xx / total requêtes",
        "Seuil d'alerte": "< 99,8 % sur 1 h",
        "Seuil critique": "< 99,0 % sur 30 min → incident P1",
        "Remédiation": (
            "ECS Fargate relance automatiquement les tâches saines. "
            "Si DB inaccessible : basculer sur mode dégradé (scores batch J-1 servis depuis cache S3)."
        ),
    },
    {
        "Indicateur (SLI)": "Drift des données (PSI)",
        "SLO cible": "PSI < 0,2 sur toutes les features en production",
        "Mesure": "Contrôle Evidently quotidien — Population Stability Index",
        "Seuil d'alerte": "PSI > 0,1 → signalement au comité trimestriel",
        "Seuil critique": "PSI > 0,2 → déclenchement du playbook de réentraînement (§13)",
        "Remédiation": (
            "Analyser la feature driftée. Si changement métier légitime : mettre à jour "
            "le pipeline et réentraîner. Si erreur de collecte : corriger la source."
        ),
    },
    {
        "Indicateur (SLI)": "Qualité du modèle (PR-AUC)",
        "SLO cible": f"PR-AUC ≥ {_pr_auc_min:.2f} sur les comptes arrivés à échéance",
        "Mesure": (
            "À chaque vague de renouvellements — calcul sur les labels devenus observables (§13)"
        ),
        "Seuil d'alerte": f"PR-AUC < {_pr_auc_min:.2f} → alerte responsable ML",
        "Seuil critique": (
            f"PR-AUC < {_pr_auc_critique:.2f} → retour au champion précédent (rollback §13)"
        ),
        "Remédiation": (
            "Analyse SHAP pour identifier les features dégradées. "
            "Déclenchement réentraînement si PSI > 0,2 confirmé. "
            "Challenger entraîné en parallèle, promu seulement s'il passe la gate de §13 "
            "(PR-AUC challenger ≥ PR-AUC champion − 1 pt)."
        ),
    },
]

df_slo = pd.DataFrame(slo_sli).set_index("Indicateur (SLI)")
display(df_slo)

# %% [markdown]
# **Ce qu'il faut retenir.** Cinq indicateurs couvrent les deux dimensions de la fiabilité :
# infrastructure (latence, disponibilité) et ML (fraîcheur, drift, qualité). Chaque SLO est
# associé à une procédure de remédiation graduée — alerte puis incident — pour éviter les décisions
# silencieuses en mode dégradé. Le seuil critique de PR-AUC déclenche un retour au champion
# précédent, ce qui garantit la continuité de service même lors d'une régression de modèle.

# %% [markdown]
# ### 11.9 Acteurs impliqués dans le fonctionnement continu (run)

# %%
acteurs_run = [
    {
        "Acteur": "Customer Success Manager (CSM)",
        "Rôle": "Utilisateur final — consulte les scores dans Salesforce, déclenche les actions de rétention",
        "Interaction avec le système": "Lecture passive (dashboard) + appels API via Salesforce Flow",
        "Formation requise": "1 h — interprétation du score et des top 3 signaux SHAP",
    },
    {
        "Acteur": "Responsable Customer Success",
        "Rôle": "Pilote la stratégie de rétention, valide les seuils de priorité, participe au comité de revue trimestriel",
        "Interaction avec le système": "Dashboard agrégé, liste priorisée du lundi, revue des KPI métier",
        "Formation requise": "2 h — lecture des métriques métier et arbitrage des seuils",
    },
    {
        "Acteur": "Équipe Data / ML",
        "Rôle": "Maintient le pipeline, surveille les SLI, déploie les nouvelles versions via CI/CD",
        "Interaction avec le système": "MLflow, Prefect, GitHub Actions, Evidently, Prometheus",
        "Formation requise": "Compétences internes — aucune formation supplémentaire",
    },
    {
        "Acteur": "Ops / SRE (mutualisé)",
        "Rôle": "Gère l'infrastructure cloud, applique les patchs de sécurité, répond aux incidents P1",
        "Interaction avec le système": "Console AWS/GCP, CloudWatch/GCP Monitoring, runbook Confluence",
        "Formation requise": "2 h — runbook d'incident spécifique à la solution churn",
    },
    {
        "Acteur": "DPO",
        "Rôle": "Valide les traitements, répond aux demandes de droit à l'explication, audite la purge",
        "Interaction avec le système": "Endpoint /explain, logs d'audit, rapport de purge mensuel",
        "Formation requise": "1 h — lecture des rapports de conformité automatisés",
    },
    {
        "Acteur": "RSSI",
        "Rôle": "Audite la conformité sécurité, valide les mises à jour d'infrastructure, revue annuelle",
        "Interaction avec le système": "SIEM (logs API exportés S3), rapports de vulnérabilités (Trivy CI)",
        "Formation requise": "Aucune — reporting automatisé vers le SIEM existant",
    },
]

df_acteurs = pd.DataFrame(acteurs_run).set_index("Acteur")
display(df_acteurs)

# %% [markdown]
# **Ce qu'il faut retenir.** Six acteurs sont impliqués en régime permanent, avec des périmètres
# clairement séparés. L'équipe Data/ML est le seul acteur qui touche au pipeline de bout en bout ;
# tous les autres ont des interfaces de lecture (dashboard, endpoint /explain, SIEM) qui ne
# requièrent pas de compétence ML. Cette séparation réduit le risque d'erreur humaine en production
# et simplifie les habilitations (principe du moindre privilège).

# %% [markdown]
# > ### 📋 Journal de bord — Architecture cible (§11)
# >
# > **Décisions retenues** — Scénario B (conteneurs managés ECS Fargate + RDS PostgreSQL + MLflow
# > auto-hébergé) retenu comme architecture cible pour la phase pilote. Cinq SLO définis avec
# > procédures de remédiation graduées. Compte-rendu d'entretien simulé avec quatre acteurs
# > (DSI, RSSI, DPO, responsable CS) documenté et assumé comme tel.
# >
# > **Alternatives écartées** — Scénario A (VM unique) : SPOF inacceptable et absence de rollback
# > modèle. Scénario C (SageMaker/Vertex/Azure ML) : budget 3–5× supérieur et lock-in vendor
# > incompatible avec la politique de réversibilité du RSSI.
# >
# > **Difficultés rencontrées** — Le chiffrage du scénario B est une estimation basée sur les
# > tarifs publics AWS 2025 ; une validation par un devis réel (console AWS Pricing Calculator)
# > est recommandée avant engagement.
# >
# > **Impact sur la suite** — Les SLO définis ici (fraîcheur < 24 h, p95 < 200 ms, PR-AUC ≥ `pr_auc_min`)
# > alimentent directement les seuils d'alerte de §12 (performance) et le playbook de
# > réentraînement de §13 (amélioration continue).
# >
# > **Temps passé** — ½ journée de travail (rédaction + diagrammes + tableau comparatif).
