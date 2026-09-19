# Architecture cible — churn SaaS B2B

Document de référence technique. Les diagrammes sont identiques à ceux du notebook §11
(source unique : `notebooks/sections/11_architecture_cible.py`).

---

## 1. Pipeline de données

```mermaid
flowchart LR
    subgraph Sources["Sources de données"]
        CRM[(CRM\nSalesforce / HubSpot)]
        SUPPORT[(Outil support\nZendesk / Freshdesk)]
        FACTURATION[(ERP facturation\nNetSuite / Stripe)]
        PRODUIT[(Events produit\nMixpanel / Amplitude)]
    end

    subgraph Ingestion["Ingestion — quotidienne 23 h 00"]
        ETL[Script ETL\nPython / dbt]
        PARQUET[(Parquet\nStockage objet)]
    end

    subgraph Pipeline["Pipeline ML — 00 h 00"]
        PREP[Préparation\n& feature engineering]
        MODELE[Modèle champion\nMLflow Registry]
        SCORES[(Scores + déciles\nbase cible)]
    end

    subgraph Exposition["Exposition"]
        API[API FastAPI\nPOST /predict]
        BATCH[Job batch\nPrefect]
        CRM_OUT[(CRM — champ\nrisque_churn)]
        DASHBOARD[Dashboard CS]
    end

    CRM --> ETL
    SUPPORT --> ETL
    FACTURATION --> ETL
    PRODUIT --> ETL
    ETL --> PARQUET
    PARQUET --> PREP
    PREP --> MODELE
    MODELE --> SCORES
    SCORES --> BATCH
    BATCH --> CRM_OUT
    SCORES --> DASHBOARD
    API --> MODELE
```

---

## 2. Architecture cible retenue — Scénario B (conteneurs managés)

```mermaid
flowchart TB
    subgraph Client["Réseau client / CSM"]
        NAVIGATEUR[Navigateur CSM]
        CRM_CLIENT[(CRM client)]
    end

    subgraph DMZ["DMZ / API Gateway"]
        GW[API Gateway\nHTTPS + JWT]
        WAF[WAF / rate-limiting]
    end

    subgraph Compute["Cluster conteneurs — ECS Fargate / Cloud Run"]
        API_CONT[Conteneur API\nFastAPI · 2 vCPU · 2 Gi]
        BATCH_CONT[Conteneur batch\nPrefect worker · 2 vCPU · 4 Gi]
        MONITORING_CONT[Conteneur monitoring\nEvidently + Prometheus]
    end

    subgraph Data["Données managées"]
        S3[(Stockage objet\nS3 / GCS — Parquet)]
        RDS[(Base relationnelle\nRDS PostgreSQL / Cloud SQL)]
    end

    subgraph MLPlatform["Registre de modèles"]
        MLFLOW[MLflow Tracking\n+ Registry — hébergé]
        ARTIFACTS_S[(Artefacts modèles\nS3 / GCS)]
    end

    subgraph CI["CI/CD — GitHub Actions"]
        PIPELINE_CI[Lint · Types · Tests · Gate qualité]
        BUILD[Build image Docker\nPublication ECR / Artifact Registry]
    end

    NAVIGATEUR -->|HTTPS| GW
    CRM_CLIENT -->|Webhook ou export CSV| S3
    GW --> WAF
    WAF --> API_CONT
    API_CONT --> MLFLOW
    MLFLOW --> ARTIFACTS_S
    BATCH_CONT --> S3
    BATCH_CONT --> MLFLOW
    BATCH_CONT --> RDS
    MONITORING_CONT --> RDS
    PIPELINE_CI --> BUILD
    BUILD --> API_CONT
    BUILD --> BATCH_CONT
```

---

## 3. Diagramme de séquence — Score à la demande (CSM → API → CRM)

```mermaid
sequenceDiagram
    actor CSM as CSM
    participant CRM as CRM (Salesforce)
    participant GW as API Gateway
    participant API as API FastAPI
    participant REG as MLflow Registry
    participant DB as Base scores

    CSM->>CRM: Ouvre la fiche compte (client_id: C-1042)
    CRM->>GW: POST /api/v1/predict\n{client_id, features…}\nAuthorization: Bearer <JWT>
    GW->>GW: Validation JWT + rate-limit
    GW->>API: Requête transmise

    API->>REG: Charger modèle champion (cache local 1 h)
    REG-->>API: Modèle XGBoost v2.3 (Production)

    API->>API: Inférence\n→ score_churn = 0.78\n→ décile = 9

    API->>API: Explication SHAP locale\n→ top 3 facteurs

    API->>DB: Persist résultat (audit trail)
    DB-->>API: OK

    API-->>GW: 200 OK\n{score: 0.78, décile: 9,\nfacteurs: [derniere_connexion, tickets, taux_adoption],\nrecommandation: "Appel de rétention prioritaire"}
    GW-->>CRM: Réponse JSON
    CRM->>CRM: Mise à jour champ risque_churn\n+ affichage bandeau alerte rouge
    CRM-->>CSM: Fiche enrichie — score risque visible\navec top 3 signaux explicatifs
```

---

## 4. Notes

- Tous les secrets (clés API, credentials DB) passent par un gestionnaire de secrets
  (AWS Secrets Manager, GCP Secret Manager ou HashiCorp Vault) — jamais en variable
  d'environnement en clair dans les images Docker.
- La souveraineté des données (hébergement UE) est contrainte par le RSSI :
  seuls des fournisseurs certifiés ISO 27001 avec région UE activée sont éligibles.
- Le scénario recommandé (B) est décrit en détail en §11 du notebook.
