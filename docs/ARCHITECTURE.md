# Architecture cible — churn SaaS B2B

Document de référence technique. Les diagrammes sont recopiés de ceux du notebook §11
(source unique : `notebooks/sections/11_architecture_cible.py`) ; en cas d'écart, la section fait
foi.

---

## 1. Pipeline de données

```mermaid
flowchart LR
    subgraph Sources["Sources de données"]
        CRM[(CRM)] & SUPPORT[(Outil support)] & FACTURATION[(ERP facturation)] & PRODUIT[(Events produit)]
    end
    ETL[Ingestion 23 h 00<br>ETL Python / dbt] --> PARQUET[(Parquet)]
    subgraph Pipeline["Pipeline ML — batch 02 h 00"]
        PREP[Préparation] --> MODELE[Modèle champion<br>registre MLflow] --> SCORES[(Scores)]
    end
    subgraph Exposition["Exposition"]
        API[API FastAPI<br>POST /predict] & BATCH[Job batch Prefect] & CRM_OUT[(CRM — champ<br>risque_churn)] & TDB[Tableau de bord CS]
    end
    CRM & SUPPORT & FACTURATION & PRODUIT --> ETL
    PARQUET --> PREP
    SCORES --> BATCH --> CRM_OUT & TDB
    API --> MODELE
```

Le batch de scoring part à 02 h 00 pour que les alertes soient dans le CRM avant 06 h 00. L'API
sert le **même** modèle champion : les vues batch et temps réel ne peuvent pas se contredire.

**Ce que le batch écrit pour chaque compte** (`flows/scoring_batch.py`, détail dans
`docs/RUNBOOK.md` §2.4) : la probabilité, le palier de la règle de décision à deux niveaux
(`ALERTE_ROUGE` ≥ 0,60, `SURVEILLANCE` au-dessus du seuil de vigilance, `OK`), le rang d'appel des
45 comptes de plus forte valeur attendue (niveau 2) et la valeur à risque. Le palier vient d'une
règle unique, `economie.niveau_risque`, partagée par l'API et le batch. Le seuil de vigilance
voyage avec le modèle : le flow de réentraînement le recalcule hors pli et l'écrit dans les
métadonnées du modèle promu.

---

## 2. Architecture cible retenue — Scénario B (conteneurs managés)

```mermaid
flowchart TB
    subgraph Client["Réseau client"]
        NAVIGATEUR[Navigateur CSM] & CRM_CLIENT[(CRM client)]
    end
    subgraph DMZ["Passerelle exposée"]
        GW[API Gateway<br>HTTPS + JWT] --> WAF[WAF / limitation de débit]
    end
    subgraph Calcul["Conteneurs managés — ECS Fargate / Cloud Run"]
        API_CONT[API FastAPI<br>2 tâches × 0,5 vCPU] & BATCH_CONT[Worker Prefect<br>2 vCPU] & MON_CONT[Surveillance<br>Evidently + Prometheus]
    end
    subgraph Donnees["Données managées"]
        S3[(Stockage objet — Parquet)] & RDS[(PostgreSQL — scores, audit)]
    end
    MLFLOW[Registre MLflow<br>et artefacts]
    IMAGE[CI/CD GitHub Actions<br>tests · gate · image Docker]
    NAVIGATEUR -->|HTTPS| GW
    CRM_CLIENT -->|export quotidien| S3
    WAF --> API_CONT --> MLFLOW
    BATCH_CONT --> S3 & MLFLOW & RDS
    MON_CONT --> RDS
    IMAGE --> API_CONT & BATCH_CONT
```

Passerelle exposée, calcul conteneurisé sans serveur à administrer, données managées et registre
de modèles alimenté **uniquement** par la CI/CD. Aucune brique ne demande Kubernetes, et chacune a
un équivalent standard ailleurs (Docker, Parquet, PostgreSQL) : c'est la réversibilité exigée par
le RSSI. L'image Docker embarque `libgomp1`, requis par LightGBM, pour que le flow puisse servir
cette famille si une future exécution du notebook la désigne (champion actuel : régression
logistique).

---

## 3. Diagramme de séquence — Score à la demande (CSM → API → CRM)

```mermaid
sequenceDiagram
    actor CSM
    participant CRM
    participant GW as API Gateway
    participant API as API FastAPI
    participant REG as Registre MLflow
    participant DB as Base scores
    CSM->>CRM: Ouvre la fiche du compte C-1042
    CRM->>GW: POST /predict {variables du compte, sans client_id} + JWT
    GW->>API: Requête validée (JWT, limitation de débit)
    API->>REG: Modèle champion @production (cache local 1 h)
    API->>API: Inférence → probabilité, décision, 3 facteurs principaux
    API->>DB: Journalise horodatage, version du modèle, score
    API-->>CRM: 200 OK {probabilite_churn, decision, facteurs_shap, facteurs_principaux}
    CRM-->>CSM: Fiche enrichie, rattachée à C-1042 par le CRM
```

Le `client_id` ne quitte jamais le CRM ; les features dérivées (ratios comme
`intensite_support`) sont calculées côté serveur, jamais demandées au client. La cible fait
évoluer par ajout l'API livrée (même route, même contrat d'entrée) :

| Élément | API livrée (§10) | Architecture cible |
|---|---|---|
| Authentification | Clé `X-API-Key`, 60 requêtes/min par IP | JWT validé par l'API Gateway |
| Facteurs de risque | Trois règles issues de l'EDA (§6) | Valeurs SHAP locales calculées en ligne |
| Explication à la demande | Absente | `POST /explain`, mêmes variables que `/predict` |
| Journal d'audit | Absent (l'API ne persiste rien) | Chaque appel journalisé dans PostgreSQL, 12 mois |

---

## 4. Notes

- Tous les secrets (clés API, credentials DB) passent par un gestionnaire de secrets
  (AWS Secrets Manager, GCP Secret Manager ou HashiCorp Vault) — jamais en variable
  d'environnement en clair dans les images Docker.
- La souveraineté des données (hébergement UE) est contrainte par le RSSI :
  seuls des fournisseurs certifiés ISO 27001 avec région UE activée sont éligibles.
- Le scénario recommandé (B) est décrit en détail en §11 du notebook.
