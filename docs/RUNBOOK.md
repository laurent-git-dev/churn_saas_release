# Runbook opérationnel — Modèle churn SaaS B2B

> Ce runbook décrit les procédures opérationnelles pour les équipes MLOps et
> Customer Success. Il couvre le démarrage, le monitoring, le réentraînement
> et le rollback du modèle de churn.

---

## 1. Démarrage rapide

### Prérequis

```bash
# Vérifier que l'environnement est prêt
make setup          # uv sync --all-extras
uv run churn-saas check-data   # vérifie les fichiers sources et le modèle
```

### Lancer l'API de prédiction

```bash
make api            # uvicorn churn_saas.api.main:app --reload
# L'API est disponible sur http://localhost:8000
# Documentation interactive : http://localhost:8000/docs
# Santé : http://localhost:8000/health
```

### Lancer le stack de monitoring complet

```bash
docker-compose up -d prometheus grafana api
# Prometheus : http://localhost:9090
# Grafana    : http://localhost:3000 (admin / voir .env)
# API        : http://localhost:8000
```

---

## 2. Opérations courantes

### 2.1 Générer les prédictions hebdomadaires (batch)

```bash
# 1. Vérifier que le gold dataset est à jour
uv run churn-saas build-gold

# 2. Appliquer le modèle sur tous les comptes
uv run churn-saas predict data/gold/gold_dataset.parquet \
    --sortie reports/predictions_$(date +%Y%m%d).csv

# 3. Vérifier les prédictions
head -5 reports/predictions_$(date +%Y%m%d).csv
```

### 2.2 Rapport de dérive (hebdomadaire)

```bash
make drift
# Génère reports/drift_report_<date>.html
# Ouvrir dans le navigateur et partager lors de la revue hebdomadaire
```

### 2.3 Évaluer le modèle en place

```bash
uv run churn-saas evaluate
# Affiche PR-AUC et ROC-AUC sur le split test reproductible
# Code de retour 0 = gate passée, 1 = gate échouée
```

---

## 3. Réentraînement

### 3.1 Réentraînement automatique (flow complet)

Le flow enchaîne : ingest → features → train → evaluate → gate → promote.

```bash
make flow
# ou, pour forcer le recalcul de toutes les étapes :
uv run python flows/retraining.py --forcer
```

Le flow écrit des marqueurs d'idempotence dans `reports/tables/flow_cache/`.
Si le flow est interrompu, il peut être relancé : les étapes déjà complétées
le même jour sont sautées.

### 3.2 Réentraînement manuel (CLI)

```bash
# Reconstruire le gold + entraîner + évaluer
uv run churn-saas build-gold --forcer
uv run churn-saas train
uv run churn-saas evaluate
uv run churn-saas retrain      # final : entraîne sur 100 % des données
```

### 3.3 Vérifier la promotion MLflow

```bash
# Lister les runs du dernier flow
uv run mlflow ui --backend-store-uri mlruns/
# Puis ouvrir http://localhost:5000
```

---

## 4. Rollback

### 4.1 Procédure de rollback immédiate (< 5 minutes)

En cas de dégradation constatée après une promotion :

```bash
# 1. Lister les archives disponibles
ls -lt artifacts/models/archive_*_best_model.pkl

# 2. Restaurer l'archive souhaitée (remplacer <HORODATAGE>)
cp artifacts/models/archive_<HORODATAGE>_best_model.pkl \
   artifacts/models/best_model.pkl

# 3. Redémarrer l'API
make api    # ou redémarrer le conteneur Docker

# 4. Vérifier la restauration
uv run churn-saas evaluate
```

### 4.2 Rollback du notebook (gold dataset)

```bash
# Restaurer une version précédente des données avec DVC
dvc checkout data/gold/gold_dataset.parquet.dvc
```

### 4.3 Déclencheurs de rollback

- Taux `ALERTE_ROUGE` qui s'écarte de plus de 10 points du taux de référence (≈ 30 %, soit
  1 508 / 5 000 au batch du 29/09/2026 — voir `reports/tables/scores_batch_*_synthese.json`),
  dans un sens ou dans l'autre
- Latence API > 500 ms en P95 (alerte Prometheus déjà déclenchée)
- Retours terrain négatifs de l'équipe CS dans les 48h suivant une promotion
- PR-AUC sur nouvelle cohorte de renouvellements < seuil a priori

---

## 5. Diagnostic — incidents fréquents

### 5.1 API ne répond pas

```bash
# Vérifier le statut du conteneur
docker ps | grep churn_saas_api
docker logs churn_saas_api --tail 50

# Si le modèle est corrompu :
uv run churn-saas check-data
```

**Alerte Prometheus associée :** `APIIndisponible` (déclenche après 1 min)

### 5.2 Latence > 200 ms

```bash
# Vérifier la charge CPU du conteneur
docker stats churn_saas_api

# Si charge CPU > 80 % : augmenter les workers uvicorn
uvicorn churn_saas.api.main:app --workers 4

# Si le modèle est trop lourd : envisager la quantification ou le cache de prédictions
```

**Alerte Prometheus associée :** `LatenceElevee` (déclenche après 5 min à > 200 ms)

### 5.3 Drift détecté (PSI > 0.20)

```bash
# Générer un rapport Evidently détaillé
make drift

# Décision : réentraîner ou non
# Si la dérive est confirmée sur plusieurs features principales :
make flow
```

### 5.4 Gate de promotion échouée

```bash
# Lire les logs du flow
# (le flow écrit dans stderr/stdout avec loguru)

# Options :
# 1. Vérifier si la dérive des données explique la dégradation
make drift

# 2. Inspecter les features construites
uv run pytest tests/test_features_build.py -v

# 3. Vérifier qu'il n'y a pas de fuite de données
uv run pytest tests/test_no_leakage.py -v

# 4. Si la dégradation est confirmée, investiguer les données brutes
# (ne jamais lire data/raw/ directement — utiliser profil_compact())
uv run python -c "
from churn_saas.data.quality import profil_compact
from churn_saas.data.loaders import charger_brut
df = charger_brut('churn_saas_complet')
print(profil_compact(df))
"
```

---

## 6. Tests de non-régression

```bash
# Suite complète (inclut la gate qualité — lente ~2 min)
make check

# Gate qualité seule
uv run pytest tests/test_model_quality_gate.py -v -m slow

# Tests rapides uniquement (CI allégée)
uv run pytest tests/ -v -m "not slow"
```

---

## 7. Contacts et escalade

| Situation | Contact |
|-----------|---------|
| Incident production (API down) | MLOps (astreinte) → escalade DSI si > 30 min |
| Dégradation de performance | Data Scientist → comité si persistant > 1 semaine |
| Question éthique / RGPD | DPO |
| Décision de rollback | MLOps (autonome si urgence) + Data Scientist (notification) |
| Décision de réentraînement | Data Scientist + validation CS Lead |
| Évolution du modèle (nouvelle feature) | PR review — Data Scientist + pair |

---

## 8. Références

| Document | Contenu |
|----------|---------|
| `docs/MODEL_CARD.md` | Caractéristiques et limites du modèle |
| `docs/DATASHEET.md` | Cycle de vie des données |
| `docs/RISK_REGISTER.md` | Registre des risques éthiques et opérationnels |
| `docs/CHECKLIST_LIVRAISON.md` | Contrôle avant envoi du livrable |
| `monitoring/prometheus.yml` | Configuration du scraping Prometheus |
| `monitoring/alerts.yml` | Règles d'alerte |
| `flows/retraining.py` | Flow de réentraînement commenté |
| `src/churn_saas/config.py` | Paramètres centralisés (seuils, cibles, chemins) |
