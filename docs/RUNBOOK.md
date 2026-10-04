# Runbook opérationnel — Modèle churn SaaS B2B

> Ce runbook décrit les procédures opérationnelles pour le data scientist,
> l'équipe Ops de la DSI et le Customer Success. Il couvre le démarrage, le monitoring, le réentraînement
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

### 2.1 Générer les prédictions à la main (hors batch nocturne)

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

En production, le déploiement Prefect `derive-hebdomadaire` (`prefect.yaml`) le lance chaque lundi
à 05 h 00 ; `make drift` le relance à la demande.

### 2.3 Évaluer le modèle en place

```bash
uv run churn-saas evaluate
# Affiche PR-AUC et ROC-AUC sur le split test reproductible
# Code de retour 0 = gate passée, 1 = gate échouée
```

### 2.4 Batch nocturne de scoring et règle à deux niveaux

```bash
uv run python flows/scoring_batch.py                 # date du jour
uv run python flows/scoring_batch.py --date-ref 2026-10-03 --forcer   # republier une date
# Écrit reports/tables/scores_batch_<date>.parquet et scores_batch_<date>_synthese.json
```

Chaque compte reçoit un **palier** (`economie.niveau_risque`, règle unique partagée avec l'API)
et une **action CS** :

| Palier | Condition | Usage |
|---|---|---|
| `ALERTE_ROUGE` | score ≥ 0,60 (`config.SEUILS_RISQUE`), jamais sous le seuil de surveillance | Appel avant échéance |
| `SURVEILLANCE` | score ≥ seuil de surveillance | Niveau 1 de la règle : revue hebdomadaire, actions automatisées |
| `OK` | sinon | Veille |

**Seuil de surveillance servi** (`ModelStore.seuil`), par ordre de priorité : `seuil_economique`
de `artifacts/models/best_model_meta.json` (le flow de réentraînement y écrit le seuil de
vigilance du challenger, recalculé hors pli sur son train), sinon le seuil de vigilance du
notebook (`reports/tables/seuil_vigilance.json`, 0,277 au 2026-10-03, recall 80 % hors pli),
sinon `config.SEUILS_RISQUE["surveillance"]` (0,40). Le seuil effectif est écrit dans le log
au chargement du modèle (`Modèle chargé depuis … seuil de surveillance : …`).

**Actions CS** (niveau 2) : les 45 comptes de plus forte valeur attendue positive sont à
`Contacter` (rang 1 à 45, capacité `config.HYPOTHESES_ECONOMIQUES["capacite_gestes_mois"]`) ;
les autres comptes à valeur attendue positive reçoivent une `Action automatisée` ; les autres
restent en `Veille`. Un compte sans MRR reste en veille, sans rang.

**Repères du batch du 2026-10-03** (`scores_batch_2026-10-03_synthese.json`, modèle déployé
appliqué au gold) : 5 000 comptes, 910 `ALERTE_ROUGE`, 911 `SURVEILLANCE`, 3 179 `OK` ;
45 à contacter, 2 920 actions automatisées ; 150 comptes sans MRR. Les 1 821 comptes au-dessus du
seuil sont proches des 1 813 de la règle mesurée hors pli (§12.6), mais pas identiques : le batch
note, avec le modèle déployé (sans leurres), des comptes vus à l'entraînement. Le contrôle de
dérive compare un batch au précédent, jamais à la mesure hors pli.

---

## 3. Réentraînement

### 3.1 Réentraînement automatique (flow complet)

Le flow enchaîne : ingest → features → train → evaluate → gate → promote. Il reprend la famille
et les hyperparamètres du champion du notebook (`train.famille_et_hyperparametres_champion`, lus
dans `reports/tables/optuna_*_meilleurs_params.json`) : régression logistique au 2026-10-03,
LightGBM possible si une future exécution du notebook le désigne (l'image Docker embarque
`libgomp1`, requis par LightGBM). Il ne relance pas la comparaison des familles.

```bash
make flow
# ou, pour forcer le recalcul de toutes les étapes :
uv run python flows/retraining.py --forcer
```

En production, le comité le lance depuis l'interface Prefect ou par
`uv run prefect deployment run reentrainement-churn/sur-decision-comite` (déploiement sans
planification). Code de retour de `make flow` : 0 = promu, 1 = gate non passée, 2 = étape en
échec après ses relances.

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
uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
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

- Taux `ALERTE_ROUGE` qui s'écarte de plus de 10 points du taux de référence (18,2 %, soit
  910 / 5 000 au batch du 2026-10-03 avec le champion régénéré — voir
  `reports/tables/scores_batch_*_synthese.json`), dans un sens ou dans l'autre
  (`config.CIBLES_PERFORMANCE["ecart_alerte_rouge_rollback_pts"]`)
- Latence moyenne de `/predict` > 200 ms pendant 5 min après une promotion (alerte
  Prometheus `LatenceElevee`) ; le p95 est suivi sur le dashboard Grafana
- Retours terrain négatifs de l'équipe CS dans les 48h suivant une promotion
- PR-AUC sur nouvelle cohorte de renouvellements < `pr_auc_critique` (0,60), **et** champion
  précédent meilleur sur la même cohorte (décision du comité ad hoc, notebook §2.6)

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
| Incident production (API down) | Ops DSI (astreinte) → Data Scientist si le modèle est en cause |
| Dégradation de performance | Data Scientist → comité si persistant > 1 semaine |
| Question éthique / RGPD | DPO |
| Décision de rollback | Data Scientist ; Ops DSI autonome en urgence, Data Scientist notifié |
| Décision de réentraînement | Data Scientist + validation CS Lead |
| Évolution du modèle (nouvelle feature) | PR review — Data Scientist + pair |

---

## 8. Références

| Document | Contenu |
|----------|---------|
| `docs/MODEL_CARD.md` | Caractéristiques et limites du modèle |
| `docs/DATASHEET.md` | Cycle de vie des données |
| `docs/RISK_REGISTER.md` | Registre des risques éthiques et opérationnels |
| `monitoring/prometheus.yml` | Configuration du scraping Prometheus |
| `monitoring/alerts.yml` | Règles d'alerte |
| `flows/retraining.py` | Flow de réentraînement commenté |
| `src/churn_saas/config.py` | Paramètres centralisés (seuils, cibles, chemins) |
