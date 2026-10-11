# Runbook opérationnel — Modèle churn SaaS B2B

> Ce runbook décrit les procédures opérationnelles pour le data scientist, l'équipe Ops de la
> DSI et le Customer Success : démarrage, scoring, surveillance, réentraînement et retour arrière
> du modèle de churn. Il suit les décisions du notebook (§10 mise en exploitation, §13
> amélioration continue). Règle de gouvernance à garder en tête : **un signal convoque un
> comité, il ne réentraîne jamais le modèle seul** ; seul le retour au champion précédent est
> immédiat.

---

## 1. Démarrage rapide

### Prérequis

```bash
make setup                     # uv sync --all-extras
uv run churn-saas check-data   # sources brutes (SHA-256), gold (colonnes interdites), modèle
```

`check-data` échoue (code 1) si une source brute manque ou si le gold contient une colonne
interdite ; l'absence du gold ou de `best_model.pkl` est signalée sans faire échouer la commande.

**D'où vient le modèle servi.** Chaque exécution du notebook réécrit
`artifacts/models/best_model.pkl` (§10.3.1) : famille et hyperparamètres du champion du §9, sans
les trois leurres, appris sur tout le gold. Il réécrit aussi `best_model_meta.json` (§10.4) :
famille, hyperparamètres, version au registre et empreinte, **sans** `seuil_economique`, si bien
que le seuil de vigilance du §12 s'applique (voir §2.4). Relancer le notebook remplace donc un
champion promu par le flow, modèle et méta ensemble, et replace l'alias `@production`.

### Lancer l'API de prédiction

```bash
make api            # uvicorn churn_saas.api.main:app --reload (développement)
# API : http://localhost:8000 — documentation : /docs — vie : /health — disponibilité : /ready
```

Sans `CHURN_API_KEY`, l'authentification est **désactivée** (mode développement) ; en
production, la définir et passer la clé dans l'en-tête `X-API-Key`. Limites :
`CHURN_RATE_LIMIT` (60 requêtes par minute et par IP, puis 429 avec `Retry-After`) et
`CHURN_MAX_BODY_BYTES` (1 Mio, puis 413).

### Lancer la pile de surveillance complète

```bash
docker compose up -d api prometheus grafana
# API        : http://localhost:8000
# Prometheus : http://localhost:9090 (collecte /metrics toutes les 30 s, règles monitoring/alerts.yml)
# Grafana    : http://localhost:3000 (admin / $GRAFANA_PASSWORD, « admin » par défaut)
```

**Fichiers montés dans le service `api`** (lecture seule) : `./artifacts` (modèle et méta),
`data/raw/catalogue_plans.csv` (sans lui, `/predict` répond 503) et
`reports/tables/seuil_vigilance.json` (sans lui, le repli 0,40 remplacerait le seuil de
vigilance). Les deux fichiers sont montés un par un, avec `create_host_path: false` : s'il en
manque un sur l'hôte (clone sans les données de l'archive, notebook jamais exécuté), le
démarrage échoue avec un message explicite au lieu de créer un répertoire vide à sa place.
Après un `git checkout` de `seuil_vigilance.json`, recréer le conteneur
(`docker compose up -d --force-recreate api`) : un fichier remplacé, et non réécrit, n'est
plus celui que voit le conteneur.

**Grafana est provisionné au démarrage** (`monitoring/grafana/provisioning/`) : la source de
données Prometheus (`http://prometheus:9090`, par défaut) et le tableau de bord
`monitoring/grafana/dashboard.json`, rangé dans le dossier « Churn SaaS ». Rien n'est à importer
à la main. Les logs doivent montrer `inserting datasource from configuration` puis
`finished to provision dashboards` ; l'erreur `plugin xychart is already registered` vient de
l'image Grafana 11.1.0 et reste sans effet. Si le tableau de bord avait été importé à la main
auparavant, le supprimer dans l'interface, sinon le provisioning entre en conflit avec lui.

**Vérifier la pile** (testée de bout en bout le 2026-10-11 sous WSL2 et Docker Desktop) :

```bash
docker compose ps                                   # api « healthy »
curl -s -o /dev/null -w '%{http_code}\n' localhost:8000/ready    # 200 (503 : catalogue absent)
docker compose logs api | grep "seuil de surveillance"           # 0.2774, pas 0.4000
```

Puis http://localhost:9090/targets (`churn_saas_api` **UP**) et, dans Grafana, **Dashboards →
Churn SaaS**. Les panneaux de débit, de latence et de prédictions calculent des taux sur 1 à
5 minutes : ils restent vides tant que l'API ne reçoit pas un trafic continu (quelques appels par
minute pendant au moins 5 minutes). Seul « API — statut » s'affiche immédiatement.

---

## 2. Opérations courantes

### 2.1 Générer les prédictions à la main (hors batch nocturne)

Le moyen recommandé est de relancer le batch : il applique la même règle que l'API (seuil de
vigilance, priorisation à deux niveaux, facteurs SHAP).

```bash
uv run churn-saas build-gold        # reconstruit le gold s'il est absent (--forcer pour le refaire)
uv run python flows/scoring_batch.py --forcer
# Écrit reports/tables/scores_batch_<date>.parquet et scores_batch_<date>_synthese.json
```

La commande CLI `predict` reste disponible pour un fichier ponctuel. Elle applique le même seuil
que l'API et le batch (ordre de priorité du §2.4), mais ne produit ni rang de priorité ni
facteurs SHAP.

```bash
uv run churn-saas predict data/gold/gold_dataset.parquet \
    --sortie reports/predictions_$(date +%Y%m%d).csv
# Colonnes : client_id, probabilite_churn, decision, seuil_applique
```

### 2.2 Rapport de dérive (hebdomadaire)

```bash
make drift
# PSI et test KS sur les variables surveillées (affichés), puis
# rapport Evidently : reports/drift_report_<date>.html
```

En production, le déploiement Prefect `derive-hebdomadaire` (`prefect.yaml`, cron `0 5 * * 1`)
le lance chaque lundi à 05 h 00 ; `make drift` le relance à la demande. **Attention** : faute de
données de production, `monitoring/drift_report.py` compare le gold à une dérive **simulée**
(scénario `adoption_chute`, §13.3) ; ses alertes sont donc attendues tant qu'aucune extraction
réelle ne le remplace.

### 2.3 Contrôler le modèle en place

```bash
uv run churn-saas evaluate
# Affiche PR-AUC et ROC-AUC sur le split 80/20 reproductible (config.RANDOM_SEED)
# Code de retour 0 = PR-AUC ≥ 0,65 (pr_auc_min), 1 = sous la cible ou modèle absent
```

**Ce n'est pas une mesure de performance.** Le modèle servi a été appris sur tout le gold
(§9.15, §10.3.1) : les 1 000 comptes du split ont servi à l'apprentissage et la note est
optimiste. Cette commande vérifie seulement que l'artefact se charge et classe correctement. La
note de référence reste celle du jeu de test du notebook (§9.14 : PR-AUC 0,760, ROC-AUC 0,881) ;
la performance réelle se mesure par cohorte de renouvellements (§13.7).

### 2.4 Batch nocturne de scoring et règle à deux niveaux

```bash
uv run python flows/scoring_batch.py                                  # date du jour
uv run python flows/scoring_batch.py --date-ref 2026-10-10 --forcer   # republier une date
```

En production : déploiement Prefect `nocturne` (cron `0 2 * * *`, fuseau Europe/Paris). Lecture
des données relancée 3 fois en cas d'échec ; un lot manqué se rejoue avec `--forcer`, et le CRM
garde les scores de la veille en attendant.

Chaque compte reçoit un **palier** (`economie.niveau_risque`, règle unique partagée avec l'API)
et une **action CS** :

| Palier | Condition | Usage |
|---|---|---|
| `ALERTE_ROUGE` | score ≥ 0,60 (`config.SEUILS_RISQUE`), jamais sous le seuil de surveillance | Priorité de revue |
| `SURVEILLANCE` | score ≥ seuil de surveillance | Niveau 1 de la règle : vigilance, actions automatisées |
| `OK` | sinon | Veille |

**Seuil de surveillance servi** (`ModelStore.seuil`), par ordre de priorité :
1. `seuil_economique` de `artifacts/models/best_model_meta.json`, s'il y figure (chaque
   promotion, par le flow ou par `churn-saas train`, y écrit le seuil de vigilance du modèle,
   recalculé hors pli sur son train par `economics.seuil_vigilance_hors_pli` ; la méta du
   notebook n'en porte pas) ;
2. sinon le seuil de vigilance du notebook (`reports/tables/seuil_vigilance.json`, 0,2774 au
   2026-10-10 : recall 80 % hors pli, 1 813 comptes, §12.6) ;
3. sinon `config.SEUILS_RISQUE["surveillance"]` (0,40).

Le seuil effectif est écrit dans le log au chargement du modèle (`Modèle chargé depuis … seuil
de surveillance : …`) et dans la colonne `seuil_applique` du batch ; le §10.10 du notebook
signale tout écart avec `seuil_vigilance.json`.

**Actions CS** (niveau 2) : les 45 comptes de plus forte valeur attendue positive sont à
`Contacter` (rang 1 à 45, capacité `config.HYPOTHESES_ECONOMIQUES["capacite_gestes_mois"]`) ;
les autres comptes à valeur attendue positive reçoivent une `Action automatisée` ; les autres
restent en `Veille`. Un compte sans MRR reste en veille, sans rang ni valeur à risque.

**Repères du batch du 2026-10-10** (`scores_batch_2026-10-10_synthese.json`, modèle déployé
appliqué au gold, seuil 0,2774) : 5 000 comptes, 910 `ALERTE_ROUGE`, 911 `SURVEILLANCE`,
3 179 `OK` ; 45 à contacter, 2 920 actions automatisées, 2 035 en veille ; 150 comptes sans MRR.
Les 1 821 comptes au-dessus du seuil sont proches des 1 813 de la règle mesurée hors pli
(§12.6), mais pas identiques : le batch note, avec le modèle déployé (sans leurres), des comptes
vus à l'entraînement. Le contrôle de dérive compare un batch au précédent, jamais à la mesure
hors pli.

---

## 3. Réentraînement

### 3.1 Réentraînement par le flow (procédure normale)

Le réentraînement est **décidé par un comité** (trimestriel, ou ad hoc après un signal du §5.3),
jamais déclenché par une alerte. Le flow enchaîne ingest → features → train → evaluate → gate →
promote. Il reprend la famille et les hyperparamètres du champion
(`train.famille_et_hyperparametres_champion` : méta du champion promu, à défaut la plus récente
étude Optuna `reports/tables/optuna_*_meilleurs_params.json`) : régression logistique au
2026-10-10, LightGBM possible si une future exécution du notebook le désigne (l'image Docker
embarque `libgomp1`, requis par LightGBM). Il ne relance ni la comparaison des familles ni le
tuning.

```bash
make flow                                  # flows/retrain_flow.py → flows/retraining.py
uv run python flows/retraining.py --forcer # ignore les marqueurs et recalcule toutes les étapes
```

En production, le comité le lance depuis l'interface Prefect ou par
`uv run prefect deployment run reentrainement-churn/sur-decision-comite` (déploiement sans
planification). Code de retour de `make flow` : 0 = promu, 1 = gate non passée (le champion
reste en place), 2 = étape en échec après ses relances.

**Ce que fait l'étape `train`** : retire colonnes interdites et leurres, découpe 80/20
stratifié, recalcule le seuil de vigilance du candidat sur ses prédictions hors pli, entraîne
sur les 80 % et mesure PR-AUC et ROC-AUC sur les 20 %. Ce split est un contrôle de
**non-régression**, pas une estimation non biaisée. Le candidat promu est celui appris sur
80 % (pas de réapprentissage sur 100 %).

**Gate** (`decision_promotion`, règle unique partagée avec le notebook) : PR-AUC ≥ 0,65 **et**
pas plus de 0,01 sous le champion. **Promotion** : archive l'ancien champion **et sa méta**
sous un même horodatage (`archive_<horodatage>_best_model.pkl` et
`archive_<horodatage>_best_model_meta.json`, `train.archiver_champion`), copie le candidat dans
`best_model.pkl`, journalise dans MLflow, déplace l'alias `@production`, puis écrit
`best_model_meta.json` (seuil de vigilance du candidat et `version_registre`, `null` si MLflow
est indisponible). Redémarrer ensuite l'API pour qu'elle charge le nouveau modèle.

Le flow écrit des marqueurs d'idempotence dans `reports/tables/flow_cache/<date>_<étape>.done` :
s'il est interrompu, il peut être relancé, et les étapes déjà faites le même jour sont sautées.

### 3.2 Entraînement en ligne de commande (sans Prefect)

```bash
uv run churn-saas build-gold --forcer   # reconstruit le gold
uv run churn-saas train                 # challenger 80/20, gate, promotion si elle passe
uv run churn-saas evaluate              # contrôle de l'artefact (voir §2.3)
```

`churn-saas train` suit **la même règle que le flow** : il entraîne un challenger
(`challenger.pkl`, seuil de vigilance hors pli sur ses 80 % d'apprentissage), le soumet à la
gate `decision_promotion` face au champion, puis le promeut par la même fonction que le flow
(`train.promouvoir_challenger` : archive du champion et de sa méta, registre MLflow, alias
`@production`, méta servie). Code de retour : 0 = promu, 1 = gate non passée (le champion reste
en place). Sur un clone neuf, sans champion, seul le plancher de 0,65 s'applique : c'est le
moyen d'obtenir un premier modèle sans exécuter le notebook. Aucune option ne contourne la gate ;
en urgence, le bon outil est `churn-saas rollback` (§4.1).

**Limite.** La méta écrite par le notebook ne porte pas de PR-AUC (mesurée sur 5 plis, elle ne
serait pas comparable au split de 20 %) : face au modèle du notebook, seul le plancher s'applique,
et un challenger CLI au-dessus de 0,65 le remplacerait. La commande `churn-saas retrain`, qui
écrivait un `best_model_final.pkl` qu'aucun service ne lisait, est supprimée : le réapprentissage
sur 100 % des données est celui du notebook (§9.15, §10.3.1).

### 3.3 Vérifier la promotion dans MLflow

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
# Ouvrir http://localhost:5000 : expérience churn_saas_classification,
# modèle enregistré churn_saas_champion, alias @production
```

---

## 4. Retour arrière (rollback)

### 4.1 Procédure de retour arrière immédiate (moins de 5 minutes)

En cas de dégradation constatée après une promotion (déclencheurs au §4.3) :

```bash
# 1. Lister les archives disponibles (une par promotion du flow)
uv run churn-saas rollback

# 2. Restaurer l'archive souhaitée
uv run churn-saas rollback <HORODATAGE>

# 3. Redémarrer l'API, puis vérifier le seuil dans le log de chargement
docker compose restart api   # ou relancer make api

# 4. Contrôler l'artefact (contrôle de chargement, pas mesure de performance, voir §2.3)
uv run churn-saas evaluate
```

`churn-saas rollback <HORODATAGE>` (`train.restaurer_archive`) fait en une fois :
1. archive le champion en place, modèle et méta : le retour arrière est lui-même réversible ;
2. remet le modèle archivé dans `best_model.pkl` et sa méta dans `best_model_meta.json`
   (seuil servi, famille, hyperparamètres). Une archive sans méta (antérieure au 11/10/2026)
   retire la méta courante : le seuil de vigilance de `seuil_vigilance.json` s'applique, ce qui
   est juste pour le modèle du notebook mais pas pour un ancien champion du flow ;
3. replace l'alias `@production` sur la `version_registre` de la méta restaurée. Si elle est
   absente (archive ancienne, MLflow indisponible à la promotion), la commande le signale et
   l'alias se replace à la main :

```bash
uv run python -c "
import mlflow
from churn_saas import config
mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
mlflow.MlflowClient().set_registered_model_alias(config.NOM_REGISTRE_MLFLOW, 'production', '<VERSION>')
"
```

Le retour arrière restaure un état déjà validé, sans réentraînement ; la cause de la
dégradation est analysée ensuite.

### 4.2 Revenir à une version antérieure du jeu gold (DVC)

```bash
git checkout <étiquette> -- data/gold/gold_dataset.parquet.dvc
uv run dvc checkout data/gold/gold_dataset.parquet.dvc
```

Aucun stockage distant DVC n'est configuré : `dvc checkout` ne restaure une version que si elle
est encore dans le cache local de la machine. Sinon, reconstruire le gold depuis les données
brutes de la version voulue (`uv run churn-saas build-gold --forcer`).

### 4.3 Déclencheurs de retour arrière

- Part d'`ALERTE_ROUGE` qui s'écarte de plus de 10 points de la référence, dans un sens ou dans
  l'autre (`config.CIBLES_PERFORMANCE["ecart_alerte_rouge_rollback_pts"]`), **après une
  promotion récente**. Référence : 18,2 %, soit 910 / 5 000 au batch du 2026-10-10
  (`reports/tables/scores_batch_*_synthese.json`). Sans promotion récente, le même écart
  convoque un comité ad hoc au lieu d'un retour arrière (§13.8 du notebook).
- PR-AUC d'une cohorte de renouvellements < `pr_auc_critique` (0,60) **et** champion précédent
  meilleur sur la même cohorte (décision du comité ad hoc, notebook §2.6 et §13.8).
- Latence moyenne de `/predict` > 200 ms pendant 5 min après une promotion (alerte Prometheus
  `LatenceElevee`) ; le p95 n'a pas d'alerte automatique et se lit sur le tableau de bord
  Grafana, en revue hebdomadaire.
- Retours terrain négatifs de l'équipe CS dans les 48 h suivant une promotion (signal
  qualitatif, à confirmer par le data scientist).

---

## 5. Diagnostic — incidents fréquents

### 5.1 L'API ne répond pas

```bash
docker compose ps api
docker compose logs api --tail 50
curl -s localhost:8000/health   # vie du processus (toujours 200 s'il tourne)
curl -s localhost:8000/ready    # 503 si le modèle ou le catalogue des plans manque

# Si le modèle est absent ou illisible :
uv run churn-saas check-data
```

**Alerte Prometheus associée :** `APIIndisponible` (`up == 0` pendant 1 min, critique). Pendant
l'incident, le CRM garde le score de la veille.

### 5.2 Latence > 200 ms

```bash
docker stats   # charge CPU du conteneur de l'API

# Si charge CPU > 80 % : augmenter les workers uvicorn (le Dockerfile en lance un seul)
uv run uvicorn churn_saas.api.main:app --workers 4
```

Le modèle seul répond en 61,9 ms au p95 (§9.13) ; de bout en bout, `POST /predict` atteint
173 ms au p95 (§10.3.3), soit 27 ms de marge. Une latence élevée vient donc d'abord de la charge
ou de la chaîne de préparation, rarement du modèle.

**Alerte Prometheus associée :** `LatenceElevee` (latence **moyenne** de `/predict` sur 5 min
> 200 ms, pendant 5 min).

### 5.3 Dérive détectée (PSI)

```bash
make drift   # rapport Evidently détaillé, PSI et KS par variable
```

| PSI sur une variable surveillée | Action |
|---|---|
| 0,10 à 0,20 | Noté pour le comité trimestriel |
| ≥ 0,20 | Comité ad hoc (data scientist + CS Lead) : analyse causale et décision sous 72 h — réentraîner (§3.1), maintenir ou geler |

**Ne pas lancer `make flow` directement.** Vérifier d'abord une panne d'intégration CRM (champ
qui n'arrive plus : compteur `churn_champs_imputes_total` sur `/metrics`) : un réentraînement
apprendrait la panne au lieu de la corriger.

### 5.4 Gate de promotion échouée

Le champion reste en place ; ce n'est pas un incident de service.

```bash
# Le motif est dans les logs du flow (loguru, sortie standard), et dans l'historique Prefect.

# 1. Vérifier si une dérive des données explique la dégradation
make drift

# 2. Inspecter les features construites
uv run pytest tests/test_features_build.py -v

# 3. Vérifier qu'il n'y a pas de fuite de données
uv run pytest tests/test_no_leakage.py -v

# 4. Si la dégradation est confirmée, examiner les données brutes
#    (ne jamais lire data/raw/ directement — utiliser profil_compact())
uv run python -c "
from churn_saas.data.quality import profil_compact
from churn_saas.data.loaders import charger_brut
df = charger_brut('churn_saas_complet')
print(profil_compact(df))
"
```

Porter le résultat au comité : une gate non passée ne se contourne pas.

---

## 6. Tests de non-régression

```bash
# Identique à la CI : lint, tests rapides, audit de rédaction, puis build du notebook
make check

# Gate qualité du modèle (tests marqués slow, exclus de make check, joués par la CI)
make test-slow
uv run pytest tests/test_model_quality_gate.py -v -m slow   # gate seule

# Tests rapides uniquement
make test
```

---

## 7. Contacts et escalade

| Situation | Contact |
|-----------|---------|
| Incident production (API indisponible) | Ops DSI (astreinte) → data scientist si le modèle est en cause |
| Dégradation de performance | Data scientist → comité ad hoc (décision sous 72 h) |
| Question éthique / RGPD | DPO |
| Décision de retour arrière | Data scientist ; Ops DSI autonome en urgence, data scientist notifié |
| Décision de réentraînement | Comité (trimestriel ou ad hoc) : data scientist + CS Lead ; la promotion relève de la gate |
| Évolution du modèle (nouvelle feature) | Pull request relue — data scientist + pair, CI verte |

---

## 8. Références

| Document | Contenu |
|----------|---------|
| `docs/MODEL_CARD.md` | Caractéristiques et limites du modèle |
| `docs/DATASHEET.md` | Cycle de vie des données |
| `docs/RISK_REGISTER.md` | Registre des risques éthiques et opérationnels |
| `prefect.yaml` | Déploiements `nocturne`, `derive-hebdomadaire`, `sur-decision-comite` |
| `docker-compose.yml` | Services `api`, `prometheus`, `grafana` |
| `monitoring/prometheus.yml` | Configuration de la collecte Prometheus |
| `monitoring/alerts.yml` | Règles d'alerte `APIIndisponible`, `LatenceElevee` |
| `flows/scoring_batch.py` | Batch nocturne de scoring |
| `flows/retraining.py` | Flow de réentraînement commenté |
| `src/churn_saas/config.py` | Paramètres centralisés (seuils, cibles, chemins) |
