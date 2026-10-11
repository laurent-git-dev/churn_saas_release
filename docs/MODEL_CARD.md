# Model Card — Modèle de détection de churn SaaS B2B

> Format inspiré de Mitchell et al. (2019) *"Model Cards for Model Reporting"*,
> adapté au contexte CISIA et aux recommandations CNIL sur la transparence des
> systèmes d'IA.

---

## 1. Informations générales

| Champ | Valeur |
|-------|--------|
| **Nom du modèle** | `churn_saas_champion` |
| **Version** | Version au registre MLflow sous l'alias `@production` : `version_registre` et `empreinte` dans `artifacts/models/best_model_meta.json`, écrits par le notebook (§10.4) et par le flow de réentraînement ; la CLI y écrit à la place `version_code` (commit git) |
| **Date d'entraînement** | `date_entrainement` dans `artifacts/models/best_model_meta.json` (notebook, flow, CLI) |
| **Auteur** | Équipe Data Science — projet CISIA |
| **Objectif** | Classification binaire : probabilité de résiliation (churn) d'un compte SaaS B2B à l'horizon de la prochaine échéance contractuelle |
| **Tâche secondaire** | Régression CLV (`valeur_vie_client_eur`) — modèle distinct (`regression_clv.joblib`) |
| **Artefacts** | `reports/tables/modele_final.joblib` : champion du notebook (§9.15), entraîné sur toutes les variables · `artifacts/models/best_model.pkl` : modèle déployé (§10.3), même algorithme et mêmes hyperparamètres, sans les trois leurres (`config.COLONNES_LEURRES_SUSPECTES`) ; réécrit, avec sa méta, à chaque exécution du notebook et à chaque promotion (flow de réentraînement ou `churn-saas train`, tous deux soumis à la gate `decision_promotion`) |
| **Tracking expériences** | MLflow — base SQLite `mlruns/mlflow.db` (`config.MLFLOW_TRACKING_URI`), modèles sérialisés au format skops ; modèle déployé enregistré au registre `churn_saas_champion` sous l'alias `@production` (§10.4) |

---

## 2. Utilisation prévue

### Cas d'usage autorisés

Le score alimente une **règle de décision à deux niveaux** (§12.6, §14.2) :

- **Niveau 1 — vigilance (revue hebdomadaire)** : comptes dont le score dépasse le seuil de
  vigilance, fixé pour détecter 80 % des churners hors pli (`config.RECALL_CIBLE_VIGILANCE`,
  seuil persisté dans `reports/tables/seuil_vigilance.json`). Palier `SURVEILLANCE` de l'API et
  du batch : actions automatisées à coût quasi nul, fiche signalée au CSM.
- **Niveau 2 — appels CSM (mensuel)** : parmi les comptes en vigilance, les 45 de plus forte
  valeur attendue (P(churn) × MRR × horizon × marge × taux de succès − coût du geste ;
  capacité `config.HYPOTHESES_ECONOMIQUES["capacite_gestes_mois"]`).
- **Alerte sur signal faible** : palier `ALERTE_ROUGE` (score ≥ 0,60, `config.SEUILS_RISQUE`,
  jamais sous le seuil de vigilance) ; pour un compte d'au moins 1 000 € de MRR, contact proactif
  sous 48 h (cas d'usage 2, §2.2).
- **Préparation de renouvellement** : argumentaire de rétention personnalisé
  basé sur les features les plus contributives (SHAP).

### Cas d'usage hors périmètre

- **Décision automatique sans revue humaine** — le score est un outil d'aide
  à la décision, pas un déclencheur automatique de résiliation ou de pénalité.
- **Évaluation individuelle de l'employé CS** — le modèle évalue le comportement
  du compte client, pas la performance de l'équipe CS.
- **Extrapolation à d'autres marchés** — entraîné sur le portefeuille d'un seul éditeur SaaS B2B
  (clients en France pour 54 %, ailleurs en Europe et au Canada) ; performances non garanties
  hors de cette distribution.

---

## 3. Algorithme et architecture

| Paramètre | Valeur |
|-----------|--------|
| **Famille** | Régression logistique régularisée L2 (`RegressionLogistiqueRecalibree`, `churn_saas.models.calibration`) |
| **Sélection** | Comparée à deux baselines (hasard stratifié, règle métier) et à deux ensembles d'arbres (forêt aléatoire, LightGBM) sur les mêmes plis ; retenue par le protocole fixé a priori en §8.8 : seuil de PR-AUC, test de Wilcoxon apparié corrigé par Holm, règle de parcimonie (§9.3.1). Après optimisation Optuna des deux candidats, elle devance LightGBM sur 13 des 15 plis du jeu de développement (PR-AUC 0,788 contre 0,777, `comparaison_modeles_optimises.parquet`), tout en étant plus simple et plus explicable. Note finale sur le jeu de test (§9.14) |
| **Bibliothèque** | scikit-learn ≥ 1.5 (1.9.1 installée, `uv.lock`) |
| **Pipeline** | Préprocesseur scikit-learn fitté dans chaque pli : numériques → imputation médiane, standardisation (`log1p` des variables asymétriques testé puis écarté, §7.8.3) ; catégorielles → normalisation casse/espaces, imputation `"inconnu"`, encodage one-hot ; écart du CSAT à la médiane du secteur appris sur le pli (`EcartAuGroupe`) → régression logistique |
| **Hyperparamètres optimisés** | Optuna (30 essais, échantillonneur TPE, 5 plis stratifiés ; le `MedianPruner` configuré n'élague aucun essai) sur la régression logistique **et** LightGBM : `C` = 0,075 et `max_iter` = 500 retenus pour le champion (`optuna_baseline__regression_logistique_meilleurs_params.json`, §9.7). Gain sur les valeurs par défaut : +0,001 de PR-AUC pour la régression logistique, +0,017 pour LightGBM (`optuna_lightgbm_meilleurs_params.json`), insuffisant pour le hisser au niveau du champion (§9.9) |
| **Gestion du déséquilibre** | `class_weight="balanced"` ; SMOTE évalué en §9.6 à titre de comparaison seulement (décalage des probabilités sans forme connue) |
| **Calibration** | Correction d'intercept analytique : la pondération décale le logit d'une quantité connue, log((1 − π)/π), retirée après l'apprentissage. Aucun paramètre appris supplémentaire ; effet mesuré par le score de Brier en §9.6 |
| **Graine** | `config.RANDOM_SEED = 42` (unique dans le projet) |

---

## 4. Données d'entraînement

| Champ | Valeur |
|-------|--------|
| **Source** | `data/raw/churn_saas_complet.csv` — données fournies par l'éditeur SaaS |
| **Volume** | ~5 000 lignes, 29 colonnes brutes ; nombre de variables après feature engineering affiché en §9.1 (champion) et §10.3 (modèle déployé, sans les leurres) |
| **Période** | Export déclaré de janvier 2022 à décembre 2023, mais souscriptions du 15/02/2022 au 10/01/2025 et extraction estimée au 04/02/2025 : incohérence du jeu fourni, à lever avec le Data Owner (§3.5, §6.11). Une seule photographie du portefeuille |
| **Taux de churn** | 28 % (1 400 churners sur 5 000 clients uniques, mesuré en §6) — déséquilibre de classes ; plancher PR-AUC d'un modèle aléatoire ≈ 0,28 |
| **Jeu de test** | 20 % des comptes (`config.PART_TEST`, 1 000 comptes), stratifié, `config.RANDOM_SEED` : mis de côté en §9.1 avant toute comparaison, ouvert une seule fois en §9.14. Même découpage dans le notebook, la CLI et le flow |
| **Validation (notebook)** | Validation croisée 5 plis × 3 répétitions (`RepeatedStratifiedKFold`) sur les 4 000 comptes de développement, pour choisir ; note de référence sur le jeu de test ; le modèle final est ensuite réentraîné sur la totalité des données (§9.15) |
| **Split (CLI et flow)** | Le même découpage 80 % / 20 % sert au contrôle de non-régression avant promotion |
| **Anti-fuite** | `config.COLONNES_INTERDITES` exclu ; toute transformation apprise dans les plis |
| **Datasheet** | `docs/DATASHEET.md` |

---

## 5. Métriques de performance

Trois mesures, toutes produites par du code exécutable :

- **Jeu de test (note de référence)** : modèle retenu appris sur le seul développement, seuil de
  vigilance fixé sur ses prédictions hors pli, puis noté une fois sur le test, avec un
  intervalle de confiance à 95 % par bootstrap (§9.14, `evaluation_test.json`) ;
- **Validation croisée** : moyenne ± écart-type sur les 15 plis du développement (§9.12,
  `comparaison_modeles_optimises.parquet`), qui sert à choisir ; prédictions out-of-fold (hors
  pli) du modèle final sur tout le portefeuille (§12.2, `probas_oof.joblib`), qui servent aux euros ;
- **CLI et flow** : même split test de 20 % (`best_model_meta.json`).

**Valeurs au 2026-10-10** (relues par `churn_saas.synthese.charger_synthese()` et dans
`reports/tables/` ; elles changent à chaque régénération, le notebook fait foi) :

| Métrique | Valeur mesurée | Source | Seuil a priori (`config.CIBLES_PERFORMANCE`) | Statut |
|----------|---------------|--------|----------------|--------|
| PR-AUC (sélection) | **0,760 sur le test** [0,710 ; 0,804] ; CV : 0,788 ± 0,019 ; hors pli : 0,784 | §9.14, §12.2.4 | ≥ 0,65 (`pr_auc_min`) | Atteint |
| ROC-AUC (co-principale) | **0,881 sur le test** [0,857 ; 0,903] ; CV : 0,889 ± 0,008 ; hors pli : 0,888 | §9.14, §12.2.4 | ≥ 0,80 (`roc_auc_min`) | Atteint |
| Recall au seuil de vigilance (co-principale) | **81,8 % sur le test** [77,4 % ; 86,3 %], seuil 0,270 fixé sur le développement ; 80,0 % hors pli au seuil livré 0,277 ; 76,1 % à 85,7 % sur le pli qui n'a pas fixé le seuil | §9.14, §12.6, `seuil_vigilance.json` | ≥ 75 % sur le test (`recall_vigilance_min`) | Atteint |
| Score de Brier (calibration) | 0,122 sur le test ; 0,116 après correction d'intercept, contre 0,135 sans (développement) | §9.6, §9.14, `comparaison_desequilibre.json` | Inférieur à la version non corrigée | Atteint |
| Latence unitaire | modèle seul : médiane 40,5 ms, p95 61,9 ms ; `POST /predict` de bout en bout : p95 173 ms | §9.13, §10.3.3, `latence_modele_final.json` | ≤ 200 ms (`latence_unitaire_ms`) | Atteinte |
| Latence batch (5 000 comptes) | 0,107 s | §9.13, `latence_modele_final.json` | ≤ 300 s (`latence_batch_5k_s`) | Atteinte |

L'écart entre test et validation croisée (−0,028 de PR-AUC) tient dans l'intervalle de confiance
du test : la validation croisée, qui a servi à choisir, n'est pas sensiblement optimiste.

**Lecture des deux niveaux** (§12.6, `economics.table_deux_niveaux`) :

| Niveau | Comptes | Part du portefeuille | Churners couverts | Recall | Precision |
|---|---|---|---|---|---|
| 1 — vigilance (seuil 0,277) | 1 813 | 36 % | 1 120 | 80,0 % | 61,8 % |
| 2 — appels CSM (capacité 45) | 45 | 0,9 % | 31 | 2,2 % | 68,9 % |

Au niveau 1, 280 churners restent sous le seuil (faux négatifs). Le recall du niveau 2 est faible
par construction : la capacité (45 gestes par mois pour 1 400 churners) en est la limite, pas le
modèle.

**Comparaison des candidats** (15 plis du jeu de développement, `comparaison_modeles.parquet` et
`comparaison_modeles_optimises.parquet`) :

| Modèle | PR-AUC par défaut | PR-AUC optimisée | ROC-AUC optimisée |
|---|---|---|---|
| Régression logistique (champion) | 0,787 | 0,788 | 0,889 |
| LightGBM | 0,762 | 0,777 | 0,884 |
| Forêt aléatoire (non optimisée) | 0,761 | — | — |
| B1 — règle métier | 0,408 | — | — |
| B0 — hasard stratifié | 0,285 | — | — |

> Ces valeurs sont une photographie : la référence reste le notebook (§9, §12), qui les recalcule
> à chaque exécution. Le contrôle de cohérence docs ↔ artefacts fait partie de la passe finale.

---

## 6. Évaluation d'équité (biais)

Protocole fixé avant l'entraînement (§4.4), mesure sur le modèle final (§12.14) : prédictions
hors pli, jeu gold dédoublonné, modalités normalisées (casse, espaces), manquants regroupés en
« inconnu ».

| Sous-groupe | Modalités | Métriques mesurées |
|-------------|-----------|-------------------|
| `pays` | France, Allemagne, Belgique, Suisse, Espagne, Canada, inconnu | TPR, FPR, ROC-AUC, écart de calibration |
| `taille_entreprise` | TPE, PME, ETI, GE | idem |
| `secteur` | Commerce, Tech, Finance, Industrie, Éducation, Santé, Public, inconnu | idem |

**Seuil d'alerte de l'audit.** TPR et FPR sont mesurés au seuil qui signale autant de comptes
qu'il y a de churners : le seuil de rentabilité signale la grande majorité du portefeuille et la
sélection sous capacité est trop petite pour comparer des segments.

**Seuils de revue** (`config.EQUITE`, choix de gouvernance) : écart de TPR > 15 points entre
modalités d'un attribut (égalité des chances, Hardt et al. 2016), ou écart de calibration
> 5 points dans une modalité. Un dépassement déclenche une revue avant déploiement (risque R01
de `docs/RISK_REGISTER.md`). Une modalité de moins de 30 churners n'est pas interprétée.

**Résultats au 2026-10-10** (§12.14, recalculés à chaque exécution ; 19 modalités, toutes avec
au moins 30 churners) :

| Attribut | Écart de TPR | Modalité la moins détectée | Écart de calibration max | Revue requise |
|---|---|---|---|---|
| `pays` | 11,0 points | Belgique | 1,0 point | non |
| `secteur` | 12,4 points | Finance | 0,5 point | non |
| `taille_entreprise` | 10,4 points | ETI | 0,2 point | non |

Aucun seuil n'est franchi ; l'écart du secteur Finance, proche des 15 points, justifie un suivi.

---

## 7. Limites connues

1. **Délai d'étiquettes** — la performance réelle n'est mesurable qu'au prochain
   renouvellement contractuel (1-12 mois). Les métriques sur le jeu de test (§9.14) ne
   reflètent pas la performance en production en temps réel (voir §13.7). Ce test unique
   compte 1 000 comptes : sa note est bruitée, d'où son intervalle de confiance.

2. **Enrichissements simulés** — les référentiels secteur et pays (§7.7) sont construits à
   partir d'ordres de grandeur publics (Gainsight, OpenView, Zuora ; EUR-Lex, IAPP) et déclarés
   comme simulés. Chaque colonne ajoutée est une fonction du secteur ou du pays : elle n'apporte
   aucune information nouvelle. Leur remplacement par une source réelle suit les fiches de
   gouvernance `FICHES_SOURCES` (`churn_saas.features.enrichissement`).

3. **Pas de modèle de survie** — le modèle prédit SI le client résilie, pas QUAND.
   La priorisation par urgence temporelle nécessite un modèle de survie complémentaire.

4. **Extrapolation hors distribution** — le modèle peut se dégrader sur des segments non
   représentés (ex. : comptes au-delà du maximum observé de 898 sièges, secteurs absents des
   données).

5. **Calibration et seuil de vigilance** — la correction d'intercept vaut pour une population
   dont le taux de churn reste proche de celui de l'entraînement. Si la prévalence change
   (dérive de la cible), les probabilités se décalent, et avec elles le seuil de vigilance et
   les montants en euros de §12 : contrôler le recall à chaque vague de renouvellements et
   recalibrer au réentraînement (§13.8). Elle est propre à la régression logistique ; un autre
   champion devrait faire établir sa propre calibration (contrôle automatique en §9.12).

6. **`intensite_support` liée à l'ancienneté** — ce ratio (tickets sur 90 jours / ancienneté)
   mêle deux échelles de temps : sa corrélation de rang avec l'ancienneté vaut −0,42, contre
   0,00 pour les tickets bruts (§7.5). Son importance se lit à côté de celle de l'ancienneté.

7. **Transfert de famille** — la CLI et le flow de réentraînement reprennent la famille et
   les hyperparamètres du champion (§9.11) ; ils ne relancent pas la comparaison de §9.3.
   Un changement de famille passe par une nouvelle exécution du notebook.

---

## 8. Considérations éthiques et réglementaires

- **RGPD** — le score porte sur des entreprises (personnes morales) : le RGPD ne s'applique
  qu'aux contacts opérationnels du CRM, hors du modèle, sur la base de l'intérêt légitime
  (art. 6.1.f). L'article 22 est très probablement hors champ ; le CSM reste décisionnaire par
  choix de conception (§4.1). AIPD non requise (revue DPO, §4.6.3).
- **AI Act** — système d'IA à risque minimal : ciblage commercial B2B, hors annexe III (§4.2).
- **Boucle de rétroaction** — les comptes contactés et sauvés apparaissent « non résiliés » au
  réentraînement suivant : le modèle sous-estimerait leur risque. Parade : groupe témoin
  d'environ 10 % laissé sans action, soumis à la validation de la Direction (§4.5, risque R04).
- **Prophétie auto-réalisatrice** — des petits comptes signalés mais non appelés faute de
  capacité partent, confirmant le score ; priorisation par valeur attendue et quota minimal
  proposé (risque R02).
- **Transparence** — les clients ne sont pas informés individuellement du score ; la mention
  aux CGU pour les contacts CRM est une réserve ouverte du DPO (§4.6.3). L'équipe CS est formée
  à utiliser le score comme outil, pas comme verdict.

---

## 9. Maintenance et cycle de vie

| Déclencheur | Action |
|-------------|--------|
| Trimestriel (calendaire) | Réentraînement lancé à l'issue du comité trimestriel (`make flow`), même famille et mêmes hyperparamètres que le champion |
| Gate de promotion | Challenger promu automatiquement si PR-AUC ≥ 0,65 et sans régression de plus de 0,01 face au champion (`decision_promotion`, §10.4) ; aucune promotion possible sans elle |
| PSI ≥ 0,20 sur au moins une variable surveillée (contrôle hebdomadaire) | Comité ad hoc sous 72 h : réentraîner, maintenir ou geler |
| PR-AUC d'une cohorte de renouvellements < 0,65 | Comité ad hoc : réentraînement + audit données |
| PR-AUC d'une cohorte de renouvellements < 0,60 | Comité ad hoc + comparaison avec le champion précédent sur la même cohorte ; retour arrière s'il fait mieux |
| Part d'`ALERTE_ROUGE` écartée de plus de 10 points après une promotion | Retour au champion précédent |
| Âge > 180 jours | Revue au comité trimestriel |
| Incident production | Retour arrière par `churn-saas rollback <horodatage>` : modèle et méta archivés, alias `@production` replacé (voir `docs/RUNBOOK.md` §4.1) |

**Contacts.** Pour toute question sur ce model card, contacter l'équipe Data Science
du projet CISIA. Pour les questions éthiques ou réglementaires, contacter le DPO.
