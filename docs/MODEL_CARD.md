# Model Card — Modèle de détection de churn SaaS B2B

> Format inspiré de Mitchell et al. (2019) *"Model Cards for Model Reporting"*,
> adapté au contexte CISIA et aux recommandations CNIL sur la transparence des
> systèmes d'IA.

---

## 1. Informations générales

| Champ | Valeur |
|-------|--------|
| **Nom du modèle** | `churn_saas_champion` |
| **Version** | `1.0.0` (tag git v1.0.0) |
| **Date d'entraînement** | Voir `artifacts/models/best_model_meta.json` |
| **Auteur** | Équipe Data Science — projet CISIA |
| **Objectif** | Classification binaire : probabilité de résiliation (churn) d'un compte SaaS B2B à l'horizon de la prochaine échéance contractuelle |
| **Tâche secondaire** | Régression CLV (`valeur_vie_client_eur`) — modèle distinct (`regression_clv.joblib`) |
| **Artefact principal** | `reports/tables/modele_final.joblib` (notebook §9) · `artifacts/models/best_model.pkl` (CLI) |
| **Tracking expériences** | MLflow — `mlruns/` |

---

## 2. Utilisation prévue

### Cas d'usage autorisés

- **Revue hebdomadaire Customer Success** : liste priorisée des comptes à risque
  (score ≥ seuil économique 0.40) pour planifier les interventions proactives.
- **Alerte avant renouvellement** : déclenchement automatique d'un appel CS
  sur les comptes `ALERTE_ROUGE` (score ≥ 0.60) à J-30 de l'échéance.
- **Préparation de renouvellement** : argumentaire de rétention personnalisé
  basé sur les features les plus contributives (SHAP).

### Cas d'usage hors périmètre

- **Décision automatique sans revue humaine** — le score est un outil d'aide
  à la décision, pas un déclencheur automatique de résiliation ou de pénalité.
- **Évaluation individuelle de l'employé CS** — le modèle évalue le comportement
  du compte client, pas la performance de l'équipe CS.
- **Extrapolation à d'autres secteurs** — entraîné sur des données SaaS B2B
  mid-market françaises ; performances non garanties sur des données hors distribution.

---

## 3. Algorithme et architecture

| Paramètre | Valeur |
|-----------|--------|
| **Famille** | Gradient Boosting sur histogrammes (HistGradientBoostingClassifier) |
| **Bibliothèque** | scikit-learn ≥ 1.4 |
| **Pipeline** | Préprocesseur sklearn (OneHotEncoder + imputation médiane) → HGB |
| **Hyperparamètres optimisés** | Optuna (budget 50 trials, TPE sampler) — voir §9 |
| **Gestion du déséquilibre** | `class_weight="balanced"` ou SMOTE (comparaison en §9) |
| **Calibration** | Non appliquée (scores bruts utilisés pour le ranking CS) |
| **Graine** | `config.RANDOM_SEED = 42` (unique dans le projet) |

---

## 4. Données d'entraînement

| Champ | Valeur |
|-------|--------|
| **Source** | `data/raw/churn_saas_complet.csv` — données fournies par l'éditeur SaaS |
| **Volume** | ~5 000 lignes, 29 colonnes brutes → ~42 features après engineering |
| **Période** | Non communiquée dans l'énoncé — supposée < 3 ans |
| **Taux de churn** | ~28 % (1 400 churners sur 5 000 clients uniques, mesuré en §6) — déséquilibre de classes ; plancher PR-AUC d'un modèle aléatoire ≈ 0.28 |
| **Split** | 80 % entraînement / 20 % test (stratifié, seed unique) |
| **Validation** | CV 5 × 3 répétitions sur le train (RepeatedStratifiedKFold) |
| **Anti-fuite** | `config.COLONNES_INTERDITES` exclu ; toute transformation apprise dans les plis |
| **Datasheet** | `docs/DATASHEET.md` |

---

## 5. Métriques de performance

Métriques mesurées sur le **split test 20 % (non vu à l'entraînement)** :

| Métrique | Valeur | Seuil a priori | Statut |
|----------|--------|----------------|--------|
| PR-AUC | voir `best_model_meta.json` | ≥ 0.65 | voir fichier |
| ROC-AUC | voir `best_model_meta.json` | ≥ 0.75 (indicatif) | voir fichier |
| Latence unitaire médiane | < 200 ms | ≤ 200 ms (SLO C4/C8) | ✓ |
| Latence batch (5 000 comptes) | < 5 min | ≤ 5 min (fenêtre maintenance) | ✓ |

> Les valeurs numériques exactes sont produites par le notebook §9 et §12 (code
> exécutable). Toute valeur recopiée ici sans lien au code serait un anti-pattern CISIA.

---

## 6. Évaluation d'équité (biais)

Le modèle a été évalué sur les sous-groupes suivants (§4 — Éthique) :

| Sous-groupe | Métriques mesurées |
|-------------|-------------------|
| `pays` (FR, DE, UK, ES, IT, NL) | TPR, FPR, précision par pays |
| `taille_entreprise` (PME / Mid / Grand) | TPR, FPR |
| `secteur` (SaaS, Finance, Industrie…) | Taux de churn prédit vs réel |

**Résultats.** Aucune disparité significative (écart TPR inter-groupe < 5 pts)
n'a été détectée sur le jeu de données fourni. Cette conclusion est conditionnelle
au volume limité (~5 000 observations) : les groupes minoritaires ont des IC larges.
Une évaluation d'équité robuste nécessiterait un jeu de données 10× plus grand.

---

## 7. Limites connues

1. **Délai d'étiquettes** — la performance réelle n'est mesurable qu'au prochain
   renouvellement contractuel (1-12 mois). Les métriques sur le split test ne
   reflètent pas la performance en production en temps réel (voir §13.7).

2. **Données simulées** — les enrichissements secteur/pays proviennent de
   référentiels simulés (Gainsight/OpenView 2023). En production, les remplacer
   par des données réelles selon les plans B documentés dans `DATASHEET.md`.

3. **Pas de modèle de survie** — le modèle prédit SI le client résilie, pas QUAND.
   La prioritisation par urgence temporelle nécessite un modèle de survie complémentaire.

4. **Extrapolation hors distribution** — le modèle peut dégrader sur des segments
   non représentés (ex. : grands comptes > 500 sièges, secteurs absents des données).

5. **Calibration** — les scores ne sont pas calibrés en probabilités réelles.
   Pour un usage décisionnel probabiliste (ex. : calcul de l'espérance de valeur
   sauvée), une calibration (Platt scaling ou isotonic regression) est recommandée.

---

## 8. Considérations éthiques et réglementaires

- **Base légale RGPD** — traitement fondé sur l'intérêt légitime de l'éditeur
  à maintenir la relation contractuelle (voir `docs/RISK_REGISTER.md`).
- **Boucle de rétroaction** — les comptes identifiés à risque et recontactés
  peuvent modifier leur comportement (prophétie auto-réalisatrice). Surveillé via
  le groupe témoin (§4).
- **Transparence** — les clients ne sont pas informés individuellement du score ;
  l'équipe CS est formée à utiliser le score comme outil, pas comme verdict.
- **Droit d'opposition** — un client qui s'oppose au traitement est retiré du
  périmètre de scoring (procédure décrite dans `docs/RISK_REGISTER.md`).

---

## 9. Maintenance et cycle de vie

| Déclencheur | Action |
|-------------|--------|
| Trimestriel (calendaire) | Réentraînement automatique (`make flow`) |
| PSI ≥ 0.20 (2 features) | Réentraînement déclenché par monitoring |
| PR-AUC < seuil | Réentraînement + audit données |
| Âge > 180 jours | Revue au comité trimestriel |
| Incident production | Rollback vers archive horodatée (voir `docs/RUNBOOK.md`) |

**Contacts.** Pour toute question sur ce model card, contacter l'équipe Data Science
du projet CISIA. Pour les questions éthiques ou réglementaires, contacter le DPO.
