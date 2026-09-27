# churn_saas_release

Détection de résiliation client (churn) SaaS B2B — pipeline ML complet de l'exploration à l'API de
prédiction, avec Python 3.12, FastAPI, scikit-learn, MLflow, Optuna, SHAP, Prefect, DVC et Evidently.

---

## Démarrage rapide

**Prérequis :** Python 3.12, [uv](https://docs.astral.sh/uv/), make

```bash
# Installer les dépendances
make setup

# Lancer la suite de tests
make test

# Régénérer et exécuter le notebook (utilise le cache)
make notebook

# Démarrer l'API de prédiction
make api
# → http://localhost:8000/docs
```

Pour forcer le recalcul de toutes les étapes lourdes (Optuna, SHAP, Evidently) :

```bash
make notebook-full
```

---

## Architecture

La vue d'ensemble de l'architecture (package Python, pipeline de données, API, monitoring) est
documentée dans [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

```
src/churn_saas/       # package Python (data, features, models, API, monitoring, CLI)
notebooks/sections/   # sources Jupytext (16 sections, format « percent »)
notebooks/build_notebook.py  # assemblage et exécution du notebook
tests/                # suite pytest (qualité, anti-fuite, drift, sécurité API)
flows/                # flow Prefect de réentraînement
monitoring/           # configuration drift + Prometheus/Grafana
```

---

## Documentation

| Document | Contenu |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Architecture système et décisions de conception |
| [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) | Fiche modèle : performances, biais, limites d'usage |
| [`docs/DATASHEET.md`](docs/DATASHEET.md) | Fiche dataset : provenance, schéma, statistiques |
| [`docs/DICTIONNAIRE_DONNEES.md`](docs/DICTIONNAIRE_DONNEES.md) | Dictionnaire des variables |
| [`docs/RUNBOOK.md`](docs/RUNBOOK.md) | Déploiement, exploitation, procédures d'urgence |
| [`docs/RISK_REGISTER.md`](docs/RISK_REGISTER.md) | Registre des risques techniques et éthiques |

---

## Données

Les données brutes ne sont pas distribuées dans ce dépôt. Le schéma complet et les statistiques
descriptives sont disponibles dans [`docs/DATASHEET.md`](docs/DATASHEET.md) et
[`docs/DICTIONNAIRE_DONNEES.md`](docs/DICTIONNAIRE_DONNEES.md).

Les pointeurs DVC (`data/**/*.dvc`) permettent de versionner la lignée des données sans stocker les
fichiers CSV dans git.

---

## Contribution

Ce dépôt est un **miroir en lecture seule** — les contributions se font via le repo source privé.

Les vérifications obligatoires avant tout commit :

```bash
make check   # ruff + black --check + mypy + pytest
```

Règles : ruff + black (ligne 100), mypy sur `src/`, pas de `print()` dans `src/` (loguru),
Conventional Commits, toutes les variables stochastiques reçoivent `config.RANDOM_SEED`.
