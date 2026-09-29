# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Contexte du projet

Démonstration de détection de résiliation client (churn) SaaS B2B.
Stack : Python 3.12 (uv), FastAPI, scikit-learn, MLflow, Optuna, SHAP, Prefect, DVC, Evidently.
Lire `docs/ARCHITECTURE.md` pour la vue d'ensemble, `docs/RUNBOOK.md` pour déployer.

## Le notebook est un artefact généré

La **source de vérité** est `notebooks/sections/NN_*.py` (format jupytext « percent »).
`notebooks/build_notebook.py` les concatène et exécute le tout vers
`notebooks/churn_saas_certification.ipynb`.

- **Ne jamais lire le `.ipynb`** — plusieurs Mo de JSON avec images en base64, il saturerait le
  contexte pour rien.
- **Ne jamais éditer le `.ipynb`** — toute modification serait écrasée à la régénération.
- Une section = un fichier = un prompt. On ne touche pas deux sections dans la même requête.

## Interdits durs

- Ne jamais lire `data/raw/*.csv` directement : passer par `churn_saas.data.quality.profil_compact()`
  qui renvoie un résumé dense. Un `df.head()` brut dans le contexte coûte cher et n'apprend rien.
- Ne jamais committer `data/`, `mlruns/`, `artifacts/`, `reports/figures/`.

## Invariants

- **Français partout** : markdown, commentaires, identifiants, titres et légendes de figures,
  messages de log, messages de commit.
- **Une seule graine** : `config.RANDOM_SEED`. Tout ce qui est stochastique la reçoit.
- **Aucun chemin en dur** : tous les chemins viennent de `src/churn_saas/config.py`.
- **Anti-fuite** : `config.COLONNES_INTERDITES` est la seule autorité. Toute transformation apprise
  (imputation, encodage, standardisation, agrégat de groupe) passe par un `Pipeline` scikit-learn
  fitté **à l'intérieur** de chaque pli de validation croisée — jamais sur le jeu complet.
  `tests/test_no_leakage.py` doit rester vert.
- **Figures** : exclusivement via `churn_saas.viz` (style unique, numérotation, légende,
  sauvegarde automatique dans `reports/figures/`).
- **Étapes lourdes** : Optuna, SHAP, CodeCarbon et Evidently passent par
  `churn_saas.cache.charger_ou_calculer()`. Si l'artefact existe dans `reports/tables/`, on le
  charge ; sinon on le calcule et on l'écrit. Cible : notebook complet en moins de 10 min.
- **ruff + black, ligne 100.** `mypy` sur `src/`. Pas de `print()` dans `src/` (loguru).
- **Conventional Commits**, un commit par fin de journée de travail au minimum.

## Commandes

Gestion de l'environnement avec **uv** (Python 3.12). Toujours `uv run ...`, jamais un `python` nu.

```bash
make setup           # uv sync --all-extras
make notebook        # régénère + exécute le notebook (cache actif)
make notebook-full   # idem en forçant le recalcul de toutes les étapes lourdes
make test            # uv run pytest -q
make lint            # ruff + black --check + mypy
make check           # lint + test + build du notebook — la porte locale, identique à la CI
make api             # uvicorn churn_saas.api.main:app --reload
make drift           # rapport Evidently
make flow            # flow Prefect de réentraînement
```
