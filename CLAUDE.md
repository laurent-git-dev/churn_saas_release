# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Nature du dépôt

Livrable d'examen pour la certification **CISIA** (« Concevoir et implémenter une solution
d'intelligence artificielle »), cas d'usage *résiliation client SaaS B2B (churn)*. Le livrable noté
est **un notebook Jupyter unique**, lisible sans explication orale, présenté devant un jury qui
remplit une grille de compétences **C1 à C9**.

Conséquence sur toutes les décisions : **la démarche prime sur le score**. Une AUC de 0,82 justifiée
vaut mieux qu'une AUC de 0,95 inexpliquée — et une AUC quasi parfaite est présumée être une fuite de
données jusqu'à preuve du contraire. Chaque choix technique doit être argumenté en texte dans le
notebook, jamais seulement implicite dans le code.

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
- Ne jamais modifier `data/raw/` (données fournies, figées) ni `docs/legacy/` (énoncé et critères).
- Ne jamais committer `data/`, `mlruns/`, `artifacts/`, `reports/figures/`.
  ⚠️ Les jeux de données font partie des **livrables** : ils sont exclus de git mais **doivent être
  présents dans le ZIP final**. Voir `docs/CHECKLIST_LIVRAISON.md`.

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
  sauvegarde automatique dans `reports/figures/`). Les mêmes fichiers alimentent le support de
  soutenance, donc aucune figure refaite à la main.
- **Étapes lourdes** : Optuna, SHAP, CodeCarbon et Evidently passent par
  `churn_saas.cache.charger_ou_calculer()`. Si l'artefact existe dans `reports/tables/`, on le
  charge ; sinon on le calcule et on l'écrit. Cible : notebook complet en moins de 10 min.
- **Journal de bord obligatoire** en fin de chaque grande section, au gabarit ci-dessous.
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

## Gabarit du journal de bord

À reproduire à l'identique en fin de chaque grande section, dans une cellule markdown :

```markdown
> ### 📋 Journal de bord — <nom de la section>
>
> **Décisions retenues** — …
> **Alternatives écartées** — … *(et pourquoi)*
> **Difficultés rencontrées** — … *(et comment résolues)*
> **Impact sur la suite** — …
> **Temps passé** — …
```

## Gabarit d'une section

```python
# %% [markdown]
# ## <N>. <Titre exact du plan imposé>
#
# <Intention de la section en 2-3 phrases : ce qu'on cherche, pourquoi c'est là.>

# %%
# <code — commenté uniquement là où le POURQUOI n'est pas évident>

# %% [markdown]
# **Ce qu'il faut retenir.** <Interprétation du résultat ci-dessus, en français, orientée métier.>

# %% [markdown]
# > ### 📋 Journal de bord — …
```

Deux règles de rédaction : sous **chaque** figure ou tableau, un paragraphe « ce qu'il faut
retenir » ; et toute affirmation chiffrée est produite par du code visible, jamais recopiée à la
main (le jury relancera le notebook).

## À lire à la demande

| Besoin | Fichier |
|---|---|
| Énoncé condensé + les 9 compétences item par item | `docs/CONTEXTE_EPREUVE.md` |
| Programme de travail jour par jour | `docs/PLAN_10_JOURS.md` |
| Prompts prêts à l'emploi, un par section | `docs/PROMPTS.md` |
| **Pièges techniques qui coûtent des points** | `docs/POINTS_DE_VIGILANCE.md` |
| Contrôle final avant envoi du ZIP | `docs/CHECKLIST_LIVRAISON.md` |
| Énoncé et critères d'origine (figés) | `docs/legacy/context/` |

Ne pas relire `docs/legacy/context/*.txt` (~42 Ko) : `docs/CONTEXTE_EPREUVE.md` en est le condensé
fidèle et suffit dans la quasi-totalité des cas.
