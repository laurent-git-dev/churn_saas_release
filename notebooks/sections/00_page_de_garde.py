# %% [markdown]
# # Détection de résiliation client SaaS B2B
#
# **Certification CISIA** — Concevoir et implémenter une solution d'intelligence artificielle
#
# Livrable d'examen : notebook Jupyter exécutable, lisible sans explication orale.
#
# Code source public : <https://github.com/laurent-git-dev/churn_saas_release>

# %%
import datetime
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version

import pandas as pd
from IPython.display import display

# Récupération du hash git courant — identifie la version exacte du code soumis
try:
    git_hash = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
    ).stdout.strip() or "inconnu"
except Exception:
    git_hash = "inconnu"


def version_safe(paquet: str) -> str:
    try:
        return version(paquet)
    except PackageNotFoundError:
        return "non installé"


# ---------------------------------------------------------------------------
# 6 champs obligatoires de la page de garde (énoncé §0)
# ---------------------------------------------------------------------------
champs_df = pd.DataFrame(
    {
        "Champ": [
            "Titre du projet",
            "Certification visée",
            "Auteur",
            "Date de remise",
            "Version du notebook",
            "Environnement d'exécution",
            "Code source public",
        ],
        "Valeur": [
            "Détection de résiliation client SaaS B2B",
            "CISIA — Concevoir et implémenter une solution d'intelligence artificielle",
            "Laurent Pottier",
            datetime.date.today().isoformat(),
            f"1.0.0 (commit {git_hash})",
            f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            f" · {sys.platform}",
            "https://github.com/laurent-git-dev/churn_saas_release",
        ],
    }
)
display(champs_df.style.hide(axis="index"))

# ---------------------------------------------------------------------------
# Versions des bibliothèques principales
# ---------------------------------------------------------------------------
paquets_principaux = [
    "pandas",
    "numpy",
    "scikit-learn",
    "scipy",
    "matplotlib",
    "seaborn",
    "xgboost",
    "mlflow",
    "jupytext",
    "nbclient",
    "optuna",
    "shap",
    "imbalanced-learn",
    "evidently",
]
env_df = pd.DataFrame(
    {
        "Bibliothèque": paquets_principaux,
        "Version installée": [version_safe(p) for p in paquets_principaux],
    }
)
display(env_df.style.hide(axis="index"))

# %% [markdown]
# **Ce qu'il faut retenir.**
# La page de garde certifie la traçabilité complète du livrable : auteur, date,
# version du code (hash git) et environnement d'exécution. Le jury peut reproduire
# exactement les résultats en installant les versions listées ci-dessus.

# %% [markdown]
# ## Grille de couverture des compétences C1 → C9
#
# Chaque ligne indique quelle section du notebook démontre la compétence
# et quel artefact en constitue la preuve. Cette table est le fil conducteur
# du jury lors de l'évaluation.
#
# | Compétence | Intitulé court | Section(s) | Artefacts de preuve |
# |---|---|---|---|
# | **C1** | Cadrage du problème IA | §2, §4, §8 | Cas d'usage, valeur attendue, colonnes interdites, cibles de performance |
# | **C2** | Données et gouvernance | §3, §5, §6 | Datasheet, contrôle qualité SHA-256, analyse EDA, détection des leurres |
# | **C3** | Préparation des données | §7 | Pipeline ColumnTransformer, anti-fuite, `test_no_leakage.py`, jeu gold |
# | **C4** | Éco-conception | §4, §9 | Tableau émissions CO₂ (CodeCarbon), arbitrage performance/carbone |
# | **C5** | Choix et entraînement du modèle | §8, §9 | Comparatif PR-AUC 5 modèles, Optuna, gestion déséquilibre, note transfert learning |
# | **C6** | Mise en exploitation | §10, §11 | TestClient (5 codes HTTP), gate MLflow, versioning 4 axes, CI/CD, Docker multi-stage |
# | **C7** | Documentation et communication | §10, §14, §15 | Contrat API, table de décision seuil→action, annexes, glossaire |
# | **C8** | Mesure de performance et impacts | §12 | PR-AUC/ROC-AUC/Brier OOF, seuil économique τ*, SHAP, verdict leurres 3 preuves, KPI |
# | **C9** | Amélioration continue | §13 | Gate qualité CI/CD, PSI/KS, Evidently, robustesse, playbook réentraînement |

# %% [markdown]
# > ### 📋 Journal de bord — Page de garde
# >
# > **Décisions retenues** — Six champs obligatoires générés par code (auteur, date,
# > version git, environnement) pour garantir leur mise à jour automatique à chaque
# > régénération. Grille de couverture C1→C9 alignée sur la nomenclature officielle CISIA
# > (réconciliée avec §15.4). L'auteur est renseigné directement : Laurent Pottier.
# >
# > **Alternatives écartées** — Champs renseignés à la main : invalidés à chaque
# > régénération (`make notebook`) et source d'erreurs humaines.
# >
# > **Difficultés rencontrées** — Aucune.
# >
# > **Impact sur la suite** — La grille de couverture sert de fil conducteur au jury ;
# > elle est reproduite en §15.4 avec le détail complet des artefacts de preuve par compétence.
# >
# > **Temps passé** — < 30 min.
