# %% [markdown]
# ## 0. Page de garde
#
# | Champ | Valeur |
# |---|---|
# | Titre du projet | **Détection de résiliation client SaaS B2B** — anticiper le départ des comptes clients pour cibler les actions de rétention |
# | Certification visée | **CISIA** — Concevoir et implémenter une solution d'intelligence artificielle |
# | Nom et prénom | Laurent Pottier |
# | Date de remise | 05/10/2026 |
# | Version du notebook | 2.0.0 |
# | Code source public | <https://github.com/laurent-git-dev/churn_saas_release> |

# %% [markdown]
# ### Environnement d'exécution
#
# Le notebook s'exécute de bout en bout : les versions ci-dessous sont relevées à l'exécution, et
# couvrent chaque étape du cycle de vie, des données brutes à la supervision en production.

# %%
import hashlib
import os
import subprocess
import sys
import warnings
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd
from IPython.display import display
from loguru import logger

from churn_saas import config
from churn_saas.format_fr import configurer_pandas, entier
from churn_saas.referentiel import grille_competences

# Tous les tableaux du notebook affichent leurs nombres au format français (« 0,28 », « 5 035 »)
configurer_pandas()

# Journal limité aux avertissements et erreurs : les messages DEBUG et INFO des modules
# (plusieurs milliers) noieraient les résultats. Ils restent actifs hors notebook (CLI, API, tests).
logger.remove()
logger.add(sys.stderr, level="WARNING")
# Les processus parallèles (n_jobs=-1 : validation croisée, forêts, permutation) réimportent
# loguru avec son niveau par défaut, DEBUG, et écrivent directement dans le terminal : ils
# lisent le niveau dans l'environnement, hérité au moment où ils sont lancés
os.environ["LOGURU_LEVEL"] = "WARNING"
# mlflow importe tqdm.auto, qui réclame ipywidgets pour ses barres de progression : on garde les
# barres texte et on masque ce seul avertissement, sans toucher aux autres
warnings.filterwarnings("ignore", message="IProgress not found")


def version_installee(paquet: str) -> str:
    try:
        return version(paquet)
    except PackageNotFoundError:
        return "non installé"


# (étape du cycle de vie, bibliothèque, rôle)
BIBLIOTHEQUES = [
    ("Données (§3, §5)", "pandas", "Chargement, nettoyage, tableaux"),
    ("Données (§3, §5)", "numpy", "Calcul numérique"),
    ("Données (§3, §5)", "pyarrow", "Stockage colonnaire Parquet"),
    ("Données (§3, §5)", "dvc", "Versionnement des jeux de données"),
    ("Analyse exploratoire (§6)", "scipy", "Tests statistiques"),
    ("Analyse exploratoire (§6)", "matplotlib", "Figures"),
    ("Modélisation (§7 à §9)", "scikit-learn", "Pipelines, modèles, validation croisée"),
    ("Modélisation (§7 à §9)", "imbalanced-learn", "Rééquilibrage des classes"),
    ("Modélisation (§7 à §9)", "lightgbm", "Famille concurrente optimisée (boosting de gradient)"),
    ("Modélisation (§7 à §9)", "optuna", "Optimisation des hyperparamètres"),
    ("Suivi et explicabilité (§9, §12)", "mlflow", "Suivi des expériences, registre de modèles"),
    ("Suivi et explicabilité (§9, §12)", "skops", "Sérialisation sûre du modèle"),
    ("Suivi et explicabilité (§9, §12)", "shap", "Explication des prédictions"),
    ("Suivi et explicabilité (§9, §12)", "codecarbon", "Mesure de l'empreinte carbone"),
    ("Mise en exploitation (§10, §11)", "fastapi", "API de prédiction"),
    ("Mise en exploitation (§10, §11)", "pydantic", "Validation des requêtes"),
    ("Mise en exploitation (§10, §11)", "uvicorn", "Serveur de l'API"),
    ("Mise en exploitation (§10, §11)", "prefect", "Orchestration du scoring et du réentraînement"),
    ("Supervision (§13)", "evidently", "Détection de dérive des données"),
    ("Supervision (§13)", "prometheus-fastapi-instrumentator", "Métriques de l'API"),
    ("Reproductibilité", "jupytext", "Génération du notebook depuis les sources"),
]
# Hash git courant : identifie la version exacte du code exécuté
try:
    git_hash = (
        subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        or "inconnu"
    )
except Exception:
    git_hash = "inconnu"

python = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
env_df = pd.DataFrame(
    [("Environnement", "Python", f"Langage ({sys.platform})", python)]
    + [("Environnement", "Commit git", "Version exacte du code exécuté", git_hash)]
    + [(etape, p, role, version_installee(p)) for etape, p, role in BIBLIOTHEQUES],
    columns=["Étape du cycle de vie", "Bibliothèque", "Rôle", "Version installée"],
)
display(env_df.style.hide(axis="index").set_properties(**{"text-align": "left"}))

# %% [markdown]
# ### Empreinte des données brutes
#
# Pour s'assurer que les résultats ont été obtenus sur exactement les mêmes fichiers, on calcule
# l'empreinte SHA-256 de chacun : une suite de 64 caractères qui change dès qu'un seul octet du
# fichier est modifié. Une relance sur des données identiques doit retrouver les mêmes empreintes.

# %%
FICHIERS_BRUTS = ["churn_saas_complet.csv", "churn_saas_echantillon.csv", "catalogue_plans.csv"]


def empreinte_sha256(chemin: Path) -> str:
    with chemin.open("rb") as flux:
        return hashlib.file_digest(flux, "sha256").hexdigest()


empreintes_df = pd.DataFrame(
    [
        (
            f"data/raw/{nom}",
            entier((config.DONNEES_BRUTES / nom).stat().st_size),
            empreinte_sha256(config.DONNEES_BRUTES / nom),
        )
        for nom in FICHIERS_BRUTS
    ],
    columns=["Fichier", "Taille (octets)", "Empreinte SHA-256"],
)
display(empreintes_df.style.hide(axis="index").set_properties(**{"text-align": "left"}))

# %% [markdown]
# ### Grille de couverture des compétences C1 → C9
#
# À titre indicatif : chaque compétence du référentiel est rattachée aux sections du notebook qui
# en apportent la preuve. La grille est recalculée à chaque exécution à partir des titres réels du
# notebook ; le détail item par item figure en annexe §15.

# %%
display(
    grille_competences()[["Compétence", "Intitulé", "Items", "Section(s)"]]
    .style.hide(axis="index")
    .set_properties(**{"text-align": "left"})
)
