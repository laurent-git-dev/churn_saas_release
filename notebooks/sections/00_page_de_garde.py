# %% [markdown]
# # Détection de résiliation client SaaS B2B
#
# **Certification CISIA** — Concevoir et implémenter une solution d'intelligence artificielle
#
# Livrable d'examen : notebook Jupyter exécutable, lisible sans explication orale.

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
        ],
        "Valeur": [
            "Détection de résiliation client SaaS B2B",
            "CISIA — Concevoir et implémenter une solution d'intelligence artificielle",
            "[NOM_PRENOM]",
            datetime.date.today().isoformat(),
            f"1.0.0 (commit {git_hash})",
            f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            f" · {sys.platform}",
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
# | **C1** | Identifier un jeu de données | §2, §3 | Entretien commanditaire simulé, dictionnaire de données, contrôles d'accès, plan B enrichissement |
# | **C2** | Risques éthiques et sociétaux | §4 | Tableau HLEG + CNIL, biais (équité TPR/FPR), dilemmes, registre des risques, fiche DPO |
# | **C3** | Préparer les données | §5, §7 | Rapport qualité, tableau renommage, DATASHEET.md, pipeline anti-fuite, journal de bord |
# | **C4** | Choisir un modèle IA | §8 | Panorama familles, build vs buy, protocole de comparaison, cibles a priori, éco-conception |
# | **C5** | Entraîner le modèle | §9 | Optuna, tableau hyperparamètres, feature engineering, réentraînement sur train+val |
# | **C6** | Implémenter le modèle | §10 | CI/CD, versioning git+DVC+MLflow, contrat d'API, job CD sur tag |
# | **C7** | Architecture cible | §11 | 3 scénarios (VM/conteneur/cloud managé), tableau coûts, CR entretien DSI/RSSI/DPO |
# | **C8** | Mesurer la performance | §8, §12 | ROC/AUC, PR-AUC, KPI métier, ROI, carbone, SLO/SLI, note de restitution |
# | **C9** | Amélioration continue | §13 | Gate qualité CI/CD, drift Evidently (PSI/KS), test robustesse, playbook réentraînement |
