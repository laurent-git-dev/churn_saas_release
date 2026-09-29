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
# Chaque ligne reprend l'intitulé officiel de la compétence du référentiel CISIA, les
# sections du notebook qui la démontrent et les principales preuves. Cette table est le
# fil conducteur du jury ; le détail **item par item** du référentiel figure en §15.4.
#
# | Compétence | Intitulé officiel | Section(s) | Preuves principales |
# |---|---|---|---|
# | **C1** | Identifier un jeu de données répondant aux besoins métiers | §2, §3 | Besoins Customer Success et 3 cas d'usage (§2.1–2.2), dictionnaire avec pertinence (§3.2), existence/accès vérifiés par code (§3.1), plan B (§3.4) |
# | **C2** | Identifier les risques éthiques et sociétaux | §4 | Chartes UE + FR (§4.3), biais TPR/FPR par sous-groupe (§4.4), dilemmes arbitrés (§4.5), registre des risques, note commanditaire et revue DPO (§4.6) |
# | **C3** | Préparer les données | §3, §5, §6, §7 | Stockage et cycle de vie soumis aux parties prenantes (§3.3–3.6), renommage (§5.9), doublons, types, dates, valeurs impossibles (§5.4–5.7), leurres (§6.10), pipeline anti-fuite (§7) |
# | **C4** | Choisir un modèle IA | §8 (+ §9, §12) | Type de résultat (§8.1), cibles a priori (§8.2), contraintes opérationnelles (§8.3), éco-conception (§8.4, §9.8–9.10), build vs buy (§8.5), familles (§8.6), protocole (§8.7–8.8), analyse ROC (§12.2), latence (§9.13, §12.3) |
# | **C5** | Entraîner le modèle | §9 (+ §7.5) | Comparatif des modèles (§9.3), Optuna et hyperparamètres (§9.7), transfert de connaissances (§9.11), réentraînement final (§9.12), feature engineering (§7.5) |
# | **C6** | Implémenter le modèle | §10 | Contrat d'API (§10.2), MLflow et versioning 4 axes (§10.4–10.5), CI + job CD (§10.6), Docker (§10.7), contrat d'échange CRM (§10.8) |
# | **C7** | Architecture cible | §11 | Compte-rendu d'entretien DSI/RSSI/DPO/CS (§11.4), 3 scénarios coût × complexité × souveraineté (§11.6), recommandation (§11.7) |
# | **C8** | Mesurer la performance et les impacts | §12 (+ §8.2, §11.8) | Cibles et SLO/SLI (§8.2, §11.8), métriques techniques, KPI métier, ROI et carbone (§12.2, §12.12–12.13), note de restitution et table de décision (§12.15) |
# | **C9** | Amélioration continue | §13 (+ §2.5) | Gate qualité en CI (§13.2), PSI/KS et Evidently (§13.3–13.4), robustesse (§13.5), obsolescence (§13.6), périodicité décidée au cadrage (§2.5, §13.11) |

# %% [markdown]
# > ### 📋 Journal de bord — Page de garde
# >
# > **Décisions retenues** — Six champs obligatoires générés par code (auteur, date,
# > version git, environnement) pour garantir leur mise à jour automatique à chaque
# > régénération. Grille de couverture C1→C9 reprenant mot pour mot les intitulés du
# > référentiel CISIA, avec renvoi aux sous-sections qui portent chaque preuve.
# > L'auteur est renseigné directement : Laurent Pottier.
# >
# > **Alternatives écartées** — Champs renseignés à la main : invalidés à chaque
# > régénération (`make notebook`) et source d'erreurs humaines.
# >
# > **Difficultés rencontrées** — La première version de la grille utilisait des intitulés
# > courts non officiels, décalés de C1 à C7 par rapport au référentiel (ex. C4 présentée
# > comme « Éco-conception » au lieu de « Choisir un modèle IA ») : elle renvoyait donc le
# > jury vers de mauvaises sections. Corrigé en réalignant la grille sur les intitulés et
# > les items du référentiel.
# >
# > **Impact sur la suite** — La grille de couverture sert de fil conducteur au jury ;
# > §15.4 la détaille item par item, avec la sous-section de preuve de chaque item.
# >
# > **Temps passé** — < 30 min.
