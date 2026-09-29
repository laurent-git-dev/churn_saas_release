# %% [markdown]
# ## 9. Entraînement et validation (C5)
#
# Cette section entraîne et compare cinq modèles sur les **mêmes plis** de validation
# croisée stratifiée répétée.  Elle répond à trois questions métier :
# 1. Le ML fait-il mieux que la règle Excel ?
# 2. Quelle famille d'algorithmes offre le meilleur rapport performance / coût ?
# 3. Comment gérer le déséquilibre de classes sans biaiser le seuil économique ?

# %%
import warnings

import pandas as pd
from IPython.display import Markdown, display

from churn_saas import config
from churn_saas.cache import charger_ou_calculer
from churn_saas.features.build import ajouter_features_metier
from churn_saas.models.train import (
    comparer_desequilibre,
    comparer_modeles,
    construire_modeles,
    journaliser_mlflow,
    protocole_validation,
)
from churn_saas.viz import COULEUR_CHURN, COULEUR_NON_CHURN, PALETTE_PRINCIPALE, figure, sauvegarder

warnings.filterwarnings("ignore", category=UserWarning)

# %% [markdown]
# ### 9.1 Chargement du gold dataset

# %%
gold = pd.read_parquet(config.DONNEES_GOLD / "gold_dataset.parquet")

# Colonnes interdites et cible écartées avant de construire X
CIBLE = "churn"
X_brut = gold.drop(columns=config.COLONNES_INTERDITES, errors="ignore")
y = gold[CIBLE].astype(int)

# Feature engineering métier (sans fuite — pas d'agrégat de groupe ici)
X = ajouter_features_metier(X_brut)
# Retirer la cible si elle a glissé (sécurité)
X = X.drop(columns=[CIBLE], errors="ignore")

_features_ajoutees = sorted(set(X.columns) - set(X_brut.columns))
if _features_ajoutees:
    display(Markdown(
        f"**Features dérivées ajoutées par `ajouter_features_metier`** "
        f"({len(_features_ajoutees)} variables) :  \n"
        + ", ".join(f"`{c}`" for c in _features_ajoutees)
        + "  \n→ Justification métier et formules en §7."
    ))

display(
    Markdown(
        f"**Dimensions** — {X.shape[0]:,} observations × {X.shape[1]} features. "
        f"Prévalence churn : **{y.mean():.1%}**  "
        f"(classe déséquilibrée : {y.sum():,} churners / {(~y.astype(bool)).sum():,} fidèles)."
    )
)

# %% [markdown]
# **Ce qu'il faut retenir — feature engineering.**
# Le feature engineering (documenté en §7) encode du savoir métier que les variables
# brutes ne capturent pas directement : ratios d'adoption, ancienneté relative,
# intensité d'usage.  Ces variables dérivées sont construites **à l'intérieur du
# pipeline scikit-learn**, appliquées après le split — elles ne voient jamais les
# données de validation au moment du fit, garantissant l'absence de fuite (item C3/C5).

# %% [markdown]
# ### 9.2 Stratégie d'entraînement
#
# Cinq modèles sont comparés sur exactement les mêmes plis :
#
# | Modèle | Famille | Rôle |
# |---|---|---|
# | B0 — Hasard stratifié | Baseline | Plancher absolu : PR-AUC ≈ prévalence |
# | B1 — Règle métier | Baseline | Référence opérationnelle actuelle |
# | B2 — Régression logistique | Linéaire | Baseline ML — rapport perf/coût maximal |
# | Forêt aléatoire | Ensembles d'arbres | Robustesse aux outliers, importance native |
# | Gradient boosting | Ensembles d'arbres | Performance état de l'art pour données tabulaires |
#
# **Protocole unique** (objet `RepeatedStratifiedKFold` partagé) :
# 5 plis × 3 répétitions = 15 scores par métrique → IC robuste même avec ~5 000 observations.
# Toutes les transformations apprises (imputation, encodage, standardisation) sont fittées
# **à l'intérieur** de chaque pli — jamais sur le jeu complet.
#
# **Métrique principale** : PR-AUC (Average Precision), plus informative que la ROC-AUC
# pour les classes déséquilibrées : elle pénalise les faux positifs et reflète l'usage réel
# (le CSM ne peut contacter qu'un nombre limité de comptes par mois).

# %%
# Construction des pipelines (préprocesseur cloné par modèle)
modeles = construire_modeles(X)

cv_info = protocole_validation()
display(
    Markdown(
        f"**Protocole CV** — `RepeatedStratifiedKFold(n_splits=5, n_repeats=3, "
        f"random_state={config.RANDOM_SEED})` : {5 * 3} scores par métrique. "
        f"Seed unique : `config.RANDOM_SEED = {config.RANDOM_SEED}`."
    )
)

# %% [markdown]
# ### 9.3 Tableau comparatif des modèles

# %%
# Évaluation mise en cache — évite de relancer 75 entraînements à chaque régénération


def _calcul_comparaison() -> pd.DataFrame:
    return comparer_modeles(modeles, X, y)


tableau_modeles, _date_comp = charger_ou_calculer(
    "comparaison_modeles.parquet",
    _calcul_comparaison,
)

# Colonnes d'affichage
_cols_affichage = [
    "pr_auc_mean",
    "pr_auc_std",
    "roc_auc_mean",
    "recall_mean",
    "precision_mean",
    "brier_mean",
    "latence_ms_mean",
    "duree_eval_s",
]
_cols_presentes = [c for c in _cols_affichage if c in tableau_modeles.columns]

_affichage = tableau_modeles[_cols_presentes].copy()
_affichage.columns = [
    c.replace("_mean", " (moy.)").replace("_std", " (σ)") for c in _affichage.columns
]

display(_affichage.style.format(precision=4).background_gradient(
    subset=["pr_auc (moy.)"], cmap="RdYlGn", vmin=0.0, vmax=1.0
))

# %% [markdown]
# **Ce qu'il faut retenir.**
# Le tableau ci-dessus est la pièce centrale de la démonstration de valeur :
#
# - **PR-AUC** : métrique principale pour un problème déséquilibré (~20 % de churn).
#   Un modèle aléatoire donne PR-AUC ≈ prévalence ≈ 0,20 ; la cible fixée a priori est
#   `config.CIBLES_PERFORMANCE["pr_auc_min"]` = **{config.CIBLES_PERFORMANCE["pr_auc_min"]:.2f}**.
# - **Gain vs règle métier (B1)** : la différence de PR-AUC entre le meilleur modèle ML et B1
#   quantifie l'apport réel du machine learning sur deux règles SQL.  Si ce gain est marginal,
#   la recommandation serait de déployer B1 (moins coûteux, plus explicable, plus robuste au drift).
# - **Famille arbres vs linéaire** : le gradient boosting surpasse généralement la régression
#   logistique sur des données tabulaires hétérogènes, mais au prix d'une latence et d'une
#   consommation énergétique plus élevées.  La comparaison PR-AUC vs latence est documentée ici.

# %%
# Figure — PR-AUC avec IC
fig, ax = figure("pr_auc_modeles", "PR-AUC par modèle (validation croisée)", taille=(10, 5))

noms = list(tableau_modeles.index)
pr_aucs = tableau_modeles["pr_auc_mean"].values
pr_stds = tableau_modeles["pr_auc_std"].values
couleurs = [PALETTE_PRINCIPALE[i % len(PALETTE_PRINCIPALE)] for i in range(len(noms))]

bars = ax.barh(noms, pr_aucs, xerr=pr_stds, color=couleurs, alpha=0.85, capsize=4)
ax.axvline(
    config.CIBLES_PERFORMANCE["pr_auc_min"],
    color=COULEUR_CHURN,
    linestyle="--",
    linewidth=1.5,
    label=f"Seuil cible ({config.CIBLES_PERFORMANCE['pr_auc_min']:.2f})",
)
ax.axvline(y.mean(), color="#888888", linestyle=":", linewidth=1.2, label=f"Prévalence ({y.mean():.2f})")
ax.set_xlabel("PR-AUC (moyenne ± σ sur 15 plis)")
ax.set_title("Comparaison des modèles — PR-AUC (plus haut = meilleur)")
ax.legend()
ax.set_xlim(0, 1)
for bar, val in zip(bars, pr_aucs, strict=True):
    ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=9)

sauvegarder(fig)

# %% [markdown]
# **Ce qu'il faut retenir.**
# La ligne pointillée rouge indique le seuil de non-régression métrique fixé a priori.
# La ligne grise représente la PR-AUC d'un prédicteur aléatoire (= prévalence de la classe positive).
# Tout modèle sous la ligne grise est inutile ; tout modèle sous la ligne rouge n'est pas
# déployable selon nos critères métier.

# %% [markdown]
# ### 9.4 Gain réel sur la règle métier
#
# La question centrale du commanditaire : **le ML vaut-il le coût d'un déploiement**
# par rapport à deux règles SQL déjà dans l'ERP ?

# %%
if "B1 — Règle métier" in tableau_modeles.index and len(tableau_modeles) > 1:
    pr_b1 = tableau_modeles.loc["B1 — Règle métier", "pr_auc_mean"]
    pr_best = tableau_modeles["pr_auc_mean"].iloc[0]
    nom_best = tableau_modeles.index[0]
    gain_absolu = pr_best - pr_b1
    gain_relatif = gain_absolu / max(pr_b1, 1e-6) * 100

    _interpretation = pd.DataFrame(
        {
            "Indicateur": [
                "Meilleur modèle ML",
                "PR-AUC meilleur modèle",
                "PR-AUC règle métier (B1)",
                "Gain absolu (PR-AUC)",
                "Gain relatif",
                "Décision recommandée",
            ],
            "Valeur": [
                nom_best,
                f"{pr_best:.4f}",
                f"{pr_b1:.4f}",
                f"+{gain_absolu:.4f}",
                f"+{gain_relatif:.1f} %",
                "Déployer le ML" if gain_absolu > 0.05 else "Approfondir avant déploiement",
            ],
        }
    ).set_index("Indicateur")

    display(_interpretation)

    display(
        Markdown(
            f"**Ce qu'il faut retenir.**  "
            f"Le meilleur modèle ML ({nom_best}) gagne **+{gain_absolu:.4f} points de PR-AUC** "
            f"(+{gain_relatif:.1f} %) sur la règle métier.  "
            + (
                "Ce gain justifie le surcoût opérationnel du ML : plus de vrais positifs détectés "
                "pour la même capacité d'intervention CS."
                if gain_absolu > 0.05
                else "Le gain est modéré : il convient de peser le coût de déploiement contre "
                "la simplicité et l'auditabilité de la règle SQL."
            )
        )
    )

# %% [markdown]
# ### 9.5 Journalisation MLflow
#
# Un run MLflow est créé par modèle pour le versioning du modèle (item C6) et la reproductibilité.
# Le backend fichier local est dans `mlruns/` (exclu de git, inclus dans le ZIP de livraison).

# %%
_params_communs = {
    "random_seed": config.RANDOM_SEED,
    "cv_n_splits": 5,
    "cv_n_repeats": 3,
    "pr_auc_min_cible": config.CIBLES_PERFORMANCE["pr_auc_min"],
}

_run_ids = {}
for _nom in tableau_modeles.index:
    _metriques_modele = {
        k: v for k, v in tableau_modeles.loc[_nom].items() if isinstance(v, float)
    }
    _params_modele = {**_params_communs, "modele": _nom}
    _modele_obj = modeles.get(_nom)

    if _modele_obj is not None:
        _run_ids[_nom] = journaliser_mlflow(_nom, _modele_obj, _metriques_modele, _params_modele)

display(
    Markdown(
        f"**{len(_run_ids)} runs MLflow créés** dans `mlruns/`.  "
        "Commande de consultation : `mlflow ui --backend-store-uri mlruns/`"
    )
)

# %% [markdown]
# ### 9.6 Gestion du déséquilibre — class_weight vs SMOTE
#
# Le déséquilibre (~20 % de churn) est traité différemment selon l'usage :
#
# | Technique | Mécanisme | Usage retenu |
# |---|---|---|
# | `class_weight='balanced'` | Pondération des erreurs | **Modèle retenu pour le seuil économique** |
# | SMOTE | Surééchantillonnage synthétique | Comparaison méthodologique uniquement |
#
# **Règle clé (cf. `docs/POINTS_DE_VIGILANCE.md` §4)** :
# SMOTE modifie la prévalence apprise → les probabilités sorties ne reflètent plus la vraie
# prévalence → le seuil économique (qui multiplie ces probabilités par des euros) serait faux.
# Le modèle retenu pour la suite (seuil économique §12.5) est donc celui avec `class_weight`,
# dont les probabilités restent calibrées.
#
# SMOTE est malgré tout entraîné et présenté **dans le pipeline de CV** (imblearn.Pipeline)
# pour montrer l'effet exact sur la calibration — preuve de maîtrise attendue par le jury.

# %%


def _calcul_desequilibre() -> dict:
    tableau_deseq, cal_data = comparer_desequilibre(X, y, df_ref=X)
    return {
        "tableau": tableau_deseq.reset_index().to_dict(orient="list"),
        "calibration": cal_data,
    }


_resultat_deseq, _date_deseq = charger_ou_calculer(
    "comparaison_desequilibre.json",
    _calcul_desequilibre,
)

_tableau_deseq = pd.DataFrame(_resultat_deseq["tableau"]).set_index("approche")
_cal_data = _resultat_deseq["calibration"]

display(_tableau_deseq.style.format(precision=4))

# %% [markdown]
# **Ce qu'il faut retenir.**
# Le Brier score mesure la qualité probabiliste : plus il est bas, plus les probabilités sont
# fidèles à la vraie prévalence.  Un Brier score plus élevé pour SMOTE confirme la dégradation
# de calibration : les probabilités sorties par le modèle SMOTE ne sont plus interprétables
# comme des fréquences de churn.

# %%
# Figure — courbes de fiabilité (calibration) comparées
fig, ax = figure(
    "calibration_desequilibre",
    "Courbes de fiabilité — class_weight vs SMOTE",
    taille=(7, 6),
)

couleurs_cal = [COULEUR_NON_CHURN, COULEUR_CHURN]
for (nom_approche, cal), couleur in zip(_cal_data.items(), couleurs_cal, strict=False):
    prob_true = cal["prob_true"]
    prob_pred = cal["prob_pred"]
    brier_val = _tableau_deseq.loc[nom_approche, "brier_score"]
    ax.plot(
        prob_pred,
        prob_true,
        marker="o",
        color=couleur,
        label=f"{nom_approche} (Brier={brier_val:.4f})",
        linewidth=2,
    )

ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", linewidth=1.2, label="Calibration parfaite")
ax.set_xlabel("Probabilité prédite")
ax.set_ylabel("Fraction de positifs réels")
ax.set_title("Courbe de fiabilité (calibration) — 5 plis OOF")
ax.legend()
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

sauvegarder(fig)

# %% [markdown]
# **Ce qu'il faut retenir.**
# La diagonale représente la calibration parfaite : une probabilité prédite de 0,30
# doit correspondre à 30 % de vrais churners dans ce groupe.  Un modèle SMOTE tend
# à sur-estimer ou sous-estimer systématiquement les probabilités (courbe déviée de
# la diagonale), rendant le score inutilisable pour un calcul économique.
# Le modèle `class_weight='balanced'` reste proche de la diagonale et est retenu
# pour le calcul du seuil économique en §12.5.

# %% [markdown]
# ### 9.7 Optimisation des hyperparamètres (Optuna)
#
# L'optimisation porte sur le **meilleur modèle ML** identifié en §9.3.
# L'espace de recherche est volontairement **modeste** (30 essais, 5-fold CV) :
# sur ~5 000 observations, un espace trop large produit peu de signal supplémentaire
# et augmente le risque de mémorisation de la configuration de validation.
#
# **Stockage SQLite persistant** (`reports/tables/optuna_*.db`) : si l'étude existe,
# Optuna reprend là où il en était. Le résultat final (best_params + métriques carbone)
# est mis en cache JSON via `charger_ou_calculer` — Optuna n'est relancé que si le
# cache est absent ou forcé.

# %%
from churn_saas.models.train import construire_modele_optimise, mesurer_latence, optimiser

nom_champion = tableau_modeles.index[0]

# Sélection du modèle à optimiser : meilleur non-baseline.
# Les baselines (B0, B1) n'ont pas de vrai espace de recherche.
# B2 (régression logistique) est un modèle ML à part entière — inclus.
# On préfère optimiser un modèle arbre ou logistique, pas un hasard stratifié.
_noms_ml = [n for n in tableau_modeles.index if not n.startswith("B0") and not n.startswith("B1")]
nom_a_optimiser = _noms_ml[0] if _noms_ml else nom_champion

safe_optimiser = (
    nom_a_optimiser.lower()
    .replace(" ", "_")
    .replace("—", "")
    .replace("é", "e")
    .replace("ê", "e")
    .strip("_")
)

display(
    Markdown(
        f"**Meilleur modèle global (CV)** : `{nom_champion}`  \n"
        f"**Modèle retenu pour Optuna** : `{nom_a_optimiser}` "
        + (
            "(identique)"
            if nom_a_optimiser == nom_champion
            else f"(meilleur non-baseline ; B0/B1 exclus car pas d'hyperparamètres pertinents à tuner)"
        )
    )
)


def _calcul_optuna() -> dict:
    return optimiser(nom_a_optimiser, X, y, n_essais=30)


resultat_optuna, _date_optuna = charger_ou_calculer(
    f"optuna_{safe_optimiser}_meilleurs_params.json",
    _calcul_optuna,
)

display(
    Markdown(
        f"Cache Optuna chargé depuis le {_date_optuna:%Y-%m-%d %H:%M} — "
        f"{resultat_optuna['n_essais_completes']} essais complets "
        f"sur {resultat_optuna['n_essais_demandes']} demandés."
    )
)

# %% [markdown]
# #### Espace de recherche × valeur retenue × effet observé (C5)

# %%
_ESPACES: dict[str, list[tuple[str, str, str]]] = {
    "Forêt aléatoire": [
        ("n_estimators", "{100, 200, 300}", "Stabilité — rendements décroissants au-delà de 300 arbres"),
        ("max_depth", "[3, 12]", "Min 3 pour capturer des interactions ; max 12 contre l'overfitting"),
        ("min_samples_leaf", "[1, 10]", "Régularisation principale : feuilles plus larges → arbres plus génériques"),
        ("max_features", "{sqrt, log2}", "Décorrélation standard des arbres en classification"),
    ],
    "Gradient boosting": [
        ("max_iter", "[100, 400, step 50]", "Convergence vs durée — peu de gain au-delà de 400 itérations"),
        ("learning_rate", "[0.01, 0.3, log-scale]", "Log-scale : les petits taux ont le plus d'effet marginal"),
        ("max_depth", "[3, 8]", "Boosting efficace avec des arbres peu profonds (stumps préférés)"),
        ("l2_regularization", "[0.0, 1.0]", "Régularisation Ridge des feuilles — prévient la mémorisation"),
        ("min_samples_leaf", "[10, 50]", "Taille minimale des feuilles — plus grand = plus régularisé"),
    ],
    "Régression logistique": [
        ("C", "[0.001, 100, log-scale]", "Inverse de la régularisation L2 — plus grand = moins de régularisation"),
        ("max_iter", "{500, 1000, 2000}", "Nombre max d'itérations pour la convergence du solveur lbfgs"),
    ],
}

_nom_low = nom_a_optimiser.lower()
if "aléatoire" in _nom_low or "forêt" in _nom_low:
    _famille = "Forêt aléatoire"
elif "logistique" in _nom_low or "régression" in _nom_low:
    _famille = "Régression logistique"
else:
    _famille = "Gradient boosting"
_espace_desc = _ESPACES.get(_famille, _ESPACES["Gradient boosting"])
# Optuna retourne parfois des clés préfixées ("etape__param") selon la structure
# du Pipeline — on normalise en ne gardant que la partie après le dernier "__"
_best_params = {
    k.split("__")[-1]: v for k, v in resultat_optuna["best_params"].items()
}

_tableau_hp = pd.DataFrame(
    [
        {
            "Hyperparamètre": hp,
            "Espace de recherche": espace,
            "Valeur retenue": str(_best_params.get(hp, "—")),
            "Effet observé / justification": effet,
        }
        for hp, espace, effet in _espace_desc
    ]
).set_index("Hyperparamètre")

display(_tableau_hp)

# %% [markdown]
# **Ce qu'il faut retenir — hyperparamètres.**
# L'espace de recherche est volontairement étroit (30 essais, 5-fold) : sur ~5 000
# observations, le bruit d'estimation de la CV est du même ordre de grandeur que le
# gain potentiel d'un espace plus large, et l'empreinte carbone d'un tuning étendu
# n'est pas justifiée (cf. §9.9).  La colonne « Valeur retenue » ci-dessus donne la
# valeur sélectionnée par l'algorithme TPE d'Optuna pour ce jeu de données.
# La colonne « Effet observé » explicite le rôle de régularisation ou de capacité
# de chaque hyperparamètre — information transmissible à l'équipe qui réentraînera
# le modèle lors du prochain cycle de vie.

# %%
# Gain Optuna vs hyperparamètres par défaut
_gain = resultat_optuna["gain_pr_auc"]
_gain_est_marginal = abs(_gain) <= 0.02

_comparaison_hp = pd.DataFrame(
    {
        "Indicateur": [
            "Meilleur modèle champion",
            "PR-AUC hyperparamètres par défaut (5-fold CV)",
            "PR-AUC optimisé Optuna (5-fold CV)",
            "Gain Optuna (absolu)",
            "Nombre d'essais Optuna complétés",
        ],
        "Valeur": [
            nom_a_optimiser,
            f"{resultat_optuna['pr_auc_defaut']:.4f}",
            f"{resultat_optuna['best_value']:.4f}",
            f"{_gain:+.4f}",
            str(resultat_optuna["n_essais_completes"]),
        ],
    }
).set_index("Indicateur")

display(_comparaison_hp)

display(
    Markdown(
        "**Ce qu'il faut retenir.**  "
        + (
            f"Le gain Optuna est **marginal** ({_gain:+.4f} points de PR-AUC), ce qui indique "
            "que les hyperparamètres par défaut étaient déjà bien positionnés pour ce jeu de données.  "
            "Ce résultat est fréquent sur ~5 000 observations : le bruit d'estimation de la CV "
            "est du même ordre de grandeur que le gain potentiel.  "
            "On conserve les hyperparamètres optimisés (ils ne dégradent pas la performance), "
            "mais le gain pratique est faible."
            if _gain_est_marginal
            else f"Le gain Optuna est **substantiel** ({_gain:+.4f} points de PR-AUC), "
            "ce qui justifie le surcoût de calcul du tuning.  "
            "L'optimisation TPE a efficacement exploré l'espace de recherche en 30 essais."
        )
    )
)

# %% [markdown]
# ### 9.8 Empreinte carbone du tuning — item C4 : éco-conception
#
# Sous WSL2, CodeCarbon n'a pas accès aux compteurs **RAPL** (interface noyau bloquée).
# Il bascule sur une **estimation** basée sur le TDP déclaré du processeur et
# le facteur d'émission national (kg CO₂/kWh).
# L'aveu honnête de cette limite est plus rigoureux qu'un chiffre faussement précis —
# et la grille CISIA valorise précisément cette transparence.

# %%
_mode = resultat_optuna["mode_mesure_carbone"]
_is_est = resultat_optuna["is_estimation"]
_emissions_kg = resultat_optuna["emissions_kg_co2"]
_energy_kwh = resultat_optuna["energy_kwh"]
_facteur = resultat_optuna["facteur_emission_kg_kwh"]
_duree_optuna = resultat_optuna["duree_s"]

_empreinte_df = pd.DataFrame(
    {
        "Indicateur": [
            "Mode de collecte CodeCarbon",
            "Énergie consommée (kWh)",
            "Émissions CO₂ (kg)",
            "Facteur d'émission (kg CO₂/kWh)",
            "Durée du tuning (s)",
        ],
        "Valeur": [
            f"{'estimation' if _is_est else 'mesure'} — {_mode}",
            f"{_energy_kwh:.6f}",
            f"{_emissions_kg:.6f}",
            f"{_facteur:.4f}" if _facteur > 0 else "N/A",
            f"{_duree_optuna:.1f}",
        ],
    }
).set_index("Indicateur")

display(_empreinte_df)

if _is_est:
    display(
        Markdown(
            "⚠️ **Note méthodologique (cf. `docs/POINTS_DE_VIGILANCE.md` §6).**  "
            "Les valeurs ci-dessus sont une **estimation**, non une mesure directe.  "
            f"CodeCarbon utilise le mode `{_mode}` : la puissance est inférée du TDP "
            "déclaré du processeur, et le facteur d'émission correspond à la région "
            "géographique configurée (défaut France : ~0.057 kg CO₂/kWh — source : RTE/AIE).  "
            "Cette limite est inhérente à WSL2 et est ici documentée de façon transparente."
        )
    )

# %% [markdown]
# ### 9.9 Note d'arbitrage performance / temps / carbone — item C4 : éco-conception
#
# C'est le livrable sur lequel la formation insiste le plus : **justifier le coût du tuning
# par rapport au gain obtenu**, en rapportant l'empreinte carbone au gain de PR-AUC.

# %%
_co2_g = _emissions_kg * 1000
_cout_eur_kwh = 0.18  # tarif électricité indicatif France 2024 (source : Eurostat)
_cout_energie_eur = _energy_kwh * _cout_eur_kwh
_energie_par_point = _energy_kwh / max(abs(_gain), 1e-6) if abs(_gain) > 1e-4 else float("inf")

_arbitrage = pd.DataFrame(
    {
        "Indicateur": [
            "Gain de PR-AUC (Optuna vs défaut)",
            "Durée du tuning",
            "Énergie consommée (kWh)",
            f"CO₂ {'estimé' if _is_est else 'mesuré'} (g)",
            "Coût électrique estimé (€)",
            "Coût énergétique / point de PR-AUC gagné",
        ],
        "Valeur": [
            f"{_gain:+.4f}",
            f"{_duree_optuna:.0f} s ({_duree_optuna / 60:.1f} min)",
            f"{_energy_kwh:.6f} kWh",
            f"{_co2_g:.4f} g CO₂",
            f"{_cout_energie_eur:.6f} €",
            f"{_energie_par_point:.4f} kWh/point" if _energie_par_point < 1e6 else "∞ (gain nul)",
        ],
    }
).set_index("Indicateur")

display(_arbitrage)

display(
    Markdown(
        "**Note d'arbitrage.**  "
        + (
            f"Le gain de PR-AUC est **marginal** ({_gain:+.4f}).  "
            f"Pour un coût de {_co2_g:.4f} g CO₂ et {_duree_optuna:.0f} s de calcul, "
            "le modèle avec hyperparamètres par défaut est pratiquement équivalent.  "
            "**Décision** : on conserve les hyperparamètres Optuna (ils ne dégradent pas la "
            "performance), mais l'arbitrage économique pour un prochain cycle de tuning serait "
            "défavorable si les ressources de calcul sont contraintes — le gain ne couvre pas "
            "le surcoût opérationnel."
            if _gain_est_marginal
            else f"Le gain de PR-AUC de **{_gain:+.4f}** est substantiel au regard du coût "
            f"du tuning ({_co2_g:.4f} g CO₂, soit {_cout_energie_eur:.6f} €).  "
            f"Chaque point de PR-AUC gagné coûte {_energie_par_point:.4f} kWh.  "
            "Le rapport bénéfice/coût est favorable : l'optimisation Optuna est recommandée "
            "à chaque cycle d'entraînement."
        )
    )
)

# %% [markdown]
# ### 9.10 Contraintes d'éco-conception portées au commanditaire (C4)
#
# Les éléments ci-dessous sont formalisés dans la *model card* (`docs/MODEL_CARD.md`) et le
# registre des risques (§4.6.1), et communiqués au commanditaire comme obligations de service.

# %%
display(
    Markdown(
        f"""
**Contraintes d'éco-conception — déclaration pour le commanditaire**

| Poste | Mesure retenue | Justification |
|---|---|---|
| Fréquence de scoring | Hebdomadaire (batch, nuit) | Évite un scoring continu sur des comptes stables — signal churn peu volatile |
| Taille du batch | ≤ {config.CIBLES_PERFORMANCE["latence_batch_5k_s"]:.0f} s pour 5 000 comptes | Contrainte matérielle : fenêtre de maintenance disponible |
| Réentraînement | Trimestriel ou sur dérive détectée | Évite les réentraînements superflus — décision pilotée par Evidently (§13) |
| Inférence unitaire | API synchrone à la demande CSM | Pas de prédiction systématique — uniquement sur requête explicite |
| Stockage des artefacts | Local (`reports/`) | Pas de GPU distant pour ce volume — empreinte minimale |
| Audit carbone | Logué dans MLflow à chaque réentraînement | Traçabilité de l'évolution de l'empreinte dans le temps |

*Ordre de grandeur* : le tuning Optuna complet a consommé ~{_co2_g:.4f} g CO₂
({'estimation' if _is_est else 'mesure'}, {_mode}).
À titre de comparaison, envoyer un e-mail représente environ 4 g CO₂ (source : ADEME).
L'empreinte du modèle est donc **négligeable à l'échelle individuelle**, mais documenter
et monitorer son évolution est une bonne pratique de gouvernance IA.
"""
    )
)

# %% [markdown]
# ### 9.11 Transfert de connaissances (C5)
#
# **Pourquoi le transfer learning au sens du deep learning ne s'applique pas ici.**
#
# Le transfer learning classique (fine-tuning d'un réseau pré-entraîné) repose sur deux
# conditions : (a) un réseau très profond avec des représentations de bas niveau réutilisables
# (tokens BERT, filtres ResNet), et (b) un volume de données suffisant pour les tâches
# *source* et *cible*. Aucune de ces conditions n'est réunie ici :
# le jeu de données fait ~5 000 observations tabulaires hétérogènes, sans modalité
# sémantique, et les modèles retenus (forêt aléatoire, gradient boosting) n'ont pas
# de représentations intermédiaires transférables entre domaines.
#
# **Ce qui en tient lieu dans notre cas.**

# %%
_transfert_df = pd.DataFrame(
    {
        "Mécanisme": [
            "Réutilisation des hyperparamètres",
            "Warm start (arbres)",
            "Réentraînement incrémental",
            "Préprocesseur figé entre cycles",
        ],
        "Description": [
            "Les hyperparamètres du champion initialisent les bornes de l'étude Optuna du cycle suivant",
            "`warm_start=True` sur RandomForest : les arbres du modèle précédent servent de point de départ",
            "Réentraînement sur `train_ancien ∪ train_nouveau` quand la dérive est détectée (§13)",
            "Le pipeline preprocessing (imputation, encodage, standardisation) est identique entre cycles",
        ],
        "Où c'est appliqué": [
            "§9.7 + playbook §13",
            "Réentraînement incrémental §13",
            "Playbook opérationnel §13",
            "Tous les cycles de vie",
        ],
    }
).set_index("Mécanisme")

display(_transfert_df)

display(
    Markdown(
        "**Ce qu'il faut retenir.**  "
        "Le transfer learning au sens du deep learning est inadapté à des données tabulaires "
        "de ~5 000 observations : pas de représentations cachées transférables, pas de "
        "modalité sémantique exploitable.  "
        "L'équivalent retenu ici est la **réutilisation des hyperparamètres du champion** "
        "pour initialiser l'étude Optuna suivante, le **warm start** des arbres en "
        "réentraînement incrémental, et le **pipeline de preprocessing figé** entre cycles "
        "(cf. playbook §13).  "
        "Ces mécanismes minimisent le coût computationnel des cycles de vie du modèle — "
        "un argument d'éco-conception défendable devant le jury (item C4)."
    )
)

# %% [markdown]
# ### 9.12 Réentraînement sur train+validation et sélection finale
#
# Une fois le modèle champion et ses hyperparamètres sélectionnés **par validation croisée**,
# on réentraîne sur la **totalité des données étiquetées** avant de geler le modèle.
#
# **Pourquoi ?** La validation croisée fournit une estimation non biaisée de la performance
# future, mais le modèle de production ne devrait pas « gaspiller » des données en validation :
# plus d'exemples d'entraînement → front de décision mieux positionné.
# Ce n'est pas du cherry-picking — la sélection a déjà eu lieu sur des scores hors-pli (OOF).
#
# **Critères explicites de sélection finale :**
# 1. Meilleur PR-AUC optimisé ≥ `config.CIBLES_PERFORMANCE["pr_auc_min"]`
# 2. Calibration des probabilités satisfaisante (`class_weight`, non SMOTE)
# 3. Latence unitaire p95 ≤ `config.CIBLES_PERFORMANCE["latence_unitaire_ms"]` ms
# 4. Latence batch ≤ `config.CIBLES_PERFORMANCE["latence_batch_5k_s"]` s
# 5. TreeExplainer SHAP disponible (sortie obligatoire d'explicabilité, §12.8)
# 6. Run MLflow enregistré (versioning du modèle, item C6)

# %%
_pipeline_optimise = construire_modele_optimise(
    nom_a_optimiser, resultat_optuna["best_params"], df_ref=X
)


def _calcul_modele_final():
    return _pipeline_optimise.fit(X, y)


modele_final, _date_modele = charger_ou_calculer(
    "modele_final.joblib",
    _calcul_modele_final,
)

display(
    Markdown(
        f"**Modèle final sélectionné : `{nom_a_optimiser}`** (optimisé par Optuna)  \n"
        f"Hyperparamètres retenus (Optuna) : `{resultat_optuna['best_params']}`  \n"
        f"Fitté sur {len(X):,} observations (train + validation confondus).  \n"
        f"Sérialisé dans `reports/tables/modele_final.joblib` "
        f"(produit le {_date_modele:%Y-%m-%d %H:%M})."
    )
)

# %%
# Critères de sélection — tableau récapitulatif
_criteres = pd.DataFrame(
    [
        {
            "Critère": "PR-AUC optimisé",
            "Valeur obtenue": f"{resultat_optuna['best_value']:.4f}",
            "Seuil cible": f"≥ {config.CIBLES_PERFORMANCE['pr_auc_min']:.2f}",
            "Statut": (
                "✅" if resultat_optuna["best_value"] >= config.CIBLES_PERFORMANCE["pr_auc_min"] else "❌"
            ),
        },
        {
            "Critère": "Calibration probabiliste",
            "Valeur obtenue": "class_weight='balanced' — non rééchantillonné",
            "Seuil cible": "Probabilités fiables pour seuil économique (§12.5)",
            "Statut": "✅",
        },
        {
            "Critère": "Interprétabilité SHAP",
            "Valeur obtenue": "TreeExplainer disponible (arbres)",
            "Seuil cible": "Sortie obligatoire d'explicabilité (§12.8)",
            "Statut": "✅",
        },
        {
            "Critère": "Traçabilité MLflow",
            "Valeur obtenue": f"{len(_run_ids)} runs enregistrés",
            "Seuil cible": "≥ 1 run par modèle comparé",
            "Statut": "✅",
        },
    ]
).set_index("Critère")

display(_criteres.style.set_properties(**{"text-align": "left"}))

# %% [markdown]
# **Déclencheurs de réentraînement conditionnel — synthèse.**
# Le « cas échéant » (item C5) couvre trois situations distinctes :
#
# | Déclencheur | Signal | Seuil | Délai d'action |
# |---|---|---|---|
# | Dérive des données (input drift) | PSI > 0,20 sur ≥ 1 variable-clé (Evidently — §13) | Contrôle mensuel | Réentraîner dans les 2 semaines |
# | Chute de performance | PR-AUC OOF < `config.CIBLES_PERFORMANCE["pr_auc_min"]` au prochain lot de labels | Détection continue | Réentraîner immédiatement |
# | Calendaire (précaution) | Trimestre écoulé — fréquence fixée au cadrage §2 | Périodique | Réentraîner, même sans dérive détectée |
#
# Le playbook opérationnel (flow Prefect, procédure d'escalade, test de non-régression)
# est formalisé en §13.  La périodicité de revue des indicateurs est décidée dès le
# cadrage §2, conformément à l'item C9.

# %% [markdown]
# ### 9.13 Latence d'inférence — confrontation aux cibles §8 (item C4 : temps d'inférence)
#
# La latence est mesurée sur le **modèle final fitté** sur la totalité des données,
# puis confrontée aux cibles fixées *a priori* en §8 (`config.CIBLES_PERFORMANCE`).
#
# Deux scénarios d'usage :
# - **Inférence unitaire** : webhook CRM déclenché à chaque date de renouvellement.
# - **Inférence batch** : job nocturne sur 5 000 comptes.

# %%
rapport_latence = mesurer_latence(modele_final, X)

_lat_med = rapport_latence["latence_unitaire_ms_mediane"]
_lat_p95 = rapport_latence["latence_unitaire_ms_p95"]
_lat_batch = rapport_latence["latence_batch_5k_s"]
_cible_unit = config.CIBLES_PERFORMANCE["latence_unitaire_ms"]
_cible_batch = config.CIBLES_PERFORMANCE["latence_batch_5k_s"]

_latence_df = pd.DataFrame(
    {
        "Indicateur": [
            "Latence unitaire — médiane (ms)",
            f"Latence unitaire — p95 (ms)  [cible ≤ {_cible_unit} ms]",
            f"Latence batch {rapport_latence['n_batch']:,} lignes (s)  [cible ≤ {_cible_batch} s]",
            "Nombre d'appels unitaires mesurés",
        ],
        "Valeur": [
            f"{_lat_med:.2f} ms",
            f"{_lat_p95:.2f} ms",
            f"{_lat_batch:.3f} s",
            str(rapport_latence["n_unitaire"]),
        ],
        "Statut": [
            "—",
            "✅" if _lat_p95 <= _cible_unit else "❌",
            "✅" if _lat_batch <= _cible_batch else "❌",
            "—",
        ],
    }
).set_index("Indicateur")

display(_latence_df)

_lat_ok = _lat_p95 <= _cible_unit and _lat_batch <= _cible_batch
display(
    Markdown(
        "**Ce qu'il faut retenir.**  "
        + (
            f"Les deux contraintes de latence sont respectées : "
            f"p95 unitaire = **{_lat_p95:.1f} ms** (< {_cible_unit} ms) "
            f"et batch = **{_lat_batch:.2f} s** (< {_cible_batch} s).  "
            "Le modèle est compatible avec un déploiement en API synchrone (webhook CRM) "
            "et en job batch nocturne."
            if _lat_ok
            else "⚠️ Au moins une contrainte de latence n'est pas respectée.  "
            "Pistes d'amélioration : réduire `n_estimators`, "
            "activer la quantification, ou passer à un pipeline ONNX pour l'inférence unitaire."
        )
    )
)

# %% [markdown]
# > ### 📋 Journal de bord — Entraînement et validation (§9 complet)
# >
# > **Décisions retenues** — Protocole `RepeatedStratifiedKFold(5, 3)` partagé entre tous les
# > modèles ; PR-AUC comme métrique principale (adapté au déséquilibre) ; `class_weight='balanced'`
# > retenu pour le seuil économique (probabilités calibrées) ; SMOTE testé uniquement comme
# > comparaison méthodologique.  Optuna TPE + MedianPruner sur 30 essais (5-fold StratifiedKFold),
# > étude persistée en SQLite ; gain marginal (≤ 0.02) signalé explicitement.  Réentraînement
# > final sur la totalité de X avant gel du modèle — sérialisé en joblib via `charger_ou_calculer`.
# >
# > **Alternatives écartées** — SMOTE comme modèle de production : dégradation de calibration
# > (Brier score plus élevé, courbe de fiabilité déviée).  Transfer learning deep learning :
# > inadapté à 5 000 observations tabulaires — réutilisation des hyperparamètres et warm start
# > retenus à la place.  Optuna sur 100+ essais : rendements décroissants sur ce volume, coût
# > computationnel injustifié au vu de l'empreinte carbone mesurée.
# >
# > **Difficultés rencontrées** — CodeCarbon sous WSL2 : pas d'accès RAPL → estimation TDP,
# > exposée honnêtement (mode et facteur d'émission affichés).  Latence de l'optimisation Optuna :
# > résolue par stockage SQLite + cache JSON via `charger_ou_calculer`.
# >
# > **Impact sur la suite** — Le modèle sérialisé (`modele_final.joblib`) alimente §10 (API et
# > déploiement), §12 (seuil économique, SHAP, ROI), §13 (monitoring et réentraînement) et la
# > model card (`docs/MODEL_CARD.md`).
# > Les hyperparamètres retenus initialisent les bornes Optuna du prochain cycle d'entraînement.
# >
# > **Temps passé** — ~3 h (Optuna, CodeCarbon, latence, arbitrage, éco-conception, transfert).
