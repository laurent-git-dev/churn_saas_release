# %% [markdown]
# ## 9. Entraînement et validation
#
# On veut savoir quel modèle prédit le mieux la résiliation, et si son avance est réelle ou due
# au hasard du découpage. Cinq modèles passent donc le même examen, sur les mêmes plis :
# deux références sans apprentissage (hasard, règle métier), la baseline (référence de base)
# régression logistique, une forêt aléatoire et LightGBM. Les deux candidats les plus
# prometteurs sont ensuite optimisés à budget égal, puis départagés une dernière fois. Le modèle
# retenu est alors jugé, une seule fois, sur un jeu de test mis de côté avant tout, puis
# réappris sur toutes les données pour la production.

# %%
import warnings

import numpy as np
import pandas as pd
from IPython.display import Markdown, display
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from churn_saas import config, viz
from churn_saas.cache import charger_ou_calculer
from churn_saas.features.build import ajouter_features_metier
from churn_saas.format_fr import entier, euros, nombre, pourcentage, scientifique, styler_fr
from churn_saas.models import economics, train
from churn_saas.models import evaluate as eval_mod

warnings.filterwarnings("ignore", category=UserWarning)

CIBLES = config.CIBLES_PERFORMANCE
_SEUIL_PR_AUC = CIBLES["pr_auc_min"]
_SEUIL_ROC_AUC = CIBLES["roc_auc_min"]
_GAIN_MIN_COMPLEXITE = CIBLES["gain_pr_auc_min_complexite"]

# %% [markdown]
# ### 9.1 Chargement du jeu de données gold
#
# Le jeu gold (§7) est relu tel quel, sans les colonnes de `config.COLONNES_INTERDITES`. Un
# modèle se juge sur des comptes qu'aucun de nos choix n'a vus : on met donc de côté, avant
# toute comparaison, un **jeu de test** (part `config.PART_TEST`), stratifié (même taux de
# churn), avec la même graine que la CLI et le flow de réentraînement. Les données servent
# ensuite à trois usages successifs, qui structurent toute la section :
#
# | Données | Usage | Ce qui s'y décide | Où |
# |---|---|---|---|
# | Jeu de développement | **Choisir** | Modèle, réglages, calibration, seuil de vigilance, tous par validation croisée | §9.2 à §9.13 |
# | Jeu de test | **Juger** | Rien : il note une seule fois le modèle choisi, appris sur le seul développement | §9.14 |
# | Toutes les données | **Produire** | Rien : la configuration figée est réapprise sur tous les comptes étiquetés | §9.15 |

# %%
gold = pd.read_parquet(config.DONNEES_GOLD / "gold_dataset.parquet")
CIBLE = "churn"
X_brut = gold.drop(columns=config.COLONNES_INTERDITES, errors="ignore")
y = gold[CIBLE].astype(int)
X = ajouter_features_metier(X_brut).drop(columns=[CIBLE], errors="ignore")
_features_ajoutees = sorted(set(X.columns) - set(X_brut.columns))
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=config.PART_TEST, stratify=y, random_state=config.RANDOM_SEED
)

display(
    Markdown(
        f"**Dimensions** — {entier(X.shape[0])} comptes × {X.shape[1]} variables, dont "
        f"{len(_features_ajoutees)} dérivées par `ajouter_features_metier` (§7.5) : "
        + ", ".join(f"`{c}`" for c in _features_ajoutees)
        + f".  \n**Prévalence du churn** : {pourcentage(y.mean(), 1)} ({entier(y.sum())} "
        f"churners pour {entier((1 - y).sum())} clients fidèles).  \n**Découpage** : "
        f"{entier(len(X_dev))} comptes de développement (churn {pourcentage(y_dev.mean(), 1)}) et "
        f"{entier(len(X_test))} comptes de test (churn {pourcentage(y_test.mean(), 1)})."
    )
)

# %% [markdown]
# **Ce qu'il faut retenir.** Les variables dérivées, calculées ligne par ligne, sont sans
# fuite ; toute transformation **apprise** (imputation, encodage, standardisation) reste dans
# le pipeline (chaîne de traitement), fitté (appris) à l'intérieur de chaque pli. Le test garde
# le taux de churn du portefeuille : sa note sera comparable à celle de la validation croisée.

# %% [markdown]
# ### 9.2 Stratégie d'entraînement
#
# Pour comparer honnêtement, tous les modèles sont notés sur les mêmes copies. On utilise une
# cross-validation (validation croisée) : le jeu de développement est coupé en 5 plis (*folds*), le
# modèle apprend sur 4 et est noté sur le 5ᵉ, à tour de rôle ; l'opération est répétée 3 fois
# avec des découpages différents, soit 15 notes par modèle et par métrique (protocole §8.8).
#
# | Modèle | Rôle |
# |---|---|
# | B0 — Hasard stratifié | Plancher : PR-AUC ≈ prévalence, ROC-AUC ≈ 0,5 |
# | B1 — Règle métier | Ce que le CSM repère aujourd'hui sans modèle (§2) |
# | Baseline — régression logistique | Référence apprise : simple, rapide, explicable |
# | Forêt aléatoire | Arbres moyennés : capte les non-linéarités, robuste aux valeurs extrêmes |
# | LightGBM | Boosting (renforcement) d'arbres : état de l'art sur données tabulaires |

# %%
modeles = train.construire_modeles(X_dev)
cv_info = train.protocole_validation()
N_PLIS_CV = cv_info.cvargs["n_splits"]
N_REPETITIONS_CV = cv_info.n_repeats
display(
    Markdown(
        f"**Protocole** — `RepeatedStratifiedKFold(n_splits={N_PLIS_CV}, "
        f"n_repeats={N_REPETITIONS_CV}, random_state={config.RANDOM_SEED})` : "
        f"{cv_info.get_n_splits()} plis partagés par les {len(modeles)} modèles.  \n"
        "**Ce qu'il faut retenir.** Un seul découpage, une seule graine : les mêmes plis pour "
        "tous, ce qui autorise en §9.3.1 une comparaison pli par pli, et non de simples moyennes."
    )
)

# %% [markdown]
# ### 9.3 Tableau comparatif des modèles
#
# Trois métriques co-principales (§8.8.1), toutes mesurées hors pli :
#
# - **PR-AUC** (aire sous la courbe précision-rappel) : qualité de la liste d'appels, métrique de
#   sélection ; le hasard vaut la prévalence ;
# - **ROC-AUC** (aire sous la courbe ROC) : probabilité qu'un churner soit classé devant un
#   client fidèle ; le hasard vaut 0,5 ;
# - **recall** (rappel) au seuil (*threshold*) de 0,5 : il dépend de l'échelle des probabilités
#   (§9.6) et ne départage pas les modèles ; le recall comparable est celui du seuil de vigilance.
#
# Contrôles : score de Brier (écart quadratique probabilité/issue, plus bas = mieux), latence.

# %%
_LIBELLES = {
    "pr_auc_mean": "PR-AUC (moy.)",
    "pr_auc_std": "PR-AUC (σ)",
    "roc_auc_mean": "ROC-AUC (moy.)",
    "recall_mean": "Recall au seuil 0,5",
    "precision_mean": "Precision au seuil 0,5",
    "brier_mean": "Brier",
    "latence_ms_mean": "Latence unitaire (ms)",
    "duree_eval_s": "Durée d'évaluation (s)",
}


def _afficher_metriques(tableau: pd.DataFrame) -> None:
    """Affiche les métriques de comparaison, PR-AUC colorée de 0 (rouge) à 1 (vert)."""
    colonnes = [c for c in _LIBELLES if c in tableau.columns]
    vue = tableau[colonnes].rename(columns=_LIBELLES)
    display(
        styler_fr(vue, precision=3).background_gradient(
            subset=["PR-AUC (moy.)"], cmap="RdYlGn", vmin=0.0, vmax=1.0
        )
    )


def _calcul_comparaison() -> pd.DataFrame:
    return train.comparer_modeles(modeles, X_dev, y_dev)


# 75 apprentissages (5 modèles × 15 plis) : résultat mis en cache
tableau_modeles, _ = charger_ou_calculer("comparaison_modeles.parquet", _calcul_comparaison)
_afficher_metriques(tableau_modeles)

# %%
fig, axes = viz.figure_grille(
    "comparaison_modeles",
    "Comparaison des modèles en validation croisée (plus haut = meilleur)",
    taille=(14, 5),
)
_ordre = tableau_modeles.sort_values("pr_auc_mean").index
_couleurs = [viz.PALETTE_PRINCIPALE[i % len(viz.PALETTE_PRINCIPALE)] for i in range(len(_ordre))]
for ax, metrique, cible, hasard, libelle_hasard in [
    (axes[0], "pr_auc", _SEUIL_PR_AUC, y_dev.mean(), "Hasard = prévalence"),
    (axes[1], "roc_auc", _SEUIL_ROC_AUC, 0.5, "Hasard"),
]:
    valeurs = tableau_modeles.loc[_ordre, f"{metrique}_mean"]
    barres = ax.barh(
        _ordre, valeurs, xerr=tableau_modeles.loc[_ordre, f"{metrique}_std"], color=_couleurs
    )
    ax.axvline(
        cible, color=viz.COULEUR_CHURN, linestyle="--", label=f"Cible a priori ({nombre(cible, 2)})"
    )
    ax.axvline(
        hasard, color="#888888", linestyle=":", label=f"{libelle_hasard} ({nombre(hasard, 2)})"
    )
    for barre, val in zip(barres, valeurs, strict=True):
        ax.text(val + 0.01, barre.get_y() + barre.get_height() / 2, nombre(val, 3), va="center")
    ax.set_xlim(0, 1.1)
    ax.set_xlabel(f"{_LIBELLES[f'{metrique}_mean']} ± σ sur {cv_info.get_n_splits()} plis")
    ax.legend(loc="lower right")
axes[1].set_yticklabels([])
viz.sauvegarder(fig)

# %%
_ARBRES = [n for n in (train.NOM_FORET, train.NOM_LIGHTGBM) if n in tableau_modeles.index]
_meilleur_arbre = tableau_modeles.loc[_ARBRES, "pr_auc_mean"].idxmax()
_lin, _arb = tableau_modeles.loc[train.NOM_BASELINE_LR], tableau_modeles.loc[_meilleur_arbre]
_ecart_pr = float(_arb["pr_auc_mean"] - _lin["pr_auc_mean"])
_atteint = tableau_modeles[["pr_auc_mean", "roc_auc_mean"]] >= [_SEUIL_PR_AUC, _SEUIL_ROC_AUC]
_n_cibles_ok = int(_atteint.all(axis=1).sum())
display(
    Markdown(
        f"**Ce qu'il faut retenir.** {_n_cibles_ok} modèle(s) sur {len(tableau_modeles)} "
        f"atteignent à la fois les cibles a priori de PR-AUC (≥ {nombre(_SEUIL_PR_AUC, 2)}) et de "
        f"ROC-AUC (≥ {nombre(_SEUIL_ROC_AUC, 2)}). Le hasard (B0) se place bien au niveau de la "
        f"prévalence ({nombre(tableau_modeles.loc['B0 — Hasard stratifié', 'pr_auc_mean'], 3)}) : "
        f"le protocole ne fabrique pas de performance. Écart de PR-AUC de `{_meilleur_arbre}` "
        f"sur la régression logistique : {nombre(_ecart_pr, 3, signe=True)}. "
        + (
            "Les arbres captent des non-linéarités que le modèle linéaire manque."
            if _ecart_pr > 0
            else "Contrairement à l'intuition de §8.6, **la régression logistique devance les "
            "arbres** : les ratios métier de §7 rendent la relation au churn largement monotone, "
            "et sur ~4 000 comptes les arbres paient leur variance."
        )
    )
)

# %% [markdown]
# #### 9.3.1 L'écart est-il significatif ? — Wilcoxon apparié et correction de Holm
#
# Une moyenne plus haute ne suffit pas : l'écart peut venir d'un découpage favorable. On
# applique la règle écrite **avant** les résultats (§8.8) :
#
# 1. **Rejet** des modèles dont la PR-AUC moyenne est sous la cible a priori ;
# 2. **Champion provisoire** : la meilleure PR-AUC moyenne parmi les non-rejetés ;
# 3. **Test de Wilcoxon apparié** : pour chaque concurrent, on regarde pli par pli qui gagne,
#    et de combien ; le test dit si le champion gagne trop souvent pour que ce soit le hasard ;
# 4. **Correction de Holm** : plusieurs tests multiplient les fausses victoires, Holm resserre
#    les seuils pour garder un risque global α ;
# 5. **Règle de complexité** : un modèle plus simple non rejeté que le champion ne bat pas
#    significativement, **ou** de moins de `gain_pr_auc_min_complexite`, est préféré (§8.4).
#
# Ordre de simplicité (`ORDRE_SIMPLICITE`) : B0 < B1 < régression logistique < ensembles
# d'arbres (forêt aléatoire et LightGBM au même niveau).

# %%
_non_rejetes = tableau_modeles.index[tableau_modeles["pr_auc_mean"] >= _SEUIL_PR_AUC].tolist()
# Si aucun modèle n'atteint la cible, on teste quand même le meilleur, mais aucun n'est déployable
champion_provisoire = _non_rejetes[0] if _non_rejetes else tableau_modeles.index[0]
tests_superiorite = train.comparer_au_champion(tableau_modeles, champion_provisoire)
_non_departages = [
    n
    for n in _non_rejetes
    if n != champion_provisoire
    and (
        not tests_superiorite.loc[n, "significatif"]
        or tests_superiorite.loc[n, "ecart_pr_auc_moyen"] < _GAIN_MIN_COMPLEXITE
    )
]
modele_retenu = min(
    [champion_provisoire, *_non_departages],
    key=lambda n: (train.ORDRE_SIMPLICITE.get(n, 99), -tableau_modeles.loc[n, "pr_auc_mean"]),
)

_LIBELLES_TESTS = {
    "ecart_pr_auc_moyen": "Écart PR-AUC moyen",
    "plis_gagnes": "Plis gagnés par le champion",
    "p_valeur_brute": "p-valeur brute",
    "p_valeur_holm": "p-valeur ajustée (Holm)",
    "significatif": f"Significatif (α = {nombre(train.ALPHA_SIGNIFICATIVITE, 2)})",
}


def _afficher_tests(tests: pd.DataFrame, champion: str) -> None:
    """Affiche les tests de supériorité du champion contre chaque concurrent."""
    vue = tests.assign(significatif=tests["significatif"].map({True: "Oui", False: "Non"}))
    vue = vue.rename(columns=_LIBELLES_TESTS)
    vue.index.name = f"Champion « {champion} » contre…"
    display(
        styler_fr(
            vue,
            {
                "Écart PR-AUC moyen": lambda v: nombre(v, 4, signe=True),
                "p-valeur brute": lambda v: scientifique(v, 2),
                "p-valeur ajustée (Holm)": lambda v: scientifique(v, 2),
            },
        )
    )


_afficher_tests(tests_superiorite, champion_provisoire)

# %%
_n_plis = sum(c.startswith(train.PREFIXE_PLI) for c in tableau_modeles.columns)
_verdict = (
    f"**`{modele_retenu}` est retenu** : aucun modèle plus simple ne le concurrence."
    if modele_retenu == champion_provisoire
    else f"Son avance sur `{modele_retenu}`, plus simple, est non significative ou sous "
    f"{nombre(_GAIN_MIN_COMPLEXITE, 2)} : **`{modele_retenu}` est retenu**."
)
display(
    Markdown(
        f"**Ce qu'il faut retenir.** {len(_non_rejetes)} modèle(s) sur {len(tableau_modeles)} "
        f"franchissent la cible de PR-AUC. Le champion provisoire `{champion_provisoire}` bat "
        f"significativement {int(tests_superiorite['significatif'].sum())} concurrent(s) sur "
        f"{len(tests_superiorite)} après correction de Holm. {_verdict} Avec {_n_plis} plis, "
        f"la plus petite p-valeur possible est 1/2^{_n_plis} ≈ {scientifique(0.5**_n_plis, 1)} "
        "(champion meilleur sur tous les plis). Limite assumée : les plis répétés partagent des "
        "données, le test est un garde-fou, pas une preuve absolue."
    )
)

# %% [markdown]
# ### 9.4 Gain réel sur la règle métier
#
# Le modèle vaut-il un déploiement, face à une règle à deux conditions qu'un filtre du CRM
# applique ? Il n'est justifié que si B1 manque la cible a priori, ou s'il le devance d'au
# moins `gain_pr_auc_min_complexite` (règle de complexité).

# %%
_pr_b1, _pr_retenu = tableau_modeles.loc[["B1 — Règle métier", modele_retenu], "pr_auc_mean"]
_gain_b1 = _pr_retenu - _pr_b1
_b1_rejete = _pr_b1 < _SEUIL_PR_AUC
_ml_justifie = not modele_retenu.startswith("B1") and (
    _b1_rejete or _gain_b1 >= _GAIN_MIN_COMPLEXITE
)
display(
    Markdown(
        f"**Ce qu'il faut retenir.** `{modele_retenu}` atteint une PR-AUC de "
        f"{nombre(_pr_retenu, 3)} contre {nombre(_pr_b1, 3)} pour la règle métier, soit "
        f"{nombre(_gain_b1, 3, signe=True)} ({pourcentage(_gain_b1 / max(_pr_b1, 1e-6), 0, signe=True)}"
        f"). La règle B1 {'manque' if _b1_rejete else 'atteint'} la cible a priori de "
        f"{nombre(_SEUIL_PR_AUC, 2)}. "
        + (
            "**Le modèle est justifié** : pour la même capacité d'appels CS, il trouve nettement "
            "plus de vrais churners."
            if _ml_justifie
            else "**La règle métier reste préférable** : plus simple et plus auditable, pour un "
            "gain insuffisant du modèle."
        )
    )
)

# %% [markdown]
# ### 9.5 Journalisation MLflow
#
# Pour retrouver et rejouer chaque résultat, chaque modèle comparé donne lieu à une exécution
# MLflow (paramètres, métriques, modèle au format skops) dans `mlruns/mlflow.db`.

# %%
_params_communs = {
    "random_seed": config.RANDOM_SEED,
    "cv_n_splits": N_PLIS_CV,
    "cv_n_repeats": N_REPETITIONS_CV,
    "pr_auc_min_cible": _SEUIL_PR_AUC,
}
_run_ids = {}
for _nom, _modele in modeles.items():
    # Scores par pli exclus : MLflow reçoit les agrégats, le détail reste dans le cache parquet
    _metriques = {
        k: v
        for k, v in tableau_modeles.loc[_nom].items()
        if isinstance(v, float) and not k.startswith(train.PREFIXE_PLI)
    }
    _run_ids[_nom] = train.journaliser_mlflow(
        _nom, _modele, _metriques, _params_communs | {"modele": _nom}
    )

display(
    Markdown(
        f"**Ce qu'il faut retenir.** {len(_run_ids)} exécutions MLflow enregistrées "
        "(`mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`) : chaque chiffre de §9.3 a "
        "une trace datée, base du registry (registre de modèles) de §10.4."
    )
)

# %% [markdown]
# ### 9.6 Gestion du déséquilibre — class_weight, correction d'intercept, SMOTE
#
# Les churners sont minoritaires : sans précaution, un modèle apprend surtout à reconnaître
# les clients fidèles. Deux remèdes classiques existent, mais tous deux faussent les
# probabilités, alors que §12 les multiplie par des euros. Il faut donc des probabilités
# **calibrées** : 30 % annoncé doit correspondre à 30 % de churners observés.
#
# | Technique | Mécanisme | Effet sur les probabilités |
# |---|---|---|
# | `class_weight='balanced'` | Les erreurs sur les churners pèsent plus lourd | Décalage **connu** : + log((1 − π)/π) sur le logit |
# | Pondération + correction d'intercept (`RegressionLogistiqueRecalibree`) | On retire ce décalage connu de l'intercept (constante) | Corrigé exactement, sans paramètre appris |
# | SMOTE | On fabrique des churners synthétiques | Décalage **sans forme connue** |
#
# Évaluation sur la régression logistique, dans le pipeline, sur les mêmes plis. La correction
# exacte est propre au modèle linéaire : la calibration des arbres n'est pas garantie (§9.12).


# %%
def _calcul_desequilibre() -> dict:
    tableau_deseq, cal_data = train.comparer_desequilibre(X_dev, y_dev, df_ref=X_dev)
    return {"tableau": tableau_deseq.reset_index().to_dict(orient="list"), "calibration": cal_data}


_resultat_deseq, _ = charger_ou_calculer("comparaison_desequilibre.json", _calcul_desequilibre)
_tableau_deseq = pd.DataFrame(_resultat_deseq["tableau"]).set_index("approche")
_cal_data = _resultat_deseq["calibration"]
display(styler_fr(_tableau_deseq, precision=4))

# %%
fig, ax = viz.figure(
    "calibration_desequilibre",
    "Courbes de fiabilité hors pli — traitements du déséquilibre",
    taille=(7, 6),
)
for (nom_approche, cal), couleur in zip(
    _cal_data.items(),
    [viz.PALETTE_PRINCIPALE[3], viz.COULEUR_NON_CHURN, viz.COULEUR_CHURN],
    strict=True,
):
    brier_val = _tableau_deseq.loc[nom_approche, "brier_score"]
    ax.plot(
        cal["prob_pred"],
        cal["prob_true"],
        marker="o",
        color=couleur,
        linewidth=2,
        label=f"{nom_approche} (Brier = {nombre(brier_val, 4)})",
    )
ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", label="Calibration parfaite")
ax.set_xlabel("Probabilité prédite")
ax.set_ylabel("Part de churners observée")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend()
viz.sauvegarder(fig)

# %%
_brier = _tableau_deseq["brier_score"]
_prauc = _tableau_deseq["pr_auc"]
_CW, _CORR, _SMOTE = (
    "class_weight='balanced'",
    "class_weight='balanced' + correction d'intercept",
    "SMOTE",
)
# Écart moyen à la diagonale : > 0 = risque surestimé
_ecart_diag = {
    nom: float(np.mean(np.asarray(cal["prob_pred"]) - np.asarray(cal["prob_true"])))
    for nom, cal in _cal_data.items()
}
display(
    Markdown(
        f"**Ce qu'il faut retenir.** La pondération seule (Brier {nombre(_brier[_CW], 4)}) et "
        f"SMOTE ({nombre(_brier[_SMOTE], 4)}) dégradent la calibration ; en moyenne sur les "
        "tranches, elles "
        + " et ".join(
            f"{'surestiment' if _ecart_diag[n] > 0 else 'sous-estiment'} le risque de "
            f"{nombre(100 * abs(_ecart_diag[n]), 1)} points"
            for n in (_CW, _SMOTE)
        )
        + f". La correction d'intercept ramène le Brier à **{nombre(_brier[_CORR], 4)}** (écart "
        f"moyen de {nombre(100 * _ecart_diag[_CORR], 1, signe=True)} point) sans changer le "
        f"classement (PR-AUC {nombre(_prauc[_CW], 4)} contre {nombre(_prauc[_CORR], 4)}). "
        "Avantage décisif sur SMOTE : son décalage est connu, donc exactement corrigeable ; "
        "c'est la version utilisée dans tout le notebook."
    )
)

# %% [markdown]
# ### 9.7 Optimisation des hyperparamètres (Optuna)
#
# Un hyperparameter (hyperparamètre) est un réglage fixé avant l'apprentissage : force de la
# régularisation, nombre et taille des arbres. Le tuning (optimisation des hyperparamètres)
# cherche les meilleurs réglages ; Optuna le fait intelligemment, en concentrant ses essais
# près des réglages déjà prometteurs (échantillonneur TPE) et en abandonnant tôt les essais
# mal partis (élagage `MedianPruner`).
#
# **Deux modèles optimisés, à budget égal** (30 essais, 5 plis chacun) : la régression
# logistique et LightGBM. Opposer un modèle réglé à un modèle par défaut serait inéquitable.
# La forêt aléatoire, même famille et même complexité que LightGBM, n'est pas optimisée :
# cela doublerait le coût du calcul pour un candidat que §9.3 place derrière.
#
# **Limite assumée.** Le meilleur de 30 essais notés sur les mêmes 5 plis est optimiste : il
# sert à **choisir** les réglages, pas à **juger** le modèle, réévalué sur 15 plis en §9.12.
#
# | Modèle | Hyperparamètre | Espace de recherche | Rôle attendu |
# |---|---|---|---|
# | Régression logistique | `C` | [0,001 ; 100], échelle log | Inverse de la régularisation : plus grand = coefficients plus libres |
# | Régression logistique | `max_iter` | {500, 1000, 2000} | Itérations du solveur, pour garantir la convergence |
# | LightGBM | `num_leaves` | [8, 64], échelle log | Taille de chaque arbre : la capacité principale du modèle |
# | LightGBM | `learning_rate` | [0,01 ; 0,3], échelle log | Taux d'apprentissage : part de correction apportée par chaque arbre |
# | LightGBM | `n_estimators` | [100, 500], pas de 50 | Nombre d'arbres, à équilibrer avec le taux d'apprentissage |
# | LightGBM | `min_child_samples` | [5, 100], échelle log | Comptes minimum par feuille : contre le surapprentissage |
# | LightGBM | `reg_lambda` | [0,001 ; 10], échelle log | Pénalité L2 sur les feuilles |
# | LightGBM | `subsample` | [0,6 ; 1] | Part des comptes tirés pour chaque arbre : décorrèle les arbres |
# | LightGBM | `colsample_bytree` | [0,5 ; 1] | Part des variables tirées pour chaque arbre |

# %%
MODELES_OPTIMISES = [train.NOM_BASELINE_LR, train.NOM_LIGHTGBM]
N_ESSAIS_OPTUNA = 30

resultats_optuna = {}
for _nom in MODELES_OPTIMISES:
    resultats_optuna[_nom], _ = charger_ou_calculer(
        f"optuna_{train.identifiant_modele(_nom)}_meilleurs_params.json",
        lambda nom=_nom: train.optimiser(nom, X_dev, y_dev, n_essais=N_ESSAIS_OPTUNA),
    )

_synthese_optuna = pd.DataFrame.from_dict(
    {
        nom: {
            "PR-AUC par défaut (5 plis)": r["pr_auc_defaut"],
            "PR-AUC du meilleur essai (optimiste)": r["best_value"],
            "Gain d'Optuna (borne haute)": r["gain_pr_auc"],
            "Essais complétés": r["n_essais_completes"],
        }
        for nom, r in resultats_optuna.items()
    },
    orient="index",
)
display(styler_fr(_synthese_optuna, precision=4))
_gains = {nom: r["gain_pr_auc"] for nom, r in resultats_optuna.items()}
_marginaux = [n for n, g in _gains.items() if abs(g) <= _GAIN_MIN_COMPLEXITE]
display(
    Markdown(
        "".join(
            f"**Réglages retenus pour `{nom}`** : "
            + ", ".join(
                f"`{hp}` = {nombre(v, 4) if isinstance(v, float) else v}" for hp, v in p.items()
            )
            + ".  \n"
            for nom, p in ((n, r["best_params"]) for n, r in resultats_optuna.items())
        )
        + "\n**Ce qu'il faut retenir.** "
        + " ; ".join(f"`{n}` gagne {nombre(g, 4, signe=True)} de PR-AUC" for n, g in _gains.items())
        + ". "
        + (
            f"Sous {nombre(_GAIN_MIN_COMPLEXITE, 2)} pour {', '.join(f'`{n}`' for n in _marginaux)}"
            " : les réglages par défaut étaient déjà bien placés. "
            if _marginaux
            else ""
        )
        + "Chaque gain est une borne haute (meilleur essai choisi sur les plis qui le notent) ; "
        "§9.12 dira si l'optimisation change le classement des deux modèles."
    )
)

# %% [markdown]
# ### 9.8 Empreinte carbone du tuning
#
# Optimiser consomme de l'énergie : on la mesure, pour vérifier qu'elle reste proportionnée au
# gain. Sous WSL2, CodeCarbon n'accède pas aux compteurs RAPL du processeur : il **estime**
# alors la consommation depuis sa puissance nominale (TDP) et le facteur d'émission du réseau.

# %%
_carbone = pd.DataFrame.from_dict(
    {
        nom: {
            "Durée (s)": r["duree_s"],
            "Énergie (kWh)": r["energy_kwh"],
            "Émissions (g CO₂)": 1000 * r["emissions_kg_co2"],
            "Coût électrique (€)": r["energy_kwh"] * config.TARIF_ELECTRICITE_EUR_KWH,
            # En deçà d'un centième de point de PR-AUC, le ratio n'a plus de sens
            "kWh par point de PR-AUC gagné": (
                r["energy_kwh"] / (100 * r["gain_pr_auc"]) if r["gain_pr_auc"] > 1e-4 else np.nan
            ),
        }
        for nom, r in resultats_optuna.items()
    },
    orient="index",
)
display(styler_fr(_carbone, precision=6))
_modes = {
    f"{'estimation' if r['is_estimation'] else 'mesure'} ({r['mode_mesure_carbone']})"
    for r in resultats_optuna.values()
}
_facteurs = {r["facteur_emission_kg_kwh"] for r in resultats_optuna.values()}
display(
    Markdown(
        f"**Ce qu'il faut retenir.** Les deux recherches ont émis au total "
        f"{nombre(_carbone['Émissions (g CO₂)'].sum(), 3)} g CO₂ en "
        f"{nombre(_carbone['Durée (s)'].sum() / 60, 1)} min, pour "
        f"{euros(_carbone['Coût électrique (€)'].sum(), 4)} d'électricité, très loin du budget "
        f"a priori de {entier(CIBLES['co2_entrainement_max_g'])} g (§8.2, bilan en §12.13). "
        f"Collecte : {', '.join(sorted(_modes))}, facteur "
        f"{', '.join(nombre(v, 4) for v in sorted(_facteurs))} kg CO₂/kWh : un ordre de grandeur. "
        "Un ratio vide signale un gain sous 0,01 point, sans rien à rapporter au coût."
    )
)

# %% [markdown]
# ### 9.9 Note d'arbitrage performance / temps / carbone
#
# Le coût de l'optimisation est négligeable, en euros comme en carbone : la vraie question est
# celle du **gain**. Sous le seuil de la règle de complexité, la recherche n'est pas relancée à
# chaque cycle : le flow réutilise les réglages trouvés ici (§9.11), jusqu'à un changement des
# données ou des variables. Le budget de 30 essais n'est pas élargi : sur ~4 000 comptes, le
# bruit de la validation croisée est du même ordre que le gain d'un espace plus large.

# %% [markdown]
# ### 9.10 Contraintes d'éco-conception portées au commanditaire
#
# Engagements repris dans la fiche modèle (`docs/MODEL_CARD.md`), conditions de service.
#
# | Poste | Mesure retenue | Pourquoi |
# |---|---|---|
# | Scoring (calcul des scores) | Batch (traitement par lot) nocturne ; API seulement à la demande d'un CSM | Les sources se rafraîchissent une fois par jour : un calcul continu ne verrait rien de neuf |
# | Réentraînement | Trimestriel ou sur dérive détectée (§13) | Pas de réentraînement superflu |
# | Tuning | Réglages réutilisés d'un cycle à l'autre (§9.11) | Une recherche n'est relancée que si les données changent |
# | Matériel | CPU du poste, sans GPU ni instance cloud | Les modèles comparés n'exploitent pas de GPU |
# | Suivi | CodeCarbon à chaque tuning (`reports/tables/codecarbon_emissions.csv`) | Évolution de l'empreinte traçable dans le temps |

# %% [markdown]
# ### 9.11 Transfert de connaissances
#
# On ne repart pas de zéro à chaque modèle. Réutiliser les couches d'un réseau de neurones ne
# s'applique pas à ~5 000 lignes tabulaires, sans texte ni image : ce qui se transmet, c'est le
# **savoir acquis sur le modèle** :
#
# | Ce qui est transmis | Comment | Où |
# |---|---|---|
# | Famille et hyperparamètres du champion | Repris par `famille_et_hyperparametres_champion()` (méta du champion promu, à défaut l'étude Optuna retenue en §9.12) : pas de nouveau tuning à chaque cycle | `flows/retraining.py`, §13.8 |
# | Schéma de préparation | Mêmes colonnes et mêmes transformations, réapprises sur les données du cycle | Tous les cycles |
# | Données cumulées | Anciens et nouveaux comptes étiquetés réunis dans le jeu gold reconstruit | §13.8 |
# | Gate (règle de promotion) | Le challenger n'est promu que s'il atteint la cible et ne régresse pas face au champion | §10.4, §13.8 |
#
# **Ce qu'il faut retenir.** Ce sont les réglages et le schéma qui passent d'un cycle à
# l'autre, pas des poids : ne pas relancer de recherche est aussi un gain d'éco-conception.

# %% [markdown]
# ### 9.12 Sélection finale du modèle
#
# Le meilleur essai d'Optuna était optimiste (§9.7) : les deux modèles optimisés sont donc
# réévalués sur les **15 plis du protocole**, ceux de §9.3, puis départagés par la même règle
# (Wilcoxon, Holm, règle de complexité). À l'issue de cette sous-section, la **configuration**
# du modèle (famille et hyperparamètres) est figée : plus rien ne sera rechoisi.


# %%
def _calcul_optimises() -> pd.DataFrame:
    tableau = train.comparer_modeles_optimises(list(resultats_optuna.values()), X_dev, y_dev)
    # Les réglages évalués accompagnent le cache, pour détecter un cache périmé
    return tableau.assign(
        best_params=[str(resultats_optuna[n]["best_params"]) for n in tableau.index]
    )


tableau_optimises, _ = charger_ou_calculer(
    "comparaison_modeles_optimises.parquet", _calcul_optimises
)
if any(
    n not in tableau_optimises.index
    or tableau_optimises.loc[n, "best_params"] != str(r["best_params"])
    for n, r in resultats_optuna.items()
):
    tableau_optimises, _ = charger_ou_calculer(
        "comparaison_modeles_optimises.parquet", _calcul_optimises, forcer=True
    )
_afficher_metriques(tableau_optimises)

nom_final, tests_optimises = train.selectionner_modele_optimise(tableau_optimises)
_champion_optimise = str(tableau_optimises["pr_auc_mean"].idxmax())
_afficher_tests(tests_optimises, _champion_optimise)

# %%
_autre = next(n for n in tableau_optimises.index if n != nom_final)
_f, _a = tableau_optimises.loc[nom_final], tableau_optimises.loc[_autre]
_motif = (
    "avec la meilleure moyenne"
    if nom_final == _champion_optimise
    else f"plus simple, `{_champion_optimise}` n'ayant pas d'avance suffisante (règle de complexité)"
)
display(
    Markdown(
        f"**Ce qu'il faut retenir.** Sur les mêmes 15 plis, `{nom_final}` obtient une PR-AUC de "
        f"{nombre(_f['pr_auc_mean'], 3)} (ROC-AUC {nombre(_f['roc_auc_mean'], 3)}), contre "
        f"{nombre(_a['pr_auc_mean'], 3)} ({nombre(_a['roc_auc_mean'], 3)}) pour `{_autre}` : "
        f"**`{nom_final}` est retenu**, {_motif}. L'optimisation "
        + ("confirme" if nom_final == modele_retenu else "renverse")
        + f" le verdict de §9.3.1 (`{modele_retenu}`)."
    )
)

# %%
# Noms relus par §10 (déploiement du modèle retenu)
nom_a_optimiser, resultat_optuna = nom_final, resultats_optuna[nom_final]
_id_final = train.identifiant_modele(nom_final)
# Performance de référence du modèle livré, relue par §14 et les annexes
_eval_protocole, _ = charger_ou_calculer(
    f"evaluation_protocole_{_id_final}_optimise.json",
    lambda: tableau_optimises.loc[nom_final].drop("best_params").astype(float).to_dict()
    | {"best_params": resultat_optuna["best_params"]},
    forcer=True,
)
# Réécrit pour être l'étude la plus récente : à défaut de champion promu,
# `famille_et_hyperparametres_champion()` (§9.11) relit la plus récente
charger_ou_calculer(
    f"optuna_{_id_final}_meilleurs_params.json", lambda: resultat_optuna, forcer=True
)

# Configuration figée, apprise une première fois sur le seul développement (§9.13, §9.14)
_pipeline_dev = train.construire_modele_optimise(
    nom_final, resultat_optuna["best_params"], df_ref=X_dev
)
modele_dev = clone(_pipeline_dev).fit(X_dev, y_dev)

# %%
_clf_final = modele_dev[-1]
_lr_recalibree = type(_clf_final).__name__ == "RegressionLogistiqueRecalibree"
_calibration_ok = _lr_recalibree and bool(_brier[_CORR] < _brier[_CW])
if hasattr(_clf_final, "feature_importances_"):
    _explainer = "TreeExplainer (arbres)"
elif hasattr(_clf_final, "coef_"):
    _explainer = "LinearExplainer (linéaire)"
else:
    _explainer = None
_criteres = pd.DataFrame.from_dict(
    {
        "Calibration des probabilités": (
            (
                f"Brier {nombre(_brier[_CORR], 4)} avec correction, {nombre(_brier[_CW], 4)} sans"
                if _lr_recalibree
                else f"Brier {nombre(_eval_protocole['brier_mean'], 4)}, sans correction établie"
            ),
            "Probabilités fiables pour les euros (§12)",
            _calibration_ok,
        ),
        "Explication de chaque score (SHAP)": (
            _explainer or "aucun explainer exact",
            "Explainer exact (§12.8)",
            _explainer is not None,
        ),
        "Traçabilité MLflow": (
            f"{len(_run_ids)} exécutions sur {len(tableau_modeles)} modèles",
            "Une exécution par modèle comparé",
            len(_run_ids) == len(tableau_modeles),
        ),
    },
    orient="index",
    columns=["Valeur obtenue", "Cible", "Statut"],
)
_criteres["Statut"] = _criteres["Statut"].map({True: "✅", False: "❌"})
display(styler_fr(_criteres).set_properties(**{"text-align": "left"}))

# %% [markdown]
# **Ce qu'il faut retenir.** Au-delà du score, chaque critère de déploiement est vérifié par le
# calcul : un ❌ signale un point à traiter avant la mise en service. La configuration est
# désormais figée. Elle est apprise une première fois sur le seul jeu de développement
# (`modele_dev`) : c'est ce modèle que mesurent la latence (§9.13) et le test (§9.14).

# %% [markdown]
# ### 9.13 Latence d'inférence — confrontation aux cibles §8
#
# Un CSM qui ouvre une fiche client attend la réponse : on mesure donc le temps de calcul d'un
# score isolé (95ᵉ centile, p95 : 95 % des appels sont plus rapides) et celui du batch nocturne,
# face aux cibles a priori de §8.2. Ce temps dépend de la configuration (nombre de coefficients
# ou d'arbres), pas du nombre de comptes appris : `modele_dev` le mesure pour le modèle livré.

# %%
# Lot de 5 000 comptes tiré du seul développement : le test reste fermé jusqu'en §9.14
_lot_latence = X_dev.sample(5_000, replace=True, random_state=config.RANDOM_SEED)
# Mesurée à chaque exécution et écrite sur disque : §12.3 et §14 relisent cette même mesure
rapport_latence, _ = charger_ou_calculer(
    "latence_modele_final.json",
    lambda: train.mesurer_latence(modele_dev, _lot_latence),
    forcer=True,
)
_lat_p95 = rapport_latence["latence_unitaire_ms_p95"]
_lat_batch = rapport_latence["latence_batch_5k_s"]
_ok_unitaire = _lat_p95 <= CIBLES["latence_unitaire_ms"]
_ok_batch = _lat_batch <= CIBLES["latence_batch_5k_s"]
display(
    Markdown(
        "| Mesure | Valeur | Cible | Statut |\n|---|---|---|---|\n"
        f"| Unitaire — médiane | {nombre(rapport_latence['latence_unitaire_ms_mediane'], 2)} ms "
        "| — | — |\n"
        f"| Unitaire — p95 | {nombre(_lat_p95, 2)} ms | ≤ {entier(CIBLES['latence_unitaire_ms'])} "
        f"ms | {'✅' if _ok_unitaire else '❌'} |\n"
        f"| Batch de {entier(rapport_latence['n_batch'])} comptes | {nombre(_lat_batch, 3)} s "
        f"| ≤ {entier(CIBLES['latence_batch_5k_s'])} s | {'✅' if _ok_batch else '❌'} |\n\n"
        "**Ce qu'il faut retenir.** "
        + (
            f"Les deux cibles sont tenues avec une large marge (mesure sur "
            f"{entier(rapport_latence['n_unitaire'])} appels unitaires) : le modèle convient à "
            "l'API synchrone comme au batch nocturne."
            if _ok_unitaire and _ok_batch
            else "⚠️ Au moins une cible de latence est manquée : alléger le préprocesseur ou le "
            "modèle avant déploiement."
        )
    )
)

# %% [markdown]
# ### 9.14 Évaluation finale sur le jeu de test
#
# Toutes les notes précédentes viennent du jeu de développement, celui qui a servi à choisir le
# modèle, ses réglages et sa calibration : elles peuvent être un peu optimistes. On ouvre donc,
# une seule fois, le test mis de côté en §9.1, dans les conditions de la production :
#
# 1. **Le modèle noté** est `modele_dev` : la configuration retenue, apprise sur le seul
#    développement ;
# 2. **Le seuil de vigilance** est fixé sans voir le test, sur les prédictions out-of-fold (hors
#    pli : chaque compte du développement est noté par un modèle qui ne l'a pas appris) ;
# 3. **La note du test** reçoit un intervalle de confiance à 95 % par bootstrap
#    (rééchantillonnage avec remise des comptes du test) : sur un échantillon de cette taille,
#    une note est bruitée.
#
# **Règle de non-retour.** Aucune décision n'est modifiée après lecture du test. Une cible
# manquée ici est documentée et portée au plan d'amélioration (§13), jamais corrigée par un
# nouveau réglage : le test deviendrait alors un second jeu de validation, et sa note perdrait
# toute valeur.


# %%
def _calcul_evaluation_test() -> dict:
    cv = StratifiedKFold(N_PLIS_CV, shuffle=True, random_state=config.RANDOM_SEED)
    oof = cross_val_predict(clone(_pipeline_dev), X_dev, y_dev, cv=cv, method="predict_proba")
    vigilance = economics.seuil_pour_recall(y_dev, oof[:, 1])
    proba = modele_dev.predict_proba(X_test)[:, 1]
    tableau = eval_mod.evaluer_sur_test(y_test, proba, vigilance["seuil"])
    return {"modele": nom_final, "vigilance_dev": vigilance, "test": tableau.to_dict("index")}


# Recalculé à chaque exécution (quelques secondes) : §12 et §14 relisent ce fichier
evaluation_test, _ = charger_ou_calculer(
    "evaluation_test.json", _calcul_evaluation_test, forcer=True
)
_t, _v = evaluation_test["test"], evaluation_test["vigilance_dev"]
_dev = {"pr_auc": _eval_protocole["pr_auc_mean"], "roc_auc": _eval_protocole["roc_auc_mean"]} | _v
_cibles_test = {"pr_auc": _SEUIL_PR_AUC, "roc_auc": _SEUIL_ROC_AUC}
_cibles_test["recall"] = CIBLES["recall_vigilance_min"]
_LIBELLES_TEST = {
    "pr_auc": "PR-AUC",
    "roc_auc": "ROC-AUC",
    "recall": "Recall au seuil de vigilance",
    "precision": "Precision au seuil de vigilance",
    "part_signalee": "Part des comptes signalés",
}
_vue_test = pd.DataFrame(_t).T.loc[list(_LIBELLES_TEST)]
_vue_test.columns = ["Test", "IC 95 % bas", "IC 95 % haut"]
_vue_test.insert(0, "Développement (hors pli)", pd.Series(_dev))
_vue_test["Cible"] = pd.Series(_cibles_test)
_atteinte = np.where(_vue_test["Test"] >= _vue_test["Cible"], "✅", "❌")
_vue_test["Statut"] = np.where(_vue_test["Cible"].isna(), "—", _atteinte)
display(styler_fr(_vue_test.rename(index=_LIBELLES_TEST), precision=3))
_ecart = _t["pr_auc"]["valeur"] - _dev["pr_auc"]
_dans_ic = _t["pr_auc"]["ic_bas"] <= _dev["pr_auc"] <= _t["pr_auc"]["ic_haut"]
_n_ok_test = sum(_t[c]["valeur"] >= v for c, v in _cibles_test.items())
display(
    Markdown(
        f"**Ce qu'il faut retenir.** Seuil de vigilance fixé sur le développement : "
        f"{nombre(_v['seuil'], 3)}. Sur {entier(len(y_test))} comptes jamais vus, `{nom_final}` "
        f"obtient une PR-AUC de {nombre(_t['pr_auc']['valeur'], 3)}, "
        f"{nombre(_ecart, 3, signe=True)} face à la validation croisée : "
        + (
            "l'écart tient dans l'intervalle de confiance du test, la sélection n'a pas rendu "
            "la validation croisée mesurablement optimiste. "
            if _dans_ic
            else "l'écart sort de l'intervalle de confiance du test, la validation croisée "
            "était optimiste et c'est la note du test qui fait foi. "
        )
        + f"{_n_ok_test} cible(s) a priori sur {len(_cibles_test)} sont tenues sur le test. "
        f"Le recall de {pourcentage(_t['recall']['valeur'], 1)} au seuil appris hors pli dit "
        "si le premier niveau de la règle de décision (§12) généralise à de nouveaux comptes. "
        "Ces notes sont les **notes de référence** du modèle, reprises en §12 et §14."
    )
)

# %% [markdown]
# ### 9.15 Modèle de production — réapprentissage sur toutes les données
#
# Le test a rempli son rôle : il a jugé la configuration retenue. Le tenir encore à l'écart
# priverait le modèle livré d'une part des exemples, sans rien gagner en rigueur, puisqu'il ne
# reste plus aucun choix à faire. La configuration figée en §9.12 est donc réapprise, telle
# quelle, sur tous les comptes étiquetés.
#
# | | Modèle noté (§9.14) | Modèle livré (ici) |
# |---|---|---|
# | Configuration | Famille et hyperparamètres figés en §9.12 | Identique |
# | Comptes appris | Jeu de développement | Développement et test, soit tous les comptes étiquetés |
# | Note propre | Jeu de test, avec intervalle de confiance | Aucune : il ne reste aucun compte étiqueté qu'il n'ait pas vu |
#
# **Contrepartie assumée.** Le modèle livré n'a pas de note propre : on lui attribue celle du
# test, une estimation plutôt prudente, car à configuration égale, plus d'exemples améliorent
# ou stabilisent un modèle, rarement l'inverse. Sa performance réelle est ensuite suivie en
# production (§13.7).

# %%
_pipeline_final = train.construire_modele_optimise(
    nom_final, resultat_optuna["best_params"], df_ref=X
)


def _meme_configuration(modele_a, modele_b) -> bool:
    """Même famille et mêmes hyperparamètres pour le dernier étage des deux pipelines."""
    return modele_a[-1].__class__ is modele_b[-1].__class__ and (
        modele_a[-1].get_params() == modele_b[-1].get_params()
    )


modele_final, _date_modele = charger_ou_calculer(
    "modele_final.joblib", lambda: clone(_pipeline_final).fit(X, y)
)
# Le nom du cache ne dépend pas du modèle : un cache d'un autre champion serait relu sans erreur
if not _meme_configuration(modele_final, _pipeline_final):
    modele_final, _date_modele = charger_ou_calculer(
        "modele_final.joblib", lambda: clone(_pipeline_final).fit(X, y), forcer=True
    )
display(
    Markdown(
        f"**Ce qu'il faut retenir.** Le modèle de production, `{nom_final}` optimisé, est appris "
        f"sur les {entier(len(X))} comptes étiquetés ({entier(len(X_dev))} de développement et "
        f"{entier(len(X_test))} de test) et sérialisé dans `reports/tables/modele_final.joblib` "
        f"(produit le {_date_modele:%Y-%m-%d %H:%M}). C'est le seul modèle relu par la suite "
        "(§10 à §13) ; ses notes de référence restent celles du test (§9.14)."
    )
)

# %% [markdown]
# > ### 📋 Journal de bord — Entraînement et validation
# >
# > **Décisions retenues** — Un seul protocole `RepeatedStratifiedKFold(5, 3)` pour les cinq
# > modèles ; PR-AUC pour la sélection, ROC-AUC et recall en co-principales ; test de Wilcoxon
# > apparié, correction de Holm et règle de complexité. Optuna sur la régression logistique
# > **et** LightGBM, à budget égal, puis sélection finale sur les 15 plis du protocole.
# > Probabilités calibrées par correction d'intercept. Trois usages des données : le
# > développement pour **choisir** (§9.2 à §9.13), le test de 20 % pour **juger**, une seule
# > fois et sans retour possible, le modèle appris sur le développement (§9.14), toutes les
# > données pour **produire** la configuration figée, sans rien rechoisir (§9.15).
# >
# > **Alternatives écartées** — Optimiser un seul modèle (comparaison inéquitable) ; optimiser
# > aussi la forêt (même famille que LightGBM, coût doublé) ; SMOTE (décalage non corrigeable) ;
# > calibration de Platt (superflue, le décalage est connu) ; transfert de réseaux profonds
# > (inadapté à 5 000 lignes tabulaires) ; recall au seuil 0,5 comme critère (dépend de
# > l'échelle des probabilités) ; validation croisée seule, sans test (aucune mesure sur des
# > données que la sélection n'a pas vues) ; livrer le modèle appris sur le seul développement
# > (20 % d'exemples en moins, sans gain de rigueur une fois tous les choix figés).
# >
# > **Difficultés rencontrées** — La régression logistique l'a emporté sur LightGBM, contre
# > l'attente : les textes écrits pour un champion à arbres ont été reformulés pour ne dépendre
# > que du résultat calculé. Le test de Wilcoxon était d'abord impossible (scores par pli non
# > conservés). Sans cache, l'exécution complète dépasse la cible de 10 minutes.
# >
# > **Impact sur la suite** — `evaluation_test.json` porte la note finale, relue par §12 et §14.
# > `modele_final.joblib` (§9.15) alimente §10 (API, registre), §12 (seuil de vigilance,
# > priorisation économique, SHAP), §13 (surveillance, réentraînement) et la fiche modèle. Les
# > réglages retenus sont réutilisés par `flows/retraining.py` (§9.11).
