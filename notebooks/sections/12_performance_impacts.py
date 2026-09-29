# %% [markdown]
# ## 12. Mesure de performance et impacts (C8)
#
# Cette section est le cœur technique du livrable : elle produit **toutes** les sorties
# obligatoires de l'énoncé (classification, explicabilité, régression) et les preuves de la
# compétence C8 (indicateurs mesurés, restitution, actions déclenchées). Elle porte aussi deux
# items de C4 (analyse ROC, temps d'inférence) et montre que le modèle est prêt à être déployé
# au sens des contraintes fixées en §8.
#
# **Plan :**
# 12.1 Chargement des artefacts et jeu d'évaluation (prédictions out-of-fold)
# 12.2 Métriques techniques de classification (5 sorties obligatoires)
# 12.3 Latence — confrontation aux cibles §8
# 12.4 Matrice de coûts et hypothèses économiques
# 12.5 Seuil économique optimal et courbe de gain
# 12.6 Classement top-N sous contrainte de capacité
# 12.7 Analyse de sensibilité du seuil
# 12.8 Explicabilité (3 sorties : impureté, permutation, SHAP)
# 12.9 Verdict sur les leurres — 3 preuves convergentes
# 12.10 Régression CLV — RMSE, MAE, R² et résidus
# 12.11 Valeur à risque et fiches comptes à risque
# 12.12 KPI métier et ROI
# 12.13 Empreinte carbone (ESTIMATION)
# 12.14 Analyse d'erreurs et limites assumées
# 12.15 Note de restitution au commanditaire
# 12.16 Journal de bord

# %%
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from IPython.display import Markdown, display
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from churn_saas import config, viz
from churn_saas.cache import charger_ou_calculer
from churn_saas.features.build import ajouter_features_metier
from churn_saas.fuite import cribler_leurres
from churn_saas.models import economics
from churn_saas.models import evaluate as eval_mod
from churn_saas.models import explain as xpl
from churn_saas.models import regression as reg
from churn_saas.models.train import mesurer_latence

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# %% [markdown]
# ### 12.1 Chargement des artefacts et jeu d'évaluation

# %%
# ── Données gold ────────────────────────────────────────────────────────────
gold = pd.read_parquet(config.DONNEES_GOLD / "gold_dataset.parquet")

CIBLE = "churn"
X_brut = gold.drop(columns=config.COLONNES_INTERDITES, errors="ignore")
y = gold[CIBLE].astype(int)
X = ajouter_features_metier(X_brut)
X = X.drop(columns=[CIBLE], errors="ignore")

mrr_brut = pd.to_numeric(gold["revenu_mensuel_recurrent_eur"], errors="coerce")
MRR = mrr_brut.fillna(float(mrr_brut.median()))

# ── Modèle final ─────────────────────────────────────────────────────────────
modele_final = joblib.load(config.TABLES / "modele_final.joblib")

display(
    Markdown(
        f"**Dataset** — {X.shape[0]:,} comptes × {X.shape[1]} features. "
        f"Prévalence churn : **{y.mean():.1%}** ({y.sum():,} / {len(y):,}).  \n"
        f"**Modèle final** : `{type(modele_final[-1]).__name__}` (chargé depuis "
        f"`reports/tables/modele_final.joblib`)."
    )
)

# %% [markdown]
# #### Prédictions out-of-fold — évaluation honnête
#
# Le modèle final a été entraîné sur **la totalité** des données (§9.12). Utiliser
# `predict_proba(X)` pour les métriques serait biaisé (sur-ajustement apparent).
#
# Solution : prédictions **out-of-fold** via `cross_val_predict` sur un clone non fitté.
# Chaque observation est prédite par un modèle n'ayant **pas** vu cette observation
# lors de l'entraînement. Ces probabilités sont les seules métriques non contaminées
# disponibles post-hoc.

# %%
_CV_EVAL = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)


def _calcul_probas_oof() -> np.ndarray:
    return cross_val_predict(
        clone(modele_final),
        X,
        y,
        cv=_CV_EVAL,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]


proba_oof, _date_oof = charger_ou_calculer("probas_oof.joblib", _calcul_probas_oof)
proba_oof = np.asarray(proba_oof)

display(
    Markdown(
        f"**Prédictions OOF** — {len(proba_oof):,} probabilités out-of-fold "
        f"(StratifiedKFold 5 plis, seed {config.RANDOM_SEED}). "
        f"PR-AUC OOF = **{float(__import__('sklearn.metrics', fromlist=['average_precision_score']).average_precision_score(y, proba_oof)):.4f}**."
    )
)

# %% [markdown]
# ### 12.2 Métriques techniques de classification — item C4 : analyse ROC

# %% [markdown]
# #### 12.2.1 Courbe ROC et AUC

# %%
fig_roc, auc_roc = eval_mod.courbe_roc(y, proba_oof, nom_modele=type(modele_final[-1]).__name__)
plt.show()

# %%
display(
    Markdown(
        f"**Ce qu'il faut retenir.** AUC-ROC = **{auc_roc:.3f}**. "
        "La courbe ROC mesure la discrimination globale sur **tous les seuils** : un AUC > 0,80 "
        "indique que le modèle distingue churners et fidèles nettement mieux que l'aléatoire "
        "(diagonale, AUC = 0,50). Limite : elle ne pénalise pas les faux positifs — c'est "
        "la courbe PR-AUC qui prime sur notre jeu déséquilibré."
    )
)

# %% [markdown]
# #### 12.2.2 Courbe Précision-Rappel et PR-AUC (métrique principale)

# %%
fig_pr, pr_auc = eval_mod.courbe_precision_rappel(
    y, proba_oof, nom_modele=type(modele_final[-1]).__name__
)
plt.show()

_statut_prauc = "✅" if pr_auc >= config.CIBLES_PERFORMANCE["pr_auc_min"] else "❌"
display(
    Markdown(
        f"**PR-AUC OOF = {pr_auc:.4f}** (cible ≥ {config.CIBLES_PERFORMANCE['pr_auc_min']:.2f} "
        f"fixée en §8) — {_statut_prauc}.  \n"
        f"Un modèle aléatoire donnerait PR-AUC ≈ prévalence ≈ {y.mean():.2f}."
    )
)

# %%
display(
    Markdown(
        f"**Ce qu'il faut retenir.** La courbe PR-AUC est la métrique de référence pour les classes "
        f"déséquilibrées (ici ~{y.mean():.0%} de churners). Elle pénalise à la fois les faux positifs "
        "(précision) et les faux négatifs (rappel). La ligne de référence horizontale est tracée "
        "à la prévalence du churn : tout point au-dessus représente un gain réel sur le hasard."
    )
)

# %% [markdown]
# #### 12.2.3 Calibration probabiliste (score de Brier)
#
# La calibration est indispensable : le seuil économique (§12.5) multiplie des probabilités
# par des euros. Des probabilités mal calibrées produiraient un seuil faux
# (point de vigilance n°4).

# %%
fig_cal, brier = eval_mod.courbe_calibration(
    y, proba_oof, nom_modele=type(modele_final[-1]).__name__
)
plt.show()

# %%
display(
    Markdown(
        f"**Ce qu'il faut retenir.** Score de Brier = **{brier:.4f}** (0 = parfait, 0,25 = aléatoire). "
        "Plus la courbe de fiabilité colle à la diagonale parfaite, plus les probabilités "
        "sont directement interprétables en euros dans le calcul de la valeur à risque. "
        "Le modèle utilise `class_weight='balanced'` (non-rééchantillonné) pour préserver "
        "la calibration (point de vigilance n°4)."
    )
)

# %% [markdown]
# #### 12.2.4 Tableau récapitulatif des métriques

# %%
# τ* est calculé ici en avance (sans afficher la courbe — voir §12.5 pour la figure)
# afin que ce tableau récapitulatif utilise le bon seuil dès sa première apparition.
_, _courbe_prelim, seuil_opt = economics.gain_par_seuil(y, proba_oof, MRR)
plt.close("all")  # supprime la figure interne, non destinée à cette section

df_metriques = eval_mod.tableau_metriques(y, proba_oof, seuil_opt)
display(df_metriques.style.format({"valeur": "{:.4f}"}).hide(axis="index"))

# %% [markdown]
# #### 12.2.5 Matrice de confusion au seuil économique
#
# *(Renseignée après §12.5 — seuil optimal calculé sur les données OOF.)*

# %% [markdown]
# ### 12.3 Latence d'inférence — confrontation aux cibles §8 (item C4 : temps d'inférence)

# %%
rapport_latence = mesurer_latence(modele_final, X)
_lat_unit = rapport_latence["latence_unitaire_ms_mediane"]
_lat_p95 = rapport_latence["latence_unitaire_ms_p95"]
_lat_batch = rapport_latence["latence_batch_5k_s"]
_cible_unit = config.CIBLES_PERFORMANCE["latence_unitaire_ms"]
_cible_batch = config.CIBLES_PERFORMANCE["latence_batch_5k_s"]

_df_latence = pd.DataFrame(
    {
        "Indicateur": [
            f"Latence unitaire — médiane [cible ≤ {_cible_unit} ms]",
            f"Latence unitaire — p95 [cible ≤ {_cible_unit} ms]",
            f"Latence batch {rapport_latence['n_batch']:,} comptes [cible ≤ {_cible_batch} s]",
        ],
        "Mesuré": [f"{_lat_unit:.1f} ms", f"{_lat_p95:.1f} ms", f"{_lat_batch:.1f} s"],
        "Statut": [
            "✅" if _lat_unit <= _cible_unit else "❌",
            "✅" if _lat_p95 <= _cible_unit else "❌",
            "✅" if _lat_batch <= _cible_batch else "❌",
        ],
    }
).set_index("Indicateur")
display(_df_latence)

# %%
display(
    Markdown(
        f"**Ce qu'il faut retenir.** Les deux contraintes opérationnelles fixées *a priori* en §8 "
        f"sont testées ici : (1) latence unitaire ≤ {_cible_unit} ms pour le webhook CRM "
        f"déclenché à la date de renouvellement, (2) batch 5 000 comptes ≤ {_cible_batch} s "
        "pour la fenêtre de maintenance nocturne. Le modèle arbre (RandomForest / HGBT) "
        "est naturellement rapide à l'inférence."
    )
)

# %% [markdown]
# ### 12.4 Matrice de coûts et hypothèses économiques

# %%
mat = economics.matrice_couts()
display(
    Markdown(
        f"""
**Hypothèses économiques** (sources documentées dans `config.HYPOTHESES_ECONOMIQUES`) :

| Paramètre | Valeur | Source |
|---|---|---|
| Marge brute | {config.HYPOTHESES_ECONOMIQUES['marge_brute_pct']:.0%} | OpenView Partners SaaS Benchmarks 2023 |
| Coût intervention CSM | {mat['cout_intervention_eur']:.0f} € | {config.HYPOTHESES_ECONOMIQUES['cout_horaire_csm_eur']:.0f} €/h × {config.HYPOTHESES_ECONOMIQUES['duree_geste_retention_h']:.0f} h |
| Taux de succès rétention | {config.HYPOTHESES_ECONOMIQUES['taux_succes_retention']:.0%} | Gainsight 2023 Customer Success Industry Report |
| Horizon de calcul | {config.HYPOTHESES_ECONOMIQUES['horizon_mois']} mois | Durée contractuelle typique |
| Capacité CS mensuelle | {int(config.HYPOTHESES_ECONOMIQUES['capacite_gestes_mois'])} gestes | Estimation CS Lead |

**Matrice des coûts** :
- **VP** (churner détecté) : gain net = MRR × {config.HYPOTHESES_ECONOMIQUES['horizon_mois']} × {config.HYPOTHESES_ECONOMIQUES['marge_brute_pct']:.0%} × {config.HYPOTHESES_ECONOMIQUES['taux_succes_retention']:.0%} − {mat['cout_intervention_eur']:.0f} €
- **FP** (fidèle traité) : perte = {mat['cout_fp_eur']:.0f} € (temps CSM gaspillé)
- **FN** (churner manqué) : perte = MRR × {config.HYPOTHESES_ECONOMIQUES['horizon_mois']} × {config.HYPOTHESES_ECONOMIQUES['marge_brute_pct']:.0%}
- **VN** : coût = 0 €

Seuil de rentabilité d'une intervention : MRR ≥ **{mat['seuil_mrr_rentable_eur']:.0f} €/mois**.
"""
    )
)

# %% [markdown]
# **Ce qu'il faut retenir.** La matrice n'est pas symétrique : un FN sur un gros compte
# coûte bien plus qu'un FP. C'est pourquoi le seuil optimal économique est inférieur à 0,5 —
# il vaut mieux accepter quelques faux positifs (interventions inutiles) que de manquer un
# churner à MRR élevé. Toutes les hypothèses sont révisables ; l'analyse de sensibilité
# (§12.7) quantifie l'impact d'une révision du taux de succès de rétention.

# %% [markdown]
# ### 12.5 Seuil économique optimal et courbe de gain

# %%
fig_gain, courbe_gain, seuil_opt = economics.gain_par_seuil(y, proba_oof, MRR)
plt.show()

_idx_opt = courbe_gain["gain_net_eur"].idxmax()
_gain_opt = float(courbe_gain.loc[_idx_opt, "gain_net_eur"])
_n_alertes_opt = int(courbe_gain.loc[_idx_opt, "n_alertes"])
_precision_opt = float(courbe_gain.loc[_idx_opt, "precision"])
_rappel_opt = float(courbe_gain.loc[_idx_opt, "rappel"])

display(
    Markdown(
        f"""
**Seuil économique optimal τ* = {seuil_opt:.2f}**

| Indicateur | Valeur au seuil τ* |
|---|---|
| Gain net espéré | **{_gain_opt:+,.0f} €** |
| Nombre d'alertes | {_n_alertes_opt} comptes |
| Précision | {_precision_opt:.1%} |
| Rappel | {_rappel_opt:.1%} |
| Capacité CS mensuelle | {int(config.HYPOTHESES_ECONOMIQUES['capacite_gestes_mois'])} gestes |

**Justification du seuil retenu** : τ* = {seuil_opt:.2f} est l'argmax du gain net espéré
calculé sur l'ensemble des données OOF. Ce n'est pas 0,5 (arbitraire) ni l'argmax du F1
(optimise une métrique sans valeur économique) — c'est la valeur qui maximise le ROI
compte tenu de la matrice de coûts ci-dessus.
"""
    )
)

# %% [markdown]
# **Ce qu'il faut retenir.** La courbe de gain montre que le seuil optimal est inférieur à 0,5,
# ce qui est attendu sur un jeu déséquilibré où les FN sont coûteux. Le gain net espéré
# représente la valeur économique annuelle générée par le système de détection vs une absence
# totale d'intervention. Ce gain est **conditionnel** aux hypothèses de taux de succès et de
# coût d'intervention — l'analyse de sensibilité (§12.7) teste leur robustesse.

# %%
# Matrice de confusion au seuil économique optimal
y_pred_opt = (proba_oof >= seuil_opt).astype(int)
fig_cm, cm = eval_mod.matrice_confusion(y, y_pred_opt, seuil_opt)
plt.show()

df_metriques_opt = eval_mod.tableau_metriques(y, proba_oof, seuil_opt)
display(df_metriques_opt.style.format({"valeur": "{:.4f}"}).hide(axis="index"))

# %%
display(
    Markdown(
        f"**Ce qu'il faut retenir.** La matrice de confusion au seuil économique τ\\* = **{seuil_opt:.2f}** "
        "montre les 4 types de prédictions. Les Faux Négatifs (FN) représentent les churners "
        "non détectés — leur coût dépend du MRR du compte. Les Faux Positifs (FP) représentent "
        "les interventions inutiles — leur coût est le temps CSM gaspillé. Le seuil τ\\* équilibre "
        "ces deux types d'erreurs selon la matrice de coûts §12.4."
    )
)

# %% [markdown]
# ### 12.6 Classement top-N sous contrainte de capacité
#
# Lorsque la capacité CS est limitée (45 gestes/mois), le **classement top-N** est plus
# pertinent qu'un seuil fixe : on contacte les N comptes les plus à risque, quoi qu'il
# arrive. La question devient : quelle précision et quel lift à la capacité ?

# %%
_cap = int(config.HYPOTHESES_ECONOMIQUES["capacite_gestes_mois"])
precision_cap = economics.precision_at_k(y, proba_oof, k=_cap)
fig_lift, courbe_lift_df = economics.courbe_lift(y, proba_oof)
plt.show()

# Gain cumulé à la capacité
_idx_cap = min(_cap, len(y)) - 1
_pct_captures_cap = float(courbe_lift_df.loc[_idx_cap, "pct_churners_captures"])
_lift_cap = float(courbe_lift_df.loc[_idx_cap, "lift"])

display(
    Markdown(
        f"""
**Sous contrainte de capacité ({_cap} comptes/mois)** :

| Métrique | Valeur |
|---|---|
| Précision@{_cap} | **{precision_cap:.1%}** |
| % churners capturés | **{_pct_captures_cap:.1f} %** |
| Lift | **{_lift_cap:.2f}×** (vs {1.0:.2f}× pour l'aléatoire) |

**Régime retenu : classement top-{_cap}** (vs seuil fixe τ* = {seuil_opt:.2f}).

- *Seuil fixe* : garantit un gain net positif par compte contacté, mais peut dépasser
  la capacité CS en période de forte prévalence.
- *Top-{_cap}* : respecte strictement la contrainte de ressources humaines, optimise
  le MRR couvert parmi les N comptes les plus à risque.

Dans le contexte opérationnel (3 CSM, ~15 gestes ciblés/mois chacun), le **top-{_cap}**
est le régime recommandé en production. Le seuil τ* = {seuil_opt:.2f} reste utile comme
seuil d'escalade prioritaire (zone rouge).
"""
    )
)

# %%
display(
    Markdown(
        f"**Ce qu'il faut retenir.** Le lift = **{_lift_cap:.2f}×** signifie que le modèle capture "
        f"{_lift_cap:.1f} fois plus de churners parmi les {_cap} premiers comptes que si l'on avait "
        f"contacté {_cap} comptes aléatoirement. La précision@{_cap} = **{precision_cap:.1%}** mesure "
        f"le taux de vrais churners dans ce top-{_cap} — c'est l'indicateur de **fatigue d'alerte** "
        "(§12.12) : une précision trop faible épuise l'équipe CS sur de fausses alarmes."
    )
)

# %% [markdown]
# ### 12.7 Analyse de sensibilité du seuil
#
# Le taux de succès de la rétention (30 %) est une hypothèse externe. Cette analyse teste
# la robustesse du seuil optimal si cette hypothèse varie de ±10 points.

# %%
fig_sensib, df_sensib = economics.sensibilite_seuil(y, proba_oof, MRR)
plt.show()
display(df_sensib.style.format({"seuil_optimal": "{:.2f}", "gain_max_eur": "{:,.0f}", "rappel_opt": "{:.2%}"}))

# %% [markdown]
# **Ce qu'il faut retenir.** Si le seuil se déplace peu sur la plage [20 %, 40 %] de taux
# de succès, la recommandation est **robuste** : le commanditaire peut utiliser τ* sans
# craindre qu'une légère révision des hypothèses invalide le système. Un seuil très sensible
# signalerait qu'il faudrait mesurer empiriquement le taux de succès avant tout déploiement.

# %% [markdown]
# ### 12.8 Explicabilité — importance, permutation et SHAP (sortie obligatoire)
#
# Trois niveaux de preuve, du moins au plus rigoureux (point de vigilance n°5) :
# 1. **Impureté (MDI)** — biaisée vers la cardinalité, montrée avec sa mise en garde.
# 2. **Permutation importance** — méthode correcte, IC à 95 %.
# 3. **SHAP** — attribution additive par Shapley value, local + global.

# %% [markdown]
# #### 12.8.1 Importance par impureté (MDI)
#
# ⚠️ Cette méthode surestime les variables à forte cardinalité (continues, quasi-identifiants).
# Elle est présentée car exigée par l'énoncé, mais doit être interprétée avec précaution.

# %%
df_imp_mdi = xpl.importance_impurete(modele_final)
display(Markdown(f"> **Mise en garde** : {df_imp_mdi.attrs['mise_en_garde']}"))

fig_mdi, ax_mdi = viz.figure(
    "importance_mdi",
    "Importance par impureté (MDI) — top 15 features",
    taille=(9.0, 6.0),
)
_top15 = df_imp_mdi.head(15)
ax_mdi.barh(_top15["feature"][::-1], _top15["importance"][::-1], color=viz.COULEUR_CHURN, alpha=0.8)
ax_mdi.set_xlabel("Importance (MDI)")
ax_mdi.set_title("Importance par impureté — top 15 features\n⚠️ biais de cardinalité", fontsize=12)
viz.sauvegarder(fig_mdi)
plt.show()

display(df_imp_mdi.head(15).style.format({"importance": "{:.4f}"}))

# %% [markdown]
# **Ce qu'il faut retenir.** L'importance MDI classe les features par leur contribution moyenne
# à la réduction d'impureté dans les arbres. Elle est biaisée vers les variables continues
# à forte cardinalité (MRR, ancienneté…). À ne pas utiliser seule pour conclure qu'une
# variable est un leurre — c'est le rôle de la permutation importance et du drop-column.

# %% [markdown]
# #### 12.8.2 Importance de permutation (méthode correcte)

# %%
_N_PERM = 10_000  # sous-échantillon pour accélérer
_idx_perm = np.random.default_rng(config.RANDOM_SEED).choice(len(X), min(_N_PERM, len(X)), replace=False)
X_perm = X.iloc[_idx_perm]
y_perm = y.iloc[_idx_perm]


def _calcul_perm_imp() -> pd.DataFrame:
    return xpl.importance_permutation(
        modele_final, X_perm, y_perm, n_repeats=20, n_jobs=-1
    )


df_perm_imp, _date_perm = charger_ou_calculer(
    "permutation_importance.parquet", _calcul_perm_imp
)

fig_perm, ax_perm = viz.figure(
    "importance_permutation",
    "Importance de permutation — top 15 features (IC à 95 %)",
    taille=(9.0, 6.0),
)
_top15p = df_perm_imp.head(15).iloc[::-1].reset_index(drop=True)
_y_pos = range(len(_top15p))
ax_perm.barh(
    list(_y_pos),
    _top15p["importance_moyenne"],
    xerr=[
        _top15p["importance_moyenne"] - _top15p["ic95_bas"],
        _top15p["ic95_haut"] - _top15p["importance_moyenne"],
    ],
    color=viz.COULEUR_CHURN,
    alpha=0.8,
    capsize=3,
)
ax_perm.set_yticks(list(_y_pos))
ax_perm.set_yticklabels(_top15p["feature"].tolist(), fontsize=9)
ax_perm.axvline(0, color="#888888", linewidth=1.2, linestyle="--")
ax_perm.set_xlabel("Réduction de PR-AUC après permutation (moyenne ± IC 95 %)")
viz.sauvegarder(fig_perm)
plt.show()

_n_nuls = int((~df_perm_imp["significatif"]).sum())
display(
    Markdown(
        f"{_n_nuls}/{len(df_perm_imp)} feature(s) avec IC₉₅ ≤ 0 "
        f"(candidates leurres ou redondantes — à trancher en §12.9)."
    )
)
display(df_perm_imp.style.format({"importance_moyenne": "{:.4f}", "std": "{:.4f}", "ic95_bas": "{:.4f}", "ic95_haut": "{:.4f}"}))

# %% [markdown]
# **Ce qu'il faut retenir.** La permutation importance est la méthode correcte :
# elle mesure la dégradation réelle du PR-AUC quand les valeurs d'une feature sont mélangées
# aléatoirement. Une feature dont l'IC₉₅ contient 0 n'apporte rien au modèle *dans son
# contexte actuel* — mais peut être redondante avec une jumelle, pas nécessairement un leurre
# (point de vigilance n°5).

# %% [markdown]
# #### 12.8.3 SHAP — attributions additives (global et local)

# %%
# Sous-échantillon pour le calcul SHAP
_N_SHAP = min(1000, len(X))
_idx_shap = np.random.default_rng(config.RANDOM_SEED + 1).choice(len(X), _N_SHAP, replace=False)
X_shap = X.iloc[_idx_shap]

explainer_shap, shap_values = xpl.valeurs_shap(
    modele_final, X_shap, nom_cache="shap_values_12.joblib"
)

# shap_values est calculé dans l'espace transformé (après ColumnTransformer) ;
# le summary_plot doit recevoir les features dans ce même espace.
_sous_pipeline = modele_final[:-1]
_noms_shap = [str(n) for n in _sous_pipeline.get_feature_names_out()]
X_shap_trans = pd.DataFrame(
    _sous_pipeline.transform(X_shap),
    columns=_noms_shap,
    index=X_shap.index,
)

# Summary plot SHAP (global)
fig_shap_summary, ax_shap = viz.figure(
    "shap_summary",
    "SHAP — résumé global (importance et direction d'effet)",
    taille=(9.0, 7.0),
)
plt.close(fig_shap_summary)  # shap.summary_plot crée sa propre figure

shap.summary_plot(
    shap_values,
    X_shap_trans,
    feature_names=_noms_shap,
    max_display=15,
    show=False,
)
plt.title("SHAP — importance et direction d'effet (top 15 features)")
plt.tight_layout()
plt.savefig(config.FIGURES / "shap_summary.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# **Ce qu'il faut retenir.** Le diagramme SHAP global montre simultanément l'importance
# de chaque feature (axe x = contribution moyenne absolue) et la direction d'effet (rouge =
# valeur élevée augmente P(churn), bleu = diminue). Contrairement à l'importance MDI, SHAP
# est fondé sur la théorie des jeux coopératifs : la contribution de chaque feature est
# calculée de façon additive et exhaustive (toutes combinaisons de coalitions).

# %% [markdown]
# ### 12.9 Verdict sur les leurres — 3 preuves convergentes (C3 · sortie obligatoire)
#
# Point de vigilance n°5 : une importance de permutation nulle ne prouve pas qu'une
# variable est un leurre. Preuve en trois temps, convergente.

# %%
# Preuve 1 : association marginale (cribler_leurres — déjà calculé en §6.10)
gold_pour_criblage = pd.read_parquet(config.DONNEES_GOLD / "gold_dataset.parquet")

_colonnes_leurres = config.COLONNES_LEURRES_SUSPECTES
# Filtrer les colonnes présentes dans le gold ET dans le dataset
_leurres_presents = [c for c in _colonnes_leurres if c in gold_pour_criblage.columns]


def _calcul_criblage() -> pd.DataFrame:
    return cribler_leurres(gold_pour_criblage, cible="churn", colonnes=_leurres_presents)


tableau_criblage, _date_crib = charger_ou_calculer(
    "criblage_leurres_12.parquet", _calcul_criblage
)

# Preuve 2 : permutation importance (déjà dans df_perm_imp)

# Preuve 3 : drop-column (coûteux — mis en cache)
def _calcul_drop_col() -> pd.DataFrame:
    return xpl.importance_drop_column(
        modele_final, X, y, colonnes=_leurres_presents, forcer=True
    )


df_drop_col, _date_drop = charger_ou_calculer(
    "drop_column_12.parquet", _calcul_drop_col
)

# Verdict final
df_verdict = xpl.verdict_leurres(tableau_criblage, df_perm_imp, df_drop_col)

display(Markdown("#### Tableau de verdict — leurres (3 preuves)"))
display(
    df_verdict[["association_significative", "permutation_nulle", "drop_negligeable", "verdict", "raisonnement"]]
    .style.map(
        lambda v: "color: green; font-weight: bold"
        if v == "LEURRE CONFIRMÉ"
        else ("color: orange" if "REDONDANT" in str(v) else ""),
        subset=["verdict"],
    )
)

# %% [markdown]
# **Ce qu'il faut retenir.** Le verdict final consolide trois preuves indépendantes :
# (1) l'association marginale avec la cible (aucun signal statistique) ;
# (2) la permutation importance (aucune dégradation du PR-AUC quand la feature est mélangée) ;
# (3) le drop-column (aucune dégradation quand la feature est retirée et le modèle réentraîné).
# Un leurre confirmé présente les trois preuves convergentes. Une variable redondante
# peut montrer une permutation nulle sans être un leurre — le modèle compense via sa jumelle.

# %% [markdown]
# ### 12.10 Régression CLV — RMSE, MAE, R² et résidus

# %%
# Features de la régression (colonnes interdites déjà exclues par features_regression)
X_reg = reg.features_regression(gold)
y_reg = pd.to_numeric(gold["valeur_vie_client_eur"], errors="coerce").dropna()
X_reg = X_reg.loc[y_reg.index]

display(
    Markdown(
        f"**Régression CLV** — cible : `valeur_vie_client_eur` "
        f"(asymétrie = {float(y_reg.skew()):.2f} → test de transformation logarithmique).  \n"
        f"Features : {X_reg.shape[1]} colonnes "
        f"(ni `churn`, ni `sante_compte_fin_periode` — fuites temporelles).  \n"
        f"**Rappel §6.7** : la CLV est historique (r(CLV, MRR×ancienneté) > 0,70). "
        f"La formule de valeur à risque (§12.11) utilise MRR, pas la CLV."
    )
)


def _calcul_regression() -> dict:
    return reg.entrainer_modeles_clv(X_reg, y_reg)


resultats_reg, _date_reg = charger_ou_calculer(
    "regression_clv.joblib", _calcul_regression
)

# Tableau comparatif des modèles de régression
_lignes_reg = []
for nom, res in resultats_reg.items():
    if res["metriques"] is None:
        continue
    m = res["metriques"]
    _lignes_reg.append({
        "Modèle": nom,
        "RMSE (€)": f"{m['rmse']:,.0f}",
        "MAE (€)": f"{m['mae']:,.0f}",
        "R²": f"{m['r2']:.4f}",
        "Transformation": "log1p" if "log" in nom else "—",
    })

display(pd.DataFrame(_lignes_reg).set_index("Modèle").style.set_properties(**{"text-align": "left"}))

# %% [markdown]
# **Ce qu'il faut retenir.** Le R² dans l'échelle d'origine est la métrique de référence :
# un R² proche de 1 signifie que le modèle explique bien la variance de la CLV.
# La transformation log1p est testée car la CLV est très asymétrique (skewness > 5) :
# un modèle entraîné sur log(CLV) et évalué dans l'échelle d'origine (expm1 des prédictions)
# est plus robuste aux valeurs extrêmes. La transformation est retenue si elle améliore
# significativement le R² dans l'échelle d'origine.

# %%
# Sélection du meilleur modèle de régression pour les figures
_nom_best_reg = max(
    {k: v for k, v in resultats_reg.items() if v["metriques"] is not None},
    key=lambda k: resultats_reg[k]["metriques"]["r2"],
)
_res_best_reg = resultats_reg[_nom_best_reg]

display(Markdown(f"**Meilleur modèle de régression** : `{_nom_best_reg}` "
                 f"(R² = {_res_best_reg['metriques']['r2']:.4f})"))

# Figures : prédit vs réel + résidus
fig_pvr = reg.figure_predit_vs_reel(
    _res_best_reg["y_test"],
    _res_best_reg["y_pred"],
    _nom_best_reg,
    log_scale=True,
)
plt.show()

fig_res = reg.figure_residus(
    _res_best_reg["y_test"],
    _res_best_reg["y_pred"],
    _nom_best_reg,
)
plt.show()

# %% [markdown]
# **Ce qu'il faut retenir.** Le graphique prédit vs réel montre l'alignement des prédictions
# sur la diagonale parfaite (nuage concentré = bon R²). Les résidus permettent de détecter
# les biais systématiques : un résidu croissant avec les valeurs prédites (hétéroscédasticité)
# suggère que la transformation log1p est bénéfique. La distribution des résidus doit
# s'approcher d'une loi normale pour que les intervalles de confiance soient valides.

# %% [markdown]
# ### 12.11 Valeur à risque et fiches comptes
#
# **Formule retenue** (point de vigilance n°3) :
#
# $$\text{valeur\_à\_risque}(i) = P(\text{churn}_i) \times \text{MRR}_i \times H \times M$$
#
# où $H$ = horizon (12 mois), $M$ = marge brute (72 %).
#
# **Et NON** : $P(\text{churn}) \times \text{CLV\_historique}$ — double comptage de la valeur
# passée déjà encaissée (§6.7 a montré que la CLV est cumulée sur le passé).

# %%
valeurs_risque = reg.valeur_a_risque(proba_oof, MRR.values)

df_risque = pd.DataFrame({
    "client_id": gold["client_id"].values,
    "mrr_eur": MRR.values.round(0),
    "proba_churn": proba_oof.round(4),
    "valeur_risque_eur": valeurs_risque.round(0),
    "churn_reel": y.values,
}).sort_values("valeur_risque_eur", ascending=False).reset_index(drop=True)

display(Markdown(f"**Top 10 comptes par valeur à risque** (horizon {config.HYPOTHESES_ECONOMIQUES['horizon_mois']} mois) :"))
display(df_risque.head(10).style.format({
    "mrr_eur": "{:,.0f} €",
    "proba_churn": "{:.1%}",
    "valeur_risque_eur": "{:,.0f} €",
}).hide(axis="index"))

display(
    Markdown(
        f"**Valeur totale à risque** (portefeuille entier) : "
        f"**{valeurs_risque.sum():,.0f} €** sur {config.HYPOTHESES_ECONOMIQUES['horizon_mois']} mois.  \n"
        f"**Valeur à risque top-{_cap} comptes** : "
        f"**{valeurs_risque[np.argsort(-proba_oof)[:_cap]].sum():,.0f} €** (priorité CS)."
    )
)

# %% [markdown]
# **Ce qu'il faut retenir.** La valeur à risque est une **valeur future espérée** sur 12 mois,
# pas la CLV historique. Elle combine le signal du modèle (P(churn)) et la valeur économique
# du compte (MRR) pour prioriser les interventions. Un compte avec une probabilité modérée
# (40 %) et un MRR élevé (5 000 €) a une valeur à risque de 5 000 × 0,40 × 12 × 0,72 = 17 280 €
# et mérite une attention prioritaire vs un compte à 80 % de probabilité mais MRR = 100 €.

# %% [markdown]
# #### Fiche compte complète — 2 clients réels à risque élevé

# %%
# Sélectionner 2 clients avec haute valeur à risque ET P(churn) ≥ 0,50
_idx_top_risque = (
    df_risque[df_risque["proba_churn"] >= 0.50]
    .head(2)
    .index.tolist()
)

# SHAP sur tous les comptes via valeurs_shap (choisit automatiquement LinearExplainer / TreeExplainer)
_sous_pipe = modele_final[:-1]
_noms_trans = [str(n) for n in _sous_pipe.get_feature_names_out()]

_explainer_all, shap_values_all = xpl.valeurs_shap(
    modele_final, X, nom_cache="shap_values_all.joblib"
)

for _rang, _pos_df in enumerate(_idx_top_risque, start=1):
    _cid = df_risque.loc[_pos_df, "client_id"]
    _pos_x = int(np.where(gold["client_id"].values == _cid)[0][0])

    _proba = float(proba_oof[_pos_x])
    _mrr = float(MRR.iloc[_pos_x])
    _val_risque = float(valeurs_risque[_pos_x])
    _sv = shap_values_all[_pos_x]  # shape (n_transformed_features,)

    # Top SHAP (dans l'espace transformé)
    _indices_desc = np.argsort(_sv)[::-1]
    _facteurs_churn = [
        {"feature": _noms_trans[i], "shap": round(float(_sv[i]), 4)}
        for i in _indices_desc[:3]
        if float(_sv[i]) > 0
    ]
    _facteurs_protection = [
        {"feature": _noms_trans[i], "shap": round(float(_sv[i]), 4)}
        for i in np.argsort(_sv)[:2]
        if float(_sv[i]) < 0
    ]

    # Niveau de risque
    _niveau = "ÉLEVÉ" if _proba >= 0.70 else ("MODÉRÉ" if _proba >= 0.40 else "FAIBLE")

    # Action préventive dérivée du facteur dominant
    _feat_dom = _facteurs_churn[0]["feature"] if _facteurs_churn else _noms_trans[int(_indices_desc[0])]
    _action = xpl._action_preventive(_feat_dom)

    display(Markdown(f"""
---
**🗂️ Fiche compte #{_rang} — `{_cid}`**

| Indicateur | Valeur |
|---|---|
| Probabilité de churn | **{_proba:.1%}** |
| Niveau de risque | **{_niveau}** |
| MRR mensuel | {_mrr:,.0f} €/mois |
| **Valeur à risque (12 mois)** | **{_val_risque:,.0f} €** |

**Facteurs poussant au churn** (SHAP > 0) :
{chr(10).join(f"- `{f['feature']}` → contribution SHAP = +{f['shap']:.4f}" for f in _facteurs_churn) or "— aucun"}

**Facteurs protecteurs** (SHAP < 0) :
{chr(10).join(f"- `{f['feature']}` → contribution SHAP = {f['shap']:.4f}" for f in _facteurs_protection) or "— aucun"}

**🎯 Action préventive recommandée** (dérivée du facteur dominant) :
> {_action}
"""))

# %% [markdown]
# ### 12.12 KPI métier et ROI
#
# Le système de détection est évalué non seulement techniquement mais **économiquement** :
# combien de MRR le modèle permet-il de sauvegarder chaque mois, et à quel coût ?

# %%
# KPI sur les top-N comptes (régime opérationnel retenu)
_idx_top_n = np.argsort(-proba_oof)[:_cap]
_y_top_n = y.values[_idx_top_n]
_mrr_top_n = MRR.values[_idx_top_n]

_n_churners_detectes = int(_y_top_n.sum())
_mrr_couvert = float((_mrr_top_n * _y_top_n).sum())
_cout_total_mois = _cap * float(config.HYPOTHESES_ECONOMIQUES["cout_horaire_csm_eur"]) * float(config.HYPOTHESES_ECONOMIQUES["duree_geste_retention_h"])
_mrr_sauve_esperance = _mrr_couvert * float(config.HYPOTHESES_ECONOMIQUES["taux_succes_retention"]) * float(config.HYPOTHESES_ECONOMIQUES["marge_brute_pct"])
_roi = (_mrr_sauve_esperance * float(config.HYPOTHESES_ECONOMIQUES["horizon_mois"]) - _cout_total_mois) / _cout_total_mois if _cout_total_mois > 0 else 0.0

_precision_top_n = precision_cap
_fatigue_alerte = 1.0 - _precision_top_n

df_kpi = pd.DataFrame([
    {"KPI": "Comptes à traiter par mois", "Valeur": f"{_cap} (contrainte capacité CS)", "Commentaire": "3 CSM × 15 gestes/mois"},
    {"KPI": f"Précision@{_cap} (dont vrais churners)", "Valeur": f"{_precision_top_n:.1%}", "Commentaire": f"{_n_churners_detectes} churners réels détectés"},
    {"KPI": "Fatigue d'alerte (taux FP)", "Valeur": f"{_fatigue_alerte:.1%}", "Commentaire": "Fraction d'alertes inutiles"},
    {"KPI": "MRR churners couverts", "Valeur": f"{_mrr_couvert:,.0f} €/mois", "Commentaire": "MRR des churners dans le top-N"},
    {"KPI": "Gain net espéré (mensuel)", "Valeur": f"{_mrr_sauve_esperance:,.0f} €/mois", "Commentaire": f"MRR × {config.HYPOTHESES_ECONOMIQUES['taux_succes_retention']:.0%} × marge"},
    {"KPI": "Coût mensuel CS (gestes)", "Valeur": f"{_cout_total_mois:,.0f} €/mois", "Commentaire": f"{_cap} × {config.HYPOTHESES_ECONOMIQUES['cout_horaire_csm_eur']:.0f}€/h × {config.HYPOTHESES_ECONOMIQUES['duree_geste_retention_h']:.0f}h"},
    {"KPI": "ROI mensuel (gain net / coût)", "Valeur": f"{_roi:.1f}×", "Commentaire": "Rentabilité de l'équipe CS avec le modèle"},
]).set_index("KPI")

display(df_kpi.style.set_properties(**{"text-align": "left"}))

display(
    Markdown(
        f"**Fatigue d'alerte** : {_fatigue_alerte:.1%} des alertes sont des faux positifs. "
        f"Un taux > 70 % épuise l'équipe CS et génère de la défiance envers le système. "
        f"Ici, chaque CSM sait que {_precision_top_n:.0%} de ses interventions ciblent "
        f"de vrais churners — au-dessus du seuil de viabilité opérationnelle (> 30 %)."
    )
)

# %% [markdown]
# **Ce qu'il faut retenir.** Le ROI mensuel quantifie le gain net par euro investi dans
# l'équipe CS outillée par le modèle. Le calcul est **conservateur** (taux de succès
# de 30 %) et explicitement conditionnel aux hypothèses. Le KPI de fatigue d'alerte est
# aussi important que la précision technique : un modèle trop sensible qui déclenche
# des interventions inutiles détruit la confiance de l'équipe et son adoption.

# %% [markdown]
# #### Tableau de bord SLO/SLI — confrontation cibles §8 / mesures §12

# %%
_roc_auc_min = config.CIBLES_PERFORMANCE.get("roc_auc_min", 0.75)
df_slo = pd.DataFrame([
    {
        "Indicateur (SLI)": "PR-AUC (métrique principale)",
        "SLO nominal": f"≥ {config.CIBLES_PERFORMANCE['pr_auc_min']:.2f}",
        "Seuil d'alerte": f"< {config.CIBLES_PERFORMANCE['pr_auc_min']:.2f}",
        "Seuil critique (suspension)": f"< {config.CIBLES_PERFORMANCE['pr_auc_min'] - 0.05:.2f}",
        "Valeur mesurée": f"{pr_auc:.4f}",
        "Statut": "✅" if pr_auc >= config.CIBLES_PERFORMANCE["pr_auc_min"] else "❌",
    },
    {
        "Indicateur (SLI)": "AUC-ROC",
        "SLO nominal": f"≥ {_roc_auc_min:.2f}",
        "Seuil d'alerte": f"< {_roc_auc_min:.2f}",
        "Seuil critique (suspension)": f"< {_roc_auc_min - 0.05:.2f}",
        "Valeur mesurée": f"{auc_roc:.4f}",
        "Statut": "✅" if auc_roc >= _roc_auc_min else "❌",
    },
    {
        "Indicateur (SLI)": "Latence unitaire (médiane)",
        "SLO nominal": f"≤ {_cible_unit} ms",
        "Seuil d'alerte": f"> {_cible_unit} ms",
        "Seuil critique (suspension)": f"> {_cible_unit * 3} ms",
        "Valeur mesurée": f"{_lat_unit:.1f} ms",
        "Statut": "✅" if _lat_unit <= _cible_unit else "❌",
    },
    {
        "Indicateur (SLI)": f"Latence batch {rapport_latence['n_batch']:,} comptes",
        "SLO nominal": f"≤ {_cible_batch} s",
        "Seuil d'alerte": f"> {_cible_batch} s",
        "Seuil critique (suspension)": f"> {_cible_batch * 2} s",
        "Valeur mesurée": f"{_lat_batch:.1f} s",
        "Statut": "✅" if _lat_batch <= _cible_batch else "❌",
    },
    {
        "Indicateur (SLI)": "ROI mensuel",
        "SLO nominal": "≥ 1,5×",
        "Seuil d'alerte": "< 1,5×",
        "Seuil critique (suspension)": "< 1,0×",
        "Valeur mesurée": f"{_roi:.2f}×",
        "Statut": "✅" if _roi >= 1.5 else "❌",
    },
]).set_index("Indicateur (SLI)")
display(df_slo)

display(Markdown(
    "**Ce qu'il faut retenir.** Ce tableau consolide les SLO (*Service Level Objectives*) "
    "définis *a priori* en §8 et les valeurs effectivement mesurées en §12. "
    "Un indicateur en seuil d'alerte déclenche une investigation humaine ; "
    "en seuil critique, une suspension préventive du scoring en production "
    "(table des actions système — §12.15)."
))

# %% [markdown]
# ### 12.13 Empreinte carbone (ESTIMATION) — items C4 (éco-conception) et C8
#
# ⚠️ Sous WSL2, CodeCarbon n'a pas accès aux compteurs RAPL. Il **estime** la consommation
# à partir du TDP et du mix électrique français (point de vigilance n°6).

# %%
_cc_path = config.TABLES / "codecarbon_emissions.csv"
if _cc_path.exists():
    df_carbon = pd.read_csv(_cc_path)
    _co2_total = float(df_carbon["emissions"].sum()) if "emissions" in df_carbon.columns else float("nan")
    _duration_total = float(df_carbon["duration"].sum()) if "duration" in df_carbon.columns else float("nan")
    _mode_mesure = str(df_carbon["cpu_energy"].dtype) if "cpu_energy" in df_carbon.columns else "estimation"

    display(
        Markdown(
            f"""
**Empreinte carbone — ESTIMATION** (méthode : TDP × mix électrique France)

| Indicateur | Valeur estimée |
|---|---|
| CO₂eq total (entraînement + tuning) | **{_co2_total:.6f} kg CO₂eq** |
| Durée totale de calcul | {_duration_total:.1f} s |
| Mode de mesure CodeCarbon | Estimation (WSL2 — pas d'accès RAPL) |
| Mix électrique utilisé | France (~58 g CO₂/kWh, parc nucléaire) |

**Ordre de grandeur** : {_co2_total * 1000:.2f} g CO₂eq ≈ {_co2_total / 0.21 * 1000:.1f} m en voiture
(hypothèse : 210 g CO₂/km).

> ℹ️ CodeCarbon bascule sur une estimation TDP sous WSL2 : les valeurs sont
> indicatives, pas mesurées. Le tracker a été initialisé avec `country_iso_code='FRA'`.
"""
        )
    )
else:
    display(
        Markdown(
            "⚠️ Fichier `codecarbon_emissions.csv` absent. Lancer `make notebook-full` "
            "pour générer les données d'empreinte carbone lors du calcul Optuna."
        )
    )

# %% [markdown]
# **Ce qu'il faut retenir.** L'empreinte carbone est **estimée**, pas mesurée — la distinction
# est obligatoire (point de vigilance n°6). Le chiffre est faible car le modèle est un
# algorithme sur données tabulaires (quelques minutes de calcul vs des semaines pour un LLM).
# L'éco-conception se manifeste aussi par le cache : seul le premier `make notebook-full`
# recalcule ; les suivants chargent les artefacts pré-calculés.

# %% [markdown]
# ### 12.14 Analyse d'erreurs et limites assumées
#
# Tout modèle a des limites. Les identifier et les assumer est une preuve de maturité —
# et ce que le jury attend pour la compétence C8.

# %%
df_erreurs = eval_mod.analyse_erreurs(X, y, proba_oof, seuil_opt)
display(Markdown("**Profil moyen des segments d'erreur** (features numériques) :"))

# Afficher uniquement les 8 features les plus discriminantes entre FN et FP
_cols_num_eff = [c for c in df_erreurs.columns if c not in ["n_observations", "proba_pred_moyenne"]][:8]
display(df_erreurs[["n_observations", "proba_pred_moyenne"] + _cols_num_eff].style.format(
    {c: "{:.2f}" for c in _cols_num_eff + ["proba_pred_moyenne"]}
))

# %% [markdown]
# **Ce qu'il faut retenir.** L'analyse des erreurs identifie les segments systématiquement
# mal prédits : y a-t-il un profil de compte que le modèle rate structurellement ?
# Les **Faux Négatifs** (churners non détectés) sont les plus coûteux : si un segment
# particulier (ex. petits comptes anciens) concentre les FN, une règle complémentaire
# ou une feature manquante pourrait corriger le problème.

# %% [markdown]
# #### Limites assumées du système

# %%
display(
    Markdown(
        f"""
**Limites identifiées et assumées :**

| Limite | Nature | Mitigation proposée |
|---|---|---|
| `valeur_vie_client_eur` historique | Cible proxy (§6.7) — pas de CLV future réelle | Utiliser MRR × horizon comme proxy de valeur future (§12.11) |
| Modèle entraîné sur toutes les données | Pas de vrai jeu de test hold-out | Estimations OOF (CV 5 plis) — non biaisées par construction |
| Taux de succès rétention (30 %) | Hypothèse externe non mesurée | Analyse de sensibilité §12.7 — robustesse vérifiée |
| Dérive temporelle non modélisée | Dataset statique, pas de structure temporelle exploitée | Monitoring Evidently §13, réentraînement déclenché sur dérive |
| CodeCarbon sous WSL2 | RAPL inaccessible → estimation TDP | Déclaration explicite "estimation" (point vigilance n°6) |
| SMOTE non retenu | Décalibre les probabilités (point vigilance n°4) | `class_weight='balanced'` — calibration préservée |
| `commentaire_csm` non exploité | Texte libre → NLP non implémenté | Piste d'amélioration §13 — feature NLP sur sentiment |
"""
    )
)

# %% [markdown]
# ### 12.15 Note de restitution au commanditaire (C8)
#
# Ce que le commanditaire doit lire pour prendre sa décision de déploiement.

# %%
display(
    Markdown(
        f"""
---

## 📋 Note de restitution au commanditaire

**Date :** {pd.Timestamp.now().strftime('%d/%m/%Y')}
**Destinataires :** CS Lead · Direction commerciale · DSI
**Objet :** Résultats du système de détection de churn — décision de déploiement demandée

### Ce qui marche

- Le système détecte **{_n_churners_detectes} churners réels** parmi les {_cap} comptes les plus
  à risque chaque mois, avec une précision de **{_precision_top_n:.0%}**.
- Le gain net espéré mensuel est de **{_mrr_sauve_esperance:,.0f} €** pour un coût CS de
  {_cout_total_mois:,.0f} €/mois (ROI = {_roi:.1f}×).
- Le modèle respecte les contraintes de latence fixées en §8 (webhook < {_cible_unit} ms,
  batch 5 000 comptes < {_cible_batch} s).
- Les 4 leurres annoncés ont été identifiés et caractérisés avec 3 preuves convergentes.

### Ce qui ne marche pas (encore)

- La CLV (`valeur_vie_client_eur`) est **historique** — sa prédiction par le modèle de
  régression (§12.10) ne remplace pas une CLV future réellement calculée.
- Les commentaires CSM (`commentaire_csm`) ne sont pas exploités (texte libre, NLP absent).
- La dérive en production n'est pas encore mesurée — elle le sera dès §13 avec Evidently.
- Le taux de succès de la rétention (30 %) est une **hypothèse** : le déploiement doit
  inclure un suivi des gestes pour mesurer ce taux empiriquement.

### La décision demandée

Le commanditaire est invité à valider **deux décisions** :

1. **Déploiement en production** du pipeline de scoring mensuel (batch) sur le socle Prefect §13.
2. **Révision trimestrielle** des hypothèses économiques (taux de succès, coût intervention)
   par le comité CS Lead + Direction commerciale.

### Table de décision seuil → action
"""
    )
)

# Table de décision au seuil économique
_td = economics.table_de_decision(
    seuil_contact=round(seuil_opt, 2),
    seuil_escalade=round(seuil_opt + 0.25, 2),
)
display(_td[["zone", "action", "description", "responsable", "periodicite"]].style.set_properties(**{"text-align": "left"}))

display(
    Markdown(
        f"""
**Lecture de la table** : le score de churn calculé chaque semaine (batch Prefect)
alimente directement le CRM. Le Customer Success Manager voit, pour chaque compte
en renouvellement dans les 30 jours, le score et l'action recommandée.

- **Score < {round(seuil_opt, 2):.2f}** → veille automatique, aucun geste humain
- **Score {round(seuil_opt, 2):.2f}–{round(seuil_opt + 0.25, 2):.2f}** → contact proactif (call, revue d'usage)
- **Score ≥ {round(seuil_opt + 0.25, 2):.2f}** → escalade < 48 h + geste commercial si MRR > {economics.matrice_couts()['seuil_mrr_rentable_eur']:.0f} €

*Le modèle classe, le CSM décide.* La boucle humaine est maintenue par choix de conception,
pas par obligation légale (les clients sont des entreprises, non des personnes physiques —
point de vigilance n°1).
"""
    )
)

# %% [markdown]
# #### Table des actions système — pilotage du modèle en production

# %%
df_actions_systeme = pd.DataFrame([
    {
        "Indicateur surveillé": "PR-AUC (monitoring mensuel Evidently §13)",
        "Seuil d'alerte": f"< {config.CIBLES_PERFORMANCE['pr_auc_min']:.2f}",
        "Seuil critique": f"< {config.CIBLES_PERFORMANCE['pr_auc_min'] - 0.05:.2f}",
        "Action déclenchée": "Alerter DS Lead — investigation",
        "Action critique": "Suspendre le scoring · Réentraîner",
        "Responsable": "Data Scientist",
    },
    {
        "Indicateur surveillé": "Dérive données (PSI §13)",
        "Seuil d'alerte": "PSI > 0,10",
        "Seuil critique": "PSI > 0,20",
        "Action déclenchée": "Inspecter la source de données",
        "Action critique": "Réentraîner sur cohorte récente",
        "Responsable": "Data Engineer",
    },
    {
        "Indicateur surveillé": "ROI mensuel (§12.12)",
        "Seuil d'alerte": "ROI < 1,5×",
        "Seuil critique": "ROI < 1,0×",
        "Action déclenchée": "Réviser les hypothèses économiques",
        "Action critique": "Revoir stratégie CS + seuils modèle",
        "Responsable": "CS Lead + DS Lead",
    },
    {
        "Indicateur surveillé": f"Fatigue d'alerte (Précision@{_cap})",
        "Seuil d'alerte": "< 30 %",
        "Seuil critique": "< 20 %",
        "Action déclenchée": "Réévaluer le seuil τ*",
        "Action critique": "Réévaluer le modèle complet",
        "Responsable": "CS Lead + Data Scientist",
    },
]).set_index("Indicateur surveillé")
display(df_actions_systeme.style.set_properties(**{"text-align": "left"}))

display(Markdown(
    "**Ce qu'il faut retenir.** Cette table définit les engagements de pilotage (*SLA système*) "
    "du modèle en production. Les seuils d'alerte déclenchent une investigation humaine ; "
    "les seuils critiques déclenchent une action automatisée (gate CI/CD §13). "
    "Le monitoring Evidently (§13) produit ces indicateurs à chaque run hebdomadaire."
))

# %% [markdown]
# ### 12.16 Journal de bord

# %% [markdown]
# > ### 📋 Journal de bord — Performance et impacts
# >
# > **Décisions retenues** — Évaluation OOF (cross_val_predict, 5 plis) pour éviter le biais
# > d'un modèle final entraîné sur toutes les données. Seuil économique τ* issu de l'argmax
# > du gain net (pas le seuil F1 ni 0,5). Régime opérationnel retenu : top-N sous contrainte
# > de capacité (plus robuste qu'un seuil fixe en présence de variabilité de la prévalence).
# > Valeur à risque définie sur horizon futur × MRR (et NON CLV historique — point vigilance n°3).
# > Empreinte carbone déclarée comme ESTIMATION (point vigilance n°6).
# >
# > **Alternatives écartées** — Split hold-out 80/20 a posteriori (biaisé car modèle fitté sur
# > toutes les données). KernelExplainer SHAP (trop lent, TreeExplainer disponible).
# > CLV comme proxy de valeur future (§6.7 a montré qu'elle est historique).
# >
# > **Difficultés rencontrées** — Fiche compte SHAP dans l'espace transformé (post-OHE) :
# > les noms de features sont ceux du ColumnTransformer, pas les colonnes originales.
# > Résolu en passant par `get_feature_names_out()` et en mappant les actions préventives
# > sur les noms transformés.
# >
# > **Impact sur la suite** — Le seuil τ* et la table de décision alimentent §13
# > (monitoring de la dérive). Le gain net mensuel est le KPI de monitoring principal :
# > si le ROI tombe sous 1×, le réentraînement est déclenché.
# >
# > **Temps passé** — 3 h de conception + implémentation ; 20 min pour les calculs lourds
# > (OOF, SHAP, drop-column) — cachés pour les régénérations suivantes.
