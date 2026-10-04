# %% [markdown]
# ## 12. Mesure de performance et impacts
#
# On veut savoir si le modèle tient ses promesses, et ce qu'il rapporte. Cette section mesure sa
# qualité technique (classement, calibration, latence), en tire une **règle de décision à deux
# niveaux** (surveiller large pour manquer peu de partants, appeler là où un geste rapporte le
# plus), explique ses prédictions, puis chiffre son impact en euros et ses limites. Toutes les
# mesures sont confrontées aux cibles fixées *a priori* en §8.

# %%
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from IPython.display import Markdown, display
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from churn_saas import config, economie, viz
from churn_saas.cache import charger, charger_ou_calculer
from churn_saas.features.build import ajouter_features_metier
from churn_saas.format_fr import entier, euros, nombre, pourcentage, styler_fr
from churn_saas.fuite import cribler_leurres
from churn_saas.models import economics
from churn_saas.models import evaluate as eval_mod
from churn_saas.models import explain as xpl
from churn_saas.models import regression as reg

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def _ok(condition: bool) -> str:
    return "✅" if condition else "❌"


def _liste(colonnes: list[str]) -> str:
    return ", ".join(f"`{c}`" for c in colonnes)


# %% [markdown]
# ### 12.1 Chargement des artefacts et jeu d'évaluation
#
# Le modèle final a été entraîné sur **toutes** les données (§9.15) : le noter sur ces mêmes
# données le flatterait (overfitting, surapprentissage). On le juge donc sur des prédictions
# **out-of-fold** (hors pli) : chaque compte est prédit par une copie du modèle entraînée sans
# lui (`cross_val_predict`, 5 plis). Elles couvrent tout le portefeuille, ce qu'exigent le bilan
# en euros et l'explicabilité. La note de référence reste celle du **jeu de test** (§9.14) :
# 20 % des comptes, mis de côté avant tout choix, notés une seule fois. On confronte les deux :
# la validation croisée de §9 a servi à choisir, ces prédictions servent à analyser, le test
# seul juge.

# %%
gold = pd.read_parquet(config.DONNEES_GOLD / "gold_dataset.parquet")
CIBLE = "churn"
X_brut = gold.drop(columns=config.COLONNES_INTERDITES, errors="ignore")
y = gold[CIBLE].astype(int)
X = ajouter_features_metier(X_brut).drop(columns=[CIBLE], errors="ignore")
mrr_brut = pd.to_numeric(gold["revenu_mensuel_recurrent_eur"], errors="coerce")
# Bilan économique sur tout le portefeuille : MRR manquant imputé par la médiane. Le batch de
# production, lui, n'impute pas (`mrr_disponible`, §10) ; l'écart est contrôlé en §12.6
_mrr_manquant = mrr_brut.isna().to_numpy()
MRR = mrr_brut.fillna(float(mrr_brut.median()))

modele_final = joblib.load(config.TABLES / "modele_final.joblib")
_classe_finale = type(modele_final[-1]).__name__
# §9 choisit la famille par calcul ; cette section en interprète une : correction d'intercept
# (§12.2.3), coefficients (§12.8.1), LinearExplainer (§12.8.3). Si un réentraînement retenait
# une autre famille, on s'arrête plutôt que d'afficher une interprétation fausse.
assert _classe_finale == "RegressionLogistiqueRecalibree", f"adapter §12 : {_classe_finale}"
nom_modele_final = "Régression logistique (recalibrée)"
_CV_EVAL = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)


def _calcul_probas_oof() -> np.ndarray:
    return cross_val_predict(
        clone(modele_final), X, y, cv=_CV_EVAL, method="predict_proba", n_jobs=-1
    )[:, 1]


proba_oof = np.asarray(charger_ou_calculer("probas_oof.joblib", _calcul_probas_oof)[0])
# Note du jeu de test, écrite par §9.14 : un modèle appris sur le seul jeu de développement
evaluation_test = charger("evaluation_test.json")
_test = evaluation_test["test"]
display(Markdown(f"""
**Ce qu'il faut retenir.** {entier(len(proba_oof))} comptes × {X.shape[1]} variables, prévalence
du churn **{pourcentage(y.mean(), 1)}** ({entier(y.sum())} partants), modèle final
`{_classe_finale}` retenu par calcul en §9. Le MRR manque pour {entier(_mrr_manquant.sum())}
comptes, imputé par la médiane ({euros(float(mrr_brut.median()))}/mois) dans le seul bilan
économique.
"""))

# %% [markdown]
# ### 12.2 Métriques techniques de classification
#
# Trois questions : le modèle **classe**-t-il bien les comptes (ROC-AUC) ? Sa liste de comptes
# signalés est-elle **fiable** malgré la rareté des partants (PR-AUC) ? Ses probabilités
# disent-elles **vrai** (calibration) ? Les métriques à seuil (recall, precision) sont traitées
# avec la règle de décision (§12.6).
#
# #### 12.2.1 Courbe ROC et AUC

# %%
fig_roc, auc_roc = eval_mod.courbe_roc(y, proba_oof, nom_modele=nom_modele_final)
plt.show()
display(Markdown(f"""
**Ce qu'il faut retenir.** ROC-AUC = **{nombre(auc_roc, 3)}** : un partant tiré au hasard reçoit
un score plus élevé qu'un fidèle tiré au hasard dans {pourcentage(auc_roc, 0)} des cas (50 % pour
un tirage au sort, la diagonale). Limite : l'axe horizontal rapporte les fausses alertes aux
{entier(int((y == 0).sum()))} fidèles, {nombre(float((y == 0).sum() / y.sum()), 1)} fois plus
nombreux que les partants : un faible taux de faux positifs peut encore faire beaucoup d'appels
inutiles.
"""))

# %% [markdown]
# #### 12.2.2 Courbe précision-rappel et PR-AUC (métrique de sélection)
#
# Quand les partants sont rares (class imbalance, déséquilibre de classes), on regarde la liste
# signalée : quelle part sont de vrais partants (precision, précision), quelle part des partants
# elle contient (recall, rappel). La PR-AUC résume ce compromis sur tous les seuils.

# %%
fig_pr, pr_auc = eval_mod.courbe_precision_rappel(y, proba_oof, nom_modele=nom_modele_final)
plt.show()
display(Markdown(f"""
**Ce qu'il faut retenir.** PR-AUC = **{nombre(pr_auc, 3)}**, contre {nombre(float(y.mean()), 2)}
pour un tirage au hasard (la prévalence, ligne horizontale) : le modèle multiplie par
{nombre(pr_auc / float(y.mean()), 1)} la qualité de la liste. C'est la métrique de sélection du
§9, car elle pénalise à la fois fausses alertes et partants manqués.
"""))

# %% [markdown]
# #### 12.2.3 Calibration probabiliste (score de Brier)
#
# Une probabilité de 30 % doit correspondre à 30 % de partants observés, car la valeur attendue
# d'un geste (§12.6) et la valeur à risque (§12.11) multiplient des probabilités par des euros.
# Le modèle est pondéré (`class_weight='balanced'`), ce qui gonfle ses probabilités ; son
# intercept (constante) est corrigé de ce décalage (§9.6). On mesure l'effet de la correction
# en recalculant les prédictions hors pli du **même** modèle sans elle.

# %%
fig_cal, brier = eval_mod.courbe_calibration(y, proba_oof, nom_modele=nom_modele_final)
plt.show()


def _calcul_probas_oof_non_corrigees() -> np.ndarray:
    # Même pipeline et mêmes hyperparamètres, sans la correction d'intercept
    modele_non_corrige = clone(modele_final)
    nom_etape, estimateur = modele_non_corrige.steps[-1]
    modele_non_corrige.steps[-1] = (nom_etape, LogisticRegression(**estimateur.get_params()))
    return cross_val_predict(
        modele_non_corrige, X, y, cv=_CV_EVAL, method="predict_proba", n_jobs=-1
    )[:, 1]


_brut, _ = charger_ou_calculer("probas_oof_non_corrigees.joblib", _calcul_probas_oof_non_corrigees)
_moy_brute = float(np.mean(_brut))
_brier_constant = float(y.mean() * (1 - y.mean()))
_ecart_pr_auc = average_precision_score(y, proba_oof) - average_precision_score(y, _brut)
display(Markdown(f"""
**Ce qu'il faut retenir.** Sans correction, le modèle annonce en moyenne
{pourcentage(_moy_brute)} de risque pour {pourcentage(y.mean())} de partants réels : toute valeur
en euros serait surestimée de {pourcentage(_moy_brute / y.mean() - 1, 0)}. Corrigé, il retrouve la
prévalence ({pourcentage(proba_oof.mean())}) ; son score de Brier passe de
{nombre(brier_score_loss(y, _brut), 4)} à **{nombre(brier, 4)}** et sa log loss (pénalité des
probabilités trop sûres d'elles) de {nombre(log_loss(y, _brut), 4)} à
{nombre(log_loss(y, proba_oof), 4)}. La bonne référence n'est pas 0,25 mais un modèle qui
annoncerait toujours la prévalence, p(1 − p) = {nombre(_brier_constant, 4)} : le modèle fait
{pourcentage(1 - brier / _brier_constant, 0)} mieux. La PR-AUC varie de
{nombre(_ecart_pr_auc, 4, signe=True)} : la correction déplace les probabilités sans changer
l'ordre des comptes. Limite : elle suppose une prévalence stable en production, d'où le suivi de
la probabilité moyenne prédite face au churn observé (§12.15).
"""))

# %% [markdown]
# #### 12.2.4 Confrontation aux cibles fixées a priori
#
# Les cibles de §8 ont été posées avant tout résultat (`config.CIBLES_PERFORMANCE`). Le statut se
# lit sur le **test** (§9.14), seule mesure sur des comptes qu'aucun choix n'a vus ; la colonne hors
# pli montre l'écart. Le recall au seuil de vigilance est confronté à sa cible en §12.6.

# %%
_c = config.CIBLES_PERFORMANCE
_atteintes = [
    _test["pr_auc"]["valeur"] >= _c["pr_auc_min"],
    _test["roc_auc"]["valeur"] >= _c["roc_auc_min"],
    _test["brier"]["valeur"] < _brier_constant,
]


def _test_ic(cle: str, decimales: int = 3) -> str:
    t = _test[cle]
    return (
        f"**{nombre(t['valeur'], decimales)}** [{nombre(t['ic_bas'], decimales)} ; "
        f"{nombre(t['ic_haut'], decimales)}]"
    )


_ecart_pr_test = _test["pr_auc"]["valeur"] - pr_auc
display(Markdown(f"""
| Métrique | Hors pli ({entier(len(y))} comptes) | Test (§9.14) [IC 95 %] | Cible ou référence | Statut (test) |
|---|---|---|---|---|
| PR-AUC (sélection) | {nombre(pr_auc, 3)} | {_test_ic('pr_auc')} | ≥ {nombre(_c['pr_auc_min'], 2)} | {_ok(_atteintes[0])} |
| ROC-AUC (co-principale) | {nombre(auc_roc, 3)} | {_test_ic('roc_auc')} | ≥ {nombre(_c['roc_auc_min'], 2)} | {_ok(_atteintes[1])} |
| Score de Brier (calibration) | {nombre(brier, 4)} | {_test_ic('brier', 4)} | < {nombre(_brier_constant, 4)} (modèle constant) | {_ok(_atteintes[2])} |

**Ce qu'il faut retenir.** {sum(_atteintes)} critère(s) sur 3 atteint(s) sur le test. La PR-AUC
du test s'écarte de {nombre(_ecart_pr_test, 3, signe=True)} de la mesure hors pli
{"et reste dans le même intervalle de confiance : la mesure hors pli, utilisée pour les euros, n'est pas sensiblement optimiste" if _test["pr_auc"]["ic_bas"] <= pr_auc <= _test["pr_auc"]["ic_haut"] else "et sort de l'intervalle de confiance du test : les euros calculés hors pli sont à lire comme une borne haute"}.
Le modèle classe bien et ses probabilités sont exploitables en euros ; reste à transformer un
score en décision (§12.4 à §12.6).
"""))
# Tableau standard des métriques au seuil de rentabilité (§12.5), réutilisé par le §13
courbe_gain, seuil_opt = economics.gain_par_seuil(y, proba_oof, MRR)
df_metriques = eval_mod.tableau_metriques(y, proba_oof, seuil_opt)

# %% [markdown]
# ### 12.3 Latence d'inférence — rappel de la mesure du §9.13
#
# La latence a été mesurée une seule fois, au §9.13, sur la même configuration : on relit son
# résultat (`reports/tables/latence_modele_final.json`) sans relancer la mesure.

# %%
rapport_latence = charger("latence_modele_final.json")
_lat_p95, _lat_batch = (
    rapport_latence["latence_unitaire_ms_p95"],
    rapport_latence["latence_batch_5k_s"],
)
_cible_unit, _cible_batch = _c["latence_unitaire_ms"], _c["latence_batch_5k_s"]
display(Markdown(f"""
**Ce qu'il faut retenir.** p95 unitaire **{nombre(_lat_p95, 1)} ms** (cible ≤ {_cible_unit} ms,
{_ok(_lat_p95 <= _cible_unit)}) ; batch (traitement par lot) de
{entier(rapport_latence['n_batch'])} comptes en **{nombre(_lat_batch, 1)} s** (cible ≤
{_cible_batch} s, {_ok(_lat_batch <= _cible_batch)}). L'inférence d'une régression logistique se
réduit au prétraitement et à un produit scalaire : le budget de l'API est surtout consommé par le
réseau et la persistance (§11.3).
"""))

# %% [markdown]
# ### 12.4 Matrice de coûts et hypothèses économiques
#
# Pour passer d'un score à une décision, il faut savoir ce que coûte chaque erreur. Tous les
# gains sont mesurés **par rapport à « ne rien faire »** (ne contacter personne vaut 0 €).

# %%
mat = economics.matrice_couts()
_hyp = config.HYPOTHESES_ECONOMIQUES
_cap = int(_hyp["capacite_gestes_mois"])
_marge, _horizon = pourcentage(_hyp["marge_brute_pct"], 0), _hyp["horizon_mois"]
_ts_ref = float(_hyp["taux_succes_retention"])
_mrr_median = float(MRR.median())
_sauvetage_median = _mrr_median * _horizon * _hyp["marge_brute_pct"] * _ts_ref
display(Markdown(f"""
| Paramètre | Valeur | Source |
|---|---|---|
| Marge brute | {_marge} | OpenView Partners SaaS Benchmarks 2023 |
| Coût d'un geste CSM | {euros(mat['cout_intervention_eur'])} | {euros(_hyp['cout_horaire_csm_eur'])}/h × {entier(_hyp['duree_geste_retention_h'])} h |
| Taux de succès d'un geste | {pourcentage(_ts_ref, 0)} | Gainsight 2023 Customer Success Industry Report |
| Horizon | {_horizon} mois | Durée contractuelle typique |
| Capacité CS | {_cap} gestes/mois | Cadrage : 3 CSM × 15 gestes (~20 % de leur temps, §2.3) |

| Issue | Valeur par rapport à « ne rien faire » |
|---|---|
| **VP** — partant contacté | MRR × {_horizon} × {_marge} × {pourcentage(_ts_ref, 0)} − {euros(mat['cout_intervention_eur'])} |
| **FP** — fidèle contacté | − {euros(mat['cout_fp_eur'])} (temps CSM perdu) |
| **FN** — partant manqué | 0 € : la perte (MRR × {_horizon} × {_marge}) est subie avec ou sans modèle |
| **VN** — fidèle laissé tranquille | 0 € |

**Ce qu'il faut retenir.** Un geste devient rentable dès {euros(mat['seuil_mrr_rentable_eur'])} de
MRR mensuel. Au MRR médian ({euros(_mrr_median)}/mois), un geste sur un partant rapporte en
espérance {euros(_sauvetage_median)} pour {euros(mat['cout_intervention_eur'])} de coût, soit
**{nombre(_sauvetage_median / mat['cout_intervention_eur'], 0)} pour 1** : un partant manqué coûte
bien plus cher qu'une fausse alerte, il faut **manquer peu de partants**. Mais la vraie contrainte
est le temps des CSM ({_cap} gestes/mois) : il faut aussi choisir *qui* appeler.
"""))

# %% [markdown]
# ### 12.5 Seuil de rentabilité hors contrainte de capacité et courbe de gain
#
# Première question : **si l'équipe n'avait pas de limite de temps**, à partir de quelle
# probabilité un geste serait-il rentable ? On contacte tout compte au-dessus d'un threshold
# (seuil) τ et on calcule le gain net ; τ\* maximise ce gain.

# %%
fig_gain = economics.tracer_gain_par_seuil(courbe_gain, seuil_opt)
plt.show()
_opt = courbe_gain.loc[courbe_gain["gain_net_eur"].idxmax()]
valeur_attendue_oof = economie.valeur_attendue_intervention(proba_oof, MRR.to_numpy())
_n_rentables = int((valeur_attendue_oof > 0).sum())
display(Markdown(f"""
**Ce qu'il faut retenir.** τ\\* = **{nombre(seuil_opt, 2)}** est bas parce qu'un geste coûte peu
face à ce qu'il sauve : il faudrait contacter {entier(_opt['n_alertes'])} comptes (precision
{pourcentage(float(_opt['precision']), 0)}, recall {pourcentage(float(_opt['rappel']), 0)}) pour un
gain de {euros(float(_opt['gain_net_eur']), signe=True)}, soit
**{nombre(_opt['n_alertes'] / _cap, 0)} fois** la capacité mensuelle. τ\\* n'est donc pas une règle
applicable : c'est un argument de dimensionnement ({entier(_n_rentables)} comptes justifieraient
un geste rentable, l'équipe en traite {_cap}).
"""))

# %% [markdown]
# ### 12.6 Règle de décision à deux niveaux — vigilance et priorisation sous capacité
#
# Un seul seuil ne peut pas servir deux objectifs opposés : ne laisser filer presque aucun partant,
# et ne mobiliser les CSM que là où un appel rapporte le plus. On les sépare :
#
# 1. **Vigilance** — le seuil le plus exigeant qui détecte au moins
#    `config.RECALL_CIBLE_VIGILANCE` des partants hors pli (`economics.seuil_pour_recall`). Les
#    comptes au-dessus passent en palier `SURVEILLANCE` (API, batch) : actions automatisées à coût
#    quasi nul, fiche signalée au CSM. Ce niveau vise les **faux négatifs**.
# 2. **Appels** — parmi les comptes en vigilance, les `capacite` comptes de plus forte **valeur
#    attendue** positive : P(churn) × MRR × horizon × marge × taux de succès − coût du geste.
#
# Le seuil de vigilance est **appris** : fixé sur les prédictions hors pli, persisté dans
# `reports/tables/seuil_vigilance.json` (réécrit dès qu'il ne correspond plus aux prédictions
# courantes), relu par `economie.niveau_risque`. Deux contrôles disent s'il généralise : le
# **jeu de test** (§9.14), où un seuil fixé sur le seul développement est appliqué à des comptes
# jamais vus, et une mesure pli par pli (fixé sur 4 plis, mesuré sur le 5ᵉ), qui donne le bruit.

# %%
_vigilance_fraiche = {
    **economics.seuil_pour_recall(y, proba_oof),
    "recall_cible": float(config.RECALL_CIBLE_VIGILANCE),
}
vigilance, _ = charger_ou_calculer(
    economie.ARTEFACT_SEUIL_VIGILANCE,
    lambda: _vigilance_fraiche,
    forcer=charger(economie.ARTEFACT_SEUIL_VIGILANCE) != _vigilance_fraiche,
)
seuil_vigilance = float(vigilance["seuil"])
_recall_hors_choix = []
for _idx_choix, _idx_mesure in _CV_EVAL.split(X, y):
    _s = economics.seuil_pour_recall(y.iloc[_idx_choix], proba_oof[_idx_choix])["seuil"]
    _partants = y.iloc[_idx_mesure].to_numpy() == 1
    _recall_hors_choix.append(float((proba_oof[_idx_mesure][_partants] >= _s).mean()))
_recall_min, _recall_moyen = float(_c["recall_vigilance_min"]), float(np.mean(_recall_hors_choix))
_recall_test = float(_test["recall"]["valeur"])
display(Markdown(f"""
Seuil de vigilance **{nombre(seuil_vigilance, 3)}** : {entier(vigilance['n_signales'])} comptes
signalés ({pourcentage(vigilance['part_signalee'], 0)} du portefeuille), recall
{pourcentage(vigilance['recall'], 1)}, precision {pourcentage(vigilance['precision'], 1)},
{entier(vigilance['n_faux_negatifs'])} partants sous le seuil. **Recall sur le jeu de test**
(§9.14) : **{pourcentage(_recall_test, 1)}** [{pourcentage(_test['recall']['ic_bas'], 1)} ;
{pourcentage(_test['recall']['ic_haut'], 1)}], cible a priori ≥ {pourcentage(_recall_min, 0)} :
{_ok(_recall_test >= _recall_min)}. Recall sur le pli qui n'a pas fixé le seuil : de
{pourcentage(min(_recall_hors_choix), 1)} à {pourcentage(max(_recall_hors_choix), 1)}, moyenne
{pourcentage(_recall_moyen, 1)}.

**Ce qu'il faut retenir.** Par construction, le seuil atteint sa cible sur les données qui l'ont
fixé. Le test dit s'il la tient sur des comptes qu'aucun choix n'a vus : c'est lui qui est
confronté à la cible de §8 ; l'écart entre plis donne le bruit à attendre en production.
"""))

# %% [markdown]
# **Matrices de confusion aux deux seuils** — rentabilité τ\* (§12.5) et vigilance : partants
# détectés (VP) ou manqués (FN), fidèles signalés à tort (FP) ou laissés tranquilles (VN).

# %%
_matrices = {}
for _nom_seuil, _seuil in (("τ*", seuil_opt), ("vigilance", seuil_vigilance)):
    _, _cm = eval_mod.matrice_confusion(y, (proba_oof >= _seuil).astype(int), _seuil)
    plt.show()
    _matrices[_nom_seuil] = dict(zip(("VN", "FP", "FN", "VP"), _cm.ravel().tolist(), strict=True))
_m_opt, _m_vig = _matrices["τ*"], _matrices["vigilance"]
display(Markdown(f"""
**Ce qu'il faut retenir.** Au seuil τ\\* = {nombre(seuil_opt, 2)}, on ne manque que
{entier(_m_opt['FN'])} partants, au prix de {entier(_m_opt['FP'])} fausses alertes. Le seuil de
vigilance ({nombre(seuil_vigilance, 2)}) signale {entier(_m_vig['VP'] + _m_vig['FP'])} comptes pour
{entier(_m_vig['FN'])} partants manqués : {entier(_m_vig['FN'] - _m_opt['FN'])} manqués de plus
pour {entier(_m_opt['FP'] - _m_vig['FP'])} fausses alertes de moins. Le compromis est acceptable
car la vigilance ne mobilise pas de temps CSM : une fausse alerte n'y coûte presque rien, alors
qu'un partant hors vigilance ne reçoit aucune attention.
"""))

# %% [markdown]
# **Ordre des appels (niveau 2).** À budget de gestes identique, trois manières de remplir
# l'agenda : par **probabilité**, par **valeur attendue**, et une **référence sans modèle**, les
# plus gros comptes par MRR, ce que ferait une équipe sans score. C'est la bonne baseline (référence
# de base) pour mesurer l'apport du modèle : « ne rien faire » le flatterait.

# %%
bilan_valeur = economics.gain_sous_capacite(y, proba_oof, MRR, capacite=_cap)
bilan_proba = economics.gain_sous_capacite(y, proba_oof, MRR, capacite=_cap, classement="proba")
bilan_sans_modele = economics.gain_sous_capacite(y, proba_oof, MRR, capacite=_cap, classement="mrr")
# Le batch n'impute pas le MRR : combien des comptes retenus ici ont un MRR imputé ?
_selection_valeur = economie.selection_sous_capacite(valeur_attendue_oof, _cap)
_n_imputes = int(_mrr_manquant[_selection_valeur].sum())
tableau_classements = pd.DataFrame(
    [bilan_sans_modele, bilan_proba, bilan_valeur],
    index=pd.Index(["sans modèle : plus gros MRR", "par probabilité", "par valeur attendue"]),
)[["n_contactes", "n_churners", "precision", "mrr_churners_couverts", "cout_gestes", "gain_net"]]
display(
    styler_fr(
        tableau_classements,
        {"n_contactes": entier, "n_churners": entier, "precision": pourcentage}
        | {c: euros for c in ("mrr_churners_couverts", "cout_gestes", "gain_net")},
    )
)
display(Markdown(f"""
**Ce qu'il faut retenir.** Classer par valeur attendue rapporte
**{euros(bilan_valeur['gain_net'])}** contre {euros(bilan_proba['gain_net'])} par probabilité :
moins de partants ({entier(bilan_valeur['n_churners'])} contre
{entier(bilan_proba['n_churners'])}) mais bien plus de MRR
({euros(bilan_valeur['mrr_churners_couverts'])}/mois contre
{euros(bilan_proba['mrr_churners_couverts'])}/mois), car le classement par probabilité remplit
l'agenda de petits comptes presque perdus. Sans modèle, appeler les {_cap} plus gros comptes
rapporterait déjà {euros(bilan_sans_modele['gain_net'])} : une large part du gain vient de la
taille des comptes. La part due au modèle est l'écart,
**{euros(bilan_valeur['gain_net'] - bilan_sans_modele['gain_net'], signe=True)}** par mois de
gestes (fourchette en §12.12).
{"Aucun compte retenu n'a de MRR imputé : la sélection est celle du batch." if _n_imputes == 0 else f"⚠️ {entier(_n_imputes)} comptes retenus ont un MRR imputé : le batch, qui les écarte, en sélectionnerait d'autres."}
"""))

# %% [markdown]
# **Bilan de la règle à deux niveaux** (`economics.table_deux_niveaux`) : comptes et partants
# couverts à chaque niveau, et partants restés hors du niveau (faux négatifs résiduels, valorisés
# à leur perte sèche MRR × horizon × marge).

# %%
deux_niveaux = economics.table_deux_niveaux(y, proba_oof, MRR, capacite=_cap).set_index("niveau")
display(
    styler_fr(
        deux_niveaux.drop(columns="seuil"),
        {c: entier for c in ("n_comptes", "n_churners", "n_faux_negatifs")}
        | {c: pourcentage for c in ("part_portefeuille", "recall", "precision")}
        | {c: euros for c in ("valeur_faux_negatifs_eur", "gain_net_eur")},
    )
)
_n1, _n2 = deux_niveaux.iloc[0], deux_niveaux.iloc[1]
# Les comptes retenus par valeur attendue sur tout le portefeuille sont-ils tous en vigilance ?
_hors_vigilance = int((proba_oof[_selection_valeur] < seuil_vigilance).sum())
display(Markdown(f"""
**Ce qu'il faut retenir.** La vigilance couvre {pourcentage(_n1['recall'], 0)} des partants en
signalant {pourcentage(_n1['part_portefeuille'], 0)} du portefeuille ;
{entier(_n1['n_faux_negatifs'])} restent hors de toute attention, soit
{euros(_n1['valeur_faux_negatifs_eur'])} de marge perdue sur {_horizon} mois. Les appels n'en
touchent que {entier(_n2['n_churners'])} (recall {pourcentage(_n2['recall'], 0)}) : c'est la
capacité CS, pas le modèle, qui limite la détection par appel, d'où l'intérêt du premier niveau.
{"Les comptes de plus forte valeur attendue sont tous en vigilance : le niveau 2 retient exactement la liste ci-dessus." if _hors_vigilance == 0 else f"⚠️ {entier(_hors_vigilance)} comptes de plus forte valeur attendue sont sous le seuil de vigilance : le niveau 2 les remplace par des comptes signalés."}
**Règle retenue** : chaque nuit, le batch classe les comptes (colonnes `rang_priorite`,
`action_cs`, §10) ; chaque mois, les CSM appellent les premiers.
"""))

# %% [markdown]
# ### 12.7 Analyse de sensibilité — taux de succès et capacité CS
#
# Deux hypothèses pèsent sur le gain : le taux de succès d'un geste (source externe, non mesuré
# ici) et la capacité de l'équipe (hypothèse de cadrage). On recalcule le gain pour chaque couple.

# %%
fig_sensib, df_sensib = economics.sensibilite_capacite(y, proba_oof, MRR)
plt.show()
_ts_min = float(df_sensib["taux_succes_retention"].min())
_gains = df_sensib.set_index(["taux_succes_retention", "capacite"])["gain_net_eur"]
_gain_pire = float(_gains[(_ts_min, _cap)])
_cap_sup = next(c for c in sorted(df_sensib["capacite"].unique()) if c > _cap)
_gain_marginal = (_gains[(_ts_ref, _cap_sup)] - _gains[(_ts_ref, _cap)]) / (_cap_sup - _cap)
display(Markdown(f"""
**Ce qu'il faut retenir.** Au taux de succès le plus prudent ({pourcentage(_ts_min, 0)}), la
règle reste {"rentable" if _gain_pire > 0 else "**déficitaire**"} ({euros(_gain_pire, signe=True)}
par mois de gestes) : la recommandation ne dépend pas d'une hypothèse optimiste. Le gain croît
avec la capacité, de moins en moins vite : entre {_cap} et {_cap_sup} gestes, un geste de plus
rapporte {euros(_gain_marginal)}, à comparer au coût d'un CSM supplémentaire. Le taux de succès
reste à **mesurer** dès le déploiement (§12.15).
"""))

# %% [markdown]
# ### 12.8 Explicabilité — importance, permutation et SHAP
#
# Sur quoi le modèle s'appuie-t-il ? Trois mesures de feature importance (importance des
# variables), de la plus rapide à la plus rigoureuse : les coefficients, l'importance de
# permutation avec son intervalle de confiance à 95 % (IC₉₅), et SHAP (valeurs de Shapley), qui
# répartit le score de chaque compte entre ses variables.
#
# #### 12.8.1 Importance native du modèle — coefficients
#
# Pour une régression logistique, c'est la valeur absolue des coefficients, calculés après
# prétraitement (numériques standardisées, catégorielles en one-hot encoding, encodage
# disjonctif). Limites : un coefficient de modalité (passage 0 → 1) ne se compare pas à celui
# d'une variable standardisée (un écart-type), et deux variables corrélées se partagent leur
# effet de façon instable.

# %%
df_imp_native = xpl.importance_native(modele_final)
fig_imp_native, ax_imp_native = viz.figure(
    "importance_native",  # nom stable : figure réutilisée hors du notebook
    "Importance native — |coefficients| de la régression logistique (top 15)",
    taille=(9.0, 6.0),
)
_top15 = df_imp_native.head(15).iloc[::-1]
ax_imp_native.barh(_top15["feature"], _top15["importance"], color=viz.COULEUR_CHURN, alpha=0.8)
ax_imp_native.set_xlabel("|coefficient|")
viz.sauvegarder(fig_imp_native)
plt.show()

# %% [markdown]
# **Ce qu'il faut retenir.** Les coefficients donnent un premier classement, lisible dans le
# modèle, mais ils mélangent des unités et éclatent une variable catégorielle en autant de lignes
# que de modalités : on ne conclut pas sur eux seuls.
#
# #### 12.8.2 Importance de permutation
#
# Si l'on mélange au hasard les valeurs d'une variable et que la PR-AUC chute, le modèle s'en
# sert. Le mélange est répété 20 fois pour obtenir un IC₉₅. La mesure, faite sur les données
# d'entraînement du modèle final, dit ce qu'il **utilise**, pas ce qui généralise (d'où le
# drop-column du §12.9).


# %%
def _calcul_perm_imp() -> pd.DataFrame:
    return xpl.importance_permutation(modele_final, X, y, n_repeats=20, n_jobs=-1)


# Le drapeau « significatif » (IC₉₅ > 0) est recalculé pour qu'un cache suive la règle en vigueur
df_perm_imp = xpl.marquer_significativite(
    charger_ou_calculer("permutation_importance.parquet", _calcul_perm_imp)[0]
)
fig_perm, ax_perm = viz.figure(
    "importance_permutation", "Importance de permutation — top 15 (IC à 95 %)", taille=(9.0, 6.0)
)
_p = df_perm_imp.head(15).iloc[::-1]
_moy = _p["importance_moyenne"]
ax_perm.barh(
    _p["feature"],
    _moy,
    xerr=[_moy - _p["ic95_bas"], _p["ic95_haut"] - _moy],
    color=viz.COULEUR_CHURN,
    alpha=0.8,
    capsize=3,
)
ax_perm.tick_params(axis="y", labelsize=9)
ax_perm.axvline(0, color="#888888", linewidth=1.2, linestyle="--")
ax_perm.set_xlabel("Réduction de PR-AUC après permutation (moyenne ± IC 95 %)")
viz.sauvegarder(fig_perm)
plt.show()
_n_non_sig = int((~df_perm_imp["significatif"]).sum())
_n_nulles = int((df_perm_imp["ic95_haut"] <= 0).sum())
display(Markdown(f"""
**Ce qu'il faut retenir.** {len(df_perm_imp) - _n_non_sig} variables sur {len(df_perm_imp)} ont
un effet démontré (IC₉₅ entièrement au-dessus de 0). Parmi les {_n_non_sig} autres, {_n_nulles}
n'apportent rien (IC₉₅ ≤ 0) et {_n_non_sig - _n_nulles} ont un effet possible mais non démontré.
Une variable sans effet peut être un leurre **ou** la jumelle redondante d'une autre : le §12.9
tranche.
"""))

# %% [markdown]
# #### 12.8.3 SHAP — attributions additives (global et local)

# %%
_idx_shap = np.random.default_rng(config.RANDOM_SEED + 1).choice(len(X), 1000, replace=False)
X_shap = X.iloc[_idx_shap]
explainer_shap, shap_values = xpl.valeurs_shap(
    modele_final, X_shap, nom_cache="shap_values_12.joblib"
)
# Les valeurs SHAP sont calculées après le ColumnTransformer : le graphique reçoit ce même espace
_sous_pipeline = modele_final[:-1]
_noms_trans = [str(n) for n in _sous_pipeline.get_feature_names_out()]
X_shap_trans = pd.DataFrame(_sous_pipeline.transform(X_shap), columns=_noms_trans)
fig_shap_summary, ax_shap = viz.figure(
    "shap_summary", "SHAP — importance et direction d'effet (top 15 variables)", taille=(9.0, 7.0)
)
shap.summary_plot(shap_values, X_shap_trans, max_display=15, plot_size=None, show=False)
ax_shap.set_xlabel("Valeur SHAP (effet sur la sortie du modèle)")
fig_shap_summary.axes[-1].set_ylabel("Valeur de la variable")
fig_shap_summary.axes[-1].set_yticklabels(["Faible", "Élevée"])
fig_shap_summary.tight_layout()
viz.sauvegarder(fig_shap_summary)
plt.show()

# %% [markdown]
# **Ce qu'il faut retenir.** Chaque point est un compte : sa position est ce que la variable
# ajoute ou retire à son score (en log-odds, logarithme de la cote de churn), sa couleur la valeur
# de la variable (rouge = élevée). Un nuage rouge à droite signifie qu'une valeur élevée pousse
# vers le churn. Pour un modèle linéaire, le `LinearExplainer` est exact : coefficient × écart de
# la valeur à la moyenne. Contrairement aux coefficients bruts, SHAP tient compte de la dispersion
# réelle des valeurs et s'explique compte par compte (fiches du §12.11).

# %% [markdown]
# ### 12.9 Verdict sur les leurres — 3 preuves convergentes
#
# Le cahier des charges signale des leurres suspectés (`config.COLONNES_LEURRES_SUSPECTES`).
# `commentaire_csm`, classé en data leakage (fuite de données) possible au §6.6, est exclu du
# modèle : son importance serait nulle par construction et ne prouverait rien. Une importance
# faible ne suffit pas non plus (une variable redondante avec une jumelle en a une aussi) : trois
# preuves doivent converger.
#
# 1. **Association marginale** avec la cible : V de Cramér, p-value corrigée par
#    Benjamini-Hochberg (comme en §6.10).
# 2. **Permutation** (§12.8.2) : la borne haute de l'IC₉₅ reste sous 1 point de PR-AUC (0,01).
# 3. **Drop-column** (retrait de colonne) : variable retirée, modèle réentraîné, PR-AUC mesurée sur
#    un jeu tenu à l'écart : perte inférieure à 1 point.
#
# On juge l'**ampleur** de l'effet, pas sa seule significativité : sur les données d'entraînement,
# une variable de pur bruit dégrade toujours un peu le score quand on la mélange.

# %%
_leurres_presents = [c for c in config.COLONNES_LEURRES_SUSPECTES if c in X.columns]
_leurres_exclus = [c for c in config.COLONNES_LEURRES_SUSPECTES if c not in X.columns]


def _calcul_criblage() -> pd.DataFrame:
    return cribler_leurres(gold, cible="churn", colonnes=_leurres_presents)


def _calcul_drop_col() -> pd.DataFrame:
    return xpl.importance_drop_column(modele_final, X, y, colonnes=_leurres_presents, forcer=True)


tableau_criblage = charger_ou_calculer("criblage_leurres_12.parquet", _calcul_criblage)[0]
# Un artefact en cache peut couvrir un périmètre plus large : on s'aligne sur les leurres testables
tableau_criblage = tableau_criblage.loc[tableau_criblage.index.isin(_leurres_presents)]
df_drop_col = charger_ou_calculer("drop_column_12.parquet", _calcul_drop_col)[0]
df_verdict = xpl.verdict_leurres(tableau_criblage, df_perm_imp, df_drop_col)
_preuves = ["association_significative", "permutation_negligeable", "drop_negligeable"]
display(styler_fr(df_verdict[[*_preuves, "verdict", "raisonnement"]]))
_rang_perm = {c: r + 1 for r, c in enumerate(df_perm_imp["feature"]) if c in _leurres_presents}
_confirmes = df_verdict.index[df_verdict["verdict"] == "LEURRE CONFIRMÉ"].tolist()
_non_confirmes = df_verdict.index[df_verdict["verdict"] != "LEURRE CONFIRMÉ"].tolist()
# Les importances sont en unités de PR-AUC ; ×100 pour les exprimer en points
_ic_haut_max = 100 * float(
    df_perm_imp.set_index("feature").loc[_leurres_presents, "ic95_haut"].max()
)
_delta_drop = 100 * df_drop_col.loc[_leurres_presents, "delta_pr_auc"]
display(Markdown(f"""
**Ce qu'il faut retenir.** Sur {len(_leurres_presents)} leurres testables,
**{len(_confirmes)} confirmé(s)** par les trois preuves ({_liste(_confirmes) or "aucun"}) ; non
confirmé(s) : {_liste(_non_confirmes) or "aucun"}.

- **Association** : {entier(int(tableau_criblage['significatif_BH'].sum()))} sur
  {entier(len(tableau_criblage))} significative(s) après correction ; redondance maximale
  {nombre(float(tableau_criblage['max_redondance'].max()), 3)} (jumelle probable au-delà de
  {nombre(config.SEUIL_REDONDANCE, 2)}) : aucune jumelle ne masque leur effet.
- **Permutation** : borne haute de l'IC₉₅ au plus {nombre(_ic_haut_max, 2)} point (seuil : 1),
  rangs {", ".join(f"{entier(r)} (`{c}`)" for c, r in _rang_perm.items())} sur
  {entier(len(df_perm_imp))}.
- **Drop-column** : Δ PR-AUC de {nombre(float(_delta_drop.min()), 2, signe=True)} à
  {nombre(float(_delta_drop.max()), 2, signe=True)} point (seuil : 1) ; un Δ négatif signifie que
  le modèle est meilleur sans la variable.

{"Le retrait anticipé au §10.3 est confirmé : le modèle déployé, réentraîné sans ces variables, est le bon ; elles restent dans le gold pour que la preuve reste reproductible." if not _non_confirmes else "⚠️ Le modèle déployé au §10.3 a retiré des variables non confirmées ici : il doit être réentraîné en les réintégrant."}
Exclue(s) en amont (§6.6) : {_liste(_leurres_exclus) or "aucune"}.
"""))

# %% [markdown]
# ### 12.10 Régression CLV — confrontation aux cibles §8, RMSE, MAE, R² et résidus
#
# Cible secondaire : `valeur_vie_client_eur`, évaluée en euros sur un jeu de test de 20 % contre
# les cibles de §8.2 (R² minimal, gain de MAE sur la baseline médiane, alerte de fuite). Règle
# posée avant le tableau : parmi les modèles qui atteignent toutes les cibles, le R² le plus élevé.
#
# **Contrôle ajouté après observation, et signalé comme tel.** Une première version retenait un
# modèle qui prédisait des CLV négatives. Or §8.1 pose la CLV comme strictement positive : une
# prédiction ≤ 0 est une valeur impossible. D'où le critère « aucune prédiction ≤ 0 », qui ne porte
# sur aucune métrique de performance. La règle est codée une seule fois
# (`reg.selectionner_champion_clv`), partagée avec §14.

# %%
X_reg = reg.features_regression(gold)  # sans churn ni sante_compte_fin_periode (fuites)
y_reg = pd.to_numeric(gold["valeur_vie_client_eur"], errors="coerce").dropna()
X_reg = X_reg.loc[y_reg.index]


def _calcul_regression() -> dict:
    return reg.entrainer_modeles_clv(X_reg, y_reg)


resultats_reg = charger_ou_calculer("regression_clv.joblib", _calcul_regression)[0]
tableau_reg, _champion_reg = reg.selectionner_champion_clv(resultats_reg)
_coche = {True: "✅", False: "❌"}
display(
    pd.DataFrame(
        {
            "RMSE (€)": tableau_reg["rmse"].map(entier),
            "MAE (€)": tableau_reg["mae"].map(entier),
            "R²": tableau_reg["r2"].map(lambda v: nombre(v, 3)),
            "Gain MAE": tableau_reg["gain_mae"].map(lambda v: pourcentage(v, 0)),
            "Prédictions ≤ 0": tableau_reg["n_pred_non_positives"].map(entier),
            f"R² ≥ {nombre(_c['clv_r2_min'], 2)}": tableau_reg["ok_r2"].map(_coche),
            f"Gain MAE ≥ {pourcentage(_c['clv_gain_mae_min'], 0)}": tableau_reg["ok_mae"].map(
                _coche
            ),
            f"R² ≤ {nombre(_c['clv_r2_alerte_fuite'], 2)}": tableau_reg["ok_fuite"].map(_coche),
            "CLV > 0": tableau_reg["ok_domaine"].map(_coche),
            "Conforme": tableau_reg["conforme"].map(_coche),
        }
    ).style.set_properties(**{"text-align": "left"})
)
_conformes_reg = tableau_reg.index[tableau_reg["conforme"]].tolist()
# Aucun modèle conforme : on le dit, et le meilleur R² n'est montré qu'à titre de diagnostic
_nom_best_reg = _champion_reg if _champion_reg is not None else str(tableau_reg["r2"].idxmax())
_m_best = tableau_reg.loc[_nom_best_reg]
_hors_domaine = tableau_reg.index[
    tableau_reg[["ok_r2", "ok_mae", "ok_fuite"]].all(axis=1) & ~tableau_reg["ok_domaine"]
].tolist()
_mae_conforme = str(tableau_reg.loc[_conformes_reg, "mae"].idxmin()) if _conformes_reg else None
# Structure de la cible : CLV additive ou multiplicative en MRR ?
_mrr_reg = pd.to_numeric(gold.loc[y_reg.index, "revenu_mensuel_recurrent_eur"], errors="coerce")
_dispo = _mrr_reg.notna() & (_mrr_reg > 0)
_corr_euros = float(np.corrcoef(y_reg[_dispo], _mrr_reg[_dispo])[0, 1])
_corr_log = float(np.corrcoef(np.log(y_reg[_dispo]), np.log(_mrr_reg[_dispo]))[0, 1])
_verdict_reg = (
    f"{len(_conformes_reg)} modèle(s) sur {len(tableau_reg)} satisfont toutes les cibles ; "
    f"champion (R² maximal parmi eux) : **`{_nom_best_reg}`**, R² = {nombre(_m_best['r2'], 3)}, "
    f"MAE = {euros(_m_best['mae'])} (gain de {pourcentage(_m_best['gain_mae'], 0)} sur la "
    "baseline médiane)."
    if _champion_reg is not None
    else "❌ **Aucun modèle ne satisfait toutes les cibles** : la régression CLV n'est pas "
    f"déployable en l'état ; `{_nom_best_reg}` n'est montré qu'en diagnostic."
)
display(Markdown(f"""
**Ce qu'il faut retenir.** {_verdict_reg} Écartés par le seul contrôle de domaine :
{_liste(_hors_domaine) or "aucun"}. La corrélation CLV–MRR passe de {nombre(_corr_euros, 2)} en
euros à **{nombre(_corr_log, 2)} en log** : la CLV est **multiplicative** (CLV ≈ MRR × durée). Un
modèle additif en euros, dominé par les très gros comptes, s'ajuste sur eux avec des corrections
négatives qui font passer de petits comptes sous zéro ; les modèles sur log(CLV) restent positifs
par construction. Les spécifications log-log, Poisson ou Gamma sont posées comme hypothèses du
prochain cycle (§13.1), et non ajoutées après lecture des résultats.
{"" if _mae_conforme in (None, _nom_best_reg) else f"La plus faible MAE revient à `{_mae_conforme}` : le R² est dominé par les gros comptes, la MAE reflète le compte typique ; la règle n'est pas changée après coup."}
Ces écarts reposent sur un seul découpage 80/20, non testé (§14.2).
"""))

# %%
_y_test_reg = np.asarray(resultats_reg[_nom_best_reg]["y_test"], dtype=float)
_y_pred_reg = np.asarray(resultats_reg[_nom_best_reg]["y_pred"], dtype=float)
fig_res = reg.figure_residus(_y_test_reg, _y_pred_reg, _nom_best_reg)
plt.show()
_erreurs_abs = np.abs(_y_pred_reg - _y_test_reg)
_q90_reel = float(np.quantile(_y_test_reg, 0.90))
_gros = _y_test_reg >= _q90_reel
_n_pred_non_pos = int((_y_pred_reg <= 0).sum())
display(Markdown(f"""
**Ce qu'il faut retenir.** Le champion se trompe en moyenne de
**{euros(float(_erreurs_abs[_gros].mean()))}** sur les 10 % de comptes à plus forte CLV
(≥ {euros(_q90_reel)}), contre {euros(float(_erreurs_abs[~_gros].mean()))} sur les autres :
l'erreur croît avec la valeur du compte, d'où l'éventail du nuage de résidus.
{f"⚠️ {entier(_n_pred_non_pos)} prédiction(s) ≤ 0 : diagnostic seulement." if _n_pred_non_pos else ""}
{"Entraîné sur log(CLV), il prédit plutôt la médiane que la moyenne et tend à sous-estimer les euros, biais corrigeable au prochain cycle." if "log" in _nom_best_reg else ""}
La CLV prédite est un ordre de grandeur pour le CSM, pas une valeur comptable.
"""))

# %% [markdown]
# ### 12.11 Valeur à risque et fiches comptes
#
# Combien un compte peut-il faire perdre s'il part ? Formule unique du projet
# (`churn_saas.economie`) :
#
# $$\text{valeur\_à\_risque}(i) = P(\text{churn}_i) \times \text{MRR}_i \times H \times M$$
#
# avec $H$ l'horizon (12 mois) et $M$ la marge brute. **Et non** P(churn) × CLV : §6.7 a montré
# que la CLV, prospective, intègre déjà le risque de départ ; la multiplier par P(churn)
# compterait ce risque deux fois.

# %%
valeurs_risque = reg.valeur_a_risque(proba_oof, MRR.values)
display(Markdown(f"""
**Ce qu'il faut retenir.** Le portefeuille expose **{euros(valeurs_risque.sum())}** sur
{_horizon} mois, dont {euros(valeurs_risque[_selection_valeur].sum())} pour les {_cap} comptes
appelés (§12.6). La valeur à risque combine le signal du modèle et le poids du compte : à 40 % de
probabilité et 5 000 €/mois, un compte expose {euros(economie.valeur_a_risque(0.40, 5_000.0))},
contre {euros(economie.valeur_a_risque(0.80, 100.0))} à 80 % et 100 €/mois. C'est le premier
qu'il faut appeler.
"""))

# %% [markdown]
# **Fiches comptes — deux clients réels à risque élevé** (plus forte valeur à risque parmi les
# comptes à P(churn) ≥ 50 %). La probabilité est la prédiction hors pli ; les facteurs SHAP
# expliquent le modèle final, celui qu'appellerait la production. Les facteurs sont présentés sous
# leur libellé métier, la contribution entre parenthèses en log-odds. Le palier vient de la même
# fonction que l'API et le batch, avec le seuil de vigilance du §12.6 : notebook et production
# parlent le même langage.


# %%
def _facteurs(indices: list[int], contributions: np.ndarray, compte: pd.Series) -> str:
    lignes = [
        f"- {xpl.libelle_facteur(_noms_trans[i], compte)} "
        f"({nombre(float(contributions[i]), 2, signe=True)})"
        for i in indices
    ]
    return "\n".join(lignes) or "— aucun"


_, shap_values_all = xpl.valeurs_shap(modele_final, X, nom_cache="shap_values_all.joblib")
_candidats = np.flatnonzero(proba_oof >= 0.50)
for _rang, _pos in enumerate(_candidats[np.argsort(-valeurs_risque[_candidats])][:2], start=1):
    _proba, _sv, _compte = float(proba_oof[_pos]), shap_values_all[_pos], X.iloc[_pos]
    _hausse = [int(i) for i in np.argsort(_sv)[::-1][:3] if _sv[i] > 0]
    _baisse = [int(i) for i in np.argsort(_sv)[:2] if _sv[i] < 0]
    display(Markdown(f"""
---
**🗂️ Fiche compte #{_rang} — `{gold['client_id'].iloc[_pos]}`**

| Indicateur | Valeur |
|---|---|
| Probabilité de churn | **{pourcentage(_proba, 1)}** |
| Palier (seuil de vigilance §12.6) | **{economie.niveau_risque(_proba)}** |
| MRR | {euros(float(MRR.iloc[_pos]))}/mois |
| **Valeur à risque ({_horizon} mois)** | **{euros(float(valeurs_risque[_pos]))}** |

**Ce qui augmente le risque :**
{_facteurs(_hausse, _sv, _compte)}

**Ce qui le réduit :**
{_facteurs(_baisse, _sv, _compte)}

**Action préventive suggérée** (facteur dominant) :
> {xpl.action_preventive(_noms_trans[_hausse[0] if _hausse else int(np.argmax(_sv))])}
"""))

# %% [markdown]
# ### 12.12 KPI métier et ROI
#
# On évalue le système en euros : combien de marge il permet de sauver chaque mois, à quel coût,
# et surtout **combien de plus qu'une équipe sans modèle**. Les KPI (indicateurs clés) sont ceux
# de la règle retenue (§12.6). Le ROI (retour sur investissement) de la cohorte compte la marge
# sauvée sur tout l'horizon par les gestes d'un mois, comme en §2.4.

# %%
_n_churners_detectes = int(bilan_valeur["n_churners"])
_precision_top_n = bilan_valeur["precision"]
_roi = bilan_valeur["roi"]
# Part du gain due au modèle : écart à la référence « plus gros MRR », sur la plage de taux de
# succès du §12.7
attribution = economics.gain_attribuable_au_modele(y, proba_oof, MRR, capacite=_cap)
_gain_modele = attribution["gain_attribuable"]
_fourchette = (
    f"{euros(attribution['gain_attribuable_min'])} à {euros(attribution['gain_attribuable_max'])}"
)
display(Markdown(f"""
| KPI | Valeur | Lecture |
|---|---|---|
| Partants sous surveillance | {pourcentage(vigilance['recall'], 0)} ({entier(vigilance['n_signales'])} comptes) | niveau 1 : recall au seuil de vigilance |
| Comptes appelés par mois | {_cap} | capacité CS (§2.3) |
| Precision des appels | {pourcentage(_precision_top_n, 1)} ({_n_churners_detectes} partants) | niveau 2 ; hasard : {pourcentage(y.mean(), 0)} |
| Fausses alertes parmi les appels | {pourcentage(1.0 - _precision_top_n, 1)} | fatigue d'alerte ; hasard : {pourcentage(1 - y.mean(), 0)} |
| Valeur sauvée par la cohorte | {euros(bilan_valeur['valeur_sauvee'])} | marge sur {_horizon} mois, pour {euros(bilan_valeur['cout_gestes'])} de gestes |
| Gain net sans modèle | {euros(attribution['reference']['gain_net'])} | les {_cap} plus gros MRR |
| **Gain attribuable au modèle** | **{euros(_gain_modele)}** | {_fourchette} (taux de succès {pourcentage(attribution['taux_succes_min'], 0)} à {pourcentage(attribution['taux_succes_max'], 0)}) |
| ROI de la cohorte | {nombre(_roi, 1)}× | (valeur sauvée − coût) / coût, contre « ne rien faire » |

**Ce qu'il faut retenir.** Le KPI qui mesure la valeur du modèle est le **gain attribuable** : ce
que la règle rapporte de plus qu'une équipe qui appellerait ses plus gros comptes sans score, à
budget identique. Le ROI de la cohorte ne compte que le temps CSM (ni remises, ni coût du projet,
chiffré plus bas) : sa valeur très élevée tient à l'asymétrie de la matrice (§12.4) et au poids de quelques gros comptes.
Il sert à détecter un effondrement, avec des seuils posés sur des repères économiques : alerte
sous 1× (le gain ne couvre plus une erreur d'un facteur 2 sur le taux de succès), critique sous 0
(perte nette). Le niveau acceptable de fatigue d'alerte sera fixé avec l'équipe CS au pilote.
"""))

# %% [markdown]
# **Le gain d'un mois ne se répète pas.** Le mois suivant, les meilleurs comptes ont déjà été
# contactés : multiplier le premier mois par 12 serait faux. On simule un an : chaque mois, les
# meilleurs comptes **non encore contactés**, avec et sans modèle, à portefeuille et scores figés
# (faute d'une seconde extraction, §6.11). La date des départs étant inconnue, deux scénarios : un
# partant contacté est encore client (retenu, prudent), ou les départs s'étalent sur l'année.

# %%
gain_annuel = economics.gain_annuel_attribuable(y, proba_oof, MRR, capacite=_cap)
_mensuel = gain_annuel["mensuel"]
fig_gain_annuel = economics.tracer_gain_sur_annee(_mensuel)
plt.show()
# Si les départs survenaient à la date anniversaire, leur taux croîtrait à son approche
_avant_anniv = (12 - pd.to_numeric(gold["anciennete_mois"], errors="coerce") % 12) % 12
_prudent = gain_annuel["annuel_departs_etales"] >= gain_annuel["annuel"]
display(Markdown(f"""
**Ce qu'il faut retenir.** Sur un an, le gain dû au modèle cumule
**{euros(gain_annuel['annuel'])}** (de {euros(gain_annuel['annuel_min'])} à
{euros(gain_annuel['annuel_max'])} selon le taux de succès), soit
{nombre(gain_annuel['annuel'] / gain_annuel['premier_mois'], 1)} fois le premier mois, et non 12.
Le modèle trouve {entier(int(_mensuel['n_churners_modele'].sum()))} partants dans l'année contre
{entier(int(_mensuel['n_churners_reference'].sum()))} sans lui, mais l'équipe sans modèle finit
par appeler les mêmes gros comptes. Une barre négative n'est donc pas une perte (à gauche, les deux
stratégies restent gagnantes) : ce mois-là, l'équipe sans modèle atteint des gros partants que le
modèle a appelés plus tôt. Ce rattrapage fait culminer le cumul au mois
{int(_mensuel['gain_attribuable_cumule'].idxmax())} : l'apport du modèle tient surtout à ce qu'il
**atteint les gros partants plus tôt**. Avec des départs étalés, le gain dû au modèle vaut
{euros(gain_annuel['annuel_departs_etales'])} : le chiffre retenu
{"est prudent" if _prudent else "est une borne haute (⚠️)"}. La date de départ ne se déduit pas de
l'anniversaire de souscription : taux de départ {pourcentage(float(y[_avant_anniv <= 2].mean()), 0)}
à 2 mois ou moins de cette date, {pourcentage(float(y[_avant_anniv >= 9].mean()), 0)} à 9 mois ou
plus.
"""))

# %% [markdown]
# **Le projet couvre-t-il son propre coût ?** Le ROI de la cohorte ne compte que le temps des CSM.
# Un investissement se juge sur son coût complet : développement amorti, infrastructure du
# scénario B (§11.7) et maintenance (`config.COUTS_PROJET`). On l'oppose au seul gain attribuable
# sur un an : sans modèle, l'équipe appellerait quand même ses plus gros comptes.

# %%
projet = economics.bilan_projet(gain_annuel)
_cp = config.COUTS_PROJET
_jours_build = projet["cout_build"] / float(_cp["cout_journalier_eur"])
_retour = projet["mois_retour"]
_ts_eq = projet["taux_succes_equilibre"]
display(Markdown(f"""
| Poste | Montant | Hypothèse |
|---|---|---|
| Développement | {euros(projet['cout_build'])} | {entier(_jours_build)} jours à {euros(_cp['cout_journalier_eur'])}, amortis sur {entier(_cp['duree_amortissement_ans'])} ans |
| Exploitation | {euros(projet['cout_run_annuel'])} par an | infrastructure {euros(_cp['infra_mensuel_eur'])}/mois, maintenance {entier(_cp['jours_maintenance_mois'])} jours/mois |
| **Coût annuel complet** | **{euros(projet['cout_annuel'])}** | amortissement + exploitation |
| Gain attribuable sur un an | {euros(gain_annuel['annuel'])} | simulation ci-dessus |
| **ROI du projet** | **{nombre(projet['roi_projet'], 1)}×** | (gain attribuable − coût complet) / coût complet |
| Délai de retour | {f"mois {_retour}" if _retour else "au-delà de 12 mois"} | gain cumulé ≥ développement + exploitation engagée |
| Taux de succès d'équilibre | {pourcentage(_ts_eq, 1)} | contre {pourcentage(_ts_ref, 0)} retenu |

**Ce qu'il faut retenir.** Coût complet compris, le projet rapporte
{nombre(projet['roi_projet'], 1)} fois sa dépense annuelle et se rembourse
{f"dès le mois {_retour}" if _retour else "au-delà de la première année (⚠️)"}. L'indicateur le plus
robuste est le taux de succès d'équilibre : le projet ne couvrirait plus son coût que si moins de
{pourcentage(_ts_eq, 1)} des gestes réussissaient, contre {pourcentage(_ts_ref, 0)} supposés.
{"Une erreur d'un facteur 10 sur les coûts, à confirmer par le contrôle de gestion, ne changerait pas la décision." if projet['roi_projet'] > 9 else "⚠️ La rentabilité dépend des hypothèses de coût, à confirmer par le contrôle de gestion."}
Le risque principal reste la valeur des plus gros comptes, contrôlée ci-dessous.
"""))

# %% [markdown]
# **Dépendance aux très gros comptes, puis confrontation au cadrage.** Le bilan pourrait ne reposer
# que sur quelques gros comptes, ou sur des MRR erronés : part du gain des cinq premiers comptes,
# gain sans le 1 % de plus gros MRR, cohérence du MRR avec sièges × prix catalogue (§7.6). Puis
# la formule du §2.4, chaque paramètre estimé avant les données remplacé par sa valeur mesurée.

# %%
concentration = economics.concentration_du_gain(y, proba_oof, MRR, capacite=_cap)
_ratio_catalogue = mrr_brut / (gold["sieges_souscrits"] * gold["prix_mensuel_par_siege_eur"])
_ratio_retenus = _ratio_catalogue.to_numpy()[_selection_valeur]
_robuste = concentration["gain_attribuable_sans_gros"] >= 0.5 * _gain_modele


def _gain_cadrage(precision: float, prevalence: float, mrr_median: float) -> float:
    """Formule du §2.4 : gain annuel dû au ciblage par rapport à des gestes faits au hasard."""
    gain_par_geste = mrr_median * _horizon * _hyp["marge_brute_pct"] * _ts_ref
    return (precision - prevalence) * _cap * 12 * gain_par_geste


_cadrage = config.PORTEFEUILLE_CADRAGE
_t0, _m0 = float(_cadrage["taux_churn"]), float(_cadrage["mrr_median_eur"])
_p0 = float(_c["pr_auc_min"])  # précision attendue au cadrage
_t1, _m1, _p1 = float(y.mean()), float(mrr_brut.median()), float(_precision_top_n)
_g0, _g1 = _gain_cadrage(_p0, _t0, _m0), _gain_cadrage(_p1, _t1, _m1)
display(Markdown(f"""
| Contrôle | Résultat |
|---|---|
| Part du gain du 1ᵉʳ mois apportée par les {entier(concentration['n_top'])} premiers comptes | {pourcentage(concentration['part_top'], 0)} |
| Gain dû au modèle (1ᵉʳ mois) : tous comptes / sans les {entier(concentration['n_exclus'])} au-delà de {euros(concentration['seuil_mrr'])}/mois | {euros(_gain_modele)} / {euros(concentration['gain_attribuable_sans_gros'])} |
| MRR / (sièges × prix catalogue) : comptes retenus / portefeuille (1ᵉʳ–99ᵉ centile) | {nombre(np.nanmin(_ratio_retenus), 2)} à {nombre(np.nanmax(_ratio_retenus), 2)} / {nombre(_ratio_catalogue.quantile(0.01), 2)} à {nombre(_ratio_catalogue.quantile(0.99), 2)} |

| Formule du §2.4 | Taux de churn | MRR médian | Precision des contactés | Gain annuel |
|---|---|---|---|---|
| Cadrage | {pourcentage(_t0, 1)} | {euros(_m0)} | {pourcentage(_p0, 1)} | {euros(_g0)} |
| Mesuré | {pourcentage(_t1, 1)} | {euros(_m1)} | {pourcentage(_p1, 1)} | **{euros(_g1)}** |

**Ce qu'il faut retenir.** Le gain est concentré, mais
{"il ne repose pas sur les seuls comptes géants : sans eux, les comptes suivants prennent leur place." if _robuste else "⚠️ il repose sur quelques comptes géants."}
Le MRR des comptes retenus est cohérent avec leurs sièges et le prix catalogue : ce ne sont pas
des erreurs de saisie, mais ces montants restent à valider par la Finance (§14.3). Mesuré, le gain
du cadrage vaut {nombre(_g1 / _g0, 2)} fois l'estimation (MRR médian : facteur
{nombre(_m1 / _m0, 2)} ; ciblage : facteur {nombre((_p1 - _t1) / (_p0 - _t0), 2)}). Cette formule
valorise chaque compte au MRR médian contre des appels au hasard ; la simulation, au MRR réel et
contre les plus gros comptes, donne {euros(gain_annuel['annuel'])} sur un an.
"""))

# %% [markdown]
# ### 12.13 Empreinte carbone (ESTIMATION)
#
# Sous WSL2, CodeCarbon n'accède pas aux compteurs de consommation du processeur : il **estime**
# l'énergie par la puissance nominale (TDP) et le mix électrique du pays. Le fichier
# `codecarbon_emissions.csv` reçoit une ligne par optimisation Optuna (§9) ; le budget de §8.2
# porte sur **une** session, à laquelle on confronte la dernière.

# %%
df_carbon = pd.read_csv(config.TABLES / "codecarbon_emissions.csv", parse_dates=["timestamp"])
_dernier = df_carbon.sort_values("timestamp").iloc[-1]
_co2_session_g = float(_dernier["emissions"]) * 1000
_co2_cumul_g = float(df_carbon["emissions"].sum()) * 1000
_budget_co2_g = _c["co2_entrainement_max_g"]
_G_CO2_PAR_KM_VOITURE = 210  # ordre de grandeur, voiture thermique moyenne
display(Markdown(f"""
| Indicateur | Valeur estimée |
|---|---|
| Dernière optimisation ({_dernier['timestamp']:%d/%m/%Y}) | **{nombre(_co2_session_g, 3)} g CO₂e** en {nombre(float(_dernier['duration']), 0)} s |
| Cumul des {entier(len(df_carbon))} optimisations du projet | {nombre(_co2_cumul_g, 3)} g CO₂e, soit {nombre(_co2_cumul_g / _G_CO2_PAR_KM_VOITURE * 1000, 1)} m en voiture |
| Facteur d'émission appliqué ({_dernier['country_name']}) | {nombre(_co2_cumul_g / float(df_carbon['energy_consumed'].sum()), 0)} g CO₂e/kWh |
| Budget a priori (§8.2) | ≤ {entier(_budget_co2_g)} g CO₂e par session |
| Statut | {_ok(_co2_session_g <= _budget_co2_g)} ({pourcentage(_co2_session_g / _budget_co2_g, 2)} du budget) |

**Ce qu'il faut retenir.** L'empreinte est **estimée**, pas mesurée, et se présente comme telle.
Le verdict reste robuste : même le cumul de toutes les relances n'atteint qu'une infime part du
budget. Réserve : la comparaison des modèles (§9.3) n'est pas instrumentée. Le cache contribue à
la sobriété : seul `make notebook-full` relance l'optimisation.
"""))

# %% [markdown]
# ### 12.14 Analyse d'erreurs, équité par sous-groupe et limites assumées
#
# Savoir où le modèle se trompe dit au commanditaire où ne pas se fier au score. On regarde
# d'abord les partants que la vigilance laisse passer (faux négatifs à son seuil), sur les 8
# variables qui les distinguent le plus des partants détectés, en écarts-types du portefeuille
# pour comparer des unités différentes.

# %%
df_erreurs = eval_mod.analyse_erreurs(X, y, proba_oof, seuil_vigilance)
_FN, _VP, _VN = "Faux Négatifs (FN)", "Vrais Positifs (VP)", "Vrais Négatifs (VN)"
_cols_num = [c for c in df_erreurs.columns if c not in ["n_observations", "proba_pred_moyenne"]]
_ecart_type = X[_cols_num].std().replace(0, np.nan)


def _ecart_standardise(segment: str) -> pd.Series:
    ecart = df_erreurs.loc[_FN, _cols_num] - df_erreurs.loc[segment, _cols_num]
    return (ecart / _ecart_type).dropna()


ecart_fn_vp = _ecart_standardise(_VP)
_top_ecarts = ecart_fn_vp.abs().sort_values(ascending=False).head(8).index.tolist()
tableau_erreurs = df_erreurs[["n_observations", "proba_pred_moyenne"] + _top_ecarts].T
tableau_erreurs["écart FN − VP (en écarts-types)"] = ecart_fn_vp.reindex(tableau_erreurs.index)
display(styler_fr(tableau_erreurs, precision=2))
_n_fn = int(df_erreurs.loc[_FN, "n_observations"])
_ecart_moyen_vp = float(ecart_fn_vp[_top_ecarts].abs().mean())
_ecart_moyen_vn = float(_ecart_standardise(_VN)[_top_ecarts].abs().mean())
_traits = ", ".join(
    f"`{c}` ({'plus élevé' if ecart_fn_vp[c] > 0 else 'plus faible'}, "
    f"{nombre(abs(ecart_fn_vp[c]), 1)} écart-type)"
    for c in _top_ecarts[:3]
)
display(Markdown(f"""
**Ce qu'il faut retenir.** Au seuil de vigilance, {entier(_n_fn)} partants échappent à la
surveillance, avec une probabilité moyenne de
{pourcentage(float(df_erreurs.loc[_FN, 'proba_pred_moyenne']), 0)} contre
{pourcentage(float(df_erreurs.loc[_VP, 'proba_pred_moyenne']), 0)} pour les détectés. Ce qui les
distingue le plus : {_traits}. Ils s'écartent en moyenne de {nombre(_ecart_moyen_vp, 2)}
écart-type des partants détectés, contre {nombre(_ecart_moyen_vn, 2)} des fidèles.
{"Ils ressemblent donc aux fidèles et partent sans trace dans les variables disponibles : baisser le seuil ne les rattraperait qu'au prix de nombreuses fausses alertes ; seule une information nouvelle (motif de départ, signaux commerciaux, §13) le pourrait." if _ecart_moyen_vn < _ecart_moyen_vp else "Ils restent plus proches des partants détectés : un seuil plus bas en rattraperait une partie, au prix de fausses alertes."}
"""))

# %% [markdown]
# #### Mesure des biais par sous-groupe
#
# Protocole fixé en §4.4 avant l'entraînement : prédictions hors pli, modalités normalisées comme
# dans le pipeline. Le taux de vrais positifs (TPR, part des partants signalés) et l'écart de
# calibration sont mesurés au seuil qui signale autant de comptes qu'il y a de partants, fixé par
# §4.4 pour que la comparaison entre segments ne dépende pas d'un réglage métier ; la ROC-AUC par
# modalité complète la mesure sans aucun seuil.

# %%
_SEUILS_EQUITE = config.EQUITE
seuil_equite = float(np.quantile(proba_oof, 1 - y.mean()))
equite = eval_mod.equite_par_sous_groupe(
    gold[["pays", "taille_entreprise", "secteur"]],
    y,
    proba_oof,
    seuil_equite,
    positifs_min=int(_SEUILS_EQUITE["churners_min"]),
)
_lignes_synthese = []
for _attribut, _bloc in equite[equite["interpretable"]].groupby(level="attribut"):
    _ecart_tpr = float(_bloc["tpr"].max() - _bloc["tpr"].min())
    _calib_max = float(_bloc["ecart_calibration"].abs().max())
    _lignes_synthese.append(
        {
            "attribut": _attribut,
            "écart de TPR max − min": _ecart_tpr,
            "modalité la moins détectée": _bloc["tpr"].idxmin()[1],
            "ROC-AUC min": float(_bloc["roc_auc"].min()),
            "écart de calibration max": _calib_max,
            "revue requise": _ecart_tpr > _SEUILS_EQUITE["ecart_tpr_max"]
            or _calib_max > _SEUILS_EQUITE["ecart_calibration_max"],
        }
    )
synthese_equite = pd.DataFrame(_lignes_synthese).set_index("attribut")


def _points(v: float) -> str:
    return nombre(100 * v, 1) + " pts"


display(
    styler_fr(
        synthese_equite,
        {"écart de TPR max − min": _points, "écart de calibration max": _points}
        | {
            "ROC-AUC min": lambda v: nombre(v, 3),
            "revue requise": lambda v: "⚠️ oui" if v else "non",
        },
    )
)
_attributs_revue = synthese_equite.index[synthese_equite["revue requise"]].tolist()
_n_non_interpretables = int((~equite["interpretable"]).sum())
_pire = synthese_equite["écart de TPR max − min"].idxmax()
_attr_auc_min = synthese_equite["ROC-AUC min"].idxmin()
display(Markdown(f"""
**Ce qu'il faut retenir.** Seuil de l'audit : {nombre(seuil_equite, 3)}.
{"Sur les trois attributs, les écarts restent sous les seuils de §4.4 : à churn égal, chaque pays, taille et secteur est détecté avec une fiabilité comparable." if not _attributs_revue else f"⚠️ Les écarts dépassent les seuils de §4.4 pour {_liste(_attributs_revue)} : revue requise avant déploiement, écart consigné au registre des risques (R01)."}
Plus grand écart de TPR : `{_pire}` ({_points(synthese_equite.loc[_pire, 'écart de TPR max − min'])},
modalité la moins détectée : {synthese_equite.loc[_pire, 'modalité la moins détectée']}). ROC-AUC
la plus basse d'une modalité : {nombre(synthese_equite.loc[_attr_auc_min, 'ROC-AUC min'], 3)}
(`{_attr_auc_min}`), contre {nombre(auc_roc, 3)} au global. {entier(_n_non_interpretables)}
modalité(s) ont moins de {entier(int(_SEUILS_EQUITE['churners_min']))} partants : non
interprétées, leur intervalle de confiance étant aussi large que l'écart recherché.
"""))

# %% [markdown]
# #### Limites assumées du système
#
# | Limite | Nature | Mitigation |
# |---|---|---|
# | Un seul jeu de test, de taille modeste | Note finale bruitée ; le modèle livré, appris sur toutes les données, n'est pas celui qui a été noté | Intervalle de confiance par bootstrap (rééchantillonnage, §9.14) ; écart test / hors pli mesuré (§12.2.4) ; même famille et mêmes réglages, seul l'échantillon d'apprentissage grandit |
# | Seuil de vigilance appris sur un portefeuille figé | Le recall tenu dépend de la distribution des scores | Recall suivi à chaque vague de renouvellements ; seuil recalculé et persisté à chaque réentraînement |
# | Partants sans signal | Les faux négatifs résiduels ressemblent aux fidèles | Nouvelles sources (motif de départ, signaux commerciaux, §13) |
# | Taux de succès (30 %) et capacité CS (45 gestes) | Hypothèses externes ou de cadrage | Sensibilité §12.7 ; mesure dès le déploiement |
# | Gain concentré sur quelques gros comptes | MRR très asymétrique | Contrôles §12.12 ; validation du MRR par la Finance |
# | MRR manquant | Imputé par la médiane dans ce bilan | Le batch n'impute pas (`mrr_disponible`, §10) ; §12.6 vérifie que la liste d'appels n'en dépend pas |
# | Correction de calibration | Suppose une prévalence stable | Probabilité moyenne prédite vs churn observé à chaque vague ; réestimation au réentraînement |
# | CLV de méthode non documentée | Prospective, intègre déjà le risque (§6.7) | Valeur à risque = MRR × horizon (§12.11) ; calcul à documenter avec la Finance |
# | `commentaire_csm` exclu | Date de rédaction inconnue : fuite non exclue (§6.6) | Réintégrable si le CRM horodate chaque commentaire |
#
# **Ce qu'il faut retenir.** Aucune limite n'invalide le modèle, mais trois conditionnent les
# euros (taux de succès, capacité CS, MRR des gros comptes) et une conditionne la promesse de
# détection (le recall au seuil de vigilance) : le suivi post-déploiement les mesure en premier.

# %% [markdown]
# ### 12.15 Note de restitution au commanditaire
#
# Ce que le commanditaire doit lire pour décider du déploiement, puis la table de décision du CSM
# et la table des actions de pilotage en production. Chiffres recalculés à chaque exécution.

# %%
_n_auto = int(vigilance["n_signales"]) - int(_n2["n_comptes"])
display(Markdown(f"""
---

## 📋 Note de restitution au commanditaire

**Destinataires :** CS Lead · Direction commerciale · DSI
**Objet :** système de détection du churn — décision de déploiement demandée

### Ce qui marche

- **Surveillance large** : chaque nuit, {entier(vigilance['n_signales'])} comptes
  ({pourcentage(vigilance['part_signalee'], 0)} du portefeuille) passent en vigilance ; ils
  regroupent {pourcentage(_recall_test, 0)} des partants d'un jeu de test que ni le modèle ni le
  seuil n'avaient vu (§9.14, §12.6). {entier(_n_auto)} d'entre eux relèvent d'actions automatisées.
- **Appels ciblés** : chaque mois, les {_cap} comptes où un geste rapporte le plus ; sur
  l'historique, {_n_churners_detectes} partants réels parmi eux (precision
  {pourcentage(_precision_top_n, 0)}).
- **Valeur** : face à une équipe qui appellerait ses {_cap} plus gros comptes sans score, le modèle
  apporte **{euros(_gain_modele)}** de marge espérée par mois de gestes ({_fourchette} selon le taux
  de succès), **{euros(gain_annuel['annuel'])}** sur un an.
- **Performance vérifiée sur des comptes jamais vus** : PR-AUC {nombre(_test['pr_auc']['valeur'], 2)},
  ROC-AUC {nombre(_test['roc_auc']['valeur'], 2)} sur le jeu de test (cibles {nombre(_c['pr_auc_min'], 2)}
  et {nombre(_c['roc_auc_min'], 2)}).
- **Exploitation** : latence conforme aux cibles de §8 ; équité :
  {"aucun écart au-delà des seuils de §4.4" if not _attributs_revue else "revue requise pour " + _liste(_attributs_revue)} ;
  {len(_confirmes)} leurre(s) sur {len(_leurres_presents)} confirmé(s) (§12.9).

### Ce qui ne marche pas (encore)

- {entier(_n1['n_faux_negatifs'])} partants restent hors surveillance, sans signal exploitable
  dans les données actuelles (§12.14).
- Le taux de succès d'un geste ({pourcentage(_ts_ref, 0)}) est une **hypothèse** : le déploiement
  doit le mesurer.
- La CLV n'a pas de méthode documentée ; `commentaire_csm` reste exclu faute d'horodatage.

### Les décisions demandées

1. **Déployer** le scoring nocturne (batch Prefect, §10) avec la règle à deux niveaux.
2. **Réviser chaque trimestre** les hypothèses économiques (comité CS Lead + Direction commerciale).

*Le modèle classe, le CSM décide* : la boucle humaine est un choix de conception, pas une
obligation légale (les clients sont des entreprises, §4.1).
"""))
_td = economics.table_de_decision(capacite=_cap, seuil_rentabilite=seuil_opt)
display(
    _td[["zone", "action", "description", "responsable", "periodicite"]].style.set_properties(
        **{"text-align": "left"}
    )
)
display(Markdown(f"""
| Indicateur surveillé | Seuil d'alerte | Seuil critique | Action | Action critique | Responsable |
|---|---|---|---|---|---|
| **PR-AUC** (à chaque vague de renouvellements, §13.7) | < {nombre(_c['pr_auc_min'], 2)} | < {nombre(_c['pr_auc_critique'], 2)} | Comité ad hoc sous 72 h | Comparaison au champion précédent sur la même cohorte ; retour arrière s'il fait mieux | Data Scientist + CS Lead |
| **Recall au seuil de vigilance** (à chaque vague) | < {pourcentage(_recall_min, 0)} | — | Recalcul du seuil sur les dernières prédictions | — | Data Scientist |
| **Dérive des entrées** (PSI hebdomadaire, §13.3) | 0,10 ≤ PSI < 0,20 | PSI ≥ 0,20 | Noté pour le comité trimestriel | Corriger la source si panne, sinon réentraîner | Data Scientist + CS Lead |
| **Part d'`ALERTE_ROUGE`** (batch nocturne, §13.8) | — | Écart > {entier(_c['ecart_alerte_rouge_rollback_pts'])} points | — | Retour au champion précédent après une promotion récente | Data Scientist |
| **ROI de la cohorte** (mensuel, §12.12) | < 1× | < 0 (perte nette) | Réviser les hypothèses économiques | Revoir la stratégie CS | CS Lead + Data Scientist |
| **Fatigue d'alerte** (precision des {_cap} appels) | < 30 % | < 20 % | Réviser la priorisation | Réévaluer le modèle | CS Lead + Data Scientist |

**Ce qu'il faut retenir.** La table de décision dit au CSM quoi faire de chaque compte ; la table
des actions dit à l'équipe quoi faire quand un indicateur se dégrade. Un signal convoque un
comité, il ne réentraîne jamais seul (§2.6) : une dérive peut venir d'une panne de données qu'un
réentraînement apprendrait au lieu de la corriger. Seul le retour au champion précédent après une
promotion récente est immédiat ; tout réentraînement passe par la gate (règle de promotion, §10.4,
§13.8). Rythmes : `ALERTE_ROUGE` chaque nuit, PSI chaque semaine, ROI chaque mois ; PR-AUC, recall
et precision à chaque vague de renouvellements, quand le churn devient observable.
"""))

# %% [markdown]
# > ### 📋 Journal de bord — Performance et impacts
# >
# > **Décisions retenues** — Cibles confrontées au jeu de test de §9.14 (note de référence) ;
# > évaluation hors pli (5 plis) pour les euros et l'explicabilité, le modèle final ayant vu
# > toutes les données. Règle à deux niveaux : vigilance au seuil qui tient un recall de 80 %
# > hors pli (persisté dans `seuil_vigilance.json`, relu par l'API et le batch), puis appels aux
# > comptes de plus forte valeur attendue dans la capacité CS. Recall du seuil contrôlé sur le
# > jeu de test et sur le pli qui ne l'a pas fixé. Calibration mesurée avant/après correction
# > d'intercept. Gain mesuré contre une référence sans modèle, simulé sur un an, avec contrôle de
# > concentration. Valeur à risque = P(churn) × MRR × horizon × marge. Régression CLV jugée
# > contre les cibles a priori et un contrôle de domaine signalé comme amendement.
# >
# > **Alternatives écartées** — Un seuil de capacité seul (laisse filer la plupart des partants).
# > Classement par probabilité seule (agenda rempli de petits comptes). Gain annuel = 12 × premier mois. CLV
# > comme valeur exposée (double compte du risque). Tableau des niveaux de service, redondant avec
# > §12.2.4, §12.3 et la table des actions.
# >
# > **Difficultés rencontrées** — Un seuil économique unique (τ\*) a d'abord été retenu : il
# > signalait bien plus de comptes que l'équipe n'en traite, d'où la règle à deux niveaux. Le
# > premier modèle CLV (meilleur R²) prédisait des valeurs négatives : contrôle de domaine ajouté
# > en amendement. Jeu de test modeste : intervalles de confiance par bootstrap.
# >
# > **Impact sur la suite** — Seuil de vigilance et priorisation appliqués par le batch (§10) ;
# > recall à chaque vague, probabilité moyenne prédite et ROI de la cohorte rejoignent le
# > monitoring (§13).
