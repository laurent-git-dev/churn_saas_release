# %% [markdown]
# ## 8. Choix du modèle
#
# Avant d'entraîner quoi que ce soit, on décide **quoi comparer, à quelle aune et sous quelles
# contraintes**. Tout est fixé ici, avant le moindre score : le type de sortie (8.1), les cibles
# a priori (8.2), les contraintes d'exploitation et de sobriété (8.3, 8.4), l'achat d'une
# solution du marché (8.5), les trois modèles retenus (8.6), les références à battre (8.7) et la
# règle qui désignera le champion (8.8). Les scores sont en §9.
#
# La cellule ci-dessous recalcule les faits cités et vérifie que les paramètres du texte sont
# ceux du code : si l'un change, le notebook s'arrête plutôt que d'afficher un texte périmé.

# %%
import pandas as pd
from IPython.display import Markdown, display

from churn_saas import config
from churn_saas.features.build import ajouter_features_metier
from churn_saas.format_fr import entier, euros, nombre, nombre_tableau, pourcentage, styler_fr
from churn_saas.models.train import SEUIL_CSAT, SEUIL_INACTIVITE_JOURS, protocole_validation

# Prévalence recalculée comme en §6 ; doublons exclus (un client = une ligne)
_prevalence = float(pd.to_numeric(df_brut.drop_duplicates()["churn"]).dropna().mean())

# Variables en entrée des modèles, construites exactement comme en §9 (avant encodage)
_gold = pd.read_parquet(config.DONNEES_GOLD / "gold_dataset.parquet")
_X = ajouter_features_metier(_gold.drop(columns=config.COLONNES_INTERDITES, errors="ignore"))
_n_variables = _X.drop(columns=["churn"], errors="ignore").shape[1]
# Lignes d'apprentissage d'un pli : (k − 1)/k du jeu de développement (test exclu, §9.1), k = 5
_obs_par_variable_pli = len(_X) * (1 - config.PART_TEST) * 4 / 5 / _n_variables

_cv, _c = protocole_validation(), config.CIBLES_PERFORMANCE
_valeurs_citees = {
    "Plis × répétitions": ((_cv.cvargs["n_splits"], _cv.n_repeats), (5, 3)),
    "Seuil d'inactivité de B1 (jours)": (SEUIL_INACTIVITE_JOURS, 30),
    "Seuil CSAT de B1": (SEUIL_CSAT, 2),
    "Gain minimal de PR-AUC du plus complexe": (_c["gain_pr_auc_min_complexite"], 0.02),
}
for _nom, (_reel, _cite) in _valeurs_citees.items():
    assert _reel == _cite, f"{_nom} : code = {_reel}, texte = {_cite} — mettre §8 à jour"
print(
    f"Prévalence du churn : {pourcentage(_prevalence, 1)} ; {entier(_n_variables)} variables en "
    f"entrée, soit {entier(_obs_par_variable_pli)} observations par variable dans un pli "
    f"d'apprentissage ; r(CLV, MRR) = {nombre(res_clv['r_mrr'], 2)} (§6.7)."
)

# %% [markdown]
# ### 8.1 Deux cibles, deux tâches : classification probabiliste et régression
#
# On veut savoir quels comptes risquent de partir et à quel point : la sortie attendue est donc
# une **probabilité**, pas un simple oui/non. Le jeu porte deux cibles, deux tâches supervisées :
#
# | | Cible principale | Cible secondaire |
# |---|---|---|
# | Variable | `churn` (1 = résilie) | `valeur_vie_client_eur` (euros) |
# | Tâche | Classification binaire : estimer P(churn = 1 \| X) | Régression |
# | Sortie | **Probabiliste** : score entre 0 et 1, d'où un classement et une décision au seuil | **Déterministe** : un montant en euros |
# | Calibration | Requise : un score de 0,7 doit correspondre à ~70 % de départs | Sans objet |
# | Métriques | PR-AUC, ROC-AUC, recall (rappel) ; Brier en contrôle | MAE, R², RMSE |
# | Modèles | Baselines B0, B1 ; régression logistique, forêt aléatoire, LightGBM | Médiane, Ridge, LightGBM (`LGBMRegressor`) |
#
# **Ce qu'il faut retenir.** Le CSM attend un score de risque (« 78 % de départ ») : il faut un
# algorithme qui fournit `predict_proba` et un contrôle de la calibration (accord entre
# probabilité annoncée et fréquence observée, §9). La valeur vie client chiffre l'enjeu d'un
# compte ; elle n'est **jamais** une feature (variable explicative) du churn (§2).

# %% [markdown]
# ### 8.2 Cibles de performance a priori
#
# Un seuil fixé après avoir vu les scores se cale toujours sur le meilleur modèle : on les fixe
# donc avant. Ils proviennent de `config.CIBLES_PERFORMANCE`, posé au cadrage (§2), et sont lus
# ici directement dans la configuration.

# %%
_lignes_cibles = [
    ("Churn — PR-AUC (validation croisée, puis test)", "≥", _c["pr_auc_min"], "Sélection"),
    ("Churn — ROC-AUC (test)", "≥", _c["roc_auc_min"], "Co-principale"),
    ("Churn — recall de vigilance visé (OOF)", "=", config.RECALL_CIBLE_VIGILANCE, "Co-principale"),
    ("Churn — recall de vigilance (test)", "≥", _c["recall_vigilance_min"], "Co-principale"),
    ("Churn — gain de PR-AUC exigé", "≥", _c["gain_pr_auc_min_complexite"], "Parcimonie"),
    ("Latence unitaire API (ms)", "≤", _c["latence_unitaire_ms"], "Exploitation"),
    ("Latence batch 5 000 comptes (s)", "≤", _c["latence_batch_5k_s"], "Exploitation"),
    ("Carbone d'un entraînement complet (g CO₂e)", "≤", _c["co2_entrainement_max_g"], "Sobriété"),
    ("CLV — R² (test, euros)", "≥", _c["clv_r2_min"], "Plancher"),
    ("CLV — gain de MAE sur la médiane", "≥", _c["clv_gain_mae_min"], "Valeur ajoutée"),
    ("CLV — R² d'alerte fuite", ">", _c["clv_r2_alerte_fuite"], "Audit obligatoire"),
]
cibles_a_priori = pd.DataFrame(_lignes_cibles, columns=["indicateur", "sens", "seuil", "rôle"])
styler_fr(cibles_a_priori, {"seuil": nombre_tableau}).hide(axis="index")

# %% [markdown]
# **Ce qu'il faut retenir.** Sans information, PR-AUC = prévalence (plus haut) et ROC-AUC = 0,5 :
# la cible de PR-AUC exige plus du double du hasard. Le recall est visé hors pli (out-of-fold :
# chaque compte scoré par un modèle qui ne l'a pas vu), avec une tolérance sur le test, car
# manquer un churner coûte bien plus qu'une surveillance inutile (§2). Les latences sont des objectifs de service (§11), pas un critère de tri. Pour la
# CLV, les cibles sont relatives à une baseline (référence de base) naïve, et un R² quasi
# parfait est présumé être une fuite. Tout champion satisfait **toutes** les cibles de sa tâche.

# %% [markdown]
# ### 8.3 Exigences d'exploitation : volumétrie, fréquence, intégration, compétences
#
# Le meilleur score ne sert à rien s'il ne s'insère pas dans le travail des équipes.
#
# | Dimension | Contrainte retenue |
# |---|---|
# | Volumétrie | ~5 000 comptes actifs : tabulaire de taille modeste |
# | Fréquence | Batch (traitement par lot) nocturne quotidien : les sources sont rafraîchies une fois par jour |
# | Déploiement | Job nocturne + API REST (FastAPI) à l'ouverture d'une fiche client (§10) |
# | Compétences | 1 data scientist à mi-temps (§2) : écosystème scikit-learn, API commune à tous les modèles |
# | Explicabilité | SHAP obligatoire : le CSM doit voir pourquoi un compte est à risque |
# | Souveraineté | Base d'analyse, modèle et scores hébergés dans l'UE (secret des affaires, RGPD pour les contacts). Les instances clientes, elles, sont aussi hébergées hors UE (`code_datacenter`) : seuls des agrégats par entreprise en sont extraits (§4.1) |
#
# **Ce qu'il faut retenir.** Le churn évolue en semaines : scorer plus souvent que les sources ne
# verrait rien de neuf. Sans explication, pas d'adoption ; la souveraineté pèse sur l'achat (§8.5).

# %% [markdown]
# ### 8.4 Leviers de sobriété et note d'arbitrage carbone
#
# Un modèle plus lourd consomme plus à chaque entraînement et chaque nuit : on ne l'accepte que
# s'il rapporte nettement plus. Leviers fixés ici, mesurés par CodeCarbon en §9 et §12.
#
# | Levier | Engagement |
# |---|---|
# | Algorithme | À gain non démontré, le modèle le plus simple l'emporte (règle de §8.8) |
# | Variables | Espace borné dès le feature engineering (ingénierie des variables, §7) ; pas d'AutoML |
# | Tuning (optimisation des hyperparamètres) | Optuna limité à 30 essais par famille optimisée, sur CPU |
# | Réentraînement | Trimestriel, anticipé seulement si une dérive est détectée (§13) : 4 entraînements par an au lieu de 12 |
# | Inférence | Batch nocturne + appels API ponctuels ; aucun flux temps réel |
#
# **Ce qu'il faut retenir.** La sobriété entre dans la règle de sélection elle-même. Note
# d'arbitrage transmise au commanditaire (Direction Produit et DSI) : un modèle plus complexe
# n'est retenu que si son gain de PR-AUC sur chaque modèle plus simple non rejeté est **à la
# fois** significatif (§8.8) **et** d'au moins 0,02 ; le budget carbone (§8.2) est confronté à
# la mesure en §12.13.

# %% [markdown]
# ### 8.5 Analyse build vs buy
#
# Avant de construire, on vérifie qu'acheter ne suffirait pas (build vs buy : développer en
# interne ou acheter). On compare sur la durée de vie de la solution : l'interne est chiffré par
# `config.COUTS_PROJET`, comme le ROI du projet (§12.12) ; licences : tarifs publics indicatifs.

# %%
_cp = config.COUTS_PROJET
_ans, _jour = int(_cp["duree_amortissement_ans"]), float(_cp["cout_journalier_eur"])
_jours_build = sum(float(_cp[f"jours_build_{p}"]) for p in ("data_scientist", "dsi", "cs"))
_build = _jours_build * _jour
_run = 12 * (float(_cp["infra_mensuel_eur"]) + float(_cp["jours_maintenance_mois"]) * _jour)
_interne = _build + _ans * _run
# Licence annuelle (bas, haut) ; l'AutoML exige en plus le même travail de préparation
_offres = {"Gainsight": (25e3, 80e3, 0), "ChurnZero": (15e3, 50e3, 0)}
_offres["AutoML managé"] = (2e3, 8e3, _build)
_cout = {n: (_ans * b + fixe, _ans * h + fixe) for n, (b, h, fixe) in _offres.items()}
_f = {n: f"{euros(b)} – {euros(h)}" for n, (b, h) in _cout.items()}
# Garde-fou du texte ci-dessous : interne dans la fourchette de ChurnZero, au-dessus de l'AutoML
_cz = _cout["ChurnZero"]
assert _cout["AutoML managé"][1] < _interne and _cz[0] <= _interne <= _cz[1]
display(Markdown(f"""
| Option | Coût sur {_ans} ans | Explicabilité | Souveraineté | Dépendance | Décision |
|---|---|---|---|---|---|
| Gainsight (suite Customer Success) | {_f['Gainsight']} | Score propriétaire, non auditable | Données aux États-Unis | Forte | Écartée |
| ChurnZero (suite Customer Success) | {_f['ChurnZero']} | Score propriétaire | Données aux États-Unis | Forte | Écartée |
| AutoML managé (Azure ML, Vertex AI, SageMaker) | {_f['AutoML managé']}, préparation des données comprise | SHAP possible, intégration lourde | Selon région | Forte (API propriétaire) | Écartée |
| **Développement interne** | **{euros(_interne)}** : {euros(_build)} de développement ({entier(_jours_build)} jours), puis {euros(_run)} par an d'infrastructure et de maintenance | SHAP natif, code auditable | Hébergement UE maîtrisé | Nulle | **Retenue** |

**Ce qu'il faut retenir.** Le coût ne départage pas : sur {_ans} ans, l'interne
({euros(_interne)}) est dans la fourchette de ChurnZero, sous Gainsight et au-dessus de l'AutoML ;
son premier poste est la maintenance ({entier(_cp['jours_maintenance_mois'])} jours par mois).
Le choix tient à l'explicabilité, à l'hébergement dans l'UE et à l'absence de dépendance ;
l'AutoML, moins cher sur le papier, échoue sur les deux derniers. Ordres de grandeur, à
confirmer par le contrôle de gestion.
"""))

# %% [markdown]
# ### 8.6 Familles candidates pour des données tabulaires
#
# On cherche le modèle le plus simple qui atteigne les cibles : une famille simple face à deux
# plus riches, toutes capables de fournir une probabilité et une explication par compte.
#
# | Modèle | Pourquoi il est retenu | Ce qu'il doit prouver |
# |---|---|---|
# | **Régression logistique** (baseline ML) | Interprétable (un coefficient par variable), rapide, sobre ; référence de toute comparaison | Que les variables métier de §7 suffisent à capter le signal |
# | **Forêt aléatoire** | Moyenne de centaines d'arbres : capte les non-linéarités et les interactions, robuste au bruit et aux valeurs extrêmes, peu de réglages | Qu'une frontière non linéaire apporte un gain |
# | **LightGBM** | Boosting d'arbres (chaque arbre corrige les erreurs des précédents), état de l'art sur données tabulaires ; gère nativement les valeurs manquantes et reste rapide | Que ce gain dépasse celui de la forêt, après tuning |
#
# | Famille écartée *avant* tout résultat | Motif d'écartement |
# |---|---|
# | SVM | Pas de probabilité native (calibration interne coûteuse) ; explication par SHAP lente et approchée ; version linéaire redondante avec la régression logistique |
# | kNN (plus proches voisins) | Probabilité par paliers de 1/k ; distances peu fiables avec plus de 50 variables encodées |
# | Bayésien naïf | Suppose les variables indépendantes, alors que usage et satisfaction sont fortement corrélés (§6) |
# | Réseaux de neurones | Surdimensionnés pour ~5 000 lignes ; déploiement et explication disproportionnés |
#
# **Ce qu'il faut retenir.** La supériorité des arbres est une **hypothèse à vérifier** : avec
# les ratios de §7 et environ 80 observations par variable et par pli (calculé plus haut), un
# modèle linéaire peut capter l'essentiel. Arbres : SHAP exact (TreeExplainer) ; régression
# logistique : ses coefficients. Les deux familles optimisées par Optuna (§9) sont **également
# réglées** avant d'être comparées.

# %% [markdown]
# ### 8.7 Les trois baselines
#
# Un modèle ne vaut que par l'écart à ce qu'on ferait sans lui. La vraie question est : le
# machine learning fait-il mieux qu'**une règle simple, applicable par un filtre du CRM** ?
#
# | Référence | Définition | Rôle |
# |---|---|---|
# | B0 — Hasard stratifié | Tirage selon la prévalence | Plancher absolu (PR-AUC ≈ prévalence) |
# | B1 — Règle métier | `derniere_connexion_jours > 30` OU `csat ≤ 2` | Une règle suffit-elle ? |
# | Baseline LR | Régression logistique L2, `class_weight="balanced"`, intercept corrigé | Mesurer le gain de la complexité |
#
# **Ce qu'il faut retenir.** Si le modèle ne bat pas nettement B1, on déploiera la règle :
# moins chère, plus lisible, plus robuste aux dérives. Ses seuils sont fixés **par convention** :
# 30 jours est la fenêtre d'activité du produit (colonnes `*_30j`), un CSAT de 1 ou 2 sur 5
# signale un client insatisfait. `class_weight="balanced"` évite d'ignorer les churners mais
# décale le logit de log((1 − π)/π) (π = prévalence) ; corriger l'intercept (constante) après
# apprentissage (`RegressionLogistiqueRecalibree`) l'annule sans changer le classement.

# %% [markdown]
# ### 8.8 Protocole de comparaison
#
# Pour qu'une comparaison soit honnête, tous les modèles passent le même examen, et la règle
# de victoire est écrite avant de connaître les copies.
#
# - **Jeu de test** : une part des comptes est mise de côté avant toute comparaison ; elle ne
#   sert qu'une fois, à noter le modèle retenu, sur des comptes qu'aucun choix n'a vus.
# - **Cross-validation (validation croisée)** : on découpe le reste en 5 plis, on apprend sur 4
#   et on évalue sur le 5ᵉ, à tour de rôle ; répété 3 fois, cela donne 15 scores par modèle.
# - **Test de Wilcoxon apparié** : dit, pli par pli, si un modèle gagne systématiquement ou si
#   l'écart peut venir du hasard. **Correction de Holm** : resserre les seuils quand les tests
#   se multiplient, pour garder un risque global de 5 %.
#
# | Élément | Décision |
# |---|---|
# | Jeu de test | `config.PART_TEST` des comptes, stratifié, mis de côté en §9.1 avec `config.RANDOM_SEED` (même découpage que la CLI et le flow) ; ouvert une seule fois, en §9.14 |
# | Découpage | Sur le jeu de développement, un seul `RepeatedStratifiedKFold` (`protocole_validation()`), 5 plis stratifiés × 3 répétitions, partagé par tous les modèles |
# | Transformations | Imputation, encodage et standardisation dans un pipeline appris dans chaque pli |
# | Métrique de sélection | PR-AUC moyenne sur les 15 plis |
# | Co-principales | ROC-AUC, recall ; le seuil de vigilance est fixé sur les prédictions out-of-fold du développement, puis appliqué au test (§9.14) |
# | Contrôles | Précision, score de Brier, latence |
# | Rejet | PR-AUC moyenne sous la cible de §8.2 |
# | Champion | Meilleure PR-AUC parmi les non-rejetés, **si** son gain sur chaque modèle plus simple est significatif (Wilcoxon unilatéral + Holm) **et** d'au moins 0,02 ; sinon le plus simple l'emporte |
#
# **Ce qu'il faut retenir.** Le test dit si l'écart est réel, le seuil de 0,02 s'il compte :
# sur 15 plis appariés, quelques millièmes peuvent être significatifs sans justifier un modèle
# plus lourd. Les plis répétés partagent des données : le test est un garde-fou, pas une preuve.
# Les plis servent à **choisir**, le jeu de test à **juger**. Le Brier n'est qu'un contrôle : une
# calibration se corrige après coup, un mauvais classement jamais.

# %% [markdown]
# #### 8.8.1 Pourquoi PR-AUC et non ROC-AUC comme métrique de rang ?
#
# La ressource rare est le temps des CSM : on veut une métrique qui baisse à chaque appel
# inutile.
#
# | | ROC-AUC (aire sous la courbe ROC) | PR-AUC (aire sous la courbe précision-rappel) |
# |---|---|---|
# | Question | Un churner pris au hasard est-il classé devant un fidèle ? | Parmi les comptes signalés, combien partent vraiment, à chaque niveau de recall ? |
# | Erreurs rapportées à | L'ensemble des fidèles (~72 % des comptes) | Les comptes signalés, c'est-à-dire la precision (précision) |
# | Hasard | 0,5, quelle que soit la prévalence | La prévalence, propre au jeu |
# | Usage ici | Co-principale : classement global, lisible, comparable à la littérature | **Sélection** : qualité de la liste d'appels |
#
# **Ce qu'il faut retenir.** La PR-AUC choisit le modèle car elle mesure ce que coûte le
# projet : les appels sur des comptes qui ne seraient pas partis. La ROC-AUC, indépendante de la
# prévalence, et le recall, que vise le seuil de vigilance (§12), restent co-principaux.

# %% [markdown]
# ### 8.9 Adéquation du modèle aux trois cas d'usage
#
# | Cas d'usage (§2) | Exigence | Conséquence |
# |---|---|---|
# | Revue hebdomadaire du portefeuille | Batch de 5 000 comptes dans la fenêtre nocturne | Aucune famille écartée (latence non discriminante) |
# | Alerte sur signal faible | Probabilité calibrée et stable d'une nuit à l'autre | Écarte kNN, bayésien naïf, SVM |
# | Préparation d'un renouvellement | Trois raisons du risque à l'écran, en moins de 200 ms | SHAP exact : régression logistique, forêt, LightGBM |
#
# **Ce qu'il faut retenir.** Les trois cas d'usage convergent vers la même exigence,
# **probabilité calibrable + explication exacte**, qui confirme la sélection de §8.6.

# %% [markdown]
# > ### 📋 Journal de bord — Choix du modèle
# >
# > **Décisions retenues** — Développement interne. Régression logistique (intercept corrigé),
# > forêt aléatoire et LightGBM comparés en §9, la supériorité des arbres restant à vérifier.
# > Jeu de test mis de côté, ouvert une seule fois ; 5 plis × 3 répétitions sur le reste. PR-AUC
# > pour la sélection, ROC-AUC et recall co-principaux, cibles lues dans `config.py` ; un modèle
# > plus complexe n'est retenu que pour un gain significatif et d'au moins 0,02.
# >
# > **Alternatives écartées** — SVM, kNN, bayésien naïf, réseaux de neurones (probabilité,
# > explication ou volume inadaptés) ; Gainsight, ChurnZero, AutoML (boîte noire, souveraineté,
# > dépendance : le coût ne les départage pas). HistGradientBoosting remplacé par LightGBM pour
# > les deux cibles, plus rapide à optimiser et de référence sur données tabulaires.
# >
# > **Difficultés rencontrées** — Le seuil CSAT de B1 valait d'abord `csat ≤ 6`, hérité d'une
# > échelle sur 10 : la règle signalait tout client noté et ne faisait pas mieux que le hasard.
# > Le protocole initial choisissait et jugeait sur les mêmes plis : le jeu de test, ajouté en
# > cours de projet, a imposé de rejouer la chaîne. Le coût interne, d'abord estimé hors
# > intégration et maintenance (6 000 à 8 000 €), est désormais calculé (`COUTS_PROJET`).
# >
# > **Impact sur la suite** — §9 applique ce protocole, optimise la régression logistique et
# > LightGBM et note le modèle retenu sur le test (§9.14) ; §12 fixe le seuil de vigilance et
# > mesure latences et carbone ; §10 déploie le champion.
