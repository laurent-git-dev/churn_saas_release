# %% [markdown]
# ## 8. Choix du modèle (C4)
#
# Cette section expose **le raisonnement qui précède toute modélisation**.
# Elle couvre les dix items de la compétence C4 dans l'ordre logique d'une démarche
# scientifique : type de problème → cibles de performance (a priori) → contraintes
# opérationnelles → panorama des familles → build vs buy → protocole de comparaison.
# Aucun résultat de modèle n'apparaît ici : les scores sont dans §9.
#
# Les valeurs chiffrées citées dans le texte (cibles de performance, règle métier, nombre de plis,
# nombre de variables, prévalence du churn) sont vérifiées par la cellule ci-dessous contre
# `config.py`, `models.train` et les données : si l'une d'elles change, le notebook refuse de
# s'exécuter plutôt que d'afficher un texte périmé.

# %%
import pandas as pd

from churn_saas import config
from churn_saas.data.loaders import charger_brut
from churn_saas.features.build import ajouter_features_metier
from churn_saas.models.train import SEUIL_CSAT, SEUIL_INACTIVITE_JOURS, protocole_validation

# Prévalence recalculée comme en §6 ; doublons exclus (un client = une ligne)
_churn = pd.to_numeric(charger_brut("churn_saas_complet").drop_duplicates()["churn"])
_prevalence = float(_churn.dropna().mean())
_PREVALENCE_CITEE = 0.28
assert abs(_prevalence - _PREVALENCE_CITEE) < 0.01, (
    f"Prévalence du churn : données = {_prevalence:.3f}, texte = {_PREVALENCE_CITEE} "
    "— mettre §8 à jour"
)

# Variables en entrée des modèles, construites exactement comme en §9 (avant encodage)
_gold = pd.read_parquet(config.DONNEES_GOLD / "gold_dataset.parquet")
_X = ajouter_features_metier(_gold.drop(columns=config.COLONNES_INTERDITES, errors="ignore"))
_n_variables = _X.drop(columns=["churn"], errors="ignore").shape[1]

_valeurs_citees = {
    "Variables en entrée des modèles": (_n_variables, 52),
    "PR-AUC minimale": (config.CIBLES_PERFORMANCE["pr_auc_min"], 0.65),
    "Latence unitaire (ms)": (config.CIBLES_PERFORMANCE["latence_unitaire_ms"], 200),
    "Latence batch 5 000 comptes (s)": (config.CIBLES_PERFORMANCE["latence_batch_5k_s"], 300),
    "Seuil d'inactivité (jours)": (SEUIL_INACTIVITE_JOURS, 30),
    "Seuil CSAT": (SEUIL_CSAT, 6),
    "Scores par métrique (plis × répétitions)": (protocole_validation().get_n_splits(), 15),
}
for _nom, (_reel, _cite) in _valeurs_citees.items():
    assert _reel == _cite, f"{_nom} : config = {_reel}, texte = {_cite} — mettre §8 à jour"
print(
    f"{len(_valeurs_citees)} valeurs citées dans §8 conformes à la configuration ; "
    f"prévalence du churn = {_prevalence:.1%} (plancher PR-AUC d'un modèle aléatoire)."
)

# %% [markdown]
# ### 8.1 Type de problème — item C4 : « type de résultat attendu »
#
# Avant de choisir un algorithme, on qualifie précisément le problème : ce choix conditionne les
# métriques, le protocole de validation et l'architecture de déploiement.
# Le jeu de données porte **deux cibles distinctes, qui donnent deux tâches de nature
# différente**, toutes deux en apprentissage supervisé (valeurs historiques disponibles) :
#
# - **cible principale** `churn` → **classification binaire**, sortie **probabiliste** ;
# - **cible secondaire** `valeur_vie_client_eur` → **régression**, sortie **déterministe**.
#
# | | **Cible principale** | **Cible secondaire** |
# |---|---|---|
# | Variable | `churn` (0 = reste, 1 = résilie) | `valeur_vie_client_eur` (en euros, continue, strictement positive) |
# | Question posée | « Ce compte va-t-il résilier ? » | « Combien vaut ce compte ? » |
# | Type de tâche | **Classification binaire** | **Régression** |
# | Sortie attendue | **Probabiliste** : score de risque entre 0 et 1, pas une étiquette 0/1 brute | **Déterministe** : un montant estimé en euros, pas une probabilité |
# | Calibration | **Requise** : le score doit être une probabilité fiable pour prioriser les comptes | Sans objet : la sortie n'est pas une probabilité |
# | Métriques d'évaluation | PR-AUC (rang), ROC-AUC, Brier (calibration) | RMSE (pénalise les grandes erreurs), MAE (robuste aux outliers), R² (variance expliquée) |
# | Modèles comparés | Régression Logistique (baseline) vs GBM classifieur | Ridge (baseline) vs GBM régresseur |
# | Fonction de perte du gradient boosting (scikit-learn) | `log_loss` (`HistGradientBoostingClassifier`) | `squared_error` (`HistGradientBoostingRegressor`) |
# | Traitée en | §9 (comparaison, calibration), §12 (performance) | §9 et §12 |
#
# **Ce qu'il faut retenir.** Deux cibles, donc deux problèmes à ne pas confondre.
#
# - **Churn (principale) : on prédit une probabilité.** Le Customer Success Manager a besoin d'un
#   *score de risque* (« ce compte a 78 % de probabilité de résilier »), pas d'une étiquette sèche.
#   D'où deux exigences : un algorithme qui fournit `predict_proba` ou un équivalent calibré, et un
#   contrôle de la calibration (courbe de fiabilité) en §9.
# - **Valeur vie client (secondaire) : on prédit un montant.** Le CSM reçoit une valeur en euros ;
#   la notion de calibration ne s'y applique pas, la qualité se juge à l'erreur (RMSE, MAE) et à la
#   variance expliquée (R²).
#
# Les deux tâches utilisent **la même famille d'algorithmes** (GBM), avec seulement un objectif
# d'optimisation différent : ce choix homogène simplifie le pipeline de déploiement (§10) et
# la maintenance.

# %% [markdown]
# ### 8.2 Cibles de performance a priori — item C4 : « performance attendue » · item C8 : indicateurs et seuils
#
# **Principe scientifique** : les seuils d'acceptabilité sont fixés *avant* de voir les résultats.
# Fixer des cibles *après* revient à une sélection opportuniste des résultats (*cherry-picking*,
# détaillé en §8.8) et invalide la démarche.
# Ces cibles proviennent de `config.CIBLES_PERFORMANCE`, défini en §2 lors du cadrage métier.
#
# | Métrique / contrainte | Seuil a priori | Justification |
# |---|---|---|
# | PR-AUC minimale | ≥ 0,65 | Modèle aléatoire ≈ 0,28 (prévalence du churn, calculée ci-dessus) ; le seuil de 0,65 représente le gain minimal justifiant le coût de déploiement |
# | Latence unitaire (API CRM) | ≤ 200 ms | Ouverture d'une fiche client dans le CRM (§2, Cas d'usage 3) — un CSM attend la réponse à l'écran |
# | Latence batch 5 000 comptes | ≤ 300 s | Fenêtre de maintenance nocturne < 5 min — contrainte d'exploitation |
# | Budget énergétique (éco-conception) | ≤ 50 g CO₂e / session d'entraînement complète (mesuré par CodeCarbon §9) | 50 g CO₂e ≈ 0,2 km en voiture — consommation négligeable vs gain de productivité CSM (~4 h/mois/personne). Seuil fixé a priori ; mesuré par CodeCarbon en §9 et reporté dans l'éco-bilan §12. |
#
# **Ce qu'il faut retenir.** Les quatre cibles ci-dessus sont contractuelles :
# tout modèle champion doit les satisfaire **simultanément**.
# Le seuil de PR-AUC de 0,65 exige plus du double du plancher d'un modèle aléatoire (0,28) :
# c'est le gain minimal pour que le projet soit économiquement viable (calcul ROI en §12).
# La latence batch de 300 s pour 5 000 comptes est compatible avec pratiquement tous les
# algorithmes scikit-learn — ce n'est pas un filtre technique mais un engagement de SLO.

# %% [markdown]
# ### 8.3 Contraintes opérationnelles — item C4 : « contraintes opérationnelles »
#
# Le choix du modèle ne se limite pas au score : il doit s'intégrer dans un écosystème
# réel avec des contraintes d'exploitation, de compétences et de gouvernance.
#
# | Dimension | Contrainte retenue |
# |---|---|
# | Volumétrie | ~5 000 comptes actifs — tabulaire de taille modeste |
# | Fréquence d'inférence | Batch nocturne quotidien (suffisant — sources rafraîchies une fois par jour) |
# | Mode de déploiement | Job batch nocturne quotidien + API REST (FastAPI) à la demande |
# | Intégration CRM | Export JSON/CSV vers CRM (Salesforce / HubSpot) — contrat d'API en §10 |
# | Compétences de l'équipe | Équipe data science de 1-2 personnes ; favoriser scikit-learn (maîtrisé, une seule bibliothèque pour toutes les familles) |
# | Explicabilité requise | SHAP obligatoire — le CSM doit comprendre pourquoi un compte est à risque |
# | Souveraineté des données | Données client B2B hébergées en France (RGPD) — solution interne préférable |
#
# **Ce qu'il faut retenir.** Le batch nocturne quotidien suffit car le churn est un
# phénomène qui évolue sur des semaines, pas des heures, et les sources (CRM, support) ne
# sont rafraîchies qu'une fois par jour : un scoring plus fréquent ne verrait rien de neuf.
# L'explicabilité (SHAP) est non négociable : sans elle, les CSM n'adopteront pas l'outil
# (résistance au changement documentée dans la littérature CX).
# La contrainte de souveraineté des données pèse directement sur le choix build vs buy (§8.5).

# %% [markdown]
# ### 8.4 Éco-conception — item C4 : « contraintes d'éco-conception portées aux acteurs »
#
# Les contraintes d'éco-conception sont évaluées et transmises au commanditaire.
# Le suivi carbone est assuré par CodeCarbon (§9, §12).
#
# | Levier | Impact carbone estimé | Décision / engagement |
# |---|---|---|
# | Choix de l'algorithme | Régression logistique nettement moins coûteuse à entraîner et à interroger qu'un ensemble de centaines d'arbres ; rapport réel mesuré en §9 (durée d'entraînement, latence) et §12 (CodeCarbon) | Comparer perf × carbone ; ne pas retenir un ensemble d'arbres si la Régression Logistique atteint la cible |
# | Fréquence de réentraînement | Trimestriel : 4 entraînements/an contre 12 en mensuel — empreinte annuelle divisée par 3 | Réentraînement trimestriel par défaut, anticipé si drift détecté (§13) |
# | Batch nocturne vs temps réel | Batch : 1 inférence/compte/nuit — coût minimal ; API : appels ponctuels uniquement | Aucun streaming temps réel — API appelée uniquement à l'ouverture d'une fiche client |
# | Nombre de variables | 52 variables en entrée des modèles (avant encodage), dont les variables métier dérivées en §7 et les 4 leurres conservés volontairement ; ~100 observations par variable : pas de réduction de dimension nécessaire | Pas d'AutoML inflationnaire ; espace de variables borné dès le feature engineering §7 |
# | Budget Optuna | Limité à 30 essais Optuna : arbitrage conscient perf/carbone | Budget fixé a priori en nombre d'essais, sur CPU (les modèles scikit-learn retenus n'utilisent pas de GPU) ; durée et empreinte réelles mesurées en §9 |
#
# **Ce qu'il faut retenir.** L'éco-conception est une contrainte de conception,
# pas un bonus optionnel.
# La note d'arbitrage performance/temps/carbone est transmise au commanditaire (§12).
# Si la régression logistique atteint la cible PR-AUC ≥ 0,65, elle est retenue en priorité
# sur un modèle plus complexe à gain marginal.
#
# **Note d'arbitrage éco-conception — transmise au commanditaire**
#
# | | |
# |---|---|
# | **Destinataire** | Direction Produit + DSI |
# | **Objet** | Arbitrage performance / temps de calcul / empreinte carbone |
# | **Engagement** | Aucun modèle plus complexe ne sera retenu pour un gain PR-AUC < 0,02 sur la baseline ML (seuil : PR-AUC ≥ 0,65). |
# | **Budget Optuna** | 30 essais maximum, sur CPU — durée et empreinte mesurées par CodeCarbon en §9 et reportées en §12. |
# | **Réentraînement** | Trimestriel par défaut ; anticipé si drift détecté (§13) — évite les entraînements inutiles. |
# | **Statut** | *(Simulée dans le cadre de l'examen — assumée comme telle)* |

# %% [markdown]
# ### 8.5 Analyse build vs buy — pertinence des solutions sur l'étagère
#
# Le commanditaire doit évaluer si un développement interne est justifié face aux solutions
# commerciales ou managées.
#
# | Solution | Principe | Coût estimé | Délai | Explicabilité | Souveraineté données | Dépendance fournisseur | Recommandation |
# |---|---|---|---|---|---|---|---|
# | **Gainsight** (SaaS Customer Success) | Plateforme CS tout-en-un avec scoring de santé intégré | ~25 000–80 000 €/an (licences per-seat + onboarding) | 3–6 mois (onboarding, migration CRM, formation) | Score propriétaire non auditable — boîte noire | Données exportées vers serveurs US — risque RGPD | Forte — migration difficile, pricing variable | Écarté — coût prohibitif pour 5 000 comptes, boîte noire, risque RGPD |
# | **ChurnZero** (SaaS Customer Success) | CRM CS avec scoring de risque churn propriétaire | ~15 000–50 000 €/an selon portefeuille | 2–4 mois | Score propriétaire — boîte noire | Données exportées vers serveurs US — risque RGPD | Forte — migration difficile | Écarté — mêmes motifs, moins de fonctionnalités CS avancées |
# | **AutoML managé** (Azure ML / Vertex AI / SageMaker) | Pipeline ML automatisé sur cloud, modèle opaque | ~2 000–8 000 €/an (compute) + coût de migration | 1–2 mois (mais lock-in cloud) | SHAP possible mais complexité d'intégration | Dépend du cloud retenu ; possible si zone EU | Forte — lock-in API propriétaire | Écarté — souveraineté non garantie, explicabilité complexe, lock-in |
# | **Développement interne** ← retenu | Pipeline scikit-learn, modèle SHAP-explicable, hébergé en propre | ~3 000–5 000 € (développement) + ~200 €/an (infra) | 6–8 semaines (déjà engagé) | SHAP natif, interprétation complète, auditabilité totale | Hébergement interne ou cloud FR/EU maîtrisé | Nulle — code source maîtrisé, portabilité totale | **Retenu** — coût minimal, explicabilité native, souveraineté totale, délai court |
#
# **Ce qu'il faut retenir.** Le développement interne est retenu sur quatre critères
# décisifs : **coût** (×10 moins cher sur 3 ans), **explicabilité** (SHAP natif,
# exigé par les CSM), **souveraineté des données** (RGPD — pas de transfert hors UE)
# et **absence de dépendance fournisseur**.
# Gainsight et ChurnZero sont pertinents pour des équipes CS de 20+ personnes
# sans compétences data ; ce n'est pas le profil du commanditaire.
# L'AutoML managé est écarté principalement pour le lock-in et la souveraineté —
# pas pour des raisons techniques.

# %% [markdown]
# ### 8.6 Panorama des familles d'algorithmes — grandes familles connues
#
# Chaque famille est évaluée sur cinq critères : principe, adéquation au tabulaire
# de ~5 000 lignes, explicabilité, coût computationnel, et motif de retenue ou d'écartement.
# Les familles **retenues** seront comparées en §9.
#
# | Famille | Principe | Adéquation (5 000 lignes, tabulaire) | Explicabilité | Coût computationnel | Motif retenue / écartement |
# |---|---|---|---|---|---|
# | Linéaires (Régression Logistique, Ridge) | Frontière de décision linéaire dans l'espace des features | Excellente — converge rapidement, pas de sur-apprentissage sur 5 k obs | Maximale — coefficients directement interprétables | Minimal (secondes sur 5 k obs) | **Retenue** — baseline ML robuste, explicabilité maximale, coût carbone minimal |
# | Arbres / Ensembles (forêt aléatoire, gradient boosting) | Combinaison de nombreux arbres de décision (bagging ou boosting) | Bonne — les GBM dominent de nombreux benchmarks tabulaires, surtout sur de gros volumes avec interactions complexes ; avantage non garanti sur 5 000 lignes | Bonne — SHAP TreeExplainer O(n), nativement supporté | Modéré — < 1 min d'entraînement sur 5 k obs sans GPU | **Retenue** — challengers : forêt aléatoire et gradient boosting (`HistGradientBoostingClassifier`, même principe que LightGBM), pour tester si les interactions non linéaires apportent un gain |
# | SVM (noyau RBF ou linéaire) | Maximisation de la marge entre classes dans un espace projeté | Correcte — mais lent à l'inférence pour des noyaux RBF en production | Limitée — coefficients uniquement pour noyau linéaire | Modéré à l'entraînement, lent à l'inférence (noyau RBF) | Écarté — latence d'inférence incompatible avec la fiche client (> 200 ms probable) |
# | kNN (k plus proches voisins) | Prédiction par vote ou moyenne des k voisins les plus proches | Faible — O(n) à l'inférence, sensible aux unités, requiert normalisation parfaite | Intuitive mais non formelle — pas de coefficients globaux | Minimal à l'entraînement, O(n) à l'inférence | Écarté — latence O(n) incompatible avec batch 5 k comptes, sensibilité à la normalisation |
# | Bayésien Naïf (GaussianNB, ComplementNB) | Hypothèse d'indépendance conditionnelle entre features | Faible — hypothèse d'indépendance rarement vérifiée sur des métriques SaaS corrélées | Directe mais limitée — log-vraisemblances par feature | Minimal | Écarté — hypothèse d'indépendance violée (corrélations fortes usage ↔ satisfaction) |
# | Réseaux de neurones (MLP, TabNet) | Composition de couches non-linéaires apprenant des représentations | Faible — sur-paramétré pour 5 000 obs ; TabNet intéressant mais complexe à déployer | Faible sans outils dédiés (SHAP KernelExplainer très lent) | Élevé — nécessite GPU pour TabNet ; MLP instable sans beaucoup de données | Écarté — sur-paramétré pour 5 k obs, complexité de déploiement disproportionnée |
#
# **Ce qu'il faut retenir.** Deux familles sont retenues pour la comparaison formelle en §9 :
# la **régression logistique** (baseline ML, explicabilité maximale, coût carbone minimal)
# et les **ensembles d'arbres** (forêt aléatoire, gradient boosting ; SHAP TreeExplainer natif).
# La réputation des arbres sur données tabulaires est une **hypothèse à vérifier, pas une
# présomption** : sur ~5 000 lignes, avec des variables métier déjà construites (ratios,
# indicateurs — §7), un modèle linéaire peut capter l'essentiel du signal. Le §9 tranche par le
# protocole de §8.8, et l'engagement de §8.4 s'applique : les arbres ne sont retenus que s'ils
# battent **significativement** la régression logistique.
# SVM, kNN et réseaux de neurones sont écartés pour des motifs opérationnels
# (latence, sous-adéquation aux 5 k obs) documentés ici *avant* les résultats.
# Le bayésien naïf est écarté pour une raison structurelle : les features SaaS
# sont fortement corrélées (usage ↔ satisfaction), violant son hypothèse fondamentale.

# %% [markdown]
# ### 8.7 Les trois baselines — démarche scientifique
#
# Un modèle ML ne se justifie que s'il fait mieux que des baselines clairement définies.
# La vraie question du commanditaire est : **« Le ML fait-il mieux que deux règles sous Excel ? »**
#
# | Baseline | Description | Rôle | Seuils / paramètres |
# |---|---|---|---|
# | B0 — Prédicteur aléatoire stratifié | Tire aléatoirement selon la prévalence observée — plancher théorique | Définir le plancher absolu : PR-AUC ≈ prévalence (~0,28) | Aucun — reproductible via `RANDOM_SEED` |
# | B1 — Règle métier (SQL / Excel) | `derniere_connexion_jours > 30` OU `csat ≤ 6` — seuils issus de l'EDA §6 | Répondre à la question métier : la règle Excel suffit-elle ? | `derniere_connexion_jours > 30`, `csat ≤ 6` (EDA §6) |
# | B2 — Régression Logistique | Modèle linéaire régularisé L2 avec pipeline anti-fuite complet — standard du secteur | Mesurer le gain de la complexité (GBM vs linéaire) — rapport perf/coût | `C=1.0`, `max_iter=1000`, `class_weight='balanced'`, `RANDOM_SEED` |
#
# **Ce qu'il faut retenir.** La baseline B1 (règle métier) est au cœur de la démonstration
# de valeur du projet : si le modèle ML n'est pas significativement meilleur qu'une règle
# `derniere_connexion_jours > 30 OR csat ≤ 6`, la recommandation au commanditaire serait de
# déployer la règle — moins coûteuse, plus explicable, plus robuste au drift.
# Les seuils viennent de l'EDA §6 (rupture nette du taux de churn),
# pas de l'intuition ni du jeu de test.

# %% [markdown]
# ### 8.8 Protocole de comparaison — démarche scientifique
#
# Le protocole est écrit **avant** de voir les résultats. L'écrire après exposerait à la
# *sélection opportuniste* des résultats (en anglais *cherry-picking*) : choisir, une fois les
# scores connus, la métrique, le seuil ou le découpage qui flatte le modèle. Chaque choix paraît
# anodin, mais le score final devient optimiste et ne se reproduit pas en production.
#
# **Vocabulaire utilisé ci-dessous.**
#
# - **Validation croisée (CV, *cross-validation*)** : le jeu d'entraînement est découpé en
#   5 parts (les *plis*) ; le modèle apprend sur 4 et est évalué sur la 5ᵉ, en faisant tourner
#   la part d'évaluation. On obtient 5 scores au lieu d'un seul. Répéter l'opération 3 fois avec
#   des découpages différents en donne 15.
# - **Écart-type entre plis** : dispersion de ces 15 scores. Il mesure à quel point le score
#   dépend du hasard du découpage ; deux modèles dont l'écart de moyenne est inférieur à cette
#   dispersion ne peuvent pas être départagés à l'œil.
# - **Test de Wilcoxon apparié** : test statistique qui compare deux modèles **pli par pli**
#   (sur les mêmes données) et indique si l'un est systématiquement meilleur ou si l'écart peut
#   relever du hasard. Il ne suppose pas que les scores suivent une loi normale, ce qui convient
#   à 15 valeurs.
# - **Correction de Holm** : lorsqu'on compare le champion à plusieurs concurrents, chaque test
#   a 5 % de chances de conclure à tort ; sur plusieurs tests, ce risque s'additionne. La
#   correction de Holm resserre le seuil pour que le risque *global* reste à 5 %.
#
# | Élément | Valeur / Décision |
# |---|---|
# | Découpage partagé | Un seul objet `RepeatedStratifiedKFold` (`models.train.protocole_validation()`), utilisé par **tous** les modèles : mêmes plis pour tous, donc comparaison équitable et appariée |
# | Nombre de plis | 5, stratifiés : chaque pli conserve la prévalence du churn (~28 %) |
# | Nombre de répétitions | 3 : trois découpages différents, pour que le score ne dépende pas d'un découpage chanceux |
# | Total de scores par métrique | 15 scores par modèle (5 plis × 3 répétitions), conservés un par un pour le test de Wilcoxon |
# | Métrique de rang | **PR-AUC** (moyenne ± écart-type sur 15 plis) — adaptée aux classes déséquilibrées (§6.1, §8.8.1) |
# | Métriques de contrôle | ROC-AUC (communication), rappel et précision au seuil 0,5 (le seuil de décision métier est fixé ensuite, en §12), score de Brier (calibration), latence (ms) |
# | Critère de rejet absolu | PR-AUC moyenne < 0,65 (`config.CIBLES_PERFORMANCE['pr_auc_min']`) : modèle écarté sans appel |
# | Critère de sélection du champion | PR-AUC moyenne la plus élevée parmi les modèles non rejetés |
# | Test de significativité | Wilcoxon apparié (*signed-rank*) sur les 15 PR-AUC, champion contre chaque autre candidat |
# | Correction pour tests multiples | Holm, pour un risque global d'erreur α = 0,05 sur l'ensemble des comparaisons |
# | Si l'écart n'est pas significatif | Le modèle **le plus simple** l'emporte (engagement d'éco-conception §8.4 : pas de complexité pour un gain non démontré) |
#
# **Pourquoi le score de Brier n'est qu'une métrique de contrôle.** La PR-AUC et le Brier ne
# mesurent pas la même chose. La PR-AUC juge le **classement** : le modèle place-t-il les
# churners devant les autres ? Le Brier juge l'**exactitude des probabilités** : quand le modèle
# annonce 70 %, le churn survient-il dans ~70 % des cas ? Or un défaut de calibration se corrige
# après coup (recalibration, par exemple `CalibratedClassifierCV`) sans modifier le classement,
# alors qu'un mauvais classement ne se rattrape pas. On choisit donc le champion sur ce qui est
# irrattrapable, et on se sert du Brier pour vérifier — et au besoin corriger — la calibration
# exigée en §8.1. Le Brier dépend en outre de la prévalence, ce qui le rend peu lisible comme
# critère unique.
#
# **Ce qu'il faut retenir.** Tous les modèles sont évalués sur les mêmes 15 découpages, avec une
# métrique de rang (PR-AUC) et un seuil de rejet (0,65) fixés avant tout résultat. Le champion
# n'est pas seulement « le meilleur score » : sa supériorité doit être statistiquement
# significative (Wilcoxon, corrigé par Holm), faute de quoi le modèle le plus simple est retenu.
# Limite assumée : les plis d'une validation croisée répétée partagent une partie de leurs
# données, donc ne sont pas totalement indépendants ; le test de Wilcoxon peut s'en trouver
# légèrement optimiste. On l'interprète donc comme un garde-fou, pas comme une preuve absolue.

# %% [markdown]
# #### 8.8.1 Pourquoi PR-AUC et non ROC-AUC comme métrique de rang ?
#
# Les deux métriques sont liées, mais se comportent différemment avec des classes déséquilibrées.
# Ce choix est explicité ici pour éviter tout malentendu à la lecture de §9 et §12.
#
# | Critère | ROC-AUC | PR-AUC |
# |---|---|---|
# | Sensibilité au déséquilibre | Faible — le FPR est dilué par les vrais négatifs (majoritaires à 72 %) | Forte — Précision et Rappel se concentrent exclusivement sur la classe positive |
# | Plancher théorique | 0,50 — un modèle naïf prédisant tout '0' y atteint 0,50 | ≈ prévalence (0,28) — plancher honnête et directement interprétable |
# | Ce qu'elle maximise | Aire TPR = f(FPR) — pénalise peu les FP quand les TN sont nombreux | Aire Précision = f(Rappel) — chaque FP et chaque FN est visible |
# | Adéquation métier | Médiocre — le coût métier porte sur la classe minoritaire (churners), pas sur les TN | Excellente — prioriser les vrais churners (Rappel) sans noyer les CSM de faux positifs (Précision) |
#
# **Ce qu'il faut retenir.** Avec une prévalence de ~28 %, la ROC-AUC peut être flatteuse
# sans que le modèle soit utile : un classificateur qui prédit « restera » pour tous les comptes
# obtient ROC-AUC = 0,50 mais rate 100 % des churners.
# La PR-AUC est immunisée contre ce biais : son plancher théorique est la prévalence elle-même
# (~0,28), ce qui rend le seuil d'acceptabilité de 0,65 directement interprétable comme un gain
# réel sur la classe minoritaire.
# La ROC-AUC reste calculée en §12 comme métrique de communication
# (plus intuitive pour un public non-data) et pour comparer avec la littérature.

# %% [markdown]
# ### 8.9 Lien avec les cas d'usage — contexte des cas d'usage
#
# Les choix techniques ci-dessus sont directement reliés aux trois cas d'usage identifiés en §2.
#
# | Cas d'usage (§2) | Exigence sur le modèle | Familles compatibles |
# |---|---|---|
# | Revue hebdomadaire du portefeuille | Batch nocturne (édition du lundi) → latence batch ≤ 300 s pour 5 k comptes | Toutes (latence batch non discriminante) |
# | Alerte sur signal faible | Score probabiliste stable d'une nuit à l'autre → seuil d'alerte et variation vs J−7 | Probabiliste → RL, GBM ; écartés : kNN, NB (pas de proba fiable par défaut) |
# | Préparation d'un renouvellement contractuel | Explicabilité SHAP + latence unitaire ≤ 200 ms → fiche client avec 3 raisons du risque | SHAP natif + inférence rapide → RL, GBM ; écarté : SVM (SHAP lent, latence RBF) |
#
# **Ce qu'il faut retenir.** Les trois cas d'usage de §2 convergent vers la même
# contrainte décisive : **probabiliste + SHAP**.
# Cela exclut kNN et bayésien naïf (probas non calibrées par défaut)
# et SVM (SHAP KernelExplainer prohibitif en production),
# ce qui confirme la sélection de §8.6 par une voie indépendante.

# %% [markdown]
# > ### 📋 Journal de bord — Choix du modèle
# >
# > **Décisions retenues** — Développement interne (build) retenu sur Gainsight/ChurnZero/AutoML
# > (coût ×10 moins cher, SHAP natif, souveraineté RGPD, délai court). Deux familles retenues
# > pour §9 : Régression Logistique (baseline ML) et ensembles d'arbres (forêt aléatoire,
# > gradient boosting), ces derniers traités comme une hypothèse à vérifier. Protocole unique
# > RepeatedStratifiedKFold(n_splits=5, n_repeats=3, RANDOM_SEED) partagé par tous les modèles.
# > Métrique de rang : PR-AUC. Sortie probabiliste assumée dès le cadrage, d'où l'exigence de
# > calibration en §9.
# >
# > **Alternatives écartées** — SVM (latence inférence > 200 ms probable sur noyau RBF),
# > kNN (O(n) à l'inférence, 5 k obs non discriminant), Bayésien Naïf (indépendance violée par
# > les corrélations usage ↔ satisfaction), Réseaux de neurones (sur-paramétrés pour 5 k obs,
# > complexité de déploiement disproportionnée). Gainsight/ChurnZero écartés pour trois motifs :
# > prix, boîte noire (non auditable), transfert de données hors UE.
# >
# > **Difficultés rencontrées** — Les seuils de la règle métier (derniere_connexion_jours > 30,
# > csat ≤ 6) proviennent de l'EDA §6 pour éviter toute fuite d'information du jeu de test
# > vers le protocole de validation.
# >
# > **Impact sur la suite** — §9 compare formellement RL vs GBM sur le protocole §8.8 ;
# > §12 mesure les latences unitaire et batch et complète l'éco-bilan ; §10 implémente
# > le contrat d'API CRM et le pipeline de déploiement continu.
