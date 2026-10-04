# %% [markdown]
# ## 14. Conclusion
#
# On veut répondre à trois questions du commanditaire : que sait faire la solution, comment
# l'utiliser chaque mois, et où ne pas s'y fier. Tous les chiffres viennent de
# `charger_synthese()`, la même fonction que le résumé exécutif (§1) : début et fin du notebook
# affichent donc les mêmes valeurs, relues depuis les artefacts des §9 et §12.

# %%
from IPython.display import Markdown, display

from churn_saas import config
from churn_saas.format_fr import entier, euros, nombre, pourcentage, styler_fr
from churn_saas.models.economics import table_deux_niveaux
from churn_saas.synthese import charger_synthese

synthese = charger_synthese()
y = synthese.y
cibles = config.CIBLES_PERFORMANCE
bilan = synthese.bilan
reference = synthese.reference
attribution = synthese.attribution
annuel = synthese.gain_annuel
concentration = synthese.concentration
latence = synthese.latence
protocole = synthese.protocole
# Note de référence : le jeu de test de §9.14, mis de côté avant tout choix
test = synthese.test

# Baselines non optimisées comparées sur les mêmes plis (§9.3)
comparaison = synthese.comparaison_modeles["pr_auc_mean"]
pr_auc_lr = comparaison["Baseline — régression logistique"]
pr_auc_lgbm = comparaison["LightGBM"]

# Cible secondaire : même règle de sélection qu'en §12.10
tableau_clv = synthese.tableau_clv
champion_clv = synthese.champion_clv
metriques_clv = tableau_clv.loc[champion_clv]
clv_conforme = synthese.champion_clv_conforme is not None

fourchette_annuelle = (
    f"{euros(annuel['annuel_min'])} à {euros(annuel['annuel_max'])} pour un taux de succès "
    f"des gestes de {pourcentage(annuel['taux_succes_min'], 0)} à "
    f"{pourcentage(annuel['taux_succes_max'], 0)}"
)


def _verdict(atteint: bool) -> str:
    return "✅" if atteint else "❌"


def _avec_ic(cle: str, format_valeur=lambda v: nombre(v, 3)) -> str:
    t = test[cle]
    return (
        f"**{format_valeur(t['valeur'])}** "
        f"[{format_valeur(t['ic_bas'])} ; {format_valeur(t['ic_haut'])}]"
    )


# %% [markdown]
# ### 14.1 Synthèse des résultats

# %%
display(Markdown(f"""
**Le modèle livré** est une **{synthese.nom_modele_final}** (choisie en §9.12, réapprise sur
toutes les données en §9.15). À protocole identique
(§9.3), LightGBM obtenait une PR-AUC moyenne de {nombre(pr_auc_lgbm, 3)} contre
{nombre(pr_auc_lr, 3)} pour la régression logistique : l'écart ne justifiait pas un modèle plus
complexe et moins lisible (règle de complexité, §9.7).

| Indicateur | Valeur [IC 95 %] | Cible fixée *a priori* (§8.2) | Atteinte |
|---|---|---|---|
| PR-AUC (aire sous la courbe précision-rappel), jeu de test | {_avec_ic('pr_auc')} | ≥ {nombre(cibles['pr_auc_min'], 2)} | {_verdict(test['pr_auc']['valeur'] >= cibles['pr_auc_min'])} |
| ROC-AUC (aire sous la courbe ROC), jeu de test | {_avec_ic('roc_auc')} | ≥ {nombre(cibles['roc_auc_min'], 2)} | {_verdict(test['roc_auc']['valeur'] >= cibles['roc_auc_min'])} |
| Recall (rappel) au seuil de vigilance, jeu de test | {_avec_ic('recall', lambda v: pourcentage(v, 1))} | ≥ {pourcentage(cibles['recall_vigilance_min'], 0)} | {_verdict(test['recall']['valeur'] >= cibles['recall_vigilance_min'])} |
| PR-AUC en validation croisée : 15 plis (moyenne ± écart-type) ; hors pli, tout le portefeuille | {nombre(protocole['pr_auc_mean'], 3)} ± {nombre(protocole['pr_auc_std'], 3)} ; {nombre(synthese.pr_auc_oof, 3)} | — | — |
| Latence unitaire, 95e centile (§9.13) | {nombre(latence['latence_unitaire_ms_p95'], 1)} ms | ≤ {entier(cibles['latence_unitaire_ms'])} ms | {_verdict(latence['latence_unitaire_ms_p95'] <= cibles['latence_unitaire_ms'])} |
| Lot de {entier(latence['n_batch'])} comptes (§9.13) | {nombre(latence['latence_batch_5k_s'], 1)} s | ≤ {entier(cibles['latence_batch_5k_s'])} s | {_verdict(latence['latence_batch_5k_s'] <= cibles['latence_batch_5k_s'])} |
| Équité par pays, taille et secteur (§12.14) | {"écarts sous les seuils du §4.4" if not synthese.attributs_equite_a_revoir else "revue requise : " + ", ".join(synthese.attributs_equite_a_revoir)} | seuils du §4.4 | {_verdict(not synthese.attributs_equite_a_revoir)} |

Le **jeu de test** (§9.14) regroupe {pourcentage(config.PART_TEST, 0)} des comptes, mis de côté avant tout choix et notés une
seule fois : c'est la note de référence, avec son intervalle de confiance (bootstrap,
rééchantillonnage des comptes du test). « Hors pli » (out-of-fold) : chaque compte est noté par
un modèle qui ne l'a pas vu ; ces prédictions couvrent tout le portefeuille et servent aux euros.
Un tri au hasard obtiendrait une PR-AUC égale à la part de churners, soit
{pourcentage(y.mean(), 0)}.

**La valeur apportée** se mesure contre ce que ferait l'équipe sans modèle, à budget égal :
appeler ses {entier(synthese.capacite)} plus gros comptes du mois (§12.12).

- Liste du modèle : **{entier(bilan['n_churners'])} churners sur {entier(bilan['n_contactes'])}
  comptes appelés**, contre {entier(reference['n_churners'])} pour la liste sans modèle.
- Marge espérée en plus : **{euros(attribution['gain_attribuable'])} le premier mois** et
  **{euros(annuel['annuel'])} sur un an** ({fourchette_annuelle}), soit
  {nombre(annuel['annuel'] / annuel['premier_mois'], 1)} fois le premier mois et non 12 fois :
  les gros comptes à risque ne se découvrent qu'une fois, le modèle les atteint surtout plus tôt.
- Gain concentré : {entier(concentration['n_top'])} comptes font
  {pourcentage(concentration['part_top'], 0)} du gain du premier mois.

**La cible secondaire**, la valeur vie client (CLV, régression, §12.10) : `{champion_clv}`
explique {pourcentage(metriques_clv['r2'], 0)} de sa variabilité (R²) et réduit l'erreur moyenne
de {pourcentage(metriques_clv['gain_mae'], 0)} par rapport à la baseline (référence de base)
médiane — {"cibles du §8.2 atteintes" if clv_conforme else "⚠️ cibles du §8.2 non atteintes, modèle non déployable"}.
Elle sert à prioriser, jamais de feature (variable explicative) du churn.
"""))

# %% [markdown]
# **Ce qu'il faut retenir.** Le modèle tient ses cibles fixées avant l'entraînement, en
# performance, mesurée sur des comptes jamais vus, comme en latence. Sa valeur ne vient pas d'un « ROI contre l'inaction » mais de
# l'écart avec la pratique actuelle : à budget d'appels égal, il trouve plus de partants parmi
# les comptes à fort enjeu, et plus tôt. Les montants en euros dépendent d'hypothèses
# économiques encore à mesurer (§14.3).

# %% [markdown]
# ### 14.2 Règle de décision à deux niveaux
#
# Un seul seuil devait arbitrer entre deux objectifs contraires : ne manquer aucun partant, et
# ne pas noyer une équipe qui ne peut passer qu'un nombre limité d'appels. On les sépare donc
# (§12.6). Le **niveau 1, vigilance**, fixe le threshold (seuil) au plus haut score qui détecte
# encore la part visée des churners : le recall (rappel) cible, mesuré hors pli. Ces comptes
# reçoivent une surveillance peu coûteuse (actions automatisées, revue hebdomadaire). Le
# **niveau 2, appels**, retient parmi eux les comptes où un geste CSM rapporte le plus en
# espérance, dans la limite de la capacité de l'équipe.

# %%
deux_niveaux = table_deux_niveaux(y, synthese.proba_oof, synthese.mrr)
colonnes = {
    "niveau": "Niveau",
    "n_comptes": "Comptes",
    "part_portefeuille": "Part du portefeuille",
    "n_churners": "Churners détectés",
    "recall": "Recall",
    "precision": "Precision",
    "n_faux_negatifs": "Churners manqués",
    "valeur_faux_negatifs_eur": "Marge des manqués",
}
display(
    Markdown(
        f"Seuil de vigilance : **{nombre(deux_niveaux.loc[0, 'seuil'], 3)}** pour un recall cible "
        f"de {pourcentage(config.RECALL_CIBLE_VIGILANCE, 0)}."
    )
)
display(
    styler_fr(
        deux_niveaux[list(colonnes)].rename(columns=colonnes).set_index("Niveau"),
        {
            "Comptes": entier,
            "Part du portefeuille": pourcentage,
            "Churners détectés": entier,
            "Recall": pourcentage,
            "Precision": pourcentage,
            "Churners manqués": entier,
            "Marge des manqués": euros,
        },
    )
)
vigilance, appels = deux_niveaux.iloc[0], deux_niveaux.iloc[1]

# %%
display(Markdown(f"""
**Ce qu'il faut retenir.** Le niveau 1 signale {entier(vigilance['n_comptes'])} comptes
({pourcentage(vigilance['part_portefeuille'], 0)} du portefeuille) et n'en laisse échapper que
{entier(vigilance['n_faux_negatifs'])} churners : c'est lui qui limite les faux négatifs. Sur le
jeu de test, un seuil appris sans lui détecte {pourcentage(test['recall']['valeur'], 0)} des
partants : la promesse tient sur de nouveaux comptes. Le
niveau 2 n'appelle que {entier(appels['n_comptes'])} comptes, avec une precision (précision) de
{pourcentage(appels['precision'], 0)} : son rôle n'est pas de couvrir tous les partants mais de
placer les heures CSM là où elles rapportent le plus. Le seuil appris est persisté dans
`seuil_vigilance.json` et relu par l'API et le batch (traitement par lot) nocturne.
"""))

# %% [markdown]
# ### 14.3 Limites assumées
#
# Connaître les limites d'un modèle, c'est savoir où ne pas s'y fier.
#
# | Famille | Limite | Conséquence | Mitigation |
# |---|---|---|---|
# | Données | Une seule extraction, tous les comptes photographiés à la même date | Aucune validation temporelle possible (§6.11) | Cross-validation (validation croisée) stratifiée répétée (§8.8) et jeu de test stratifié (§9.14) ; validation temporelle sur extractions successives (§13.7) |
# | Données | `commentaire_csm` exclu | Date de rédaction inconnue : data leakage (fuite de données) non exclu (§6.6) | Réintégrable si le CRM horodate chaque commentaire |
# | Données | `valeur_vie_client_eur` de méthode non documentée | CLV qui intègre déjà le risque de départ (§6.7) | Valeur à risque = MRR × horizon (§12.11) ; méthode à documenter avec la Finance |
# | Économie | Taux de succès d'un geste issu d'un rapport sectoriel | Gain en euros conditionnel, pas mesuré | Sensibilité (§12.7) ; mesure par groupe témoin |
# | Économie | Gain concentré sur quelques gros comptes | Une erreur sur leur MRR fausse le bilan (§12.12) | Validation du MRR par la Finance |
# | Économie | Coût d'un geste limité au temps CSM ; coût du projet estimé, non devisé | Remises non comptées ; ROI du projet conditionnel | Valeur mesurée contre une liste sans modèle, à coût égal ; taux de succès d'équilibre (§12.12) ; coûts à confirmer par le contrôle de gestion |
# | Modèle | Cible sans horizon défini | Suivi en production difficile (§13.7) | Horizon à fixer avec le métier avant le réentraînement |
# | Modèle | Étiquettes connues 1 à 12 mois après la prédiction | Performance non mesurable en temps réel | Dérive des entrées suivie immédiatement (PSI/KS, §13.3) ; évaluation par cohorte de renouvellements (§13.7) |
# | Modèle | Calibration (accord score / fréquence observée) liée à la part de churners actuelle | Si elle change, scores et euros se décalent | Contrôle à chaque vague de renouvellements ; réestimation au réentraînement |
# | Modèle | Un seul jeu de test, de taille modeste | Note de référence bruitée | Intervalle de confiance par bootstrap (§9.14) ; évaluation par cohorte en production (§13.7) |
# | Modèle | Régression CLV évaluée sur un seul découpage (test de 20 %) | Pas d'intervalle de confiance sur R² et MAE | Validation croisée répétée au prochain cycle (§13.1) |
# | Technique | Monitoring (surveillance) démontré sur données simulées (§13.9) | Seuils d'alerte non confrontés au trafic réel | Calage sur les premiers mois de production |
# | Technique | CodeCarbon sous WSL2 | Empreinte carbone estimée, non mesurée (§9.8) | Déclaration explicite, facteur d'émission France |
#
# **Ce qu'il faut retenir.** Aucune limite n'invalide le modèle, mais deux familles pèsent plus
# que les autres : les hypothèses économiques conditionnent les montants annoncés, le délai des
# étiquettes et l'horizon de la cible conditionnent la mesure de la performance réelle. La
# feuille de route vise d'abord ces deux familles.

# %% [markdown]
# ### 14.4 Recommandations et feuille de route
#
# **Avant le déploiement**
#
# 1. **Déployer le scoring (calcul des scores) nocturne** (`flows/scoring_batch.py`) : les
#    scores et le niveau de chaque compte alimentent le CRM chaque nuit ; le passage du dimanche
#    nourrit la revue CS du lundi.
# 2. **Mettre en place un groupe témoin** (environ 10 % des comptes signalés, tirés au sort et
#    laissés sans geste, §4.5). Les gestes changent l'issue des comptes traités : seul ce groupe
#    fournit des étiquettes non biaisées et mesure l'effet net des gestes.
# 3. **Mesurer les hypothèses économiques** dans le CRM : durée et nombre de gestes dès le
#    premier mois, taux de succès au fil des renouvellements, par comparaison au groupe témoin.
#
# **Chantiers de données en parallèle** : faire valider par la Finance le MRR des gros comptes ;
# fixer avec le CS Lead l'horizon de la cible (3, 6 ou 12 mois) ; horodater `commentaire_csm`.
#
# **En production** : un signal de dérive convoque le comité trimestriel (§13.11), il ne
# réentraîne jamais seul. Un nouveau modèle n'est promu que s'il passe la gate (règle de
# promotion) du §10.4 et si son recall au seuil de vigilance reste au-dessus du minimum du §8.2.
#
# **Feuille de route** — pistes classées par rapport impact / effort (§13.1) ; les gains de
# PR-AUC sont des ordres de grandeur à confirmer par expérimentation.
#
# | Priorité | Piste | Gain supposé | Effort |
# |---|---|---|---|
# | 1 | Données d'usage fines (événements par fonctionnalité au lieu d'agrégats sur 30 jours) : voir la baisse d'engagement dès son apparition | +0,04 à +0,08 de PR-AUC | 2 mois-ingénieur |
# | 2 | Historique sur 3 ans (durée de conservation du §4.1) : mieux capter les cycles de renouvellement longs | +0,03 à +0,06 de PR-AUC | 1 mois-ingénieur |
# | 3 | Modèle de survie : prédire *quand* le client part, pour prioriser les échéances proches | autre métrique (C-index) | 3 mois-ingénieur |
#
# Écartés à court terme : un modèle de langage sur `commentaire_csm`, qui n'est qu'un jeu de
# libellés normalisés et dont l'exclusion tient à la fuite, pas à la technique (§6.6) ;
# l'inférence en flux continu, inutile pour un phénomène qui évolue sur des semaines avec des
# sources rafraîchies une fois par jour.

# %%
display(Markdown(f"""
**Ce qu'il faut retenir.** Le modèle place les futurs partants en tête de liste : PR-AUC de
{nombre(test['pr_auc']['valeur'], 3)} sur des comptes jamais vus, contre {nombre(y.mean(), 2)} au
hasard. La règle à deux
niveaux surveille {entier(vigilance['n_comptes'])} comptes pour ne manquer que
{pourcentage(1 - vigilance['recall'], 0)} des churners, et concentre les
{entier(appels['n_comptes'])} appels mensuels là où ils rapportent le plus : environ
{euros(annuel['annuel'])} de marge espérée sur un an de plus qu'une équipe sans modèle, sous
réserve des hypothèses économiques. *Le modèle classe, le CSM décide* : aucune action n'est
déclenchée sans intervention humaine. **Décision attendue** : lancer le scoring nocturne avec
groupe témoin, puis réviser les hypothèses à chaque comité trimestriel.
"""))
