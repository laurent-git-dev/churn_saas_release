# %% [markdown]
# ## 1. Résumé exécutif
#
# Une page pour décider : le problème, la démarche, trois chiffres clés, la recommandation et
# les limites. Chaque chiffre est calculé ici à partir des résultats produits plus loin (§9 à
# §12) ; la section citée entre parenthèses donne le détail.
#
# > **Note de lecture.** `make notebook` exécute cette section en dernier puis l'affiche en
# > tête : ses chiffres viennent de la même exécution que le reste, d'où ses numéros
# > d'exécution (`[n]`) plus élevés.

# %%
import warnings

from IPython.display import Markdown, display

from churn_saas import config
from churn_saas.format_fr import entier, euros, nombre, pourcentage
from churn_saas.models.economics import table_deux_niveaux
from churn_saas.synthese import charger_synthese

warnings.filterwarnings("ignore")

# Mêmes chiffres que la conclusion (§14) : même fonction de synthèse, mêmes artefacts. S'il en
# manque un, la section s'arrête au lieu d'afficher des valeurs de substitution.
synthese = charger_synthese()
y = synthese.y
_cap = synthese.capacite
_pr_b1 = float(synthese.comparaison_modeles["pr_auc_mean"].get("B1 — Règle métier", float("nan")))
_cibles = config.CIBLES_PERFORMANCE
deux_niveaux = table_deux_niveaux(y, synthese.proba_oof, synthese.mrr)  # règle du §12.6
vigilance, appels = deux_niveaux.iloc[0], deux_niveaux.iloc[1]
_annuel = synthese.gain_annuel
_mensuel = _annuel["mensuel"]
_test = synthese.test  # note du jeu de test (§9.14)

# %% [markdown]
# ### 1.1 Le problème métier

# %%
display(Markdown(f"""
Un éditeur SaaS B2B suit **{entier(len(y))} comptes** (PME et grandes entreprises
européennes, abonnements annuels) ; **{pourcentage(y.mean(), 1)}** sont partis sur
l'historique. Son équipe Customer Success (CS), trois chargés de clientèle (CSM), ne peut mener
qu'environ **{_cap} actions de rétention par mois**, et les signaux de départ sont repérés trop
tard, quand le client a déjà décidé. L'objectif : un **score de risque de départ** recalculé
chaque nuit, pour placer ces actions là où elles rapportent le plus, avant l'échéance du
contrat (§2).
"""))

# %% [markdown]
# **Ce qu'il faut retenir.** La ressource rare n'est pas la donnée mais le temps des CSM. Le
# modèle ne remplace pas leur jugement : il leur dit **qui regarder et qui appeler en premier**.

# %% [markdown]
# ### 1.2 La démarche
#
# - **Assainir avant de calculer (§5 à §7).** Doublons, valeurs impossibles et formats
#   incohérents sont corrigés. Surtout, le data leakage (fuite de données) est écarté : des
#   colonnes qui « connaissent » déjà la réponse, comme un indicateur de santé calculé *après*
#   la résiliation (§6.5) ou des commentaires CSM sans date fiable (§6.6). Trois variables
#   *leurres*, sans lien avec le départ, sont détectées puis retirées (§12.9).
# - **Battre une règle simple (§8, §9).** Deux baselines (références de base) sans
#   apprentissage — le hasard et une règle métier applicable dans le CRM — sont comparées à
#   trois modèles (régression logistique, forêt aléatoire, LightGBM) sur les mêmes découpages
#   de cross-validation (validation croisée). Toute transformation est apprise à l'intérieur de
#   chaque découpage : rien du jeu d'évaluation ne fuit vers le modèle.
# - **Juger sur des comptes jamais vus (§9.14).** Un jeu de test, mis de côté avant tout choix,
#   n'est ouvert qu'une fois, pour noter le modèle retenu : c'est sa note de référence.
# - **Préférer le plus simple (§9.3.1).** Un modèle plus complexe n'est retenu que s'il fait
#   significativement mieux, d'un écart minimal fixé à l'avance.
# - **Décider en deux niveaux (§12.6).** Un seul seuil ne peut pas à la fois manquer peu de
#   partants et respecter la capacité de l'équipe. Le niveau 1 (*vigilance*) surveille
#   largement pour limiter les partants manqués ; le niveau 2 (*appels*) y choisit les comptes
#   où une action rapporte le plus (probabilité × valeur du compte − coût de l'action).
#
# **Ce qu'il faut retenir.** Les règles du jeu (mesures, seuils, protocole) sont fixées *avant*
# de voir les résultats : on cherche une liste d'appels fiable et un outil maintenable par une
# petite équipe, pas le meilleur score possible.

# %% [markdown]
# ### 1.3 Trois chiffres clés
#
# La qualité du modèle est mesurée sur le **jeu de test** : des comptes qu'aucun choix n'a vus,
# comme un compte nouveau en production ; l'intervalle de confiance à 95 % donne la marge
# d'incertitude. Volumes et euros portent sur tout le portefeuille, mesurés *out-of-fold* (hors
# pli) : chaque compte y est noté par un modèle qui ne l'a pas vu à l'entraînement.

# %%
display(Markdown(f"""
| Question | Chiffre clé | Repère |
|---|---|---|
| Le modèle ({synthese.nom_modele_final}) sait-il repérer les partants ? | **ROC-AUC {nombre(_test["roc_auc"]["valeur"], 3)}** [{nombre(_test["roc_auc"]["ic_bas"], 2)} ; {nombre(_test["roc_auc"]["ic_haut"], 2)}] · PR-AUC {nombre(_test["pr_auc"]["valeur"], 3)} [{nombre(_test["pr_auc"]["ic_bas"], 2)} ; {nombre(_test["pr_auc"]["ic_haut"], 2)}], jeu de test | Cibles fixées à l'avance : {nombre(_cibles["roc_auc_min"], 2)} et {nombre(_cibles["pr_auc_min"], 2)} · règle métier : PR-AUC {nombre(_pr_b1, 3)} · hasard : {nombre(y.mean(), 2)} (§9.3, §9.14) |
| Combien de comptes surveiller pour manquer peu de partants ? | **{pourcentage(vigilance["part_portefeuille"], 0)} du portefeuille** ({entier(vigilance["n_comptes"])} comptes) pour un recall de {pourcentage(vigilance["recall"], 0)} | Recall (rappel) visé : {pourcentage(config.RECALL_CIBLE_VIGILANCE, 0)}, tenu sur le jeu de test ({pourcentage(_test["recall"]["valeur"], 0)}) ; {entier(vigilance["n_faux_negatifs"])} partants restent hors surveillance (§12.6) |
| Que rapporte le modèle sur un an ? | **{euros(_annuel["annuel"])}** de marge espérée | De {euros(_annuel["annuel_min"])} à {euros(_annuel["annuel_max"])} selon le taux de succès des actions, par rapport à une équipe qui appelle ses plus gros comptes (§12.12) |
"""))

# %%
_atteintes = sum(_test[m]["valeur"] >= _cibles[f"{m}_min"] for m in ("roc_auc", "pr_auc"))
_verdict = {2: "atteint ses deux cibles", 1: "n'atteint qu'une de ses deux cibles"}.get(
    _atteintes, "n'atteint aucune de ses cibles"
)
_arbres = synthese.comparaison_modeles["pr_auc_mean"].reindex(["Forêt aléatoire", "LightGBM"])
_ecart = synthese.protocole["pr_auc_mean"] - _arbres.max()
_simple = (
    f"Ce modèle linéaire, le plus simple, devance même les modèles à arbres "
    f"({nombre(_ecart, 3, signe=True)} de PR-AUC) : la complexité n'apportait rien ici. "
    if _ecart > 0 and "logistique" in synthese.nom_modele_final
    else ""
)
display(Markdown(f"""
**Ce qu'il faut retenir.** Sur des comptes jamais vus, le classement {_verdict}, fixées
avant l'entraînement, et fait nettement mieux que la règle métier, sa vraie concurrente.
{_simple}Un partant reçoit un score plus haut qu'un client fidèle dans {pourcentage(_test["roc_auc"]["valeur"], 0)} des cas (hasard : 50 %). Le
niveau 1 ramène la surveillance à {entier(vigilance["n_comptes"])} comptes ; le niveau 2 n'en
appelle que {entier(appels["n_comptes"])} par mois, avec une precision (part de vrais partants)
de {pourcentage(appels["precision"], 0)}. Sur un an, le modèle contacte
{entier(_mensuel["n_churners_modele"].sum())} partants contre
{entier(_mensuel["n_churners_reference"].sum())} sans lui ; son gain annuel vaut
{nombre(_annuel["annuel"] / _annuel["premier_mois"], 1)} fois le premier mois, et non 12 fois,
car les comptes à fort enjeu ne se présentent qu'une fois : **il les atteint surtout plus tôt**.
{entier(synthese.n_rentables)} comptes justifieraient une action rentable pour {_cap}
possibles : **la limite est la capacité de l'équipe**, pas le modèle (§12.7).
"""))

# %% [markdown]
# ### 1.4 Recommandation
#
# 1. **Déployer le calcul nocturne des scores** et la règle à deux niveaux : les comptes en
#    vigilance alimentent la revue hebdomadaire du portefeuille et des actions automatisées ;
#    les appels CSM du mois suivent la liste du niveau 2 (§10, §14.2).
# 2. **Mettre en place un groupe témoin dès le premier mois** : une petite part des comptes à
#    risque, tirée au sort et laissée sans action. Sans lui, impossible de mesurer l'effet réel
#    des actions, puisqu'elles modifient l'issue des comptes contactés (§4.5).
# 3. **Valider les hypothèses économiques** (taux de succès d'une action, revenu des gros
#    comptes) et **fixer avec le métier l'horizon du départ** à prédire avant le prochain
#    réentraînement (§14.4).
#
# **Ce qu'il faut retenir.** Le modèle est prêt pour un déploiement progressif ; les montants en
# euros resteront des estimations tant que le groupe témoin n'aura pas mesuré l'effet réel des
# actions. *Le modèle classe, le CSM décide* : aucune action n'est déclenchée sans humain.

# %% [markdown]
# ### 1.5 Limites principales

# %%
display(Markdown(f"""
| Limite | Conséquence | Réponse prévue |
|---|---|---|
| Horizon du départ non défini, une seule extraction datée | Pas de validation sur une période future (§6.11) | Horizon fixé avec le métier ; validation sur les extractions successives (§13.7) |
| Un départ ne s'observe qu'à l'échéance du contrat | Performance non mesurable en temps réel | Surveillance de la dérive des données (§13.3), puis bilan par vague de renouvellements (§13.7) |
| Taux de succès d'une action ({pourcentage(config.HYPOTHESES_ECONOMIQUES["taux_succes_retention"], 0)}) issu d'un rapport sectoriel | Les euros sont conditionnels | Sensibilité (§12.7) ; mesure par le groupe témoin |
| Gain concentré sur les comptes à fort revenu, simulé à portefeuille figé | Une erreur sur leur revenu fausserait le bilan | Contrôle sièges × prix catalogue (§12.12) ; validation par la Finance (§14.4) |
| Probabilités et seuil de vigilance réglés sur la part actuelle de partants | Si elle change, scores, seuil et euros se décalent | Contrôle du recall à chaque vague, réglage au réentraînement (§13.8) |
"""))

# %% [markdown]
# **Ce qu'il faut retenir.** Aucune de ces limites n'invalide le modèle : les premières touchent
# à la **mesure** (horizon, délai d'observation), les suivantes aux **montants en euros**. Le
# groupe témoin et la fixation de l'horizon répondent aux deux ; la liste complète est en §14.3.

# %% [markdown]
# > ### 📋 Journal de bord — Résumé exécutif
# >
# > **Décisions retenues** — Cinq blocs pour un décideur, trois chiffres clés alignés sur les
# > mesures co-principales : qualité du classement (ROC-AUC, PR-AUC) et recall au seuil de
# > vigilance, lus sur le jeu de test (§9.14), puis valeur annuelle. Tout est relu depuis les
# > artefacts de §9 et §12 par la même fonction que la conclusion (§14).
# >
# > **Alternatives écartées** — Chiffres écrits à la main : faux dès la régénération suivante.
# > Meilleure valeur d'Optuna : optimiste (§9.7). Mesure hors pli comme note de référence : la
# > sélection du modèle a vu ces comptes. Gain du seul premier mois : il invitait à
# > multiplier par 12. ROI face à « ne rien faire » : ordre de grandeur peu crédible, laissé au
# > §12.12.
# >
# > **Difficultés rencontrées** — La première version relisait des résultats calculés plus loin
# > (§9, §12) et plantait sur une exécution vierge ; le repli sur des valeurs de substitution
# > affichait des euros fictifs, remplacé par un arrêt explicite. La note de référence a changé
# > deux fois (protocole, hors pli, puis test) : la sélection avait déjà vu ces comptes.
# >
# > **Impact sur la suite** — Point d'entrée du lecteur ; il reste aligné sur la conclusion.
