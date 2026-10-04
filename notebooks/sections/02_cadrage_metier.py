# %% [markdown]
# ## 2. Cadrage métier et cas d'usage
#
# Avant toute ligne de code, il faut répondre à la question : *pourquoi* ce projet ? Cette section
# identifie le besoin, décrit trois cas d'usage, chiffre la valeur attendue, traduit le besoin en
# problème de machine learning et pose les hypothèses métier que l'exploration (§6) confrontera
# aux données.

# %% [markdown]
# ### 2.1 Contexte de l'éditeur SaaS B2B
#
# L'éditeur vend une plateforme **SaaS B2B** à des PME et grandes entreprises européennes, sur
# abonnement annuel à renouvellement tacite. Sa croissance repose sur l'acquisition de comptes et
# sur leur **rétention** : chaque **churn (résiliation)** fait perdre un revenu récurrent.
#
# **Problème.** L'équipe Customer Success (CS) suit environ 5 000 comptes : à ce volume, le suivi
# manuel est impossible et les signaux arrivent souvent après la décision du client.
#
# **Finalité.** Un **score d'aide à la priorisation** : il classe les comptes par risque pour que
# l'équipe CS agisse avant l'échéance. Il ne prend **aucune décision automatique** : le CSM
# (gestionnaire du compte) choisit toujours l'action.

# %% [markdown]
# ### 2.2 Trois cas d'usage opérationnels
#
# | Dimension | 1. Revue hebdomadaire du portefeuille | 2. Alerte sur signal faible | 3. Préparation d'un renouvellement |
# |---|---|---|---|
# | **Qui** | Le CS Lead (responsable de l'équipe) | Le CSM du compte | Le CSM, avec le CS Lead si MRR ≥ 5 000 € |
# | **Quand** | Lundi, avant la réunion de 9 h | Chaque matin, si un compte ≥ 1 000 € de MRR passe en `ALERTE_ROUGE` | 90 jours avant l'échéance du contrat |
# | **Décision** | Choisir les 10 comptes prioritaires de chaque CSM | Contact proactif sous 48 h | Stratégie de renouvellement : migration de plan, remise, formation, escalade |
# | **Donnée reçue** | Liste triée par score décroissant, variation sur 7 jours, MRR, CSM en charge | Notification CRM : score, score à J−7, 3 variables les plus contributives (SHAP), derniers tickets | Fiche compte : score, valeur vie estimée (CLV), variables explicatives |
# | **Mode de calcul** | Batch (traitement par lot) nocturne ; passage du dimanche disponible à 8 h 30 | Même batch, alertes dans le CRM avant 6 h | API appelée à l'ouverture de la fiche, 95 % des réponses en moins de 200 ms |
# | **Valeur** | Préparation de 2 h ramenée à 15 min ; fin du choix « de mémoire » | Agir avant que l'inactivité ne devienne irréversible | Meilleur renouvellement des comptes à forte valeur |
#
# **Ce qu'il faut retenir.** Les sources n'étant rafraîchies qu'une fois par jour, un calcul
# « temps réel » n'apporterait aucun signal nouveau : le batch nocturne sert les cas 1 et 2,
# l'API ne sert qu'à la consultation à l'écran (cas 3).

# %% [markdown]
# ### 2.3 Hypothèses
#
# | Hypothèse économique | Valeur | Source |
# |---|---|---|
# | Marge brute SaaS B2B | 72 % | OpenView Partners, *SaaS Benchmarks* 2023 |
# | Coût horaire CSM chargé | 55 €/h | Glassdoor 2024 × coefficient de charges 1,45 |
# | Durée d'un geste de rétention | 2 h | Cadrage : préparation, appel, suivi CRM (à mesurer en production) |
# | Taux de succès d'un geste | 30 % | Gainsight, *Customer Success Report* 2023 |
# | Horizon de valeur | 12 mois | Contrats annuels |
# | Capacité mensuelle CS | 45 gestes | Cadrage : 3 CSM × 15 gestes par mois (~20 % de leur temps) |
# | Portefeuille : comptes, taux de churn, MRR médian | 5 000 ; 28 % ; 450 € | Commanditaire (`config.PORTEFEUILLE_CADRAGE`), vérifié en §12 |
#
# **Ce qu'il faut retenir.** Ces valeurs sont rangées dans `config.HYPOTHESES_ECONOMIQUES` et
# `config.PORTEFEUILLE_CADRAGE` : tout calcul les lit de là, et §12 les confronte aux mesures.

# %% [markdown]
# ### 2.4 Valeur attendue chiffrée
#
# On veut savoir combien rapporte un meilleur ciblage. Un geste de rétention ne sauve un compte
# que s'il vise un vrai churner : la valeur du modèle tient donc à la part de vrais churners
# parmi les comptes contactés. Sans modèle, cette part vaut le taux de churn (choix au hasard) ;
# au cadrage, on retient pour le modèle la cible de PR-AUC (aire sous la courbe
# précision-rappel), mesurée réellement en §12.

# %%
import pandas as pd
from IPython.display import Markdown, display

from churn_saas import config
from churn_saas.format_fr import entier, euros, nombre, pourcentage

hyp = config.HYPOTHESES_ECONOMIQUES
portefeuille = config.PORTEFEUILLE_CADRAGE
taux_churn_observe = float(portefeuille["taux_churn"])
gestes_par_an = int(hyp["capacite_gestes_mois"]) * 12
valeur_par_compte_sauve_eur = float(portefeuille["mrr_median_eur"]) * hyp["horizon_mois"]
precision_ciblage_cadrage = config.CIBLES_PERFORMANCE["pr_auc_min"]
taux_sauvetage_avec_modele = precision_ciblage_cadrage * hyp["taux_succes_retention"]
taux_sauvetage_sans_modele = taux_churn_observe * hyp["taux_succes_retention"]
cout_gestes_an_eur = gestes_par_an * hyp["cout_horaire_csm_eur"] * hyp["duree_geste_retention_h"]
# Marge brute du revenu sauvé si chaque geste sauvait son compte ; le temps CSM est une charge
# déduite en totalité, identique avec ou sans modèle : il s'annule dans le gain incrémental
marge_potentielle_eur = gestes_par_an * valeur_par_compte_sauve_eur * hyp["marge_brute_pct"]
valeur_nette_an_eur = taux_sauvetage_avec_modele * marge_potentielle_eur - cout_gestes_an_eur
gain_incremental_eur = (
    taux_sauvetage_avec_modele - taux_sauvetage_sans_modele
) * marge_potentielle_eur
display(
    Markdown(
        "| Grandeur | Valeur |\n|---|---|\n"
        f"| Churners attendus sans intervention | {entier(portefeuille['nb_comptes'] * taux_churn_observe)} comptes |\n"
        f"| Gestes de rétention par an | {entier(gestes_par_an)} |\n"
        f"| Valeur d'un compte sauvé (MRR médian × horizon) | {euros(valeur_par_compte_sauve_eur)} |\n"
        f"| Comptes sauvés par geste : hasard → ciblage | {pourcentage(taux_sauvetage_sans_modele, 1)} → {pourcentage(taux_sauvetage_avec_modele, 1)} |\n"
        f"| Coût annuel des gestes (temps CSM) | {euros(cout_gestes_an_eur)} |\n"
        f"| Valeur nette annuelle avec ciblage | {euros(valeur_nette_an_eur)} |\n"
        f"| **Gain incrémental du ciblage** | **{euros(gain_incremental_eur)}/an** |\n\n"
        "**Ce qu'il faut retenir.** Le levier unique est la part de vrais churners contactés : "
        f"{pourcentage(taux_churn_observe, 0)} au hasard, {pourcentage(precision_ciblage_cadrage, 0)} "
        f"visés avec le modèle. Le gain estimé, **{euros(gain_incremental_eur)}/an** face à une "
        "approche non ciblée, est volontairement prudent (ni effet d'image, ni renouvellements "
        "induits) ; §12 le recalcule avec la précision réellement obtenue."
    )
)

# %% [markdown]
# ### 2.5 Formulation machine learning et critères de réussite
#
# On veut transformer « quels comptes appeler ? » en une question à laquelle un modèle sait
# répondre. C'est une **classification binaire supervisée** : le modèle apprend sur des comptes
# dont on connaît l'issue (`churn` = 1 s'il a résilié, 0 sinon) et estime pour un nouveau compte
# la probabilité **P(churn = 1 | X)**, où X regroupe les **features (variables explicatives)**
# connues au moment de la prédiction, en cinq familles :
#
# | Famille | Exemples de variables |
# |---|---|
# | Usage | `taux_adoption_pct`, `heures_usage_30j`, `fonctionnalites_utilisees`, `nb_integrations` |
# | Activité | `connexions_30j`, `derniere_connexion_jours`, `utilisateurs_actifs` |
# | Support | `tickets_support_90j`, `delai_reponse_support_h`, `csat` |
# | Facturation | `retards_paiement_12m`, `revenu_mensuel_recurrent_eur` |
# | Contrat et profil | `plan`, `sieges_souscrits`, `anciennete_mois`, `secteur`, `taille_entreprise` |
#
# **Trois sorties**, une par usage : la **probabilité** (fiche compte), le **classement** des
# comptes par risque (revue hebdomadaire) et la **décision** au **threshold (seuil)** métier fixé
# en §12 (alerte). **Problème secondaire** : une régression de `valeur_vie_client_eur` (CLV) sert
# à pondérer la priorisation ; cette valeur dépend du comportement futur du client, elle n'est
# **jamais une feature du churn** (data leakage, fuite de données : `config.COLONNES_INTERDITES`).
#
# **Objectif de décision.** Un churner manqué (faux négatif) coûte bien plus qu'un appel inutile
# (faux positif) : on cherche d'abord à **limiter les faux négatifs**. D'où trois métriques
# co-principales : le **recall (rappel)**, part des churners détectés ; la **PR-AUC**, qualité
# de la liste signalée, retenue pour la sélection car elle reste exigeante quand les churners
# sont minoritaires ; la **ROC-AUC (aire sous la courbe ROC)**, capacité à classer un churner
# au-dessus d'un client fidèle. Contraintes : trois variables explicatives par score (SHAP), un
# data scientist à mi-temps (la DSI exploite, sans profil ML), RGPD à la marge (§4.1).

# %%
cibles = config.CIBLES_PERFORMANCE
display(
    Markdown(
        "| Critère fixé avant toute modélisation | Seuil | Mesure |\n|---|---|---|\n"
        f"| PR-AUC (sélection du modèle) | ≥ {nombre(cibles['pr_auc_min'], 2)} | Cross-validation (validation croisée) pour choisir, puis jeu de test mis de côté avant tout choix (§9.14) |\n"
        f"| ROC-AUC | ≥ {nombre(cibles['roc_auc_min'], 2)} | Idem : le statut se lit sur le test (§9.14, §12.2.4) |\n"
        f"| Recall au seuil de vigilance | {pourcentage(config.RECALL_CIBLE_VIGILANCE, 0)} visé hors pli (out-of-fold), ≥ {pourcentage(cibles['recall_vigilance_min'], 0)} sur le test | Seuil fixé sans le test, appliqué au test (§9.14, §12.6) |\n"
        f"| Latence unitaire / batch de 5 000 comptes | ≤ {entier(cibles['latence_unitaire_ms'])} ms / ≤ {entier(cibles['latence_batch_5k_s'])} s | Mesure directe, §9 |\n"
        f"| Variables suspectes tranchées | {entier(len(config.COLONNES_LEURRES_SUSPECTES))} sur {entier(len(config.COLONNES_LEURRES_SUSPECTES))} | §6 et §12 |\n"
        "| Aucune fuite de données | Test automatisé | `tests/test_no_leakage.py` |\n\n"
        "**Ce qu'il faut retenir.** Les cibles sont écrites avant tout résultat : impossible de "
        f"choisir après coup la métrique la plus flatteuse. Le recall cible de "
        f"{pourcentage(config.RECALL_CIBLE_VIGILANCE, 0)} fonde le premier niveau de la règle de "
        "décision (§12) : surveiller large pour manquer peu de churners, la capacité des CSM "
        "étant gérée ensuite par priorisation."
    )
)

# %% [markdown]
# ### 2.6 Périodicité de revue des indicateurs — décision de cadrage
#
# Un modèle vieillit avec le produit et le marché : on fixe dès maintenant quand l'interroger.
# Règle : **aucun signal statistique ne réentraîne le modèle seul ; il convoque un
# comité**, car une dérive peut venir d'une panne d'intégration CRM qu'un réentraînement
# apprendrait au lieu de la corriger. Aucun modèle n'entre en production sans passer la
# **gate (règle de promotion)** automatique (§10).
#
# | Instance | Participants | Ordre du jour |
# |---|---|---|
# | **Comité trimestriel** (mars, juin, sept., déc.) | CS Lead, data scientist, DPO | Dérive (PSI, KS), performance sur les cohortes de renouvellement, âge du modèle (revue dès 180 jours, échéance connue à l'entraînement) → réentraîner, maintenir ou geler |
# | **Comité ad hoc** (seuil franchi) | Data scientist, CS Lead | Analyse causale et décision sous 72 h |
#
# | Signal | Contrôle | Seuil | Action |
# |---|---|---|---|
# | Part de comptes `ALERTE_ROUGE` | Chaque nuit | Écart > 10 points à la référence | Retour au modèle précédent s'il vient d'être promu, sinon comité ad hoc |
# | Dérive des entrées (PSI) | Chaque semaine | 0,10 à 0,20 / ≥ 0,20 | Noté pour le trimestre / comité ad hoc |
# | PR-AUC d'une cohorte de renouvellements | À chaque vague observée | < 0,65 / < 0,60 | Comité ad hoc / comparaison avec le modèle précédent et retour arrière s'il fait mieux |

# %%
# Les seuils cités dans le tableau ci-dessus doivent rester ceux de la configuration
_seuils_cites = {"pr_auc_min": 0.65, "pr_auc_critique": 0.60, "ecart_alerte_rouge_rollback_pts": 10}
for _cle, _valeur in _seuils_cites.items():
    assert config.CIBLES_PERFORMANCE[_cle] == _valeur, f"§2.6 à mettre à jour : {_cle}"

# %% [markdown]
# **Ce qu'il faut retenir.** La PR-AUC se mesure par cohorte : une résiliation ne s'observe qu'à
# l'échéance. Le rythme trimestriel suit la saisonnalité des renouvellements et la capacité d'un
# data scientist à mi-temps ; il est outillé en §13.

# %% [markdown]
# ### 2.7 Hypothèses métier à tester
#
# Avant de regarder les données, on écrit ce que le métier croit savoir des départs, pour le
# vérifier plutôt que chercher des motifs au hasard : §6 teste chaque hypothèse et rend un verdict.

# %%
_colonnes = ["Hypothèse", "Variable", "Si la variable augmente, le risque…", "Énoncé"]
hypotheses = pd.DataFrame(config.HYPOTHESES_METIER, columns=_colonnes)
hypotheses[_colonnes[2]] = hypotheses[_colonnes[2]].map({"+": "augmente", "-": "diminue"})
_protection = hypotheses["Énoncé"].str.contains("rédui")
hypotheses.insert(1, "Nature", _protection.map({True: "Protection", False: "Risque"}))
display(hypotheses.style.hide(axis="index"))
_nature = hypotheses["Nature"].value_counts()
display(
    Markdown(
        f"**Ce qu'il faut retenir.** {entier(len(hypotheses))} hypothèses : "
        f"{entier(_nature['Risque'])} facteurs de risque et {entier(_nature['Protection'])} facteurs "
        f"de protection. {', '.join(sorted(config.HYPOTHESES_HAUT_DISTRIBUTION))} porte sur le haut "
        f"de la distribution (au-delà du quantile {nombre(config.QUANTILE_HAUT_DISTRIBUTION, 2)}) "
        "pour ne pas doubler le test du CSAT faible. Ce tableau est la colonne vertébrale de "
        "l'exploration (§6) et oriente le feature engineering (ingénierie des variables, §7)."
    )
)

# %% [markdown]
# > ### 📋 Journal de bord — Cadrage métier
# >
# > **Décisions retenues** — Trois cas d'usage (revue hebdomadaire, alerte, renouvellement).
# > Problème posé en classification binaire P(churn = 1 | X), régression de la CLV en problème
# > secondaire hors features. Objectif : limiter les faux négatifs, suivi par recall, PR-AUC et
# > ROC-AUC. Hypothèses H1 à H14 générées depuis `config.HYPOTHESES_METIER`. Comité trimestriel :
# > un signal convoque un comité, il ne réentraîne jamais seul.
# >
# > **Alternatives écartées** — Décision automatique (le score aide, le CSM décide). CLV comme
# > feature (connue après coup : fuite). Accuracy (exactitude) comme métrique (trompeuse quand
# > les churners sont minoritaires). Revue mensuelle (trop lourde pour un data scientist à
# > mi-temps) ou annuelle (trop tardive). Réentraînement automatique sur alerte de dérive.
# >
# > **Difficultés rencontrées** — Paramètres du portefeuille fournis avant tout accès aux données :
# > de simples estimations, confrontées en §5 et §12. Suivi du PR-AUC sur 30 jours glissants
# > abandonné : une résiliation n'est connue qu'au renouvellement du contrat.
# >
# > **Impact sur la suite** — Hypothèses → §6 et §7 ; métriques et cibles → §8 et §9 ; recall
# > cible → règle à deux niveaux de §12 ; périodicité → §13.
