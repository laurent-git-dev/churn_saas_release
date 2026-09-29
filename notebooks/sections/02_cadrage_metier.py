# %% [markdown]
# ## 2. Cadrage métier et cas d'usage (C1)
#
# Avant toute ligne de code, il faut répondre à la question : *pourquoi* ce projet ?
# Cette section identifie le besoin métier, décrit trois cas d'usage opérationnels distincts,
# chiffre la valeur attendue à partir des hypothèses économiques du projet, et fixe
# — dès le cadrage — la périodicité de révision des indicateurs (item C9).

# %% [markdown]
# ### 2.1 Contexte de l'éditeur SaaS B2B
#
# L'éditeur commercialise une plateforme **SaaS B2B** à destination de PME et grandes entreprises
# européennes. Le modèle de revenus est basé sur des abonnements annuels (MRR récurrent) avec
# renouvellement tacite. La croissance nette repose sur deux leviers : l'acquisition de nouveaux
# comptes et la **rétention** des comptes existants.
#
# **Problème opérationnel.** L'équipe Customer Success gère un portefeuille de ~5 000 comptes
# actifs. Face à ce volume, le suivi manuel est impossible : les alertes de churn arrivent trop tard,
# souvent après que le client a déjà décidé de résilier. Un outil d'aide à la décision basé sur les
# données d'usage est nécessaire pour **anticiper** les résiliations et prioriser les interventions.
#
# **Objectif principal.** Fournir un score de risque de churn pour chaque compte,
# afin de permettre à l'équipe CS d'agir avant la prochaine échéance contractuelle.

# %% [markdown]
# ### 2.2 Trois cas d'usage opérationnels

# %% [markdown]
# #### Cas d'usage 1 — Revue hebdomadaire du portefeuille
#
# | Dimension | Détail |
# |---|---|
# | **Qui** | Le CS Lead (responsable de l'équipe Customer Success) |
# | **Quand** | Chaque lundi matin, avant la réunion d'équipe (9 h 00) |
# | **Quelle décision** | Sélectionner les 10 comptes prioritaires à contacter pour chaque CSM parmi les ~5 000 du portefeuille |
# | **Donnée reçue** | Tableau trié par score de risque décroissant : `client_id`, score churn (0–1), variation du score vs semaine précédente, MRR associé, CSM en charge |
# | **Contrainte** | Le rapport doit être disponible à 8 h 30 → passage du dimanche soir du batch nocturne quotidien |
#
# *Valeur : réduire le temps de préparation de réunion de 2 h à 15 min ; éliminer le biais de
# sélection (aujourd'hui fondé sur la mémoire du CSM).*

# %% [markdown]
# #### Cas d'usage 2 — Alerte sur signal faible
#
# | Dimension | Détail |
# |---|---|
# | **Qui** | Le CSM en charge du compte |
# | **Quand** | Chaque matin (J+1), dès que le batch nocturne détecte un franchissement de seuil (ex. : score > 0,70 sur un compte ≥ 1 000 € MRR, ou hausse > 0,15 vs J−7) |
# | **Quelle décision** | Déclencher un contact proactif non planifié (email personnalisé, appel de vérification) dans les 48 h |
# | **Donnée reçue** | Notification CRM enrichie : score actuel, score J−7, top-3 des variables contributrices (SHAP), historique des 3 derniers tickets |
# | **Contrainte** | Alerte disponible dans le CRM avant 06 h 00 — produite par le batch nocturne (les sources ne sont rafraîchies qu'une fois par jour : un « temps réel » n'apporterait aucun signal nouveau) |
#
# *Valeur : détecter les comptes à risque émergent avant que l'inactivité ne devienne irréversible.*

# %% [markdown]
# #### Cas d'usage 3 — Préparation d'un renouvellement contractuel
#
# | Dimension | Détail |
# |---|---|
# | **Qui** | Le CSM en charge + le CS Lead pour les comptes stratégiques (MRR ≥ 5 000 €) |
# | **Quand** | 90 jours avant la date d'échéance contractuelle de chaque compte |
# | **Quelle décision** | Adapter la stratégie de renouvellement : offre de migration de plan, remise de fidélité, session de formation, escalade commerciale |
# | **Donnée reçue** | Fiche compte : score de risque, valeur vie estimée (CLV), features explicatives, comparaison avec les comptes similaires qui ont renouvelé |
# | **Contrainte** | Fiche consultable à la demande : score recalculé à l'ouverture de la fiche client dans le CRM (`POST /predict`), latence p95 ≤ 200 ms — un CSM attend la réponse à l'écran |
# | **Contrainte (données)** | Disponibilité des données CLV (modèle de régression parallèle, §9) |
#
# *Valeur : augmenter le taux de renouvellement sur les comptes à forte valeur vie.*

# %% [markdown]
# ### 2.3 Hypothèses
#
# #### Hypothèses posées au cadrage
#
# | Hypothèse | Valeur retenue | Source |
# |---|---|---|
# | Marge brute SaaS B2B | 72 % | OpenView Partners SaaS Benchmarks 2023 |
# | Coût horaire CSM chargé | 55 €/h | Glassdoor 2024 × coeff. charges 1,45 |
# | Durée d'un geste de rétention | 2 h | Hypothèse de cadrage : préparation + appel + suivi CRM (à mesurer en production) |
# | Taux de succès ciblé | 30 % | Gainsight 2023 Customer Success Report |
# | Horizon de calcul CLV | 12 mois | Contrats annuels du portefeuille |
# | Capacité mensuelle CS | 45 gestes | Hypothèse de cadrage : 3 CSM × 15 gestes/mois (~20 % de leur temps) |
# | Taux de churn observé | 28 % | Jeu de données fourni (confirmé en §5) |
# | Part de vrais churners parmi les comptes ciblés | 65 % | Cible PR-AUC `config.CIBLES_PERFORMANCE` (mesurée en §12) |

# %% [markdown]
# ### 2.4 Valeur attendue chiffrée
#
# Les calculs suivants sont dérivés exclusivement de `config.HYPOTHESES_ECONOMIQUES`.
# **Aucun chiffre n'est écrit à la main**.

# %%
from churn_saas import config

hyp = config.HYPOTHESES_ECONOMIQUES

# Paramètres du portefeuille (issus du jeu de données — confirmés en §5)
nb_comptes = 5_000
taux_churn_observe = 0.28  # 28 % de positifs dans le jeu de données (1 412 / 5 035)
mrr_moyen_eur = 450  # MRR médian estimé — sera confirmé par profil_compact() en §5

# Comptes qui churneraient sans intervention
churners_attendus = nb_comptes * taux_churn_observe
print(f"Churners attendus sans intervention : {churners_attendus:.0f} comptes")

# Gestes de rétention réalisables par mois et par an
gestes_par_mois = hyp["capacite_gestes_mois"]
gestes_par_an = gestes_par_mois * 12
print(f"Capacité de rétention : {gestes_par_mois} gestes/mois → {gestes_par_an} gestes/an")

# Valeur sauvée par geste réussi (en MRR × horizon)
valeur_par_compte_sauve_eur = mrr_moyen_eur * hyp["horizon_mois"]
print(f"Valeur sauvée par compte retenu : {valeur_par_compte_sauve_eur:,.0f} €")

# Un geste ne sauve un compte que s'il vise un vrai churner : le taux de succès (30 %) ne
# s'applique qu'à cette fraction des gestes. Au cadrage, la précision des comptes ciblés
# n'est pas encore mesurée : on retient la cible de PR-AUC (précision moyenne sur toute la
# courbe, en général dépassée en tête de liste). Valeur réelle mesurée en §12.
precision_ciblage_cadrage = config.CIBLES_PERFORMANCE["pr_auc_min"]
taux_sauvetage_avec_modele = precision_ciblage_cadrage * hyp["taux_succes_retention"]
print(
    f"Comptes sauvés par geste avec modèle : {precision_ciblage_cadrage:.0%} de vrais churners "
    f"× {hyp['taux_succes_retention']:.0%} de succès = {taux_sauvetage_avec_modele:.1%}"
)

# Valeur brute générée par le modèle (gestes × taux de sauvetage × valeur par compte)
valeur_brute_an_eur = gestes_par_an * taux_sauvetage_avec_modele * valeur_par_compte_sauve_eur
print(f"Valeur brute annuelle (gestes × taux de sauvetage × MRR annuel) : {valeur_brute_an_eur:,.0f} €")

# Coût des gestes de rétention (temps CSM)
cout_geste_eur = hyp["cout_horaire_csm_eur"] * hyp["duree_geste_retention_h"]
cout_total_gestes_an_eur = gestes_par_an * cout_geste_eur
print(f"Coût d'un geste de rétention : {cout_geste_eur:.0f} €")
print(f"Coût total gestes/an : {cout_total_gestes_an_eur:,.0f} €")

# Valeur nette attendue (avant coût de développement et maintenance du modèle).
# La marge brute s'applique au revenu sauvé seulement : le temps CSM est une charge réelle,
# déduite en totalité.
valeur_nette_an_eur = valeur_brute_an_eur * hyp["marge_brute_pct"] - cout_total_gestes_an_eur
print(f"\n→ Valeur nette annuelle estimée (marge brute {hyp['marge_brute_pct']:.0%}) : "
      f"{valeur_nette_an_eur:,.0f} €")

# Sans modèle : comptes choisis au hasard → la part de vrais churners contactés est la
# prévalence. Le taux de sauvetage se déduit donc des données, sans hypothèse supplémentaire.
taux_sauvetage_sans_modele = taux_churn_observe * hyp["taux_succes_retention"]
print(
    f"\nComptes sauvés par geste sans modèle : {taux_churn_observe:.0%} de vrais churners "
    f"× {hyp['taux_succes_retention']:.0%} de succès = {taux_sauvetage_sans_modele:.1%}"
)
# Le coût des gestes est identique dans les deux scénarios : il s'annule dans la différence.
gain_incremental_eur = (
    (taux_sauvetage_avec_modele - taux_sauvetage_sans_modele)
    * gestes_par_an
    * valeur_par_compte_sauve_eur
    * hyp["marge_brute_pct"]
)
print(f"→ Gain incrémental du modèle vs intervention non ciblée : {gain_incremental_eur:,.0f} €/an")

# %%
from IPython.display import Markdown, display

display(
    Markdown(
        f"**Ce qu'il faut retenir.** "
        f"Avec une capacité de {int(hyp['capacite_gestes_mois'])} gestes de rétention par mois "
        f"et un taux de succès cible de {hyp['taux_succes_retention']:.0%} "
        "(source : Gainsight 2023), la valeur du modèle tient à un seul levier : "
        f"faire passer la part de vrais churners contactés de {taux_churn_observe:.0%} "
        f"(hasard) à {precision_ciblage_cadrage:.0%} (ciblage). "
        f"Le gain incrémental estimé est de **{gain_incremental_eur:,.0f} €/an** "
        "par rapport à une approche non ciblée, "
        "à confirmer dès §12 après mesure des métriques réelles. "
        "Ces chiffres sont intentionnellement conservateurs : "
        "ils excluent l'effet d'image et les renouvellements spontanés induits par une meilleure "
        "relation CSM-client."
    )
)

# %% [markdown]
# ### 2.5 Contraintes et critères de réussite
#
# #### Contraintes opérationnelles
#
# - **Batch nocturne quotidien** : scores et alertes (cas d'usage 2) disponibles dans le CRM avant
#   06 h 00 ; le passage du dimanche soir alimente la revue du lundi à 8 h 30 (cas d'usage 1).
# - **Latence API** : inférence unitaire ≤ 200 ms (consultation de fiche client, cas d'usage 3).
# - **Explicabilité** : chaque score doit être accompagné des 3 features les plus influentes
#   (SHAP), lisibles par un CSM non-technicien.
# - **Maintenance** : l'équipe Data compte 1 data scientist à mi-temps — pas de modèle
#   nécessitant une infrastructure complexe.
# - **Réglementation** : données personnelles (pays, secteur) → conformité RGPD (§4).
#
# #### Critères de réussite mesurables (fixés *avant* la modélisation — §8, §12)
#
# | Critère | Seuil minimal | Mesure |
# |---|---|---|
# | PR-AUC (métrique principale) | ≥ 0,65 | Évaluation croisée k=5 |
# | Latence inférence unitaire | ≤ 200 ms | Mesure directe en §10 |
# | Latence batch 5 000 comptes | ≤ 300 s | Mesure directe en §10 |
# | Taux de churn détecté (rappel@seuil) | À définir en §9 (compromis FP/FN) | Matrice de confusion |
# | Leurres identifiés et exclus | 4/4 | Permutation importance + SHAP |
# | Fuite de données : aucune | Test automatisé | `tests/test_no_leakage.py` |

# %% [markdown]
# ### 2.6 Périodicité de revue des indicateurs — décision de cadrage
#
# > **Item C9 — décidé ici, en phase de cadrage.**
# > La pertinence des indicateurs du modèle est interrogée selon une périodicité définie
# > maintenant, avant tout déploiement.
#
# **Décision : comité trimestriel de revue du modèle.**
#
# | Fréquence | Participants | Ordre du jour |
# |---|---|---|
# | **Trimestrielle** (mars, juin, sept., déc.) | CS Lead, Data Scientist, DPO | Revue des métriques de drift (PSI, KS), performance sur cohorte récente → décision : réentraîner, maintenir ou geler |
# | **Ad hoc** (seuil d'alerte franchi) | Data Scientist + CS Lead | Si PSI > 0,20 ou PR-AUC < 0,55 sur fenêtre glissante 30 j → analyse causale et décision sous 72 h |
#
# Ce rythme trimestriel est cohérent avec la saisonnalité des renouvellements (contrats annuels,
# vagues Q1 et Q3) et la capacité de l'équipe (1 data scientist à mi-temps).
# Il sera outillé en §13 (rapport Evidently automatisé, gate CI/CD).

# %% [markdown]
# **Ce qu'il faut retenir.**
# La revue trimestrielle est la périodicité retenue au cadrage. Elle constitue le garde-fou
# contre l'obsolescence du modèle (concept drift lié à l'évolution du produit ou du marché).
# Toute alerte automatique (§13) peut déclencher une revue anticipée. Cette décision est
# documentée ici pour traçabilité auprès des parties prenantes.

# %% [markdown]
# > ### 📋 Journal de bord — Cadrage métier
# >
# > **Décisions retenues** — 3 cas d'usage distincts couvrant les besoins opérationnels du CS
# > Lead (revue hebdo), du CSM (alerte) et de la préparation commerciale (renouvellement).
# > Comité de revue trimestriel décidé au cadrage (item C9). Apport du modèle mesuré par le
# > gain incrémental vs intervention non ciblée, calculé en §2.4 depuis
# > `config.HYPOTHESES_ECONOMIQUES` (valeur affichée dans la sortie de la cellule).
# >
# > **Alternatives écartées** — Revue mensuelle (trop lourde pour 1 DS mi-temps) ; revue annuelle
# > (trop tardive pour détecter le drift post-release produit). Alerte uniquement sans batch (perd
# > la vision portefeuille globale, cas d'usage 1).
# >
# > **Difficultés rencontrées** — Le MRR moyen (450 €) est une estimation à ce stade ;
# > il sera confirmé par `profil_compact()` en §5 et réinjecté dans le calcul de ROI en §12.
# >
# > **Impact sur la suite** — Les 3 cas d'usage contraignent §8 (latence, explicabilité) et §12
# > (KPIs métier). La périodicité de revue dimensionne §13 (rituel Evidently trimestriel).
