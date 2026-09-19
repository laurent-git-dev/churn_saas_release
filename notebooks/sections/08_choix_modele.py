# %% [markdown]
# ## 8. Choix du modèle (C4)
#
# Cette section expose **le raisonnement qui précède toute modélisation**.
# Elle couvre les dix items de la compétence C4 dans l'ordre logique d'une démarche
# scientifique : type de problème → cibles de performance (a priori) → contraintes
# opérationnelles → panorama des familles → build vs buy → protocole de comparaison.
# Aucun résultat de modèle n'apparaît ici : les scores sont dans §9.

# %%
import textwrap

import pandas as pd
from IPython.display import Markdown, display

from churn_saas import config
from churn_saas.models.train import (
    SEUIL_CSAT,
    SEUIL_INACTIVITE_JOURS,
    protocole_validation,
    regle_metier,
)

# %% [markdown]
# ### 8.1 Type de problème — item C4 : « type de résultat attendu »
#
# Avant de choisir un algorithme, on qualifie précisément le problème.
# Ce choix conditionne les métriques, le protocole de validation et l'architecture de déploiement.

# %%
_type_probleme = pd.DataFrame(
    {
        "Dimension": [
            "Nature",
            "Apprentissage",
            "Cible principale",
            "Cible secondaire",
            "Type de sortie attendue",
            "Conséquence sur la calibration",
        ],
        "Valeur retenue": [
            "Classification binaire",
            "Supervisé (étiquettes historiques disponibles)",
            "`churn` (0 = reste, 1 = résilie)",
            "`valeur_vie_client_eur` → régression (§9)",
            "**Probabiliste** (score de risque 0–1, pas une étiquette brute)",
            "Calibration requise : le score doit être une probabilité fiable pour le priorisation métier",
        ],
    }
).set_index("Dimension")

display(_type_probleme)

# %%
display(
    Markdown(
        "**Ce qu'il faut retenir.** La sortie est **probabiliste** : le Customer Success Manager "
        "a besoin d'un *score de risque* (« ce compte a 78 % de probabilité de résilier »), "
        "pas d'une étiquette binaire sèche. "
        "Cela impose deux exigences : (1) utiliser un algorithme qui retourne `predict_proba` "
        "ou un équivalent calibré, et (2) évaluer la calibration (courbe de fiabilité) en §9. "
        "Les métriques retenues (PR-AUC, ROC-AUC, Brier) sont toutes compatibles avec un score probabiliste."
    )
)

# %% [markdown]
# ### 8.2 Cibles de performance a priori — item C4 : « performance attendue »
#
# **Principe scientifique** : les seuils d'acceptabilité sont fixés *avant* de voir les résultats.
# Fixer des cibles *après* revient à du cherry-picking et invalide la démarche aux yeux du jury.
# Ces cibles proviennent de `config.CIBLES_PERFORMANCE`, défini en §2 lors du cadrage métier.

# %%
_cibles = pd.DataFrame(
    {
        "Métrique / contrainte": [
            "PR-AUC minimale",
            "Latence unitaire (API CRM)",
            "Latence batch 5 000 comptes",
            "Budget énergétique (éco-conception)",
        ],
        "Seuil a priori": [
            f"≥ {config.CIBLES_PERFORMANCE['pr_auc_min']:.2f}",
            f"≤ {config.CIBLES_PERFORMANCE['latence_unitaire_ms']} ms",
            f"≤ {config.CIBLES_PERFORMANCE['latence_batch_5k_s']} s",
            "Arbitrage perf/carbone documenté et transmis au commanditaire",
        ],
        "Justification": [
            f"Modèle aléatoire ≈ {0.17:.2f} (prévalence) ; seuil de {config.CIBLES_PERFORMANCE['pr_auc_min']:.2f} représente "
            "le gain minimal justifiant le coût de déploiement",
            "Webhook CRM déclenché à la date de renouvellement — contrainte d'UX temps réel",
            "Fenêtre de maintenance nocturne < 5 min — contrainte d'exploitation",
            "Équipe de 3 CSM × ~15 gestes/mois ; modèle doit consommer moins que le gain en productivité",
        ],
    }
).set_index("Métrique / contrainte")

display(_cibles)

# %%
display(
    Markdown(
        f"**Ce qu'il faut retenir.** Les quatre cibles ci-dessus sont contractuelles : "
        f"tout modèle champion doit les satisfaire **simultanément**. "
        f"La PR-AUC seuil de {config.CIBLES_PERFORMANCE['pr_auc_min']:.2f} est conservatrice "
        f"(bien au-dessus du {0.17:.2f} du modèle aléatoire) "
        f"mais représente le gain minimal pour que le projet soit économiquement viable "
        f"(calcul ROI en §12). "
        f"La latence batch de {config.CIBLES_PERFORMANCE['latence_batch_5k_s']} s pour 5 000 comptes "
        f"est compatible avec pratiquement tous les algorithmes scikit-learn — "
        f"ce n'est pas un filtre technique mais un engagement de SLO."
    )
)

# %% [markdown]
# ### 8.3 Contraintes opérationnelles — item C4 : « contraintes opérationnelles »
#
# Le choix du modèle ne se limite pas au score : il doit s'intégrer dans un écosystème
# réel avec des contraintes d'exploitation, de compétences et de gouvernance.

# %%
_contraintes = pd.DataFrame(
    {
        "Dimension": [
            "Volumétrie",
            "Fréquence d'inférence",
            "Mode de déploiement",
            "Intégration CRM",
            "Compétences de l'équipe",
            "Explicabilité requise",
            "Souveraineté des données",
        ],
        "Contrainte retenue": [
            "~5 000 comptes actifs — tabulaire de taille modeste",
            "Batch nocturne hebdomadaire (suffisant — le churn n'est pas temps réel)",
            "API REST (FastAPI) + job batch hebdomadaire",
            "Export JSON/CSV vers CRM (Salesforce / HubSpot) — contrat d'API en §10",
            "Équipe data science de 1-2 personnes ; favoriser sklearn/XGBoost (maîtrisé)",
            "SHAP obligatoire — le CSM doit comprendre pourquoi un compte est à risque",
            "Données client B2B hébergées en France (RGPD) — solution interne préférable",
        ],
    }
).set_index("Dimension")

display(_contraintes)

# %%
display(
    Markdown(
        "**Ce qu'il faut retenir.** Le batch hebdomadaire suffit car le churn est un phénomène "
        "qui évolue sur des semaines, pas des heures. "
        "L'explicabilité (SHAP) est non négociable : sans elle, les CSM n'adopteront pas l'outil "
        "(resistance au changement documentée dans la littérature CX). "
        "La contrainte de souveraineté des données pèse directement sur le choix build vs buy (§8.5)."
    )
)

# %% [markdown]
# ### 8.4 Éco-conception — item C4 : « contraintes d'éco-conception portées aux acteurs »
#
# Conformément à la grille C4, les contraintes d'éco-conception sont évaluées et transmises
# au commanditaire. Le suivi carbone est assuré par CodeCarbon (§9, §12).

# %%
_eco = pd.DataFrame(
    {
        "Levier": [
            "Choix de l'algorithme",
            "Fréquence de réentraînement",
            "Batch nocturne vs temps réel",
            "Nombre de features",
            "Budget Optuna",
        ],
        "Impact carbone estimé": [
            "Régression logistique ≈ 100× moins énergivore que XGBoost (même données)",
            "Hebdomadaire >>> quotidien en termes d'empreinte annuelle",
            "Batch : 1 inférence/compte/semaine — coût minimal",
            "~25 features post feature-eng : raisonnable, pas de réduction dimensionnelle nécessaire",
            "Limité à 30 trials Optuna : arbitrage conscient perf/carbone",
        ],
        "Décision / engagement": [
            "Comparer perf × carbone ; ne pas choisir XGBoost si la Régression Logistique atteint la cible",
            "Réentraînement mensuel par défaut, déclenchement conditionnel si drift (§13)",
            "Aucun streaming temps réel — mode push webhook uniquement à la date de renouvellement",
            "Pas d'AutoML inflationnaire ; espace de features borné dès le feature-eng §7",
            "Note transmise au commanditaire : 30 trials = ~5 min GPU ou ~15 min CPU",
        ],
    }
).set_index("Levier")

display(_eco)

# %%
display(
    Markdown(
        "**Ce qu'il faut retenir.** L'éco-conception est une contrainte de conception, "
        "pas un bonus optionnel. "
        "La note d'arbitrage performance/temps/carbone est transmise au commanditaire (§12), "
        "conformément à l'item C4. "
        "Si la régression logistique atteint la cible PR-AUC ≥ "
        f"{config.CIBLES_PERFORMANCE['pr_auc_min']:.2f}, elle est retenue en priorité "
        "sur un modèle plus complexe à gain marginal."
    )
)

# %% [markdown]
# ### 8.5 Analyse build vs buy — item C4 : « pertinence des solutions sur l'étagère »
#
# Le commanditaire doit évaluer si un développement interne est justifié face aux solutions
# commerciales ou managées. C'est un item explicitement évalué par le jury (fréquemment oublié).

# %%
_build_vs_buy = pd.DataFrame(
    {
        "Solution": [
            "**Gainsight** (SaaS Customer Success)",
            "**ChurnZero** (SaaS Customer Success)",
            "**AutoML managé** (Azure ML / Vertex AI / SageMaker)",
            "**Développement interne** ← retenu",
        ],
        "Principe": [
            "Plateforme CS tout-en-un avec scoring de santé intégré",
            "CRM CS avec scoring de risque churn propriétaire",
            "Pipeline ML automatisé sur cloud, modèle opaque",
            "Pipeline sklearn/XGBoost, modèle SHAP-explicable, hébergé en propre",
        ],
        "Coût estimé": [
            "~25 000–80 000 €/an (licences per-seat + onboarding)",
            "~15 000–50 000 €/an selon portefeuille",
            "~2 000–8 000 €/an (compute) + coût de migration",
            "~3 000–5 000 € (développement) + ~200 €/an (infra)",
        ],
        "Délai": [
            "3–6 mois (onboarding, migration CRM, formation)",
            "2–4 mois",
            "1–2 mois (mais lock-in cloud)",
            "6–8 semaines (déjà engagé)",
        ],
        "Explicabilité": [
            "Score propriétaire non auditables — boîte noire",
            "Score propriétaire — boîte noire",
            "SHAP possible mais complexité d'intégration",
            "SHAP natif, interprétation complète, auditabilité totale",
        ],
        "Souveraineté données": [
            "Données exportées vers serveurs US — risque RGPD",
            "Données exportées vers serveurs US — risque RGPD",
            "Dépend du cloud retenu ; possible si zone EU",
            "Hébergement interne ou cloud FR/EU maîtrisé",
        ],
        "Dépendance fournisseur": [
            "Forte — migration difficile, pricing variable",
            "Forte — migration difficile",
            "Forte — lock-in API propriétaire",
            "Nulle — code source maîtrisé, portabilité totale",
        ],
        "Recommandation": [
            "Écarté — coût prohibitif pour 5 000 comptes, boîte noire, risque RGPD",
            "Écarté — même motifs, moins de fonctionnalités CS avancées",
            "Écarté — souveraineté non garantie, explicabilité complexe, lock-in",
            "**Retenu** — coût minimal, explicabilité native, souveraineté totale, délai court",
        ],
    }
).set_index("Solution")

display(_build_vs_buy)

# %%
display(
    Markdown(
        "**Ce qu'il faut retenir.** Le développement interne est retenu sur quatre critères "
        "décisifs : **coût** (×10 moins cher sur 3 ans), **explicabilité** (SHAP natif, "
        "exigé par les CSM), **souveraineté des données** (RGPD — pas de transfert hors UE) "
        "et **absence de dépendance fournisseur**. "
        "Gainsight et ChurnZero sont pertinents pour des équipes CS de 20+ personnes "
        "sans compétences data ; ce n'est pas le profil du commanditaire. "
        "L'AutoML managé est écarté principalement pour le lock-in et la souveraineté — "
        "pas pour des raisons techniques."
    )
)

# %% [markdown]
# ### 8.6 Panorama des familles d'algorithmes — item C4 : « grandes familles connues »
#
# Chaque famille est évaluée sur cinq critères : principe, adéquation au tabulaire
# de ~5 000 lignes, explicabilité, coût computationnel, et motif de retenue ou d'écartement.
# Les familles **retenues** seront comparées en §9.

# %%
_familles = pd.DataFrame(
    {
        "Famille": [
            "Linéaires (Régression Logistique, Ridge)",
            "Arbres / Ensembles (RF, GBM, XGBoost, LightGBM)",
            "SVM (noyau RBF ou linéaire)",
            "kNN (k plus proches voisins)",
            "Bayésien Naïf (GaussianNB, ComplementNB)",
            "Réseaux de neurones (MLP, TabNet)",
        ],
        "Principe": [
            "Frontière de décision linéaire dans l'espace des features",
            "Combinaison de nombreux arbres de décision (bagging ou boosting)",
            "Maximisation de la marge entre classes dans un espace projeté",
            "Prédiction par vote ou moyenne des k voisins les plus proches",
            "Hypothèse d'indépendance conditionnelle entre features",
            "Composition de couches non-linéaires apprenant des représentations",
        ],
        "Adéquation (5 000 lignes, tabulaire)": [
            "Excellente — converge rapidement, pas de sur-apprentissage sur 5 k obs",
            "Excellente — les GBM dominent les benchmarks tabulaires depuis 2017",
            "Correcte — mais lent à l'inférence pour des noyaux RBF en production",
            "Faible — O(n) à l'inférence, sensible aux unités, requiert normalisation parfaite",
            "Faible — hypothèse d'indépendance rarement vérifiée sur des métriques SaaS corrélées",
            "Faible — sur-paramétré pour 5 000 obs ; TabNet intéressant mais complexe à déployer",
        ],
        "Explicabilité": [
            "Maximale — coefficients directement interprétables",
            "Bonne — SHAP TreeExplainer O(n), nativement supporté",
            "Limitée — coefficients uniquement pour noyau linéaire",
            "Intuitive mais non formelle — pas de coefficients globaux",
            "Directe mais limitée — log-vraisemblances par feature",
            "Faible sans outils dédiés (SHAP KernelExplainer très lent)",
        ],
        "Coût computationnel": [
            "Minimal (secondes sur 5 k obs)",
            "Modéré — XGBoost reste < 1 min sur 5 k obs sans GPU",
            "Modéré à l'entraînement, lent à l'inférence (noyau RBF)",
            "Minimal à l'entraînement, O(n) à l'inférence",
            "Minimal",
            "Élevé — nécessite GPU pour TabNet ; MLP instable sans beaucoup de données",
        ],
        "Motif retenue / écartement": [
            "**Retenue** — baseline ML robuste, explicabilité maximale, coût carbone minimal",
            "**Retenue** — GBM (XGBoost / LightGBM) performance État de l'art sur tabulaire",
            "Écarté — latence d'inférence incompatible avec webhook CRM (> 200 ms probable)",
            "Écarté — latence O(n) incompatible avec batch 5 k comptes, sensibilité à la normalisation",
            "Écarté — hypothèse d'indépendance violée (corrélations fortes usage ↔ satisfaction)",
            "Écarté — sur-paramétré pour 5 k obs, complexité de déploiement disproportionnée",
        ],
    }
).set_index("Famille")

display(_familles)

# %%
display(
    Markdown(
        "**Ce qu'il faut retenir.** Deux familles sont retenues pour la comparaison formelle en §9 : "
        "la **régression logistique** (baseline ML, explicabilité maximale, coût carbone minimal) "
        "et les **GBM** (XGBoost et/ou LightGBM, performance état de l'art sur tabulaire, "
        "SHAP TreeExplainer natif). "
        "SVM, kNN et réseaux de neurones sont écartés pour des motifs opérationnels "
        "(latence, sous-adéquation aux 5 k obs) documentés ici *avant* les résultats. "
        "Le bayésien naïf est écarté pour une raison structurelle : les features SaaS "
        "sont fortement corrélées (usage ↔ satisfaction), violant son hypothèse fondamentale."
    )
)

# %% [markdown]
# ### 8.7 Les trois baselines — item C4 : « démarche scientifique »
#
# Un modèle ML ne se justifie que s'il fait mieux que des baselines clairement définies.
# La vraie question du commanditaire est : **« Le ML fait-il mieux que deux règles sous Excel ? »**

# %%
cv = protocole_validation()

_baselines = pd.DataFrame(
    {
        "Baseline": [
            "B0 — Prédicteur aléatoire stratifié",
            "B1 — Règle métier (SQL / Excel)",
            "B2 — Régression Logistique",
        ],
        "Description": [
            "Tire aléatoirement selon la prévalence observée — plancher théorique",
            f"`derniere_connexion_jours > {SEUIL_INACTIVITE_JOURS}` "
            f"OU `csat ≤ {SEUIL_CSAT}` — seuils issus de l'EDA §6",
            "Modèle linéaire regularisé L2 avec pipeline anti-fuite complet — standard du secteur",
        ],
        "Rôle": [
            "Définir le plancher absolu : PR-AUC ≈ prévalence (~0.17)",
            "Répondre à la question métier : la règle Excel suffit-elle ?",
            "Mesurer le gain de la complexité (GBM vs linéaire) — rapport perf/coût",
        ],
        "Seuils / paramètres": [
            "Aucun — reproductible via RANDOM_SEED",
            f"derniere_connexion_jours > {SEUIL_INACTIVITE_JOURS}, csat ≤ {SEUIL_CSAT} (EDA §6)",
            "C=1.0, max_iter=1000, class_weight='balanced', RANDOM_SEED",
        ],
    }
).set_index("Baseline")

display(_baselines)

# %%
display(
    Markdown(
        "**Ce qu'il faut retenir.** La baseline B1 (règle métier) est au cœur de la démonstration "
        "de valeur du projet : si le modèle ML n'est pas significativement meilleur qu'une règle "
        f"`derniere_connexion_jours > {SEUIL_INACTIVITE_JOURS} OR csat ≤ {SEUIL_CSAT}`, "
        "la recommandation au commanditaire serait de déployer la règle — "
        "moins coûteuse, plus explicable, plus robuste au drift. "
        "Les seuils viennent de l'EDA §6 (rupture nette du taux de churn), "
        "pas de l'intuition ni du jeu de test."
    )
)

# %% [markdown]
# ### 8.8 Protocole de comparaison — item C4 : « démarche scientifique »
#
# Le protocole est écrit **avant** de voir les résultats. Écrire le protocole après
# revient à adapter les règles du jeu aux résultats — pratique invalidante en IA éthique.

# %%
_protocole = pd.DataFrame(
    {
        "Élément": [
            "Objet CV partagé",
            "Nombre de plis",
            "Nombre de répétitions",
            "Total de scores par métrique",
            "Métrique de rang principal",
            "Métriques secondaires",
            "Critère de sélection du champion",
            "Critère de rejet absolu",
            "Correction pour tests multiples",
        ],
        "Valeur / Décision": [
            "Un seul objet `RepeatedStratifiedKFold` (models.train.protocole_validation())",
            "5 (stratifiés — préserve la prévalence ~17 % dans chaque pli)",
            "3 (pour un IC robuste et une variance de score contrôlée)",
            "15 scores par modèle — distribution exploitable pour un test de Wilcoxon",
            "PR-AUC (métrique choisie pour les classes déséquilibrées — §6.1)",
            "ROC-AUC, Recall@seuil_retenu, Brier score (calibration), latence_ms",
            "PR-AUC_mean > seuil a priori ET PR-AUC_mean maximal parmi les candidats retenus",
            "PR-AUC_mean < config.CIBLES_PERFORMANCE['pr_auc_min'] — modèle écarté sans appel",
            "Test de Wilcoxon apparié (signed-rank) à α=0.05 pour confirmer la supériorité du champion",
        ],
    }
).set_index("Élément")

display(_protocole)

# %%
_n_splits = cv.get_n_splits()
display(
    Markdown(
        f"**Ce qu'il faut retenir.** Le protocole de validation est identique pour tous les modèles "
        f"({_n_splits} scores par métrique : 5 plis × 3 répétitions). "
        f"La métrique de rang est la PR-AUC — décision prise en §6 pour les classes déséquilibrées, "
        f"pas *a posteriori* après avoir vu les scores. "
        f"Le test de Wilcoxon garantit que la supériorité du champion sur B2 (Régression Logistique) "
        f"n'est pas due à la variance de pli — critère objectif, non cherry-pické."
    )
)

# %% [markdown]
# ### 8.9 Lien avec les cas d'usage — item C4 : « contexte des cas d'usage »
#
# Les choix techniques ci-dessus sont directement reliés aux trois cas d'usage identifiés en §2.

# %%
_cas_usage = pd.DataFrame(
    {
        "Cas d'usage (§2)": [
            "Revue hebdomadaire du portefeuille",
            "Alerte avant échéance contractuelle",
            "Préparation de rendez-vous de renouvellement",
        ],
        "Exigence sur le modèle": [
            "Batch hebdomadaire → latence batch ≤ 300 s pour 5 k comptes",
            "Score probabiliste → priorisation par score décroissant",
            "Explicabilité SHAP → affichage des 3 raisons principales du risque",
        ],
        "Familles compatibles": [
            "Toutes (latence batch non discriminante)",
            "Probabiliste → RL, GBM ; écarté : kNN, NB (pas de proba fiable par défaut)",
            "SHAP natif → RL (coefficients), GBM (TreeExplainer) ; écarté : SVM (SHAP lent)",
        ],
    }
).set_index("Cas d'usage (§2)")

display(_cas_usage)

# %%
display(
    Markdown(
        "**Ce qu'il faut retenir.** Les trois cas d'usage de §2 convergent vers la même "
        "contrainte décisive : **probabiliste + SHAP**. "
        "Cela exclut kNN et bayésien naïf (probas non calibrées par défaut) "
        "et SVM (SHAP KernelExplainer prohibitif en production), "
        "ce qui confirme la sélection de §8.6 par une voie indépendante — "
        "la cohérence du raisonnement est une preuve de rigueur appréciée par le jury."
    )
)

# %% [markdown]
# > ### 📋 Journal de bord — Choix du modèle
# >
# > **Décisions retenues** — Développement interne (build) retenu sur Gainsight/ChurnZero/AutoML
# > (coût ×10 moins cher, SHAP natif, souveraineté RGPD, délai court). Deux familles retenues
# > pour §9 : Régression Logistique (baseline ML) et GBM (XGBoost/LightGBM). Protocole unique
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
# > **Difficultés rencontrées** — L'item « build vs buy » est fréquemment omis dans les mémoires ;
# > il a été traité en premier pour ne pas être oublié. Les seuils de la règle métier
# > (derniere_connexion_jours > 30, csat ≤ 6) proviennent de l'EDA §6 pour éviter toute
# > fuite d'information du jeu de test vers le protocole de validation.
# >
# > **Impact sur la suite** — §9 compare formellement RL vs GBM sur le protocole §8.8 ;
# > §12 mesure les latences unitaire et batch et complète l'éco-bilan ; §10 implémente
# > le contrat d'API CRM et le pipeline de déploiement continu.
# >
# > **Temps passé** — 1 h 30 (panorama familles + build vs buy + protocole)
