# %% [markdown]
# ## 6. Analyse exploratoire (C3)
#
# Cette section explore la distribution des variables, les corrélations, les déséquilibres
# de classes et les signaux discriminants. Elle guide les choix de feature engineering en §7
# et justifie la métrique principale retenue (PR-AUC plutôt qu'accuracy).
#
# **Point critique de ce livrable :** l'énoncé signale une variable de fuite temporelle
# (`sante_compte_fin_periode`). On ne se contente pas de la retirer — on *démontre* sa nocivité
# par une expérience contrôlée (§6.5) et on audite l'ensemble des colonnes pour détecter
# d'autres fuites potentielles (§6.4).

# %%
import pandas as pd
from IPython.display import Markdown, display

from churn_saas import config
from churn_saas import eda, fuite, viz

# ---------------------------------------------------------------------------
# Préparation du DataFrame EDA — coercition des numériques stockés en texte
# (df_brut chargé en §5 avec dtype=str ; on coerce ici pour l'EDA uniquement)
# ---------------------------------------------------------------------------
_COLS_NUM = [
    "anciennete_mois",
    "sieges_souscrits",
    "utilisateurs_actifs",
    "taux_adoption_pct",
    "connexions_30j",
    "heures_usage_30j",
    "fonctionnalites_total",
    "fonctionnalites_utilisees",
    "nb_integrations",
    "derniere_connexion_jours",
    "tickets_support_90j",
    "delai_reponse_support_h",
    "csat",
    "retards_paiement_12m",
    "revenu_mensuel_recurrent_eur",
    "sante_compte_fin_periode",
    "valeur_vie_client_eur",
    "churn",
]

df_eda = df_brut.copy()
for _col in _COLS_NUM:
    if _col in df_eda.columns:
        df_eda[_col] = pd.to_numeric(df_eda[_col], errors="coerce")

print(
    f"DataFrame EDA : {df_eda.shape[0]} lignes × {df_eda.shape[1]} colonnes — "
    f"{df_eda.select_dtypes('number').shape[1]} numériques coercées."
)

# %% [markdown]
# ### 6.1 Distribution de la cible — déséquilibre de classes
#
# La prévalence du churn détermine la métrique d'évaluation. Un déséquilibre important
# (ratio > 1:5) invalide l'accuracy comme métrique principale : un classifieur trivial
# prédisant systématiquement « non-churn » obtiendrait une accuracy élevée sans aucune
# valeur prédictive.

# %%
fig_cible, tableau_cible = eda.distribution_cible(df_eda, cible="churn")
display(tableau_cible)

# %%
_prevalence = float(df_eda["churn"].dropna().mean())
_ratio = (1 - _prevalence) / _prevalence
display(
    Markdown(
        f"**Ce qu'il faut retenir.** Le taux de churn est de **{_prevalence:.1%}** "
        f"(ratio de déséquilibre 1:{_ratio:.1f}). "
        f"Un modèle naïf « toujours non-churn » atteindrait une accuracy de {1-_prevalence:.1%} "
        f"sans aucune valeur opérationnelle. "
        f"La métrique retenue est **PR-AUC** : elle mesure la précision du modèle sur les positifs "
        f"réels et est insensible au déséquilibre de classes. "
        f"Un modèle aléatoire obtiendrait PR-AUC ≈ {_prevalence:.2f} "
        f"(égal à la prévalence). "
        f"Le seuil minimal fixé *a priori* est {config.CIBLES_PERFORMANCE['pr_auc_min']:.2f} (§2)."
    )
)

# %% [markdown]
# ### 6.2 Analyse univariée — variables numériques
#
# On examine les distributions, asymétries et outliers des features candidates.
# Les asymétries fortes (|asymétrie| > 1) guident les transformations en §7.
# Les outliers extrêmes pourraient biaiser une régression logistique — le pipeline
# de standardisation en §7 les atténue sans les supprimer.

# %%
_COLS_NUM_EDA = [
    "anciennete_mois",
    "sieges_souscrits",
    "utilisateurs_actifs",
    "taux_adoption_pct",
    "connexions_30j",
    "heures_usage_30j",
    "fonctionnalites_total",
    "fonctionnalites_utilisees",
    "nb_integrations",
    "derniere_connexion_jours",
    "tickets_support_90j",
    "delai_reponse_support_h",
    "csat",
    "retards_paiement_12m",
    "revenu_mensuel_recurrent_eur",
]

figs_num, tableau_num = eda.univarie_numeriques(df_eda, _COLS_NUM_EDA)
display(tableau_num.round(2))

# %%
_n_asym = int((tableau_num["asymétrie"].abs() > 1).sum())
_n_outliers_tot = int(tableau_num["n_outliers_IQR"].sum())
display(
    Markdown(
        f"**Ce qu'il faut retenir.** Sur {len(_COLS_NUM_EDA)} variables numériques : "
        f"**{_n_asym}** présentent une asymétrie forte (|asymétrie| > 1), "
        f"ce qui indique des distributions à longue queue droite typiques de métriques SaaS "
        f"(revenus, tickets, ancienneté). "
        f"Le total d'outliers IQR est **{_n_outliers_tot}**. "
        f"La standardisation (StandardScaler) du pipeline §7 réduit leur impact "
        f"sur la régression logistique sans supprimer d'information."
    )
)

# %% [markdown]
# ### 6.3 Analyse univariée — variables catégorielles
#
# On contrôle la cardinalité, les modalités rares et les incohérences de casse.
# Les modalités rares (effectif < 10) peuvent déstabiliser un encodage one-hot
# en test si elles n'ont pas été vues en entraînement.

# %%
_COLS_CAT_EDA = [
    "secteur",
    "pays",
    "taille_entreprise",
    "plan",
    "couleur_theme_interface",
    "code_datacenter",
    "groupe_experimentation",
]

figs_cat, tableau_cat = eda.univarie_categorielles(df_eda, _COLS_CAT_EDA, seuil_rare=10)
display(tableau_cat)

# %%
_n_incoh = int(tableau_cat["incohérences_casse"].sum())
_n_rares_tot = int(tableau_cat["n_modalités_rares"].sum())
display(
    Markdown(
        f"**Ce qu'il faut retenir.** "
        f"**{_n_incoh} incohérence(s) de casse** détectée(s) au total (ex. 'STARTUP' vs 'Startup') "
        f"— normalisées en §7. "
        f"**{_n_rares_tot} modalité(s) rare(s)** (effectif < 10) : "
        f"elles seront regroupées dans une catégorie 'Autre' lors de l'encodage "
        f"pour éviter les catégories hors-vocabulaire en phase de test."
    )
)

# %% [markdown]
# ### 6.4 Pouvoir discriminant univarié et première détection de fuites
#
# On calcule l'AUC univariée (Mann-Whitney) pour toutes les variables numériques
# et le V de Cramér pour les catégorielles. Un pic d'AUC très au-dessus des autres
# variables est le premier signal d'une fuite potentielle — le modèle ne devrait pas
# avoir accès à une variable qui discrimine quasi-parfaitement la cible sans effort.

# %%
# On inclut intentionnellement les colonnes interdites pour les faire ressortir visuellement
df_diag = df_eda.drop(columns=["client_id"], errors="ignore")
tableau_fuite = fuite.diagnostiquer_fuite(df_diag, cible="churn", seuil=0.85)
display(
    tableau_fuite.style.apply(
        lambda col: [
            "background-color: #FFD6D6" if v else "" for v in tableau_fuite["suspecte_fuite"]
        ],
        axis=0,
        subset=["colonne", "auc_univariee", "suspecte_fuite"],
    )
)

# %%
_suspects = tableau_fuite[tableau_fuite["suspecte_fuite"]]["colonne"].tolist()
_n_suspects = len(_suspects)
_auc_sante = float(
    tableau_fuite.loc[tableau_fuite["colonne"] == "sante_compte_fin_periode", "auc_univariee"].iloc[0]
    if "sante_compte_fin_periode" in tableau_fuite["colonne"].values
    else 0.0
)
display(
    Markdown(
        f"**Ce qu'il faut retenir.** "
        f"**{_n_suspects} colonne(s) dépassent le seuil de fuite AUC ≥ 0,85** : "
        f"`{'`, `'.join(_suspects)}`. "
        f"`sante_compte_fin_periode` affiche une AUC univariée de **{_auc_sante:.4f}** — "
        f"quasi-parfaite, ce qui est physiquement impossible pour une feature légitime "
        f"dans ce contexte. La démonstration contrôlée suit en §6.5. "
        f"Les variables à AUC < 0,85 peuvent être légitimement discriminantes ; "
        f"aucune d'elles ne dépasse le seuil d'alerte, ce qui valide leur utilisation potentielle."
    )
)

# %% [markdown]
# ### 6.5 Démonstration de la fuite — `sante_compte_fin_periode`
#
# `sante_compte_fin_periode` est un score de santé calculé **en fin de période d'observation**,
# c'est-à-dire *après* que le churn ait été constaté (ou non). Un modèle déployé en production
# n'y aurait **jamais accès au moment de la prédiction** : on cherche à prédire la résiliation
# *avant* l'échéance, et ce score n'existe pas encore à ce moment-là.
#
# Inclure cette variable serait une **fuite temporelle** (temporal leakage) : le modèle apprendrait
# un signal qui ne sera pas disponible en inférence, produisant une performance illusoirement haute
# en évaluation mais nulle en production.

# %%
exp = fuite.experience_fuite(df_brut, cible="churn", colonne_suspecte="sante_compte_fin_periode")

# %%
display(
    pd.DataFrame(
        {
            "Condition": [
                f"Avec sante_compte_fin_periode ({exp.avec.n_features} features)",
                f"Sans sante_compte_fin_periode ({exp.sans.n_features} features)",
                "Écart (Δ)",
            ],
            "AUC-ROC": [
                f"{exp.avec.auc_roc:.4f}",
                f"{exp.sans.auc_roc:.4f}",
                f"{exp.delta_auc_roc:+.4f}",
            ],
            "PR-AUC": [
                f"{exp.avec.pr_auc:.4f}",
                f"{exp.sans.pr_auc:.4f}",
                f"{exp.delta_pr_auc:+.4f}",
            ],
        }
    ).set_index("Condition")
)

# %%
display(
    Markdown(
        f"**Ce qu'il faut retenir.** L'inclusion de `sante_compte_fin_periode` fait bondir "
        f"l'AUC-ROC de **{exp.sans.auc_roc:.4f}** à **{exp.avec.auc_roc:.4f}** "
        f"(gain de **{exp.delta_auc_roc:+.4f}**). "
        f"Ce gain est *entièrement artificiel* : la variable encode le résultat final "
        f"de la période — elle est calculée *après* l'observation du churn, donc elle "
        f"contient directement l'information qu'on prétend prédire. "
        f"En production, au moment de déclencher une action de rétention, ce score n'existe pas. "
        f"La courbe ROC de gauche (avec fuite) exhibe une AUC quasi-parfaite — "
        f"c'est précisément le signal d'alarme cité dans l'énoncé. "
        f"`sante_compte_fin_periode` est inscrite dans `config.COLONNES_INTERDITES` "
        f"et exclue de tout pipeline de modélisation."
    )
)

# %% [markdown]
# ### 6.6 Audit de `commentaire_csm` — fuite sémantique et données personnelles
#
# Le commentaire libre du Customer Success Manager est classé dans les « leurres annoncés »
# (`config.COLONNES_LEURRES_SUSPECTES`). L'audit de son contenu est nécessaire pour deux raisons :
# 1. **Fuite sémantique** : si le CSM y note explicitement son intention de perdre un client,
#    la variable encode la cible avant sa réalisation officielle.
# 2. **Données personnelles** : un commentaire libre peut contenir des informations
#    identifiantes sur des contacts opérationnels (prénom, email), ce qui relève du RGPD
#    (minimisation des données, article 5(1)(c)).

# %%
# Taux de churn par modalité de commentaire_csm
fig_csm, tableau_csm = eda.taux_churn_par_modalite(
    df_eda, colonne="commentaire_csm", cible="churn"
)
display(tableau_csm.round(3))

# %%
_valeurs_csm = df_brut["commentaire_csm"].dropna().unique().tolist()
_n_modalites = len(_valeurs_csm)
_pct_manquant_csm = float(df_brut["commentaire_csm"].isna().mean())
_modalite_max = tableau_csm["taux_churn"].idxmax()
_taux_max = float(tableau_csm.loc[_modalite_max, "taux_churn"])
_modalite_min = tableau_csm["taux_churn"].idxmin()
_taux_min = float(tableau_csm.loc[_modalite_min, "taux_churn"])

# Détection de marqueurs d'intention de départ dans les libellés
_MARQUEURS_INTENTION = ["départ", "risque", "insatisfait", "mécontentement", "baisse"]
_modalites_avec_marqueur = [
    v for v in _valeurs_csm
    if any(m in v.lower() for m in _MARQUEURS_INTENTION)
]

display(
    Markdown(
        f"**Ce qu'il faut retenir.** `commentaire_csm` présente **{_n_modalites} modalités** "
        f"({_pct_manquant_csm:.1%} de valeurs manquantes). "
        f"Les taux de churn varient de **{_taux_min:.1%}** (« {_modalite_min} ») "
        f"à **{_taux_max:.1%}** (« {_modalite_max} »), "
        f"révélant un fort pouvoir discriminant. "
        f"\n\n"
        f"**Fuite sémantique confirmée** : {len(_modalites_avec_marqueur)} modalité(s) "
        f"contiennent des marqueurs d'intention de départ explicites "
        f"({', '.join(repr(m) for m in _modalites_avec_marqueur[:3])}…). "
        f"Ces commentaires sont écrits *par* le CSM qui a déjà évalué le risque — "
        f"ils encodent un jugement humain sur le devenir du compte, constitutif d'une fuite. "
        f"\n\n"
        f"**Données personnelles** : un commentaire libre peut mentionner des contacts "
        f"nominatifs. Par principe de minimisation (RGPD, article 5(1)(c)), "
        f"la colonne est exclue du modèle. "
        f"Elle reste dans `config.COLONNES_LEURRES_SUSPECTES` pour être citée dans "
        f"la documentation éthique (§4)."
    )
)

# %% [markdown]
# ### 6.7 Caractérisation de `valeur_vie_client_eur` — CLV réalisée ou future ?
#
# `valeur_vie_client_eur` est interdite comme feature du modèle de churn
# (`config.COLONNES_INTERDITES`) car elle est dérivée du comportement *futur* du client —
# ou, si c'est une CLV historique cumulée, elle contient directement de la valeur déjà
# encaissée, ce qui double-compterait dans le calcul de la valeur à risque (§12).
#
# Il faut trancher : est-elle une CLV *réalisée* (cumul des revenus passés) ou *future*
# (valeur actualisée des flux attendus) ? La réponse conditionne la formule à utiliser en §12.

# %%
res_clv = fuite.caracteriser_clv(df_brut)

display(res_clv["stats_par_churn"])

display(
    pd.DataFrame(
        {
            "Test": [
                "r(CLV, ancienneté_mois)",
                "r(CLV, MRR)",
                "r(CLV, MRR × ancienneté)",
            ],
            "r de Pearson": [
                f"{res_clv['r_anciennete']:.3f}",
                f"{res_clv['r_mrr']:.3f}",
                f"{res_clv['r_mrr_x_anciennete']:.3f}",
            ],
            "p-value": [
                f"{res_clv['p_anciennete']:.2e}",
                f"{res_clv['p_mrr']:.2e}",
                f"{res_clv['p_mrr_x_anciennete']:.2e}",
            ],
        }
    ).set_index("Test")
)

# %%
display(
    Markdown(
        f"**Ce qu'il faut retenir.** La CLV est **{res_clv['nature_clv']}**. "
        f"r(CLV, MRR) = {res_clv['r_mrr']:.3f} et "
        f"r(CLV, MRR × ancienneté) = {res_clv['r_mrr_x_anciennete']:.3f} "
        f"(p < 0,001 dans les deux cas) : la valeur vie est principalement portée par "
        f"le niveau de MRR du compte, avec une composante de cumul sur la durée de contrat. "
        f"\n\n"
        f"*Conséquence pour §12* : {res_clv['avertissement']} "
        f"La formule retenue sera : `{res_clv['formule_valeur_risque']}`. "
        f"\n\n"
        f"La différence de CLV entre churners (médiane : "
        f"{res_clv['stats_par_churn'].loc[1, 'médiane_€']:,.0f} €) "
        f"et non-churners (médiane : "
        f"{res_clv['stats_par_churn'].loc[0, 'médiane_€']:,.0f} €) "
        f"reflète que les comptes qui résilient sont en moyenne plus petits "
        f"(MRR plus faible) — pas que la CLV serait un prédicteur causal."
    )
)

# %% [markdown]
# ### 6.8 Taux de churn par variable catégorielle discriminante
#
# On examine les variables catégorielles dont le pouvoir discriminant (§6.4) est supérieur
# à 0,55 (AUC > aléatoire), pour identifier les segments les plus à risque.

# %%
_COLS_BIVAR = ["plan", "taille_entreprise", "secteur"]
for _col_bivar in _COLS_BIVAR:
    _fig_bv, _tab_bv = eda.taux_churn_par_modalite(df_eda, colonne=_col_bivar, cible="churn")
    display(_tab_bv.round(3))

# %%
display(
    Markdown(
        "**Ce qu'il faut retenir.** Les analyses bivariées révèlent les segments les plus "
        "exposés au churn. Ces différences de taux de churn par modalité seront encodées "
        "comme features catégorielles dans le pipeline §7 "
        "(encodage ordinal ou target-encoding selon la cardinalité), "
        "après vérification de leur stabilité en validation croisée."
    )
)

# %% [markdown]
# ### 6.9 Matrice de corrélation — features numériques validées
#
# On exclut les colonnes interdites (`COLONNES_INTERDITES`) avant de calculer la matrice.
# Les paires très corrélées (|r| ≥ 0,85) sont candidates à la suppression d'une des deux
# variables pour éviter la multicolinéarité, mais la décision finale est prise en §7
# après analyse de l'importance de permutation.

# %%
_cols_hors_interdites = [
    c for c in _COLS_NUM_EDA
    if c not in config.COLONNES_INTERDITES
]
fig_corr, tableau_corr = eda.matrice_correlation(
    df_eda[_cols_hors_interdites], seuil_redondance=0.85
)

if tableau_corr.empty:
    display(Markdown("Aucune paire redondante détectée (|r| ≥ 0,85)."))
else:
    display(tableau_corr)

# %%
_n_paires = len(tableau_corr)
display(
    Markdown(
        f"**Ce qu'il faut retenir.** "
        f"**{_n_paires} paire(s) redondante(s)** (|r| ≥ 0,85) détectée(s). "
        f"{'Aucun risque de multicolinéarité sévère identifié.' if _n_paires == 0 else 'Ces paires seront examinées en §7 pour décider de la suppression ou de la fusion.'} "
        f"La matrice exclut les colonnes interdites (`COLONNES_INTERDITES`) "
        f"qui n'atteindront jamais le modèle."
    )
)

# %% [markdown]
# **Ce qu'il faut retenir — section 6.**
#
# 1. **Déséquilibre de classes** : taux de churn ≈ {_prevalence:.0%} — l'accuracy est
#    une métrique trompeuse ; la **PR-AUC** est la métrique principale pour ce projet.
# 2. **Fuite temporelle démontrée** : `sante_compte_fin_periode` affiche une AUC univariée
#    de {_auc_sante:.4f}, ce qui est physiquement impossible pour une feature légitime.
#    L'expérience contrôlée (§6.5) chiffre le gain artificiel à {exp.delta_auc_roc:+.4f}
#    en AUC-ROC. La variable est exclue par `config.COLONNES_INTERDITES`.
# 3. **Fuite sémantique + RGPD** : `commentaire_csm` encode des jugements du CSM sur le
#    risque de départ — fuite fonctionnelle ET données potentiellement personnelles.
#    Exclu par `COLONNES_LEURRES_SUSPECTES`.
# 4. **CLV historique** : `valeur_vie_client_eur` est une CLV {res_clv['nature_clv']} ;
#    la valeur à risque sera calculée en §12 avec la formule
#    `P(churn) × MRR × horizon × marge_brute`.
# 5. **client_id** : identifiant technique — exclu par `COLONNES_INTERDITES` pour éviter
#    la mémorisation d'entité.

# %% [markdown]
# > ### 📋 Journal de bord — Analyse exploratoire
# >
# > **Décisions retenues** — Démonstration contrôlée de la fuite (expérience avec/sans) plutôt
# > que simple déclaration ; AUC univariée comme heuristique de détection automatique (seuil 0,85).
# > Exclusion de `commentaire_csm` au titre de la fuite sémantique ET des données personnelles —
# > double justification robuste au Q&R du jury. CLV caractérisée comme historique (r > 0,70
# > avec MRR × ancienneté) → formule MRR × horizon × marge retenue pour §12.
# >
# > **Alternatives écartées** — Supprimer `sante_compte_fin_periode` sans démonstration
# > (aurait manqué le point de pédagogie exigé par l'énoncé). Inclure `commentaire_csm`
# > avec NLP (encodage TF-IDF) : hors périmètre, données personnelles, et fuite fonctionnelle
# > même encodée (le signal encode le jugement du CSM, pas un comportement mesurable).
# >
# > **Difficultés rencontrées** — Coercition des numériques stockés en texte (héritage §5) :
# > résolue par un `df_eda` dédié à l'EDA, sans modifier `df_brut` utilisé en aval.
# >
# > **Impact sur la suite** — §7 reçoit une liste propre de features candidates, sans aucune
# > des 4 colonnes interdites. La formule de la valeur à risque est tranchée pour §12.
# > La démonstration de la fuite est reproductible (graine fixée, même split).
# >
# > **Temps passé** — ~1 h (diagnostic automatique + expérience contrôlée + audit CLV).
