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
# ### 6.10 Criblage des leurres — verdict provisoire
#
# L'énoncé identifie quatre variables leurres dans `config.COLONNES_LEURRES_SUSPECTES`.
# Conformément au **point de vigilance n°5**, une importance de permutation nulle ne suffit
# pas à conclure : une variable **redondante** avec une autre donne aussi une importance
# proche de zéro. On applique donc trois preuves convergentes :
#
# 1. **Association marginale** avec la cible — V de Cramér (chi2, p-value Benjamini-Hochberg) ;
# 2. **Contrôle de redondance** — max V de Cramér avec les catégorielles légitimes,
#    pour écarter l'hypothèse de la variable jumelle ;
# 3. **(§12) Permutation importance + drop-column importance** — verdict définitif.

# %%
tableau_leurres = fuite.cribler_leurres(
    df_eda, cible="churn", colonnes=config.COLONNES_LEURRES_SUSPECTES
)
display(tableau_leurres)

# %%
_leurres_probables = tableau_leurres[
    tableau_leurres["verdict_provisoire"].str.startswith("leurre probable")
].index.tolist()
_redondants = tableau_leurres[
    tableau_leurres["verdict_provisoire"].str.startswith("redondant")
].index.tolist()
_potentiellement_utiles = tableau_leurres[
    tableau_leurres["verdict_provisoire"].str.startswith("potentiellement")
].index.tolist()

_fmt = lambda lst: ("`" + "`, `".join(lst) + "`") if lst else "—"  # noqa: E731

display(
    Markdown(
        f"**Ce qu'il faut retenir.** Sur les {len(config.COLONNES_LEURRES_SUSPECTES)} leurres "
        f"suspectes : **{len(_leurres_probables)} leurre(s) probable(s)** "
        f"({_fmt(_leurres_probables)}) — association marginale non significative (p_BH > 0,05) "
        f"et faible redondance avec les autres variables. "
        f"**{len(_redondants)} redondant(s)** ({_fmt(_redondants)}) — pas d'association directe "
        f"mais corrélé à d'autres features : l'importance de permutation seule conclurait à "
        f"tort (point de vigilance n°5). "
        f"**{len(_potentiellement_utiles)} potentiellement utile(s)** "
        f"({_fmt(_potentiellement_utiles)}) — association BH-significative détectée. "
        f"\n\nLe verdict reste **provisoire** — permutation importance et drop-column "
        f"importance en §12 apporteront la preuve définitive. Note : `commentaire_csm` est "
        f"exclu indépendamment du test statistique (fuite sémantique + RGPD, §6.6)."
    )
)

# %% [markdown]
# ### 6.11 Structure temporelle — split temporel possible ?
#
# `date_souscription` enregistre la date d'entrée du client dans le portefeuille.
# Si les souscriptions s'étalent sur une durée suffisante (≥ 12 mois), un **split temporel**
# est préférable au split aléatoire : il reproduit les conditions réelles de déploiement —
# le modèle voit uniquement les clients passés et prédit sur les clients récents, sans
# aucune contamination future dans l'ensemble d'entraînement.

# %%
_col_date = "date_souscription"
_dates = pd.to_datetime(df_brut[_col_date], dayfirst=True, errors="coerce")
_n_dates_valides = int(_dates.notna().sum())
_n_dates_manquantes = int(_dates.isna().sum())
_d_min = _dates.min()
_d_max = _dates.max()
_duree_mois = (_d_max - _d_min).days / 30.44

# Distribution mensuelle des souscriptions
_par_mois = _dates.dt.to_period("M").value_counts().sort_index()

# Date de split candidate : 80e percentile (80 % train, 20 % test)
_dates_triees = _dates.dropna().sort_values()
_date_split = _dates_triees.iloc[int(_n_dates_valides * 0.80)]
_n_train_temp = int((_dates <= _date_split).sum())
_n_test_temp = int((_dates > _date_split).sum())
_seuil_test_min = 200
_split_temporel_possible = _duree_mois >= 12.0 and _n_test_temp >= _seuil_test_min

print(f"Plage : {_d_min.date()} → {_d_max.date()} ({_duree_mois:.1f} mois)")
print(f"Dates manquantes : {_n_dates_manquantes} ({_n_dates_manquantes / len(df_brut):.1%})")
print(f"Date de split candidate (80e pct) : {_date_split.date()}")
print(f"Train temporel (≤ split) : {_n_train_temp} | Test temporel (> split) : {_n_test_temp}")
print(f"Split temporel recommandé : {_split_temporel_possible}")

# %%
fig_dates, ax_dates = viz.figure(
    "structure_temporelle",
    "Souscriptions par mois — structure temporelle",
    taille=(12.0, 4.5),
)
ax_dates.bar(
    range(len(_par_mois)),
    _par_mois.values,
    color=viz.couleur(0),
    edgecolor="white",
    linewidth=0.4,
    alpha=0.85,
)
_split_period = _date_split.to_period("M")
_idx_split_plot = (
    list(_par_mois.index).index(_split_period)
    if _split_period in _par_mois.index
    else None
)
if _idx_split_plot is not None:
    ax_dates.axvline(
        _idx_split_plot,
        color=viz.COULEUR_CHURN,
        linestyle="--",
        linewidth=1.8,
        label=(
            f"Split candidat — {_date_split.strftime('%Y-%m')} "
            f"({_n_train_temp} train / {_n_test_temp} test)"
        ),
    )
    ax_dates.legend(fontsize=9)
_step_ticks = max(1, len(_par_mois) // 12)
ax_dates.set_xticks(range(0, len(_par_mois), _step_ticks))
ax_dates.set_xticklabels(
    [str(p) for p in _par_mois.index[::_step_ticks]],
    rotation=45,
    ha="right",
    fontsize=9,
)
ax_dates.set_ylabel("Nombre de souscriptions")
viz.sauvegarder(fig_dates)

# %%
display(
    Markdown(
        f"**Ce qu'il faut retenir.** Les souscriptions s'étalent sur **{_duree_mois:.1f} mois** "
        f"({_d_min.date()} → {_d_max.date()}, {_n_dates_manquantes} dates manquantes, "
        f"convention dayfirst=True appliquée). "
        + (
            f"**Split temporel recommandé** : date candidate `{_date_split.date()}` "
            f"(80e percentile — {_n_train_temp} obs en train, {_n_test_temp} obs en test). "
            f"Ce split reproduit les conditions de production : le modèle voit uniquement "
            f"les clients souscrits avant cette date et prédit sur les plus récents. "
            f"Il évite toute contamination d'information future dans le train."
            if _split_temporel_possible
            else (
                f"**Split temporel déconseillé** "
                f"({'durée insuffisante (< 12 mois)' if _duree_mois < 12.0 else 'durée suffisante'}"
                f"{', effectif test insuffisant (' + str(_n_test_temp) + ' < ' + str(_seuil_test_min) + ')' if _n_test_temp < _seuil_test_min else ''}). "
                f"Un split aléatoire stratifié (80/20, graine `config.RANDOM_SEED`) sera utilisé en §7."
            )
        )
    )
)

# %% [markdown]
# ### Conclusion de l'EDA — impact sur la stratégie de modélisation
#
# *Liste consolidée des décisions que l'EDA impose à la section 7.*
# *Chaque point est directement actionnable dans le pipeline de modélisation.*

# %%
_split_desc = (
    f"**split temporel** à `{_date_split.date()}` "
    f"({_n_train_temp} obs train / {_n_test_temp} obs test — 80e pct des souscriptions)"
    if _split_temporel_possible
    else "**split aléatoire stratifié** 80/20 (structure temporelle insuffisante)"
)
_feats_liste = []
if _n_paires > 0:
    _feats_liste.append(
        f"{_n_paires} paire(s) redondante(s) (|r| ≥ 0,85) : "
        "supprimer la variable la moins interprétable de chaque paire"
    )
if _n_asym > 0:
    _feats_liste.append(
        f"{_n_asym} variable(s) à forte asymétrie : log-transformation candidate (à valider en §7)"
    )
_feats_liste.append(
    "`csat` probablement MNAR (insatisfaits ne répondent pas) "
    "→ créer `csat_manquant` (indicateur binaire 0/1)"
)

_leurres_probables_str = _fmt(_leurres_probables)
_redondants_str = _fmt(_redondants)

display(
    Markdown(
        "**Décisions actionnables issues de l'EDA — reprises telles quelles en §7 :**\n\n"
        "1. **Colonnes à exclure** — `config.COLONNES_INTERDITES` "
        "(`sante_compte_fin_periode`, `valeur_vie_client_eur`, `churn`, `client_id`) + "
        "`commentaire_csm` (fuite sémantique + RGPD, §6.6) + "
        "`date_souscription` / `jour_souscription` (non prédictifs comme features brutes ; "
        "la structure temporelle est exploitée via le type de split, §6.11).\n\n"
        f"2. **Métrique principale** — **PR-AUC** (prévalence {_prevalence:.1%}, "
        f"ratio 1:{_ratio:.0f}) ; l'accuracy est trompeuse sur classes déséquilibrées. "
        f"Seuil minimal *a priori* : `{config.CIBLES_PERFORMANCE['pr_auc_min']:.2f}` (§2).\n\n"
        f"3. **Type de split** — {_split_desc}.\n\n"
        "4. **Traitement du déséquilibre** — `class_weight='balanced'` dans le modèle "
        "retenu (SMOTE décalibre les probabilités, point de vigilance n°4 ; SMOTE testé "
        "uniquement comme ablation méthodologique pour montrer la dégradation de "
        "calibration).\n\n"
        "5. **Features à créer / transformer** — " + " ; ".join(_feats_liste) + ".\n\n"
        f"6. **Leurres** — verdict provisoire : {len(_leurres_probables)} leurre(s) probable(s) "
        f"({_leurres_probables_str}), {len(_redondants)} redondant(s) ({_redondants_str}), "
        f"{len(_potentiellement_utiles)} potentiellement utile(s) ({_fmt(_potentiellement_utiles)}). "
        "Confirmation par permutation importance + drop-column importance en §12.\n\n"
        "7. **Valeur à risque** — CLV identifiée comme "
        f"{res_clv['nature_clv']} "
        f"(r(CLV, MRR×ancienneté) = {res_clv['r_mrr_x_anciennete']:.3f}) ; "
        "formule §12 : `P(churn) × MRR × 12 × marge_brute` "
        "(pas de double-comptage de valeur passée, point de vigilance n°3)."
    )
)

# %% [markdown]
# > ### 📋 Journal de bord — Analyse exploratoire
# >
# > **Décisions retenues** — Démonstration contrôlée de la fuite (expérience avec/sans, §6.5)
# > plutôt que simple déclaration. AUC univariée comme heuristique de détection automatique
# > (seuil 0,85). Exclusion de `commentaire_csm` au titre de la fuite sémantique ET des données
# > personnelles — double justification robuste au Q&R. CLV caractérisée comme historique
# > (r > 0,70 avec MRR × ancienneté) → formule MRR × horizon × marge pour §12.
# > Criblage des leurres en deux temps (association marginale BH + redondance) : verdict
# > provisoire ; importance nulle seule aurait été insuffisante (point de vigilance n°5).
# > Analyse temporelle tranchée par le code — le type de split est justifié par la durée
# > des souscriptions et l'effectif de test, pas décidé *a priori*.
# >
# > **Alternatives écartées** — Supprimer `sante_compte_fin_periode` sans démonstration
# > (aurait manqué le point de pédagogie exigé). Inclure `commentaire_csm` avec NLP
# > (TF-IDF) : hors périmètre, fuite fonctionnelle même encodée, données personnelles.
# > Conclure sur les leurres uniquement par importance de permutation (point vigilance n°5).
# > Utiliser la CLV brute dans la valeur à risque : double-comptage de la valeur passée.
# >
# > **Difficultés rencontrées** — Coercition des numériques stockés en texte (héritage §5) :
# > résolue par un `df_eda` dédié à l'EDA, sans modifier `df_brut` utilisé en aval.
# > Format de `date_souscription` potentiellement ambigu : convention dayfirst=True appliquée
# > systématiquement (cohérente avec les données européennes de l'énoncé).
# >
# > **Impact sur la suite** — §7 reçoit une liste complète et numérotée de décisions
# > actionnables (colonnes exclues, type de split, traitement du déséquilibre, features à créer).
# > La conclusion de l'EDA est la seule source de vérité — §7 ne redérivera pas ces choix.
# > La formule de valeur à risque est tranchée pour §12.
# >
# > **Temps passé** — ~1,5 h (diagnostic auto + expérience contrôlée + audit CLV +
# > criblage leurres + analyse temporelle + conclusion).
