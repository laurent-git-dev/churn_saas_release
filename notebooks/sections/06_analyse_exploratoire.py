# %% [markdown]
# ## 6. Analyse exploratoire
#
# Avant de modéliser, on vérifie que les données soutiennent les intuitions métier du §2. On
# cherche aussi ce qui fausserait un modèle : une classe rare, des variables connues trop tard
# (fuites), des variables sans information (leurres) ou une structure temporelle trompeuse.
# L'ordre suit cette logique : cible (§6.1), **test des hypothèses H1 à H14** (§6.2),
# catégorielles (§6.3), puis détection et démonstration des fuites (§6.4 à §6.7), segments
# (§6.8), redondances (§6.9), leurres (§6.10) et temps (§6.11).
#
# Toute l'EDA (*exploratory data analysis*, analyse exploratoire) porte sur les lignes
# dédoublonnées (§5.5) : un client compté deux fois biaiserait la prévalence et pourrait se
# retrouver à la fois en entraînement et en test dans l'expérience du §6.5.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display
from matplotlib.patches import Patch

from churn_saas import config, eda, fuite, viz
from churn_saas.data.quality import coercer_numeriques, parser_dates
from churn_saas.features.build import COLONNES_NON_MODELISEES, ajouter_features_metier
from churn_saas.format_fr import entier, euros, nombre, pourcentage, scientifique, styler_fr

# df_brut (§5) est lu en texte : même coercition robuste qu'en §5.6, sinon « 12,5 » ou
# « 45.2 % » deviendraient NaN et fausseraient manquants et statistiques
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

df_brut_dedup = df_brut.drop_duplicates()
df_eda, _rapport_coercition_eda = coercer_numeriques(df_brut_dedup, _COLS_NUM)

print(
    f"Dédoublonnage : {entier(len(df_brut))} → {entier(len(df_brut_dedup))} lignes "
    f"({entier(len(df_brut) - len(df_brut_dedup))} doublon(s) exact(s) écarté(s) de l'EDA)"
)
print(
    f"DataFrame EDA : {entier(df_eda.shape[0])} lignes × {df_eda.shape[1]} colonnes — "
    f"{df_eda.select_dtypes('number').shape[1]} numériques coercées "
    f"({entier(_rapport_coercition_eda['nb_reparees'].sum())} valeur(s) réparée(s), "
    f"{entier(_rapport_coercition_eda['nb_irrecuperables'].sum())} irrécupérable(s))."
)

# %% [markdown]
# ### 6.1 Distribution de la cible — déséquilibre de classes
#
# La part de clients qui résilient (prévalence) décide de la façon de mesurer un modèle. Le
# ratio de déséquilibre compte les non-churners pour un churner : `(1 − prévalence) /
# prévalence`. Lecture usuelle : classe minoritaire sous 10 % → déséquilibre **fort** ; de 10 à
# 40 % → **modéré** ; au-delà → classes quasi équilibrées.
#
# Le critère décisif reste la comparaison avec un **modèle naïf** qui prédit toujours
# « non-churn » : si son accuracy (exactitude) est élevée alors qu'il ne trouve aucun churner,
# l'accuracy ne mesure pas ce qui intéresse le métier.

# %%
fig_cible, tableau_cible = eda.distribution_cible(df_eda, cible="churn")
display(tableau_cible)

# %%
_prevalence = float(df_eda["churn"].dropna().mean())
_ratio = (1 - _prevalence) / _prevalence
_part_minoritaire = min(_prevalence, 1 - _prevalence)
if _part_minoritaire < 0.10:
    _niveau_desequilibre = "fort"
elif _part_minoritaire < 0.40:
    _niveau_desequilibre = "modéré"
else:
    _niveau_desequilibre = "faible (classes quasi équilibrées)"

display(
    Markdown(
        f"**Ce qu'il faut retenir.** Le taux de churn est de **{pourcentage(_prevalence)}**, soit "
        f"{nombre(_ratio, 1)} non-churners pour 1 churner : déséquilibre "
        f"**{_niveau_desequilibre}**. Un modèle naïf « toujours non-churn » aurait une accuracy "
        f"de {pourcentage(1 - _prevalence)} sans détecter un seul client à risque : l'accuracy "
        f"est écartée.\n\n"
        f"Trois métriques la remplacent (cibles fixées *a priori* au §2) :\n\n"
        f"- **PR-AUC** (aire sous la courbe précision-rappel), métrique de sélection : elle juge "
        f"la qualité de la liste des comptes signalés. Le hasard vaut la prévalence "
        f"({nombre(_prevalence)}) ; cible ≥ {nombre(config.CIBLES_PERFORMANCE['pr_auc_min'])} ;\n"
        f"- **ROC-AUC**, indépendante de la prévalence (hasard = 0,50) : cible ≥ "
        f"{nombre(config.CIBLES_PERFORMANCE['roc_auc_min'])} ;\n"
        f"- **recall** (rappel) au seuil de vigilance, car un churner manqué (faux négatif) coûte "
        f"bien plus qu'une surveillance inutile : seuil réglé pour un recall de "
        f"{nombre(config.RECALL_CIBLE_VIGILANCE)}, plancher "
        f"{nombre(config.CIBLES_PERFORMANCE['recall_vigilance_min'])}."
    )
)

# %% [markdown]
# ### 6.2 Test des hypothèses métier H1 à H14
#
# Le §2 a posé quatorze intuitions du métier (« peu de connexions annoncent un départ »…). On
# les confronte aux données **avant** de modéliser : une hypothèse confirmée justifie une
# variable, une hypothèse infirmée signale une intuition fausse ou une donnée suspecte.
#
# Pour chaque hypothèse, `eda.tester_hypotheses` compare churners et non-churners :
#
# | Indicateur | Question posée | Lecture |
# |---|---|---|
# | médianes churn / non-churn | Les churners ont-ils des valeurs plus hautes ou plus basses ? | donne le sens observé, comparé au sens attendu |
# | test de Mann-Whitney | L'écart peut-il venir du hasard ? | test sur les rangs, sans supposer de loi normale (variables asymétriques) |
# | correction de Holm | Quatorze tests font-ils naître un faux positif ? | p-value ajustée ; risque global d'erreur ≤ 5 % |
# | AUC orientée | L'effet est-il fort ? | probabilité qu'un churner soit du côté « risqué » ; 0,50 = aucun effet |
#
# Verdict : **confirmée** (significative, sens attendu), **infirmée** (significative, sens
# contraire) ou **non concluante**. Avec plusieurs milliers de clients, un effet minime peut
# être significatif : l'AUC orientée dit s'il compte. H13 porte sur le **haut** de la
# distribution du CSAT (quartile supérieur), sinon elle doublerait H7.
#
# H9, H10 et H12 portent sur des ratios du §7 (`ajouter_features_metier`). Ils sont calculés
# ligne par ligne, sans paramètre appris : les calculer ici ne crée aucune fuite.

# %%
_SEUIL_EFFET_NOTABLE = 0.60  # AUC orientée : en deçà, effet réel mais faible (convention a priori)

_df_derive = ajouter_features_metier(df_eda)
_cols_derivees = [c for c in _df_derive.columns if c not in df_eda.columns]
df_hypotheses = df_eda.join(_df_derive[_cols_derivees])
fig_hypotheses, tableau_hypotheses = eda.tester_hypotheses(df_hypotheses, cible="churn")

display(
    styler_fr(
        tableau_hypotheses.drop(columns="p_value_brute"),
        {
            "mediane_churn": lambda v: nombre(v, 2),
            "mediane_non_churn": lambda v: nombre(v, 2),
            "p_value_holm": scientifique,
            "auc_orientee": lambda v: nombre(v, 3),
        },
    )
)

# %%
_verdicts = tableau_hypotheses["verdict"].value_counts()
_confirmees = tableau_hypotheses[tableau_hypotheses["verdict"] == "confirmée"]
_non_confirmees = tableau_hypotheses[tableau_hypotheses["verdict"] != "confirmée"]
_effets_faibles = _confirmees[_confirmees["auc_orientee"] < _SEUIL_EFFET_NOTABLE]
_plus_fortes = _confirmees["auc_orientee"].nlargest(3)


def _liste_hyp(ids: pd.Index) -> str:
    return ", ".join(
        f"{h} `{tableau_hypotheses.loc[h, 'variable']}` "
        f"({nombre(tableau_hypotheses.loc[h, 'auc_orientee'], 3)})"
        for h in ids
    )


_auc_h1 = float(tableau_hypotheses.loc["H1", "auc_orientee"])
_auc_h9 = float(tableau_hypotheses.loc["H9", "auc_orientee"])
display(
    Markdown(
        f"**Ce qu'il faut retenir.** Sur {len(tableau_hypotheses)} hypothèses : "
        + ", ".join(f"**{n}** {v}(s)" for v, n in _verdicts.items())
        + ". "
        + (
            f"Non confirmées : {_liste_hyp(_non_confirmees.index)} : l'intuition métier n'est "
            f"pas soutenue par les données et la variable devra faire ses preuves au §12. "
            if len(_non_confirmees)
            else "Toutes les intuitions du métier vont dans le sens observé. "
        )
        + f"\n\n- **Effets les plus nets** (AUC orientée) : {_liste_hyp(_plus_fortes.index)}.\n"
        + (
            f"- **Effets réels mais faibles** (AUC < {nombre(_SEUIL_EFFET_NOTABLE)}) : "
            f"{_liste_hyp(_effets_faibles.index)}. Utiles en complément, insuffisants seuls.\n"
            if len(_effets_faibles)
            else ""
        )
        + f"- **H9 double H1** (AUC {nombre(_auc_h9, 3)} contre {nombre(_auc_h1, 3)}) : "
        f"`taux_utilisation_sieges` est le taux d'adoption ramené entre 0 et 1, et le contrôle "
        f"de cohérence du §7 le confirme. Redondance attendue, à garder en tête pour lire "
        f"l'importance des variables au §12.\n\n"
        f"Aucune hypothèse prise seule n'approche une séparation parfaite : le risque de départ "
        f"se lit dans la **combinaison** de signaux d'usage, de support, de facturation et de "
        f"contrat, ce qui justifie un modèle multivarié plutôt qu'une règle sur une variable."
    )
)

# %% [markdown]
# **Forme des distributions.** Les boîtes ci-dessus montrent des queues longues. On mesure
# leur asymétrie (0 = symétrique, au-delà de 1 = forte, seuil `config.SEUIL_ASYMETRIE_LOG`) et
# le taux de churn parmi les valeurs extrêmes (*outliers*, hors de [Q1 − 1,5 × IQR ;
# Q3 + 1,5 × IQR], règle de Tukey). Si ce taux diffère nettement du reste, l'extrême décrit un
# profil réel et ne doit pas être supprimé.

# %%
_COLS_NUM_EDA = [c for c in _COLS_NUM if c not in config.COLONNES_INTERDITES]
_MIN_OUTLIERS = 30  # effectif minimal pour qu'un taux de churn soit interprétable


def _profil_extremes(serie: pd.Series) -> dict[str, float]:
    q1, q3 = serie.quantile([0.25, 0.75])
    extreme = (serie < q1 - 1.5 * (q3 - q1)) | (serie > q3 + 1.5 * (q3 - q1))
    churn = df_eda.loc[serie.index, "churn"]
    return {
        "asymétrie": float(serie.skew()),
        "n_outliers": int(extreme.sum()),
        "churn_outliers": float(churn[extreme].mean()) if extreme.any() else np.nan,
        "churn_autres": float(churn[~extreme].mean()),
    }


tableau_formes = pd.DataFrame(
    {c: _profil_extremes(df_eda[c].dropna()) for c in _COLS_NUM_EDA}
).T.sort_values("asymétrie", ascending=False)
tableau_formes["écart_points"] = np.where(
    tableau_formes["n_outliers"] >= _MIN_OUTLIERS,
    100 * (tableau_formes["churn_outliers"] - tableau_formes["churn_autres"]),
    np.nan,
)
display(
    styler_fr(
        tableau_formes,
        {
            "asymétrie": lambda v: nombre(v, 2),
            "n_outliers": entier,
            "churn_outliers": pourcentage,
            "churn_autres": pourcentage,
            "écart_points": lambda v: nombre(v, 1, signe=True),
        },
    )
)

# %%
_SEUIL_ECART = 10  # points de pourcentage
_asym_fortes = tableau_formes.index[
    tableau_formes["asymétrie"].abs() > config.SEUIL_ASYMETRIE_LOG
].tolist()
_extremes_informatifs = tableau_formes.index[
    tableau_formes["écart_points"].abs() >= _SEUIL_ECART
].tolist()


def _liste_cols(cols: pd.Index | list[str]) -> str:
    return ", ".join(f"`{c}`" for c in cols)


display(
    Markdown(
        f"**Ce qu'il faut retenir.** **{len(_asym_fortes)}** des {len(_COLS_NUM_EDA)} variables "
        f"sont fortement asymétriques ({_liste_cols(_asym_fortes)}) : comptages, durées et "
        f"montants, avec beaucoup de petites valeurs et une queue de comptes extrêmes. Pour "
        f"**{len(_extremes_informatifs)}** variables ({_liste_cols(_extremes_informatifs)}), "
        f"le taux de churn des extrêmes s'écarte d'au moins {_SEUIL_ECART} points de celui des "
        f"autres clients : ces extrêmes sont les profils à repérer.\n\n"
        f"Conséquences : **aucun outlier n'est supprimé ni plafonné** ; l'imputation se fait par "
        f"la **médiane**, peu sensible aux extrêmes ; les arbres ne voient que l'ordre des "
        f"valeurs et ignorent l'asymétrie. Pour la régression logistique, une standardisation ne "
        f"réduit pas une queue (transformation affine) : seul un `log1p` le fait, hypothèse "
        f"testée au §7.8.3."
    )
)

# %% [markdown]
# ### 6.3 Variables catégorielles — qualité avant encodage
#
# Avant l'encodage one-hot du §7 (une colonne par modalité), trois contrôles : la
# **cardinalité** (nombre de colonnes créées), les **variantes typographiques** (« France » et
# « france␣ » deviendraient deux colonnes, normalisées par `.strip().lower()` au §7.3.5) et
# les **modalités rares** (effectif < 10), qui risquent de manquer dans un pli de
# cross-validation (validation croisée). La figure montre les modalités brutes ;
# `commentaire_csm` est audité à part (§6.6).

# %%
_COLS_CAT_EDA = [
    "secteur",
    "pays",
    "taille_entreprise",
    "plan",
    "couleur_theme_interface",
    "code_datacenter",
    "groupe_experimentation",
    "jour_souscription",
]
_SEUIL_RARE = 10

fig_cat, tableau_cat = eda.univarie_categorielles(df_eda, _COLS_CAT_EDA, seuil_rare=_SEUIL_RARE)
display(
    styler_fr(
        tableau_cat.drop(columns="seuil_rare"),
        {"fréquence_mode_pct": lambda v: pourcentage(v / 100)},
    )
)

# %%
_n_variantes = int(tableau_cat["variantes_typographiques"].sum())
_cols_variantes = tableau_cat[tableau_cat["variantes_typographiques"] > 0]
_col_exemple = _cols_variantes["variantes_typographiques"].idxmax() if _n_variantes else None
_n_rares_tot = int(tableau_cat["n_modalités_rares"].sum())
_n_rares_norm = int(tableau_cat["n_rares_normalisées"].sum())
_card_brute = int(tableau_cat["cardinalité"].sum())
_card_norm = int(tableau_cat["cardinalité_normalisée"].sum())
_cols_manquants = tableau_cat[tableau_cat["n_manquants"] > 0]

display(
    Markdown(
        "**Ce qu'il faut retenir.**\n\n"
        + (
            f"- **{_n_variantes} variantes typographiques** sur {len(_cols_variantes)} colonnes "
            f"({_liste_cols(_cols_variantes.index)}) ; par exemple `{_col_exemple}` contient "
            f"{_cols_variantes.loc[_col_exemple, 'exemple_variantes']} (« ␣ » = espace). Sans "
            f"normalisation : {entier(_card_brute)} modalités au lieu de {entier(_card_norm)}, "
            f"soit {entier(_card_brute - _card_norm)} colonnes one-hot en trop.\n"
            if _n_variantes
            else "- **Aucune variante typographique.**\n"
        )
        + (
            f"- **{_n_rares_tot} modalité(s) rare(s)**, {_n_rares_norm} après normalisation.\n"
            if _n_rares_tot
            else f"- **Aucune modalité rare** (effectif < {_SEUIL_RARE}). Une modalité inconnue "
            "en production serait codée par un vecteur nul (`handle_unknown='ignore'`).\n"
        )
        + (
            "- **Manquants** : "
            + ", ".join(
                f"`{c}` ({pourcentage(ligne['n_manquants'] / len(df_eda))})"
                for c, ligne in _cols_manquants.iterrows()
            )
            + ", imputés par une modalité `inconnu`.\n"
            if len(_cols_manquants)
            else "- **Aucun manquant.**\n"
        )
    )
)

# %% [markdown]
# ### 6.4 Pouvoir discriminant univarié et première détection de fuites
#
# Une variable qui prédit presque parfaitement le churn **à elle seule**, sans modèle, est en
# général une information connue *après* la résiliation : c'est une data leakage (fuite de
# données). On mesure donc le lien de chaque variable avec la cible, prise isolément.
#
# **Indicateur : la ROC-AUC univariée.** Si l'on tire au hasard un churner et un non-churner,
# c'est la probabilité que la variable les classe dans le bon ordre. Elle est préférée ici à
# la PR-AUC parce que sa référence est la même pour toutes les variables (0,50 = aucun lien)
# et qu'elle est symétrique : on retient `max(AUC, 1 − AUC)` et la colonne `sens` indique si
# les valeurs élevées vont avec plus (« ↑ churn ») ou moins (« ↓ churn ») de churn. La PR-AUC
# reste la métrique de sélection du modèle (§6.1) : elle répond à une autre question.
#
# Pour une catégorielle, le score est le taux de churn de la modalité (normalisée, §7.3.5) :
# AUC légèrement optimiste, effet faible vu la cardinalité (§6.3).
#
# **Seuil d'alerte : ROC-AUC ≥ 0,85**, fixé *a priori*, niveau que des variables d'usage, de
# support ou de facturation n'atteignent pas seules. Dépasser le seuil est une **alerte**, pas
# une preuve : la preuve est temporelle. Rester sous le seuil ne garantit rien non plus : les
# colonnes suspectes sont examinées une à une (§6.5 à §6.7).

# %%
_SEUIL_FUITE = 0.85
# Colonnes non évaluées, avec la raison affichée dans le commentaire
_EXCLUES_DIAG = {
    "client_id": "identifiant technique, sans valeur prédictive",
    "commentaire_csm": "texte libre, audité au §6.6",
    "date_souscription": "date brute, non comparable telle quelle ; étudiée au §6.11",
}

# Les colonnes interdites sont gardées volontairement pour les faire ressortir
df_diag = df_eda.copy()
# Même normalisation qu'au §7.3.5 : une modalité = un taux de churn
for _col_diag in _COLS_CAT_EDA:
    df_diag[_col_diag] = df_diag[_col_diag].str.strip().str.lower()
tableau_fuite = fuite.diagnostiquer_fuite(
    df_diag, cible="churn", seuil=_SEUIL_FUITE, exclure=set(_EXCLUES_DIAG)
)
display(
    styler_fr(
        tableau_fuite.drop(columns="seuil").set_index("colonne"),
        {"auc_univariee": lambda v: nombre(v, 3), "part_manquants": pourcentage},
    )
)

# %%
_fuite_idx = tableau_fuite.set_index("colonne")
_suspects = _fuite_idx.index[_fuite_idx["suspecte_fuite"]].tolist()
# Variables légitimes : celles qui ne sont pas interdites par config
_legitimes = _fuite_idx.drop(index=config.COLONNES_INTERDITES, errors="ignore")
_meilleure_legitime = _legitimes["auc_univariee"].idxmax()
_auc_meilleure_legitime = float(_legitimes["auc_univariee"].max())
# Colonnes interdites que le seuil ne détecte pas
_interdites_sous_seuil = [
    c
    for c in config.COLONNES_INTERDITES
    if c in _fuite_idx.index and not _fuite_idx.loc[c, "suspecte_fuite"]
]
_max_manquants = _legitimes.loc[_legitimes["type"] == "numérique", "part_manquants"]


def _liste_auc(tableau: pd.DataFrame) -> str:
    return ", ".join(f"`{c}` ({nombre(v, 3)})" for c, v in tableau["auc_univariee"].items())


display(
    Markdown(
        "**Ce qu'il faut retenir.**\n\n"
        + (
            f"- **Fuite signalée** : {_liste_auc(_fuite_idx.loc[_suspects])} atteint le seuil "
            f"de {nombre(_SEUIL_FUITE)}, alors que la meilleure variable légitime, "
            f"`{_meilleure_legitime}`, plafonne à {nombre(_auc_meilleure_legitime, 3)}. "
            f"L'argument décisif est temporel : `sante_compte_fin_periode` est calculée en "
            f"**fin** de période, après la résiliation. Démonstration au §6.5.\n"
            if _suspects
            else f"- **Aucune colonne** n'atteint le seuil de {nombre(_SEUIL_FUITE)}.\n"
        )
        + (
            "- **Le seuil ne suffit pas** : "
            + _liste_auc(_fuite_idx.loc[_interdites_sous_seuil])
            + " reste sous le seuil alors que la colonne est interdite, car connue après la "
            "décision du client (§6.7). Chaque variable doit aussi être disponible *avant* la "
            "prédiction.\n"
            if _interdites_sous_seuil
            else ""
        )
        + "- **Signaux légitimes** : les sens observés recoupent les hypothèses confirmées au "
        "§6.2 ; les catégorielles sont reprises au §6.8.\n"
        f"- **Manquants** : l'AUC des numériques porte sur les lignes renseignées (au plus "
        f"{pourcentage(_max_manquants.max())} de manquants, pour `{_max_manquants.idxmax()}`). "
        f"Le signal porté par l'absence elle-même est capté par les indicateurs de manquance "
        f"du pipeline (§7).\n"
        "- **Colonnes non évaluées** : "
        + " ; ".join(f"`{c}` ({raison})" for c, raison in _EXCLUES_DIAG.items())
        + "."
    )
)

# %% [markdown]
# ### 6.5 Démonstration de la fuite — `sante_compte_fin_periode`
#
# Le §6.4 a donné l'**alerte** : prise seule, `sante_compte_fin_periode` sépare presque
# parfaitement churners et non-churners. Cette section **mesure** ce que la variable change dans
# un vrai modèle.
#
# **Pourquoi c'est une fuite.** Selon le cahier des charges et le dictionnaire de données (§5), ce score de
# santé est calculé **en fin de période d'observation**, c'est-à-dire après que le churn a été
# constaté (ou non). On cherche à prédire la résiliation *avant* l'échéance, pour laisser le temps
# d'agir. À ce moment-là, le score n'existe pas encore. Un modèle qui l'utilise apprend donc un
# signal indisponible en production : c'est une **fuite temporelle** (*temporal leakage*).
#
# **Protocole.** On entraîne le même modèle deux fois, avec et sans la variable ; seule cette
# variable change entre les deux conditions :
#
# - **Données** : `df_eda`, c'est-à-dire les lignes dédoublonnées dont les numériques ont été
#   convertis de façon robuste au début du §6 ;
# - **Variables** : les numériques non interdites. Les catégorielles sont écartées pour garder un
#   pipeline minimal : la démonstration porte sur une seule colonne, pas sur le meilleur modèle ;
# - **Split** : un seul découpage 80/20 stratifié sur la cible, avec la graine du projet, et le
#   même pour les deux conditions ;
# - **Pipeline** : imputation par la médiane, standardisation, régression logistique
#   (`class_weight="balanced"`), fitté sur le train seulement ;
# - **Mesures** : **PR-AUC**, métrique de sélection retenue au §6.1, et ROC-AUC en complément.
#
# C'est une démonstration contrôlée, pas le modèle final. Celui-ci est évalué en validation
# croisée au §9. Un seul split ne donne pas d'intervalle de confiance, mais il suffit ici :
# l'écart attendu est bien plus grand que la variabilité d'un découpage à l'autre.
#
# La figure produite par la cellule suivante montre les deux courbes ROC : **avec** la variable
# à gauche, **sans** à droite.

# %%
exp = fuite.experience_fuite(df_eda, cible="churn", colonne_suspecte="sante_compte_fin_periode")

# %%
display(
    pd.DataFrame(
        {
            "Condition": [
                f"Avec sante_compte_fin_periode ({exp.avec.n_features} variables)",
                f"Sans sante_compte_fin_periode ({exp.sans.n_features} variables)",
                "Écart (Δ)",
            ],
            "PR-AUC": [
                nombre(exp.avec.pr_auc, 4),
                nombre(exp.sans.pr_auc, 4),
                nombre(exp.delta_pr_auc, 4, signe=True),
            ],
            "ROC-AUC": [
                nombre(exp.avec.auc_roc, 4),
                nombre(exp.sans.auc_roc, 4),
                nombre(exp.delta_auc_roc, 4, signe=True),
            ],
        }
    ).set_index("Condition")
)

# %%
display(
    Markdown(
        f"**Ce qu'il faut retenir.** Ajouter `sante_compte_fin_periode` fait passer la "
        f"**PR-AUC** de **{nombre(exp.sans.pr_auc, 4)}** à **{nombre(exp.avec.pr_auc, 4)}** "
        f"(**{nombre(exp.delta_pr_auc, 4, signe=True)}**), et la ROC-AUC de "
        f"{nombre(exp.sans.auc_roc, 4)} à {nombre(exp.avec.auc_roc, 4)} "
        f"({nombre(exp.delta_auc_roc, 4, signe=True)}). Pour comparaison, un modèle aléatoire "
        f"obtiendrait une PR-AUC égale à la prévalence, {nombre(_prevalence, 2)}.\n\n"
        f"Sans la variable, le modèle détecte déjà nettement les churners. Avec elle, la "
        f"détection devient presque parfaite (courbe de gauche de la figure), ce qui n'est pas "
        f"plausible pour un problème de churn. Ce gain est **artificiel** : il vient d'une "
        f"information qui n'existera pas au moment de prédire. Le modèle ne pourrait pas "
        f"tourner tel quel en production, faute de cette colonne. Et même en remplaçant la "
        f"valeur manquante, il n'atteindrait jamais la performance mesurée ici.\n\n"
        f"Le cahier des charges signalait cette fuite. La démarche la **confirme par la mesure**, "
        f"en deux temps : l'alerte univariée au §6.4, puis l'écart de performance ci-dessus. "
        f"`sante_compte_fin_periode` est inscrite dans `config.COLONNES_INTERDITES` et exclue "
        f"de tout pipeline de modélisation."
    )
)

# %% [markdown]
# ### 6.6 Audit de `commentaire_csm` — fuite sémantique et données personnelles
#
# Le commentaire du Customer Success Manager (CSM) figure parmi les leurres suspectés, mais il
# pose deux risques de plus :
#
# 1. **Fuite sémantique** : rédigé *après* la décision du client, il encoderait la cible. Un
#    mot alarmant ne le prouve pas (un CSM qui repère un risque à temps fournit un signal
#    légitime) ; la question est chronologique, et la date de rédaction est inconnue.
# 2. **Données personnelles** : un champ saisi à la main peut citer un nom, un e-mail ou un
#    téléphone (minimisation, RGPD art. 5(1)(c)).
#
# L'audit procède en trois temps : nature du champ, signal par modalité, **provenance** de ce
# signal (résumé de variables déjà présentes, ou information postérieure au churn).

# %%
_csm = df_eda["commentaire_csm"]
_csm_renseigne = _csm.dropna()
_longueurs_csm = _csm_renseigne.str.len()
_taux_manquant_csm = float(_csm.isna().mean())
_churn_si_manquant = float(df_eda.loc[_csm.isna(), "churn"].mean())
_churn_si_renseigne = float(df_eda.loc[_csm.notna(), "churn"].mean())

# Recherche automatique des données personnelles détectables par motif
_MOTIFS_PII = {
    "adresse e-mail": r"[\w.+-]+@[\w-]+\.[\w.]+",
    "numéro de téléphone": r"(?:\+\d{1,3}[\s.]?)?\d(?:[\s.-]?\d{2}){4}",
}
_n_pii = {
    nom: int(_csm_renseigne.str.contains(motif, regex=True).sum())
    for nom, motif in _MOTIFS_PII.items()
}

display(
    pd.Series(
        {
            "Commentaires renseignés": entier(len(_csm_renseigne)),
            "Valeurs manquantes": pourcentage(_taux_manquant_csm),
            "Libellés distincts": entier(_csm_renseigne.nunique()),
            "Longueur (caractères) min – max": (
                f"{entier(_longueurs_csm.min())} – {entier(_longueurs_csm.max())}"
            ),
            "Taux de churn si commentaire manquant": pourcentage(_churn_si_manquant),
            "Taux de churn si commentaire renseigné": pourcentage(_churn_si_renseigne),
            **{f"Libellés contenant un(e) {nom}": entier(n) for nom, n in _n_pii.items()},
        },
        name="Valeur",
    )
    .rename_axis("Indicateur")
    .to_frame()
)

# %%
display(
    Markdown(
        f"**Ce qu'il faut retenir.** `commentaire_csm` n'est **pas du texte libre** : ses "
        f"{entier(len(_csm_renseigne))} valeurs se répartissent en "
        f"**{entier(_csm_renseigne.nunique())} libellés normalisés** "
        f"({entier(_longueurs_csm.min())} à {entier(_longueurs_csm.max())} caractères), comme "
        f"un menu déroulant. Vide dans {pourcentage(_taux_manquant_csm)} des cas, il donne "
        f"{pourcentage(_churn_si_manquant)} de churn quand il manque et "
        f"{pourcentage(_churn_si_renseigne)} sinon : l'absence n'est pas un signal en soi. "
        f"**Données personnelles** : "
        + (
            "aucun libellé ne contient d'e-mail ni de téléphone"
            if sum(_n_pii.values()) == 0
            else ", ".join(f"{nom} : {entier(n)} libellé(s)" for nom, n in _n_pii.items())
        )
        + " ; le vocabulaire étant fermé, le contrôle est exhaustif (tableau suivant)."
    )
)

# %% [markdown]
# **Signal et provenance de chaque libellé.** Pour chaque libellé : taux de churn avec son
# intervalle de Wilson à 95 % (un taux extrême sur quelques dizaines de comptes est moins sûr
# qu'il n'en a l'air), médianes de cinq variables que le modèle reçoit déjà et, **à titre de
# diagnostic seulement**, la part de comptes dont `sante_compte_fin_periode` (interdite, §6.5)
# est nulle, témoin de ce qui n'est connu qu'en fin de période.

# %%
_VARS_PROFIL_CSM = [
    "taux_adoption_pct",
    "heures_usage_30j",
    "derniere_connexion_jours",
    "tickets_support_90j",
    "csat",
]
fig_csm, tableau_csm = eda.taux_churn_par_modalite(df_eda, colonne="commentaire_csm", cible="churn")
_grp_csm = df_eda.groupby("commentaire_csm")
profil_csm = tableau_csm[["effectif", "taux_churn", "ic_bas", "ic_haut"]].join(
    _grp_csm[_VARS_PROFIL_CSM].median()
)
profil_csm["part_sante_nulle"] = _grp_csm["sante_compte_fin_periode"].apply(
    lambda s: float(s.dropna().eq(0).mean())
)
display(
    styler_fr(
        profil_csm,
        {
            "effectif": entier,
            **{v: pourcentage for v in ["taux_churn", "ic_bas", "ic_haut", "part_sante_nulle"]},
            **{v: (lambda x: nombre(x, 1)) for v in _VARS_PROFIL_CSM},
        },
    )
)

# %%
_SEUIL_CHURN_EXTREME = 0.90
_modalite_max = tableau_csm["taux_churn"].idxmax()
_mod_extremes = tableau_csm.index[tableau_csm["taux_churn"] >= _SEUIL_CHURN_EXTREME].tolist()
_masque_extremes = df_eda["commentaire_csm"].isin(_mod_extremes)
_part_sante_nulle_ext = float(
    df_eda.loc[_masque_extremes, "sante_compte_fin_periode"].dropna().eq(0).mean()
)
_churn_sante_nulle = float(df_eda.loc[df_eda["sante_compte_fin_periode"].eq(0), "churn"].mean())

display(
    Markdown(
        f"**Ce qu'il faut retenir.** Le taux de churn va de "
        f"{pourcentage(tableau_csm['taux_churn'].min())} à "
        f"**{pourcentage(tableau_csm.loc[_modalite_max, 'taux_churn'])}** (« {_modalite_max} », "
        f"IC₉₅ [{pourcentage(tableau_csm.loc[_modalite_max, 'ic_bas'])} ; "
        f"{pourcentage(tableau_csm.loc[_modalite_max, 'ic_haut'])}]), pour "
        f"{pourcentage(_prevalence)} en moyenne : ce n'est pas un leurre au sens strict. Mais "
        f"les libellés traduisent en mots les variables voisines (adoption basse, tickets "
        f"nombreux, CSAT bas) : le commentaire est d'abord un **résumé qualitatif** de "
        f"variables déjà présentes (redondance mesurée au §6.10).\n\n"
        f"Les {len(_mod_extremes)} libellé(s) à plus de {pourcentage(_SEUIL_CHURN_EXTREME, 0)} "
        f"de churn ({', '.join(f'« {m} »' for m in _mod_extremes)}, "
        f"{entier(_masque_extremes.sum())} comptes) décrivent des **comptes à l'arrêt** "
        f"(dernière connexion il y a "
        f"{entier(df_eda.loc[_masque_extremes, 'derniere_connexion_jours'].median())} jours en "
        f"médiane), dont **{pourcentage(_part_sante_nulle_ext)}** ont un score de santé de fin "
        f"de période nul, score qui s'accompagne de {pourcentage(_churn_sante_nulle)} de churn. "
        f"Ils décrivent très probablement des clients **déjà perdus** : la fuite est plausible."
    )
)

# %% [markdown]
# **Décision : `commentaire_csm` est exclue du modèle** (`config.COLONNES_INTERDITES`), pour
# trois raisons dont aucune ne repose sur un mot alarmant :
#
# 1. **Redondance** : son information est déjà portée, plus finement, par l'usage et le
#    support (profil ci-dessus, jumelle identifiée au §6.10) ;
# 2. **Fuite non exclue** : la date de rédaction est inconnue et les libellés les plus
#    prédictifs coïncident avec un score de santé de fin de période nul ;
# 3. **Précaution RGPD** : sans donnée personnelle ici, mais saisi à la main en production ;
#    l'exclure dès la conception (*privacy by design*, §4) évite de devoir le filtrer.
#
# Exclue du prétraitement, son importance au §12 sera **nulle par construction** : ce zéro ne
# prouve rien, le verdict est celui de ce paragraphe.

# %% [markdown]
# ### 6.7 Caractérisation de `valeur_vie_client_eur` — CLV réalisée ou future ?
#
# La CLV (*customer lifetime value*, valeur vie client) est exclue des variables du churn dans
# tous les cas : *prospective*, elle dépend du comportement futur (fuite) ; *réalisée*, elle
# n'apprend rien d'utile. Reste à savoir ce qu'elle mesure, car elle sert de cible secondaire
# (§2, §12).
#
# Une CLV réalisée (cumul des revenus encaissés ≈ MRR × ancienneté) a deux signatures : elle
# ne dépasse presque jamais MRR × ancienneté, et son élasticité à l'ancienneté (c dans
# log CLV = a + b·log MRR + c·log ancienneté) vaut environ 1. Une CLV prospective (MRR × durée
# de vie estimée) dépasse souvent le CA encaissé et dépend peu de l'ancienneté. Les seuils
# sont fixés *a priori* dans `fuite.caracteriser_clv` ; les corrélations de Pearson sont
# descriptives seulement.

# %%
res_clv = fuite.caracteriser_clv(df_eda)
_seuils_clv = res_clv["seuils"]

_criteres_clv = {
    "Part des comptes où CLV > MRR × ancienneté": (
        pourcentage(res_clv["part_clv_sup_cumul"]),
        f"≤ {pourcentage(_seuils_clv['part_max_realisee'], 0)}",
        f"≥ {pourcentage(_seuils_clv['part_min_prospective'], 0)}",
    ),
    "Élasticité de la CLV à l'ancienneté (c)": (
        nombre(res_clv["elasticite_anciennete"], 2),
        f"≥ {nombre(_seuils_clv['elasticite_min_realisee'])}",
        f"< {nombre(_seuils_clv['elasticite_max_prospective'])}",
    ),
    "Élasticité de la CLV au MRR (b)": (nombre(res_clv["elasticite_mrr"], 2), "≈ 1", "≈ 1"),
}
display(
    pd.DataFrame.from_dict(
        _criteres_clv,
        orient="index",
        columns=["Observé", "Attendu si réalisée", "Attendu si prospective"],
    ).rename_axis("Critère")
)
display(styler_fr(res_clv["stats_par_churn"], precision=0))

# %%
_stats_clv = res_clv["stats_par_churn"]
_mois_mrr = res_clv["mois_mrr"]


def _p(valeur: float) -> str:
    # « p = 3,11e-07 », ou « p < 1e-300 » quand scipy renvoie 0 (sous-dépassement)
    texte = scientifique(valeur)
    return f"p {texte}" if texte.startswith("<") else f"p = {texte}"


display(
    Markdown(
        f"**Ce qu'il faut retenir.** La CLV est **{res_clv['nature_clv']}** : elle dépasse le CA "
        f"déjà encaissé pour {pourcentage(res_clv['part_clv_sup_cumul'])} des comptes, ce "
        f"qu'une somme de revenus passés ne peut pas faire, et son élasticité à l'ancienneté "
        f"n'est que de {nombre(res_clv['elasticite_anciennete'], 2)} (contre "
        f"{nombre(res_clv['elasticite_mrr'], 2)} au MRR). r(CLV, MRR × ancienneté) = "
        f"{nombre(res_clv['r_mrr_x_anciennete'], 3)} est même *inférieur* à r(CLV, MRR) = "
        f"{nombre(res_clv['r_mrr'], 3)} ({_p(res_clv['p_mrr'])}). Elle se lit comme **MRR × "
        f"durée de vie estimée** : {nombre(_mois_mrr.loc[0.5], 0)} mois de MRR en médiane "
        f"({nombre(_mois_mrr.loc[0.05], 0)} à {nombre(_mois_mrr.loc[0.95], 0)} entre les 5e et "
        f"95e percentiles).\n\n"
        f"Cette durée implicite est plus courte chez les churners (ρ de Spearman "
        f"{nombre(res_clv['rho_mois_churn'], 2)}, {_p(res_clv['p_mois_churn'])}) : la CLV "
        f"intègre déjà le risque de départ, nouvelle raison de l'exclure des variables. "
        f"{res_clv['avertissement']} La valeur à risque du §12 s'écrit donc sans elle : "
        f"`{res_clv['formule_valeur_risque']}`.\n\n"
        f"L'écart de CLV entre churners ({euros(_stats_clv.loc[1, 'CLV médiane_€'])} en médiane) "
        f"et non-churners ({euros(_stats_clv.loc[0, 'CLV médiane_€'])}) suit surtout celui du "
        f"MRR ({euros(_stats_clv.loc[1, 'MRR médian_€'])} contre "
        f"{euros(_stats_clv.loc[0, 'MRR médian_€'])}) : les comptes qui partent sont plus petits."
    )
)

# %% [markdown]
# ### 6.8 Taux de churn par variable catégorielle discriminante
#
# On cherche les segments les plus à risque. On retient d'abord les catégorielles dont le lien
# avec le churn est **statistiquement établi**, puis on lit leur taux de churn par modalité.
#
# **Critère : test du χ² d'indépendance corrigé par Benjamini-Hochberg (BH).** Le **V de
# Cramér** mesure la force du lien (0 à 1) ; la p-value du χ² dit s'il se distingue du hasard ;
# la correction BH évite qu'une variable sans lien passe le seuil de 5 % une fois sur vingt,
# puisqu'on teste plusieurs variables. La ROC-AUC du §6.4 (colonne `auc_6_4`) n'est pas utilisée
# comme critère : pour une catégorielle, elle est mécaniquement un peu au-dessus de 0,50 même
# sans lien, et d'autant plus que les modalités sont nombreuses. Les calculs portent sur
# `df_diag` (modalités normalisées au §6.4).

# %%
_ALPHA_BIVAR = 0.05
# Toutes les catégorielles du §6.3 sont testées ensemble : la correction BH porte sur ce lot
tableau_assoc_cat = eda.association_categorielles(
    df_diag, _COLS_CAT_EDA, cible="churn", alpha=_ALPHA_BIVAR
).join(tableau_fuite.set_index("colonne")["auc_univariee"].rename("auc_6_4"))
_COLS_BIVAR = tableau_assoc_cat.index[tableau_assoc_cat["significatif_BH"]].tolist()
_cols_ecartees_bivar = tableau_assoc_cat[~tableau_assoc_cat["significatif_BH"]]
# Significatives avant correction mais plus après : ce que la correction BH évite de retenir
_rejetees_par_bh = _cols_ecartees_bivar.index[
    _cols_ecartees_bivar["p_value_brute"] < _ALPHA_BIVAR
].tolist()
_leurres_testes = [c for c in config.COLONNES_LEURRES_SUSPECTES if c in tableau_assoc_cat.index]
_leurres_tous_non_signif = bool(_leurres_testes) and not bool(
    tableau_assoc_cat.loc[_leurres_testes, "significatif_BH"].any()
)

display(
    styler_fr(
        tableau_assoc_cat,
        {
            "n_modalites": entier,
            "v_cramer": lambda v: nombre(v, 3),
            "p_value_brute": scientifique,
            "p_value_BH": scientifique,
            "auc_6_4": lambda v: nombre(v, 3),
        },
    )
)

# %%
display(
    Markdown(
        "**Ce qu'il faut retenir.** "
        + (
            f"{len(_COLS_BIVAR)} variable(s) sur {len(tableau_assoc_cat)} ont un lien significatif "
            f"après correction BH : "
            + ", ".join(
                f"`{c}` (V = {nombre(tableau_assoc_cat.loc[c, 'v_cramer'], 3)})"
                for c in _COLS_BIVAR
            )
            + f". Le V le plus élevé reste modeste "
            f"({nombre(tableau_assoc_cat['v_cramer'].max(), 3)}) : lien réel, mais aucune ne "
            f"sépare seule churners et non-churners. "
            if _COLS_BIVAR
            else f"Aucune des {len(tableau_assoc_cat)} catégorielles n'a de lien significatif. "
        )
        + (
            f"La correction écarte {_liste_cols(_rejetees_par_bh)}, significative(s) avant "
            f"correction seulement : trop peu net pour exclure un faux positif. "
            if _rejetees_par_bh
            else ""
        )
        + (
            f"Aucun leurre suspecté ({_liste_cols(_leurres_testes)}) n'a de lien significatif."
            if _leurres_tous_non_signif
            else ""
        )
    )
)

# %% [markdown]
# Pour chaque variable retenue : taux de churn par modalité et **intervalle de Wilson à 95 %**
# (fiable même sur un petit effectif). Intervalles qui se chevauchent : écart non établi.

# %%
tableaux_bivar = {
    c: eda.taux_churn_par_modalite(df_diag, colonne=c, cible="churn")[1] for c in _COLS_BIVAR
}


def _modalites_hors_moyenne(tableau: pd.DataFrame) -> str:
    # Intervalle de Wilson entièrement d'un côté du taux moyen : écart établi
    au_dessus = tableau.index[tableau["ic_bas"] > _prevalence].tolist()
    au_dessous = tableau.index[tableau["ic_haut"] < _prevalence].tolist()
    return (
        f"au-dessus : {', '.join(au_dessus) or 'aucune'} ; "
        f"au-dessous : {', '.join(au_dessous) or 'aucune'}"
    )


display(
    Markdown(
        "**Ce qu'il faut retenir.** "
        + " ".join(
            f"`{c}` : de {pourcentage(t['taux_churn'].max())} ({t['taux_churn'].idxmax()}) à "
            f"{pourcentage(t['taux_churn'].min())} ({t['taux_churn'].idxmin()}) ; modalités dont "
            f"l'intervalle exclut le taux moyen — {_modalites_hors_moyenne(t)}."
            for c, t in tableaux_bivar.items()
        )
        + " Les modalités au-dessus sont les segments à cibler en premier.\n\n"
        f"Les {len(_cols_ecartees_bivar)} autres variables (V entre "
        f"{nombre(_cols_ecartees_bivar['v_cramer'].min(), 3)} et "
        f"{nombre(_cols_ecartees_bivar['v_cramer'].max(), 3)}) ne sont pas exclues pour ce seul "
        f"motif : une variable sans effet isolé peut compter en combinaison (mesure au §12.8). "
        f"Seule `jour_souscription` est écartée, faute d'hypothèse métier (conclusion de l'EDA)."
    )
)

# %% [markdown]
# **Encodage retenu : one-hot** (`OneHotEncoder(handle_unknown='ignore')`), fitté dans chaque
# pli, pour toutes les catégorielles admises, significatives ou non. La cardinalité est faible
# après normalisation (§6.3). Un encodage ordinal de `plan` ou `taille_entreprise` imposerait
# un écart de risque constant entre niveaux, que les intervalles ci-dessus ne permettent pas de
# vérifier. Le target encoding (encodage par la cible) n'apporte rien à cette cardinalité et
# ouvre un risque de fuite (§7.9).

# %% [markdown]
# ### 6.9 Matrice de corrélation — variables numériques validées
#
# Deux variables presque colinéaires portent **la même information**. Pour un modèle linéaire,
# le poids se répartit arbitrairement entre elles (coefficients instables) ; pour
# l'explicabilité (§12.8), permuter l'une seule dégrade peu le modèle, puisque sa « jumelle »
# compense. Trois mesures, sur les numériques du §6.2 :
#
# | Mesure | Ce qu'elle détecte | Seuil |
# |---|---|---|
# | r de Pearson (heatmap) | lien **linéaire** entre deux variables | \|r\| ≥ 0,85 |
# | ρ de Spearman | lien **monotone**, insensible aux extrêmes (variables asymétriques, §6.2) | \|ρ\| ≥ 0,85 |
# | VIF (*variance inflation factor*) | variable reconstituée par **plusieurs** autres : 1 / (1 − R²) | VIF ≥ 10 |

# %%
fig_corr, tableau_corr = eda.matrice_correlation(df_eda[_COLS_NUM_EDA], seuil_redondance=0.85)

# Contrôle de robustesse : les mêmes paires ressortent-elles avec un coefficient de rang ?
_spearman = df_eda[_COLS_NUM_EDA].corr(method="spearman")
tableau_corr["ρ_spearman"] = [
    round(float(_spearman.loc[v1, v2]), 4)
    for v1, v2 in zip(tableau_corr["variable_1"], tableau_corr["variable_2"], strict=True)
]

# VIF = diagonale de l'inverse de la matrice de corrélation de Pearson
_corr_pearson = df_eda[_COLS_NUM_EDA].corr(method="pearson")
tableau_vif = (
    pd.Series(np.diag(np.linalg.inv(_corr_pearson.to_numpy())), index=_corr_pearson.columns)
    .rename("VIF")
    .sort_values(ascending=False)
    .to_frame()
)
_vif_eleves = tableau_vif[tableau_vif["VIF"] >= 10]

if tableau_corr.empty:
    display(Markdown("Aucune paire redondante détectée (|r| ≥ 0,85)."))
else:
    display(
        styler_fr(
            tableau_corr,
            {"corrélation": lambda v: nombre(v, 2), "ρ_spearman": lambda v: nombre(v, 2)},
        )
    )
display(styler_fr(tableau_vif.head(6), {"VIF": lambda v: nombre(v, 1)}))

# %%
_n_paires = len(tableau_corr)
_liste_paires = " ; ".join(
    f"`{r.variable_1}` ↔ `{r.variable_2}` (r = {nombre(r.corrélation, 2)}, "
    f"ρ = {nombre(r.ρ_spearman, 2)})"
    for r in tableau_corr.itertuples()
)
_texte_vif = (
    f"**{len(_vif_eleves)} variable(s)** ont un VIF ≥ 10 ("
    + ", ".join(f"`{c}` : {nombre(v, 1)}" for c, v in _vif_eleves["VIF"].items())
    + ")"
    if len(_vif_eleves)
    else f"aucune variable n'a un VIF ≥ 10 (maximum : {nombre(tableau_vif['VIF'].max(), 1)})"
)
display(
    Markdown(
        f"**Ce qu'il faut retenir.** **{_n_paires} paire(s)** dépassent |r| ≥ 0,85"
        + (f" : {_liste_paires}." if _n_paires else ".")
        + f" Côté colinéarité multiple, {_texte_vif}."
    )
)

# %% [markdown]
# Ces redondances ont une **explication métier** : le MRR est un prix par siège multiplié par
# les sièges ; sièges et utilisateurs actifs mesurent la taille du compte ; connexions et heures
# mesurent la même intensité d'usage.
#
# **Décision : on conserve les deux variables de chaque paire.** Aucune n'est une fuite ; la
# pénalité L2 de la régression logistique (`C` réglé par Optuna, §9) stabilise les coefficients
# et les arbres n'y sont pas sensibles ; chaque variable sert aux ratios du §7. L'importance des
# variables (§12.8) et le verdict des leurres (§12.9) se lisent **en tenant compte des paires**.

# %% [markdown]
# ### 6.10 Criblage des leurres — verdict provisoire
#
# Une variable leurre ne porte aucune information sur le churn. Le cahier des charges en
# suspecte quatre (`config.COLONNES_LEURRES_SUSPECTES`). Une importance de permutation nulle ne
# suffit pas à conclure : une variable **redondante** avec une jumelle a aussi une importance
# proche de zéro. On réunit donc trois preuves convergentes :
#
# 1. **Association avec le churn** : V de Cramér (χ²) ou AUC (Mann-Whitney), p-values corrigées
#    par BH sur la famille des variables criblées (d'où des valeurs différentes du §6.8) ;
# 2. **Redondance** avec **toutes** les autres variables, par une mesure entre 0 et 1 adaptée
#    au couple de types : V de Cramér (catégorielle × catégorielle), rapport de corrélation η
#    (catégorielle × numérique), \|r\| de Pearson (numérique × numérique) ;
# 3. **Permutation importance et drop-column importance** sur le modèle : verdict au §12.9.
#
# | Condition (dans l'ordre) | Verdict provisoire |
# |---|---|
# | redondance ≥ 0,70 avec une autre variable | redondant : une importance nulle au §12 ne prouvera rien |
# | sinon, association significative (p_BH < 0,05) | potentiellement utile |
# | sinon | leurre probable |


# %%
tableau_criblage_leurres = fuite.cribler_leurres(
    df_eda, cible="churn", colonnes=config.COLONNES_LEURRES_SUSPECTES
)
# Les colonnes d'importance ne sont renseignées qu'au §12 : inutile d'afficher des colonnes vides
display(
    styler_fr(
        tableau_criblage_leurres.drop(columns=["permutation_imp", "drop_column_imp"]),
        {
            "association_marginale": lambda v: nombre(v, 3),
            "p_value_BH": scientifique,
            "max_redondance": lambda v: nombre(v, 3),
        },
    )
)

# %%
_leurres_probables = tableau_criblage_leurres[
    tableau_criblage_leurres["verdict_provisoire"].str.startswith("leurre probable")
].index.tolist()
_redondants = tableau_criblage_leurres[
    tableau_criblage_leurres["verdict_provisoire"].str.startswith("redondant")
].index.tolist()
_potentiellement_utiles = tableau_criblage_leurres[
    tableau_criblage_leurres["verdict_provisoire"].str.startswith("potentiellement")
].index.tolist()


def _detail_criblage(col: str) -> str:
    """Résumé chiffré d'une ligne du criblage : association, p_BH et jumelle la plus proche."""
    ligne = tableau_criblage_leurres.loc[col]
    mesure_assoc = "V de Cramér" if ligne["type"] == "catégorielle" else "AUC"
    return (
        f"`{col}` ({mesure_assoc} {nombre(ligne['association_marginale'], 3)}, "
        f"p_BH {scientifique(ligne['p_value_BH'])} ; redondance max "
        f"{nombre(ligne['max_redondance'], 3)}, {ligne['mesure_redondance']} avec "
        f"`{ligne['variable_jumelle']}`)"
    )


_phrases_criblage = [
    f"**Ce qu'il faut retenir.** Sur les {len(tableau_criblage_leurres)} variables suspectées :"
]
if _leurres_probables:
    _phrases_criblage.append(
        f"- **{len(_leurres_probables)} leurre(s) probable(s)** : "
        + " ; ".join(_detail_criblage(c) for c in _leurres_probables)
        + ". Ni lien avec le churn, ni jumelle : elles n'apprennent rien, seules ou par "
        "l'intermédiaire d'une autre variable."
    )
if _redondants:
    _phrases_criblage.append(
        f"- **{len(_redondants)} redondant(s)** : "
        + " ; ".join(_detail_criblage(c) for c in _redondants)
        + ". Leur information est portée par la jumelle : une importance faible ne prouverait "
        "pas un leurre."
    )
if _potentiellement_utiles:
    _phrases_criblage.append(
        f"- **{len(_potentiellement_utiles)} potentiellement utile(s)** : "
        + " ; ".join(_detail_criblage(c) for c in _potentiellement_utiles)
        + ". Lien significatif et pas de jumelle : ce n'est pas un leurre."
    )
if "commentaire_csm" in tableau_criblage_leurres.index:
    _phrases_criblage.append(
        "\n`commentaire_csm` n'est pas un leurre au sens strict : lié au churn, il est rattaché "
        f"à `{tableau_criblage_leurres.loc['commentaire_csm', 'variable_jumelle']}`, ce qui "
        "confirme la redondance relevée au §6.6. Exclu de toute façon (§6.6), il n'est pas "
        "réévalué au §12.9."
    )
if "code_datacenter" in tableau_criblage_leurres.index:
    _regions = df_eda["code_datacenter"].astype(str)
    _regions_hors_ue = sorted(r for r in _regions.unique() if not r.startswith("eu-"))
    _phrases_criblage.append(
        f"\n`code_datacenter` héberge {pourcentage(_regions.isin(_regions_hors_ue).mean())} des "
        f"comptes hors UE ({', '.join(f'`{r}`' for r in _regions_hors_ue)}), sans lien avec leur "
        "pays : aucun hébergeur ne répartit ainsi ses clients, autre signe d'une variable "
        "fabriquée. L'hébergement hors UE est traité au §4.1."
    )
_phrases_criblage.append(
    "\nVerdict **provisoire** : la preuve fondée sur le modèle vient au §12.9."
)
display(Markdown("\n".join(_phrases_criblage)))

# %% [markdown]
# ### 6.11 Structure temporelle — split temporel possible ?
#
# Entraîner sur le passé et évaluer sur le futur (split temporel) est souvent plus fidèle à la
# production, s'il existe un axe qui sépare un *passé* d'un *futur*. Or `date_souscription` est
# une **date d'entrée**, et tous les clients sont photographiés à la même date (§5). Une coupure
# au 80e percentile isolerait-elle une période future, ou seulement des clients récents ?

# %%
_col_date = "date_souscription"
# Même lecture qu'en §5.7 (cascade de formats valeur par valeur) : un pd.to_datetime unique
# ne reconnaîtrait qu'un seul des trois formats présents et fausserait la plage temporelle
_dates = parser_dates(df_brut_dedup, [_col_date])[0][_col_date]
_d_min = _dates.min()
_d_max = _dates.max()
_duree_mois = (_d_max - _d_min).days / 30.44

_date_split = _dates.quantile(0.80)  # coupure candidate : 80 % des souscriptions avant
_cote_train = _dates <= _date_split
_cote_test = _dates > _date_split
_anciennete = df_eda["anciennete_mois"]
_churn = df_eda["churn"]
_taux_train_temp = float(_churn[_cote_train].mean())
_taux_test_temp = float(_churn[_cote_test].mean())
_anc_train_temp = float(_anciennete[_cote_train].median())
_anc_test_temp = float(_anciennete[_cote_test].median())
# Corrélation de rang : la date de souscription n'est-elle qu'une ancienneté renversée ?
_rho_date_anciennete = float((_dates - _d_min).dt.days.corr(_anciennete, method="spearman"))

fig_dates, ax_trim = viz.figure(
    "structure_temporelle", "Taux de churn par trimestre de souscription", taille=(12.0, 4.8)
)
_stats_trim = _churn.groupby(_dates.dt.to_period("Q")).agg(["mean", "size"])
_effectif_min_trim = 30  # en deçà, taux non interprétable : barre grisée
_trim_fiables = _stats_trim["size"] >= _effectif_min_trim
ax_trim.bar(
    range(len(_stats_trim)),
    _stats_trim["mean"].values,
    color=[viz.COULEUR_CHURN if ok else "lightgrey" for ok in _trim_fiables],
    alpha=0.85,
)
ax_trim.axhline(
    _prevalence, color="grey", linestyle=":", label=f"Taux global — {pourcentage(_prevalence)}"
)
ax_trim.axvline(
    list(_stats_trim.index).index(_date_split.to_period("Q"))
    - 0.5
    + (_date_split - _date_split.to_period("Q").start_time).days / 91.3,
    color=viz.couleur(0),
    linestyle="--",
    label=f"Coupure candidate — {_date_split:%m/%Y}",
)
_poignees, _libelles = ax_trim.get_legend_handles_labels()
if not _trim_fiables.all():
    _poignees.append(Patch(color="lightgrey"))
    _libelles.append(f"Effectif < {entier(_effectif_min_trim)} clients (non interprétable)")
ax_trim.legend(_poignees, _libelles, fontsize=9)
ax_trim.set_xticks(range(len(_stats_trim)))
ax_trim.set_xticklabels(
    [f"T{p.quarter} {p.year}" for p in _stats_trim.index], rotation=45, ha="right", fontsize=9
)
ax_trim.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: pourcentage(v, 0)))
ax_trim.set_ylabel("Taux de churn")
viz.sauvegarder(fig_dates)

# %%
display(
    Markdown(
        f"**Ce qu'il faut retenir.** Les souscriptions couvrent **{nombre(_duree_mois, 1)} mois** "
        f"({_d_min:%d/%m/%Y} → {_d_max:%d/%m/%Y}), assez pour un split temporel en volume. Mais "
        f"la date de souscription est l'ancienneté retournée (Spearman "
        f"{nombre(_rho_date_anciennete, 3)}) : tous les clients sont observés à la même date "
        f"d'extraction ({date_reference:%d/%m/%Y}).\n\n"
        f"Couper au {_date_split:%d/%m/%Y} ne sépare donc pas un passé d'un futur ; cela isole "
        f"les **clients récents** (ancienneté médiane {nombre(_anc_test_temp, 0)} mois contre "
        f"{nombre(_anc_train_temp, 0)}, churn **{pourcentage(_taux_test_temp)}** contre "
        f"{pourcentage(_taux_train_temp)}). Le modèle y serait évalué sur une population qu'il "
        f"n'a pas vue, alors qu'en production il score chaque nuit **tout le portefeuille**.\n\n"
        f"**Décision : split temporel écarté** au profit d'une cross-validation stratifiée "
        f"répétée (`RepeatedStratifiedKFold`, §8.8), dont chaque pli garde le mélange "
        f"d'anciennetés. L'effet de la date d'entrée reste porté par `anciennete_mois` (H14). "
        f"Une vraie validation temporelle demande plusieurs extractions successives (apprendre "
        f"à *t*, évaluer à *t + 1*) : elle relève du suivi en production (§13.3, §13.7)."
    )
)

# %% [markdown]
# ### Conclusion de l'EDA — impact sur la stratégie de modélisation

# %%
_n_confirmees = int((tableau_hypotheses["verdict"] == "confirmée").sum())
display(
    Markdown(
        "**Ce qu'il faut retenir — décisions transmises au §7.**\n\n"
        "1. **Hypothèses métier** — "
        f"{_n_confirmees} sur {len(tableau_hypotheses)} confirmées (§6.2) ; les plus nettes : "
        f"{_liste_hyp(_plus_fortes.index)}. Les ratios qui servent H9, H10 et H12 sont "
        "construits au §7.\n\n"
        "2. **Colonnes exclues** — `config.COLONNES_INTERDITES` ("
        + _liste_cols(config.COLONNES_INTERDITES)
        + ", dont la fuite démontrée au §6.5 et `commentaire_csm` au §6.6), puis "
        "`features.build.COLONNES_NON_MODELISEES` ("
        + _liste_cols(sorted(COLONNES_NON_MODELISEES))
        + ") : `date_souscription` n'apporte que l'ancienneté (§6.11) ; `jour_souscription` n'a "
        "ni hypothèse métier ni lien significatif (p_BH "
        f"{nombre(tableau_assoc_cat.loc['jour_souscription', 'p_value_BH'], 2)}, §6.8).\n\n"
        f"3. **Métriques** — PR-AUC pour la sélection (prévalence {pourcentage(_prevalence)}), "
        f"ROC-AUC et recall au seuil de vigilance en co-principaux (§6.1).\n\n"
        f"4. **Validation** — cross-validation stratifiée répétée ; split temporel écarté "
        f"(§6.11).\n\n"
        "5. **Déséquilibre** — `class_weight='balanced'`, complété au §9.6 par une correction "
        "d'intercept ; SMOTE seulement en comparaison méthodologique.\n\n"
        f"6. **Transformations** — extrêmes conservés et imputation par la médiane ; "
        f"{len(_asym_fortes)} variable(s) asymétrique(s) : `log1p` à tester (§7.8.3) ; "
        f"{_n_paires} paire(s) corrélée(s) conservée(s), lues ensemble au §12.8 ; one-hot "
        f"fitté pli par pli.\n\n"
        f"7. **Leurres** — {len(_leurres_probables)} leurre(s) probable(s) "
        f"({_liste_cols(_leurres_probables) or '—'}), {len(_redondants)} redondant(s) "
        f"({_liste_cols(_redondants) or '—'}) ; confirmation au §12.9.\n\n"
        f"8. **Valeur à risque** — CLV {res_clv['nature_clv']}, donc hors formule : "
        f"`{res_clv['formule_valeur_risque']}`."
    )
)

# %% [markdown]
# > ### 📋 Journal de bord — Analyse exploratoire
# >
# > **Décisions retenues** — Hypothèses H1 à H14 testées avant toute modélisation
# > (Mann-Whitney, Holm, AUC orientée) pour orienter le feature engineering ; PR-AUC pour la
# > sélection, ROC-AUC et recall en co-principaux ; fuite de `sante_compte_fin_periode`
# > démontrée par la mesure ; `commentaire_csm` exclu pour sa provenance ; CLV prospective, hors
# > variables et hors valeur à risque ; extrêmes et paires corrélées conservés ; one-hot fitté
# > pli par pli ; leurres criblés (association puis redondance).
# >
# > **Alternatives écartées** — Univarié exhaustif *(remplacé par le test des hypothèses, qui ne
# > garde que ce qui sert une décision)* ; suppression de `sante_compte_fin_periode` sans
# > démonstration *(la fuite serait restée une affirmation)* ; détection de fuite par mots-clés
# > *(un mot ne dit rien de la chronologie)* ; plafonnement des extrêmes *(retirerait les
# > churners typiques)* ; encodage ordinal, target encoding, suppression d'une variable par paire
# > corrélée ; split temporel *(n'isole que les clients récents)*.
# >
# > **Difficultés rencontrées** — Sans dédoublonnage, un même client tombait des deux côtés de
# > l'expérience de fuite (§6.5) : d'où la copie dédoublonnée. Plusieurs hypothèses métier ne
# > sont pas confirmées par les données : l'intuition a été corrigée, pas forcée. Une seule
# > extraction datée interdit le split temporel.
# >
# > **Impact sur la suite** — Le §7 reçoit la liste numérotée de décisions ci-dessus et ne la
# > redérive pas ; la valeur à risque est fixée pour le §12, où l'importance des variables sera
# > lue en tenant compte des paires corrélées.
