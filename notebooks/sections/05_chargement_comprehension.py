# %% [markdown]
# ## 5. Chargement et compréhension des données
#
# Avant de prédire quoi que ce soit, il faut savoir ce que contiennent vraiment les fichiers :
# doublons, nombres écrits comme du texte, dates mélangées, valeurs impossibles, trous. Cette
# section **diagnostique** sans rien transformer : chaque anomalie est mesurée ici, puis corrigée
# en §7, seul point de transformation du pipeline (chaîne de traitement). Le tableau de bord du
# §5.11 récapitule le tout.

# %%
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display
from matplotlib import ticker as mticker

from churn_saas import config, viz
from churn_saas.data.loaders import charger_brut
from churn_saas.data.quality import (
    analyser_doublons,
    analyser_manquance,
    coercer_numeriques,
    detecter_valeurs_impossibles,
    diagnostiquer_anciennete,
    estimer_date_reference,
    parser_dates,
    profil_compact,
    proposer_renommage,
)
from churn_saas.format_fr import entier, euros, nombre, pourcentage, styler_fr

# %% [markdown]
# ### 5.1 Chargement des sources et contrôle d'intégrité
#
# L'existence, les droits, la volumétrie et l'empreinte SHA-256 des trois fichiers ont été
# contrôlés au §3.1 ; on n'y revient pas. Chaque source reçoit ici un rôle : le **jeu principal**
# sert à l'apprentissage, le **catalogue des plans** (référentiel statique, sans client ni cible,
# donc sans risque de data leakage, ou fuite de données) est joint en §7.6, et
# l'**échantillon** est écarté une fois prouvé qu'il n'apporte rien de nouveau.
#
# Le jeu principal est lu **tout en texte** : pandas ne convertit rien en silence, et chaque
# défaut de format reste visible pour le diagnostic. Les marqueurs de valeur manquante écrits en
# toutes lettres (chaîne vide, « n/a », « null »…) sont ramenés à `NaN` dès le chargement, sinon
# `isna()` les ignorerait et tous les taux de manquants seraient sous-estimés.

# %%
ct = resultats["Jeu principal"]  # contrôle des sources fait au §3.1

df_brut = charger_brut("churn_saas_complet")
_MARQUEURS_NA = ["", "n/a", "na", "nan", "null", "none", "#n/a", "-", "nd", "nr", "inconnu"]
df_brut = df_brut.replace({m: np.nan for m in _MARQUEURS_NA})

# Extrait tiré au hasard (graine unique), transposé : une ligne par colonne, lisible en largeur.
display(df_brut.sample(5, random_state=config.RANDOM_SEED).T)

_nb_colonnes_texte = sum(not pd.api.types.is_numeric_dtype(t) for t in df_brut.dtypes)
display(
    Markdown(
        f"**Ce qu'il faut retenir.** Le jeu principal compte **{entier(df_brut.shape[0])} lignes** "
        f"× {df_brut.shape[1]} colonnes, encodé en `{ct['encodage']}` ; ses empreintes figées au "
        f"§3.1 garantissent à chaque relance que les données n'ont pas changé. L'extrait ci-dessus "
        f"montre cinq clients tels qu'ils sont livrés, valeurs manquantes déjà ramenées à `NaN` : "
        f"**{_nb_colonnes_texte} colonnes sur {df_brut.shape[1]}** sont lues comme du texte, "
        f"montants et dates compris. Cinq lignes ne prouvent rien : les défauts entrevus "
        f"ici sont mesurés sur tout le jeu de §5.4 à §5.9."
    )
)

# %% [markdown]
# Le **catalogue des plans** est le référentiel commercial qui sera joint au jeu principal par la
# colonne `plan`. Il faut donc vérifier dès maintenant que chaque libellé de plan du jeu principal
# y trouve une correspondance.

# %%
catalogue_brut = charger_brut("catalogue_plans")
display(catalogue_brut.set_index("plan"))

_plans_principal = set(df_brut["plan"].dropna().unique())
_plans_orphelins = sorted(_plans_principal - set(catalogue_brut["plan"]))

display(
    Markdown(
        f"**Ce qu'il faut retenir.** Le catalogue décrit "
        f"**{len(catalogue_brut)} plans** sur {catalogue_brut.shape[1] - 1} attributs commerciaux "
        f"(prix par siège, délai de réponse garanti, quota…). Le jeu principal contient "
        f"**{len(_plans_principal)} libellés de plan distincts**, "
        + (
            "tous présents dans le catalogue : la jointure sera complète."
            if not _plans_orphelins
            else f"dont {len(_plans_orphelins)} absents du catalogue "
            f"({', '.join(f'« {x} »' for x in _plans_orphelins)}) : variantes d'écriture à "
            f"normaliser en §7 **avant** la jointure, sinon ces clients resteraient sans prix."
        )
    )
)

# %% [markdown]
# **L'échantillon est-il un sous-ensemble strict du jeu principal ?** S'il contenait des lignes
# absentes du complet, l'écarter ferait perdre de l'information ; s'il y figure déjà, l'ajouter
# créerait des doublons et surpondérerait ces clients. On compare ligne à ligne, après la même
# normalisation des marqueurs manquants (pandas apparie `NaN` avec `NaN` dans une jointure).

# %%
df_echantillon = charger_brut("churn_saas_echantillon").replace({m: np.nan for m in _MARQUEURS_NA})
_memes_colonnes = list(df_echantillon.columns) == list(df_brut.columns)
_complet_brut = df_brut.drop_duplicates()
_jointure = df_echantillon.merge(
    _complet_brut, how="left", on=list(df_echantillon.columns), indicator=True
)
_nb_lignes_retrouvees = int((_jointure["_merge"] == "both").sum())
_sous_ensemble_strict = (
    _memes_colonnes
    and _nb_lignes_retrouvees == len(df_echantillon)
    and len(df_echantillon) < len(_complet_brut)
)
_controles = {
    "Lignes de l'échantillon": entier(len(df_echantillon)),
    "Lignes du jeu principal (hors doublons exacts)": entier(len(_complet_brut)),
    "Schéma identique (noms et ordre des colonnes)": _memes_colonnes,
    "Lignes de l'échantillon retrouvées à l'identique": entier(_nb_lignes_retrouvees),
    "Tous ses client_id présents dans le complet": bool(
        df_echantillon["client_id"].isin(df_brut["client_id"]).all()
    ),
    "Sous-ensemble strict": _sous_ensemble_strict,
}
display(pd.Series(_controles, name="Résultat").to_frame())
assert _sous_ensemble_strict, "L'échantillon contient des lignes absentes du jeu principal"

display(
    Markdown(
        f"**Ce qu'il faut retenir.** Les **{entier(_nb_lignes_retrouvees)} lignes** de "
        f"l'échantillon figurent à l'identique, sur les {df_echantillon.shape[1]} colonnes, dans "
        f"le jeu principal ({pourcentage(len(df_echantillon) / len(_complet_brut), 1)} de "
        f"celui-ci). L'échantillon n'apporte **aucune information nouvelle** : il est écarté. "
        f"L'`assert` fait échouer le notebook si une future livraison rompait cette propriété."
    )
)

# %% [markdown]
# ### 5.2 Premier regard sur le portefeuille
#
# Avant tout diagnostic, on regarde le portefeuille comme le ferait la direction : combien de
# petits et de gros comptes, quel revenu au total, quel poids pour les comptes partis. L'aperçu
# est volontairement **naïf** : copies exactes retirées, MRR (revenu mensuel récurrent) converti
# en nombre, rien d'autre. C'est une copie de travail, pas une transformation (§7 reste l'unique
# point de transformation) ; les chiffres sont des ordres de grandeur, la prévalence définitive
# est établie en §6.1 et le bilan économique en §12.
#
# Pour comparer petits et gros comptes sans seuil arbitraire, on les range par **quartile** de
# MRR : quatre tranches contenant chacune un quart des comptes. Le **revenu annuel récurrent**
# vaut MRR × 12.

# %%
_df_mrr, _ = coercer_numeriques(_complet_brut, ["revenu_mensuel_recurrent_eur"])
apercu = pd.DataFrame(
    {
        "mrr": _df_mrr["revenu_mensuel_recurrent_eur"],
        "churn": pd.to_numeric(_complet_brut["churn"], errors="coerce"),
    }
)
_nb_sans_mrr, _nb_sans_churn = (int(apercu[c].isna().sum()) for c in ("mrr", "churn"))
apercu = apercu.dropna()
apercu["revenu_annuel"] = apercu["mrr"] * 12
apercu["revenu_partants"] = apercu["revenu_annuel"] * apercu["churn"]
_QUARTILES = ["Q1 — plus petits", "Q2", "Q3", "Q4 — plus gros"]
apercu["quartile"] = pd.qcut(apercu["mrr"], 4, labels=_QUARTILES)

_revenu_total = float(apercu["revenu_annuel"].sum())


def _resumer(g: pd.DataFrame) -> pd.Series:
    return pd.Series(
        {
            "Comptes": len(g),
            "MRR min.": g["mrr"].min(),
            "MRR max.": g["mrr"].max(),
            "MRR médian": g["mrr"].median(),
            "Revenu annuel": g["revenu_annuel"].sum(),
            "Part du revenu": g["revenu_annuel"].sum() / _revenu_total,
            "Taux de churn": g["churn"].mean(),
            "Revenu annuel des partants": g["revenu_partants"].sum(),
        }
    )


_par_quartile = {str(q): _resumer(g) for q, g in apercu.groupby("quartile", observed=True)}
tableau_portefeuille = pd.DataFrame({**_par_quartile, "Total": _resumer(apercu)}).T
_euros = ["MRR min.", "MRR max.", "MRR médian", "Revenu annuel", "Revenu annuel des partants"]
display(
    styler_fr(
        tableau_portefeuille,
        {"Comptes": entier, "Part du revenu": pourcentage, "Taux de churn": pourcentage}
        | {c: euros for c in _euros},
    )
)

_q1, _q4 = tableau_portefeuille.loc[_QUARTILES[0]], tableau_portefeuille.loc[_QUARTILES[-1]]
display(
    Markdown(
        f"**Ce qu'il faut retenir.** {entier(len(apercu))} comptes exploitables "
        f"({entier(_nb_sans_mrr)} sans MRR et {entier(_nb_sans_churn)} sans cible mis de côté) "
        f"génèrent **{euros(_revenu_total)} de revenu annuel récurrent**. Les écarts de taille "
        f"sont considérables : le quart des plus petits comptes paie au plus "
        f"{euros(_q1['MRR max.'])} par mois et pèse **{pourcentage(_q1['Part du revenu'], 1)}** "
        f"du revenu ; le quart des plus gros paie au moins {euros(_q4['MRR min.'])} et en pèse "
        f"**{pourcentage(_q4['Part du revenu'], 1)}**. Le churn touche "
        f"{pourcentage(_q1['Taux de churn'], 1)} des petits comptes contre "
        f"{pourcentage(_q4['Taux de churn'], 1)} des gros."
    )
)

# %%
fig_mrr, (ax_hist, ax_conc) = viz.figure_grille(
    "apercu_portefeuille_mrr", "Distribution du MRR et concentration du revenu", 1, 2
)
_bornes = np.logspace(np.log10(apercu["mrr"].min()), np.log10(apercu["mrr"].max()), 40)
ax_hist.hist(
    [apercu.loc[apercu["churn"] == 0, "mrr"], apercu.loc[apercu["churn"] == 1, "mrr"]],
    bins=_bornes,
    stacked=True,
    color=[viz.COULEUR_NON_CHURN, viz.COULEUR_CHURN],
    label=["Restés", "Partis"],
    edgecolor="white",
    linewidth=0.4,
)
_mrr_median = float(apercu["mrr"].median())
ax_hist.axvline(_mrr_median, color=viz.couleur(5), linestyle="--", linewidth=1.5)
ax_hist.text(_mrr_median * 1.1, ax_hist.get_ylim()[1] * 0.92, f"Médiane : {euros(_mrr_median)}")
ax_hist.set_xscale("log")
ax_hist.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: entier(v)))
ax_hist.set(xlabel="MRR (€ par mois, échelle logarithmique)", ylabel="Nombre de comptes")
ax_hist.legend()

# Courbe de concentration : comptes triés du plus gros au plus petit MRR
_cumul = np.cumsum(np.sort(apercu["revenu_annuel"].to_numpy())[::-1]) / _revenu_total
_part_comptes = np.arange(1, len(_cumul) + 1) / len(_cumul)
_part_top20 = float(_cumul[int(np.ceil(0.2 * len(_cumul))) - 1])
ax_conc.plot(_part_comptes, _cumul, color=viz.couleur(6), linewidth=2.5, label="Portefeuille")
ax_conc.plot([0, 1], [0, 1], color="grey", linestyle=":", label="Revenu également réparti")
ax_conc.scatter([0.2], [_part_top20], color=viz.couleur(5), zorder=3)
ax_conc.annotate(
    f"20 % des comptes = {pourcentage(_part_top20, 0)} du revenu",
    (0.2, _part_top20),
    xytext=(0.3, _part_top20 - 0.2),
    arrowprops={"arrowstyle": "->", "color": "grey"},
)
for _axe in (ax_conc.xaxis, ax_conc.yaxis):
    _axe.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
ax_conc.set(
    xlabel="Part des comptes (du plus gros au plus petit MRR)",
    ylabel="Part cumulée du revenu annuel",
    xlim=(0, 1),
    ylim=(0, 1.02),
)
ax_conc.legend(loc="lower right")
fig_mrr.tight_layout()
viz.sauvegarder(fig_mrr)
plt.show()

_mrr_cadrage = float(config.PORTEFEUILLE_CADRAGE["mrr_median_eur"])
display(
    Markdown(
        f"**Ce qu'il faut retenir.** Le MRR s'étale de {euros(apercu['mrr'].min())} à "
        f"{euros(apercu['mrr'].max())} par mois : seule une échelle logarithmique rend la "
        f"distribution lisible. Le revenu est **très concentré** : **20 % des comptes portent "
        f"{pourcentage(_part_top20, 0)} du revenu annuel**. Le MRR médian observé "
        f"({euros(_mrr_median)}) "
        + (
            f"s'écarte nettement des {euros(_mrr_cadrage)} annoncés au cadrage (§2) : "
            "l'hypothèse de valorisation d'un compte sauvé est à revoir avec ces données (§12)."
            if abs(_mrr_median / _mrr_cadrage - 1) > 0.2
            else f"est cohérent avec les {euros(_mrr_cadrage)} annoncés au cadrage (§2)."
        )
    )
)

# %%
fig_partants, (ax_quart, ax_poids) = viz.figure_grille(
    "apercu_portefeuille_partants", "Poids des comptes partis, en nombre et en revenu", 1, 2
)
_t = tableau_portefeuille.loc[_QUARTILES]
_x = np.arange(len(_QUARTILES))
ax_quart.bar(_x - 0.2, _t["Comptes"] / len(apercu), 0.4, color=viz.couleur(7), label="Comptes")
ax_quart.bar(_x + 0.2, _t["Part du revenu"], 0.4, color=viz.couleur(6), label="Revenu annuel")
for _i, _taux in enumerate(_t["Taux de churn"]):
    _haut = max(_t["Comptes"].iloc[_i] / len(apercu), _t["Part du revenu"].iloc[_i])
    ax_quart.text(_i, _haut + 0.02, f"churn {pourcentage(_taux, 0)}", ha="center")
ax_quart.set_xticks(_x, _QUARTILES)
ax_quart.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
ax_quart.set(xlabel="Quartile de MRR", ylabel="Part du portefeuille", ylim=(0, 1.05))
ax_quart.legend(loc="upper left")

_part_partants = {
    "Comptes": float(apercu["churn"].mean()),
    "Revenu annuel": float(apercu["revenu_partants"].sum()) / _revenu_total,
}
_restes = [1 - p for p in _part_partants.values()]
_barres_restes = ax_poids.bar(
    list(_part_partants), _restes, 0.5, color=viz.COULEUR_NON_CHURN, label="Restés"
)
_barres_partis = ax_poids.bar(
    list(_part_partants),
    list(_part_partants.values()),
    0.5,
    bottom=_restes,
    color=viz.COULEUR_CHURN,
    label="Partis",
)
for _barres in (_barres_restes, _barres_partis):
    ax_poids.bar_label(
        _barres,
        labels=[pourcentage(b.get_height(), 1) for b in _barres],
        label_type="center",
        color="white",
        fontweight="bold",
    )
ax_poids.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
ax_poids.set(ylabel="Part du total", ylim=(0, 1))
ax_poids.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
fig_partants.tight_layout()
viz.sauvegarder(fig_partants)
plt.show()

_q_max = _t["Taux de churn"].idxmax()
display(
    Markdown(
        f"**Ce qu'il faut retenir.** Les comptes partis représentent "
        f"**{pourcentage(_part_partants['Comptes'], 1)} des comptes** mais "
        f"**{pourcentage(_part_partants['Revenu annuel'], 1)} du revenu annuel**, soit "
        f"{euros(apercu['revenu_partants'].sum())} par an. "
        + (
            "Les partants sont donc plutôt de petits comptes : "
            if _part_partants["Revenu annuel"] < _part_partants["Comptes"]
            else "Les partants pèsent plus lourd en revenu qu'en nombre : "
        )
        + f"le churn culmine dans le quartile « {_q_max} » "
        f"({pourcentage(_t.loc[_q_max, 'Taux de churn'], 1)}). Deux lectures à garder en tête : "
        "compter les comptes sauvés favoriserait les petits clients, compter le revenu sauvé les "
        "gros. C'est le dilemme des petits comptes (§4) et la raison pour laquelle la "
        "priorisation combine probabilité de départ et MRR (§12.6)."
    )
)

# %% [markdown]
# **Limites.** *Données brutes* : seules les copies exactes sont retirées ; les valeurs
# impossibles ou mal formatées éventuelles sont diagnostiquées plus loin. *MRR manquants* :
# exclus des sommes, le revenu total est donc légèrement sous-estimé. *MRR des partants* :
# c'est le dernier MRR connu, un ordre de grandeur du revenu en jeu, pas une perte comptable.

# %% [markdown]
# ### 5.3 Dictionnaire de données
#
# **C'est le dictionnaire de référence du projet** : les 29 colonnes, groupées par famille, avec
# leur type attendu et leur statut **final** dans le pipeline (la section qui tranche est
# indiquée). Le tableau du §3.2 n'en est qu'un extrait, centré sur la pertinence métier.
#
# | Colonne | Famille | Type attendu | Description | Statut |
# |---|---|---|---|---|
# | `client_id` | Client | texte | Identifiant unique du compte | identifiant — exclu |
# | `date_souscription` | Client | date | Date de début de contrat | non modélisée — n'apporte que l'ancienneté, déjà dans `anciennete_mois` (§6.11) |
# | `jour_souscription` | Client | catégorielle | Jour de semaine à la souscription | non modélisée — sans hypothèse métier ni lien avec le churn (§6.8) |
# | `secteur` | Client | catégorielle | Secteur d'activité | explicative |
# | `pays` | Client | catégorielle | Pays du siège social | explicative |
# | `taille_entreprise` | Client | catégorielle | Taille (effectif) | explicative |
# | `plan` | Client | catégorielle | Formule d'abonnement souscrite | explicative |
# | `anciennete_mois` | Usage produit | numérique | Ancienneté du contrat (mois) | explicative |
# | `sieges_souscrits` | Usage produit | numérique | Nombre de licences souscrites | explicative |
# | `utilisateurs_actifs` | Usage produit | numérique | Utilisateurs actifs (30 j) | explicative |
# | `taux_adoption_pct` | Usage produit | numérique [0-100] | Part des sièges utilisés (%) | explicative |
# | `connexions_30j` | Usage produit | numérique | Nombre de connexions (30 j) | explicative |
# | `heures_usage_30j` | Usage produit | numérique | Heures d'utilisation (30 j) | explicative |
# | `fonctionnalites_total` | Usage produit | numérique | Fonctionnalités disponibles dans le plan | explicative — redondante avec `plan`, conservée pour le taux de couverture fonctionnelle (§7.5) |
# | `fonctionnalites_utilisees` | Usage produit | numérique | Fonctionnalités effectivement utilisées | explicative |
# | `nb_integrations` | Usage produit | numérique | Intégrations tierces actives | explicative |
# | `derniere_connexion_jours` | Usage produit | numérique | Jours depuis la dernière connexion | explicative |
# | `tickets_support_90j` | Support | numérique | Tickets ouverts sur 90 jours | explicative |
# | `delai_reponse_support_h` | Support | numérique | Délai moyen de réponse support (h) | explicative |
# | `csat` | Support | numérique [1-5] | Score de satisfaction client | explicative — hypothèse « les mécontents ne répondent pas » testée en §5.9 |
# | `retards_paiement_12m` | Facturation | numérique | Retards de paiement sur 12 mois | explicative |
# | `revenu_mensuel_recurrent_eur` | Facturation | numérique | Revenu mensuel récurrent du compte (€) | explicative |
# | `couleur_theme_interface` | Leurres | catégorielle | Couleur du thème de l'interface | leurre suspecté — cosmétique ; gardé pour la preuve (§12.9), retiré du modèle déployé (§10.3) |
# | `code_datacenter` | Leurres | catégorielle | Identifiant du centre de données | leurre suspecté — localisation technique ; idem |
# | `groupe_experimentation` | Leurres | catégorielle | Groupe de test A/B | leurre suspecté — biais de sélection possible ; idem |
# | `commentaire_csm` | Leurres | texte libre | Commentaire libre du chargé de clientèle | suspecte (leurre ou fuite ?) — audit §6.6 : interdite |
# | `sante_compte_fin_periode` | Fuite | numérique | Score de santé calculé en FIN de période | ⚠ fuite temporelle — exclue |
# | `valeur_vie_client_eur` | Cible | numérique | Valeur vie client estimée (€) | cible de régression — jamais variable explicative |
# | `churn` | Cible | binaire (0/1) | Résiliation observée (1 = churn) | cible principale |
#
# **Ce qu'il faut retenir.** **19 variables** entrent dans le modèle ; les 2 colonnes de date de
# souscription restent dans le jeu préparé sans être modélisées. Les **4 suspectes** ne sont pas
# supprimées a priori : les trois leurres probables reçoivent un verdict provisoire au §6.10 et
# définitif au §12.9 ; `commentaire_csm`, liée à la cible, relève de l'audit de fuite du §6.6.
# `sante_compte_fin_periode` est une fuite pure (score calculé après le churn) : inscrite dans
# `config.COLONNES_INTERDITES`, elle n'atteindra jamais le modèle.

# %% [markdown]
# ### 5.4 Profil compact du jeu de données principal
#
# On veut une radiographie rapide avant les diagnostics détaillés : `profil_compact()` donne, par
# colonne, le taux de manquants, le nombre de valeurs distinctes, les valeurs dominantes et un
# indicateur `numerique_en_texte` pour les nombres stockés comme du texte.

# %%
profil = profil_compact(df_brut)
display(profil)

nb_avec_manquants = int((profil["taux_manquants"] > 0).sum())
display(
    Markdown(
        f"**Ce qu'il faut retenir.** Sur {df_brut.shape[1]} colonnes, **{nb_avec_manquants}** "
        f"ont des valeurs manquantes (mécanisme étudié au §5.9) et "
        f"**{int(profil['numerique_en_texte'].sum())}** sont des nombres stockés en texte "
        f"(réparation testée au §5.6)."
    )
)

# %% [markdown]
# ### 5.5 Doublons — exacts vs clé métier
#
# Deux sortes de doublons, deux traitements. Une copie **exacte** (toutes colonnes identiques)
# se supprime sans risque. Un même `client_id` avec des données **différentes** n'a pas de
# traitement sûr : mise à jour (garder la plus récente), contacts multiples (agréger) ou erreur de
# jointure (corriger la source). Comme la bonne réponse dépend de la cause, le pipeline s'arrête
# s'il en rencontre, plutôt que de choisir une ligne à l'aveugle.

# %%
rapport_doublons = analyser_doublons(df_brut, cle_metier="client_id")
nb_exact = rapport_doublons["nb_doublons_exacts"]
nb_cle_non_exact = rapport_doublons["nb_doublons_cle_metier_non_exacts"]
n_total = rapport_doublons["nb_total_lignes"]

# analyser_doublons() compte toutes les lignes d'un groupe (original compris) ; les lignes
# redondantes sont celles qu'une déduplication retirerait
nb_redondantes_exactes = n_total - len(_complet_brut)
nb_redondantes_cle = int(_complet_brut.duplicated(subset=["client_id"]).sum())

if nb_exact > 0:
    display(rapport_doublons["exemples_doublons_exacts"])

display(
    Markdown(
        f"**Ce qu'il faut retenir.** Sur {entier(n_total)} lignes, **{entier(nb_exact)}** "
        f"appartiennent à un groupe de doublons exacts, soit **{entier(nb_redondantes_exactes)} "
        f"copie(s) redondante(s)** à supprimer ({entier(len(_complet_brut))} lignes restantes). "
        + (
            "**Aucune ligne** ne partage un `client_id` avec des données divergentes : une fois "
            "les copies retirées, chaque compte apparaît une seule fois. "
            if nb_cle_non_exact == 0
            else f"⚠️ **{entier(nb_cle_non_exact)} ligne(s)** partagent un `client_id` avec des "
            f"données divergentes ({entier(nb_redondantes_cle)} redondante(s)) : la construction "
            f"du jeu préparé s'arrêtera tant que leur cause n'est pas établie. "
        )
        + "Déduplication et contrôle d'unicité sont appliqués en §7.3.1 et dans "
        "`construire_gold_dataset()`."
    )
)

# %% [markdown]
# ### 5.6 Colonnes numériques stockées en texte
#
# Un nombre écrit `12,5`, `45.2 €` ou `12.5 h` est du texte pour pandas : il faut le réparer
# avant tout calcul. `coercer_numeriques()` gère virgule ou point décimal, espaces insécables,
# symboles (€, %) et unités accolées. Piège évité : la convertibilité est mesurée **après**
# réparation ; un `pd.to_numeric` brut échouerait justement sur les colonnes les plus mal
# formatées, qui passeraient alors sous le seuil de détection (80 %) sans jamais être examinées.

# %%
cols_num_texte = list(profil[profil["numerique_en_texte"]].index)
total_repare = total_irrecup = 0
cols_reparees: list[str] = []

if cols_num_texte:
    _, rapport_coercition = coercer_numeriques(df_brut, cols_num_texte)
    display(rapport_coercition)
    total_repare = int(rapport_coercition["nb_reparees"].sum())
    total_irrecup = int(rapport_coercition["nb_irrecuperables"].sum())
    cols_reparees = list(rapport_coercition.index[rapport_coercition["nb_reparees"] > 0])

display(
    Markdown(
        f"**Ce qu'il faut retenir.** {len(cols_num_texte)} colonne(s) numériques stockées en "
        f"texte, dont **{len(cols_reparees)} mal formatée(s)** "
        f"(`{'`, `'.join(cols_reparees) or 'aucune'}`). La coercition répare "
        f"**{entier(total_repare)} valeur(s)** et en laisse **{entier(total_irrecup)} "
        f"irrécupérable(s)**, mises à `NaN`. Appliquée en §7.3, c'est une règle de réécriture "
        f"valeur par valeur qui n'apprend rien des données : l'appliquer avant le découpage en "
        f"plis ne crée pas de fuite."
    )
)

# %% [markdown]
# ### 5.7 Dates multi-formats
#
# `date_souscription` mélange plusieurs écritures (`2021-02-01`, `01/02/2021`…). Appliquer un
# seul format « dominant » laisserait vides toutes les dates écrites autrement : `parser_dates()`
# essaie donc une cascade de formats **valeur par valeur**. Il compte aussi les dates ambiguës
# (`01/02/2021` : 1er février ou 2 janvier ?) et documente la convention retenue pour elles.

# %%
_, rapport_dates = parser_dates(df_brut, ["date_souscription"])
_ligne_dates = rapport_dates.loc["date_souscription"]
_formats = _ligne_dates["repartition_formats"] or ""
display(
    rapport_dates[
        ["repartition_formats", "convention_retenue", "nb_parsees", "nb_ambigues"]
        + ["nb_irrecuperables"]
    ]
)

# Effectifs par format (« %d/%m/%Y : 1621 ») avec séparateur de milliers
_repartition = re.sub(r"\d{4,}", lambda m: entier(int(m.group())), _formats)
display(
    Markdown(
        f"**Ce qu'il faut retenir.** **{len(_formats.split(' | ')) if _formats else 0} formats** "
        f"coexistent (`{_repartition}`). Les {entier(_ligne_dates['nb_ambigues'])} dates ambiguës "
        f"sont lues selon la convention *{_ligne_dates['convention_retenue']}*, comme toutes les "
        f"dates à barre oblique. **{entier(_ligne_dates['nb_parsees'])} dates lues**, "
        f"{entier(_ligne_dates['nb_irrecuperables'])} irrécupérable(s). Même lecture en §7.3, "
        f"sans apprentissage donc sans fuite."
    )
)

# %% [markdown]
# ### 5.8 Valeurs métier impossibles
#
# Certaines valeurs sont fausses par construction, sans aucune statistique :
# `detecter_valeurs_impossibles()` évalue quatre familles de règles.
#
# - **Bornes** : `taux_adoption_pct` ∈ [0, 100], `csat` ∈ [1, 5], `churn` ∈ {0, 1}.
# - **Positivité** des colonnes structurellement positives (compteurs, durées, montants).
# - **Relations** : utilisateurs ≤ sièges, fonctionnalités utilisées ≤ disponibles, taux
#   d'adoption = 100 × utilisateurs / sièges, au moins une connexion sur 30 jours ⇒ dernière
#   connexion il y a 30 jours ou moins.
# - **Cohérence temporelle** : souscription antérieure à l'extraction, ancienneté déclarée
#   cohérente avec la date de souscription.
#
# Le tableau liste **toutes** les règles, même celles qui ne relèvent rien : un contrôle qui ne
# trouve rien ne vaut que si l'on voit ce qui a été contrôlé.
#
# **Quelle date de référence pour l'ancienneté ?** `anciennete_mois` est comptée jusqu'à une
# date d'extraction **T** non fournie. Comparer à la date du jour serait faux (le résultat
# changerait à chaque relance). Chaque ligne donne sa propre estimation de T
# (`date_souscription + anciennete_mois`) ; on en prend la **médiane**, insensible tant que moins
# de la moitié des lignes sont aberrantes, donc capable de les isoler. Elle est crédible si les
# estimations individuelles tiennent dans une fenêtre d'environ un mois (ancienneté arrondie au
# mois) et si T est postérieure à la dernière souscription.

# %%
date_reference = estimer_date_reference(df_brut)
anomalies = detecter_valeurs_impossibles(df_brut, date_reference=date_reference)

df_regles = pd.DataFrame(anomalies.attrs["regles_controlees"]).rename(
    columns={"regle": "Règle", "colonne_ou_paire": "Colonne(s)", "nb_lignes": "Lignes en violation"}
)
# Traitements alignés sur ce que §7.3.4 applique réellement
_TRAITEMENTS = {
    "taux_adoption_pct ∉": "Borné à [0, 100] en §7.3.4",
    "utilisateurs_actifs > sieges_souscrits": "Borné à sieges_souscrits en §7.3.4",
}
df_regles["Traitement"] = [
    (
        "—"
        if n == 0
        else next(
            (t for cle, t in _TRAITEMENTS.items() if r.startswith(cle)),
            "À examiner avant la préparation",
        )
    )
    for r, n in zip(df_regles["Règle"], df_regles["Lignes en violation"], strict=True)
]
display(df_regles.set_index("Règle"))

total_lignes_anomalies = int(df_regles["Lignes en violation"].sum())
_regle_taux = df_regles["Règle"].str.startswith("taux_adoption_pct ≠")
_taux_redondant = (
    bool(_regle_taux.any()) and int(df_regles.loc[_regle_taux, "Lignes en violation"].iloc[0]) == 0
)

display(
    Markdown(
        f"**Ce qu'il faut retenir.** **{len(df_regles)} règles** évaluées, "
        f"**{int((df_regles['Lignes en violation'] > 0).sum())} violée(s)**, "
        f"**{entier(total_lignes_anomalies)} ligne(s)** concernée(s). "
        + (
            "Aucune correction n'est nécessaire ; les bornages de §7.3.4 restent en place comme "
            "garde-fou pour les données futures reçues par l'API et le batch (traitement par lot). "
            if total_lignes_anomalies == 0
            else "La colonne « Traitement » indique le sort de chaque règle violée. "
        )
        + (
            "Point notable : `taux_adoption_pct` vaut exactement 100 × `utilisateurs_actifs` / "
            "`sieges_souscrits` sur toutes les lignes renseignées. Elle est **redondante** : ses "
            "trous se **recalculent exactement** (mieux qu'une imputation), et son poids sera "
            "partagé avec ses deux colonnes sources dans les corrélations (§6) et l'importance "
            "des variables (§12.8)."
            if _taux_redondant
            else ""
        )
    )
)

# %%
diag_anciennete = diagnostiquer_anciennete(df_brut, date_reference=date_reference)
display(
    pd.Series(
        {
            "Date d'extraction estimée (médiane)": f"{diag_anciennete['date_reference']:%d/%m/%Y}",
            "Estimation individuelle la plus ancienne": f"{diag_anciennete['implicite_min']:%d/%m/%Y}",
            "Estimation individuelle la plus récente": f"{diag_anciennete['implicite_max']:%d/%m/%Y}",
            "Étendue des estimations (jours)": entier(diag_anciennete["etendue_jours"]),
            "Écart interquartile des estimations (jours)": entier(diag_anciennete["iqr_jours"]),
            "Dernière souscription observée": f"{diag_anciennete['date_souscription_max']:%d/%m/%Y}",
            "Ancienneté déclarée en mois entiers": (
                "oui" if diag_anciennete["anciennete_entiere"] else "non"
            ),
        },
        name="Valeur",
    ).to_frame()
)

_ecarts = diag_anciennete["repartition_ecarts"]
display(
    Markdown(
        f"**Ce qu'il faut retenir.** Les estimations de la date d'extraction tiennent dans "
        f"**{entier(diag_anciennete['etendue_jours'])} jours** : les deux colonnes décrivent une "
        f"même extraction, datée du **{diag_anciennete['date_reference']:%d/%m/%Y}**, postérieure "
        f"à la dernière souscription. L'ancienneté déclarée s'écarte de l'ancienneté recalculée "
        f"de "
        + ", ".join(f"**{k} mois pour {entier(v)} ligne(s)**" for k, v in _ecarts.items())
        + f" ; l'écart maximal ({max(abs(k) for k in _ecarts)} mois) reste sous la tolérance de "
        f"{diag_anciennete['tolerance_mois']} mois : `anciennete_mois` est utilisable sans "
        f"correction."
    )
)

# %% [markdown]
# **Limites du contrôle d'ancienneté.** *Circularité* : T est estimée sur les données qu'on
# contrôle ; le test isole les lignes incohérentes avec les autres, mais pas un décalage commun à
# toutes. *Tolérance* : 3 mois est un choix, pas une mesure ; il laisse une marge aux données
# futures sans masquer une vraie erreur de saisie (une année). *Arrondi* : 30,44 jours est une
# durée moyenne de mois, qui explique à elle seule les écarts d'un mois.

# %% [markdown]
# ### 5.9 Mécanisme de manquance (MCAR / MAR / MNAR)
#
# Une valeur manquante n'est pas qu'un trou à boucher : **la raison pour laquelle elle manque**
# décide du traitement. La boucher avec la médiane est sans danger si le trou est dû au hasard ;
# c'est une erreur si les trous touchent surtout un type de client.
#
# | Mécanisme | Définition | Exemple pour `csat` | Conséquence |
# |---|---|---|---|
# | **MCAR** — manquant complètement au hasard | La manquance ne dépend de rien. | Un incident d'export a effacé des notes au hasard. | Les clients renseignés restent représentatifs : imputation simple (médiane) sans biais. |
# | **MAR** — manquant au hasard **une fois connues les autres colonnes** | La manquance dépend de colonnes **observées**, pas de la valeur manquante. | L'enquête n'est envoyée qu'aux plans Business et Enterprise. | Imputer **en tenant compte** de ces colonnes ; une médiane globale serait biaisée. |
# | **MNAR** — manquant **pas** au hasard | La manquance dépend de la **valeur manquante elle-même**. | Les mécontents ne répondent pas à l'enquête. | Aucune imputation ne corrige tout ; le fait de manquer est une information, gardée en indicateur (`csat_manquant`). |
#
# **Ce qu'on peut tester.** *MCAR contre MAR* : si la manquance d'une colonne est associée à une
# autre colonne observée (V de Cramér pour les catégorielles, corrélation point-bisériale pour
# les numériques), elle n'est pas MCAR ; 0,15 sépare une association négligeable d'une
# association à prendre en compte. *MNAR ne se prouve pas* : la valeur dont dépend la manquance
# est justement celle qu'on n'a pas. On cherche un indice : si les mécontents ne répondaient pas,
# les clients sans note résilieraient **plus** ; un test du χ² compare le churn des lignes
# manquantes et renseignées.
#
# `analyser_manquance()` applique ces tests dans l'ordre : manquance liée au churn (p < 0,05) →
# **MNAR**, à lire « manquance qui renseigne sur le churn » (indice, pas preuve) ; sinon
# association ≥ 0,15 → **MAR** ; sinon **MCAR**, retenu faute de preuve contraire. Les colonnes
# quasi uniques (identifiant, date brute) sont exclues des associations : le V de Cramér y vaut
# mécaniquement ≈ 1 et ferait conclure à tort à MAR. L'analyse porte sur les lignes dédoublonnées.

# %%
df_manquance = _complet_brut
rapport_manquance = analyser_manquance(df_manquance, cible="churn")
_mecanismes = rapport_manquance.get("mecanisme_propose", pd.Series(dtype=str))
nb_mnar, nb_mar, nb_mcar = (int((_mecanismes == m).sum()) for m in ("MNAR", "MAR", "MCAR"))
cols_mnar = list(_mecanismes.index[_mecanismes == "MNAR"])

if not rapport_manquance.empty:
    display(
        rapport_manquance[
            ["taux_manquants", "taux_churn_si_manquant", "taux_churn_si_renseigne"]
            + ["p_value_cible", "top3_associations", "mecanisme_propose"]
        ].rename(
            columns={
                "taux_manquants": "Part manquante",
                "taux_churn_si_manquant": "Churn si manquant",
                "taux_churn_si_renseigne": "Churn si renseigné",
                "p_value_cible": "p-value (χ² manquance × churn)",
                "top3_associations": "Associations les plus fortes",
                "mecanisme_propose": "Mécanisme",
            }
        )
    )
    _assoc_max = (
        rapport_manquance["top3_associations"].str.extract(r"=(\d+\.\d+)")[0].astype(float).max()
    )
    _p_min = float(rapport_manquance["p_value_cible"].min())
    display(
        Markdown(
            f"**Ce qu'il faut retenir.** Sur {len(rapport_manquance)} colonnes incomplètes : "
            f"**{nb_mnar} MNAR** (`{'`, `'.join(cols_mnar) or 'aucune'}`), **{nb_mar} MAR**, "
            f"**{nb_mcar} MCAR**. Plus petite p-value : {nombre(_p_min, 2)} (seuil 0,05) ; "
            f"association la plus forte : {nombre(_assoc_max, 2)} (seuil 0,15). "
            + (
                "Les trous se comportent comme s'ils étaient répartis au hasard : une imputation "
                "(remplacement par une valeur plausible) simple suffit — médiane pour les "
                'numériques, modalité `"inconnu"` pour les catégorielles — apprise **dans chaque '
                "pli** de cross-validation (validation croisée) en §7.4. "
                if nb_mnar == 0 and nb_mar == 0
                else "Les colonnes MAR demandent une imputation conditionnelle, les colonnes MNAR "
                "un indicateur de manquance en plus de l'imputation. "
            )
            + "Colonnes exclues des associations car quasi uniques : "
            f"`{'`, `'.join(rapport_manquance.attrs.get('colonnes_exclues_associations', [])) or 'aucune'}`."
        )
    )

# %% [markdown]
# **Zoom sur `csat` : « les mécontents ne répondent pas » ?** C'est l'hypothèse MNAR la plus
# plausible. Si elle était vraie, les clients sans note ressembleraient aux clients mal notés, et
# leur taux de churn serait proche de celui des notes 1 et 2.

# %%
_csat_num = pd.to_numeric(df_manquance["csat"], errors="coerce")
_churn_num = pd.to_numeric(df_manquance["churn"], errors="coerce")
df_churn_csat = (
    pd.DataFrame({"csat": _csat_num.map(lambda v: "manquant" if pd.isna(v) else entier(v))})
    .assign(churn=_churn_num)
    .groupby("csat")["churn"]
    .agg(clients="count", taux_churn="mean")
)
display(styler_fr(df_churn_csat, {"taux_churn": pourcentage}))

_churn_manquant = float(df_churn_csat.loc["manquant", "taux_churn"])
_churn_insatisfaits = float(_churn_num[_csat_num <= 2].mean())
_churn_global = float(_churn_num.mean())
_ressemble_mecontents = abs(_churn_manquant - _churn_insatisfaits) < abs(
    _churn_manquant - _churn_global
)
_notes_proches = (
    df_churn_csat.drop(index="manquant")["taux_churn"].sub(_churn_manquant).abs().nsmallest(2).index
)
display(
    Markdown(
        f"**Ce qu'il faut retenir.** Les clients sans note résilient à "
        f"**{pourcentage(_churn_manquant, 1)}**, contre {pourcentage(_churn_insatisfaits, 1)} pour "
        f"les notes 1 ou 2 et {pourcentage(_churn_global, 1)} en moyenne ; ils se comportent comme "
        f"les notes {' et '.join(sorted(_notes_proches))}. "
        + (
            "Ils ressemblent aux mécontents : les données sont compatibles avec l'hypothèse, sans "
            "la prouver. "
            if _ressemble_mecontents
            else "Ils ne ressemblent pas aux mécontents : l'argument principal en faveur d'un "
            "mécanisme MNAR tombe, sans exclure un mécanisme plus subtil. "
        )
        + "Les indicateurs `csat_manquant`, `heures_usage_30j_manquant` et "
        "`delai_reponse_support_h_manquant` sont tout de même créés en §7.5 : coût quasi nul, "
        "aucune fuite (règle sans apprentissage), et un effet combiné reste possible. "
        "L'importance de permutation (§12.8) tranchera."
    )
)

# %% [markdown]
# **Limites.** *MCAR par défaut* : ne pas trouver d'association prouve seulement qu'elle est
# trop faible pour être détectée avec ces effectifs. *Tests multiples* : dix colonnes testées à
# 5 % donnent en moyenne une demi-fausse alerte ; une future détection isolée se lira avec
# prudence. *Seuils* : 0,05 et 0,15 sont des conventions, pas des valeurs optimisées.
# *`commentaire_csm`* manque à plus de 50 %, ce qui justifierait d'habitude de la supprimer ;
# mais l'absence de commentaire peut porter de l'information : l'audit de fuite du §6.6 tranche.

# %% [markdown]
# ### 5.10 Renommage — convention de nommage
#
# Des noms explicites et homogènes rendent le pipeline lisible. Convention : **snake_case sans
# accent, unités suffixées** (`_eur`, `_pct`, `_mois`, `_jours`, `_h`), réservées aux **grandeurs
# mesurées** : comptages (`sieges_souscrits`), scores sans dimension (`csat`) et catégories n'ont
# pas d'unité, et `_30j`, `_90j`, `_12m` sont des **fenêtres d'observation** (`tickets_support_90j`
# compte des tickets sur 90 jours). Le tableau avant/après sert de contrat pour tout le code aval.

# %%
df_renommage = proposer_renommage(df_brut)
display(df_renommage)
nb_modifies = int(df_renommage["modifie"].sum())

display(
    Markdown(
        f"**Ce qu'il faut retenir.** **{nb_modifies} colonne(s) à renommer** sur "
        f"{len(df_renommage)}"
        + (
            " : toutes les grandeurs mesurées portent déjà leur unité. Seule entorse tolérée, "
            "`heures_usage_30j` porte la sienne en tête ; la renommer casserait le contrat de "
            "l'API et du modèle entraîné sans gain de clarté."
            if nb_modifies == 0
            else ", renommées en §7 avant tout feature engineering (ingénierie des variables)."
        )
    )
)

# %% [markdown]
# ### 5.11 Synthèse — tableau de bord qualité
#
# Toutes les anomalies de la section en un seul tableau : volume mesuré, traitement retenu et
# section où il est appliqué. C'est le contrat de données de §7, et la référence à vérifier à
# chaque relance du notebook.

# %%
_CV = "§7.4, dans les plis"
_renom = ("Unités suffixées", "§7") if nb_modifies else ("Aucun (noms conformes)", "—")
_vol = {
    "exacts": f"{entier(nb_redondantes_exactes)} copie(s) / {entier(nb_exact)} ligne(s)",
    "cle": f"{entier(nb_redondantes_cle)} copie(s) / {entier(nb_cle_non_exact)} ligne(s)",
    "texte": f"{entier(total_repare)} réparée(s), {entier(total_irrecup)} irrécupérable(s)",
    "dates": f"{entier(_ligne_dates['nb_parsees'])} lue(s), "
    f"{entier(_ligne_dates['nb_ambigues'])} ambiguë(s)",
    "anciennete": f"écart max. {max(abs(k) for k in _ecarts)} mois "
    f"(tolérance {diag_anciennete['tolerance_mois']})",
}
df_synthese = pd.DataFrame(
    [
        ("Doublons exacts", _vol["exacts"], "Suppression", "§7.3.1"),
        ("Doublons sur `client_id`", _vol["cle"], "Contrôle bloquant d'unicité", "§7.3.1"),
        ("Numériques en texte", _vol["texte"], "`coercer_numeriques()`", "§7.3"),
        ("Dates multi-formats", _vol["dates"], "`parser_dates()` en cascade", "§7.3"),
        ("Valeurs impossibles", f"{entier(total_lignes_anomalies)} ligne(s)", "Bornage", "§7.3.4"),
        ("Ancienneté incohérente", _vol["anciennete"], "Aucun (contrôle seul)", "§7.3"),
        ("Manquants MNAR", f"{nb_mnar} col.", "Indicateurs `_manquant`", "§7.5"),
        ("Manquants MAR", f"{nb_mar} col.", "Imputation conditionnelle" if nb_mar else "—", _CV),
        ("Manquants MCAR", f"{nb_mcar} col.", "Médiane / « inconnu »", _CV),
        ("Renommage", f"{nb_modifies} col.", *_renom),
        ("Colonnes interdites", f"{len(config.COLONNES_INTERDITES)} col.", "Exclusion", "§7, §9"),
    ],
    columns=["Anomalie", "Volume mesuré", "Traitement retenu", "Appliqué en"],
).set_index("Anomalie")
display(df_synthese)

# %% [markdown]
# **Ce qu'il faut retenir.** Chaque anomalie mesurée dans cette section est reliée à la section
# qui la traite : on peut vérifier que l'analyse et le pipeline disent la même chose. Rien n'est
# transformé ici ; §7 est l'unique point de transformation, ce qui garantit la traçabilité et
# l'absence de fuite. Les transformations qui apprennent des données (imputation) sont confinées
# aux plis de cross-validation ; les autres sont des règles fixes, sans risque de fuite.

# %% [markdown]
# > ### 📋 Journal de bord — Chargement et compréhension des données
# >
# > **Décisions retenues** — Lecture tout en texte, marqueurs de valeur manquante ramenés à `NaN`
# > dès le chargement : aucun défaut de format masqué, un seul `isna()` pour tous les taux de
# > manquants. Trois sources, trois rôles ; l'échantillon écarté sur preuve calculée (sous-ensemble
# > strict, verrouillé par un `assert`). Premier regard naïf sur le portefeuille avant tout
# > diagnostic, par quartile de MRR plutôt que par seuils arbitraires : revenu très concentré,
# > partants surtout parmi les petits comptes. Valeurs impossibles contrôlées par un jeu explicite
# > de règles, toutes affichées ; ancienneté contrôlée contre une date d'extraction estimée sur les
# > données. Convention de nommage vérifiée : aucun renommage nécessaire.
# >
# > **Alternatives écartées** — Concaténer l'échantillon (autant de doublons que de lignes,
# > clients surpondérés). Laisser pandas inférer les types (conversions silencieuses qui
# > masqueraient les défauts). Comparer l'ancienneté à la date du jour (résultat faux et variable
# > d'une relance à l'autre). Renommer `heures_usage_30j` en `usage_30j_h` (contrat de l'API et du
# > modèle cassé pour un gain de clarté nul).
# >
# > **Difficultés rencontrées** — Seule la forme de date dominante était d'abord appliquée, les
# > autres devenaient `NaT` : cascade valeur par valeur (`parser_dates()`). Un `pd.to_numeric` brut
# > ne convertissait qu'une valeur sur cinq et masquait la colonne au diagnostic : convertibilité
# > mesurée après réparation. Faux « MAR » (V de Cramér ≈ 1) sur les colonnes quasi uniques :
# > exclues. Chaque point a son test de non-régression.
# >
# > **Impact sur la suite** — Aucune manquance liée au churn démontrée : imputation simple dans les
# > plis (§7.4), indicateurs `_manquant` gardés comme hypothèse et départagés en §12.8.
# > `taux_adoption_pct` recalculée exactement plutôt qu'imputée (§7) ; `plan` normalisé **avant**
# > la jointure du catalogue (§7.6). Le tableau de bord §5.11 sert de contrat de données pour §7.
