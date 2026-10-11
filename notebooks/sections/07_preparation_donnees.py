# %% [markdown]
# ## 7. Préparation des données
#
# On veut donner aux modèles des données propres et des variables qui parlent le langage du
# métier, sans jamais laisser filtrer d'information du jeu de test. Cette section enchaîne le
# nettoyage des défauts repérés au §5, le feature engineering (ingénierie des variables) guidé
# par les hypothèses du §6.2, l'enrichissement externe et le versionnement du jeu *gold* qui
# alimente les §8 et §9. La stratégie anti-fuite est posée **avant** le code.

# %%
import hashlib
import json
import subprocess

import numpy as np
import pandas as pd
from IPython.display import Markdown, display
from scipy.stats import spearmanr, wilcoxon
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from churn_saas import config
from churn_saas.cache import charger_ou_calculer
from churn_saas.cli import construire_gold_dataset
from churn_saas.data.quality import (
    analyser_doublons,
    analyser_manquance,
    coercer_numeriques,
    detecter_valeurs_impossibles,
    parser_dates,
    proposer_renommage,
)
from churn_saas.features.build import (
    ajouter_features_metier,
    construire_preprocesseur,
    controle_coherence_adoption,
    joindre_catalogue,
    reconstituer_taux_adoption,
)
from churn_saas.features.enrichissement import (
    ALIAS_SECTEURS,
    FICHES_SOURCES,
    REFERENTIEL_PAYS,
    REFERENTIEL_SECTORIEL,
    enrichir_par_pays,
    enrichir_par_secteur,
)
from churn_saas.format_fr import entier, nombre, pourcentage, styler_fr
from churn_saas.models.train import construire_modeles, evaluer_modele, protocole_validation

# %% [markdown]
# ### 7.1 Rappel des décisions issues de l'EDA (§6)
#
# Chaque transformation de cette section remonte à un constat des §5-§6 ; aucune n'est
# appliquée « par défaut ».
#
# | Constat | Conséquence en §7 |
# |---|---|
# | Cible déséquilibrée (§6.1) | Aucun rééchantillonnage ici : `class_weight='balanced'` dans le `Pipeline` du §9. |
# | Fuite `sante_compte_fin_periode` (§6.5), audit de `commentaire_csm` (§6.6) | Inscrites dans `config.COLONNES_INTERDITES`, écartées par le `ColumnTransformer` (`remainder='drop'`). |
# | Leurres suspectés (§6.10) | Gardés dans le gold et en entrée du modèle : leur inutilité se **prouve** au §12.9 (une variable absente aurait une importance nulle par construction, ce qui ne prouverait rien). Le modèle déployé est réentraîné sans eux (§10.3). |
# | Hypothèses H1 à H14 (§6.2) | Les ratios du §7.5 servent les hypothèses H3, H6, H9, H10 et H12. |
# | Manquance (§5.9) | Pas de lien avec le churn démontré ; indicateurs de manquance gardés comme hypothèse, départagés au §12.8. |
# | Variables asymétriques (§6.2) | Extrêmes conservés ; un log1p signé est testé, puis écarté (§7.8.3). |
# | Numériques en texte, dates multi-formats, valeurs impossibles (§5.6 à §5.8) | Coercition, parsing en cascade et bornage métier (§7.3). |

# %% [markdown]
# ### 7.2 Stratégie anti-fuite — explication avant le code
#
# **L'intuition.** Un examen ne vaut que si l'élève n'a pas vu les questions : le modèle ne doit
# rien apprendre, même indirectement, des clients sur lesquels on le note.
#
# **La notion exacte.** Le data leakage (fuite de données) survient quand une statistique
# calculée sur tout le jeu (médiane, moyenne, modalités) sert à préparer les données : elle
# contient de l'information sur les lignes de validation et gonfle les scores. Le remède : toute
# transformation *apprise* est placée dans un `Pipeline` scikit-learn (chaîne de traitement)
# dont le `fit()` (apprentissage) ne voit que la partie d'entraînement du pli courant ; le pli
# de validation est seulement transformé. `tests/test_no_leakage.py` le vérifie à chaque CI.
#
# | Peut précéder le split (découpage) | Doit être dans le `Pipeline`, fitté par pli |
# |---|---|
# | Coercition numérique, parsing de dates (aucune statistique) | Imputation par la médiane ou la constante `"inconnu"` |
# | Bornes métier connues a priori (taux ∈ [0, 100], utilisateurs ≤ sièges) | Standardisation (`StandardScaler`) |
# | Ratios ligne par ligne (§7.5), indicateurs de manquance | Encodage one-hot des catégorielles |
# | Jointure sur un référentiel statique (catalogue, secteur, pays) | Agrégats de groupe : `ecart_csat_secteur` (`EcartAuGroupe`) |
# | | Toute sélection de colonnes fondée sur les données (log du §7.8.3) |

# %% [markdown]
# ### 7.3 Nettoyage
#
# Les étapes portent sur `df_prep`, copie de `df_brut` (§5.1) qui reste intact, et rejouent ce
# que `construire_gold_dataset()` enchaîne (contrôle au §7.8). Première étape : appliquer le
# renommage proposé au §5.10 (snake_case, sans accent, unités suffixées).

# %%
df_prep = df_brut.copy()
_renommage = proposer_renommage(df_prep).query("modifie")
_renames = dict(zip(_renommage["nom_original"], _renommage["nom_propose"], strict=True))
df_prep = df_prep.rename(columns=_renames)
display(
    Markdown(
        f"**Ce qu'il faut retenir.** {entier(len(_renames))} colonne(s) renommée(s)"
        + (": " + ", ".join(f"`{k}` → `{v}`" for k, v in _renames.items()) if _renames else "")
        + ". Les noms du dictionnaire fourni sont repris tels quels par tout le pipeline aval."
    )
)

# %% [markdown]
# #### 7.3.1 Déduplication

# %%
rapport_doublons = analyser_doublons(df_prep, cle_metier="client_id")
n_avant = rapport_doublons["nb_total_lignes"]
df_prep = df_prep.drop_duplicates(keep="first")
# Contrôle bloquant : un client_id répété avec des données divergentes n'a pas de traitement sûr
# par défaut (mise à jour, contacts multiples, erreur de jointure). Même contrôle dans la CLI.
_cles_repetees = int(df_prep["client_id"].duplicated().sum())
assert _cles_repetees == 0, f"{_cles_repetees} client_id répétés : cause à investiguer (§5.5)"

display(
    Markdown(
        f"**Ce qu'il faut retenir.** {entier(n_avant)} → {entier(len(df_prep))} lignes : "
        f"{entier(n_avant - len(df_prep))} copie(s) exacte(s) retirée(s), sans information "
        "nouvelle ; chaque `client_id` est unique. Un compte en double aux données divergentes "
        "arrêterait la préparation : sa règle de résolution se décide une fois la cause connue."
    )
)

# %% [markdown]
# #### 7.3.2 Coercition des numériques

# %%
_COLS_NUM = [
    *["anciennete_mois", "sieges_souscrits", "utilisateurs_actifs", "taux_adoption_pct"],
    *["connexions_30j", "heures_usage_30j", "nb_integrations", "derniere_connexion_jours"],
    *["fonctionnalites_total", "fonctionnalites_utilisees", "csat", "churn"],
    *["tickets_support_90j", "delai_reponse_support_h", "retards_paiement_12m"],
    *["revenu_mensuel_recurrent_eur", "valeur_vie_client_eur"],
]
df_prep, rapport_coercition = coercer_numeriques(df_prep, _COLS_NUM)
display(rapport_coercition)

# %% [markdown]
# **Ce qu'il faut retenir.** Séparateurs, symboles monétaires et espaces insécables sont réparés ;
# l'irrécupérable devient manquant, imputé dans le `Pipeline`.

# %% [markdown]
# #### 7.3.3 Parsing des dates

# %%
df_prep, rapport_dates = parser_dates(df_prep, ["date_souscription"])
display(rapport_dates)

# Vérification : ancienneté recalculée depuis la date d'extraction estimée en §5.8 (pas
# aujourd'hui, sinon le résultat dériverait à chaque relance)
_anc_calculee = ((date_reference - df_prep["date_souscription"]).dt.days / 30.44).round(0)
_coherence = float(((df_prep["anciennete_mois"] - _anc_calculee).abs() <= 3).mean())

# %%
display(
    Markdown(
        "**Ce qu'il faut retenir.** Chaque date est lue avec le premier format de la cascade qui "
        "la reconnaît (dates ambiguës : « premier champ > 12 → JJ/MM/AAAA »). Ancienneté déduite "
        f"de la date et ancienneté déclarée concordent à ±3 mois sur **{pourcentage(_coherence)}**"
        " des comptes : les deux colonnes décrivent la même réalité."
    )
)

# %% [markdown]
# #### 7.3.4 Correction des valeurs impossibles

# %%
rapport_anomalies = detecter_valeurs_impossibles(df_prep, date_reference=date_reference)
display(rapport_anomalies[["regle", "colonne_ou_paire", "nb_lignes_concernees"]])

# Bornes métier connues a priori, sans statistique apprise : elles peuvent précéder le split
df_prep["taux_adoption_pct"] = df_prep["taux_adoption_pct"].clip(0, 100)
df_prep["utilisateurs_actifs"] = df_prep["utilisateurs_actifs"].clip(
    lower=0, upper=df_prep["sieges_souscrits"]
)

# %% [markdown]
# **Ce qu'il faut retenir.** Les bornages (taux ∈ [0, 100], utilisateurs ≤ sièges) sont appliqués
# quel que soit le volume ci-dessus : ce sont des contraintes métier, qui protègent aussi les
# données futures reçues par l'API et le batch.

# %% [markdown]
# #### 7.3.5 Normalisation de casse et d'espaces (catégorielles)
#
# « Finance », « finance » et « FINANCE␣ » deviendraient trois colonnes one-hot. La normalisation
# n'apprend rien. Pour le modèle, elle est faite dans le `Pipeline` (`NormalisationCategorielle`),
# donc à l'identique en entraînement, batch et API ; le gold garde les libellés d'origine.

# %%
_COLS_CAT = ["secteur", "pays", "taille_entreprise", "plan", "jour_souscription"]
_COLS_CAT += ["couleur_theme_interface", "code_datacenter", "groupe_experimentation"]
_rapport_casse = {}
for _col in _COLS_CAT:
    _avant = df_prep[_col].dropna().nunique()
    df_prep[_col] = df_prep[_col].astype(str).str.strip().str.lower().replace("nan", np.nan)
    _rapport_casse[_col] = {"modalités avant": _avant, "modalités après": df_prep[_col].nunique()}
display(pd.DataFrame(_rapport_casse).T.rename_axis("colonne"))

# %% [markdown]
# **Ce qu'il faut retenir.** Toute baisse du nombre de modalités correspond à des doublons
# typographiques fusionnés, sans perte d'information métier.

# %% [markdown]
# ### 7.4 Traitement des valeurs manquantes
#
# **Recalculer avant d'imputer.** `taux_adoption_pct` vaut 100 × `utilisateurs_actifs` /
# `sieges_souscrits` (contrôle au §7.5) : un taux manquant se recalcule exactement au lieu de
# recevoir la médiane d'un client « typique ». `reconstituer_taux_adoption()` le fait ligne par
# ligne, et `ajouter_features_metier()` l'appelle aussi : même traitement en gold, batch et API.
#
# **Imputer le reste, dans le `Pipeline`** : médiane pour les numériques (robuste aux extrêmes),
# constante `"inconnu"` pour les catégorielles, qui garde la trace de la manquance.

# %%
# Contrôle mesuré avant le recalcul : après, les taux reconstitués seraient cohérents par
# construction et gonfleraient artificiellement le bilan
bilan_coherence_adoption = controle_coherence_adoption(df_prep)
_nb_taux_avant = int(df_prep["taux_adoption_pct"].isna().sum())
df_prep, _nb_taux_reconstitues = reconstituer_taux_adoption(df_prep)
_nb_taux_apres = int(df_prep["taux_adoption_pct"].isna().sum())

_df_manquance = df_prep.drop(columns=["client_id", "valeur_vie_client_eur"])
rapport_manquance = analyser_manquance(_df_manquance, cible="churn")
display(
    styler_fr(
        rapport_manquance[["taux_manquants", "mecanisme_propose", "lien_cible_significatif"]],
        {"taux_manquants": lambda v: pourcentage(v, 1)},
    )
)

# %%
_repartition_mecanismes = rapport_manquance["mecanisme_propose"].value_counts()
_cols_mnar = list(rapport_manquance.index[rapport_manquance["mecanisme_propose"] == "MNAR"])
display(
    Markdown(
        f"**Ce qu'il faut retenir.** `taux_adoption_pct` : **{entier(_nb_taux_reconstitues)}** "
        f"valeur(s) recalculée(s) sur {entier(_nb_taux_avant)} manquante(s) ; il en reste "
        f"{entier(_nb_taux_apres)} (sièges nuls ou utilisateurs inconnus). Mécanismes de "
        "manquance : "
        + ", ".join(f"**{entier(n)} {m}**" for m, n in _repartition_mecanismes.items())
        + " ; colonnes dont la manquance est liée au churn : "
        + (", ".join(f"`{c}`" for c in _cols_mnar) or "**aucune**")
        + ". Les indicateurs `*_manquant` du §7.5 restent une hypothèse métier, départagée par "
        "l'importance de permutation au §12.8."
    )
)

# %% [markdown]
# ### 7.5 Feature engineering — ratios métier
#
# **L'intuition.** Dix tickets support n'ont pas le même sens pour un compte de 3 utilisateurs et
# pour un compte de 300. Un ratio rapporte un volume brut à la taille ou à l'âge du compte, et
# fait apparaître le comportement plutôt que la taille.
#
# Toutes les variables ci-dessous sont calculées par `ajouter_features_metier()` **ligne par
# ligne**, sans statistique apprise sur d'autres clients : elles ne peuvent pas faire fuiter
# d'information entre plis. Seule exception, `ecart_csat_secteur`, calculée dans le `Pipeline`.
#
# **Les cinq ratios préconisés.**
#
# | Ratio | Formule | Hypothèse servie (§6.2) | Lecture métier |
# |---|---|---|---|
# | `taux_utilisation_sieges` | `utilisateurs_actifs / sieges_souscrits` | H9 (miroir de H1) | Sièges payés mais non déployés : la facture devient un argument de départ. |
# | `taux_couverture_fonctionnelle` | `fonctionnalites_utilisees / fonctionnalites_total` | H12 (prolonge H4) | Peu de fonctions explorées sur celles du plan : valeur perçue faible. |
# | `intensite_usage_par_utilisateur` | `heures_usage_30j / utilisateurs_actifs` | H3 | Usage réel de chaque utilisateur, indépendamment de la taille du compte. |
# | `arpu_par_siege` | `revenu_mensuel_recurrent_eur / sieges_souscrits` | aucune hypothèse dédiée (famille facturation) | Revenu par siège bas : remise forte ou plan surdimensionné, relation commerciale fragile. |
# | `intensite_support` | `tickets_support_90j / anciennete_mois` | H6 | Frictions rapportées à la durée de la relation. |
#
# Pour chaque ratio, on compare son AUC univariée orientée (probabilité qu'un churner soit du
# côté « risqué » ; 0,50 = aucun signal) à celle de sa variable brute. Mesure descriptive, sans
# modèle : elle dit si la normalisation garde ou renforce le signal.

# %%
df_prep = ajouter_features_metier(df_prep, csat_median_par_secteur=None)
_features_creees = [c for c in df_prep.columns if c not in df_brut.columns]

_RATIOS_PRECONISES = {
    # ratio : (variable brute comparée, sens de risque attendu, hypothèse)
    "taux_utilisation_sieges": ("utilisateurs_actifs", "-", "H9"),
    "taux_couverture_fonctionnelle": ("fonctionnalites_utilisees", "-", "H12"),
    "intensite_usage_par_utilisateur": ("heures_usage_30j", "-", "H3"),
    "arpu_par_siege": ("revenu_mensuel_recurrent_eur", "-", "—"),
    "intensite_support": ("tickets_support_90j", "+", "H6"),
}


def _auc_orientee(colonne: str, sens: str) -> float:
    """AUC univariée de ``colonne`` contre le churn, orientée dans le sens de risque attendu."""
    valides = df_prep[[colonne, "churn"]].dropna()
    auc = roc_auc_score(valides["churn"], valides[colonne])
    return float(auc if sens == "+" else 1 - auc)


tableau_ratios = pd.DataFrame(
    {
        ratio: {
            "hypothèse": hyp,
            "variable brute": brute,
            "auc_ratio": _auc_orientee(ratio, sens),
            "auc_brute": _auc_orientee(brute, sens),
            "manquants": df_prep[ratio].isna().mean(),
        }
        for ratio, (brute, sens, hyp) in _RATIOS_PRECONISES.items()
    }
).T.rename_axis("ratio")
display(
    styler_fr(
        tableau_ratios,
        {"auc_ratio": lambda v: nombre(v, 3), "auc_brute": lambda v: nombre(v, 3)}
        | {"manquants": pourcentage},
    )
)

# %%
_gagnants = tableau_ratios.index[tableau_ratios["auc_ratio"] > tableau_ratios["auc_brute"]]
_sans_signal = tableau_ratios.index[tableau_ratios["auc_ratio"] < 0.55]
display(
    Markdown(
        f"**Ce qu'il faut retenir.** {entier(len(_features_creees))} variables créées. "
        f"{entier(len(_gagnants))} des cinq ratios portent plus de signal que leur variable brute ("
        + (", ".join(f"`{c}`" for c in _gagnants) or "aucun")
        + ") ; pour les autres, la variable brute reste plus discriminante et les deux sont "
        "gardées. Signal faible (AUC < 0,55) : "
        + (", ".join(f"`{c}`" for c in _sans_signal) or "aucun")
        + ". Le poids réel, en combinaison, se mesure au §12.8 ; les manquants (dénominateur nul, "
        "composante absente) sont imputés dans le `Pipeline`."
    )
)

# %% [markdown]
# **Contrôle de cohérence de l'adoption.** `taux_utilisation_sieges` × 100 devrait reproduire
# `taux_adoption_pct` ; si oui, les deux variables sont jumelles (H1 et H9) et le recalcul du §7.4
# est légitime. `controle_coherence_adoption()` mesure l'écart sur les lignes renseignées **avant**
# recalcul (incohérence au-delà de `config.SEUIL_INCOHERENCE_ADOPTION_PTS` point).

# %%
display(styler_fr(bilan_coherence_adoption, precision=4))
_bilan = bilan_coherence_adoption.iloc[0]
_correlation_jumelles = df_prep["taux_utilisation_sieges"].corr(df_prep["taux_adoption_pct"])
display(
    Markdown(
        f"**Ce qu'il faut retenir.** {entier(_bilan['n_incoherences'])} incohérence(s) sur "
        f"{entier(_bilan['n_comparables'])} lignes (écart maximal "
        f"{nombre(_bilan['ecart_max_pts'], 2)} point), corrélation "
        f"{nombre(_correlation_jumelles, 4)} : c'est la même information à un facteur 100 près. "
        "Les deux sont gardées (la régularisation absorbe la redondance, les arbres n'en "
        "souffrent pas), mais leurs importances se partagent et se lisent **ensemble** au §12.8."
    )
)

# %% [markdown]
# **Limite d'`intensite_support`.** Le numérateur couvre une fenêtre de 90 jours, le dénominateur
# toute la vie du compte : deux échelles de temps différentes. Un compte de deux mois avec trois
# tickets obtient un ratio bien plus élevé qu'un compte de quatre ans avec les mêmes trois tickets,
# alors que la friction récente est identique. Le ratio est donc lié à l'ancienneté **par
# construction** ; la cellule suivante le mesure.

# %%
_rho_ratio, _rho_brut = (
    spearmanr(df_prep[c], df_prep["anciennete_mois"], nan_policy="omit").statistic
    for c in ["intensite_support", "tickets_support_90j"]
)
display(
    Markdown(
        f"**Ce qu'il faut retenir.** Corrélation de rang avec l'ancienneté : "
        f"{nombre(_rho_ratio, 2)} pour `intensite_support`, contre {nombre(_rho_brut, 2)} pour "
        "les tickets bruts : une partie de son signal est celui de l'ancienneté (H14). Elle est "
        "gardée car elle pose une question métier distincte (le support pèse-t-il lourd **pour "
        "l'âge du compte** ?), mais son importance au §12.8 se lit à côté de l'ancienneté. La "
        "version rigoureuse exigerait l'historique complet des tickets, absent des données."
    )
)

# %% [markdown]
# **Les autres variables créées, en une ligne chacune.**
#
# | Variable | Formule | Justification |
# |---|---|---|
# | `surdimensionnement` | `max(0, sieges − utilisateurs)` | Sièges vides payés, en nombre : ce que voit l'acheteur sur la facture. |
# | `connexions_par_utilisateur` | `connexions_30j / utilisateurs_actifs` | H10 : régularité d'usage, indépendante de la taille du compte. |
# | `recence_normalisee` | `derniere_connexion_jours / (anciennete_mois × 30)` | H5 rapportée à l'âge du compte : décrochage relatif. |
# | `compte_dormant` | `derniere_connexion_jours > 30` | H5 en signal binaire : un mois ≈ un cycle de reporting client. |
# | `pression_support` | `tickets_support_90j / utilisateurs_actifs` | H6 rapportée à la taille du compte, sur la même fenêtre de 90 jours. |
# | `ecart_csat_secteur` | `csat − médiane du secteur` (`EcartAuGroupe`, fitté par pli) | H7 relative : moins satisfait que ses pairs, donc plus exposé à la concurrence. |
# | `tranche_anciennete` | `anciennete_mois` en 3 tranches (0-2, 3-12, > 12 mois) | H14 : prise en main, installation et maturité ont des risques différents. |
# | `tranche_integrations` | `nb_integrations` en 4 tranches (0, 1-3, 4-8, > 8) | H11 : chaque intégration renchérit la migration vers un concurrent. |
# | `csat_manquant`, `heures_usage_30j_manquant`, `delai_reponse_support_h_manquant` | `colonne.isna()` | Hypothèse « silence = désengagement », non confirmée au §5.9, gardée à coût nul. |

# %% [markdown]
# ### 7.6 Jointure catalogue et variables commerciales
#
# Le catalogue (référentiel statique, donc sans fuite) donne le prix par siège, d'où
# `remise_consentie` (1 − MRR / (prix × sièges) : forte remise = négociation tendue) et
# `adequation_plan` (usage < 50 % des sièges : surdimensionné ; ≥ 90 % : montée en gamme).

# %%
# catalogue_brut est chargé et décrit en §5.1 ; on en repart sans relire le fichier
catalogue, _ = coercer_numeriques(
    catalogue_brut.copy(),
    [
        *["prix_mensuel_par_siege_eur", "fonctionnalites_incluses"],
        "sla_reponse_h",
        "quota_stockage_go",
    ],
)
df_prep = joindre_catalogue(df_prep, catalogue)
_sans_prix = int(df_prep["prix_mensuel_par_siege_eur"].isna().sum())
_sans_plan = int(df_prep["plan"].isna().sum())
display(
    Markdown(
        "**Ce qu'il faut retenir.** "
        + (
            "Tous les clients ont un prix catalogue : la normalisation de `plan` suffit."
            if _sans_prix == 0
            else f"{entier(_sans_prix)} client(s) sans prix catalogue (dont {entier(_sans_plan)} "
            "sans plan) : leur `remise_consentie` sera imputée dans le `Pipeline`."
        )
    )
)

# %% [markdown]
# ### 7.7 Enrichissement externe et gouvernance
#
# Deux référentiels externes (secteur et pays) sont ajoutés, **construits et documentés comme
# tels** : ce sont des ordres de grandeur plausibles tirés de rapports publics, pas des sources
# réelles. Leurs fiches de gouvernance indiquent la source de production qui les remplacera ; les
# colonnes concernées sont tracées dans les métadonnées du gold (§7.8.1).

# %%
fiches = pd.DataFrame(FICHES_SOURCES).T.set_index("nom").T.rename_axis("champ")
display(fiches)
display(REFERENTIEL_SECTORIEL.drop(index=["_repli_"]))
display(REFERENTIEL_PAYS.drop(index=["_repli_"]))

# %% [markdown]
# **Ce qu'il faut retenir.** Chaque source a une licence, une fraîcheur, un coût et un plan B
# identifiés : remplacer la simulation par la source réelle ne demande aucun changement de code,
# seulement un réentraînement.

# %%
df_prep = enrichir_par_pays(enrichir_par_secteur(df_prep))

# Couverture de la jointure sectorielle : quels libellés tombent sur la valeur de repli ?
_cles_secteur = {str(k).casefold() for k in REFERENTIEL_SECTORIEL.index} | set(ALIAS_SECTEURS)
_secteur_cle = df_prep["secteur"].str.strip().str.casefold()
_sur_repli = _secteur_cle.notna() & ~_secteur_cle.isin(_cles_secteur)
_libelles_repli = _secteur_cle[_sur_repli].value_counts()
_sans_secteur = int(_secteur_cle.isna().sum())

# Redondance : chaque colonne ajoutée est-elle une fonction du seul libellé secteur / pays ?
# Nombre maximal de valeurs distinctes par libellé (1 = fonction déterministe).
_constante = df_prep["taux_adoption_pct"] - df_prep["ecart_adoption_secteur"]
_cle_par_colonne = {"taux_churn_median_saas_pct": "secteur", "dynamique_croissance": "secteur"}
_cle_par_colonne |= {c: "pays" for c in ["zone_reglementaire", "langue_support_fr"]}
_cle_par_colonne |= {"decalage_horaire_paris_h": "pays"}
_valeurs_par_libelle = {
    col: df_prep.groupby(cle, dropna=False)[col].nunique(dropna=False).max()
    for col, cle in _cle_par_colonne.items()
}
# ecart_adoption_secteur = taux_adoption_pct − constante du secteur
_valeurs_par_libelle["ecart_adoption_secteur"] = (
    _constante.round(9).groupby(df_prep["secteur"], dropna=False).nunique().max()
)
_redondance = pd.Series(_valeurs_par_libelle, name="valeurs distinctes max par libellé")
display(_redondance)
# L'affirmation de redondance qui suit repose sur ce contrôle
assert (_redondance == 1).all(), _redondance

# %%
display(
    Markdown(
        "**Ce qu'il faut retenir.** Ces colonnes n'apportent **aucune information nouvelle** : "
        "chacune est fixée par le secteur ou le pays, déjà dans le modèle. Elles sont gardées "
        "car elles démontrent la chaîne complète (jointure, repli, traçabilité, contrat de "
        "l'API) que les sources réelles emprunteront, sans injecter d'information inventée.\n\n"
        "**Couverture.** La jointure ignore casse et espaces et rattache les abréviations ("
        + ", ".join(f"« {a} » → « {c} »" for a, c in ALIAS_SECTEURS.items())
        + f"). {entier(int(_sur_repli.sum()))} compte(s) ({pourcentage(float(_sur_repli.mean()))})"
        " restent sur la valeur de repli : "
        + (", ".join(f"« {s} » ({entier(n)})" for s, n in _libelles_repli.items()) or "aucun")
        + f", plus {entier(_sans_secteur)} comptes sans secteur. Le repli est une valeur "
        "générique : une limite de la simulation, pas une information."
    )
)

# %% [markdown]
# ### 7.8 Jeu de données gold — schéma, volumétrie et versioning
#
# Le gold, écrit par `construire_gold_dataset()`, est l'unique point d'entrée des §8 à §13 : il
# contient tout ce qui se calcule avant le split, les transformations apprises restant dans le
# `Pipeline`.

# %%
chemin_gold = construire_gold_dataset(forcer=False)  # idempotent : ne réécrit pas si présent
df_gold = pd.read_parquet(chemin_gold)
with (config.DONNEES_GOLD / "gold_metadata.json").open(encoding="utf-8") as f:
    meta = json.load(f)

_origines = {c: "brute" for c in df_brut.columns}
_origines |= {c: "catalogue" for c in [*catalogue.columns, "remise_consentie", "adequation_plan"]}
_origines |= {c: "enrichissement_simule" for c in meta["colonnes_enrichissement_simule"]}
_origines["plan"] = "brute"
schema = pd.DataFrame(
    {
        "dtype": df_gold.dtypes.astype(str),
        "manquants": df_gold.isna().mean(),
        "origine": [_origines.get(c, "feature_metier") for c in df_gold.columns],
    }
).rename_axis("colonne")
display(styler_fr(schema, {"manquants": pourcentage}))

# %%
display(
    Markdown(
        f"**Ce qu'il faut retenir.** Gold : {entier(df_gold.shape[0])} lignes × "
        f"{entier(df_gold.shape[1])} colonnes, dont "
        + ", ".join(f"{entier(n)} {o}" for o, n in schema["origine"].value_counts().items())
        + ". La colonne `origine` trace la provenance de chaque variable et repère celles à "
        "remplacer par les sources de production ; seules les variables à manquants gardent un "
        "taux non nul, l'imputation étant différée dans le `Pipeline`."
    )
)

# %% [markdown]
# **Contrôle : démonstration contre gold.** Seul le gold alimente les §8-9 : `df_prep` et le gold
# doivent coïncider ligne à ligne, à casse et espaces près.


# %%
def _forme_comparable(serie: pd.Series) -> pd.Series:
    """Valeurs comparables entre chemins : numériques en flottant, texte sans casse ni espaces."""
    if pd.api.types.is_numeric_dtype(serie) and not pd.api.types.is_bool_dtype(serie):
        serie = serie.astype(float).round(9)
    return serie.astype("string").str.strip().str.lower().fillna("<manquant>")


_communes = [c for c in df_gold.columns if c in df_prep.columns]
_prep_cmp, _gold_cmp = df_prep.reset_index(drop=True), df_gold.reset_index(drop=True)
_meme_volume = len(_prep_cmp) == len(_gold_cmp)
_ecarts = {
    c: int((_forme_comparable(_prep_cmp[c]) != _forme_comparable(_gold_cmp[c])).sum())
    for c in (_communes if _meme_volume else [])
}
_divergentes = {c: n for c, n in _ecarts.items() if n > 0}
_hors_gold = sorted(set(df_prep.columns) - set(df_gold.columns))
_branches_pre = [nom for nom, _, _ in construire_preprocesseur(df_gold).transformers]
_verdict = (
    "les deux chemins sont **identiques** : la démonstration décrit fidèlement le gold"
    if _meme_volume and not _divergentes
    else "**écart détecté** : "
    + (", ".join(f"`{c}` ({entier(n)})" for c, n in _divergentes.items()) or "volumes différents")
)
display(
    Markdown(
        f"**Ce qu'il faut retenir.** Sur {entier(len(_communes))} colonnes communes, {_verdict}. "
        "Colonnes de `df_prep` absentes du gold : "
        + (", ".join(f"`{c}`" for c in _hors_gold) or "aucune")
        + ". `ecart_csat_secteur` est "
        + ("absente du gold" if "ecart_csat_secteur" not in df_gold.columns else "**dans le gold**")
        + (
            " et produite par une branche du préprocesseur, fittée dans chaque pli."
            if "ecart_csat_secteur" in _branches_pre
            else " mais **absente du préprocesseur**."
        )
    )
)

# %% [markdown]
# #### 7.8.1 Métadonnées et versioning

# %%
display(
    pd.Series(
        {
            "date de construction": meta["date_construction"],
            "version du code (git)": meta["version_code"],
            **{f"SHA-256 {s}": h[:16] + "…" for s, h in meta["sources"].items()},
            "lignes × colonnes": f"{entier(meta['nb_lignes'])} × {entier(meta['nb_colonnes'])}",
            "colonnes simulées": ", ".join(meta["colonnes_enrichissement_simule"]),
            "note de gouvernance": meta["note_gouvernance"],
        },
        name="valeur",
    ).to_frame()
)

# %% [markdown]
# **Ce qu'il faut retenir.** Empreintes des sources et version git relient chaque gold aux
# fichiers et au code qui l'ont produit : il se reconstruit à l'identique.

# %% [markdown]
# #### 7.8.2 Versioning DVC
#
# **L'intuition.** git suit le code, mais un gros fichier de données n'a rien à y faire. DVC
# (*Data Version Control*) ne met dans git qu'un pointeur, `data/gold/gold_dataset.parquet.dvc`
# (empreinte MD5 et taille du Parquet) : toute modification du gold change l'empreinte. Après
# chaque reconstruction : `uv run dvc add data/gold/gold_dataset.parquet`, puis commit du pointeur.
#
# **Choix assumé : aucun stockage distant DVC dans ce projet.** Le gold se reconstruit à
# l'identique depuis les données brutes figées et le code versionné, et il est livré dans
# l'archive du projet. Le stockage distant sera mis en place en phase de production
# (`dvc remote add` vers un stockage objet, `dvc push` à chaque réentraînement, §13.10), quand
# plusieurs personnes et machines partageront les versions du gold.
#
# La cellule compare les empreintes du pointeur commité et du gold du disque, sans `dvc status` :
# la règle `/data/*` du `.gitignore` lui fait croire `data/` ignoré et lui masque le pointeur.

# %%
_pointeur_dvc = config.DONNEES_GOLD / "gold_dataset.parquet.dvc"
_texte_pointeur = _pointeur_dvc.read_text() if _pointeur_dvc.exists() else ""
# Le pointeur est un court YAML : la ligne « - md5: … » porte l'empreinte calculée par dvc add
_md5_pointeur = next(
    (x.split(":")[1].strip() for x in _texte_pointeur.splitlines() if "md5:" in x), None
)
_md5_disque = hashlib.md5((config.DONNEES_GOLD / "gold_dataset.parquet").read_bytes()).hexdigest()
_commande = ["uv", "run", "dvc", "remote", "list"]
_remotes_dvc = subprocess.run(
    _commande, capture_output=True, text=True, cwd=config.RACINE
).stdout.split()
print(f"Empreinte MD5 — pointeur commité : {_md5_pointeur} ; gold sur disque : {_md5_disque}")

if _md5_pointeur is None:
    _etat_dvc = "**le gold n'est pas versionné** (aucun pointeur)."
elif _md5_pointeur == _md5_disque:
    _etat_dvc = "le pointeur commité **correspond** au gold présent sur le disque."
else:
    _etat_dvc = "**le pointeur est périmé** : relancer `dvc add`, committer le pointeur."
_distant = ", ".join(f"`{r}`" for r in _remotes_dvc) or "**aucun, par choix** (local et archive)"

display(
    Markdown(
        f"**Ce qu'il faut retenir.** État DVC du gold : {_etat_dvc} Stockage distant : {_distant}. "
        "Le stockage distant et l'appel à `dvc add` puis `dvc push` par le flow de réentraînement "
        "relèvent de la mise en production (§13.10)."
    )
)

# %% [markdown]
# #### 7.8.3 Variables asymétriques — log1p testé puis écarté
#
# **Hypothèse (§6.2).** Comptages, durées et montants ont une longue queue droite ; pour un modèle
# linéaire, un log devrait réduire le poids des extrêmes. `LogAsymetrique` applique
# `signe(x) × log(1 + |x|)` aux colonnes dont l'asymétrie, mesurée sur le train de chaque pli,
# dépasse `config.SEUIL_ASYMETRIE_LOG`. **Règle fixée avant le résultat** : les trois familles du
# §9 sont évaluées avec et sans log sur les 15 mêmes plis ; le log n'est retenu que si la PR-AUC
# de la régression logistique progresse avec un Wilcoxon apparié significatif (p < 0,05).

# %%
_X_illustration = df_gold.drop(columns=config.COLONNES_INTERDITES, errors="ignore")
_X_modele = ajouter_features_metier(_X_illustration).drop(columns=["churn"], errors="ignore")
_y_modele = df_gold["churn"].astype(int)
_FAMILLES = ["Baseline — régression logistique", "Forêt aléatoire", "LightGBM"]


def _comparer_avec_sans_log() -> pd.DataFrame:
    modeles = construire_modeles(_X_modele)
    pre_log = construire_preprocesseur(_X_modele, log_asymetrique=True)
    lignes = []
    for nom in _FAMILLES:
        variantes = {"sans": modeles[nom], "avec": clone(modeles[nom]).set_params(pre=pre_log)}
        plis = {}
        for cle, modele in variantes.items():
            res = evaluer_modele(modele, _X_modele, _y_modele, protocole_validation())
            plis[cle] = np.array([res[f"pr_auc_pli_{i:02d}"] for i in range(15)])
        ecarts = plis["avec"] - plis["sans"]
        lignes.append(
            {
                "modèle": nom,
                "pr_auc_sans_log": plis["sans"].mean(),
                "pr_auc_avec_log": plis["avec"].mean(),
                "écart": ecarts.mean(),
                "plis_gagnés_par_le_log": int((ecarts > 0).sum()),
                "p_wilcoxon": 1.0 if np.allclose(ecarts, 0) else wilcoxon(ecarts).pvalue,
            }
        )
    return pd.DataFrame(lignes).set_index("modèle")


tableau_ablation_log, _ = charger_ou_calculer("ablation_log_7.parquet", _comparer_avec_sans_log)
display(
    styler_fr(
        tableau_ablation_log,
        {"écart": lambda v: nombre(v, 3, signe=True), "p_wilcoxon": lambda v: nombre(v, 4)},
        precision=3,
    )
)

# %%
_lr = tableau_ablation_log.loc["Baseline — régression logistique"]
_log_retenu = bool(_lr["écart"] > 0 and _lr["p_wilcoxon"] < 0.05)
_arbres = tableau_ablation_log.drop(index="Baseline — régression logistique")
display(
    Markdown(
        f"**Ce qu'il faut retenir.** Pour la régression logistique, le log fait passer la PR-AUC "
        f"de {nombre(_lr['pr_auc_sans_log'], 3)} à {nombre(_lr['pr_auc_avec_log'], 3)} (écart "
        f"{nombre(_lr['écart'], 3, signe=True)}, {entier(_lr['plis_gagnés_par_le_log'])} pli(s) "
        f"gagné(s) sur 15, Wilcoxon p = {nombre(_lr['p_wilcoxon'], 4)}). Pour les arbres, l'écart "
        f"ne dépasse pas {nombre(_arbres['écart'].abs().max(), 3)}, comme attendu d'une "
        "transformation qui conserve l'ordre des valeurs.\n\n"
        + (
            "**Décision : le log est retenu** dans le `Pipeline`."
            if _log_retenu
            else "**Décision : le log est écarté**, selon la règle fixée avant le test : "
            "l'hypothèse de l'EDA est réfutée par la validation. Explication la plus probable : "
            "les valeurs extrêmes (inactivité longue, retards, support saturé) portent le signal "
            "de churn, et le log les rapproche du reste. Le `Pipeline` reste **imputation → "
            "standardisation** ; `LogAsymetrique` reste dans le code pour que l'expérience soit "
            "reproductible."
        )
    )
)

# %% [markdown]
# ### 7.9 Alternatives écartées
#
# | Alternative | Raison du rejet |
# |---|---|
# | Imputation KNN | Bénéfice non démontré face à la médiane pour des taux de manquance < 15 %, et `Pipeline` alourdi (les distances exigent standardisation et encodage préalables). Piste à évaluer sur les plis du §9. |
# | PCA avant modélisation | Détruit l'interprétabilité alors que chaque score doit être expliqué (§12.8) ; le nombre de variables ne justifie pas de réduction. |
# | Target encoding (encodage par la cible) | Fuite directe hors `Pipeline` ; possible avec `TargetEncoder` fitté par pli, mais le one-hot reste plus lisible pour un nombre de modalités modeste. |
# | Suppression des leurres avant modélisation | Une preuve chiffrée (§12.9) vaut mieux qu'une suppression aveugle. |
# | Log1p, RobustScaler ou plafonnement (winsorisation) | Log testé au §7.8.3 ; RobustScaler est affine, donc sans effet sur la forme ; le plafonnement écrase les extrêmes, qui portent le signal. |
# | Standardisation ou SMOTE avant le split | Fuite : les statistiques ou exemples synthétiques du train contaminent le test. Tout est dans le `Pipeline`, à l'intérieur des plis. |

# %% [markdown]
# > ### 📋 Journal de bord — Préparation des données
# >
# > **Décisions retenues** — Nettoyage par règles sans statistique apprise. `taux_adoption_pct`
# > recalculé plutôt qu'imputé, justifié par le contrôle de cohérence. Cinq ratios reliés aux
# > hypothèses du §6.2 et comparés à leur variable brute ; `intensite_support` gardée malgré son
# > lien mesuré avec l'ancienneté. Imputation dans le `Pipeline`. Enrichissement simulé gardé en
# > connaissance de cause (redondant, contrôlé). Gold versionné par DVC et métadonnées, sans
# > stockage distant (choix assumé, prévu en production).
# >
# > **Alternatives écartées** — KNN *(bénéfice non démontré)*, PCA *(explicabilité)*, target
# > encoding et standardisation ou SMOTE avant split *(fuite)*, RobustScaler et plafonnement
# > *(sans effet ou destructeur de signal)*.
# >
# > **Difficultés rencontrées** — log1p, suggéré par les asymétries de l'EDA, dégradait la
# > régression logistique (§7.8.3) : écarté selon une règle fixée avant l'essai et verrouillé par
# > un test. Le premier test anti-fuite passait toujours (données uniformes) ; réécrit sur données
# > bimodales, il a révélé un compteur de replis toujours nul (comptage après `fillna`).
# >
# > **Impact sur la suite** — Les §8-9 reçoivent le gold ; le `ColumnTransformer` y écarte les
# > colonnes interdites. Au §12.8, les paires jumelles (`taux_adoption_pct` /
# > `taux_utilisation_sieges`, `intensite_support` / ancienneté) se lisent ensemble.
