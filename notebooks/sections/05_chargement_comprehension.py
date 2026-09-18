# %% [markdown]
# ## 5. Chargement et compréhension des données (C3)
#
# Cette section charge les trois sources, vérifie leur intégrité physique (hash, volumétrie,
# encodage), analyse la qualité des données brutes (doublons, types, dates, valeurs impossibles,
# mécanismes de manquance) et pose la convention de nommage.
# Elle constitue la traçabilité exigée par C3 avant toute transformation — chaque anomalie
# détectée ici est tracée jusqu'à la section §7 où elle est corrigée.

# %%
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

from churn_saas import config
from churn_saas.data.loaders import charger_brut
from churn_saas.data.quality import (
    analyser_doublons,
    analyser_manquance,
    coercer_numeriques,
    controler_source,
    detecter_valeurs_impossibles,
    parser_dates,
    profil_compact,
    proposer_renommage,
)

# %% [markdown]
# ### 5.1 Chargement des sources et contrôle d'intégrité
#
# Trois fichiers constituent le socle de données. On vérifie existence, taille, encodage
# et empreinte SHA-256 avant tout traitement. Cette empreinte est reproduite dans le ZIP
# de livraison pour certifier l'absence d'altération entre l'entraînement et la livraison.

# %%
_SOURCES = {
    "churn_saas_complet.csv": config.DONNEES_BRUTES / "churn_saas_complet.csv",
    "churn_saas_echantillon.csv": config.DONNEES_BRUTES / "churn_saas_echantillon.csv",
    "catalogue_plans.csv": config.DONNEES_BRUTES / "catalogue_plans.csv",
}

controles = {nom: controler_source(chemin) for nom, chemin in _SOURCES.items()}

df_controles = pd.DataFrame(
    [
        {
            "Fichier": nom,
            "Lignes (en-tête inclus)": c.get("nb_lignes", "—"),
            "Colonnes": c.get("nb_colonnes", "—"),
            "Taille (Ko)": round(c["taille_octets"] / 1024, 1) if c.get("existe") else "—",
            "Encodage": c.get("encodage", "—"),
            "SHA-256 (8 car.)": c["sha256"][:8] + "…" if c.get("sha256") else "—",
            "Date modification": c.get("date_modification", "—"),
        }
        for nom, c in controles.items()
    ]
).set_index("Fichier")

display(df_controles)

ct = controles["churn_saas_complet.csv"]
n_obs = ct["nb_lignes"] - 1  # soustrait l'en-tête

display(
    Markdown(
        f"**Ce qu'il faut retenir.** Les 3 sources sont présentes et accessibles. "
        f"Le dataset principal compte **{n_obs} observations** ({ct['nb_colonnes']} colonnes), "
        f"encodé en `{ct['encodage']}`. "
        f"Les empreintes SHA-256 sont figées ici et vérifiées à la livraison pour certifier "
        f"que les données soumises au jury sont identiques à celles utilisées pour l'entraînement."
    )
)

# %%
# Chargement en préservant tous les formats d'origine (dtype=str, pas de conversion implicite)
df_brut = charger_brut("churn_saas_complet")

# Normalise les marqueurs textuels de valeur manquante en NaN.
# Sans cette étape, isna() retourne False sur les chaînes vides et les marqueurs comme "n/a",
# ce qui fausse les taux de manquants dans profil_compact() et analyser_manquance().
_MARQUEURS_NA = [
    "", "n/a", "na", "nan", "null", "none", "#n/a", "-", "nd", "nr", "inconnu"
]
df_brut = df_brut.replace({m: np.nan for m in _MARQUEURS_NA})

print(f"Dataset principal chargé : {df_brut.shape[0]} lignes × {df_brut.shape[1]} colonnes")

# %% [markdown]
# ### 5.2 Dictionnaire de données
#
# Le tableau recense les 29 colonnes, groupées par famille fonctionnelle, avec leur type attendu
# et leur rôle dans le pipeline. La colonne « Statut » distingue les features candidates,
# les identifiants, les leurres annoncés et la fuite temporelle — item C3 (pertinence des données).

# %%
_DICTIONNAIRE = [
    # --- Famille Client ---
    ("client_id", "Client", "texte", "Identifiant unique du compte", "identifiant — exclu"),
    ("date_souscription", "Client", "date", "Date de début de contrat", "explicative"),
    ("jour_souscription", "Client", "catégorielle", "Jour de semaine à la souscription", "explicative"),
    ("secteur", "Client", "catégorielle", "Secteur d'activité", "explicative"),
    ("pays", "Client", "catégorielle", "Pays du siège social", "explicative"),
    ("taille_entreprise", "Client", "catégorielle", "Taille (effectif)", "explicative"),
    ("plan", "Client", "catégorielle", "Formule d'abonnement souscrite", "explicative"),
    # --- Famille Usage produit ---
    ("anciennete_mois", "Usage produit", "numérique", "Ancienneté du contrat (mois)", "explicative"),
    ("sieges_souscrits", "Usage produit", "numérique", "Nombre de licences souscrites", "explicative"),
    (
        "utilisateurs_actifs",
        "Usage produit",
        "numérique",
        "Utilisateurs actifs (30 j)",
        "explicative",
    ),
    (
        "taux_adoption_pct",
        "Usage produit",
        "numérique [0-100]",
        "Part des sièges utilisés (%)",
        "explicative",
    ),
    (
        "connexions_30j",
        "Usage produit",
        "numérique",
        "Nombre de connexions (30 j)",
        "explicative",
    ),
    (
        "heures_usage_30j",
        "Usage produit",
        "numérique",
        "Heures d'utilisation (30 j)",
        "explicative",
    ),
    (
        "fonctionnalites_total",
        "Usage produit",
        "numérique",
        "Fonctionnalités disponibles dans le plan",
        "explicative",
    ),
    (
        "fonctionnalites_utilisees",
        "Usage produit",
        "numérique",
        "Fonctionnalités effectivement utilisées",
        "explicative",
    ),
    (
        "nb_integrations",
        "Usage produit",
        "numérique",
        "Intégrations tierces actives",
        "explicative",
    ),
    (
        "derniere_connexion_jours",
        "Usage produit",
        "numérique",
        "Jours depuis la dernière connexion",
        "explicative",
    ),
    # --- Famille Support ---
    (
        "tickets_support_90j",
        "Support",
        "numérique",
        "Tickets ouverts sur 90 jours",
        "explicative",
    ),
    (
        "delai_reponse_support_h",
        "Support",
        "numérique",
        "Délai moyen de réponse support (h)",
        "explicative",
    ),
    (
        "csat",
        "Support",
        "numérique [1-5]",
        "Score de satisfaction client",
        "explicative — MNAR probable",
    ),
    # --- Famille Facturation ---
    (
        "retards_paiement_12m",
        "Facturation",
        "numérique",
        "Retards de paiement sur 12 mois",
        "explicative",
    ),
    (
        "revenu_mensuel_recurrent_eur",
        "Facturation",
        "numérique",
        "MRR du compte (€)",
        "explicative",
    ),
    # --- Leurres annoncés ---
    (
        "couleur_theme_interface",
        "Leurres",
        "catégorielle",
        "Couleur du thème UI",
        "leurre — cosmétique",
    ),
    (
        "code_datacenter",
        "Leurres",
        "catégorielle",
        "Identifiant du datacenter",
        "leurre — localisation technique",
    ),
    (
        "groupe_experimentation",
        "Leurres",
        "catégorielle",
        "Groupe d'A/B test",
        "leurre — biais de sélection possible",
    ),
    (
        "commentaire_csm",
        "Leurres",
        "texte libre",
        "Commentaire libre du CSM",
        "leurre — non structuré",
    ),
    # --- Fuite temporelle ---
    (
        "sante_compte_fin_periode",
        "Fuite",
        "numérique",
        "Score de santé calculé en FIN de période",
        "⚠ fuite temporelle — exclue",
    ),
    # --- Cibles ---
    (
        "valeur_vie_client_eur",
        "Cible",
        "numérique",
        "Valeur vie client estimée (€)",
        "cible régression — exclue des features",
    ),
    ("churn", "Cible", "binaire (0/1)", "Résiliation observée (1 = churn)", "cible principale"),
]

df_dico = pd.DataFrame(
    _DICTIONNAIRE, columns=["Colonne", "Famille", "Type attendu", "Description", "Statut"]
)
display(df_dico.set_index("Colonne"))

nb_explicatives = df_dico[df_dico["Statut"].str.startswith("explicative")].shape[0]
nb_leurres = df_dico[df_dico["Famille"] == "Leurres"].shape[0]

display(
    Markdown(
        f"**Ce qu'il faut retenir.** Les {len(_DICTIONNAIRE)} colonnes sont réparties en "
        f"6 familles fonctionnelles. **{nb_explicatives} variables** sont candidates comme "
        f"features explicatives. **{nb_leurres} leurres** sont annoncés : leur inutilité sera "
        f"prouvée par permutation importance (§9) — on ne les supprime pas a priori. "
        f"La variable `sante_compte_fin_periode` est une fuite temporelle pure (score calculé "
        f"après observation du churn) ; elle est inscrite dans `config.COLONNES_INTERDITES` "
        f"et n'atteindra jamais le modèle."
    )
)

# %% [markdown]
# ### 5.3 Profil compact du dataset principal
#
# `profil_compact()` produit une ligne par colonne : taux de manquants, cardinalité, valeurs
# dominantes et un flag signalant les colonnes numériques stockées en texte.
# C'est la radiographie initiale — elle oriente tous les diagnostics suivants.

# %%
profil = profil_compact(df_brut)
display(profil)

nb_avec_manquants = int((profil["taux_manquants"] > 0).sum())
nb_num_texte_flag = int(profil["numerique_en_texte"].sum())

display(
    Markdown(
        f"**Ce qu'il faut retenir.** Sur {df_brut.shape[0]} observations "
        f"et {df_brut.shape[1]} colonnes : "
        f"**{nb_avec_manquants} colonne(s)** présentent des valeurs manquantes et "
        f"**{nb_num_texte_flag} colonne(s)** sont des numériques stockées en texte "
        f"(flag `numerique_en_texte = True`). "
        f"Ces deux familles d'anomalies sont diagnostiquées dans les blocs §5.5 et §5.8 "
        f"puis corrigées en §7."
    )
)

# %% [markdown]
# ### 5.4 Doublons — exacts vs clé métier
#
# Deux types de doublons ont des traitements distincts. Les doublons exacts (toutes colonnes
# identiques) sont supprimables sans risque. Les doublons sur clé métier seulement (même
# `client_id`, données différentes) peuvent représenter des mises à jour historiques — ils
# nécessitent une investigation avant décision.

# %%
rapport_doublons = analyser_doublons(df_brut, cle_metier="client_id")

nb_exact = rapport_doublons["nb_doublons_exacts"]
nb_cle_non_exact = rapport_doublons["nb_doublons_cle_metier_non_exacts"]
n_total = rapport_doublons["nb_total_lignes"]

df_resume_doublons = pd.DataFrame(
    [
        {
            "Type de doublon": "Exacts (toutes colonnes identiques)",
            "Lignes concernées": nb_exact,
            "Décision": "drop_duplicates(keep='first')",
            "Justification": "Redondance pure — aucune information supplémentaire",
        },
        {
            "Type de doublon": "Clé métier seulement (client_id différent dans les données)",
            "Lignes concernées": nb_cle_non_exact,
            "Décision": "Conserver la ligne la plus récente (date_souscription max)",
            "Justification": "Peut indiquer une mise à jour du compte — ne pas supprimer sans vérif.",
        },
    ]
).set_index("Type de doublon")

display(df_resume_doublons)

if nb_exact > 0:
    display(
        Markdown(f"*Exemples de doublons exacts ({nb_exact} lignes) :*")
    )
    display(rapport_doublons["exemples_doublons_exacts"])

display(
    Markdown(
        f"**Ce qu'il faut retenir.** Sur {n_total} lignes : "
        f"**{nb_exact} doublon(s) exact(s)** (suppression sécurisée) et "
        f"**{nb_cle_non_exact} doublon(s)** sur `client_id` avec données divergentes "
        f"(conservation de la ligne la plus récente). "
        f"Le traitement différencié est justifié : un doublon partiellement distinct "
        f"peut être la version la plus à jour d'un compte — le supprimer aveuglément "
        f"introduirait un biais. Application en §7."
    )
)

# %% [markdown]
# ### 5.5 Colonnes numériques stockées en texte
#
# `profil_compact()` signale les colonnes `object` convertibles à plus de 80 % en numérique.
# `coercer_numeriques()` gère les séparateurs décimaux mixtes (virgule/point), les espaces
# insécables, les symboles (€, %) et les marqueurs textuels de valeur manquante.

# %%
cols_num_texte = list(profil[profil["numerique_en_texte"]].index)
total_repare = 0
total_irrecup = 0
rapport_coercition = pd.DataFrame()

if cols_num_texte:
    _, rapport_coercition = coercer_numeriques(df_brut, cols_num_texte)
    display(rapport_coercition)
    total_repare = int(rapport_coercition["nb_reparees"].sum())
    total_irrecup = int(rapport_coercition["nb_irrecuperables"].sum())
    display(
        Markdown(
            f"**Ce qu'il faut retenir.** {len(cols_num_texte)} colonne(s) numériques stockées "
            f"en texte : `{', '.join(cols_num_texte)}`. "
            f"La coercition répare **{total_repare} valeur(s)** (séparateur décimal virgule, "
            f"symboles €/%, espaces insécables) et laisse **{total_irrecup} valeur(s) "
            f"irrécupérable(s)** converties en NaN. "
            f"La conversion est appliquée en §7 via `coercer_numeriques()` dans le pipeline "
            f"sklearn, fitté à l'intérieur des plis de validation pour éviter toute fuite."
        )
    )
else:
    display(
        Markdown(
            "**Ce qu'il faut retenir.** Aucune colonne numérique stockée en texte n'a été "
            "détectée par `profil_compact()` (seuil : 80 % de valeurs convertibles). "
            "Le dataset est déjà correctement typé à ce niveau."
        )
    )

# %% [markdown]
# ### 5.6 Dates multi-formats
#
# La colonne `date_souscription` peut mélanger des formats (YYYY-MM-DD, DD/MM/YYYY…).
# `parser_dates()` détecte le format dominant, quantifie les dates ambiguës (ex. 01/02/2021 :
# JJ/MM ou MM/JJ ?) et documente la convention retenue — indispensable pour prouver
# l'absence de contamination temporelle lors du split train/test.

# %%
_COLONNES_DATES = ["date_souscription"]

_, rapport_dates = parser_dates(df_brut, _COLONNES_DATES)
display(
    rapport_dates[
        ["format_detecte", "convention_retenue", "nb_parsees", "nb_ambigues", "nb_irrecuperables"]
    ]
)

for col in _COLONNES_DATES:
    row = rapport_dates.loc[col]
    display(
        Markdown(
            f"**Ce qu'il faut retenir — `{col}`.** Format dominant détecté : "
            f"`{row['format_detecte']}`. Convention retenue : *{row['convention_retenue']}*. "
            f"**{row['nb_parsees']} date(s) parsée(s)**, {row['nb_ambigues']} ambiguë(s), "
            f"{row['nb_irrecuperables']} irrécupérable(s). "
            f"La convention DD/MM/YYYY par défaut est justifiée par le contexte français du projet. "
            f"Le parsing est appliqué en §7 via `parser_dates()` dans le pipeline."
        )
    )

# %% [markdown]
# ### 5.7 Valeurs métier impossibles
#
# Certaines plages de valeurs sont physiquement impossibles : `utilisateurs_actifs` ne peut
# dépasser `sieges_souscrits`, `taux_adoption_pct` doit être dans [0, 100], l'ancienneté
# doit être cohérente avec la date de souscription. Chaque violation est une anomalie
# documentée avec la décision de traitement associée.

# %%
anomalies = detecter_valeurs_impossibles(df_brut)

if anomalies.empty:
    display(Markdown("Aucune valeur métier impossible détectée selon les règles définies."))
    total_lignes_anomalies = 0
else:
    _decisions = {
        "utilisateurs_actifs > sieges_souscrits": "Borner utilisateurs_actifs à sieges_souscrits",
        "taux_adoption_pct": "Borner à [0, 100]",
        "anciennete_mois": "Passer en NaN si écart > 6 mois (incohérence trop forte)",
        "< 0": "Passer en NaN (valeur absurde pour un indicateur positif)",
    }

    def _choisir_decision(regle: str) -> str:
        for cle, dec in _decisions.items():
            if cle in regle:
                return dec
        return "Passer en NaN (valeur absurde)"

    df_anomalies = anomalies[["regle", "colonne_ou_paire", "nb_lignes_concernees"]].copy()
    df_anomalies["Décision"] = df_anomalies["regle"].apply(_choisir_decision)
    display(df_anomalies.set_index("regle"))

    total_lignes_anomalies = int(anomalies["nb_lignes_concernees"].sum())
    nb_regles = len(anomalies)
    display(
        Markdown(
            f"**Ce qu'il faut retenir.** **{nb_regles} règle(s) métier** violées, "
            f"affectant **{total_lignes_anomalies} ligne(s)** au total. "
            f"Le traitement est individualisé par règle (bornage ou passage en NaN) "
            f"pour préserver le maximum d'information tout en écartant les valeurs absurdes. "
            f"Application en §7."
        )
    )

# %% [markdown]
# ### 5.8 Mécanisme de manquance (MCAR / MAR / MNAR)
#
# Le mécanisme de manquance détermine la stratégie d'imputation : un manquant MNAR
# (Missing Not At Random) est corrélé à la cible — il est lui-même informatif et doit
# être encodé comme feature binaire avant imputation, sous peine de perdre un signal prédictif.
# L'hypothèse testée ici : un `csat` manquant indique souvent un client insatisfait.

# %%
rapport_manquance = analyser_manquance(df_brut, cible="churn")
nb_mnar = nb_mar = nb_mcar = 0
cols_mnar: list[str] = []

if rapport_manquance.empty:
    display(Markdown("Aucune colonne avec valeurs manquantes détectée après normalisation."))
else:
    nb_mnar = int((rapport_manquance["mecanisme_propose"] == "MNAR").sum())
    nb_mar = int((rapport_manquance["mecanisme_propose"] == "MAR").sum())
    nb_mcar = int((rapport_manquance["mecanisme_propose"] == "MCAR").sum())
    cols_mnar = list(rapport_manquance[rapport_manquance["mecanisme_propose"] == "MNAR"].index)

    display(
        rapport_manquance[
            [
                "taux_manquants",
                "mecanisme_propose",
                "confiance",
                "taux_churn_si_manquant",
                "taux_churn_si_renseigne",
                "recommandation",
            ]
        ]
    )

    cols_mnar_str = f"`{'`, `'.join(cols_mnar)}`" if cols_mnar else "aucune"
    display(
        Markdown(
            f"**Ce qu'il faut retenir.** "
            f"**{nb_mnar} colonne(s) MNAR** (corrélées au churn — indicateur de manquance "
            f"à conserver comme feature) : {cols_mnar_str}. "
            f"**{nb_mar} colonne(s) MAR** (imputation conditionnelle recommandée). "
            f"**{nb_mcar} colonne(s) MCAR** (imputation simple). "
            f"En §7, les colonnes MNAR reçoivent un indicateur binaire `_est_manquant` créé "
            f"AVANT imputation dans le pipeline sklearn, fitté dans chaque pli de cross-validation "
            f"pour éviter toute fuite de la distribution des manquants."
        )
    )

# %% [markdown]
# ### 5.9 Renommage — convention de nommage (item C3)
#
# L'item C3 exige que les données soient « correctement nommées ou renommées ».
# Convention adoptée : **snake_case sans accent, unités suffixées** (_eur, _pct, _mois, _jours).
# Le tableau avant/après sert de contrat — tout code ultérieur utilise les noms proposés.

# %%
df_renommage = proposer_renommage(df_brut)
display(df_renommage)

nb_modifies = int(df_renommage["modifie"].sum())

display(
    Markdown(
        f"**Ce qu'il faut retenir.** Convention : **snake_case, sans accent, "
        f"unités suffixées** (_eur, _pct, _mois, _jours). "
        f"**{nb_modifies} colonne(s) renommée(s)** sur {len(df_renommage)} — "
        f"les noms d'origine sont déjà quasi conformes (issus d'un dictionnaire structuré). "
        f"Les renommages seront appliqués en §7 avant tout feature engineering, "
        f"pour garantir la cohérence des noms dans l'ensemble du pipeline."
    )
)

# %% [markdown]
# ### 5.10 Synthèse — tableau de bord qualité
#
# Ce tableau récapitule toutes les anomalies détectées, leur volume, le traitement retenu
# et la section où il est appliqué. Il constitue la « documentation traçable des traitements »
# exigée par C3 et servira de référence lors de la relance du notebook par le jury.

# %%
_vol_num_texte = f"{total_repare} valeur(s) réparée(s)" if cols_num_texte else "aucune"
_vol_dates = int(rapport_dates["nb_parsees"].sum())
_vol_anomalies = int(anomalies["nb_lignes_concernees"].sum()) if not anomalies.empty else 0

_lignes_synthese = [
    {
        "Anomalie": "Doublons exacts",
        "Volume": f"{rapport_doublons['nb_doublons_exacts']} ligne(s)",
        "Traitement retenu": "drop_duplicates(keep='first')",
        "Section appliquée": "§7 — Préparation",
    },
    {
        "Anomalie": "Doublons sur clé métier (client_id)",
        "Volume": f"{rapport_doublons['nb_doublons_cle_metier_non_exacts']} ligne(s)",
        "Traitement retenu": "Conserver la ligne la plus récente",
        "Section appliquée": "§7 — Préparation",
    },
    {
        "Anomalie": f"Numériques en texte ({len(cols_num_texte)} col.)",
        "Volume": _vol_num_texte,
        "Traitement retenu": "coercer_numeriques() dans pipeline sklearn",
        "Section appliquée": "§7 — Préparation",
    },
    {
        "Anomalie": "Dates multi-formats (date_souscription)",
        "Volume": f"{_vol_dates} date(s) parsée(s)",
        "Traitement retenu": f"parser_dates() — {rapport_dates['format_detecte'].iloc[0]}",
        "Section appliquée": "§7 — Préparation",
    },
    {
        "Anomalie": "Valeurs métier impossibles",
        "Volume": f"{_vol_anomalies} ligne(s)",
        "Traitement retenu": "Bornage ou passage en NaN selon règle",
        "Section appliquée": "§7 — Préparation",
    },
    {
        "Anomalie": f"Valeurs manquantes — MNAR ({nb_mnar} col.)",
        "Volume": f"{nb_mnar} colonne(s) corrélées au churn",
        "Traitement retenu": "Indicateur binaire `_est_manquant` + imputation médiane",
        "Section appliquée": "§7 — Pipeline sklearn (dans les plis CV)",
    },
    {
        "Anomalie": f"Valeurs manquantes — MAR ({nb_mar} col.)",
        "Volume": f"{nb_mar} colonne(s) conditionnelles",
        "Traitement retenu": "KNNImputer conditionné",
        "Section appliquée": "§7 — Pipeline sklearn (dans les plis CV)",
    },
    {
        "Anomalie": f"Valeurs manquantes — MCAR ({nb_mcar} col.)",
        "Volume": f"{nb_mcar} colonne(s) aléatoires",
        "Traitement retenu": "SimpleImputer (moyenne / mode)",
        "Section appliquée": "§7 — Pipeline sklearn (dans les plis CV)",
    },
    {
        "Anomalie": f"Renommage ({nb_modifies} col.)",
        "Volume": f"{nb_modifies} colonne(s) modifiée(s)",
        "Traitement retenu": "snake_case, sans accent, unités suffixées",
        "Section appliquée": "§7 — Préparation",
    },
    {
        "Anomalie": f"Variables interdites ({len(config.COLONNES_INTERDITES)} col.)",
        "Volume": f"{len(config.COLONNES_INTERDITES)} colonne(s)",
        "Traitement retenu": "Exclusion via config.COLONNES_INTERDITES",
        "Section appliquée": "§7 & §9 — Pipeline & Modélisation",
    },
]

df_synthese = pd.DataFrame(_lignes_synthese).set_index("Anomalie")
display(df_synthese)

display(
    Markdown(
        "**Ce qu'il faut retenir.** Ce tableau couvre les **4 exigences techniques obligatoires** "
        "de l'énoncé : doublons ✓, valeurs manquantes ✓, numériques en texte ✓, "
        "dates multi-formats ✓. Chaque anomalie est tracée jusqu'à la section où elle est "
        "appliquée, ce qui permet au jury de vérifier la cohérence entre l'analyse et le pipeline. "
        "Aucune transformation n'est appliquée dans cette section — §7 est l'unique point "
        "de transformation, ce qui garantit la traçabilité et l'anti-fuite."
    )
)

# %% [markdown]
# > ### 📋 Journal de bord — Chargement et compréhension des données
# >
# > **Décisions retenues** — Normalisation des marqueurs textuels de valeur manquante en NaN
# > dès le chargement (avant tout appel aux fonctions quality.py), pour garantir la cohérence
# > de isna() dans profil_compact() et analyser_manquance(). Convention de nommage snake_case
# > avec suffixes d'unités (_eur, _pct) adoptée pour l'ensemble du pipeline aval.
# >
# > **Alternatives écartées** — Laisser les chaînes vides "" en l'état : aurait faussé tous
# > les taux de manquants (isna() = False sur ""). Charger avec dtype inféré par pandas :
# > aurait converti silencieusement certains champs et masqué les incohérences de format que
# > profil_compact() doit signaler. Importer _MARQUEURS_MANQUANTS depuis quality.py (privé) :
# > préféré une liste explicite dans le notebook, plus lisible pour le jury.
# >
# > **Difficultés rencontrées** — La détection MNAR de `csat` nécessite que la cible `churn`
# > soit présente dans le DataFrame brut en tant que chaîne — résolu : analyser_manquance()
# > applique pd.to_numeric() en interne sur la cible. Les valeurs impossibles sont détectées
# > sur dtype=str via pd.to_numeric(..., errors="coerce") dans detecter_valeurs_impossibles().
# >
# > **Impact sur la suite** — Les colonnes MNAR identifiées ici reçoivent un indicateur binaire
# > `_est_manquant` en §7 ; ces indicateurs sont inclus dans le pipeline sklearn fitté dans
# > chaque pli de cross-validation (anti-fuite). La synthèse §5.10 sert de contrat de données
# > pour §7 et de référence pour le jury lors de la relance du notebook.
# >
# > **Temps passé** — ~1 h (rédaction, mise au point des appels de fonctions, vérifications).
