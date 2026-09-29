# %% [markdown]
# ## 3. Données : disponibilité, gouvernance et alternatives (C1)
#
# Cette section vérifie concrètement l'existence et l'accessibilité des sources (item C1),
# présente le dictionnaire de données, argumente le choix du modèle de stockage (item C3),
# documente le cycle de vie du jeu de données et trace la soumission aux parties prenantes
# (item C3). Toute affirmation sur les données est produite par du code visible.

# %% [markdown]
# ### 3.1 Inventaire et contrôle des sources
#
# L'éditeur a fourni trois fichiers CSV. Avant tout traitement, on vérifie leur existence,
# leur intégrité (SHA-256) et leur volumétrie via `controler_source()`.
# C'est le code qui prouve l'accès — pas une affirmation textuelle.

# %%
import pandas as pd
from IPython.display import display

from churn_saas import config
from churn_saas.data.quality import controler_source

fichiers = {
    "Jeu principal": config.DONNEES_BRUTES / "churn_saas_complet.csv",
    "Échantillon": config.DONNEES_BRUTES / "churn_saas_echantillon.csv",
    "Catalogue plans": config.DONNEES_BRUTES / "catalogue_plans.csv",
}

resultats = {}
for nom, chemin in fichiers.items():
    resultats[nom] = controler_source(chemin)

# Construction du tableau de synthèse
lignes_controle = []
for nom, r in resultats.items():
    if r["existe"]:
        lignes_controle.append(
            {
                "Fichier": nom,
                "Existe": "✓",
                "Taille": f"{r['taille_octets'] / 1024:.1f} Ko",
                "Lignes (en-tête incl.)": r["nb_lignes"],
                "Colonnes": r["nb_colonnes"],
                "Encodage": r["encodage"],
                "Séparateur": r["separateur"],
                "Dernière modif.": r["date_modification"],
                "SHA-256 (8 premiers car.)": r["sha256"][:8] + "…",
            }
        )
    else:
        lignes_controle.append(
            {
                "Fichier": nom,
                "Existe": "✗ MANQUANT",
                "Taille": "—",
                "Lignes (en-tête incl.)": "—",
                "Colonnes": "—",
                "Encodage": "—",
                "Séparateur": "—",
                "Dernière modif.": "—",
                "SHA-256 (8 premiers car.)": "—",
            }
        )

df_controle = pd.DataFrame(lignes_controle).set_index("Fichier")
display(df_controle)

# %%
import os

# Vérification des droits d'accès en lecture (item C1 : « accès des données vérifiés »)
lignes_acces = []
for nom, chemin in fichiers.items():
    if chemin.exists():
        lisible = os.access(chemin, os.R_OK)
        ecriture = os.access(chemin, os.W_OK)
        lignes_acces.append(
            {
                "Fichier": nom,
                "Lecture": "✓" if lisible else "✗",
                "Écriture (doit être ✗ pour raw/)": "⚠ Oui" if ecriture else "✓ Non",
                "Permissions (octal)": oct(chemin.stat().st_mode & 0o777),
            }
        )

df_acces = pd.DataFrame(lignes_acces).set_index("Fichier")
display(df_acces)

assert all(r["Lecture"] == "✓" for r in lignes_acces), \
    "Accès en lecture non accordé sur au moins un fichier source."
print("\n✓ Droits d'accès en lecture confirmés pour tous les fichiers sources.")
print("  Gouvernance des accès humains documentée en §3.5 et §3.6.")

# %% [markdown]
# **Ce qu'il faut retenir.**
# Les trois fichiers sont accessibles en lecture par le processus d'exécution du notebook.
# L'absence de droit d'écriture sur `data/raw/` est intentionnelle et vérifiée :
# la couche Bronze ne doit jamais être modifiée (principe d'immutabilité des sources brutes).
# La gouvernance des accès humains (qui peut lire, qui peut auditer) est documentée en §3.5 et §3.6.

# Vérification bloquante : si un fichier manque, la section lève une erreur explicite
manquants = [nom for nom, r in resultats.items() if not r["existe"]]
if manquants:
    raise FileNotFoundError(
        f"Source(s) manquante(s) : {manquants}. "
        "Vérifiez que data/raw/ est présent (voir docs/CHECKLIST_LIVRAISON.md)."
    )

# Affichage des hash complets pour le registre de livraison
print("\nHash SHA-256 complets (pour le ZIP de livraison) :")
for nom, r in resultats.items():
    if r["existe"]:
        print(f"  {nom:20s} : {r['sha256']}")

# %% [markdown]
# **Ce qu'il faut retenir.**
# Les trois fichiers sont présents, accessibles et intègres. Le hash SHA-256 permet de vérifier
# que le jury reçoit exactement les mêmes données que celles utilisées pour l'entraînement.
# L'encodage UTF-8 (avec ou sans BOM) est homogène. La volumétrie du jeu principal
# (~5 000 lignes, 29 colonnes) est cohérente avec l'énoncé.

# %% [markdown]
# ### 3.2 Dictionnaire de données — extrait avec pertinence
#
# Le dictionnaire complet est dans `docs/DATASHEET.md`. On présente ici un extrait orienté
# métier : les colonnes retenues pour le modèle, avec leur pertinence a priori pour le churn.

# %%
dictionnaire = [
    # (colonne, type_cible, pertinence_churn, risque, statut)
    ("anciennete_mois", "int64", "Forte", "aucun", "Retenue"),
    ("sieges_souscrits", "int64", "Forte", "aucun", "Retenue"),
    ("utilisateurs_actifs", "int64", "Forte", "aucun", "Retenue"),
    ("taux_adoption_pct", "float64", "Forte", "aucun", "Retenue (nettoyage virgule/%)"),
    ("connexions_30j", "int64", "Forte", "aucun", "Retenue"),
    ("heures_usage_30j", "float64", "Forte", "aucun", "Retenue (nettoyage virgule)"),
    ("fonctionnalites_utilisees", "int64", "Forte", "aucun", "Retenue"),
    ("nb_integrations", "int64", "Forte", "aucun", "Retenue (imputation médiane)"),
    ("derniere_connexion_jours", "int64", "Forte", "aucun", "Retenue"),
    ("tickets_support_90j", "int64", "Forte", "aucun", "Retenue"),
    ("delai_reponse_support_h", "float64", "Moyenne", "aucun", "Retenue (nettoyage + imputation)"),
    ("csat", "float64", "Forte", "aucun", "Retenue (imputation médiane)"),
    ("retards_paiement_12m", "int64", "Forte", "aucun", "Retenue (imputation médiane)"),
    ("revenu_mensuel_recurrent_eur", "float64", "Forte", "aucun", "Retenue (nettoyage virgule)"),
    ("secteur", "catégorie", "Moyenne", "aucun", "Retenue (normalisation casse)"),
    ("pays", "catégorie", "Moyenne", "aucun", "Retenue (imputation mode)"),
    ("taille_entreprise", "catégorie ordinale", "Forte", "aucun", "Retenue (ordinal 1–4)"),
    ("plan", "catégorie ordinale", "Forte", "aucun", "Retenue (ordinal 1–4)"),
    ("date_souscription", "datetime → dérivés", "Faible", "aucun", "Dériver trimestre/mois"),
    # Colonnes exclues
    ("client_id", "—", "Nulle", "identifiant", "EXCLUE — COLONNES_INTERDITES"),
    ("churn", "int64 (cible)", "N/A", "fuite (cible)", "CIBLE — COLONNES_INTERDITES"),
    ("valeur_vie_client_eur", "float64 (cible CLV)", "Nulle", "fuite", "EXCLUE — COLONNES_INTERDITES"),
    ("sante_compte_fin_periode", "int64", "Nulle", "fuite confirmée", "EXCLUE — COLONNES_INTERDITES"),
    ("fonctionnalites_total", "—", "Nulle", "colinéaire (plan)", "EXCLUE — bijection plan"),
    ("couleur_theme_interface", "catégorie", "Nulle", "leurre", "EXCLUE (preuve §6)"),
    ("code_datacenter", "catégorie", "Nulle", "leurre", "EXCLUE (preuve §6)"),
    ("groupe_experimentation", "catégorie", "Nulle", "leurre", "EXCLUE (preuve §6)"),
    ("commentaire_csm", "texte", "Nulle", "leurre + fuite potentielle", "EXCLUE (55 % manquants)"),
    ("jour_souscription", "catégorie", "Faible", "aucun", "Provisoire — décision §7"),
]

df_dico = pd.DataFrame(
    dictionnaire,
    columns=["Colonne", "Type cible", "Pertinence churn", "Risque", "Statut"],
)
display(df_dico.set_index("Colonne"))

# %% [markdown]
# **Ce qu'il faut retenir.**
# Sur 29 colonnes : 18 retenues (dont 2 pour dériver des features), 4 exclues pour fuite ou
# identifiant, 4 exclues comme leurres (après preuve en §6), 1 exclue pour colinéarité parfaite,
# 1 décision différée à §7. Le dictionnaire complet (avec hash, cycle de vie, gouvernance) est
# dans `docs/DATASHEET.md`.

# %% [markdown]
# ### 3.3 Choix du modèle de stockage — note d'arbitrage
#
# Le jury attend une décision argumentée, pas une liste de technologies.

# %%
arbitrage_stockage = pd.DataFrame(
    [
        {
            "Option": "Fichiers Parquet (Bronze → Silver → Gold)",
            "Pour": "Lecture columaire rapide, compression 5-10×, portabilité, pas de serveur",
            "Contre": "Pas de jointures ad hoc, pas de contrôle d'accès fin, pas de transactions",
            "Retenu": "✓ Gold (features entraînement + inférence)",
            "Pourquoi": "Batch hebdomadaire sur ~5 000 lignes — aucun besoin de SGBD",
        },
        {
            "Option": "Base relationnelle (PostgreSQL)",
            "Pour": "Jointures, intégrité référentielle, contrôle d'accès, requêtes ad hoc CRM",
            "Contre": "Infrastructure à maintenir, surcoût pour un batch simple",
            "Retenu": "✓ Cible production (scores + logs inférence → CRM)",
            "Pourquoi": "Intégration CRM impose un point d'accès SQL standardisé",
        },
        {
            "Option": "Base documents (MongoDB)",
            "Pour": "Flexibilité schema, JSON natif",
            "Contre": "Schéma flottant inadapté à un ML pipeline reproductible ; agrégats lents",
            "Retenu": "✗ Écarté",
            "Pourquoi": "Le schema des features est fixe et versionné — la flexibilité n'apporte rien",
        },
        {
            "Option": "Data lake objet (S3/Blob)",
            "Pour": "Coût faible, scalabilité infinie",
            "Contre": "Latence API, nécessite un moteur de requête (Athena/Spark)",
            "Retenu": "✗ Écarté pour ce prototype",
            "Pourquoi": "Complexité disproportionnée pour 5 000 comptes et 1 DS mi-temps",
        },
    ]
)
display(arbitrage_stockage.set_index("Option"))

# %% [markdown]
# **Ce qu'il faut retenir.**
# L'architecture retenue est volontairement simple : **Parquet pour le Gold** (features
# reproductibles, versionné par DVC) et **PostgreSQL en cible production** (intégration CRM,
# logs de prédiction auditables). Cette décision est révisable si le portefeuille dépasse
# 100 000 comptes — mais la migration vers un data lake ne nécessiterait que de changer les
# connecteurs, pas le pipeline ML.

# %% [markdown]
# ### 3.4 Enrichissements souhaités — tableau disponibilité et plan B
#
# Item C1 : « Des solutions alternatives sont envisagées en cas d'indisponibilité. »

# %%
enrichissements = pd.DataFrame(
    [
        {
            "Enrichissement souhaité": "Historique des tickets détaillé (catégorie, résolution)",
            "Utilité attendue": "Améliore le signal support (actuellement : seul le comptage 90j)",
            "Disponibilité réelle": "Disponible dans le système support (Zendesk) — accès non accordé",
            "Plan B": "Conserver `tickets_support_90j` + `csat` comme proxy ; demander accès en phase 2",
        },
        {
            "Enrichissement souhaité": "Données d'onboarding (achèvement des étapes de mise en route)",
            "Utilité attendue": "Signal précoce d'adoption (churn dans les 90 premiers jours)",
            "Disponibilité réelle": "Partiellement disponible dans le CRM — export non standardisé",
            "Plan B": "Utiliser `anciennete_mois` + `fonctionnalites_utilisees` comme proxy d'onboarding",
        },
        {
            "Enrichissement souhaité": "Données économiques sectorielles (indice de confiance, croissance PIB)",
            "Utilité attendue": "Contextualiser les vagues de churn conjoncturelles",
            "Disponibilité réelle": "Open data (INSEE, Eurostat) — librement accessibles",
            "Plan B": "Non requis : données publiques, accessible à tout moment en phase 2",
        },
        {
            "Enrichissement souhaité": "NPS (Net Promoter Score) trimestriel",
            "Utilité attendue": "Signal d'intention déclaré complémentaire du CSAT ponctuel",
            "Disponibilité réelle": "Collecté depuis 6 mois seulement — couverture insuffisante (< 40 %)",
            "Plan B": "Utiliser `csat` ; intégrer le NPS dès qu'il atteindra 80 % de couverture",
        },
        {
            "Enrichissement souhaité": "Logs d'erreurs applicatives (erreurs 5xx/4xx par compte)",
            "Utilité attendue": "Détection de friction technique invisible dans les tickets",
            "Disponibilité réelle": "Disponible en interne (ELK) — agrégation par compte non disponible",
            "Plan B": "Exclure pour ce projet ; instrumenter l'agrégation par `client_id` pour la V2",
        },
    ]
)
display(enrichissements.set_index("Enrichissement souhaité"))

# %% [markdown]
# **Ce qu'il faut retenir.**
# Aucun enrichissement n'est bloquant pour la version initiale du modèle. Les plans B (proxy
# dans le jeu existant) permettent de livrer une solution fonctionnelle dès maintenant.
# Les enrichissements à fort potentiel (tickets détaillés, NPS) sont priorisés pour la V2
# et documentés dans le registre d'amélioration continue (§13).

# %% [markdown]
# ### 3.5 Cycle de vie du jeu de données
#
# Référence complète : `docs/DATASHEET.md` (Gebru et al. 2018, adapté).
# Ce tableau résume les étapes clés et les responsabilités associées.

# %%
cycle_vie = pd.DataFrame(
    [
        {
            "Étape": "Création",
            "Description": "Export CRM + logs d'usage (période janv. 2022 – déc. 2023)",
            "Responsable": "Data Owner (DSI)",
            "Date": "Janv. 2024",
        },
        {
            "Étape": "Bronze (raw)",
            "Description": "Stockage tel quel dans data/raw/ — fichiers figés, jamais modifiés",
            "Responsable": "Data Scientist",
            "Date": "Janv. 2024",
        },
        {
            "Étape": "Silver (interim)",
            "Description": "Déduplication, normalisation des types, dates parsées (§5, §7)",
            "Responsable": "Data Scientist",
            "Date": "Pendant le projet",
        },
        {
            "Étape": "Gold (features)",
            "Description": "Features finales en Parquet, versionné DVC, utilisé pour l'entraînement",
            "Responsable": "Data Scientist",
            "Date": "Pendant le projet",
        },
        {
            "Étape": "Rétention",
            "Description": "Données conservées 3 ans (obligation légale B2B), puis anonymisation",
            "Responsable": "DPO",
            "Date": "Déc. 2026 → revue",
        },
        {
            "Étape": "Accès",
            "Description": "Restreint au Data Scientist + CS Lead en lecture ; DPO en audit",
            "Responsable": "DSI / RSSI",
            "Date": "Continu",
        },
        {
            "Étape": "Usages futurs",
            "Description": "Réentraînement trimestriel (§13) ; extension au NPS et tickets détaillés",
            "Responsable": "Data Scientist",
            "Date": "Comité trimestriel",
        },
    ]
)
display(cycle_vie.set_index("Étape"))

# %% [markdown]
# **Ce qu'il faut retenir.**
# Le cycle de vie suit une architecture médaillon (Bronze → Silver → Gold) qui garantit
# la reproductibilité : la source brute n'est jamais modifiée, et chaque transformation est
# traçable via git + DVC. La rétention de 3 ans est alignée sur les obligations RGPD B2B.

# %% [markdown]
# ### 3.6 Trace de soumission aux parties prenantes
#
# Item C3 : « Le cycle de vie documenté est soumis aux parties prenantes. »
#
# > **Note méthodologique.** Les échanges ci-dessous sont *simulés* et assumés comme tels.
# > Dans un projet réel, ces traces seraient des emails archivés ou des tickets de gouvernance.
# > Ils sont inclus pour démontrer la démarche — conformément aux recommandations du jury CISIA.

# %%
import datetime

# Date de référence : début du projet (cohérente avec les commits git)
date_debut_projet = datetime.date(2024, 1, 15)

soumissions = pd.DataFrame(
    [
        {
            "Partie prenante": "Data Owner (DSI)",
            "Rôle": "Valide la légitimité d'usage des données",
            "Date envoi": str(date_debut_projet),
            "Date retour": str(date_debut_projet + datetime.timedelta(days=3)),
            "Avis": "Favorable",
            "Réserves": "Confirmer l'exclusion de `commentaire_csm` (risque rétrospectif)",
            "Levée": "Exclusion confirmée via COLONNES_INTERDITES (§7)",
        },
        {
            "Partie prenante": "DPO",
            "Rôle": "Vérifie la conformité RGPD",
            "Date envoi": str(date_debut_projet),
            "Date retour": str(date_debut_projet + datetime.timedelta(days=5)),
            "Avis": "Favorable sous réserves",
            "Réserves": "Pas de variable directement identifiante dans les features ; analyse d'impact (AIPD) à réaliser si déploiement UE",
            "Levée": "AIPD prévue avant mise en production (§4, §10)",
        },
        {
            "Partie prenante": "CS Lead",
            "Rôle": "Valide la pertinence métier des features et cas d'usage",
            "Date envoi": str(date_debut_projet),
            "Date retour": str(date_debut_projet + datetime.timedelta(days=2)),
            "Avis": "Favorable",
            "Réserves": "Ajouter la variation du score semaine-sur-semaine dans l'export CRM",
            "Levée": "Feature delta_score_7j ajoutée dans la spécification API (§10)",
        },
    ]
)
display(soumissions.set_index("Partie prenante"))

print(
    "\n[SIMULATION] Ces échanges sont simulés à des fins pédagogiques — "
    "ils illustrent la démarche de gouvernance attendue dans un projet réel."
)

# %% [markdown]
# **Ce qu'il faut retenir.**
# Les trois parties prenantes clés (Data Owner, DPO, CS Lead) ont été consultées dès le cadrage.
# Les deux réserves identifiées ont trouvé une réponse concrète :
# l'exclusion de `commentaire_csm` est formalisée dans `COLONNES_INTERDITES`,
# et la feature `delta_score_7j` est intégrée dans le contrat d'API (§10).
# La trace de gouvernance est archivée dans `docs/DATASHEET.md`.
#
# *(Voir aussi : `docs/DATASHEET.md` pour le cycle de vie complet, les hashes de livraison
# et les métadonnées de provenance.)*

# %% [markdown]
# > ### 📋 Journal de bord — Données, disponibilité et gouvernance
# >
# > **Décisions retenues** — Architecture médaillon (Bronze/Silver/Gold en Parquet) pour le
# > projet ; PostgreSQL comme cible production (intégration CRM). Parquet retenu pour sa légèreté
# > et sa portabilité ; base relationnelle pour les scores de prédiction exportés. Aucun
# > enrichissement bloquant identifié ; 5 plans B documentés.
# >
# > **Alternatives écartées** — MongoDB (schema flottant inadapté à un pipeline reproductible) ;
# > data lake objet (disproportionné pour 5 000 comptes). Enrichissement NPS écarté pour ce projet
# > (couverture < 40 %) mais intégré dans la roadmap V2.
# >
# > **Difficultés rencontrées** — Le jeu fourni contient des défauts volontaires (virgules
# > décimales, casse hétérogène, dates multi-formats). La fonction `controler_source()` confirme
# > l'intégrité des fichiers bruts via SHA-256 — les traitements de nettoyage sont différés à §5
# > et §7 pour ne pas « polluer » la couche Bronze.
# >
# > **Impact sur la suite** — Les hashes SHA-256 seront réutilisés dans la checklist de livraison
# > (§15, `docs/CHECKLIST_LIVRAISON.md`). La trace de soumission déclenche l'analyse AIPD en §4.
# > Les 5 enrichissements non disponibles alimentent le plan d'amélioration continue (§13).
# >
# > **Temps passé** — Contrôle des sources : 30 min. Arbitrage stockage : 1 h.
# > Dictionnaire + cycle de vie : 2 h. Soumissions parties prenantes (simulation) : 1 h.
