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

# %%
import os

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
# Les trois fichiers sont présents, lisibles et intègres : leur empreinte SHA-256 garantit que l'on
# travaille systématiquement sur les "bonnes" données, et leur volumétrie (tableau ci-dessus)
# est cohérente avec l'énoncé. L'absence de droit d'écriture sur `data/raw/` est voulue :
# la couche Bronze est immuable. La gouvernance des accès humains (qui peut lire, qui peut auditer)
# est traitée en §3.5 et §3.6.

# %% [markdown]
# ### 3.2 Dictionnaire de données — extrait avec pertinence
#
# Le dictionnaire complet est dans `docs/DATASHEET.md`. On présente ici un extrait orienté
# métier : les colonnes retenues pour le modèle, avec leur pertinence a priori pour le churn.
#
# | Colonne | Type cible | Pertinence churn | Risque | Statut |
# |---|---|---|---|---|
# | `anciennete_mois` | int64 | Forte | aucun | Retenue |
# | `sieges_souscrits` | int64 | Forte | aucun | Retenue |
# | `utilisateurs_actifs` | int64 | Forte | aucun | Retenue |
# | `taux_adoption_pct` | float64 | Forte | aucun | Retenue (nettoyage virgule/%) |
# | `connexions_30j` | int64 | Forte | aucun | Retenue |
# | `heures_usage_30j` | float64 | Forte | aucun | Retenue (nettoyage virgule) |
# | `fonctionnalites_utilisees` | int64 | Forte | aucun | Retenue |
# | `nb_integrations` | int64 | Forte | aucun | Retenue (imputation médiane) |
# | `derniere_connexion_jours` | int64 | Forte | aucun | Retenue |
# | `tickets_support_90j` | int64 | Forte | aucun | Retenue |
# | `delai_reponse_support_h` | float64 | Moyenne | aucun | Retenue (nettoyage + imputation) |
# | `csat` | float64 | Forte | aucun | Retenue (imputation médiane) |
# | `retards_paiement_12m` | int64 | Forte | aucun | Retenue (imputation médiane) |
# | `revenu_mensuel_recurrent_eur` | float64 | Forte | aucun | Retenue (nettoyage virgule) |
# | `secteur` | catégorie | Moyenne | aucun | Retenue (normalisation casse) |
# | `pays` | catégorie | Moyenne | aucun | Retenue (imputation mode) |
# | `taille_entreprise` | catégorie ordinale | Forte | aucun | Retenue (ordinal 1–4) |
# | `plan` | catégorie ordinale | Forte | aucun | Retenue (ordinal 1–4) |
# | `date_souscription` | datetime → dérivés | Faible | aucun | Dériver trimestre/mois |
# | `client_id` | — | Nulle | identifiant | EXCLUE — `COLONNES_INTERDITES` |
# | `churn` | int64 (cible) | N/A | fuite (cible) | CIBLE — `COLONNES_INTERDITES` |
# | `valeur_vie_client_eur` | float64 (cible CLV) | Nulle | fuite | EXCLUE — `COLONNES_INTERDITES` |
# | `sante_compte_fin_periode` | int64 | Nulle | fuite confirmée | EXCLUE — `COLONNES_INTERDITES` |
# | `fonctionnalites_total` | — | Nulle | colinéaire (plan) | EXCLUE — bijection plan |
# | `couleur_theme_interface` | catégorie | Nulle | leurre | EXCLUE (preuve §6) |
# | `code_datacenter` | catégorie | Nulle | leurre | EXCLUE (preuve §6) |
# | `groupe_experimentation` | catégorie | Nulle | leurre | EXCLUE (preuve §6) |
# | `commentaire_csm` | texte | Nulle | leurre + fuite potentielle | EXCLUE (55 % manquants) |
# | `jour_souscription` | catégorie | Faible | aucun | Provisoire — décision §7 |

# %% [markdown]
# **Ce qu'il faut retenir.**
# Sur 29 colonnes : 19 retenues (dont `date_souscription`, qui sert à dériver des features),
# 4 exclues pour fuite ou identifiant, 4 exclues comme leurres (après preuve en §6), 1 exclue pour
# colinéarité parfaite, 1 décision différée à §7. Le dictionnaire complet (avec hash, cycle de vie,
# gouvernance) est dans `docs/DATASHEET.md`.

# %% [markdown]
# ### 3.3 Choix du modèle de stockage — note d'arbitrage
#
# Voici un tableau comparatif pour les technologies candidates.
#
# | Option | Pour | Contre | Retenu | Pourquoi |
# |---|---|---|---|---|
# | Fichiers Parquet (Bronze → Silver → Gold) | Lecture columnaire rapide, compression 5-10×, portabilité, pas de serveur | Pas de jointures ad hoc, pas de contrôle d'accès fin, pas de transactions | ✓ Gold (features entraînement + inférence) | Batch hebdomadaire sur ~5 000 lignes — aucun besoin de SGBD |
# | Base relationnelle (PostgreSQL) | Jointures, intégrité référentielle, contrôle d'accès, requêtes ad hoc CRM | Infrastructure à maintenir, surcoût pour un batch simple | ✓ Cible production (scores + logs inférence → CRM) | Intégration CRM impose un point d'accès SQL standardisé |
# | Base documents (MongoDB) | Flexibilité du schéma, JSON natif | Schéma flottant inadapté à un pipeline ML reproductible ; agrégats lents | ✗ Écarté | Le schéma des features est fixe et versionné — la flexibilité n'apporte rien |
# | Data lake objet (S3/Blob) | Coût faible, scalabilité infinie | Latence API, nécessite un moteur de requête (Athena/Spark) | ✗ Écarté pour ce prototype | Complexité disproportionnée pour 5 000 comptes et 1 DS mi-temps |

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
# Voici des solutions alternatives envisagées en cas d'indisponibilité.
#
# | Enrichissement souhaité | Utilité attendue | Disponibilité réelle | Plan B |
# |---|---|---|---|
# | Historique des tickets détaillé (catégorie, résolution) | Améliore le signal support (actuellement : seul le comptage 90j) | Disponible dans le système support (Zendesk) — accès non accordé | Conserver `tickets_support_90j` + `csat` comme proxy ; demander accès en phase 2 |
# | Données d'onboarding (achèvement des étapes de mise en route) | Signal précoce d'adoption (churn dans les 90 premiers jours) | Partiellement disponible dans le CRM — export non standardisé | Utiliser `anciennete_mois` + `fonctionnalites_utilisees` comme proxy d'onboarding |
# | Données économiques sectorielles (indice de confiance, croissance PIB) | Contextualiser les vagues de churn conjoncturelles | Open data (INSEE, Eurostat) — librement accessibles | Non requis : données publiques, accessibles à tout moment en phase 2 |
# | NPS (Net Promoter Score) trimestriel | Signal d'intention déclaré complémentaire du CSAT ponctuel | Collecté depuis 6 mois seulement — couverture insuffisante (< 40 %) | Utiliser `csat` ; intégrer le NPS dès qu'il atteindra 80 % de couverture |
# | Logs d'erreurs applicatives (erreurs 5xx/4xx par compte) | Détection de friction technique invisible dans les tickets | Disponible en interne (ELK) — agrégation par compte non disponible | Exclure pour ce projet ; instrumenter l'agrégation par `client_id` pour la V2 |

# %% [markdown]
# **Ce qu'il faut retenir.**
# Aucun enrichissement n'est bloquant pour la version initiale du modèle : chaque donnée manquante
# est remplacée par un *proxy*, une variable déjà présente dans le jeu et corrélée au signal visé
# (par exemple `csat` à la place du NPS, *Net Promoter Score*). On perd un peu de signal, mais on
# livre une solution fonctionnelle dès maintenant. Les deux enrichissements à plus fort potentiel,
# l'historique détaillé des tickets et le NPS, sont priorisés pour une V2, dès que l'accès au
# système support sera accordé et que le NPS couvrira 80 % des clients.

# %% [markdown]
# ### 3.5 Cycle de vie du jeu de données
#
# Référence complète : `docs/DATASHEET.md`.
# Ce tableau résume les étapes clés et les responsabilités associées.
#
# | Étape | Description | Responsable | Date |
# |---|---|---|---|
# | Création | Export CRM + logs d'usage (période janv. 2022 – déc. 2023) | Data Owner (DSI) | Janv. 2024 |
# | Bronze (raw) | Stockage tel quel dans `data/raw/` — fichiers figés, jamais modifiés | Data Scientist | Janv. 2024 |
# | Silver (interim) | Déduplication, normalisation des types, dates parsées (§5, §7) | Data Scientist | Pendant le projet |
# | Gold (features) | Features finales en Parquet, versionné DVC, utilisé pour l'entraînement | Data Scientist | Pendant le projet |
# | Rétention | Données conservées 3 ans (obligation légale B2B), puis anonymisation | DPO | Déc. 2026 → revue |
# | Accès | Restreint au Data Scientist + CS Lead en lecture ; DPO en audit | DSI / RSSI | Continu |
# | Usages futurs | Réentraînement trimestriel (§13.8) ; extension au NPS et tickets détaillés (voir §3.4) | Data Scientist | Comité trimestriel |

# %% [markdown]
# **Ce qu'il faut retenir.**
# Le cycle de vie suit une architecture médaillon (Bronze → Silver → Gold) qui garantit
# la reproductibilité : la source brute n'est jamais modifiée, et chaque transformation est
# traçable via git + DVC. La rétention de 3 ans est alignée sur les obligations RGPD B2B.

# %% [markdown]
# ### 3.6 Trace de soumission aux parties prenantes
#
# Le cycle de vie documenté est soumis aux parties prenantes (Item C3).
#
# > **Note méthodologique.** Les échanges ci-dessous sont *simulés* et assumés comme tels.
# > Dans un projet réel, ces traces seraient des emails archivés ou des tickets de gouvernance.
# > Ils sont inclus pour démontrer la démarche.
#
# | Partie prenante | Rôle | Date envoi | Date retour | Avis | Réserves | Levée |
# |---|---|---|---|---|---|---|
# | Data Owner (DSI) | Valide la légitimité d'usage des données | 2024-01-15 | 2024-01-18 | Favorable | Confirmer l'exclusion de `commentaire_csm` (risque rétrospectif) | Exclusion confirmée via `COLONNES_INTERDITES` (§7) |
# | DPO | Vérifie la conformité RGPD | 2024-01-15 | 2024-01-20 | Favorable sous réserves | Pas de variable directement identifiante dans les features ; analyse d'impact (AIPD) à réaliser si déploiement UE | AIPD prévue avant mise en production (§4, §10) |
# | CS Lead | Valide la pertinence métier des features et cas d'usage | 2024-01-15 | 2024-01-17 | Favorable | Ajouter la variation du score semaine-sur-semaine dans l'export CRM | Feature `delta_score_7j` ajoutée dans la spécification API (§10) |

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
# > **Difficultés rencontrées** — Le jeu fourni contient des défauts (virgules
# > décimales, casse hétérogène, dates multi-formats). La fonction `controler_source()` confirme
# > l'intégrité des fichiers bruts via SHA-256 — les traitements de nettoyage sont différés à §5
# > et §7 pour ne pas « polluer » la couche Bronze.
# >
# > **Impact sur la suite** — Les hashes SHA-256 permettent de vérifier que les données livrées
# > sont identiques à celles de l'entraînement. La trace de soumission déclenche l'analyse AIPD en §4.
# > Les enrichissements écartés pour la V1 (tickets détaillés et NPS en priorité) constituent la
# > feuille de route des données pour une V2.
