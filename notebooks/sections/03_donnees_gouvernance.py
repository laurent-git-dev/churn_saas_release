# %% [markdown]
# ## 3. Données : disponibilité, gouvernance et alternatives
#
# Avant d'analyser des données, on s'assure qu'elles existent, sont lisibles et intactes. Puis on
# résume leur pertinence, on justifie le stockage, on prévoit des replis et on documente leur
# cycle de vie, soumis aux parties prenantes.

# %% [markdown]
# ### 3.1 Inventaire et contrôle des sources
#
# L'éditeur a fourni trois fichiers CSV. Pour être sûr de toujours travailler sur les mêmes
# données, on calcule leur **empreinte SHA-256** : une signature de 64 caractères qui change dès
# qu'un seul octet du fichier change. `controler_source()` mesure aussi la volumétrie, l'encodage
# et le séparateur ; la cellule échoue si un fichier manque ou n'est pas lisible.

# %%
import os

import pandas as pd
from IPython.display import display

from churn_saas import config
from churn_saas.data.quality import controler_source
from churn_saas.format_fr import entier, nombre

fichiers = {
    "Jeu principal": config.DONNEES_BRUTES / "churn_saas_complet.csv",
    "Échantillon": config.DONNEES_BRUTES / "churn_saas_echantillon.csv",
    "Catalogue plans": config.DONNEES_BRUTES / "catalogue_plans.csv",
}
resultats = {nom: controler_source(chemin) for nom, chemin in fichiers.items()}
manquants = [nom for nom, r in resultats.items() if not r["existe"]]
assert not manquants, f"Sources introuvables : {manquants}"
assert all(os.access(c, os.R_OK) for c in fichiers.values()), "Lecture refusée sur une source."

df_controle = pd.DataFrame(
    {
        nom: {
            "Taille": f"{nombre(resultats[nom]['taille_octets'] / 1024, 1)} Ko",
            "Lignes (en-tête incl.)": entier(resultats[nom]["nb_lignes"]),
            "Colonnes": entier(resultats[nom]["nb_colonnes"]),
            "Encodage": resultats[nom]["encodage"],
            "Séparateur": resultats[nom]["separateur"],
            "Dernière modif.": resultats[nom]["date_modification"],
            "SHA-256 (8 premiers car.)": resultats[nom]["sha256"][:8] + "…",
            "Écriture possible": "oui" if os.access(chemin, os.W_OK) else "non",
            "Permissions": oct(chemin.stat().st_mode & 0o777),
        }
        for nom, chemin in fichiers.items()
    }
).T.rename_axis("Fichier")
display(df_controle)

# %% [markdown]
# **Ce qu'il faut retenir.** Les trois fichiers sont présents et lisibles ; leur volumétrie est
# celle du tableau. La colonne « Écriture possible » montre que, sur le poste de travail, rien
# n'empêche techniquement de les modifier : l'immuabilité de la couche brute (`data/raw/`) tient
# à une règle (aucun code n'y écrit) et se contrôle par l'empreinte SHA-256, consignée dans
# `docs/DATASHEET.md`. En production, l'accès en lecture seule est accordé par la DSI (§3.5).

# %% [markdown]
# ### 3.2 Dictionnaire de données — extrait avec pertinence
#
# On veut savoir, avant toute analyse, quelles colonnes ont une chance d'expliquer le churn
# (résiliation) et lesquelles posent un risque. Le dictionnaire de référence, avec le statut final
# de chaque colonne, est au §5.3 (et dans `docs/DATASHEET.md`) ; voici l'extrait orienté métier.
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
# | `pays` | catégorie | Moyenne | aucun | Retenue (manquants → modalité « inconnu ») |
# | `taille_entreprise` | catégorie | Forte | aucun | Retenue (normalisation casse, encodage disjonctif) |
# | `plan` | catégorie | Forte | aucun | Retenue (normalisation casse, encodage disjonctif) |
# | `fonctionnalites_total` | int64 | Faible | redondante avec `plan` (une valeur par plan) | Retenue (sert au taux de couverture fonctionnelle, §7.5) |
# | `date_souscription` | date | Faible | redondante (ancienneté) | Non modélisée (§6.11) |
# | `jour_souscription` | catégorie | Faible | aucun | Non modélisée (§6.8) |
# | `couleur_theme_interface` | catégorie | Nulle | leurre suspecté | Gardée pour la preuve (§6.10, §12.9), retirée du modèle déployé (§10.3) |
# | `code_datacenter` | catégorie | Nulle | leurre suspecté | Gardée pour la preuve (§6.10, §12.9), retirée du modèle déployé (§10.3) |
# | `groupe_experimentation` | catégorie | Nulle | leurre suspecté | Gardée pour la preuve (§6.10, §12.9), retirée du modèle déployé (§10.3) |
# | `client_id` | — | Nulle | identifiant | EXCLUE — `COLONNES_INTERDITES` |
# | `churn` | int64 (cible) | N/A | fuite (cible) | CIBLE — `COLONNES_INTERDITES` |
# | `valeur_vie_client_eur` | float64 | Nulle | fuite (cible CLV) | EXCLUE — `COLONNES_INTERDITES` |
# | `sante_compte_fin_periode` | int64 | Nulle | fuite confirmée | EXCLUE — `COLONNES_INTERDITES` |
# | `commentaire_csm` | texte | Nulle | fuite potentielle (date de rédaction inconnue) | Suspecte — audit §6.6, puis interdite |

# %% [markdown]
# **Ce qu'il faut retenir.** Les variables d'usage, de support, de facturation et de contrat
# forment le cœur du jeu. Deux colonnes de date ne sont pas modélisées, trois leurres suspectés
# sont gardés le temps de prouver leur inutilité, et les colonnes de `config.COLONNES_INTERDITES`
# (identifiant, deux cibles, fuites) n'entrent jamais dans le modèle.

# %% [markdown]
# ### 3.3 Choix du modèle de stockage — note d'arbitrage
#
# Le bon stockage est le plus simple qui couvre l'usage : un calcul nocturne sur quelques milliers
# de comptes (§3.1), puis un envoi des scores au CRM.
#
# | Option | Pour | Contre | Retenu | Pourquoi |
# |---|---|---|---|---|
# | Fichiers Parquet (Bronze → Silver → Gold) | Lecture par colonne rapide, compression, portabilité, pas de serveur | Pas de jointures ad hoc, pas de contrôle d'accès fin, pas de transactions | ✓ Gold (variables d'entraînement et d'inférence) | Traitement par lot nocturne sur quelques milliers de lignes (§10) : aucun besoin de SGBD |
# | Base relationnelle (PostgreSQL) | Jointures, intégrité référentielle, contrôle d'accès, requêtes du CRM | Infrastructure à maintenir | ✓ Cible production (scores et journal des prédictions → CRM) | L'intégration CRM impose un point d'accès SQL standard |
# | Base documents (MongoDB) | Schéma souple, JSON natif | Schéma mouvant inadapté à une chaîne reproductible ; agrégats lents | ✗ Écarté | Le schéma des variables est fixe et versionné : la souplesse n'apporte rien |
# | Stockage objet (S3/Blob) + moteur de requête | Coût faible, capacité quasi illimitée | Nécessite un moteur de requête (Athena/Spark), latence | ✗ Écarté pour ce prototype | Complexité disproportionnée pour ce volume et un data scientist à mi-temps |

# %% [markdown]
# **Ce qu'il faut retenir.** L'architecture est volontairement simple : **Parquet pour le Gold**
# (reproductible, versionné par DVC) et **PostgreSQL en production** (intégration CRM, prédictions
# auditables). Le choix est révisable si le portefeuille dépasse 100 000 comptes ; passer au
# stockage objet ne changerait que les connecteurs, pas la chaîne de modélisation.

# %% [markdown]
# ### 3.4 Enrichissements souhaités — tableau disponibilité et plan B
#
# Certaines données utiles n'existent pas encore ou ne sont pas accessibles ; pour chacune, on
# prévoit une solution de repli.
#
# | Enrichissement souhaité | Utilité attendue | Disponibilité réelle | Plan B |
# |---|---|---|---|
# | Historique des tickets détaillé (catégorie, résolution) | Affiner le signal support (aujourd'hui : seul le comptage sur 90 jours) | Dans l'outil support (Zendesk), accès non accordé | `tickets_support_90j` + `csat` comme substituts ; accès demandé pour la V2 |
# | Données d'onboarding (étapes de mise en route achevées) | Signal précoce d'adoption (départs des 90 premiers jours) | Partiellement dans le CRM, export non standardisé | `anciennete_mois` + `fonctionnalites_utilisees` comme substituts |
# | Données économiques sectorielles (confiance, croissance) | Expliquer les vagues de départ conjoncturelles | Données publiques (INSEE, Eurostat) | Non requis en V1 ; accessibles à tout moment |
# | NPS (Net Promoter Score) trimestriel | Intention déclarée, complémentaire du CSAT ponctuel | Collecté depuis 6 mois, couverture < 40 % | `csat` ; intégrer le NPS à 80 % de couverture |
# | Journaux d'erreurs applicatives par compte | Friction technique invisible dans les tickets | En interne (ELK), sans agrégation par compte | Exclu ; agrégation par `client_id` à instrumenter pour la V2 |

# %% [markdown]
# **Ce qu'il faut retenir.** Aucun enrichissement n'est bloquant : chaque donnée absente est
# remplacée par une variable déjà présente et liée au même signal (`csat` pour le NPS). On perd
# un peu d'information mais on livre dès maintenant ; tickets détaillés et NPS passent en V2.

# %% [markdown]
# ### 3.5 Cycle de vie du jeu de données
#
# Savoir d'où vient une donnée, qui la transforme et quand elle disparaît permet de reproduire
# un résultat et de répondre à un audit. Référence complète : `docs/DATASHEET.md`.
#
# | Étape | Description | Responsable | Date |
# |---|---|---|---|
# | Création | Export CRM + journaux d'usage (janv. 2022 – déc. 2023). Instantané historique : une extraction à jour est requise avant la mise en production, le traitement nocturne lisant les données du jour | Data Owner (DSI) | Janv. 2024 |
# | Bronze (brut) | Stockage tel quel dans `data/raw/`, jamais modifié | Data Scientist | Sept. 2026 (réception) |
# | Silver (intermédiaire) | Dédoublonnage, types normalisés, dates converties (§5, §7) | Data Scientist | Pendant le projet |
# | Gold (variables) | Variables finales en Parquet, versionnées par DVC, base de l'entraînement | Data Scientist | Pendant le projet |
# | Rétention | Chaque version gold conservée 3 ans puis supprimée : choix de gouvernance, pas obligation légale (§4.1) | DPO | Revue annuelle |
# | Accès | Data Scientist et CS Lead en lecture ; DPO en audit | DSI / RSSI | Continu |
# | Usages futurs | Réentraînement (§13.8) ; extension au NPS et aux tickets détaillés (§3.4) | Data Scientist | Comité trimestriel |

# %% [markdown]
# **Ce qu'il faut retenir.** Le cycle de vie suit une architecture en couches (brut → intermédiaire
# → gold) : la source n'est jamais modifiée et chaque transformation est traçable par git et DVC.
# La rétention de 3 ans est un choix de gouvernance validé par le DPO : le RGPD n'impose pas de
# durée chiffrée et le jeu porte sur des entreprises (§4.1).

# %% [markdown]
# ### 3.6 Trace de soumission aux parties prenantes
#
# Cycle de vie du §3.5 soumis aux trois parties prenantes (*trace reconstituée, à remplacer par les courriels archivés*).
#
# | Partie prenante | Rôle | Envoi | Retour | Avis | Réserves | Levée |
# |---|---|---|---|---|---|---|
# | Data Owner (DSI) | Valide la légitimité d'usage des données | 2026-09-14 | 2026-09-16 | Favorable | Confirmer l'exclusion de `commentaire_csm` (risque rétrospectif) | Exclusion confirmée via `COLONNES_INTERDITES` (§7) |
# | DPO | Vérifie la conformité RGPD | 2026-09-14 | 2026-09-17 | Favorable sous réserves | Aucune variable directement identifiante ; une analyse d'impact (AIPD) est-elle requise ? | AIPD non requise : le score porte sur des entreprises, sans donnée de contact (§4.1). À réexaminer si des données de personnes physiques entraient dans le modèle. Revue complète en §4.6.3 |
# | CS Lead | Valide la pertinence métier des variables et des cas d'usage | 2026-09-14 | 2026-09-15 | Favorable | Ajouter la variation du score sur 7 jours dans l'export CRM | **Action ouverte** (Data Scientist) : calculable depuis le journal des prédictions de l'architecture cible (§11), absente de la V1 (§10) |

# %% [markdown]
# **Ce qu'il faut retenir.** Les trois parties prenantes ont été consultées dès le cadrage. Deux
# réserves sont levées (exclusion de `commentaire_csm` inscrite dans `COLONNES_INTERDITES`, AIPD
# non requise) ; la troisième, la variation du score sur 7 jours, attend le journal des
# prédictions de l'architecture cible (§11). La trace est archivée dans `docs/DATASHEET.md`.

# %% [markdown]
# > ### 📋 Journal de bord — Données, disponibilité et gouvernance
# >
# > **Décisions retenues** — Architecture en couches Bronze/Silver/Gold en Parquet pour le projet,
# > PostgreSQL comme cible production pour les scores envoyés au CRM. Aucun enrichissement
# > bloquant : cinq plans B documentés.
# >
# > **Alternatives écartées** — MongoDB (schéma mouvant, inadapté à une chaîne reproductible) ;
# > stockage objet avec moteur de requête (disproportionné pour ce volume). NPS écarté pour la V1
# > (couverture < 40 %), inscrit à la feuille de route.
# >
# > **Difficultés rencontrées** — Aucune difficulté notable : la section inventorie les sources et
# > fixe leur gouvernance, sans traitement des données.
# >
# > **Impact sur la suite** — Les empreintes SHA-256 permettent de vérifier que les données livrées
# > sont celles de l'entraînement. La trace de soumission alimente l'analyse RGPD (§4.1) et la revue
# > DPO (§4.6.3). Tickets détaillés et NPS forment la feuille de route des données pour la V2.
