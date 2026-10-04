# %% [markdown]
# ## 11. Architecture cible
#
# Un modèle n'a de valeur que s'il tourne chaque nuit, fiable, sécurisé et à coût acceptable.
# Cette section décrit le flux de données, compare trois architectures et leur coût, rend compte
# des contraintes de la DSI, du RSSI, du DPO et du Customer Success, puis fixe les SLO.

# %% [markdown]
# ### 11.1 Pipeline de données — vue d'ensemble
#
# Des quatre sources au CRM : ingestion de nuit, pipeline de scoring, puis exposition.

# %%
from IPython.display import Markdown, display

from churn_saas import config, schemas, viz
from churn_saas.format_fr import nombre, pourcentage

_ = viz.sauvegarder(schemas.schema_pipeline_donnees())

# %% [markdown]
# **Ce qu'il faut retenir.** Le batch (traitement par lot) de scoring (calcul des scores) part à
# 02 h 00 (§10) pour que les alertes soient dans le CRM avant 06 h 00 (§2). L'API sert le
# **même** modèle champion : les vues batch et temps réel ne peuvent pas se contredire.

# %% [markdown]
# ### 11.2 Architecture cible — scénario B retenu
#
# Qui parle à qui en production : chaque cadre est une zone de responsabilité distincte.

# %%
_ = viz.sauvegarder(schemas.schema_architecture_cible())

# %% [markdown]
# **Ce qu'il faut retenir.** Passerelle exposée, calcul conteneurisé sans serveur à administrer,
# données managées et registre (registry) de modèles alimenté **uniquement** par la CI/CD. Aucune
# brique ne demande Kubernetes et chacune a un équivalent standard ailleurs (Docker, Parquet,
# PostgreSQL) : c'est la réversibilité exigée par le RSSI.

# %% [markdown]
# ### 11.3 Diagramme de séquence — score à la demande (CSM → API → CRM)
#
# Cas d'usage 3 du §2 : un CSM ouvre une fiche compte ; le CRM interroge l'API et affiche le
# score pendant le chargement de la fiche.

# %%
_ = viz.sauvegarder(schemas.schema_sequence_score())

# %% [markdown]
# **Ce qu'il faut retenir.** L'inférence pèse peu dans le budget de 200 ms (mesure au §9.13,
# rappelée au §12.3) ; l'explication SHAP réutilise le même passage du prétraitement. Le budget
# va surtout au réseau et à la persistance. Le `client_id` ne quitte jamais le CRM (§10.8). La
# cible fait **évoluer par ajout** l'API du §10 (même route, même contrat d'entrée) :
#
# | Élément | API livrée (§10) | Architecture cible |
# |---|---|---|
# | Authentification | Clé `X-API-Key`, 60 requêtes/min par IP | JWT validé par l'API Gateway |
# | Raisons du score | 3 variables SHAP exactes (`facteurs_shap`) et 3 signaux métier issus de l'EDA (§6) | Inchangé |
# | Explication détaillée | Absente (3 raisons du risque seulement) | `POST /explain` : toutes les variables, facteurs protecteurs compris |
# | Journal d'audit | Absent (l'API ne persiste rien) | Chaque appel journalisé dans PostgreSQL, 12 mois |

# %% [markdown]
# ### 11.4 Compte-rendu d'entretien avec les acteurs
#
# *Trace reconstituée à partir des contraintes typiques d'un éditeur SaaS B2B français de taille
# intermédiaire ; elle est à remplacer par le compte-rendu archivé des entretiens.*
#
# | Acteur | Contraintes remontées | Conséquence sur l'architecture |
# |---|---|---|
# | **DSI** | Budget pilote ≤ 600 €/mois.<br>Équipe ops sans compétence Kubernetes.<br>API REST standard, sans SDK propriétaire.<br>Disponibilité ≥ 99,5 % de 06 h à 20 h. | Services entièrement managés (ECS Fargate ou Cloud Run).<br>Scénario A écarté (pas de haute disponibilité), scénario C écarté (budget, dépendance fournisseur). |
# | **RSSI** | Hébergement UE, fournisseur certifié ISO 27001.<br>Chiffrement en transit (TLS 1.2+) et au repos (AES-256).<br>Journalisation des appels conservée 12 mois. | Région Paris imposée (`eu-west-3` ou `europe-west9`), jamais la région par défaut d'un SDK.<br>Une région UE fixe le lieu, pas le droit applicable : un fournisseur américain reste soumis au Cloud Act. Accepté pour des indicateurs d'entreprise (R10, §4) ; un cloud qualifié SecNumCloud s'imposerait si des données personnelles entraient dans le périmètre.<br>WAF obligatoire, journal d'audit en base exporté vers le SIEM. |
# | **DPO** | Scoring de personnes morales : article 22 du RGPD hors champ (§4), mais les contacts CRM restent des données personnelles.<br>Minimisation ; conservation des scores 24 mois (§4.1). | Base scores limitée à `client_id`, sans nom ni e-mail.<br>Purge hebdomadaire des scores de plus de 24 mois.<br>Explication de chaque score maintenue **par choix de gouvernance** (humain dans la boucle, §4). |
# | **Responsable Customer Success** | Score visible dans Salesforce, avec 3 signaux explicatifs.<br>Alertes disponibles avant 06 h 00.<br>Indicateur d'incertitude pour les comptes de moins de 3 mois. | Champ Salesforce alimenté par le batch et par `POST /predict` à l'ouverture de la fiche.<br>Batch à 02 h 00 (§10).<br>Champ `incertitude_score` prévu en cible. |
#
# La **généralisation** est la capacité du système à fonctionner au-delà du pilote (5 000
# comptes, un produit, un data scientist à mi-temps). Les entretiens en ont fixé les limites :
#
# | Axe | Contrainte | Réponse architecturale |
# |---|---|---|
# | Volume (5 k → 50 k comptes) | Monter en charge sans refonte | Mise à l'échelle automatique des conteneurs ; scénario C réévalué au-delà de 50 k (§11.7) |
# | Portabilité | Pas de dépendance à un fournisseur | Images Docker, Parquet et PostgreSQL standards |
# | Réplicabilité (autres produits) | Séparer configuration et code | Un `config.py` par produit, pipeline réutilisé tel quel |
# | Robustesse temporelle | Détecter la dérive avant la dégradation | PSI hebdomadaire et SLO de qualité par cohorte (§11.8, §13.8) |
# | Compétences | Pas de spécialiste MLOps | Services managés et runbook d'incident (`docs/RUNBOOK.md`) |

# %% [markdown]
# **Ce qu'il faut retenir.** Les quatre acteurs convergent vers le même scénario : conteneurs
# managés, région Paris, intégration native dans Salesforce. L'explication de chaque score reste
# un choix de gouvernance, mis en œuvre au §12.8.

# %% [markdown]
# ### 11.5 Contraintes techniques, réglementaires et organisationnelles
#
# En résumé, chaque contrainte devient une décision vérifiable. **Techniques** : p95 < 200 ms
# (pipeline en mémoire, cache d'une heure), alertes avant 06 h 00 (batch à 02 h 00, alerte à
# 04 h 00), reproductibilité (`config.RANDOM_SEED`, DVC, MLflow). **Réglementaires** : région
# Paris explicite, base scores réduite au `client_id`, purge à 24 mois, journaux 12 mois.
# **Organisationnelles** : orchestration managée et budget ≤ 600 €/mois, qui départage les
# trois scénarios ci-dessous.

# %% [markdown]
# ### 11.6 Trois scénarios comparatifs
#
# | Critère | A — VM unique + cron | **B — Conteneurs managés ★** | C — Plateforme ML managée |
# |---|---|---|---|
# | Description | Une VM Linux, scoring par cron, API simple | ECS Fargate ou Cloud Run, PostgreSQL managé, MLflow, CI/CD GitHub Actions | SageMaker, Vertex AI ou Azure ML : inférence, registre et A/B test intégrés |
# | Coût mensuel (pilote 5 k comptes) | 30 – 60 € | **90 – 110 €** | 800 – 2 500 € |
# | Compétences requises | Linux, Python, cron | Docker, cloud de base, CI/CD | SDK propriétaire du fournisseur |
# | Hébergement UE | Oui | Oui, région Paris imposée | Partiel : services annexes hors UE par défaut |
# | Haute disponibilité, retour arrière du modèle | ❌ | ✅ | ✅ |
# | Réversibilité | Maximale | Haute (Docker, Parquet) | Faible (pipelines propriétaires) |
# | Limite critique | Point unique de défaillance, inadapté au-delà de 20 k comptes | MLflow à maintenir (2 – 4 h/mois) | Budget au moins 7 fois celui de B, dépendance fournisseur |

# %% [markdown]
# **Ce qu'il faut retenir.** A suffit pour une preuve de concept, pas pour la production ; C
# dépasse le budget et crée la dépendance que refuse le RSSI. **B** apporte haute disponibilité,
# retour arrière du modèle et CI/CD pour quelques dizaines d'euros de plus que A.

# %% [markdown]
# ### 11.7 Recommandation argumentée
#
# **Scénario retenu : B**, seul à respecter budget, compétences, hébergement UE et réversibilité
# tout en offrant haute disponibilité et gate (règle de promotion) automatisée.
#
# **Coûts du scénario B** (AWS `eu-west-3`, pilote 5 000 comptes, tarifs publics 2025) :
#
# | Poste | Coût mensuel estimé |
# |---|---|
# | Calcul — API 24 h/24 (≈ 30 €) et batch de moins de 5 min par nuit (≈ 1 €) | ≈ 31 € |
# | PostgreSQL managé (≈ 25 €), stockage objet (≈ 10 €), MLflow (5 – 15 €) | ≈ 40 – 50 € |
# | API Gateway, WAF et journaux | ≈ 20 – 30 € |
# | **Total** (CI/CD incluse dans l'offre gratuite de GitHub Actions) | **≈ 90 – 110 €** |
#
# À 20 k comptes avec redondance multi-zones, prévoir 300 – 450 €/mois ; une réservation d'un an
# réduit la facture de calcul de 30 à 40 %. Ces montants sont à confirmer par un devis.
# **Trajectoire** : local (`Makefile`) → ECS Fargate à T+2 mois → Evidently et test champion /
# challenger à T+6 mois → scénario C réévalué à T+12 mois au-delà de 50 k comptes.
#
# > 📧 **Arbitrage — comité du 22 septembre 2026 (DSI, direction générale, RSSI).** Scénario B
# > validé ; ~100 €/mois, faible face au gain de la cohorte mensuelle (§12.12). Le RSSI ajoute
# > 15 – 20 €/mois (chiffrement au repos, journaux 12 mois). *Trace reconstituée.*

# %% [markdown]
# ### 11.8 SLO / SLI et procédures de remédiation
#
# On veut savoir chaque jour si le service tient ses promesses. Un **SLO** (objectif de niveau
# de service) fixe le niveau visé ; le **SLI** (indicateur) le mesure. Les seuils du modèle
# viennent de `config.CIBLES_PERFORMANCE`. Le PSI (indice de stabilité de population) mesure
# l'écart de distribution d'une variable entre production et entraînement.

# %%
_c = config.CIBLES_PERFORMANCE
_pr_auc_min, _pr_auc_critique = nombre(_c["pr_auc_min"], 2), nombre(_c["pr_auc_critique"], 2)
_recall_min = pourcentage(_c["recall_vigilance_min"], 0)

display(Markdown(f"""
| SLI | SLO | Alerte | Critique | Remédiation |
|---|---|---|---|---|
| **Fraîcheur des scores** | Âge ≤ 26 h | Batch du jour absent à 04 h 00 | Absent à 06 h 00 : incident P1, alertes non livrées | Logs Prefect, relance manuelle ; source indisponible → CSM prévenus |
| **Latence API p95** | < 200 ms (fenêtre 1 h) | > 150 ms | > 200 ms | Charge CPU, rechargement du cache, tâche supplémentaire |
| **Disponibilité API** | 99,5 % de 06 h à 20 h | < 99,8 % sur 1 h | < 99,0 % sur 30 min : P1 | Relance automatique ; sinon scores de la veille servis en mode dégradé |
| **Dérive (PSI)** | < 0,20 sur chaque variable | 0,10 à 0,20 : comité trimestriel | ≥ 0,20 : comité ad hoc (§2.6) | Erreur de collecte → corriger la source ; changement réel → réentraînement (§13.8) |
| **Qualité (PR-AUC)** | ≥ {_pr_auc_min} par cohorte de renouvellements (§13.7) | < {_pr_auc_min} : comité ad hoc | < {_pr_auc_critique} : comparaison au champion précédent | Retour arrière s'il fait mieux ; challenger promu seulement via la gate (§13.8) |
| **Recall au seuil de vigilance** | ≥ {_recall_min} des churners signalés (§12.6) | < {_recall_min} : comité ad hoc | — | Recalcul du seuil de vigilance sur la cohorte récente |
"""))

# %% [markdown]
# **Ce qu'il faut retenir.** Six indicateurs couvrent l'infrastructure et le modèle, chacun avec
# une remédiation graduée pour ne jamais décider en silence en mode dégradé. Le recall (rappel)
# au seuil de vigilance surveille directement les faux négatifs que la règle à deux niveaux veut
# minimiser. Un signal sur le modèle convoque un comité plutôt qu'un réentraînement seul (§2.6,
# table d'actions du §12.15).

# %% [markdown]
# ### 11.9 Acteurs impliqués dans le fonctionnement continu (run)
#
# | Acteur | Rôle | Interface | Formation |
# |---|---|---|---|
# | CSM | Consulte les scores, mène les actions de rétention | Salesforce, tableau de bord | 1 h : lire le score et ses 3 facteurs |
# | Responsable CS | Valide les seuils, participe au comité trimestriel | Liste priorisée, KPI métier | 2 h : arbitrage des seuils |
# | Équipe Data / ML | Maintient le pipeline, surveille les SLI, déploie via la CI/CD | MLflow, Prefect, Evidently | Aucune |
# | Ops / SRE | Infrastructure, correctifs, incidents P1 | Console cloud, `docs/RUNBOOK.md` | 2 h : runbook |
# | DPO | Registre des traitements, audit de la purge | Journaux d'audit, rapport de purge | 1 h |
# | RSSI | Conformité sécurité, revue annuelle | SIEM, scan des images Docker (à ajouter à la CI) | Aucune |

# %% [markdown]
# **Ce qu'il faut retenir.** Seule l'équipe Data / ML touche au pipeline ; les autres acteurs
# lisent, sans compétence ML requise : c'est le principe du moindre privilège.

# %% [markdown]
# > ### 📋 Journal de bord — Architecture cible
# >
# > **Décisions retenues** — Scénario B (conteneurs managés, région Paris, ≈ 90 – 110 €/mois) ;
# > six SLO gradués, dont le recall au seuil de vigilance ; écarts API livrée / cible au §11.3.
# >
# > **Alternatives écartées** — Scénario A : point unique de défaillance, pas de retour arrière.
# > Scénario C : au moins 800 €/mois et dépendance fournisseur contraire à la réversibilité.
# >
# > **Difficultés rencontrées** — Chiffrage sur tarifs publics, non devisé ; entretiens
# > reconstitués. L'API livrée reste en deçà de la cible : ni authentification JWT, ni route
# > `/explain`, ni journal d'audit (§11.3).
# >
# > **Impact sur la suite** — Les SLO (alertes avant 06 h 00, p95 < 200 ms, PR-AUC et recall
# > minimaux) alimentent le monitoring (surveillance) et le plan de réentraînement du §13.
