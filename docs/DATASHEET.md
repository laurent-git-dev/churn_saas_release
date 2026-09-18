# Datasheet — Jeu de données churn SaaS B2B

> Adapté de *Datasheets for Datasets* (Gebru et al., 2018).
> Ce document est la référence de gouvernance du jeu de données utilisé dans le projet CISIA.
> Il est référencé depuis `notebooks/sections/03_donnees_gouvernance.py`.

---

## 1. Motivation

**Pourquoi ce jeu de données a-t-il été créé ?**
Pour permettre l'entraînement d'un modèle de prédiction de résiliation (churn) client dans un
contexte SaaS B2B. L'objectif est de fournir à l'équipe Customer Success un outil d'aide à la
décision basé sur les données d'usage, de support et de facturation.

**Qui a créé ce jeu de données ?**
L'éditeur SaaS (fictif, cadre CISIA). Export réalisé par l'équipe DSI à partir du CRM interne
et des logs d'usage applicatif.

**Financement ?**
Projet interne — aucun financement externe.

---

## 2. Composition

| Champ | Valeur |
|---|---|
| Fichier principal | `churn_saas_complet.csv` |
| Fichier d'échantillon | `churn_saas_echantillon.csv` (50 lignes) |
| Fichier annexe | `catalogue_plans.csv` (tarifs et fonctionnalités par plan) |
| Nombre d'instances | 5 035 lignes (5 000 comptes uniques + 35 doublons) |
| Nombre de variables | 29 colonnes |
| Période couverte | Janvier 2022 – Décembre 2023 |
| Niveau de granularité | Une ligne = un compte client à un instant T |
| Variable cible principale | `churn` (binaire : 1 = résiliation, 0 = renouvellement) |
| Variable cible secondaire | `valeur_vie_client_eur` (régression CLV) |
| Taux de positifs (churn=1) | ~28 % (1 412 / 5 035) |
| Données personnelles | `pays`, `secteur` — pas de donnée directement identifiante |

### Intégrité des fichiers (SHA-256)

> Les hash ci-dessous sont générés par `controler_source()` lors de l'exécution du notebook (§3).
> Les valeurs exactes sont affichées dans la cellule de sortie correspondante.

| Fichier | SHA-256 |
|---|---|
| `churn_saas_complet.csv` | *(voir sortie §3)* |
| `churn_saas_echantillon.csv` | *(voir sortie §3)* |
| `catalogue_plans.csv` | *(voir sortie §3)* |

---

## 3. Processus de collecte

**Comment les données ont-elles été collectées ?**
Export automatisé depuis trois systèmes source :
1. **CRM** (Salesforce) → colonnes client, plan, facturation, dates
2. **Logs d'usage applicatif** → connexions, heures d'usage, fonctionnalités utilisées
3. **Système support** (Zendesk) → tickets, délais de réponse, CSAT

**Mécanisme de collecte :** requêtes SQL planifiées, export CSV mensuel, consolidé en un fichier
unique par la DSI. Aucune collecte via scraping ou API tierce.

**Qui a collecté les données ?**
Équipe DSI (Data Owner). Validation par le CS Lead pour la pertinence métier.

**Cadre temporel :**
Snapshot statique couvrant janvier 2022 à décembre 2023. Non mis à jour depuis.

---

## 4. Pré-traitement et nettoyage

**Des données brutes ont-elles été prétraitées ?**
Non — `data/raw/` contient les fichiers fournis tels quels (figés). Tous les traitements
sont appliqués dans le pipeline scikit-learn (§7) :

| Anomalie | Traitement | Section |
|---|---|---|
| 35 doublons (`client_id`) | `drop_duplicates()` avant tout traitement | §5 |
| Casse hétérogène (3–4 variantes/modalité) | `.str.strip().str.title()` + mapping | §5, §7 |
| Virgule décimale francophone | `.str.replace(",", ".", regex=False)` + cast float | §5, §7 |
| Double format `%` dans `taux_adoption_pct` | Suppression `%` + virgule→point + cast | §5, §7 |
| Dates multi-formats (3 formats) | `pd.to_datetime(..., format="mixed")` | §5, §7 |
| Valeurs manquantes | Imputation médiane/mode **dans le pipeline**, dans chaque pli | §7 |

**Le code source est conservé ?**
Oui — `data/raw/` est immuable. Les transformations sont versionées dans `src/churn_saas/`.

---

## 5. Défauts volontaires (contexte CISIA)

L'énoncé annonce explicitement les défauts suivants, à détecter et traiter dans le notebook :

| Défaut | Variable | Traitement prévu |
|---|---|---|
| Fuite de données | `sante_compte_fin_periode` | Exclusion via `COLONNES_INTERDITES` + démonstration §5 |
| Fuite potentielle | `commentaire_csm` | Exclusion + documentation de l'anomalie §5, §7 |
| Leurres (4) | `couleur_theme_interface`, `code_datacenter`, `groupe_experimentation`, `commentaire_csm` | Preuve chiffrée (importance nulle) en §6 |
| Colinéarité parfaite | `fonctionnalites_total` ↔ `plan` | Exclusion documentée en §7 |
| Doublons | 35 paires sur `client_id` | Déduplication en §5 |

---

## 6. Utilisations

**Usage prévu :** entraînement et évaluation d'un modèle de classification binaire (churn) et
d'un modèle de régression (CLV), dans un cadre de certification CISIA.

**Usages non prévus :**
- Profilage individuel de personnes physiques
- Prise de décision entièrement automatisée sans supervision humaine
- Revente ou diffusion à des tiers

**Limites connues :**
- Snapshot statique : ne reflète pas les évolutions du produit post-2023
- Cohorte unique : un seul éditeur, non généralisable à d'autres marchés SaaS sans validation
- Absence de données d'onboarding détaillées (proxy via `anciennete_mois`)
- NPS non disponible (couverture < 40 %) — proxy via `csat`

---

## 7. Distribution

**Qui peut accéder à ces données ?**
- Accès lecture : Data Scientist en charge du projet, CS Lead (scores uniquement)
- Accès audit : DPO
- Accès externe : aucun (données confidentielles)

**Les données contiennent-elles des informations confidentielles ?**
Oui — données commerciales sensibles (MRR, taux de churn, comportement client).
Elles ne doivent pas être publiées ou partagées hors du cadre du projet.

**Format de distribution pour la certification :**
ZIP livrable CISIA — inclus dans `docs/CHECKLIST_LIVRAISON.md`.

---

## 8. Maintenance

**Qui est responsable du jeu de données ?**
Data Owner (DSI) pour la source brute. Data Scientist pour les versions Silver/Gold.

**Comment le jeu de données sera-t-il mis à jour ?**
Réentraînement trimestriel : export CRM + logs actualisés, versionné via DVC.
Décision de réentraînement prise lors du comité trimestriel (§2, §13).

**Durée de rétention :**
3 ans à compter de la date de création (obligation légale B2B en France).
Après janvier 2027 : anonymisation ou suppression selon décision DPO.

---

## 9. Considérations éthiques et RGPD

**Données personnelles ?**
`pays` et `secteur` sont des données indirectement identifiantes (agrégées au niveau entreprise,
pas individu). Aucune donnée directement identifiante (nom, email, téléphone) n'est présente.

**Analyse d'impact (AIPD) :**
Requise avant déploiement en production si le système influence des décisions commerciales
avec effet juridique ou significatif sur les clients. Prévue en §4 et §10.

**Biais identifiés :**
- Biais de couverture : seuls les clients actuels sont représentés (pas les prospects)
- Biais de sélection : les clients churners avant la période d'observation sont absents
- Biais de mesure : `csat` est déclaratif (non-réponse non aléatoire)

**Évaluation d'équité :**
Mesure des taux de faux positifs/négatifs par sous-groupe (`pays`, `taille_entreprise`, `secteur`)
en §4 et §12.

---

## 10. Trace de soumission aux parties prenantes

*(Simulé — voir note dans `notebooks/sections/03_donnees_gouvernance.py`)*

| Partie prenante | Rôle | Date envoi | Avis | Réserves levées |
|---|---|---|---|---|
| Data Owner (DSI) | Légitimité d'usage | 2024-01-15 | Favorable | Exclusion `commentaire_csm` confirmée |
| DPO | Conformité RGPD | 2024-01-15 | Favorable (sous réserves) | AIPD avant production |
| CS Lead | Pertinence métier | 2024-01-15 | Favorable | Feature `delta_score_7j` ajoutée |

---

*Document rédigé dans le cadre de la certification CISIA — Concevoir et implémenter une solution d'intelligence artificielle.*
*Référence : Gebru, T. et al. (2018). Datasheets for Datasets. arXiv:1803.09010.*
