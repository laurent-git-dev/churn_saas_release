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
| Nombre d'instances | 5 035 lignes : 5 000 comptes uniques et 35 copies exactes (70 lignes en groupes de doublons) |
| Nombre de variables | 29 colonnes |
| Période couverte | Export déclaré de janvier 2022 à décembre 2023 ; souscriptions observées du 15/02/2022 au 10/01/2025 et extraction estimée au 04/02/2025 (§3.5, §5.8, §6.11) : incohérence à lever avec le Data Owner |
| Niveau de granularité | Une ligne = un compte client à un instant T |
| Variable cible principale | `churn` (binaire : 1 = résiliation, 0 = renouvellement) |
| Variable cible secondaire | `valeur_vie_client_eur` (régression CLV) |
| Taux de positifs (churn=1) | 28,0 % après dédoublonnage (1 400 / 5 000) |
| Données personnelles | Aucune dans le jeu modélisé : les comptes sont des entreprises ; `commentaire_csm` (13 libellés normalisés, sans nom, e-mail ni téléphone) est exclu par minimisation (§6.6) |

### Intégrité des fichiers (SHA-256)

> Empreintes de référence relevées le 2026-10-11 par `controler_source()`, la fonction que le
> notebook exécute en §3.1 : une empreinte différente signale un fichier modifié.

| Fichier | Lignes (en-tête incl.) | SHA-256 |
|---|---|---|
| `churn_saas_complet.csv` | 5 036 | `2fc45e8c3c74d3ddcdbda162c5bb50184587201b46d877cc06742afd13063844` |
| `churn_saas_echantillon.csv` | 51 | `680f5371c406f47a3a67f1611cb016c17efd61e93a52d1b4b8dbd3a39dc2f296` |
| `catalogue_plans.csv` | 5 | `c94f6f52652532c578cb7a44d752b8b197ef84f5e6c6e17a58579fa195e9ad08` |

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
Photographie unique du portefeuille, déclarée comme couvrant janvier 2022 à décembre 2023 ; les
dates de souscription vont pourtant jusqu'en janvier 2025 (voir Composition). Non mise à jour
depuis.

---

## 4. Pré-traitement et nettoyage

**Des données brutes ont-elles été prétraitées ?**
Non — `data/raw/` contient les fichiers fournis tels quels (figés). Les corrections **sans
apprentissage** sont faites à la construction du jeu gold (`churn_saas.cli.construire_gold_dataset`,
§7.3) ; tout ce qui **apprend** des données (imputation, encodage, standardisation, agrégat de
groupe) reste dans le pipeline scikit-learn, appris dans chaque pli (§7.2, §7.4) :

| Anomalie | Traitement | Section |
|---|---|---|
| 35 copies exactes (70 lignes en groupes) ; aucun doublon divergent sur `client_id` | `drop_duplicates(keep="first")` ; contrôle bloquant si un doublon sur `client_id` apparaissait | §5.5, §7.3.1 |
| Casse et espaces hétérogènes (`secteur` 28 → 7 modalités, `plan` 16 → 4, `taille_entreprise` 12 → 4) | `NormalisationCategorielle` dans le pipeline (même règle à l'entraînement, dans le batch et l'API) ; le gold garde les libellés d'origine | §6.3, §7.3.5 |
| Nombres stockés en texte (virgule décimale, symbole `%`) dans 4 colonnes | `coercer_numeriques()` : 6 529 valeurs réparées, aucune irrécupérable | §5.6, §7.3.2 |
| Dates en 3 formats, 962 ambiguës | `parser_dates()` : cascade de formats valeur par valeur ; convention `JJ/MM/AAAA` démontrée par les données | §5.7, §7.3.3 |
| Valeurs hors domaine | Aucune violation ; bornages `taux_adoption_pct` ∈ [0, 100] et `utilisateurs_actifs` ≤ `sieges_souscrits` | §5.8, §7.3.4 |
| Valeurs manquantes (10 colonnes, toutes MCAR) | Médiane pour les numériques, modalité `inconnu` pour les catégorielles, **dans le pipeline** ; `taux_adoption_pct` recalculé exactement quand c'est possible | §5.9, §7.4 |
| Feature engineering | 15 variables calculées ligne par ligne, dont les cinq ratios reliés aux hypothèses H1 à H14 (`config.HYPOTHESES_METIER`) ; écart du CSAT au secteur appris dans le pipeline (`EcartAuGroupe`) | §7.5 |

**Jeu gold produit.** `data/gold/gold_dataset.parquet` (5 000 lignes après déduplication,
57 colonnes au 2026-10-10 : 29 brutes, variables dérivées, catalogue et enrichissement),
versionné par DVC avec ses métadonnées (`gold_metadata.json`, §7.8).

**Le code source est conservé ?**
Oui — `data/raw/` est immuable. Les transformations sont versionnées dans `src/churn_saas/`.

---

## 5. Défauts volontaires (contexte CISIA)

L'énoncé annonce explicitement les défauts suivants, à détecter et traiter dans le notebook :

| Défaut | Variable | Traitement prévu |
|---|---|---|
| Fuite de données | `sante_compte_fin_periode` | Exclusion via `COLONNES_INTERDITES` + démonstration §6.5 |
| Fuite potentielle | `commentaire_csm` | Audit §6.6 (date de rédaction inconnue, redondance, minimisation) → `COLONNES_INTERDITES`. Pas un leurre au sens strict |
| Leurres (3) | `couleur_theme_interface`, `code_datacenter`, `groupe_experimentation` | Criblage §6.10, trois preuves convergentes en §12.9 (association marginale, permutation, drop-column : 3 leurres confirmés sur 3), retrait du modèle déployé §10.3 |
| Redondance | `fonctionnalites_total` ↔ `plan` (bijection) | Conservée : sert au taux de couverture fonctionnelle (§7.5) ; colinéarité absorbée par la régularisation de la régression logistique |
| Doublons | 35 copies exactes | Déduplication en §5.5 et §7.3.1 |

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
- Tickets support disponibles sur 90 jours seulement, sans historique : les ratios qui les
  rapportent à l'ancienneté (`intensite_support`) mêlent deux échelles de temps (corrélation de
  rang −0,42 avec `anciennete_mois`, §7.5)
- Une seule extraction datée et horizon du départ non précisé : pas de validation sur une
  période future (§6.11)

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
ZIP livrable CISIA, qui embarque les données ; elles restent hors du dépôt git.

---

## 8. Maintenance

**Qui est responsable du jeu de données ?**
Data Owner (DSI) pour la source brute. Data Scientist pour les versions Silver/Gold.

**Comment le jeu de données sera-t-il mis à jour ?**
Réentraînement trimestriel : export CRM + logs actualisés, versionné via DVC.
Décision de réentraînement prise lors du comité trimestriel (§2, §13).

**Durée de rétention :**
3 ans à compter de la création de chaque version, puis suppression. Choix de gouvernance validé
par le DPO et revu chaque année : le RGPD n'impose pas de durée chiffrée (art. 5.1.e), et le jeu
porte sur des personnes morales. Repère : référentiel CNIL « gestion commerciale » (3 ans).
Scores et prédictions : 24 mois ; journaux techniques : 12 mois (notebook §4.1).

---

## 9. Considérations éthiques et RGPD

**Données personnelles ?**
Non dans le jeu modélisé : `pays`, `secteur` et `taille_entreprise` sont des attributs
d'entreprise, pas des données personnelles (§4.4). Le seul champ à risque, `commentaire_csm`, ne
contient aucun nom, e-mail ni téléphone (13 libellés normalisés, contrôle exhaustif, §6.6) et est
exclu par minimisation. Les coordonnées des contacts opérationnels restent dans le CRM, hors du
modèle (§4.1).

**Analyse d'impact (AIPD) :**
Non requise : le score porte sur des entreprises, sans donnée de contact (§3.6, §4.1). À
réexaminer si des données de personnes physiques entraient dans le modèle (revue DPO, §4.6.3).

**Biais identifiés :**
- Biais de couverture : seuls les clients actuels sont représentés (pas les prospects)
- Biais de sélection : les clients churners avant la période d'observation sont absents
- Biais de mesure : `csat` est déclaratif ; sa non-réponse n'est pas liée au churn de façon détectable (MCAR, 25,8 % de churn sans note, §5.9), ce qui n'exclut pas un mécanisme plus subtil

**Évaluation d'équité :**
Écart de recall (TPR) et de calibration entre modalités de `pays`, `taille_entreprise` et
`secteur` : protocole fixé en §4.4 (seuils 15 et 5 points), mesure sur le modèle final en §12.14.
Au 2026-10-10, aucun seuil franchi ; plus grand écart de TPR : 12,4 points (secteur Finance).

---

## 10. Trace de soumission aux parties prenantes

*(Trace reconstituée, à remplacer par les courriels archivés — détail en §3.6 du notebook)*

| Partie prenante | Rôle | Envoi | Retour | Avis | Réserve | Statut |
|---|---|---|---|---|---|---|
| Data Owner (DSI) | Légitimité d'usage | 2026-09-14 | 2026-09-16 | Favorable | Confirmer l'exclusion de `commentaire_csm` | Levée : exclusion via `COLONNES_INTERDITES` |
| DPO | Conformité RGPD | 2026-09-14 | 2026-09-17 | Favorable sous réserves | Une AIPD est-elle requise ? | Levée : AIPD non requise (entreprises, sans donnée de contact) |
| CS Lead | Pertinence métier | 2026-09-14 | 2026-09-15 | Favorable | Ajouter la variation du score sur 7 jours dans l'export CRM | **Ouverte** : absente de la V1, attend le journal des prédictions (§11) |

---

*Document rédigé dans le cadre de la certification CISIA — Concevoir et implémenter une solution d'intelligence artificielle.*
*Référence : Gebru, T. et al. (2018). Datasheets for Datasets. arXiv:1803.09010.*
