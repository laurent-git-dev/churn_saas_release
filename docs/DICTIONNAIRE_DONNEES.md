# Dictionnaire des données — churn_saas_complet.csv

Source : profil produit par `churn_saas.data.quality.profil_compact()` sur
`data/raw/churn_saas_complet.csv` (5 035 lignes, 29 colonnes, 35 copies exactes). La colonne
« traitement » décrit ce que fait réellement le code (§5 à §7), valeurs au 2026-10-10.

> **Légende risques** — `fuite` : variable connue après la décision de prédiction ;
> `identifiant` : clé technique sans valeur prédictive ;
> `leurre` : variable annoncée sans lien avec le churn (à démontrer chiffres à l'appui) ;
> `leurre + fuite potentielle` : leurre annoncé, mais dont les libellés pourraient avoir été saisis après le départ ;
> `colinéaire` : information déjà portée par une autre variable ;
> `donnée personnelle` : donnée à caractère personnel (RGPD) ;
> `aucun` : aucun risque identifié.

---

## Tableau principal (29 colonnes)

| # | nom | type observé | type cible | unité | sémantique métier | pertinence a priori pour le churn | risque | traitement prévu |
|---|---|---|---|---|---|---|---|---|
| 1 | `client_id` | str | — | — | Identifiant unique du compte client (format CLI-XXXXXX). 5 000 valeurs distinctes pour 5 035 lignes → 35 copies exactes, aucun doublon divergent. | **Nulle** — identifiant technique, aucune valeur prédictive ; risque de mémorisation d'entité si conservé. | `identifiant` | Exclue via `COLONNES_INTERDITES`. Clé du contrôle d'unicité (bloquant) après déduplication (§5.5, §7.3.1). |
| 2 | `date_souscription` | str | datetime | — | Date de début de contrat. **Trois formats coexistent** : `JJ Mon AAAA` (1 709 lignes), `JJ/MM/AAAA` (1 621), `AAAA-MM-JJ` (1 705). 0 % manquants. | **Faible** par elle-même ; utile pour dériver la saisonnalité (mois/trimestre de souscription) et vérifier la cohérence avec `anciennete_mois`. | `aucun` | `parser_dates()` : cascade de formats valeur par valeur, convention `JJ/MM/AAAA` démontrée par les données (962 dates ambiguës, §5.7). **Non modélisée** : n'apporte que l'ancienneté (ρ = −0,998, §6.11). |
| 3 | `jour_souscription` | str (catégorie) | str → catégorie | — | Jour de la semaine de la souscription (lundi–dimanche, 7 valeurs). 0 % manquants. Redondant avec `date_souscription` une fois parsée. | **Faible** — le jour de la semaine n'a pas de justification métier directe comme signal de churn ; redondance avec `date_souscription`. | `aucun` | **Non modélisée** (`features.build.COLONNES_NON_MODELISEES`) : sans hypothèse métier ni association significative avec le churn (§6.8). |
| 4 | `secteur` | str | str → catégorie | — | Secteur d'activité du client. **28 valeurs observées** pour 7 secteurs réels, dues à la casse et aux espaces hétérogènes (Commerce / COMMERCE / commerce). 5,0 % manquants. | **Moyenne** — certains secteurs ont des cycles économiques propres (ex. : la Tech renouvelle différemment du Public). | `aucun` | Normalisation casse/espaces dans le pipeline (`NormalisationCategorielle`) ; manquants → modalité `inconnu` ; one-hot. Clé de l'écart CSAT au secteur (`EcartAuGroupe`) et de l'enrichissement sectoriel (§7.5, §7.7). |
| 5 | `pays` | str (catégorie) | str → catégorie | — | Pays d'implantation du client (6 valeurs : France, Belgique, Allemagne…). 4,0 % manquants. | **Moyenne** — effets réglementaires, culturels et économiques sur l'adoption et le renouvellement. | `aucun` | Manquants → modalité `inconnu` ; one-hot (6 modalités). Clé de l'enrichissement par pays (§7.7). |
| 6 | `taille_entreprise` | str | str → catégorie | — | Taille de l'entreprise (TPE / PME / ETI / GE). **12 valeurs observées** pour 4 catégories réelles : variantes de casse et espaces préfixes (` PME`, `pme`, `PME`). 0 % manquants. | **Forte** — la taille conditionne la capacité d'adoption, l'implication du IT, et la criticité du produit. | `aucun` | Normalisation casse/espaces (12 → 4 modalités) ; one-hot (aucun encodage ordinal, §6.3). |
| 7 | `plan` | str | str → catégorie | — | Plan souscrit (Starter / Pro / Business / Enterprise). **16 valeurs observées** pour 4 plans réels : variantes de casse et espaces. 0 % manquants. | **Forte** — détermine le MRR, les fonctionnalités accessibles et les engagements contractuels. Corrélé à `fonctionnalites_total`. | `aucun` | Normalisation casse/espaces (16 → 4 modalités) ; one-hot ; clé de jointure du catalogue des plans (§7.6). |
| 8 | `anciennete_mois` | int64 | int64 | mois | Durée en mois depuis la souscription (1–36 mois). 0 % manquants. | **Forte** — l'ancienneté est un proxy de la loyauté ; les churners en début de cycle ont un profil distinct (onboarding raté). | `aucun` | Conservée ; sert à `tranche_anciennete` (H14), `intensite_support` et `recence_normalisee` (§7.5). |
| 9 | `sieges_souscrits` | int64 | int64 | sièges | Nombre de licences achetées (1–898). 0 % manquants. Distribution longue queue. | **Forte** — proxy de l'engagement financier et de la taille de déploiement interne. | `aucun` | Conservée ; sert à `taux_utilisation_sieges`, `surdimensionnement` et `arpu_par_siege` (§7.5) ; borne de `utilisateurs_actifs`. |
| 10 | `utilisateurs_actifs` | int64 | int64 | utilisateurs | Nombre d'utilisateurs actifs sur le compte (0–829). 0 % manquants. Jamais supérieur à `sieges_souscrits` (ratio max = 1,0). | **Forte** — signal direct d'adoption réelle ; un faible ratio siège/actifs signale un déploiement en échec. | `aucun` | Conservée, bornée à `sieges_souscrits` (§7.3.4) ; sert à `taux_utilisation_sieges` (H9). |
| 11 | `taux_adoption_pct` | str | float64 | % | Taux d'adoption (0–100). **Double format** : séparateur décimal mixte (`.` vs `,`) **et** présence optionnelle du symbole `%` (`50.0%`, `50,0`, `50.0`). 5,0 % manquants. Seulement 57,8 % des valeurs non-nulles sont convertibles directement. | **Forte** — indicateur composite d'engagement ; faible adoption → risque de churn élevé. | `aucun` | `coercer_numeriques()` (2 020 valeurs réparées) ; bornée à [0, 100] ; recalculée exactement par `utilisateurs_actifs / sieges_souscrits` (`reconstituer_taux_adoption`, §7.4). |
| 12 | `connexions_30j` | int64 | int64 | connexions | Nombre de connexions sur les 30 derniers jours (0–156). 0 % manquants. | **Forte** — signal d'activité récent le plus direct ; 0 connexion est un signal d'alarme fort. | `aucun` | Conservée ; sert à `connexions_par_utilisateur` (H10). |
| 13 | `heures_usage_30j` | str | float64 | heures | Heures d'utilisation sur les 30 derniers jours (0,0–170,2). **Virgule décimale francophone** sur 1 402 valeurs (30 % des non-nuls). 6,1 % manquants. | **Forte** — signal d'intensité d'usage, complémentaire de `connexions_30j` (durée vs fréquence). | `aucun` | `coercer_numeriques()` ; imputation médiane dans le pipeline ; indicateur `heures_usage_30j_manquant` (§7.5). |
| 14 | `fonctionnalites_total` | int64 | int64 | nombre | Nombre total de fonctionnalités disponibles dans le plan (4 valeurs : 8, 16, 26, 40). 0 % manquants. **Entièrement déterminé par `plan`** : Starter=8, Pro=16, Business=26, Enterprise=40 (bijection parfaite confirmée). | **Nulle** — colinéaire à 100 % avec `plan` ; aucune information ajoutée après encodage du plan. | `colinéaire` | **Conservée** comme dénominateur de `taux_couverture_fonctionnelle` (H12, §7.5) ; la régularisation L2 absorbe la redondance avec `plan`. |
| 15 | `fonctionnalites_utilisees` | int64 | int64 | nombre | Nombre de fonctionnalités effectivement utilisées (0–40). 0 % manquants. | **Forte** — mesure de la profondeur d'adoption (breadth of use) ; complémentaire à `taux_adoption_pct`. | `aucun` | Conservée ; sert à `taux_couverture_fonctionnelle` (H12). |
| 16 | `nb_integrations` | float64 | int64 | intégrations | Nombre d'intégrations tierces actives (0–16). 4,0 % manquants. | **Forte** — les intégrations créent de la *stickiness* (coût de migration élevé) ; puissant signal anti-churn. | `aucun` | Imputation médiane dans le pipeline ; sert à `tranche_integrations` (H11). |
| 17 | `derniere_connexion_jours` | int64 | int64 | jours | Nombre de jours depuis la dernière connexion (0–200). 0 % manquants. | **Forte** — signal d'inactivité ; clé du churn précoce. Complémentaire de `connexions_30j`. | `aucun` | Conservée ; sert à `compte_dormant` (> 30 jours, H5) et `recence_normalisee`. |
| 18 | `tickets_support_90j` | int64 | int64 | tickets | Nombre de tickets support ouverts sur 90 jours (0–17). 0 % manquants. | **Forte** — signal de friction (direction ambiguë : peu de tickets peut signaler peu d'usage *ou* faible friction). | `aucun` | Conservée ; sert à `intensite_support` (H6) et `pression_support`. |
| 19 | `delai_reponse_support_h` | str | float64 | heures | Délai moyen de réponse du support (0,5–56,5 h). **Virgule décimale francophone** dans une partie des valeurs : 1 394 réparées, aucune irrécupérable (§5.6). 10,0 % manquants. | **Moyenne** — la qualité du support influence la satisfaction ; signal secondaire. | `aucun` | `coercer_numeriques()` ; imputation médiane dans le pipeline ; indicateur `delai_reponse_support_h_manquant`. |
| 20 | `csat` | float64 | float64 | score (1–5) | Score de satisfaction client CSAT (1–5, 5 valeurs entières). 8,0 % manquants. | **Forte** — signal direct et déclaratif de la satisfaction ; corrélé au risque de churn. | `aucun` | Imputation médiane dans le pipeline (manquance MCAR : 25,8 % de churn sans note, §5.9) ; indicateur `csat_manquant` ; sert à `ecart_csat_secteur` (H7). |
| 21 | `retards_paiement_12m` | float64 | int64 | retards | Nombre de retards de paiement sur 12 mois (0–7). 5,0 % manquants. | **Forte** — signal de friction financière et d'intention de non-renouvellement. | `aucun` | Imputation médiane dans le pipeline. |
| 22 | `revenu_mensuel_recurrent_eur` | str | float64 | EUR | MRR mensuel en euros. **Double format décimal** : point anglosaxon (`171.02`) et virgule francophone (`40,0`, `2000,67`). 1 713 valeurs réparées. 3,0 % manquants. | **Forte** — détermine l'importance économique du compte ; corrélé au plan et à la taille. Utile pour prioriser les interventions. | `aucun` | `coercer_numeriques()` ; imputation médiane pour le modèle, **jamais pour un montant** : sans MRR, valeur à risque `null` et compte en veille (§10.2, §12.6). Sert à `arpu_par_siege`. |
| 23 | `couleur_theme_interface` | str (catégorie) | — (exclure) | — | Couleur de l'interface choisie par l'utilisateur (5 valeurs : clair, violet, bleu…). 0 % manquants, répartition uniforme. | **Nulle** — préférence cosmétique sans lien causal avec la résiliation. Leurre annoncé. | `leurre` | Conserver dans le gold et le champion pour produire la preuve chiffrée d'inutilité (association marginale, permutation importance, drop-column importance, §12.9) ; exclure du modèle déployé (§10.3). |
| 24 | `code_datacenter` | str (catégorie) | — (exclure) | — | Région d'hébergement (4 valeurs : eu-w1, eu-w3, us-e1, ap-s1, dont deux hors UE, risque R10). 0 % manquants. Aucune redondance avec une autre variable, `pays` compris (redondance maximale 0,08, §6.10). | **Nulle** — choix d'infrastructure sans corrélation causale avec le churn. Leurre annoncé. | `leurre` | Même traitement que `couleur_theme_interface`. |
| 25 | `groupe_experimentation` | str (catégorie) | — (exclure) | — | Appartenance à un groupe d'A/B test (control / A / B, quasi-équirépartis : 34 % / 33 % / 33 %). 0 % manquants. | **Nulle** — leurre annoncé. Si l'A/B test avait un effet réel sur le churn, inclure cette variable biaiserait le modèle en production (les futurs clients n'appartiennent à aucun groupe). | `leurre` | Même traitement que `couleur_theme_interface`. |
| 26 | `commentaire_csm` | str (13 valeurs) | — (exclure) | — | Annotation textuelle du CSM. **55,4 % manquants**. Seulement 13 valeurs distinctes malgré 2 245 non-nuls. ⚠️ Les libellés encodent explicitement le risque perçu (« Client insatisfait, risque de départ. », « Usage en forte baisse ce trimestre. »). | **Suspecte** — bien que leurre annoncé, les 13 catégories sont fortement associées au churn (V de Cramér 0,435, §6.10). **Risque : le CSM a pu annoter après observation de la résiliation** → fuite rétrospective potentielle. Redondant avec l'usage (η = 0,83 avec `derniere_connexion_jours`, §6.10) : ce n'est pas un leurre au sens strict. | `leurre + fuite potentielle` | Exclure via `COLONNES_INTERDITES` : fuite non exclue + précaution RGPD + 55 % manquants. Audit au §6.6, criblage au §6.10. |
| 27 | `sante_compte_fin_periode` | int64 | — (exclure) | score (0–100) | Score de santé du compte calculé **en fin de période** (après que le churn soit observé). **Fuite confirmée** : churners µ = 11,7 (σ = 10,0) vs non-churners µ = 60,1 (σ = 12,2) — quasi-aucun chevauchement. | **Nulle** — fuite de données pure ; un modèle réel n'y aurait pas accès au moment de la prédiction. | `fuite` | Exclue via `COLONNES_INTERDITES`. Fuite détectée en §6.4 (ROC-AUC univariée 0,999, seuil d'alerte 0,85) et démontrée en §6.5 (PR-AUC 0,9995 avec, 0,7659 sans). |
| 28 | `valeur_vie_client_eur` | int64 | float64 | EUR | Valeur vie du client (300–2 000 000 EUR). Cible du modèle de régression (§8.1, §12.10). 0 % manquants. | **Nulle pour le churn** — cible secondaire ; CLV prospective (MRR × durée de vie estimée) qui intègre déjà le risque de départ (§6.7) → fuite si utilisée comme feature du churn, double comptage si multipliée par P(churn). | `fuite` | Exclure via `COLONNES_INTERDITES` du modèle de churn. Utiliser uniquement comme cible de la régression CLV. |
| 29 | `churn` | int64 | int64 | catégorie binaire (0/1) | Variable cible principale : 1 = résiliation, 0 = renouvellement. **28,0 % de positifs** après dédoublonnage (1 400 / 5 000) — déséquilibre modéré. 0 % manquants. | **N/A** — variable à prédire. | `fuite` (cible) | Exclue des variables (`COLONNES_INTERDITES`). Déséquilibre : `class_weight="balanced"` + correction d'intercept ; SMOTE évalué et écarté (§9.6). |

---

## Anomalies transverses détectées

| Problème | Colonnes concernées | Gravité | Traitement |
|---|---|---|---|
| Doublons exacts (35 copies, 70 lignes en groupes) | toutes | Haute | `drop_duplicates(keep="first")` ; contrôle bloquant d'unicité sur `client_id` (§5.5, §7.3.1) |
| Casse et espaces hétérogènes | `taille_entreprise`, `plan`, `secteur` | Haute | `NormalisationCategorielle` dans le pipeline (§7.3.5) |
| Nombres stockés en texte (virgule décimale, symbole `%`) | `taux_adoption_pct`, `heures_usage_30j`, `delai_reponse_support_h`, `revenu_mensuel_recurrent_eur` | Haute | `coercer_numeriques()` : 6 529 valeurs réparées, aucune irrécupérable (§5.6) |
| Dates multi-formats (3 formats, 962 ambiguës) | `date_souscription` | Moyenne | `parser_dates()`, convention `JJ/MM/AAAA` démontrée (§5.7) |
| Valeurs manquantes > 5 % | `commentaire_csm` (55 %), `heures_usage_30j` (6 %), `delai_reponse_support_h` (10 %), `csat` (8 %) | Variable | Médiane (numériques) et `inconnu` (catégorielles) dans le pipeline, jamais sur le jeu complet ; 10 colonnes MCAR (§5.9) |

---

## Colonnes écartées ou conservées sous condition, et pourquoi

Cette section est reprise directement dans le notebook (§7 — Préparation des données).

### Écartées pour risque de fuite de données

| Colonne | Motif | Preuve |
|---|---|---|
| `sante_compte_fin_periode` | Score calculé **après** observation du churn (fin de période) → fuite temporelle pure | Churners : µ = 11,7 vs non-churners : µ = 60,1 ; quasi-aucun chevauchement ; corrélation de Pearson avec `churn` ≈ −0,88 (jeu gold) |
| `valeur_vie_client_eur` | Cible de la régression CLV ; CLV **prospective** qui intègre déjà le risque de départ | Dépasse le CA encaissé pour 79,7 % des comptes, élasticité à l'ancienneté 0,21 (§6.7) |
| `commentaire_csm` | Les 13 valeurs encodent explicitement le risque perçu par le CSM ; annotation pouvant être rétrospective | 55 % manquants + libellés prédictifs directs (« Client insatisfait, risque de départ. ») |

### Écartées comme identifiant technique

| Colonne | Motif |
|---|---|
| `client_id` | Clé primaire — aucune valeur prédictive, risque de mémorisation d'entité |

### Écartées comme cible

| Colonne | Motif |
|---|---|
| `churn` | Variable à prédire |

### Écartées comme leurres (preuve en trois temps au §12.9 : association, permutation, drop-column)

Les trois leurres restent en entrée du champion du notebook, pour que leur inutilité soit mesurée,
puis sont retirés du modèle déployé (`config.COLONNES_LEURRES_SUSPECTES`). Mesures au 2026-10-10 :
aucune association marginale significative après correction de Benjamini-Hochberg (p ajustée
0,97 pour les trois, `criblage_leurres_12.parquet`) ; les retirer change la PR-AUC de −0,003 à
+0,001 (`drop_column_12.parquet`), écart jugé négligeable.

| Colonne | Motif | Anomalie observée |
|---|---|---|
| `couleur_theme_interface` | Préférence cosmétique — aucun lien causal attendu | Distribution uniforme sur 5 valeurs, 0 % manquants → pas d'anomalie supplémentaire |
| `code_datacenter` | Localisation infrastructure — aucun lien causal attendu | 4 valeurs ; aucune redondance avec une autre variable, `pays` compris (§6.10) |
| `groupe_experimentation` | Appartenance A/B test — biaiserait le modèle en production | Distribution quasi-uniforme (34/33/33 %) — pas de signal suspect de sélection |

### Conservée malgré sa colinéarité

| Colonne | Motif | Décision |
|---|---|---|
| `fonctionnalites_total` | Bijection parfaite avec `plan` : Starter=8, Pro=16, Business=26, Enterprise=40 (vérifiée sur les 5 035 lignes) | Conservée : dénominateur de `taux_couverture_fonctionnelle` (H12) ; redondance absorbée par la régularisation |

### Non modélisées

| Colonne | Motif |
|---|---|
| `date_souscription` | N'apporte que l'ancienneté, déjà portée par `anciennete_mois` (§6.11) |
| `jour_souscription` | Sans hypothèse métier ni association significative avec le churn (§6.8) |

---

## Variables dérivées — ratios métier (§7.5)

Calculées ligne par ligne par `features.build.ajouter_features_metier()`, sans statistique apprise
sur d'autres clients (aucune fuite entre plis). Seule exception, l'écart du CSAT à la médiane du
secteur, appris dans le `Pipeline` (`EcartAuGroupe`). Chaque ratio prolonge une hypothèse métier
de `config.HYPOTHESES_METIER` (§6.2). L'API les calcule côté serveur : le client ne les envoie
jamais.

| Ratio | Formule | Hypothèse | Lecture métier |
|---|---|---|---|
| `taux_utilisation_sieges` | `utilisateurs_actifs / sieges_souscrits` | H9 | Sièges payés mais non déployés : la facture devient un argument de départ |
| `taux_couverture_fonctionnelle` | `fonctionnalites_utilisees / fonctionnalites_total` | H12 | Peu de fonctions explorées sur celles du plan : valeur perçue faible |
| `intensite_usage_par_utilisateur` | `heures_usage_30j / utilisateurs_actifs` | H3 | Usage réel par utilisateur, indépendamment de la taille du compte |
| `arpu_par_siege` | `revenu_mensuel_recurrent_eur / sieges_souscrits` | — | Revenu par siège bas : remise forte ou plan surdimensionné |
| `intensite_support` | `tickets_support_90j / anciennete_mois` | H6 | Frictions rapportées à la durée de la relation |

Dix autres variables complètent ces ratios (§7.5) : `surdimensionnement`, `connexions_par_utilisateur`,
`recence_normalisee`, `compte_dormant`, `pression_support`, `tranche_anciennete`,
`tranche_integrations` et les trois indicateurs de manquance (`csat_manquant`,
`heures_usage_30j_manquant`, `delai_reponse_support_h_manquant`). La jointure du catalogue des
plans (§7.6) et l'enrichissement simulé par secteur et par pays (§7.7) ajoutent 13 colonnes, d'où
les 57 colonnes du gold.

**Limite d'`intensite_support`.** Le numérateur couvre 90 jours, le dénominateur toute la vie du
compte : le ratio est lié à l'ancienneté par construction. Sur le jeu gold, corrélation de rang
−0,42 avec `anciennete_mois`, contre 0,00 pour `tickets_support_90j`. Son importance se lit donc à
côté de celle de l'ancienneté (§12.8). Elle est gardée pour la question métier distincte qu'elle
pose (le support pèse-t-il lourd pour l'âge du compte ?).
