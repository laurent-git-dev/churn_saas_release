# Dictionnaire des données — churn_saas_complet.csv

Source : profil produit par `churn_saas.data.quality.profil_compact()` sur
`data/raw/churn_saas_complet.csv` (5 035 lignes, 29 colonnes, 35 doublons détectés).

> **Légende risques** — `fuite` : variable connue après la décision de prédiction ;
> `identifiant` : clé technique sans valeur prédictive ;
> `leurre` : variable annoncée sans lien avec le churn (à démontrer chiffres à l'appui) ;
> `leurre + fuite potentielle` : leurre annoncé mais encoding suspect du jugement rétrospectif ;
> `colinéaire` : information déjà portée par une autre variable ;
> `donnée personnelle` : donnée à caractère personnel (RGPD) ;
> `aucun` : aucun risque identifié.

---

## Tableau principal (29 colonnes)

| # | nom | type observé | type cible | unité | sémantique métier | pertinence a priori pour le churn | risque | traitement prévu |
|---|---|---|---|---|---|---|---|---|
| 1 | `client_id` | str | — | — | Identifiant unique du compte client (format CLI-XXXXXX). 5 000 valeurs distinctes pour 5 035 lignes → 35 doublons. | **Nulle** — identifiant technique, aucune valeur prédictive ; risque de mémorisation d'entité si conservé. | `identifiant` | Exclure via `COLONNES_INTERDITES`. Utiliser pour déduplication avant tout autre traitement. |
| 2 | `date_souscription` | str | datetime | — | Date de début de contrat. **Trois formats coexistent** : `JJ Mon AAAA` (1 709 lignes), `JJ/MM/AAAA` (1 621), `AAAA-MM-JJ` (1 705). 0 % manquants. | **Faible** par elle-même ; utile pour dériver la saisonnalité (mois/trimestre de souscription) et vérifier la cohérence avec `anciennete_mois`. | `aucun` | Parser avec `pd.to_datetime` en mode `dayfirst=True` + format multiple. Dériver : mois et trimestre de souscription. |
| 3 | `jour_souscription` | str (catégorie) | str → catégorie | — | Jour de la semaine de la souscription (lundi–dimanche, 7 valeurs). 0 % manquants. Redondant avec `date_souscription` une fois parsée. | **Faible** — le jour de la semaine n'a pas de justification métier directe comme signal de churn ; redondance avec `date_souscription`. | `aucun` | **Non modélisée** (`features.build.COLONNES_NON_MODELISEES`) : sans hypothèse métier ni association significative avec le churn (§6.8). |
| 4 | `secteur` | str | str → catégorie | — | Secteur d'activité du client. **28 valeurs observées** pour ~10–14 secteurs réels, dues à la casse hétérogène (Commerce / COMMERCE / commerce). 5,0 % manquants. | **Moyenne** — certains secteurs ont des cycles économiques propres (ex. : la Tech renouvelle différemment du Public). | `aucun` | Normaliser : `.str.strip().str.title()`. Imputer les 5 % manquants par la modalité la plus fréquente. OneHot ou target encoding. |
| 5 | `pays` | str (catégorie) | str → catégorie | — | Pays d'implantation du client (6 valeurs : France, Belgique, Allemagne…). 4,0 % manquants. | **Moyenne** — effets réglementaires, culturels et économiques sur l'adoption et le renouvellement. | `aucun` | Imputer les 4 % manquants par mode ou catégorie `"Inconnu"`. OneHot (6 modalités — acceptable). |
| 6 | `taille_entreprise` | str | str → catégorie ordinale | — | Taille de l'entreprise (TPE / PME / ETI / GE). **12 valeurs observées** pour 4 catégories réelles : variantes de casse et espaces préfixes (` PME`, `pme`, `PME`). 0 % manquants. | **Forte** — la taille conditionne la capacité d'adoption, l'implication du IT, et la criticité du produit. | `aucun` | Normaliser en 4 catégories (`.str.strip().str.upper()`). Encoder en ordinal (TPE=1, PME=2, ETI=3, GE=4). |
| 7 | `plan` | str | str → catégorie ordinale | — | Plan souscrit (Starter / Pro / Business / Enterprise). **16 valeurs observées** pour 4 plans réels : variantes de casse et espaces. 0 % manquants. | **Forte** — détermine le MRR, les fonctionnalités accessibles et les engagements contractuels. Corrélé à `fonctionnalites_total`. | `aucun` | Normaliser en 4 catégories. Encoder en ordinal croissant (Starter=1 … Enterprise=4). |
| 8 | `anciennete_mois` | int64 | int64 | mois | Durée en mois depuis la souscription (1–36 mois). 0 % manquants. | **Forte** — l'ancienneté est un proxy de la loyauté ; les churners en début de cycle ont un profil distinct (onboarding raté). | `aucun` | Conserver tel quel. Vérifier la cohérence avec `date_souscription` parsée. |
| 9 | `sieges_souscrits` | int64 | int64 | sièges | Nombre de licences achetées (1–898). 0 % manquants. Distribution longue queue. | **Forte** — proxy de l'engagement financier et de la taille de déploiement interne. | `aucun` | Conserver. Dériver : `taux_occupation_siege = utilisateurs_actifs / sieges_souscrits`. |
| 10 | `utilisateurs_actifs` | int64 | int64 | utilisateurs | Nombre d'utilisateurs actifs sur le compte (0–829). 0 % manquants. Jamais supérieur à `sieges_souscrits` (ratio max = 1,0). | **Forte** — signal direct d'adoption réelle ; un faible ratio siège/actifs signale un déploiement en échec. | `aucun` | Dériver `taux_occupation_siege`. Conserver la valeur brute en parallèle. |
| 11 | `taux_adoption_pct` | str | float64 | % | Taux d'adoption (0–100). **Double format** : séparateur décimal mixte (`.` vs `,`) **et** présence optionnelle du symbole `%` (`50.0%`, `50,0`, `50.0`). 5,0 % manquants. Seulement 57,8 % des valeurs non-nulles sont convertibles directement. | **Forte** — indicateur composite d'engagement ; faible adoption → risque de churn élevé. | `aucun` | Nettoyage : (1) supprimer le suffixe `%` ; (2) remplacer `,` par `.` ; (3) cast en float. Imputer les 5 % manquants par médiane dans le pipeline. |
| 12 | `connexions_30j` | int64 | int64 | connexions | Nombre de connexions sur les 30 derniers jours (0–156). 0 % manquants. | **Forte** — signal d'activité récent le plus direct ; 0 connexion est un signal d'alarme fort. | `aucun` | Conserver. Créer feature binaire : `inactif_30j = (connexions_30j == 0)`. |
| 13 | `heures_usage_30j` | str | float64 | heures | Heures d'utilisation sur les 30 derniers jours (0,0–170,2). **Virgule décimale francophone** sur 1 402 valeurs (30 % des non-nuls). 6,1 % manquants. | **Forte** — signal d'intensité d'usage, complémentaire de `connexions_30j` (durée vs fréquence). | `aucun` | Remplacer `,` par `.` ; cast en float. 6 % manquants → imputer par médiane dans le pipeline. |
| 14 | `fonctionnalites_total` | int64 | int64 | nombre | Nombre total de fonctionnalités disponibles dans le plan (4 valeurs : 8, 16, 26, 40). 0 % manquants. **Entièrement déterminé par `plan`** : Starter=8, Pro=16, Business=26, Enterprise=40 (bijection parfaite confirmée). | **Nulle** — colinéaire à 100 % avec `plan` ; aucune information ajoutée après encodage du plan. | `colinéaire` | **Conservée** comme dénominateur de `taux_couverture_fonctionnelle` (H12, §7.5) ; la régularisation L2 absorbe la redondance avec `plan`. |
| 15 | `fonctionnalites_utilisees` | int64 | int64 | nombre | Nombre de fonctionnalités effectivement utilisées (0–40). 0 % manquants. | **Forte** — mesure de la profondeur d'adoption (breadth of use) ; complémentaire à `taux_adoption_pct`. | `aucun` | Dériver : `ratio_fonctionnalites = fonctionnalites_utilisees / fonctionnalites_total`. |
| 16 | `nb_integrations` | float64 | int64 | intégrations | Nombre d'intégrations tierces actives (0–16). 4,0 % manquants. | **Forte** — les intégrations créent de la *stickiness* (coût de migration élevé) ; puissant signal anti-churn. | `aucun` | 4 % manquants → imputer par médiane. Cast en int après imputation. |
| 17 | `derniere_connexion_jours` | int64 | int64 | jours | Nombre de jours depuis la dernière connexion (0–200). 0 % manquants. | **Forte** — signal d'inactivité ; clé du churn précoce. Complémentaire de `connexions_30j`. | `aucun` | Conserver. Dériver : `inactif_60j = (derniere_connexion_jours > 60)`. |
| 18 | `tickets_support_90j` | int64 | int64 | tickets | Nombre de tickets support ouverts sur 90 jours (0–17). 0 % manquants. | **Forte** — signal de friction (direction ambiguë : peu de tickets peut signaler peu d'usage *ou* faible friction). | `aucun` | Conserver. Croiser avec `csat` pour lever l'ambiguïté directionnelle. |
| 19 | `delai_reponse_support_h` | str | float64 | heures | Délai moyen de réponse du support (0,5–max). **Virgule décimale francophone** dans une partie des valeurs. 10,0 % manquants. 87 % convertibles après remplacement virgule→point. | **Moyenne** — la qualité du support influence la satisfaction ; signal secondaire. | `aucun` | Remplacer `,` par `.` ; cast en float. 10 % manquants → imputer par médiane dans le pipeline. |
| 20 | `csat` | float64 | float64 | score (1–5) | Score de satisfaction client CSAT (1–5, 5 valeurs entières). 8,0 % manquants. | **Forte** — signal direct et déclaratif de la satisfaction ; corrélé au risque de churn. | `aucun` | 8 % manquants → imputer par médiane dans le pipeline. |
| 21 | `retards_paiement_12m` | float64 | int64 | retards | Nombre de retards de paiement sur 12 mois (0–7). 5,0 % manquants. | **Forte** — signal de friction financière et d'intention de non-renouvellement. | `aucun` | 5 % manquants → imputer par médiane. Cast en int. |
| 22 | `revenu_mensuel_recurrent_eur` | str | float64 | EUR | MRR mensuel en euros. **Double format décimal** : point anglosaxon (`171.02`) et virgule francophone (`40,0`, `2000,67`). 1 019 valeurs avec virgule. 3,0 % manquants. | **Forte** — détermine l'importance économique du compte ; corrélé au plan et à la taille. Utile pour prioriser les interventions. | `aucun` | Remplacer `,` par `.` ; cast en float. 3 % manquants → imputer par médiane. |
| 23 | `couleur_theme_interface` | str (catégorie) | — (exclure) | — | Couleur de l'interface choisie par l'utilisateur (5 valeurs : clair, violet, bleu…). 0 % manquants, répartition uniforme. | **Nulle** — préférence cosmétique sans lien causal avec la résiliation. Leurre annoncé. | `leurre` | Conserver dans le gold et le champion pour produire la preuve chiffrée d'inutilité (association marginale, permutation importance, drop-column importance, §12.9) ; exclure du modèle déployé (§10.3). |
| 24 | `code_datacenter` | str (catégorie) | — (exclure) | — | Région d'hébergement (4 valeurs : eu-w1, us-e1, ap-s1 + 1 autre). 0 % manquants. Aucune redondance avec une autre variable, `pays` compris (redondance maximale 0,08, §6.10). | **Nulle** — choix d'infrastructure sans corrélation causale avec le churn. Leurre annoncé. | `leurre` | Même traitement que `couleur_theme_interface`. |
| 25 | `groupe_experimentation` | str (catégorie) | — (exclure) | — | Appartenance à un groupe d'A/B test (control / A / B, quasi-équirépartis : 34 % / 33 % / 33 %). 0 % manquants. | **Nulle** — leurre annoncé. Si l'A/B test avait un effet réel sur le churn, inclure cette variable biaiserait le modèle en production (les futurs clients n'appartiennent à aucun groupe). | `leurre` | Même traitement que `couleur_theme_interface`. |
| 26 | `commentaire_csm` | str (13 valeurs) | — (exclure) | — | Annotation textuelle du CSM. **55,4 % manquants**. Seulement 13 valeurs distinctes malgré 2 245 non-nuls. ⚠️ Les libellés encodent explicitement le risque perçu (« Client insatisfait, risque de départ. », « Usage en forte baisse ce trimestre. »). | **Suspecte** — bien que leurre annoncé, les 13 catégories prédisent quasi-directement le churn. **Risque : le CSM a pu annoter après observation de la résiliation** → fuite rétrospective potentielle. Redondant avec l'usage (η = 0,83 avec `derniere_connexion_jours`, §6.10) : ce n'est pas un leurre au sens strict. | `leurre + fuite potentielle` | Exclure via `COLONNES_INTERDITES` : fuite non exclue + précaution RGPD + 55 % manquants. Audit au §6.6, criblage au §6.10. |
| 27 | `sante_compte_fin_periode` | int64 | — (exclure) | score (0–100) | Score de santé du compte calculé **en fin de période** (après que le churn soit observé). **Fuite confirmée** : churners µ = 11,7 (σ = 10,0) vs non-churners µ = 60,1 (σ = 12,2) — quasi-aucun chevauchement. | **Nulle** — fuite de données pure ; un modèle réel n'y aurait pas accès au moment de la prédiction. | `fuite` | Exclure via `COLONNES_INTERDITES`. Démontrer la fuite en §5 (corrélation avec `churn`, distribution bimodale). |
| 28 | `valeur_vie_client_eur` | int64 | float64 | EUR | Valeur vie du client (300–2 000 000 EUR). Cible du modèle de régression (§9). 0 % manquants. | **Nulle pour le churn** — cible secondaire, connue après décision de résiliation → fuite temporelle si utilisée comme feature du modèle de churn. | `fuite` | Exclure via `COLONNES_INTERDITES` du modèle de churn. Utiliser uniquement comme cible de la régression CLV. |
| 29 | `churn` | int64 | int64 | catégorie binaire (0/1) | Variable cible principale : 1 = résiliation, 0 = renouvellement. **28 % de positifs** (1 412 / 5 035) — déséquilibre modéré. 0 % manquants. | **N/A** — variable à prédire. | `fuite` (cible) | Exclure via `COLONNES_INTERDITES`. Traiter le déséquilibre de classes (class_weight ou sur-échantillonnage SMOTE). |

---

## Anomalies transverses détectées

| Problème | Colonnes concernées | Gravité | Traitement |
|---|---|---|---|
| Doublons (35 paires, 70 lignes) | `client_id` | Haute | `df.drop_duplicates()` avant tout traitement |
| Casse hétérogène (3–4 variantes par modalité) | `taille_entreprise`, `plan`, `secteur` | Haute | `.str.strip().str.title()` puis mapping explicite |
| Virgule comme séparateur décimal | `heures_usage_30j`, `delai_reponse_support_h`, `revenu_mensuel_recurrent_eur` | Haute | `.str.replace(",", ".", regex=False)` puis cast float |
| Double format décimal + symbole % | `taux_adoption_pct` | Haute | Supprimer `%` puis remplacer `,` → `.` puis cast float |
| Dates multi-formats (3 formats) | `date_souscription` | Moyenne | `pd.to_datetime(..., dayfirst=True, format="mixed")` |
| Valeurs manquantes > 5 % | `commentaire_csm` (55 %), `heures_usage_30j` (6 %), `delai_reponse_support_h` (10 %), `csat` (8 %) | Variable | Imputation médiane/mode dans le pipeline (jamais sur le jeu complet) |

---

## Colonnes écartées ou conservées sous condition, et pourquoi

Cette section est reprise directement dans le notebook (§7 — Préparation des données).

### Écartées pour risque de fuite de données

| Colonne | Motif | Preuve |
|---|---|---|
| `sante_compte_fin_periode` | Score calculé **après** observation du churn (fin de période) → fuite temporelle pure | Churners : µ = 11,7 vs non-churners : µ = 60,1 ; quasi-aucun chevauchement ; corrélation de Pearson avec `churn` ≈ −0,90 |
| `valeur_vie_client_eur` | Cible de la régression CLV, dérivée du comportement **futur** du client → connue après décision | Règle explicite de l'énoncé + logique temporelle |
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
puis sont retirés du modèle déployé (`config.COLONNES_LEURRES_SUSPECTES`). Mesures au 2026-10-03 :
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

**Limite d'`intensite_support`.** Le numérateur couvre 90 jours, le dénominateur toute la vie du
compte : le ratio est lié à l'ancienneté par construction. Sur le jeu gold, corrélation de rang
−0,42 avec `anciennete_mois`, contre 0,00 pour `tickets_support_90j`. Son importance se lit donc à
côté de celle de l'ancienneté (§12.8). Elle est gardée pour la question métier distincte qu'elle
pose (le support pèse-t-il lourd pour l'âge du compte ?).
