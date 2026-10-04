# Registre des risques — Modèle de prédiction du churn SaaS B2B

**Projet :** Churn SaaS B2B — certification CISIA
**Version :** 1.1 (révision du 3 octobre 2026 : règle de décision à deux niveaux, risque R09 ; révision du 4 octobre 2026 : risque R10)
**Date de création :** 19 septembre 2026
**Prochaine revue :** 19 décembre 2026 (comité trimestriel)
**Propriétaire du registre :** Data Scientist / Équipe projet

Ce registre est produit en application de l'item C2 du référentiel CISIA et des lignes
directrices HLEG (exigence 7 — Responsabilité). Il est mis à jour à chaque cycle de
réentraînement du modèle et soumis au comité de revue trimestriel.

---

## Grille de cotation

| Niveau | Probabilité | Impact |
|---|---|---|
| 1 — Faible | < 20 % de survenance sur 12 mois | Perturbation mineure, récupérable rapidement |
| 2 — Moyen | 20–60 % | Dégradation mesurable, nécessite une action corrective |
| 3 — Élevé | > 60 % ou déjà observé | Impact significatif sur les résultats ou la conformité |

**Score = Probabilité × Impact (max 9).**
Seuil d'escalade : score ≥ 6 → revue immédiate + information Direction.

---

## Registre

### R01 — Biais commercial par sous-groupe (pays, taille, secteur)

| Champ | Valeur |
|---|---|
| **Description** | Le modèle présente des taux de détection (TPR) significativement différents selon le pays, la taille d'entreprise ou le secteur. Certains segments sont systématiquement sur- ou sous-signalés, indépendamment de leur comportement réel. |
| **Probabilité** | 2 — Moyen (biais de représentation fréquents dans les données SaaS B2B, identifiés en §4.4, mesurés en §12.14) |
| **Impact** | 2 — Moyen (discrimination commerciale perçue, risque réputationnel) |
| **Score** | 4 |
| **Mitigation** | Audit d'équité (TPR, FPR, ROC-AUC et calibration par sous-groupe) sur le modèle final (§12.14) et à chaque réentraînement. Seuils de revue (`config.EQUITE`) : écart de TPR > 15 points entre modalités ou écart de calibration > 5 points → revue obligatoire avant déploiement. |
| **Propriétaire** | Data Scientist |
| **Statut** | Mesuré — en surveillance continue |

---

### R02 — Prophétie auto-réalisatrice sur les petits comptes

| Champ | Valeur |
|---|---|
| **Description** | Les petits comptes (MRR faible) reçoivent des scores de risque élevés. L'équipe CS, à capacité limitée, priorise les grands comptes. Les petits comptes signalés, non accompagnés, finissent par résilier — confirmant la prédiction du modèle. Le modèle crée la réalité qu'il prétend observer. |
| **Probabilité** | 3 — Élevé (mécanisme structurel dès que la capacité CS est contrainte) |
| **Impact** | 2 — Moyen (perte de comptes faible MRR individuellement, mais effet cumulatif significatif) |
| **Score** | 6 → **Escalade Direction** |
| **Mitigation** | Règle de priorisation : score × MRR × coût d'intervention (§12), non score seul. Quota minimal de petits comptes traités par semaine (décision managériale, transmise en note de synthèse). Groupe témoin 10 % pour détecter l'effet causal de la non-intervention. |
| **Propriétaire** | CS Lead + Data Scientist |
| **Statut** | Ouvert — mitigation en cours d'implémentation |

---

### R03 — Incitation perverse liée aux remises commerciales

| Champ | Valeur |
|---|---|
| **Description** | Une politique de remise systématique offerte aux comptes signalés à risque élevé peut être *gamée* : les clients apprennent que simuler un comportement de départ (réduction des connexions, ouverture de tickets) déclenche une offre avantageuse. Effet secondaire : *deadweight loss* sur les comptes qui auraient résilié de toute façon. |
| **Probabilité** | 2 — Moyen (nécessite une politique de remise automatique ou peu encadrée) |
| **Impact** | 2 — Moyen (dégradation de marge, corruption progressive des labels d'entraînement) |
| **Score** | 4 |
| **Mitigation** | Décision de remise obligatoirement humaine (CSM + validation manager). Monitoring du taux de remise par cohorte de score (tableau de bord §12). Détection du gaming via drift des features (connexions_30j, tickets_support_90j avant renouvellement — Evidently §13). |
| **Propriétaire** | Directeur Commercial + CS Lead |
| **Statut** | Ouvert — suivi trimestriel |

---

### R04 — Boucle de rétroaction et corruption de la distribution d'apprentissage

| Champ | Valeur |
|---|---|
| **Description** | Les interventions CS modifient le comportement des comptes à risque. Ces comptes apparaissent comme *non churné* au prochain réentraînement, sans que le comportement contrefactuel (aurait-il résilié sans intervention ?) soit connu. Le modèle sous-estime progressivement le risque réel des comptes qui font l'objet d'interventions récurrentes. Dégradation silencieuse et difficile à détecter sans protocole expérimental. |
| **Probabilité** | 3 — Élevé (mécanisme inévitable dès que le modèle alimente des actions) |
| **Impact** | 3 — Élevé (dégradation irréversible des labels → modèle inutilisable sans reboot) |
| **Score** | 9 → **Escalade Direction — Risque critique** |
| **Mitigation principale** | Groupe témoin non traité : ~10 % des comptes à risque élevé exclus des interventions CS à chaque cycle (tirage aléatoire stratifié). Leur devenir fournit les labels contrefactuels non biaisés. Protocole documenté en §13. **Décision de validation requise de la Direction avant déploiement.** |
| **Mitigation secondaire** | Monitoring PSI/KS et rapport Evidently sur la distribution des features et des scores (§13.3, §13.4). PSI ≥ 0,20 → comité ad hoc sous 72 h (réentraîner, maintenir ou geler) ; aucun réentraînement automatique sur signal statistique seul. |
| **Propriétaire** | Data Scientist (protocole) + Direction (validation groupe témoin) |
| **Statut** | Ouvert — validation Direction en attente |

---

### R05 — Fuite de données personnelles via `commentaire_csm`

| Champ | Valeur |
|---|---|
| **Description** | Le champ `commentaire_csm` est un texte libre rédigé par les CSM. Il peut contenir des données personnelles des contacts opérationnels du client (noms, postes, opinions, situations personnelles). Son inclusion dans le jeu d'entraînement constituerait un traitement de données personnelles non déclaré et non minimisé. |
| **Probabilité** | 3 — Élevé (texte libre non structuré, pratique courante chez les CSM) |
| **Impact** | 3 — Élevé (non-conformité RGPD, risque de sanction CNIL, risque réputationnel) |
| **Score** | 9 → **Escalade DPO — Risque critique** |
| **Mitigation** | Variable exclue du modèle (`config.COLONNES_INTERDITES`, audit en §6.6). Validation DPO documentée (fiche de revue §4.6.3). Si analyse NLP de ce champ envisagée à l'avenir : analyse d'impact (AIPD) obligatoire au préalable. |
| **Propriétaire** | DPO |
| **Statut** | **Clos** — mitigation en place (exclusion effective) |

---

### R06 — Sur-confiance dans les probabilités (modèle non calibré)

| Champ | Valeur |
|---|---|
| **Description** | Un modèle bien discriminant (AUC élevée) peut rendre des probabilités mal calibrées : une probabilité de 0,8 ne correspond pas nécessairement à un taux de churn réel de 80 %. Si les CSM interprètent le score comme une probabilité absolue, ils peuvent sur- ou sous-estimer le risque réel. |
| **Probabilité** | 2 — Moyen (problème fréquent avec les ensembles et les modèles non calibrés) |
| **Impact** | 2 — Moyen (mauvaise allocation des ressources CS, calcul de valeur à risque faux) |
| **Score** | 4 |
| **Mitigation** | Correction d'intercept analytique après `class_weight="balanced"` : score de Brier 0,116 contre 0,136 sans correction (`comparaison_desequilibre.json`, valeurs au 2026-10-03, §9.6) ; courbe de fiabilité en §12.2.3. Contrôle à chaque vague de renouvellements : la correction suppose une prévalence stable (§13.8). Mention explicite dans le guide d'usage CSM : le score est un *indicateur de risque relatif*, non une probabilité absolue. |
| **Propriétaire** | Data Scientist |
| **Statut** | Mesuré — calibration documentée en §9.6 et §12.2.3 |

---

### R07 — Obsolescence non détectée du modèle

| Champ | Valeur |
|---|---|
| **Description** | La distribution des données évolue (nouveaux segments, changement de produit, crise sectorielle). Le modèle, entraîné sur des données historiques, peut devenir progressivement inadapté sans que la dégradation soit visible si aucun monitoring n'est en place. |
| **Probabilité** | 2 — Moyen (inévitable à long terme, délai variable selon la stabilité du contexte) |
| **Impact** | 3 — Élevé (décisions CS basées sur un modèle inadapté = gestion des churns manquée) |
| **Score** | 6 → **Escalade Direction** |
| **Mitigation** | Monitoring Evidently (PSI/KS) sur les features et les prédictions (§13). Indicateur d'obsolescence intégré au tableau de bord (§13). PSI ≥ 0,20 ou PR-AUC d'une cohorte de renouvellements < 0,65 → comité ad hoc (§13.8). Comité de revue trimestriel. |
| **Propriétaire** | Data Scientist |
| **Statut** | Ouvert — outillage de monitoring en place (§13) |

---

### R08 — Usage inadapté du score par les CSM

| Champ | Valeur |
|---|---|
| **Description** | Les CSM peuvent utiliser le score de deux façons contre-productives : (a) l'ignorer complètement (rejet de l'outil) ; (b) le suivre aveuglément sans contextualisation (abandon du jugement). Dans les deux cas, la valeur ajoutée du modèle est nulle ou négative. |
| **Probabilité** | 2 — Moyen (résistance au changement courante lors de l'introduction d'un outil ML) |
| **Impact** | 2 — Moyen (ROI du projet non atteint) |
| **Score** | 4 |
| **Mitigation** | Formation structurée des CSM à l'interprétation du score et des SHAP locaux (§10). Guide d'usage disponible dans l'interface (§10). Suivi du taux d'utilisation de l'outil dans le CRM (KPI de déploiement §12). Retours CSM intégrés au comité trimestriel. |
| **Propriétaire** | CS Lead |
| **Statut** | Ouvert — formation planifiée avant déploiement |

---

### R09 — Churners manqués sous le seuil de vigilance (faux négatifs résiduels)

| Champ | Valeur |
|---|---|
| **Description** | Le niveau 1 de la règle de décision (vigilance) vise un recall de 80 % : par construction, environ un churner sur cinq reste sous le seuil et ne reçoit aucune attention. Mesuré hors pli au 2026-10-03 (`economics.table_deux_niveaux`, §12.6) : 280 churners sur 1 400, pour une perte sèche estimée à 7,1 M€ (MRR × horizon × marge). Si la prévalence ou la distribution des scores dérive, le seuil fixé (0,277, `seuil_vigilance.json`) peut ne plus garantir ce recall. |
| **Probabilité** | 3 — Élevé (certain par construction ; seule l'ampleur varie) |
| **Impact** | 2 — Moyen (perte de revenu concentrée sur les comptes non signalés ; pas d'atteinte aux personnes) |
| **Score** | 6 → **Escalade Direction** |
| **Mitigation** | Recall cible fixé au cadrage, avant les résultats (`config.RECALL_CIBLE_VIGILANCE`) ; tenue vérifiée sur le jeu de test, avec un seuil fixé sans lui (81,8 % [77,4 % ; 86,3 %], cible ≥ 75 %, §9.14), et sur des plis qui n'ont pas fixé le seuil (76,1 % à 85,7 %). Recall contrôlé à chaque vague de renouvellements ; seuil recalculé hors pli à chaque réentraînement et embarqué dans les métadonnées du modèle promu (§13.8). Les comptes `OK` restent visibles dans le CRM : le jugement du CSM peut toujours les remonter. |
| **Propriétaire** | Data Scientist (seuil) + CS Lead (choix du recall cible) |
| **Statut** | Ouvert — accepté au cadrage, surveillé à chaque vague |

---

### R10 — Données d'usage produites hors UE

| Champ | Valeur |
|---|---|
| **Description** | Les instances clientes sont réparties sur quatre régions, dont deux hors UE (`us-e1`, `ap-s1`, colonne `code_datacenter`). Les journaux de connexion, d'où sont tirées les variables d'usage, y sont produits : ils relèvent du droit local (Cloud Act pour les États-Unis) et, pour la part qui concerne des utilisateurs identifiables, des règles de transfert du RGPD (chapitre V). |
| **Probabilité** | 1 — Faible (le jeu modélisé ne contient que des indicateurs d'entreprise) |
| **Impact** | 2 — Moyen (secret des affaires des clients ; non-conformité si des journaux bruts quittaient leur région) |
| **Score** | 2 |
| **Mitigation** | Agrégation par entreprise dans chaque région avant extraction : seuls des indicateurs d'entreprise rejoignent l'UE (§4.1). Encadrement des instances hors UE (décision d'adéquation ou clauses contractuelles types) vérifié par le DPO. Base d'analyse, modèle et scores hébergés en région UE (§11). |
| **Propriétaire** | DPO + RSSI |
| **Statut** | Ouvert — contrôle de l'extraction à documenter avant déploiement |

---

## Synthèse

| ID | Risque | Score | Statut |
|---|---|---|---|
| R04 | Boucle de rétroaction | **9** — Critique | Validation Direction en attente |
| R05 | Fuite données perso (commentaire_csm) | **9** — Critique | **Clos** (exclusion effective) |
| R02 | Prophétie auto-réalisatrice | **6** — Élevé | Mitigation en cours |
| R07 | Obsolescence non détectée | **6** — Élevé | Outillage en place |
| R09 | Churners manqués sous le seuil de vigilance | **6** — Élevé | Accepté, surveillé par vague |
| R01 | Biais par sous-groupe | 4 | Surveillance continue |
| R03 | Remises perverses | 4 | Suivi trimestriel |
| R06 | Modèle non calibré | 4 | Calibration documentée |
| R08 | Usage inadapté (CSM) | 4 | Formation planifiée |
| R10 | Données d'usage produites hors UE | 2 | Agrégation en région avant extraction |

**Points d'attention avant déploiement :**
1. R04 — Validation du protocole de groupe témoin par la Direction (condition bloquante).
2. R05 — Clos ; à rouvrir si analyse NLP de `commentaire_csm` envisagée.
3. R02 / R08 — Formation CSM et règle de priorisation à finaliser.

---

*Ce registre est produit conformément à l'item C2 du référentiel CISIA et à l'exigence 7
(Responsabilité) des lignes directrices HLEG de la Commission européenne. Il est référencé
dans le notebook (§4.6) et fait partie des livrables du ZIP final.*
