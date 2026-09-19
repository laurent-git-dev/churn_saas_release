# Registre des risques — Modèle de prédiction du churn SaaS B2B

**Projet :** Churn SaaS B2B — certification CISIA
**Version :** 1.0
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
| **Probabilité** | 2 — Moyen (biais de représentation fréquents dans les données SaaS B2B, mesurés en §4.4) |
| **Impact** | 2 — Moyen (discrimination commerciale perçue, risque réputationnel) |
| **Score** | 4 |
| **Mitigation** | Audit de fairness (TPR/FPR par sous-groupe) à chaque réentraînement. Seuil d'alerte : écart de TPR > 15 points entre sous-groupes → revue obligatoire avant déploiement. Résultats reportés dans §12 et §13. |
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
| **Mitigation secondaire** | Monitoring Evidently (PSI/KS) sur la distribution des features et des scores (§13). Réentraînement déclenché si dérive > seuil. |
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
| **Mitigation** | Variable exclue du modèle (config.COLONNES_LEURRES_SUSPECTES, §7). Validation DPO documentée (fiche de revue §4.6.3). Si analyse NLP de ce champ envisagée à l'avenir : analyse d'impact (AIPD) obligatoire au préalable. |
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
| **Mitigation** | Calibration systématique du modèle champion (courbe de fiabilité + score de Brier — §9). Mention explicite dans le guide d'usage CSM : le score est un *indicateur de risque relatif*, non une probabilité absolue. |
| **Propriétaire** | Data Scientist |
| **Statut** | Mesuré — calibration documentée en §9 |

---

### R07 — Obsolescence non détectée du modèle

| Champ | Valeur |
|---|---|
| **Description** | La distribution des données évolue (nouveaux segments, changement de produit, crise sectorielle). Le modèle, entraîné sur des données historiques, peut devenir progressivement inadapté sans que la dégradation soit visible si aucun monitoring n'est en place. |
| **Probabilité** | 2 — Moyen (inévitable à long terme, délai variable selon la stabilité du contexte) |
| **Impact** | 3 — Élevé (décisions CS basées sur un modèle inadapté = gestion des churns manquée) |
| **Score** | 6 → **Escalade Direction** |
| **Mitigation** | Monitoring Evidently (PSI/KS) sur les features et les prédictions (§13). Indicateur d'obsolescence intégré au tableau de bord (§13). Déclenchement automatique du réentraînement si PSI > 0,2 ou AUC de production < seuil (§13). Comité de revue trimestriel. |
| **Propriétaire** | Data Scientist / MLOps |
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

## Synthèse

| ID | Risque | Score | Statut |
|---|---|---|---|
| R04 | Boucle de rétroaction | **9** — Critique | Validation Direction en attente |
| R05 | Fuite données perso (commentaire_csm) | **9** — Critique | **Clos** (exclusion effective) |
| R02 | Prophétie auto-réalisatrice | **6** — Élevé | Mitigation en cours |
| R07 | Obsolescence non détectée | **6** — Élevé | Outillage en place |
| R01 | Biais par sous-groupe | 4 | Surveillance continue |
| R03 | Remises perverses | 4 | Suivi trimestriel |
| R06 | Modèle non calibré | 4 | Calibration documentée |
| R08 | Usage inadapté (CSM) | 4 | Formation planifiée |

**Points d'attention avant déploiement :**
1. R04 — Validation du protocole de groupe témoin par la Direction (condition bloquante).
2. R05 — Clos ; à rouvrir si analyse NLP de `commentaire_csm` envisagée.
3. R02 / R08 — Formation CSM et règle de priorisation à finaliser.

---

*Ce registre est produit conformément à l'item C2 du référentiel CISIA et à l'exigence 7
(Responsabilité) des lignes directrices HLEG de la Commission européenne. Il est référencé
dans le notebook (§4.6) et fait partie des livrables du ZIP final.*
