from pathlib import Path

# Racine du projet : src/churn_saas/config.py → parents[2] = racine du dépôt
RACINE: Path = Path(__file__).resolve().parents[2]

# Chemins dérivés de la racine — jamais de chemins codés en dur ailleurs dans le projet
DONNEES_BRUTES: Path = RACINE / "data" / "raw"
DONNEES_INTERIM: Path = RACINE / "data" / "interim"
DONNEES_GOLD: Path = RACINE / "data" / "gold"
ARTIFACTS: Path = RACINE / "artifacts"
FIGURES: Path = RACINE / "reports" / "figures"
TABLES: Path = RACINE / "reports" / "tables"
MLRUNS: Path = RACINE / "mlruns"
# Sources jupytext du notebook, parcourues pour la traçabilité du référentiel (§0, §15.4)
SECTIONS_NOTEBOOK: Path = RACINE / "notebooks" / "sections"
# Backend de suivi MLflow : SQLite local (le backend fichier pur est déprécié depuis MLflow 2.14)
MLFLOW_TRACKING_URI: str = f"sqlite:///{MLRUNS / 'mlflow.db'}"
# Nom du modèle au registre MLflow ; l'alias « production » désigne la version déployée
NOM_REGISTRE_MLFLOW: str = "churn_saas_champion"

# Graine unique pour tout ce qui est stochastique (split, modèles, Optuna, SHAP)
RANDOM_SEED: int = 42

# Part du jeu de test, mise de côté avant toute modélisation (§9.1), commune au notebook,
# à la CLI et au flow de réentraînement : même découpage stratifié partout
PART_TEST: float = 0.20

# Asymétrie au-delà de laquelle une variable numérique est dite fortement asymétrique (§6.2).
# Sert aussi à LogAsymetrique, log1p signé testé puis écarté en §7.8.3
SEUIL_ASYMETRIE_LOG: float = 1.0

# Redondance (|r de Pearson|, rapport de corrélation η ou V de Cramér) au-delà de laquelle une
# variable est la « jumelle » probable d'une autre : son importance nulle ne prouve alors rien
# Commun au criblage des leurres (§6.10), au verdict (§12.9) et à
# l'audit de commentaire_csm (§6.6)
SEUIL_REDONDANCE: float = 0.70

# Écart (en points de pourcentage) entre taux_adoption_pct et 100 × utilisateurs / sièges
# au-delà duquel une ligne est jugée incohérente (controle_coherence_adoption)
SEUIL_INCOHERENCE_ADOPTION_PTS: float = 1.0

# ---------------------------------------------------------------------------
# Colonnes interdites — ne doivent JAMAIS atteindre le modèle de churn.
# Toute modification ici se propage automatiquement à l'ensemble du pipeline.
# ---------------------------------------------------------------------------
COLONNES_INTERDITES: list[str] = [
    # Identifiant technique : aucune valeur prédictive, risque de mémorisation d'entité
    "client_id",
    # Cible du modèle de churn : fuite directe — c'est ce qu'on cherche à prédire
    "churn",
    # Cible secondaire (régression CLV) : dérivée du comportement futur du client,
    # donc connue *après* la décision de résilier ou de rester
    "valeur_vie_client_eur",
    # Score de santé calculé en FIN de période (après que le churn soit observé) :
    # fuite temporelle pure — un modèle réel n'y aurait pas accès au moment de la prédiction
    "sante_compte_fin_periode",
    # Commentaire du CSM : date de rédaction inconnue, libellés les plus prédictifs liés à un
    # score de santé de fin de période nul — fuite non exclue (audit §6.6)
    "commentaire_csm",
]

# ---------------------------------------------------------------------------
# Colonnes leurres suspectes (4 variables signalées par le cahier des charges).
# À NE PAS supprimer d'office : une preuve chiffrée d'inutilité est plus probante
# (association marginale, permutation et drop-column importance, §6.10 et §12.9)
# qu'une suppression aveugle.
# ---------------------------------------------------------------------------
COLONNES_LEURRES_SUSPECTES: list[str] = [
    # Préférence cosmétique de l'interface utilisateur — sans lien attendu avec le churn
    "couleur_theme_interface",
    # Localisation technique de l'infrastructure hébergeant les données — non liée à la rétention
    "code_datacenter",
    # Affectation à un groupe d'A/B test — peut induire un biais de sélection non contrôlé
    "groupe_experimentation",
    # Commentaire du CSM : suspect criblé avec les leurres, mais pas un leurre au sens strict
    # (13 libellés normalisés, liés au churn, redondants avec l'usage et le support). Requalifié
    # en colonne interdite pour fuite non exclue : voir COLONNES_INTERDITES (§6.6)
    "commentaire_csm",
]

# ---------------------------------------------------------------------------
# Hypothèses économiques utilisées pour le ROI métier (§2, §8, §12).
# Chaque valeur est révisable ; un tableau de sensibilité est prévu en §12.
# Le protocole de révision (comité trimestriel) est décrit en §2.
# ---------------------------------------------------------------------------
HYPOTHESES_ECONOMIQUES: dict[str, float | int] = {
    # Marge brute moyenne du portefeuille SaaS B2B
    # Source : OpenView Partners SaaS Benchmarks 2023 (fourchette observée : 70–75 %)
    "marge_brute_pct": 0.72,
    # Coût horaire chargé d'un Customer Success Manager (France, segment mid-market)
    # Source : Glassdoor 2024 × coefficient charges patronales ≈ 1,45 → ~55 €/h
    "cout_horaire_csm_eur": 55,
    # Durée d'un geste de rétention complet : préparation + appel + suivi CRM
    # Hypothèse de cadrage (absente du cahier des charges) : ~30 min de préparation, ~1 h d'appel,
    # ~30 min de compte rendu CRM et de suivi. À mesurer sur les premiers mois de déploiement.
    "duree_geste_retention_h": 2,
    # Taux de succès d'une intervention ciblée sur un compte identifié à risque élevé
    # Source : Gainsight 2023 Customer Success Industry Report — intervention proactive
    "taux_succes_retention": 0.30,
    # Horizon de calcul de la valeur sauvée, en mois de MRR
    # Source : durée contractuelle typique du portefeuille (renouvellements annuels)
    "horizon_mois": 12,
    # Capacité mensuelle de l'équipe CS (nombre de gestes de rétention réalisables)
    # Hypothèse de cadrage (absente du cahier des charges) : 3 CSM × 15 gestes/mois = 45.
    # 15 gestes × 2 h = 30 h/mois, soit ~20 % d'un temps plein (~150 h/mois) : le reste va à
    # l'onboarding, au support et aux renouvellements. 3 CSM pour ~5 000 comptes (~1 700 par
    # CSM) est très au-delà des repères low-touch (~150 comptes/CSM) : ce sous-dimensionnement
    # est le problème que le modèle adresse, pas une norme.
    "capacite_gestes_mois": 45,
}

# ---------------------------------------------------------------------------
# Paramètres du portefeuille communiqués par le commanditaire au cadrage (§2.4), avant tout
# accès aux données. Confrontés aux valeurs mesurées sur le jeu gold en §12.12.
# ---------------------------------------------------------------------------
PORTEFEUILLE_CADRAGE: dict[str, float | int] = {
    "nb_comptes": 5000,
    # Taux de churn annuel historique, d'après le suivi des renouvellements de la Direction CS
    "taux_churn": 0.28,
    # MRR médian par compte, d'après la facturation (la médiane résiste aux très gros comptes)
    "mrr_median_eur": 450,
}

# ---------------------------------------------------------------------------
# Durées de conservation (§4.1) — choix de gouvernance validés par le DPO, revus chaque année.
# Le RGPD (art. 5.1.e) n'impose aucune durée chiffrée, seulement une durée proportionnée à la
# finalité ; le jeu porte d'ailleurs sur des personnes morales.
# ---------------------------------------------------------------------------
DUREES_CONSERVATION: dict[str, int] = {
    # Scores et prédictions, à compter de `date_prediction` : l'étiquette arrive jusqu'à 12 mois
    # après la prédiction (contrats annuels + fenêtre de 90 jours), puis un an de comparaison
    # d'une année sur l'autre. En deçà, le suivi de performance (§2.6, §13.8) est impossible.
    "predictions_mois": 24,
    # Versions gold des données d'entraînement, à compter de leur création : trois cycles de
    # renouvellement annuel. Repère : référentiel CNIL « gestion commerciale » (3 ans).
    "donnees_entrainement_ans": 3,
    # Journaux techniques (appels API, accès) : repère CNIL pour les journaux de sécurité.
    "journaux_mois": 12,
}

# ---------------------------------------------------------------------------
# Paliers de risque affichés au CSM (API, batch, fiches comptes §12.11) — autorité unique,
# appliquée par `economie.niveau_risque`. Ce sont des repères de lecture du score, fixés avec le
# CS Lead : SURVEILLANCE dès que le risque dépasse nettement la prévalence (~28 %), ALERTE_ROUGE
# dès que le départ est plus probable que le maintien, avec une marge. Ils ne décident pas qui
# appeler : c'est la priorisation par valeur attendue sous capacité (§12.6, `rang_priorite`).
# ---------------------------------------------------------------------------
SEUILS_RISQUE: dict[str, float] = {
    "surveillance": 0.40,
    "alerte_rouge": 0.60,
}

# ---------------------------------------------------------------------------
# Seuils de la règle métier — fixés a priori, jamais ajustés sur le jeu de test
#
# • `derniere_connexion_jours > 30` : la fenêtre d'activité de référence du produit
#   (colonnes `*_30j`) ; au-delà, le compte n'a plus aucune activité sur la période.
# • `csat <= 2` : convention standard du CSAT sur 5 points — 1 et 2 = « insatisfait »,
#   3 = neutre, 4 et 5 = « satisfait ». Le jeu fourni note le CSAT de 1 à 5 (contrôlé en §5.8).
#   Un seuil conventionnel, et non optimisé sur les données, ne peut pas avoir été calé sur le
#   jeu de test.
# Partagés par la baseline B1 (`models.train.regle_metier`) et les facteurs de l'API : ils vivent
# ici pour que l'API n'importe pas le module d'entraînement (MLflow, Optuna absents de l'image).
# ---------------------------------------------------------------------------
SEUIL_INACTIVITE_JOURS: int = 30
SEUIL_CSAT: int = 2

# ---------------------------------------------------------------------------
# Audit d'équité (protocole §4.4, mesure sur le modèle final §12.14) — seuils fixés a priori.
# Critère : égalité des chances (Hardt, Price et Srebro, 2016) — à churn égal, un compte doit
# avoir la même probabilité d'être signalé, quel que soit son pays, sa taille ou son secteur.
# Aucune norme ne chiffre l'écart tolérable : les valeurs ci-dessous sont des choix de
# gouvernance, validés avec le CS Lead et revus chaque année, au même titre que les durées de
# conservation.
# ---------------------------------------------------------------------------
EQUITE: dict[str, float | int] = {
    # Écart maximal de TPR (rappel) entre modalités d'un même attribut au seuil de décision.
    # Au-delà : revue obligatoire avant déploiement (registre des risques, R01).
    "ecart_tpr_max": 0.15,
    # Écart maximal |probabilité moyenne prédite − taux de churn réel| dans une modalité. La
    # priorisation par valeur attendue (§12.6) multiplie la probabilité par le MRR : un segment
    # mal calibré serait systématiquement sur- ou sous-priorisé.
    "ecart_calibration_max": 0.05,
    # Nombre minimal de churners pour interpréter un TPR : en deçà, l'intervalle de confiance
    # (±16 points environ pour 30 churners) est aussi large que l'écart que l'on cherche à détecter.
    "churners_min": 30,
}

# ---------------------------------------------------------------------------
# Hypothèses métier (§2), formulées avec la Direction CS avant toute exploration des données,
# puis confrontées aux données en EDA (§6). Tuple (id, variable, sens_attendu, libellé).
# `sens_attendu` est le sens de l'association entre la variable et le churn : « + » = plus la
# variable est élevée, plus le risque de churn est élevé ; « - » = l'inverse.
# H1 à H8 décrivent des signaux de risque élevé, H9 à H14 des signaux de risque faible.
# ---------------------------------------------------------------------------
HYPOTHESES_METIER: list[tuple[str, str, str, str]] = [
    # --- Risque élevé ---
    ("H1", "taux_adoption_pct", "-", "Un faible taux d'adoption augmente le risque de churn"),
    ("H2", "connexions_30j", "-", "Peu de connexions sur 30 jours augmentent le risque de churn"),
    ("H3", "heures_usage_30j", "-", "Peu d'heures d'usage sur 30 jours augmentent le risque"),
    (
        "H4",
        "fonctionnalites_utilisees",
        "-",
        "Peu de fonctionnalités utilisées augmentent le risque de churn",
    ),
    (
        "H5",
        "derniere_connexion_jours",
        "+",
        "Une longue période depuis la dernière connexion augmente le risque de churn",
    ),
    ("H6", "tickets_support_90j", "+", "De nombreux tickets support augmentent le risque"),
    ("H7", "csat", "-", "Une faible satisfaction (CSAT) augmente le risque de churn"),
    ("H8", "retards_paiement_12m", "+", "Des retards de paiement augmentent le risque de churn"),
    # --- Risque faible ---
    # Quasi-miroir de H1 : taux_adoption_pct ≈ 100 × utilisateurs / sièges (contrôle de
    # cohérence en §5). Gardée car elle porte sur la feature dérivée vue par le modèle, mais
    # le test sera pratiquement le même que H1.
    (
        "H9",
        "taux_utilisation_sieges",
        "-",
        "Une forte adoption (sièges utilisés / souscrits) réduit le risque de churn",
    ),
    # Distincte de H2 : rapportée au nombre d'utilisateurs actifs, elle mesure la régularité
    # de l'usage et non le volume brut, qui croît mécaniquement avec la taille du compte.
    (
        "H10",
        "connexions_par_utilisateur",
        "-",
        "Un usage régulier (connexions par utilisateur actif) réduit le risque de churn",
    ),
    ("H11", "nb_integrations", "-", "De nombreuses intégrations réduisent le risque de churn"),
    # Distincte de H4 : rapportée au nombre de fonctionnalités du plan souscrit.
    (
        "H12",
        "taux_couverture_fonctionnelle",
        "-",
        "Une forte couverture fonctionnelle réduit le risque de churn",
    ),
    # Miroir exact de H7 si elle était testée sur le CSAT brut : elle porte sur le haut de la
    # distribution (CSAT ≥ quantile QUANTILE_HAUT_DISTRIBUTION), pour vérifier qu'une
    # satisfaction élevée protège, et pas seulement qu'une satisfaction faible expose.
    (
        "H13",
        "csat",
        "-",
        "Une satisfaction élevée (haut de la distribution du CSAT) réduit le risque de churn",
    ),
    ("H14", "anciennete_mois", "-", "Une ancienneté importante réduit le risque de churn"),
]

# Quantile délimitant le « haut de distribution » d'une variable (H13 : quartile supérieur)
QUANTILE_HAUT_DISTRIBUTION: float = 0.75
# Hypothèses testées sur le haut de distribution (indicateur variable ≥ quantile ci-dessus)
# plutôt que sur la variable brute : sans cela, H13 serait un doublon exact de H7
HYPOTHESES_HAUT_DISTRIBUTION: frozenset[str] = frozenset({"H13"})

# ---------------------------------------------------------------------------
# Recall cible du seuil de vigilance (premier niveau de la règle de décision, §12) : le seuil
# est choisi sur les prédictions out-of-fold pour détecter 80 % des churners.
# Pourquoi 0,80 : un faux négatif coûte bien plus cher qu'une surveillance inutile. D'après
# `models.economics.matrice_couts`, un churner manqué perd MRR × horizon × marge (≈ 8,6 mois
# de MRR, soit ≈ 3 900 € au MRR médian de cadrage), alors qu'une surveillance inutile coûte un
# geste CSM (≈ 110 €) : un rapport d'environ 35 pour 1. Viser plus haut (0,90) ferait exploser
# le nombre de comptes signalés pour peu de churners supplémentaires ; la capacité de l'équipe
# CS est gérée au second niveau (priorisation par valeur attendue).
# ---------------------------------------------------------------------------
RECALL_CIBLE_VIGILANCE: float = 0.80

# ---------------------------------------------------------------------------
# Coûts du projet de ML lui-même, pour le ROI du projet (§12.12) : développement amorti,
# infrastructure et maintenance. Hypothèses de cadrage, à confirmer par le contrôle de gestion.
# ---------------------------------------------------------------------------
COUTS_PROJET: dict[str, float | int] = {
    # Coût journalier chargé d'un profil technique (data scientist, ingénieur DSI) ;
    # ordre de grandeur du marché français, appliqué aussi au temps interne
    "cout_journalier_eur": 600,
    # Développement : ~3 mois d'un data scientist à mi-temps (§2)
    "jours_build_data_scientist": 60,
    # Intégration DSI : déploiement du scénario B, champ Salesforce, batch nocturne (§10, §11)
    "jours_build_dsi": 10,
    # Cadrage et recette par l'équipe Customer Success
    "jours_build_cs": 5,
    # Amortissement linéaire : durée de vie attendue de la solution avant refonte
    "duree_amortissement_ans": 3,
    # Infrastructure du scénario B, haut de la fourchette de §11.7
    "infra_mensuel_eur": 110,
    # Maintenance : surveillance, réentraînement, incidents, MLflow (2 – 4 h/mois, §11.6)
    "jours_maintenance_mois": 2,
}

# ---------------------------------------------------------------------------
# Cibles de performance — fixées A PRIORI, avant toute modélisation.
# Principe : les seuils sont définis avant les résultats pour éviter
# le cherry-picking de métriques a posteriori.
# ---------------------------------------------------------------------------
CIBLES_PERFORMANCE: dict[str, float | int] = {
    # PR-AUC : métrique principale pour la classification déséquilibrée (~28 % de churn, §6).
    # Un modèle aléatoire donnerait PR-AUC ≈ 0.28 (la prévalence) ; le seuil de 0.65 représente
    # le gain minimal justifiant le coût de déploiement et d'exploitation.
    "pr_auc_min": 0.65,
    # ROC-AUC : probabilité qu'un churner tiré au hasard reçoive un score plus élevé qu'un
    # fidèle tiré au hasard (0.5 = hasard). 0.80 est le seuil usuel d'un bon pouvoir de
    # classement ; co-principale avec la PR-AUC, insensible à la prévalence.
    "roc_auc_min": 0.80,
    # Recall minimal mesuré sur le jeu de test au seuil de vigilance. Le seuil vise
    # RECALL_CIBLE_VIGILANCE (0.80) sur les prédictions out-of-fold ; 5 points de tolérance
    # absorbent le bruit d'estimation du passage out-of-fold → test (intervalle de confiance
    # à 95 % de ±5 points environ pour ~280 churners de test). En deçà, le seuil ne tient pas
    # sa promesse de détection.
    "recall_vigilance_min": 0.75,
    # Latence d'inférence unitaire (ms, p95) — mode API synchrone, un compte à la fois.
    # Contrainte : appel à l'ouverture d'une fiche client dans le CRM (CU3), un CSM attend.
    "latence_unitaire_ms": 200,
    # Latence du batch pour 5 000 comptes (secondes) — job nocturne quotidien.
    # Contrainte : le batch doit tenir dans la fenêtre de maintenance (< 5 min).
    "latence_batch_5k_s": 300,
    # Gain minimal de PR-AUC moyenne (validation croisée) qu'un modèle plus complexe doit
    # apporter sur un modèle plus simple non rejeté, en plus d'un écart significatif
    # (Wilcoxon + Holm) : engagement d'éco-conception transmis au commanditaire (§8.4, §8.8).
    "gain_pr_auc_min_complexite": 0.02,
    # Tolérance de régression de la gate de promotion (flow de réentraînement, §10.4, §13.8) :
    # un challenger est promu s'il atteint `pr_auc_min` et ne perd pas plus de 0.01 de PR-AUC
    # face au champion sur le split de contrôle — un réentraînement sur données à jour ne doit
    # pas être bloqué par le bruit d'estimation, mais une vraie régression l'est.
    "tolerance_regression_promotion": 0.01,
    # Seuil critique de PR-AUC en production, mesurée sur une cohorte de renouvellements
    # (§2.6, §13.8) : sous `pr_auc_min`, comité ad hoc ; sous ce seuil, comité ad hoc et
    # comparaison avec le champion précédent sur la même cohorte, retour arrière s'il fait mieux.
    "pr_auc_critique": 0.60,
    # Écart (en points de pourcentage) de la part de décisions ALERTE_ROUGE du batch nocturne
    # par rapport à la référence au-delà duquel on revient au champion précédent après une
    # promotion récente (`docs/RUNBOOK.md` §4.3, §13.8) : dérive de sortie visible chaque nuit.
    "ecart_alerte_rouge_rollback_pts": 10,
    # Budget carbone d'une session d'entraînement complète (comparaison + tuning), en g CO₂e.
    # Fixé a priori (§8.2), confronté à la mesure CodeCarbon en §12.13.
    "co2_entrainement_max_g": 50,
    # --- Cible secondaire : régression de valeur_vie_client_eur (CLV), jeu de test, échelle € ---
    # R² minimal : le modèle doit expliquer au moins la moitié de la variance de la CLV. C'est un
    # plancher prudent : §6.7 montre que le MRR seul en explique déjà une large part
    # (r(CLV, MRR) > 0.80) ; l'exigence de valeur ajoutée est portée par le gain de MAE.
    "clv_r2_min": 0.50,
    # Gain minimal de MAE vs la baseline naïve (médiane pour tous, DummyRegressor) : 30 %.
    # En deçà, une valeur par défaut suffit et le modèle ne justifie pas sa maintenance.
    "clv_gain_mae_min": 0.30,
    # Garde-fou : un R² au-delà de ce seuil est présumé être une fuite (reconstruction mécanique
    # de la cible) et impose un audit avant toute acceptation du modèle.
    "clv_r2_alerte_fuite": 0.99,
}

# Tarif indicatif de l'électricité en France (€/kWh, 2024, source : Eurostat) : sert à
# convertir en euros l'énergie consommée par le tuning (§9.9). Ordre de grandeur seulement.
TARIF_ELECTRICITE_EUR_KWH: float = 0.18
