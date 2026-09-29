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

# Graine unique pour tout ce qui est stochastique (split, modèles, Optuna, SHAP)
RANDOM_SEED: int = 42

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
]

# ---------------------------------------------------------------------------
# Colonnes leurres suspectes (4 leurres annoncés dans l'énoncé).
# À NE PAS supprimer d'office : le jury attend une preuve chiffrée d'inutilité
# (permutation importance nulle, SHAP ≈ 0), pas une suppression aveugle.
# ---------------------------------------------------------------------------
COLONNES_LEURRES_SUSPECTES: list[str] = [
    # Préférence cosmétique de l'interface utilisateur — sans lien attendu avec le churn
    "couleur_theme_interface",
    # Localisation technique de l'infrastructure hébergeant les données — non liée à la rétention
    "code_datacenter",
    # Affectation à un groupe d'A/B test — peut induire un biais de sélection non contrôlé
    "groupe_experimentation",
    # Texte libre non structuré, très hétérogène — importance marginale sans NLP dédié
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
    # Hypothèse de cadrage (l'énoncé ne la fournit pas) : ~30 min de préparation, ~1 h d'appel,
    # ~30 min de compte rendu CRM et de suivi. À mesurer sur les premiers mois de déploiement.
    "duree_geste_retention_h": 2,
    # Taux de succès d'une intervention ciblée sur un compte identifié à risque élevé
    # Source : Gainsight 2023 Customer Success Industry Report — intervention proactive
    "taux_succes_retention": 0.30,
    # Horizon de calcul de la valeur sauvée, en mois de MRR
    # Source : durée contractuelle typique du portefeuille (renouvellements annuels)
    "horizon_mois": 12,
    # Capacité mensuelle de l'équipe CS (nombre de gestes de rétention réalisables)
    # Hypothèse de cadrage (l'énoncé ne la fournit pas) : 3 CSM × 15 gestes/mois = 45.
    # 15 gestes × 2 h = 30 h/mois, soit ~20 % d'un temps plein (~150 h/mois) : le reste va à
    # l'onboarding, au support et aux renouvellements. 3 CSM pour ~5 000 comptes (~1 700 par
    # CSM) est très au-delà des repères low-touch (~150 comptes/CSM) : ce sous-dimensionnement
    # est le problème que le modèle adresse, pas une norme.
    "capacite_gestes_mois": 45,
}

# ---------------------------------------------------------------------------
# Cibles de performance — fixées A PRIORI, avant toute modélisation.
# Principe CISIA : les seuils sont définis avant les résultats pour éviter
# le cherry-picking de métriques a posteriori.
# ---------------------------------------------------------------------------
CIBLES_PERFORMANCE: dict[str, float | int] = {
    # PR-AUC : métrique principale pour la classification déséquilibrée (~28 % de churn, §6).
    # Un modèle aléatoire donnerait PR-AUC ≈ 0.28 (la prévalence) ; le seuil de 0.65 représente
    # le gain minimal justifiant le coût de déploiement et d'exploitation.
    "pr_auc_min": 0.65,
    # Latence d'inférence unitaire (ms, p95) — mode API synchrone, un compte à la fois.
    # Contrainte : appel à l'ouverture d'une fiche client dans le CRM (CU3), un CSM attend.
    "latence_unitaire_ms": 200,
    # Latence du batch pour 5 000 comptes (secondes) — job nocturne quotidien.
    # Contrainte : le batch doit tenir dans la fenêtre de maintenance (< 5 min).
    "latence_batch_5k_s": 300,
}
