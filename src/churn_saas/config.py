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
    # Source : estimation CS Lead consolidée sur 3 mois de données de tickets internes
    "duree_geste_retention_h": 2,
    # Taux de succès d'une intervention ciblée sur un compte identifié à risque élevé
    # Source : Gainsight 2023 Customer Success Industry Report — intervention proactive
    "taux_succes_retention": 0.30,
    # Horizon de calcul de la valeur sauvée, en mois de MRR
    # Source : durée contractuelle typique du portefeuille (renouvellements annuels)
    "horizon_mois": 12,
    # Capacité mensuelle de l'équipe CS (nombre de gestes de rétention réalisables)
    # Source : CS Lead — 3 CSM × ~15 gestes ciblés/mois = 45
    "capacite_gestes_mois": 45,
}

# ---------------------------------------------------------------------------
# Cibles de performance — fixées A PRIORI, avant toute modélisation.
# Principe CISIA : les seuils sont définis avant les résultats pour éviter
# le cherry-picking de métriques a posteriori.
# ---------------------------------------------------------------------------
CIBLES_PERFORMANCE: dict[str, float | int] = {
    # PR-AUC : métrique principale pour la classification déséquilibrée (~15–20 % de churn).
    # Un modèle aléatoire donnerait PR-AUC ≈ 0.17 ; le seuil de 0.65 représente
    # le gain minimal justifiant le coût de déploiement et d'exploitation.
    "pr_auc_min": 0.65,
    # Latence d'inférence unitaire (ms) — mode API synchrone, un compte à la fois.
    # Contrainte : intégration dans un webhook CRM déclenché à la date de renouvellement.
    "latence_unitaire_ms": 200,
    # Latence du batch pour 5 000 comptes (secondes) — job nocturne hebdomadaire.
    # Contrainte : le batch doit tenir dans la fenêtre de maintenance (< 5 min).
    "latence_batch_5k_s": 300,
}
