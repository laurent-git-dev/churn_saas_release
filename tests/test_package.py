"""Test trivial : vérifie que le paquet est importable et que la config est cohérente."""
import churn_saas
import churn_saas.config as cfg


def test_paquet_importable() -> None:
    """Le paquet churn_saas doit s'importer sans erreur."""
    assert churn_saas is not None


def test_random_seed_fixe() -> None:
    """La graine aléatoire doit valoir 42 (invariant unique du projet)."""
    assert cfg.RANDOM_SEED == 42


def test_colonnes_interdites_non_vides() -> None:
    """La liste des colonnes interdites doit contenir au moins la cible churn."""
    assert "churn" in cfg.COLONNES_INTERDITES
    assert "client_id" in cfg.COLONNES_INTERDITES
