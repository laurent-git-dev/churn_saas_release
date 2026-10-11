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


def test_hypotheses_metier_bien_formees() -> None:
    """H1 à H14 : identifiants uniques et ordonnés, sens ∈ {+, -}, libellé non vide."""
    ids = [h[0] for h in cfg.HYPOTHESES_METIER]
    assert ids == [f"H{i}" for i in range(1, 15)]
    for _, variable, sens, libelle in cfg.HYPOTHESES_METIER:
        assert variable and libelle
        assert sens in {"+", "-"}
    assert 0.5 < cfg.QUANTILE_HAUT_DISTRIBUTION < 1


def test_hypotheses_metier_hors_colonnes_interdites() -> None:
    """Aucune hypothèse ne porte sur une colonne interdite (cible, fuite, identifiant)."""
    variables = {h[1] for h in cfg.HYPOTHESES_METIER}
    assert not variables & set(cfg.COLONNES_INTERDITES)


def test_cibles_recall_et_roc_auc() -> None:
    """Recall de vigilance et ROC-AUC minimale fixés a priori, dans ]0, 1[."""
    assert cfg.RECALL_CIBLE_VIGILANCE == 0.80
    cibles = cfg.CIBLES_PERFORMANCE
    assert 0.5 < cibles["roc_auc_min"] < 1
    assert 0 < cibles["recall_vigilance_min"] <= cfg.RECALL_CIBLE_VIGILANCE
