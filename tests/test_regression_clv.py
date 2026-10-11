"""Tests de l'entraînement des modèles CLV (`entrainer_modeles_clv`).

Le challenger non linéaire de la régression CLV est LightGBM, comme pour la cible principale :
une seule famille de boosting dans tout le projet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from churn_saas import config
from churn_saas.models import regression as reg


def _jeu_synthetique(n: int = 200) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(config.RANDOM_SEED)
    mrr = rng.uniform(100, 5_000, n)
    X = pd.DataFrame(
        {
            "revenu_mensuel_recurrent_eur": mrr,
            "anciennete_mois": rng.integers(1, 60, n).astype(float),
            "segment": rng.choice(["PME", "ETI", "GE"], n),
        }
    )
    y = pd.Series(mrr * rng.uniform(10, 30, n), name="valeur_vie_client_eur")
    return X, y


def test_challenger_clv_est_lightgbm() -> None:
    X, y = _jeu_synthetique()
    resultats = reg.entrainer_modeles_clv(X, y, forcer=True)

    assert set(resultats) == {"baseline", "ridge", "lgbm", "ridge_log", "lgbm_log"}
    for nom in ("lgbm", "lgbm_log"):
        estimateur = resultats[nom]["modele"].named_steps["reg"]
        assert isinstance(estimateur, LGBMRegressor)
        assert estimateur.random_state == config.RANDOM_SEED
