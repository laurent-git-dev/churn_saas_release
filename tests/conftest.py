"""Fixtures partagées — rend la suite exécutable sans les données brutes (CI).

`data/raw/` fait partie des livrables mais n'est pas versionné : sur un runner GitHub,
le catalogue des plans est absent. Les tests qui utilisent un modèle mocké n'ont pas
besoin des *vraies* valeurs du catalogue — seulement d'un catalogue de même schéma —,
d'où le repli synthétique ci-dessous. Les tests qui exigent les vrais artefacts
(`tests/test_api_model_contract.py`) sont, eux, ignorés faute de `best_model.pkl`.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest

from churn_saas.api import main as api_main

# Même schéma que data/raw/catalogue_plans.csv — valeurs synthétiques, ordres de grandeur
# réalistes. Ne sert qu'aux tests qui n'évaluent pas le vrai modèle.
_CATALOGUE_SYNTHETIQUE = pd.DataFrame(
    {
        "plan": ["Starter", "Pro", "Business", "Enterprise"],
        "prix_mensuel_par_siege_eur": [15, 30, 50, 80],
        "fonctionnalites_incluses": [10, 20, 30, 40],
        "sla_reponse_h": [24, 12, 8, 4],
        "quota_stockage_go": [50, 250, 1000, 2000],
        "support_dedie": ["non", "non", "oui", "oui"],
    }
)


@pytest.fixture(autouse=True)
def catalogue_de_reference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[None]:
    """Garantit un catalogue des plans exploitable, réel s'il existe, synthétique sinon.

    En environnement complet (poste de développement), les tests s'exécutent contre les
    vraies données : la fixture ne fait rien. En CI, elle substitue un catalogue de même
    schéma pour que l'API reste scorable — sans jamais masquer l'absence des données au
    reste du code, puisque `_CHEMIN_CATALOGUE` pointe alors vers un fichier bien réel.
    """
    api_main._catalogue.cache_clear()
    if not api_main._CHEMIN_CATALOGUE.exists():
        chemin: Path = tmp_path_factory.mktemp("donnees_reference") / "catalogue_plans.csv"
        _CATALOGUE_SYNTHETIQUE.to_csv(chemin, index=False)
        monkeypatch.setattr(api_main, "_CHEMIN_CATALOGUE", chemin)
    yield
    api_main._catalogue.cache_clear()
