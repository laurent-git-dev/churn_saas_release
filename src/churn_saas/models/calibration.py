"""Régression logistique pondérée dont les probabilités restent calibrées.

`class_weight='balanced'` rééquilibre les classes pendant l'apprentissage, mais il décale
les probabilités : pour une régression logistique, la pondération ajoute
``log(w₁ / w₀)`` au logit, soit ``log((1 − π) / π)`` avec π la prévalence — les cotes sont
multipliées par ≈ 2,6 pour π = 28 %. Le modèle surestime alors le risque (§12.2.3), ce qui
fausse toute grandeur en euros calculée à partir de P(churn) (valeur à risque, valeur attendue
d'une intervention).

La correction est analytique (« prior correction », King & Zeng, 2001) : on retranche
``log(w₁ / w₀)`` à l'intercept après l'apprentissage. Aucun paramètre supplémentaire n'est
appris, les poids de classes sont ceux du jeu d'entraînement du pli, et le modèle reste une
`LogisticRegression` (mêmes coefficients) : explicabilité, SHAP et API inchangés.

Module volontairement léger (numpy + scikit-learn) : il est importé par le conteneur
d'inférence au chargement du modèle sérialisé.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight


class RegressionLogistiqueRecalibree(LogisticRegression):  # type: ignore[misc]
    """`LogisticRegression` pondérée, intercept corrigé pour des probabilités calibrées.

    Mêmes paramètres que `LogisticRegression`. Classification binaire uniquement.
    Les coefficients sont ceux du modèle pondéré ; seul l'intercept est corrigé de
    ``correction_intercept_ = log(w₁ / w₀)``, nul si ``class_weight`` vaut ``None``.
    Le classement des comptes (PR-AUC, ROC-AUC) est donc inchangé.

    ``moyenne_entree_`` mémorise la moyenne des variables d'apprentissage : c'est la référence
    des valeurs SHAP exactes calculées par l'API (`models.explication_locale`).
    """

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> RegressionLogistiqueRecalibree:
        super().fit(X, y, sample_weight=sample_weight)
        if len(self.classes_) != 2:
            raise ValueError("RegressionLogistiqueRecalibree : classification binaire uniquement.")
        if self.class_weight is None:
            self.correction_intercept_ = 0.0
        else:
            poids = compute_class_weight(self.class_weight, classes=self.classes_, y=np.asarray(y))
            self.correction_intercept_ = float(np.log(poids[1] / poids[0]))
        # intercept_ est posé par LogisticRegression.fit, non typé côté scikit-learn
        self.intercept_ = self.intercept_ - self.correction_intercept_  # type: ignore[has-type]
        # Référence des valeurs SHAP linéaires servies par l'API (`models.explication_locale`) :
        # moyenne de chaque variable sur les données d'apprentissage, comme LinearExplainer
        self.moyenne_entree_ = np.asarray(X.mean(axis=0), dtype=float).ravel()
        return self
