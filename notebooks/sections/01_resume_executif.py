# %% [markdown]
# ## 1. Résumé exécutif
#
# Synthèse à l'intention d'un lecteur pressé — contexte, approche, résultats clés,
# recommandation et limites. Tous les chiffres sont produits par le code visible ;
# le jury peut relancer le notebook pour les vérifier.

# %%
import glob
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import Markdown, display
from sklearn.metrics import average_precision_score, roc_auc_score

from churn_saas import config
from churn_saas.cache import charger_ou_calculer
from churn_saas.features.build import ajouter_features_metier
from churn_saas.models.economics import gain_par_seuil, precision_at_k

warnings.filterwarnings("ignore")

# ── Données gold ─────────────────────────────────────────────────────────────
gold = pd.read_parquet(config.DONNEES_GOLD / "gold_dataset.parquet")
CIBLE = "churn"
X_brut = gold.drop(columns=config.COLONNES_INTERDITES, errors="ignore")
y = gold[CIBLE].astype(int)
X = ajouter_features_metier(X_brut)
X = X.drop(columns=[CIBLE], errors="ignore")
MRR = pd.to_numeric(gold["revenu_mensuel_recurrent_eur"], errors="coerce")
MRR = MRR.fillna(float(MRR.median()))

# ── Artefacts produits en §9 ──────────────────────────────────────────────────
tableau_modeles, _ = charger_ou_calculer("comparaison_modeles.parquet", lambda: pd.DataFrame())

_nom_champion = tableau_modeles.index[0] if len(tableau_modeles) > 0 else "N/A"
_pr_auc_cv_champion = (
    float(tableau_modeles["pr_auc_mean"].iloc[0]) if "pr_auc_mean" in tableau_modeles.columns else float("nan")
)
_pr_b1 = (
    float(tableau_modeles.loc["B1 — Règle métier", "pr_auc_mean"])
    if "B1 — Règle métier" in tableau_modeles.index
    else float("nan")
)

_optuna_files = sorted(Path(config.TABLES).glob("optuna_*_meilleurs_params.json"))
_optuna = json.loads(_optuna_files[0].read_text(encoding="utf-8")) if _optuna_files else {}
_pr_auc_optuna = float(_optuna.get("best_value", float("nan")))
_best_params = _optuna.get("best_params", {})
_n_essais = int(_optuna.get("n_essais_completes", 0))

# ── Artefacts produits en §12 ─────────────────────────────────────────────────
# probas_oof.joblib est écrit par §12. Sur un premier run propre (make notebook-full),
# §12 n'a pas encore tourné ; la lambda retourne None et le fallback ci-dessous
# évite le crash. Sur make notebook (cache actif), le fichier existe déjà.
proba_oof, _ = charger_ou_calculer(
    "probas_oof.joblib",
    lambda: None,
)

if proba_oof is None:
    display(
        Markdown(
            "⚠️ **`probas_oof.joblib` absent** — ce fichier est produit par §12. "
            "Les métriques OOF affichées ci-dessous sont des **valeurs de substitution** "
            "(prédicteur aléatoire à la prévalence). "
            "Relancer `make notebook` après une première exécution complète pour obtenir "
            "les vraies valeurs."
        )
    )
    # Fallback : prédicteur aléatoire constant (PR-AUC ≈ prévalence, ROC-AUC ≈ 0,50)
    proba_oof = np.full(len(y), float(y.mean()), dtype=float)

proba_oof = np.asarray(proba_oof, dtype=float)

pr_auc_oof = float(average_precision_score(y, proba_oof))
roc_auc_oof = float(roc_auc_score(y, proba_oof))

_, courbe_gain, seuil_opt = gain_par_seuil(y, proba_oof, MRR)
_idx_opt = courbe_gain["gain_net_eur"].idxmax()
_gain_net_eur = float(courbe_gain.loc[_idx_opt, "gain_net_eur"])
_precision_seuil = float(courbe_gain.loc[_idx_opt, "precision"])
_rappel_seuil = float(courbe_gain.loc[_idx_opt, "rappel"])

_cap = int(config.HYPOTHESES_ECONOMIQUES["capacite_gestes_mois"])
precision_cap = precision_at_k(y, proba_oof, k=_cap)
_idx_top_n = np.argsort(-proba_oof)[:_cap]
_y_top_n = y.values[_idx_top_n]
_mrr_top_n = MRR.values[_idx_top_n]
_n_churners_detectes = int(_y_top_n.sum())
_mrr_couvert = float((_mrr_top_n * _y_top_n).sum())
_mrr_sauve = (
    _mrr_couvert
    * config.HYPOTHESES_ECONOMIQUES["taux_succes_retention"]
    * config.HYPOTHESES_ECONOMIQUES["marge_brute_pct"]
)
_gain_annuel = _mrr_sauve * config.HYPOTHESES_ECONOMIQUES["horizon_mois"]
_cout_mensuel_cs = (
    _cap
    * config.HYPOTHESES_ECONOMIQUES["cout_horaire_csm_eur"]
    * config.HYPOTHESES_ECONOMIQUES["duree_geste_retention_h"]
)
# ROI par cohorte (définition de §12.12) : les gestes d'un mois retiennent des contrats qui
# rapportent sur tout l'horizon, d'où la comparaison de _gain_annuel au coût du mois.
_roi = (_gain_annuel - _cout_mensuel_cs) / _cout_mensuel_cs if _cout_mensuel_cs > 0 else 0.0

# %%
display(
    Markdown(
        f"### Problème métier\n\n"
        f"Un éditeur SaaS B2B gère un portefeuille de **{len(y):,} comptes** actifs "
        "(PME et grandes entreprises européennes, abonnements annuels). L'équipe Customer Success "
        "(3 CSM) ne peut pas surveiller manuellement l'ensemble du portefeuille : "
        "les signaux de résiliation arrivent trop tard, après que le client a déjà décidé de partir. "
        "L'objectif est de produire, chaque semaine, un **score de risque de churn à 30 jours** "
        "pour chaque compte, afin de prioriser les interventions CS avant l'échéance contractuelle.\n\n"
        f"La prévalence de churn dans le dataset est de **{y.mean():.1%}** (classe déséquilibrée) : "
        "c'est pourquoi la **PR-AUC** est la métrique principale — elle pénalise à la fois les faux "
        "positifs et les faux négatifs, contrairement à la ROC-AUC insensible au déséquilibre."
    )
)

# %% [markdown]
# ### Approche retenue
#
# Cinq modèles comparés sur le **même protocole** (RepeatedStratifiedKFold 5×3 = 15 plis)
# pour garantir des estimations fiables malgré la taille modeste du dataset (~5 000 observations).
# Toutes les transformations sont apprises **à l'intérieur** de chaque pli (anti-fuite).
# La métrique principale est la PR-AUC ; le seuil de décision est choisi par maximisation du
# gain net espéré (et non 0,5 ni l'argmax du F1, qui n'ont pas de sens économique).

# %%
display(
    Markdown(
        f"""
**Architecture de la solution :**

| Composant | Valeur retenue | Justification |
|---|---|---|
| Modèle champion | `{_nom_champion}` | Meilleur PR-AUC CV parmi 5 candidats |
| Protocole de validation | RepeatedStratifiedKFold(5, 3) | 15 scores → IC robuste sur ~5 000 obs. |
| Optimisation | Optuna TPE, {_n_essais} essais | Cache SQLite, gain carbone documenté |
| Seuil de décision | τ* = {seuil_opt:.2f} | Argmax du gain net espéré (§12.5) |
| Régime opérationnel | Top-{_cap} comptes/mois | Contrainte capacité CS (3 × 15 gestes) |
| Anti-fuite | `COLONNES_INTERDITES` + Pipeline sklearn | `test_no_leakage.py` bloquant en CI |
| Déploiement | FastAPI + Docker + Prefect | Batch nocturne quotidien (revue du lundi, alertes) + API à l'ouverture d'une fiche client |
"""
    )
)

# %% [markdown]
# ### Principaux résultats

# %%
display(
    Markdown(
        f"""
**Métriques de classification (évaluation out-of-fold, non biaisée) :**

| Métrique | Valeur OOF | Référence |
|---|---|---|
| **PR-AUC** (métrique principale) | **{pr_auc_oof:.4f}** | Cible ≥ {config.CIBLES_PERFORMANCE["pr_auc_min"]:.2f} · Aléatoire ≈ {y.mean():.2f} |
| ROC-AUC | {roc_auc_oof:.4f} | Aléatoire = 0,50 |
| PR-AUC CV champion (Optuna) | {_pr_auc_optuna:.4f} | {'+' if (_pr_auc_optuna - _pr_b1) >= 0 else ''}{_pr_auc_optuna - _pr_b1:.4f} vs règle métier B1 |
| Seuil économique τ* | {seuil_opt:.2f} | Précision {_precision_seuil:.1%} · Rappel {_rappel_seuil:.1%} |

**KPI opérationnel (régime top-{_cap} comptes/mois) :**

| KPI | Valeur |
|---|---|
| Précision@{_cap} | **{precision_cap:.1%}** ({_n_churners_detectes} churners réels dans le top-{_cap}) |
| MRR churners couverts | {_mrr_couvert:,.0f} €/mois |
| Coût mensuel CS (gestes) | {_cout_mensuel_cs:,.0f} €/mois |
| **Valeur sauvée par la cohorte du mois** ({config.HYPOTHESES_ECONOMIQUES['horizon_mois']} mois) | **{_gain_annuel:,.0f} €** |
| ROI de la cohorte mensuelle ((valeur − coût) / coût) | **{_roi:.1f}×** |
"""
    )
)

# %%
display(
    Markdown(
        f"**Ce qu'il faut retenir.**  "
        f"Le modèle `{_nom_champion}` atteint une PR-AUC de **{pr_auc_oof:.4f}** en "
        f"évaluation out-of-fold, au-dessus du seuil de déploiement fixé a priori à "
        f"{config.CIBLES_PERFORMANCE['pr_auc_min']:.2f}.  "
        f"En régime opérationnel (top-{_cap} comptes/mois), il identifie "
        f"**{_n_churners_detectes} churners réels** sur {_cap} interventions "
        f"(précision : **{precision_cap:.1%}**).  "
        f"Les gestes d'un mois coûtent {_cout_mensuel_cs:,.0f} € et sauvent une marge espérée de "
        f"**{_gain_annuel:,.0f} €** sur {config.HYPOTHESES_ECONOMIQUES['horizon_mois']} mois "
        f"(ROI de la cohorte = {_roi:.1f}×), "
        f"sous les hypothèses économiques documentées en §12.4."
    )
)

# %% [markdown]
# ### Limites identifiées

# %%
display(
    Markdown(
        f"""
**Limites assumées — ce que la solution ne sait pas faire :**

| Limite | Nature | Impact |
|---|---|---|
| `valeur_vie_client_eur` historique | Cible proxy (valeur passée, §6.7) | CLV future inconnue — on utilise MRR × horizon à la place |
| Taux de succès rétention ({config.HYPOTHESES_ECONOMIQUES['taux_succes_retention']:.0%}) | Hypothèse externe, non mesurée empiriquement | Le ROI est conditionnel — sensibilité testée en §12.7 |
| Absence d'étiquettes en temps réel | Le churn s'observe à l'échéance contractuelle (1–12 mois) | Monitoring par PSI/KS et cohortes en attendant les labels |
| Commentaires CSM non exploités | Texte libre — NLP absent | Signal faible perdu ; piste d'amélioration §13.1 |
| CodeCarbon sous WSL2 | RAPL inaccessible → estimation TDP | Empreinte carbone indicative, non mesurée |
| Taille du dataset (~5 000 obs.) | Gains marginaux au-delà de 30 essais Optuna | Performances à surveiller si le portefeuille croît |
"""
    )
)

# %% [markdown]
# > ### 📋 Journal de bord — Résumé exécutif
# >
# > **Décisions retenues** — Section rédigée après §9 et §12 : tous les chiffres sont
# > chargés depuis les artefacts mis en cache (aucun chiffre recopié à la main).
# > Résumé structuré en 4 blocs (problème, approche, résultats, limites) pour couvrir
# > la totalité de la grille C1→C9 en une seule lecture.
# >
# > **Alternatives écartées** — Résumé textuel sans tableau : moins lisible pour un jury
# > qui scanne rapidement. Chiffres insérés à la main : invalidés à chaque régénération.
# >
# > **Difficultés rencontrées** — La section 01 précède §9 et §12 dans l'ordre du notebook,
# > mais les métriques finales ne sont disponibles qu'après ; résolu en chargeant les
# > artefacts mis en cache via `charger_ou_calculer` (lecture seule, sans recalcul).
# >
# > **Impact sur la suite** — Ce résumé est le point d'entrée du jury ; il doit être
# > relancé en dernier (après `make notebook`) pour afficher les valeurs définitives.
# >
# > **Temps passé** — ~1 h (rédaction + câblage des artefacts).
