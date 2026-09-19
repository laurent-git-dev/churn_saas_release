"""Analyse économique du seuil de décision — justification obligatoire C4/C8.

Démarche : le seuil n'est ni 0,5 (arbitraire) ni l'argmax du F1 (optimise une
métrique sans valeur économique). Il est dérivé du gain net espéré par compte,
calculé depuis les hypothèses documentées dans config.HYPOTHESES_ECONOMIQUES.

Point de vigilance n°3 : la valeur à risque est définie comme
    MRR_mensuel × horizon_mois × marge_brute
et NON valeur_vie_client_eur (qui contient la valeur passée — double comptage).

Point de vigilance n°4 : le seuil économique est calculé sur un modèle
CALIBRÉ et NON RÉÉCHANTILLONNÉ. SMOTE décalibre les probabilités ; le modèle
retenu utilise class_weight='balanced'.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from matplotlib.figure import Figure

from churn_saas import config, viz

# ---------------------------------------------------------------------------
# Helpers internes
# ---------------------------------------------------------------------------


def _params_cout() -> tuple[float, float, float]:
    """Extrait (cout_intervention, mult_mrr, taux_succes) depuis config."""
    h = config.HYPOTHESES_ECONOMIQUES
    cout_intervention = float(h["cout_horaire_csm_eur"]) * float(h["duree_geste_retention_h"])
    mult_mrr = float(h["horizon_mois"]) * float(h["marge_brute_pct"])
    ts = float(h["taux_succes_retention"])
    return cout_intervention, mult_mrr, ts


# ---------------------------------------------------------------------------
# API publique
# ---------------------------------------------------------------------------


def matrice_couts() -> dict[str, Any]:
    """Construit la matrice des coûts depuis config.HYPOTHESES_ECONOMIQUES.

    Quatre issues possibles, chacune documentée avec son hypothèse :

    **TP** — churner détecté, intervention → sauvegarde probable.
        Gain net = MRR × horizon × marge × taux_succès − coût_intervention.
        Peut être négatif pour les très petits comptes : le modèle est rentable
        seulement si MRR > ``seuil_mrr_rentable_eur``.

    **FP** — fidèle traité comme churner → intervention inutile.
        Coût = coût_intervention (temps CSM + geste commercial éventuel).
        Hypothèse : pas de geste commercial supplémentaire pour simplifier
        (la sensibilité à ce coût est testée dans sensibilite_seuil).

    **FN** — churner non détecté → perte de MRR sur l'horizon.
        Perte = MRR × horizon × marge.
        C'est le coût dominant sur les gros comptes.

    **TN** — fidèle non alerté → aucune action, coût = 0.

    Note : FN et TP dépendent du MRR individuel du compte. Passer mrr comme
    vecteur à gain_par_seuil() pour un calcul compte par compte.
    """
    h = config.HYPOTHESES_ECONOMIQUES
    cout_intervention, mult_mrr, ts = _params_cout()
    seuil_rentabilite = cout_intervention / (mult_mrr * ts) if (mult_mrr * ts) > 0 else float("inf")

    matrice: dict[str, Any] = {
        # Coût fixe à chaque prédiction positive (TP ou FP)
        "cout_intervention_eur": cout_intervention,
        # FP : uniquement le temps CSM gaspillé
        "cout_fp_eur": cout_intervention,
        # FN : multiplicateur du MRR → perte de marge future sur l'horizon
        #   application : perte_fn_i = mrr_i × mult_mrr_fn
        "mult_mrr_fn": mult_mrr,
        # TP : gain brut = marge récupérée × taux de succès de la rétention
        #   application : gain_brut_tp_i = mrr_i × mult_mrr_tp − cout_intervention
        "mult_mrr_tp": mult_mrr * ts,
        # MRR minimum pour que l'intervention sur un TP soit rentable
        #   si mrr_i < seuil_mrr_rentable, le gain net TP est négatif
        "seuil_mrr_rentable_eur": round(seuil_rentabilite, 2),
        "cout_tn_eur": 0.0,
        "hypotheses": dict(h),
    }

    logger.info(
        "Matrice des coûts — intervention={:.0f} €, mult_mrr_fn={:.2f}, "
        "mult_mrr_tp={:.2f}, seuil_rentabilité_mrr={:.0f} €",
        cout_intervention,
        mult_mrr,
        mult_mrr * ts,
        seuil_rentabilite,
    )
    return matrice


def gain_par_seuil(
    y: pd.Series | np.ndarray,
    proba: np.ndarray,
    mrr: pd.Series | np.ndarray,
    *,
    n_seuils: int = 200,
    taux_succes_override: float | None = None,
) -> tuple[Figure, pd.DataFrame, float]:
    """Calcule le gain net espéré du portefeuille pour chaque seuil de décision.

    Formule par compte i au seuil τ :
        gain_i(τ) =
            TP_i × (mrr_i × H × M × ts − C)   [gain si sauvegarde réussie]
          − FP_i × C                             [coût intervention inutile]
          − FN_i × (mrr_i × H × M)              [marge perdue si non détecté]

    où H=horizon_mois, M=marge_brute, ts=taux_succes_retention, C=cout_intervention.

    Paramètres
    ----------
    y :
        Vérité terrain (0/1).
    proba :
        Probabilités prédites — doit venir d'un modèle calibré non rééchantillonné.
    mrr :
        MRR mensuel par compte (même longueur que y), en euros.
    n_seuils :
        Nombre de seuils évalués dans [0, 1].
    taux_succes_override :
        Si renseigné, remplace config.HYPOTHESES_ECONOMIQUES["taux_succes_retention"].
        Utilisé par sensibilite_seuil().

    Retourne
    --------
    (fig, courbe, seuil_optimal)
        fig : figure matplotlib (gain net €  en fonction du seuil).
        courbe : DataFrame (seuil, gain_net_eur, n_alertes, precision, rappel).
        seuil_optimal : float — argmax du gain net.
    """
    y_arr = np.asarray(y, dtype=float)
    mrr_arr = np.asarray(mrr, dtype=float)
    seuils = np.linspace(0.0, 1.0, n_seuils)

    cout_intervention, mult_mrr, ts = _params_cout()
    if taux_succes_override is not None:
        ts = float(taux_succes_override)

    gains = np.empty(n_seuils)
    n_alertes = np.empty(n_seuils, dtype=int)
    precisions = np.empty(n_seuils)
    rappels = np.empty(n_seuils)

    total_positifs = float(y_arr.sum()) or 1.0  # éviter la division par zéro

    for i, seuil in enumerate(seuils):
        pred = (proba >= seuil).astype(float)
        tp = y_arr * pred
        fp = (1.0 - y_arr) * pred
        fn = y_arr * (1.0 - pred)

        gain_tp = (tp * (mrr_arr * mult_mrr * ts - cout_intervention)).sum()
        cout_fp = (fp * cout_intervention).sum()
        perte_fn = (fn * mrr_arr * mult_mrr).sum()

        gains[i] = gain_tp - cout_fp - perte_fn
        n_alertes[i] = int(pred.sum())
        n_pred = pred.sum() or 1.0
        precisions[i] = tp.sum() / n_pred
        rappels[i] = tp.sum() / total_positifs

    idx_opt = int(np.argmax(gains))
    seuil_optimal = float(seuils[idx_opt])
    gain_opt = float(gains[idx_opt])
    gain_a_05 = float(gains[int(np.abs(seuils - 0.5).argmin())])

    courbe = pd.DataFrame(
        {
            "seuil": seuils,
            "gain_net_eur": gains,
            "n_alertes": n_alertes,
            "precision": precisions.round(4),
            "rappel": rappels.round(4),
        }
    )

    # ── Figure ──────────────────────────────────────────────────────────────
    fig, ax = viz.figure(
        "gain_par_seuil",
        "Gain net espéré du portefeuille selon le seuil de décision",
        taille=(9.0, 5.5),
    )

    ax.plot(seuils, gains / 1_000, color=viz.COULEUR_NON_CHURN, linewidth=2.5)
    ax.axvline(
        seuil_optimal,
        color=viz.COULEUR_CHURN,
        linewidth=1.8,
        linestyle="--",
        label=f"Seuil optimal τ* = {seuil_optimal:.2f}",
    )
    ax.axvline(
        0.5,
        color="#888888",
        linewidth=1.2,
        linestyle=":",
        label=f"Seuil 0,50 (gain = {gain_a_05 / 1_000:+.1f} k€)",
    )
    ax.axhline(0, color="#444444", linewidth=0.8, linestyle="-")

    ax.scatter([seuil_optimal], [gain_opt / 1_000], color=viz.COULEUR_CHURN, s=80, zorder=5)
    ax.text(
        seuil_optimal + 0.02,
        gain_opt / 1_000,
        f"τ* = {seuil_optimal:.2f}\n{gain_opt / 1_000:+.1f} k€",
        fontsize=9,
        color=viz.COULEUR_CHURN,
        va="center",
    )

    ax.set_xlabel("Seuil de décision τ")
    ax.set_ylabel("Gain net espéré (k€)")
    ax.legend(loc="lower left")
    ax.set_xlim(-0.02, 1.02)

    viz.sauvegarder(fig)
    logger.info(
        "Gain par seuil — τ* = {:.2f}, gain optimal = {:+.0f} €, gain à 0.5 = {:+.0f} €",
        seuil_optimal,
        gain_opt,
        gain_a_05,
    )
    return fig, courbe, seuil_optimal


def courbe_lift(
    y: pd.Series | np.ndarray,
    proba: np.ndarray,
) -> tuple[Figure, pd.DataFrame]:
    """Courbe de gain cumulé et ratio de lift — version sous contrainte de capacité.

    Lorsque l'équipe CS ne peut traiter que N comptes par mois (config.HYPOTHESES_ECONOMIQUES
    « capacite_gestes_mois »), on passe d'un seuil à un classement top-N.

    Retourne
    --------
    (fig, courbe)
        courbe : DataFrame (pct_contactes, pct_churners_captures, lift, n_comptes).
    """
    y_arr = np.asarray(y, dtype=float)
    n = len(y_arr)
    total_churners = float(y_arr.sum()) or 1.0
    capacite = int(config.HYPOTHESES_ECONOMIQUES["capacite_gestes_mois"])

    idx_tri = np.argsort(-proba)
    y_trie = y_arr[idx_tri]

    # Calcul cumulatif compte par compte
    cumul_churners = np.cumsum(y_trie)
    ks = np.arange(1, n + 1)
    pct_contactes = ks / n * 100.0
    pct_captures = cumul_churners / total_churners * 100.0
    lift = pct_captures / (pct_contactes + 1e-9) * 100.0 / 100.0  # lift = captures / aléatoire

    courbe = pd.DataFrame(
        {
            "n_comptes": ks,
            "pct_contactes": pct_contactes.round(2),
            "pct_churners_captures": pct_captures.round(2),
            "lift": lift.round(3),
        }
    )

    # Lift à la capacité de l'équipe CS
    k_cap = min(capacite, n)
    lift_a_capacite = float(lift[k_cap - 1])
    precision_cap = float(cumul_churners[k_cap - 1] / k_cap)

    fig, axes = viz.figure_grille(
        "courbe_lift",
        "Courbe de gain cumulé et lift — contrainte de capacité CS",
        nlignes=1,
        ncols=2,
        taille=(13.0, 5.5),
    )
    ax_gain, ax_lift = axes[0], axes[1]

    # Panel gauche : gain cumulé
    ax_gain.plot(
        pct_contactes, pct_captures, color=viz.COULEUR_CHURN, linewidth=2.0, label="Modèle"
    )
    ax_gain.plot([0, 100], [0, 100], "--", color="#888888", linewidth=1.2, label="Aléatoire")
    # Contrainte capacité
    cap_pct = k_cap / n * 100.0
    cap_capture = float(cumul_churners[k_cap - 1] / total_churners * 100.0)
    ax_gain.axvline(
        cap_pct,
        color=viz.COULEUR_NON_CHURN,
        linewidth=1.5,
        linestyle="--",
        label=f"Capacité CS ({capacite} comptes)",
    )
    ax_gain.scatter([cap_pct], [cap_capture], color=viz.COULEUR_NON_CHURN, s=70, zorder=5)
    ax_gain.text(
        cap_pct + 1.5,
        cap_capture - 5,
        f"{cap_capture:.0f} % des churners\ncaptés",
        fontsize=9,
        color=viz.COULEUR_NON_CHURN,
    )
    ax_gain.set_xlabel("% comptes contactés")
    ax_gain.set_ylabel("% churners capturés")
    ax_gain.legend(loc="lower right", fontsize=9)
    ax_gain.set_xlim(0, 100)
    ax_gain.set_ylim(0, 105)

    # Panel droit : ratio de lift
    ax_lift.plot(pct_contactes, lift, color=viz.COULEUR_CHURN, linewidth=2.0)
    ax_lift.axhline(1.0, color="#888888", linewidth=1.2, linestyle="--", label="Aléatoire")
    ax_lift.axvline(
        cap_pct, color=viz.COULEUR_NON_CHURN, linewidth=1.5, linestyle="--", label="Capacité CS"
    )
    ax_lift.scatter([cap_pct], [lift_a_capacite], color=viz.COULEUR_NON_CHURN, s=70, zorder=5)
    ax_lift.text(
        cap_pct + 1.5,
        lift_a_capacite + 0.05,
        f"Lift = {lift_a_capacite:.2f}×",
        fontsize=9,
        color=viz.COULEUR_NON_CHURN,
    )
    ax_lift.set_xlabel("% comptes contactés")
    ax_lift.set_ylabel("Ratio de lift (modèle / aléatoire)")
    ax_lift.legend(loc="upper right", fontsize=9)
    ax_lift.set_xlim(0, 100)
    ax_lift.set_ylim(0, None)

    viz.sauvegarder(fig)
    logger.info(
        "Courbe de lift — capacité CS = {} comptes ({:.1f} %) : "
        "{:.0f} % churners capturés, lift = {:.2f}, précision = {:.1f} %",
        k_cap,
        cap_pct,
        cap_capture,
        lift_a_capacite,
        precision_cap * 100,
    )
    return fig, courbe


def precision_at_k(
    y: pd.Series | np.ndarray,
    proba: np.ndarray,
    k: int,
) -> float:
    """Précision dans le top-k comptes les plus à risque.

    Utile sous contrainte de capacité : si l'équipe CS ne peut contacter
    que k comptes par mois, quelle fraction sont de vrais churners ?

    Paramètres
    ----------
    k :
        Nombre de comptes à contacter (classement par score décroissant).

    Retourne
    --------
    Précision@k en [0, 1].
    """
    if k <= 0:
        return 0.0
    y_arr = np.asarray(y, dtype=float)
    k = min(k, len(y_arr))
    idx_top = np.argsort(-proba)[:k]
    p_at_k = float(y_arr[idx_top].mean())
    logger.debug("Précision@{} = {:.4f}", k, p_at_k)
    return p_at_k


def sensibilite_seuil(
    y: pd.Series | np.ndarray,
    proba: np.ndarray,
    mrr: pd.Series | np.ndarray,
    plage_taux_succes: list[float] | np.ndarray | None = None,
    *,
    n_seuils: int = 200,
) -> tuple[Figure, pd.DataFrame]:
    """Analyse la sensibilité du seuil optimal au taux de succès de la rétention.

    Objectif : montrer que le seuil recommandé est robuste et répondre à la
    question « d'où sortent vos chiffres ? ». Si le seuil se déplace peu
    sur la plage [20 %, 40 %], la recommandation est stable.

    Comportement attendu : plus le taux de succès est élevé, plus l'intervention
    est rentable → on abaisse le seuil (on intervient sur plus de comptes).

    Paramètres
    ----------
    plage_taux_succes :
        Liste de taux à tester. Défaut : 0.20, 0.25, 0.30, 0.35, 0.40.

    Retourne
    --------
    (fig, tableau)
        tableau : DataFrame (taux_succes, seuil_optimal, gain_max_eur, n_alertes_opt).
    """
    if plage_taux_succes is None:
        plage_taux_succes = [0.20, 0.25, 0.30, 0.35, 0.40]

    lignes = []
    for ts in plage_taux_succes:
        _, courbe, seuil_opt = gain_par_seuil(
            y, proba, mrr, n_seuils=n_seuils, taux_succes_override=float(ts)
        )
        idx_opt = courbe["gain_net_eur"].idxmax()
        lignes.append(
            {
                "taux_succes_retention": round(float(ts), 3),
                "seuil_optimal": round(seuil_opt, 4),
                "gain_max_eur": round(float(courbe.loc[idx_opt, "gain_net_eur"]), 0),
                "n_alertes_opt": int(courbe.loc[idx_opt, "n_alertes"]),
                "rappel_opt": round(float(courbe.loc[idx_opt, "rappel"]), 4),
            }
        )

    tableau = pd.DataFrame(lignes)

    fig, ax = viz.figure(
        "sensibilite_seuil",
        "Sensibilité du seuil optimal au taux de succès de la rétention",
        taille=(8.0, 5.0),
    )

    seuils_opt = tableau["seuil_optimal"].values
    taux = tableau["taux_succes_retention"].values * 100.0  # en %

    ax.plot(taux, seuils_opt, "o-", color=viz.COULEUR_CHURN, linewidth=2.0, markersize=8)
    ax.axhline(0.5, color="#888888", linewidth=1.0, linestyle=":", label="Seuil 0,50 (référence)")

    # Valeur de référence du config
    ts_ref = float(config.HYPOTHESES_ECONOMIQUES["taux_succes_retention"])
    _, _, seuil_ref = gain_par_seuil(y, proba, mrr, n_seuils=n_seuils)
    ax.scatter(
        [ts_ref * 100],
        [seuil_ref],
        color=viz.COULEUR_NON_CHURN,
        s=100,
        zorder=6,
        label=f"Hypothèse retenue (ts = {ts_ref:.0%})",
    )

    ax.set_xlabel("Taux de succès de la rétention (%)")
    ax.set_ylabel("Seuil optimal τ*")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper right")

    # Annotation de la monotonie
    delta_seuil = float(seuils_opt[0] - seuils_opt[-1])
    sens = "↓ seuil quand ts ↑" if delta_seuil > 0 else "↑ seuil quand ts ↑"
    ax.text(
        0.05,
        0.10,
        f"Variation : {delta_seuil:+.2f} ({sens})",
        transform=ax.transAxes,
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
    )

    viz.sauvegarder(fig)
    logger.info(
        "Sensibilité du seuil — plage taux_succes [{:.0%}, {:.0%}] → "
        "seuil [{:.2f}, {:.2f}], variation {:.2f}",
        float(min(plage_taux_succes)),
        float(max(plage_taux_succes)),
        float(seuils_opt.min()),
        float(seuils_opt.max()),
        delta_seuil,
    )
    return fig, tableau


def table_de_decision(
    seuil_contact: float = 0.40,
    seuil_escalade: float = 0.65,
) -> pd.DataFrame:
    """Table de décision opérationnelle pour l'équipe Customer Success.

    Traduit le score de risque en action concrète, évitant les décisions
    ad hoc. Sert aussi à C8 (actions déclenchées en fonction des indicateurs).

    Paramètres
    ----------
    seuil_contact :
        Seuil à partir duquel un contact proactif est déclenché.
        Recommandé : issu de l'analyse gain_par_seuil() sur les données de validation.
    seuil_escalade :
        Seuil à partir duquel une escalade prioritaire est déclenchée.
        Recommandé : seuil à haute précision (peu de FP tolerés).

    Retourne
    --------
    DataFrame avec colonnes : zone, score_min, score_max, action,
    description, responsable, periodicite.
    """
    lignes = [
        {
            "zone": f"Faible risque (score < {seuil_contact:.2f})",
            "score_min": 0.0,
            "score_max": seuil_contact,
            "action": "Surveiller",
            "description": (
                "Revue trimestrielle automatique. Aucun geste proactif. "
                "Indicateur de dérive suivi par le batch hebdomadaire."
            ),
            "responsable": "Système (batch hebdomadaire)",
            "periodicite": "Trimestrielle",
        },
        {
            "zone": f"Risque modéré ({seuil_contact:.2f} ≤ score < {seuil_escalade:.2f})",
            "score_min": seuil_contact,
            "score_max": seuil_escalade,
            "action": "Contacter",
            "description": (
                "Appel de prévention, revue d'usage, proposition de formation "
                "ou de session d'onboarding complémentaire."
            ),
            "responsable": "CSM assigné au compte",
            "periodicite": "Mensuelle",
        },
        {
            "zone": f"Risque élevé (score ≥ {seuil_escalade:.2f})",
            "score_min": seuil_escalade,
            "score_max": 1.0,
            "action": "Escalader",
            "description": (
                "Intervention prioritaire dans les 48 h. Geste commercial si MRR "
                f"> {matrice_couts()['seuil_mrr_rentable_eur']:.0f} €/mois. "
                "Réunion CS Lead + Direction commerciale."
            ),
            "responsable": "CS Lead + Direction commerciale",
            "periodicite": "Immédiate (< 48 h)",
        },
    ]

    df = pd.DataFrame(lignes)
    logger.info(
        "Table de décision — seuils : surveiller < {:.2f} ≤ contacter < {:.2f} ≤ escalader",
        seuil_contact,
        seuil_escalade,
    )
    return df
