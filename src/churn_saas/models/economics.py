"""Analyse économique de la décision de contact — justification obligatoire C4/C8.

Démarche : la décision n'est ni un seuil à 0,5 (arbitraire) ni l'argmax du F1 (métrique sans
valeur économique). Les gains sont mesurés **par rapport à « ne rien faire »**, depuis les
hypothèses documentées dans config.HYPOTHESES_ECONOMIQUES.

Deux régimes :

- **hors contrainte de capacité** : ``gain_par_seuil`` donne le seuil de rentabilité τ*,
  au-delà duquel tout geste rapporte plus qu'il ne coûte en espérance ;
- **sous contrainte de capacité** (régime réel, ``capacite_gestes_mois``) : on contacte les
  comptes de plus forte valeur attendue (``economie.selection_sous_capacite``), évalués par
  ``gain_sous_capacite``. C'est la règle de décision retenue (§12.6).

Point de vigilance n°3 : la valeur à risque est définie comme
    MRR_mensuel × horizon_mois × marge_brute
et NON valeur_vie_client_eur (CLV prospective qui intègre déjà le risque de départ, §6.7 —
double comptage).

Point de vigilance n°4 : les calculs en euros supposent des probabilités CALIBRÉES. Le
modèle retenu est pondéré (class_weight='balanced'), non rééchantillonné, et son intercept
est corrigé du décalage dû à la pondération (``models.calibration``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from matplotlib.figure import Figure

from churn_saas import config, economie, viz
from churn_saas.format_fr import entier, nombre, pourcentage

# Plage de taux de succès testée par les analyses de sensibilité (§12.7, §12.12)
PLAGE_TAUX_SUCCES: list[float] = [0.20, 0.25, 0.30, 0.35, 0.40]

# Durée d'application simulée de la règle de décision (§12.12) : une année d'exploitation
MOIS_PAR_AN: int = 12

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

    **FN** — churner non détecté → perte de MRR sur l'horizon (MRR × horizon × marge).
        Cette perte est subie **avec ou sans modèle** : elle n'entre pas dans le gain,
        mesuré par rapport à « ne rien faire ». La compter reviendrait à pénaliser deux fois
        le churner manqué et à fausser le seuil vers le bas.

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
        # FN : perte de marge subie sur l'horizon — information, hors du calcul de gain
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
) -> tuple[pd.DataFrame, float]:
    """Gain net du portefeuille, par rapport à « ne rien faire », pour chaque seuil.

    Régime **sans contrainte de capacité** : tout compte au-dessus du seuil est contacté.
    Calcul seul, sans figure : la figure est produite par :func:`tracer_gain_par_seuil`.

    Formule par compte i au seuil τ :
        gain_i(τ) =
            TP_i × (mrr_i × H × M × ts − C)   [marge sauvée en espérance, moins le geste]
          − FP_i × C                             [geste inutile]

    où H=horizon_mois, M=marge_brute, ts=taux_succes_retention, C=cout_intervention.
    FN et VN valent 0 : sans contact, la situation est celle de « ne rien faire ». Le gain
    vaut donc 0 à τ = 1 (personne n'est contacté).

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

    Retourne
    --------
    (courbe, seuil_optimal)
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

        gain_tp = (tp * (mrr_arr * mult_mrr * ts - cout_intervention)).sum()
        cout_fp = (fp * cout_intervention).sum()

        gains[i] = gain_tp - cout_fp
        n_alertes[i] = int(pred.sum())
        n_pred = pred.sum() or 1.0
        precisions[i] = tp.sum() / n_pred
        rappels[i] = tp.sum() / total_positifs

    seuil_optimal = float(seuils[int(np.argmax(gains))])

    courbe = pd.DataFrame(
        {
            "seuil": seuils,
            "gain_net_eur": gains,
            "n_alertes": n_alertes,
            "precision": precisions.round(4),
            "rappel": rappels.round(4),
        }
    )
    logger.debug(
        "Gain par seuil — τ* = {:.2f}, gain optimal = {:+.0f} €", seuil_optimal, gains.max()
    )
    return courbe, seuil_optimal


def tracer_gain_par_seuil(courbe: pd.DataFrame, seuil_optimal: float) -> Figure:
    """Trace le gain net espéré selon le seuil, à partir du résultat de :func:`gain_par_seuil`.

    Appelée une seule fois dans le notebook (§12.5) : une seule figure numérotée.
    """
    seuils = courbe["seuil"].to_numpy()
    gains = courbe["gain_net_eur"].to_numpy()
    gain_opt = float(gains[int(np.abs(seuils - seuil_optimal).argmin())])
    gain_a_05 = float(gains[int(np.abs(seuils - 0.5).argmin())])

    fig, ax = viz.figure(
        "gain_par_seuil",
        "Gain net du portefeuille selon le seuil, sans contrainte de capacité",
        taille=(9.0, 5.5),
    )

    ax.plot(seuils, gains / 1_000, color=viz.COULEUR_NON_CHURN, linewidth=2.5)
    ax.axvline(
        seuil_optimal,
        color=viz.COULEUR_CHURN,
        linewidth=1.8,
        linestyle="--",
        label=f"Seuil de rentabilité τ* = {nombre(seuil_optimal, 2)}",
    )
    ax.axvline(
        0.5,
        color="#888888",
        linewidth=1.2,
        linestyle=":",
        label=f"Seuil 0,50 (gain = {nombre(gain_a_05 / 1_000, 1, signe=True)} k€)",
    )
    ax.axhline(0, color="#444444", linewidth=0.8, linestyle="-")

    ax.scatter([seuil_optimal], [gain_opt / 1_000], color=viz.COULEUR_CHURN, s=80, zorder=5)
    # Étiquette décalée sous le point : au seuil optimal, la courbe forme souvent un plateau
    ax.annotate(
        f"τ* = {nombre(seuil_optimal, 2)}\n{nombre(gain_opt / 1_000, 1, signe=True)} k€",
        xy=(seuil_optimal, gain_opt / 1_000),
        xytext=(12, -30),
        textcoords="offset points",
        fontsize=9,
        color=viz.COULEUR_CHURN,
        va="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9, "edgecolor": "none"},
    )

    ax.set_xlabel("Seuil de décision τ")
    ax.set_ylabel("Gain net vs « ne rien faire » (k€)")
    ax.legend(loc="lower left")
    ax.set_xlim(-0.02, 1.02)

    viz.sauvegarder(fig)
    return fig


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
        f"{entier(cap_capture)} % des churners\ncaptés",
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
        f"Lift = {nombre(lift_a_capacite, 2)}×",
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


def gain_sous_capacite(
    y: pd.Series | np.ndarray,
    proba: np.ndarray,
    mrr: pd.Series | np.ndarray,
    *,
    capacite: int | None = None,
    classement: str = "valeur_attendue",
    taux_succes: float | None = None,
    cout: float | None = None,
) -> dict[str, float]:
    """Bilan d'un mois de gestes quand l'équipe CS ne peut contacter que ``capacite`` comptes.

    Paramètres
    ----------
    classement :
        ``"valeur_attendue"`` (règle retenue) : les comptes de plus forte valeur attendue
        positive (``economie.selection_sous_capacite``) ; ``"proba"`` : les ``capacite`` plus
        fortes probabilités de churn, pour comparaison ; ``"mrr"`` : les ``capacite`` plus gros
        comptes, sans modèle — ce que ferait une équipe CS qui priorise sur le seul MRR.
    taux_succes, cout :
        Remplacent les hypothèses de ``config`` (analyse de sensibilité). Le classement par
        valeur attendue utilise ces mêmes valeurs.

    Retourne
    --------
    dict : n_contactes, n_churners, precision, mrr_churners_couverts (€/mois),
    valeur_sauvee (€ sur l'horizon, en espérance), cout_gestes (€), gain_net (€), roi.
    Le gain est mesuré par rapport à « ne rien faire », avec la vérité terrain ``y``.
    """
    h = config.HYPOTHESES_ECONOMIQUES
    if capacite is None:
        capacite = int(h["capacite_gestes_mois"])
    cout_geste, mult_mrr, ts_defaut = _params_cout()
    ts = ts_defaut if taux_succes is None else float(taux_succes)
    cout_geste = cout_geste if cout is None else float(cout)

    y_arr = np.asarray(y, dtype=float)
    mrr_arr = np.asarray(mrr, dtype=float)
    proba_arr = np.asarray(proba, dtype=float)
    if classement == "valeur_attendue":
        valeurs = economie.valeur_attendue_intervention(proba_arr, mrr_arr, ts, cout_geste)
        idx = economie.selection_sous_capacite(np.asarray(valeurs), capacite)
    elif classement == "proba":
        idx = np.argsort(-proba_arr, kind="stable")[:capacite]
    elif classement == "mrr":
        idx = np.argsort(-mrr_arr, kind="stable")[:capacite]
    else:
        raise ValueError(f"Classement inconnu : {classement!r} (valeur_attendue | proba | mrr).")

    return _bilan_selection(idx, y_arr, mrr_arr, ts=ts, cout_geste=cout_geste, mult_mrr=mult_mrr)


def _bilan_selection(
    idx: np.ndarray,
    y_arr: np.ndarray,
    mrr_arr: np.ndarray,
    *,
    ts: float,
    cout_geste: float,
    mult_mrr: float,
    presence: float = 1.0,
) -> dict[str, float]:
    """Bilan des gestes sur les comptes ``idx``, mesuré par rapport à « ne rien faire ».

    ``presence`` : probabilité qu'un partant soit encore client au moment du geste ; un
    partant déjà parti ne peut plus être retenu, mais le geste a été préparé et coûte.
    """
    n_contactes = len(idx)
    churners = y_arr[idx]
    mrr_couverts = float((mrr_arr[idx] * churners).sum())
    valeur_sauvee = mrr_couverts * presence * mult_mrr * ts
    cout_gestes = n_contactes * cout_geste
    gain_net = valeur_sauvee - cout_gestes
    return {
        "n_contactes": float(n_contactes),
        "n_churners": float(churners.sum()),
        "precision": float(churners.mean()) if n_contactes else 0.0,
        "mrr_churners_couverts": mrr_couverts,
        "valeur_sauvee": valeur_sauvee,
        "cout_gestes": cout_gestes,
        "gain_net": gain_net,
        "roi": gain_net / cout_gestes if cout_gestes > 0 else 0.0,
    }


def gain_attribuable_au_modele(
    y: pd.Series | np.ndarray,
    proba: np.ndarray,
    mrr: pd.Series | np.ndarray,
    *,
    capacite: int | None = None,
    plage_taux_succes: list[float] | None = None,
) -> dict[str, Any]:
    """Gain de la règle retenue au-delà de ce que l'équipe CS obtiendrait sans modèle.

    ``gain_sous_capacite`` mesure le gain par rapport à « ne rien faire ». Or, sans modèle,
    une équipe CS ne reste pas inactive : elle contacte en priorité ses plus gros comptes. La
    part du gain due au modèle est donc l'écart entre la règle retenue (valeur attendue) et
    cette référence (classement ``"mrr"``), à budget de gestes identique : les coûts sont les
    mêmes, seul le choix des comptes diffère.

    Retourne
    --------
    dict : ``modele`` et ``reference`` (bilans de ``gain_sous_capacite`` aux hypothèses de
    ``config``), ``gain_attribuable`` (€, écart de gain net), ``gain_attribuable_min`` et
    ``gain_attribuable_max`` (€, sur ``plage_taux_succes``), ``taux_succes_min`` et
    ``taux_succes_max``.
    """
    if plage_taux_succes is None:
        plage_taux_succes = PLAGE_TAUX_SUCCES
    modele = gain_sous_capacite(y, proba, mrr, capacite=capacite)
    reference = gain_sous_capacite(y, proba, mrr, capacite=capacite, classement="mrr")
    ecarts = [
        gain_sous_capacite(y, proba, mrr, capacite=capacite, taux_succes=ts)["gain_net"]
        - gain_sous_capacite(y, proba, mrr, capacite=capacite, classement="mrr", taux_succes=ts)[
            "gain_net"
        ]
        for ts in plage_taux_succes
    ]
    gain_attribuable = modele["gain_net"] - reference["gain_net"]
    logger.debug(
        "Gain attribuable au modèle = {:.0f} € (plage {:.0f} – {:.0f} €)",
        gain_attribuable,
        min(ecarts),
        max(ecarts),
    )
    return {
        "modele": modele,
        "reference": reference,
        "gain_attribuable": gain_attribuable,
        "gain_attribuable_min": float(min(ecarts)),
        "gain_attribuable_max": float(max(ecarts)),
        "taux_succes_min": float(min(plage_taux_succes)),
        "taux_succes_max": float(max(plage_taux_succes)),
    }


def gain_sur_annee(
    y: pd.Series | np.ndarray,
    proba: np.ndarray,
    mrr: pd.Series | np.ndarray,
    *,
    capacite: int | None = None,
    n_mois: int = MOIS_PAR_AN,
    taux_succes: float | None = None,
    departs_etales: bool = False,
) -> pd.DataFrame:
    """Gain mois par mois quand la règle est appliquée chaque mois pendant ``n_mois``.

    ``gain_sous_capacite`` chiffre **un** mois de gestes : celui qui choisit les meilleurs
    comptes du portefeuille. Le mois suivant, ces comptes ont déjà été contactés ; la liste
    descend vers des comptes de moindre valeur attendue. Multiplier le premier mois par 12
    surestimerait donc l'année. Ici, chaque mois retient les ``capacite`` meilleurs comptes
    **non encore contactés**, pour la règle retenue comme pour la référence sans modèle (plus
    gros MRR non encore contactés).

    Hypothèses (à expliciter dans le texte) : portefeuille et scores figés sur l'année — pas de
    nouveaux comptes, pas de mise à jour des scores.

    ``departs_etales`` : par défaut, un partant contacté au mois m est supposé encore présent.
    Cela gonfle les gains bruts, et davantage ceux de la référence, qui atteint ses partants
    plus tard : le gain attribuable est alors une estimation prudente. Avec ``True``, les
    départs sont répartis uniformément sur l'année : un partant n'est encore présent au mois m
    qu'avec une probabilité 1 − (m − 1) / ``n_mois`` (scénario de sensibilité, sans recours à
    une date d'échéance que les données ne fournissent pas).

    Retourne
    --------
    DataFrame indexé par ``mois`` (1 à ``n_mois``) : n_churners_modele, n_churners_reference,
    gain_net_modele, gain_net_reference, gain_attribuable, gain_attribuable_cumule (€).
    """
    h = config.HYPOTHESES_ECONOMIQUES
    if capacite is None:
        capacite = int(h["capacite_gestes_mois"])
    cout_geste, mult_mrr, ts_defaut = _params_cout()
    ts = ts_defaut if taux_succes is None else float(taux_succes)

    y_arr = np.asarray(y, dtype=float)
    mrr_arr = np.asarray(mrr, dtype=float)
    valeurs = economie.valeur_attendue_intervention(
        np.asarray(proba, dtype=float), mrr_arr, ts, cout_geste
    )
    # Scores figés : retirer les comptes déjà contactés puis reprendre les meilleurs revient à
    # parcourir un classement unique, tranche de ``capacite`` par tranche
    ordre_modele = economie.selection_sous_capacite(np.asarray(valeurs), len(y_arr))
    ordre_reference = np.argsort(-np.nan_to_num(mrr_arr, nan=-np.inf), kind="stable")

    lignes = []
    for mois in range(1, n_mois + 1):
        tranche = slice((mois - 1) * capacite, mois * capacite)
        presence = 1.0 - (mois - 1) / n_mois if departs_etales else 1.0
        modele, reference = (
            _bilan_selection(
                ordre[tranche],
                y_arr,
                mrr_arr,
                ts=ts,
                cout_geste=cout_geste,
                mult_mrr=mult_mrr,
                presence=presence,
            )
            for ordre in (ordre_modele, ordre_reference)
        )
        lignes.append(
            {
                "mois": mois,
                "n_churners_modele": int(modele["n_churners"]),
                "n_churners_reference": int(reference["n_churners"]),
                "gain_net_modele": modele["gain_net"],
                "gain_net_reference": reference["gain_net"],
                "gain_attribuable": modele["gain_net"] - reference["gain_net"],
            }
        )
    tableau = pd.DataFrame(lignes).set_index("mois")
    tableau["gain_attribuable_cumule"] = tableau["gain_attribuable"].cumsum()
    logger.debug(
        "Gain attribuable sur {} mois = {:.0f} € (premier mois : {:.0f} €)",
        n_mois,
        float(tableau["gain_attribuable"].sum()),
        float(tableau["gain_attribuable"].iloc[0]),
    )
    return tableau


def gain_annuel_attribuable(
    y: pd.Series | np.ndarray,
    proba: np.ndarray,
    mrr: pd.Series | np.ndarray,
    *,
    capacite: int | None = None,
    plage_taux_succes: list[float] | None = None,
) -> dict[str, Any]:
    """Gain attribuable au modèle cumulé sur un an (``gain_sur_annee``), avec sa fourchette.

    Retourne
    --------
    dict : ``mensuel`` (tableau de ``gain_sur_annee`` aux hypothèses de ``config``),
    ``premier_mois`` et ``annuel`` (€), ``annuel_min`` et ``annuel_max`` (€, sur
    ``plage_taux_succes``), ``taux_succes_min`` et ``taux_succes_max`` ; scénario de
    sensibilité à départs étalés sur l'année : ``mensuel_departs_etales`` et
    ``annuel_departs_etales`` (€).
    """
    if plage_taux_succes is None:
        plage_taux_succes = PLAGE_TAUX_SUCCES
    mensuel = gain_sur_annee(y, proba, mrr, capacite=capacite)
    mensuel_etales = gain_sur_annee(y, proba, mrr, capacite=capacite, departs_etales=True)
    annuels = [
        float(
            gain_sur_annee(y, proba, mrr, capacite=capacite, taux_succes=ts)[
                "gain_attribuable"
            ].sum()
        )
        for ts in plage_taux_succes
    ]
    return {
        "mensuel": mensuel,
        "premier_mois": float(mensuel["gain_attribuable"].iloc[0]),
        "annuel": float(mensuel["gain_attribuable"].sum()),
        "annuel_min": min(annuels),
        "annuel_max": max(annuels),
        "taux_succes_min": float(min(plage_taux_succes)),
        "taux_succes_max": float(max(plage_taux_succes)),
        "mensuel_departs_etales": mensuel_etales,
        "annuel_departs_etales": float(mensuel_etales["gain_attribuable"].sum()),
    }


def bilan_projet(gain_annuel: dict[str, Any]) -> dict[str, Any]:
    """ROI annuel du projet de ML, coûts de développement et de maintenance compris.

    Le ROI de la cohorte (``gain_sous_capacite``) ne compte que le temps CSM. Ici, le gain
    retenu est le gain **attribuable** au modèle sur un an (``gain_annuel_attribuable``) : sans
    modèle, l'équipe appellerait quand même ses plus gros comptes, au même coût de gestes. On
    lui oppose le coût complet du projet (``config.COUTS_PROJET``) : développement amorti
    linéairement, infrastructure et maintenance.

    Le taux de succès d'équilibre exploite la linéarité du gain attribuable en taux de succès :
    les coûts de gestes, identiques dans les deux listes, s'annulent, et le classement par
    valeur attendue ne dépend pas du taux de succès.

    Retourne
    --------
    dict : ``cout_build``, ``cout_run_annuel`` (infrastructure + maintenance) et
    ``cout_annuel`` (amortissement + run), en € ; ``roi_projet`` ; ``mois_retour`` (premier
    mois où le gain attribuable cumulé couvre le build et le run engagés, ``None`` si jamais
    sur l'horizon simulé) ; ``taux_succes_equilibre`` (taux annulant le gain net du projet).
    """
    c = config.COUTS_PROJET
    jour = float(c["cout_journalier_eur"])
    jours_build = (
        float(c["jours_build_data_scientist"])
        + float(c["jours_build_dsi"])
        + float(c["jours_build_cs"])
    )
    cout_build = jours_build * jour
    run_mensuel = float(c["infra_mensuel_eur"]) + float(c["jours_maintenance_mois"]) * jour
    cout_run_annuel = run_mensuel * MOIS_PAR_AN
    cout_annuel = cout_build / float(c["duree_amortissement_ans"]) + cout_run_annuel

    annuel = float(gain_annuel["annuel"])
    cumul = gain_annuel["mensuel"]["gain_attribuable_cumule"]
    engage = cout_build + run_mensuel * cumul.index.to_numpy()
    couverts = cumul.index[cumul.to_numpy() >= engage]
    mois_retour = int(couverts[0]) if len(couverts) else None
    ts_ref = float(config.HYPOTHESES_ECONOMIQUES["taux_succes_retention"])
    taux_equilibre = ts_ref * cout_annuel / annuel if annuel > 0 else float("nan")
    logger.debug(
        "Projet : coût annuel {:.0f} €, ROI {:.1f}, retour au mois {}",
        cout_annuel,
        (annuel - cout_annuel) / cout_annuel,
        mois_retour,
    )
    return {
        "cout_build": cout_build,
        "cout_run_annuel": cout_run_annuel,
        "cout_annuel": cout_annuel,
        "roi_projet": (annuel - cout_annuel) / cout_annuel,
        "mois_retour": mois_retour,
        "taux_succes_equilibre": taux_equilibre,
    }


def tracer_gain_sur_annee(mensuel: pd.DataFrame) -> Figure:
    """Trace le gain net mensuel du modèle et de la référence, et le cumul attribuable.

    Appelée une seule fois dans le notebook (§12.12) : une seule figure numérotée.
    """
    fig, axes = viz.figure_grille(
        "gain_sur_annee",
        "Gain des gestes mois après mois, règle appliquée pendant un an",
        nlignes=1,
        ncols=2,
        taille=(14.0, 5.0),
    )
    mois = mensuel.index.to_numpy()
    ax_mois, ax_cumul = axes
    ax_mois.plot(
        mois,
        mensuel["gain_net_modele"] / 1_000,
        marker="o",
        color=viz.COULEUR_CHURN,
        label="Avec le modèle (valeur attendue)",
    )
    ax_mois.plot(
        mois,
        mensuel["gain_net_reference"] / 1_000,
        marker="s",
        color="#888888",
        label="Sans modèle (plus gros MRR)",
    )
    ax_mois.set_xlabel("Mois d'application de la règle")
    ax_mois.set_ylabel("Gain net du mois vs « ne rien faire » (k€)")
    ax_mois.set_title("Gain mensuel des gestes")
    ax_mois.legend(loc="upper right")

    ax_cumul.bar(
        mois,
        mensuel["gain_attribuable"] / 1_000,
        color=viz.COULEUR_NON_CHURN,
        label="Écart du mois",
    )
    ax_cumul.plot(
        mois,
        mensuel["gain_attribuable_cumule"] / 1_000,
        marker="o",
        color=viz.COULEUR_CHURN,
        label="Cumul",
    )
    ax_cumul.set_xlabel("Mois d'application de la règle")
    ax_cumul.set_ylabel("Gain attribuable au modèle (k€)")
    ax_cumul.set_title("Part due au modèle")
    ax_cumul.legend(loc="upper right")
    for ax in axes:
        ax.set_xticks(mois)
        ax.axhline(0, color="#444444", linewidth=0.8)

    viz.sauvegarder(fig)
    return fig


def concentration_du_gain(
    y: pd.Series | np.ndarray,
    proba: np.ndarray,
    mrr: pd.Series | np.ndarray,
    *,
    capacite: int | None = None,
    n_top: int = 5,
    quantile_mrr: float = 0.99,
) -> dict[str, float]:
    """Dépendance du gain du premier mois à quelques très gros comptes.

    Deux indicateurs :

    - ``part_top`` : part du gain net du mois (règle retenue) apportée par les ``n_top``
      comptes qui y contribuent le plus ;
    - ``gain_attribuable_sans_gros`` : gain attribuable au modèle recalculé après retrait des
      comptes dont le MRR dépasse le quantile ``quantile_mrr`` du portefeuille — pour la règle
      comme pour la référence. Un gain qui tient sans eux est un gain robuste.
    """
    h = config.HYPOTHESES_ECONOMIQUES
    if capacite is None:
        capacite = int(h["capacite_gestes_mois"])
    cout_geste, mult_mrr, ts = _params_cout()
    y_arr = np.asarray(y, dtype=float)
    mrr_arr = np.asarray(mrr, dtype=float)
    proba_arr = np.asarray(proba, dtype=float)

    valeurs = economie.valeur_attendue_intervention(proba_arr, mrr_arr, ts, cout_geste)
    idx = economie.selection_sous_capacite(np.asarray(valeurs), capacite)
    contributions = mrr_arr[idx] * y_arr[idx] * mult_mrr * ts - cout_geste
    gain_net = float(contributions.sum())
    part_top = float(np.sort(contributions)[::-1][:n_top].sum() / gain_net) if gain_net else 0.0

    seuil_mrr = float(np.nanquantile(mrr_arr, quantile_mrr))
    garde = ~(mrr_arr > seuil_mrr)
    sans_gros = gain_attribuable_au_modele(
        y_arr[garde], proba_arr[garde], mrr_arr[garde], capacite=capacite
    )
    return {
        "part_top": part_top,
        "n_top": float(n_top),
        "seuil_mrr": seuil_mrr,
        "n_exclus": float((~garde).sum()),
        "gain_attribuable_sans_gros": float(sans_gros["gain_attribuable"]),
    }


def sensibilite_capacite(
    y: pd.Series | np.ndarray,
    proba: np.ndarray,
    mrr: pd.Series | np.ndarray,
    plage_taux_succes: list[float] | None = None,
    plage_capacites: list[int] | None = None,
) -> tuple[Figure, pd.DataFrame]:
    """Sensibilité du gain de la règle retenue au taux de succès et à la capacité CS.

    Pour chaque couple (taux de succès, capacité), gain net du mois avec la priorisation par
    valeur attendue. Répond à deux questions du commanditaire : la recommandation tient-elle
    si le taux de succès est plus faible qu'espéré ? que rapporterait un CSM de plus ?

    Retourne
    --------
    (fig, tableau)
        tableau : DataFrame (taux_succes_retention, capacite, n_churners, gain_net_eur, roi).
    """
    h = config.HYPOTHESES_ECONOMIQUES
    cap_ref = int(h["capacite_gestes_mois"])
    ts_ref = float(h["taux_succes_retention"])
    if plage_taux_succes is None:
        plage_taux_succes = PLAGE_TAUX_SUCCES
    if plage_capacites is None:
        plage_capacites = sorted({15, 30, cap_ref, 60, 90, 120})

    lignes = []
    for ts in plage_taux_succes:
        for cap in plage_capacites:
            bilan = gain_sous_capacite(y, proba, mrr, capacite=cap, taux_succes=ts)
            lignes.append(
                {
                    "taux_succes_retention": float(ts),
                    "capacite": int(cap),
                    "n_churners": int(bilan["n_churners"]),
                    "gain_net_eur": round(bilan["gain_net"], 0),
                    "roi": round(bilan["roi"], 2),
                }
            )
    tableau = pd.DataFrame(lignes)

    fig, ax = viz.figure(
        "sensibilite_capacite",
        "Gain net du mois selon la capacité CS et le taux de succès de la rétention",
        taille=(9.0, 5.5),
    )
    # Rouge réservé à l'hypothèse retenue ; les autres taux prennent des couleurs non rouges
    autres_couleurs = [c for c in viz.PALETTE_PRINCIPALE if c != viz.COULEUR_CHURN]
    for k, ts in enumerate(plage_taux_succes):
        sous = tableau[tableau["taux_succes_retention"] == float(ts)]
        est_ref = abs(float(ts) - ts_ref) < 1e-9
        ax.plot(
            sous["capacite"],
            sous["gain_net_eur"] / 1_000,
            "o-",
            color=viz.COULEUR_CHURN if est_ref else autres_couleurs[k % len(autres_couleurs)],
            linewidth=2.5 if est_ref else 1.4,
            label=f"ts = {pourcentage(float(ts), 0)}" + (" (hypothèse retenue)" if est_ref else ""),
        )
    ax.axvline(cap_ref, color="#888888", linestyle=":", linewidth=1.2)
    ax.text(cap_ref, ax.get_ylim()[1], f" capacité actuelle ({cap_ref})", va="top", fontsize=9)
    ax.axhline(0, color="#444444", linewidth=0.8)
    ax.set_xlabel("Capacité CS (gestes de rétention par mois)")
    ax.set_ylabel("Gain net du mois vs « ne rien faire » (k€)")
    ax.legend(loc="lower right", fontsize=9)
    viz.sauvegarder(fig)
    return fig, tableau


def seuil_pour_recall(
    y: pd.Series | np.ndarray,
    proba: np.ndarray,
    recall_cible: float = config.RECALL_CIBLE_VIGILANCE,
) -> dict[str, float]:
    """Seuil de vigilance : le plus haut seuil qui détecte au moins ``recall_cible`` des churners.

    Premier niveau de la règle de décision. Un compte est signalé si ``proba >= seuil`` ; le
    seuil retenu est le plus exigeant qui tient la cible, donc celui qui signale le moins de
    comptes. À calculer sur les prédictions out-of-fold, jamais sur le jeu de test.

    Retourne
    --------
    dict : seuil, recall, precision, part_signalee (part du portefeuille signalée),
    n_signales, n_faux_negatifs (churners sous le seuil).
    """
    y_arr = np.asarray(y, dtype=float)
    proba_arr = np.asarray(proba, dtype=float)
    n_churners = float(y_arr.sum())
    if n_churners == 0:
        raise ValueError("Aucun churner dans y : le recall n'est pas défini.")

    ordre = np.argsort(-proba_arr, kind="stable")
    recall_cumule = np.cumsum(y_arr[ordre]) / n_churners
    rang = int(np.searchsorted(recall_cumule, recall_cible - 1e-12))
    seuil = float(proba_arr[ordre][min(rang, len(ordre) - 1)])

    # Les ex æquo au seuil sont signalés aussi : le recall effectif peut dépasser la cible
    signales = proba_arr >= seuil
    n_signales = int(signales.sum())
    vrais_positifs = float(y_arr[signales].sum())
    resultat = {
        "seuil": seuil,
        "recall": vrais_positifs / n_churners,
        "precision": vrais_positifs / n_signales,
        "part_signalee": n_signales / len(y_arr),
        "n_signales": float(n_signales),
        "n_faux_negatifs": float(n_churners - vrais_positifs),
    }
    logger.debug(
        "Seuil de vigilance = {:.4f} — recall {:.3f}, precision {:.3f}, {} comptes signalés",
        seuil,
        resultat["recall"],
        resultat["precision"],
        n_signales,
    )
    return resultat


def table_deux_niveaux(
    y: pd.Series | np.ndarray,
    proba: np.ndarray,
    mrr: pd.Series | np.ndarray,
    capacite: int | None = None,
    recall_cible: float = config.RECALL_CIBLE_VIGILANCE,
) -> pd.DataFrame:
    """Bilan de la règle de décision à deux niveaux, en entonnoir.

    - **Niveau 1 — vigilance** : comptes au-dessus du seuil de ``seuil_pour_recall`` ; on
      surveille large pour manquer peu de churners.
    - **Niveau 2 — appels** : parmi les comptes signalés, les ``capacite`` de plus forte valeur
      attendue positive (``gain_sous_capacite``), seuls à recevoir un geste CSM.

    Les faux négatifs résiduels sont les churners hors du niveau : leur valeur est la perte
    sèche ``MRR × horizon × marge`` (point de vigilance n°3), un MRR absent comptant pour zéro.

    Retourne
    --------
    DataFrame, une ligne par niveau : niveau, seuil, n_comptes, part_portefeuille, n_churners,
    recall, precision, n_faux_negatifs, valeur_faux_negatifs_eur, gain_net_eur (niveau 2
    seulement : la vigilance seule ne déclenche pas de geste CSM).
    """
    if capacite is None:
        capacite = int(config.HYPOTHESES_ECONOMIQUES["capacite_gestes_mois"])
    _, mult_mrr, _ = _params_cout()
    y_arr = np.asarray(y, dtype=float)
    proba_arr = np.asarray(proba, dtype=float)
    mrr_arr = np.asarray(mrr, dtype=float)
    n_churners = float(y_arr.sum())

    vigilance = seuil_pour_recall(y_arr, proba_arr, recall_cible)
    idx_signales = np.flatnonzero(proba_arr >= vigilance["seuil"])
    bilan = gain_sous_capacite(
        y_arr[idx_signales], proba_arr[idx_signales], mrr_arr[idx_signales], capacite=capacite
    )
    valeurs = economie.valeur_attendue_intervention(proba_arr[idx_signales], mrr_arr[idx_signales])
    idx_appels = idx_signales[economie.selection_sous_capacite(np.asarray(valeurs), capacite)]

    def _ligne(niveau: str, idx: np.ndarray, gain_net: float) -> dict[str, Any]:
        retenus = np.zeros(len(y_arr), dtype=bool)
        retenus[idx] = True
        manques = (y_arr == 1) & ~retenus
        detectes = float(y_arr[retenus].sum())
        return {
            "niveau": niveau,
            "seuil": vigilance["seuil"],
            "n_comptes": len(idx),
            "part_portefeuille": len(idx) / len(y_arr),
            "n_churners": int(detectes),
            "recall": detectes / n_churners,
            "precision": detectes / len(idx) if len(idx) else 0.0,
            "n_faux_negatifs": int(manques.sum()),
            "valeur_faux_negatifs_eur": float(np.nansum(mrr_arr[manques])) * mult_mrr,
            "gain_net_eur": gain_net,
        }

    table = pd.DataFrame(
        [
            _ligne("1 — vigilance (seuil de recall)", idx_signales, float("nan")),
            _ligne(f"2 — appels (capacité {capacite})", idx_appels, bilan["gain_net"]),
        ]
    )
    logger.debug(
        "Deux niveaux — {} comptes surveillés, {} appelés, {} churners manqués au niveau 2",
        len(idx_signales),
        len(idx_appels),
        int(table.loc[1, "n_faux_negatifs"]),
    )
    return table


def table_de_decision(capacite: int, seuil_rentabilite: float) -> pd.DataFrame:
    """Table de décision opérationnelle pour l'équipe Customer Success (C8).

    Trois zones, dérivées de la règle de priorisation sous capacité (§12.6) :

    - **Contacter** : les ``capacite`` comptes de plus forte valeur attendue ;
    - **Action automatisée** : valeur attendue positive mais hors capacité — un geste humain
      serait rentable, l'équipe ne peut pas l'assurer ; on déclenche une action à coût quasi
      nul (parcours d'adoption, e-mail, message in-app) ;
    - **Veille** : valeur attendue négative ou nulle — un geste coûterait plus qu'il ne
      rapporte en espérance.

    ``seuil_rentabilite`` (τ*, §12.5) n'est rappelé qu'à titre indicatif : c'est la
    probabilité au-delà de laquelle un compte au MRR type devient rentable à contacter.
    """
    lignes = [
        {
            "zone": f"Priorité (rang ≤ {capacite} par valeur attendue)",
            "action": "Contacter",
            "description": (
                "Geste de rétention par le CSM : appel, revue d'usage, plan d'action. "
                "Escalade à la direction commerciale pour les plus fortes valeurs à risque."
            ),
            "responsable": "CSM assigné · CS Lead pour l'escalade",
            "periodicite": "Mensuelle (liste du batch)",
        },
        {
            "zone": f"Rentable hors capacité (valeur attendue > 0, rang > {capacite})",
            "action": "Action automatisée",
            "description": (
                "Parcours d'adoption, e-mail ciblé ou message in-app, sans temps CSM. "
                "Le compte remonte dans la liste si sa valeur attendue augmente."
            ),
            "responsable": "Système (batch) · Marketing produit",
            "periodicite": "Mensuelle",
        },
        {
            "zone": "Non rentable (valeur attendue ≤ 0)",
            "action": "Veille",
            "description": (
                "Aucun geste : son coût dépasse le gain espéré. Score recalculé à chaque "
                f"batch (repère : τ* = {nombre(seuil_rentabilite, 2)} pour un compte au MRR type)."
            ),
            "responsable": "Système (batch)",
            "periodicite": "À chaque batch",
        },
    ]
    return pd.DataFrame(lignes)
