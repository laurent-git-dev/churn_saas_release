"""Simulation des métriques de monitoring du tableau de bord Grafana.

Sans stack Prometheus en marche, on génère deux heures de métriques synthétiques (pas de 30 s,
le ``scrape_interval`` de ``prometheus.yml``) pour les six panels de données de
``monitoring/grafana/dashboard.json``, dans leurs unités : statut, débit, latence, taux
d'erreur, prédictions par décision et champs absents des demandes.

Trois incidents sont injectés, un par niveau de surveillance :

- une panne de l'API puis un pic de latence (règles ``APIIndisponible`` et ``LatenceElevee``) ;
- une dérive de sortie : la part ``ALERTE_ROUGE`` part du dernier batch réel et gagne
  ``DERIVE_SORTIE_PTS`` points, au-delà de la tolérance du runbook (§4.3) ;
- une rupture d'intégration CRM : ``secteur`` cesse d'être transmis.

Usage autonome : ``uv run python monitoring/simulation_grafana.py`` (figure dans
``reports/figures/``). Le notebook (§13.9) appelle les deux fonctions publiques.
"""

from __future__ import annotations

import json

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib.figure import Figure

from churn_saas import config, viz
from churn_saas.format_fr import nombre

PAS_S = 30
NB_PAS = 240  # 240 × 30 s = 2 heures
DEBIT_REQ_S = 12
PANNE = slice(100, 122)
PIC_LATENCE = slice(80, 100)
DERIVE_SORTIE_PTS = 12
IDX_RUPTURE_CRM = 150


def _repartition_dernier_batch() -> tuple[float, float]:
    """Parts ``ALERTE_ROUGE`` et ``SURVEILLANCE`` du dernier batch de scoring publié (§10)."""
    synthese = json.loads(
        sorted(config.TABLES.glob("scores_batch_*_synthese.json"))[-1].read_text(encoding="utf-8")
    )
    n = synthese["n_comptes"]
    return synthese["nb_alertes_rouges"] / n, synthese["nb_surveillances"] / n


def simuler_metriques() -> pd.DataFrame:
    """Génère les séries synthétiques des six panels du tableau de bord.

    Returns
    -------
    pd.DataFrame
        Une ligne par pas de 30 s. ``attrs`` porte les paramètres des incidents
        (``duree_panne_min``, ``taux_alerte_ref``, ``taux_alerte_fin``, ``tolerance_pts``).
    """
    rng = np.random.default_rng(config.RANDOM_SEED)
    ts = pd.date_range("2026-09-27 00:00:00", periods=NB_PAS, freq=f"{PAS_S}s")

    up = np.ones(NB_PAS, dtype=int)
    up[PANNE] = 0
    req = rng.poisson(DEBIT_REQ_S, NB_PAS).astype(float) * up

    p50 = 75.0 + rng.normal(0, 5, NB_PAS)
    p50[PIC_LATENCE] += rng.normal(220, 20, PIC_LATENCE.stop - PIC_LATENCE.start).clip(min=0)
    p50 = p50.clip(min=20)
    p95 = (p50 * 1.5 + rng.normal(0, 8, NB_PAS)).clip(min=30)
    p99 = (p50 * 2.0 + rng.normal(0, 12, NB_PAS)).clip(min=40)
    for serie in (p50, p95, p99):
        serie[PANNE] = 0

    erreurs = (0.5 + rng.normal(0, 0.1, NB_PAS)).clip(min=0)
    erreurs[PANNE] = 100.0

    taux_alerte_ref, taux_surv = _repartition_dernier_batch()
    taux_alerte_fin = taux_alerte_ref + DERIVE_SORTIE_PTS / 100
    taux_alerte = np.linspace(taux_alerte_ref, taux_alerte_fin, NB_PAS)

    def debit_classe(taux: np.ndarray) -> np.ndarray:
        debit: np.ndarray = rng.poisson(DEBIT_REQ_S * PAS_S * taux) / PAS_S * up
        return debit

    # `csat` absent de ~8 % des demandes (nominal) ; `secteur` passe de 5 % à 60 %
    taux_csat = (0.08 + rng.normal(0, 0.006, NB_PAS)).clip(min=0)
    taux_secteur = np.where(np.arange(NB_PAS) < IDX_RUPTURE_CRM, 0.05, 0.60)
    taux_secteur = (taux_secteur + rng.normal(0, 0.006, NB_PAS)).clip(min=0)

    df = pd.DataFrame(
        {
            "ts": ts,
            "up": up,
            "req_par_s": req,
            "latence_p50": p50,
            "latence_p95": p95,
            "latence_p99": p99,
            "erreurs_pct": erreurs,
            "pred_ok": debit_classe(1.0 - taux_alerte - taux_surv),
            "pred_surveillance": debit_classe(np.full(NB_PAS, taux_surv)),
            "pred_alerte_rouge": debit_classe(taux_alerte),
            "manque_csat_par_h": req * 3600 * taux_csat,
            "manque_secteur_par_h": req * 3600 * taux_secteur,
        }
    )
    df.attrs = {
        "duree_panne_min": (PANNE.stop - PANNE.start) * PAS_S / 60,
        "taux_alerte_ref": taux_alerte_ref,
        "taux_alerte_fin": taux_alerte_fin,
        "tolerance_pts": config.CIBLES_PERFORMANCE["ecart_alerte_rouge_rollback_pts"],
    }
    return df


def tracer_tableau_de_bord(df: pd.DataFrame) -> Figure:
    """Trace les quatre vues du tableau de bord simulé et sauvegarde la figure.

    Parameters
    ----------
    df : pd.DataFrame
        Sortie de :func:`simuler_metriques`.

    Returns
    -------
    Figure
        Figure 2 × 2 : disponibilité et erreurs, latence, prédictions par décision,
        champs absents.
    """
    fig, axes = viz.figure_grille(
        "monitoring_tableau_de_bord_simule",
        "Tableau de bord de surveillance simulé — un incident par niveau",
        nlignes=2,
        ncols=2,
        taille=(15, 9),
    )
    ax_dispo, ax_latence, ax_pred, ax_manque = axes.ravel()
    ts = df["ts"]

    ax_dispo.fill_between(ts, df["up"], step="post", color=viz.PALETTE_PRINCIPALE[2], alpha=0.6)
    ax_dispo.set_ylabel("Statut (1 = disponible)")
    ax_err = ax_dispo.twinx()
    ax_err.plot(ts, df["erreurs_pct"], color=viz.COULEUR_CHURN, linewidth=1.2)
    ax_err.set_ylabel("Taux d'erreur 4xx + 5xx (%)", color=viz.COULEUR_CHURN)
    ax_dispo.set_title(
        f"Panne de {nombre(df.attrs['duree_panne_min'], 0)} min → alerte APIIndisponible"
    )

    for col, etiquette, idx in [
        ("latence_p50", "p50", 0),
        ("latence_p95", "p95", 3),
        ("latence_p99", "p99", 1),
    ]:
        ax_latence.plot(ts, df[col], label=etiquette, color=viz.couleur(idx), linewidth=1.2)
    slo = config.CIBLES_PERFORMANCE["latence_unitaire_ms"]
    ax_latence.axhline(slo, linestyle="--", color=viz.COULEUR_CHURN, label=f"SLO {slo} ms")
    ax_latence.set_ylabel("Latence /predict (ms)")
    ax_latence.set_title("Pic de latence soutenu → alerte LatenceElevee")
    ax_latence.legend(loc="upper left", fontsize=9)

    ax_pred.stackplot(
        ts,
        df["pred_ok"],
        df["pred_surveillance"],
        df["pred_alerte_rouge"],
        labels=["OK", "SURVEILLANCE", "ALERTE_ROUGE"],
        colors=[viz.COULEUR_NON_CHURN, viz.PALETTE_PRINCIPALE[3], viz.COULEUR_CHURN],
        alpha=0.8,
    )
    ax_pred.set_ylabel("Prédictions par seconde")
    ax_pred.set_title(f"Dérive de sortie : ALERTE_ROUGE +{DERIVE_SORTIE_PTS} points")
    ax_pred.legend(loc="upper left", fontsize=9)

    ax_manque.plot(ts, df["manque_csat_par_h"], label="csat (nominal)", color=viz.couleur(0))
    ax_manque.plot(ts, df["manque_secteur_par_h"], label="secteur", color=viz.COULEUR_CHURN)
    ax_manque.axvline(ts.iloc[IDX_RUPTURE_CRM], linestyle="--", color=viz.COULEUR_CHURN)
    ax_manque.set_ylabel("Demandes par heure avec champ absent")
    ax_manque.set_title("Rupture d'intégration CRM : secteur n'est plus transmis")
    ax_manque.legend(loc="upper left", fontsize=9)

    for ax in axes.ravel():
        ax.set_xlabel("Heure")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))  # type: ignore[no-untyped-call]
    fig.tight_layout()
    viz.sauvegarder(fig)
    return fig


if __name__ == "__main__":
    chemin = viz.sauvegarder(tracer_tableau_de_bord(simuler_metriques()))
    logger.info("Tableau de bord simulé : {}", chemin)
