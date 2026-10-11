"""Détection de dérive et tests de robustesse du modèle de churn — C9.

Architecture des signaux de surveillance
-----------------------------------------
La dérive se manifeste sur deux axes indépendants, avec des délais de détection très différents :

* **Dérive des entrées** (``psi``, ``ks_test``, ``rapport_evidently``) :
  détectable *immédiatement*, car elle ne nécessite que les données d'entrée —
  aucune étiquette réelle n'est requise. C'est le signal précoce.

* **Dégradation de la performance** (PR-AUC, précision, rappel) :
  détectable uniquement avec les *vraies étiquettes*, qui arrivent en décalé
  (ici : au prochain renouvellement contractuel, soit 1 à 12 mois plus tard).
  C'est le signal tardif mais définitif.

Conséquence opérationnelle : surveiller en continu la dérive des entrées comme
alerte précoce ; déclencher une révision de performance dès que les labels
réels sont disponibles. Ne jamais confondre les deux mécanismes.
"""

from __future__ import annotations

import datetime
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from sklearn.metrics import average_precision_score

from churn_saas import config
from churn_saas.format_fr import nombre

# ---------------------------------------------------------------------------
# Seuils d'interprétation PSI — issus de la pratique de gestion du risque
# crédit (Siddiqi 2006) et repris par la littérature MLOps. Ces seuils
# restent des conventions empiriques : les adapter selon le contexte métier.
# ---------------------------------------------------------------------------
_SEUILS_PSI = {
    "stable": 0.10,  # PSI < 0.10 : distribution stable, pas d'action requise
    "attention": 0.20,  # 0.10 ≤ PSI < 0.20 : dérive modérée, surveillance renforcée
    # PSI ≥ 0.20 : dérive significative, révision du modèle recommandée
}

# Seuil KS — p-value en dessous de laquelle on rejette l'hypothèse que les
# deux échantillons proviennent de la même distribution.
_SEUIL_KS_PVALUE = 0.05

# Seuil d'obsolescence — âge du modèle (en jours) au-delà duquel une révision
# est recommandée, même en l'absence de signal de dérive. Justification :
# les comportements SaaS B2B évoluent typiquement sur un cycle annuel ;
# un modèle de plus de 180 jours a vu passer un demi-cycle de renouvellements.
_SEUIL_OBSOLESCENCE_JOURS = 180


# ---------------------------------------------------------------------------
# 1. PSI (Population Stability Index)
# ---------------------------------------------------------------------------


def psi(
    reference: pd.Series | np.ndarray,
    courant: pd.Series | np.ndarray,
    *,
    n_bins: int = 10,
    epsilon: float = 1e-6,
) -> dict[str, Any]:
    """Calcule le PSI entre une distribution de référence et une distribution courante.

    Le PSI mesure le changement global de distribution d'une variable continue.
    Il est asymétrique : PSI(ref→courant) ≠ PSI(courant→ref).

    **Limites connues** :
    * Sensible au nombre de bins (``n_bins``) — un choix trop faible masque
      les dérives locales, un choix trop élevé amplifie le bruit sur petits échantillons.
    * Instable si certains bins sont vides dans l'une des distributions (géré par ``epsilon``).
    * Conçu pour les variables continues ; pour les variables catégorielles, préférer
      le test du chi-deux ou la distance de Hellinger.

    Parameters
    ----------
    reference :
        Vecteur de la période de référence (entraînement).
    courant :
        Vecteur de la période courante (production).
    n_bins :
        Nombre de bins équidistants calculés sur la référence.
    epsilon :
        Valeur minimale ajoutée aux proportions pour éviter la division par zéro.

    Returns
    -------
    dict avec les clés :
        ``psi`` (float), ``interpretation`` (str), ``action`` (str),
        ``bins`` (DataFrame avec les proportions par bin).
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(courant, dtype=float)

    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]

    bornes = np.percentile(ref, np.linspace(0, 100, n_bins + 1))
    bornes[0] -= 1e-9
    bornes[-1] += 1e-9

    p_ref = np.histogram(ref, bins=bornes)[0] / len(ref)
    p_cur = np.histogram(cur, bins=bornes)[0] / len(cur)

    p_ref = np.maximum(p_ref, epsilon)
    p_cur = np.maximum(p_cur, epsilon)

    psi_par_bin = (p_cur - p_ref) * np.log(p_cur / p_ref)
    valeur_psi = float(np.sum(psi_par_bin))

    if valeur_psi < _SEUILS_PSI["stable"]:
        interpretation = "stable"
        action = "Aucune action requise."
    elif valeur_psi < _SEUILS_PSI["attention"]:
        interpretation = "attention"
        action = "Surveillance renforcée. Planifier une revue de performance."
    else:
        interpretation = "derive_significative"
        action = "Dérive significative. Révision du modèle recommandée."

    df_bins = pd.DataFrame(
        {
            "borne_inf": bornes[:-1],
            "borne_sup": bornes[1:],
            "p_reference": p_ref,
            "p_courant": p_cur,
            "psi_bin": psi_par_bin,
        }
    )

    return {
        "psi": valeur_psi,
        "interpretation": interpretation,
        "action": action,
        "bins": df_bins,
    }


# ---------------------------------------------------------------------------
# 2. Test de Kolmogorov-Smirnov
# ---------------------------------------------------------------------------


def ks_test(
    reference: pd.Series | np.ndarray,
    courant: pd.Series | np.ndarray,
) -> dict[str, Any]:
    """Test KS bilatéral entre une distribution de référence et une distribution courante.

    Le test KS mesure la distance maximale entre les fonctions de répartition empiriques.
    Il est plus sensible aux dérives localisées que le PSI, mais :

    **Limites connues** :
    * Ne capture pas les changements de forme complexes (par ex. variance augmente
      mais moyenne identique → PSI peut le voir, KS moins).
    * Sur de très grands échantillons, rejette quasiment tout (p < 0.05 pour des
      différences négligeables). Interpréter la *statistique D* en parallèle.
    * Conçu pour les distributions continues. Sur des données discrètes ou très
      groupées, le test devient conservateur.

    Returns
    -------
    dict avec les clés :
        ``statistique`` (float), ``p_value`` (float), ``derive_detectee`` (bool),
        ``interpretation`` (str).
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(courant, dtype=float)
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        resultat = stats.ks_2samp(ref, cur)

    derive = bool(resultat.pvalue < _SEUIL_KS_PVALUE)
    if not derive:
        interpretation = f"Pas de dérive détectée (p={nombre(resultat.pvalue, 4)} ≥ {nombre(_SEUIL_KS_PVALUE, 2)})."
    else:
        interpretation = (
            f"Dérive détectée (D={nombre(resultat.statistic, 4)}, p={nombre(resultat.pvalue, 4)} "
            f"< {nombre(_SEUIL_KS_PVALUE, 2)}). Vérifier la distribution courante."
        )

    return {
        "statistique": float(resultat.statistic),
        "p_value": float(resultat.pvalue),
        "derive_detectee": derive,
        "interpretation": interpretation,
    }


# ---------------------------------------------------------------------------
# 3. Rapport Evidently
# ---------------------------------------------------------------------------


def rapport_evidently(
    reference: pd.DataFrame,
    courant: pd.DataFrame,
    *,
    chemin_sortie: Path | None = None,
    forcer: bool = False,
) -> Path:
    """Génère un rapport de dérive des données via Evidently et l'écrit en HTML.

    Le rapport inclut un ``DataDriftPreset`` (dérive par feature) et un
    ``DataSummaryPreset`` (valeurs manquantes, types, statistiques de base).

    Le fichier HTML est écrit dans ``reports/`` (ou dans ``chemin_sortie`` si fourni).
    Si le fichier existe déjà et ``forcer=False``, il est rechargé sans recalcul —
    même logique que ``cache.charger_ou_calculer()``.

    Parameters
    ----------
    reference :
        DataFrame de la période de référence (données d'entraînement ou baseline).
    courant :
        DataFrame de la période courante (données de production récentes).
    chemin_sortie :
        Chemin complet du fichier HTML produit. Par défaut :
        ``reports/drift_report_<date>.html``.
    forcer :
        Si True, recalcule même si le fichier existe.

    Returns
    -------
    Path du fichier HTML produit.
    """
    import evidently  # noqa: F401 — import tardif pour ne pas bloquer si optionnel
    from evidently import Report, presets

    if chemin_sortie is None:
        date_str = datetime.date.today().isoformat()
        chemin_sortie = config.RACINE / "reports" / f"drift_report_{date_str}.html"

    chemin_sortie = Path(chemin_sortie)

    if chemin_sortie.exists() and not forcer:
        logger.info("Rapport Evidently déjà présent → {}", chemin_sortie.name)
        return chemin_sortie

    logger.info("Génération du rapport Evidently …")
    chemin_sortie.parent.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rapport = Report(metrics=[presets.DataDriftPreset(), presets.DataSummaryPreset()])
        snapshot = rapport.run(reference_data=reference, current_data=courant)
        snapshot.save_html(str(chemin_sortie))

    logger.info("Rapport Evidently écrit → {}", chemin_sortie)
    return chemin_sortie


# ---------------------------------------------------------------------------
# 4. Simulateur de dérive réaliste
# ---------------------------------------------------------------------------


def simuler_derive(
    df: pd.DataFrame,
    type_de_derive: str,
    *,
    intensite: float = 1.0,
    graine: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """Fabrique une dérive réaliste sur le DataFrame courant pour démontrer la détection.

    Types disponibles
    -----------------
    ``"adoption_chute"``
        Baisse généralisée du taux d'adoption et des connexions — simule une
        vague de désengagement utilisateur (nouvelle interface, incident produit).

    ``"nouveau_segment"``
        Arrivée d'un nouveau segment de clients (ex. grands comptes avec beaucoup
        de sièges souscrits mais faible adoption initiale). Modifie la distribution
        de ``sieges_souscrits`` et ``utilisateurs_actifs``.

    ``"anciennete_rajeunissement"``
        Afflux de nouveaux clients — l'ancienneté moyenne chute, les comptes
        établis sont proportionnellement moins représentés.

    ``"support_degradation"``
        Hausse du volume de tickets support et allongement des délais de réponse —
        simule une dégradation de la qualité du support.

    Parameters
    ----------
    df :
        DataFrame source (copie — l'original n'est pas modifié).
    type_de_derive :
        Identifiant du scénario (voir ci-dessus).
    intensite :
        Facteur multiplicatif de l'amplitude de la dérive (1.0 = scénario standard).
    graine :
        Graine aléatoire pour la reproductibilité.

    Returns
    -------
    DataFrame avec la dérive appliquée.
    """
    rng = np.random.default_rng(graine)
    # Normalisation vers dtypes numpy standard.
    # Les DataFrames chargés depuis CSV peuvent avoir des colonnes numériques
    # en ArrowDtype (large_string, double[pyarrow]) qui ne supportent pas
    # l'arithmétique numpy directe. On tente np.asarray(float) colonne par colonne ;
    # en cas d'échec (colonne catégorielle), on conserve la version string.
    data: dict[str, Any] = {}
    for col in df.columns:
        try:
            arr = np.asarray(df[col], dtype=float)
            data[col] = arr
        except (ValueError, TypeError):
            data[col] = df[col].to_numpy(dtype=object, na_value=None)
    derive = pd.DataFrame(data, index=df.index)
    n = len(derive)

    if type_de_derive == "adoption_chute":
        # Baisse de l'adoption : taux_adoption_pct chute de 20 pt en moyenne
        if "taux_adoption_pct" in derive.columns:
            bruit = rng.normal(0, 5 * intensite, n)
            derive["taux_adoption_pct"] = (
                derive["taux_adoption_pct"] - 20 * intensite + bruit
            ).clip(0, 100)
        if "connexions_30j" in derive.columns:
            facteur = rng.uniform(0.4, 0.7, n) ** intensite
            derive["connexions_30j"] = (derive["connexions_30j"] * facteur).clip(0).round()

    elif type_de_derive == "nouveau_segment":
        # 30 % des lignes remplacées par un profil grand compte
        n_nouveaux = int(n * 0.30 * intensite)
        idx = rng.choice(n, n_nouveaux, replace=False)
        if "sieges_souscrits" in derive.columns:
            derive.loc[derive.index[idx], "sieges_souscrits"] = rng.integers(50, 200, n_nouveaux)
        if "utilisateurs_actifs" in derive.columns:
            derive.loc[derive.index[idx], "utilisateurs_actifs"] = rng.integers(5, 20, n_nouveaux)
        if "revenu_mensuel_recurrent_eur" in derive.columns:
            derive.loc[derive.index[idx], "revenu_mensuel_recurrent_eur"] = rng.uniform(
                5000, 20000, n_nouveaux
            )

    elif type_de_derive == "anciennete_rajeunissement":
        # L'ancienneté moyenne chute de moitié
        if "anciennete_mois" in derive.columns:
            derive["anciennete_mois"] = (
                derive["anciennete_mois"] * (1 - 0.5 * intensite) + rng.normal(0, 3, n)
            ).clip(0)

    elif type_de_derive == "support_degradation":
        # Hausse des tickets et allongement du délai de réponse
        if "tickets_support_90j" in derive.columns:
            derive["tickets_support_90j"] = (
                derive["tickets_support_90j"] * (1 + 1.5 * intensite) + rng.poisson(2, n)
            ).clip(0)
        if "delai_reponse_support_h" in derive.columns:
            derive["delai_reponse_support_h"] = (
                derive["delai_reponse_support_h"] * (1 + 2.0 * intensite) + rng.exponential(10, n)
            ).clip(0)

    else:
        types_valides = [
            "adoption_chute",
            "nouveau_segment",
            "anciennete_rajeunissement",
            "support_degradation",
        ]
        raise ValueError(
            f"Type de dérive inconnu : '{type_de_derive}'. " f"Valeurs acceptées : {types_valides}"
        )

    return derive


# ---------------------------------------------------------------------------
# 5. Test de robustesse
# ---------------------------------------------------------------------------


def tester_robustesse(
    modele: Any,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    niveaux_bruit: list[float] | None = None,
    taux_manquants: list[float] | None = None,
    graine: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """Mesure la dégradation de PR-AUC sous deux types de perturbation à l'inférence.

    Item C9 — robustesse : le modèle doit être évalué non seulement sur des données
    propres, mais aussi sous des conditions dégradées réalistes (données de production
    bruitées ou incomplètes).

    Perturbations testées
    ----------------------
    * **Bruit gaussien croissant** sur les variables numériques :
      à chaque niveau, un bruit N(0, σ × std_feature) est ajouté.
      Simule des erreurs de capteur, des arrondis, ou des transformations imparfaites.

    * **Injection de valeurs manquantes** à des taux croissants :
      chaque cellule numérique est mise à NaN avec la probabilité ``taux``.
      Le modèle doit comporter un imputer dans son pipeline pour y survivre.

    Parameters
    ----------
    modele :
        Modèle scikit-learn avec ``predict_proba``. Doit gérer les NaN
        s'il contient un imputer dans son pipeline.
    X :
        Features du jeu d'évaluation (non vues à l'entraînement).
    y :
        Étiquettes réelles correspondantes.
    niveaux_bruit :
        Liste de σ (en fraction de l'écart-type de chaque feature).
        Défaut : [0.0, 0.1, 0.25, 0.5, 1.0].
    taux_manquants :
        Liste de taux de NaN injectés.
        Défaut : [0.0, 0.05, 0.10, 0.20, 0.30].
    graine :
        Graine pour la reproductibilité.

    Returns
    -------
    DataFrame avec les colonnes :
        ``type_perturbation``, ``niveau``, ``pr_auc``, ``degradation_relative_pct``.

    Notes
    -----
    La dégradation relative est calculée par rapport au PR-AUC sans perturbation
    (niveau 0.0). Une dégradation > 10 % est considérée comme significative.
    """
    if niveaux_bruit is None:
        niveaux_bruit = [0.0, 0.1, 0.25, 0.5, 1.0]
    if taux_manquants is None:
        taux_manquants = [0.0, 0.05, 0.10, 0.20, 0.30]

    rng = np.random.default_rng(graine)
    y_arr = np.asarray(y)
    cols_num = X.select_dtypes(include="number").columns.tolist()

    def _pr_auc(X_perturbe: pd.DataFrame) -> float:
        try:
            proba = modele.predict_proba(X_perturbe)[:, 1]
            return float(average_precision_score(y_arr, proba))
        except Exception as exc:
            logger.warning("Erreur lors du calcul PR-AUC : {}", exc)
            return float("nan")

    # Référence : PR-AUC sans perturbation
    pr_auc_ref = _pr_auc(X)
    logger.info("PR-AUC de référence (sans perturbation) : {:.4f}", pr_auc_ref)

    lignes: list[dict[str, Any]] = []

    # --- Bruit gaussien ---
    for sigma in niveaux_bruit:
        if sigma == 0.0:
            pr = pr_auc_ref
        else:
            X_bruit = X.copy()
            for col in cols_num:
                std = float(X[col].std())
                X_bruit[col] = X[col] + rng.normal(0, sigma * std, len(X))
            pr = _pr_auc(X_bruit)

        degr = (pr_auc_ref - pr) / pr_auc_ref * 100 if pr_auc_ref > 0 else float("nan")
        lignes.append(
            {
                "type_perturbation": "bruit_gaussien",
                "niveau": sigma,
                "pr_auc": round(pr, 4),
                "degradation_relative_pct": round(degr, 2),
            }
        )
        logger.debug("Bruit σ={:.2f} → PR-AUC={:.4f} (−{:.1f}%)", sigma, pr, degr)

    # --- Valeurs manquantes ---
    for taux in taux_manquants:
        if taux == 0.0:
            pr = pr_auc_ref
        else:
            X_nan = X.copy()
            masque = rng.random(X_nan[cols_num].shape) < taux
            X_nan[cols_num] = X_nan[cols_num].where(~masque, other=np.nan)
            pr = _pr_auc(X_nan)

        degr = (pr_auc_ref - pr) / pr_auc_ref * 100 if pr_auc_ref > 0 else float("nan")
        lignes.append(
            {
                "type_perturbation": "valeurs_manquantes",
                "niveau": taux,
                "pr_auc": round(pr, 4),
                "degradation_relative_pct": round(degr, 2),
            }
        )
        logger.debug("NaN taux={:.0%} → PR-AUC={:.4f} (−{:.1f}%)", taux, pr, degr)

    return pd.DataFrame(lignes)


# ---------------------------------------------------------------------------
# 6. Indicateur d'obsolescence
# ---------------------------------------------------------------------------


def indicateur_obsolescence(
    date_entrainement: datetime.date | datetime.datetime | str,
    date_courante: datetime.date | datetime.datetime | str | None = None,
) -> dict[str, Any]:
    """Calcule l'âge du modèle et évalue son niveau d'obsolescence.

    L'âge seul ne prouve pas l'obsolescence (un modèle sur un marché stable peut
    tenir longtemps), mais il fournit un signal de déclenchement de revue en
    l'absence de dérive mesurable. Seuil par défaut : 180 jours (demi-cycle
    de renouvellement contractuel annuel typique en SaaS B2B).

    Parameters
    ----------
    date_entrainement :
        Date ou datetime d'entraînement du modèle. Peut être une chaîne ISO 8601.
    date_courante :
        Date de référence pour le calcul. Par défaut : aujourd'hui.

    Returns
    -------
    dict avec les clés :
        ``age_jours`` (int), ``seuil_jours`` (int), ``statut`` (str),
        ``message`` (str), ``date_revue_recommandee`` (date).
    """

    def _to_date(val: datetime.date | datetime.datetime | str) -> datetime.date:
        if isinstance(val, datetime.datetime):
            return val.date()
        if isinstance(val, datetime.date):
            return val
        # datetime.fromisoformat accepte une date seule comme un horodatage complet : les
        # métadonnées de `churn-saas train` portent l'heure (« 2026-10-01T00:10:38 »)
        return datetime.datetime.fromisoformat(str(val)).date()

    d_train = _to_date(date_entrainement)
    d_courant = _to_date(date_courante) if date_courante is not None else datetime.date.today()
    age = (d_courant - d_train).days

    date_revue = d_train + datetime.timedelta(days=_SEUIL_OBSOLESCENCE_JOURS)

    if age < _SEUIL_OBSOLESCENCE_JOURS:
        statut = "ok"
        message = (
            f"Modèle âgé de {age} jours. "
            f"Revue recommandée à partir du {date_revue.isoformat()}."
        )
    else:
        statut = "revision_recommandee"
        message = (
            f"Modèle âgé de {age} jours (seuil : {_SEUIL_OBSOLESCENCE_JOURS} j). "
            "Révision recommandée — déclencher un audit de dérive et de performance."
        )

    return {
        "age_jours": age,
        "seuil_jours": _SEUIL_OBSOLESCENCE_JOURS,
        "statut": statut,
        "message": message,
        "date_revue_recommandee": date_revue,
    }
