"""Construction du préprocesseur sklearn et des features métier pour le pipeline.

Ce module est la SEULE autorité sur quelles colonnes atteignent le modèle.
Toute modification du périmètre des features passe ici — jamais dispersée
dans le notebook ou dans les scripts d'entraînement.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_saas import config
from churn_saas.features.transformers import (
    EcartAuGroupe,
    LogAsymetrique,
    NormalisationCategorielle,
    est_texte,
)

# Colonnes présentes dans le gold mais écartées du modèle sans être des fuites : la structure
# temporelle est exploitée par le type de split (§6.11), pas comme feature brute
COLONNES_NON_MODELISEES: frozenset[str] = frozenset({"date_souscription", "jour_souscription"})


def colonnes_features(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Retourne (colonnes_numériques, colonnes_catégorielles) admissibles au modèle.

    C'est la SEULE fonction qui décide quelles colonnes atteignent le modèle.
    Elle lève une exception si une colonne de ``config.COLONNES_INTERDITES`` est présente,
    forçant l'appelant à épurer le DataFrame avant de continuer.

    Parameters
    ----------
    df:
        DataFrame épuré des colonnes interdites. Si elles sont encore présentes,
        la fonction lève plutôt que de les ignorer silencieusement.

    Returns
    -------
    (numeriques, categorielles) — listes de noms de colonnes, sans les interdites.

    Raises
    ------
    ValueError
        Si une colonne de ``config.COLONNES_INTERDITES`` est présente dans ``df``.
    """
    presentes_interdites = set(df.columns) & set(config.COLONNES_INTERDITES)
    if presentes_interdites:
        raise ValueError(
            f"Colonnes interdites détectées dans le DataFrame : {sorted(presentes_interdites)}. "
            "Épurer avec df.drop(columns=config.COLONNES_INTERDITES, errors='ignore') avant "
            "d'appeler colonnes_features()."
        )

    numeriques = [
        col
        for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col])
    ]
    categorielles = [
        col
        for col in df.columns
        if col not in COLONNES_NON_MODELISEES
        and (est_texte(df[col]) or isinstance(df[col].dtype, pd.CategoricalDtype))
    ]
    logger.info(
        "colonnes_features — {} numériques, {} catégorielles sélectionnées",
        len(numeriques),
        len(categorielles),
    )
    return numeriques, categorielles


def construire_preprocesseur(df: pd.DataFrame, log_asymetrique: bool = False) -> ColumnTransformer:
    """Construit un ColumnTransformer prêt à s'intégrer dans un Pipeline sklearn.

    Traitements appliqués :
    - **Numériques** : imputation médiane → StandardScaler.
    - **Catégorielles** : normalisation casse/espaces → imputation par ``"inconnu"`` → OneHotEncoder
      avec ``handle_unknown="ignore"`` (modalités inconnues au transform → vecteur nul).
    - **``ecart_csat_secteur``** (si ``secteur`` et ``csat`` sont présents) : CSAT − médiane du
      secteur apprise sur le train du pli (``EcartAuGroupe``) → imputation médiane →
      StandardScaler.
    - **Colonnes interdites** : exclues via ``remainder="drop"`` — elles n'atteignent
      jamais le modèle même si le DataFrame d'entrée les contient.

    Le schéma des colonnes est figé à la construction (pas au fit), ce qui garantit
    que les noms de features sont stables entre les plis de validation croisée.

    Parameters
    ----------
    df:
        DataFrame de référence pour déduire le schéma (colonnes + types).
        Peut contenir les colonnes interdites : elles sont retirées en interne
        avant de construire le ColumnTransformer.
    log_asymetrique:
        Insère ``LogAsymetrique`` (log1p signé des colonnes asymétriques, choisies sur le
        train du pli) entre l'imputation et la standardisation. **Désactivé** : testé en §7.8.3,
        il dégrade la régression logistique sur tous les plis. L'option ne sert qu'à
        reproduire cette comparaison.

    Returns
    -------
    ColumnTransformer configuré, non encore fitté.
    """
    df_features = df.drop(columns=config.COLONNES_INTERDITES, errors="ignore")
    numeriques, categorielles = colonnes_features(df_features)

    etapes_numeriques: list[tuple[str, Any]] = [("imputation", SimpleImputer(strategy="median"))]
    if log_asymetrique:
        # Après l'imputation (entrée sans NaN), avant la standardisation
        etapes_numeriques.append(
            ("log_asymetrique", LogAsymetrique(seuil_asymetrie=config.SEUIL_ASYMETRIE_LOG))
        )
    etapes_numeriques.append(("standardisation", StandardScaler()))
    pipeline_numerique = Pipeline(steps=etapes_numeriques)
    pipeline_categoriel = Pipeline(
        steps=[
            ("normalisation", NormalisationCategorielle()),
            ("imputation", SimpleImputer(strategy="constant", fill_value="inconnu")),
            (
                "encodage",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    transformateurs: list[tuple[str, Any, list[str]]] = [
        ("numerique", pipeline_numerique, numeriques),
        ("categoriel", pipeline_categoriel, categorielles),
    ]
    # Écart au CSAT médian du secteur : la médiane par secteur est une statistique apprise,
    # donc calculée ici, dans le pli, et jamais dans le gold (§7.2)
    if {"secteur", "csat"} <= set(df_features.columns):
        pipeline_ecart = Pipeline(
            steps=[
                ("agregat", EcartAuGroupe("secteur", "csat", "median", nom="ecart_csat_secteur")),
                ("imputation", SimpleImputer(strategy="median")),
                ("standardisation", StandardScaler()),
            ]
        )
        transformateurs.append(("ecart_csat_secteur", pipeline_ecart, ["secteur", "csat"]))

    preprocesseur = ColumnTransformer(
        transformers=transformateurs,
        # Toutes les colonnes non listées (interdites, dates…) tombent ici et sont supprimées
        remainder="drop",
        verbose_feature_names_out=False,
    )
    logger.info(
        "construire_preprocesseur — schéma : {} numériques, {} catégorielles, reste exclu",
        len(numeriques),
        len(categorielles),
    )
    return preprocesseur


# ---------------------------------------------------------------------------
# Feature engineering métier
# ---------------------------------------------------------------------------


def reconstituer_taux_adoption(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Recalcule ``taux_adoption_pct`` manquant à partir de sa définition exacte.

    Sur le jeu fourni, ``taux_adoption_pct`` vaut exactement
    ``100 × utilisateurs_actifs / sieges_souscrits`` (contrôle de §5.8). Une valeur manquante
    est donc recalculée sans erreur, ce qui vaut mieux qu'une imputation par la médiane.
    Règle ligne par ligne, sans paramètre appris : aucune fuite, applicable avant le split,
    en batch comme dans l'API. Les valeurs renseignées ne sont jamais modifiées.

    Returns
    -------
    (df_complété, nombre_de_valeurs_reconstituées) — ``df`` n'est pas modifié en place.
    """
    df = df.copy()
    colonnes = {"taux_adoption_pct", "utilisateurs_actifs", "sieges_souscrits"}
    if not colonnes <= set(df.columns):
        return df, 0
    taux = pd.to_numeric(df["taux_adoption_pct"], errors="coerce")
    sieges = pd.to_numeric(df["sieges_souscrits"], errors="coerce")
    recalcule = (
        100 * pd.to_numeric(df["utilisateurs_actifs"], errors="coerce") / sieges.where(sieges > 0)
    )
    a_combler = taux.isna() & recalcule.notna()
    df["taux_adoption_pct"] = taux.where(~a_combler, recalcule.clip(0, 100))
    return df, int(a_combler.sum())


def controle_coherence_adoption(df: pd.DataFrame) -> pd.DataFrame:
    """Mesure l'écart entre ``taux_adoption_pct`` et sa définition ``100 × utilisateurs / sièges``.

    Justifie la reconstitution de ``reconstituer_taux_adoption`` : si l'égalité tient sur les
    lignes renseignées, un taux manquant peut être recalculé plutôt qu'imputé. Contrôle ligne
    par ligne, sans paramètre appris. Une ligne est incohérente si l'écart absolu dépasse
    ``config.SEUIL_INCOHERENCE_ADOPTION_PTS`` points ; seules les lignes où les deux valeurs
    existent (taux renseigné, sièges > 0, utilisateurs connus) sont comparées.

    Parameters
    ----------
    df:
        DataFrame contenant ``taux_adoption_pct``, ``utilisateurs_actifs`` et
        ``sieges_souscrits``. Non modifié.

    Returns
    -------
    Bilan d'une ligne : ``n_comparables``, ``ecart_moyen_pts``, ``ecart_max_pts``,
    ``n_incoherences``, ``part_incoherences`` (NaN si aucune ligne comparable).
    """
    colonnes = {"taux_adoption_pct", "utilisateurs_actifs", "sieges_souscrits"}
    if colonnes <= set(df.columns):
        taux = pd.to_numeric(df["taux_adoption_pct"], errors="coerce")
        sieges = pd.to_numeric(df["sieges_souscrits"], errors="coerce")
        utilisateurs = pd.to_numeric(df["utilisateurs_actifs"], errors="coerce")
        ecart = (100 * utilisateurs / sieges.where(sieges > 0) - taux).abs().dropna()
    else:
        logger.warning("controle_coherence_adoption — colonnes absentes : {}", sorted(colonnes))
        ecart = pd.Series(dtype=float)

    n_comparables = len(ecart)
    n_incoherences = int((ecart > config.SEUIL_INCOHERENCE_ADOPTION_PTS).sum())
    bilan = pd.DataFrame(
        {
            "n_comparables": [n_comparables],
            "ecart_moyen_pts": [ecart.mean() if n_comparables else np.nan],
            "ecart_max_pts": [ecart.max() if n_comparables else np.nan],
            "n_incoherences": [n_incoherences],
            "part_incoherences": [n_incoherences / n_comparables if n_comparables else np.nan],
        }
    )
    logger.info(
        "controle_coherence_adoption — {} incohérences sur {} lignes comparables",
        n_incoherences,
        n_comparables,
    )
    return bilan


def ajouter_features_metier(
    df: pd.DataFrame,
    csat_median_par_secteur: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Crée ~15 variables dérivées à partir des colonnes brutes.

    Toutes les features sont calculées ligne par ligne — aucune statistique apprise
    sur le jeu d'entraînement n'est nécessaire ici.

    Exception : ``ecart_csat_secteur`` requiert les médianes CSAT par secteur
    apprises sur le train via ``AgregatParGroupe`` — passées via
    ``csat_median_par_secteur``. Si ce paramètre est None, la colonne n'est
    pas créée, garantissant l'absence de fuite.

    Convention division par zéro : NaN (sera imputé dans le pipeline sklearn).

    Parameters
    ----------
    df:
        DataFrame contenant les colonnes brutes source.
    csat_median_par_secteur:
        Dict ``{secteur: médiane_csat}`` appris **uniquement sur le train**
        (via ``AgregatParGroupe``). Si None, ``ecart_csat_secteur`` n'est pas créée.

    Returns
    -------
    DataFrame enrichi des features dérivées (copie — l'original n'est pas modifié).
    """
    # Avant toute feature : un taux d'adoption déductible ne doit pas être imputé en aval
    df, _ = reconstituer_taux_adoption(df)
    _initiales = set(df.columns)

    def _num(col: str) -> pd.Series:
        """Série numérique ou NaN si la colonne est absente."""
        if col not in df.columns:
            logger.warning("ajouter_features_metier — colonne '{}' absente, NaN substitué", col)
            return pd.Series(float("nan"), index=df.index, dtype=float)
        return pd.to_numeric(df[col], errors="coerce")

    sieges = _num("sieges_souscrits")
    utilisateurs = _num("utilisateurs_actifs")
    heures = _num("heures_usage_30j")
    connexions = _num("connexions_30j")
    fonc_util = _num("fonctionnalites_utilisees")
    fonc_total = _num("fonctionnalites_total")
    mrr = _num("revenu_mensuel_recurrent_eur")
    recence = _num("derniere_connexion_jours")
    anciennete = _num("anciennete_mois")
    tickets = _num("tickets_support_90j")
    integrations = _num("nb_integrations")

    # Dénominateurs sécurisés : 0 → NaN pour éviter inf et ZeroDivisionError
    sieges_d = sieges.where(sieges > 0)
    utilisateurs_d = utilisateurs.where(utilisateurs > 0)
    fonc_total_d = fonc_total.where(fonc_total > 0)
    anciennete_jours_d = (anciennete * 30).where(anciennete > 0)

    # 1. Taux d'utilisation des sièges
    # Sous-utilisation → sièges payés non déployés → risque de non-renouvellement
    df["taux_utilisation_sieges"] = utilisateurs / sieges_d

    # 2. Surdimensionnement absolu (sièges vides payés)
    # Nombre visible sur la facture → renforce l'argument de résiliation
    df["surdimensionnement"] = (sieges - utilisateurs).clip(lower=0)

    # 3. Intensité d'usage par utilisateur actif
    # Usage concentré sur peu d'utilisateurs = ancrage précaire dans l'organisation
    df["intensite_usage_par_utilisateur"] = heures / utilisateurs_d

    # 4. Connexions par utilisateur actif
    # Faible ratio = dépendance partielle au produit, usage concentré ou intermittent
    df["connexions_par_utilisateur"] = connexions / utilisateurs_d

    # 5. Taux de couverture fonctionnelle (breadth of use)
    # Peu de fonctionnalités explorées = valeur perçue faible → risque churn faute de ROI
    df["taux_couverture_fonctionnelle"] = fonc_util / fonc_total_d

    # 6. ARPU par siège (Average Revenue Per Unit)
    # ARPU bas = remise commerciale ou plan sous-utilisé → relation contractuelle fragile
    df["arpu_par_siege"] = mrr / sieges_d

    # 7. Récence normalisée (fraction de vie sans connexion)
    # Inactivité longue / ancienneté totale = décrochage progressif, plus fort que les jours bruts
    df["recence_normalisee"] = recence / anciennete_jours_d

    # 8. Compte dormant (inactif depuis > 30 jours)
    # Signal binaire de décrochage — 30 j ≈ un cycle de reporting client
    df["compte_dormant"] = recence > 30

    # 9. Pression support par utilisateur actif
    # Beaucoup de tickets par utilisateur = frictions répétées → prédictif si CSAT faible
    df["pression_support"] = tickets / utilisateurs_d

    # 9 bis. Intensité support rapportée à l'ancienneté (tickets sur 90 j / mois d'ancienneté)
    # Limite : la fenêtre de 90 j est rapportée à l'ancienneté totale, deux échelles de temps
    # différentes. Le ratio gonfle mécaniquement pour les comptes récents (petit dénominateur) :
    # il est corrélé par construction à l'ancienneté et à tranche_anciennete
    df["intensite_support"] = tickets / anciennete.where(anciennete > 0)

    # 10. Écart au CSAT médian du secteur (ANTI-FUITE : agrégat externe obligatoire)
    # Un client en dessous de la médiane sectorielle est vulnérable à la concurrence.
    # Les médianes DOIVENT venir d'un AgregatParGroupe fitté sur le train uniquement —
    # jamais calculées sur le jeu complet, ce qui constituerait une fuite de données.
    if csat_median_par_secteur is not None and "csat" in df.columns and "secteur" in df.columns:
        csat_ref = df["secteur"].map(csat_median_par_secteur)
        df["ecart_csat_secteur"] = _num("csat") - csat_ref

    # 11. Tranche d'ancienneté (3 phases du cycle de vie client)
    # Onboarding (<3 mois), consolidation (3-12 mois) et maturité (>12 mois) ont
    # des profils de risque structurellement différents
    df["tranche_anciennete"] = pd.cut(
        anciennete,
        bins=[-0.001, 2, 12, float("inf")],
        labels=["onboarding", "installation", "mature"],
    ).astype(object)

    # 12. Tranche d'intégrations tierces (proxy de stickiness)
    # Chaque intégration augmente le coût de migration et réduit le risque de churn
    df["tranche_integrations"] = pd.cut(
        integrations,
        bins=[-0.001, 0, 3, 8, float("inf")],
        labels=["aucune", "faible", "moderee", "forte"],
    ).astype(object)

    # 13-15. Indicateurs de manquance — hypothèse métier (un CSAT manquant révélerait un client
    # insatisfait). Le test de manquance du notebook (§5.9) ne la confirme pas sur les données
    # fournies ; les indicateurs sont conservés, sans fuite, et départagés par l'importance
    # de permutation (§12.8)
    for _col_mnar in ["csat", "heures_usage_30j", "delai_reponse_support_h"]:
        if _col_mnar in df.columns:
            df[f"{_col_mnar}_manquant"] = df[_col_mnar].isna()

    _nouvelles = sorted(set(df.columns) - _initiales)
    logger.info(
        "ajouter_features_metier — {} features ajoutées : {}",
        len(_nouvelles),
        _nouvelles,
    )
    return df


def joindre_catalogue(df: pd.DataFrame, catalogue: pd.DataFrame) -> pd.DataFrame:
    """Joint le catalogue des plans et dérive les features commerciales.

    Dérive :
    - ``remise_consentie`` : écart entre MRR réel et prix catalogue théorique.
      Une remise forte peut signaler une négociation difficile → risque de churn.
    - ``adequation_plan`` : positionnement du client par rapport au plan souscrit.
      Sur-dimensionné (sièges sous-utilisés) = candidat au churn (perçoit un gaspillage) ;
      sous-dimensionné (quasi-saturation) = candidat à l'upsell, pas au churn.

    Convention division par zéro : NaN.

    Parameters
    ----------
    df:
        DataFrame client avec au minimum ``plan``, ``revenu_mensuel_recurrent_eur``,
        ``sieges_souscrits``, ``utilisateurs_actifs``.
    catalogue:
        DataFrame du catalogue (4 lignes, une par plan).
        Colonnes attendues : ``plan``, ``prix_mensuel_par_siege_eur``.

    Returns
    -------
    DataFrame enrichi des colonnes catalogue et des deux features dérivées.
    """
    df = df.copy()

    # Normalisation de la clé de jointure (casse hétérogène dans le jeu réel)
    df["_plan_norm"] = df["plan"].str.strip().str.title() if "plan" in df.columns else None
    cat = catalogue.copy()
    cat["_plan_norm"] = cat["plan"].str.strip().str.title()

    cols_enrichissement = [c for c in cat.columns if c not in {"plan", "_plan_norm"}]
    df = df.merge(
        cat[["_plan_norm"] + cols_enrichissement],
        on="_plan_norm",
        how="left",
    ).drop(columns=["_plan_norm"])

    # Remise consentie = 1 − (MRR réel / prix catalogue × sièges)
    mrr = pd.to_numeric(df.get("revenu_mensuel_recurrent_eur"), errors="coerce")
    prix = pd.to_numeric(df.get("prix_mensuel_par_siege_eur"), errors="coerce")
    sieges = pd.to_numeric(df.get("sieges_souscrits"), errors="coerce")
    prix_catalogue_total = prix * sieges
    # Division par zéro (prix=0 ou sieges=0) → NaN
    df["remise_consentie"] = 1.0 - mrr / prix_catalogue_total.where(prix_catalogue_total > 0)

    # Adéquation au plan : classification du taux d'utilisation des sièges
    utilisateurs = pd.to_numeric(df.get("utilisateurs_actifs"), errors="coerce")
    taux_util = utilisateurs / sieges.where(sieges > 0)
    # np.select ne tolère pas None dans les choix (contrainte mypy) ; on construit en deux temps
    _adequation: pd.Series = pd.Series(
        np.select(
            [taux_util >= 0.9, taux_util < 0.5],
            ["sous-dimensionne", "sur-dimensionne"],
            default="adapte",
        ),
        index=df.index,
        dtype=object,
    )
    _adequation[taux_util.isna()] = None  # sieges=0 ou utilisateurs NaN → valeur manquante
    df["adequation_plan"] = _adequation

    logger.info(
        "joindre_catalogue — {} lignes enrichies, {} colonnes catalogue + 2 features dérivées",
        len(df),
        len(cols_enrichissement),
    )
    return df
