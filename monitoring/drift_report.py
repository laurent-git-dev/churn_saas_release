"""Flow Prefect 3 hebdomadaire — rapport de dérive (``make drift``).

Planifié chaque lundi à 05 h 00 par le déploiement ``derive-hebdomadaire`` de
``prefect.yaml``, avant la revue du lundi ; ``make drift`` le lance à la demande en local runner.

Génère un rapport Evidently de dérive des données en comparant la distribution
de référence (entraînement) à une distribution courante simulée.

En production, remplacer ``df_courant`` par les données réelles de la fenêtre
de monitoring (ex. les 30 derniers jours de scores envoyés à l'API).
"""

import datetime
import warnings

warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402
from loguru import logger  # noqa: E402
from prefect import flow  # noqa: E402

from churn_saas import config  # noqa: E402
from churn_saas.monitoring.drift import (  # noqa: E402
    indicateur_obsolescence,
    ks_test,
    psi,
    rapport_evidently,
    simuler_derive,
)


def _charger_donnees_reference() -> pd.DataFrame:
    """Charge le gold dataset (données nettoyées) et écarte les colonnes interdites.

    On utilise le gold dataset plutôt que le CSV brut : les features sont déjà
    typées en float64, les formats hétérogènes résolus, les doublons supprimés.
    En production, remplacer par les données de la fenêtre de monitoring courante.
    """

    chemin_gold = config.DONNEES_GOLD / "gold_dataset.parquet"
    if chemin_gold.exists():
        df = pd.read_parquet(chemin_gold)
    else:
        # Fallback CSV brut si le gold n'est pas encore généré
        df = pd.read_csv(
            config.DONNEES_BRUTES / "churn_saas_complet.csv",
            sep=None,
            engine="python",
            on_bad_lines="skip",
        )
    cols_interdites = set(config.COLONNES_INTERDITES) | set(config.COLONNES_LEURRES_SUSPECTES)
    df = df.drop(columns=[c for c in cols_interdites if c in df.columns])
    # Ne conserver que les colonnes numériques pour l'audit de dérive
    return df.select_dtypes(include="number")


@flow(
    name="rapport-derive-hebdomadaire",
    description="PSI, test KS, âge du modèle et rapport Evidently sur la fenêtre écoulée.",
)
def main() -> None:
    logger.info("=== Rapport de dérive — make drift ===")

    df_ref = _charger_donnees_reference()
    logger.info("Référence : {} lignes, {} colonnes", len(df_ref), df_ref.shape[1])

    # En production : remplacer par les données réelles de la fenêtre de monitoring.
    df_courant = simuler_derive(df_ref, "adoption_chute", intensite=1.0)
    logger.info("Distribution courante simulée (scénario : baisse d'adoption).")

    features_audit = [
        "taux_adoption_pct",
        "connexions_30j",
        "anciennete_mois",
        "tickets_support_90j",
        "revenu_mensuel_recurrent_eur",
    ]

    print("\n" + "=" * 60)
    print("  AUDIT DE DÉRIVE — PSI et KS")
    print("=" * 60)

    for feat in features_audit:
        if feat not in df_ref.columns or feat not in df_courant.columns:
            continue
        res_psi = psi(df_ref[feat].dropna(), df_courant[feat].dropna())
        res_ks = ks_test(df_ref[feat].dropna(), df_courant[feat].dropna())
        alerte = (
            "⚠️" if res_psi["interpretation"] != "stable" or res_ks["derive_detectee"] else "✓ "
        )
        print(
            f"  {alerte} {feat:<35s} "
            f"PSI={res_psi['psi']:.4f} [{res_psi['interpretation']}]  "
            f"KS p={res_ks['p_value']:.4f} "
            f"({'dérive' if res_ks['derive_detectee'] else 'stable'})"
        )

    print("=" * 60)

    date_entrainement_simulee = datetime.date.today() - datetime.timedelta(days=200)
    obs = indicateur_obsolescence(date_entrainement_simulee)
    statut_emoji = "⚠️" if obs["statut"] == "revision_recommandee" else "✓ "
    print(f"\n{statut_emoji} Obsolescence : {obs['message']}")

    # df_ref est déjà filtré sur les colonnes numériques (select_dtypes dans _charger_donnees_reference)
    cols_communs = [c for c in df_ref.columns if c in df_courant.columns]
    chemin_html = rapport_evidently(df_ref[cols_communs], df_courant[cols_communs], forcer=True)
    print(f"\n✓  Rapport Evidently écrit → {chemin_html.relative_to(config.RACINE)}")
    logger.success("make drift terminé.")


if __name__ == "__main__":
    main()
