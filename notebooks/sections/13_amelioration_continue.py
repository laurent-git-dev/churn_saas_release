# %% [markdown]
# ## 13. Amélioration continue
#
# Un modèle mis en production se dégrade sans bruit : les clients changent, les données aussi.
# Cette section outille sa surveillance dans la durée : contrôles automatiques avant toute mise
# en production, indicateurs de dérive, de robustesse et d'âge, plan de réentraînement et
# rythme de revue fixé dès le cadrage (§2.6). Elle traite deux contraintes propres au churn
# (résiliation) B2B : les étiquettes n'arrivent qu'à l'échéance du contrat, et des alertes trop
# fréquentes finissent ignorées.

# %%
from __future__ import annotations

import ast
import datetime
import re
import sys
import warnings

import joblib
import pandas as pd
import yaml
from IPython.display import Markdown, display
from loguru import logger
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from churn_saas import config
from churn_saas.cache import charger_ou_calculer
from churn_saas.features.build import ajouter_features_metier
from churn_saas.format_fr import entier, nombre, pourcentage, scientifique, styler_fr
from churn_saas.monitoring import (
    indicateur_obsolescence,
    ks_test,
    psi,
    rapport_evidently,
    simuler_derive,
    tester_robustesse,
)

sys.path.insert(0, str(config.RACINE))  # rend `monitoring/` importable
from monitoring.simulation_grafana import (  # noqa: E402
    simuler_metriques,
    tracer_tableau_de_bord,
)

# Mêmes variables qu'en §9 : jeu gold (§7) et modèle final (§9)
gold = pd.read_parquet(config.DONNEES_GOLD / "gold_dataset.parquet")
y_complet = gold["churn"].astype(int)
X_complet = ajouter_features_metier(gold.drop(columns=config.COLONNES_INTERDITES, errors="ignore"))
X_complet = X_complet.drop(columns=["churn"], errors="ignore")

chemin_modele_final = config.TABLES / "modele_final.joblib"
modele_final = joblib.load(chemin_modele_final)

# Référence (80 %) = période d'apprentissage ; courant (20 %) = production récente simulée
X_ref, X_courant, y_ref, y_courant = train_test_split(
    X_complet, y_complet, test_size=0.20, stratify=y_complet, random_state=config.RANDOM_SEED
)
logger.info("Référence : {} lignes, courant : {} lignes.", len(X_ref), len(X_courant))

# %% [markdown]
# ---
# ### 13.1 Pistes d'amélioration — analyse coût/bénéfice
#
# On veut savoir où investir le prochain effort. Les gains de PR-AUC (aire sous la courbe
# précision-rappel) ci-dessous sont des **ordres de grandeur supposés**, à confirmer par
# expérience : ils servent à arbitrer les priorités en comité (§13.11), pas à promettre.
#
# | Priorité | Piste | Description | Gain de PR-AUC supposé | Effort (mois-ingé) | Prérequis |
# |---|---|---|---|---|---|
# | 1 | Données d'usage granulaires | Événements par fonctionnalité (clics, exports, appels d'API) plutôt qu'agrégats 30 j | +0,04 à +0,08 | 2 | Instrumentation produit, chargement dans le batch (traitement par lot) nocturne |
# | 2 | Historique sur 3 ans | Le jeu couvre 2 ans ; l'étendre à la durée de conservation de 3 ans (§4.1) capte mieux les cycles de renouvellement | +0,03 à +0,06 | 1 | Archivage ; au-delà de 3 ans, accord du DPO |
# | 3 | Modèle de survie | Prédire **quand** le client résilie, pas seulement **si** : priorise les renouvellements imminents | Sans objet (autre métrique : C-index) | 3 | Dates exactes de résiliation, accord du DPO |
# | 4 | Analyse du texte des tickets | Signaux faibles dans le texte des tickets de support. Les commentaires CSM n'y entrent qu'une fois horodatés : sans date de rédaction, la fuite n'est pas exclue (§6.6) | +0,02 à +0,05 | 4 | GPU, horodatage CRM, base légale RGPD, DPO |
# | 5 | Données externes sur les entreprises clientes | Santé financière et effectif (base Sirene) : un client en difficulté résilie plus souvent | +0,01 à +0,03 | 1 | Numéro SIREN dans le CRM, licence de la source |

# %% [markdown]
# **Ce qu'il faut retenir.** La piste au meilleur rapport impact/effort est l'**instrumentation
# fine du produit** : des événements datés montreraient une baisse d'engagement dès qu'elle
# commence, avant que les agrégats sur 30 jours ne la reflètent. Le modèle de survie répond à
# une autre question (*quand* plutôt que *si*) et demande un outillage distinct.
#
# **Cible secondaire (CLV).** Le §12.10 a montré qu'un modèle additif en euros peut prédire des
# CLV négatives, car la CLV est multiplicative (≈ MRR × durée de vie). Le prochain cycle
# comparera, avec une règle posée *avant* les résultats : régression log-log, gradient boosting
# à perte Gamma (moyenne positive par construction), et modélisation de la durée puis
# CLV = MRR × durée prédite, le tout en validation croisée répétée. « Aucune CLV prédite ≤ 0 »
# devient une cible *a priori*.

# %% [markdown]
# ---
# ### 13.2 Évaluation automatisée dans la chaîne d'intégration continue
#
# On veut qu'aucune régression du modèle n'atteigne la production sans être vue. Deux verrous
# s'enchaînent :
#
# 1. **Test de non-régression** — `tests/test_model_quality_gate.py` entraîne une régression
#    logistique en cross-validation (validation croisée) sur un jeu synthétique et exige
#    PR-AUC ≥ `pr_auc_min`. L'étape « Gate qualité modèle » de `.github/workflows/ci.yml`
#    le lance à chaque push et pull request : s'il échoue, la fusion est bloquée.
# 2. **Gate (règle de promotion) du réentraînement** — `stage_gate()` dans
#    `flows/retraining.py` compare le modèle candidat au champion (modèle en place). Il n'est
#    promu que s'il atteint `pr_auc_min` **et** ne perd pas plus de
#    `tolerance_regression_promotion` face au champion ; sinon le champion reste en place.

# %%
_ci = (config.RACINE / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
_test_gate = (config.RACINE / "tests" / "test_model_quality_gate.py").read_text(encoding="utf-8")
_flow = (config.RACINE / "flows" / "retraining.py").read_text(encoding="utf-8")
controles_ci = {
    "La CI lance les tests lents (`make test-slow`)": "make test-slow" in _ci,
    "Le test de non-régression est marqué `slow`": "pytest.mark.slow" in _test_gate,
    "Le test lit le seuil `pr_auc_min` dans la config": "pr_auc_min" in _test_gate,
    "Le réentraînement passe par `stage_gate`": "def stage_gate" in _flow,
    "Le réentraînement est orchestré par Prefect (`@flow`)": "@flow(" in _flow,
}
display(Markdown("\n".join(f"- {'✅' if ok else '❌'} {c}" for c, ok in controles_ci.items())))

# %% [markdown]
# **Ce qu'il faut retenir.** Les deux verrous sont en place et lisent leurs seuils (§13.8) dans
# la même configuration. Le premier protège le **code** (préparation, variables) à chaque modification ;
# le second protège la **production**, sur le vrai modèle réentraîné avec les hyperparamètres
# du champion.

# %% [markdown]
# ---
# ### 13.3 Dérive des entrées — PSI et test de Kolmogorov-Smirnov
#
# On veut savoir si les clients d'aujourd'hui ressemblent encore à ceux de l'apprentissage :
# c'est le data drift (dérive des données). Il se mesure **sans attendre les étiquettes**, ce
# qui en fait le signal précoce de la surveillance. Deux mesures complémentaires :
#
# - le **PSI** (indice de stabilité de population) compare la répartition d'une variable par
#   tranches : < 0,10 stable, 0,10 à 0,20 à surveiller, ≥ 0,20 dérive significative
#   (conventions du risque crédit, Siddiqi 2006) ;
# - le **test de Kolmogorov-Smirnov** mesure l'écart maximal entre deux distributions et dit
#   s'il peut être dû au hasard (p < 0,05 : dérive confirmée).
#
# `simuler_derive()` propose quatre scénarios ; on présente `adoption_chute`, le plus réaliste
# pour un SaaS : −20 points d'adoption en moyenne, connexions réduites de 30 à 60 %.

# %%
features_surveillees = [
    f
    for f in ["taux_adoption_pct", "connexions_30j", "anciennete_mois", "tickets_support_90j"]
    if f in X_ref.columns
]
X_derive = simuler_derive(X_courant, "adoption_chute", intensite=1.0)

lignes_psi = []
for col in features_surveillees:
    res_psi, res_ks = psi(X_ref[col], X_derive[col]), ks_test(X_ref[col], X_derive[col])
    lignes_psi.append(
        {
            "Variable": col,
            "PSI": res_psi["psi"],
            "Interprétation PSI": res_psi["interpretation"],
            "KS D": res_ks["statistique"],
            "KS p": res_ks["p_value"],
            "Dérive KS": "Oui" if res_ks["derive_detectee"] else "Non",
        }
    )
df_psi = pd.DataFrame(lignes_psi)
display(
    styler_fr(
        df_psi,
        {"PSI": lambda v: nombre(v, 4), "KS D": lambda v: nombre(v, 4), "KS p": scientifique},
    ).hide(axis="index")
)


def _liste(variables: pd.Series) -> str:
    return ", ".join(f"`{v}`" for v in variables) if len(variables) else "aucune"


_interp = df_psi.set_index("Variable")["Interprétation PSI"]
display(
    Markdown(
        f"- PSI ≥ 0,20 : {_liste(_interp[_interp == 'derive_significative'].index.to_series())}\n"
        f"- 0,10 ≤ PSI < 0,20 : {_liste(_interp[_interp == 'attention'].index.to_series())}\n"
        f"- Dérive confirmée par KS : {_liste(df_psi.loc[df_psi['Dérive KS'] == 'Oui', 'Variable'])}"
    )
)

# %% [markdown]
# **Ce qu'il faut retenir.** Le scénario ne touche que `taux_adoption_pct` et `connexions_30j` ;
# les deux autres variables sont des témoins. La liste ci-dessus dit lesquelles franchissent
# les seuils : c'est ce signal qui convoque le comité *avant* que la performance ne soit
# mesurable (§13.7). Les seuils PSI sont des conventions, révisables si le métier le justifie.

# %% [markdown]
# ---
# ### 13.4 Rapport Evidently — dérive sur toutes les variables
#
# Le PSI suit quatre variables choisies ; on veut aussi une vue d'ensemble. Evidently teste
# chaque variable numérique et écrit un rapport HTML autonome (`reports/drift_report_demo.html`)
# : dérive par variable et résumé de qualité (manquants, types). `rapport_evidently()` ne le
# recalcule que s'il est absent. En exploitation, la chaîne Prefect `monitoring/drift_report.py`
# le produit chaque lundi sur la fenêtre écoulée (déploiement planifié, §13.8).

# %%
cols_communes = [
    c
    for c in X_ref.columns
    if c in X_derive.columns
    and pd.api.types.is_numeric_dtype(X_ref[c])
    and not pd.api.types.is_bool_dtype(X_ref[c])
]
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    chemin_html = rapport_evidently(
        X_ref[cols_communes].astype(float),
        X_derive[cols_communes].astype(float),
        chemin_sortie=config.RACINE / "reports" / "drift_report_demo.html",
    )
display(
    Markdown(
        f"Rapport écrit : `{chemin_html.relative_to(config.RACINE)}` "
        f"({entier(len(cols_communes))} variables testées)."
    )
)

# %% [markdown]
# **Ce qu'il faut retenir.** Au-delà de 1 000 lignes de référence, Evidently retient la
# distance de Wasserstein normalisée pour les variables continues et la divergence de
# Jensen-Shannon pour celles à peu de valeurs ; il conclut à une dérive globale si plus de la
# moitié des variables dérivent. Les rapports hebdomadaires du trimestre sont les pièces
# jointes du comité de revue (§13.11).

# %% [markdown]
# ---
# ### 13.5 Robustesse — dégradation sous perturbation
#
# On veut savoir si le modèle tient quand les données arrivent abîmées : erreurs de saisie,
# cellules vides lors de la synchronisation CRM. Deux perturbations croissantes : un **bruit
# gaussien** (σ × écart-type de chaque variable) et des **valeurs manquantes** injectées.
#
# **Protocole.** Le modèle final a été fitté (appris) sur tout le jeu : l'évaluer sur
# `X_courant` mesurerait sa mémoire. On évalue donc un **clone** (même famille, mêmes
# hyperparamètres) réentraîné sur `X_ref` seul, sur les 20 % qu'il n'a jamais vus. Une perte
# relative de PR-AUC de plus de 10 % est jugée significative.


# %%
def _calculer_robustesse() -> pd.DataFrame:
    modele_clone = clone(modele_final).fit(X_ref, y_ref)
    return tester_robustesse(
        modele_clone,
        X_courant,
        y_courant,
        niveaux_bruit=[0.0, 0.1, 0.25, 0.5, 1.0],
        taux_manquants=[0.0, 0.05, 0.10, 0.20, 0.30],
    )


df_robustesse, _ = charger_ou_calculer("robustesse_13_holdout.joblib", _calculer_robustesse)
SEUIL_DEGRADATION_PCT = 10

tableau_rob = df_robustesse.pivot(
    index="niveau", columns="type_perturbation", values="degradation_relative_pct"
).rename(columns={"bruit_gaussien": "Bruit gaussien (%)", "valeurs_manquantes": "Manquants (%)"})
tableau_rob.index.name = "Niveau (σ ou taux de NaN)"
display(styler_fr(tableau_rob, precision=1))

lignes_conclusion = []
for type_pert, libelle in [("bruit_gaussien", "Bruit"), ("valeurs_manquantes", "Manquants")]:
    sous = df_robustesse[df_robustesse["type_perturbation"] == type_pert]
    depasse = sous[sous["degradation_relative_pct"] > SEUIL_DEGRADATION_PCT]
    franchi = (
        f"seuil franchi dès le niveau {nombre(depasse['niveau'].iloc[0], 2)}"
        if len(depasse)
        else "seuil jamais franchi"
    )
    lignes_conclusion.append(
        f"- {libelle} : perte maximale {nombre(sous['degradation_relative_pct'].max(), 1)} %, "
        f"{franchi}."
    )
pr_auc_clone = df_robustesse.loc[df_robustesse["niveau"] == 0.0, "pr_auc"].iloc[0]
display(
    Markdown(
        f"PR-AUC du clone sans perturbation : **{nombre(pr_auc_clone, 4)}** "
        f"(cible *a priori* : {nombre(config.CIBLES_PERFORMANCE['pr_auc_min'], 2)}).\n\n"
        + "\n".join(lignes_conclusion)
    )
)

# %% [markdown]
# **Ce qu'il faut retenir.** Le tableau donne la perte relative de PR-AUC par niveau de
# perturbation. Grâce à l'imputation (remplacement des manquants) intégrée au pipeline (chaîne
# de traitement), le modèle répond toujours ; la conclusion chiffre ce que cela coûte. Elle
# fixe un engagement de qualité des données en amont : le taux de manquants toléré est le plus
# haut niveau testé qui reste sous 10 % de perte.

# %% [markdown]
# ---
# ### 13.6 Indicateur d'obsolescence
#
# Même sans dérive visible, un modèle vieillit. Le seuil de 180 jours vaut un demi-cycle de
# renouvellement annuel : au-delà, le modèle n'a pas vu une partie des comportements récents.

# %%
# Date d'entraînement = date de production de `modele_final.joblib` (celle affichée en §9)
date_train = datetime.datetime.fromtimestamp(chemin_modele_final.stat().st_mtime)
obs = indicateur_obsolescence(date_train, date_courante=datetime.date.today())
display(
    Markdown(
        f"Entraîné le {date_train.date().isoformat()} : **{entier(obs['age_jours'])} jours** "
        f"(seuil {entier(obs['seuil_jours'])}), statut `{obs['statut']}`, revue recommandée le "
        f"{obs['date_revue_recommandee'].isoformat()}.\n\n> {obs['message']}"
    )
)

# %% [markdown]
# **Ce qu'il faut retenir.** L'âge du modèle est un filet de sécurité qui complète le PSI : il
# déclenche une revue même quand aucune dérive n'est mesurée. Le seuil est documenté dans
# `monitoring/drift.py` et révisable en comité trimestriel.

# %% [markdown]
# ---
# ### 13.7 Le délai d'obtention des étiquettes
#
# Un client résilie à l'échéance de son contrat, et les contrats sont annuels (§2.1).
# L'étiquette `churn = 1` n'est donc connue que de quelques jours à 12 mois après la
# prédiction. On ne peut pas mesurer la performance en temps réel : le modèle prédit
# aujourd'hui, on saura plus tard s'il avait raison.
#
# | Signal | Disponibilité | Usage |
# |---|---|---|
# | Qualité des entrées reçues (champs absents) | Immédiate | Détecter une rupture d'intégration |
# | Dérive des entrées (PSI, KS) | Immédiate | Alerte précoce, sans étiquette |
# | Dérive de sortie (part `ALERTE_ROUGE`) | Immédiate | Alerte précoce, sans étiquette |
# | Performance réelle (PR-AUC sur étiquettes) | Décalée de 1 à 12 mois | Évaluation définitive |
#
# **Parades.**
#
# 1. **Signaux sans étiquette** (§13.3, §13.9) : ils ne mesurent pas la performance, mais
#    signalent qu'elle *risque* de baisser et déclenchent une revue anticipée.
# 2. **Évaluation par cohortes** : à chaque vague de renouvellements, on compare le taux de
#    départ réel des comptes signalés et non signalés. C'est la mesure de vérité.
# 3. **Groupe témoin non traité (~10 %, §4.5)** : une action réussie fait passer un vrai
#    churner pour une fausse alerte. Seuls les comptes tirés au sort et laissés sans action
#    donnent des étiquettes non biaisées ; la comparaison témoin/contactés mesure en plus
#    l'effet des actions.
#
# **Limite.** La cible fournie n'a pas d'horizon défini. Le prochain cycle le fixera (par
# exemple « résiliation au renouvellement dans les 90 jours », la fenêtre d'action de §2), ce
# qui bornera aussi le délai des étiquettes.

# %% [markdown]
# ---
# ### 13.8 Plan de réentraînement — déclencheurs et retour arrière
#
# On veut savoir, pour chaque signal, qui fait quoi et quand. Les seuils sont ceux du cadrage
# (§2.6) et de `config.CIBLES_PERFORMANCE`.
#
# | Indicateur | Vérification | Action | Responsable |
# |---|---|---|---|
# | Part `ALERTE_ROUGE` écartée de plus de 10 points de la référence (`ecart_alerte_rouge_rollback_pts`, RUNBOOK §4.3) | Chaque nuit (synthèse du batch, dashboard) | Après une promotion récente : **retour au champion précédent** ; sinon comité ad hoc | Data Scientist |
# | 0,10 ≤ PSI < 0,20 sur une variable surveillée | Hebdomadaire (déploiement Prefect du lundi) | Noté pour le comité trimestriel | Data Scientist |
# | PSI ≥ 0,20 sur au moins une variable surveillée | Hebdomadaire | Comité ad hoc : analyse causale et décision sous 72 h (réentraîner, maintenir ou geler) | Data Scientist + CS Lead |
# | PR-AUC < `pr_auc_min` sur une cohorte de renouvellements | À chaque vague observée (§13.7) | Comité ad hoc : réentraînement et audit des données | Data Scientist + CS Lead |
# | PR-AUC < `pr_auc_critique` sur une cohorte | Idem | Comité ad hoc et comparaison au champion précédent sur la même cohorte : retour arrière s'il fait mieux | Data Scientist + CS Lead |
# | Âge du modèle ≥ 180 jours (§13.6) | À chaque comité trimestriel (échéance connue dès l'entraînement) | Réentraîner, maintenir ou geler | Comité |
# | Calendrier trimestriel | Après le comité de mars, juin, septembre, décembre | Réentraînement (déploiement Prefect lancé à la main) | Data Scientist, sur décision du comité |
# | Gate de promotion (`stage_gate`) | Après chaque réentraînement | Promotion si elle passe ; sinon candidat rejeté | Automatique |

# %%
_seuils = {
    k: config.CIBLES_PERFORMANCE[k]
    for k in [
        "pr_auc_min",
        "pr_auc_critique",
        "tolerance_regression_promotion",
        "ecart_alerte_rouge_rollback_pts",
    ]
}
_deploiements = yaml.safe_load((config.RACINE / "prefect.yaml").read_text(encoding="utf-8"))
_arbre = ast.parse(_flow)
etapes = [
    f"`{n.name}` — {(ast.get_docstring(n) or 'voir code').splitlines()[0]}"
    for n in _arbre.body
    if isinstance(n, ast.FunctionDef) and n.name.startswith("stage_")
]
display(
    Markdown(
        "Seuils en vigueur : "
        + ", ".join(f"`{k}` = {nombre(v, 2)}" for k, v in _seuils.items())
        + ".\n\nÉtapes de `flows/retraining.py`, lues dans le code :\n\n"
        + "\n".join(f"{i}. {e}" for i, e in enumerate(etapes, 1))
        + "\n\nDéploiements Prefect déclarés dans `prefect.yaml` :\n\n"
        + "\n".join(
            f"- `{d['name']}` → `{d['entrypoint']}`, "
            + (f"cron `{d['schedule']['cron']}`" if d.get("schedule") else "**sans planification**")
            for d in _deploiements["deployments"]
        )
    )
)

# %% [markdown]
# **Ce qu'il faut retenir.** Aucun signal ne réentraîne le modèle seul : une dérive peut venir
# d'une panne de données, qu'un réentraînement apprendrait au lieu de la corriger. Les rôles
# sont séparés : le comité décide de **lancer** un réentraînement, la gate décide de la
# **promotion**, et personne ne la contourne. Seul le retour au champion précédent est
# immédiat, car il restaure un état déjà validé : chaque promotion archive l'ancien modèle
# (`archive_<horodatage>_best_model.pkl`), qu'il suffit de recopier avant de redémarrer l'API
# (`docs/RUNBOOK.md` §4.1).
#
# **Orchestration.** Prefect 3, l'orchestrateur du scoring (calcul des scores) nocturne (§10),
# pilote aussi la surveillance et le réentraînement : chaque étape est une tâche Prefect
# (`@task`), relancée automatiquement en cas d'échec passager (lecture des sources,
# entraînement) avec une attente croissante ; la promotion, elle, n'est jamais relancée seule.
# Les trois déploiements traduisent la séparation des rôles : le scoring et le rapport de
# dérive sont planifiés, le réentraînement ne l'est **pas**. Le comité le lance à la main
# depuis l'interface Prefect, qui garde l'état de chaque étape et l'historique des exécutions.
# Une étape déjà faite le jour même n'est pas refaite (marqueur daté, sauf `--forcer`).

# %% [markdown]
# ---
# ### 13.9 Surveillance du service — alertes, compteurs et fatigue d'alerte
#
# Le monitoring (surveillance) en production repose sur trois services de
# `docker-compose.yml` : l'API expose ses métriques sur `/metrics`, Prometheus les collecte
# toutes les 30 s et évalue les règles d'alerte de `monitoring/alerts.yml`, Grafana les
# affiche (`monitoring/grafana/dashboard.json` : disponibilité, latence, sorties du modèle,
# qualité des entrées).
#
# **Fatigue d'alerte.** Une alerte qui sonne trop souvent finit ignorée. Les deux règles sont
# donc filtrées deux fois : une fenêtre de calcul qui lisse les pics, puis un délai de
# persistance (`for`).
#
# | Règle | Condition | Justification |
# |---|---|---|
# | `APIIndisponible` | API muette (`up == 0`) | Critique : délai court, l'astreinte doit agir tout de suite |
# | `LatenceElevee` | Latence **moyenne** de `/predict` sur 5 min > 200 ms | Garde-fou de l'objectif de latence de §8 ; la moyenne est stable, le p95 et le p99 restent suivis sur le dashboard (tableau de bord) |
#
# Contrepartie assumée : une moyenne sous 200 ms ne garantit pas le p95, qui se lit en revue
# hebdomadaire sans alerte automatique.

# %%
_alertes = (config.RACINE / "monitoring" / "alerts.yml").read_text(encoding="utf-8")
regles = re.findall(r"- alert: (\w+).*?\n\s+for: (\w+)", _alertes, flags=re.DOTALL)
_api = (config.RACINE / "src" / "churn_saas" / "api" / "main.py").read_text(encoding="utf-8")
compteurs = re.findall(r'Counter\(\s*"(\w+)"', _api)
display(
    Markdown(
        "Règles d'alerte et délai de persistance : "
        + ", ".join(f"`{nom}` ({delai})" for nom, delai in regles)
        + ".\n\nCompteurs métier exposés sur `/metrics` : "
        + ", ".join(f"`{c}`" for c in compteurs)
        + "."
    )
)

# %% [markdown]
# **Ce qu'il faut retenir.** Les deux compteurs métier surveillent le modèle, pas seulement le
# service :
#
# - `churn_predictions_total{decision}` suit la part de chaque décision : si `ALERTE_ROUGE`
#   s'envole, les **sorties** dérivent, même sans dérive visible des entrées ;
# - `churn_champs_imputes_total{champ}` compte les champs absents des demandes : un champ que
#   le CRM cesse de transmettre se voit ici **avant** que les scores ne bougent.
#
# Avec le PSI (§13.3) et les cohortes (§13.7), la surveillance a **quatre niveaux**, du plus
# précoce au plus tardif : qualité des entrées reçues, dérive des entrées, dérive des sorties,
# performance réelle. L'ordre compte : une donnée manquante fausse les variables avant de
# déplacer les scores.
#
# **Tableau de bord simulé.** Sans stack Prometheus en marche, `monitoring/simulation_grafana.py`
# génère deux heures de métriques synthétiques (pas de 30 s) aux unités du dashboard, avec un
# incident par niveau : une panne, un pic de latence, une dérive de la part `ALERTE_ROUGE`
# partie du dernier batch réel (§10), et un champ `secteur` qui cesse d'être transmis.

# %%
df_metriques = simuler_metriques()
fig = tracer_tableau_de_bord(df_metriques)
display(
    Markdown(
        f"Part `ALERTE_ROUGE` : {pourcentage(df_metriques.attrs['taux_alerte_ref'], 0)} → "
        f"{pourcentage(df_metriques.attrs['taux_alerte_fin'], 0)} (tolérance du runbook : "
        f"±{entier(df_metriques.attrs['tolerance_pts'])} points)."
    )
)

# %% [markdown]
# **Ce qu'il faut retenir.** Chaque incident n'est visible que sur son panel. La panne et le pic
# de latence, plus longs que leur délai `for`, déclenchent leur alerte, alors qu'un pic de
# quelques secondes ne le ferait pas. La dérive de sortie dépasse la tolérance du runbook et
# déclenche le retour arrière (§13.8). La rupture CRM ne produit **aucune erreur** (le champ
# est facultatif, la réponse reste un succès) et ne déplace pas tout de suite les scores, car
# le pipeline impute : seul le compteur de champs absents la révèle à temps.

# %% [markdown]
# ---
# ### 13.10 Versioning et gouvernance — les quatre dimensions
#
# Pour expliquer une prédiction après coup, il faut retrouver le code, les données, le modèle
# et la configuration qui l'ont produite.
#
# | Dimension | Outil | Commande | Responsable | Rétention |
# |---|---|---|---|---|
# | Code | Git + tags `vMAJEUR.MINEUR.CORRECTIF` | `git tag v1.2.0 && git push --tags` | Data Scientist | Illimitée |
# | Données | DVC (`data/gold/gold_dataset.parquet.dvc`), stockage objet cible | `dvc add … && dvc push` | Data Scientist | 3 ans, puis suppression (§4.1) |
# | Modèle | Registry (registre de modèles) MLflow : une version par promotion, alias `@production` | `stage_promote` | Data Scientist, validation CS Lead | Toutes les versions et archives, sans purge |
# | Configuration | `src/churn_saas/config.py`, seule source de vérité | Pull request revue | Data Scientist, revue par un pair | Historique git |

# %% [markdown]
# **Ce qu'il faut retenir.** À partir du `run_id` MLflow, on retrouve la version du code, celle
# des données (empreinte DVC) et les hyperparamètres. Le registre utilise des **alias** plutôt
# que les anciens statuts `Staging`/`Production`, dépréciés : revenir en arrière revient à
# replacer `@production` sur une version antérieure. Le comité décide de lancer un
# réentraînement ; seule la gate déplace l'alias. On évite ainsi un réentraînement sans
# discussion comme une mise en production d'un modèle qui régresse.

# %% [markdown]
# ---
# ### 13.11 Périodicité de revue — comité trimestriel
#
# Un rythme de revue fixé après coup s'adapterait aux résultats : il a donc été décidé au
# cadrage (§2.6) et il est repris ici à l'identique. Revue trimestrielle (mars, juin,
# septembre, décembre), plus un comité ad hoc dès qu'un seuil de §13.8 est franchi, avec
# décision sous 72 h. Membres : CS Lead (cohérence avec le terrain), Data Scientist
# (métriques), DPO (conformité RGPD du trimestre) ; l'équipe Ops est invitée pour
# l'exploitation, le commanditaire quand un investissement est à arbitrer.
#
# **Ordre du jour (60 min).** Alertes du trimestre (10) ; rapports de dérive hebdomadaires
# (10) ; performance sur les cohortes de renouvellement (15) ; **valeur réalisée** (10) ; âge
# du modèle et décision réentraîner, maintenir ou geler (10) ; pistes de §13.1 (5).
#
# **Valeur réalisée.** Le gain annoncé en §12.12 est une **simulation** (portefeuille figé,
# taux de succès des actions repris d'un rapport sectoriel). Le comité le remplace peu à peu
# par une mesure :
#
# | Élément | Définition |
# |---|---|
# | Mesure | Par vague de renouvellements : taux de départ des comptes **contactés** contre celui du **groupe témoin** (§4.5). L'écart, valorisé en MRR × marge × horizon, moins le coût des actions, donne le gain réel |
# | Taux de succès mesuré | (départs témoin − départs contactés) / départs témoin : remplace l'hypothèse de `config.HYPOTHESES_ECONOMIQUES` |
# | Première lecture | À la première vague dont les résiliations sont observées (1 à 12 mois, §13.7) |
# | Règle de décision | Sous la borne basse de la fourchette de §12.12 deux comités de suite : mise à jour du taux de succès, recalcul du bilan, revue de la règle d'action avec le CS Lead. Au-dessus de la borne haute : vérifier d'abord le tirage du groupe témoin |
#
# Tant que cette mesure n'existe pas, les montants en euros restent des ordres de grandeur.

# %% [markdown]
# > ### 📋 Journal de bord — Amélioration continue
# >
# > **Décisions retenues** — Deux verrous : test de non-régression dans la CI et gate de
# > promotion du réentraînement, sur les mêmes seuils de configuration. Prefect, déjà en place
# > pour le scoring, orchestre le rapport de dérive (planifié chaque lundi) et le réentraînement
# > (déclenché à la main par le comité). Robustesse mesurée sur un clone réentraîné sur 80 % des
# > données, le modèle final ayant tout vu. Surveillance à quatre niveaux, dont la qualité des
# > entrées reçues : un champ que le CRM ne transmet plus ne produit ni erreur ni dérive
# > immédiate du score. Deux alertes filtrées contre la fatigue d'alerte. Délai des étiquettes
# > traité par signaux sans étiquette, cohortes et groupe témoin. Valeur réalisée mesurée en
# > comité contre le gain simulé.
# >
# > **Alternatives écartées** — Réentraînement automatique sur alerte PSI ou planifié sans
# > comité : une panne de données se corrige, elle ne s'apprend pas. Airflow ou Prefect Cloud :
# > infrastructure non justifiée pour un pilote, le serveur Prefect auto-hébergé suffit. Alerte
# > sur le p99 : trop instable à faible volume. Alerte sur les champs absents : pas encore de
# > référence de production, seuil arbitraire. Quatre figures de monitoring simulé, réduites à
# > une seule, le code allant dans `monitoring/simulation_grafana.py`.
# >
# > **Difficultés rencontrées** — Le réentraînement à relances codées à la main a cédé la place à
# > Prefect, dont le cache, incapable d'identifier de façon fiable un pipeline scikit-learn, est
# > désactivé au profit d'un marqueur daté (`--forcer`). Surveillance et dérive démontrées sur
# > données simulées : seuils d'alerte jamais confrontés à un trafic réel.
# >
# > **Impact sur la suite** — Le dispositif de suivi est complet ; la conclusion (§14) s'appuie
# > dessus pour juger l'exploitabilité du projet.
