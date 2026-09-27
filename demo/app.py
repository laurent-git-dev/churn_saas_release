"""IHM de démonstration Churn Radar — à destination des CSM."""
import json
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

from profils import LABELS, PROFILS

API_URL = "http://localhost:8000"
DATA_DIR = Path(__file__).parent / "data"

SECTEURS = [
    "Technologie", "Finance", "Santé", "Industrie", "Services",
    "Commerce", "Éducation", "Immobilier", "Télécommunications",
    "Médias", "Logistique", "Énergie",
]
PAYS = [
    "France", "Allemagne", "Espagne", "Italie", "Pays-Bas", "Belgique",
    "Portugal", "Suède", "Pologne", "Autriche", "Danemark", "Finlande",
    "Suisse", "Royaume-Uni", "États-Unis", "Canada", "Australie",
    "Japon", "Brésil", "Inde",
]
TAILLES = ["TPE", "PME", "ETI", "GE"]
PLANS = ["Starter", "Pro", "Business", "Enterprise"]

COULEURS_DECISION = {
    "ALERTE_ROUGE": "🔴",
    "SURVEILLANCE": "🟠",
    "OK": "🟢",
}


def _verifier_api() -> bool:
    try:
        httpx.get(f"{API_URL}/health", timeout=3).raise_for_status()
        return True
    except Exception:
        return False


def _lire_valeur(cle: str, defaut):
    vals = st.session_state.get("form_values", {})
    return vals.get(cle, defaut)


st.set_page_config(page_title="Churn Radar — Démo CSM", layout="wide", page_icon="📡")
st.title("📡 Churn Radar")
st.caption("Démo CSM — Détection de résiliation client SaaS B2B")

if not _verifier_api():
    st.error("API non joignable — lancez `make api` dans un terminal puis rechargez cette page.")
    st.stop()

onglet1, onglet2 = st.tabs(["🔍 Analyse client", "📊 Portefeuille"])

# ─────────────────────────────────────────────────────────────────────────────
# Onglet 1 — Analyse d'un compte unique
# ─────────────────────────────────────────────────────────────────────────────
with onglet1:
    st.subheader("Profils pré-remplis")
    col_sain, col_surv, col_alerte = st.columns(3)
    for col, cle in zip([col_sain, col_surv, col_alerte], ["sain", "surveillance", "alerte"]):
        with col:
            if st.button(LABELS[cle], width="stretch"):
                st.session_state["form_values"] = PROFILS[cle].copy()

    st.divider()
    st.subheader("Données du compte")

    with st.expander("📈 Usage", expanded=True):
        c1, c2, c3 = st.columns(3)
        anciennete = c1.number_input(
            "Ancienneté (mois)", 0, 240, _lire_valeur("anciennete_mois", 12)
        )
        sieges = c2.number_input(
            "Sièges souscrits", 1, 10000, _lire_valeur("sieges_souscrits", 10)
        )
        actifs = c3.number_input(
            "Utilisateurs actifs", 0, 10000, _lire_valeur("utilisateurs_actifs", 5)
        )
        c4, c5, c6 = st.columns(3)
        adoption = c4.number_input(
            "Taux d'adoption (%)", 0.0, 100.0, float(_lire_valeur("taux_adoption_pct", 50.0))
        )
        connexions = c5.number_input(
            "Connexions (30 j)", 0, 10000, _lire_valeur("connexions_30j", 20)
        )
        integrations = c6.number_input(
            "Intégrations actives", 0, 100, _lire_valeur("nb_integrations", 2)
        )
        c7, c8, c9 = st.columns(3)
        fonc_total = c7.number_input(
            "Fonctionnalités total", 0, 500, _lire_valeur("fonctionnalites_total", 20)
        )
        fonc_utilisees = c8.number_input(
            "Fonctionnalités utilisées", 0, 500, _lire_valeur("fonctionnalites_utilisees", 10)
        )
        derniere_co = c9.number_input(
            "Dernière connexion (jours)", 0, 3650, _lire_valeur("derniere_connexion_jours", 5)
        )
        heures_inconnu = st.checkbox(
            "Heures d'usage inconnues", value=(_lire_valeur("heures_usage_30j", 0.0) is None)
        )
        heures = None
        if not heures_inconnu:
            heures = st.number_input(
                "Heures d'usage (30 j)", 0.0, 10000.0,
                float(_lire_valeur("heures_usage_30j", 0.0) or 0.0),
            )

    with st.expander("🎧 Support"):
        c1, c2, c3 = st.columns(3)
        tickets = c1.number_input(
            "Tickets support (90 j)", 0, 1000, _lire_valeur("tickets_support_90j", 0)
        )
        delai_inconnu = c2.checkbox(
            "Délai réponse inconnu",
            value=(_lire_valeur("delai_reponse_support_h", None) is None),
        )
        delai = None
        if not delai_inconnu:
            delai = c2.number_input(
                "Délai réponse support (h)", 0.0, 8760.0,
                float(_lire_valeur("delai_reponse_support_h", 0.0) or 0.0),
            )
        csat_inconnu = c3.checkbox(
            "CSAT inconnu", value=(_lire_valeur("csat", None) is None)
        )
        csat = None
        if not csat_inconnu:
            csat = c3.number_input(
                "Score CSAT (0–10)", 0.0, 10.0,
                float(_lire_valeur("csat", 7.0) or 7.0),
            )

    with st.expander("💳 Facturation"):
        c1, c2 = st.columns(2)
        retards = c1.number_input(
            "Retards de paiement (12 m)", 0, 120, _lire_valeur("retards_paiement_12m", 0)
        )
        mrr = c2.number_input(
            "Revenu mensuel récurrent (€)", 0.0, 1_000_000.0,
            float(_lire_valeur("revenu_mensuel_recurrent_eur", 1000.0)),
        )

    with st.expander("🏢 Profil client"):
        c1, c2, c3, c4 = st.columns(4)
        secteur_val = _lire_valeur("secteur", "Technologie")
        secteur = c1.selectbox(
            "Secteur", SECTEURS,
            index=SECTEURS.index(secteur_val) if secteur_val in SECTEURS else 0,
        )
        pays_val = _lire_valeur("pays", "France")
        pays = c2.selectbox(
            "Pays", PAYS, index=PAYS.index(pays_val) if pays_val in PAYS else 0
        )
        taille_val = _lire_valeur("taille_entreprise", "PME")
        taille = c3.selectbox(
            "Taille", TAILLES,
            index=TAILLES.index(taille_val) if taille_val in TAILLES else 1,
        )
        plan_val = _lire_valeur("plan", "Starter")
        plan = c4.selectbox(
            "Plan", PLANS, index=PLANS.index(plan_val) if plan_val in PLANS else 0
        )

    if st.button("🔍 Analyser ce compte", type="primary", width="stretch"):
        payload = {
            "anciennete_mois": anciennete,
            "sieges_souscrits": sieges,
            "utilisateurs_actifs": actifs,
            "taux_adoption_pct": adoption,
            "connexions_30j": connexions,
            "heures_usage_30j": heures,
            "fonctionnalites_total": fonc_total,
            "fonctionnalites_utilisees": fonc_utilisees,
            "nb_integrations": integrations,
            "derniere_connexion_jours": derniere_co,
            "tickets_support_90j": tickets,
            "delai_reponse_support_h": delai,
            "csat": csat,
            "retards_paiement_12m": retards,
            "revenu_mensuel_recurrent_eur": mrr,
            "secteur": secteur,
            "pays": pays,
            "taille_entreprise": taille,
            "plan": plan,
        }
        with st.spinner("Analyse en cours…"):
            rep = httpx.post(f"{API_URL}/predict", json=payload, timeout=10)

        if rep.status_code == 200:
            res = rep.json()
            proba = res["probabilite_churn"]
            valeur = res["valeur_a_risque_eur"]
            decision = res["decision"]
            facteurs = res["facteurs_principaux"]

            st.divider()
            col_proba, col_valeur, col_facteurs = st.columns([1, 1, 2])

            with col_proba:
                st.metric("Probabilité de churn", f"{proba:.1%}")
                icone = COULEURS_DECISION.get(decision, "")
                if decision == "ALERTE_ROUGE":
                    st.error(f"{icone} **{decision}**")
                elif decision == "SURVEILLANCE":
                    st.warning(f"{icone} **{decision}**")
                else:
                    st.success(f"{icone} **{decision}**")

            with col_valeur:
                # L'API renvoie null si le MRR est absent : on n'affiche pas un montant imputé
                if valeur is None:
                    st.metric("Valeur à risque", "n/d")
                    st.caption(res.get("motif_valeur_a_risque") or "MRR absent de la demande.")
                else:
                    st.metric("Valeur à risque", f"{valeur:,.0f} €")

            with col_facteurs:
                st.subheader("Facteurs de risque")
                if facteurs:
                    for f in facteurs:
                        st.markdown(f"• {f}")
                else:
                    st.markdown("Aucun facteur de risque majeur détecté.")

            imputes = res.get("champs_imputes") or []
            if imputes:
                st.info(
                    "Champs absents reconstruits par le pipeline (imputation) : "
                    + ", ".join(f"`{c}`" for c in imputes)
                    + " — le score reste exploitable, mais repose en partie sur des "
                    "valeurs estimées."
                )
        else:
            st.error(f"Erreur API ({rep.status_code}) : {rep.text}")

# ─────────────────────────────────────────────────────────────────────────────
# Onglet 2 — Analyse du portefeuille (batch)
# ─────────────────────────────────────────────────────────────────────────────
with onglet2:
    st.subheader("Source des données")
    source = st.radio(
        "Source",
        ["Charger les comptes de démo (10 comptes)", "Importer un CSV"],
        horizontal=True,
        label_visibility="collapsed",
    )

    df_portefeuille: pd.DataFrame | None = None

    if source == "Charger les comptes de démo (10 comptes)":
        df_portefeuille = pd.read_csv(DATA_DIR / "demo_clients.csv")
    else:
        fichier = st.file_uploader(
            "Fichier CSV (colonnes identiques à demo_clients.csv)", type=["csv"]
        )
        if fichier is not None:
            df_portefeuille = pd.read_csv(fichier)

    if df_portefeuille is not None:
        st.caption(f"{len(df_portefeuille)} comptes chargés.")

        if st.button("📊 Analyser le portefeuille", type="primary", width="stretch"):
            client_ids = (
                df_portefeuille["client_id"].tolist()
                if "client_id" in df_portefeuille.columns
                else [f"#{i+1}" for i in range(len(df_portefeuille))]
            )
            cols_modele = [c for c in df_portefeuille.columns if c != "client_id"]
            # to_json convertit NaN → null, json.loads restitue None — fiable pour colonnes nullable
            payload_batch = json.loads(
                df_portefeuille[cols_modele].to_json(orient="records", force_ascii=False)
            )

            with st.spinner(f"Analyse de {len(payload_batch)} comptes…"):
                rep = httpx.post(
                    f"{API_URL}/predict-batch",
                    json=payload_batch,
                    timeout=60,
                )

            if rep.status_code == 200:
                res = rep.json()
                predictions = res["predictions"]

                col_r, col_s, col_ok = st.columns(3)
                col_r.metric("🔴 ALERTE ROUGE", res["nb_alertes_rouges"])
                col_s.metric("🟠 SURVEILLANCE", res["nb_surveillances"])
                col_ok.metric("🟢 OK", res["nb_ok"])

                df_res = pd.DataFrame(
                    [
                        {
                            "client_id": cid,
                            "probabilite_churn": p["probabilite_churn"],
                            "decision": p["decision"],
                            "valeur_a_risque_eur": p["valeur_a_risque_eur"],
                            "champs_imputes": ", ".join(p.get("champs_imputes") or []),
                        }
                        for cid, p in zip(client_ids, predictions)
                    ]
                ).sort_values("probabilite_churn", ascending=False)

                nb_incomplets = int((df_res["champs_imputes"] != "").sum())
                if nb_incomplets:
                    st.caption(
                        f"{nb_incomplets} compte(s) sur {len(df_res)} comportent des champs "
                        "absents, reconstruits par le pipeline (colonne « champs_imputes »). "
                        "La valeur à risque est marquée « n/d » quand le MRR manque."
                    )

                df_res["probabilite_churn"] = df_res["probabilite_churn"].map("{:.1%}".format)
                df_res["valeur_a_risque_eur"] = df_res["valeur_a_risque_eur"].map(
                    lambda v: "n/d" if v is None or pd.isna(v) else f"{v:,.0f} €"
                )

                st.dataframe(df_res, width="stretch", hide_index=True)
            else:
                st.error(f"Erreur API ({rep.status_code}) : {rep.text}")
