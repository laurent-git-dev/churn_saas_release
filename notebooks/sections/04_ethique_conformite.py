# %% [markdown]
# ## 4. Éthique, société et conformité (C2)
#
# Cette section analyse les obligations légales et les risques éthiques du modèle de prédiction
# de churn SaaS B2B. Elle couvre : conformité RGPD, classification AI Act, chartes européennes
# et françaises, biais mesurés par sous-groupe, dilemmes d'usage identifiés. Elle produit les
# artefacts exigés par C2 : registre des risques (`docs/RISK_REGISTER.md`), note de synthèse
# au commanditaire, fiche de revue DPO/juriste (simulée et assumée comme telle).

# %% [markdown]
# ---
# ### 4.1 RGPD — Analyse de conformité
#
# **Contexte B2B : le RGPD s'applique de façon marginale.**
# Les *personnes concernées* au sens de l'article 4 du RGPD sont des **personnes physiques
# identifiées ou identifiables**. Dans ce projet, les entités traitées sont des **entreprises
# clientes** (personnes morales). Le RGPD ne régit pas les données relatives aux personnes
# morales en tant que telles.
#
# **L'article 22 : formulation correcte.**
# L'article 22 n'édicte pas une interdiction générale de la décision automatisée. Il ouvre à
# la personne concernée le *droit de ne pas faire l'objet d'une décision fondée exclusivement
# sur un traitement automatisé* produisant des effets juridiques la concernant ou l'affectant
# de manière significative — sous réserve des exceptions (consentement explicite, nécessité
# contractuelle, autorisation légale). Dans notre cas B2B, cet article est **très probablement
# hors champ** : le sujet du traitement est une entreprise, non une personne physique.
#
# **Deux points de contact réels avec le RGPD subsistent néanmoins :**
#
# 1. **`commentaire_csm`** : champ texte libre rédigé par les Customer Success Managers.
#    Il peut contenir des données personnelles des contacts opérationnels du client (noms,
#    postes, opinions personnelles). Au titre de la **minimisation** (article 5.1.c), cette
#    variable est exclue du modèle — décision actée dans `config.COLONNES_LEURRES_SUSPECTES`.
# 2. **Contacts opérationnels associés aux comptes** : les coordonnées des interlocuteurs
#    CRM sont des données personnelles. Elles ne sont pas dans le jeu modélisé mais entrent
#    dans le périmètre du traitement global lors de l'envoi des alertes CS.
#
# **Humain dans la boucle : un choix de conception, non une obligation.**
# Le modèle classifie ; le Customer Success Manager décide. Ce choix n'est pas imposé par
# l'article 22 (hors champ), il relève d'un **choix de gouvernance** : responsabilité humaine
# assumée, réduction du risque d'erreur systématique, meilleure acceptabilité interne.

# %%
import pandas as pd
from IPython.display import display

# Tableau de conformité RGPD
rgpd_items = [
    {
        "Exigence RGPD": "Base légale (art. 6)",
        "Application au projet": "Données entreprises : hors champ RGPD. "
        "Pour les contacts opérationnels associés : intérêt légitime (art. 6.1.f) — "
        "prévention de la résiliation dans une relation contractuelle préexistante.",
        "Mesure retenue": "Mention dans le registre de traitements du DPO. "
        "Analyse d'intérêt légitime documentée.",
    },
    {
        "Exigence RGPD": "Minimisation (art. 5.1.c)",
        "Application au projet": "`commentaire_csm` : texte libre pouvant contenir "
        "des données personnelles de contacts identifiables.",
        "Mesure retenue": "Variable exclue du modèle (config.COLONNES_LEURRES_SUSPECTES). "
        "Justification documentée dans §7.",
    },
    {
        "Exigence RGPD": "Finalité (art. 5.1.b)",
        "Application au projet": "Prédiction du risque de résiliation pour orientation "
        "des actions de rétention CS — finalité explicitement définie.",
        "Mesure retenue": "Finalité inscrite dans la fiche de traitement. "
        "Réutilisation à d'autres fins (scoring crédit, tarification discriminante) interdite.",
    },
    {
        "Exigence RGPD": "Durée de conservation (art. 5.1.e)",
        "Application au projet": "Prédictions et scores stockés pour suivi des actions CS.",
        "Mesure retenue": "Durée maximale : 24 mois (alignée sur le cycle contractuel annuel "
        "+ 1 an de comparaison). Suppression automatique à l'expiration.",
    },
    {
        "Exigence RGPD": "Information (art. 13/14)",
        "Application au projet": "Les contacts opérationnels dont les coordonnées transitent "
        "dans le CRM doivent être informés du traitement.",
        "Mesure retenue": "Mention ajoutée aux CGU / politique de confidentialité de l'éditeur. "
        "Hors périmètre du présent modèle — action transmise au DPO.",
    },
    {
        "Exigence RGPD": "Article 22 — décision automatisée",
        "Application au projet": "Hors champ B2B (personnes morales). "
        "Humain dans la boucle maintenu par choix de conception.",
        "Mesure retenue": "Le modèle produit un score de risque ; "
        "la décision d'action est prise par le CSM.",
    },
]

df_rgpd = pd.DataFrame(rgpd_items).set_index("Exigence RGPD")
display(df_rgpd.style.set_properties(**{"text-align": "left", "white-space": "pre-wrap"}))

# %% [markdown]
# **Ce qu'il faut retenir.** Le RGPD s'applique marginalement à ce projet B2B : les cibles
# du modèle sont des entreprises (personnes morales), non des personnes physiques. L'article 22
# est très probablement hors champ. Deux points de vigilance réels : `commentaire_csm` (exclu
# par minimisation) et les contacts opérationnels CRM (couverts par le registre de traitements
# du DPO). L'humain dans la boucle est un choix de conception, pas une contrainte légale.

# %% [markdown]
# ---
# ### 4.2 AI Act — Classification du système et obligations applicables
#
# Le règlement (UE) 2024/1689 sur l'intelligence artificielle (AI Act) est entré en application
# progressivement depuis août 2024. Son champ d'application est large.

# %%
# Tableau de classification AI Act
ai_act_items = [
    {
        "Étape d'analyse": "Art. 3(1) — Le système est-il un « système d'IA » ?",
        "Analyse": "Oui, sans ambiguïté. L'article 3(1) définit un système d'IA comme "
        "un système à base de machine conçu pour fonctionner avec des niveaux d'autonomie "
        "variables, qui génère des résultats tels que des prédictions, recommandations ou "
        "décisions influençant des environnements réels. Une régression logistique, un "
        "gradient boosting ou toute autre méthode ML entre dans cette définition. "
        "L'AI Act ne se limite pas au deep learning.",
        "Conclusion": "Système d'IA — AI Act applicable.",
    },
    {
        "Étape d'analyse": "Art. 5 — Pratiques interdites",
        "Analyse": "L'article 5 prohibe notamment : la manipulation subliminale, "
        "l'exploitation de vulnérabilités, la notation sociale généralisée, "
        "la biométrie de masse. "
        "Si le score alimente une politique tarifaire différenciée, un risque de "
        "manipulation commerciale existe. Notre usage : aide à la priorisation CS, "
        "sans tarification automatique ni exploitation de vulnérabilités. "
        "Pas de notation sociale (périmètre B2B). Pas de biométrie.",
        "Conclusion": "Non concerné par les pratiques interdites. "
        "À surveiller si le score est couplé à une tarification automatique.",
    },
    {
        "Étape d'analyse": "Annexe III — Systèmes à haut risque",
        "Analyse": "L'annexe III liste 8 domaines haut risque : biométrie, "
        "infrastructures critiques, éducation, emploi et gestion des travailleurs, "
        "accès à des services essentiels (crédit, assurance, sécurité sociale), "
        "répression, justice, démocratie. "
        "Notre système : ciblage commercial B2B. "
        "Pas d'emploi, pas de scoring de crédit, pas d'accès à un service essentiel, "
        "pas de biométrie, pas de répression. Le ciblage commercial B2B n'y figure pas.",
        "Conclusion": "Hors haut risque (Annexe III non applicable).",
    },
    {
        "Étape d'analyse": "Classification finale et obligations",
        "Analyse": "Risque minimal. Les obligations sont limitées : "
        "pas de conformité obligatoire au titre du haut risque. "
        "Néanmoins, alignement volontaire sur les exigences de transparence "
        "et de documentation technique (model card, datasheet, registre des risques) — "
        "pratiques que ce projet produit intégralement.",
        "Conclusion": "Risque minimal. Alignement volontaire sur la transparence.",
    },
]

df_ai_act = pd.DataFrame(ai_act_items).set_index("Étape d'analyse")
display(df_ai_act.style.set_properties(**{"text-align": "left", "white-space": "pre-wrap"}))

# %% [markdown]
# **Ce qu'il faut retenir.** Le système est bien un « système d'IA » au sens de l'article 3(1),
# y compris en régression logistique. L'analyse de l'annexe III conclut à l'absence de haut
# risque : le ciblage commercial B2B n'y figure pas. L'article 5 n'est pas déclenché pour
# l'usage prévu. Les obligations effectives relèvent du risque minimal ; le projet s'aligne
# volontairement sur les standards de transparence et de documentation du haut risque.

# %% [markdown]
# ---
# ### 4.3 Chartes éthiques — volet européen et volet français
#
# #### 4.3.1 Lignes directrices HLEG — 7 exigences pour une IA digne de confiance (2019)
#
# Le Groupe d'experts de haut niveau sur l'IA (HLEG) de la Commission européenne a publié
# en 2019 les lignes directrices pour une IA digne de confiance (« Ethics Guidelines for
# Trustworthy AI »). Elles définissent 7 exigences clés.

# %%
# Tableau des 7 exigences HLEG × mesures concrètes du projet
hleg_items = [
    {
        "Exigence HLEG": "1. Primauté de l'humain et surveillance",
        "Description": "Les systèmes d'IA doivent soutenir l'autonomie humaine "
        "et permettre une supervision effective.",
        "Mesure concrète dans le projet": "• Score de risque interprétable (SHAP §9). "
        "• Décision finale déléguée au CSM, jamais au modèle seul. "
        "• Tableau de bord avec indicateur de confiance et alertes de drift (§13). "
        "• Possibilité de suspendre le système si performance dégradée (playbook §13).",
    },
    {
        "Exigence HLEG": "2. Robustesse technique et sécurité",
        "Description": "Le système doit être fiable, précis et résistant aux perturbations.",
        "Mesure concrète dans le projet": "• Test de robustesse à l'inférence (bruit gaussien "
        "et valeurs manquantes ajoutés — §13). "
        "• Graine unique (config.RANDOM_SEED=42) pour la reproductibilité. "
        "• Pipeline scikit-learn fitté dans chaque pli de CV : zéro fuite de données.",
    },
    {
        "Exigence HLEG": "3. Vie privée et gouvernance des données",
        "Description": "Les données personnelles doivent être protégées et leur usage gouverné.",
        "Mesure concrète dans le projet": "• Exclusion de `commentaire_csm` (minimisation). "
        "• config.COLONNES_INTERDITES appliquée automatiquement dans le pipeline. "
        "• Datasheet documentée (docs/DATASHEET.md). "
        "• Durée de conservation définie (24 mois — cf. §4.1).",
    },
    {
        "Exigence HLEG": "4. Transparence",
        "Description": "Traçabilité des systèmes, explicabilité des décisions, "
        "communication ouverte sur les capacités et limites.",
        "Mesure concrète dans le projet": "• SHAP values par compte (§9) — explicabilité locale. "
        "• Model card (MLflow Registry §10). "
        "• Notebook entièrement exécutable — toute affirmation chiffrée est produite par le code. "
        "• Limites assumées explicitement (§14).",
    },
    {
        "Exigence HLEG": "5. Diversité, non-discrimination et équité",
        "Description": "Éviter les biais injustes et garantir l'accessibilité.",
        "Mesure concrète dans le projet": "• Mesure des biais par pays, taille et secteur (§4.4). "
        "• Groupe témoin non traité (~10 %) pour évaluer l'impact réel des interventions (§4.5). "
        "• Revue de fairness documentée dans la fiche DPO (§4.6).",
    },
    {
        "Exigence HLEG": "6. Bien-être sociétal et environnemental",
        "Description": "Prendre en compte les impacts sur la société et l'environnement.",
        "Mesure concrète dans le projet": "• Estimation de l'empreinte carbone via CodeCarbon (§9/§12). "
        "• Note d'arbitrage performance/coût/carbone transmise au commanditaire (§8). "
        "• Analyse des 3 dilemmes éthiques sociétaux (§4.5).",
    },
    {
        "Exigence HLEG": "7. Responsabilité",
        "Description": "Mécanismes d'imputabilité, d'audit et de recours.",
        "Mesure concrète dans le projet": "• Registre des risques (docs/RISK_REGISTER.md). "
        "• Propriétaires assignés à chaque risque. "
        "• Fiche DPO/juriste simulée (§4.6). "
        "• Versioning modèle + données (MLflow + DVC §10). "
        "• Playbook de réponse aux incidents (§13).",
    },
]

df_hleg = pd.DataFrame(hleg_items).set_index("Exigence HLEG")
display(df_hleg.style.set_properties(**{"text-align": "left", "white-space": "pre-wrap"}))

# %% [markdown]
# **Ce qu'il faut retenir.** Les 7 exigences HLEG sont toutes couvertes par des mesures
# concrètes et traçables dans le projet. La transparence et la responsabilité constituent
# les axes les plus documentés, conformément aux recommandations du jury CISIA.

# %% [markdown]
# #### 4.3.2 Volet français — Rapport Villani, recommandations CNIL et charte Impact AI
#
# Au-delà du cadre européen, trois références françaises structurent l'approche éthique du projet.

# %%
volet_francais = [
    {
        "Référence": "Rapport Villani\n« Donner un sens à l'IA »\n(2018)",
        "Contenu pertinent": "Mission interministérielle co-pilotée par Cédric Villani. "
        "Recommandations structurantes : transparence algorithmique, explicabilité des décisions "
        "à fort impact, développement d'une IA « de confiance » respectueuse des droits "
        "fondamentaux, formation des équipes, et investissement dans la recherche en éthique de l'IA.",
        "Application au projet": "• Explicabilité locale SHAP (§9) — répond à l'enjeu "
        "de transparence algorithmique. "
        "• Formation documentée des CSM à l'interprétation du score (§10). "
        "• Ce notebook assume les limites du modèle et les documente explicitement (§14).",
    },
    {
        "Référence": "Recommandations CNIL\nsur l'IA\n(2022 et suiv.)",
        "Contenu pertinent": "La CNIL a publié une série de recommandations sur les systèmes "
        "d'IA : privacy by design, droit à l'explication, vigilance renforcée sur les biais, "
        "transparence sur les traitements automatisés, et consignes spécifiques sur "
        "l'IA générative et le profilage. "
        "Note : ces recommandations s'appliquent aux traitements de données personnelles ; "
        "leur portée directe est marginale en contexte B2B, mais leur esprit oriente les bonnes "
        "pratiques retenues.",
        "Application au projet": "• Privacy by design : exclusion préventive de `commentaire_csm`. "
        "• Droit à l'explication : SHAP par compte + tableau de bord interprétable. "
        "• Biais : mesure systématique par sous-groupe (§4.4). "
        "• Fiche de revue DPO documentée (§4.6).",
    },
    {
        "Référence": "Charte Impact AI\n(coalition française)",
        "Contenu pertinent": "Coalition d'organisations françaises (entreprises, institutions, "
        "associations) engagées pour un développement responsable de l'IA. "
        "La charte engage les signataires sur : transparence, équité, responsabilité, "
        "formation des équipes, gouvernance des données et réduction de l'empreinte "
        "environnementale. "
        "Note : la charte est une démarche volontaire sans valeur réglementaire. "
        "Elle est citée ici comme référence d'engagement sectoriel, non comme obligation.",
        "Application au projet": "• Transparence : notebook entièrement exécutable. "
        "• Équité : registre des risques et mesure des biais. "
        "• Environnement : estimation CodeCarbon (§9/§12). "
        "• Formation : guide d'usage CSM (§10).",
    },
]

df_fr = pd.DataFrame(volet_francais).set_index("Référence")
display(df_fr.style.set_properties(**{"text-align": "left", "white-space": "pre-wrap"}))

# %% [markdown]
# **Ce qu'il faut retenir.** Les trois références françaises — rapport Villani (stratégie
# nationale), recommandations CNIL (protection des données et biais) et charte Impact AI
# (engagement sectoriel volontaire) — convergent sur les mêmes exigences : transparence,
# explicabilité, mesure des biais, formation des équipes. Ce projet les couvre intégralement.

# %% [markdown]
# ---
# ### 4.4 Mesure des biais par sous-groupe
#
# **Périmètre de l'analyse.** Les variables `pays`, `taille_entreprise` et `secteur` sont
# des **attributs d'entreprise**, non des données personnelles au sens du RGPD. Néanmoins,
# un modèle qui présenterait des taux de détection significativement différents selon ces
# groupes induirait un **risque de discrimination commerciale** : certains segments
# seraient systématiquement sur- ou sous-signalés, indépendamment de leur comportement réel.
#
# **Méthode.** On entraîne ici un modèle de référence simple (régression logistique,
# prétraitement minimal) sur un split stratifié 80/20, dans le seul but de disposer
# de prédictions pour calculer les métriques d'équité. Ce n'est pas le modèle champion
# (§9) — c'est un **modèle dédié à l'audit de fairness**. Le calcul est mis en cache
# via `charger_ou_calculer()` pour ne pas alourdir le notebook.

# %%
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from churn_saas import config
from churn_saas.cache import charger_ou_calculer

warnings.filterwarnings("ignore", category=FutureWarning)


def _calculer_biais() -> dict:
    """
    Entraîne un modèle logistique minimal sur un split 80/20 et retourne les métriques
    d'équité (taux de churn, TPR, FPR) par pays, taille_entreprise et secteur.
    Fonction interne à la section éthique — ne pas réutiliser comme modèle de production.
    """
    df = pd.read_csv(
        config.DONNEES_BRUTES / "churn_saas_complet.csv",
        sep=None,
        engine="python",
        on_bad_lines="skip",
    )

    features_groupe = ["pays", "taille_entreprise", "secteur"]
    features_num = [
        "anciennete_mois",
        "taux_adoption_pct",
        "connexions_30j",
        "tickets_support_90j",
        "retards_paiement_12m",
        "revenu_mensuel_recurrent_eur",
        "derniere_connexion_jours",
    ]
    cible = "churn"

    colonnes_requises = features_groupe + features_num + [cible]
    df = df[[c for c in colonnes_requises if c in df.columns]].copy()

    # Nettoyage minimal : coercition numérique, suppression des lignes sans cible
    for col in features_num:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[cible])
    df[cible] = df[cible].astype(int)

    # Remplissage des manquants : médiane pour le numérique, mode pour le catégoriel
    for col in features_num:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    for col in features_groupe:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "inconnu")

    X = df[features_num].copy()
    X_groupe = df[features_groupe].copy()
    y = df[cible]

    X_train, X_test, y_train, y_test, Xg_train, Xg_test = train_test_split(
        X, y, X_groupe, test_size=0.20, random_state=config.RANDOM_SEED, stratify=y
    )

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=config.RANDOM_SEED))])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    resultats = {}
    for var in features_groupe:
        lignes = []
        for val in sorted(Xg_test[var].unique()):
            masque = Xg_test[var] == val
            if masque.sum() < 10:
                continue
            y_v = y_test[masque]
            p_v = pd.Series(y_pred)[masque.values]
            taux_churn = y_v.mean()
            if y_v.nunique() < 2:
                continue
            tn, fp, fn, tp = confusion_matrix(y_v, p_v, labels=[0, 1]).ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
            lignes.append(
                {
                    "Valeur": str(val),
                    "N (test)": int(masque.sum()),
                    "Taux churn réel": round(float(taux_churn), 3),
                    "TPR (Sensibilité)": round(float(tpr), 3),
                    "FPR (1 – Spécificité)": round(float(fpr), 3),
                }
            )
        resultats[var] = pd.DataFrame(lignes).set_index("Valeur")
    return resultats


tables_biais, _ = charger_ou_calculer("biais_sous_groupes.joblib", _calculer_biais)

# %% [markdown]
# #### 4.4.1 Biais par pays

# %%
print("=== Métriques d'équité par PAYS ===")
display(tables_biais["pays"])

# %% [markdown]
# #### 4.4.2 Biais par taille d'entreprise

# %%
print("=== Métriques d'équité par TAILLE D'ENTREPRISE ===")
display(tables_biais["taille_entreprise"])

# %% [markdown]
# #### 4.4.3 Biais par secteur

# %%
print("=== Métriques d'équité par SECTEUR ===")
display(tables_biais["secteur"])

# %% [markdown]
# #### 4.4.4 Analyse des écarts et seuil d'alerte

# %%
seuil_alerte_tpr = 0.15  # Écart de TPR toléré entre sous-groupes

for var, df_biais in tables_biais.items():
    if "TPR (Sensibilité)" not in df_biais.columns or df_biais["TPR (Sensibilité)"].isna().all():
        continue
    ecart_tpr = df_biais["TPR (Sensibilité)"].max() - df_biais["TPR (Sensibilité)"].min()
    ecart_fpr = df_biais["FPR (1 – Spécificité)"].max() - df_biais["FPR (1 – Spécificité)"].min()
    alerte = "⚠️ ALERTE" if ecart_tpr > seuil_alerte_tpr else "✓ OK"
    print(f"[{var}] Écart TPR = {ecart_tpr:.3f} | Écart FPR = {ecart_fpr:.3f} | {alerte}")

print(f"\nSeuil d'alerte : écart de TPR > {seuil_alerte_tpr:.0%} entre sous-groupes.")
print("Tout écart significatif est documenté dans docs/RISK_REGISTER.md (R01).")

# %% [markdown]
# **Ce qu'il faut retenir.** Les métriques d'équité sont calculées sur un modèle de référence
# logistique entraîné spécifiquement pour cet audit (pas le modèle champion de §9). Les variables
# analysées (`pays`, `taille_entreprise`, `secteur`) sont des attributs d'entreprise, non des
# données personnelles. Un écart de TPR supérieur à 15 points entre sous-groupes déclencherait
# une revue obligatoire avant déploiement — seuil choisi en référence aux pratiques de fairness
# ML (cf. IBM AI Fairness 360). La même analyse sera répétée sur le modèle champion en §12.

# %% [markdown]
# ---
# ### 4.5 Dilemmes éthiques identifiés et arbitrages
#
# Trois dilemmes structurels sont inhérents à un système de prédiction du churn utilisé
# pour orienter des actions commerciales. Ils sont exposés ici, avec leurs mécanismes
# et les mesures de mitigation retenues.

# %% [markdown]
# #### Dilemme 1 — La prophétie auto-réalisatrice sur les petits comptes
#
# **Mécanisme.** Le modèle attribue un score de risque élevé aux petits comptes (MRR faible,
# faible adoption). L'équipe CS, dont la capacité est limitée, priorise les comptes à fort MRR.
# Les petits comptes signalés à risque ne reçoivent aucune intervention. Privés d'accompagnement,
# ils finissent effectivement par résilier — confirmant la prédiction du modèle.
# Le modèle n'a pas prédit le churn : il l'a *organisé*.
#
# **Gravité.** Élevée — boucle causale difficile à détecter sans groupe témoin.
#
# **Mitigation retenue.**
# - La règle de priorisation CS intègre explicitement le ratio *risque × MRR × coût d'intervention*
#   (modélisé en §12), et non le score seul.
# - Un quota minimal de petits comptes traités par semaine est défini (action managériale, hors
#   périmètre technique du modèle — transmis dans la note de synthèse au commanditaire).
# - Le groupe témoin (~10 % des comptes non traités — cf. Dilemme 3) permet de détecter
#   si la non-intervention causale de résiliation.

# %% [markdown]
# #### Dilemme 2 — L'incitation perverse de la remise offerte aux comptes à risque
#
# **Mécanisme.** Pour retenir un compte signalé à risque élevé, le CSM propose une remise
# commerciale. Si cette pratique se généralise, deux effets pervers apparaissent :
# (a) les clients apprennent que simuler un comportement de départ (réduire les connexions,
# ouvrir des tickets) déclenche une offre commerciale avantageuse — *gaming* du modèle ;
# (b) la marge est dégradée sur les comptes qui seraient partis de toute façon (*deadweight loss*).
#
# **Gravité.** Moyenne — risque réel mais observable via le monitoring du taux de remise.
#
# **Mitigation retenue.**
# - La décision d'offrir une remise reste humaine (CSM + validation manager) et n'est jamais
#   automatisée par le modèle.
# - Le tableau de bord de §12 suit le taux de remise par cohorte de score — toute dérive
#   détectée déclenche une revue de la politique commerciale (non du modèle).
# - Le gaming est suivi via le monitoring de la distribution des features d'entrée (Evidently §13) :
#   une variation anormale des connexions_30j ou tickets_support_90j dans les semaines précédant
#   le renouvellement sera signalée.

# %% [markdown]
# #### Dilemme 3 — La boucle de rétroaction et la corruption de la distribution d'apprentissage
#
# **Mécanisme.** Le modèle prédit le churn d'un compte. Une intervention CS est déclenchée.
# Cette intervention modifie le comportement du compte : il ne résilie pas. Au prochain cycle
# d'entraînement, ce compte apparaît comme *non churné* — mais son vrai comportement contrefactuel
# (aurait-il résilié sans intervention ?) reste inconnu. Le modèle apprend sur une distribution
# biaisée par ses propres actions passées. Avec le temps, il sous-estime le risque réel des
# comptes qui font l'objet d'interventions récurrentes.
#
# **Gravité.** Élevée — dégradation silencieuse et progressive, difficile à détecter sans
# protocole expérimental explicite.
#
# **Mitigation retenue : groupe témoin non traité (~10 %).**
# - À chaque cycle de prédiction, environ 10 % des comptes à risque élevé sont *volontairement*
#   exclus de la liste d'intervention (tirage aléatoire stratifié par segment).
# - Ces comptes constituent le groupe de contrôle : leur devenir réel (churn ou non) sans
#   intervention fournit les labels non biaisés nécessaires au réentraînement.
# - Ce protocole est documenté dans §13 (amélioration continue) et transmis au commanditaire
#   dans la note de synthèse ci-dessous.
# - **Limite assumée** : ce groupe témoin implique de laisser certains comptes sans intervention
#   malgré un risque élevé identifié. Ce choix est éthiquement discutable et doit être validé
#   par le commanditaire. Il est présenté ici comme condition nécessaire à la robustesse
#   scientifique du système, non comme une décision automatique.

# %%
# Synthèse des dilemmes
dilemmes = [
    {
        "Dilemme": "1 — Prophétie auto-réalisatrice",
        "Mécanisme": "Petits comptes signalés → dépriorisés → résiliation confirmée",
        "Probabilité": "Élevée",
        "Impact": "Élevé",
        "Mitigation": "Règle de priorisation MRR×risque + quota petits comptes + groupe témoin",
        "Référence RISK_REGISTER": "R02",
    },
    {
        "Dilemme": "2 — Incitation perverse (remise)",
        "Mécanisme": "Remise systématique → gaming du modèle + deadweight loss commercial",
        "Probabilité": "Moyenne",
        "Impact": "Moyen",
        "Mitigation": "Décision humaine obligatoire + monitoring taux de remise + drift features",
        "Référence RISK_REGISTER": "R03",
    },
    {
        "Dilemme": "3 — Boucle de rétroaction",
        "Mécanisme": "Actions CS modifient l'issue → labels biaisés → dégradation silencieuse du modèle",
        "Probabilité": "Élevée",
        "Impact": "Élevé",
        "Mitigation": "Groupe témoin 10 % non traité pour labels contrefactuels",
        "Référence RISK_REGISTER": "R04",
    },
]

df_dilemmes = pd.DataFrame(dilemmes).set_index("Dilemme")
display(df_dilemmes.style.set_properties(**{"text-align": "left", "white-space": "pre-wrap"}))

# %% [markdown]
# **Ce qu'il faut retenir.** Les trois dilemmes identifiés sont des risques systémiques inhérents
# à tout système de prédiction du churn couplé à des actions de rétention. Aucune solution
# technique ne les élimine complètement ; ils nécessitent une gouvernance humaine active.
# Le groupe témoin non traité (~10 %) est la mitigation clé du dilemme 3, mais suppose
# une décision éthique délibérée du commanditaire, documentée dans la note ci-dessous.

# %% [markdown]
# ---
# ### 4.6 Artefacts de gouvernance
#
# #### 4.6.1 Registre des risques
#
# Le registre complet des risques est maintenu dans `docs/RISK_REGISTER.md`.
# Il couvre 8 risques identifiés, avec probabilité, impact, mitigation et propriétaire assigné.
# Ce registre est l'artefact principal de traçabilité des risques pour le jury C2.

# %%
print("Registre des risques : docs/RISK_REGISTER.md")
print("Pour consultation, ouvrir le fichier directement.")
print()
# Résumé court en tableau pour le notebook
risques_resume = [
    {"ID": "R01", "Risque": "Biais commercial par sous-groupe", "Propriétaire": "Data Scientist"},
    {"ID": "R02", "Risque": "Prophétie auto-réalisatrice (petits comptes)", "Propriétaire": "CS Lead"},
    {"ID": "R03", "Risque": "Incitation perverse (remises automatiques)", "Propriétaire": "Dir. Commercial"},
    {"ID": "R04", "Risque": "Boucle de rétroaction (distribution corrompue)", "Propriétaire": "Data Scientist"},
    {"ID": "R05", "Risque": "Fuite de données perso via commentaire_csm", "Propriétaire": "DPO"},
    {"ID": "R06", "Risque": "Sur-confiance (modèle non calibré)", "Propriétaire": "Data Scientist"},
    {"ID": "R07", "Risque": "Obsolescence non détectée du modèle", "Propriétaire": "MLOps / Data Scientist"},
    {"ID": "R08", "Risque": "Usage inadapté du score par les CSM", "Propriétaire": "CS Lead"},
]
display(pd.DataFrame(risques_resume).set_index("ID"))

# %%
from churn_saas import config

_risk_register = config.RACINE / "docs" / "RISK_REGISTER.md"
assert _risk_register.exists(), (
    "docs/RISK_REGISTER.md introuvable — vérifier que le fichier est inclus dans le ZIP de livraison."
)
_taille = _risk_register.stat().st_size
_lignes = len(_risk_register.read_text(encoding="utf-8").splitlines())
print(f"✓ docs/RISK_REGISTER.md présent ({_taille} octets, {_lignes} lignes).")
print("  Ce registre constitue la trace de communication des risques aux acteurs concernés (item C2).")
print("  La note de synthèse datée au commanditaire est reproduite ci-dessous (§4.6.2).")

# %% [markdown]
# #### 4.6.2 Note de synthèse au commanditaire
#
# > ---
# > **NOTE DE SYNTHÈSE — RISQUES ÉTHIQUES ET LÉGAUX**
# > Modèle de prédiction du churn SaaS B2B
# >
# > **Date :** 19 septembre 2026
# > **Destinataire :** Direction Générale / Commanditaire du projet
# > **Auteur :** Équipe Data Science
# > **Objet :** Risques éthiques, légaux et opérationnels portés à la connaissance des acteurs
# >
# > ---
# >
# > **1. Conformité légale**
# > Le présent système est classifié **à risque minimal** au sens de l'AI Act (règlement
# > UE 2024/1689). Le RGPD s'applique marginalement (clients B2B = personnes morales).
# > Deux points de vigilance ont été traités : exclusion de `commentaire_csm` (texte libre)
# > et coordination avec le DPO pour les contacts opérationnels CRM. Aucune obligation
# > de conformité haut risque n'est déclenchée.
# >
# > **2. Risques éthiques opérationnels**
# > Trois dilemmes systémiques sont identifiés et documentés (§4.5) :
# > - **Prophétie auto-réalisatrice** : le score peut aggraver la situation des petits comptes
# >   s'il n'est pas couplé à une règle de priorisation équilibrée (MRR × risque).
# >   **Décision demandée : définir un quota minimal de petits comptes traités par semaine.**
# > - **Remises perverses** : une politique de remise systématique aux comptes signalés
# >   peut être *gamée* et dégrade la marge. La décision de remise doit rester humaine.
# > - **Groupe témoin non traité** : pour garantir la robustesse scientifique à long terme,
# >   ~10 % des comptes à risque élevé seront volontairement exclus des interventions CS
# >   à chaque cycle. **Cette décision implique de laisser certains comptes sans accompagnement
# >   malgré un risque identifié. Elle requiert une validation explicite de la Direction.**
# >
# > **3. Biais commerciaux**
# > Des écarts de taux de détection entre sous-groupes (pays, taille, secteur) ont été mesurés.
# > Tout écart de TPR supérieur à 15 points déclenche une revue avant déploiement (R01).
# >
# > **4. Registre des risques**
# > Le registre complet (8 risques, probabilités, impacts, mitigations, propriétaires) est
# > disponible dans `docs/RISK_REGISTER.md`. Il est mis à jour à chaque cycle de réentraînement.
# >
# > **5. Prochaines étapes**
# > - Validation du groupe témoin par la Direction (décision demandée avant déploiement).
# > - Formation des CSM à l'interprétation du score (§10).
# > - Comité de revue trimestriel des métriques de fairness et de performance (§13).
# >
# > *Cette note constitue la traçabilité de la communication des risques aux acteurs concernés,
# > conformément à l'item C2 de la certification CISIA.*
# > ---

# %% [markdown]
# #### 4.6.3 Fiche de revue DPO / juriste
#
# *La fiche ci-dessous est **simulée** et assumée explicitement comme telle. Dans un projet
# réel, elle serait complétée et signée par le DPO ou le conseil juridique de l'entreprise.
# Elle est présentée ici pour démontrer la démarche de vérification par les acteurs concernés,
# conformément à l'item C2 du référentiel CISIA.*
#
# > ---
# > **FICHE DE REVUE DPO / JURISTE**
# > *(Simulée — assumée comme telle)*
# >
# > **Projet :** Modèle de prédiction du churn SaaS B2B
# > **Date de revue :** 19 septembre 2026
# > **Rôle du relecteur :** Délégué à la Protection des Données (DPO) / Juriste interne
# > **Périmètre examiné :**
# > - Jeu de données `churn_saas_complet.csv` (29 colonnes, ~5 000 lignes)
# > - Pipeline de traitement et de modélisation (sections §5 à §9 du notebook)
# > - API de prédiction (§10) et architecture cible (§11)
# > - Registre des risques `docs/RISK_REGISTER.md`
# >
# > **Avis rendu :** *Favorable sous réserve des levées de réserve listées ci-dessous.*
# >
# > **Réserves identifiées :**
# >
# > | N° | Réserve | Levée |
# > |---|---|---|
# > | 1 | `commentaire_csm` : texte libre susceptible de contenir des données personnelles | Exclue du modèle (config.COLONNES_LEURRES_SUSPECTES) ✓ |
# > | 2 | Contacts opérationnels CRM : coordonnées de personnes physiques | Mention à ajouter aux CGU de l'éditeur — **action ouverte, propriétaire : DPO** |
# > | 3 | Durée de conservation des prédictions non définie initialement | Définie à 24 mois (§4.1) ✓ |
# > | 4 | Finalité du traitement non formalisée | Inscrite dans la fiche de traitement et le registre des activités ✓ |
# > | 5 | Groupe témoin non traité : dimension éthique à valider | Soumise à validation de la Direction (note de synthèse §4.6.2) — **en attente** |
# >
# > **Levées de réserve :**
# > Réserves 1, 3, 4 levées. Réserve 2 et 5 ouvertes, avec propriétaires assignés.
# >
# > **Signature (simulée) :** DPO / Juriste interne — *[Non signé — document de démonstration
# > pédagogique]*
# > ---

# %% [markdown]
# **Ce qu'il faut retenir.** Les six items de C2 sont couverts : chartes éthiques (HLEG + volet
# français), impacts éthiques et sociétaux (3 dilemmes arbitrés), biais mesurés (TPR/FPR par
# sous-groupe), dilemmes identifiés, risques portés à la connaissance des acteurs (RISK_REGISTER +
# note de synthèse datée), vérification par les acteurs (fiche DPO simulée et assumée). L'ensemble
# des affirmations juridiques est conforme aux formulations de `docs/POINTS_DE_VIGILANCE.md`.

# %% [markdown]
# > ### 📋 Journal de bord — Éthique, société et conformité
# >
# > **Décisions retenues** — Formulation RGPD strictement conforme à l'article 22
# > (droit, non interdiction) ; hors champ B2B assumé et argumenté. Classification AI Act en
# > risque minimal après analyse de l'annexe III et de l'article 5. Alignement volontaire sur
# > les exigences de transparence. Groupe témoin 10 % retenu comme mitigation principale de la
# > boucle de rétroaction — soumis à validation commanditaire. Modèle dédié à l'audit de fairness
# > (LogReg minimal) distinct du modèle champion (§9).
# >
# > **Alternatives écartées** — Citer l'article 22 comme interdiction générale (formulation
# > incorrecte et contre-productive à l'oral). Prétendre que l'AI Act ne concerne pas la
# > régression logistique (l'article 3(1) est explicite). Présenter l'humain dans la boucle
# > comme une obligation légale (c'est un choix de conception). Utiliser les prédictions du
# > modèle champion pour l'audit de fairness en §4 (le champion n'est pas encore entraîné à
# > ce stade du notebook — modèle dédié retenu à la place).
# >
# > **Difficultés rencontrées** — Tension entre la position chronologique de §4 (avant
# > l'entraînement §9) et la nécessité de calculer TPR/FPR qui requièrent un modèle. Résolue
# > par un modèle logistique dédié à l'audit, mis en cache via `charger_ou_calculer()`, et une
# > note explicite précisant que le champion sera réévalué en §12.
# > Incertitude sur les formulations exactes de la charte Impact AI : référencée à son niveau
# > d'engagement général, sans citation d'articles spécifiques non vérifiés.
# >
# > **Impact sur la suite** — §12 (mesure de performance) reprend l'analyse de fairness sur
# > le modèle champion. §13 (amélioration continue) documente le protocole du groupe témoin
# > et la revue trimestrielle des métriques d'équité. La note de synthèse au commanditaire
# > conditionne le déploiement (groupe témoin à valider).
# >
# > **Temps passé** — Environ 3 h (analyse juridique, rédaction des artefacts, code de
# > mesure des biais).
