# Churn Radar — Mode d'emploi démo CSM

Interface de démonstration de la solution **churn_saas** à destination des Customer Success
Managers. Permet d'analyser le risque de résiliation d'un compte unitaire ou d'un portefeuille
entier, en appelant l'API FastAPI locale.

---

## 1. Prérequis

```bash
# Depuis la racine du projet
uv sync --extra demo
# (ou make setup si tous les extras sont déjà installés)
```

---

## 2. Lancement

```bash
make demo
```

Cela démarre :
- l'**API FastAPI** sur `http://localhost:8000` (documentation : `/docs`)
- l'**IHM Streamlit** sur `http://localhost:8501`

Ouvrez `http://localhost:8501` dans votre navigateur.

---

## 3. Scénario de démonstration (5 minutes)

### Étape 1 — Montrer une alerte critique

1. Onglet **🔍 Analyse client**
2. Cliquer **🚨 Compte en alerte** → le formulaire se pré-remplit
3. Cliquer **Analyser ce compte** → résultat attendu : `ALERTE_ROUGE`, ~72 % de probabilité,
   ~27 600 € à risque, 3 facteurs (inactivité élevée, satisfaction faible, retards de paiement)

### Étape 2 — Montrer l'effet d'une action corrective

4. Modifier **Taux d'adoption** de `22 %` → `55 %`
5. Modifier **Dernière connexion** de `42` → `10` jours
6. Cliquer à nouveau **Analyser ce compte** → décision bascule en `SURVEILLANCE` (~45 %)
   → illustre l'impact d'une intervention CSM réussie

### Étape 3 — Vue portefeuille

7. Onglet **📊 Portefeuille**
8. Cliquer **Analyser le portefeuille** (données de démo pré-chargées)
9. Observer la répartition : 4 comptes ALERTE_ROUGE, 3 SURVEILLANCE, 3 OK
10. Identifier les comptes à plus forte valeur à risque dans le tableau

### Étape 4 — Import d'un CSV réel

11. Sélectionner **Importer un CSV**, charger son propre fichier
    (format identique à `demo/data/demo_clients.csv`, voir section 4)

---

## 4. Format du CSV d'import

Le fichier doit contenir les 19 colonnes suivantes (sans `client_id`, ou avec — il sera ignoré
lors de l'appel à l'API et utilisé uniquement pour l'affichage) :

| Colonne | Type | Exemple |
|---|---|---|
| `anciennete_mois` | entier | `24` |
| `sieges_souscrits` | entier | `15` |
| `utilisateurs_actifs` | entier | `10` |
| `taux_adoption_pct` | décimal | `45.0` |
| `connexions_30j` | entier | `30` |
| `heures_usage_30j` | décimal ou vide | `60.0` |
| `fonctionnalites_total` | entier | `20` |
| `fonctionnalites_utilisees` | entier | `8` |
| `nb_integrations` | entier | `2` |
| `derniere_connexion_jours` | entier | `7` |
| `tickets_support_90j` | entier | `3` |
| `delai_reponse_support_h` | décimal ou vide | `12.0` |
| `csat` | décimal ou vide | `7.5` |
| `retards_paiement_12m` | entier | `0` |
| `revenu_mensuel_recurrent_eur` | décimal | `2000.0` |
| `secteur` | texte | `Technologie` |
| `pays` | texte | `France` |
| `taille_entreprise` | texte | `PME` |
| `plan` | texte | `Business` |

**Valeurs autorisées :**
- `secteur` : Technologie, Finance, Santé, Industrie, Services, Commerce, Éducation, Immobilier, Télécommunications, Médias, Logistique, Énergie
- `plan` : Starter, Pro, Business, Enterprise
- `taille_entreprise` : TPE, PME, ETI, GE
- Colonnes nullable (`heures_usage_30j`, `delai_reponse_support_h`, `csat`) : laisser vide si inconnu
