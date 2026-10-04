# %% [markdown]
# ## 4. Éthique, société et conformité
#
# Un score de risque oriente des actions commerciales : il faut savoir quelles règles s'imposent
# et quels effets indésirables il peut produire. Cette section analyse le RGPD et l'AI Act,
# confronte le projet aux chartes éthiques, fixe le protocole de mesure des biais et arbitre les
# dilemmes d'usage. Elle produit le registre des risques (`docs/RISK_REGISTER.md`), une note au
# commanditaire et la fiche de revue du DPO (trace reconstituée).

# %% [markdown]
# ---
# ### 4.1 RGPD — Analyse de conformité
#
# **Contexte B2B : le RGPD s'applique de façon marginale.** Le RGPD protège les *personnes
# concernées*, c'est-à-dire des **personnes physiques** identifiées ou identifiables (article 4).
# Ici, les comptes notés sont des **entreprises clientes** (personnes morales), que le RGPD ne
# régit pas en tant que telles.
#
# **L'article 22, formulé correctement.** Il n'interdit pas la décision automatisée : il ouvre à
# la personne concernée le *droit de ne pas faire l'objet d'une décision fondée exclusivement sur
# un traitement automatisé* produisant des effets juridiques la concernant ou l'affectant de
# manière significative, sous réserve d'exceptions (consentement explicite, nécessité
# contractuelle, autorisation légale). Le sujet du traitement étant une entreprise, cet article
# est **très probablement hors champ**.
#
# **Deux points de contact réels subsistent :**
#
# 1. **`commentaire_csm`** : texte libre des Customer Success Managers (CSM), qui peut contenir
#    des données personnelles des contacts du client (noms, postes, opinions). Au titre de la
#    **minimisation** (article 5.1.c), il est exclu du modèle ; la décision, prise à l'audit du
#    §6.6, est inscrite dans `config.COLONNES_INTERDITES`.
# 2. **Contacts opérationnels des comptes** : leurs coordonnées CRM sont des données
#    personnelles. Absentes du jeu modélisé, elles entrent dans le traitement global lors de
#    l'envoi des alertes aux CSM.
#
# **Humain dans la boucle : un choix de conception, non une obligation.** Le modèle classe, le
# CSM décide. L'article 22 ne l'impose pas ici ; c'est un **choix de gouvernance** :
# responsabilité humaine assumée, moindre risque d'erreur systématique, meilleure acceptation.
#
# | Exigence RGPD | Application au projet | Mesure retenue |
# |---|---|---|
# | **Base légale (art. 6)** | Données d'entreprises : hors champ. Contacts opérationnels : intérêt légitime (art. 6.1.f), prévenir la résiliation dans une relation contractuelle existante. | Mention au registre des traitements du DPO ; analyse d'intérêt légitime documentée. |
# | **Minimisation (art. 5.1.c)** | `commentaire_csm` peut contenir des données de contacts identifiables. | Variable exclue (`config.COLONNES_INTERDITES`), justifiée au §6.6. |
# | **Finalité (art. 5.1.b)** | Prédire le risque de résiliation pour orienter les actions de rétention. | Finalité inscrite dans la fiche de traitement ; réutilisation (crédit, tarification discriminante) interdite. |
# | **Conservation (art. 5.1.e)** | Scores conservés pour suivre les actions CS. | Durées dans le tableau ci-dessous ; purge automatique hebdomadaire des prédictions. |
# | **Information (art. 13/14)** | Les contacts dont les coordonnées transitent dans le CRM doivent être informés. | Mention aux CGU / politique de confidentialité de l'éditeur ; action transmise au DPO. |
# | **Transferts hors UE (chap. V)** | Les instances clientes sont réparties sur quatre régions, dont deux hors UE (`code_datacenter`, §6.10) : les journaux de connexion des utilisateurs y sont produits. | Agrégation par entreprise dans chaque région avant extraction : seuls des indicateurs d'entreprise rejoignent l'UE. Encadrement des instances hors UE (décision d'adéquation ou clauses contractuelles types) suivi par le DPO (R10). |
# | **Art. 22 — décision automatisée** | Hors champ B2B ; humain dans la boucle par choix de conception. | Le modèle produit un score ; le CSM décide de l'action. |
#
# **Durées de conservation retenues.** Aucune n'est une obligation légale : l'article 5.1.e exige
# une durée proportionnée à la finalité, sans la chiffrer. Ce sont des choix de gouvernance,
# validés par le DPO, revus chaque année, et repris par toutes les sections.
#
# | Données | Durée | Point de départ | À l'expiration | Justification |
# |---|---|---|---|---|
# | Scores et prédictions (table de scoring, CRM, journal des prédictions) | 24 mois | `date_prediction` | Purge automatique hebdomadaire | L'étiquette arrive jusqu'à 12 mois après la prédiction (contrats annuels + fenêtre de 90 jours), puis un an de comparaison d'une année sur l'autre. Plus court, le suivi de performance (§2.6) deviendrait impossible |
# | Données d'entraînement (versions gold DVC) | 3 ans | Création de la version | Suppression | Trois cycles de renouvellement annuel. Repère : référentiel CNIL « gestion commerciale » (3 ans) |
# | Journaux techniques (appels API, accès) | 12 mois | Écriture de l'entrée | Suppression | Repère CNIL pour les journaux de sécurité |
#
# La cellule suivante vérifie que ces durées sont bien celles de la configuration.

# %%
from churn_saas import config
from churn_saas.format_fr import entier

_durees_citees = {"predictions_mois": 24, "donnees_entrainement_ans": 3, "journaux_mois": 12}
for _cle, _valeur in _durees_citees.items():
    assert config.DUREES_CONSERVATION[_cle] == _valeur, f"§4.1 à mettre à jour : {_cle}"
print(f"✓ {entier(len(_durees_citees))} durées de conservation conformes à la configuration.")

# %% [markdown]
# **Ce qu'il faut retenir.** Le contrôle passe (durées conformes à la configuration). Le RGPD
# ne touche ce projet B2B qu'à la marge ; ses deux vrais points de contact sont traités.

# %% [markdown]
# ---
# ### 4.2 AI Act — Classification du système et obligations applicables
#
# L'AI Act (règlement (UE) 2024/1689, applicable progressivement depuis août 2024) proportionne
# les obligations au risque. On le parcourt : système d'IA ? pratique interdite ? haut risque ?
#
# | Étape d'analyse | Analyse | Conclusion |
# |---|---|---|
# | **Art. 3(1) — Est-ce un « système d'IA » ?** | Oui. L'article 3(1) vise tout système à base de machine, doté d'un certain niveau d'autonomie, qui génère des prédictions, recommandations ou décisions influençant un environnement réel. Une régression logistique ou un modèle de boosting (renforcement) y entrent : l'AI Act ne se limite pas à l'apprentissage profond. | Système d'IA : AI Act applicable. |
# | **Art. 5 — Pratiques interdites** | L'article 5 prohibe notamment la manipulation subliminale, l'exploitation des vulnérabilités, la notation sociale et certains usages biométriques. Notre usage : aide à la priorisation CS, sans tarification automatique ni exploitation de vulnérabilités, sans notation sociale (B2B) ni biométrie. | Non concerné. À réexaminer si le score était couplé à une tarification automatique. |
# | **Annexe III — Haut risque** | Huit domaines : biométrie, infrastructures critiques, éducation, emploi, accès aux services essentiels (crédit, assurance, prestations), répression, migration, justice et démocratie. Le ciblage commercial B2B n'y figure pas. | Hors haut risque. |
# | **Classification et obligations** | Risque minimal : aucune obligation de conformité au titre du haut risque. Alignement volontaire sur la transparence et la documentation technique (fiche modèle, datasheet, registre des risques), que le projet produit. | Risque minimal, transparence volontaire. |

# %% [markdown]
# **Ce qu'il faut retenir.** Le système est bien un « système d'IA » au sens de l'article 3(1),
# même avec une régression logistique. Il n'entre ni dans les pratiques interdites (article 5) ni
# dans le haut risque (annexe III) : ses obligations sont celles du risque minimal, complétées
# volontairement par la documentation qu'exigerait le haut risque.

# %% [markdown]
# ---
# ### 4.3 Chartes éthiques — volet européen et volet français
#
# Au-delà de la loi, des chartes décrivent ce qu'est une IA « digne de confiance ». On vérifie
# que chacune de leurs exigences se traduit par une mesure concrète du projet.
#
# #### 4.3.1 Lignes directrices HLEG — 7 exigences pour une IA digne de confiance (2019)
#
# Le groupe d'experts de haut niveau sur l'IA (HLEG) de la Commission européenne a publié en 2019
# des lignes directrices (« Ethics Guidelines for Trustworthy AI ») qui posent sept exigences.
#
# | Exigence HLEG | Description | Mesure concrète dans le projet |
# |---|---|---|
# | **1. Primauté de l'humain et surveillance** | L'IA soutient l'autonomie humaine et reste supervisable. | • Score explicable (SHAP, valeurs de Shapley, §12.8).<br>• Décision finale au CSM, jamais au modèle seul.<br>• Surveillance des dérives et alertes (§13).<br>• Suspension possible si la performance se dégrade (§13.8). |
# | **2. Robustesse technique et sécurité** | Système fiable, précis, résistant aux perturbations. | • Test de robustesse au bruit et aux valeurs manquantes (§13.5).<br>• Graine unique (`config.RANDOM_SEED`).<br>• Pipeline (chaîne de traitement) appris dans chaque pli de validation croisée : pas de fuite de données. |
# | **3. Vie privée et gouvernance des données** | Données personnelles protégées, usage gouverné. | • Exclusion de `commentaire_csm` (minimisation).<br>• `config.COLONNES_INTERDITES` appliquée dans la chaîne.<br>• Datasheet (`docs/DATASHEET.md`).<br>• Durées de conservation (§4.1). |
# | **4. Transparence** | Traçabilité, explicabilité, communication sur les limites. | • Valeurs SHAP par compte (§12.8).<br>• Registre de modèles MLflow (§10.4).<br>• Notebook exécutable : chaque chiffre est produit par le code.<br>• Limites assumées (§14). |
# | **5. Diversité, non-discrimination et équité** | Éviter les biais injustes. | • Biais par pays, taille et secteur : protocole §4.4, mesure §12.14.<br>• Groupe témoin non traité (~10 %, §4.5).<br>• Revue d'équité dans la fiche DPO (§4.6.3). |
# | **6. Bien-être sociétal et environnemental** | Impacts sur la société et l'environnement. | • Empreinte carbone estimée par CodeCarbon (§9.8, §12.13).<br>• Note d'arbitrage performance / coût / carbone (§9.9).<br>• Trois dilemmes sociétaux analysés (§4.5). |
# | **7. Responsabilité** | Imputabilité, audit, recours. | • Registre des risques avec propriétaires (`docs/RISK_REGISTER.md`).<br>• Fiche de revue DPO (§4.6.3).<br>• Versioning modèle + données (MLflow, DVC, §10.5).<br>• Plan de retour arrière (§13.8). |

# %% [markdown]
# **Ce qu'il faut retenir.** Chaque exigence HLEG a au moins une mesure concrète et traçable.

# %% [markdown]
# #### 4.3.2 Volet français — Rapport Villani, recommandations CNIL et charte Impact AI
#
# | Référence | Contenu pertinent | Application au projet |
# |---|---|---|
# | **Rapport Villani**<br>« Donner un sens à l'IA »<br>(2018) | Mission confiée par le Premier ministre à Cédric Villani : transparence algorithmique, explicabilité des décisions à fort impact, IA respectueuse des droits fondamentaux, formation des équipes. | • Explicabilité locale SHAP (§12.8).<br>• Transfert de connaissances aux CSM (§9.11).<br>• Limites du modèle documentées (§14). |
# | **Recommandations CNIL**<br>sur l'IA<br>(2022 et suiv.) | Protection des données dès la conception, droit à l'explication, vigilance sur les biais, transparence des traitements automatisés. *Portée directe marginale en B2B (données de personnes morales), mais leur esprit guide les pratiques retenues.* | • Exclusion préventive de `commentaire_csm`.<br>• Explication par compte (SHAP).<br>• Biais : protocole a priori (§4.4), mesure sur le modèle final (§12.14).<br>• Fiche de revue DPO (§4.6.3). |
# | **Charte Impact AI**<br>(coalition française) | Engagement volontaire d'organisations françaises : transparence, équité, responsabilité, formation, gouvernance des données, sobriété environnementale. *Sans valeur réglementaire : citée comme engagement sectoriel.* | • Notebook entièrement exécutable.<br>• Registre des risques et mesure des biais.<br>• Estimation CodeCarbon (§9.8).<br>• Transfert de connaissances (§9.11). |

# %% [markdown]
# **Ce qu'il faut retenir.** Rapport Villani (stratégie nationale), CNIL (données et biais) et
# charte Impact AI (engagement volontaire) convergent : transparence, explicabilité, mesure des
# biais, formation des équipes. Le projet répond à chacun de ces points.

# %% [markdown]
# ---
# ### 4.4 Mesure des biais par sous-groupe
#
# **Périmètre.** `pays`, `taille_entreprise` et `secteur` sont des **attributs d'entreprise**,
# pas des données personnelles. Mais un modèle qui repérerait moins bien les départs d'un pays
# ou d'un secteur ferait subir à ces clients un **traitement commercial inégal** : moins
# d'appels de rétention, à risque égal.
#
# **Biais potentiels identifiés avant toute modélisation.**
#
# | Mécanisme | Où il apparaît | Effet attendu sur le modèle |
# |---|---|---|
# | Sous-représentation | Pays hors France et grandes entreprises : quelques centaines de comptes chacun (effectifs en §6.3) | Moins d'exemples de départ appris : détection potentiellement plus faible dans ces segments |
# | Modalités manquantes | `pays` et `secteur` manquants pour une partie des comptes (§5.9) | Un groupe « inconnu » hétérogène, dont le traitement doit être vérifié |
# | Variable relais | `taille_entreprise` est liée au MRR, au nombre de sièges et à l'usage | Le modèle peut pénaliser une taille via ces variables, même si on retirait l'attribut |
# | Biais d'étiquette historique | Les départs passés dépendent des actions CS passées, décidées de mémoire (§2.2) | Le modèle apprend aussi les angles morts de l'ancienne pratique |
# | Politique de décision | Priorisation sous capacité par valeur attendue (§12.6) : P(churn) × MRR | Les petits comptes sont moins contactés : **choix économique assumé** (dilemme 1, §4.5), distinct d'un biais du modèle |
#
# **Protocole d'audit, fixé avant l'entraînement.** On veut mesurer les biais du modèle
# réellement déployé, avec sa vraie règle de décision : la mesure est donc faite en **§12.14**,
# sur les prédictions out-of-fold (hors pli : faites par un modèle qui n'a pas vu le compte) du
# modèle final et le jeu gold nettoyé. Un modèle de substitution entraîné ici sur les données
# brutes auditerait un autre modèle. Les règles d'interprétation, elles, sont fixées dès
# maintenant pour ne pas être choisies après avoir vu les résultats :
#
# | Critère | Mesure | Seuil de revue | Pourquoi |
# |---|---|---|---|
# | Égalité des chances | Écart de recall (rappel : part des churners signalés) entre modalités d'un attribut, au seuil qui signale autant de comptes qu'il y a de churners | > 15 points | À départ égal, un compte doit avoir la même chance d'être signalé (Hardt, Price et Srebro, 2016) |
# | Calibration par segment | Écart entre probabilité moyenne prédite et taux de départ réel dans la modalité | > 5 points | La priorisation multiplie la probabilité par le MRR : un segment mal calibré serait sur- ou sous-priorisé |
# | Effectif interprétable | Nombre de churners dans la modalité | < 30 : non interprété | Avec 30 churners, l'intervalle de confiance d'un recall fait déjà ±16 points environ |
#
# Aucune norme ne chiffre l'écart tolérable : ces seuils sont des **choix de gouvernance**,
# validés avec le CS Lead et inscrits dans `config.EQUITE`. Un dépassement déclenche une revue
# avant déploiement, consignée au registre des risques (R01, propriétaire : Data Scientist).

# %%
_seuils_cites = {"ecart_tpr_max": 0.15, "ecart_calibration_max": 0.05, "churners_min": 30}
for _cle, _valeur in _seuils_cites.items():
    assert config.EQUITE[_cle] == _valeur, f"§4.4 à mettre à jour : {_cle}"
print(f"✓ {entier(len(_seuils_cites))} seuils d'équité conformes à la configuration.")

# %% [markdown]
# **Ce qu'il faut retenir.** Le contrôle passe : les seuils du tableau sont ceux de
# `config.EQUITE`, qu'appliquera le §12.14, figés avant tout résultat. Quatre biais possibles
# sont identifiés ; le cinquième, la préférence pour les gros comptes, est un choix économique.

# %% [markdown]
# ---
# ### 4.5 Dilemmes éthiques identifiés et arbitrages
#
# Un score qui déclenche des actions commerciales modifie le comportement qu'il prédit. Trois
# dilemmes en découlent ; aucun n'a de solution purement technique.
#
# #### Dilemme 1 — La prophétie auto-réalisatrice sur les petits comptes
#
# **Mécanisme.** Le modèle signale des petits comptes (MRR faible, faible adoption). L'équipe CS,
# de capacité limitée, priorise les gros comptes. Privés d'accompagnement, les petits comptes
# signalés finissent par résilier, ce qui confirme la prédiction : le modèle n'a pas prédit le
# départ, il l'a *organisé*. **Gravité élevée** : boucle invisible sans groupe témoin.
#
# **Mitigation.** La priorisation combine risque, MRR et coût d'intervention (§12.6), pas le
# score seul ; un quota hebdomadaire minimal de petits comptes traités est proposé au
# commanditaire (décision managériale) ; le groupe témoin (dilemme 3) révèle si l'absence
# d'intervention cause elle-même des départs.
#
# #### Dilemme 2 — L'incitation perverse de la remise offerte aux comptes à risque
#
# **Mécanisme.** Pour retenir un compte signalé, le CSM propose une remise. Généralisée, la
# pratique a deux effets pervers : (a) les clients apprennent qu'un comportement de départ
# simulé (moins de connexions, plus de tickets) déclenche une offre, et **manipulent le
# modèle** ; (b) la remise est consentie à des comptes qui seraient restés (**effet d'aubaine**),
# ce qui dégrade la marge. **Gravité moyenne** : observable via le taux de remise.
#
# **Mitigation.** La remise reste une décision humaine (CSM + validation du manager), jamais
# automatisée ; le taux de remise par tranche de score est suivi, et une dérive déclenche une
# revue de la politique commerciale ; une variation anormale de `connexions_30j` ou de
# `tickets_support_90j` avant renouvellement est signalée par la surveillance des dérives (§13).
#
# #### Dilemme 3 — La boucle de rétroaction et la corruption de la distribution d'apprentissage
#
# **Mécanisme.** Le modèle signale un compte, le CSM intervient, le compte reste. Au
# réentraînement suivant, il apparaît *non résilié*, sans qu'on sache s'il serait parti sans
# intervention. Le modèle apprend sur des données façonnées par ses propres actions et finit par
# sous-estimer le risque des comptes régulièrement traités. **Gravité élevée** : dégradation
# silencieuse et progressive.
#
# **Mitigation : groupe témoin non traité (~10 %).** À chaque cycle, environ 10 % des comptes à
# risque élevé sont *volontairement* exclus de la liste d'intervention (tirage aléatoire
# stratifié par segment). Leur devenir fournit des étiquettes non biaisées pour le
# réentraînement et mesure l'effet réel des actions (§13). **Limite assumée** : laisser des
# comptes à risque sans intervention est éthiquement discutable ; c'est une condition de
# robustesse soumise à la validation du commanditaire, pas une décision automatique.
#
# | Dilemme | Mécanisme | Probabilité | Impact | Mitigation | Registre |
# |---|---|---|---|---|---|
# | **1 — Prophétie auto-réalisatrice** | Petits comptes signalés → dépriorisés → départ confirmé | Élevée | Élevé | Priorisation MRR × risque + quota petits comptes + groupe témoin | R02 |
# | **2 — Incitation perverse (remise)** | Remise systématique → manipulation + effet d'aubaine | Moyenne | Moyen | Décision humaine + suivi du taux de remise + dérive des variables | R03 |
# | **3 — Boucle de rétroaction** | Actions CS → étiquettes biaisées → dégradation silencieuse | Élevée | Élevé | Groupe témoin de 10 % non traité | R04 |

# %% [markdown]
# **Ce qu'il faut retenir.** La technique atténue ces dilemmes, seule une gouvernance humaine les
# maîtrise ; le groupe témoin exige une décision explicite du commanditaire.

# %% [markdown]
# ---
# ### 4.6 Artefacts de gouvernance
#
# #### 4.6.1 Registre des risques
#
# Un risque n'est maîtrisé que s'il a un propriétaire. `docs/RISK_REGISTER.md` détaille
# probabilité, impact et mitigation de chacun, et sert de trace de communication aux acteurs.
#
# | ID | Risque | Propriétaire |
# |---|---|---|
# | R01 | Biais commercial par sous-groupe | Data Scientist |
# | R02 | Prophétie auto-réalisatrice (petits comptes) | CS Lead |
# | R03 | Incitation perverse (remises automatiques) | Dir. Commercial |
# | R04 | Boucle de rétroaction (distribution corrompue) | Data Scientist |
# | R05 | Fuite de données personnelles via `commentaire_csm` | DPO |
# | R06 | Sur-confiance (modèle non calibré) | Data Scientist |
# | R07 | Obsolescence non détectée du modèle | Data Scientist |
# | R08 | Usage inadapté du score par les CSM | CS Lead |
# | R09 | Churners manqués sous le seuil de vigilance | Data Scientist + CS Lead |
# | R10 | Données d'usage produites hors UE (instances `us-e1`, `ap-s1`) | DPO + RSSI |

# %%
_risk_register = config.RACINE / "docs" / "RISK_REGISTER.md"
assert (
    _risk_register.exists()
), "docs/RISK_REGISTER.md introuvable : il doit accompagner le notebook."
_lignes = len(_risk_register.read_text(encoding="utf-8").splitlines())
print(f"✓ docs/RISK_REGISTER.md présent ({entier(_lignes)} lignes).")

# %% [markdown]
# #### 4.6.2 Note de synthèse au commanditaire
#
# > ---
# > **NOTE DE SYNTHÈSE — RISQUES ÉTHIQUES ET LÉGAUX** — Modèle de prédiction du churn SaaS B2B
# >
# > **19 septembre 2026** · Direction générale (commanditaire) · Équipe Data Science
# >
# > **1. Conformité légale.** Système **à risque minimal** au sens de l'AI Act ; RGPD applicable
# > à la marge (clients personnes morales). `commentaire_csm` est exclu et les contacts CRM sont
# > suivis avec le DPO. Aucune obligation de haut risque n'est déclenchée.
# >
# > **2. Risques éthiques opérationnels (§4.5).**
# > - *Prophétie auto-réalisatrice* : le score peut aggraver la situation des petits comptes.
# >   **Décision demandée : un quota minimal de petits comptes traités par semaine.**
# > - *Remises* : une remise systématique serait manipulée et dégraderait la marge ; elle reste
# >   une décision humaine.
# > - *Groupe témoin* : ~10 % des comptes à risque élevé exclus des interventions à chaque cycle,
# >   donc laissés sans accompagnement malgré un risque identifié. **Validation explicite de la
# >   Direction requise.**
# >
# > **3. Biais commerciaux.** Protocole fixé avant l'entraînement (§4.4), mesure sur le modèle
# > final (§12.14) : un écart de recall supérieur à 15 points ou de calibration supérieur à
# > 5 points entre sous-groupes déclenche une revue avant déploiement (R01).
# >
# > **4. Prochaines étapes.** Validation du groupe témoin avant déploiement ; transfert de
# > connaissances aux CSM (§9.11) ; comité trimestriel de revue équité et performance (§13.11).
# >
# > ---

# %% [markdown]
# #### 4.6.3 Fiche de revue DPO / juriste
#
# *Trace reconstituée, à remplacer par la fiche complétée et signée par le DPO ou le conseil
# juridique de l'entreprise.*
#
# > ---
# > **FICHE DE REVUE DPO / JURISTE** — Modèle de prédiction du churn SaaS B2B
# >
# > **19 septembre 2026** · Délégué à la protection des données (DPO) / juriste interne
# >
# > **Périmètre examiné :** jeu `churn_saas_complet.csv` (volumétrie au §3.1) ; chaîne de
# > traitement et de modélisation (§5 à §9) ; API de prédiction (§10) et architecture cible
# > (§11) ; registre des risques `docs/RISK_REGISTER.md`.
# >
# > **Avis rendu :** *favorable sous réserve des levées ci-dessous.*
# >
# > | N° | Réserve | Levée |
# > |---|---|---|
# > | 1 | `commentaire_csm` : texte libre susceptible de contenir des données personnelles | Exclu du modèle (`config.COLONNES_INTERDITES`, audit §6.6) ✓ |
# > | 2 | Contacts opérationnels CRM : coordonnées de personnes physiques | Mention à ajouter aux CGU de l'éditeur — **action ouverte, propriétaire : DPO** |
# > | 3 | Durée de conservation des prédictions non définie initialement | 24 mois, avec les autres durées (§4.1) ✓ |
# > | 4 | Finalité du traitement non formalisée | Inscrite dans la fiche de traitement et le registre des activités ✓ |
# > | 5 | Groupe témoin non traité : dimension éthique à valider | Soumise à la Direction (note §4.6.2) — **en attente** |
# > | 6 | Instances clientes hébergées hors UE (`code_datacenter`) | Agrégation par entreprise en région avant extraction (§4.1, R10) ✓ |
# >
# > Réserves 1, 3, 4 et 6 levées ; réserves 2 et 5 ouvertes, avec propriétaires assignés.
# >
# > ---

# %% [markdown]
# **Ce qu'il faut retenir.** Le DPO valide le traitement sous deux réserves ouvertes, chacune
# avec un propriétaire : l'information des contacts CRM et la validation du groupe témoin.

# %% [markdown]
# > ### 📋 Journal de bord — Éthique, société et conformité
# >
# > **Décisions retenues** — Article 22 présenté comme un droit, non une interdiction, et hors
# > champ B2B argumenté. AI Act : risque minimal après analyse de l'article 5 et de l'annexe III,
# > avec alignement volontaire sur la transparence. Groupe témoin de 10 % comme mitigation de la
# > boucle de rétroaction, soumis au commanditaire. Protocole d'équité défini ici
# > (`config.EQUITE`) et appliqué en §12.14 au modèle final, avec sa vraie règle de décision.
# >
# > **Alternatives écartées** — Citer l'article 22 comme interdiction générale (juridiquement
# > faux). Dire que l'AI Act ne concerne pas la régression logistique (l'article 3(1) est
# > explicite). Présenter l'humain dans la boucle comme une obligation légale.
# >
# > **Difficultés rencontrées** — Une première mesure des biais, sur un modèle d'audit dédié, a
# > été abandonnée : doublons et casse non normalisée faussaient les sous-groupes, au seuil
# > arbitraire de 0,5 et sur un autre modèle que celui livré. D'où la séparation protocole ici /
# > mesure en §12.14. Revue DPO reconstituée, deux réserves ouvertes (mention CGU, groupe témoin).
# >
# > **Impact sur la suite** — §12.14 mesure l'équité selon ce protocole. §13 reprend le groupe
# > témoin et la revue trimestrielle des métriques d'équité. Le déploiement reste conditionné à la
# > validation du groupe témoin par le commanditaire.
