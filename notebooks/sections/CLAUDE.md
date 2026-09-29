# Règles propres aux sections du notebook

Ces fichiers `NN_*.py` (jupytext « percent ») sont la source du notebook généré. Les règles
ci-dessous complètent le `CLAUDE.md` racine.

## Texte statique → markdown, jamais du code

Une cellule `# %%` ne doit exister que si elle **calcule** quelque chose (données, modèle, figure,
vérification sur un fichier…). Tout contenu rédigé à la main s'écrit directement dans une cellule
`# %% [markdown]` :

- tableau d'information (conformité, chartes, risques, synthèses) → **tableau markdown**, pas une
  liste de dictionnaires passée à `pd.DataFrame(...)` puis `display()` ;
- `print()` d'un texte fixe (titre, rappel, renvoi vers un document) → phrase dans le markdown
  voisin ;
- dans une cellule de calcul, seuls les `print()` qui affichent une **valeur calculée** sont
  admis.

Pourquoi : le jury lit le notebook comme un rapport. Du texte caché dans du code est plus dur à
lire, à relire et à maintenir, et laisse croire à un résultat calculé là où il n'y en a pas.

Mise en forme des tableaux markdown : une ligne par rangée (E501 est ignoré par ruff), `<br>` pour
les retours à la ligne dans une cellule, identifiants et chemins entre backticks.
