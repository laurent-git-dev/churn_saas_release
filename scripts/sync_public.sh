#!/usr/bin/env bash
set -euo pipefail

PUBLIC_REMOTE="https://github.com/laurent-git-dev/churn_saas_release.git"
PRIVATE_REPO="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

# 1. Clone local vers temp (--no-local = vrai clone, pas de hardlinks)
git clone --no-local "$PRIVATE_REPO" "$TMP_DIR/public"

# 2. Supprimer les chemins sensibles de tout l'historique
git -C "$TMP_DIR/public" filter-repo --invert-paths \
  --path 'docs/legacy' \
  --path 'docs/CONTEXTE_EPREUVE.md' \
  --path 'docs/PROMPTS.md' \
  --path 'docs/PLAN_10_JOURS.md' \
  --path 'docs/CHECKLIST_LIVRAISON.md' \
  --path 'docs/POINTS_DE_VIGILANCE.md' \
  --path 'docs/POINTS_DE_VIGILANCE.pdf' \
  --path 'docs/QUESTIONS_JURY.md' \
  --path 'docs/QUESTIONS_JURY.pdf' \
  --path 'docs/RAPPORT_COMPETENCES_C1_C9.md' \
  --path 'docs/RAPPORT_COMPETENCES_C1_C9.pdf' \
  --path 'docs/RAPPORT_TECHNIQUE_COMPLET.md' \
  --path 'docs/RAPPORT_TECHNIQUE_COMPLET.pdf' \
  --path 'livraison' \
  --path '.claude' \
  --path '.claudeignore'

# 3. Substituer CLAUDE.md et ajouter README.md
cp "$PRIVATE_REPO/scripts/CLAUDE.public.md" "$TMP_DIR/public/CLAUDE.md"
cp "$PRIVATE_REPO/scripts/README.public.md"  "$TMP_DIR/public/README.md"
git -C "$TMP_DIR/public" add CLAUDE.md README.md
git -C "$TMP_DIR/public" \
  -c user.email="lpottier@chapsvision.com" \
  -c user.name="Laurent Pottier" \
  commit -m "docs: guide développeur et README pour le dépôt public"

# 4. Vérification anti-fuite AVANT tout push
LEAK=0
for path in docs/legacy docs/CONTEXTE_EPREUVE.md docs/PROMPTS.md \
            docs/PLAN_10_JOURS.md docs/CHECKLIST_LIVRAISON.md \
            docs/POINTS_DE_VIGILANCE.md docs/POINTS_DE_VIGILANCE.pdf \
            docs/QUESTIONS_JURY.md docs/QUESTIONS_JURY.pdf \
          	docs/RAPPORT_COMPETENCES_C1_C9.md docs/RAPPORT_COMPETENCES_C1_C9.pdf \
	          docs/RAPPORT_TECHNIQUE_COMPLET.md docs/RAPPORT_TECHNIQUE_COMPLET.pdf \
            livraison .claude .claudeignore; do
  count=$(git -C "$TMP_DIR/public" log --all --oneline -- "$path" | wc -l)
  [[ "$count" -gt 0 ]] && { echo "ERREUR FUITE : $path ($count commits)" >&2; LEAK=1; }
done
[[ "$LEAK" -eq 0 ]] || { echo "Push annulé." >&2; exit 1; }

# 5. Push vers le repo public (force requis car les SHAs sont réécrits à chaque run)
git -C "$TMP_DIR/public" remote add public "$PUBLIC_REMOTE"
git -C "$TMP_DIR/public" push public main --force --tags

echo "Synchronisation publique terminée."
