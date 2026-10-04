# ── Étape 1 : installation des dépendances ──────────────────────────────────
FROM python:3.12-slim AS builder

# Même version que le poste de développement : uv.lock est au format « revision = 3 »,
# que uv 0.4 ne sait pas lire
RUN pip install --no-cache-dir uv==0.11.29

WORKDIR /app

# Copier uniquement les manifestes pour profiter du cache Docker
COPY pyproject.toml uv.lock ./

# Installer les dépendances (hors code source) dans un venv isolé
# --frozen : le lock fait foi, jamais résolu à nouveau pendant le build
RUN uv sync --frozen --extra api --no-install-project --no-cache

# Maintenant copier le code source et l'installer
COPY src/ ./src/
RUN uv sync --frozen --extra api --no-cache

# ── Étape 2 : image finale allégée ──────────────────────────────────────────
FROM python:3.12-slim AS final

# libgomp1 : runtime OpenMP exigé par LightGBM, absent de l'image slim. Le champion actuel est
# une régression logistique, mais le flow de réentraînement peut promouvoir un LightGBM :
# sans cette bibliothèque, sa désérialisation ferait échouer le démarrage de l'API.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Utilisateur non-root pour limiter la surface d'attaque
RUN groupadd --system appgroup && useradd --system --gid appgroup appuser

WORKDIR /app

# Copier le venv construit dans l'étape builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src

# Volumes attendus (montés via docker-compose) :
#   /app/artifacts/models/  — best_model.pkl et best_model_meta.json (famille, seuil)
RUN mkdir -p /app/artifacts/models && chown -R appuser:appgroup /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "churn_saas.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
