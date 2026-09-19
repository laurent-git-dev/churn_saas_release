# ── Étape 1 : installation des dépendances ──────────────────────────────────
FROM python:3.12-slim AS builder

RUN pip install --no-cache-dir uv==0.4.29

WORKDIR /app

# Copier uniquement les manifestes pour profiter du cache Docker
COPY pyproject.toml uv.lock ./

# Installer les dépendances (hors code source) dans un venv isolé
RUN uv sync --extra api --no-install-project --no-cache

# Maintenant copier le code source et l'installer
COPY src/ ./src/
RUN uv sync --extra api --no-cache

# ── Étape 2 : image finale allégée ──────────────────────────────────────────
FROM python:3.12-slim AS final

# Utilisateur non-root pour limiter la surface d'attaque
RUN groupadd --system appgroup && useradd --system --gid appgroup appuser

WORKDIR /app

# Copier le venv construit dans l'étape builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src

# Volumes attendus (montés via docker-compose) :
#   /app/artifacts/models/  — contient best_model.pkl
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
