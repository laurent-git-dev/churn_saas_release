"""Sécurité de l'API : authentification par clé, limitation de débit, taille de corps."""

from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import Annotated

from fastapi import Header, HTTPException, Request, status
from fastapi.responses import JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

# Historique des timestamps par IP pour la fenêtre glissante
_historique: dict[str, list[float]] = defaultdict(list)


def _get_api_key() -> str:
    """Lit la clé d'API depuis l'environnement à chaque appel (testable via monkeypatch)."""
    return os.environ.get("CHURN_API_KEY", "")


def _get_rate_limit() -> int:
    """Lit la limite de débit (requêtes/minute) depuis l'environnement."""
    return int(os.environ.get("CHURN_RATE_LIMIT", "60"))


def _get_max_body_bytes() -> int:
    """Lit la taille maximale du corps de requête (octets) depuis l'environnement."""
    return int(os.environ.get("CHURN_MAX_BODY_BYTES", str(1 * 1024 * 1024)))


async def verifier_cle_api(
    x_api_key: Annotated[str | None, Header()] = None,
) -> None:
    """Dépendance FastAPI — vérifie la présence et la validité de l'en-tête X-API-Key.

    Si CHURN_API_KEY n'est pas définie, l'authentification est désactivée (mode dev/CI).
    """
    cle_configuree = _get_api_key()
    if not cle_configuree:
        return
    if x_api_key is None or x_api_key != cle_configuree:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Clé d'API manquante ou invalide.",
            headers={"WWW-Authenticate": "ApiKey"},
        )


async def verifier_rate_limit(request: Request) -> None:
    """Dépendance FastAPI — limite à CHURN_RATE_LIMIT requêtes par minute par IP.

    Implémentation fenêtre glissante en mémoire.
    Suffisant pour un déploiement single-instance ; un Redis serait nécessaire en multi-pod.
    """
    ip = request.client.host if request.client else "unknown"
    limite = _get_rate_limit()
    maintenant = time.monotonic()
    fenetre = 60.0

    # Nettoyage des entrées hors fenêtre
    _historique[ip] = [t for t in _historique[ip] if maintenant - t < fenetre]

    if len(_historique[ip]) >= limite:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Quota dépassé : {limite} requêtes/minute maximum par IP.",
            headers={"Retry-After": "60"},
        )
    _historique[ip].append(maintenant)


class LimiteCorpsMiddleware(BaseHTTPMiddleware):
    """Middleware ASGI — rejette avec 413 les corps dépassant CHURN_MAX_BODY_BYTES."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        max_octets = _get_max_body_bytes()
        longueur = request.headers.get("content-length")
        if longueur and int(longueur) > max_octets:
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={"detail": f"Corps trop volumineux — maximum {max_octets // 1024} Kio."},
            )
        return await call_next(request)
