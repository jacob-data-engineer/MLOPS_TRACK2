"""Login gate: guarded by `PPE_APP_PASSWORD`, `hmac`-compared, fails closed
(an unset/empty env var blocks every login rather than letting anyone in).
A successful login issues a session token (see `session_store.py`) that
doubles as both the bearer credential and the key into this process's
in-memory session state — every other route depends on `require_session`
to validate it.

`MAX_LOGIN_ATTEMPTS` is a soft throttle keyed by client host, since no
session exists to key it on before a token is issued."""

from __future__ import annotations

import hmac
import os
import threading

from fastapi import Header, HTTPException, Request
from pydantic import BaseModel

from .session_store import SESSIONS

MAX_LOGIN_ATTEMPTS = 5

_failed_attempts: dict[str, int] = {}
_failed_attempts_lock = threading.Lock()


class LoginRequest(BaseModel):
    password: str


class LoginResponse(BaseModel):
    token: str


def login(payload: LoginRequest, request: Request) -> LoginResponse:
    expected = os.environ.get("PPE_APP_PASSWORD")
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="This API requires PPE_APP_PASSWORD to be set before it can be used.",
        )

    client_key = request.client.host if request.client else "unknown"
    with _failed_attempts_lock:
        attempts = _failed_attempts.get(client_key, 0)
    if attempts >= MAX_LOGIN_ATTEMPTS:
        raise HTTPException(status_code=429, detail="Too many incorrect attempts.")

    if not hmac.compare_digest(payload.password.encode("utf-8"), expected.encode("utf-8")):
        with _failed_attempts_lock:
            _failed_attempts[client_key] = attempts + 1
        raise HTTPException(status_code=401, detail="Incorrect password.")

    with _failed_attempts_lock:
        _failed_attempts.pop(client_key, None)
    token = SESSIONS.create()
    return LoginResponse(token=token)


def require_session(authorization: str | None = Header(default=None)) -> tuple[str, dict]:
    """Dependency for every non-auth route: validates the bearer token and
    returns `(token, session_dict)` so a route can both key its own state
    and hand the token to session-scoped helpers."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or malformed Authorization header.")
    token = authorization.removeprefix("Bearer ").strip()
    session = SESSIONS.get(token)
    if session is None:
        raise HTTPException(status_code=401, detail="Invalid or expired session token.")
    return token, session
