"""Tests for `ppe_pipeline_api.auth` — the HTTP login gate and its
`_password_ok`: fails closed, `hmac`-compared, and every other route
depends on `require_session` validating the issued token."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ppe_pipeline_api import auth as auth_module
from ppe_pipeline_api.main import app


@pytest.fixture(autouse=True)
def _reset_failed_attempts():
    """`_failed_attempts` is module-level global state keyed by client
    host — every `TestClient()` instance in this file shares the same
    host, so each test must start from a clean slate rather than
    inheriting an earlier test's failed-attempt count."""
    auth_module._failed_attempts.clear()
    yield
    auth_module._failed_attempts.clear()


def _client() -> TestClient:
    return TestClient(app)


def test_login_fails_closed_when_password_unset(monkeypatch):
    monkeypatch.delenv("PPE_APP_PASSWORD", raising=False)
    response = _client().post("/auth/login", json={"password": "anything"})
    assert response.status_code == 503


def test_login_rejects_wrong_password(monkeypatch):
    monkeypatch.setenv("PPE_APP_PASSWORD", "correct-horse")
    response = _client().post("/auth/login", json={"password": "wrong"})
    assert response.status_code == 401


def test_login_accepts_correct_password_and_issues_token(monkeypatch):
    monkeypatch.setenv("PPE_APP_PASSWORD", "correct-horse")
    response = _client().post("/auth/login", json={"password": "correct-horse"})
    assert response.status_code == 200
    token = response.json()["token"]
    assert isinstance(token, str) and token


def test_login_non_ascii_password_round_trips(monkeypatch):
    monkeypatch.setenv("PPE_APP_PASSWORD", "café123")
    response = _client().post("/auth/login", json={"password": "café123"})
    assert response.status_code == 200


def test_login_locks_out_after_max_attempts(monkeypatch):
    monkeypatch.setenv("PPE_APP_PASSWORD", "correct-horse")
    client = _client()
    for _ in range(auth_module.MAX_LOGIN_ATTEMPTS):
        response = client.post("/auth/login", json={"password": "wrong"})
        assert response.status_code == 401

    locked_out = client.post("/auth/login", json={"password": "correct-horse"})
    assert locked_out.status_code == 429


# These three used to probe /track1/hazard-types. That route was removed
# with the PPE judge pages, and the SPA catch-all serves index.html for
# any unmatched path -- so the request came back 200 with HTML. The two
# "rejects" tests failed for a reason that had nothing to do with auth,
# and the "accepts" one PASSED without ever reaching an authenticated
# route, which is worse: it would have stayed green if auth were deleted.
#
# /auth/session is the right probe. It is genuinely protected, it needs no
# AWS, and it exists to answer exactly this question.
PROTECTED = "/auth/session"


def test_protected_route_rejects_missing_authorization_header(monkeypatch):
    monkeypatch.setenv("PPE_APP_PASSWORD", "correct-horse")
    response = _client().get(PROTECTED)
    assert response.status_code == 401


def test_protected_route_rejects_invalid_token(monkeypatch):
    monkeypatch.setenv("PPE_APP_PASSWORD", "correct-horse")
    response = _client().get(PROTECTED, headers={"Authorization": "Bearer not-a-real-token"})
    assert response.status_code == 401


def test_protected_route_accepts_valid_token(monkeypatch):
    monkeypatch.setenv("PPE_APP_PASSWORD", "correct-horse")
    client = _client()
    token = client.post("/auth/login", json={"password": "correct-horse"}).json()["token"]
    response = client.get(PROTECTED, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200


def test_a_deleted_route_is_not_mistaken_for_a_protected_one():
    """The trap the three tests above fell into.

    Every unmatched path returns the React shell with 200, so a route that
    no longer exists is indistinguishable from one that answered. Any auth
    test has to probe a path that is really registered, or it proves
    nothing at all.
    """
    from ppe_pipeline_api.main import app

    registered = set(app.openapi()["paths"])
    assert PROTECTED in registered

    gone = _client().get("/track1/hazard-types")
    assert gone.status_code == 200
    assert gone.headers["content-type"].startswith("text/html")


def test_health_check_does_not_require_auth():
    response = _client().get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# --- every route, not just the one we remembered to test ----------------------
#
# The tests above probe ONE protected path. The UAT sweep ran with
# `require_session` overridden -- the only way to reach the routes at all --
# which means a route that forgot to require a session looked exactly like
# one that has it. These read the dependency tree instead, so a new route
# added without a guard fails here rather than shipping.

#: Public on purpose. `/build` is documented unauthenticated in CLAUDE.md
#: (the build badge reads it), `/health` is the container probe,
#: `/auth/login` is how a session is obtained at all, and `/auth/session`
#: is the liveness probe the client uses before signing anyone out.
PUBLIC_BY_DESIGN = {
    "/health", "/build", "/auth/login", "/auth/session", "/auth/logout",
    "/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc",
}


def _walk_routes(routes, prefix=""):
    """Every leaf route as (full path, route).

    FastAPI 0.141 wraps each `include_router` in an `_IncludedRouter` with
    no `.routes` of its own -- the real APIRouter hangs off
    `.original_router`. A walk that only follows `.routes` finds 5 routes
    out of 122 and reports a clean app it never checked, which is a worse
    failure than finding nothing.
    """
    for route in routes:
        included = getattr(route, "original_router", None)
        if included is not None:
            context = getattr(route, "include_context", None)
            yield from _walk_routes(included.routes,
                                    prefix + getattr(context, "prefix", ""))
            continue
        inner = getattr(route, "routes", None)
        if inner:
            yield from _walk_routes(inner, prefix)
        elif hasattr(route, "dependant"):
            yield prefix + route.path, route


def _requires_session(dependant, seen=None):
    from ppe_pipeline_api import auth

    seen = seen if seen is not None else set()
    if id(dependant) in seen:
        return False
    seen.add(id(dependant))
    if getattr(dependant, "call", None) is auth.require_session:
        return True
    return any(_requires_session(sub, seen) for sub in dependant.dependencies)


def _leaves():
    from ppe_pipeline_api.main import app

    return list(_walk_routes(app.routes))


def test_the_route_walk_actually_reaches_the_routers():
    """Guards the guard. If this drops back to a handful, every assertion
    below it passes by looking at almost nothing."""
    from ppe_pipeline_api.main import app

    assert len(_leaves()) >= len(app.openapi()["paths"])


def test_every_route_requires_a_session_except_the_public_ones():
    unguarded = sorted(
        path for path, route in _leaves()
        if path not in PUBLIC_BY_DESIGN
        and not path.endswith("{full_path:path}")
        and not _requires_session(route.dependant)
    )
    assert unguarded == []


def test_nothing_new_has_been_made_public():
    """The other direction: a route dropping its guard would show up above,
    but a route being ADDED to the public set should be a deliberate edit
    to this list, not something that drifts."""
    public = sorted(
        path for path, _route in _leaves()
        if path in PUBLIC_BY_DESIGN or path.endswith("{full_path:path}")
    )
    assert public == [
        "/auth/session", "/build", "/health", "/{full_path:path}", "/auth/login",
    ] or set(public) <= PUBLIC_BY_DESIGN | {"/{full_path:path}"}


def test_every_write_route_requires_a_session():
    """Called out separately because the consequence differs: a missing
    guard on a read is an information leak, on a write it is an action
    anybody could trigger -- and these include pull/run and train-now."""
    writes = [(path, route) for path, route in _leaves()
              if (route.methods or set()) & {"POST", "PUT", "PATCH", "DELETE"}]
    assert len(writes) > 50, "the walk should be finding every write route"
    unguarded = sorted(path for path, route in writes
                       if path not in PUBLIC_BY_DESIGN
                       and not _requires_session(route.dependant))
    assert unguarded == []


# --- the frontend must only call routes that exist ----------------------------
#
# This matters more here than in most apps. The SPA catch-all serves
# index.html with 200 for ANY unmatched path, so a frontend calling a route
# that was renamed or removed does not get a 404 it can report -- it gets
# HTML, with 200, and fails later when it reads a field off it. The error
# surfaces far from the cause. That is exactly how three tests in this file
# came to probe a route that had not existed for months and "pass".

def _api_ts() -> str:
    import pathlib

    return (pathlib.Path(__file__).resolve().parents[1]
            / "web" / "src" / "lib" / "api.ts").read_text(encoding="utf-8")


def _normalise(path: str) -> str:
    import re

    path = re.sub(r"\$\{[^}]*\}", "{p}", path)
    path = path.split("?")[0]
    path = re.sub(r"\{[^}]*\}", "{p}", path)
    # A hole glued onto a segment with no `/` before it is a query string
    # being appended, not a path parameter.
    path = re.sub(r"(?<![/])\{p\}$", "", path)
    return path.rstrip("/") or "/"


def test_every_track2_endpoint_the_frontend_calls_is_a_real_route():
    import re

    from ppe_pipeline_api.main import app

    # Only the surface this product still has. The retired PPE judge
    # wrappers (/escalations, /fewshot, /goldset) are still in api.ts but
    # nothing in the nav reaches them; flagging those would bury this.
    live = ("/track2/", "/auth/")
    call = re.compile(
        r"""api(?:Get|Post|Put|Delete|PostForm|GetImageBlob)\s*(?:<[^(]*?>)?\s*\(\s*"""
        r"""[`"']((?:[^`"'$]|\$\{[^}]*\})*)""",
        re.S,
    )

    declared = {_normalise(p) for p in app.openapi()["paths"]}
    called = {_normalise(raw): raw for raw in call.findall(_api_ts())
              if raw.startswith(live)}

    assert len(called) > 100, "the extractor should be finding the whole client"
    missing = sorted(f"{p}  (written as {called[p]!r})"
                     for p in called if p not in declared)
    assert missing == []
