"""Typing a page URL must give you the page, not an API error.

Ten of the SPA's routes are spelled exactly like the API routers behind
them -- `/track2/models` is both the page and its data. The routers are
registered first and FastAPI takes the first match, so the catch-all that
serves `index.html` never ran for them. Clicking through the nav always
worked, because React Router never asks the server; only a reload, a
bookmark or a typed URL hit this, which is why it survived so long.

Checked on CONTENT-TYPE, never on the status alone: this API answers 200
with the React shell for any unmatched path, so a 200 on its own proves
nothing about whether a route exists.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ppe_pipeline_api.main import app  # noqa: E402

# Every SPA route that an API router also claims.
COLLIDING = [
    "/track2/models",
    "/track2/metrics",
    "/track2/deploy",
    "/track2/eval",
    "/track2/export",
    "/track2/cost",
    "/track2/schedule",
    "/track2/lineage",
    "/track2/notifications",
    "/track2/review",
]

# SPA routes with no API router of the same name -- these always worked,
# and are here so a future router added under one of them fails loudly.
NO_COLLISION = ["/track2/help", "/track2/synthetic", "/track2/multi-auto-training"]

BROWSER = {"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}


@pytest.fixture
def client():
    return TestClient(app)


class TestABrowserGetsThePage:
    @pytest.mark.parametrize("path", COLLIDING + NO_COLLISION)
    def test_every_page_url_serves_the_app(self, client, path):
        res = client.get(path, headers=BROWSER)
        assert res.status_code == 200, path
        # The content-type is the real assertion. A JSON 401 and the React
        # shell are both "a response"; only one of them is the page.
        assert res.headers["content-type"].startswith("text/html"), path

    def test_the_root_still_serves_the_app(self, client):
        res = client.get("/", headers=BROWSER)
        assert res.status_code == 200
        assert res.headers["content-type"].startswith("text/html")

    def test_a_deep_link_that_matches_nothing_still_serves_the_app(self, client):
        res = client.get("/track2/models/anything/deeper", headers=BROWSER)
        assert res.status_code == 200
        assert res.headers["content-type"].startswith("text/html")


class TestTheAppsOwnCallsAreUnaffected:
    """`fetch` with no Accept set sends */*, which must still get JSON.

    If this ever starts returning HTML, the SPA's 401 handling breaks
    silently: it reads `body.detail` to decide whether the session ended,
    and an HTML body makes every expired session look like a network
    blip instead.
    """

    @pytest.mark.parametrize("path", COLLIDING)
    def test_an_unauthenticated_api_call_still_gets_a_json_401(self, client, path):
        res = client.get(path, headers={"Accept": "*/*"})
        assert res.status_code == 401, path
        assert res.headers["content-type"].startswith("application/json"), path
        assert "detail" in res.json(), path

    def test_an_explicit_json_accept_gets_json(self, client):
        res = client.get("/track2/models", headers={"Accept": "application/json"})
        assert res.status_code == 401
        assert res.headers["content-type"].startswith("application/json")

    def test_no_accept_header_at_all_gets_json(self, client):
        # httpx sends */* by default; this pins that the fallback is JSON
        # rather than HTML for anything that is not asking for a page.
        res = client.get("/track2/models")
        assert res.status_code == 401
        assert res.headers["content-type"].startswith("application/json")


class TestItIsAuthOnly:
    """Only 401 and 403 become the page.

    A genuine API 404 or 409 that silently turned into the React shell
    would be far harder to debug than the bug this fixes -- the caller
    would get a 200 full of HTML and no hint that anything went wrong.

    Called directly rather than over a route, because every authenticated
    route raises 401 BEFORE it can reach its own 404 -- so there is no
    request that reaches this handler with a 404 on a real path.
    """

    @staticmethod
    def _handler():
        from fastapi import HTTPException

        from ppe_pipeline_api.main import app

        handler = app.exception_handlers.get(HTTPException)
        assert handler is not None, "the handler is not registered"
        return handler

    @pytest.mark.parametrize("status", [404, 409, 422, 500])
    def test_other_errors_keep_their_status_and_json(self, status):
        import asyncio

        from fastapi import HTTPException
        from starlette.requests import Request

        scope = {
            "type": "http", "method": "GET", "path": "/track2/models",
            "headers": [(b"accept", b"text/html")], "query_string": b"",
        }
        res = asyncio.run(self._handler()(Request(scope), HTTPException(status, "nope")))
        assert res.status_code == status
        assert res.media_type == "application/json"

    @pytest.mark.parametrize("status", [401, 403])
    def test_auth_errors_become_the_page(self, status):
        import asyncio

        from fastapi import HTTPException
        from starlette.requests import Request

        scope = {
            "type": "http", "method": "GET", "path": "/track2/models",
            "headers": [(b"accept", b"text/html")], "query_string": b"",
        }
        res = asyncio.run(self._handler()(Request(scope), HTTPException(status, "nope")))
        assert res.status_code == 200
        assert str(res.media_type or "").startswith("text/html")
