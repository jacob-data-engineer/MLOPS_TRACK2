"""Which build a container is running.

Two containers now run side by side -- a stable one and a development one,
on different ports against the same tables. That is only safe while the
difference is visible, so this endpoint is what the badge reads.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from ppe_pipeline_api.main import app


def test_build_reports_the_tag_and_role(monkeypatch):
    monkeypatch.setenv("APP_BUILD_TAG", "20260924-034612")
    monkeypatch.setenv("APP_ROLE", "dev")
    monkeypatch.setenv("APP_PORT", "8090")
    got = TestClient(app).get("/build")
    assert got.status_code == 200
    assert got.json() == {"tag": "20260924-034612", "role": "dev", "port": "8090"}


def test_build_needs_no_session():
    """Unauthenticated on purpose.

    It carries no data, and the moment you most need to know which build
    you are looking at is before you have logged into it.
    """
    assert TestClient(app).get("/build").status_code == 200


def test_build_is_empty_rather_than_absent_when_unset(monkeypatch):
    """An older image has no APP_BUILD_TAG; the badge then shows nothing.

    Blank rather than an error, so a container deployed before this
    existed does not render a broken page.
    """
    for name in ("APP_BUILD_TAG", "APP_ROLE", "APP_PORT"):
        monkeypatch.delenv(name, raising=False)
    assert TestClient(app).get("/build").json() == {"tag": "", "role": "", "port": ""}
