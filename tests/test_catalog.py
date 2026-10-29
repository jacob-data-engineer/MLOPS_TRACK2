"""The central data catalog, step 1 (2026-10-29): its tables and routes."""
from __future__ import annotations

import pathlib

from ppe_pipeline_api.data import catalog

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_nothing_in_the_schema_drops_or_deletes():
    text = " ".join(catalog.SCHEMA).upper()
    for word in ("DROP ", "DELETE ", "TRUNCATE"):
        assert word not in text
    assert all("IF NOT EXISTS" in s for s in catalog.SCHEMA)


def test_the_split_and_decay_live_on_the_version_not_the_file():
    members = next(s for s in catalog.SCHEMA if "dataset_members" in s and "CREATE TABLE" in s)
    for column in ("split", "included", "reason", "decay_weight", "label_id"):
        assert column in members
    files = next(s for s in catalog.SCHEMA if "TABLE IF NOT EXISTS files" in s)
    assert "split" not in files


def test_labels_are_versioned_not_overwritten():
    labels = next(s for s in catalog.SCHEMA if "TABLE IF NOT EXISTS labels" in s)
    assert "UNIQUE (sha256, version)" in labels and "class_names" in labels


def test_unconfigured_says_so_rather_than_failing(monkeypatch):
    monkeypatch.setattr(catalog, "HOST", "")
    assert catalog.status() == {"configured": False, "reachable": False, "tables": {}}


def test_the_routes_and_the_deploy_wiring_exist():
    from ppe_pipeline_api.main import app
    paths = app.openapi()["paths"]
    assert "/track2/catalog/status" in paths and "/track2/catalog/schema" in paths
    deploy = (ROOT / "deploy.sh").read_text(encoding="utf-8")
    assert "-e CATALOG_HOST='$CATALOG_HOST'" in deploy
    tf = (ROOT / "terraform" / "app" / "catalog.tf").read_text(encoding="utf-8")
    assert "deletion_protection        = true" in tf and "publicly_accessible    = false" in tf


def test_the_connection_is_verified_against_the_shipped_rds_roots():
    import inspect
    assert catalog.CA_BUNDLE.exists()
    assert catalog.CA_BUNDLE.read_text().count("BEGIN CERTIFICATE") > 50
    source = inspect.getsource(catalog.connect)
    assert "cafile=str(CA_BUNDLE)" in source and "CERT_NONE" not in source
