"""Review & Metrics / Data Catalog improvements (2026-11-18): catalog
filtering, measured export progress, export lookup by record, the
read-only dataset availability check, and the additive migration."""
from __future__ import annotations

import threading

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError

from ppe_pipeline_api.data import catalog
from ppe_pipeline_api.data import catalog_view as cv
from test_catalog import _S3Zip  # noqa: E402  (pytest puts tests/ on the path)


# --- filtering: parameterised, Singapore dates, across the whole table ---------------

def test_user_text_is_only_ever_a_bound_parameter():
    hostile = "x'; DROP TABLE training_runs; --"
    where, params = cv._browse_filter("run", search=hostile, status="Completed",
                                      from_date="2026-11-01", to_date="2026-11-18")
    assert hostile not in where and "DROP" not in where
    assert f"%{hostile}%" in params and "Completed" in params
    assert where.count("%s") == len(params)


def test_date_boundaries_are_singapore_calendar_days_inclusive():
    where, params = cv._browse_filter("run", from_date="2026-11-18", to_date="2026-11-18")
    assert "AT TIME ZONE 'Asia/Singapore'" in where
    assert ">= %s::date" in where and "<= %s::date" in where
    assert params == ("2026-11-18", "2026-11-18")


def test_a_bad_date_is_refused_not_ignored():
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        cv._browse_filter("version", from_date="18/11/2026")


def test_search_covers_run_number_id_model_and_dataset():
    where, _ = cv._browse_filter("run", search="v12")
    for column in ("run_id", "run_number", "detector", "architecture", "version_id"):
        assert f"r.{column}" in where


def test_the_filter_reaches_the_count_and_the_page(monkeypatch):
    calls = []
    monkeypatch.setattr(cv, "_rows", lambda sql, params=(): calls.append((sql, params)) or [{"n": 7}])
    cv.runs(25, 50, search="helmet", status="", from_date="", to_date="")
    count, page = calls
    assert " WHERE " in count[0] and " WHERE " in page[0]
    assert count[1] == ("%helmet%",) * 5
    assert page[1][-2:] == (25, 50), "limit and offset reach the query: older records stay reachable"


# --- exports: latest for one record, measured progress, final states -----------------

def test_exports_can_be_narrowed_to_one_record(monkeypatch):
    calls = []
    monkeypatch.setattr(cv, "_rows", lambda sql, params=(): calls.append((sql, params)) or [])
    cv.exports(5, run_id="run-9")
    assert "WHERE run_id = %s" in calls[-1][0] and calls[-1][1] == ("run-9", 5)
    cv.exports(5, version_id="ver-2")
    assert "WHERE version_id = %s" in calls[-1][0]
    cv.exports(5)
    assert "WHERE" not in calls[-1][0]


def _run_build(monkeypatch, entries, *, fmt="zip"):
    s3 = _S3Zip({("source", "a.jpg"): b"image", ("source", "a.txt"): b"0 .5 .5 .1 .1"})
    monkeypatch.setattr(cv.storage, "s3", s3)
    monkeypatch.setattr(cv, "_rows", lambda *a: [{"state": "building"}])
    slots = threading.BoundedSemaphore(1)
    slots.acquire()
    monkeypatch.setattr(cv, "_building", slots)
    results = []
    monkeypatch.setattr(cv, "_finish", lambda *args: results.append(args))
    cv._build("p1", "run", "r", "run-1", entries, ["helmet"], "s3", "dest", "out", fmt)
    return results


def test_progress_names_its_stage_and_counts_real_files(monkeypatch):
    results = _run_build(monkeypatch, [("images/train/a.jpg", "s3://source/a.jpg", {}),
                                      ("labels/train/a.txt", "s3://source/a.txt", {})])
    stages = [r[2][0] for r in results if "SET stage=%s" in r[1]]
    assert stages[0] == "Reading source files"
    assert "Writing metadata and manifest" in stages
    assert "Finalizing ZIP upload" in stages, "packing and finalizing are told apart"
    totals = {r[2][2] for r in results if "SET stage=%s" in r[1]}
    assert totals == {2}, "the total is the files planned, never a guess"
    final = results[-1]
    assert final[2][0] == "ready" and "stage = %s" in final[1]
    assert final[2][6] == "Finished" and final[2][7] == 2


def test_missing_files_finish_as_incomplete_never_ready(monkeypatch):
    results = _run_build(monkeypatch, [("images/train/a.jpg", "s3://source/a.jpg", {}),
                                      ("images/val/gone.jpg", "s3://source/gone.jpg", {})])
    final = results[-1][2]
    assert final[0] == "partial" and final[4] == 1
    assert final[6] == "Finished with missing files"


def test_a_failure_is_recorded_with_its_stage(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("disk full")
    monkeypatch.setattr(cv, "_numbered", boom)
    results = _run_build(monkeypatch, [("images/train/a.jpg", "s3://source/a.jpg", {})])
    assert "state = 'failed'" in results[-1][1] and "stage = 'Failed'" in results[-1][1]
    assert "disk full" in results[-1][2][0]


def test_progress_only_updates_a_building_export():
    import inspect
    src = inspect.getsource(cv._build)
    assert "WHERE id=%s AND state='building'" in src, "a stopped export is never written back to building"


# --- the migration is additive and safe for existing rows --------------------------

def test_migration_9_only_adds_nullable_columns():
    assert catalog.SCHEMA_VERSION == max(catalog.MIGRATIONS) == 9
    for statement in catalog.MIGRATIONS[9]:
        assert statement.startswith("ALTER TABLE exports ADD COLUMN IF NOT EXISTS")
        assert "NOT NULL" not in statement.upper() and "DROP" not in statement.upper()


# --- availability: read-only, and honest about why ---------------------------------

class _ReadOnlyS3:
    """Only head_object exists: any copy, put or delete would raise."""

    def __init__(self, etags, error=None):
        self.etags, self.error = etags, error

    def head_object(self, Bucket, Key):
        if self.error:
            raise self.error
        if (Bucket, Key) not in self.etags:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        return {"ETag": self.etags[(Bucket, Key)]}


STORE = "s3://b/central/objects/replay/"


def _version(**over):
    v = {"version_id": "v1", "version_number": 4, "train_images": 1, "val_images": 0,
         "central_recipe": {"bucket": "b", "folder": "_training_inputs/" + "0" * 32 + "/dataset",
                            "store_uri": "s3://b/central"},
         "file_provenance": [{"central_image": STORE + "i.jpg", "central_image_etag": '"I"',
                              "central_label": STORE + "l.txt", "central_label_etag": '"L"'}]}
    v.update(over)
    return v


def _availability(monkeypatch, version, s3, members=None, run_version="v1"):
    from ppe_pipeline_api.data import dataset_versions, central_recipe, prepared_training
    monkeypatch.setattr(dataset_versions, "get", lambda vid: version)
    monkeypatch.setattr(central_recipe.storage, "s3", s3)
    monkeypatch.setattr(prepared_training, "verify_members", members or (lambda v: None))
    monkeypatch.setattr(cv, "_rows", lambda sql, params=(): [{"version_id": run_version}])
    return cv.dataset_availability("run", "run-1")


def test_ready_when_every_preserved_file_matches(monkeypatch):
    got = _availability(monkeypatch, _version(), _ReadOnlyS3({("b", "central/objects/replay/i.jpg"): '"I"',
                                                             ("b", "central/objects/replay/l.txt"): '"L"'}))
    assert got["state"] == "ready" and got["version_number"] == 4
    assert "checked separately" in got["detail"]


def test_a_changed_file_is_unavailable(monkeypatch):
    got = _availability(monkeypatch, _version(), _ReadOnlyS3({("b", "central/objects/replay/i.jpg"): '"OTHER"',
                                                             ("b", "central/objects/replay/l.txt"): '"L"'}))
    assert got["state"] == "unavailable" and got["problem"] == "files"


def test_a_missing_file_is_unavailable(monkeypatch):
    got = _availability(monkeypatch, _version(), _ReadOnlyS3({}))
    assert got["state"] == "unavailable" and "missing or changed" in got["detail"]


def test_an_incomplete_sql_list_is_unavailable(monkeypatch):
    def short(v):
        raise ValueError("The SQL dataset file list is incomplete")
    got = _availability(monkeypatch, _version(), _ReadOnlyS3({}), members=short)
    assert got["state"] == "unavailable" and got["problem"] == "sql_membership"


def test_access_denied_is_a_failed_check_not_missing_files(monkeypatch):
    denied = ClientError({"Error": {"Code": "AccessDenied"}}, "HeadObject")
    got = _availability(monkeypatch, _version(), _ReadOnlyS3({}, error=denied))
    assert got["state"] == "check_failed" and "access was refused" in got["detail"]
    assert "not evidence that files are missing" in got["detail"]


def test_a_network_failure_is_a_failed_check(monkeypatch):
    down = EndpointConnectionError(endpoint_url="https://s3")
    got = _availability(monkeypatch, _version(), _ReadOnlyS3({}, error=down))
    assert got["state"] == "check_failed"


def test_a_record_without_a_recipe_is_legacy(monkeypatch):
    got = _availability(monkeypatch, _version(central_recipe=None), _ReadOnlyS3({}))
    assert got["state"] == "legacy"


def test_a_run_without_a_version_is_legacy(monkeypatch):
    got = _availability(monkeypatch, _version(), _ReadOnlyS3({}), run_version=None)
    assert got["state"] == "legacy"


def test_a_catalog_outage_is_a_failed_check(monkeypatch):
    def down(*a, **k):
        raise ConnectionError("db unreachable")
    monkeypatch.setattr(cv, "_rows", down)
    got = cv.dataset_availability("run", "run-1")
    assert got["state"] == "check_failed"


def test_the_availability_route_exists():
    from ppe_pipeline_api.main import app
    paths = app.openapi()["paths"]
    assert "/track2/catalog/dataset-availability" in paths
    params = {p["name"] for p in paths["/track2/catalog/exports"]["get"]["parameters"]}
    assert {"run_id", "version_id"} <= params
    params = {p["name"] for p in paths["/track2/catalog/runs"]["get"]["parameters"]}
    assert {"search", "status", "from_date", "to_date", "limit", "offset"} <= params


# --- the page: what it must and must not show (static checks) ----------------------

import pathlib  # noqa: E402

WEB = pathlib.Path(__file__).resolve().parents[1] / "web" / "src"
PAGE = (WEB / "routes" / "Track2MetricsPage.tsx").read_text(encoding="utf-8")
CATALOG = (WEB / "routes" / "Track2CatalogPage.tsx").read_text(encoding="utf-8")
META = (WEB / "components" / "Track2CatalogMetadata.tsx").read_text(encoding="utf-8")
PROGRESS = (WEB / "components" / "Track2ExportProgress.tsx").read_text(encoding="utf-8")


def test_review_and_metrics_first_and_the_catalog_last():
    assert PAGE.index('id="comparison"') < PAGE.index('id="model-export"') < PAGE.index('id="data-catalog"')
    assert PAGE.rstrip().endswith("}") and PAGE.index("<Track2CatalogPage") > PAGE.index("<Track2MetricsExportCard")


def test_jump_links_reach_every_section():
    for target in ("comparison", "model-export", "data-catalog", "catalog-export", "catalog-metadata"):
        assert f'id: "{target}"' in PAGE
    assert 'id="catalog-export"' in META and 'id="catalog-metadata"' in META


def test_refresh_all_reaches_the_catalog_and_keeps_selection():
    assert "Refresh all" in PAGE and "setCatalogRefresh" in PAGE
    assert "refreshKey={catalogRefresh}" in PAGE
    assert "refreshKey={refreshKey}" in CATALOG, "the selected record and its exports refresh too"


def test_the_exports_tab_and_removed_sections_stay_gone():
    assert '"exports"' not in CATALOG and "catalog-export-history" not in CATALOG
    for gone in ("Files in Run", "Queue same run again", "getTrack2CatalogRunFiles"):
        assert gone not in CATALOG and gone not in META


def test_no_second_sticky_bar_and_the_toolbar_clears_the_banner():
    assert "sticky top-0 z-10" not in CATALOG
    assert "var(--stale-banner-h, 0px)" in PAGE
    layout = (WEB / "components" / "Layout.tsx").read_text(encoding="utf-8")
    assert '"--stale-banner-h"' in layout


def test_failures_are_never_shown_as_empty_lists():
    assert "setVersions([])" not in CATALOG and "setExportsList([])" not in CATALOG
    assert 'status: "failed"' in CATALOG and "Retry" in CATALOG
    assert "runsRequest.current" in CATALOG and "versionsRequest.current" in CATALOG, "stale answers dropped"


def test_filters_go_to_the_server_with_load_more():
    assert "getTrack2CatalogRuns(offset, filter, limit)" in CATALOG
    assert "Load ${Math.min(PAGE" in CATALOG
    assert "From (SGT)" in CATALOG


def test_the_two_export_workflows_are_named_apart():
    card = (WEB / "components" / "Track2MetricsExportCard.tsx").read_text(encoding="utf-8")
    assert "Model export" in card and "Manual" in card and "Auto" in card
    assert "Catalog export: the selected run or dataset version" in META


def test_an_old_export_is_labelled_as_previous():
    assert "Previous export -- made with different options" in PROGRESS
    assert "Download previous ZIP" in META
    for field in ("Source record", "Contents", "Destination", "Created", "Export ID"):
        assert field in PROGRESS


def test_progress_is_measured_never_estimated():
    import re
    shown = re.sub(r"/\*.*?\*/|//[^\n]*", "", PROGRESS, flags=re.S)   # what renders, not the comments
    for word in ("ETA", "MB/s", "remaining", "estimated"):
        assert word not in shown
    assert "processed_files" in PROGRESS and "updated_at" in PROGRESS
    assert 'exp.state === "ready"' in PROGRESS, "ready only when the server says so"
    assert "Stop export" in PROGRESS


def test_metadata_says_whose_it_is():
    assert "Selected catalog record:" in META and "Opened external file:" in META
    assert "was closed because" in META


def test_availability_is_read_only_and_explains_itself():
    for state in ("Not checked", "Checking", "Ready to rebuild", "Unavailable", "Legacy record", "Check failed"):
        assert state in META
    assert "Model-class compatibility is checked separately" in META


def test_one_timestamp_format():
    fmt = (WEB / "lib" / "format.ts").read_text(encoding="utf-8")
    assert "export function formatSgt" in fmt and "SGT (GMT+8)" in fmt
    for src in (CATALOG, META, PROGRESS, PAGE):
        assert "formatSgt" in src


def test_migrations_are_serialised_and_recording_one_twice_is_harmless():
    import inspect
    src = inspect.getsource(catalog.ensure_schema)
    assert "pg_advisory_lock" in src
    assert src.count("ON CONFLICT (version) DO NOTHING") == 2


def test_the_new_columns_are_migrated_before_they_are_read_or_written(monkeypatch):
    """Deployed 2026-11-18: the export list 400'd with "processed_files does
    not exist" -- in PostgreSQL mode nothing had run migration 9 yet."""
    import inspect
    order = []
    monkeypatch.setattr(catalog, "ensure_schema_once", lambda: order.append("migrate"))
    monkeypatch.setattr(cv, "_rows", lambda *a: order.append("read") or [])
    cv.exports(5)
    assert order == ["migrate", "read"]
    assert "catalog.ensure_schema_once()" in inspect.getsource(cv.start_export)
