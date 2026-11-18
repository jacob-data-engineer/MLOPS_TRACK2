"""Compare -> best -> export, with the run that made it (2026-11-18).

Every compared or best version shows its run number and can open that run
in the Data Catalog; Model export can send the best version's .pt with the
run's metadata embedded."""
import pathlib

import pytest
from fastapi import HTTPException

from ppe_pipeline_api.routers import track2_metrics as router

WEB = pathlib.Path(__file__).resolve().parents[1] / "web" / "src"
ARN = "arn:aws:sagemaker:x:1:model-package/yolo11s-helmet-detector/32"


def _setup(monkeypatch, runs, exported):
    from ppe_pipeline_api.data import catalog_view
    rows = [{"version": "32", "model_package_arn": ARN}]
    monkeypatch.setattr(router.mx, "group_versions", lambda group, **k: rows)
    monkeypatch.setattr(router.mx, "pick_best", lambda rows, criterion: (rows[0], ""))
    monkeypatch.setattr(router.mx, "describe_pick", lambda best, criterion: {"version": best["version"], "model_package_arn": ARN})
    monkeypatch.setattr(catalog_view, "runs_by_model", lambda: runs)
    monkeypatch.setattr(catalog_view, "export_run_weights",
                        lambda run_id, **kw: exported.append((run_id, kw)) or {"uri": "s3://dest/run-58.pt"})


def _request(**over):
    body = {"group_name": "yolo11s-helmet-detector", "criterion": "mAP50", "destination": "s3",
            "bucket": "dest", "prefix": "out", "include_metadata": True,
            "metadata_groups": ["Identity", "Evaluation"], **over}
    return router.WeightsExportRequest(**body)


def test_the_best_version_goes_out_as_its_run_with_metadata(monkeypatch):
    exported = []
    _setup(monkeypatch, {ARN: {"run_number": 58, "run_id": "run-58"}}, exported)
    out = router.export_best_weights(_request(), session=None)
    assert exported == [("run-58", {"destination": "s3", "bucket": "dest", "prefix": "out",
                                    "metadata_groups": ["Identity", "Evaluation"]})]
    assert out["run_number"] == 58 and out["picked"]["version"] == "32"
    assert out["written_to"] == "s3://dest/run-58.pt"


def test_a_version_with_no_run_is_refused_not_sent_plain(monkeypatch):
    exported = []
    _setup(monkeypatch, {}, exported)
    with pytest.raises(HTTPException, match="no recorded training run"):
        router.export_best_weights(_request(), session=None)
    assert not exported


def test_no_categories_is_refused(monkeypatch):
    exported = []
    _setup(monkeypatch, {ARN: {"run_number": 58, "run_id": "run-58"}}, exported)
    with pytest.raises(HTTPException, match="at least one metadata category"):
        router.export_best_weights(_request(metadata_groups=[]), session=None)


def test_without_the_tick_the_plain_export_is_unchanged(monkeypatch):
    called = []
    monkeypatch.setattr(router.mx, "run_export", lambda *a, **k: called.append(k) or {"ok": True})
    router.export_best_weights(_request(include_metadata=False), session=None)
    assert called and "metadata_groups" not in called[0]


def test_the_choice_is_saved_in_the_preset():
    fields = router.ExportSettings.model_fields
    assert "include_metadata" in fields and "metadata_groups" in fields


def test_run_numbers_and_the_way_to_the_run_are_on_the_page():
    page = (WEB / "routes" / "Track2MetricsPage.tsx").read_text(encoding="utf-8")
    card = (WEB / "components" / "Track2MetricsExportCard.tsx").read_text(encoding="utf-8")
    picker = (WEB / "components" / "Track2ModelPicker.tsx").read_text(encoding="utf-8")
    catalog = (WEB / "routes" / "Track2CatalogPage.tsx").read_text(encoding="utf-8")
    assert "getTrack2RunsByModel()" in page and "runLabel(run)" in page
    assert "Open run in catalog" in page and "Open run in catalog" in card
    assert "openRun={openRun}" in page and "openRun?.nonce" in catalog
    assert "Run ${runs[r.model_package_arn].run_number}" in picker
    assert "Include the training run&apos;s metadata in the .pt" in card
    assert "METADATA_CATEGORIES" in card
