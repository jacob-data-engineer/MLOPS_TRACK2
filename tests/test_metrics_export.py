"""Exporting the registry's metrics as a CSV.

ONE builder for both destinations. A download and an S3 copy of the same
selection have to be byte-identical, or "I exported it" stops meaning one
thing -- and the obvious shortcut, letting the browser build the download
from what the page happens to be showing while the server builds the S3
copy from the registry, is exactly how two formats of one export come to
exist.
"""
from __future__ import annotations

import csv
import importlib
import io

import pytest

mod = importlib.import_module("src.ppe_pipeline_api.routers.track2_metrics")

ROWS = [
    {"group_name": "yolo11s-helmet-detector", "registry": "helmet",
     "generation": "11", "size": "s", "version": "4", "status": "Approved",
     "model_package_arn": "arn:...:model-package/yolo11s-helmet-detector/4",
     "metrics": {"mAP50": 0.995, "mAP50-95": 0.872, "precision": 0.982,
                 "recall": 1.0, "f1": 0.991, "accuracy": None,
                 "classes": ["helmet"],
                 "dataset": {"validation_count": 5,
                             "dataset_version": "sha256-abc"}}},
    {"group_name": "yolo11s-helmet-detector", "registry": "helmet",
     "generation": "11", "size": "s", "version": "1", "status": "Approved",
     "model_package_arn": "arn:...:model-package/yolo11s-helmet-detector/1",
     "metrics": None},
]


@pytest.fixture(autouse=True)
def registry(monkeypatch):
    monkeypatch.setattr(mod.t2, "scan_all_model_metrics", lambda: [dict(r) for r in ROWS])


def parse(text):
    return list(csv.reader(io.StringIO(text)))


def test_every_selected_version_is_a_row():
    text, count = mod.build_metrics_csv([], [])
    rows = parse(text)
    assert count == 2
    assert len(rows) == 3, "header plus two versions"


def test_only_the_requested_versions():
    text, count = mod.build_metrics_csv(
        ["arn:...:model-package/yolo11s-helmet-detector/4"], [])
    assert count == 1
    assert parse(text)[1][4] == "4"


def test_only_the_requested_metrics():
    text, _ = mod.build_metrics_csv([], ["mAP50"])
    header = parse(text)[0]
    assert "mAP50" in header
    assert "recall" not in header and "precision" not in header


def test_the_column_order_is_fixed_not_the_callers():
    """Two exports of the same metrics must diff cleanly, so the order
    comes from METRIC_KEYS rather than from however the request listed
    them."""
    a, _ = mod.build_metrics_csv([], ["recall", "mAP50"])
    b, _ = mod.build_metrics_csv([], ["mAP50", "recall"])
    assert parse(a)[0] == parse(b)[0]
    assert parse(a)[0].index("mAP50") < parse(a)[0].index("recall")


def test_an_unevaluated_version_exports_BLANK_not_zero():
    """A version that was never evaluated is not one that scored zero,
    and a spreadsheet cannot tell the difference once a 0 is written."""
    text, _ = mod.build_metrics_csv([], ["mAP50"])
    v1 = [r for r in parse(text)[1:] if r[4] == "1"][0]
    assert v1[parse(text)[0].index("mAP50")] == ""


def test_the_context_columns_are_always_there():
    """The reason this is a review export and not a leaderboard: a 0.995
    measured on five images is not a better model than a 0.400 measured
    on two hundred, and a CSV sorted on mAP50 alone says it is."""
    header = parse(mod.build_metrics_csv([], ["mAP50"])[0])[0]
    for column in ("validation_images", "classes", "dataset_version",
                   "model_package_arn"):
        assert column in header, column


def test_no_metrics_selected_is_refused():
    with pytest.raises(ValueError, match="at least one metric"):
        mod.build_metrics_csv([], ["not-a-metric"])


def test_a_selection_matching_nothing_is_refused():
    """Rather than writing a header-only file to S3 and reporting
    success."""
    with pytest.raises(ValueError, match="No versions matched"):
        mod.build_metrics_csv(["arn:nothing"], [])


def test_both_destinations_build_the_same_bytes(monkeypatch):
    """The whole point of one builder."""
    written = {}
    monkeypatch.setattr(mod.t2, "s3", type("S3", (), {
        "put_object": staticmethod(lambda **kw: written.update(kw) or {}),
    })())
    downloaded = mod.export_metrics(
        mod.MetricsExportRequest(metrics=["mAP50"]), session=None)
    mod.export_metrics(
        mod.MetricsExportRequest(metrics=["mAP50"], bucket="b", prefix="x/y"),
        session=None)
    assert downloaded["csv"].encode("utf-8") == written["Body"]


def test_the_s3_key_sits_under_the_chosen_prefix(monkeypatch):
    written = {}
    monkeypatch.setattr(mod.t2, "s3", type("S3", (), {
        "put_object": staticmethod(lambda **kw: written.update(kw) or {}),
    })())
    out = mod.export_metrics(
        mod.MetricsExportRequest(bucket="track2-s3", prefix="/exports/"),
        session=None)
    assert written["Bucket"] == "track2-s3"
    assert written["Key"].startswith("exports/track2-metrics-")
    assert written["Key"].endswith(".csv")
    assert out["written_to"] == f"s3://track2-s3/{written['Key']}"
    assert out["csv"] == "", "the text is not echoed back on an S3 write"


def test_the_bucket_root_is_a_valid_destination(monkeypatch):
    written = {}
    monkeypatch.setattr(mod.t2, "s3", type("S3", (), {
        "put_object": staticmethod(lambda **kw: written.update(kw) or {}),
    })())
    mod.export_metrics(mod.MetricsExportRequest(bucket="b"), session=None)
    assert written["Key"].startswith("track2-metrics-")


def test_no_bucket_hands_the_text_back():
    out = mod.export_metrics(mod.MetricsExportRequest(), session=None)
    assert out["written_to"] == ""
    assert out["csv"].startswith("group,registry,")
    assert out["filename"].endswith(".csv")
