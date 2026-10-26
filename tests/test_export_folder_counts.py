"""Review & metrics' automatic export shows how many .pt files sit in its
folder, and how many in archive/ (asked for, 2026-10-26)."""
import inspect
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ppe_pipeline_api import metrics_export as mx  # noqa: E402
from ppe_pipeline_api.routers import track2_metrics as tm  # noqa: E402

CARD = " ".join((pathlib.Path(__file__).resolve().parents[1] / "web" / "src" / "components"
                 / "Track2MetricsExportCard.tsx").read_text(encoding="utf-8").split())


def test_counts_the_folder_and_its_archive_with_the_same_rule(monkeypatch):
    seen = []
    monkeypatch.setattr(mx, "export_siblings",
                        lambda bucket, prefix, group: seen.append(prefix) or ["a"] * (3 if "archive" in prefix else 1))
    got = mx.export_counts("b", "exports/helmet", "yolo11s-helmet-detector")
    assert (got["in_folder"], got["archived"]) == (1, 3)
    assert seen == ["exports/helmet", "exports/helmet/archive"]


def test_the_status_route_carries_them():
    assert '"counts": counts' in inspect.getsource(tm.get_auto_export)


def test_the_card_words_them_for_the_rule_in_force():
    assert "In the folder" in CARD
    assert 'watched.retention === "archive_oldest" && (' in CARD
    assert "archived in" in CARD


def test_the_newest_file_names_the_version_in_the_folder(monkeypatch):
    """By the stamp in the name -- v9 written after v10 is the one in use."""
    monkeypatch.setattr(mx, "export_siblings", lambda bucket, prefix, group: [] if "archive" in prefix else [
        "f/yolo11-s-helmet-v10-20261020-090000.pt", "f/yolo11-s-helmet-v9-20261026-120000.pt"])
    got = mx.export_counts("b", "f", "yolo11s-helmet-detector")["latest"]
    assert got["version"] == "9" and got["stamp"] == "20261026-120000"


def test_the_route_scores_it_and_names_the_best():
    src = inspect.getsource(tm.get_auto_export)
    assert "mx.group_versions(plan[\"group_name\"], max_age=60)" in src
    assert 'counts["best"]' in src


def test_the_card_shows_both_and_flags_a_thin_score():
    assert "Version in the folder" in CARD
    assert "Best by {watchedCriterionLabel}" in CARD
    assert "too few for that score to mean much" in CARD
