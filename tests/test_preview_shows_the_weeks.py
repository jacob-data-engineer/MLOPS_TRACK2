"""The Assemble preview shows Pull Control's weeks, as the run uses them.

Asked for on 2026-10-26. The preview named the Assemble folders and the
golden set, and left out the week folders the run also trains on. One
function now names the weeks for both, so they cannot drift apart.
"""
import inspect
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppe_pipeline_api import retrain_schedule as rs  # noqa: E402
from ppe_pipeline_api import retained_lists as rl  # noqa: E402
from ppe_pipeline_api import dataset_yaml as dy  # noqa: E402
from ppe_pipeline_api.routers import track2_models as tm  # noqa: E402


def test_retention_on_names_the_list_and_counts_the_selection(monkeypatch):
    monkeypatch.setattr(rl, "select", lambda **kw: {
        "train": ["a"] * 7, "train_available": 10,
        "weeks": [{"week": "week_1", "keep": 3, "available": 4},
                  {"week": "week_2", "keep": 4, "available": 6},
                  {"week": "week_3", "keep": 0, "available": 0}]})
    monkeypatch.setattr(rl, "write", lambda **kw: (_ for _ in ()).throw(AssertionError("wrote")))
    out = rs.pull_train_entries({"retention": {"enabled": True}}, "retraining/_retained", write=False)
    assert out["entries"] == [{"images": "retraining/_retained/train.txt", "labels": ""}]
    assert out["retained"]["images"] == 7 and out["retained"]["available"] == 10
    # Every week the list draws from, by folder; an empty week is left out.
    assert out["retained"]["weeks"] == [
        {"folder": "retraining/_retained/week_1", "keep": 3, "available": 4},
        {"folder": "retraining/_retained/week_2", "keep": 4, "available": 6}]


def test_retention_off_names_every_week_whole(monkeypatch):
    monkeypatch.setattr(dy, "scan_root", lambda root, bucket="": [
        {"week": "week_2", "images_path": f"{root}/week_2/images/train",
         "labels_path": f"{root}/week_2/labels/train"},
        {"week": "week_3", "images_path": f"{root}/week_3/images/train",
         "labels_path": f"{root}/week_3/labels/train"}])
    out = rs.pull_train_entries({"retention": {"enabled": False}}, "r", write=False)
    assert [e["images"] for e in out["entries"]] == ["r/week_2/images/train", "r/week_3/images/train"]
    assert out["retained"] is None


def test_the_chain_uses_the_same_function_and_writes():
    src = inspect.getsource(rs._run_chain_body)
    assert "pull_train_entries(config, root, write=True)" in src


def test_the_preview_route_takes_it_and_the_write_route_does_not():
    assert "include_pull" in tm.DatasetYamlInPlaceRequest.model_fields
    preview = inspect.getsource(tm.dataset_yaml_preview)
    write = inspect.getsource(tm.dataset_yaml_write)
    assert "rs.pull_train_entries(config, root, write=False)" in preview
    assert "include_pull" not in write
