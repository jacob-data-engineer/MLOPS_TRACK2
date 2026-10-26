"""A week unticked on the Auto Training card is left out of training.

`auto_training.excluded` was saved by that card and read by no server
code, so an unticked week_2 trained anyway (reported 2026-10-26).
"""
import inspect
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ppe_pipeline_api import dataset_yaml as dy  # noqa: E402
from ppe_pipeline_api import retained_lists as rl  # noqa: E402
from ppe_pipeline_api import retrain_schedule as rs  # noqa: E402
from ppe_pipeline_api.routers import track2_retention as tr  # noqa: E402


def _inventory(monkeypatch):
    weeks = [{"name": f"week_{i}",
              "train": {f"s{i}_{j}": f"r/week_{i}/images/train/s{i}_{j}.jpg" for j in range(4)},
              "val": {f"v{i}": f"r/week_{i}/images/val/v{i}.jpg"}} for i in (1, 2, 3)]
    monkeypatch.setattr(rl.rt, "read_inventory", lambda root, **kw: weeks)
    monkeypatch.setattr(rl.rt, "plan_inventory", lambda inv, cfg: {"weeks": [
        {"week": w["name"], "stems": sorted(w["train"]), "age_weeks": 0, "fraction": 1.0}
        for w in inv]})


def test_the_list_skips_an_unticked_week(monkeypatch):
    _inventory(monkeypatch)
    out = rl.select(root="r", bucket="b", config={"excluded_weeks": ["week_2"]})
    assert not [p for p in out["train"] if "week_2" in p]
    assert not [p for p in out["val"] if "week_2" in p]
    assert len(out["train"]) == 8 and out["train_available"] == 8


def test_the_other_weeks_keep_what_they_had(monkeypatch):
    _inventory(monkeypatch)
    every = rl.select(root="r", bucket="b", config={})
    some = rl.select(root="r", bucket="b", config={"excluded_weeks": ["week_2"]})
    for week in ("week_1", "week_3"):
        assert [p for p in every["train"] if week in p] == [p for p in some["train"] if week in p]


def test_the_chain_reads_the_card():
    assert rs.excluded_weeks({"auto_training": {"excluded": ["week_2"]}}) == ["week_2"]
    body = inspect.getsource(rs._run_chain_body)
    assert 'retention["excluded_weeks"] = excluded_weeks(config)' in body
    assert "excluded_weeks=retention[\"excluded_weeks\"]" in body


def test_retention_off_skips_the_folder_too(monkeypatch):
    monkeypatch.setattr(dy, "scan_root", lambda root, bucket="": [
        {"week": "week_1", "images_path": "r/week_1/images/train", "labels_path": "r/week_1/labels/train"},
        {"week": "week_2", "images_path": "r/week_2/images/train", "labels_path": "r/week_2/labels/train"}])
    out = rs.pull_train_entries({"retention": {"enabled": False},
                                 "auto_training": {"excluded": ["week_2"]}}, "r", write=False)
    assert [e["images"] for e in out["entries"]] == ["r/week_1/images/train"]


def test_the_week_folders_build_skips_it():
    assert "excluded_weeks=()" in inspect.getsource(dy.build)


def test_the_retention_card_previews_see_it(monkeypatch):
    monkeypatch.setattr(tr.pc, "read_config", lambda: {"auto_training": {"excluded": ["week_2"]}})
    assert tr._with_prefix({})["excluded_weeks"] == ["week_2"]
