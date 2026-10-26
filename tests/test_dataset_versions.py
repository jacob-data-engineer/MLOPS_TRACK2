"""Dataset versions (Option B, 2026-10-26): every queued run trains on a
dataset nobody can change.

Proved against the REAL consumers, not strings: the trainer's download,
label filter and preflight; Ultralytics' own list reader; and the
evaluator's filter over the layout its download produces.
"""
from __future__ import annotations

import inspect
import os
import pathlib
import sys
import types

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "model_training" / "training-container"))
sys.path.insert(0, str(ROOT / "tests"))

from ppe_pipeline_api import dataset_snapshots as ds   # noqa: E402
from ppe_pipeline_api import dataset_versions as dsv   # noqa: E402
from ppe_pipeline_api import dataset_yaml as dy        # noqa: E402
from ppe_pipeline_api import pull_control as pc        # noqa: E402
from ppe_pipeline_api import retrain_schedule as rs    # noqa: E402
from test_review_f1_f4_f5_dataset_contract import FakeS3, _dataset, S  # noqa: E402

train = pytest.importorskip("train")


class FakeTable:
    def __init__(self):
        self.rows = {}

    def put_item(self, Item):
        self.rows[Item["snapshot_id"]] = dict(Item)

    def get_item(self, Key):
        row = self.rows.get(Key["snapshot_id"])
        return {"Item": dict(row)} if row else {}

    def scan(self, **kw):
        items = list(self.rows.values())
        if "FilterExpression" in kw:
            want = kw["ExpressionAttributeValues"][":t"]
            items = [r for r in items if r.get("record_type") == want]
        return {"Items": items}

    def update_item(self, **kw):
        # The app's role has no UpdateItem on this table; using it is a bug.
        raise AssertionError("UpdateItem is not granted on the snapshot table")


@pytest.fixture
def world(monkeypatch):
    table = FakeTable()
    monkeypatch.setattr(ds, "_table", lambda: table)

    def make(files):
        s3 = FakeS3(files)
        monkeypatch.setattr(pc, "s3", s3)
        monkeypatch.setattr(pc, "_list_objects", lambda prefix, bucket="", **kw: [
            {"Key": k, "Size": 1} for (b, k) in sorted(s3.files) if b == (bucket or "home") and k.startswith(prefix)])
        monkeypatch.setattr(pc, "destination_bucket", lambda config=None: "home")
        return s3
    make.table = table
    return make


CLASSES = {0: "helmet"}


def _composed(s3):
    """Testing_Dataset trains; its val plus golden val validate."""
    return dy.write_in_place(
        splits={"train": S("ds/images/train"), "val": S("ds/images/val", "golden_set/images/val")},
        classes=CLASSES, bucket="home", overwrite=True, label_scan=0)


def _download(monkeypatch, s3, tmp_path, uris, yaml_uri):
    monkeypatch.setattr(train.time, "sleep", lambda s: None)
    monkeypatch.setattr(train.boto3, "client", lambda *a, **k: s3)
    dest = tmp_path / "dataset"
    dest.mkdir(exist_ok=True)
    return train.download_dataset_with_progress(uris, "ap-southeast-1", str(dest),
                                                data_yaml_s3_uri=yaml_uri)


def _world_files():
    return {**_dataset("home", "ds"), **_dataset("home", "golden_set", ("val",), "g")}


# --- what a version is -------------------------------------------------------------

def test_a_version_lists_the_exact_images_and_is_written_once(world):
    s3 = world(_world_files())
    out = _composed(s3)
    v = dsv.for_run(home="home", root=out["folder"], train_data_uris=out["download_prefix"],
                    data_yaml_uri=out["uri"], eval_root=out["eval"]["folder"])
    assert (v["train_images"], v["val_images"]) == (1, 2)
    keys = {k for (b, k) in s3.files if k.startswith(v["folder"] + "/")}
    assert keys == {f"{v['folder']}/{n}" for n in (
        "data.yaml", "train/train.txt", "train/val.txt", "eval/data.yaml", "eval/val.txt",
        "manifest.json")}
    assert world.table.rows[v["version_id"]]["record_type"] == "dataset_version"


def test_it_trains_from_the_version_on_the_real_trainer(world, tmp_path, monkeypatch):
    s3 = world(_world_files())
    out = _composed(s3)
    v = dsv.for_run(home="home", root=out["folder"], train_data_uris=out["download_prefix"],
                    data_yaml_uri=out["uri"])
    local = _download(monkeypatch, s3, tmp_path, v["train_data_uris"], v["data_yaml_uri"])
    new_yaml, rep = train.build_filtered_dataset(str(tmp_path / "dataset"), local)
    report = train.preflight_dataset(new_yaml)
    assert report["train"]["images"] == 1 and report["val"]["images"] == 2
    # Ultralytics' OWN list reader resolves every line to a file that exists.
    from ultralytics.data.base import BaseDataset
    doc = yaml.safe_load(open(new_yaml))
    for split, n in (("train", 1), ("val", 2)):
        lst = pathlib.Path(doc["path"]) / doc[split][0]
        files = BaseDataset.get_img_files(types.SimpleNamespace(prefix="", fraction=1.0), str(lst))
        assert len(files) == n and all(os.path.isfile(f) for f in files)


def test_the_evaluator_reads_the_version_in_its_own_layout(world, tmp_path):
    s3 = world(_world_files())
    out = _composed(s3)
    v = dsv.for_run(home="home", root=out["folder"], train_data_uris=out["download_prefix"],
                    data_yaml_uri=out["uri"])
    data = tmp_path / "data"
    # The ProcessingInput: the eval/ prefix, stripped, at data/.
    eb, ek = "home", v["eval_dataset_folder"]   # a key in the app's bucket
    for (b, k), body in list(s3.files.items()):
        if b == eb and k.startswith(ek + "/"):
            p = data / k[len(ek) + 1:]
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(body)
    # fetch_extra_val: every extra folder at data/<key>.
    for uri in v["eval_extra_uris"].split(","):
        b, prefix = uri[5:].split("/", 1)
        for (fb, k), body in list(s3.files.items()):
            if fb == b and k.startswith(prefix):
                p = data / k
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(body)
    sys.path.insert(0, str(ROOT / "model_training" / "custom_training"))
    ev = pytest.importorskip("evaluate_any")
    new_yaml, rep = ev.build_filtered_dataset(str(data), str(data / "data.yaml"),
                                              out_root=str(tmp_path / "eval_filtered"))
    assert rep["images"] == 2


# --- why it exists -------------------------------------------------------------------

def test_a_later_build_cannot_change_a_queued_version(world):
    s3 = world(_world_files())
    out = _composed(s3)
    v = dsv.for_run(home="home", root=out["folder"], train_data_uris=out["download_prefix"],
                    data_yaml_uri=out["uri"])
    frozen = {k: s3.files[k] for k in s3.files if k[1].startswith(v["folder"] + "/")}
    # A different build -- golden only -- rewrites the shared slots.
    dy.write_in_place(splits={"train": S("ds/images/train"), "val": S("ds/images/val")},
                      classes={0: "helmet", 1: "vest"}, bucket="home", overwrite=True, label_scan=0)
    assert {k: s3.files[k] for k in frozen} == frozen


def test_images_arriving_after_queueing_are_not_trained_on(world):
    s3 = world(_world_files())
    out = _composed(s3)
    v = dsv.for_run(home="home", root=out["folder"], train_data_uris=out["download_prefix"],
                    data_yaml_uri=out["uri"])
    s3.files[("home", "ds/images/train/late.jpg")] = b"img"
    lines = s3.files[("home", f"{v['folder']}/train/train.txt")].decode().split()
    assert len(lines) == 1 and not any("late.jpg" in l for l in lines)


def test_an_older_single_folder_yaml_with_a_separate_eval_folder(world):
    s3 = world({**_dataset("home", "tr", ("train",), "t"), **_dataset("home", "ev", ("val",), "e"),
                ("home", "tr/data.yaml"): "path: x\ntrain: images/train\nnames:\n  0: helmet\n",
                ("home", "ev/data.yaml"): "path: x\ntrain: images/val\nval: images/val\nnames:\n  0: helmet\n"})
    v = dsv.for_run(home="home", root="tr", eval_root="ev")
    assert (v["train_images"], v["val_images"]) == (1, 1)


def test_train_and_val_overlapping_is_refused(world):
    s3 = world({**_dataset("home", "tr", ("train",), "t"),
                ("home", "tr/data.yaml"): "train: images/train\nval: images/train\nnames:\n  0: helmet\n"})
    with pytest.raises(ValueError, match="both the training and the validation"):
        dsv.for_run(home="home", root="tr")


def test_a_remap_follows_the_images_into_both_layouts(world):
    s3 = world({**_world_files(), ("home", "ds/data.yaml"): "names:\n  0: person\n  1: helmet\n"})
    out = dy.write_in_place(
        splits={"train": S("ds/images/train"), "val": S("ds/images/val", "golden_set/images/val")},
        classes={0: "helmet", 1: "person"}, bucket="home", overwrite=True, label_scan=0)
    v = dsv.for_run(home="home", root=out["folder"], train_data_uris=out["download_prefix"],
                    data_yaml_uri=out["uri"])
    t = yaml.safe_load(s3.files[("home", f"{v['folder']}/data.yaml")].decode())
    e = yaml.safe_load(s3.files[("home", f"{v['folder']}/eval/data.yaml")].decode())
    assert t["x_label_remap"]["ds/labels/train"] == {0: 1, 1: 0}
    assert e["x_label_remap"]["ds/labels/val"] == {0: 1, 1: 0}


# --- wiring -------------------------------------------------------------------------

def test_snapshot_listings_do_not_show_versions(world):
    world.table.rows["dv-x"] = {"snapshot_id": "dv-x", "record_type": "dataset_version"}
    world.table.rows["snap"] = {"snapshot_id": "snap", "root": "r"}
    assert [r["snapshot_id"] for r in ds.listing()] == ["snap"]


def test_the_chain_freezes_a_version_before_it_hands_over():
    code = inspect.getsource(rs._run_chain_body)
    assert code.index("dsv.for_run(") < code.index("# 3. submit")
    assert '"dataset_version_id": version_id,' in code
    assert "fields = dsv.run_fields(version)" in code


def test_the_manual_queue_add_freezes_one_too():
    from ppe_pipeline_api.routers import track2_models as tm
    code = inspect.getsource(tm)
    assert code.index("dsv.for_run(") < code.index("t2.queue_add(entry)")


def test_queue_again_replays_the_settings_on_the_same_version(world, monkeypatch):
    from ppe_pipeline_api.routers import track2_retention as tr
    s3 = world(_world_files())
    out = _composed(s3)
    v = dsv.for_run(home="home", root=out["folder"], train_data_uris=out["download_prefix"],
                    data_yaml_uri=out["uri"])
    dsv.mark(v["version_id"], entry_template={"epochs": 7, "notes": "first", "custom_classes": True})
    queued = []
    monkeypatch.setattr(tr.t2, "queue_add", lambda e: queued.append(e))
    res = tr.queue_version(v["version_id"], session=None)
    assert queued[0]["epochs"] == 7 and queued[0]["dataset_version_id"] == v["version_id"]
    assert queued[0]["data_yaml_uri"] == v["data_yaml_uri"] and res["queued"] == queued[0]["id"]


def test_the_card_lists_views_and_queues_again():
    page = (ROOT / "web" / "src" / "routes" / "Track2ModelsPage.tsx").read_text(encoding="utf-8")
    card = (ROOT / "web" / "src" / "components" / "Track2DatasetVersionsPanel.tsx").read_text(encoding="utf-8")
    assert "<Track2DatasetVersionsPanel />" in page
    for call in ("listTrack2DatasetVersions", "getTrack2DatasetVersion", "queueTrack2DatasetVersion"):
        assert call in card


def test_marking_uses_only_what_the_app_role_is_granted(world):
    s3 = world(_world_files())
    out = _composed(s3)
    v = dsv.for_run(home="home", root=out["folder"], train_data_uris=out["download_prefix"],
                    data_yaml_uri=out["uri"])
    dsv.mark(v["version_id"], state="queued", entry_template={"epochs": 3, "lr": 0.5})
    row = world.table.rows[v["version_id"]]
    assert row["state"] == "queued" and row["train_images"] == 1
    assert dsv.get(v["version_id"])["entry_template"]["lr"] == 0.5


def test_the_run_start_check_accepts_the_version(world, monkeypatch):
    """The first versioned run was SKIPPED by the queue: its eval folder was
    an s3:// URI and `validate_dataset_folder` reads a key in the app's
    bucket. Run through the real check, not a stub."""
    from ppe_pipeline_api import track2_shared as t2
    s3 = world(_world_files())
    monkeypatch.setattr(t2, "s3", s3)
    monkeypatch.setattr(t2, "BUCKET", "home")
    monkeypatch.setattr(t2, "_s3_object_exists",
                        lambda key, bucket="": ("home", key) in s3.files)
    out = _composed(s3)
    v = dsv.for_run(home="home", root=out["folder"], train_data_uris=out["download_prefix"],
                    data_yaml_uri=out["uri"])
    f = dsv.run_fields(v)
    assert not f["eval_dataset_folder"].startswith("s3://")
    for folder in (f["dataset_folder"], f["eval_dataset_folder"]):
        t2.validate_dataset_folder(folder, used_for_evaluation=True, custom_classes=True)
    # A version stored before the fix is normalised when handed to a run.
    old = {**v, "eval_dataset_folder": f"s3://home/{v['folder']}/eval"}
    assert dsv.run_fields(old)["eval_dataset_folder"] == f"{v['folder']}/eval"


def test_the_queue_card_names_where_the_images_come_from(world):
    s3 = world(_world_files())
    out = _composed(s3)
    v = dsv.for_run(home="home", root=out["folder"], train_data_uris=out["download_prefix"],
                    data_yaml_uri=out["uri"])
    f = dsv.run_fields(v)
    assert f["dataset_source_folders"] == ["ds/images/train"]
    assert f["eval_source_folders"] == ["ds/images/val", "golden_set/images/val"]
    assert (f["dataset_version_train_images"], f["dataset_version_val_images"]) == (1, 2)


def test_an_entry_queued_before_that_is_filled_in(world, monkeypatch):
    from ppe_pipeline_api import track2_shared as t2
    s3 = world(_world_files())
    out = _composed(s3)
    v = dsv.for_run(home="home", root=out["folder"], train_data_uris=out["download_prefix"],
                    data_yaml_uri=out["uri"])
    # As the earlier versions were stored: no folders on the row.
    row = world.table.rows[v["version_id"]]
    row.pop("train_folders"); row.pop("val_folders")
    dsv._card_cache.clear()
    monkeypatch.setattr(t2, "_training_queue", [{"id": "e1", "dataset_version_id": v["version_id"],
                                                 "dataset_folder": v["folder"]}])
    got = t2.queue_snapshot()["pending"][0]
    assert got["dataset_source_folders"] == ["ds/images/train"]
    assert got["dataset_version_val_images"] == 2


def test_the_card_shows_the_version_and_drops_the_folder_caveat():
    page = (ROOT / "web" / "src" / "routes" / "Track2ModelsPage.tsx").read_text(encoding="utf-8")
    assert "Dataset version: <span" in page
    assert "this run trains on the dataset version above" in page


def test_each_folder_is_counted_for_the_queue_card():
    """2026-10-26: the card shows how many images each folder gives."""
    from ppe_pipeline_api import dataset_versions as dsv
    imgs = [("b", "r/week_1/images/train/a.jpg"), ("b", "r/week_1/images/train/c.jpg"),
            ("b", "r/week_2/images/train/d.jpg"), ("o", "x/images/val/e.jpg")]
    assert dsv.folder_counts(imgs, "b") == {
        "r/week_1/images/train": 2, "r/week_2/images/train": 1, "s3://o/x/images/val": 1}
    f = dsv.run_fields({"eval_dataset_folder": "_datasets/v/eval", "folder": "_datasets/v",
                        "train_data_uris": "", "data_yaml_uri": "", "eval_extra_uris": "",
                        "version_id": "v", "bucket": "b",
                        "train_folder_counts": {"r/week_1/images/train": 2}})
    assert f["dataset_source_counts"] == {"r/week_1/images/train": 2}
