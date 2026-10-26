"""A frozen snapshot has to be trainable as frozen.

"Train on this" restored the snapshot's yaml and reported success. When
that yaml named WEEK FOLDERS -- which is what the chain writes whenever
retention is off -- the restored definition said
`train: [week_1/images/train, ...]`, so the run trained on whatever those
folders held that minute. For a corpus still being pulled into, that is
not the dataset anybody froze. It looked broken while reporting success,
which is worse than failing.

The manifest held every file name the whole time. It was simply never
turned back into something a run could read.
"""
from __future__ import annotations

import importlib

import pytest
import yaml

ds = importlib.import_module("src.ppe_pipeline_api.dataset_snapshots")

FILES = [
    "week_1/images/train/a.jpg",
    "week_1/images/val/v1.jpg",
    "week_1/labels/train/a.txt",
    "week_1/labels/val/v1.txt",
    "week_2/images/train/b.jpg",
    "week_2/labels/train/b.txt",
]


def _fake_s3(**methods):
    def missing(name):
        def refuse(**kwargs):
            pytest.fail(f"unexpected s3.{name}({kwargs.get('Key')})")
        return refuse
    for name in ("put_object", "copy_object", "get_object"):
        methods.setdefault(name, missing(name))
    return type("FakeS3", (), {k: staticmethod(v) for k, v in methods.items()})()


def test_images_are_split_by_their_own_path():
    got = ds.lists_from_manifest(FILES)
    assert got["train"] == ["./week_1/images/train/a.jpg", "./week_2/images/train/b.jpg"]
    assert got["val"] == ["./week_1/images/val/v1.jpg"]


def test_labels_are_not_listed():
    """Ultralytics finds a label by swapping /images/ for /labels/ in the
    image's own path. A label in the list is an image it cannot read."""
    got = ds.lists_from_manifest(FILES)
    assert not any(".txt" in line for lines in got.values() for line in lines)


def test_every_line_carries_the_leading_dot_slash():
    """`get_img_files` expands `./` against THE LIST FILE'S OWN
    directory. Without it a line is read relative to the process's
    working directory, which is not the download root."""
    got = ds.lists_from_manifest(FILES)
    assert all(line.startswith("./") for lines in got.values() for line in lines)


def test_the_lists_live_beside_data_yaml_not_in_a_subfolder():
    """The same trap from the other side: a list at
    <root>/_retained/train.txt holding ./week_1/... resolves to
    <root>/_retained/week_1/..., which does not exist. It cost a run."""
    rl = importlib.import_module("src.ppe_pipeline_api.retained_lists")
    assert rl.FOLDER == ""


def test_the_order_is_stable():
    """Two restores of one snapshot must produce identical lists, or the
    dataset fingerprint moves and a repeat is a different run."""
    assert ds.lists_from_manifest(FILES) == ds.lists_from_manifest(list(reversed(FILES)))


def test_a_custom_images_subpath_is_honoured():
    files = ["week_1/frames/train/a.jpg", "week_1/images/train/b.jpg"]
    got = ds.lists_from_manifest(files, images_subpath="frames")
    assert got["train"] == ["./week_1/frames/train/a.jpg"]


def test_rubbish_entries_are_skipped_not_fatal():
    got = ds.lists_from_manifest(["", "week_1", "week_1/images", "a/b/c/d.jpg"])
    assert got["train"] == [] and got["val"] == []


def test_an_empty_manifest_gives_empty_lists():
    assert ds.lists_from_manifest([]) == {"train": [], "val": []}
    assert ds.lists_from_manifest(None) == {"train": [], "val": []}


def test_the_rewritten_yaml_names_the_lists():
    original = ("path: s3://b/retraining\n"
                "train:\n- week_1/images/train\n"
                "val:\n- week_1/images/val\n"
                "names:\n  0: helmet\n")
    doc = yaml.safe_load(ds._yaml_naming_lists(original, "train.txt", "val.txt"))
    assert doc["train"] == "train.txt"
    assert doc["val"] == "val.txt"
    # Everything else survives -- the taxonomy above all, since a run
    # scored against the wrong names is worse than one that fails.
    assert doc["names"] == {0: "helmet"}
    assert doc["path"] == "s3://b/retraining"


def test_the_rewritten_yaml_says_what_it_is():
    got = ds._yaml_naming_lists("train:\n- w/images/train\n", "train.txt", "val.txt")
    assert "pinned by name" in got


def test_a_yaml_that_is_not_a_mapping_does_not_crash():
    got = ds._yaml_naming_lists("- just\n- a\n- list\n", "train.txt", "val.txt")
    assert yaml.safe_load(got)["train"] == "train.txt"


def test_restore_pins_when_the_snapshot_has_no_lists(monkeypatch):
    """The whole fix, end to end."""
    row = {
        "snapshot_id": "s1", "root": "retraining", "bucket": "b", "lists": {},
        "yaml_key": "snap/data.yaml", "manifest_key": "snap/manifest.json",
        "live_key": "retraining/data.yaml", "download_prefix": "s3://b/retraining/",
        "manifest_uri": "s3://b/snap/manifest.json", "selected": 1638,
        "names_from": "folder",
    }
    monkeypatch.setattr(ds, "get", lambda sid, with_yaml=True: row)
    monkeypatch.setattr(ds, "_manifest_files", lambda r: FILES)
    monkeypatch.setattr(ds, "_read_text",
                        lambda bucket, key: "train:\n- week_1/images/train\nnames:\n  0: helmet\n")
    written: dict = {}
    monkeypatch.setattr(ds.pc, "s3", _fake_s3(
        put_object=lambda **kw: written.update({kw["Key"]: kw["Body"]}) or {},
    ))
    got = ds.restore("s1")
    assert got["names_from"] == "list"
    assert got["pinned_from_manifest"] is True
    assert got["selected"] == 3, "two train images and one val"
    assert "retraining/train.txt" in written
    assert "retraining/val.txt" in written
    assert b"./week_1/images/train/a.jpg" in written["retraining/train.txt"]
    # And the live yaml now names the lists rather than the folders.
    assert yaml.safe_load(written["retraining/data.yaml"])["train"] == "train.txt"


def test_restore_still_copies_the_lists_a_snapshot_already_had(monkeypatch):
    """The retention path is unchanged: when the yaml already named
    lists, the archived ones are what go back."""
    row = {
        "snapshot_id": "s2", "root": "retraining", "bucket": "b",
        "lists": {"train": {"key": "snap/train.txt",
                            "live_key": "retraining/train.txt", "lines": 5}},
        "yaml_key": "snap/data.yaml", "manifest_key": "snap/manifest.json",
        "live_key": "retraining/data.yaml", "download_prefix": "s3://b/retraining/",
        "manifest_uri": "s3://b/snap/manifest.json", "selected": 5,
        "names_from": "list",
    }
    monkeypatch.setattr(ds, "get", lambda sid, with_yaml=True: row)
    copied: list = []
    monkeypatch.setattr(ds.pc, "s3", _fake_s3(
        copy_object=lambda **kw: copied.append(kw["Key"]) or {},
    ))
    got = ds.restore("s2")
    assert got["pinned_from_manifest"] is False
    assert copied == ["retraining/train.txt", "retraining/data.yaml"]


def test_an_unreadable_manifest_falls_back_rather_than_failing(monkeypatch):
    """The live slot must still get a yaml. Failing outright leaves the
    previous definition in place with an error nobody can act on."""
    row = {
        "snapshot_id": "s3", "root": "retraining", "bucket": "b", "lists": {},
        "yaml_key": "snap/data.yaml", "manifest_key": "snap/missing.json",
        "live_key": "retraining/data.yaml", "download_prefix": "s3://b/retraining/",
        "manifest_uri": "", "selected": 0, "names_from": "folder",
    }
    monkeypatch.setattr(ds, "get", lambda sid, with_yaml=True: row)

    def gone(**kwargs):
        raise RuntimeError("no such key")

    copied: list = []
    monkeypatch.setattr(ds.pc, "s3", _fake_s3(
        get_object=gone,
        copy_object=lambda **kw: copied.append(kw["Key"]) or {},
    ))
    got = ds.restore("s3")
    assert got["pinned_from_manifest"] is False
    assert "retraining/data.yaml" in copied


def test_the_train_endpoint_restores_before_it_runs():
    """Order matters: running first would train on whatever the card had
    selected, while reporting the snapshot's id.

    The restore itself lives in `restore_snapshot` now -- one path, so
    the two endpoints cannot disagree about what a restore records -- so
    what is checked here is that the call comes first."""
    import inspect
    router = importlib.import_module("src.ppe_pipeline_api.routers.track2_retention")
    source = inspect.getsource(router.train_on_snapshot)
    assert source.index("restore_snapshot(") < source.index("rs.run_chain")
    # Queued, not started: the account has one GPU, and a direct start
    # fails outright with SAGEMAKER_RESOURCE_LIMIT rather than waiting.
    assert 'submit="queue"' in source


def test_the_train_endpoint_points_the_config_at_the_restored_root():
    """Via the restore it delegates to, which is where the write now is.
    Asserted across BOTH functions so moving it again cannot quietly
    drop it."""
    import inspect
    router = importlib.import_module("src.ppe_pipeline_api.routers.track2_retention")
    train = inspect.getsource(router.train_on_snapshot)
    restore = inspect.getsource(router.restore_snapshot)
    assert '"default_dataset"' in restore
    assert "restore_snapshot(" in train
    assert train.index("restore_snapshot(") < train.index("rs.run_chain")


def test_restore_records_which_definition_is_now_live():
    """The card reads `default_dataset`, not the bucket. Without this the
    restore moved the yaml and nothing else knew: the box went on saying
    "frozen as <the chain's snapshot>" over a slot holding a different
    one, so loading looked like it had done nothing -- and named the
    wrong dataset while the run would have used the right one."""
    import inspect
    router = importlib.import_module("src.ppe_pipeline_api.routers.track2_retention")
    source = inspect.getsource(router.restore_snapshot)
    assert '"default_dataset"' in source
    assert '"set_by": "snapshot"' in source
    assert source.index("ds.restore") < source.index('"default_dataset"')


def test_a_failed_selection_does_not_fail_a_restore_that_happened():
    """The yaml IS back. Reporting failure would be a lie, and would stop
    the caller reading the result it needs."""
    import inspect
    router = importlib.import_module("src.ppe_pipeline_api.routers.track2_retention")
    source = inspect.getsource(router.restore_snapshot)
    assert "selection_error" in source


def test_the_train_endpoint_refuses_when_the_selection_failed():
    """Opposite call, same fact: the chain is about to READ
    `default_dataset`, so a failed write there means it would assemble
    whatever the card last had."""
    import inspect
    router = importlib.import_module("src.ppe_pipeline_api.routers.track2_retention")
    source = inspect.getsource(router.train_on_snapshot)
    assert "selection_error" in source
    assert source.index("selection_error") < source.index("rs.run_chain")


def test_the_train_endpoint_reuses_the_restore_endpoint():
    """One path, so the two can never disagree about what a restore
    records."""
    import inspect
    router = importlib.import_module("src.ppe_pipeline_api.routers.track2_retention")
    source = inspect.getsource(router.train_on_snapshot)
    assert "restore_snapshot(" in source


def test_restore_reports_when_the_snapshot_was_frozen(monkeypatch):
    """Not when it was put back. Those are different dates, and the first
    is the only one that tells one frozen dataset from another -- a
    snapshot taken on 30 September was reading as 'at 2026-10-06 06:59'
    because the card showed the restore time."""
    row = {
        "snapshot_id": "s4", "root": "retraining", "bucket": "b", "lists": {},
        "created_at": "2026-09-30T02:53:00+00:00", "note": "after-pull retrain",
        "yaml_key": "snap/data.yaml", "manifest_key": "snap/manifest.json",
        "live_key": "retraining/data.yaml", "download_prefix": "s3://b/retraining/",
        "manifest_uri": "s3://b/snap/manifest.json", "selected": 1496,
        "names_from": "folder",
    }
    monkeypatch.setattr(ds, "get", lambda sid, with_yaml=True: row)
    monkeypatch.setattr(ds, "_manifest_files", lambda r: FILES)
    monkeypatch.setattr(ds, "_read_text", lambda bucket, key: "train:\n- w/images/train\n")
    monkeypatch.setattr(ds.pc, "s3", _fake_s3(put_object=lambda **kw: {}))
    got = ds.restore("s4")
    assert got["created_at"] == "2026-09-30T02:53:00+00:00"
    assert got["note"] == "after-pull retrain"


def test_the_unpinned_path_reports_it_too(monkeypatch):
    """Both return paths, or the date appears only for some snapshots."""
    row = {
        "snapshot_id": "s5", "root": "retraining", "bucket": "b",
        "lists": {"train": {"key": "k", "live_key": "lk", "lines": 1}},
        "created_at": "2026-09-29T05:37:00+00:00", "note": "",
        "yaml_key": "snap/data.yaml", "manifest_key": "snap/manifest.json",
        "live_key": "retraining/data.yaml", "download_prefix": "s3://b/retraining/",
        "manifest_uri": "", "selected": 5, "names_from": "list",
    }
    monkeypatch.setattr(ds, "get", lambda sid, with_yaml=True: row)
    monkeypatch.setattr(ds.pc, "s3", _fake_s3(copy_object=lambda **kw: {}))
    assert ds.restore("s5")["created_at"] == "2026-09-29T05:37:00+00:00"


def test_the_selection_carries_the_freeze_date_separately_from_the_restore_time():
    import inspect
    router = importlib.import_module("src.ppe_pipeline_api.routers.track2_retention")
    source = inspect.getsource(router.restore_snapshot)
    assert '"snapshot_created_at"' in source
    # And `at` -- the restore time -- is still there, because the two
    # answer different questions.
    assert '"at":' in source


def test_the_live_or_frozen_choice_is_part_of_the_saved_settings():
    """A settings file that restored the folders, the detector and the
    epochs but not this would reproduce the wrong one of two runs that
    look identical on paper."""
    pc = importlib.import_module("src.ppe_pipeline_api.pull_control")
    assert pc.DEFAULTS["data_source"] == "live"
    router = importlib.import_module("src.ppe_pipeline_api.routers.track2_pull_control")
    assert "data_source" in router.ConfigRequest.model_fields


def test_a_settings_file_round_trips_the_choice():
    """It rides in the config blob, so Save and Load carry it with
    everything else -- the point of putting it there rather than in the
    browser."""
    pc = importlib.import_module("src.ppe_pipeline_api.pull_control")
    body = dict(pc.DEFAULTS)
    body["data_source"] = "snapshot"
    assert dict(pc.DEFAULTS, **body)["data_source"] == "snapshot"
    # And a file written before the key existed loads as "live" rather
    # than as undefined, which the radio could not render.
    older = {k: v for k, v in pc.DEFAULTS.items() if k != "data_source"}
    assert {**pc.DEFAULTS, **older}["data_source"] == "live"


def test_the_selection_records_whether_the_run_reads_a_pinned_list():
    """"Which images will this train on" is not a question that should
    expire with a React state variable. The note set by the click is gone
    on the next reload; this is read every time the card renders."""
    import inspect
    router = importlib.import_module("src.ppe_pipeline_api.routers.track2_retention")
    source = inspect.getsource(router.restore_snapshot)
    assert '"pinned"' in source
    assert "pinned_from_manifest" in source
    # Either route to a pinned list counts: one built from the manifest
    # here, or one the snapshot already carried.
    assert 'names_from") == "list"' in source
