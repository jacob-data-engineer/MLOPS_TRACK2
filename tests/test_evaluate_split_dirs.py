"""The evaluator has to read the same data.yaml the trainer reads.

The retrain chain writes its splits as a LIST of week folders. Ultralytics
trains on the union; `split_dirs` only ever checked for a `str`, so it
resolved nothing, and every scheduled retrain failed at EvaluateYolo with
"no images found under images/train or images/val" -- AFTER a successful
seven-minute training step. Training passing and evaluation failing on one
dataset is the signature: two readers, two sets of rules, one file.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "model_training" / "custom_training" / "evaluate_any.py"


@pytest.fixture(scope="module")
def ev():
    """Imported from the file, because it ships to S3 as a file -- the
    thing that runs in the processing job is this exact module."""
    spec = importlib.util.spec_from_file_location("evaluate_any_under_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:                                  # pragma: no cover
        pytest.skip(f"evaluate_any.py needs deps not installed here: {exc}")
    return module


def _week(root: Path, week: str, split: str, names: list) -> None:
    images = root / week / "images" / split
    labels = root / week / "labels" / split
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    for name in names:
        (images / f"{name}.jpg").write_bytes(b"\xff\xd8\xff")
        (labels / f"{name}.txt").write_text("0 0.5 0.5 0.2 0.2\n")


def test_a_list_of_week_folders_resolves_to_all_of_them(ev, tmp_path):
    _week(tmp_path, "week_1", "train", ["a", "b"])
    _week(tmp_path, "week_2", "train", ["c"])
    manifest = {"train": ["week_1/images/train", "week_2/images/train"]}
    got = ev.split_dirs(tmp_path, manifest)
    assert sorted(str(p.relative_to(tmp_path)).replace("\\", "/") for p in got["train"]) == [
        "week_1/images/train", "week_2/images/train",
    ]


def test_the_union_is_what_gets_evaluated(ev, tmp_path):
    """Not the first entry. Scoring a model on one week of a corpus it
    trained on in full reports a number for a dataset that never existed."""
    _week(tmp_path, "week_1", "train", ["a", "b"])
    _week(tmp_path, "week_2", "train", ["c"])
    _week(tmp_path, "week_1", "val", ["v1"])
    manifest = {"train": ["week_1/images/train", "week_2/images/train"],
                "val": ["week_1/images/val"]}
    files = ev.dataset_files(tmp_path, manifest)
    assert len(files) == 4
    assert sum(1 for split, _i, _l in files if split == "train") == 3
    assert sum(1 for split, _i, _l in files if split == "val") == 1
    # Every image found its label by the /images/ -> /labels/ swap, across
    # week folders -- the thing a per-directory pairing gets wrong.
    assert all(label is not None for _s, _i, label in files)


def test_a_week_that_no_longer_exists_is_skipped_not_fatal(ev, tmp_path):
    """Retention retires the oldest week while the yaml still names it."""
    _week(tmp_path, "week_1", "train", ["a"])
    manifest = {"train": ["week_1/images/train", "week_9/images/train"]}
    got = ev.split_dirs(tmp_path, manifest)
    assert len(got["train"]) == 1


def test_a_list_naming_nothing_real_does_NOT_fall_back_silently(ev, tmp_path):
    """The fallback exists for a yaml that named nothing. When SOME of a
    list resolves, swapping in images/<split> would evaluate a different
    dataset from the trained one -- so the fallback is only for none."""
    _week(tmp_path, "week_1", "train", ["a"])
    (tmp_path / "images" / "train").mkdir(parents=True)
    (tmp_path / "images" / "train" / "other.jpg").write_bytes(b"\xff\xd8\xff")
    partly = ev.split_dirs(tmp_path, {"train": ["week_1/images/train", "week_9/images/train"]})
    assert [p.name for p in partly["train"]] == ["train"]
    assert partly["train"][0].parts[-3] == "week_1"
    # Nothing real at all -> the conventional layout.
    none = ev.split_dirs(tmp_path, {"train": ["week_8/images/train"]})
    assert none["train"][0].parts[-3] != "week_1"


def test_a_plain_string_split_still_works(ev, tmp_path):
    """Roboflow-style. The case that worked before must keep working."""
    (tmp_path / "train" / "images").mkdir(parents=True)
    (tmp_path / "train" / "images" / "a.jpg").write_bytes(b"\xff\xd8\xff")
    got = ev.split_dirs(tmp_path, {"train": "../train/images"})
    assert got["train"][0] == tmp_path / "train" / "images"


def test_no_manifest_falls_back_to_the_conventional_layout(ev, tmp_path):
    (tmp_path / "images" / "val").mkdir(parents=True)
    (tmp_path / "images" / "val" / "a.jpg").write_bytes(b"\xff\xd8\xff")
    got = ev.split_dirs(tmp_path, {})
    assert got["val"][0] == tmp_path / "images" / "val"


def test_the_fingerprint_order_does_not_depend_on_the_yaml_order(ev, tmp_path):
    """Re-ordering the week list must not read as new data."""
    _week(tmp_path, "week_1", "train", ["a"])
    _week(tmp_path, "week_2", "train", ["b"])
    one = ev.dataset_files(tmp_path, {"train": ["week_1/images/train", "week_2/images/train"]})
    two = ev.dataset_files(tmp_path, {"train": ["week_2/images/train", "week_1/images/train"]})
    assert [str(i) for _s, i, _l in one] == [str(i) for _s, i, _l in two]


def test_the_real_chain_yaml_shape_is_covered(ev, tmp_path):
    """The exact shape pull control writes, weeks 1/2/3/6/7 -- the one
    that failed in production on 2026-10-06."""
    for week in ("week_1", "week_2", "week_3", "week_6", "week_7"):
        _week(tmp_path, week, "train", [f"{week}_a"])
        _week(tmp_path, week, "val", [f"{week}_v"])
    manifest = {
        "path": "s3://track2-s3/retraining",
        "train": [f"{w}/images/train" for w in ("week_1", "week_2", "week_3", "week_6", "week_7")],
        "val": [f"{w}/images/val" for w in ("week_1", "week_2", "week_3", "week_6", "week_7")],
        "names": {0: "helmet", 2: "person"},
    }
    files = ev.dataset_files(tmp_path, manifest)
    assert len(files) == 10, "every week folder, both splits"


def _list_file(root, name, lines):
    path = root / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_a_split_named_as_a_LIST_FILE_is_read(ev, tmp_path):
    """The regression that followed the week-folder one. A pinned
    snapshot writes `train: train.txt`, Ultralytics reads that as a list
    of image paths, and evaluation only ever looked for directories --
    so the run died with "no images found" after training succeeded on
    the very same file."""
    _week(tmp_path, "week_1", "train", ["a", "b"])
    _list_file(tmp_path, "train.txt",
               ["./week_1/images/train/a.jpg", "./week_1/images/train/b.jpg"])
    files = ev.dataset_files(tmp_path, {"train": "train.txt"})
    assert len(files) == 2
    assert all(split == "train" for split, _i, _l in files)
    # Labels still found by swapping /images/ for /labels/ in each
    # image's OWN path, so a list may cross week folders freely.
    assert all(label is not None for _s, _i, label in files)


def test_a_list_resolves_against_its_own_directory(ev, tmp_path):
    """`get_img_files` expands `./` against the LIST FILE's directory,
    not against `path:`. This is the rule that put a list at
    <root>/_retained/train.txt looking for <root>/_retained/week_1/..."""
    _week(tmp_path, "week_1", "train", ["a"])
    inner = tmp_path / "lists"
    inner.mkdir()
    # The same line, from a list one level down, must NOT resolve.
    _list_file(inner, "train.txt", ["./week_1/images/train/a.jpg"])
    assert ev.dataset_files(tmp_path, {"train": "lists/train.txt"}) == []
    # From the root, it does.
    _list_file(tmp_path, "train.txt", ["./week_1/images/train/a.jpg"])
    assert len(ev.dataset_files(tmp_path, {"train": "train.txt"})) == 1


def test_both_splits_as_lists(ev, tmp_path):
    _week(tmp_path, "week_1", "train", ["a"])
    _week(tmp_path, "week_2", "val", ["v"])
    _list_file(tmp_path, "train.txt", ["./week_1/images/train/a.jpg"])
    _list_file(tmp_path, "val.txt", ["./week_2/images/val/v.jpg"])
    files = ev.dataset_files(tmp_path, {"train": "train.txt", "val": "val.txt"})
    assert sorted(split for split, _i, _l in files) == ["train", "val"]


def test_a_list_naming_a_missing_image_is_skipped(ev, tmp_path):
    """Ultralytics drops a line it cannot read without a word, so a stale
    list must not make the count disagree with what will train."""
    _week(tmp_path, "week_1", "train", ["a"])
    _list_file(tmp_path, "train.txt",
               ["./week_1/images/train/a.jpg", "./week_1/images/train/gone.jpg"])
    assert len(ev.dataset_files(tmp_path, {"train": "train.txt"})) == 1


def test_blank_lines_in_a_list_are_ignored(ev, tmp_path):
    _week(tmp_path, "week_1", "train", ["a"])
    _list_file(tmp_path, "train.txt", ["", "./week_1/images/train/a.jpg", "  "])
    assert len(ev.dataset_files(tmp_path, {"train": "train.txt"})) == 1


def test_a_list_file_suppresses_the_conventional_fallback(ev, tmp_path):
    """A yaml that names a real list has answered the question. Falling
    back to images/<split> because the list resolved no DIRECTORIES would
    evaluate a different dataset from the one trained on."""
    _week(tmp_path, "week_1", "train", ["a"])
    (tmp_path / "images" / "train").mkdir(parents=True)
    (tmp_path / "images" / "train" / "other.jpg").write_bytes(b"\xff\xd8\xff")
    _list_file(tmp_path, "train.txt", ["./week_1/images/train/a.jpg"])
    files = ev.dataset_files(tmp_path, {"train": "train.txt"})
    assert len(files) == 1
    assert "week_1" in str(files[0][1])


def test_directories_and_lists_can_be_mixed(ev, tmp_path):
    """Nothing forbids it, and silently dropping one half would be the
    same class of bug again."""
    _week(tmp_path, "week_1", "train", ["a"])
    _week(tmp_path, "week_2", "train", ["b"])
    _list_file(tmp_path, "train.txt", ["./week_2/images/train/b.jpg"])
    files = ev.dataset_files(tmp_path, {"train": ["week_1/images/train", "train.txt"]})
    assert len(files) == 2
