"""What a run consumed, kept where it cannot move afterwards."""

from __future__ import annotations

import pytest

from ppe_pipeline_api import run_artifacts as ra


LIST_YAML = """\
path: s3://track2-s3/retraining
train: _retained/train.txt
val: _retained/val.txt
names:
  0: helmet
"""

FOLDER_YAML = """\
path: s3://track2-s3/retraining
train:
  - week_1/images/train
  - week_2/images/train
val:
  - week_1/images/val
names:
  0: helmet
"""

TRAIN_LINES = """\
./week_1/images/train/a.jpg
./week_2/images/train/b.jpg
./week_7/images/train/c.jpg
"""

VAL_LINES = "./week_1/images/val/d.jpg\n"


class S3:
    def __init__(self, objects: dict):
        self.objects = dict(objects)
        self.copies: list = []

    def get_object(self, *, Bucket, Key):
        if Key not in self.objects:
            raise KeyError(Key)

        class Body:
            def __init__(self, blob):
                self.blob = blob

            def read(self):
                return self.blob

        return {"Body": Body(self.objects[Key].encode("utf-8"))}

    def copy_object(self, *, Bucket, Key, CopySource, **_kw):
        if CopySource["Key"] not in self.objects:
            raise KeyError(CopySource["Key"])
        self.copies.append((CopySource["Key"], Key))
        self.objects[Key] = self.objects[CopySource["Key"]]


@pytest.fixture
def listed(monkeypatch):
    fake = S3({
        "retraining/data.yaml": LIST_YAML,
        "retraining/_retained/train.txt": TRAIN_LINES,
        "retraining/_retained/val.txt": VAL_LINES,
    })
    monkeypatch.setattr(ra.pc, "s3", fake)
    monkeypatch.setattr(ra.pc, "destination_bucket", lambda: "track2-s3")
    return fake


@pytest.fixture
def foldered(monkeypatch):
    fake = S3({"retraining/data.yaml": FOLDER_YAML})
    monkeypatch.setattr(ra.pc, "s3", fake)
    monkeypatch.setattr(ra.pc, "destination_bucket", lambda: "track2-s3")
    return fake


def test_the_selection_is_read_from_the_lists(listed):
    """These lists ARE what the run consumes, so they are what the audit
    should record -- not a recursive listing of a root that holds far
    more."""
    found = ra.read_selection("retraining")
    assert found["images"] == [
        "week_1/images/train/a.jpg",
        "week_2/images/train/b.jpg",
        "week_7/images/train/c.jpg",
        "week_1/images/val/d.jpg",
    ]
    assert found["yaml_key"] == "retraining/data.yaml"
    assert set(found["lists"]) == {"train", "val"}


def test_labels_are_named_by_the_same_swap_ultralytics_makes(listed):
    """A .txt IS the supervision the run learned from. Inferring it any
    other way would record a file the run did not read."""
    found = ra.read_selection("retraining")
    assert found["labels"] == [
        "week_1/labels/train/a.txt",
        "week_2/labels/train/b.txt",
        "week_7/labels/train/c.txt",
        "week_1/labels/val/d.txt",
    ]


def test_a_folder_list_yaml_yields_no_selection(foldered):
    """A list of week folders means "everything under here", which the
    recursive listing already describes. Returning names here would be
    claiming a selection that was never made."""
    found = ra.read_selection("retraining")
    assert found["images"] == []
    assert found["lists"] == {}
    assert found["yaml_key"] == "retraining/data.yaml"


def test_the_artifacts_are_copied_not_referenced(listed):
    """The live slots are overwritten by the next Save. A pointer to them
    answers "what does that path hold now", which is a different question
    from "what did this run train on"."""
    out = ra.capture("abc123", "retraining")
    assert out["yaml_key"] == "_runs/abc123/data.yaml"
    assert out["train_list_key"] == "_runs/abc123/train.txt"
    assert out["val_list_key"] == "_runs/abc123/val.txt"
    assert out["train_list_lines"] == 3
    assert out["val_list_lines"] == 1
    assert listed.copies == [
        ("retraining/data.yaml", "_runs/abc123/data.yaml"),
        ("retraining/_retained/train.txt", "_runs/abc123/train.txt"),
        ("retraining/_retained/val.txt", "_runs/abc123/val.txt"),
    ]

    # And the copy survives the live slot being rewritten.
    listed.objects["retraining/_retained/train.txt"] = "./week_9/images/train/z.jpg\n"
    assert listed.objects["_runs/abc123/train.txt"] == TRAIN_LINES


def test_run_folders_are_outside_the_training_prefix(listed):
    """`train.py` downloads every object beneath the prefix it is given,
    so a history of runs kept inside it would be fetched by every later
    run."""
    out = ra.capture("abc123", "retraining")
    assert out["folder"] == "_runs/abc123"
    assert not out["folder"].startswith("retraining/")


def test_a_missing_yaml_is_empty_handed_not_an_exception(monkeypatch):
    """This runs inside an audit write. Losing the audit because its
    extra detail could not be gathered trades something whole for
    something better."""
    monkeypatch.setattr(ra.pc, "s3", S3({}))
    monkeypatch.setattr(ra.pc, "destination_bucket", lambda: "track2-s3")
    assert ra.read_selection("retraining")["images"] == []
    assert ra.capture("abc123", "retraining") == {}


def test_a_missing_list_does_not_lose_the_other_one(monkeypatch):
    fake = S3({
        "retraining/data.yaml": LIST_YAML,
        "retraining/_retained/train.txt": TRAIN_LINES,
    })
    monkeypatch.setattr(ra.pc, "s3", fake)
    monkeypatch.setattr(ra.pc, "destination_bucket", lambda: "track2-s3")
    found = ra.read_selection("retraining")
    assert set(found["lists"]) == {"train"}
    assert len(found["images"]) == 3


def test_blank_lines_are_not_recorded_as_files(monkeypatch):
    fake = S3({
        "retraining/data.yaml": LIST_YAML,
        "retraining/_retained/train.txt": "./week_1/images/train/a.jpg\n\n\n",
        "retraining/_retained/val.txt": VAL_LINES,
    })
    monkeypatch.setattr(ra.pc, "s3", fake)
    monkeypatch.setattr(ra.pc, "destination_bucket", lambda: "track2-s3")
    found = ra.read_selection("retraining")
    assert found["images"] == ["week_1/images/train/a.jpg", "week_1/images/val/d.jpg"]
