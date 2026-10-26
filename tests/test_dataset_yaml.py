"""Generating a data.yaml over the week folders.

The cases that matter are the silent ones. A yaml that names label
folders, or that gets the class ORDER wrong, trains happily and produces
a model that is wrong in a way no error message mentions.
"""

from __future__ import annotations

import yaml

from ppe_pipeline_api import dataset_yaml as dy


class FakeS3:
    """Enough S3 to scan folders and fetch one data.yaml."""

    def __init__(self, objects: dict, files: dict | None = None):
        self.objects = dict(objects)
        self.files = dict(files or {})
        self.put: list = []

    def list_objects_v2(self, **kwargs):
        prefix = kwargs["Prefix"]
        contents = [{"Key": key, "Size": size}
                    for key, size in sorted(self.objects.items())
                    if key.startswith(prefix)]
        return {"Contents": contents, "IsTruncated": False}

    def get_object(self, Bucket, Key):
        if Key not in self.files:
            raise KeyError(Key)

        class Body:
            def __init__(self, data):
                self.data = data

            def read(self):
                return self.data

        return {"Body": Body(self.files[Key].encode("utf-8"))}

    def put_object(self, **kwargs):
        self.put.append((kwargs["Key"], kwargs["Body"]))
        return {}


SOURCE_YAML = """
path: .
train: images/train
val: images/val
names:
  0: helmet
  1: gloves
  2: person
"""


def _world(monkeypatch, objects, files=None):
    s3 = FakeS3(objects, files)
    monkeypatch.setattr(dy.pc, "s3", s3)
    monkeypatch.setattr(dy.pc, "destination_bucket", lambda config=None: "track2-s3")
    return s3


def _weeks(root, *specs):
    """`(week, images, labels)` triples as S3 keys."""
    out = {}
    for name, images, labels in specs:
        for i in range(images):
            out[f"{root}/{name}/images/{name}_{i}.jpg"] = 100
        for i in range(labels):
            out[f"{root}/{name}/labels/{name}_{i}.txt"] = 10
    return out


def test_weeks_are_ordered_by_number_not_by_name(monkeypatch):
    """`week_10` after `week_9`, not between `week_1` and `week_2`.

    Alphabetical ordering would put the newest week in the middle, and
    every "which week is newest" question after that is wrong.
    """
    _world(monkeypatch, _weeks("train_root",
                               ("week_1", 2, 2), ("week_2", 2, 2),
                               ("week_9", 2, 2), ("week_10", 2, 2)))
    got = [w["week"] for w in dy.scan_root("train_root")]
    assert got == ["week_1", "week_2", "week_9", "week_10"]


def test_any_prefix_orders_the_same_way(monkeypatch):
    """Only the digits are read, so day_ and batch- work unchanged."""
    _world(monkeypatch, {**_weeks("r", ("day_1", 1, 1), ("day_10", 1, 1),
                                  ("day_2", 1, 1))})
    assert [w["week"] for w in dy.scan_root("r")] == ["day_1", "day_2", "day_10"]


def test_the_yaml_names_image_folders_and_never_label_folders(monkeypatch):
    """Ultralytics finds labels by swapping /images/ for /labels/.

    Listing the label folders as data paths would have it look for
    labels/labels/, find none, and train every image as a background
    example with no annotations -- silently.
    """
    _world(monkeypatch,
           {**_weeks("train_root", ("week_1", 3, 3)),
            **_weeks("eval_root", ("week_1", 2, 2))},
           {"src/data.yaml": SOURCE_YAML})

    built = dy.build(training_root="train_root", eval_root="eval_root",
                     class_sources=["src"])
    document = yaml.safe_load(built["yaml"])

    assert document["train"] == ["train_root/week_1/images"]
    assert document["val"] == ["eval_root/week_1/images"]
    for split in ("train", "val"):
        assert not any("labels" in path for path in document[split])


def test_nothing_is_written_by_building(monkeypatch):
    """No folder, no copy. That is the whole constraint."""
    s3 = _world(monkeypatch,
                {**_weeks("train_root", ("week_1", 3, 3)),
                 **_weeks("eval_root", ("week_1", 2, 2))},
                {"src/data.yaml": SOURCE_YAML})
    dy.build(training_root="train_root", eval_root="eval_root",
             class_sources=["src"])
    assert s3.put == []


def test_class_names_come_from_the_source_dataset(monkeypatch):
    """The indices in the labels came from that file, so it is the only
    source guaranteed to agree with them."""
    _world(monkeypatch,
           {**_weeks("train_root", ("week_1", 1, 1)),
            **_weeks("eval_root", ("week_1", 1, 1))},
           {"ds/data.yaml": SOURCE_YAML})

    # Picked as `ds/images/train`, two levels below the yaml.
    built = dy.build(training_root="train_root", eval_root="eval_root",
                     class_sources=["ds/images/train"])
    assert built["classes"]["names"] == {0: "helmet", 1: "gloves", 2: "person"}
    assert built["classes"]["source"].endswith("ds/data.yaml")
    assert yaml.safe_load(built["yaml"])["names"] == {0: "helmet", 1: "gloves", 2: "person"}


def test_no_class_names_is_blocking(monkeypatch):
    """A label line is only an index. Without names it means nothing."""
    _world(monkeypatch, {**_weeks("train_root", ("week_1", 1, 1)),
                         **_weeks("eval_root", ("week_1", 1, 1))})
    built = dy.build(training_root="train_root", eval_root="eval_root",
                     class_sources=["nowhere"])
    assert built["ready"] is False
    assert any("class names" in p["text"] for p in built["problems"]
               if p["level"] == "blocking")


def test_a_week_with_no_labels_is_reported(monkeypatch):
    """Legitimate to want, catastrophic by accident: every image becomes
    a background example."""
    _world(monkeypatch,
           {**_weeks("train_root", ("week_1", 5, 0)),
            **_weeks("eval_root", ("week_1", 1, 1))},
           {"src/data.yaml": SOURCE_YAML})
    built = dy.build(training_root="train_root", eval_root="eval_root",
                     class_sources=["src"])
    assert built["train_weeks"][0]["unlabelled"] is True
    assert any("background example" in p["text"] for p in built["problems"])


def test_no_evaluation_root_is_blocking(monkeypatch):
    """Without a val split, training cannot measure itself at all."""
    _world(monkeypatch, _weeks("train_root", ("week_1", 2, 2)),
           {"src/data.yaml": SOURCE_YAML})
    built = dy.build(training_root="train_root", class_sources=["src"])
    assert built["ready"] is False
    assert any("val split" in p["text"] for p in built["problems"])


def test_the_download_prefix_is_the_parent_of_both_roots(monkeypatch):
    """The trainer downloads ONE prefix, so the paths must sit inside it.

    Two top-level roots share only the bucket, which is reported: a yaml
    whose paths cannot be resolved is worse than one that refuses.
    """
    _world(monkeypatch,
           {**_weeks("retraining/training", ("week_1", 1, 1)),
            **_weeks("retraining/evaluation", ("week_1", 1, 1))},
           {"src/data.yaml": SOURCE_YAML})
    nested = dy.build(training_root="retraining/training",
                      eval_root="retraining/evaluation", class_sources=["src"])
    assert nested["download_prefix"] == "s3://track2-s3/retraining/"
    assert nested["write_to"] == "retraining/data.yaml"
    # Paths are written relative to that prefix, so they resolve locally.
    assert yaml.safe_load(nested["yaml"])["train"] == ["training/week_1/images"]
    assert not any("whole bucket" in p["text"] for p in nested["problems"])

    _world(monkeypatch,
           {**_weeks("train_root", ("week_1", 1, 1)),
            **_weeks("eval_root", ("week_1", 1, 1))},
           {"src/data.yaml": SOURCE_YAML})
    flat = dy.build(training_root="train_root", eval_root="eval_root",
                    class_sources=["src"])
    assert flat["download_prefix"] == "s3://track2-s3/"
    assert any("whole bucket" in p["text"] for p in flat["problems"])


def test_saving_writes_one_file_and_only_that(monkeypatch):
    s3 = _world(monkeypatch,
                {**_weeks("train_root", ("week_1", 2, 2)),
                 **_weeks("eval_root", ("week_1", 1, 1))},
                {"src/data.yaml": SOURCE_YAML})
    built = dy.build(training_root="train_root", eval_root="eval_root",
                     class_sources=["src"])
    dy.save(key=built["write_to"], text=built["yaml"])

    assert len(s3.put) == 1
    key, body = s3.put[0]
    assert key == "data.yaml"
    assert yaml.safe_load(body.decode("utf-8"))["train"] == ["train_root/week_1/images"]


def _split_weeks(root, *specs):
    """`(week, train_images, val_images)` as keys in the single-root shape."""
    out = {}
    for name, train, val in specs:
        for i in range(train):
            out[f"{root}/{name}/images/train/{name}_{i}.jpg"] = 100
            out[f"{root}/{name}/labels/train/{name}_{i}.txt"] = 10
        for i in range(val):
            out[f"{root}/{name}/images/val/{name}_v{i}.jpg"] = 100
            out[f"{root}/{name}/labels/val/{name}_v{i}.txt"] = 10
    return out


def test_one_root_reads_both_splits_from_inside_each_week(monkeypatch):
    """The layout the pull now writes.

    The split is a folder inside the week, so train and val are read from
    the same tree -- and must not be confused for one another.
    """
    _world(monkeypatch,
           _split_weeks("retraining", ("week_1", 3, 2), ("week_2", 4, 1)),
           {"src/data.yaml": SOURCE_YAML})

    built = dy.build(root="retraining", class_sources=["src"])
    document = yaml.safe_load(built["yaml"])

    assert document["train"] == ["week_1/images/train", "week_2/images/train"]
    assert document["val"] == ["week_1/images/val", "week_2/images/val"]
    assert built["train_images"] == 7
    assert built["val_images"] == 3
    # Still no label folder anywhere in the data paths.
    for split in ("train", "val"):
        assert not any("labels" in path for path in document[split])


def test_one_root_means_one_download_prefix(monkeypatch):
    """The point of the merge.

    Two top-level roots shared only the bucket, so a yaml naming both
    could only be resolved by fetching everything in it.
    """
    _world(monkeypatch,
           _split_weeks("retraining", ("week_1", 1, 1)),
           {"src/data.yaml": SOURCE_YAML})
    built = dy.build(root="retraining", class_sources=["src"])

    assert built["download_prefix"] == "s3://track2-s3/retraining/"
    assert built["write_to"] == "retraining/data.yaml"
    assert not any("whole bucket" in p["text"] for p in built["problems"])


def test_a_root_with_no_val_split_is_blocking(monkeypatch):
    """Images but nothing held out: training cannot measure itself."""
    _world(monkeypatch,
           _split_weeks("retraining", ("week_1", 5, 0)),
           {"src/data.yaml": SOURCE_YAML})
    built = dy.build(root="retraining", class_sources=["src"])

    assert built["ready"] is False
    assert any("images/val" in p["text"] for p in built["problems"]
               if p["level"] == "blocking")


def test_val_images_are_never_counted_as_training_ones(monkeypatch):
    """`images/` is a prefix of `images/train/` and `images/val/`.

    Matching by substring would count every val image on both sides, and
    the yaml would train on its own held-out set.
    """
    _world(monkeypatch,
           _split_weeks("retraining", ("week_1", 2, 7)),
           {"src/data.yaml": SOURCE_YAML})
    built = dy.build(root="retraining", class_sources=["src"])
    assert built["train_images"] == 2
    assert built["val_images"] == 7
