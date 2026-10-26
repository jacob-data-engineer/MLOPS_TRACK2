"""The retention policy, written where the trainer obeys it."""

from __future__ import annotations

import pytest

from ppe_pipeline_api import retained_lists as rl


WEEKS = {
    "week_1": {"train": 24, "val": 2},
    "week_2": {"train": 310, "val": 82},
    "week_3": {"train": 166, "val": 37},
    "week_4": {"train": 0, "val": 0},
    "week_5": {"train": 0, "val": 0},
    "week_6": {"train": 7, "val": 1},
    "week_7": {"train": 92, "val": 27},
}

POLICY = {
    "enabled": True, "curve": "exponential", "weekly_rate": 0.01,
    "full_weeks": 1, "use_floor": True, "floor": 0.10, "min_images": 25,
}


@pytest.fixture
def bucket(monkeypatch):
    """The week folders, as keys."""
    keys = []
    for week, sides in WEEKS.items():
        for split, count in sides.items():
            for i in range(count):
                keys.append(f"retraining/{week}/images/{split}/{week}_{split}_{i:04d}.jpg")
    written: dict = {}

    class S3:
        def put_object(self, *, Bucket, Key, Body, **_kw):
            written[Key] = Body

    def basenames(prefix, extensions):
        prefix = prefix.strip("/") + "/"
        return {k.rsplit("/", 1)[-1].rsplit(".", 1)[0]: k
                for k in keys if k.startswith(prefix)}

    monkeypatch.setattr(rl.pc, "s3", S3())
    monkeypatch.setattr(rl.pc, "destination_bucket", lambda: "track2-s3")

    from ppe_pipeline_api import track2_shared as t2
    monkeypatch.setattr(t2, "_basenames", basenames)
    # The inventory (`retention.read_inventory`) lists keys, not stems.
    monkeypatch.setattr(t2, "_list_keys",
                        lambda prefix: sorted(k for k in keys if k.startswith(prefix)))
    monkeypatch.setattr(t2, "list_s3_folders", lambda root: list(WEEKS))
    return written


def test_the_list_crosses_week_folders(bucket):
    """The property the whole idea rests on. Ultralytics reads a split
    FILE as a flat list of image paths, and resolves each one's label
    independently by swapping /images/ for /labels/ -- so entries may sit
    in different weeks and each still finds its own."""
    chosen = rl.select(root="retraining", config=POLICY)
    weeks_in_list = {line.split("/")[1] for line in chosen["train"]}
    assert weeks_in_list == {"week_1", "week_2", "week_3", "week_6", "week_7"}
    assert all(line.startswith("./week_") for line in chosen["train"])
    assert all("/images/train/" in line for line in chosen["train"])


def test_the_policy_actually_thins_the_train_side(bucket):
    """Until this existed the decay reached nothing: a folder list trains
    on every image in it whatever the card says."""
    chosen = rl.select(root="retraining", config=POLICY)
    assert chosen["train_available"] == 599
    assert len(chosen["train"]) == 587           # what the card reports
    assert len(chosen["train"]) < chosen["train_available"]

    by_week = {r["week"]: r for r in chosen["weeks"]}
    # Smaller than the 25-image minimum, so kept whole however old.
    assert by_week["week_1"]["keep"] == 24
    assert by_week["week_6"]["keep"] == 7
    assert by_week["week_2"]["keep"] == 301


def test_the_val_side_is_never_decayed(bucket):
    """An evaluation set that shrinks week by week makes every run's
    numbers incomparable with the last."""
    chosen = rl.select(root="retraining", config=POLICY)
    assert len(chosen["val"]) == chosen["val_available"] == 149
    assert {line.split("/")[1] for line in chosen["val"]} == {
        "week_1", "week_2", "week_3", "week_6", "week_7"}


def test_an_empty_week_contributes_nothing_and_ages_nothing(bucket):
    chosen = rl.select(root="retraining", config=POLICY)
    by_week = {r["week"]: r for r in chosen["weeks"]}
    assert by_week["week_4"]["keep"] == 0
    assert by_week["week_5"]["keep"] == 0
    # week_1 is four weeks of DATA back, not six of calendar.
    assert by_week["week_1"]["age_weeks"] == 4
    assert by_week["week_7"]["age_weeks"] == 0


def test_paths_are_relative_with_the_dot_slash_ultralytics_resolves(bucket):
    """`get_img_files` replaces a leading './' with the list file's own
    parent directory, so the same list reads correctly in the bucket and
    in the container."""
    chosen = rl.select(root="retraining", config=POLICY)
    assert chosen["train"][0].startswith("./")
    assert "retraining/" not in chosen["train"][0]


def test_the_written_line_count_is_checked_against_the_selection(bucket, monkeypatch):
    """`get_img_files` drops a line whose extension is not an image
    format WITHOUT a word -- it filters on the extension before anything
    checks the file exists. A typo is therefore a quietly smaller
    training set, and counting back is the only thing that catches it."""
    out = rl.write(root="retraining", config=POLICY)
    assert out["written"]["train.txt"]["lines"] == len(out["train"]) == 587
    assert out["written"]["val.txt"]["lines"] == len(out["val"]) == 149
    assert out["train_entry"] == "train.txt"
    assert out["val_entry"] == "val.txt"

    # And it is a real check, not a formality.
    monkeypatch.setattr(rl, "render", lambda paths: "\n".join(paths[:-1]) + "\n")
    with pytest.raises(ValueError, match="lines for"):
        rl.write(root="retraining", config=POLICY)


def test_the_lists_sit_at_the_root_the_paths_are_relative_to(bucket):
    """Not a subfolder, and this is correctness rather than tidiness.

    `get_img_files` expands a leading `./` against THE LIST FILE'S OWN
    DIRECTORY, not against `path:`. A list at `<root>/_retained/train.txt`
    holding `./week_1/...` therefore resolves to
    `<root>/_retained/week_1/...`, which does not exist -- and that is
    exactly how a real run failed: "No valid images found in
    .../_retained/week_1/labels/train.cache".

    So the lines are relative to the root, and the list belongs at the
    root. This test encodes the pairing, because the two are only
    correct together and either alone looks reasonable.
    """
    out = rl.write(root="retraining", config=POLICY)
    for entry in out["written"].values():
        key = entry["key"]
        assert key.startswith("retraining/")
        # One segment after the root: the file itself, no folder between.
        assert key[len("retraining/"):].count("/") == 0, key

    # And the lines are relative to that same directory.
    assert out["train"][0].startswith("./week_")


def test_render_writes_no_comments(bucket):
    """A header comment would not be an annotation, it would be an
    invisible reduction of the dataset -- the extension filter drops the
    line and says nothing."""
    text = rl.render(["./week_1/images/train/a.jpg", "./week_2/images/train/b.jpg"])
    assert text == "./week_1/images/train/a.jpg\n./week_2/images/train/b.jpg\n"
    assert "#" not in text
    assert "" not in text.splitlines()


def test_a_disabled_policy_keeps_everything(bucket):
    chosen = rl.select(root="retraining", config={**POLICY, "enabled": False})
    assert len(chosen["train"]) == chosen["train_available"] == 599
    assert chosen["enabled"] is False
