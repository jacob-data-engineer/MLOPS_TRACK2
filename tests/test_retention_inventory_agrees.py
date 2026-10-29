"""The card, the grid and the trainer count the SAME images.

The preview and the schedule grid used to size each week from the whole
`images/` subtree -- train AND val -- while `retained_lists.select`,
which writes the list the trainer actually reads, decays `images/train`
only and keeps `images/val` whole. The same policy function was handed
two different populations, so a week of 100 train + 100 val images at
50% read "100 of 200" on the card while the run kept 50 of 100, and a
validation-only week aged everything before it on the card while the
trainer stepped over it as empty.

One fake bucket drives all three here: preview, matrix and select.
"""
from __future__ import annotations

import pytest

from ppe_pipeline_api import retained_lists as rl
from ppe_pipeline_api import retention as rt
from ppe_pipeline_api import track2_shared as t2
from ppe_pipeline_api.routers import track2_retention as tr
from _sides import patch_everywhere

ROOT = "retraining"

POLICY = {
    **rt.DEFAULTS,
    "enabled": True, "curve": "exponential", "weekly_rate": 0.25,
    "full_weeks": 1, "use_floor": True, "floor": 0.10, "min_images": 25,
    "use_golden": False,
}


def _keys() -> tuple[list, dict]:
    keys: list = []
    expected: dict = {}

    def add(week, split, names):
        for name in names:
            keys.append(f"{ROOT}/{week}/images/{split}/{name}")
        expected.setdefault(week, {"train": 0, "val": 0})[split] += len(names)
        # Labels beside them, which must never be counted as images.
        for name in names:
            keys.append(f"{ROOT}/{week}/labels/{split}/{name.rsplit('.', 1)[0]}.txt")

    # Mixed, and the val stems are DISTINCT from the train ones.
    add("week_1", "train", [f"w1_t{i}.jpg" for i in range(30)])
    add("week_1", "val", [f"w1_v{i}.jpg" for i in range(10)])
    # The example from the finding: 100 + 100.
    add("week_2", "train", [f"w2_t{i}.jpg" for i in range(100)])
    add("week_2", "val", [f"w2_v{i}.jpg" for i in range(100)])
    # Validation only: nothing to train on, so it must not age anything.
    add("week_3", "train", [])
    add("week_3", "val", [f"w3_v{i}.jpg" for i in range(20)])
    # week_4 exists and is empty.
    expected["week_4"] = {"train": 0, "val": 0}
    # Under the 25-image minimum: kept whole however old.
    add("week_5", "train", [f"w5_t{i}.jpg" for i in range(8)])
    add("week_5", "val", [f"w5_v{i}.jpg" for i in range(2)])
    # The same basenames in two camera subfolders -- different images,
    # different labels. Counting by bare stem would hide fifteen.
    add("week_6", "train", [f"cam_a/f{i}.jpg" for i in range(15)]
        + [f"cam_b/f{i}.jpg" for i in range(15)]
        + [f"flat{i}.png" for i in range(30)])
    add("week_6", "val", [f"w6_v{i}.jpg" for i in range(15)])
    # Numeric order: newest, though it sorts between 1 and 2 as text.
    add("week_10", "train", [f"w10_t{i}.jpg" for i in range(40)])
    add("week_10", "val", [f"w10_v{i}.jpg" for i in range(5)])
    # Not a week.
    keys.append(f"{ROOT}/_retained/images/train/stray.jpg")
    keys.append(f"{ROOT}/train.txt")
    return keys, expected


KEYS, EXPECTED = _keys()
FOLDERS = ["_retained", "week_1", "week_10", "week_2", "week_3", "week_4",
           "week_5", "week_6"]


@pytest.fixture
def bucket(monkeypatch):
    def list_keys(prefix):
        return sorted(k for k in KEYS if k.startswith(prefix))

    def basenames(prefix, extensions):
        prefix = prefix.strip("/") + "/"
        return {k.rsplit("/", 1)[-1].rsplit(".", 1)[0]: k
                for k in list_keys(prefix) if k.lower().endswith(extensions)}

    patch_everywhere(monkeypatch, "_list_keys", list_keys)
    patch_everywhere(monkeypatch, "_basenames", basenames)
    patch_everywhere(monkeypatch, "list_s3_folders",
                        lambda root: list(FOLDERS) if root.strip("/") == ROOT else [])
    monkeypatch.setattr(rl.pc, "destination_bucket", lambda: "track2-s3")
    monkeypatch.setattr(tr.pc, "read_config", lambda: {
        "retention": {k: v for k, v in POLICY.items()}, "week_prefix": "week_"})


def _select():
    return rl.select(root=ROOT, config={**POLICY, "week_prefix": "week_"})


def _preview():
    return tr.preview(tr.PreviewRequest(root=ROOT), session=None)


def _matrix():
    return tr.matrix(tr.MatrixRequest(root=ROOT, ahead=0), session=None)


def test_preview_rows_are_exactly_what_the_selector_keeps(bucket):
    chosen = {r["week"]: r for r in _select()["weeks"]}
    shown = {r["week"]: r for r in _preview()["weeks"]}
    assert set(shown) == set(chosen) == {
        "week_1", "week_2", "week_3", "week_4", "week_5", "week_6", "week_10"}
    for week, row in shown.items():
        sel = chosen[week]
        assert row["available"] == sel["available"] == EXPECTED[week]["train"], week
        assert row["keep"] == sel["keep"], week
        assert row["age_weeks"] == sel["age_weeks"], week
        effective = round(sel["keep"] / sel["available"], 4) if sel["available"] else 0.0
        assert row["effective"] == effective, week
        assert row["val"] == sel["val"] == EXPECTED[week]["val"], week


def test_the_finding_example_is_counted_on_the_train_side_only(bucket):
    week_2 = next(r for r in _preview()["weeks"] if r["week"] == "week_2")
    assert week_2["available"] == 100          # not 200
    assert week_2["val"] == 100                # reported, separately


def test_preview_totals_match_the_lists_the_trainer_reads(bucket):
    chosen = _select()
    shown = _preview()
    assert shown["available"] == chosen["train_available"]
    assert shown["keep"] == len(chosen["train"])
    assert shown["val_available"] == chosen["val_available"] == len(chosen["val"])


def test_a_validation_only_week_ages_nothing(bucket):
    shown = {r["week"]: r for r in _preview()["weeks"]}
    assert shown["week_3"]["available"] == 0 and shown["week_3"]["keep"] == 0
    # Weeks holding TRAIN data, newest first: 10, 6, 5, 2, 1.
    assert shown["week_10"]["age_weeks"] == 0
    assert shown["week_6"]["age_weeks"] == 1
    assert shown["week_5"]["age_weeks"] == 2
    assert shown["week_2"]["age_weeks"] == 3
    assert shown["week_1"]["age_weeks"] == 4


def test_small_weeks_are_held_by_the_minimum_identically(bucket):
    shown = {r["week"]: r for r in _preview()["weeks"]}
    assert shown["week_5"]["keep"] == 8        # under the minimum: whole
    assert shown["week_1"]["keep"] == 25       # 30 at age 4, held at 25
    assert shown["week_1"]["floored"] is True


def test_validation_is_kept_whole_and_listed(bucket):
    chosen = _select()
    want = sorted("./" + k[len(ROOT) + 1:] for k in KEYS
                  if "/images/val/" in k and not k.startswith(f"{ROOT}/_"))
    assert sorted(chosen["val"]) == want
    assert len(want) == sum(e["val"] for e in EXPECTED.values())
    assert not any("/images/val/" in line for line in chosen["train"])


def test_matrix_sizes_are_the_training_inventory(bucket):
    grid = _matrix()
    chosen = {r["week"]: r for r in _select()["weeks"]}
    assert grid["weeks"] == ["week_1", "week_2", "week_3", "week_4",
                             "week_5", "week_6", "week_10"]
    newest = next(r for r in grid["rows"] if r["week"] == "week_10")
    for cell in newest["cells"]:
        week = cell["week"]
        assert cell["available"] == EXPECTED[week]["train"], week
        assert cell["keep"] == chosen[week]["keep"], week
        if EXPECTED[week]["train"]:
            # An empty week's "age" is not used by either (nothing to
            # keep); only weeks holding training data must agree.
            assert cell["age_weeks"] == chosen[week]["age_weeks"], week
    assert next(c for c in newest["cells"] if c["week"] == "week_3")["empty"] is True


def test_same_basename_in_two_subfolders_is_two_images(bucket):
    chosen = rl.select(root=ROOT, config={**POLICY, "enabled": False})
    week_6 = [line for line in chosen["train"] if line.startswith("./week_6/")]
    assert len(week_6) == 60
    assert "./week_6/images/train/cam_a/f0.jpg" in week_6
    assert "./week_6/images/train/cam_b/f0.jpg" in week_6
    shown = {r["week"]: r for r in _preview()["weeks"]}
    assert shown["week_6"]["available"] == 60


def test_non_week_folders_stay_out(bucket):
    chosen = _select()
    assert not any("_retained" in line for line in chosen["train"] + chosen["val"])
