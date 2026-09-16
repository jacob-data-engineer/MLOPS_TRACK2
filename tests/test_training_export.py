"""Tests for the YOLO training export.

The format assertions are checked against the REAL dataset's own files
(C:\\dev\\Testing_Dataset\\object_detection_dataset), not against what this
module happens to produce -- a self-consistent exporter that Track2 cannot
read is the failure mode worth guarding.
"""

from __future__ import annotations

import io
import json

import pytest
from PIL import Image

from ppe_pipeline import training_export
from ppe_pipeline.config import PipelineConfig
from ppe_pipeline.models import BoundingBox
from ppe_pipeline.training_export import (
    audited_crop_label,
    CLASS_NAMES,
    HELMET_CLASS,
    PERSON_CLASS,
    build_export,
    build_frame_export,
    DEFAULT_EXPORT_PREFIX,
    FRAME_EXPORT_PREFIX,
    data_yaml,
    label_text,
    load_export_index,
    to_yolo,
    write_export,
    ExportedBox,
)


def _row(**overrides) -> dict:
    row = {
        "alarm_id": "alarm-1",
        "hazard_type": "no_helmet",
        "outcome": "CONFIRM_FP",
        "reason_code": "helmet_worn",
        "confidence": 0.95,
        "timestamp": "2026-08-27T10:00:00+00:00",
        "image_key": "eval/alarm_images/alarm-1.jpg",
        "image_width": 1920,
        "image_height": 1080,
        "person_box": {"x1": 100, "y1": 100, "x2": 300, "y2": 400},
        "crop_box": {"x1": 100, "y1": 100, "x2": 300, "y2": 400},
        "helmet_box": {"x1": 180, "y1": 160, "x2": 220, "y2": 200},
        "helmet_box_source": "cv",
        "helmet_box_confidence": 0.8,
        "judge_provider": "bedrock",
        "judge_model": "qwen.qwen3-vl-235b-a22b-instruct",
    }
    row.update(overrides)
    return row


# --- format, against the real dataset ------------------------------------


def test_class_map_matches_the_real_data_yaml():
    """Renumbering to a compact 0=helmet/1=person would silently corrupt any
    later merge with the real dataset."""
    assert CLASS_NAMES == {
        0: "helmet",
        1: "gloves",
        2: "person",
        3: "working_boots",
        4: "man_cage",
        5: "safe_area",
    }
    assert HELMET_CLASS == 0
    assert PERSON_CLASS == 2


def test_data_yaml_has_the_real_files_shape():
    """`names` must be an int-keyed MAP: Track2's load_class_names does
    `.items()` on it and would break on a list. The real file also has no
    `nc:`, no `path:` and no `test:`."""
    text = data_yaml()

    assert "names:\n  0: helmet\n" in text
    assert "  5: safe_area\n" in text
    assert "train: images/train\n" in text
    assert "nc:" not in text
    assert "path:" not in text
    assert "test:" not in text
    assert "- helmet" not in text  # i.e. not a list


def test_a_label_line_matches_the_real_labels_shape():
    """Real line, from labels/val/Helmet_Wharf_11_frame_0099.txt:
        2 0.255747 0.508773 0.037818 0.136102
    Five whitespace-separated fields, 6dp, no confidence column."""
    text = label_text(
        [ExportedBox(PERSON_CLASS, (0.255747, 0.508773, 0.037818, 0.136102), (0, 0, 1, 1), (0, 0, 1, 1), "cv")]
    )

    line = text.strip()
    fields = line.split()
    assert len(fields) == 5
    assert fields[0] == "2"
    assert line == "2 0.255747 0.508773 0.037818 0.136102"
    assert text.endswith("\n")


def test_no_boxes_produces_an_empty_but_valid_file():
    assert label_text([]) == ""


# --- coordinate conversion ------------------------------------------------


def test_a_box_centred_in_the_crop_normalizes_to_the_centre():
    crop = BoundingBox(x1=100, y1=100, x2=300, y2=400)  # 200x300

    cx, cy, w, h = to_yolo([180, 160, 220, 200], crop)

    assert cx == pytest.approx(0.5)
    assert cy == pytest.approx(0.2 + (40 / 2) / 300)
    assert w == pytest.approx(40 / 200)
    assert h == pytest.approx(40 / 300)


def test_conversion_normalizes_against_the_rounded_crop():
    """`crop_bytes` cuts on the rounded tuple, so that is the exported
    image's real pixel size -- normalizing against the unrounded width would
    offset every label by up to a pixel, differently per case."""
    crop = BoundingBox(x1=100.4, y1=100.4, x2=300.6, y2=400.6)
    x1, y1, x2, y2 = crop.as_tuple()

    _, _, w, h = to_yolo([150, 150, 250, 250], crop)

    assert w == pytest.approx(100 / (x2 - x1))
    assert h == pytest.approx(100 / (y2 - y1))


def test_a_zero_area_crop_is_refused():
    with pytest.raises(ValueError):
        to_yolo([1, 2, 3, 4], BoundingBox(x1=10, y1=10, x2=10.2, y2=10.2))


# --- what gets exported ---------------------------------------------------


def test_a_pipeline_settled_case_is_exported_with_both_boxes():
    result = build_export([_row()], {}, config=PipelineConfig())

    assert result.written == 1
    case = result.cases[0]
    assert {b.class_id for b in case.boxes} == {HELMET_CLASS, PERSON_CLASS}
    assert case.image_name == "alarm-1.jpg"


def test_a_human_box_wins_over_the_models():
    """That is the entire point of the review queue."""
    reviews = {
        "alarm-1": {
            "helmet_box_status": "drawn",
            "helmet_box": [190, 170, 210, 190],
        }
    }
    result = build_export([_row(helmet_box_source="vlm")], reviews, config=PipelineConfig())

    helmet = next(b for b in result.cases[0].boxes if b.class_id == HELMET_CLASS)
    assert helmet.source == "human"
    # the human's box, not the model's
    assert helmet.yolo[2] == pytest.approx(20 / 200)


def test_the_person_box_is_labelled_as_a_pseudo_label():
    """It comes from the CV detector that raised the alarm, so it must not be
    recorded as if a human verified it."""
    result = build_export([_row()], {}, config=PipelineConfig())

    person = next(b for b in result.cases[0].boxes if b.class_id == PERSON_CLASS)
    assert person.source == "cv"


def test_no_helmet_visible_skips_the_case_entirely():
    """Not "export the person box alone", which was the tempting reading.

    The case reached the box queue because the judge cleared it as
    helmet_worn. A human then finding no helmet to draw means one of two
    things -- the helmet is there but unboxable (occluded, too small), or
    there is genuinely no helmet and the verdict is wrong -- and this option
    cannot distinguish them. Exporting a person-only label would, in the
    first case, teach the model that a helmet which IS present is background,
    which is the exact harm the crop-not-frame decision exists to avoid. So
    the whole case is skipped, and the reason is reported."""
    reviews = {"alarm-1": {"helmet_box_status": "no_helmet_visible"}}

    result = build_export([_row(helmet_box_source="vlm")], reviews, config=PipelineConfig())

    assert result.written == 0
    assert "no helmet visible" in result.skipped[0]["reason"]


def test_cannot_determine_is_not_exported_and_says_why():
    reviews = {"alarm-1": {"verdict": "cannot_determine"}}

    result = build_export([_row(outcome="ESCALATE")], reviews, config=PipelineConfig())

    assert result.written == 0
    assert "cannot determine" in result.skipped[0]["reason"]


def test_a_case_still_awaiting_a_human_is_not_exported():
    result = build_export([_row(outcome="ESCALATE")], {}, config=PipelineConfig())

    assert result.written == 0
    assert result.skipped == []  # not settled at all, so not a skip -- just absent


@pytest.mark.parametrize(
    "overrides,reason",
    [
        ({"image_key": "streamlit-legacy.jpg"}, "no stored frame"),
        ({"crop_box": None}, "no recorded crop region"),
    ],
)
def test_cases_that_cannot_be_exported_are_reported_not_dropped(overrides, reason):
    """A small export must be explicable, or it reads as broken."""
    result = build_export([_row(**overrides)], {}, config=PipelineConfig())

    assert result.written == 0
    assert reason in result.skipped[0]["reason"]


def test_a_box_outside_the_crop_is_dropped_rather_than_written():
    """An out-of-range YOLO coordinate is accepted silently by the trainer and
    learned as garbage."""
    result = build_export(
        [_row(helmet_box={"x1": 5000, "y1": 5000, "x2": 5100, "y2": 5100})],
        {},
        config=PipelineConfig(),
    )

    assert {b.class_id for b in result.cases[0].boxes} == {PERSON_CLASS}


def test_a_rerun_alarm_is_exported_once():
    rows = [
        _row(timestamp="2026-08-01T00:00:00+00:00"),
        _row(timestamp="2026-08-27T00:00:00+00:00", helmet_box_confidence=0.9),
    ]

    result = build_export(rows, {}, config=PipelineConfig())

    assert result.written == 1


# --- metadata -------------------------------------------------------------


def test_metadata_carries_the_provenance_track2_needs():
    reviews = {"alarm-1": {"verdict": "no_violation", "reviewed_by": "Han"}}
    result = build_export([_row(outcome="ESCALATE")], reviews, config=PipelineConfig())

    meta = result.cases[0].metadata

    assert meta["alarm_id"] == "alarm-1"
    assert meta["image_name"] == "alarm-1.jpg"
    assert meta["hazard_type"] == "no_helmet"
    assert meta["judge_model"] == "qwen.qwen3-vl-235b-a22b-instruct"
    assert meta["judge_provider"] == "bedrock"
    assert meta["human_verdict"] == "no_violation"
    assert meta["reviewed_by"] == "Han"
    assert meta["settled_by"] == "human"
    assert meta["frame_width"] == 1920
    assert meta["crop_box"] == [100, 100, 300, 400]
    assert {b["source"] for b in meta["boxes"]} <= {"human", "cv"}


def test_metadata_distinguishes_pipeline_settled_from_human_settled():
    """A disagreement between judge and human has to stay visible, so both
    are recorded rather than one overwriting the other."""
    auto = build_export([_row()], {}, config=PipelineConfig()).cases[0].metadata
    assert auto["settled_by"] == "pipeline"
    assert auto["human_verdict"] is None
    assert auto["judge_outcome"] == "CONFIRM_FP"


# --- writing --------------------------------------------------------------


@pytest.fixture
def frame(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    key = tmp_path / "eval" / "alarm_images"
    key.mkdir(parents=True)
    buffer = io.BytesIO()
    Image.new("RGB", (1920, 1080), color="white").save(buffer, format="JPEG")
    (key / "alarm-1.jpg").write_bytes(buffer.getvalue())
    return tmp_path


def test_write_export_lays_out_a_real_yolo_tree(frame):
    result = build_export([_row()], {}, config=PipelineConfig())

    written = write_export(result, prefix="out")

    assert (frame / "out" / "images" / "train" / "alarm-1.jpg").is_file()
    label = (frame / "out" / "labels" / "train" / "alarm-1.txt").read_text(encoding="utf-8")
    assert len(label.strip().splitlines()) == 2
    assert (frame / "out" / "data.yaml").read_text(encoding="utf-8") == data_yaml()
    assert written["manifest"]["case_count"] == 1


def test_the_exported_image_is_the_crop_not_the_frame(frame):
    """A full frame is 91.3% likely to contain another worker this export
    cannot label, which would train them as background."""
    result = build_export([_row()], {}, config=PipelineConfig())
    write_export(result, prefix="out")

    with Image.open(frame / "out" / "images" / "train" / "alarm-1.jpg") as image:
        assert image.size == (200, 300)


def test_a_batch_manifest_records_what_went_out(frame):
    result = build_export([_row()], {}, config=PipelineConfig())
    write_export(result, prefix="out")

    path = frame / "out" / "batches" / f"{result.batch_id}.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))

    assert manifest["classes"]["0"] == "helmet"
    # Which splits this batch actually touched. A single global "train" was a
    # lie once a val split existed.
    assert manifest["splits"] == [result.cases[0].metadata["split"]]
    assert manifest["cases"][0]["alarm_id"] == "alarm-1"


def test_exporting_twice_tops_up_rather_than_replacing(frame):
    write_export(build_export([_row()], {}, config=PipelineConfig()), prefix="out")
    second = _row(alarm_id="alarm-2", image_key="eval/alarm_images/alarm-1.jpg")
    write_export(build_export([second], {}, config=PipelineConfig()), prefix="out")

    index = load_export_index("out")

    assert set(index) == {"alarm-1", "alarm-2"}
    assert (frame / "out" / "images" / "train" / "alarm-1.jpg").is_file()
    assert (frame / "out" / "images" / "train" / "alarm-2.jpg").is_file()


def test_the_index_is_empty_before_the_first_export(frame):
    assert load_export_index("out") == {}


# --- the train/val split ---------------------------------------------------


def test_the_split_is_held_out_by_scene_not_by_frame():
    """Consecutive frames of one clip are near-duplicates: the same worker,
    the same helmet, seconds apart. Splitting per frame puts them in both
    halves, so the val score measures memorisation and reads highest exactly
    when the model has generalised least."""
    from ppe_pipeline.training_export import scene_of, split_for

    frames = [f"Helmet_Hatch_21_frame_{i:04d}" for i in range(12)]

    assert {scene_of(f) for f in frames} == {"Helmet_Hatch_21"}
    # Every frame of one clip lands on the same side.
    assert len({split_for(f) for f in frames}) == 1


def test_the_split_is_stable_and_order_independent():
    """Cases are exported ONE AT A TIME as they settle, so there is no moment
    when the whole set is in hand to divide. The assignment has to be a pure
    function of the case."""
    from ppe_pipeline.training_export import split_for

    assert split_for("Helmet_Hatch_04_frame_0245") == split_for("Helmet_Hatch_04_frame_0245")
    # A different frame of the same clip, asked in a different order.
    assert split_for("Helmet_Hatch_04_frame_0001") == split_for("Helmet_Hatch_04_frame_0245")


def test_both_splits_actually_get_used_across_many_scenes():
    """A policy that put everything in one split would pass every test above
    and still be useless."""
    from ppe_pipeline.training_export import SPLITS, split_for

    assigned = {split_for(f"Helmet_Hatch_{i:02d}_frame_0001") for i in range(60)}

    assert assigned == set(SPLITS)


def test_data_yaml_points_val_at_its_own_directory():
    """It used to point at `images/train`, so any run validated on its own
    training images and reported a meaningless score."""
    from ppe_pipeline.training_export import data_yaml

    text = data_yaml()

    assert "train: images/train" in text
    assert "val: images/val" in text


def test_republishing_moves_a_case_rather_than_duplicating_it(frame, monkeypatch):
    """Everything exported before the val split existed sits under train. A
    re-sync must move those, not leave a copy in both, which is the leakage
    the split exists to prevent."""
    import ppe_pipeline.training_export as te

    result = build_export([_row()], {}, config=PipelineConfig())
    alarm_id = result.cases[0].alarm_id

    # Publish it as train, then force the same case to val and republish.
    monkeypatch.setattr(te, "split_for", lambda _a, **_k: "train")
    write_export(build_export([_row()], {}, config=PipelineConfig()), prefix="out")
    assert (frame / "out" / "images" / "train" / f"{alarm_id}.jpg").is_file()

    monkeypatch.setattr(te, "split_for", lambda _a, **_k: "val")
    write_export(build_export([_row()], {}, config=PipelineConfig()), prefix="out")

    assert (frame / "out" / "images" / "val" / f"{alarm_id}.jpg").is_file()
    assert not (frame / "out" / "images" / "train" / f"{alarm_id}.jpg").is_file()
    assert not (frame / "out" / "labels" / "train" / f"{alarm_id}.txt").is_file()


# --- the audit's verdict reaching the crops of a crowded image -------------
#
# The live feed and the offline gold run both settle on the IMAGE. A training
# sample is one PERSON. `audited_crop_label` is the whole of what crosses
# that gap, and these are its edges.


def test_a_cleared_image_clears_every_person_in_it():
    """"Nobody here was in violation" is true of each of them separately."""
    assert audited_crop_label("no_violation", crop_fired=False, people_in_image=9) == "no_violation"
    assert audited_crop_label("no_violation", crop_fired=True, people_in_image=9) == "no_violation"


def test_a_violation_image_labels_only_the_person_the_judge_fired_on():
    """At least one person was in violation, and the human never said which.
    The judge's attribution is the only pointer to who -- which is what
    per-crop calls buy and what a batched call would lose."""
    assert audited_crop_label("violation", crop_fired=True, people_in_image=9) == "violation"
    assert audited_crop_label("violation", crop_fired=False, people_in_image=9) is None


def test_a_one_person_image_has_no_gap_to_bridge():
    """Every live alarm and every gold case today. The verdict is about them,
    so nothing here may change what those export."""
    assert audited_crop_label("violation", crop_fired=False, people_in_image=1) == "violation"
    assert audited_crop_label("no_violation", crop_fired=False) == "no_violation"


def _cv_crop_row(person_index: int, outcome: str) -> dict:
    frame = "unk__unk__20260427T070224Z__aaaa__f000450"
    candidate = f"{frame}__p{person_index:02d}"
    return {
        **_row(alarm_id=f"exec__{candidate}", outcome=outcome),
        "row_kind": "helmet_cv",
        "cv": {"frame_name": f"{frame}.jpg", "candidate_id": candidate,
               "person_index": person_index, "people_in_frame": 3},
    }


def test_a_bystander_in_a_violation_frame_is_kept_out_of_the_dataset():
    """Seven of the eight people in an alarm frame are usually bystanders. A
    human calling the FRAME a violation says nothing about them, and labelling
    them from it would be up to seven wrong labels per right one."""
    frame_key = "unk__unk__20260427T070224Z__aaaa__f000450"
    rows = [
        _cv_crop_row(0, "CONFIRM_TP"),
        _cv_crop_row(1, "CONFIRM_FP"),
        _cv_crop_row(2, "CONFIRM_FP"),
    ]
    audits = {frame_key: {"decision": "agree", "audited_verdict": "violation",
                          "ground_truth_verdict": "violation"}}

    result = build_export(rows, {}, config=PipelineConfig(), audits=audits)

    assert [case.alarm_id for case in result.cases] == [f"exec__{frame_key}__p00"]
    assert len(result.skipped) == 2
    assert "not the one" in result.skipped[0]["reason"]


def test_without_the_audit_sidecar_nothing_changes():
    """The parameter is optional and an ordinary alarm must not notice it."""
    rows = [_cv_crop_row(0, "CONFIRM_TP"), _cv_crop_row(1, "CONFIRM_FP")]

    assert len(build_export(rows, {}, config=PipelineConfig()).cases) == 2


# --- a human answering ONE crop of a crowded image -------------------------
#
# The training-label gap, and the product argument for per-crop verdicts: a
# reviewer resolving a hedge on a violation frame produced NO training sample
# for the person they had just looked at, because that person did not fire and
# the frame's verdict was not about them.


FRAME_KEY = "unk__unk__20260427T070224Z__aaaa__f000450"

_VIOLATION_AUDIT = {
    FRAME_KEY: {
        "decision": "agree",
        "audited_verdict": "violation",
        "ground_truth_verdict": "violation",
    }
}


def test_a_crop_the_reviewer_answered_is_labelled_by_that_answer():
    """Whatever the frame verdict says, and whether or not the crop fired."""
    rows = [
        _cv_crop_row(0, "CONFIRM_TP"),
        _cv_crop_row(1, "ESCALATE"),
        _cv_crop_row(2, "CONFIRM_FP"),
    ]
    reviews = {
        FRAME_KEY: {
            "verdict": "violation",
            "crop_verdicts": {f"{FRAME_KEY}__p01": "no_violation"},
        }
    }

    result = build_export(rows, reviews, config=PipelineConfig(), audits=_VIOLATION_AUDIT)

    exported = {case.alarm_id: case for case in result.cases}
    assert f"exec__{FRAME_KEY}__p01" in exported
    assert exported[f"exec__{FRAME_KEY}__p01"].metadata["human_verdict"] == "no_violation"
    assert exported[f"exec__{FRAME_KEY}__p01"].metadata["settled_by"] == "human"


def test_the_hedge_a_human_answered_used_to_produce_no_training_sample():
    """The same rows with only a FRAME answer: the hedge is settled, and then
    dropped as a bystander of a violation frame nobody attributed to it."""
    rows = [_cv_crop_row(0, "CONFIRM_TP"), _cv_crop_row(1, "ESCALATE")]
    reviews = {FRAME_KEY: {"verdict": "violation"}}

    result = build_export(rows, reviews, config=PipelineConfig(), audits=_VIOLATION_AUDIT)

    assert [case.alarm_id for case in result.cases] == [f"exec__{FRAME_KEY}__p00"]
    assert "no honest label" in result.skipped[0]["reason"]


def test_a_crop_answer_settles_a_hedge_the_frame_never_answered():
    """No frame verdict at all: the crop's own answer is the whole reason this
    row is exportable."""
    rows = [_cv_crop_row(0, "CONFIRM_TP"), _cv_crop_row(1, "ESCALATE")]
    reviews = {FRAME_KEY: {"crop_verdicts": {f"{FRAME_KEY}__p01": "violation"}}}

    result = build_export(rows, reviews, config=PipelineConfig())

    assert sorted(case.alarm_id for case in result.cases) == [
        f"exec__{FRAME_KEY}__p00",
        f"exec__{FRAME_KEY}__p01",
    ]


def test_a_crop_answered_cannot_determine_is_finished_work_and_not_a_label():
    """It clears the queue and stays out of the dataset, exactly as the frame
    verdict of the same name does -- and it is skipped for THAT reason, not as
    a bystander nobody spoke for."""
    rows = [_cv_crop_row(0, "CONFIRM_TP"), _cv_crop_row(1, "ESCALATE")]
    reviews = {
        FRAME_KEY: {"crop_verdicts": {f"{FRAME_KEY}__p01": "cannot_determine"}}
    }

    result = build_export(rows, reviews, config=PipelineConfig(), audits=_VIOLATION_AUDIT)

    assert [case.alarm_id for case in result.cases] == [f"exec__{FRAME_KEY}__p00"]
    assert result.skipped == [
        {
            "alarm_id": f"exec__{FRAME_KEY}__p01",
            "reason": "settled without a label (cannot determine / no helmet visible)",
        }
    ]


def test_a_crop_answer_is_the_label_even_on_a_crop_the_judge_fired_on():
    assert audited_crop_label(
        "violation", crop_fired=True, people_in_image=9, crop_verdict="no_violation"
    ) == "no_violation"
    assert audited_crop_label(
        "violation", crop_fired=False, people_in_image=9, crop_verdict="violation"
    ) == "violation"


def test_a_crop_nobody_answered_still_inherits_nothing():
    """The parameter is optional and the existing rule is untouched without
    it."""
    assert audited_crop_label(
        "violation", crop_fired=False, people_in_image=9, crop_verdict=None
    ) is None


# --- the FULL-FRAME dataset ------------------------------------------------
#
# The crop rule existed because this repo only kept the largest person box per
# frame, so a full frame would have taught the model that every other worker
# was background. The helmet-CV path asks the judge about EVERY person the
# detector finds, so that objection is about data this arm no longer has.


FRAME = "unk__unk__20260427T070224Z__aaaa__f000450"


def _frame_row(person_index: int, outcome: str, *, person, helmet=None, reason=None) -> dict:
    candidate = f"{FRAME}__p{person_index:02d}"
    row = _row(
        alarm_id=f"exec__{candidate}",
        outcome=outcome,
        reason_code=reason or ("helmet_worn" if outcome == "CONFIRM_FP" else "no_helmet"),
        image_key="eval/alarm_images/frame.jpg",
        person_box={"x1": person[0], "y1": person[1], "x2": person[2], "y2": person[3]},
        helmet_box=(
            {"x1": helmet[0], "y1": helmet[1], "x2": helmet[2], "y2": helmet[3]}
            if helmet
            else None
        ),
        helmet_box_source="cv" if helmet else None,
        helmet_box_confidence=0.9 if helmet else None,
    )
    row["row_kind"] = "helmet_cv"
    row["cv"] = {"frame_name": f"{FRAME}.jpg", "candidate_id": candidate,
                 "person_index": person_index, "people_in_frame": 0,
                 # The precondition: a box nobody asked about may hold a ship.
                 "gate_checked": True}
    return row


def _frame_rows(*rows: dict) -> list[dict]:
    """Stamp `people_in_frame` to match the rows given.

    The completeness guard compares the two, so a fixture that claims three
    people and supplies two is a frame the export correctly refuses -- which
    is what these rows would have been without this.
    """
    for row in rows:
        row["cv"]["people_in_frame"] = len(rows)
    return list(rows)


def test_the_frame_export_labels_every_person_in_frame_coordinates():
    """One sample per IMAGE, and the boxes are the frame's own pixels
    normalized against the frame -- not a crop's."""
    rows = _frame_rows(
        _frame_row(0, "CONFIRM_FP", person=(100, 100, 300, 400), helmet=(180, 110, 220, 150)),
        _frame_row(1, "CONFIRM_TP", person=(900, 480, 1010, 760)),
    )

    result = build_frame_export(rows, {}, config=PipelineConfig())

    assert [case.alarm_id for case in result.cases] == [FRAME]
    case = result.cases[0]
    assert case.metadata["unit"] == "full_frame"
    assert case.metadata["people_labelled"] == 2
    assert case.metadata["helmets_labelled"] == 1
    # Normalized against the 1920x1080 FRAME: the first person's centre x is
    # (100+300)/2 / 1920.
    person_boxes = [b for b in case.boxes if b.class_id == PERSON_CLASS]
    assert person_boxes[0].yolo[0] == pytest.approx(200 / 1920)
    assert person_boxes[0].yolo[1] == pytest.approx(250 / 1080)


def test_a_box_the_gate_emptied_is_not_labelled():
    """CEO: "Removed the bounding boxes where VLM says theres no human in
    it." A box that holds a ship would teach the detector to find ships."""
    rows = _frame_rows(
        _frame_row(0, "CONFIRM_FP", person=(100, 100, 300, 400), helmet=(180, 110, 220, 150)),
        _frame_row(1, "CONFIRM_FP", person=(20, 25, 45, 52), reason="no_person_present"),
    )

    case = build_frame_export(rows, {}, config=PipelineConfig()).cases[0]

    assert case.metadata["people_labelled"] == 1
    assert case.metadata["boxes_gated"] == 1
    assert [list(b.frame) for b in case.boxes if b.class_id == PERSON_CLASS] == [
        [100.0, 100.0, 300.0, 400.0]
    ]


def test_a_helmet_box_only_comes_from_a_person_the_judge_cleared():
    """On a person it called a violation there is no helmet to box, and a
    model-drawn one would contradict the verdict it ships with."""
    rows = _frame_rows(
        _frame_row(0, "CONFIRM_TP", person=(100, 100, 300, 400), helmet=(180, 110, 220, 150)),
    )

    case = build_frame_export(rows, {}, config=PipelineConfig()).cases[0]

    assert case.metadata["people_labelled"] == 1
    assert case.metadata["helmets_labelled"] == 0


def test_a_live_alarm_is_not_a_full_frame_sample():
    """It carries ONE person box for a frame that may hold eight people, so
    exporting it whole reintroduces the unlabelled-worker problem the crop
    dataset exists to avoid. Those cases stay in the crop dataset."""
    result = build_frame_export([_row()], {}, config=PipelineConfig())

    assert result.cases == []


def test_a_frame_whose_every_box_was_gated_is_skipped_with_a_reason():
    rows = _frame_rows(_frame_row(0, "CONFIRM_FP", person=(20, 25, 45, 52), reason="no_person_present"))

    result = build_frame_export(rows, {}, config=PipelineConfig())

    assert result.cases == []
    assert "gated" in result.skipped[0]["reason"]


def test_the_frame_dataset_is_written_beside_the_crop_one_not_into_it():
    """Same bucket, own prefix. Same class indices, incompatible geometry --
    mixed in one tree they train a model on two definitions of "image"."""
    assert FRAME_EXPORT_PREFIX != DEFAULT_EXPORT_PREFIX
    assert FRAME_EXPORT_PREFIX.startswith("track2/")


def test_settling_one_case_publishes_its_frame_too(frame, monkeypatch):
    """CEO: "the data is auto exported if VLM settled, and after the
    escalation and helmet box settled". Wired inside `sync_case` rather than
    at each call site, so every path that publishes a crop on settle
    publishes the frame too and no new one can forget to."""
    calls = []
    monkeypatch.setattr(
        training_export,
        "sync_frame",
        lambda case_id, *a, **k: calls.append(case_id) or {"synced": 1, "case_id": case_id},
    )
    rows = _frame_rows(
        _frame_row(0, "CONFIRM_FP", person=(100, 100, 300, 400), helmet=(180, 110, 220, 150))
    )
    rows[0]["image_key"] = "eval/alarm_images/alarm-1.jpg"

    result = training_export.sync_case(
        rows[0]["alarm_id"], rows, {}, config=PipelineConfig()
    )

    assert calls == [FRAME]
    assert result["frame"]["synced"] == 1


def test_a_human_verdict_that_contradicts_the_judge_drops_the_helmet_labels():
    """A person is a person whatever the verdict. But if a human says a frame
    holds a violation where the judge cleared everyone, at least one helmet
    box it drew is wrong and nothing says which."""
    rows = _frame_rows(
        _frame_row(0, "CONFIRM_FP", person=(100, 100, 300, 400), helmet=(180, 110, 220, 150)),
        _frame_row(1, "CONFIRM_FP", person=(900, 480, 1010, 760), helmet=(940, 490, 980, 530)),
    )
    audits = {FRAME: {"decision": "disagree", "corrected_verdict": "violation",
                      "audited_verdict": "no_violation"}}

    case = build_frame_export(rows, {}, config=PipelineConfig(), audits=audits).cases[0]

    assert case.metadata["helmets_disputed"] is True
    assert case.metadata["helmets_labelled"] == 0
    # The people stay: that is most of this dataset's value.
    assert case.metadata["people_labelled"] == 2


def test_a_human_agreeing_leaves_the_labels_alone():
    rows = _frame_rows(
        _frame_row(0, "CONFIRM_FP", person=(100, 100, 300, 400), helmet=(180, 110, 220, 150))
    )
    audits = {FRAME: {"decision": "agree", "audited_verdict": "no_violation"}}

    case = build_frame_export(rows, {}, config=PipelineConfig(), audits=audits).cases[0]

    assert case.metadata["helmets_disputed"] is False
    assert case.metadata["helmets_labelled"] == 1


def test_a_frame_that_stops_qualifying_is_retracted_not_left_stale(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    """The whole point of re-syncing on a decision: a frame that used to
    export and now cannot must come OUT rather than keep the version it
    had."""
    rows = [_frame_row(0, "CONFIRM_FP", person=(20, 25, 45, 52), reason="no_person_present")]

    result = training_export.sync_frame(FRAME, rows, {}, config=PipelineConfig())

    assert result["synced"] == 0
    assert "retracted" in result


def test_a_frame_missing_a_person_is_not_a_full_frame_sample():
    """MEASURED on the real log: built from settled rows only, 105 of 128
    frames would have labelled fewer people than the detector found, and alarm
    00e90bfd would have shipped ONE person out of twelve -- eleven workers
    taught as background."""
    rows = [_frame_row(0, "CONFIRM_FP", person=(100, 100, 300, 400), helmet=(180, 110, 220, 150))]
    rows[0]["cv"]["people_in_frame"] = 12  # the judge only ever answered for one

    result = build_frame_export(rows, {}, config=PipelineConfig())

    assert result.cases == []
    assert "background" in result.skipped[0]["reason"]


def test_an_escalated_person_still_contributes_a_person_box():
    """The box is the DETECTOR's output; it does not wait on a verdict. Only
    the helmet box does."""
    rows = _frame_rows(
        _frame_row(0, "CONFIRM_FP", person=(100, 100, 300, 400), helmet=(180, 110, 220, 150)),
        _frame_row(1, "ESCALATE", person=(900, 480, 1010, 760)),
    )

    case = build_frame_export(rows, {}, config=PipelineConfig()).cases[0]

    assert case.metadata["people_labelled"] == 2
    assert case.metadata["helmets_labelled"] == 1


def test_a_frame_nobody_gate_checked_is_exported_anyway():
    """The gate WAS a precondition here, and when it was removed on 2 Sep this
    refused everything: no row carried an answer any more, so the full-frame
    dataset silently shipped nothing -- 0 of 37 frames on the live corpus.

    Boxes now come from the vendor CV's own annotation rather than a
    detector's guess, and an auditor can strike one out by hand, so the
    certification the gate provided is provided better elsewhere."""
    rows = _frame_rows(
        _frame_row(0, "CONFIRM_FP", person=(100, 100, 300, 400), helmet=(180, 110, 220, 150))
    )
    rows[0]["cv"]["gate_checked"] = False

    result = build_frame_export(rows, {}, config=PipelineConfig())

    assert len(result.cases) == 1
    assert result.cases[0].metadata["people_labelled"] == 1


def test_a_box_nobody_could_make_out_is_not_a_person_label():
    """MEASURED on alarm 00e90bfd: eleven of twelve boxes came back `unclear`
    at 37x40 to 67x39 pixels -- "no identifiable body parts". Such a box is
    not a `person` label, and a frame holding one cannot be fully labelled,
    because an unlabelled worker is a worker taught as background."""
    rows = _frame_rows(
        _frame_row(0, "CONFIRM_FP", person=(100, 100, 300, 400), helmet=(180, 110, 220, 150)),
        _frame_row(1, "CONFIRM_FP", person=(900, 480, 940, 520)),
    )
    rows[0]["cv"]["gate_answer"] = "person"
    rows[1]["cv"]["gate_answer"] = "unclear"

    result = build_frame_export(rows, {}, config=PipelineConfig())

    assert result.cases == []
    assert "too small or too blurred" in result.skipped[0]["reason"]


def test_a_frame_the_gate_could_read_end_to_end_exports():
    """The shape that actually passes: every box came back `person`."""
    rows = _frame_rows(
        _frame_row(0, "CONFIRM_FP", person=(100, 100, 300, 400), helmet=(180, 110, 220, 150)),
        _frame_row(1, "CONFIRM_TP", person=(900, 480, 1010, 760)),
    )
    for row in rows:
        row["cv"]["gate_answer"] = "person"

    case = build_frame_export(rows, {}, config=PipelineConfig()).cases[0]

    assert case.metadata["people_labelled"] == 2
    assert case.metadata["helmets_labelled"] == 1


# --- crops the auditor struck out ----------------------------------------
#
# CEO, 2 Sep: "under audit where it shows person 1 2 3 4 5 6 etc. make those
# options selectable to the auditor can remove them so it gets removed from
# the exported training data or doesnt get exported."


def test_a_struck_out_crop_leaves_the_crop_dataset():
    """The auditor looked at this exact picture and said the box is not worth
    training on. That outranks anything the pipeline worked out about it."""
    frame_key = "unk__unk__20260427T070224Z__aaaa__f000450"
    rows = [_cv_crop_row(0, "CONFIRM_FP"), _cv_crop_row(1, "CONFIRM_FP")]
    audits = {
        frame_key: {
            "decision": "agree",
            "audited_verdict": "no_violation",
            "ground_truth_verdict": "no_violation",
            "excluded_people": [f"{frame_key}__p01"],
        }
    }

    result = build_export(rows, {}, config=PipelineConfig(), audits=audits)

    assert [case.alarm_id for case in result.cases] == [f"exec__{frame_key}__p00"]
    assert result.skipped[0]["reason"] == "excluded by the auditor"


def test_a_struck_out_crops_boxes_leave_the_full_frame_sample():
    """Same strike-out, other dataset. A whole-frame sample carries every
    person box in the picture, so one struck out has to disappear from it or
    the exclusion means nothing here."""
    rows = _frame_rows(
        _frame_row(0, "CONFIRM_FP", person=(100, 100, 300, 400), helmet=(180, 110, 220, 150)),
        _frame_row(1, "CONFIRM_FP", person=(900, 480, 1010, 760), helmet=(940, 490, 980, 530)),
    )
    audits = {
        FRAME: {
            "decision": "agree",
            "audited_verdict": "no_violation",
            "excluded_people": [f"{FRAME}__p01"],
        }
    }

    case = build_frame_export(rows, {}, config=PipelineConfig(), audits=audits).cases[0]

    assert case.metadata["people_labelled"] == 1
    assert case.metadata["helmets_labelled"] == 1


def test_striking_out_nothing_changes_nothing():
    """An empty list is how a previous strike-out is cleared, so it must read
    as "export everything" rather than as a missing field."""
    rows = _frame_rows(
        _frame_row(0, "CONFIRM_FP", person=(100, 100, 300, 400), helmet=(180, 110, 220, 150)),
        _frame_row(1, "CONFIRM_FP", person=(900, 480, 1010, 760)),
    )
    audits = {FRAME: {"decision": "agree", "audited_verdict": "no_violation",
                      "excluded_people": []}}

    case = build_frame_export(rows, {}, config=PipelineConfig(), audits=audits).cases[0]

    assert case.metadata["people_labelled"] == 2


def test_a_box_drawn_on_a_crowded_image_is_not_copied_onto_everyone():
    """A verdict about an image is about everyone in it; a helmet box is one
    person's head. Reviews became resolvable by image so an escalation
    verdict could settle its crops -- carrying the BOX across the same way
    would label one worker's helmet on another."""
    rows = _frame_rows(
        _frame_row(0, "CONFIRM_FP", person=(100, 100, 300, 400)),
        _frame_row(1, "CONFIRM_FP", person=(900, 480, 1010, 760)),
    )
    reviews = {
        FRAME: {
            "verdict": "no_violation",
            "helmet_box_status": "drawn",
            "helmet_box": {"x1": 180, "y1": 110, "x2": 220, "y2": 150},
        }
    }

    result = build_export(rows, reviews, config=PipelineConfig())

    human_boxes = [
        box for case in result.cases for box in case.boxes if box.source == "human"
    ]
    assert human_boxes == []


def test_a_box_drawn_on_a_single_person_image_still_travels():
    """Where the image holds one person, the image's box IS that person's."""
    rows = _frame_rows(_frame_row(0, "CONFIRM_FP", person=(100, 100, 300, 400)))
    reviews = {
        FRAME: {
            "verdict": "no_violation",
            "helmet_box_status": "drawn",
            "helmet_box": {"x1": 180, "y1": 110, "x2": 220, "y2": 150},
        }
    }

    result = build_export(rows, reviews, config=PipelineConfig())

    sources = [box.source for case in result.cases for box in case.boxes]
    assert "human" in sources


def test_a_drawn_box_lands_in_the_right_place_in_BOTH_datasets():
    """The whole coordinate chain, which no single test covered.

    A reviewer draws on the CROP, because that is the image they are served.
    The server maps it back to original frame pixels on save
    (`judge_box_to_original`), and that is what the sidecar stores -- the crop
    is a review-time rendering decision and a box stored against it would go
    stale the moment crop padding is retuned.

    Each dataset then normalises that frame-pixel box against its OWN image:
    the full-frame dataset against the whole frame, the crop dataset against
    the crop, because its images are crops. So "everything is in full-frame
    coordinates" is true of the STORED box and of one of the two datasets.
    """
    from ppe_pipeline.cropping import judge_box_to_original

    # A 200x400 crop cut from a 1920x1080 frame at (500, 200).
    crop = BoundingBox(500, 200, 700, 600)
    drawn = (80.0, 30.0, 120.0, 70.0)  # on the crop, as the reviewer sees it
    stored = judge_box_to_original(
        [drawn[0] / 200, drawn[1] / 400, drawn[2] / 200, drawn[3] / 400], crop
    )

    assert stored.as_tuple() == (580, 230, 620, 270)  # frame pixels

    frame = BoundingBox(0, 0, 1920, 1080)
    fx, fy, fw, fh = to_yolo(list(stored.as_tuple()), frame)
    assert (fx * 1920, fy * 1080) == pytest.approx((600.0, 250.0))
    assert (fw * 1920, fh * 1080) == pytest.approx((40.0, 40.0))

    cx, cy, cw, ch = to_yolo(list(stored.as_tuple()), crop)
    assert (cx * 200, cy * 400) == pytest.approx((100.0, 50.0))
    assert (cw * 200, ch * 400) == pytest.approx((40.0, 40.0))


def _writable(tmp_path):
    """A storage backend rooted in tmp_path, with the frame already in it.

    An explicit backend, because `LocalStorage()` roots at the working
    directory and these tests write images and labels.
    """
    import io

    from PIL import Image

    from ppe_pipeline.storage import LocalStorage

    backend = LocalStorage(tmp_path)
    buffer = io.BytesIO()
    Image.new("RGB", (1920, 1080), (90, 90, 90)).save(buffer, format="JPEG")
    backend.put_bytes("eval/alarm_images/frame.jpg", buffer.getvalue(), content_type="image/jpeg")
    return backend


def test_a_single_case_republish_writes_no_batch_manifest(tmp_path):
    """Saving one decision is not a batch. It was writing a manifest object and
    appending to the index's batch list every time, which had grown to 92
    entries for 72 cases, each one making the index slower to read and write.

    What it still writes is the case's image, label and index entry, which is
    the whole of what changed.
    """
    backend = _writable(tmp_path)
    rows = _frame_rows(
        _frame_row(0, "CONFIRM_FP", person=(100, 100, 300, 400), helmet=(180, 110, 220, 150))
    )
    result = build_frame_export(rows, {}, config=PipelineConfig())

    training_export.write_frame_export(result, backend=backend, partial=True)

    prefix = training_export.FRAME_EXPORT_PREFIX
    index, _ = backend.read_json(f"{prefix}/index.json", default={})
    assert list(index["cases"]) == [FRAME]
    assert index["batches"] == []
    assert backend.list_keys(f"{prefix}/batches") == []


def test_a_full_publish_still_records_its_batch(tmp_path):
    """The history of what went out when, which a real publish is."""
    backend = _writable(tmp_path)
    rows = _frame_rows(
        _frame_row(0, "CONFIRM_FP", person=(100, 100, 300, 400), helmet=(180, 110, 220, 150))
    )
    result = build_frame_export(rows, {}, config=PipelineConfig())

    training_export.write_frame_export(result, backend=backend)

    prefix = training_export.FRAME_EXPORT_PREFIX
    index, _ = backend.read_json(f"{prefix}/index.json", default={})
    assert len(index["batches"]) == 1
    assert backend.list_keys(f"{prefix}/batches")


def test_a_failed_image_upload_is_reported_not_swallowed(tmp_path):
    """The publish IS the image and the label. A helper that swallowed those
    would turn a failed upload into a case that silently never reached the
    dataset while the screen said it had.

    A stale copy in the other split is different: it 404s on almost every
    case, because almost every case has only ever been in one split.
    """
    backend = _writable(tmp_path)
    boom = RuntimeError("s3 said no")

    def fail():
        raise boom

    with pytest.raises(RuntimeError):
        training_export._in_parallel([fail])

    # ...while an optional job that fails is not fatal.
    training_export._in_parallel([lambda: None], [fail])


def test_clearing_a_frame_the_judge_fired_on_KEEPS_its_helmet_labels():
    """The other direction, which used to be treated the same and should not
    be. A helmet box exists only on a crop the judge CLEARED, and a human
    saying "no violation" is overruling the crop that FIRED -- which carries
    no helmet box in the first place. The cleared crops are untouched by that
    finding, and the human agreeing nobody was in violation is consistent
    with their helmets being worn.

    Measured on frame 1821747e: seven helmet boxes were dropped because a
    human overruled the one crop that fired.
    """
    rows = _frame_rows(
        _frame_row(0, "CONFIRM_TP", person=(100, 100, 300, 400)),
        _frame_row(1, "CONFIRM_FP", person=(900, 480, 1010, 760), helmet=(940, 490, 980, 530)),
    )
    audits = {FRAME: {"decision": "disagree", "corrected_verdict": "no_violation",
                      "audited_verdict": "violation"}}

    case = build_frame_export(rows, {}, config=PipelineConfig(), audits=audits).cases[0]

    assert case.metadata["helmets_disputed"] is False
    assert case.metadata["helmets_labelled"] == 1
    assert case.metadata["people_labelled"] == 2
