"""Tests for the escalation queues — the pure derivation logic.

These exercise `ppe_pipeline.escalations` directly against synthetic audit
rows. The HTTP layer is covered separately in test_api_escalations.py.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ppe_pipeline.config import PipelineConfig
from ppe_pipeline.escalations import (
    BOX_STATUS_OPTIONS,
    CROP_VERDICTS,
    VERDICT_OPTIONS,
    audit_reopened_box,
    awaiting_verdict_crops,
    build_queue,
    crop_verdict,
    frame_available,
    is_exportable,
    needs_box,
    needs_verdict,
    queue_counts,
    save_review,
    load_reviews,
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
        "person_box": {"x1": 10, "y1": 20, "x2": 110, "y2": 220},
        "judge_model": "qwen.qwen3-vl-235b-a22b-instruct",
    }
    row.update(overrides)
    return row


# --- which cases need a human -------------------------------------------


def test_an_escalated_alarm_needs_a_verdict():
    assert needs_verdict(_row(outcome="ESCALATE")) is True


@pytest.mark.parametrize("outcome", ["CONFIRM_TP", "CONFIRM_FP"])
def test_a_settled_alarm_does_not_need_a_verdict(outcome):
    assert needs_verdict(_row(outcome=outcome)) is False


def test_a_vlm_fallback_box_needs_a_human_box():
    """The fallback box is the judge's loose inline one — not trainable."""
    config = PipelineConfig()
    assert needs_box(_row(helmet_box_source="vlm"), config) is True


def test_a_low_confidence_detector_box_needs_a_human_box():
    config = PipelineConfig()  # review_confidence defaults to 0.5
    row = _row(helmet_box_source="cv", helmet_box_confidence=0.4)

    assert needs_box(row, config) is True


def test_a_confident_detector_box_does_not():
    config = PipelineConfig()
    row = _row(helmet_box_source="cv", helmet_box_confidence=0.8)

    assert needs_box(row, config) is False


def test_the_confidence_floor_comes_from_config_not_a_literal():
    """Raising review_confidence must pull more cases into the queue."""
    row = _row(helmet_box_source="cv", helmet_box_confidence=0.6)
    lenient = PipelineConfig()
    strict = PipelineConfig()
    strict.helmet_box.review_confidence = 0.9

    assert needs_box(row, lenient) is False
    assert needs_box(row, strict) is True


def test_a_case_with_no_box_at_all_needs_one():
    assert needs_box(_row(helmet_box_source=None), PipelineConfig()) is True


@pytest.mark.parametrize(
    "overrides",
    [
        {"outcome": "CONFIRM_TP"},              # a real violation: no helmet to box
        {"outcome": "ESCALATE"},                # unsettled: belongs in the other queue
        {"reason_code": "bare_head"},           # cleared for a different reason
        {"hazard_type": "too_close_to_vehicle"},  # not a helmet case at all
    ],
)
def test_non_helmet_clearances_never_enter_the_box_queue(overrides):
    assert needs_box(_row(helmet_box_source="vlm", **overrides), PipelineConfig()) is False


# --- queue construction --------------------------------------------------


def test_queues_are_disjoint():
    """A case is in exactly one queue — an escalation has no verdict to
    clear a helmet, and a cleared helmet needs no verdict."""
    rows = [
        _row(alarm_id="esc", outcome="ESCALATE"),
        _row(alarm_id="box", helmet_box_source="vlm"),
    ]

    verdict_ids = {i.alarm_id for i in build_queue(rows, "verdict", {})}
    box_ids = {i.alarm_id for i in build_queue(rows, "box", {})}

    assert verdict_ids == {"esc"}
    assert box_ids == {"box"}
    assert verdict_ids.isdisjoint(box_ids)


def test_a_reviewed_case_drops_out_of_the_queue():
    """This is what makes saving advance to the next case."""
    rows = [_row(alarm_id="esc", outcome="ESCALATE")]
    reviews = {"esc": {"verdict": "violation"}}

    assert build_queue(rows, "verdict", reviews) == []
    assert len(build_queue(rows, "verdict", reviews, include_reviewed=True)) == 1


def test_a_half_filled_review_does_not_count_as_reviewed():
    """A stray key must not silently clear a case from someone's queue."""
    rows = [_row(alarm_id="esc", outcome="ESCALATE")]

    assert len(build_queue(rows, "verdict", {"esc": {"reviewed_by": "Han"}})) == 1


def test_a_drawn_box_clears_the_box_queue():
    rows = [_row(alarm_id="b", helmet_box_source="vlm")]
    reviews = {"b": {"helmet_box_status": "drawn", "helmet_box": [1, 2, 3, 4]}}

    assert build_queue(rows, "box", reviews) == []


def test_no_helmet_visible_also_clears_the_box_queue():
    """A reviewer who cannot find a helmet has still finished the case."""
    rows = [_row(alarm_id="b", helmet_box_source="vlm")]
    reviews = {"b": {"helmet_box_status": "no_helmet_visible"}}

    assert build_queue(rows, "box", reviews) == []


def test_newest_first():
    rows = [
        _row(alarm_id="old", outcome="ESCALATE", timestamp="2026-08-01T00:00:00+00:00"),
        _row(alarm_id="new", outcome="ESCALATE", timestamp="2026-08-27T00:00:00+00:00"),
    ]

    assert [i.alarm_id for i in build_queue(rows, "verdict", {})] == ["new", "old"]


def test_rows_without_an_alarm_id_are_skipped():
    """There is nothing to key a decision on, so such a row is unreviewable
    and unexportable — surfacing it would be a dead end in the UI."""
    rows = [_row(alarm_id="", outcome="ESCALATE"), _row(alarm_id=None, outcome="ESCALATE")]

    assert build_queue(rows, "verdict", {}) == []


@pytest.mark.parametrize("image_key", [None, "", "streamlit-5458985889.jpg"])
def test_a_case_whose_frame_was_never_stored_is_not_offered_as_work(image_key):
    """Nothing a reviewer does can settle a case with no image, so listing it
    made an entirely unworkable queue look like a backlog.

    The bare-filename form is the one that actually matters: every row in
    production has one, because the old writer recorded an `image_key` it
    never stored bytes at. Checking the key's PRESENCE therefore passed them
    all as reviewable and the reviewer got "Image not found"."""
    rows = [_row(alarm_id="e", outcome="ESCALATE", image_key=image_key)]

    assert build_queue(rows, "verdict", {}) == []
    # ...but it is counted, so an empty queue is explicable.
    assert queue_counts(rows, {})["verdict"]["unreviewable"] == 1
    # ...and still reachable on purpose, so the API can explain itself.
    shown = build_queue(rows, "verdict", {}, include_unreviewable=True)
    assert len(shown) == 1 and shown[0].frame_available is False


def test_a_stored_frame_makes_a_case_workable():
    rows = [_row(alarm_id="e", outcome="ESCALATE",
                 image_key="eval/alarm_images/e.jpg")]

    items = build_queue(rows, "verdict", {})

    assert len(items) == 1
    assert items[0].frame_available is True


def test_malformed_rows_do_not_break_the_queue():
    rows = ["not a dict", None, _row(alarm_id="ok", outcome="ESCALATE")]

    assert [i.alarm_id for i in build_queue(rows, "verdict", {})] == ["ok"]


def test_item_carries_the_context_a_reviewer_needs():
    rows = [_row(alarm_id="e", outcome="ESCALATE", reasoning="Head is occluded.")]

    item = build_queue(rows, "verdict", {})[0]

    assert item.person_box == [10.0, 20.0, 110.0, 220.0]
    assert item.image_width == 1920 and item.image_height == 1080
    assert item.judge_reasoning == "Head is occluded."
    assert item.judge_model == "qwen.qwen3-vl-235b-a22b-instruct"
    assert item.violation_type == "helmet"


def test_counts_cover_both_queues():
    rows = [
        _row(alarm_id="e1", outcome="ESCALATE"),
        _row(alarm_id="e2", outcome="ESCALATE"),
        _row(alarm_id="b1", helmet_box_source="vlm"),
    ]
    reviews = {"e1": {"verdict": "violation"}}

    counts = queue_counts(rows, reviews)

    assert counts["verdict"] == {"pending": 1, "total": 2, "unreviewable": 0}
    assert counts["box"] == {"pending": 1, "total": 1, "unreviewable": 0}


# --- persistence ---------------------------------------------------------


def test_save_review_stamps_reviewer_and_time(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    when = datetime(2026, 8, 27, 12, 0, tzinfo=timezone.utc)

    entry = save_review("a1", {"verdict": "violation"}, reviewer="Han", path="reviews.json", now=when)

    assert entry["verdict"] == "violation"
    assert entry["reviewed_by"] == "Han"
    assert entry["reviewed_at"] == when.isoformat()


def test_save_review_merges_rather_than_replaces(tmp_path, monkeypatch):
    """A box decision must not wipe a verdict recorded earlier on the same
    alarm, and vice versa."""
    monkeypatch.chdir(tmp_path)
    save_review("a1", {"verdict": "violation"}, path="reviews.json")

    entry = save_review("a1", {"helmet_box_status": "no_helmet_visible"}, path="reviews.json")

    assert entry["verdict"] == "violation"
    assert entry["helmet_box_status"] == "no_helmet_visible"


def test_save_review_leaves_other_cases_alone(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    save_review("a1", {"verdict": "violation"}, path="reviews.json")
    save_review("a2", {"verdict": "no_violation"}, path="reviews.json")

    reviews = load_reviews("reviews.json")

    assert reviews["a1"]["verdict"] == "violation"
    assert reviews["a2"]["verdict"] == "no_violation"


def test_load_reviews_of_a_missing_file_is_empty(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert load_reviews("nope.json") == {}


def test_the_vocabularies_are_what_the_ui_and_export_expect():
    assert VERDICT_OPTIONS == ["violation", "no_violation", "cannot_determine"]
    assert BOX_STATUS_OPTIONS == [
        "drawn",
        "accepted_cv",
        "no_helmet_present",
        "no_helmet_visible",
    ]


# --- regressions from the adversarial review ----------------------------


@pytest.mark.parametrize("confidence", ["", "n/a", [], {}, None, float("nan")])
def test_an_unusable_confidence_queues_the_case_rather_than_500ing(confidence):
    """One malformed row used to take the whole endpoint down, hiding every
    good case. An unreadable confidence is also not evidence of a good box,
    so the case must be queued, not dropped -- NaN especially, since every
    comparison against it is False."""
    row = _row(helmet_box_source="cv", helmet_box_confidence=confidence)

    assert needs_box(row, PipelineConfig()) is True


def test_confidence_exactly_at_the_floor_is_trusted():
    """The boundary the config knob controls: `< floor` needs review, so
    equal to the floor does not."""
    row = _row(helmet_box_source="cv", helmet_box_confidence=0.5)

    assert needs_box(row, PipelineConfig()) is False


def test_a_rerun_alarm_is_listed_once_not_twice():
    """The audit log is append-only, so re-running an alarm appends a second
    row for the same id -- and the review sidecar is keyed by id alone.
    Listing both double-counted the badge and let one save record a verdict
    against a row the reviewer never saw."""
    rows = [
        _row(alarm_id="A", outcome="ESCALATE", timestamp="2026-08-01T00:00:00+00:00",
             image_width=4000, image_height=3000),
        _row(alarm_id="A", outcome="ESCALATE", timestamp="2026-08-27T00:00:00+00:00",
             image_width=64, image_height=48),
    ]

    items = build_queue(rows, "verdict", {})

    assert len(items) == 1
    # The newest row wins -- it is also the run whose frame is on disk, since
    # a re-run overwrites the frame at the same key.
    assert (items[0].image_width, items[0].image_height) == (64, 48)
    assert queue_counts(rows, {})["verdict"] == {"pending": 1, "total": 1, "unreviewable": 0}


def test_a_non_string_timestamp_does_not_break_sorting():
    rows = [
        _row(alarm_id="a", outcome="ESCALATE", timestamp=17),
        _row(alarm_id="b", outcome="ESCALATE", timestamp="2026-08-27T00:00:00+00:00"),
    ]

    assert len(build_queue(rows, "verdict", {})) == 2


def test_a_hand_edited_review_entry_does_not_break_the_queue():
    """The sidecar is a JSON file someone may edit; only its top level was
    ever type-checked."""
    rows = [_row(alarm_id="a", outcome="ESCALATE")]

    assert len(build_queue(rows, "verdict", {"a": "not a dict"})) == 1


def test_an_escalation_a_human_cleared_reaches_the_box_queue():
    """`_resolve_outcome` downgrades a low-confidence FP to ESCALATE, so a
    worn helmet routinely arrives as an escalation. Gating the box queue on
    the judge's outcome alone let those cases fall out of BOTH queues and
    ship with only the loose VLM fallback box -- exactly what the box queue
    exists to prevent."""
    rows = [_row(alarm_id="e", outcome="ESCALATE", reason_code="helmet_worn",
                 helmet_box_source="vlm")]

    assert build_queue(rows, "box", {}) == []

    cleared = {"e": {"verdict": "no_violation"}}
    assert [i.alarm_id for i in build_queue(rows, "box", cleared)] == ["e"]


def test_confirming_a_real_violation_does_not_ask_for_a_helmet_box():
    """There is no helmet to draw on a genuine bare-head violation."""
    rows = [_row(alarm_id="e", outcome="ESCALATE", reason_code="helmet_worn",
                 helmet_box_source="vlm")]

    assert build_queue(rows, "box", {"e": {"verdict": "violation"}}) == []


# --- accepting the detector's box ------------------------------------------


def test_an_accepted_box_is_a_label():
    """Redrawing a box that is already right is wasted work and worse data: a
    hand-drawn copy of a good detector box is less accurate than the box it
    replaces. The queue asks for a human because the detector was UNSURE, not
    because it was wrong."""
    assert is_exportable({"helmet_box_status": "accepted_cv"}) is True
    assert is_exportable({"helmet_box_status": "drawn"}) is True
    assert is_exportable({"helmet_box_status": "no_helmet_visible"}) is False


def test_an_accepted_case_counts_as_reviewed_and_leaves_the_queue():
    rows = [_row(alarm_id="acc", reason_code="helmet_worn", helmet_box_source="vlm")]
    reviews = {"acc": {"helmet_box_status": "accepted_cv"}}

    pending = build_queue(rows, "box", reviews, config=PipelineConfig())

    assert pending == []


# --- an audit that rejects only the box ------------------------------------


def test_a_box_wrong_audit_sends_the_case_back_for_a_box():
    """The verdict stands; the label does not. Recording it as a disagreement
    would blame the judge for a verdict it got right."""
    assert audit_reopened_box({"decision": "box_wrong"}) is True


def test_a_box_wrong_audit_reopens_a_case_that_was_already_boxed():
    """This is the normal case: the box being rejected is usually one someone
    already drew or accepted."""
    rows = [_row(alarm_id="redo", reason_code="helmet_worn", helmet_box_source="vlm")]
    reviews = {"redo": {"helmet_box_status": "drawn", "helmet_box": [1, 2, 3, 4]}}
    audits = {"redo": {"decision": "box_wrong"}}

    pending = build_queue(rows, "box", reviews, config=PipelineConfig(), audits=audits)

    assert [i.alarm_id for i in pending] == ["redo"]


def test_an_escalation_a_human_cleared_still_needs_a_helmet_box():
    """"No violation" on a helmet alarm MEANS a helmet is worn, so there is a
    helmet to box. The judge's reason code explains the judge's own verdict,
    so an escalation's reason is why it gave up (`head_occluded`) and can
    never be `helmet_worn`. Gating on it made this route unreachable: the case
    fell out of both queues and shipped a person box with NO helmet box,
    teaching the detector "no helmet here" on a case a human had just cleared.
    """
    row = _row(
        alarm_id="esc-cleared",
        outcome="ESCALATE",
        reason_code="head_occluded",
        helmet_box_source=None,
        helmet_box_confidence=None,
    )
    review = {"verdict": "no_violation"}

    assert needs_box(row, PipelineConfig(), review) is True


def test_such_a_case_does_not_export_until_the_box_is_drawn():
    """The export gate reads `needs_box`, so the same fix keeps the wrong
    label out of the training set."""
    from ppe_pipeline.training_export import build_export

    row = _row(
        alarm_id="esc-cleared",
        outcome="ESCALATE",
        reason_code="head_occluded",
        image_key="eval/alarm_images/esc-cleared.jpg",
        helmet_box_source=None,
        helmet_box_confidence=None,
    )
    reviews = {"esc-cleared": {"verdict": "no_violation"}}

    result = build_export([row], reviews, config=PipelineConfig())

    assert result.cases == []


def test_a_helmet_violation_does_not_ask_for_a_helmet_box():
    """The judge confirming a violation means no helmet is worn, so there is
    nothing to box and the case is settled as it stands."""
    row = _row(alarm_id="tp", outcome="CONFIRM_TP", reason_code="bare_head")

    assert needs_box(row, PipelineConfig(), {}) is False


def test_a_non_helmet_hazard_never_asks_for_a_helmet_box():
    row = _row(alarm_id="glove", hazard_type="glove", outcome="CONFIRM_FP",
               reason_code="helmet_worn")

    assert needs_box(row, PipelineConfig(), {}) is False


def test_an_absolute_key_is_still_a_key():
    """The alarm image directory may be configured absolute -- the test suite
    configures exactly that -- so this function cannot reject absolute keys.
    Whether an absolute key belongs to this deployment is answered against
    the configuration, in `routers/escalations._frame_is_fetchable_here`."""
    assert frame_available("eval/alarm_images/case0.jpg") is True
    assert frame_available("C:/Users/shali/tmp/alarm_images/case0.jpg") is True
    # Unchanged: a bare filename is a row from before frames were persisted.
    assert frame_available("streamlit-5458985889.jpg") is False
    assert frame_available(None) is False


def test_a_case_reopened_by_an_audit_says_so():
    """A VIOLATION stamp beside "draw the helmet" reads as a contradiction.
    The case is here because an audit corrected it to no_violation -- a human
    saying the helmet IS worn -- and the sheet has to be able to say that,
    or a reviewer sees only the overruled verdict.

    Real case: `Helmet Hatch_23_frame_0011`, judged CONFIRM_TP / bare_head,
    audited `disagree` -> `no_violation`.
    """
    rows = [_row(alarm_id="hatch", outcome="CONFIRM_TP", reason_code="bare_head")]
    audits = {"hatch": {"decision": "disagree", "corrected_verdict": "no_violation",
                        "audited_verdict": "violation"}}

    items = build_queue(rows, "box", {}, audits=audits)

    assert len(items) == 1
    assert items[0].requeued_by_audit is True
    assert items[0].audit_corrected_verdict == "no_violation"


def test_an_ordinary_box_case_is_not_flagged_as_reopened():
    """The flag has to mean something: a cleared helmet needing a box is the
    queue's normal population and nothing overruled it."""
    rows = [_row(alarm_id="b", helmet_box_source="vlm")]

    items = build_queue(rows, "box", {})

    assert len(items) == 1 and items[0].requeued_by_audit is False


# --- a reopened box has to be answerable ---------------------------------


def _audited_no_violation(at="2026-09-02T03:04:45+00:00") -> dict:
    return {
        "decision": "disagree",
        "corrected_verdict": "no_violation",
        "audited_at": at,
    }


def test_an_audit_reopens_the_box_when_no_box_has_been_saved_since():
    """The case was exported as a violation and an auditor says the helmet is
    worn, so somebody has to draw one."""
    assert audit_reopened_box(_audited_no_violation()) is True
    assert audit_reopened_box(_audited_no_violation(), {"helmet_box_status": None}) is True


def test_a_box_saved_after_the_audit_answers_it():
    """MEASURED on the live corpus, 2 Sep: two cases audited at 03:04:45 and
    given a box at 03:34:55 were still in the queue. The audit entry keeps
    saying "this case needs a box" for as long as it exists, so without a way
    to answer it the case is pulled back every time the queue is built,
    however many boxes are saved."""
    review = {"helmet_box_status": "accepted_cv", "reviewed_at": "2026-09-02T03:34:55+00:00"}

    assert audit_reopened_box(_audited_no_violation(), review) is False


def test_a_box_saved_BEFORE_the_audit_does_not_answer_it():
    """That box is what the auditor was overruling."""
    review = {"helmet_box_status": "accepted_cv", "reviewed_at": "2026-09-02T02:00:00+00:00"}

    assert audit_reopened_box(_audited_no_violation(), review) is True


def test_answering_the_verdict_does_not_answer_the_box():
    """Saving a verdict on the same case touches `reviewed_at` too, and says
    nothing about the box."""
    review = {"verdict": "no_violation", "reviewed_at": "2026-09-02T03:34:55+00:00"}

    assert audit_reopened_box(_audited_no_violation(), review) is True


def test_a_collapsed_image_is_routed_by_its_crops_not_its_outcome():
    """An image's outcome is an OR, so a frame the judge fired on AND hedged
    about reads CONFIRM_TP -- and it still owes an answer about the crop it
    hedged on. The review answers the crop (CEO, 5 Sep), so nothing about
    another crop short-circuits it."""
    fired_and_hedged = _row(outcome="CONFIRM_TP", awaiting_verdict=["frame__p01"])

    assert needs_verdict(fired_and_hedged) is True
    assert needs_verdict(_row(outcome="ESCALATE", awaiting_verdict=[])) is False


def test_a_live_alarm_has_no_crop_list_and_keeps_the_old_rule():
    """One row, one crop, its own image: its outcome IS the whole answer."""
    assert needs_verdict(_row(outcome="ESCALATE")) is True
    assert needs_verdict(_row(outcome="CONFIRM_TP")) is False


# --- a verdict about ONE crop ---------------------------------------------
#
# A frame answer says what the picture is. Two people the judge could not read
# need two answers, and the export needs a per-crop one to produce a training
# sample for a hedge resolved on a crowded violation frame.


def _hedge(person_index: int, outcome: str = "ESCALATE") -> dict:
    candidate = f"frame__p{person_index:02d}"
    return _row(
        alarm_id=f"exec__{candidate}",
        outcome=outcome,
        cv={"candidate_id": candidate, "person_index": person_index},
    )


def test_a_crop_verdict_is_read_by_candidate_id():
    entry = {CROP_VERDICTS: {"frame__p01": "violation"}}

    assert crop_verdict(entry, _hedge(1)) == "violation"
    assert crop_verdict(entry, _hedge(2)) is None


@pytest.mark.parametrize(
    "stored",
    ["frame__p01", ["frame__p01"], {"frame__p01": "maybe"}, {"frame__p01": None}],
)
def test_a_hand_edited_crop_verdict_is_not_an_answer(stored):
    """The sidecar is JSON a human can edit, and a verdict spelled some other
    way must not reach a screen as one."""
    assert crop_verdict({CROP_VERDICTS: stored}, _hedge(1)) is None


def test_a_crop_with_its_own_answer_is_no_longer_owed():
    """The point of the whole change: one hedge answered, its neighbour still
    open, on a frame nobody has given an overall verdict for."""
    crops = [_hedge(1), _hedge(2)]
    review = {CROP_VERDICTS: {"frame__p01": "violation"}}

    assert awaiting_verdict_crops(crops, review, crops) == ["frame__p02"]


def test_the_crop_answer_beats_the_frame_answer_it_contradicts():
    """A frame `no_violation` covers the crops nobody singled out; a crop the
    reviewer DID single out is answered by what they said about it."""
    crops = [_hedge(1), _hedge(2)]
    review = {
        "verdict": "no_violation",
        "reviewed_at": "2026-08-27T11:00:00+00:00",
        CROP_VERDICTS: {"frame__p01": "violation"},
    }

    assert awaiting_verdict_crops(crops, review, crops) == []
    assert crop_verdict(review, _hedge(1)) == "violation"
    assert crop_verdict(review, _hedge(2)) is None


def test_a_crop_answer_does_not_improve_the_auto_annotation_rate():
    """`rolling_metrics._images` passes NO review, because a frame a human has
    since answered still needed one. That must not change because the answer
    is now per crop."""
    crops = [_hedge(1), _hedge(2)]

    assert awaiting_verdict_crops(crops) == ["frame__p01", "frame__p02"]


def test_a_crop_verdict_is_not_answered_away_by_a_later_re_run():
    """`_reviewed_before` refuses a frame answer for a crop that did not exist
    yet. A crop answer needs no such guard -- it is filed against a candidate
    id, so a person a re-run added has no answer to inherit."""
    added_later = _hedge(2)
    added_later["timestamp"] = "2026-08-27T12:00:00+00:00"
    review = {
        "verdict": "no_violation",
        "reviewed_at": "2026-08-27T11:00:00+00:00",
        CROP_VERDICTS: {"frame__p01": "no_violation"},
    }
    crops = [_hedge(1), added_later]

    assert awaiting_verdict_crops(crops, review, crops) == ["frame__p02"]
