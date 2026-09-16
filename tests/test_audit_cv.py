"""Tests for the helmet-CV seam in the audit module.

CV rows share `eval/track1_audit_log.json` with live alarms (CEO decision),
so most of what is under test here is separation: a CV row must never reach
the live audit queue, and a live alarm must never reach the CV comparison.
The rest is that one human decision scores both models through the SAME
`audited_ground_truth` and `metrics.classify` — two scoring functions would
be two definitions of right.
"""

from __future__ import annotations

import pytest

from ppe_pipeline.audit import (
    _crop_review,
    build_audit_cases,
    build_audit_queue,
    cv_audit_items,
    cv_confusion,
    cv_verdict,
    cv_vs_judge,
    arm_metrics,
    group_by_frame,
    ground_truth_key,
    is_cv_row,
    reviewer_verdict,
    vlm_confusion,
    vlm_frame_verdicts,
)
from ppe_pipeline.config import PipelineConfig
from ppe_pipeline.helmet_cv.records import execution_id

FRAME = "unk__unk__20260427T070224Z__aaaa__f000450.jpg"
#: The human decision is keyed by the FRAME, not the person in it, so this is
#: what an audits dict is keyed by throughout this file.
FRAME_KEY = "unk__unk__20260427T070224Z__aaaa__f000450"
CANDIDATE = f"{FRAME_KEY}__p00"


def _cv_block(**overrides) -> dict:
    block = {
        "cv_run_id": "cv-1",
        "candidate_id": CANDIDATE,
        "frame_name": "unk__unk__20260427T070224Z__aaaa__f000450.jpg",
        "clip_alarm_id": "aaaa",
        "frame_role": "alarm",
        "person_index": 0,
        "person_box": [10.0, 10.0, 50.0, 200.0],
        "person_confidence": 0.9,
        "people_in_frame": 1,
        "helmets_in_frame": 0,
    }
    block.update(overrides)
    return block


def _cv_row(alarm_id: str = CANDIDATE, outcome: str = "CONFIRM_TP", **overrides) -> dict:
    row = {
        "alarm_id": alarm_id,
        "hazard_type": "no_helmet",
        "outcome": outcome,
        "reason_code": "no_helmet",
        "confidence": 0.9,
        "timestamp": "2026-08-30T10:00:00+00:00",
        "image_key": "eval/alarm_images/candidate.jpg",
        "crop_box": {"x1": 0, "y1": 0, "x2": 200, "y2": 300},
        "row_kind": "helmet_cv",
        "cv": _cv_block(),
    }
    row.update(overrides)
    return row


def _live_row(alarm_id: str = "live-1") -> dict:
    return {
        "alarm_id": alarm_id,
        "hazard_type": "no_helmet",
        "outcome": "CONFIRM_FP",
        "reason_code": "helmet_worn",
        "timestamp": "2026-08-30T10:00:00+00:00",
        "image_key": "eval/alarm_images/live.jpg",
        "crop_box": {"x1": 0, "y1": 0, "x2": 200, "y2": 300},
        "helmet_box_source": "cv",
        "helmet_box_confidence": 0.9,
    }


def _audit(decision: str, *, audited_verdict: str | None, corrected: str | None = None) -> dict:
    return {
        "decision": decision,
        "corrected_verdict": corrected,
        "audited_verdict": audited_verdict,
        "source": "helmet_cv_review",
        "audited_by": "Han",
        "note": "",
        "audited_at": "2026-08-31T09:00:00+00:00",
    }


# --- separation -----------------------------------------------------------


def test_a_cv_row_is_recognisable_from_its_top_level_marker():
    """Filterable without reaching into the nested block."""
    assert is_cv_row(_cv_row())
    assert not is_cv_row(_live_row())


def test_a_cv_row_never_reaches_the_live_audit_queue():
    """It is a frame from a clip somebody already alarmed on, not an alarm.
    Counting it would put a synthetic case in the shadow-mode agreement rate
    the 85% target is read off."""
    items = build_audit_queue([_live_row(), _cv_row()], {}, {}, config=PipelineConfig())

    assert [item.alarm_id for item in items] == ["live-1"]


def test_a_live_alarm_never_reaches_the_cv_comparison():
    assert [item.alarm_id for item in cv_audit_items([_live_row(), _cv_row()], {})] == [CANDIDATE]


def test_the_cv_block_is_carried_nested_not_flattened():
    """The detector's geometry and the judge's boxes are different claims by
    different models; flat, one would overwrite the other with no way to tell
    which."""
    item = cv_audit_items([_cv_row()], {})[0]

    assert item.cv["cv_run_id"] == "cv-1"
    assert item.cv["frame_role"] == "alarm"
    assert item.cv["person_box"] == [10.0, 10.0, 50.0, 200.0]


def test_a_live_alarm_carries_an_empty_cv_block():
    item = build_audit_queue([_live_row()], {}, {}, config=PipelineConfig())[0]

    assert item.cv == {}


def test_an_escalated_cv_candidate_is_still_in_the_comparison():
    """`build_audit_queue` would drop it -- an ESCALATE is unsettled for a
    live alarm. There is no escalation queue for the pilot, and dropping
    every escalated candidate would remove a large share of the corpus
    invisibly."""
    items = cv_audit_items([_cv_row(outcome="ESCALATE")], {})

    assert len(items) == 1
    assert items[0].judge_outcome == "ESCALATE"


def test_the_newest_row_per_candidate_wins():
    old = _cv_row(timestamp="2026-08-30T10:00:00+00:00", outcome="CONFIRM_TP")
    new = _cv_row(timestamp="2026-08-31T10:00:00+00:00", outcome="CONFIRM_FP")

    items = cv_audit_items([old, new], {})

    assert len(items) == 1
    assert items[0].judge_outcome == "CONFIRM_FP"


# --- what the gate claimed ------------------------------------------------


# --- the review encoding --------------------------------------------------


def test_a_reviewers_verdict_is_recoverable_from_an_agreement():
    """Reviews are recorded blind: the human states a verdict, the server
    encodes it as agree/disagree. This reverses the encoding."""
    assert reviewer_verdict(_audit("agree", audited_verdict="violation")) == "violation"


def test_a_reviewers_verdict_is_recoverable_from_a_disagreement():
    entry = _audit("disagree", audited_verdict="violation", corrected="no_violation")

    assert reviewer_verdict(entry) == "no_violation"


# --- scoring --------------------------------------------------------------


def test_cv_is_scored_against_the_human_not_the_judge():
    """One reviewer's answer scores both arms, through the same
    `audited_ground_truth` the VLM arm is scored with."""
    audits = {FRAME_KEY: _audit("agree", audited_verdict="violation")}

    confusion = cv_confusion([_cv_row()], audits)

    assert confusion["overall"]["counts"]["true_positive"] == 1
    assert confusion["overall"]["precision"] == pytest.approx(1.0)


def test_an_alarm_the_human_cleared_is_a_cv_false_positive():
    audits = {
        FRAME_KEY: _audit("disagree", audited_verdict="violation", corrected="no_violation")
    }

    confusion = cv_confusion([_cv_row()], audits)

    assert confusion["overall"]["counts"]["false_positive"] == 1
    assert confusion["overall"]["precision"] == pytest.approx(0.0)


def test_a_cannot_determine_review_produces_no_label():
    """A reviewer finding the evidence never supported a verdict is a real
    finding, but not a ground truth to score against."""
    audits = {
        FRAME_KEY: _audit("disagree", audited_verdict="violation", corrected="cannot_determine")
    }

    assert cv_confusion([_cv_row()], audits)["overall"]["n"] == 0


def test_an_unreviewed_candidate_contributes_nothing():
    assert cv_confusion([_cv_row()], {})["overall"]["n"] == 0


def test_the_population_caveat_travels_with_the_numbers():
    """The payload has to say what its numbers cover, and say it in ONE place.

    Two things belong here: the coverage limit (footage that never fired is
    absent) and the bias (the queue is all CV-positive frames, so CV's precision
    is unblinded). The first was there; the second was not, and they are
    different failures.
    """
    confusion = cv_confusion([_cv_row()], {FRAME_KEY: _audit("agree", audited_verdict="violation")})

    caveat = confusion["population"]
    assert "never raised an alarm" in caveat
    assert "Frames production CV fired on" in caveat
    # The bias disclosure has to travel with the numbers too. The queue is all
    # CV-positive frames, so CV's precision is unblinded, and that is a
    # different failure from the coverage limit above.
    assert "AGREEMENT-AFTER-SEEING" in caveat


def test_every_rate_is_none_rather_than_zero_on_an_empty_denominator():
    """`ConfusionCounts` returns 0.0 for all four when nothing is decided, and
    0.0 on the wire reads as "got everything wrong" rather than "not measured
    yet"."""
    metrics = arm_metrics(cv_confusion([_cv_row()], {})["overall"])

    assert metrics == {
        "n": 0,
        "sample_is_thin": True,
        "counts": {"true_positive": 0, "false_negative": 0, "false_positive": 0,
                   "true_negative": 0, "uncertain": 0},
        "precision": None,
        "recall": None,
        "f1": None,
        "accuracy": None,
    }


def test_each_rate_is_nulled_against_its_own_denominator():
    """A corpus of nothing but true negatives has a real accuracy and no
    precision and no recall at all, so the four cannot share one null test.

    Scored on the VLM arm, because the CV arm cannot reach a true negative:
    every alarm in this corpus is one production CV raised, so CV asserted a
    violation on all of them -- see `cv_verdict`."""
    cleared = _cv_row(outcome="CONFIRM_FP")
    audits = {FRAME_KEY: _audit("agree", audited_verdict="no_violation")}

    metrics = arm_metrics(vlm_confusion([cleared], audits))

    assert metrics["counts"]["true_negative"] == 1
    assert metrics["precision"] is None
    assert metrics["recall"] is None
    assert metrics["f1"] is None
    assert metrics["accuracy"] == pytest.approx(1.0)


def test_one_stable_label_scores_two_runs_without_stale_invalidation():
    left_cv = _cv_block(cv_run_id="run-a")
    left_cv.update({"ground_truth_id": CANDIDATE, "execution_id": execution_id("run-a", CANDIDATE)})
    right_cv = _cv_block(cv_run_id="run-b")
    right_cv.update({"ground_truth_id": CANDIDATE, "execution_id": execution_id("run-b", CANDIDATE)})
    rows = [
        _cv_row(alarm_id=left_cv["execution_id"], outcome="CONFIRM_TP", cv=left_cv),
        _cv_row(alarm_id=right_cv["execution_id"], outcome="CONFIRM_FP", cv=right_cv),
    ]
    audits = {
        FRAME_KEY: {
            **_audit("agree", audited_verdict="violation"),
            "ground_truth_verdict": "violation",
            "revision": 2,
        }
    }

    left = cv_audit_items(rows, audits, "run-a")[0]
    right = cv_audit_items(rows, audits, "run-b")[0]
    assert left.audited and right.audited
    assert not left.stale_audit and not right.stale_audit
    assert cv_confusion(rows, audits, "run-a")["overall"]["n"] == 1
    assert cv_confusion(rows, audits, "run-b")["overall"]["n"] == 1


# --- MULTI-CROP FRAMES ----------------------------------------------------
#
# This file had ZERO tests with two people in one frame, and that is why the
# frame-level unit was applied in two places and skipped in seven. At one crop
# per frame, per-crop and per-frame arithmetic are IDENTICAL, so every per-crop
# site passed. `vlm_frame_verdicts` -- the mechanism the whole redesign exists
# for -- had no test that exercised the OR at all, because OR over a one-element
# list is the identity function.
#
# Every test below needs at least two crops on one frame to say anything.


def _crop(
    person_index: int, outcome: str = "CONFIRM_TP", frame: str = FRAME_KEY, **overrides
) -> dict:
    """One judged crop of `frame`. Several of these share one human decision."""
    candidate = f"{frame}__p{person_index:02d}"
    return _cv_row(
        alarm_id=f"exec__{candidate}",
        outcome=outcome,
        **overrides,
        cv=_cv_block(
            candidate_id=candidate,
            execution_id=f"exec__{candidate}",
            frame_name=f"{frame}.jpg",
            person_index=person_index,
            people_in_frame=3,
        ),
    )


def test_one_verdict_on_a_crowded_frame_counts_once():
    """The bug this file could not previously express.

    Measured on the shipping run: three of eighteen alarm frames carried 46% of
    the CV arm's headline precision, because each was scored once per person in
    it. A single reviewer's verdict on the nine-crop frame moved the number
    about twenty points.
    """
    rows = [_crop(i) for i in range(5)]
    audits = {FRAME_KEY: _audit("disagree", audited_verdict="violation",
                                corrected="no_violation")}

    counts = cv_confusion(rows, audits, "cv-1")["overall"]["counts"]

    assert counts["false_positive"] == 1
    assert arm_metrics(cv_confusion(rows, audits, "cv-1")["overall"])["n"] == 1


def test_any_person_in_violation_makes_the_frame_a_violation():
    """The production question is "should this frame have raised an alarm?",
    and a frame is not cleared because two of its three people wear helmets."""
    rows = [_crop(0, "CONFIRM_FP"), _crop(1, "CONFIRM_FP"), _crop(2, "CONFIRM_TP")]

    frames = vlm_frame_verdicts(cv_audit_items(rows, {}, "cv-1"), {})

    assert len(frames) == 1
    assert frames[0].verdict == "TP"
    assert frames[0].crops == 3
    assert frames[0].fired_candidates == [f"{FRAME_KEY}__p02"]


def test_a_confident_violation_beats_a_hedge_in_the_same_frame():
    rows = [_crop(0, "ESCALATE"), _crop(1, "CONFIRM_TP")]

    assert vlm_frame_verdicts(cv_audit_items(rows, {}, "cv-1"), {})[0].verdict == "TP"


def test_a_hedge_beats_a_clearance_in_the_same_frame():
    """TP > uncertain > FP. The judge declining to answer about somebody is
    not the judge clearing the frame, and this site has to say so in the same
    order as `case_outcome`, `_IMAGE_OUTCOMES` and `_CASE_OUTCOMES` -- four
    screens read the four, and one frame must not have two answers."""
    rows = [_crop(0, "ESCALATE"), _crop(1, "CONFIRM_FP")]

    assert vlm_frame_verdicts(cv_audit_items(rows, {}, "cv-1"), {})[0].verdict == "uncertain"


def test_a_frame_of_nothing_but_hedges_is_uncertain():
    rows = [_crop(0, "ESCALATE"), _crop(1, "ESCALATE")]

    assert vlm_frame_verdicts(cv_audit_items(rows, {}, "cv-1"), {})[0].verdict == "uncertain"


def test_a_frame_everyone_cleared_is_a_no_violation():
    rows = [_crop(0, "CONFIRM_FP"), _crop(1, "CONFIRM_FP")]

    assert vlm_frame_verdicts(cv_audit_items(rows, {}, "cv-1"), {})[0].verdict == "FP"


def test_the_vlm_arm_scores_a_crowded_frame_once():
    """`ground_truth_verdict` is set, as `review_frame` always sets it.

    Without it the frame's staleness is checked against ONE arbitrary crop's
    settled verdict rather than the frame's OR, so a frame whose first crop was
    cleared drops out of scoring entirely. Real reviews always carry the field
    so this cannot bite in production, but a LEGACY audit that predates it is
    checked the fragile way -- see the handoff.
    """
    rows = [_crop(0, "CONFIRM_FP"), _crop(1, "CONFIRM_TP"), _crop(2, "CONFIRM_TP")]
    audits = {FRAME_KEY: {**_audit("agree", audited_verdict="violation"),
                          "ground_truth_verdict": "violation"}}

    metrics = arm_metrics(vlm_confusion(rows, audits, "cv-1"))

    assert metrics["n"] == 1
    assert metrics["counts"]["true_positive"] == 1


def test_the_attribution_does_not_decide_the_verdict():
    """`fired` is which crops fired; it must never BE the answer. Deriving the
    verdict from it meant a violation crop with no `candidate_id` silently
    flipped the frame to no-violation."""
    rows = [_crop(0, "CONFIRM_TP"), _crop(1, "CONFIRM_FP")]
    for row in rows:
        if row["cv"]["person_index"] == 0:
            row["cv"]["candidate_id"] = ""

    frame = vlm_frame_verdicts(cv_audit_items(rows, {}, "cv-1"), {})[0]

    assert frame.verdict == "TP"
    assert frame.fired_candidates == []


def test_two_frames_of_one_alarm_are_two_units():
    """Grouping must key on the frame, not the alarm: an alarm holds many
    frames and each carries its own human decision."""
    rows = [_crop(0), _crop(1), _crop(0, frame=FRAME_KEY.replace("f000450", "f000451"))]

    assert len(group_by_frame(cv_audit_items(rows, {}, "cv-1"))) == 2


def test_a_frame_short_of_its_people_is_not_scored_as_cleared():
    """An OR over a HALF-judged frame reads as "nobody was in violation", and
    afterwards it is indistinguishable from a frame everyone was judged in.
    A stopped live feed and a failed judge call both produce this shape, so
    the frame is reported as incomplete and kept out of the scored population
    rather than averaged in as a suppression."""
    rows = [_crop(0, "CONFIRM_FP"), _crop(1, "CONFIRM_FP")]  # people_in_frame is 3

    frame = vlm_frame_verdicts(cv_audit_items(rows, {}, "cv-1"), {})[0]
    assert frame.people_in_frame == 3
    assert frame.complete is False

    comparison = cv_vs_judge(rows, {}, "cv-1")
    assert comparison["decided_total"] == 0
    assert comparison["suppressed"] == 0
    assert comparison["incomplete_frames"] == 1

    audits = {FRAME_KEY: {**_audit("agree", audited_verdict="violation"),
                          "ground_truth_verdict": "violation"}}
    assert arm_metrics(vlm_confusion(rows, audits, "cv-1"))["n"] == 0


def test_a_whole_frame_still_scores_normally():
    """The completeness guard must not quietly shrink a complete corpus."""
    rows = [_crop(0, "CONFIRM_FP"), _crop(1, "CONFIRM_FP"), _crop(2, "CONFIRM_FP")]

    comparison = cv_vs_judge(rows, {}, "cv-1")

    assert comparison["decided_total"] == 1
    assert comparison["suppressed"] == 1
    assert comparison["incomplete_frames"] == 0


def test_a_legacy_row_with_no_people_count_is_complete():
    """`people_in_frame` defaults to 0, which is "unrecorded", not "empty"."""
    rows = [_crop(0, "CONFIRM_TP")]
    rows[0]["cv"]["people_in_frame"] = 0

    frame = vlm_frame_verdicts(cv_audit_items(rows, {}, "cv-1"), {})[0]

    assert frame.complete is True
    assert cv_vs_judge(rows, {}, "cv-1")["decided_total"] == 1


# --- the review a crop is settled by --------------------------------------
#
# `set_verdict` files a human decision against the FRAME -- that is what the
# escalation queue lists and what the review screen posts -- while the audit
# queue looked one up by the CROP's own alarm id. Measured on the live corpus
# on 6 Sep: 0 of 23 review entries matched a crop id, 23 of 23 matched a frame
# key. Every answered frame still read "no verdict yet" on the audit screen,
# and the `human_settled` bucket had been empty since the collapse shipped.


def test_a_crop_is_settled_by_the_verdict_filed_against_its_frame():
    rows = [_crop(0, "CONFIRM_FP"), _crop(1, "ESCALATE")]
    reviews = {FRAME_KEY: {**_ANSWERED, "verdict": "no_violation", "reviewed_by": "Han"}}

    case = build_audit_cases(rows, reviews, {}, config=PipelineConfig())[0]

    assert case.settled_verdict == "no_violation"
    assert case.settled_by == "human"
    assert case.awaiting_verdict == 0
    assert [person.reviewed_by for person in case.people] == ["Han", "Han"]


def test_a_crop_with_no_review_anywhere_is_still_unsettled():
    """The lookup widened, and nothing else did."""
    rows = [_crop(0, "ESCALATE"), _crop(1, "ESCALATE")]

    case = build_audit_cases(rows, {}, {}, config=PipelineConfig())[0]

    assert case.settled_verdict is None
    assert case.settled_by == "pipeline"
    assert case.people_awaiting == 2


def test_a_frames_helmet_rectangle_is_not_handed_to_its_other_people():
    """`OPEN_TASKS.md` task 15: one worker's helmet box exported as every
    other worker's label. The box STATUS is a fact about the frame's box
    question and travels; the drawn rectangle is one person's head and does
    not."""
    frame_crops = [_crop(0, "ESCALATE"), _crop(1, "CONFIRM_FP")]
    review = {"helmet_box_status": "drawn", "helmet_box": [1.0, 2.0, 3.0, 4.0]}

    carried = _crop_review({FRAME_KEY: review}, frame_crops[0], frame_crops)

    assert carried["helmet_box_status"] == "drawn"
    assert "helmet_box" not in carried


def test_a_frame_holding_one_person_keeps_its_helmet_rectangle():
    """Where the frame and the person are the same thing, so are their boxes."""
    row = _crop(0, "ESCALATE")
    review = {"helmet_box_status": "drawn", "helmet_box": [1.0, 2.0, 3.0, 4.0]}

    carried = _crop_review({FRAME_KEY: review}, row, [row])

    assert carried["helmet_box"] == [1.0, 2.0, 3.0, 4.0]


def test_a_crops_own_review_beats_its_frames():
    """A live alarm's review is filed under its own id, and stays the answer."""
    frame_crops = [_crop(0, "ESCALATE"), _crop(1, "CONFIRM_FP")]
    reviews = {
        FRAME_KEY: {"verdict": "no_violation"},
        frame_crops[0]["alarm_id"]: {"verdict": "violation"},
    }

    assert _crop_review(reviews, frame_crops[0], frame_crops)["verdict"] == "violation"


# --- an unanswered hedge blocks the image from settling --------------------
#
# CEO, 5 Sep: a frame with any escalated crop cannot settle until that crop is
# reviewed, even when another crop already fired. The review answers the CROP,
# not only the frame. One `outcome` field was answering three questions -- what
# did the judge conclude, did this frame need a human, does it still need one
# -- and only the first of them is the OR.


#: A review the app itself would have written: `save_review` stamps every one
#: with `reviewed_at`, and the backfill needs it to place the answer against
#: the crop it is supposed to be about.
_ANSWERED = {"reviewed_at": "2026-09-02T00:00:00+00:00"}


def _hedged(*outcomes: str, reviews: dict | None = None):
    rows = [_crop(index, outcome) for index, outcome in enumerate(outcomes)]
    return build_audit_cases(rows, reviews or {}, {}, config=PipelineConfig())[0]


def test_an_unanswered_hedge_leaves_a_cleared_frame_unsettled():
    case = _hedged("CONFIRM_FP", "ESCALATE")

    assert case.settled_verdict is None
    assert case.awaiting_verdict == 1


def test_a_blocked_violation_frame_keeps_its_verdict():
    """Nulling it would cost the judge a confirmed true positive it got right
    and retire the audits recorded against it for a finding that cannot exist:
    no answer to the hedge can lower an OR that a violation has already won."""
    case = _hedged("CONFIRM_TP", "CONFIRM_FP", "ESCALATE")

    assert case.settled_verdict == "violation"
    assert case.judge_outcome == "CONFIRM_TP"
    # And it still owes a human an answer -- that is the routing signal, not
    # the verdict being absent.
    assert case.awaiting_verdict == 1


def test_the_judges_own_answer_puts_a_hedge_above_a_clearance():
    assert _hedged("CONFIRM_FP", "ESCALATE").judge_outcome == "ESCALATE"
    assert _hedged("CONFIRM_TP", "ESCALATE").judge_outcome == "CONFIRM_TP"
    assert _hedged("CONFIRM_FP", "CONFIRM_FP").judge_outcome == "CONFIRM_FP"


def test_a_frame_level_verdict_answers_a_single_hedged_crop():
    """The backfill rule, and why no migration was needed: the three verdicts
    already in the live sidecar were all given on frames with one hedge in
    them, and a frame-level answer is an answer about that crop."""
    case = _hedged(
        "CONFIRM_FP", "ESCALATE", reviews={FRAME_KEY: {**_ANSWERED, "verdict": "no_violation"}}
    )

    assert case.awaiting_verdict == 0
    assert case.settled_verdict == "no_violation"


# --- what a frame-level answer can and cannot have been about --------------
#
# Found by an adversarial pass over the change above. One answer was closing
# every hedge in its frame, an answer given days earlier was closing a hedge
# created by a later re-run, and a `violation` about a picture was being
# stamped onto every person in it -- including seven the judge had cleared, on
# a real nine-crop frame in the live corpus.


def test_one_answer_closes_every_hedge_the_reviewer_was_SHOWN():
    """A FRAME WITH TWO HEDGES HAS TO BE CLEARABLE. The reviewer is shown the
    whole picture and every crop of it, so the answer they give is about the
    people in front of them -- all of them.

    This was briefly the other way, on the truer reading that a frame-level
    answer cannot say WHICH of two hedged people was the violator. It made the
    screen unusable: 19 of the live corpus's 34 blocked frames hold two hedges
    or more, nothing else could answer them, and the reviewer's save left the
    case exactly where it was. Something else can now -- `crop_verdicts`, below
    -- and this stays as what covers the crops a reviewer did not single
    out."""
    case = _hedged(
        "ESCALATE", "ESCALATE", reviews={FRAME_KEY: {**_ANSWERED, "verdict": "no_violation"}}
    )

    assert case.awaiting_verdict == 0
    assert case.settled_verdict == "no_violation"


def test_an_answer_cannot_reach_a_crop_that_did_not_exist_yet():
    """A re-run appends a row for a person the earlier review never saw. The
    original bug, arriving by a different door: a crop the judge hedged on
    settling with nobody having looked at it. The crop the reviewer DID see
    stays answered -- only the new one is asked."""
    rows = [
        _crop(0, "ESCALATE"),
        _crop(1, "ESCALATE", timestamp="2026-09-05T00:00:00+00:00"),
    ]
    reviews = {FRAME_KEY: {"verdict": "no_violation", "reviewed_at": "2026-09-01T00:00:00+00:00"}}

    case = build_audit_cases(rows, reviews, {}, config=PipelineConfig())[0]

    assert case.awaiting_verdict == 1
    # The crop they DID answer is cleared and has moved on to owing a helmet
    # box; only the new one is still owed a verdict.
    assert [person.awaiting for person in case.people] == ["box", "verdict"]
    assert case.settled_verdict is None


def test_an_answer_with_no_timestamp_cannot_be_placed_and_does_not_backfill():
    """Cannot tell, so the crop is asked again -- and that is recoverable
    rather than a dead end: the frame is still in the verdict queue, and every
    review the app writes carries `reviewed_at` (`save_review`), so the next
    save resolves it. Only entries written before the field existed land
    here."""
    case = _hedged("CONFIRM_FP", "ESCALATE", reviews={FRAME_KEY: {"verdict": "no_violation"}})

    assert case.awaiting_verdict == 1


def test_a_violation_about_a_picture_is_not_a_violation_about_everyone_in_it():
    """Live corpus, 6 Sep: one nine-crop frame stamped `violation` onto seven
    people the judge had correctly cleared. A worker standing beside a
    violator is not thereby a violator. `training_export.audited_crop_label`
    has drawn this line since task 15."""
    rows = [_crop(0, "CONFIRM_FP"), _crop(1, "CONFIRM_TP"), _crop(2, "CONFIRM_FP")]
    reviews = {FRAME_KEY: {**_ANSWERED, "verdict": "violation"}}

    case = build_audit_cases(rows, reviews, {}, config=PipelineConfig())[0]

    assert [person.settled_verdict for person in case.people] == [
        "no_violation",
        "violation",
        "no_violation",
    ]
    assert case.settled_verdict == "violation"


def test_a_clearance_about_a_picture_is_about_everyone_in_it():
    """The other polarity, and the reason the rule is not "never propagate":
    "nobody here was in violation" overrules even the crop the judge fired
    on."""
    rows = [_crop(0, "CONFIRM_TP"), _crop(1, "CONFIRM_FP")]
    reviews = {FRAME_KEY: {**_ANSWERED, "verdict": "no_violation"}}

    case = build_audit_cases(rows, reviews, {}, config=PipelineConfig())[0]

    assert [person.settled_verdict for person in case.people] == ["no_violation"] * 2
    assert case.settled_verdict == "no_violation"


def test_a_frame_answered_cannot_determine_is_still_a_case():
    """It used to drop every crop of the picture, and a case with no crops is
    no case at all: the frame left the audit queue, the case endpoint 404'd,
    and the audit already recorded against it was in no bucket and no count.
    Gone, rather than decided."""
    rows = [_crop(0, "CONFIRM_FP"), _crop(1, "ESCALATE")]
    reviews = {FRAME_KEY: {**_ANSWERED, "verdict": "cannot_determine"}}

    cases = build_audit_cases(rows, reviews, {}, config=PipelineConfig())

    assert [case.case_id for case in cases] == [FRAME_KEY]
    assert cases[0].settled_verdict == "cannot_determine"


def test_a_live_alarms_own_unlabelled_answer_still_drops_it():
    """The guard that dropped it is not gone, it is narrower: a decision filed
    against the ROW itself has nothing to check against the dataset, which is
    the case it was written for."""
    row = _cv_row(alarm_id="live-1", outcome="ESCALATE")
    row.pop("row_kind")
    row["cv"] = {}

    assert build_audit_queue(
        [row], {"live-1": {"verdict": "cannot_determine"}}, {}, config=PipelineConfig()
    ) == []


# --- an answer about ONE crop ---------------------------------------------
#
# The frame answer above stays, and covers the crops nobody singled out. What
# a reviewer says about a NAMED person beats it: 19 of the 34 blocked frames
# on 6 Sep held two hedges or more, and one answer for two people the judge
# could not read says what the frame is, not which person was which.


def _per_crop(*verdicts: str, frame: str | None = None) -> dict:
    entry = {**_ANSWERED, "crop_verdicts": {
        f"{FRAME_KEY}__p{index:02d}": verdict
        for index, verdict in enumerate(verdicts)
        if verdict
    }}
    if frame:
        entry["verdict"] = frame
    return {FRAME_KEY: entry}


def test_two_hedges_can_be_answered_two_different_ways():
    case = _hedged("ESCALATE", "ESCALATE", reviews=_per_crop("violation", "no_violation"))

    assert case.awaiting_verdict == 0
    assert [person.settled_verdict for person in case.people] == [
        "violation",
        "no_violation",
    ]
    assert case.settled_verdict == "violation"


def test_the_crop_answer_wins_over_the_frame_answer():
    """The specific beats the general. The crop nobody named still takes the
    frame's answer, which is what keeps the frame-level fallback working."""
    case = _hedged(
        "ESCALATE", "ESCALATE", reviews=_per_crop("violation", None, frame="no_violation")
    )

    assert [person.settled_verdict for person in case.people] == [
        "violation",
        "no_violation",
    ]


def test_a_crop_answer_alone_settles_its_person():
    """No frame verdict at all: one hedge answered, the other still owed."""
    case = _hedged("ESCALATE", "ESCALATE", reviews=_per_crop("no_violation"))

    assert case.awaiting_verdict == 1
    assert [person.awaiting for person in case.people] == ["box", "verdict"]
    assert [person.settled_by for person in case.people] == ["human", "pipeline"]


def test_a_crop_answer_cannot_reach_a_person_it_does_not_name():
    """Filed against a candidate id, so unlike a frame answer it cannot spread
    onto a person a later re-run added -- and needs no timestamp guard."""
    rows = [
        _crop(0, "ESCALATE"),
        _crop(1, "ESCALATE", timestamp="2026-09-05T00:00:00+00:00"),
    ]

    case = build_audit_cases(rows, _per_crop("no_violation"), {}, config=PipelineConfig())[0]

    assert case.awaiting_verdict == 1
    assert [person.settled_verdict for person in case.people] == ["no_violation", None]


def test_a_struck_out_hedge_does_not_block_the_frame():
    """The crop has left the population, so there is nobody left to ask about
    -- the same rule `included_people` applies to the OR itself."""
    rows = [_crop(0, "CONFIRM_FP"), _crop(1, "ESCALATE")]
    audits = {FRAME_KEY: {"excluded_people": [f"{FRAME_KEY}__p01"]}}

    case = build_audit_cases(rows, {}, audits, config=PipelineConfig())[0]

    assert case.awaiting_verdict == 0
    assert case.settled_verdict == "no_violation"
