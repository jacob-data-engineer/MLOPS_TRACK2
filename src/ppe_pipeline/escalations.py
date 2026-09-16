"""Human follow-up on what the automated pipeline could not settle.

Two queues, both derived from the Track1 audit log rather than kept as their
own list of work:

* **verdict** -- alarms the judge escalated (`outcome == "ESCALATE"`). A
  human says whether the violation is real. Nothing else is asked: the
  point of this review is to produce a correct label for CV retraining, so
  a rationale, reason code, evidence rating and difficulty (all of which
  the *gold set* collects, because it grades the VLM's reasoning) would be
  work nobody downstream reads.

* **box** -- alarms where the helmet turned out to BE worn, whose helmet
  box is not good enough to train on: the stage-2 detector found nothing
  (so the looser VLM inline box was used), or it found something below
  `helmet_box.review_confidence`. A human draws the box. No verdict is
  asked here -- it is already settled, and a bad box is not evidence of a
  bad verdict.

  "Worn" covers both a judge clearance (`CONFIRM_FP`) and an escalation a
  human resolved to `no_violation`. The second matters because
  `_resolve_outcome` downgrades a low-confidence FP to ESCALATE: gating on
  the judge's outcome alone let those cases fall out of both queues.

  A case therefore moves verdict -> box as it is worked, so the two are
  disjoint among *pending* cases but a single alarm can legitimately be
  reviewed in both.

Deriving the queues from the audit log, instead of appending to a separate
work list when a run happens, means a case cannot go missing because a
session ended or a write raced: the audit log is the record of what the
pipeline did, and these queues are a view over it. It also means a case
that has already been reviewed simply stops matching.

Human decisions live in their own sidecar (`escalation_reviews.json`),
keyed by `alarm_id`. The audit log itself is append-only history of what
the *models* decided and is never rewritten -- a human correction is a
separate fact, not an edit to the record of the original verdict.

Everything here stores boxes in ORIGINAL frame pixels, deliberately. The
training export emits person-*crops* rather than full frames, because a
full frame cannot be exported safely: this repo kept only the largest
`person` box per case and discarded the rest of each source label file, so
a second worker in frame would go unlabelled and train as background. That
crop is a rendering decision made at export time, not a storage decision
-- so when the export later moves to full frames, or to a different crop
rule, nothing stored needs migrating and no review needs redoing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from . import storage
from .config import PipelineConfig, get_config
from .models import ViolationType
from .track1_adapter import VIOLATION_TYPE_TO_HAZARD_LABEL

DEFAULT_REVIEWS_PATH = "eval/escalation_reviews.json"

QueueName = Literal["verdict", "box"]

#: Human verdicts. The same words the gold set uses for a settled label,
#: so a downstream consumer never has to learn a second vocabulary.
#:
#: "cannot_determine" exists because a case is escalated precisely WHEN the
#: evidence is insufficient -- `head_occluded` is the commonest reason. With
#: only two options a reviewer who genuinely cannot tell is forced to guess,
#: and that guess becomes a CV training label. It clears the queue and is
#: excluded from the training export, mirroring what `no_helmet_visible`
#: does on the box side.
VERDICT_OPTIONS = ["violation", "no_violation", "cannot_determine"]

#: Verdicts that produce a usable training label. `cannot_determine` is a
#: settled review but not a usable label.
EXPORTABLE_VERDICTS = ["violation", "no_violation"]

#: Outcomes of a box review. "no_helmet_visible" is not a verdict change --
#: it records that a human could not find a helmet to draw, which makes the
#: case unusable as a helmet training example and is worth knowing.
#: What a reviewer can conclude about a helmet box.
#:
#: `accepted_cv` exists because redrawing a box that is already right is
#: wasted work AND worse data: a hand-drawn copy of a good detector box is
#: less accurate than the box it replaces, not more. The queue asks for a
#: human because the detector was unsure, not because the detector was
#: wrong, and it is often not wrong. Accepting records the SAME coordinates
#: with a human's endorsement behind them.
BOX_STATUS_OPTIONS = ["drawn", "accepted_cv", "no_helmet_present", "no_helmet_visible"]

#: "There is no helmet in this image."
#:
#: A case only reaches the box queue because something concluded a helmet IS
#: worn. A reviewer finding none is not reporting a box problem, they are
#: overturning the verdict: this is a violation the judge cleared, which is a
#: FALSE NEGATIVE and the worst error this pipeline makes.
#:
#: It is separate from `no_helmet_visible` because that means the opposite
#: thing about the verdict: a helmet is worn but cannot be boxed (occluded,
#: cut off by the crop). One says the judge was right and the label is
#: impossible; the other says the judge was wrong. Conflating them threw the
#: false negative away silently, and left the case counted as a confident
#: correct clearance in the auto-annotation rate.
NO_HELMET_PRESENT = "no_helmet_present"

#: The statuses that yield a helmet label. `no_helmet_visible` is finished
#: work but not a label.
LABELLING_BOX_STATUSES = ("drawn", "accepted_cv")

HAZARD_LABEL_TO_VIOLATION_TYPE = {
    label: violation_type for violation_type, label in VIOLATION_TYPE_TO_HAZARD_LABEL.items()
}


@dataclass(frozen=True)
class QueueItem:
    """One case awaiting a human, flattened for the API/UI.

    Boxes are in ORIGINAL frame pixels (`[x1, y1, x2, y2]`), the same space
    the audit log records and the same space the drawing tool works in, so
    nothing has to convert between coordinate systems mid-review.
    """

    alarm_id: str
    queue: QueueName
    hazard_type: str
    violation_type: str | None
    timestamp: str | None
    image_key: str | None
    image_width: int | None
    image_height: int | None
    person_box: list[float] | None
    helmet_box: list[float] | None
    # The region the judge was shown, in original pixels, and the same
    # boxes expressed relative to that crop. The review UI shows the CROP,
    # not the frame -- a 70px-tall worker in a 1920x1080 frame cannot be
    # boxed accurately -- so it needs the overlays in crop pixels, which
    # are the natural pixels of the served image (crop_bytes does not
    # resize). Null when the run predates crop_box being recorded.
    crop_box: list[float] | None
    person_box_in_crop: list[float] | None
    helmet_box_in_crop: list[float] | None
    helmet_box_source: str | None
    helmet_box_confidence: float | None
    # What the models said, shown to the reviewer as context. Never editable:
    # a human correction is recorded separately, in the review sidecar.
    judge_outcome: str | None
    judge_reason: str | None
    judge_reasoning: str | None
    judge_confidence: float | None
    judge_model: str | None
    # Whether this case's frame can actually be fetched -- i.e. whether it
    # is reviewable at all. See `frame_available`.
    frame_available: bool = False
    reviewed: bool = False
    review: dict = field(default_factory=dict)
    #: This case is in the BOX queue because an audit corrected it to
    #: `no_violation` -- a human saying the helmet IS worn on a case the judge
    #: called a violation. Without it the sheet shows a VIOLATION stamp beside
    #: a request to draw the helmet and reads as a contradiction; the reviewer
    #: cannot see that the verdict has since been overruled. See
    #: `audit_reopened_box`.
    requeued_by_audit: bool = False
    #: The corrected verdict that reopened it, so the sheet can quote the
    #: finding rather than assert it.
    audit_corrected_verdict: str | None = None
    #: Every crop of this image, when the row stands for a whole frame -- the
    #: escalation queues collapse a frame's crops to one item, and a reviewer
    #: cannot strike one out without being shown it. Empty for a live alarm,
    #: which is one crop and has no strip. Written by
    #: `routers/escalations._one_row_per_image`.
    candidates: list[dict] = field(default_factory=list)
    #: Which of those crops still owe a human a verdict, by candidate id. The
    #: collapsed row and the audit case have carried this since escalated
    #: crops began blocking their frame; the queue item did not, so the screen
    #: that ASKS the question could not say which person it was about. Empty
    #: for a live alarm, and empty for a frame whose hedges are all answered.
    awaiting_verdict: list[str] = field(default_factory=list)
    #: WHICH PERSON the box question is about, by candidate id -- the crop
    #: `box_row` re-anchored this item on. Carried because a box decision can
    #: be an answer about that person and not only about the picture:
    #: `no_helmet_present` overturns their clearance, and a crop answer is how
    #: that reaches them. Null for a live alarm, which is one crop.
    box_crop_id: str | None = None


def _finite_float(value: object) -> float | None:
    """A usable float, or None for anything that is not one.

    Audit rows are JSON written by past versions of the pipeline and by
    hand; a confidence has turned up as "", "n/a", a list, and NaN. Any of
    those reaching `float()` unguarded took the whole queue endpoint down
    with a 500 -- one bad row hid every good one. NaN is rejected too:
    every comparison against it is False, which would quietly mark an
    untrustworthy box as trustworthy.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _entry_for(reviews: dict, alarm_id: str) -> dict:
    """One case's review entry, tolerating a hand-edited sidecar."""
    entry = reviews.get(alarm_id)
    return entry if isinstance(entry, dict) else {}


def frame_available(image_key: object) -> bool:
    """Whether a row's frame can be fetched, judged from the key alone.

    Rows written before frame persistence existed carry a BARE filename
    (`streamlit-5458985889.jpg`) because the old writer recorded a key it
    never stored bytes at. The current writer always produces a key under
    the frame directory, and sets it to None if the write failed -- so a key
    containing a separator means the bytes really were written.

    A key alone rather than a storage HEAD per row on purpose: a queue can
    be long, and one round trip per case to draw a list is not worth it.

    An ABSOLUTE key is legitimate here and stays legitimate: the alarm image
    directory may be configured as an absolute path (`PPE_EVAL_DATA_DIR`, and
    the test suite does exactly that). Whether a given absolute key belongs to
    THIS deployment is a question about configuration, so it is answered where
    the configuration lives -- `routers/escalations._frame_is_fetchable_here`
    -- not here, where all this function has is the key.
    """
    if not image_key or not isinstance(image_key, str):
        return False
    return "/" in image_key.replace("\\", "/")


def _shift_into_crop(box: list[float] | None, crop: list[float] | None) -> list[float] | None:
    """A box in original pixels, expressed relative to `crop`.

    `crop_bytes` cuts without resizing, so a crop pixel IS an original pixel
    offset by the crop origin -- the conversion is a translation, not a
    scale. Clamped to the crop, since a person box is padded outward and can
    extend past the crop it produced.
    """
    if box is None or crop is None:
        return None
    # Rounded, matching `BoundingBox.as_tuple()` -- the served crop is cut
    # on the rounded tuple (see `crop_bytes`) and the box POST validates
    # against it, so shifting on raw floats would leave display and
    # validation on subtly different conventions.
    cx1, cy1, cx2, cy2 = (round(v) for v in crop)
    width, height = cx2 - cx1, cy2 - cy1
    if width <= 0 or height <= 0:
        return None
    shifted = [box[0] - cx1, box[1] - cy1, box[2] - cx1, box[3] - cy1]
    return [
        max(0.0, min(width, shifted[0])),
        max(0.0, min(height, shifted[1])),
        max(0.0, min(width, shifted[2])),
        max(0.0, min(height, shifted[3])),
    ]


def _box_to_list(box: dict | None) -> list[float] | None:
    if not isinstance(box, dict):
        return None
    try:
        return [float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])]
    except (KeyError, TypeError, ValueError):
        return None


def load_reviews(path: str | Path = DEFAULT_REVIEWS_PATH) -> dict[str, dict]:
    value, _ = storage.get_storage().read_json(
        Path(path).as_posix(), default={}, revalidate=True
    )
    return value if isinstance(value, dict) else {}


def save_review(
    alarm_id: str,
    updates: dict,
    *,
    reviewer: str = "",
    path: str | Path = DEFAULT_REVIEWS_PATH,
    now: datetime | None = None,
) -> dict:
    """Merge `updates` into one case's review entry, concurrency-safely.

    Goes through `storage.update_json`, which re-reads and retries on a
    losing compare-and-swap, so two reviewers clearing the same queue at
    once cannot silently drop one another's work (the same guarantee
    `gold_set.update_case_annotation` relies on).

    EVERY SAVE MERGES OVER THE PREVIOUS ENTRY, so a key `updates` carries is a
    key it OVERWRITES -- and `CROP_VERDICTS` is a whole map, so writing it
    replaces every crop's answer at once. A caller with no opinion about them
    must therefore leave the key OUT (`None`), which is not the same as
    sending `{}` to clear them. `save_audit` learned that distinction the hard
    way with `excluded_people`: for one release a box reviewer erased an
    auditor's strike-outs by pressing their own button, which flipped a
    verdict back and re-published a crop a human had pulled. Two callers here
    -- `routers/escalations.set_box` and the `no_helmet_present` verdict it
    records -- write this sidecar knowing nothing about crop verdicts, and
    stay correct only by not naming the key.
    """
    stamped = dict(updates)
    stamped["reviewed_by"] = reviewer
    stamped["reviewed_at"] = (now or datetime.now(timezone.utc)).isoformat()

    def mutate(current: dict) -> dict:
        entry = dict(current.get(alarm_id, {}))
        entry.update(stamped)
        return {**current, alarm_id: entry}

    updated = storage.get_storage().update_json(Path(path).as_posix(), mutate, default={})
    return updated[alarm_id]


#: On a review entry: what a human answered about EACH crop of the frame, by
#: candidate id -- `{candidate_id: verdict}`, beside the frame's own `verdict`.
#:
#: A frame answer says what the PICTURE is, which is the right answer to the
#: wrong question when the judge could not read two people in it: 19 of the 34
#: frames blocked on 6 Sep held two hedges or more, and one save covering both
#: never says which person was which. It also left the export nothing to work
#: with -- `training_export.audited_crop_label` gives a non-firing crop of a
#: violation frame no label at all, so resolving a hedge on a crowded violation
#: frame produced no training sample.
#:
#: In THIS sidecar and not the audit one, which is where a strike-out lives.
#: `training_export._settled_cases` gates a crop's exportability on a verdict
#: read from here, so a crop verdict stored anywhere else would be a human
#: answer the export never sees, and the crop would stay unexported forever.
CROP_VERDICTS = "crop_verdicts"

#: On a row standing for a whole image: the crops of it that still owe a human
#: a verdict, by candidate id. Written by `awaiting_verdict_crops` below,
#: beside `audit.NO_VALID_CROPS`.
AWAITING_VERDICT_CROPS = "awaiting_verdict"


#: On a row standing for a whole image: the crop its BOX question is about.
#: Written by `routers/escalations._one_row_per_image`, read by `box_row`.
BOX_CROP = "box_crop"

#: What the box question takes from that crop rather than from the image's
#: own row. The geometry is obvious; the outcome and the words come with it
#: because they are the judge's answer ABOUT THAT PERSON, and a sheet showing
#: one person's crop under another's reasoning is the contradiction
#: `audit._deciding_person` exists to prevent.
_BOX_ANCHOR_FIELDS = (
    "outcome",
    "reason_code",
    "reasoning",
    "confidence",
    "crop_box",
    "person_box",
    "helmet_box",
    "helmet_box_source",
    "helmet_box_confidence",
)


def box_crop_id(row: dict) -> str | None:
    """The candidate id of the crop this image's box question is about."""
    crop = row.get(BOX_CROP)
    if not isinstance(crop, dict):
        return None
    return _crop_id(crop) or None


def box_row(row: dict) -> dict:
    """This image's row, re-anchored on the crop that owes the helmet box.

    ONE IMAGE, TWO QUESTIONS, TWO ANCHORS. The verdict queue asks "is this
    frame a violation", and its row is the crop that DECIDED that. The box
    queue asks "where is this person's helmet", which is a different crop --
    and since the outcome precedence put ESCALATE above CONFIRM_FP, routinely
    a different one. Anchoring both on the verdict's crop meant the box sheet
    offered the hedged person: `accepted_cv` 400'd because that crop has no
    detector box to accept, and a drawn rectangle was measured against a
    person the judge never asked about.

    The image's identity stays -- `alarm_id` is still the frame, because that
    is what one box decision is filed under. Only what the question is ABOUT
    moves.
    """
    crop = row.get(BOX_CROP)
    if not isinstance(crop, dict):
        return row
    return {**row, **{field: crop.get(field) for field in _BOX_ANCHOR_FIELDS}}


def _reviewed_before(row: dict, entry: dict) -> bool:
    """Whether this crop already existed when that review was written.

    A human can only have been answering about a crop that was in front of
    them. A re-run appends a row for a person the earlier review never saw, so
    without this an old frame verdict silently answers a NEW hedge -- which is
    the bug this whole change exists to remove, arriving by a different door.

    Both timestamps must parse. An entry with no `reviewed_at` predates the
    field, and a row with no timestamp cannot be placed against it; in either
    case the honest answer is "cannot tell", and the crop goes to the queue.
    Under-answering costs a reviewer one question; over-answering settles a
    person nobody looked at.
    """
    judged, reviewed = row.get("timestamp"), entry.get("reviewed_at")
    if not judged or not reviewed:
        return False
    try:
        return datetime.fromisoformat(str(judged)) <= datetime.fromisoformat(str(reviewed))
    except (TypeError, ValueError):
        return False


def frame_verdict_answers(row: dict, entry: dict, frame_crops: list[dict]) -> bool:
    """Whether the FRAME's verdict is an answer about THIS crop.

    Three different questions wear one word here, and the polarity decides
    which:

    * **A crop the judge HEDGED on** is answered, unless it is NEWER than the
      review. The reviewer is shown the whole picture and every crop of it --
      the strip carries them all -- so the answer they give is an answer about
      the people in front of them, including the ones the judge could not
      read. The single exception is a crop a later re-run added, which they
      never saw.

      This was briefly narrower: a frame answer resolved a hedge only when it
      was the frame's ONLY one. That is a truer reading of what a frame-level
      answer knows -- it says what the FRAME is, not which of two hedged
      people was the violator -- and it made the screen unusable, because 19
      of the 34 blocked frames hold two hedges or more and NOTHING else could
      answer them: the reviewer saves, the verdict is stored, no crop is
      resolved, and the case never leaves the queue.

      Something else CAN answer them now -- `CROP_VERDICTS`, the per-crop
      answer, which is asked first everywhere both exist. The frame answer
      stays as what covers the crops a reviewer did not single out, because
      it is what clears a frame today and per-crop is an addition rather than
      a replacement.
    * **A `violation`** is about SOMEONE in the picture, never everyone. Only
      the crops the judge FIRED on inherit it; a worker the judge cleared,
      standing beside a violator, is not thereby a violator. Live corpus, 6
      Sep: one nine-crop frame stamped `violation` onto seven people the judge
      had correctly cleared. `training_export.audited_crop_label` has drawn
      this line since task 15 and is the precedent here.
    * **Anything else** -- `no_violation`, `cannot_determine` -- IS about
      everyone in the picture. "Nobody here was in violation" and "this cannot
      be read" are both statements about the whole frame, so every crop
      inherits them.
    """
    if row.get("outcome") == "ESCALATE":
        return _reviewed_before(row, entry)
    return frame_answer_reaches(row.get("outcome"), entry.get("verdict"))


def frame_answer_reaches(outcome: str | None, verdict: str | None) -> bool:
    """The polarity above, without the question of whether the crop existed.

    Named on its own because the SAVE has to apply it before there is a stored
    review to date a crop against: `routers/escalations` refuses a frame
    answer its crops contradict, and to know what a frame answer settles each
    crop to it needs this rule and not the staleness guard. Every crop it
    weighs is one the reviewer was just shown, so "did this crop exist when
    the answer was given" is answered by construction there.

    Two spellings of the polarity is exactly what let the refusal disagree
    with what `audit.case_outcome` settled a moment later: the check credited
    every unnamed crop with the frame's verdict, so a cleared crop propped up
    a `violation` in the check and contributed `no_violation` in reality.
    """
    if outcome == "ESCALATE":
        return True
    if verdict == "violation":
        return outcome == "CONFIRM_TP"
    return True


def crop_verdicts(entry: dict) -> dict[str, str]:
    """The per-crop answers on a review entry, hand-edited junk dropped.

    Same tolerance the rest of this module extends to the sidecar: it is JSON
    a human can edit, and a string where a map belongs, or a verdict spelled
    some other way, must not reach a screen as an answer.
    """
    value = entry.get(CROP_VERDICTS) if isinstance(entry, dict) else None
    if not isinstance(value, dict):
        return {}
    return {
        str(crop): verdict
        for crop, verdict in value.items()
        if verdict in VERDICT_OPTIONS
    }


def crop_verdict(entry: dict, row: dict) -> str | None:
    """The human's answer about THIS crop, if they gave one about it alone.

    THE SPECIFIC ANSWER, which beats the general one wherever both exist: a
    frame verdict is a statement about the picture, and this is a statement
    about the person. No `_reviewed_before` guard is needed or wanted -- a
    crop verdict is filed against a candidate id, so unlike a frame answer it
    cannot spread onto a person a later re-run added.
    """
    return crop_verdicts(entry).get(_crop_id(row))


def awaiting_verdict_crops(
    kept: list[dict],
    review: dict | None = None,
    frame_crops: list[dict] | None = None,
) -> list[str]:
    """Which of an image's KEPT crops still owe a human a verdict, by id.

    Two sites collapse an image's crops into one row, and they ask this for
    two different reasons -- which is why `review` is a parameter and not a
    lookup:

    * the escalation queues (`routers/escalations._one_row_per_image`) ask
      "does anyone still have to answer for this picture", and PASS the
      frame's review, so a hedge the reviewer already answered stops being
      work;
    * the metrics page (`rolling_metrics._images`) asks "did the pipeline
      need a human for this frame at all", and passes NOTHING, because a
      frame a human has since answered still needed one. That is the
      `auto_annotation_rate` question and the answer must not improve when
      somebody works the queue.

    Over the KEPT crops only: a struck-out crop has left the population, so
    there is nobody left to ask about it. `frame_crops` is the whole picture,
    struck crops included: it is what the reviewer was looking at when they
    answered.

    A crop is owed until EITHER answer covers it: its own, or the frame's.
    Its own first, because that is the one asked about it -- and because a
    reviewer may answer a crop on a frame they gave no overall verdict for.
    """
    population = kept if frame_crops is None else frame_crops
    answered = review if review and review.get("verdict") else None
    return [
        _crop_id(crop)
        for crop in kept
        if crop.get("outcome") == "ESCALATE"
        and not (review and crop_verdict(review, crop))
        and not (answered and frame_verdict_answers(crop, answered, population))
    ]


def newest_per_crop(rows: list[dict]) -> list[dict]:
    """One row per crop of one image, newest wins, in the order first seen.

    THE RULE EVERY OTHER READER OF THIS LOG ALREADY APPLIES, and the one place
    it was missing. The log is append-only, so re-judging a frame appends a
    second row for each of its people rather than replacing them: `build_queue`,
    `audit.build_audit_queue` and `training_export._settled_cases` all keep the
    newest per alarm id for exactly that reason, and the crop collapse did not
    -- so the review strip showed one worker twice, on 7 of the live corpus's
    186 frames, one of them with 13 cards for 6 people.

    Keyed by `_crop_id`, which is the candidate id where there is one: that is
    what a strike-out is filed against and what survives a re-run, while the
    alarm id is free to change with the execution that produced it.
    """
    newest: dict[str, dict] = {}
    for row in rows:
        key = _crop_id(row)
        previous = newest.get(key)
        if previous is None or _sort_key(row) >= _sort_key(previous):
            newest[key] = row
    return list(newest.values())


def _crop_id(crop: dict) -> str:
    """A crop's candidate id, or the alarm id of a row that has no `cv` block.

    `cv` is whatever a past writer put there: it has arrived as a string on a
    hand-edited row, and reaching into it unguarded took both escalation
    queues down with a 500 -- one bad row hiding every good one.
    """
    cv = crop.get("cv")
    cv = cv if isinstance(cv, dict) else {}
    return str(cv.get("candidate_id") or crop.get("alarm_id") or "")


def needs_verdict(row: dict) -> bool:
    """Whether a human still owes this row a verdict.

    OFF THE CROP LIST WHERE THERE IS ONE, not off the image's outcome string.
    An image's outcome is an OR, so a frame holding one crop the judge fired
    on and one it hedged about reads CONFIRM_TP -- and it still owes an answer
    about the second crop. The review answers the CROP (CEO, 5 Sep), so a
    violation elsewhere in the picture does not short-circuit it, and routing
    on the collapsed outcome hid every such frame from the verdict queue.

    A row with no crop list is one crop -- a live alarm, which is its own
    image -- and its own outcome is the whole answer.
    """
    crops = row.get(AWAITING_VERDICT_CROPS)
    if crops is not None:
        return bool(crops)
    return row.get("outcome") == "ESCALATE"


def _box_answered_after(audit: dict, review: dict | None) -> bool:
    """Whether a box decision was saved AFTER the audit that asked for one.

    The reopen has to be answerable, and nothing else answers it: the audit
    entry keeps saying "this case needs a box" for as long as it exists, so
    without this the case is pulled back into the queue every time it is
    built, however many boxes are saved.

    MEASURED on the live corpus, 2 Sep: two cases audited at 03:04:45 and
    given a box at 03:34:55, still sitting in the queue afterwards -- which is
    exactly how it was reported ("two cases i already settled why they havent
    left the queue").

    A box decision, not merely a later save. Answering the VERDICT question
    on the same case touches `reviewed_at` too, and that says nothing about
    the box.
    """
    if not isinstance(review, dict) or not review.get("helmet_box_status"):
        return False
    drawn_at, audited_at = review.get("reviewed_at"), audit.get("audited_at")
    if not drawn_at or not audited_at:
        # One of them predates the field. Treat the audit as unanswered: a
        # case queued once too often is recoverable, one that silently ships
        # the box an auditor rejected is not.
        return False
    try:
        return datetime.fromisoformat(str(drawn_at)) > datetime.fromisoformat(str(audited_at))
    except ValueError:
        return False


def audit_reopened_box(audit: dict | None, review: dict | None = None) -> bool:
    """Whether an audit has invalidated this case's helmet box.

    Two audits do it. An auditor correcting a case to `no_violation` is
    saying the helmet IS worn, so the case needs a helmet box, and whatever
    box decision came before (including a `no_helmet_visible`, or the
    absence of any box on a case exported as a violation) has just been
    overruled. And a `box_wrong` audit says so directly: the verdict stands,
    the box does not.

    Without this the finding would be recorded, the case pulled from the
    dataset, and then nothing would ever ask anyone to draw the box that
    puts it back. `build_queue` reads this to treat an already-reviewed case
    as unreviewed again, which is what lets a rejected box be redrawn while
    the rejected one stays on the record.
    """
    if not isinstance(audit, dict):
        return False
    reopened = audit.get("decision") == "box_wrong" or (
        audit.get("decision") == "disagree"
        and audit.get("corrected_verdict") == "no_violation"
    )
    return reopened and not _box_answered_after(audit, review)


def _human_verdict_for(row: dict, entry: dict) -> str | None:
    """The human answer `needs_box` should read off this row.

    AT THE GRAIN OF THE ROW IT IS ASKED ABOUT, which is two different things
    here and was one:

    * A ROW THAT IS ONE PERSON -- a live alarm, a crop of a frame, the crop
      `box_row` anchors a sheet on -- is answered by ITS OWN verdict where a
      human gave it one, and by the frame's otherwise. The specific answer
      beats the general one everywhere else a human verdict is read
      (`crop_verdict`, `audit._crop_review`), and it did not reach here at all:
      a hedged crop a reviewer answered `no_violation` was asked for no helmet
      box by anybody, and shipped with a person box and no helmet box --
      teaching the detector that a helmeted head is background, which is
      verbatim the failure the comment below says this rule exists to prevent.
    * A ROW THAT STANDS FOR A WHOLE IMAGE has no "own" answer to read: the
      per-crop answers on its review belong to the people in the picture, not
      to it, and the row is only whichever of them decided the verdict. Its
      own frame answer is the whole of what it can be asked. Which of its
      people a human cleared one at a time is a question about a crop, and
      `_a_person_a_human_cleared_owes_a_box` asks it of the crop.

    Reading the second as if it were the first put a frame's deciding crop in
    charge of the whole image: one `cannot_determine` on the person who
    happened to decide the verdict, and the frame left the box queue with a
    cleared worker in it still owing a box.
    """
    if row.get(AWAITING_VERDICT_CROPS) is not None:
        return entry.get("verdict")
    return crop_verdict(entry, row) or entry.get("verdict")


def _a_person_a_human_cleared_owes_a_box(row: dict, entry: dict) -> bool:
    """Whether the crop this image's box question is about was cleared BY A
    HUMAN, one person at a time.

    THE IMAGE'S OWN OUTCOME CANNOT ANSWER THIS, for the reason `needs_verdict`
    gives about the other queue: the outcome is an OR, so a frame holding one
    crop the judge fired on and one it hedged over reads `CONFIRM_TP`, and
    routing on that string hid every such frame -- here, every frame whose
    hedge a reviewer had just answered "the helmet is worn". Those crops fell
    out of BOTH queues and shipped with a person box and no helmet box.

    ASKED ONLY OF THE ANCHOR, AND ONLY OF A HUMAN ANSWER. `box_row`'s crop is
    already one that owes a box, and the human answer is what bounds the
    population: asking the whole question per crop instead would put every
    frame holding any weakly-boxed cleared worker in this queue (measured on
    the live corpus, pending 16 -> 94, task 42), which is a workload decision
    and not this one. A crop a reviewer answered `no_violation` is the same
    statement that has always brought a frame back here -- an escalation
    resolved to "the helmet is worn" -- arriving on the person rather than on
    the picture.
    """
    crop = row.get(BOX_CROP)
    if not isinstance(crop, dict):
        return False
    return crop_verdict(entry, crop) == "no_violation"


def needs_box(
    row: dict,
    config: PipelineConfig,
    review: dict | None = None,
    audit: dict | None = None,
) -> bool:
    """A helmet that is worn, whose box is not good enough to train on.

    Two ways for the box to qualify, both meaning "no trustworthy detector
    box": it came from the VLM fallback (the detector found nothing, or
    could not run), or the detector did find one but scored it below
    `review_confidence`.

    Two ways for the CASE to qualify. The judge clearing it
    (`CONFIRM_FP`) is the obvious one. The other is an escalation a human
    has since resolved to `no_violation` -- which means "the helmet is
    worn", making it a helmet training example exactly like a judge
    clearance. `_resolve_outcome` downgrades a low-confidence FP to
    ESCALATE, so this is a routine path, not an edge case; gating only on
    the judge's own outcome let those cases fall out of both queues and
    ship with nothing but the loose fallback box.
    """
    if audit_reopened_box(audit, review):
        # An auditor has said the helmet is worn. That settles "is there a
        # helmet to box" regardless of what the judge or an earlier reviewer
        # concluded, and it is the only route by which a case exported as a
        # violation comes back for a box.
        return True
    if row.get("hazard_type") != VIOLATION_TYPE_TO_HAZARD_LABEL[ViolationType.HELMET]:
        return False

    outcome = row.get("outcome")
    human_verdict = _human_verdict_for(row, review or {})
    # Two routes to "a helmet is worn on this case", and the reason code gates
    # only the first of them. The judge's `reason_code` explains its OWN
    # verdict, so it says `helmet_worn` when the judge cleared the alarm for
    # that reason. An escalation's reason code says why the judge gave up
    # (`head_occluded`, and so on) and can never be `helmet_worn` -- so
    # requiring it up front made the second route unreachable, which is a
    # silent data bug rather than a missing feature: the case fell out of both
    # queues and shipped with a person box and NO helmet box, teaching the
    # detector "no helmet here" on a case a human had just said was clear.
    # For that route the human's verdict IS the statement that a helmet is
    # worn, whatever the judge escalated for.
    judge_cleared = outcome == "CONFIRM_FP" and row.get("reason_code") == "helmet_worn"
    human_cleared = outcome == "ESCALATE" and human_verdict == "no_violation"
    if not (judge_cleared or human_cleared):
        return False
    if row.get("helmet_box_source") != "cv":
        return True
    confidence = _finite_float(row.get("helmet_box_confidence"))
    if confidence is None:
        return True
    return confidence < config.helmet_box.review_confidence


def _sort_key(row: dict) -> str:
    """Timestamps come from JSON and are not guaranteed to be strings; a bare
    `<` between an int and a str took the endpoint down with a 500."""
    return str(row.get("timestamp") or "")


def _is_reviewed(queue: QueueName, entry: dict, row: dict | None = None) -> bool:
    """Whether the question this queue asks about this row has been answered.

    FOR THE VERDICT QUEUE THAT IS THE CROPS, not the presence of an entry.
    A frame answered once and re-judged afterwards carries both a verdict and
    a crop the judge hedged on that nobody has seen: reading the entry alone
    called it reviewed and hid it from the queue, while the audit case held it
    back as unanswered -- so nobody could answer it and nothing could be
    audited. It stays IN the queue either way, so `include_reviewed` still
    shows the frames a reviewer has already worked.
    """
    if queue == "verdict":
        if row is not None and AWAITING_VERDICT_CROPS in row:
            return not row[AWAITING_VERDICT_CROPS]
        return entry.get("verdict") in VERDICT_OPTIONS
    return entry.get("helmet_box_status") in BOX_STATUS_OPTIONS


def is_exportable(entry: dict) -> bool:
    """Whether a settled review yields a usable training label.

    A `cannot_determine` verdict and a `no_helmet_visible` box are both
    finished work, but neither is a label -- exporting either would put a
    guess into the training set, which is the thing both options exist to
    avoid. A box the reviewer ACCEPTED is a label: the coordinates came from
    the detector, but a human confirmed them, which is the distinction that
    matters downstream."""
    if entry.get("verdict") is not None:
        return entry.get("verdict") in EXPORTABLE_VERDICTS
    return entry.get("helmet_box_status") in LABELLING_BOX_STATUSES


def item_for(row: dict, queue: QueueName, entry: dict) -> QueueItem:
    """One row as a queue item, whether or not the queue would list it.

    Public because a case can be legitimately reachable without being work:
    an image an auditor struck every crop out of has nothing to ask a human,
    so `build_queue` drops it, and the reviewer still has to be able to reach
    it to put a crop back.
    """
    return _to_item(row, queue, entry)


def _to_item(row: dict, queue: QueueName, entry: dict) -> QueueItem:
    hazard_type = row.get("hazard_type") or ""
    violation_type = HAZARD_LABEL_TO_VIOLATION_TYPE.get(hazard_type)
    crop_box = _box_to_list(row.get("crop_box"))
    person_box = _box_to_list(row.get("person_box"))
    helmet_box = _box_to_list(row.get("helmet_box"))
    return QueueItem(
        alarm_id=row.get("alarm_id") or "",
        queue=queue,
        hazard_type=hazard_type,
        violation_type=violation_type.value if violation_type else None,
        timestamp=row.get("timestamp"),
        image_key=row.get("image_key"),
        image_width=row.get("image_width"),
        image_height=row.get("image_height"),
        person_box=person_box,
        helmet_box=helmet_box,
        crop_box=crop_box,
        person_box_in_crop=_shift_into_crop(person_box, crop_box),
        helmet_box_in_crop=_shift_into_crop(helmet_box, crop_box),
        helmet_box_source=row.get("helmet_box_source"),
        helmet_box_confidence=row.get("helmet_box_confidence"),
        judge_outcome=row.get("outcome"),
        judge_reason=row.get("reason_code"),
        judge_reasoning=row.get("reasoning"),
        judge_confidence=row.get("confidence"),
        judge_model=row.get("judge_model"),
        frame_available=frame_available(row.get("image_key")),
        reviewed=_is_reviewed(queue, entry, row),
        review=entry,
        candidates=row.get("candidates") if isinstance(row.get("candidates"), list) else [],
        awaiting_verdict=(
            row[AWAITING_VERDICT_CROPS]
            if isinstance(row.get(AWAITING_VERDICT_CROPS), list)
            else []
        ),
        box_crop_id=box_crop_id(row),
    )


def build_queue(
    rows: list[dict],
    queue: QueueName,
    reviews: dict[str, dict],
    *,
    config: PipelineConfig | None = None,
    include_reviewed: bool = False,
    include_unreviewable: bool = False,
    audits: dict[str, dict] | None = None,
) -> list[QueueItem]:
    """The cases awaiting a human in one queue, newest first.

    A row with no `alarm_id` is skipped rather than surfaced: it cannot be
    reviewed (there is nothing to key the decision on) and cannot be
    exported.

    A row whose frame cannot be fetched is also excluded by default. Those
    are history, not work: nothing a reviewer does can settle them, because
    there is no image to look at. Every row written before frame persistence
    existed is in this state, so listing them made a queue that was entirely
    unworkable look like a backlog. They are counted separately (see
    `queue_counts`) so nothing is hidden, and `include_unreviewable=True`
    brings them back.
    """
    config = config or get_config()

    # One item per alarm, not per row. The audit log is append-only, so
    # re-running an alarm appends a SECOND row for the same id -- and the
    # review sidecar is keyed by alarm id alone. Listing both rows showed
    # the case twice, double-counted the badge, and let a single save
    # record a verdict against a row the reviewer never saw. The newest row
    # wins, which is also the run whose frame is on disk (a re-run
    # overwrites the frame at the same key).
    newest: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        alarm_id = row.get("alarm_id")
        # No id means nothing to key a decision on: unreviewable and
        # unexportable, so surfacing it would be a dead end.
        if not alarm_id or not isinstance(alarm_id, str):
            continue
        previous = newest.get(alarm_id)
        if previous is None or _sort_key(row) >= _sort_key(previous):
            newest[alarm_id] = row

    items = []
    for alarm_id, row in newest.items():
        entry = _entry_for(reviews, alarm_id)
        audit = _entry_for(audits or {}, alarm_id)
        if queue == "verdict":
            # A frame a reviewer has already answered stays LISTED -- that is
            # what `include_reviewed` is for -- even once its crops are all
            # resolved and `needs_verdict` is False. An image an auditor struck
            # every crop out of carries no outcome and is the exception: there
            # is nothing left the answer could have been about, and it leaves
            # both queues on its own (it stays reachable by id).
            answered = bool(entry.get("verdict")) and bool(row.get("outcome"))
            if not needs_verdict(row) and not answered:
                continue
        elif not (
            needs_box(row, config, entry, audit)
            or _a_person_a_human_cleared_owes_a_box(row, entry)
        ):
            continue
        # WHICH FRAMES are in the box queue is unchanged -- asked of the image,
        # as it always was. WHICH PERSON the sheet is about is not: it is the
        # crop that owes the box, which is rarely the crop that decided the
        # frame's verdict. See `box_row`.
        item = _to_item(box_row(row) if queue == "box" else row, queue, entry)
        if queue == "box" and audit_reopened_box(audit, entry):
            item = replace(
                item,
                requeued_by_audit=True,
                audit_corrected_verdict=audit.get("corrected_verdict"),
            )
        # An audit correction supersedes the earlier box decision, so the case
        # counts as unreviewed again -- otherwise it would be re-queued and
        # then immediately filtered back out as "already decided".
        reviewed = item.reviewed and not audit_reopened_box(audit, entry)
        if reviewed and not include_reviewed:
            continue
        if not item.frame_available and not include_unreviewable:
            continue
        items.append(item)

    # Newest first: an escalation is an operational loose end, and the most
    # recent one is the one someone is most likely waiting on.
    return sorted(items, key=lambda i: str(i.timestamp or ""), reverse=True)


def queue_counts(
    rows: list[dict],
    reviews: dict[str, dict],
    *,
    config: PipelineConfig | None = None,
    audits: dict[str, dict] | None = None,
) -> dict[str, dict[str, int]]:
    """Per queue: pending work, reviewed total, and how many cases cannot be
    worked at all because their frame was never stored.

    `unreviewable` is reported rather than silently dropped: it is the
    honest answer to "why is my queue empty when the audit log has 49
    rows".
    """
    config = config or get_config()
    counts = {}
    for queue in ("verdict", "box"):
        everything = build_queue(
            rows,
            queue,
            reviews,
            config=config,
            include_reviewed=True,
            include_unreviewable=True,
            audits=audits,
        )
        workable = [item for item in everything if item.frame_available]
        counts[queue] = {
            "pending": len([i for i in workable if not i.reviewed]),
            "total": len(workable),
            "unreviewable": len(everything) - len(workable),
        }
    return counts


def audit_readiness(
    rows: list[dict],
    reviews: dict[str, dict],
    *,
    config: PipelineConfig | None = None,
    exported: dict[str, dict] | None = None,
    exportable_now: int | None = None,
    audited: int = 0,
) -> dict:
    """Whether cases can be audited yet, and what is standing in the way.

    An audit spot-checks *settled* cases -- both the ones a human decided and
    the ones the pipeline settled alone -- against what was actually prepared
    for training. Its output is the ground truth for live data, where there
    is none: `rolling_metrics.compute_shadow_mode_agreement` already measures
    agent-vs-human agreement against an 85% target, and audit decisions are
    exactly that signal for cases with no gold label.

    So auditing needs a training export to audit, which does not exist yet.
    Rather than hide the tab until then, this reports real progress toward it
    -- how many cases are settled, how many would yield a label, how many
    have been exported -- because "what do I have to do before I can audit"
    is a more useful answer than a disabled tab.

    `exported` is the export manifest, keyed by alarm id, once there is one.

    `exportable_now` is how many cases the EXPORT would actually write.
    Settled is not the same as exportable: the export additionally needs a
    stored frame and a recorded crop region, which rows judged before those
    were added do not have. Counting settled cases as exportable made this
    panel claim 13 where a run would write 1 -- so the real figure is
    passed in from the exporter itself rather than re-derived here, which
    is also what keeps the two from drifting apart again.
    """
    config = config or get_config()
    settled_by_human = 0
    labelled_by_human = 0
    auto_settled = 0
    excluded = 0

    seen: set[str] = set()
    for queue in ("verdict", "box"):
        for item in build_queue(
            rows, queue, reviews, config=config, include_reviewed=True
        ):
            if item.alarm_id in seen:
                continue
            seen.add(item.alarm_id)
            if not item.reviewed:
                continue
            settled_by_human += 1
            if is_exportable(item.review):
                labelled_by_human += 1
            else:
                excluded += 1

    # Everything else with a usable frame the pipeline settled on its own:
    # no escalation, and a helmet box good enough that no human was asked.
    # Those are the cases with the LEAST oversight, which is what makes them
    # worth auditing at all.
    for row in rows:
        if not isinstance(row, dict):
            continue
        alarm_id = row.get("alarm_id")
        if not alarm_id or alarm_id in seen or not frame_available(row.get("image_key")):
            continue
        if needs_verdict(row) or needs_box(row, config, _entry_for(reviews, alarm_id)):
            continue
        seen.add(alarm_id)
        auto_settled += 1

    settled_with_a_label = auto_settled + labelled_by_human
    exportable = (
        settled_with_a_label if exportable_now is None else exportable_now
    )
    exported_count = len(exported or {})

    blockers = []
    if exported_count == 0:
        blockers.append(
            {
                "code": "no_export",
                "message": (
                    "No training data has been exported yet, so there is nothing "
                    "prepared to audit against. The export is the next thing to build."
                ),
            }
        )
    if exportable == 0:
        blockers.append(
            {
                "code": "nothing_exportable",
                "message": (
                    "No settled case yields a training label yet. Judge some alarms on "
                    "Run pipeline, and settle anything that lands in the two queues."
                ),
            }
        )

    return {
        "ready": not blockers,
        "blockers": blockers,
        "counts": {
            "auto_settled": auto_settled,
            # Settled AND label-yielding, but not necessarily exportable --
            # see `exportable`, which is the number a run would write.
            "settled_with_a_label": settled_with_a_label,
            "settled_by_human": settled_by_human,
            "labelled_by_human": labelled_by_human,
            "excluded_by_reviewer": excluded,
            "exportable": exportable,
            "exported": exported_count,
            "audited": audited,
        },
        # No sampling. The audit queue is every settled case, oldest first,
        # capped at `audit.AUDIT_LIMIT` -- a bound on human workload rather
        # than a statistical sample, so there is no rate to report. The
        # earlier 20%-keyed-on-alarm-id scheme was replaced before it shipped;
        # reporting it here would describe a queue that does not exist.
    }
