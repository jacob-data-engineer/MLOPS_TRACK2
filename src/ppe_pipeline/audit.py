"""Auditing settled cases — the ground truth for data that has none.

An audit spot-checks cases that are already SETTLED, against what was
prepared for training. It exists because live data has no gold label: for an
alarm arriving off a camera there is nothing to score the judge against, so
a human's after-the-fact agreement IS the measurement.
`rolling_metrics.compute_shadow_mode_agreement` already scores agent-vs-human
agreement against an 85% target; audit decisions are that same signal for
cases with no gold label.

Two kinds of case are worth auditing, for different reasons:

* **pipeline-settled** -- nothing human ever looked at them. No escalation,
  and a helmet box confident enough that nobody was asked. These have the
  least oversight in the whole system, which is exactly why they need some.
* **human-settled** -- a reviewer decided them. Auditing these audits the
  reviewer, not the model, and both are worth knowing about.

WHAT AN AUDITOR RECORDS

Agree or disagree with the settled outcome, and on disagreement, what the
verdict should have been. The correction matters as much as the
disagreement: "wrong" alone gives you an error rate, while "wrong, and it
was actually a violation" also gives you a usable label.

WHEN AN AUDIT GOES STALE

An audit decision is about the JUDGE'S OUTPUT, not about the image -- unlike
a reviewer's verdict or drawn box, which are facts about the photograph and
stay true however often the case is re-run. Alarm ids are the image's own
identity (an eval case id is its file stem), so re-running a case supersedes
the earlier row rather than duplicating it -- which is what we want, except
that an "agree" recorded against the old verdict would silently read as
agreeing with the new one. So an audit records the verdict it was agreeing
or disagreeing WITH, and stops counting once that verdict changes: the case
returns to the queue to be audited again. Entries written before this was
recorded have no `audited_verdict` and are left alone -- there is no way to
know what they agreed with, and guessing would be worse than trusting them.

CV ROWS SHARE THIS LOG

The helmet-CV emulation (`helmet_cv/`) writes its judge rows into the SAME
`eval/track1_audit_log.json` as live alarms, by CEO decision — one screen,
one human decision, both models scored against it. Those rows carry
`row_kind: "helmet_cv"` at the top level and a nested `cv` block, and every
query here selects one kind or the other (`kind="live"` by default). A CV row
must never reach the live audit queue: it is a frame from a clip somebody
already alarmed on, not an alarm, and counting it would put a synthetic case
in the shadow-mode agreement rate that the 85% target is read off.

**`POST /metrics/reset` wipes this file.** It clears produced state, and by
this decision the CV comparison corpus is now produced state living inside
it — so a reset destroys the pilot's judge rows along with the test run it
was aimed at. `reset` has no concept of row kind and is not being taught one
here. Take a copy of the audit log before resetting during a pilot.

THE 400 CAP

The queue shows at most `AUDIT_LIMIT` cases, oldest first, and fewer when
fewer exist. That is a workload cap, not a statistical sample: it bounds how
much is in front of a person at once, and because audited cases drop out,
finishing some lets more in. Oldest-first keeps membership stable -- a new
case joins the end of the queue rather than displacing something a person
may already be part-way through.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from . import storage
from .config import PipelineConfig, get_config
from .eval.metrics import ConfusionCounts, classify
from .escalations import (
    VERDICT_OPTIONS,
    _entry_for,
    crop_verdict,
    frame_available,
    frame_verdict_answers,
    is_exportable,
    needs_box,
    needs_verdict,
)

DEFAULT_AUDITS_PATH = "eval/audit_reviews.json"
_GROUND_TRUTH_UNSET = object()

#: Most cases shown at once. A cap on human workload, not a sample size --
#: see the module docstring.
AUDIT_LIMIT = 400

#: Below this many audited cases, an agreement rate is not worth reading:
#: at n=3 one disagreement moves it 33 points. The rate is still shown --
#: withholding it would be worse than showing it with a caveat -- but the
#: sample size is flagged, and the threshold is reported to the UI rather
#: than hardcoded there, so there is one source of truth for it.
AUDIT_MIN_SAMPLES = 10

#: The same floor for a block counted in CROPS. The same human decisions
#: produce several times as many crops as frames (8.54 people per alarm frame
#: on `pilot-v3`), so a crop `n` clears the frame-tuned floor above on
#: identical evidence and a screen would present a confident-looking number
#: derived from three human answers.
AUDIT_CROP_MIN_SAMPLES = AUDIT_MIN_SAMPLES * 4

#: What an auditor can conclude about the settled outcome.
#:
#: `box_wrong` is the case the other two cannot express: the verdict is
#: right, and the helmet box the case was labelled with is not. Recording
#: that as a `disagree` would corrupt the verdict metrics (it would claim
#: the judge decided wrongly when it did not); recording it as an `agree`
#: would leave a bad label sitting in the training set. It endorses the
#: verdict and sends the case back for a box.
AUDIT_DECISIONS = ["agree", "disagree", "box_wrong"]

#: The decisions that endorse the settled VERDICT, whatever they say about
#: the box. What the confusion matrix scores the judge on.
VERDICT_ENDORSING_DECISIONS = ("agree", "box_wrong")

#: An audit written BY the decision it is agreeing with -- a reviewer
#: resolving an escalation, which settles the case and audits it in one act
#: (CEO, 2 Sep). The entry is a real ground truth and counts everywhere a
#: ground truth counts; what it is NOT is an independent check, so the
#: agreement rate excludes it. Left in `audited` deliberately: the case is
#: decided and must not read as pending work.
SELF_SETTLED_SOURCE = "escalation_review"

#: On disagreement, what it should have been. Same vocabulary as the
#: escalation review, so nothing downstream learns a second one --
#: `cannot_determine` included, because an auditor can equally find that the
#: evidence never supported any verdict.
CORRECTED_VERDICTS = list(VERDICT_OPTIONS)

#: What a person carried into a case is still waiting on.
#:
#: The two are NOT interchangeable, and one rule turns on the difference: a
#: person awaiting a VERDICT has no answer of any kind, so they contribute no
#: verdict to the image's OR -- and they BLOCK it from settling at all unless
#: it is already a violation (`case_outcome`, CEO 5 Sep). The review answers
#: the CROP, not only the frame, so a frame does not stop needing one because
#: somebody else in it fired.
#:
#: A person awaiting a BOX has been answered --
#: the judge cleared them -- and what is owed is a helmet rectangle, which is
#: a labelling gap, not a verdict gap. Treating both as "no answer" meant that
#: striking out the one violating crop of a frame whose remaining crops were
#: cleared-but-unboxed left the image with NO verdict at all, where the honest
#: answer is that it is cleared.
AWAITING_VERDICT = "verdict"
AWAITING_BOX = "box"

SettledBy = Literal["pipeline", "human"]


@dataclass(frozen=True)
class AuditItem:
    alarm_id: str
    settled_by: SettledBy
    hazard_type: str
    timestamp: str | None
    image_key: str | None
    crop_box: list[float] | None
    #: Both in ORIGINAL FRAME PIXELS, like `crop_box`, so a caller drawing over
    #: the served crop subtracts the crop origin. Carried so an auditor can SEE
    #: the box the pipeline settled on: judging "was this verdict right" from
    #: the crop alone, with the evidence the judge acted on left undrawn, is
    #: asking someone to check work they have not been shown.
    helmet_box: list[float] | None
    person_box: list[float] | None
    #: What was settled, and by whom -- the thing being audited.
    settled_verdict: str | None
    judge_outcome: str | None
    judge_reason: str | None
    judge_reasoning: str | None
    judge_model: str | None
    helmet_box_source: str | None
    helmet_box_confidence: float | None
    reviewed_by: str | None
    #: Whether this case reached the training export. An audit is meant to
    #: check what was PREPARED, so an unexported case is auditable but its
    #: findings say nothing about the dataset.
    exported: bool = False
    audited: bool = False
    #: A decision exists, but the verdict it was about has since changed, so it
    #: no longer counts and the case is back in the queue. Surfaced rather than
    #: hidden so an auditor seeing a case twice knows why.
    stale_audit: bool = False
    audit: dict = field(default_factory=dict)
    #: The helmet-CV emulation's own answer for this same frame, when the row
    #: came from that pilot -- `{}` for every live alarm. Nested exactly as it
    #: sits on the row: CV's `helmet_box` and the judge's are different claims
    #: by different models and must not share a key.
    cv: dict = field(default_factory=dict)
    #: What this person is still waiting on, when they are carried into a
    #: case without being settled: `"verdict"` for a judge that hedged.
    #: `None` for everyone else. See `build_audit_queue`'s `include_unsettled`.
    awaiting: str | None = None
    #: Where this row came from: `{source: "live_cv" | "offline_gold", ...}`.
    #: The live feed and the offline gold set converge on this one queue, and
    #: an auditor deciding a case is entitled to know which arm produced it.
    #: `{}` for rows written before lineage existed.
    lineage: dict = field(default_factory=dict)


def load_audits(path: str | Path = DEFAULT_AUDITS_PATH) -> dict[str, dict]:
    value, _ = storage.get_storage().read_json(
        Path(path).as_posix(), default={}, revalidate=True
    )
    return value if isinstance(value, dict) else {}


def save_audit(
    alarm_id: str,
    decision: str,
    *,
    corrected_verdict: str | None = None,
    audited_verdict: str | None = None,
    source: str = "audit",
    auditor: str = "",
    note: str = "",
    path: str | Path = DEFAULT_AUDITS_PATH,
    now: datetime | None = None,
    ground_truth_verdict: str | None | object = _GROUND_TRUTH_UNSET,
    excluded_people: list[str] | None = None,
) -> dict:
    """Record one audit decision, concurrency-safely.

    Kept in its own sidecar rather than merged into the escalation reviews:
    an audit is a judgement ABOUT a settled case, including cases no
    reviewer ever touched, so folding it into the review record would
    conflate "what was decided" with "whether that decision was right".

    EVERY SAVE MERGES OVER THE PREVIOUS ENTRY, so a key this call writes is a
    key it OVERWRITES. `excluded_people=None` means "I am not touching the
    strike-outs" and `[]` means "clear them" -- two different intentions that
    a bare list default could not tell apart. It could not for one release,
    and the cost was real: a box reviewer or a helmet-CV reviewer pressing
    their own button silently erased an auditor's strike-outs, which flipped
    the verdict back and re-published a crop a human had pulled.
    """
    entry = {
        "decision": decision,
        "corrected_verdict": corrected_verdict,
        # The verdict this decision was ABOUT. Recorded so the decision can be
        # retired if that verdict later changes -- see the module docstring.
        "audited_verdict": audited_verdict,
        # `audit` for a decision made on the Audit screen, `box_review` for one
        # a helmet-box reviewer produced as a side effect of finding no helmet.
        # Both are human findings and both count; the distinction is worth
        # keeping because only one of them was someone's whole job at the time.
        "source": source,
        "note": note,
        "audited_by": auditor,
        "audited_at": (now or datetime.now(timezone.utc)).isoformat(),
        # CROPS THE AUDITOR STRUCK OUT, by candidate id (CEO, 2 Sep: "make
        # those options selectable to the auditor can remove them so it gets
        # removed from the exported training data or doesnt get exported").
        #
        # A separate act from the verdict, and deliberately so. The verdict
        # says what the PICTURE was; this says which boxes in it are not worth
        # training on -- a box on a hose, a crop too small to read, a person
        # already labelled by an overlapping box. Recorded per crop rather
        # than as a redrawn box: the auditor is not a labeller, and "this one
        # is junk" is a judgement they can make reliably.
    }
    if ground_truth_verdict is not _GROUND_TRUTH_UNSET:
        entry["ground_truth_verdict"] = ground_truth_verdict
    if excluded_people is not None:
        entry["excluded_people"] = sorted(set(excluded_people))

    def mutate(current: dict) -> dict:
        previous = _entry_for(current, alarm_id)
        revision = int(previous.get("revision", 0)) + 1
        revised = {
            **previous,
            **entry,
            "revision": revision,
            "revises_audited_at": previous.get("audited_at"),
        }
        return {**current, alarm_id: revised}

    updated = storage.get_storage().update_json(Path(path).as_posix(), mutate, default={})
    return updated[alarm_id]


def _settled_verdict(row: dict, review: dict) -> str | None:
    """The verdict that stands on this case.

    A human's decision wins over the judge's, because that is what settling
    means. Mapped onto the review vocabulary so an auditor compares like
    with like rather than against a raw CONFIRM_TP/CONFIRM_FP.

    WHICH human decision -- this crop's own answer or its frame's -- is
    settled before it gets here, by `_crop_review`. `review` is already the
    one that applies to this crop.
    """
    human = review.get("verdict")
    if human:
        return human
    outcome = row.get("outcome")
    if outcome == "CONFIRM_TP":
        return "violation"
    if outcome == "CONFIRM_FP":
        return "no_violation"
    return None


def _is_audited(entry: dict) -> bool:
    return entry.get("decision") in AUDIT_DECISIONS


def _audit_is_stale(entry: dict, settled_verdict: str | None) -> bool:
    """Whether this decision was made against a verdict that no longer stands.

    An entry with no `audited_verdict` predates the field, so there is nothing
    to compare and it is taken at face value.
    """
    if not _is_audited(entry):
        return False
    if entry.get("source") in ("box_review", SELF_SETTLED_SOURCE):
        # This audit IS the reason the settled verdict is what it is -- a
        # reviewer finding no helmet on a case the judge cleared, or one
        # resolving an escalation. Comparing it against the verdict it
        # produced would retire it the instant it was written.
        return False
    if "audited_verdict" not in entry:
        # Predates the field. There is no way to know what it agreed with and
        # guessing would be worse than trusting it.
        return False
    # PRESENT AND NULL IS NOT ABSENT. A case whose only remaining crops are
    # awaiting a human has no settled verdict, and an audit recorded against
    # that `None` used to short-circuit here and become permanently
    # unretirable -- endorsing a verdict that did not exist, and still
    # endorsing it after a re-run produced a real one.
    return entry.get("audited_verdict") != settled_verdict


def _sort_key(row: dict) -> tuple[str, str]:
    """Oldest first, alarm id breaking ties, so the order is total and
    stable -- a queue that reshuffles under a reviewer is worse than one
    that is merely long."""
    return (str(row.get("timestamp") or ""), str(row.get("alarm_id") or ""))


def _box_list(box) -> list[float] | None:
    if not isinstance(box, dict) or not all(k in box for k in ("x1", "y1", "x2", "y2")):
        return None
    return [float(box[k]) for k in ("x1", "y1", "x2", "y2")]


#: Row kinds, and what each selects. "live" is every ordinary alarm row --
#: the ones with no `row_kind` at all, which is every row written before the
#: helmet-CV pilot existed.
ROW_KINDS = ("live", "cv", "all")


def is_cv_row(row: dict) -> bool:
    """Whether this audit-log row came from the helmet-CV emulation."""
    return row.get("row_kind") == "helmet_cv"


def _matches_kind(row: dict, kind: str) -> bool:
    if kind == "all":
        return True
    return is_cv_row(row) if kind == "cv" else not is_cv_row(row)


#: A drawn helmet rectangle, which belongs to ONE person's head -- see
#: `_crop_review`. The status beside it is a fact about the frame's box
#: question; these two are the answer itself.
_BOX_LABEL_FIELDS = ("helmet_box", "helmet_box_in_crop")


def _crop_review(reviews: dict[str, dict], row: dict, frame_crops: list[dict]) -> dict:
    """The human decision that applies to this crop, by its own id or its image's.

    An ordinary alarm is one row and one review, both under the same id. A
    helmet-CV row is one PERSON, and the decision about it is filed against the
    IMAGE -- `set_verdict` is called with the frame key, which is what the
    escalation queue lists and what the review screen writes. A crop looked up
    by its own id alone never found the verdict a reviewer had just given for
    the picture it belongs to: measured on the live corpus, 0 of 23 review
    entries matched a crop id and 23 of 23 matched a frame key, so every
    answered frame still read `settled_verdict: null` on the audit screen and
    nothing at all landed in the `human_settled` bucket.

    WHAT TRAVELS FROM THE FRAME TO A CROP IS NOT ONE RULE BUT THREE.

    * The VERDICT travels by `escalations.frame_verdict_answers`, which is
      where the polarity lives: a clearance is about everyone, a violation is
      about somebody, and a hedge is answered only when the answer could have
      been about it.
    * The box STATUS travels: "the box question on this frame has been
      answered" is a fact about the frame, and the box queue asks it once per
      frame.
    * The drawn RECTANGLE does not, unless the frame holds one person. It is
      one person's head, and carrying it would label one worker's helmet on
      another (`OPEN_TASKS.md` task 15).

    AND THIS CROP'S OWN ANSWER OVERRIDES ALL OF IT. `escalations.CROP_VERDICTS`
    is a human answering about THIS person, so where it exists there is
    nothing to carry and no polarity to apply -- the specific answer beats the
    general one. Resolved here, once, rather than at each reader: this is the
    function that decides what a human said about a crop, and everything that
    turns on that (`_settled_verdict`, whether the case counts as
    human-settled, `needs_box`, `is_exportable`) reads it through here.
    """
    own = _entry_for(reviews, row.get("alarm_id") or "")
    case = own or _entry_for(reviews, _person_case_key(row))
    if not case:
        return {}
    mine = crop_verdict(case, row)
    if own:
        return {**own, "verdict": mine} if mine else own
    carried = (
        dict(case)
        if len(frame_crops) <= 1
        else {k: v for k, v in case.items() if k not in _BOX_LABEL_FIELDS}
    )
    if mine:
        carried["verdict"] = mine
    elif case.get("verdict") and not frame_verdict_answers(row, case, frame_crops):
        carried.pop("verdict", None)
    return carried


def build_audit_queue(
    rows: list[dict],
    reviews: dict[str, dict],
    audits: dict[str, dict],
    *,
    config: PipelineConfig | None = None,
    exported: dict[str, dict] | None = None,
    settled_by: str = "all",
    include_audited: bool = False,
    limit: int = AUDIT_LIMIT,
    kind: str = "live",
    include_unsettled: bool = False,
) -> list[AuditItem]:
    """Settled cases awaiting an audit, oldest first, at most `limit`.

    `settled_by` filters to `pipeline` or `human`; the two are worth looking
    at separately, since auditing a pipeline-settled case checks the model
    and auditing a human-settled one checks the reviewer.

    `kind` selects live alarms (the default) or the helmet-CV pilot's rows,
    which share this log. Defaulting to `live` is deliberate: every existing
    caller wants alarms, and a CV row leaking into the live queue would put a
    frame nobody alarmed on into the shadow-mode agreement rate.

    `include_unsettled` carries a HELMET-CV person who is not settled into
    the result anyway, marked `awaiting`, instead of dropping them. It applies
    to CV rows only, and the asymmetry is not arbitrary:

    * a live alarm the judge ESCALATED is waiting on a human, and there is a
      queue for exactly that. Auditing it before that human answers would be
      auditing a decision nobody has made.
    * a helmet-CV person the judge escalated is waiting on nobody. The human
      step for that frame IS this audit, and the question asked here is about
      the frame, which a person can answer whether or not the judge could.
      Dropping them also made the frame look SHORT of its people -- an
      escalation read on screen as a person nobody had judged.
    """
    config = config or get_config()
    exported = exported or {}

    newest: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict) or not _matches_kind(row, kind):
            continue
        alarm_id = row.get("alarm_id")
        if not alarm_id or not isinstance(alarm_id, str):
            continue
        previous = newest.get(alarm_id)
        if previous is None or _sort_key(row) >= _sort_key(previous):
            newest[alarm_id] = row

    # The crops of each image, so a frame-level answer can be weighed against
    # what else is in the picture -- how many people, and how many of them the
    # judge hedged on. See `_crop_review`.
    frames: dict[str, list[dict]] = {}
    for row in newest.values():
        frames.setdefault(_person_case_key(row), []).append(row)

    items: list[AuditItem] = []
    for alarm_id, row in newest.items():
        frame_crops = frames[_person_case_key(row)]
        review = _crop_review(reviews, row, frame_crops)
        settled_by_human = bool(review.get("verdict") or review.get("helmet_box_status"))

        # Still waiting on a human is not settled, so there is nothing to
        # audit yet -- unless this is a CV person being carried into their
        # frame's case as context, which is what `include_unsettled` is for.
        carry_unsettled = include_unsettled and is_cv_row(row)
        awaiting: str | None = None
        if needs_verdict(row) and not review.get("verdict"):
            if not carry_unsettled:
                continue
            awaiting = AWAITING_VERDICT
        if needs_box(row, config, review) and not review.get("helmet_box_status"):
            if not carry_unsettled:
                continue
            awaiting = awaiting or AWAITING_BOX
        # A case with no frame cannot be looked at, so it cannot be audited.
        if not frame_available(row.get("image_key")):
            continue
        # A human decision that deliberately produced no label
        # (cannot_determine / no_helmet_visible) has nothing to check against
        # the dataset -- it was excluded from it on purpose.
        #
        # ONLY THIS CROP'S OWN ANSWER DROPS IT. A frame-level `cannot_determine`
        # would otherwise drop every crop of the picture, and a case with no
        # crops is no case at all: the frame left the audit queue, the case
        # endpoint 404'd, and the audit somebody had already recorded against
        # it was in no bucket and no count -- gone, rather than decided. Live
        # corpus, 6 Sep: one frame, and the reason `rolling_metrics` counted
        # 186 images where the audit summary counted 185.
        if settled_by_human and not is_exportable(review) and _entry_for(reviews, alarm_id):
            continue

        origin: SettledBy = "human" if settled_by_human else "pipeline"
        if settled_by in ("pipeline", "human") and settled_by != origin:
            continue

        settled_verdict = _settled_verdict(row, review)
        entry = _entry_for(audits, alarm_id)
        stale = _audit_is_stale(entry, settled_verdict)
        # A stale decision does not count as audited: it is back in the queue.
        audited = _is_audited(entry) and not stale
        if audited and not include_audited:
            continue

        items.append(
            AuditItem(
                alarm_id=alarm_id,
                settled_by=origin,
                hazard_type=row.get("hazard_type") or "",
                timestamp=row.get("timestamp"),
                image_key=row.get("image_key"),
                crop_box=_box_list(row.get("crop_box")),
                helmet_box=_box_list(row.get("helmet_box")),
                person_box=_box_list(row.get("person_box")),
                settled_verdict=settled_verdict,
                judge_outcome=row.get("outcome"),
                judge_reason=row.get("reason_code"),
                judge_reasoning=row.get("reasoning"),
                judge_model=row.get("judge_model"),
                helmet_box_source=row.get("helmet_box_source"),
                helmet_box_confidence=row.get("helmet_box_confidence"),
                reviewed_by=review.get("reviewed_by"),
                exported=alarm_id in exported,
                audited=audited,
                stale_audit=stale,
                audit=entry,
                awaiting=awaiting,
                cv=row.get("cv") if isinstance(row.get("cv"), dict) else {},
                lineage=row.get("lineage") if isinstance(row.get("lineage"), dict) else {},
            )
        )

    items.sort(key=lambda i: (str(i.timestamp or ""), i.alarm_id))
    return items[: max(0, limit)]


def case_verdict(verdicts: list[str | None]) -> str | None:
    """OR one image's per-person verdicts into the image's verdict.

    A frame is a violation if ANY person in it is; it is cleared only when
    every person was cleared; and it has no answer at all only when nothing
    in it was decided either way.

    IT ORS VERDICT STRINGS, AND THAT IS ALL IT CAN SEE. "This crop has no
    answer" does not reach here, and neither does WHY -- a crop the judge
    hedged on and a crop a human answered `cannot_determine` arrive
    identically. So the rule that an unanswered hedge blocks the image from
    settling lives one level up, in `case_outcome`, which knows which crops
    are still owed a human. Pushing it down here would make a frame a HUMAN
    has answered `cannot_determine` unsettleable forever.
    """
    if "violation" in verdicts:
        return "violation"
    if "no_violation" in verdicts:
        return "no_violation"
    if "cannot_determine" in verdicts:
        return "cannot_determine"
    return None


def case_key(item: AuditItem) -> str:
    """THE IMAGE this row's crop came from -- the unit of one audit.

    CEO decision, 1 Sep: "All the crops from 1 image should be part of 1
    audit." Every arm reaches the judge the same way -- an image plus the
    boxes to crop out of it -- so every arm groups the same way here:

    * a helmet-CV row (live feed or pilot) by its FRAME, the key
      `ground_truth_key` already defines and the helmet-CV review screen
      already writes human decisions against;
    * anything else by `lineage.frame_key`, the image the run recorded
      judging, falling back to the alarm itself.

    Deliberately NOT `image_key`. That is where the judged frame was STORED,
    and a writer is free to store one image per row -- which the helmet-CV
    path does -- so it identifies a copy, not a picture. `lineage.frame_key`
    is the picture, recorded by the arm that judged it.

    For a live alarm and for today's gold cases that is one row per image, so
    the case is a group of one and its id stays the alarm id -- which is what
    keeps every audit decision ever recorded still attached to its case.
    """
    if item.cv:
        return ground_truth_key(item.cv) or item.alarm_id
    return item.lineage.get("frame_key") or item.alarm_id


@dataclass(frozen=True)
class AuditCase:
    """One IMAGE, with every person the judge was asked about nested in it.

    The human decides the image once. Scoring a per-person row would count
    that one decision once per person, which inflates `n` and weights the
    result by how crowded each image happened to be -- measured on the pilot,
    three of eighteen frames carried 46% of one arm's headline precision.
    `group_by_frame` exists for that reason on the helmet-CV side; this is the
    same rule applied to the audit queue itself.
    """

    case_id: str
    #: Every crop of this image, in the order the judge saw them.
    people: list[AuditItem]
    settled_by: SettledBy
    hazard_type: str
    timestamp: str | None
    #: The image's verdict: the OR of the people an auditor KEPT. What an
    #: auditor agrees or disagrees with. `NO_VALID_CROPS` when they struck out
    #: every one of them -- see `case_settled_verdict`.
    settled_verdict: str | None
    #: The judge's own answer for the image, OR'd the same way, in the audit
    #: log's vocabulary.
    judge_outcome: str | None
    #: Which arm produced this case -- `live_cv`, `offline_gold`, or `""` for
    #: a row written before lineage was recorded.
    source: str
    #: False when some of this image's people were never judged (a failed
    #: call, or a live feed stopped part way). Their absence would otherwise
    #: read as "nobody else was in violation".
    complete: bool
    people_expected: int
    #: How many of this image's crops the first-pass person gate discarded as
    #: holding no person. Shown to the auditor, because "12 boxes, 2 people"
    #: is a fact about the detector they are entitled to see.
    people_gated: int = 0
    #: People in this image the judge declined to answer about. They are
    #: shown, not hidden: an auditor deciding the picture is entitled to see
    #: where the model gave up.
    people_awaiting: int = 0
    #: Kept people the judge hedged on and nobody has answered yet. Above
    #: zero, this image still needs a human, whatever its verdict says -- see
    #: `CaseOutcome.awaiting_verdict`, and `routers/escalations.get_audit_queue`
    #: for the one place that filter belongs.
    awaiting_verdict: int = 0
    #: People an auditor struck out of this image. They stay in `people` --
    #: the strike-out is reversible and the auditor has to see what they
    #: struck -- but they are out of the OR that produced `settled_verdict`.
    people_excluded: int = 0
    exported: bool = False
    audited: bool = False
    stale_audit: bool = False
    audit: dict = field(default_factory=dict)
    #: The words behind that answer, taken from THE CROP THAT PRODUCED IT.
    #:
    #: Never the first crop. The image's outcome is an OR across its people,
    #: so on a violation image the answer belongs to whichever person fired --
    #: rarely person 1. Measured on the pilot, showing crops[0]'s reasoning
    #: under the image's stamp put a "clearly wearing a white helmet"
    #: explanation beneath a VIOLATION verdict on 93% of violation frames.
    #: `routers/helmet_cv._deciding_crop` fixed that on its own screen; the
    #: audit screen read `people[0]` until 2 Sep and showed the same
    #: contradiction (CEO: "Make sure its the FINAL verdict from seeing all
    #: crops, not the first crop").
    judge_reason: str | None = None
    judge_reasoning: str | None = None
    judge_model: str | None = None
    #: Which of `people` the three fields above came from, so a screen can
    #: point at it rather than making the auditor hunt for it.
    deciding_person: int = 0

    @property
    def alarm_id(self) -> str:
        """The id an audit decision is recorded against."""
        return self.case_id


def build_audit_cases(
    rows: list[dict],
    reviews: dict[str, dict],
    audits: dict[str, dict],
    *,
    config: PipelineConfig | None = None,
    exported: dict[str, dict] | None = None,
    settled_by: str = "all",
    include_audited: bool = False,
    limit: int = AUDIT_LIMIT,
    kind: str = "all",
) -> list[AuditCase]:
    """The audit queue, one case per IMAGE, oldest first, at most `limit`.

    `kind` defaults to `all` here, unlike `build_audit_queue`: this IS the one
    audit process the live feed, the pilot and the offline gold set all land
    in, so excluding an arm would put it outside the only measurement live
    data has. The per-row function keeps its `live` default, because its
    remaining callers are the ones that must not see a synthetic case.
    """
    items = build_audit_queue(
        rows,
        reviews,
        audits,
        config=config,
        exported=exported,
        settled_by=settled_by,
        # Auditedness is a property of the CASE, not of a crop: the decision
        # is recorded against the image. Filtering per row here would drop the
        # people of a decided image one at a time and leave a case full of
        # holes, so everything is collected and the filter applied below.
        include_audited=True,
        limit=10**9,
        kind=kind,
        # A frame's escalated people belong IN their frame's case. The human
        # is deciding the picture, and "the judge could not tell" is part of
        # what they are looking at.
        include_unsettled=True,
    )

    grouped: dict[str, list[AuditItem]] = {}
    for item in items:
        grouped.setdefault(case_key(item), []).append(item)

    cases: list[AuditCase] = []
    for people in grouped.values():
        first = people[0]
        if all(person.awaiting for person in people) and not first.cv:
            # Nothing here has an answer of any kind. Only reachable for a
            # non-CV row, which never carries unsettled anyway; kept as the
            # explicit statement of the rule.
            continue
        # A helmet-CV case is keyed by its frame, which is what the review
        # screen already writes decisions against; anything else keeps the
        # alarm id, so decisions recorded before this queue existed still
        # resolve to their case.
        #
        # THE INVARIANT THIS RESTS ON: no group mixes a CV row with a non-CV
        # row that shares its picture. `case_id` is taken from `first`, while
        # `included_people`'s own fallback, `rolling_metrics._images` and
        # `routers/escalations._one_row_per_image` all derive the key per row
        # -- and for a CV row that is the frame stem. So a group whose FIRST
        # row is non-CV (a live alarm carrying `lineage.frame_key` pointing at
        # a frame a CV row also covers) would file a strike-out under the live
        # alarm's id while those three look for it under the frame stem, and
        # the exclusion would apply on the audit screen and nowhere else. No
        # writer produces that shape today: a picture is judged by one arm.
        # If you make one that does, the case key has to move into a single
        # function that both this and `_case_key_of` call.
        case_id = case_key(first) if first.cv else first.alarm_id
        # THE ENTRY BEFORE THE VERDICT. The verdict is OR'd over the crops
        # this entry did not strike out, so the filter cannot run until the
        # entry has been read; it used to be fetched one line later.
        entry = _entry_for(audits, case_id)
        # A person the judge HEDGED on contributes no verdict to the OR: no
        # human has spoken for them yet. A struck-out one contributes none
        # either -- it has left the population. One merely awaiting a helmet
        # BOX does contribute, because it has an answer (see AWAITING_VERDICT).
        answer = case_outcome(people, audits, case_id=case_id)
        settled_verdict = answer.settled_verdict
        outcome = answer.judge_outcome
        deciding = answer.deciding_person
        stale = _audit_is_stale(entry, settled_verdict)
        audited = _is_audited(entry) and not stale
        if audited and not include_audited:
            continue
        expected = max(int(person.cv.get("people_in_frame") or 0) for person in people)
        gated = max(int(person.cv.get("people_gated") or 0) for person in people)
        cases.append(
            AuditCase(
                case_id=case_id,
                people=people,
                settled_by="human" if any(p.settled_by == "human" for p in people) else "pipeline",
                hazard_type=first.hazard_type,
                timestamp=min(str(person.timestamp or "") for person in people) or None,
                settled_verdict=settled_verdict,
                judge_outcome=outcome,
                judge_reason=people[deciding].judge_reason,
                judge_reasoning=people[deciding].judge_reasoning,
                judge_model=people[deciding].judge_model,
                deciding_person=deciding,
                source=str(first.lineage.get("source") or ""),
                complete=len(people) + gated >= expected,
                people_expected=max(expected, len(people)),
                people_gated=gated,
                people_awaiting=sum(1 for person in people if person.awaiting),
                awaiting_verdict=answer.awaiting_verdict,
                people_excluded=answer.people_excluded,
                # The export unit stays the CROP -- one person per training
                # sample -- so an image counts as exported once any of its
                # crops reached the dataset.
                exported=any(person.exported for person in people),
                audited=audited,
                stale_audit=stale,
                audit=entry,
            )
        )

    cases.sort(key=lambda case: (str(case.timestamp or ""), case.case_id))
    return cases[: max(0, limit)]


def _deciding_person(people: list[AuditItem], outcome: str | None) -> int:
    """Which crop produced the IMAGE's outcome.

    The image's outcome is an OR across its people, so the crop that decided
    it is the first one whose own answer matches -- on a violation image, the
    person who fired, which is rarely person 1. Taking `people[0]` instead put
    a "clearly wearing a white helmet" reasoning under a VIOLATION stamp on
    93% of the pilot's violation frames.

    When nothing was decided, person 1 stands in: there is no decider, and
    every crop is equally unhelpful.
    """
    if outcome is None:
        return 0
    for index, person in enumerate(people):
        if person.judge_outcome == outcome:
            return index
    return 0


#: How a judge outcome maps onto the verdict vocabulary `classify` speaks.
#: ESCALATE is `uncertain` -- the judge declining to answer -- and is kept as
#: its own bucket rather than scored as a miss, exactly as the offline eval
#: treats it.
_OUTCOME_TO_VERDICT = {"CONFIRM_TP": "TP", "CONFIRM_FP": "FP", "ESCALATE": "uncertain"}


def excluded_people(entry: dict | None) -> set[str]:
    """Candidate ids an auditor struck out of this case's training data.

    Read in one place because three callers need it and each would otherwise
    reach into the audit record for a key it does not own: the crop export,
    the full-frame export, and the screen that has to show them struck out.
    """
    if not isinstance(entry, dict):
        return set()
    return {str(c) for c in entry.get("excluded_people") or []}


#: Why a person is not in the population an image's verdict is OR'd over.
#:
#: `excluded_by_auditor` is the only one anything produces. The other two name
#: outcomes of the first-pass person gate, which is not built and was dropped
#: again on 3 Sep (`OPEN_TASKS.md` task 33) -- they are defined here so the
#: vocabulary is fixed in one place rather than invented at each site if that
#: feature ever returns.
EXCLUDED_BY_AUDITOR = "excluded_by_auditor"
NO_PERSON = "no_person"
BELOW_MIN_AREA = "below_min_area"
EXCLUSION_REASONS = (EXCLUDED_BY_AUDITOR, NO_PERSON, BELOW_MIN_AREA)

#: The image's outcome when an auditor struck out EVERY one of its crops.
#:
#: Its own outcome, not `no_violation` (CEO, 3 Sep). "None of these boxes is
#: worth looking at" is a different finding from "nobody in this picture was
#: in violation", and folding the first into the second would put a frame
#: carrying no evidence at all into the cleared bucket, where it would count
#: as a judge clearance somebody agreed with.
#:
#: `case_verdict([])` already returned `None` and every scorer already skipped
#: it, so this names an accidental behaviour rather than inventing one. Never
#: something an auditor can pick: it is derived from their strike-outs.
NO_VALID_CROPS = "no_valid_crops"


def _person_cv(person) -> dict:
    """One person's `cv` block, off an `AuditItem` or off a raw log row."""
    cv = person.get("cv") if isinstance(person, dict) else getattr(person, "cv", None)
    return cv if isinstance(cv, dict) else {}


def _person_case_key(person) -> str:
    """The case id a strike-out about this person is filed under.

    `case_key`'s rule, read off a raw audit-log row as well as an `AuditItem`,
    because two of the sites below group rows rather than items.
    """
    if not isinstance(person, dict):
        return case_key(person)
    cv = _person_cv(person)
    alarm_id = str(person.get("alarm_id") or "")
    if cv:
        return ground_truth_key(cv) or alarm_id
    lineage = person.get("lineage") if isinstance(person.get("lineage"), dict) else {}
    return str(lineage.get("frame_key") or alarm_id)


def included_people(case_or_items, audits: dict[str, dict], *, case_id: str | None = None):
    """THE population an image's verdict is OR'd over, and who was left out.

    Returns `(kept, [(person, reason)])`.

    ONE FILTER, READ BY EVERY SITE. The OR precedence is spelled out at five
    of them -- `case_verdict`, `build_audit_cases`, `vlm_frame_verdicts`,
    `rolling_metrics._images` and `routers/escalations._one_row_per_image` --
    and an exclusion changes the POPULATION each of them runs over rather than
    the precedence itself. Filter at one site only and the audit screen, the
    escalation queue, the metrics page and the CV comparison disagree about
    the same frame while each looks individually correct. So nothing inlines
    this.

    The rule, from the CEO (3 Sep): "same logic if after remove, as long as
    there is one violation, then the frame is violation". The struck crop
    leaves the population; the existing precedence then runs over what is
    left.

    Accepts an `AuditCase`, a list of `AuditItem`, or a list of raw audit-log
    rows. `case_id` names the case the strike-outs are filed under, and is
    passed by callers that have already decided it; derived per person
    otherwise, which is the same answer for every row that could carry a
    strike-out (a candidate id exists only on a CV row, and a CV row's case is
    its frame).
    """
    if isinstance(case_or_items, AuditCase):
        people = list(case_or_items.people)
        case_id = case_id or case_or_items.case_id
    else:
        people = list(case_or_items or [])

    struck_for: dict[str, set[str]] = {}
    kept: list = []
    dropped: list[tuple[object, str]] = []
    for person in people:
        key = case_id if case_id is not None else _person_case_key(person)
        if key not in struck_for:
            struck_for[key] = excluded_people(_entry_for(audits, key))
        candidate = str(_person_cv(person).get("candidate_id") or "")
        if candidate and candidate in struck_for[key]:
            dropped.append((person, EXCLUDED_BY_AUDITOR))
        else:
            kept.append(person)
    return kept, dropped


@dataclass(frozen=True)
class CaseOutcome:
    """What one image settles to under a given set of strike-outs.

    Every answer that depends on WHICH CROPS ARE IN, in one object with one
    author. `build_audit_cases` fills a queue from it, `record_audit` records
    the verdict a decision is being made against from it, and the preview
    endpoint answers "what would this become" from it -- so a reviewer's
    unsaved strike-outs and the save that follows cannot disagree, and no
    client has to re-implement the precedence in order to show it.
    """

    #: `NO_VALID_CROPS` when every crop was struck out. `None` while any kept
    #: crop is still waiting on a human: that is no verdict yet, which is not
    #: the same as a clearance. A VIOLATION survives that wait -- see
    #: `case_outcome` -- because no answer to the hedge can lower it.
    settled_verdict: str | None
    #: The judge's own answer for the image, OR'd over the kept crops.
    judge_outcome: str | None
    #: Which of the FULL people list the image's answer came from, so a screen
    #: can point at it. A struck-out crop is never the decider.
    deciding_person: int
    #: Candidate ids that left the population.
    excluded: list[str]
    #: Kept crops the judge hedged on and no human has answered. THE ROUTING
    #: SIGNAL: whether this image still needs a person is this being above
    #: zero, never `settled_verdict is None` -- a blocked violation frame has
    #: a verdict and still needs the answer. Also what makes a `None` verdict
    #: explicable on screen: "still waiting on a human", not "nothing here".
    awaiting_verdict: int

    @property
    def people_excluded(self) -> int:
        return len(self.excluded)


def case_outcome(
    people: list[AuditItem], audits: dict[str, dict], *, case_id: str | None = None
) -> CaseOutcome:
    """THE answer to "what is this image, given who is still in it".

    Everything that reads a verdict off a set of crops comes through here.
    Two spellings of it would let an audit retire itself the instant it was
    written, and a seventh spelling -- in the browser, to preview unsaved
    strike-outs -- would be one that no test on either side of the wire could
    catch drifting.
    """
    kept, dropped = included_people(people, audits, case_id=case_id)
    blocked = [person for person in kept if person.awaiting == AWAITING_VERDICT]
    if people and not kept:
        settled = NO_VALID_CROPS
    else:
        # Only a person awaiting a VERDICT contributes nothing -- see
        # AWAITING_VERDICT. One awaiting a box has an answer; it is the
        # rectangle that is missing.
        settled = case_verdict(
            [
                person.settled_verdict
                for person in kept
                if person.awaiting != AWAITING_VERDICT
            ]
        )
        # AN UNANSWERED HEDGE BLOCKS THE IMAGE FROM SETTLING (CEO, 5 Sep):
        # the review answers the CROP, not only the frame, so another crop
        # firing first does not short-circuit it.
        #
        # A VIOLATION IS THE EXCEPTION, and keeps its verdict. The frame is a
        # violation whatever the hedge turns out to be -- the OR cannot go
        # back down -- so nulling it would cost the judge a confirmed true
        # positive it got right and retire the audits recorded against it for
        # a finding that cannot exist. The frame is still held out of the
        # audit QUEUE, which routes on `awaiting_verdict` rather than on the
        # verdict being absent.
        if blocked and settled != "violation":
            settled = None

    # TP > ESCALATE > FP: what did the JUDGE conclude about this image. A
    # hedge outranks a clearance because it is the judge declining to answer
    # about somebody, and an image where it declined is not one it cleared.
    # Whether the image still NEEDS a human is a different question, answered
    # by `awaiting_verdict`, and how much of the corpus it annotated alone is
    # a third one, answered in `rolling_metrics` by the same field.
    judged = [_OUTCOME_TO_VERDICT.get(person.judge_outcome or "") for person in kept]
    if "TP" in judged:
        judge_outcome = "CONFIRM_TP"
    elif "uncertain" in judged:
        judge_outcome = "ESCALATE"
    elif "FP" in judged:
        judge_outcome = "CONFIRM_FP"
    else:
        judge_outcome = None

    # Chosen among the crops that are still in the population, then located in
    # the full list a screen renders: a struck-out crop with the same answer
    # must not be credited with deciding the image.
    deciding = 0
    if kept:
        chosen = kept[_deciding_person(kept, judge_outcome)]
        deciding = next(index for index, person in enumerate(people) if person is chosen)

    return CaseOutcome(
        settled_verdict=settled,
        judge_outcome=judge_outcome,
        deciding_person=deciding,
        excluded=[
            str(_person_cv(person).get("candidate_id") or "") for person, _ in dropped
        ],
        awaiting_verdict=len(blocked),
    )


def case_settled_verdict(
    people: list[AuditItem], audits: dict[str, dict], *, case_id: str | None = None
) -> str | None:
    """One image's settled verdict over the crops an auditor KEPT.

    A named shorthand for the one field `record_audit` needs off
    `case_outcome`, which is where the rule lives.
    """
    return case_outcome(people, audits, case_id=case_id).settled_verdict


def audited_ground_truth(item: AuditItem) -> str | None:
    """What an audit says the case ACTUALLY was, or `None` if it says nothing.

    Agreement endorses the verdict that stood; a disagreement replaces it with
    the correction, which is the whole reason a correction is required. A
    `cannot_determine` either way yields no label -- an auditor finding the
    evidence never supported any verdict is a real finding, but not a
    ground truth to score against.
    """
    if not item.audited:
        return None
    stable_truth = item.audit.get("ground_truth_verdict")
    if stable_truth:
        return None if stable_truth == "cannot_determine" else stable_truth
    decision = item.audit.get("decision")
    verdict = (
        item.settled_verdict
        if decision in VERDICT_ENDORSING_DECISIONS
        else item.audit.get("corrected_verdict")
    )
    return verdict if verdict in ("violation", "no_violation") else None


def audit_confusion(
    rows: list[dict],
    reviews: dict[str, dict],
    audits: dict[str, dict],
    *,
    config: PipelineConfig | None = None,
    exported: dict[str, dict] | None = None,
) -> dict:
    """Precision/recall/F1/accuracy for the JUDGE, scored against audits.

    This is the whole point of auditing live data. An offline eval scores the
    judge against a gold set; live alarms have no gold set, so an auditor's
    confirmation becomes the ground truth and the same confusion matrix falls
    out of it. Deliberately the same `ConfusionCounts` the offline eval uses,
    so the two are read the same way and cannot drift into two definitions of
    precision.

    Only cases the judge actually decided contribute a TP/FP/FN/TN; an
    ESCALATE is counted as `uncertain`, split by what the case turned out to
    be, since "how often did it hedge on real violations" is a different
    question from "how often was it wrong".

    A case whose settled verdict came from a human (an escalation the judge
    declined) still counts: the judge's own answer there was ESCALATE, and
    that is exactly what lands in the uncertain buckets.
    """
    config = config or get_config()
    # ONE ROW PER IMAGE. A decision is recorded against the image, so scoring
    # the crops would count that single human answer once per person and
    # weight the whole matrix by how crowded each image happened to be.
    items = build_audit_cases(
        rows,
        reviews,
        audits,
        config=config,
        exported=exported,
        include_audited=True,
        limit=10**9,
    )

    overall = ConfusionCounts()
    per_hazard: dict[str, ConfusionCounts] = {}
    for item in items:
        # An image whose every crop was struck out has no population left to
        # have been right or wrong about. Out of every denominator here, and
        # counted in its own bucket by `audit_summary` instead -- the same
        # "decided, but not a rate" treatment `box_rejected` gets.
        if item.settled_verdict == NO_VALID_CROPS:
            continue
        truth = audited_ground_truth(item)
        if truth is None:
            continue
        verdict = _OUTCOME_TO_VERDICT.get(item.judge_outcome or "")
        if verdict is None:
            continue
        is_violation = truth == "violation"
        outcome = classify(is_violation, verdict)
        overall.record(outcome, is_violation)
        bucket = per_hazard.setdefault(item.hazard_type or "unknown", ConfusionCounts())
        bucket.record(outcome, is_violation)

    return {
        "overall": _confusion_dict(overall),
        "per_hazard_type": {k: _confusion_dict(v) for k, v in sorted(per_hazard.items())},
        "min_samples": AUDIT_MIN_SAMPLES,
        #: The unit of everything above, on the response rather than in a
        #: reader's head. The block below is in a different one and the two
        #: must never be summed.
        "unit": "images",
        "per_crop": _crop_attribution(items, audits),
    }


def _crop_attribution(cases: list[AuditCase], audits: dict[str, dict]) -> dict:
    """Where the judge's per-crop answers landed, scored against the only
    per-crop answer a human ever gives: keep it, or strike it out.

    ATTRIBUTION, NOT A DERIVED PRECISION. There is no per-crop human ground
    truth. An auditor records ONE verdict for the picture, and
    `training_export.audited_crop_label` is explicit that on a violation image
    only the crop that FIRED inherits the label while every other crop
    inherits nothing. Pushing the image's verdict down onto its crops would
    label every bystander in every violation frame -- up to seven wrong labels
    per right one at the pilot's measured 8.54 people a frame.

    So the question this answers is the one the crops can answer: **did the
    judge fire on a crop the auditor kept, or on one they struck out?**
    `fired_on_struck` is the judge pointing at a box a human says holds
    nothing worth training on.

    ADDITIVE, AND NEVER SUMMED WITH `overall`. That block is in IMAGES and is
    the number the 85% shadow-mode target is read off (`CONTRACT.md`); this
    one is in CROPS and answers a different question. `fired_kept_rate` is the
    only rate here, and only because its denominator -- the crops the judge
    fired on -- is a real population. There is deliberately no accuracy: "the
    judge cleared a crop the auditor kept" is the ordinary case and not an
    error, so an accuracy over these counts would read as a judge failure
    rate.

    A NOTE ON `excluded_crops`: the struck-out crops are IN `n`, because they
    are this block's negative class and holding them out would leave the
    headline question unanswerable. They are reported separately so the
    held-out-looking population is still visible, the treatment
    `cv_vs_judge.incomplete_frames` gets. The accounting closes:
    `excluded_crops` is exactly the three `_on_struck` counts summed, so a
    crop that never entered `n` is never reported as excluded from it either.

    ONLY DECISIONS TAKEN WITH THE STRIP IN FRONT OF THE PERSON COUNT. An
    auditor who was never offered a strike-out did not "keep" anything, and
    scoring them as if they had reads a confident 100% off the fact that the
    feature did not exist yet: measured on six legacy frames carrying a plain
    `{"decision": "agree"}`, this block reported `fired_kept_rate: 1.0` -- and
    at ~8.54 crops a frame, legacy audits alone clear `AUDIT_CROP_MIN_SAMPLES`
    on their own, so the thin-sample gate does not catch it. The marker is the
    `excluded_people` KEY on the entry, which `save_audit` writes only when a
    caller manages strike-outs at all (see its docstring) -- so a legacy
    audit, a box review and a helmet-CV review are all out.
    """
    counts = ConfusionCounts()
    struck_total = 0
    for case in cases:
        # Crops of AUDITED images only: a keep is only a human answer where a
        # human actually looked at the strip.
        if not case.audited or "excluded_people" not in case.audit:
            continue
        _, dropped = included_people(case, audits)
        struck = {id(person) for person, reason in dropped if reason == EXCLUDED_BY_AUDITOR}
        for person in case.people:
            verdict = _OUTCOME_TO_VERDICT.get(person.judge_outcome or "")
            if verdict is None:
                # No judge answer to attribute. Counted nowhere, on either
                # side, so `n` and `excluded_crops` describe one population.
                continue
            auditor_kept = id(person) not in struck
            if not auditor_kept:
                struck_total += 1
            counts.record(classify(auditor_kept, verdict), auditor_kept)

    fired = counts.true_positive + counts.false_positive
    return {
        "unit": "crops",
        "n": counts.total,
        "excluded_crops": struck_total,
        "sample_is_thin": counts.total < AUDIT_CROP_MIN_SAMPLES,
        "min_samples": AUDIT_CROP_MIN_SAMPLES,
        # Named for what they are rather than TP/FP/FN/TN: "cleared a crop
        # the auditor kept" is not a false negative in any sense a reader of
        # that word would take.
        "counts": {
            "fired_on_kept": counts.true_positive,
            "fired_on_struck": counts.false_positive,
            "cleared_on_kept": counts.false_negative,
            "cleared_on_struck": counts.true_negative,
            "escalated": counts.uncertain,
            # BOTH HALVES. `escalated` alone cannot attribute a hedge, so a
            # card carrying it had to rule escalations off below the table
            # with a note that the split was unavailable. The struck half is
            # also what makes the accounting close: the three `_on_struck`
            # counts sum to `excluded_crops` exactly.
            "escalated_on_kept": counts.uncertain_violation,
            "escalated_on_struck": counts.uncertain_no_violation,
        },
        "fired_total": fired,
        "fired_kept_rate": (counts.true_positive / fired) if fired else None,
    }


def shadow_mode_agreement_from_audits(confusion: dict, *, min_samples: int = AUDIT_MIN_SAMPLES) -> dict:
    """The shadow-mode benchmark, measured against audits instead of a gold set.

    Deliberately the SAME definition the gold-set version used -- of the cases
    the judge actually committed to a verdict on, how often did it match the
    human -- with only the source of the human answer changed. Escalations are
    excluded from both numerator and denominator, exactly as before, because
    the judge deferring is not a verdict to agree or disagree with.

    Keeping the definition fixed and swapping only the label source is what
    makes the two numbers comparable at all. It also means this is a headline
    view of `audit_confusion`'s accuracy over decided cases, not a second,
    subtly different statistic reported beside it.

    Why audits rather than gold: offline eval exists to rehearse the live
    feed, and a live alarm has no gold label. Gold stays where a gold label
    genuinely exists -- scoring an eval run against ground truth.

    Returns the same shape the gold-set version did, so the benchmark tile
    reads it unchanged: `{"insufficient_data": True, "matched_count": N}`
    below `min_samples`, otherwise the counts and the rate as well.
    """
    counts = confusion["counts"]
    matched = confusion["decided_total"]
    agreed = counts["true_positive"] + counts["true_negative"]
    if matched < min_samples:
        return {"insufficient_data": True, "matched_count": matched}
    return {
        "insufficient_data": False,
        "matched_count": matched,
        "agreement_count": agreed,
        "agreement_rate": agreed / matched,
    }


def _confusion_dict(counts: ConfusionCounts) -> dict:
    """One confusion matrix, flattened for the API.

    `n` is the audited case count the whole block rests on, so the UI can mark
    a thin sample rather than presenting a rate from three cases as though it
    were a measurement.
    """
    return {
        "unit": "images",
        "n": counts.total,
        "sample_is_thin": counts.total < AUDIT_MIN_SAMPLES,
        "counts": {
            "true_positive": counts.true_positive,
            "false_negative": counts.false_negative,
            "false_positive": counts.false_positive,
            "true_negative": counts.true_negative,
            "uncertain": counts.uncertain,
        },
        "precision": counts.precision,
        "recall": counts.recall,
        "f1": counts.f1,
        "accuracy": counts.accuracy,
        "decided_total": counts.decided_total,
        "uncertain_rate": counts.uncertain_rate,
        # "What really was a violation, and how did the judge answer it" --
        # the distribution that says WHERE the errors are, which a single
        # accuracy number cannot.
        "ground_truth_violation_total": counts.ground_truth_violation_total,
        "ground_truth_no_violation_total": counts.ground_truth_no_violation_total,
        "violation_verdict_distribution": counts.violation_verdict_distribution,
        "no_violation_verdict_distribution": counts.no_violation_verdict_distribution,
    }


def audit_summary(
    rows: list[dict],
    reviews: dict[str, dict],
    audits: dict[str, dict],
    *,
    config: PipelineConfig | None = None,
    exported: dict[str, dict] | None = None,
) -> dict:
    """Counts and the agreement rate, per origin and overall.

    The agreement rate is the point of the whole exercise: for live data it
    is the only available measure of whether the pipeline is right, and it
    is what `rolling_metrics`' 85% shadow-mode target is measured against.
    Reported as `None` rather than 0 when nothing has been audited -- a 0%
    agreement rate and "no data yet" are very different claims.
    """
    config = config or get_config()
    # Counted in IMAGES, the unit a person actually decides, so "12 pending"
    # means twelve decisions to make rather than twelve crops belonging to
    # two pictures.
    everything = build_audit_cases(
        rows,
        reviews,
        audits,
        config=config,
        exported=exported,
        include_audited=True,
        limit=10**9,
    )

    def bucket(items: list[AuditCase]) -> dict:
        audited = [i for i in items if i.audited]
        # PENDING, BUT NOT LISTED. `get_audit_queue` holds a case owed a
        # verdict on one of its crops out of the queue, and this count is what
        # is left of `pending` after it does -- so the badge above the list and
        # the list itself agree. Without it the screen said 115 pending over 86
        # rows on the live corpus, which is a number somebody chases.
        blocked = [i for i in items if i.awaiting_verdict and not i.audited]
        # Settled and audited in one act by the same person -- see
        # SELF_SETTLED_SOURCE. Counted as audited, kept out of the rate.
        self_settled = [i for i in audited if i.audit.get("source") == SELF_SETTLED_SOURCE]
        # Every crop struck out, so there is nothing left the judge could have
        # been right or wrong about -- see NO_VALID_CROPS.
        no_valid_crops = [i for i in audited if i.settled_verdict == NO_VALID_CROPS]
        # AN ENDORSEMENT OF A VERDICT THAT DOES NOT EXIST. A case whose only
        # remaining crops are awaiting a human has no settled verdict, so
        # "agree" agreed with nothing. `audit_confusion` already drops these
        # (there is no ground truth to score), and counting them here as
        # agreement put 100% agreement on the page beside n=0.
        # `record_audit` now refuses to write one; this covers what is already
        # in the sidecar.
        no_verdict = [
            i
            for i in audited
            if i.settled_verdict is None
            and i.audit.get("decision") in VERDICT_ENDORSING_DECISIONS
            and i not in self_settled
        ]
        checked = [
            i
            for i in audited
            if i not in self_settled and i not in no_valid_crops and i not in no_verdict
        ]
        agreed = [i for i in checked if i.audit.get("decision") == "agree"]
        box_rejected = [i for i in checked if i.audit.get("decision") == "box_wrong"]
        return {
            "auditable": len(items),
            "audited": len(audited),
            "pending": len(items) - len(audited),
            # Of that `pending`, the ones a human owes a verdict on before an
            # auditor can be asked anything. `pending - blocked` is the number
            # of cases the audit queue actually lists.
            "blocked": len(blocked),
            # Decisions retired because the verdict they were about changed.
            # Counted separately so a drop in `audited` is explicable rather
            # than looking like lost work.
            "stale": len([i for i in items if i.stale_audit]),
            "agreed": len(agreed),
            # Its own bucket, not folded into either side. It endorses the
            # verdict (so `audit_confusion` scores it as agreement) while
            # rejecting the label, so counting it as agreement here would
            # hide a bad box and counting it as disagreement would blame the
            # judge for a verdict it got right.
            "box_rejected": len(box_rejected),
            # A reviewer resolving an escalation settles the case and audits
            # it in one act. Its own bucket, and out of the rate below: a
            # person agreeing with the verdict they themselves just gave is
            # tautological, and folding those in would drift every rate
            # towards 100% as reviewing continued.
            "self_settled": len(self_settled),
            # An image whose crops were ALL struck out. Its own bucket beside
            # the two above, and out of the rate for the same reason: it is
            # decided -- a human has finished with it -- but there is no
            # verdict left to have agreed or disagreed with.
            "no_valid_crops": len(no_valid_crops),
            # Audited, but endorsing a verdict the case does not have. Its own
            # bucket for the same reason as the two above: decided, and not
            # part of any rate.
            "no_verdict": len(no_verdict),
            "disagreed": len(checked) - len(agreed) - len(box_rejected),
            "agreement_rate": (len(agreed) / len(checked)) if checked else None,
            # Whether the rate rests on enough cases to mean anything.
            "sample_is_thin": len(checked) < AUDIT_MIN_SAMPLES,
        }

    return {
        "limit": AUDIT_LIMIT,
        "min_samples": AUDIT_MIN_SAMPLES,
        "overall": bucket(everything),
        "pipeline_settled": bucket([i for i in everything if i.settled_by == "pipeline"]),
        "human_settled": bucket([i for i in everything if i.settled_by == "human"]),
        "exported": bucket([i for i in everything if i.exported]),
        "decisions": AUDIT_DECISIONS,
        "corrected_verdicts": CORRECTED_VERDICTS,
    }


# --- the helmet-CV comparison ---------------------------------------------


def reviewer_verdict(entry: dict) -> str | None:
    """The verdict a reviewer actually stated, recovered from an audit entry.

    Reviews on the CV screen are recorded BLIND: the reviewer is shown the
    frame and states what it is, never "do you agree with X". The server then
    encodes that as agree/disagree against the settled verdict, because that
    is the vocabulary `audited_ground_truth` and every existing metric speak.
    This reverses the encoding, for a caller that wants the human's own words
    back without inferring them.
    """
    if entry.get("ground_truth_verdict"):
        return entry["ground_truth_verdict"]
    if entry.get("decision") in VERDICT_ENDORSING_DECISIONS:
        return entry.get("audited_verdict")
    return entry.get("corrected_verdict")


#: The frame role production CV actually fired on. Only these frames carry a
#: CV claim; see `records.FRAME_ROLE_ALARM`.
FRAME_ROLE_ALARM = "alarm"


def ground_truth_key(cv: dict) -> str | None:
    """THE unit a human decision is keyed by, in one place on purpose.

    A verdict is recorded against the FRAME, not against each person in it.
    The human is asked "is this frame a violation?", the judge's per-crop
    answers are OR'd up to the same unit, and the crop that fired is carried
    alongside as attribution rather than scored separately.

    THE SAVING, ON THE LIVE CORPUS: **76.2% less reviewing** -- 186 verdicts
    instead of 781, at the measured 4.2 people per alarm frame (6 Sep). It
    falls to **68.2%** once each open crop is answered on its own as well
    (186 + 62 = 248 decisions), which is where per-crop review takes it.

    The 88% previously quoted here is `pilot-v3`'s, at 8.54 people a frame:
    316 verdicts instead of 2699. It is not wrong, it is a different corpus --
    the saving is a function of how crowded the frames are, and the live feed's
    are half as crowded as the pilot's. This figure had already been wrong
    twice (1.52, then 3.86) by quoting a superseded run, which is why the
    docstring says to check the run. Both are kept so the next reader can see
    what makes them differ.

    Isolated here because it is the pivot of the whole measurement. If the
    unit ever has to become the person again, this function is the change.
    """
    from .helmet_cv.records import frame_stem

    frame = cv.get("frame_name")
    return frame_stem(frame) if frame else None


def cv_verdict(item: AuditItem) -> str | None:
    """What production CV claimed about this frame.

    There is no gate to read any more, and that is the point: this corpus
    exists BECAUSE production CV fired. So CV's claim on an alarm frame is
    always "violation" -- `"TP"` here means "CV asserted a violation", and the
    human's answer is what turns it into a true or false one.

    Which means the CV arm has PRECISION AND NOTHING ELSE. CV never asserted
    a negative anywhere in this corpus, so there is no denominator for recall
    and no true negative to count. Reporting a recall for it would be
    inventing one.

    `None` for a clip frame. Production made no claim about those -- they were
    decoded out of the clip by us -- so scoring them against CV would credit
    or blame it for an answer it never gave. They feed the VLM arm alone,
    which is what lets the VLM have recall while CV does not.
    """
    if not item.cv or item.cv.get("frame_role") != FRAME_ROLE_ALARM:
        return None
    return "TP"


#: THE ONE PLACE THIS SENTENCE EXISTS. The UI renders it verbatim, so it is the
#: authority on what these numbers cover, and it has to be true of the run that
#: ships rather than of the design that produced it.
#:
#: It previously said "the sample is drawn without reference to the gate, so
#: cleared candidates are audited too" — a description of the gate-blind
#: sampler over a cascade that was deleted. There is no gate, and the pilot run
#: scores alarm frames only.
#:
#: It also covered only GENERALISATION (what footage is absent) and said nothing
#: about LABEL CONTAMINATION, which is the live risk: every frame in the queue is
#: one production CV fired on, so a reviewer learns that at their first reveal
#: and every subsequent verdict carries acquiescence bias — on the very
#: decisions that are ground truth for both arms. That is a different failure
#: from coverage and it belongs next to the numbers, not in a handoff.
#: KEPT SHORT ON PURPOSE, because it is rendered verbatim next to the numbers
#: and on-screen copy is terse. Three claims have to survive any edit: the
#: population is CV-positive frames only, so CV has precision and no recall
#: over all footage; the VLM is asked about every frame, so it has one; and
#: both arms are agreement-after-seeing. The closing restatement ("not
#: independent measurements of the models") said the third one twice.
CV_POPULATION_CAVEAT = (
    "Frames production CV fired on. CV asserted a violation on all of them and "
    "cleared none, so precision is the only figure it can have; footage that never "
    "raised an alarm is absent, so nothing here measures its recall. The VLM is "
    "asked about every frame, including ones it clears, so it has a recall. "
    "READ BOTH ARMS AS AGREEMENT-AFTER-SEEING: the reviewer is shown CV's record "
    "and the judge's verdict before deciding, so deferring to either inflates it."
)


def _cv_confusion_dict(counts: ConfusionCounts) -> dict:
    """One confusion matrix over the CV corpus, flattened.

    Reports precision, recall, F1 and accuracy -- the four the pipeline spec
    asks for, for both arms, off the same audit ground truth.

    Kept separate from `_confusion_dict` only so the CV corpus's own caveat
    travels with its numbers (see `CV_POPULATION_CAVEAT`); the counts and the
    definitions are `ConfusionCounts`', unchanged, so the CV arm, the VLM arm
    and the live-alarm audit cannot drift into three definitions of
    precision.
    """
    return {
        "n": counts.total,
        "sample_is_thin": counts.total < AUDIT_MIN_SAMPLES,
        "counts": {
            "true_positive": counts.true_positive,
            "false_negative": counts.false_negative,
            "false_positive": counts.false_positive,
            "true_negative": counts.true_negative,
            "uncertain": counts.uncertain,
        },
        "precision": counts.precision,
        "recall": counts.recall,
        "f1": counts.f1,
        "accuracy": counts.accuracy,
        "decided_total": counts.decided_total,
        "ground_truth_violation_total": counts.ground_truth_violation_total,
        "ground_truth_no_violation_total": counts.ground_truth_no_violation_total,
    }


def cv_audit_items(
    rows: list[dict], audits: dict[str, dict], cv_run_id: str | None = None
) -> list[AuditItem]:
    """One `AuditItem` per judged CV candidate, newest row per candidate.

    Deliberately NOT `build_audit_queue`. That function encodes what
    "settled" means for a LIVE alarm -- an ESCALATE is not settled until a
    human resolves it in the escalation queue, and an unexportable case is
    dropped. Neither applies here: a CV candidate is a frame from a finished
    clip, there is no escalation queue for it, and an ESCALATE is a perfectly
    good judge answer to score (it lands in the `uncertain` bucket, exactly as
    it does for live alarms). Routing these through the live rules would
    silently drop every escalated candidate from the comparison -- which is a
    large share of them, and invisible in the output.

    What IS shared is everything that matters for scoring:
    `audited_ground_truth` and `metrics.classify` are used unchanged, so one
    reviewer's answer scores the gate and the judge by the same definition of
    right.
    """
    newest: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict) or not is_cv_row(row):
            continue
        cv = row.get("cv") if isinstance(row.get("cv"), dict) else {}
        row_run_id = cv.get("cv_run_id") or "legacy-unscoped"
        if cv_run_id is not None and row_run_id != cv_run_id:
            continue
        alarm_id = cv.get("execution_id") or row.get("alarm_id")
        if not alarm_id or not isinstance(alarm_id, str):
            continue
        previous = newest.get(alarm_id)
        if previous is None or _sort_key(row) >= _sort_key(previous):
            newest[alarm_id] = row

    items: list[AuditItem] = []
    for alarm_id, row in newest.items():
        # No reviews dict: a CV candidate has no escalation review, so the
        # settled verdict is the judge's own answer and nothing else.
        settled_verdict = _settled_verdict(row, {})
        # Re-derived for THIS loop's row, and named apart from the filtering
        # loop's `cv` above on purpose: sharing that name reads as a
        # redundant assignment, and deleting it as one silently scores every
        # candidate against the last row of the previous loop.
        row_cv = row.get("cv") if isinstance(row.get("cv"), dict) else {}
        # The FRAME, not the person -- see `ground_truth_key`. Every crop of a
        # frame reads the same human decision, which is what lets the judge's
        # per-crop answers be OR'd up to the unit the human actually labelled.
        ground_truth_id = ground_truth_key(row_cv) or row_cv.get("candidate_id") or alarm_id
        entry = _entry_for(audits, ground_truth_id)
        stale = False if entry.get("ground_truth_verdict") else _audit_is_stale(entry, settled_verdict)
        items.append(
            AuditItem(
                alarm_id=alarm_id,
                settled_by="pipeline",
                hazard_type=row.get("hazard_type") or "",
                timestamp=row.get("timestamp"),
                image_key=row.get("image_key"),
                crop_box=_box_list(row.get("crop_box")),
                helmet_box=_box_list(row.get("helmet_box")),
                person_box=_box_list(row.get("person_box")),
                settled_verdict=settled_verdict,
                judge_outcome=row.get("outcome"),
                judge_reason=row.get("reason_code"),
                judge_reasoning=row.get("reasoning"),
                judge_model=row.get("judge_model"),
                helmet_box_source=row.get("helmet_box_source"),
                helmet_box_confidence=row.get("helmet_box_confidence"),
                reviewed_by=entry.get("audited_by"),
                audited=_is_audited(entry) and not stale,
                stale_audit=stale,
                audit=entry,
                cv=row.get("cv") if isinstance(row.get("cv"), dict) else {},
                lineage=row.get("lineage") if isinstance(row.get("lineage"), dict) else {},
            )
        )
    items.sort(key=lambda item: (str(item.timestamp or ""), item.alarm_id))
    return items


def cv_confusion(rows: list[dict], audits: dict[str, dict], cv_run_id: str | None = None) -> dict:
    """PRECISION for production CV, scored against the same human decisions the
    VLM arm is scored against.

    Reuses `audited_ground_truth` unchanged and `metrics.classify` unchanged,
    on purpose. The whole value of putting these rows in this log is that one
    reviewer's answer scores both models; two scoring functions would be two
    definitions of "right" and the comparison would be meaningless.

    PRECISION AND NOTHING ELSE, and read the recall field as meaningless here.
    Every alarm in this corpus exists because production CV fired, so CV
    asserted a violation on all of them and asserted a negative on none. There
    is no false negative to count and no true negative either. What this
    measures is exactly one thing: **of the alarms production raised, what
    share were real?** That is the question the corpus can answer.

    `per_gate_reason` is gone with the cascade. There is no rung to attribute
    an error to any more -- production's own reasoning is not in the corpus,
    only the fact that it fired.

    Scored on alarm frames only, since those are the only frames production
    claimed anything about -- see `cv_verdict`.
    """
    items = cv_audit_items(rows, audits, cv_run_id)

    return {
        # ONCE PER FRAME. Scoring per crop counted one human verdict once per
        # person in the frame -- see `group_by_frame`.
        "overall": _arm_confusion(items, cv_verdict),
        "min_samples": AUDIT_MIN_SAMPLES,
        # What these numbers are numbers about. Travels with them so a rate
        # cannot be quoted as production CV's recall over all footage, which
        # is the one thing this corpus cannot measure.
        "population": CV_POPULATION_CAVEAT,
    }


def cv_vs_judge(rows: list[dict], audits: dict[str, dict], cv_run_id: str | None = None) -> dict:
    """What the VLM would have done to the alarms production actually raised.

    Alarm frames only, because those are the only frames production CV
    claimed anything about. CV's claim there is always "violation", which
    changes the shape of this comparison from the cascade version:

    * `suppressed` -- CV raised it and the VLM cleared it. **This is the
      product.** Every one of these is an alarm an operator would not have
      had to look at, and the human verdict on the same frame says whether
      the VLM was right to drop it.
    * `confirmed` -- both called it a violation.
    * `escalated` -- the VLM declined to answer. Not a disagreement; reported
      separately, the same treatment `shadow_mode_agreement_from_audits`
      gives it, which is what keeps the two numbers comparable.

    There is deliberately no `judge_only` cell. It would mean "CV cleared it
    and the VLM held it", and CV cleared nothing in this corpus -- the corpus
    is the alarms it raised. A cell that can never be populated is worse than
    an absent one, because a reader takes its zero for a finding.
    """
    alarm_frames = [
        frame
        for frame in vlm_frame_verdicts(cv_audit_items(rows, audits, cv_run_id), audits)
        if frame.frame_role == FRAME_ROLE_ALARM
    ]
    # A frame the judge never finished cannot be OR'd. Reported, not dropped
    # in silence: an interrupted live feed leaves these behind and the reader
    # needs to see that the population is short.
    frames = [frame for frame in alarm_frames if frame.complete]
    incomplete = len(alarm_frames) - len(frames)

    confirmed = suppressed = escalated = 0
    for frame in frames:
        if frame.verdict is None:
            continue
        if frame.verdict == "uncertain":
            escalated += 1
        elif frame.verdict == "TP":
            confirmed += 1
        else:
            suppressed += 1

    decided = confirmed + suppressed
    return {
        "decided_total": decided,
        "confirmed": confirmed,
        "suppressed": suppressed,
        "judge_escalated": escalated,
        "suppression_rate": (suppressed / decided) if decided else None,
        "incomplete_frames": incomplete,
        "population": CV_POPULATION_CAVEAT,
    }


def _judge_verdict(item: AuditItem) -> str | None:
    """The judge's answer about ONE crop, as a verdictual string."""
    return _OUTCOME_TO_VERDICT.get(item.judge_outcome or "")


def group_by_frame(items: list[AuditItem]) -> dict[str, list[AuditItem]]:
    """`cv_audit_items` output, grouped into the unit a human actually labelled.

    **Every arm must go through this.** The unit of truth is the frame
    (`ground_truth_key`) while the unit of a row is one crop, so anything that
    iterates `cv_audit_items` directly scores one human verdict once per person
    in the frame. That inflates `n`, and worse, weights the result by how
    crowded each frame happened to be: measured on `r1920-p025`, three of the
    eighteen alarm frames carry 46% of the CV arm's headline precision, and a
    single verdict on the nine-crop frame moves it about twenty points.

    Insertion order is preserved, so `[0]` is the frame's first crop and is a
    stable representative for anything frame-level (the human decision, the
    frame role, the provenance) rather than an arbitrary one.
    """
    grouped: dict[str, list[AuditItem]] = {}
    for item in items:
        key = ground_truth_key(item.cv or {})
        if key:
            grouped.setdefault(key, []).append(item)
    return grouped


def _arm_confusion(items: list[AuditItem], verdict_of) -> dict:
    """One arm's confusion matrix, scored ONCE PER FRAME.

    `verdict_of` takes the frame's representative crop. Restored after being
    deleted in the two-arm rewrite -- `cv_paired_intersection` still called it,
    so `/helmet-cv/summary?compare_run_id=` was a 500.
    """
    counts = ConfusionCounts()
    for crops in group_by_frame(items).values():
        item = crops[0]
        truth = audited_ground_truth(item)
        if truth is None:
            continue
        verdict = verdict_of(item)
        if verdict is None:
            continue
        is_violation = truth == "violation"
        counts.record(classify(is_violation, verdict), is_violation)
    return _cv_confusion_dict(counts)


@dataclass(frozen=True)
class FrameVerdict:
    """The judge's answer about one FRAME, OR'd up from its crops.

    `fired_candidates` is the attribution: which crops the judge called a
    violation. It is carried, not scored -- the human's verdict is about the
    frame, so a frame both of them call a violation counts as agreement even
    if the judge pointed at the wrong worker. That gap is real and is why the
    crops are judged one call each: with per-crop calls the attribution is at
    least trustworthy, whereas a batched call can swap content between crops
    and mislabel which worker fired while the frame answer stays right.
    """

    ground_truth_id: str
    frame_name: str
    frame_role: str
    verdict: str | None
    fired_candidates: list[str]
    crops: int
    item: AuditItem
    #: How many people the detector found in this frame, and whether a judged
    #: crop came back for each of them. An OR over a SHORT set reads as "no
    #: violation" and is indistinguishable from a complete frame afterwards,
    #: so an incomplete frame is carried and then excluded from the scored
    #: population rather than silently averaged in. Legacy rows carry no
    #: `people_in_frame`; those count as complete, since 0 is "unrecorded".
    people_in_frame: int = 0
    complete: bool = True
    #: Crops an auditor struck out. They are not in `crops`, not in the OR
    #: and not in `fired_candidates`; carried so the held-out population is
    #: visible rather than silently absent. `crops == 0` with this above zero
    #: is the `NO_VALID_CROPS` frame -- every crop struck, no verdict left.
    excluded_crops: int = 0


def vlm_frame_verdicts(items: list[AuditItem], audits: dict[str, dict]) -> list[FrameVerdict]:
    """One verdict per frame, OR'd over the crops an auditor KEPT.

    ANY crop the judge calls a violation makes the frame a violation. That is
    the production question -- "should this frame have raised an alarm?" -- and
    a frame is not cleared just because three of the four people in it are
    wearing helmets.

    TP > uncertain > FP, the same order as `case_outcome.judge_outcome`,
    `rolling_metrics._IMAGE_OUTCOMES` and
    `routers/escalations._CASE_OUTCOMES`. A violation still wins outright --
    the judge committed to it, and no hedge elsewhere in the frame can lower
    it. A hedge beats a clearance, because the judge declining to answer
    about somebody is not the judge clearing the frame; four screens read
    these four sites, and one of them answering differently is one frame with
    two answers on it.

    `audits` is required rather than defaulted: this is one of the five sites
    the OR is written at, and a site that could silently score a struck-out
    crop is exactly the drift `included_people` exists to prevent.
    """
    verdicts: list[FrameVerdict] = []
    for key, crop_items in group_by_frame(items).items():
        crops, excluded = included_people(crop_items, audits, case_id=key)
        answers = [(_judge_verdict(crop), crop) for crop in crops]
        # The VERDICT comes from the answers; `fired` is only the attribution.
        # Deriving the verdict from `fired` meant a violation crop with no
        # `candidate_id` flipped the whole frame to "no violation".
        any_violation = any(answer == "TP" for answer, _ in answers)
        fired = [
            crop.cv.get("candidate_id")
            for answer, crop in answers
            if answer == "TP" and crop.cv.get("candidate_id")
        ]
        if any_violation:
            verdict = "TP"
        elif any(answer == "uncertain" for answer, _ in answers):
            verdict = "uncertain"
        elif any(answer == "FP" for answer, _ in answers):
            verdict = "FP"
        else:
            verdict = None
        # The frame's own facts come from every crop of it, struck out or
        # not: the detector found those people whatever an auditor thinks of
        # the boxes, so holding a struck crop out here would make the frame
        # read as SHORT of its people.
        first = crop_items[0]
        people = max(int(crop.cv.get("people_in_frame") or 0) for crop in crop_items)
        # A gated crop is accounted for, not missing: the first pass found
        # nobody in it, so there was no helmet question to ask. Only an
        # UNEXPLAINED shortfall makes the frame incomplete.
        gated = max(int(crop.cv.get("people_gated") or 0) for crop in crop_items)
        verdicts.append(
            FrameVerdict(
                ground_truth_id=key,
                frame_name=first.cv.get("frame_name") or "",
                frame_role=first.cv.get("frame_role") or "",
                verdict=verdict,
                fired_candidates=fired,
                crops=len(crops),
                excluded_crops=len(excluded),
                people_in_frame=people,
                complete=len(crop_items) + gated >= people,
                # Any crop of the frame carries the frame's human decision,
                # since the decision is keyed by frame. The first is as good
                # as any and avoids a second lookup.
                item=first,
            )
        )
    return verdicts


def vlm_confusion(rows: list[dict], audits: dict[str, dict], cv_run_id: str | None = None) -> dict:
    """The VLM arm: precision, recall, F1 and accuracy over FRAMES.

    Full metrics, unlike the CV arm, and the asymmetry is not an oversight.
    The VLM is asked about every frame in the sample, including frames whose
    crops it clears, so a human verdict of "violation" on a frame the VLM
    cleared is a genuine false negative and recall has a denominator. CV
    asserted a violation on every alarm in the corpus and never asserted a
    negative anywhere, so it has precision only.

    One ceiling to keep in mind reading these numbers: the detector supplies
    every crop, so a worker it never found is a worker the judge was never
    shown, and this arm cannot see that miss. See `records`'s module
    docstring.
    """
    counts = ConfusionCounts()
    for frame in vlm_frame_verdicts(cv_audit_items(rows, audits, cv_run_id), audits):
        truth = audited_ground_truth(frame.item)
        if truth is None or frame.verdict is None or not frame.complete:
            continue
        is_violation = truth == "violation"
        counts.record(classify(is_violation, frame.verdict), is_violation)
    return _cv_confusion_dict(counts)


def cv_frame_count(rows: list[dict], audits: dict[str, dict], cv_run_id: str | None = None) -> int:
    """How many DISTINCT frames carry a human decision.

    One number for "frames reviewed", because the ledger and the summary were
    each computing their own and disagreeing by the crowding factor.
    """
    return sum(
        1 for crops in group_by_frame(cv_audit_items(rows, audits, cv_run_id)).values()
        if crops[0].audited
    )


def cv_paired_intersection(
    rows: list[dict], audits: dict[str, dict], left_run_id: str, right_run_id: str
) -> dict:
    """Metrics for two runs over only the FRAMES present in both.

    Keyed by `ground_truth_key`, not by the record's `ground_truth_id` field.
    That field is the per-candidate id, and `records`' own docstring says the
    person index "means nothing across runs -- there is no tracker", so pairing
    on it matched unrelated people between runs. The frame is the only identity
    that survives a re-run.
    """
    left = group_by_frame(cv_audit_items(rows, audits, left_run_id))
    right = group_by_frame(cv_audit_items(rows, audits, right_run_id))
    shared = sorted(set(left) & set(right))
    return {
        "ground_truth_ids": shared,
        "n": len(shared),
        "left": _arm_confusion([left[key][0] for key in shared], cv_verdict),
        "right": _arm_confusion([right[key][0] for key in shared], cv_verdict),
    }


def arm_metrics(confusion: dict) -> dict:
    """One arm's four headline metrics, each `None` when its own denominator
    is empty.

    Every one of `ConfusionCounts`' rate properties returns 0.0 rather than
    raising on an empty denominator, and 0.0 on the wire reads as "this model
    got everything wrong" rather than "nobody has audited anything yet".
    Those are opposite claims, so each rate is nulled against the denominator
    it actually rests on, not against one shared count:

      * precision needs TP+FP -- cases the model called a violation
      * recall needs TP+FN -- cases that really were violations
      * accuracy needs `decided_total` -- everything except the judge's own
        hedges, which is the same denominator `audit_confusion` uses
      * F1 needs both of its parents, so it is null if either is

    `n` is the audited candidate count the whole block rests on, so a caller
    can mark a thin sample instead of presenting a rate from three cases as
    though it were a measurement.
    """
    counts = confusion["counts"]
    precision_denominator = counts["true_positive"] + counts["false_positive"]
    recall_denominator = counts["true_positive"] + counts["false_negative"]

    precision = confusion["precision"] if precision_denominator else None
    recall = confusion["recall"] if recall_denominator else None
    return {
        "n": confusion["n"],
        "sample_is_thin": confusion["sample_is_thin"],
        "counts": dict(counts),
        "precision": precision,
        "recall": recall,
        "f1": confusion["f1"] if (precision is not None and recall is not None) else None,
        "accuracy": confusion["accuracy"] if confusion["decided_total"] else None,
    }
