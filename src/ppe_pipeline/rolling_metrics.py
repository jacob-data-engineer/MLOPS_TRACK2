"""Metrics for the `/metrics` screen, computed from Track1's local
audit-log JSON file (`track1_adapter.run_track1_judge_inference`'s audit
rows) and, for shadow-mode agreement, test_proj's own gold-set annotations
sidecar.

These are the concrete measurables named in
`Project Documents/JP_TimeLine.pdf`'s Track 1 benchmark section: shadow-
mode agreement rate (target >= 85%, agent vs. human) and label output
volume/week (target >= 400/week), plus the auto-annotation gate outcome
#6 relies on (share of rows resolved CONFIRM_TP/CONFIRM_FP by
`track1_adapter._resolve_outcome` rather than escalated).

Every function here degrades to zeroed/empty results rather than raising
when the audit log (or gold-set sidecar) doesn't exist yet or is empty —
expected, since neither file exists until the revamped Run tab has
actually been used at least once.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import storage
from .eval import gold_set

DEFAULT_AUDIT_LOG_PATH = "eval/track1_audit_log.json"

# The hazard-type keys the per-hazard breakdown is pre-seeded with, so the
# UI shows every type consistently even at 0. These mirror Track1's
# `hazard_validation.models.HazardType` values, which this module used to
# import directly — inlined so JP_UI carries no dependency on the Track1
# repo (see the README's "Related repos").
#
# Deliberately NOT `track1_adapter.VIOLATION_TYPE_TO_HAZARD_LABEL.values()`,
# which is these four plus "glove": glove has never had a Track1 HazardType,
# so glove rows have never appeared in this breakdown. Adding it here would
# silently change what the /metrics screen reports.
TRACK1_HAZARD_TYPES = (
    "no_helmet",
    "under_suspended_cargo",
    "hatch_fall_zone",
    "too_close_to_vehicle",
)
DEFAULT_GOLD_ANNOTATIONS_PATH = "eval/gold_annotations.json"

_OUTCOMES = ("CONFIRM_TP", "CONFIRM_FP", "ESCALATE")

#: WHY THE ZERO IS NOT A RESULT, on screen, in as few words as it takes.
#: The pass this counts is Track1's periodic FN-coverage sampling (outcome #7,
#: FLAG_MISS); naming it bought the reader nothing, and the only thing they
#: can act on is that the number is not a measurement.
FLAG_MISS_NOTE = (
    "Missed-violation sampling is not built yet, so this is always 0. "
    "It does not measure missed violations."
)


def _read_json_list(path: str | Path, label: str) -> list:
    """Read a JSON array via `ppe_pipeline`'s storage abstraction,
    treating a corrupt/malformed file the same as a missing one (empty
    list) rather than raising — `LocalStorage.read_json`'s `default=`
    only covers the file-doesn't-exist case, not "file exists but its
    content is invalid JSON" (or valid JSON but not a list), so both are
    guarded here explicitly.

    Emits a `RuntimeWarning` rather than staying silent about it: corrupt
    data quietly treated as "no data yet" is itself worth surfacing to
    whoever's running this, even though it doesn't crash the caller.
    """
    try:
        rows, _ = storage.get_storage().read_json(Path(path).as_posix(), default=[])
    except json.JSONDecodeError:
        warnings.warn(f"{label} at {path!s} contains invalid JSON — treating as empty.", RuntimeWarning, stacklevel=2)
        return []
    if not isinstance(rows, list):
        warnings.warn(f"{label} at {path!s} is not a JSON array — treating as empty.", RuntimeWarning, stacklevel=2)
        return []
    return rows


def _read_audit_log_rows(audit_log_path: str | Path) -> list[dict]:
    return _read_json_list(audit_log_path, "audit log")


#: An image's outcome from its crops', in the order a tie is broken. The same
#: order as `audit.case_outcome`, `live_feed.frame_verdict` and
#: `routers/escalations._CASE_OUTCOMES`: one question -- what did the judge
#: conclude about this image -- and it must have one answer wherever it is
#: asked. A hedge outranks a clearance because an image the judge declined to
#: answer about is not one it cleared.
_IMAGE_OUTCOMES = ("CONFIRM_TP", "ESCALATE", "CONFIRM_FP")


#: Reason codes for a row that settled with nobody in the box. Its box is not
#: a `person` label, whichever of the two produced it.
_NO_PERSON_REASONS = ("no_person_present", "no_person_detected")


def _image_key(row: dict) -> str:
    """The IMAGE a row belongs to. An ordinary alarm is its own image."""
    from .audit import ground_truth_key

    # `isinstance`, not `or {}`: a hand-edited row can carry `cv` as a string
    # or a list, and `or {}` only catches a falsy one. `ground_truth_key` then
    # calls `.get` on it and one bad row 500s the whole metrics page, hiding
    # every good one. The collapse in `routers/escalations` guards the same
    # way, for the same reason.
    cv = row.get("cv")
    return str(ground_truth_key(cv if isinstance(cv, dict) else {}) or row.get("alarm_id") or "")


def _images(rows: list[dict], audits: dict[str, dict]) -> list[dict]:
    """One row per IMAGE, standing for the crops of it an auditor KEPT.

    A helmet-CV row is one person in one frame; an ordinary alarm row is
    already its own image and passes through untouched.

    `audits` is required, not defaulted: this is one of the five sites the OR
    is written at, and the population it runs over is `audit.included_people`'
    -- a struck-out crop must not decide an image here while the audit screen
    and the escalation queue have already dropped it.
    """
    from .audit import NO_VALID_CROPS, included_people, is_cv_row
    from .escalations import AWAITING_VERDICT_CROPS, newest_per_crop, awaiting_verdict_crops

    grouped: dict[str, list[dict]] = {}
    singles: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if not is_cv_row(row):
            singles.append(row)
            continue
        key = _image_key(row)
        if not key:
            continue
        grouped.setdefault(str(key), []).append(row)

    collapsed = []
    for crops in grouped.values():
        kept, _ = included_people(crops, audits)
        outcomes = {crop.get("outcome") for crop in kept}
        outcome = next((o for o in _IMAGE_OUTCOMES if o in outcomes), None)
        # An image with every crop struck out keeps a representative row so it
        # still counts as an image, and carries no outcome -- it is in none of
        # the outcome buckets and out of the auto-annotation rate.
        deciding = next(
            (c for c in kept if c.get("outcome") == outcome), (kept or crops)[0]
        )
        collapsed.append(
            {
                **deciding,
                "outcome": outcome,
                NO_VALID_CROPS: not kept,
                # WHICH KEPT CROPS THE JUDGE HEDGED ON, carried because the
                # outcome above cannot say: TP outranks ESCALATE, so a frame
                # the judge fired on AND hedged about reads CONFIRM_TP.
                # `auto_annotation_rate` is a claim about how much of the
                # corpus needed no human, and that frame did.
                AWAITING_VERDICT_CROPS: awaiting_verdict_crops(kept),
            }
        )

    # AND THE UNGROUPED ROWS GO THROUGH THE SAME FILTER. A row is grouped only
    # when it is a helmet-CV row, but a row can carry a `cv` block -- and so a
    # candidate id, and so a strike-out -- without carrying `row_kind`, and
    # `audit.case_key` groups on the `cv` block alone. Passing these through
    # unfiltered made the audit screen honour a strike-out that this page
    # ignored. A group of one that is struck out keeps its place in the image
    # count and loses its outcome, exactly as a struck group does.
    for row in singles:
        kept, _ = included_people([row], audits)
        collapsed.append(
            {**row, AWAITING_VERDICT_CROPS: awaiting_verdict_crops(kept)}
            if kept
            else {**row, "outcome": None, NO_VALID_CROPS: True, AWAITING_VERDICT_CROPS: []}
        )
    return collapsed


def compute_track1_metrics(
    audit_log_path: str | Path = DEFAULT_AUDIT_LOG_PATH,
    audits_path: str | Path | None = None,
) -> dict:
    """Aggregate metrics over every row in `audit_log_path`.

    COUNTED IN IMAGES, not rows. A helmet-CV row is one PERSON, and one
    alarm frame holds several, so counting rows reported a run of 31 images
    as "3 violation, 73 no violation, 8 escalate" -- 84, the crop count. It
    read as though a reset had not cleared anything, and it weighted the whole
    distribution by how crowded each frame happened to be. Every other surface
    in the app already counts images (`audit.build_audit_cases`, the audit
    summary, the escalation queues) for exactly that reason.

    An image's outcome is the OR of its crops', the same precedence used by
    `live_feed.frame_verdict` and `audit.case_outcome`: any person in
    violation makes the image a violation, and a person the judge declined to
    answer about leaves it hedged rather than cleared. Over the crops an
    auditor KEPT -- `audits_path` is the strike-out sidecar, and defaults to
    the one `audit.load_audits` reads.

    `label_volume_this_week` is the exception and stays in ROWS, because a
    label IS one person's crop -- that is the unit the 400-a-week target
    counts, and the unit the training set is built from.

    THE UNIT IS ROWS; THE POPULATION IS STILL THE KEPT CROPS. Every count in
    that block -- the volume, the distinct images, the person boxes and the
    helmet boxes -- runs over `audit.included_people`, because it is claiming
    what was DELIVERED and a struck-out crop is retracted from both datasets
    (`training_export._is_excluded`, and the `struck` skip in
    `build_frame_export`). Counting one would report a label the export had
    pulled. The direction is over-report and it grows with how much the
    strike-out feature is used, so it reads correct until auditors start using
    it -- and `MetricsPage` grades this figure against the 400-a-week target.
    If you add a count to that loop, it belongs on the same population.

    Returns a dict with:
    - `total_alarms`: image count.
    - `outcome_counts`: dict of CONFIRM_TP/CONFIRM_FP/ESCALATE -> count, per
      image.
    - `auto_annotation_rate`: share of images the pipeline needed no human
      for, or `0.0` if there are no rows. THE NUMERATOR IS IMAGES WITH NO
      UNANSWERED HEDGE IN THEM, counted off the crops rather than read off
      the image's outcome: the outcome is an OR that puts CONFIRM_TP above
      ESCALATE, so a frame the judge fired on AND hedged about reads
      CONFIRM_TP while a reviewer is queued to answer for it. Over images
      that HAVE an answer: one whose every crop an auditor struck out is in
      neither the numerator nor the denominator, and is reported as
      `no_valid_crops`. Still derived from what the run recorded (not
      re-derived from a hardcoded confidence threshold) so it can never
      drift out of sync with whatever confirm/escalate threshold
      `_resolve_outcome` actually applied when the row was written.
    - `per_hazard_type`: dict of hazard type value -> {"total":
      int, "outcome_counts": {...}}, one entry per `TRACK1_HAZARD_TYPES`
      entry (present even at 0, so the UI can show every type
      consistently).
    - `label_volume_this_week`: rows whose `timestamp` falls within the
      last 7 days, MINUS the crops an auditor struck out. One per person
      crop, which is what the >=400/week target counts and is left
      unchanged so the benchmark keeps its meaning.
    - `label_volume_breakdown`: that same week, over the same kept crops,
      split three ways -- `images` (distinct frames), `person` (boxes that
      hold a person; a row settled as "nobody here" carries none) and
      `helmet` (boxes on people the judge cleared, the only ones a helmet
      label can come from).
    - `flag_miss_count`: always `0` — see `flag_miss_note`.
    - `flag_miss_note`: explains why `flag_miss_count` is hardcoded rather
      than measured.
    """
    from .audit import (
        DEFAULT_AUDITS_PATH,
        NO_VALID_CROPS,
        included_people,
        load_audits,
    )
    from .escalations import AWAITING_VERDICT_CROPS, newest_per_crop

    rows = _read_audit_log_rows(audit_log_path)
    audits = load_audits(audits_path or DEFAULT_AUDITS_PATH)
    images = _images(rows, audits)
    total_alarms = len(images)
    outcome_counts = {outcome: 0 for outcome in _OUTCOMES}
    per_hazard_type = {
        hazard_type: {"total": 0, "outcome_counts": {outcome: 0 for outcome in _OUTCOMES}}
        for hazard_type in TRACK1_HAZARD_TYPES
    }
    auto_annotated_count = 0
    no_valid_crops = 0
    label_volume_this_week = 0
    person_labels = 0
    helmet_labels = 0
    week_ago = datetime.now(timezone.utc) - timedelta(days=7)

    for row in images:
        outcome = row.get("outcome")
        if outcome in outcome_counts:
            outcome_counts[outcome] += 1
        if row.get(NO_VALID_CROPS):
            no_valid_crops += 1

        hazard_type = row.get("hazard_type")
        if hazard_type in per_hazard_type:
            per_hazard_type[hazard_type]["total"] += 1
            if outcome in per_hazard_type[hazard_type]["outcome_counts"]:
                per_hazard_type[hazard_type]["outcome_counts"][outcome] += 1

        # DID THIS IMAGE NEED A HUMAN -- its own question, and not readable
        # off the outcome. TP outranks ESCALATE there, so a frame the judge
        # fired on AND hedged about reads CONFIRM_TP while still owing an
        # answer about the crop it hedged on. Counting that as auto-annotated
        # claimed a frame a reviewer is queued to look at.
        if outcome in _OUTCOMES and not row.get(AWAITING_VERDICT_CROPS):
            auto_annotated_count += 1

    # LABEL VOLUME, SPLIT BY WHAT THE LABELS ACTUALLY ARE (CEO, 2 Sep:
    # "label volume should split Total number images | Persons | Helmet").
    #
    # One number could not say whether a week's work was many pictures or one
    # crowded one, and the three units are genuinely different things: an
    # IMAGE is a sample in the full-frame dataset, a PERSON box is a sample in
    # the crop dataset and a label in the full-frame one, and a HELMET box is
    # a label that exists only where somebody was cleared.
    #
    # OVER THE CROPS THE EXPORT ACTUALLY WRITES. A crop an auditor struck out
    # is dropped by both datasets (`training_export._is_excluded` for the crop
    # one, the `struck` skip in `build_frame_export` for the frames), so
    # counting it here reported a label that had been retracted -- and the
    # error grew in proportion to how much the strike-out feature was used, so
    # it read correct until auditors started using it.
    labelled_images: set[str] = set()
    # ONE LABEL PER CROP, not per row. Re-judging a frame appends a row for
    # each of its people instead of replacing them, and this block claims what
    # was DELIVERED -- the export writes the newest row of a crop and no other.
    # Measured on the live corpus, 6 Sep: 821 rows over 781 crops, so the week
    # was claiming 808 labels where 781 exist, against a 400-a-week target.
    delivered, _ = included_people(newest_per_crop(rows), audits)
    for row in delivered:
        # IN ROWS, deliberately: a label is one person's crop.
        if not isinstance(row, dict):
            continue
        timestamp = row.get("timestamp")
        if not timestamp:
            continue
        try:
            ts = datetime.fromisoformat(timestamp)
        except (ValueError, TypeError):
            # fromisoformat raises TypeError (not ValueError) when
            # `timestamp` isn't a string at all -- e.g. a JSON number
            # landed in that field from a malformed audit-log row.
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if ts < week_ago:
            continue

        label_volume_this_week += 1
        labelled_images.add(_image_key(row))
        # A row settled as "nobody in this box" carries no person box and must
        # not be counted as one: labelling it would teach the detector that
        # whatever is there IS a person.
        if row.get("reason_code") not in _NO_PERSON_REASONS and row.get("person_box"):
            person_labels += 1
        # A helmet box only exists where the judge CLEARED the person. On
        # someone it called a violation there is no helmet to box.
        if row.get("outcome") == "CONFIRM_FP" and row.get("helmet_box"):
            helmet_labels += 1

    # AN ALL-STRUCK IMAGE IS OUT OF THE DENOMINATOR, not just out of the
    # numerator. An auditor striking every crop of a frame says nothing about
    # whether the pipeline needed a human for it, and leaving it below the
    # line turned a frame that had auto-annotated at 1.0 into a rate of 0.0 --
    # so striking crops dragged a headline number down. `audit_summary` pulls
    # the same case out of both sides of its own rate; these two screens have
    # to agree.
    decidable = total_alarms - no_valid_crops
    auto_annotation_rate = auto_annotated_count / decidable if decidable else 0.0

    return {
        "total_alarms": total_alarms,
        "outcome_counts": outcome_counts,
        "auto_annotation_rate": auto_annotation_rate,
        # Images an auditor struck every crop out of. Still images, so still
        # in `total_alarms`; out of the rate above, and reported so the
        # held-out population is visible rather than merely absent.
        "no_valid_crops": no_valid_crops,
        "per_hazard_type": per_hazard_type,
        "label_volume_this_week": label_volume_this_week,
        # The same week, split by what the labels ARE. One number could not
        # say whether a week was many pictures or one crowded one.
        "label_volume_breakdown": {
            "images": len(labelled_images),
            "person": person_labels,
            "helmet": helmet_labels,
        },
        "flag_miss_count": 0,
        "flag_miss_note": FLAG_MISS_NOTE,
    }


def compute_shadow_mode_agreement(
    audit_log_path: str | Path = DEFAULT_AUDIT_LOG_PATH,
    gold_annotations_path: str | Path = DEFAULT_GOLD_ANNOTATIONS_PATH,
) -> dict:
    """Join `audit_log_path` rows against `gold_annotations_path` entries
    (test_proj's `gold_set.load_gold_annotations` loader), matching
    Track1's `alarm_id` directly against the gold sidecar's `case_id` keys
    — the two are literally the same identifier for cases sourced from the
    gold set (see `Track1/eval/mock_alarm_feed.json`, built by remapping
    test_proj's gold-set cases 1:1, `alarm_id` == `source_detection_id` ==
    `case_id`).

    Only rows resolved to CONFIRM_TP/CONFIRM_FP are counted as "matched" —
    an ESCALATE row is the agent deferring to a human, so there is no
    agent verdict yet to compare against the gold label. A row only
    counts once it also has a resolved gold `final_label` that isn't
    `cannot_determine` (an unresolved/undetermined gold case has no human
    verdict to compare against either).

    Returns `{"insufficient_data": True, "matched_count": N}` when fewer
    than 10 matched cases exist (per CEO decision — the rate is withheld,
    not shown, below this threshold). Otherwise returns
    `{"insufficient_data": False, "matched_count": N, "agreement_count":
    M, "agreement_rate": M / N}`. Always returns a dict, never bare
    `None`, so callers don't need a second null-check on top of the
    `insufficient_data` flag.
    """
    rows = _read_audit_log_rows(audit_log_path)
    try:
        annotations = gold_set.load_gold_annotations(Path(gold_annotations_path))
    except json.JSONDecodeError:
        warnings.warn(
            f"gold annotations at {gold_annotations_path!s} contain invalid JSON — treating as empty.",
            RuntimeWarning,
            stacklevel=2,
        )
        annotations = {}
    if not isinstance(annotations, dict):
        warnings.warn(
            f"gold annotations at {gold_annotations_path!s} is not a JSON object — treating as empty.",
            RuntimeWarning,
            stacklevel=2,
        )
        annotations = {}

    matched = 0
    agreements = 0
    for row in rows:
        if not isinstance(row, dict):
            continue

        outcome = row.get("outcome")
        if outcome not in ("CONFIRM_TP", "CONFIRM_FP"):
            continue

        entry = annotations.get(row.get("alarm_id"))
        if not isinstance(entry, dict):
            continue
        final_label = entry.get("final_label")
        if not final_label:
            continue
        try:
            ground_truth = gold_set.to_ground_truth(final_label)
        except ValueError:
            continue
        if ground_truth is None:
            continue

        matched += 1
        agent_verdict = "TP" if outcome == "CONFIRM_TP" else "FP"
        if agent_verdict == ground_truth:
            agreements += 1

    if matched < 10:
        return {"insufficient_data": True, "matched_count": matched}

    return {
        "insufficient_data": False,
        "matched_count": matched,
        "agreement_count": agreements,
        "agreement_rate": agreements / matched,
    }
