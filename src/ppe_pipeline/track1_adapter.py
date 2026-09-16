"""The Track1 integration seam: `run_track1_judge_inference` (+
`Track1RunResult`/`_resolve_outcome`/`VIOLATION_TYPE_TO_HAZARD_LABEL`).

Judges one alarm through `ppe_pipeline`'s own judge — few-shot examples,
`crop_box_for_violation` cropping, and the geometric too-near-forklift
pre-filter — the same method "Evaluate against ground truth"'s stepper
uses. This is what the API's `/track1/run` endpoint calls.

An older `hazard_validation`-typed seam (`run_track1_inference`, judging
through Track1's real `worker.process_alarm`: zero-shot, uncropped, no
few-shot, no geometric filter) used to sit alongside it, kept alive only
because the Streamlit `app.py` called it directly. It went out with
`app.py`. The live `hazard_validation` worker itself still not using
few-shot is tracked separately (Jira SCRUM-28, cross-repo).
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from . import helmet_detector, storage
from .config import PipelineConfig, get_config
from .cropping import PersonCrop, crop_box_for_violation, crop_bytes, judge_box_to_original
from .few_shot import FewShotLibrary
from .forklift_rule import evaluate_too_near_forklift
from .helmet_detector import resolve_weights_path
from .judge_client import JudgeClient
from .judge_prompt import build_forklift_operator_context
from .models import BoundingBox, ViolationType

logger = logging.getLogger(__name__)

DEFAULT_AUDIT_LOG_PATH = "eval/track1_audit_log.json"
# Where a judged alarm's frame is kept, keyed by the row's `image_key`.
# Escalation review and YOLO export both need the original frame back long
# after the run, and an eval-dataset path is not enough: the Run screen is
# meant to grow a manual-upload source, whose bytes exist nowhere else.
DEFAULT_ALARM_IMAGE_DIR = "eval/alarm_images"

# Characters allowed in the filename component built from an alarm id.
# `alarm_id` is a client-supplied form field on POST /track1/run, and it is
# interpolated straight into a storage key -- so anything path-like in it
# (`..`, a separator, a Windows reserved name) must not survive into the
# key. storage.validate_key is the backstop; this keeps the key sane in the
# first place so a legitimate id still produces a readable filename.
_SAFE_KEY_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
_WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def safe_image_stem(alarm_id: str) -> str:
    """A filename-safe stem for an alarm id, never empty and never a
    Windows reserved device name (which would make the write fail, and the
    frame silently vanish)."""
    stem = _SAFE_KEY_CHARS.sub("_", str(alarm_id)).strip("._-")
    if not stem:
        stem = "alarm"
    if stem.upper() in _WINDOWS_RESERVED:
        stem = f"{stem}_"
    return stem

_MEDIA_TYPE_SUFFIXES = {"image/png": ".png", "image/jpeg": ".jpg"}

#: Top-level marker on a row produced by the helmet-CV emulation
#: (`helmet_cv/`) rather than by a live alarm. It shares this log — the CEO's
#: call, so CV and the judge are audited against one human decision through
#: one screen — which makes a filterable tag mandatory, not decorative: an
#: untagged CV row would be counted as an alarm by `rolling_metrics`, appear
#: in the escalation queue, and be exported as a training label for a frame
#: nobody ever alarmed on. The nested `row["cv"]` block carries the same run
#: id; this is at the top level so a filter never has to reach into it.
CV_ROW_KIND = "helmet_cv"
ROW_KIND_FIELD = "row_kind"



# The 4 hazard types the original hazard_validation.models.HazardType scheme
# covers, plus "glove" (no real equivalent there — glove isn't a current
# Track1 focus). Used only to label audit-log rows/API responses; nothing
# here depends on hazard_validation's HazardType enum anymore.
VIOLATION_TYPE_TO_HAZARD_LABEL: dict[ViolationType, str] = {
    ViolationType.HELMET: "no_helmet",
    ViolationType.DANGEROUS_PLANE: "hatch_fall_zone",
    ViolationType.TOO_NEAR_FORKLIFT: "too_close_to_vehicle",
    ViolationType.SUSPENDED_LOAD: "under_suspended_cargo",
    ViolationType.GLOVE: "glove",
}

# A confidence_score of exactly 0.5 must resolve to CONFIRM_TP, not
# ESCALATE (CEO decision) — comparing with `> 0.49` rather than `>= 0.5`
# gets that boundary right without relying on exact float equality.
_CONFIRM_THRESHOLD = 0.49

# The geometric proximity filter's decision is deterministic, not a
# probabilistic guess — 1.0 stands in for "fully confident" on a
# filter-dropped row, matching its "high" categorical confidence_level.
_FILTERED_CONFIDENCE_SCORE = 1.0


def _resolve_outcome(verdict: str, confidence_score: float | None) -> str:
    """Mirrors `hazard_validation.verdict_engine.resolve_alarm`'s original
    truth table, just fed a derived/reported score instead of a native float:
    "FP" always confirms; "TP" confirms only above the confidence threshold,
    otherwise escalates to human review; anything else ("uncertain") always
    escalates. `confidence_score` is `JudgeResult.confidence_score`, which is
    `Optional` (Bedrock's chat-completions endpoint offers only JSON mode,
    not a structured-output guarantee) — treat a
    missing score as unknown confidence and escalate, the same safe default
    as a low score, rather than raising."""
    if verdict == "FP":
        return "CONFIRM_FP"
    if verdict == "TP":
        return "CONFIRM_TP" if confidence_score is not None and confidence_score > _CONFIRM_THRESHOLD else "ESCALATE"
    return "ESCALATE"


def _box_to_dict(box: BoundingBox) -> dict:
    return {"x1": box.x1, "y1": box.y1, "x2": box.x2, "y2": box.y2}


@dataclass
class Track1RunResult:
    alarm_id: str
    hazard_type: str
    violation_type: str
    outcome: str
    verdict: str
    confidence_level: str
    confidence_score: float | None
    reason: str
    explanation: str
    # The image the judge was actually shown, and the region of the original
    # it was cut from — so a caller can render exactly what was judged
    # instead of re-deriving the crop and risking drift from this function's.
    judged_image_bytes: bytes = b""
    crop_box: BoundingBox | None = None
    # Where the judge said the helmet is, on a helmet alarm it cleared as
    # "helmet_worn": normalized to `judged_image_bytes`, and the same box
    # mapped back onto the original frame. None for everything else.
    helmet_box: list[float] | None = None
    helmet_box_original: BoundingBox | None = None
    # IoU between two independent localization asks — the signal that says
    # whether this box is worth training on. See helmet_detector.py.
    helmet_box_confidence: float | None = None
    # "cv" (stage-2 YOLO11s) or "vlm" (the judge's inline box, the
    # fallback). See eval/runner.py's CaseOutcome for the full rationale.
    helmet_box_source: str | None = None
    # The judge that produced this verdict, and the detector weights behind
    # a "cv" box. Recorded so a label can be traced to the models that made it.
    judge_provider: str | None = None
    judge_model: str | None = None
    helmet_detector_weights: str | None = None
    # Storage key the judged frame was saved under, and its pixel size.
    # Both exist for what happens *after* the run: escalation review needs
    # the frame back, and a YOLO label needs the dimensions to normalize
    # boxes that are recorded in original pixels.
    image_key: str | None = None
    image_width: int | None = None
    image_height: int | None = None


#: The reason code a crop the first-pass gate emptied is settled with. Its
#: own code, never `helmet_worn`: "nobody was here" and "the worker here is
#: wearing a helmet" are different findings and a training set must be able to
#: tell them apart.
NO_PERSON_REASON = "no_person_present"

#: The DETECTOR found nobody in the frame at all -- a different finding from
#: the gate's, and it needs its own code.
#:
#: Before this, such a frame was judged on the WHOLE FRAME: the judge was
#: asked "is this person wearing a helmet" about a picture with no identified
#: person, and answered accordingly. Measured on one live run, 7 of 99 rows
#: were that case and 6 of them came back `CONFIRM_FP / helmet_worn` -- "the
#: helmet is worn" about a frame where nothing was detected. Confidently
#: wrong, and headed for the training export.
#:
#: Settled as No Violation for the same reason a gated crop is: no person
#: means no person in violation. NOT proof nobody was there -- a detector
#: that misses people is exactly what was found on 1 Sep -- so the frame
#: stays visible in the audit for a human to overturn.
NO_PERSON_DETECTED_REASON = "no_person_detected"


def record_no_person_row(
    image_bytes: bytes,
    media_type: str,
    *,
    alarm_id: str,
    source_detection_id: str,
    person_box: BoundingBox | None,
    crop_box: BoundingBox | None,
    gate_reason: str,
    cv_decision: dict,
    reason: str = NO_PERSON_DETECTED_REASON,
    answered_by: str = "person_detector",
    lineage: dict | None = None,
    audit_log_path: str | Path = DEFAULT_AUDIT_LOG_PATH,
    alarm_image_dir: str | Path | None = None,
    image_key: str | None = None,
    config: PipelineConfig | None = None,
) -> dict:
    """Settle one crop with nobody in it. NO JUDGE CALL.

    One caller since 2 Sep: no rectangle could be recovered from the
    snapshot, so there is no box to crop and nothing to ask about. The
    first-pass person gate was the other, and it is gone.

    This writes a row rather than nothing, and that is the point. A settled
    crop with no row has no case, so no image, so it appeared in the feed's
    results and then 404'd when opened. A row makes it an ordinary case: an
    image, a verdict, a reason, and a human able to disagree.

    `CONFIRM_FP` because there is no person, so there is no person in
    violation -- the alarm was false. `reason_code` is `no_person_present`,
    NOT `helmet_worn`: they are different findings and the export must be
    able to tell them apart. The judge fields say who answered -- the gate,
    with its own words -- so nothing downstream mistakes this for a helmet
    verdict.

    `image_key` may be passed to share ONE stored copy of the frame across
    every gated crop of that frame, rather than storing the same 1920x1080
    JPEG once per blob.
    """
    config = config or get_config()
    alarm_image_dir = alarm_image_dir if alarm_image_dir is not None else DEFAULT_ALARM_IMAGE_DIR

    image_width = image_height = None
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image_width, image_height = image.size
    except Exception:  # noqa: BLE001 - a row with no size is worth more than no row
        logger.warning("could not read frame size for %s", alarm_id)

    if image_key is None:
        suffix = ".png" if media_type == "image/png" else ".jpg"
        image_key = f"{Path(alarm_image_dir).as_posix()}/{safe_image_stem(alarm_id)}{suffix}"
        try:
            storage.get_storage().put_bytes(image_key, image_bytes, content_type=media_type)
        except Exception:  # noqa: BLE001 - same treatment as the judge path
            logger.exception("could not persist alarm frame at %s", image_key)
            image_key = None

    row = {
        "alarm_id": alarm_id,
        "hazard_type": "no_helmet",
        "outcome": "CONFIRM_FP",
        "reason_code": reason,
        # The gate answers a yes/no question and reports no score of its own.
        # A number invented here would be indistinguishable from the judge's.
        "confidence": None,
        "confidence_level": "high",
        "reasoning": gate_reason or "No person was found, so there is no helmet question to ask.",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_detection_id": source_detection_id,
        "person_box": _box_to_dict(person_box) if person_box is not None else None,
        "image_width": image_width,
        "image_height": image_height,
        "crop_box": _box_to_dict(crop_box) if crop_box is not None else None,
        # WHO answered. Not the judge model: this verdict never reached it.
        "judge_provider": answered_by,
        "judge_model": config.judge.model,
        ROW_KIND_FIELD: CV_ROW_KIND,
        "cv": cv_decision,
    }
    if lineage is not None:
        row["lineage"] = dict(lineage)
        row["source"] = lineage.get("source")
    if image_key is not None:
        row["image_key"] = image_key

    storage.get_storage().update_json(
        Path(audit_log_path).as_posix(), lambda rows: rows + [row], default=[]
    )
    return row


def run_track1_judge_inference(
    image_bytes: bytes,
    media_type: str,
    violation_type: ViolationType,
    person_box: BoundingBox,
    alarm_id: str,
    source_detection_id: str,
    forklift_box: BoundingBox | None = None,
    region_box: BoundingBox | None = None,
    provider: str | None = None,
    model: str | None = None,
    few_shot_library: FewShotLibrary | None = None,
    config: PipelineConfig | None = None,
    audit_log_path: str | Path = DEFAULT_AUDIT_LOG_PATH,
    alarm_image_dir: str | Path | None = None,
    helmet_box_mode: str | None = None,
    cv_decision: dict | None = None,
    prepared_crop: PersonCrop | None = None,
    lineage: dict | None = None,
) -> Track1RunResult:
    """Judges one alarm through `ppe_pipeline`'s real judge pipeline
    (crop -> geometric too-near-forklift pre-filter -> few-shot-primed
    `JudgeClient`), synchronously, and appends a row to the audit log —
    the interactive spot-check tool behind "Run Track1 inference on this
    alarm".

    `helmet_box_mode` picks what draws the helmet box on a cleared helmet:
    "cv" (default) runs the stage-2 YOLO11s detector; "vlm" skips it and uses
    the judge's own inline `helmet_box` field. The VLM path is the automatic
    fallback anyway, so this only forces it -- useful for deliberately
    producing the loose boxes the review queue is meant to catch, which are
    otherwise hard to come by (the detector clears most cases well above the
    review threshold).

    `cv_decision`, when given, is the helmet-CV emulation's own answer for
    this same frame (`helmet_cv.records.CandidateRecord.cv_block`). It is
    attached to the audit row as a NESTED `cv` block and the row is marked
    `row_kind: "helmet_cv"`. Nested rather than flattened because the row
    already has `confidence`, `helmet_box`, `helmet_box_source` and
    `helmet_box_confidence` of its own, all meaning the JUDGE's — a flat CV
    `helmet_box` would silently overwrite the judge's and leave no way to
    tell which model drew it.

    `forklift_box` only applies to `TOO_NEAR_FORKLIFT`; `region_box` only
    to `GLOVE`/`HELMET` (see `crop_box_for_violation`). `provider`/`model`
    left `None` default to `config.judge`'s own provider/model.
    `few_shot_library` left `None` runs the judge zero-shot.

    A `TOO_NEAR_FORKLIFT` candidate the geometric proximity filter finds
    not close enough (`not is_violation and not is_operator`) never reaches
    the judge at all — it's synthesized as a "FP"/"high" verdict directly,
    the same short-circuit `eval.runner.run_eval` and the eval stepper's
    `_compute_pending` both apply for this case.

    Returns a `Track1RunResult` and appends one row to `audit_log_path`
    (a local JSON array, default `eval/track1_audit_log.json`) — same
    field shape `rolling_metrics.py` already reads back for the metrics
    tab, with `confidence` now the genuine numeric `confidence_score` and
    a new additive `confidence_level` (the categorical word) alongside it.
    """
    config = config or get_config()
    # Resolved at call time rather than as a default argument: a default is
    # bound at import, which would make the destination impossible to
    # redirect (tests would write frames into the repo).
    alarm_image_dir = alarm_image_dir if alarm_image_dir is not None else DEFAULT_ALARM_IMAGE_DIR
    # Resolved here, not read off the client's privates, because both end up
    # in the audit row: a verdict is only interpretable later if you know
    # which model produced it, and the model changes over the project's life.
    judge_provider = provider if provider is not None else config.judge.provider
    judge_model = model if model is not None else config.judge.model
    client = JudgeClient(provider=provider, model=model, few_shot_library=few_shot_library, config=config)

    if prepared_crop is None:
        with Image.open(io.BytesIO(image_bytes)) as image:
            orig_size = image.size
    else:
        orig_size = (prepared_crop.original_width, prepared_crop.original_height)
    image_width, image_height = orig_size

    if prepared_crop is None:
        box = crop_box_for_violation(
            violation_type,
            person_box,
            orig_size,
            region_box=region_box,
            forklift_box=forklift_box,
            config=config,
        )
        cropped_bytes, cropped_media_type = crop_bytes(image_bytes, box)
    else:
        box = prepared_crop.crop_box
        cropped_bytes = prepared_crop.image_bytes
        cropped_media_type = prepared_crop.media_type

    filter_result = None
    if violation_type is ViolationType.TOO_NEAR_FORKLIFT and forklift_box is not None:
        filter_result = evaluate_too_near_forklift(person_box, forklift_box, config=config)

    if filter_result is not None and not filter_result.is_violation and not filter_result.is_operator:
        verdict = "FP"
        confidence_level = "high"
        confidence_score = _FILTERED_CONFIDENCE_SCORE
        reason = "geometric_proximity_filter"
        explanation = (
            "Filtered before reaching the judge: geometric proximity check found the "
            "person and forklift not near enough."
        )
        # too_near_forklift only, and no judge call was made — nothing to report.
        helmet_box = None
        helmet_box_confidence = None
        helmet_box_source = None
    else:
        extra_context = None
        if filter_result is not None and filter_result.is_operator:
            extra_context = build_forklift_operator_context(filter_result.containment_ratio)

        call = client.judge(
            image_bytes=cropped_bytes,
            media_type=cropped_media_type,
            person_id=alarm_id,
            violation_types=[violation_type],
            extra_context=extra_context,
        )
        result = call.response.results[0]
        verdict = result.verdict
        confidence_level = result.confidence
        confidence_score = result.confidence_score
        reason = result.reason
        explanation = result.explanation

        # Same stage-2 detector pass the eval runner makes on a cleared
        # helmet, for the same reason: the inline `helmet_box` field is too
        # loose to label from. This screen is the eval path over one image, so
        # it has to ask the same question the same way — otherwise a
        # single-image spot check and the eval run disagree about the box for
        # the identical case.
        helmet_box = result.helmet_box
        helmet_box_confidence = None
        helmet_box_source = "vlm" if helmet_box is not None else None
        if (
            violation_type is ViolationType.HELMET
            and verdict == "FP"
            and reason == "helmet_worn"
        ):
            # "vlm" forces the fallback: no detector call, so the judge's
            # own inline box stands and helmet_box_source stays "vlm".
            cv_mode = helmet_box_mode != "vlm"
            detector = None if not cv_mode else helmet_detector.get_helmet_detector(config)
            located = detector.locate(cropped_bytes, cropped_media_type) if detector else None
            if located is not None:
                helmet_box = located.box
                helmet_box_confidence = located.confidence
                helmet_box_source = "cv"
            elif cv_mode:
                # The detector is the chosen source, so its silence is the
                # answer: no box. The judge's inline box is NOT a fallback
                # here (CEO decision). Keeping it would record a box nobody
                # asked for, draw it on the review crop, and -- since a
                # reviewer can now accept a detector box in one press -- let
                # someone endorse a VLM guess while believing they were
                # endorsing YOLO11s. The case still reaches Helmet boxes,
                # because `needs_box` queues anything whose source is not
                # "cv", and it now arrives with nothing drawn on it.
                helmet_box = None
                helmet_box_confidence = None
                helmet_box_source = None

    outcome = _resolve_outcome(verdict, confidence_score)
    helmet_box_original = (
        judge_box_to_original(helmet_box, box) if helmet_box is not None else None
    )
    hazard_type = VIOLATION_TYPE_TO_HAZARD_LABEL[violation_type]
    suffix = _MEDIA_TYPE_SUFFIXES.get(media_type, ".jpg")
    image_key = f"{Path(alarm_image_dir).as_posix()}/{safe_image_stem(alarm_id)}{suffix}"

    # Written before the audit row, so a row never points at a frame that
    # is not there. A failure here must not lose the verdict -- the run is
    # already done and paid for -- so it degrades to a row with no image_key
    # rather than raising; the case then simply cannot be reviewed later.
    try:
        storage.get_storage().put_bytes(image_key, image_bytes, content_type=media_type)
    except Exception:  # noqa: BLE001 - a sound verdict outweighs a saved frame
        logger.exception("could not persist alarm frame at %s", image_key)
        image_key = None

    row = {
        "alarm_id": alarm_id,
        "hazard_type": hazard_type,
        "outcome": outcome,
        "reason_code": reason,
        "confidence": confidence_score,
        "confidence_level": confidence_level,
        "reasoning": explanation,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_detection_id": source_detection_id,
        "person_box": _box_to_dict(person_box),
        # Original frame size. Boxes above are in original pixels, and a
        # YOLO label needs them normalized -- without this the export would
        # have to re-open every image just to divide by a width.
        "image_width": image_width,
        "image_height": image_height,
        # The exact region the judge was shown, in original pixels.
        # Recorded rather than recomputed at review time because the crop
        # depends on config pad factors AND on region_box, which is not
        # stored -- so a later config change, or a case with a head region,
        # would silently reconstruct a different crop from the one judged,
        # and a box drawn on it would map back to the wrong place.
        "crop_box": _box_to_dict(box),
        # Which models produced this row. Track2 consumes these downstream to
        # tell one generation of labels from another, and a verdict with no
        # recorded judge is not reproducible.
        "judge_provider": judge_provider,
        "judge_model": judge_model,
    }
    if cv_decision is not None:
        row[ROW_KIND_FIELD] = CV_ROW_KIND
        row["cv"] = cv_decision
    if lineage is not None:
        row["lineage"] = dict(lineage)
        row["source"] = lineage.get("source")
    if helmet_box_source == "cv":
        row["helmet_detector_weights"] = resolve_weights_path(config.helmet_box.weights).name
    if image_key is not None:
        row["image_key"] = image_key
    if forklift_box is not None:
        row["vehicle_box"] = _box_to_dict(forklift_box)
    if helmet_box_confidence is not None:
        row["helmet_box_confidence"] = helmet_box_confidence
    if helmet_box_source is not None:
        row["helmet_box_source"] = helmet_box_source
    if helmet_box_original is not None:
        # Original-image coordinates, not the normalized crop-relative ones:
        # the crop this was judged on isn't reproducible from the row (pad
        # factors are config, and config moves), so a normalized box would go
        # stale the moment cropping is retuned.
        row["helmet_box"] = _box_to_dict(helmet_box_original)

    storage.get_storage().update_json(
        Path(audit_log_path).as_posix(), lambda rows: rows + [row], default=[]
    )

    return Track1RunResult(
        alarm_id=alarm_id,
        hazard_type=hazard_type,
        violation_type=violation_type.value,
        outcome=outcome,
        verdict=verdict,
        confidence_level=confidence_level,
        confidence_score=confidence_score,
        reason=reason,
        explanation=explanation,
        judged_image_bytes=cropped_bytes,
        crop_box=box,
        helmet_box=helmet_box,
        helmet_box_original=helmet_box_original,
        helmet_box_confidence=helmet_box_confidence,
        helmet_box_source=helmet_box_source,
        judge_provider=judge_provider,
        judge_model=judge_model,
        helmet_detector_weights=(
            resolve_weights_path(config.helmet_box.weights).name if helmet_box_source == "cv" else None
        ),
        image_key=image_key,
        image_width=image_width,
        image_height=image_height,
    )
