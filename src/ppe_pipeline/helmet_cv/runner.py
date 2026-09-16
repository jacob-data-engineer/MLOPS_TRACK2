"""Running the judge over a sample of
what it produced.

TWO PASSES, DELIBERATELY SEPARATE

The gate pass is local, free, and repeatable: a YOLO11s model on the CPU
over frames already on disk. `judge_candidates` costs money and minutes per
call. Keeping them apart means the expensive pass runs over a chosen sample
of a finished CV pass, rather than the two being interleaved and a killed
run leaving half a comparison.

RESUME AND LAYOUT MIRROR `extract`

    <out_dir>/cv_runs/<run>/cv/<alarm_id>.json  per-alarm resume marker
    <out_dir>/cv_runs/<run>/cv_candidates.jsonl merged candidates

Same reasoning as `extract`'s: a sidecar per alarm means concurrent workers
never contend for a lock, an existing sidecar means that alarm is done, and
the sidecar is written through a temp file and `os.replace`d so a run killed
mid-write cannot leave truncated JSON that reads as "done".

ONE PASS IS THREADED AND THE OTHER IS NOT

The gate pass is single threaded on purpose: the model sits behind an
inference lock (see `pose`), so a pool would serialise on that lock
and buy nothing but a harder-to-read traceback.

`judge_candidates` IS pooled, because its time goes on waiting for Bedrock
rather than on the CPU. It also groups its work by FRAME rather than by
candidate, so a frame holding four people is read from disk once instead of
four times.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from ..track1_adapter import NO_PERSON_DETECTED_REASON
from .records import (
    FRAME_ROLE_CLIP,
    CandidateRecord,
    CvProvenance,
    build_record,
    execution_id,
    validate_cv_run_id,
)

logger = logging.getLogger(__name__)

#: Bumped with `records.SCHEMA_VERSION` when the cascade was removed.
SCHEMA_VERSION = 3
CANDIDATE_MANIFEST = "cv_candidates.jsonl"
RUN_METADATA = "run.json"
LEGACY_RUN_ID = "legacy-unscoped"


def new_cv_run_id() -> str:
    """A fresh id for one pass. Stamped on every record and on every audit
    row the judge pass writes, so one run's rows can always be told from
    another's — the two minimum-height arms otherwise look identical."""
    return f"cv-{uuid.uuid4().hex[:12]}"


def run_dir(out_dir: Path, cv_run_id: str) -> Path:
    return out_dir / "cv_runs" / validate_cv_run_id(cv_run_id)


def run_metadata_path(out_dir: Path, cv_run_id: str) -> Path:
    return run_dir(out_dir, cv_run_id) / RUN_METADATA


def artifact_root(out_dir: Path, cv_run_id: str | None) -> Path:
    return out_dir if cv_run_id in (None, LEGACY_RUN_ID) else run_dir(out_dir, cv_run_id)


def _ensure_run_metadata(out_dir: Path, provenance: CvProvenance) -> dict:
    if provenance.cv_run_id == LEGACY_RUN_ID:
        raise ValueError(f"{LEGACY_RUN_ID!r} is read-only; choose a new cv_run_id")
    path = run_metadata_path(out_dir, provenance.cv_run_id)
    expected = asdict(provenance)
    if path.is_file():
        current = json.loads(path.read_text(encoding="utf-8"))
        if current.get("provenance") != expected:
            raise ValueError(
                f"CV run {provenance.cv_run_id!r} metadata does not match this configuration; "
                "choose a new --cv-run-id. Run metadata is immutable."
            )
        return current
    payload = {
        "schema": SCHEMA_VERSION,
        "cv_run_id": provenance.cv_run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "provenance": expected,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temp, path)
    return payload


def available_runs(out_dir: Path) -> list[dict]:
    runs: list[dict] = []
    root = out_dir / "cv_runs"
    if root.is_dir():
        for path in sorted(root.glob(f"*/{RUN_METADATA}")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                validate_cv_run_id(payload.get("cv_run_id", ""))
                runs.append(payload)
            except (OSError, ValueError, TypeError) as exc:
                logger.warning("unreadable CV run metadata %s: %s", path, exc)
    if (out_dir / "cv").is_dir() or (out_dir / CANDIDATE_MANIFEST).is_file():
        runs.append({"cv_run_id": LEGACY_RUN_ID, "legacy": True, "provenance": {}})
    return runs


def sidecar_path(out_dir: Path, alarm_id: str, cv_run_id: str | None = None) -> Path:
    base = artifact_root(out_dir, cv_run_id)
    return base / "cv" / f"{alarm_id}.json"


def is_scored(out_dir: Path, alarm_id: str, cv_run_id: str | None = None) -> bool:
    return sidecar_path(out_dir, alarm_id, cv_run_id).is_file()


def iter_frames(frames_dir: Path) -> Iterator[dict]:
    """Every extracted frame, from `extract`'s merged manifest."""
    manifest = frames_dir / "frames.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"no frame manifest at {manifest} — run `helmet_cv.cli extract` first, "
            "or `helmet_cv.cli manifest` to rebuild it from existing sidecars."
        )
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def group_by_alarm(frames: Iterable[dict]) -> dict[str, list[dict]]:
    """Frames per alarm, each list in frame-index order — the sidecar's own
    unit, and the order a reviewer reads a clip in."""
    grouped: dict[str, list[dict]] = {}
    for frame in frames:
        grouped.setdefault(frame.get("alarm_id", ""), []).append(frame)
    for rows in grouped.values():
        rows.sort(key=lambda row: row.get("frame_index", 0))
    return grouped


def records_from_boxes(
    frame_row: dict,
    boxes: list[tuple[float, float, float, float]],
    *,
    provenance: CvProvenance,
    box_source: str = "vendor_cv",
) -> list[CandidateRecord]:
    """Candidates from boxes SOMEONE ELSE drew -- the vendor CV's own.

    At least one record, always, so a
    snapshot whose annotation could not be read still leaves a row rather than
    vanishing from the corpus. That row is judged on the whole frame exactly
    as a no-detection frame is.

    No confidence: the vendor publishes a box, not a score. `None` says so
    rather than inventing a number that would sit in the same field as our
    detector's real ones.
    """
    if not boxes:
        return [
            build_record(
                frame=frame_row,
                person_index=0,
                person_box=None,
                person_confidence=None,
                people_in_frame=0,
                helmets_in_frame=None,
                provenance=provenance,
                box_source=box_source,
            )
        ]
    return [
        build_record(
            frame=frame_row,
            person_index=index,
            person_box=[float(v) for v in box],
            person_confidence=None,
            people_in_frame=len(boxes),
            helmets_in_frame=None,
            provenance=provenance,
            box_source=box_source,
        )
        for index, box in enumerate(boxes)
    ]


def _write_sidecar(out_dir: Path, alarm_id: str, payload: dict) -> None:
    """Atomic, for the same reason `extract._write_sidecar` is: a truncated
    sidecar reads as "done" and is never retried."""
    path = sidecar_path(out_dir, alarm_id, payload.get("cv_run_id"))
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temp_path, path)


def iter_cv_sidecars(out_dir: Path, cv_run_id: str | None = None) -> Iterator[dict]:
    """Every CV sidecar, alarm-id order. An unreadable one is skipped with a
    warning rather than ending the walk."""
    cv_dir = artifact_root(out_dir, cv_run_id) / "cv"
    if not cv_dir.is_dir():
        return
    for path in sorted(cv_dir.glob("*.json")):
        try:
            yield json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("unreadable cv sidecar %s: %s", path, exc)


def build_candidate_manifest(
    out_dir: Path, manifest_path: Path | None = None, *, cv_run_id: str | None = None
) -> int:
    """Merge the per-alarm sidecars into one JSONL of candidates."""
    base = artifact_root(out_dir, cv_run_id)
    manifest_path = manifest_path or (base / CANDIDATE_MANIFEST)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    temp_path = manifest_path.with_suffix(".jsonl.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        for sidecar in iter_cv_sidecars(out_dir, cv_run_id):
            for record in sidecar.get("candidates", ()):
                handle.write(json.dumps(record) + "\n")
                rows += 1
    os.replace(temp_path, manifest_path)
    return rows


def iter_candidates(
    out_dir: Path, manifest_path: Path | None = None, *, cv_run_id: str | None = None
) -> Iterator[dict]:
    """Every candidate from the merged manifest."""
    base = artifact_root(out_dir, cv_run_id)
    manifest_path = manifest_path or (base / CANDIDATE_MANIFEST)
    if not manifest_path.is_file():
        return
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


# --- the judge pass -------------------------------------------------------


def _frame_bytes(frames_dir: Path, candidate: dict) -> bytes:
    return (frames_dir / candidate["relative_path"]).read_bytes()


def _execution_key(candidate: dict) -> str:
    return candidate.get("execution_id") or candidate.get("candidate_id", "")


def group_by_frame(candidates: Iterable[dict]) -> dict[str, list[dict]]:
    """Candidates keyed by the frame they came from, person order preserved.

    The unit of WORK is a frame even though the unit of COMPARISON is a
    person: a 1920x1080 JPEG read and decoded once serves however many people
    are standing in it, and four routed workers in one frame otherwise cost
    four reads of the same file.
    """
    grouped: dict[str, list[dict]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.get("frame_name", ""), []).append(candidate)
    for rows in grouped.values():
        rows.sort(key=lambda row: row.get("person_index", 0))
    return grouped


def _full_frame_box(image_bytes: bytes) -> list[float]:
    """The whole frame as a person box, for a candidate CV found nobody in."""
    import cv2
    import numpy as np

    decoded = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    height, width = decoded.shape[:2]
    return [0.0, 0.0, float(width), float(height)]


def _record_no_person(
    image_bytes: bytes,
    cv_candidate: dict,
    crop,
    gate_reason: str,
    judge_kwargs: dict,
    shared_frame_key: str | None,
    reason: str | None = None,
    answered_by: str = "person_detector",
) -> str | None:
    """Write one settled "nobody here" row, returning the frame key to reuse.

    One caller now: no rectangle was recovered for this frame at all. It used
    to have a second -- the first-pass person gate, removed 2 Sep -- which is
    why the row still names its answerer rather than assuming.
    """
    from ..models import BoundingBox
    from ..track1_adapter import NO_PERSON_DETECTED_REASON, record_no_person_row

    execution = cv_candidate["execution_id"]
    person_box = cv_candidate.get("person_box")
    row = record_no_person_row(
        image_bytes,
        "image/jpeg",
        alarm_id=execution,
        source_detection_id=execution,
        person_box=BoundingBox(*person_box) if person_box else None,
        crop_box=crop.crop_box if crop is not None else None,
        gate_reason=gate_reason,
        reason=reason or NO_PERSON_DETECTED_REASON,
        answered_by=answered_by,
        cv_decision=CandidateRecord(**cv_candidate).cv_block(),
        lineage=judge_kwargs.get("lineage"),
        audit_log_path=judge_kwargs.get("audit_log_path", "eval/track1_audit_log.json"),
        alarm_image_dir=judge_kwargs.get("alarm_image_dir"),
        image_key=shared_frame_key,
        config=judge_kwargs.get("config"),
    )
    return shared_frame_key or row.get("image_key")


def _judge_one_frame(
    candidates: list[dict],
    *,
    frames_dir: Path,
    judge,
    judge_kwargs: dict,
    already_judged: set[str],
    should_stop: Callable[[], bool] | None = None,
    on_crop_judged: Callable[[], None] | None = None,
) -> dict:
    """Judge every candidate of ONE frame, reading that frame once.

    `should_stop` is checked BETWEEN per-person calls, so a stop costs at most
    the call already in flight rather than the whole frame. That deliberately
    leaves the frame short of `people_in_frame`, which is why
    `audit.vlm_frame_verdicts` reports completeness: an interrupted frame must
    not read as "nobody else was in violation" -- see `_truncate_to_limit`,
    which refuses to create the same shape on purpose.
    """
    from ..cropping import crops_for_frame
    from ..models import BoundingBox, ViolationType

    # `gated` and `gate_calls` are kept at zero rather than dropped: they are
    # read off stored rows written before the first-pass gate was removed
    # (2 Sep), and a missing key there reads as a different kind of record.
    counts = {"judged": 0, "skipped": 0, "failed": 0, "stopped": 0, "gated": 0, "gate_calls": 0, "tp": 0, "fp": 0, "escalated": 0, "no_person": 0}

    def settled(n: int = 1) -> None:
        """`n` crops of this frame are now ANSWERED, however they were.

        THE CONTRACT IS "ANSWERED", NOT "JUDGED", and the caller's total is
        `len(candidates)` -- so every candidate has to arrive here exactly
        once or the screen's "3 of 9" stops short and reads as a hang. Three
        ways a crop settles without a judge call, all of them real:
        the detector found nobody and the crop is answered by reason code
        (measured at 7 of 99 rows on one live run), the frame could not be
        read or decoded, or the crop was judged on an earlier pass.

        A crop dropped by `should_stop` is deliberately NOT counted: it was
        never attempted, the run is ending, and a total that completes itself
        on the way out would claim work nobody did.
        """
        if on_crop_judged is None:
            return
        for _ in range(n):
            on_crop_judged()

    pending = [c for c in candidates if _execution_key(c) not in already_judged]
    counts["skipped"] = len(candidates) - len(pending)
    settled(counts["skipped"])
    if not pending:
        return counts

    try:
        image_bytes = _frame_bytes(frames_dir, pending[0])
    except OSError as exc:
        # One unreadable frame costs its own candidates, not the pass.
        logger.warning("could not read frame for %s: %s", pending[0]["candidate_id"], exc)
        counts["failed"] = len(pending)
        settled(len(pending))
        return counts

    full_frame: list[float] | None = None
    try:
        prepared = {
            crop.candidate_id: crop
            for crop in crops_for_frame(
                image_bytes,
                pending,
                ViolationType.HELMET,
                judge_kwargs.get("config"),
                full_frame_for_missing=True,
            )
        }
    except Exception as exc:  # noqa: BLE001 - a bad decode must not end a PAID pass
        # `_frame_bytes` above is guarded, but the DECODE was not, and PIL
        # raises here on a truncated JPEG. That propagated out of the frame loop
        # and killed a run mid-flight with money already spent and no resume
        # marker for the frames it never reached. `extract_clip` anticipates
        # truncated files; so must this.
        logger.warning("could not decode frame for %s: %s", pending[0]["candidate_id"], exc)
        counts["failed"] = len(pending)
        settled(len(pending))
        return counts
    # ONE STORED COPY of the frame for every crop of it that settles without
    # a judge call, rather than the same 1920x1080 JPEG once per row.
    shared_frame_key: str | None = None

    # THE CROPS THAT NEED NO JUDGE CALL ARE SETTLED FIRST, and sequentially.
    # Each writes a row and they share ONE stored copy of the frame, which a
    # race would duplicate.
    judgeable: list[dict] = []
    for index, candidate in enumerate(pending):
        if should_stop is not None and should_stop():
            counts["stopped"] = len(pending) - index
            return counts
        candidate_id = candidate["candidate_id"]
        if candidate.get("person_box") is None:
            # NO BOX WAS RECOVERED FOR THIS FRAME. `records_from_boxes` emits one
            # candidate with no box so the frame stays visible rather than
            # vanishing from the corpus -- and this used to judge it on the
            # WHOLE FRAME, which asks "is this person wearing a helmet" about
            # a picture with no identified person. Measured on one live run: 7
            # of 99 rows, 6 of them answered `helmet_worn`.
            #
            # No person means no person in violation, so it settles here, with
            # its own reason and the DETECTOR named as the answerer. No judge
            # call. It is not proof nobody was there, so the frame stays in
            # the audit for a human to overturn.
            counts["no_person"] += 1
            cv_candidate = _cv_candidate_for(candidate)
            try:
                shared_frame_key = _record_no_person(
                    image_bytes,
                    cv_candidate,
                    prepared.get(candidate_id),
                    "The detector found no person in this frame.",
                    judge_kwargs,
                    shared_frame_key,
                    reason=NO_PERSON_DETECTED_REASON,
                    answered_by="person_detector",
                )
            except Exception as exc:  # noqa: BLE001 - a lost row must not end the pass
                logger.warning("could not record the empty frame %s: %s", candidate_id, exc)
            settled()
            continue
        judgeable.append(candidate)

    # CROPS OF ONE FRAME RUN CONCURRENTLY.
    #
    # MEASURED on the deployed feed, 1 Sep: one alarm took about four minutes
    # -- 9 judge calls at ~25s each, strictly one after another -- while the
    # clip download and the detector together took 5.1 seconds. The judge
    # calls ARE the wall clock, and they were the one part not running
    # concurrently: `max_workers` counts FRAMES in flight, and a live alarm is
    # one frame, so it never helped here.
    #
    # Each task makes its own call and writes its own row through the same
    # `run_track1_judge_inference`, so nothing about a row changes -- only the
    # order they land in, and rows are independent. Storage appends are
    # already concurrent today (frames run in a pool), so `update_json`'s
    # compare-and-swap is already doing this job.
    #
    # SAVES NO TOKENS. This is wall clock only; batching is the token lever
    # and it is a separate, unshipped decision.
    def judge_one(candidate: dict) -> dict:
        candidate_id = candidate["candidate_id"]
        # STOP MEANS STOP, checked as each task starts rather than only
        # between frames. Concurrency weakens the guarantee honestly: it was
        # "at most the one call in flight", it is now "at most `crop_workers`
        # in flight", because a task already dispatched cannot be recalled.
        # A crop skipped this way is `stopped`, never a verdict.
        if should_stop is not None and should_stop():
            return {"stopped": 1}
        try:
            box = candidate.get("person_box") or _full_frame_box(image_bytes)
            cv_candidate = _cv_candidate_for(candidate)
            execution = cv_candidate["execution_id"]
            judged = judge(
                image_bytes,
                "image/jpeg",
                ViolationType.HELMET,
                BoundingBox(*box),
                execution,
                execution,
                cv_decision=CandidateRecord(**cv_candidate).cv_block(),
                prepared_crop=prepared.get(candidate_id),
                **judge_kwargs,
            )
            result = {"judged": 1, "outcome": getattr(judged, "outcome", None)}
        except Exception as exc:  # noqa: BLE001 - one failed call must not end the pass
            logger.warning("judging %s failed: %s", candidate_id, exc)
            result = {"failed": 1, "outcome": None}
        # A COMPLETION, reported as it happens rather than counted afterwards:
        # the caller is a screen watching an alarm that takes minutes. A crop
        # dropped by `should_stop` returned above and is not one. The counter
        # takes a lock, because these land from `crop_workers` threads at once.
        settled()
        return result

    config = judge_kwargs.get("config")
    workers = getattr(getattr(config, "helmet_cv", None), "crop_workers", 1) or 1
    if len(judgeable) > 1 and workers > 1:
        with ThreadPoolExecutor(max_workers=min(int(workers), len(judgeable))) as pool:
            results = list(pool.map(judge_one, judgeable))
    else:
        results = [judge_one(candidate) for candidate in judgeable]

    for result in results:
        counts["judged"] += result.get("judged", 0)
        counts["failed"] += result.get("failed", 0)
        counts["stopped"] += result.get("stopped", 0)
        # The answers, tallied as flat ints so they still SUM across frames
        # the way every other key here does. A caller can OR them into the
        # frame's verdict without re-reading the audit log it just wrote --
        # which is what a live feed needs to report an alarm the moment it is
        # judged.
        outcome = result.get("outcome")
        if outcome == "CONFIRM_TP":
            counts["tp"] += 1
        elif outcome == "CONFIRM_FP":
            counts["fp"] += 1
        elif outcome == "ESCALATE":
            counts["escalated"] += 1
    return counts


def _cv_candidate_for(candidate: dict) -> dict:
    """One candidate's cv block fields.

    It used to fill the gate's answer in here too, for two paths that differed
    only in that. With the gate gone there is one path and no answer to carry.
    """
    cv_candidate = dict(candidate)
    stable_id = cv_candidate.setdefault("ground_truth_id", candidate["candidate_id"])
    cv_candidate.setdefault(
        "execution_id",
        execution_id(
            cv_candidate.get("provenance", {}).get("cv_run_id", LEGACY_RUN_ID), stable_id
        ),
    )
    return cv_candidate


def _truncate_to_limit(
    grouped: dict[str, list[dict]], limit: int, already_judged: set[str]
) -> dict[str, list[dict]]:
    """Drop work beyond `limit` judge calls, in WHOLE FRAMES.

    Applied before the pool starts, so the cap is deterministic. Racing a
    counter inside the pool would make the cap depend on thread scheduling, and
    a cap that depends on scheduling is not a cap.

    The cap is therefore approximate by design: it stops at the last frame that
    fits entirely rather than splitting one to hit the number exactly. At 8.54
    people per alarm frame it can undershoot by up to a frame's worth, which is
    the right trade -- a half-judged frame produces a wrong frame verdict that
    looks like a right one.
    """
    kept: dict[str, list[dict]] = {}
    budget = max(0, limit)
    for frame_name, frame_candidates in grouped.items():
        done = [c for c in frame_candidates if _execution_key(c) in already_judged]
        pending = [c for c in frame_candidates if _execution_key(c) not in already_judged]
        if not pending:
            # Nothing to spend budget on; keeping it costs nothing and keeps
            # the `skipped` count honest.
            kept[frame_name] = frame_candidates
            continue
        if budget <= 0:
            break
        if len(pending) > budget:
            # WHOLE FRAMES ONLY. Taking `pending[:budget]` judged some people in
            # a frame and not others, and `vlm_frame_verdicts` then ORs over the
            # incomplete set -- a frame whose violating worker fell outside the
            # budget reads as "no violation", and the rows are indistinguishable
            # from a fully judged frame afterwards. That is the exact corruption
            # `cli._sample_whole_frames` exists to prevent; a budget must not
            # reintroduce it. Stop instead: an unjudged frame is visibly
            # unjudged, a half-judged one is not.
            break
        kept[frame_name] = done + pending
        budget -= len(pending)
    return kept


def judge_candidates(
    candidates: Iterable[dict],
    *,
    frames_dir: Path,
    config=None,
    few_shot_library=None,
    audit_log_path=None,
    alarm_image_dir=None,
    judge=None,
    limit: int | None = None,
    already_judged: set[str] | None = None,
    max_workers: int | None = None,
    lineage: dict | None = None,
    should_stop: Callable[[], bool] | None = None,
    on_crop_judged: Callable[[], None] | None = None,
) -> dict:
    """Ask the judge about each sampled candidate, writing one audit row each.

    Goes through the ordinary `track1_adapter.run_track1_judge_inference`
    rather than a parallel path, so the judge answers a CV candidate exactly
    the way it answers a live alarm — same crop, same few-shot examples, same
    prompt. Anything else and the comparison measures two different judges.

    The row lands in the SHARED audit log, tagged (see
    `track1_adapter.CV_ROW_KIND`) so it stays filterable. `alarm_id` is the
    candidate id, which is what makes one person in one frame the unit the
    audit adjudicates.

    A candidate CV found nobody in is still judged, on the whole frame: the
    person box is the frame. That is the only way to ask the question at all
    when there is no box to crop to, and it is the one case where the judge
    is shown something materially different from what it would see live.

    WORK IS GROUPED BY FRAME, AND FRAMES RUN CONCURRENTLY

    Candidates are grouped by `frame_name`, so a frame is read from disk once
    however many people are standing in it — four routed workers in one frame
    used to cost four reads of the same 1920x1080 JPEG. The groups run on a
    bounded pool (`config.helmet_cv.max_workers`, default 4): threads because
    the time goes on waiting for Bedrock, bounded because an unbounded
    fan-out at a judge's rate limit turns a slow pass into a failed one.

    Still ONE judge call per person. Batched multi-crop inference was
    measured and deferred, not forgotten — see `cropping.CropBatch`.

    EVERY SAMPLED CANDIDATE IS JUDGED, INCLUDING ONES THE GATE CLEARED

    In the real pipeline only a CV violation reaches the VLM — a cleared
    person simply ends. This pass deliberately does not reproduce that
    routing, because the pilot has to measure the VLM arm's RECALL, and a
    judge that is only ever asked about frames CV already flagged can never
    be observed missing one. Judging the whole gate-blind sample is what
    fills the `judge_only` cell in `audit.cv_vs_judge` and what makes
    `audit.cv_pipeline_confusion` pure computation rather than a second pass.

    So `gate_fired` on a record says what production WOULD route; it does not
    say what was judged here.
    """
    from ..config import get_config
    from ..track1_adapter import run_track1_judge_inference

    config = config or get_config()
    judge = judge or run_track1_judge_inference
    already_judged = already_judged or set()

    judge_kwargs: dict = {"few_shot_library": few_shot_library, "config": config}
    if audit_log_path is not None:
        judge_kwargs["audit_log_path"] = audit_log_path
    if alarm_image_dir is not None:
        judge_kwargs["alarm_image_dir"] = alarm_image_dir
    if lineage is not None:
        judge_kwargs["lineage"] = lineage

    grouped = group_by_frame(candidates)
    if limit is not None:
        grouped = _truncate_to_limit(grouped, limit, already_judged)

    workers = max_workers if max_workers is not None else config.helmet_cv.max_workers
    # `gated` and `gate_calls` are kept at zero rather than dropped: they are
    # read off stored rows written before the first-pass gate was removed
    # (2 Sep), and a missing key there reads as a different kind of record.
    counts = {"judged": 0, "skipped": 0, "failed": 0, "stopped": 0, "gated": 0, "gate_calls": 0, "tp": 0, "fp": 0, "escalated": 0, "no_person": 0}

    def run(frame_candidates: list[dict]) -> dict:
        return _judge_one_frame(
            frame_candidates,
            frames_dir=frames_dir,
            judge=judge,
            judge_kwargs=judge_kwargs,
            already_judged=already_judged,
            should_stop=should_stop,
            on_crop_judged=on_crop_judged,
        )

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for result in pool.map(run, list(grouped.values())):
            for key, value in result.items():
                counts[key] += value

    return counts
