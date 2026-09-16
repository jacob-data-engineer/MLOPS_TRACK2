"""CLI for the helmet frame extractor and the CV gate that runs over it.

`list`/`extract`/`manifest`/`status` read the live alarm bucket, so they need
AWS credentials with read access to s3://safetyai-cadit. `cv` and
everything after it are purely local — extracted frames plus two committed
checkpoints — except `cv-judge`, which calls Bedrock and costs money.

    uv run python -m ppe_pipeline.helmet_cv.cli list --limit 5
    uv run python -m ppe_pipeline.helmet_cv.cli extract --out eval/helmet_frames --limit 50
    uv run python -m ppe_pipeline.helmet_cv.cli extract --out eval/helmet_frames --retry-failed
    uv run python -m ppe_pipeline.helmet_cv.cli manifest --out eval/helmet_frames
    uv run python -m ppe_pipeline.helmet_cv.cli status --out eval/helmet_frames
    uv run python -m ppe_pipeline.helmet_cv.cli cv --out eval/helmet_frames
    uv run python -m ppe_pipeline.helmet_cv.cli cv --out eval/helmet_frames --imgsz 1920
    uv run python -m ppe_pipeline.helmet_cv.cli cv-status --out eval/helmet_frames --cv-run-id RUN
    uv run python -m ppe_pipeline.helmet_cv.cli cv-sample --out eval/helmet_frames --cv-run-id RUN --target 200
    uv run python -m ppe_pipeline.helmet_cv.cli cv-judge --out eval/helmet_frames --cv-run-id RUN --target 200 --confirm-paid

`list` is the dry run: it resolves and filters clips without downloading
anything, which is the cheap way to check a date window before committing to
a pass over 27k clips.

THE DETECTOR IS A CROP PROVIDER, NOT A GATE

`cv` finds people and stops. It reaches no verdict, so there is nothing here
to route: every candidate goes to the judge, and the judge's answers are
aggregated up to a frame verdict by OR. The ported JPSAI cascade this command
used to run was deleted after measurement showed it could not operate on this
footage -- see `thresholds`.

`--imgsz` is the one parameter worth thinking about. Unset means ultralytics'
640, which scales this corpus's ~83px workers to ~28px and costs about half
the persons and 6/7 of the helmets. Since the detector is the only source of
crops, a person missed here is a person the judge is never asked about, so
detector recall is a ceiling on VLM recall. It is recorded in provenance; use
a fresh `--cv-run-id` when you change it.

"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from .clips import (
    DEFAULT_BUCKET,
    DEFAULT_PREFIX,
    DEFAULT_REGION,
    DEFAULT_VIOLATION_TYPE,
    get_s3_client,
    iter_clips,
    snapshot_alarm_ids,
)
from .runner import CANDIDATE_MANIFEST
from .extract import (
    DEFAULT_JPEG_QUALITY,
    DEFAULT_STRIDE_SECONDS,
    DEFAULT_WORKERS,
    UNKNOWN_FIELD,
    build_manifest,
    extract_corpus,
    is_extracted,
    iter_sidecars,
)


def _parse_date(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected YYYY-MM-DD, got {value!r}") from exc


def _add_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument(
        "--violation-type",
        default=DEFAULT_VIOLATION_TYPE,
        help="Key prefix filter, e.g. helmet_violation. Pass 'all' for every type.",
    )
    parser.add_argument("--since", type=_parse_date, help="Only clips starting on/after this UTC date.")
    parser.add_argument("--until", type=_parse_date, help="Only clips starting before this UTC date.")
    parser.add_argument("--limit", type=int, help="Stop after this many clips (the OLDEST "
                                                  "this many -- key order is chronological).")
    parser.add_argument(
        "--require-snapshot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only clips whose alarm has a snapshot/ still. DEFAULT ON: an "
             "alarm with no still has no locatable alarm frame, so it can never "
             "carry a CV claim, and extracting it spends minutes to produce "
             "frames only the VLM arm could use. Measured: only 52%% of clips "
             "sampled across the corpus have one. Pass --no-require-snapshot to "
             "include them anyway.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        help="Take this many clips spread across the whole corpus, ordered by "
             "md5(alarm_id). Use this rather than --limit for anything measured: "
             "--limit returns the oldest N, which over-represents a few days and "
             "cameras.",
    )


def _selected_clips(args):
    client = get_s3_client(args.region)
    clips = iter_clips(
        client,
        bucket=args.bucket,
        prefix=args.prefix,
        violation_type=None if args.violation_type == "all" else args.violation_type,
        start_on_or_after=args.since,
        start_before=args.until,
    )
    if getattr(args, "require_snapshot", False):
        # BEFORE sampling, so the sample is drawn from clips that can actually
        # carry a CV claim. Filtering afterwards would silently shrink a
        # `--sample 110` to whatever fraction happened to have a still -- which
        # is what produced 61 alarm frames from 130 clips.
        viable = snapshot_alarm_ids(
            client,
            bucket=args.bucket,
            violation_type=None if args.violation_type == "all" else args.violation_type,
        )
        print(f"{len(viable)} alarms have a snapshot still", file=sys.stderr)
        clips = (clip for clip in clips if clip.alarm_id in viable)
    if getattr(args, "sample", None):
        clips = _md5_sample(clips, args.sample)
    if args.limit is not None:
        clips = (clip for _, clip in zip(range(args.limit), clips, strict=False))
    return client, clips


def _md5_sample(clips, count: int):
    """`count` clips spread across the whole corpus, deterministically.

    `--limit` takes the FIRST N in key order, and key order is chronological, so
    it returns the oldest N — the first pilot's twenty clips were all from two
    days and a handful of cameras. That is fine for a smoke test and wrong for
    a precision estimate, which needs the sample to look like the corpus.

    Ordered by `md5(alarm_id)` rather than a seeded shuffle: it is the house
    sampling convention (`eval/gold_set.py`, and NOT `hash()`, which is
    per-process randomised), and it needs no seed to reproduce. The membership
    can shift if the corpus grows, so the chosen alarm ids are printed —
    a sample nobody wrote down is a sample nobody can re-audit.
    """
    import hashlib

    ordered = sorted(clips, key=lambda c: hashlib.md5(c.alarm_id.encode("utf-8")).hexdigest())
    picked = ordered[: max(0, count)]
    print(
        f"sampled {len(picked)} of {len(ordered)} clips by md5(alarm_id)",
        file=sys.stderr,
    )
    for clip in picked:
        print(f"sampled {clip.alarm_id}", file=sys.stderr)
    return picked


def _cmd_list(args) -> int:
    _, clips = _selected_clips(args)
    count = 0
    for clip in clips:
        count += 1
        print(f"{clip.alarm_id}  {clip.start:%Y-%m-%d %H:%M:%S}Z  {clip.duration_seconds:6.1f}s  {clip.key}")
    print(f"\n{count} clip(s)", file=sys.stderr)
    return 0


def _cmd_extract(args) -> int:
    out_dir = Path(args.out)
    client, clips = _selected_clips(args)

    counts = extract_corpus(
        clips,
        client=client,
        bucket=args.bucket,
        out_dir=out_dir,
        workers=args.workers,
        retry_failed=args.retry_failed,
        re_extract=args.re_extract,
        site=args.site,
        camera=args.camera,
        stride_seconds=args.stride,
        jpeg_quality=args.jpeg_quality,
        max_frames=args.max_frames_per_clip,
    )
    print(
        f"ok={counts['ok']} failed={counts['failed']} skipped={counts['skipped']} -> {out_dir}",
        file=sys.stderr,
    )

    if not args.no_manifest:
        rows = build_manifest(out_dir)
        print(f"manifest: {rows} frame(s) -> {out_dir / 'frames.jsonl'}", file=sys.stderr)
    return 0


def _cmd_manifest(args) -> int:
    out_dir = Path(args.out)
    rows = build_manifest(out_dir, Path(args.manifest) if args.manifest else None)
    print(f"{rows} frame(s)", file=sys.stderr)
    return 0


def _cmd_status(args) -> int:
    out_dir = Path(args.out)
    ok = failed = frames = 0
    reasons: dict[str, int] = {}
    for sidecar in iter_sidecars(out_dir):
        if sidecar.get("status") == "ok":
            ok += 1
            frames += len(sidecar.get("frames", ()))
        else:
            failed += 1
            reason = (sidecar.get("error") or "unknown").split(":")[0]
            reasons[reason] = reasons.get(reason, 0) + 1

    print(f"clips extracted : {ok}")
    print(f"clips failed    : {failed}")
    print(f"frames written  : {frames}")
    for reason, count in sorted(reasons.items(), key=lambda item: -item[1]):
        print(f"  {reason}: {count}")
    return 0


def _sample_whole_frames(candidates: list[dict], target: int, rate: int | None):
    """Select whole FRAMES, and every crop of the ones selected.

    **Sampling by `candidate_id` is wrong now and would corrupt both arms.**
    Truth is keyed by the frame and the judge's answers are OR'd across a
    frame's crops, so a candidate-level sample judges SOME people in a frame and
    not others -- and the OR then runs over an incomplete set. A frame whose
    violating worker happened not to be sampled reads as "no violation",
    understating the VLM's violations and, through the shared human decision,
    distorting the CV arm too. At 8.54 people per alarm frame that is not an
    edge case, it is most frames.

    So the predicate keys on the FRAME STEM. `md5` and the same `is_sampled`
    as before, just applied to the unit that is actually being measured.
    """
    from .records import frame_stem
    from .sample import is_sampled, rate_for_target

    by_frame: dict[str, list[dict]] = {}
    for candidate in candidates:
        by_frame.setdefault(frame_stem(candidate["frame_name"]), []).append(candidate)

    rate = rate or rate_for_target(len(by_frame), target)
    frames = [stem for stem in by_frame if is_sampled(stem, rate)]
    picked = [c for stem in frames for c in by_frame[stem]]
    return picked, frames, by_frame, rate


def _cmd_cv_sample(args) -> int:
    from .runner import iter_candidates

    frames_dir = Path(args.out)
    candidates = list(iter_candidates(frames_dir, cv_run_id=args.cv_run_id))
    picked, frames, by_frame, rate = _sample_whole_frames(candidates, args.target, args.rate)

    print(
        f"rate=1-in-{rate}  frames={len(frames)}/{len(by_frame)}  "
        f"crops={len(picked)}/{len(candidates)}",
        file=sys.stderr,
    )
    # Printed because it is the number that says whether the sample is honest:
    # the sampled snapshot share should track the corpus's, since the
    # predicate never looks at the frame role. Read from the constant, not a
    # literal: this check silently reported 0/0 for a while because the string
    # here was left as "snapshot" after the role was renamed to "alarm".
    from .records import FRAME_ROLE_ALARM

    snap = sum(1 for c in picked if c.get("frame_role") == FRAME_ROLE_ALARM)
    corpus_snap = sum(1 for c in candidates if c.get("frame_role") == FRAME_ROLE_ALARM)
    print(
        f"alarm frames: {snap}/{len(picked)} sampled vs "
        f"{corpus_snap}/{len(candidates)} corpus",
        file=sys.stderr,
    )
    for candidate in picked:
        print(candidate["candidate_id"])
    return 0


def _cmd_cv_judge(args) -> int:
    """Judge the sampled candidates. Costs money and needs Bedrock."""
    from .. import storage
    from ..config import get_config
    from ..few_shot import load_few_shot_library
    from .records import execution_id
    from .runner import iter_candidates, judge_candidates, run_metadata_path

    frames_dir = Path(args.out)
    candidates = list(iter_candidates(frames_dir, cv_run_id=args.cv_run_id))
    if not candidates:
        print(f"no candidates found for run {args.cv_run_id!r}", file=sys.stderr)
        return 2
    execution_ids = [candidate.get("execution_id") for candidate in candidates]
    if any(not value for value in execution_ids) or len(set(execution_ids)) != len(execution_ids):
        print("candidate manifest has missing or duplicate execution identities", file=sys.stderr)
        return 2
    if any(candidate.get("provenance", {}).get("cv_run_id") != args.cv_run_id for candidate in candidates):
        print("candidate manifest mixes cv_run_id values", file=sys.stderr)
        return 2
    metadata_path = run_metadata_path(frames_dir, args.cv_run_id)
    try:
        import json

        run_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        print(f"missing or unreadable immutable run metadata at {metadata_path}", file=sys.stderr)
        return 2
    expected_provenance = run_metadata.get("provenance")
    if any(candidate.get("provenance") != expected_provenance for candidate in candidates):
        print("candidate provenance does not match immutable run metadata", file=sys.stderr)
        return 2
    if any(
        candidate.get("execution_id")
        != execution_id(args.cv_run_id, candidate.get("ground_truth_id") or "")
        for candidate in candidates
    ):
        print("candidate execution identity does not match its run and ground truth", file=sys.stderr)
        return 2
    picked, frames, by_frame, rate = _sample_whole_frames(candidates, args.target, args.rate)
    print(
        f"rate=1-in-{rate}  frames={len(frames)}/{len(by_frame)}  "
        f"crops={len(picked)}/{len(candidates)}",
        file=sys.stderr,
    )
    if not picked:
        print("sampling selected zero candidates; adjust --rate or --target", file=sys.stderr)
        return 2

    existing, _ = storage.get_storage().read_json(args.audit_log, default=[])
    existing_rows = existing if isinstance(existing, list) else []
    identities: dict[str, str | None] = {}
    for row in existing_rows:
        if not isinstance(row, dict) or row.get("row_kind") != "helmet_cv":
            continue
        cv = row.get("cv") if isinstance(row.get("cv"), dict) else {}
        execution = cv.get("execution_id") or row.get("alarm_id")
        ground_truth = cv.get("ground_truth_id") or cv.get("candidate_id")
        if execution in identities and identities[execution] != ground_truth:
            print(f"audit log has conflicting identity for {execution!r}", file=sys.stderr)
            return 2
        identities[execution] = ground_truth
    already = set()
    for row in existing_rows:
        if not isinstance(row, dict) or row.get("row_kind") != "helmet_cv":
            continue
        cv = row.get("cv") if isinstance(row.get("cv"), dict) else {}
        if cv.get("cv_run_id") != args.cv_run_id:
            continue
        already.add(cv.get("execution_id") or row.get("alarm_id"))

    pending = sum(1 for candidate in picked if candidate.get("execution_id") not in already)
    # `--limit` is a hard cap on pending calls (`runner._truncate_to_limit`
    # spends a budget, it does not race a counter), so the count printed
    # before the operator confirms must respect it. Printing the uncapped
    # number would overstate the bill they are agreeing to.
    if args.limit is not None:
        pending = min(pending, max(0, args.limit))
    print(f"pending paid judge calls: {pending}", file=sys.stderr)
    if not args.confirm_paid:
        print("refusing paid run without --confirm-paid", file=sys.stderr)
        return 2

    counts = judge_candidates(
        picked,
        frames_dir=frames_dir,
        config=get_config(),
        few_shot_library=load_few_shot_library(Path(args.few_shot_dir) / "manifest.json"),
        audit_log_path=args.audit_log,
        limit=args.limit,
        already_judged=already,
    )
    print(
        f"judged={counts['judged']} skipped={counts['skipped']} failed={counts['failed']}",
        file=sys.stderr,
    )
    return 0


def _cmd_cv_status(args) -> int:
    from .runner import iter_cv_sidecars

    alarms = candidates = no_person = 0
    roles: dict[str, int] = {}
    per_frame: dict[str, int] = {}
    for sidecar in iter_cv_sidecars(Path(args.out), args.cv_run_id):
        alarms += 1
        for record in sidecar.get("candidates", ()):
            candidates += 1
            role = str(record.get("frame_role"))
            roles[role] = roles.get(role, 0) + 1
            if record.get("person_box") is None:
                no_person += 1
            frame = str(record.get("frame_name"))
            per_frame[frame] = per_frame.get(frame, 0) + 1

    frames = len(per_frame) or 1
    print(f"alarms detected : {alarms}")
    print(f"frames          : {len(per_frame)}")
    print(f"candidates      : {candidates}")
    # The detector is the VLM's only source of crops, so a frame it found
    # nobody in is a frame the judge only ever sees whole. Printed because it
    # is the ceiling on VLM recall, not a curiosity.
    print(f"no person found : {no_person} ({100.0 * no_person / max(1, candidates):.1f}%)")
    print(f"people/frame    : {candidates / frames:.2f} mean")
    print("frame roles:")
    for role, count in sorted(roles.items(), key=lambda item: -item[1]):
        print(f"  {role}: {count}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract named frames from raw UAT alarm clips.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="Resolve clips without downloading (dry run).")
    _add_selection_args(list_parser)
    list_parser.set_defaults(func=_cmd_list)

    extract_parser = subparsers.add_parser("extract", help="Download, sample, and write frames.")
    _add_selection_args(extract_parser)
    extract_parser.add_argument("--out", required=True, help="Output directory.")
    extract_parser.add_argument("--stride", type=float, default=DEFAULT_STRIDE_SECONDS,
                                help="Seconds between sampled frames.")
    extract_parser.add_argument("--jpeg-quality", type=int, default=DEFAULT_JPEG_QUALITY)
    extract_parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    extract_parser.add_argument("--site", default=UNKNOWN_FIELD,
                                help="Site name for the frame name. No mapping exists yet.")
    extract_parser.add_argument("--camera", default=UNKNOWN_FIELD,
                                help="Camera name for the frame name. No mapping exists yet.")
    extract_parser.add_argument("--max-frames-per-clip", type=int)
    extract_parser.add_argument("--retry-failed", action="store_true",
                                help="Re-attempt clips whose sidecar recorded a failure.")
    extract_parser.add_argument("--re-extract", action="store_true",
                                help="Re-extract clips that already have an OK sidecar. Needed "
                                     "after a sidecar schema change: a v1 sidecar is wrong rather "
                                     "than done, because it holds no snapshot frame.")
    extract_parser.add_argument("--no-manifest", action="store_true",
                                help="Skip rebuilding frames.jsonl afterwards.")
    extract_parser.set_defaults(func=_cmd_extract)

    manifest_parser = subparsers.add_parser("manifest", help="Merge sidecars into frames.jsonl.")
    manifest_parser.add_argument("--out", required=True)
    manifest_parser.add_argument("--manifest", help="Manifest path (default <out>/frames.jsonl).")
    manifest_parser.set_defaults(func=_cmd_manifest)

    status_parser = subparsers.add_parser("status", help="Summarise what has been extracted.")
    status_parser.add_argument("--out", required=True)
    status_parser.set_defaults(func=_cmd_status)

    cv_status_parser = subparsers.add_parser("cv-status", help="Summarise what the gate decided.")
    cv_status_parser.add_argument("--out", required=True)
    cv_status_parser.add_argument("--cv-run-id", required=True)
    cv_status_parser.set_defaults(func=_cmd_cv_status)

    cv_sample_parser = subparsers.add_parser(
        "cv-sample", help="Print the candidate ids the judge would be asked about (dry run)."
    )
    cv_sample_parser.add_argument("--out", required=True)
    cv_sample_parser.add_argument("--cv-run-id", required=True)
    cv_sample_parser.add_argument("--target", type=int, default=200,
                                  help="Aim for roughly this many FRAMES. Every crop of a "
                                       "selected frame is judged, because the frame verdict "
                                       "is an OR across them.")
    cv_sample_parser.add_argument("--rate", type=int,
                                  help="Force a 1-in-N rate instead of deriving it from --target.")
    cv_sample_parser.set_defaults(func=_cmd_cv_sample)

    cv_judge_parser = subparsers.add_parser(
        "cv-judge", help="Judge the sampled candidates. Calls Bedrock; costs money."
    )
    cv_judge_parser.add_argument("--out", required=True)
    cv_judge_parser.add_argument("--cv-run-id", required=True)
    cv_judge_parser.add_argument(
        "--confirm-paid", action="store_true", help="Confirm that Bedrock charges are authorised."
    )
    cv_judge_parser.add_argument("--target", type=int, default=200)
    cv_judge_parser.add_argument("--rate", type=int)
    cv_judge_parser.add_argument("--limit", type=int, help="Hard cap on judge calls this run.")
    cv_judge_parser.add_argument("--audit-log", default="eval/track1_audit_log.json",
                                 help="Shared with live alarms; CV rows are tagged row_kind.")
    cv_judge_parser.add_argument("--few-shot-dir", default="few_shot")
    cv_judge_parser.set_defaults(func=_cmd_cv_judge)


    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
