"""Turning alarm clips into named, sampled frames.

The unit the audit adjudicates has to be a *frame*, not a clip: a 60-second
`helmet_violation` clip contains violating and non-violating frames, and
scoring CV against the judge only means anything if both saw the same
picture. So extraction is the step that fixes the unit of comparison, and
every frame carries enough identity to be traced back to its alarm.

FRAME NAMES

    unk__unk__20260427T070224Z__726f88cb-a3d0-4907-9d8d-111055ebc676__f000450.jpg
    └site┘└cam┘└── frame UTC ──┘└──────────── alarm id ────────────┘ └ index ┘

`__` separates fields because a single `_` already appears inside the source
timestamps. The charset stays `[A-Za-z0-9._-]`, which satisfies
`track1_adapter.safe_image_stem` and `storage.validate_key`, and is legal on
Windows — `uncaptured_hazard/`'s own names use colons and are not.

Site and camera are `unk` placeholders, not omitted fields: the bucket
carries no camera mapping (see `clips`), and a name with the slot already in
it can be regenerated when one appears without every downstream reference
changing shape.

The frame index is in the name as well as the timestamp because a sub-second
stride would otherwise put two frames in the same second under one name.

TIMESTAMPS ARE COMPUTED, NOT READ

A frame's UTC is `clip_start + index / fps`, not OpenCV's `CAP_PROP_POS_MSEC`.
Backends disagree on whether that property reports the frame just decoded or
the next one, and a silently-off-by-one-frame timestamp is worse than an
assumption stated out loud: this assumes constant frame rate, which CCTV
exports are.

LAYOUT

    <out_dir>/frames/<alarm_id>/<frame name>.jpg
    <out_dir>/index/<alarm_id>.json

Frames are grouped per alarm because the helmet corpus at a 1s stride is
~1.6M files, and one flat directory of that size is painful on NTFS. The
per-alarm sidecar is both the manifest shard and the resume marker — a clip
whose sidecar exists is skipped, so a killed run resumes by re-running.
Sidecars are written per clip rather than appended to one manifest so that
concurrent workers never contend for a lock; `build_manifest` merges them.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from .clips import ClipRef
from .records import FRAME_ROLE_ALARM, FRAME_ROLE_CLIP

logger = logging.getLogger(__name__)

#: Bumped from 1 when frames gained a role and the snapshot still joined the
#: corpus: a v1 sidecar has no snapshot frame in it at all.
SCHEMA_VERSION = 2

#: How closely a clip frame must agree with the annotated still before we
#: believe it is the frame the still was drawn on.
#:
#: **0.94, not 0.90.** The true match scores 0.95-0.99 and a wrong frame in a
#: moving scene reaches 0.92, so 0.90 sat INSIDE the overlap: on a 130-clip
#: sample six of 61 accepted matches fell in 0.90-0.94, i.e. 10% of the CV
#: arm's entire population was in the band where a wrong frame is possible.
#: Below this we decline to name an alarm frame rather than label the wrong one,
#: because a wrong alarm frame scores CV against a frame it never saw — which is
#: worse than a smaller sample, since it is invisible in the result.
ALARM_FRAME_MIN_AGREEMENT = 0.94

#: Per-pixel tolerance when comparing a clip frame against the annotated still.
#: JPEG round-tripping moves values by a few levels, so an exact compare finds
#: nothing.
_ALARM_FRAME_PIXEL_TOLERANCE = 8

DEFAULT_STRIDE_SECONDS = 1.0
DEFAULT_JPEG_QUALITY = 90
DEFAULT_WORKERS = 4
UNKNOWN_FIELD = "unk"

# Believed only if the container's own figure is sane. A broken or missing
# CAP_PROP_FPS reads as 0 or nan, and dividing by it would put every frame in
# the clip at the same timestamp.
#
# The floor is deliberately below 1: this corpus is ~1fps CCTV export, not
# 25fps video. Six sampled helmet clips ran 0.974-1.855 fps over 35-96
# frames, so a 24fps floor would reject every real value in the bucket.
_MIN_PLAUSIBLE_FPS = 0.2
_MAX_PLAUSIBLE_FPS = 240.0

# Last resort, and chosen for THIS corpus rather than for video in general —
# see above. Only reached when the container reports nothing usable *and* the
# frame count is unreadable too.
_ASSUMED_FPS = 1.0

_SAFE_FIELD = str.maketrans({" ": "-", "/": "-", "\\": "-", ":": "-", "_": "-"})


def safe_field(value: str) -> str:
    """A name component reduced to `[A-Za-z0-9.-]`, never empty.

    Underscores are folded to `-` as well: they are the field separator's own
    character, and one in a site name would fake an extra field.
    """
    cleaned = "".join(
        ch for ch in str(value).translate(_SAFE_FIELD) if ch.isalnum() or ch in ".-"
    ).strip(".-")
    return cleaned or UNKNOWN_FIELD


@dataclass(frozen=True)
class FrameRecord:
    """One extracted frame, and everything needed to place it without
    re-parsing its name or reopening its clip."""

    frame_name: str
    relative_path: str
    alarm_id: str
    clip_key: str
    violation_type: str
    #: `snapshot` for the still production CV actually fired on, `clip` for a
    #: frame we decoded out of the video ourselves. The CV arm is scored on
    #: the first only -- production made no claim about the rest of the clip.
    frame_role: str
    site: str
    camera: str
    frame_index: int
    offset_ms: int
    frame_utc: str
    width: int
    height: int


@dataclass
class ClipExtraction:
    """The sidecar for one clip: its outcome, the settings that produced it,
    and the frames it yielded."""

    schema: int
    alarm_id: str
    clip_key: str
    violation_type: str
    clip_start_utc: str
    clip_end_utc: str
    status: str  # "ok" | "failed"
    fps: float | None = None
    fps_source: str | None = None  # "container" | "derived" | "assumed"
    frames_decoded: int = 0
    #: Which clip frame production CV fired on, and how confidently it was
    #: identified. `None` means no still existed or no frame matched well
    #: enough, and this alarm then contributes to the VLM arm only -- see
    #: `_locate_alarm_frame`. The agreement is recorded so nobody has to trust
    #: the match on faith.
    alarm_frame_index: int | None = None
    alarm_frame_agreement: float | None = None
    #: The SECOND-best agreement. Recorded, never enforced -- see
    #: `_locate_alarm_frame`. A match at 0.96 with a runner-up at 0.95 is a tie
    #: between near-identical frames; one at 0.96 with a runner-up at 0.80 is
    #: unambiguous. Only the sidecar can tell them apart afterwards.
    alarm_frame_runner_up: float | None = None
    error: str | None = None
    params: dict = field(default_factory=dict)
    frames: list[dict] = field(default_factory=list)


def frame_name(
    *,
    site: str,
    camera: str,
    frame_utc: datetime,
    alarm_id: str,
    frame_index: int,
) -> str:
    stamp = frame_utc.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{safe_field(site)}__{safe_field(camera)}__{stamp}__{alarm_id}__f{frame_index:06d}.jpg"


def _resolve_fps(raw_fps: float, *, frame_count: float, duration_seconds: float) -> tuple[float, str]:
    """This clip's frame rate, and where the figure came from.

    Falls back to the clip's own decoded length over the span in its
    filename, which is a real measurement rather than a guess: across six
    sampled helmet clips it tracked the container's figure to within 0.6%.
    Only when the frame count is unreadable too does a bare constant apply.
    """
    if _MIN_PLAUSIBLE_FPS <= raw_fps <= _MAX_PLAUSIBLE_FPS:
        return raw_fps, "container"

    if frame_count > 0 and duration_seconds > 0:
        derived = frame_count / duration_seconds
        if _MIN_PLAUSIBLE_FPS <= derived <= _MAX_PLAUSIBLE_FPS:
            return derived, "derived"

    return _ASSUMED_FPS, "assumed"


def sidecar_path(out_dir: Path, alarm_id: str) -> Path:
    return out_dir / "index" / f"{alarm_id}.json"


def is_extracted(out_dir: Path, alarm_id: str) -> bool:
    return sidecar_path(out_dir, alarm_id).is_file()


def _download_to_temp(client, bucket: str, key: str) -> Path:
    """The clip on local disk. OpenCV needs a seekable path — it cannot read
    a streaming body — so the download is not avoidable."""
    handle = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    try:
        with handle:
            client.download_fileobj(bucket, key, handle)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise
    return Path(handle.name)


def _sample_frames(
    video_path: Path,
    clip: ClipRef,
    *,
    frames_dir: Path,
    site: str,
    camera: str,
    stride_seconds: float,
    jpeg_quality: int,
    max_frames: int | None,
) -> tuple[list[FrameRecord], float, str, int]:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {clip.key}")

    try:
        fps, fps_source = _resolve_fps(
            float(capture.get(cv2.CAP_PROP_FPS) or 0.0),
            frame_count=float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0),
            duration_seconds=clip.duration_seconds,
        )
        # At ~1fps the default 1s stride keeps every frame. That is the
        # intended behaviour for this corpus — a clip is only 35-96 frames —
        # but it means `--stride` downsamples nothing until it exceeds ~1.0.
        step = max(1, round(fps * stride_seconds))

        records: list[FrameRecord] = []
        index = 0
        while True:
            # grab() advances without decoding. Skipped frames cost demuxing
            # only, which is most of the win at a 1s stride over 25fps
            # footage — 24 of every 25 frames are never decoded.
            if not capture.grab():
                break
            if index % step == 0 and (max_frames is None or len(records) < max_frames):
                ok, frame = capture.retrieve()
                if ok and frame is not None:
                    records.append(
                        _write_frame(
                            frame,
                            clip=clip,
                            frames_dir=frames_dir,
                            site=site,
                            camera=camera,
                            index=index,
                            fps=fps,
                            jpeg_quality=jpeg_quality,
                        )
                    )
            index += 1

        return records, fps, fps_source, index
    finally:
        capture.release()


def _write_frame(
    frame,
    *,
    clip: ClipRef,
    frames_dir: Path,
    site: str,
    camera: str,
    index: int,
    fps: float,
    jpeg_quality: int,
    frame_role: str = FRAME_ROLE_CLIP,
) -> FrameRecord:
    import cv2

    offset_ms = round(index * 1000.0 / fps)
    frame_utc = clip.start + timedelta(milliseconds=offset_ms)
    name = frame_name(
        site=site, camera=camera, frame_utc=frame_utc, alarm_id=clip.alarm_id, frame_index=index
    )
    destination = frames_dir / name
    if not cv2.imwrite(str(destination), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]):
        raise RuntimeError(f"failed to write {destination}")

    height, width = frame.shape[:2]
    return FrameRecord(
        frame_name=name,
        relative_path=f"frames/{clip.alarm_id}/{name}",
        alarm_id=clip.alarm_id,
        clip_key=clip.key,
        violation_type=clip.violation_type,
        frame_role=frame_role,
        site=safe_field(site),
        camera=safe_field(camera),
        frame_index=index,
        offset_ms=offset_ms,
        frame_utc=frame_utc.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        width=width,
        height=height,
    )


def _write_frame_at_index(
    video_path: Path,
    index: int,
    *,
    clip: ClipRef,
    frames_dir: Path,
    site: str,
    camera: str,
    fps: float,
    jpeg_quality: int,
) -> FrameRecord | None:
    """Decode ONE frame by index and write it, for a full-scan alarm frame.

    Seeks rather than re-reading the whole clip, and falls back to a sequential
    read when the container will not seek accurately — some of these are
    variable-frame-rate hevc, where `CAP_PROP_POS_FRAMES` lands nearby rather
    than exactly. An off-by-one here would write the WRONG frame under the
    alarm role, so the sequential path is the safe default when seeking is not
    exact.
    """
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        landed = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        if not ok or landed != index:
            capture.release()
            capture = cv2.VideoCapture(str(video_path))
            for _ in range(index):
                if not capture.read()[0]:
                    return None
            ok, frame = capture.read()
        if not ok:
            return None
    finally:
        capture.release()

    return _write_frame(
        frame,
        clip=clip,
        frames_dir=frames_dir,
        site=site,
        camera=camera,
        index=index,
        fps=fps,
        jpeg_quality=jpeg_quality,
        frame_role=FRAME_ROLE_ALARM,
    )


def _scan_video_for_still(video_path: Path, still) -> tuple[int, float, float | None] | None:
    """Best-matching frame over the WHOLE clip, by the same pixel agreement.

    Decodes every frame rather than the sampled subset. Local and already
    downloaded, so the cost is CPU on a clip that is open anyway; the benefit is
    the alarm frames the stride sieve drops.

    Returns `(frame_index, agreement, runner_up)`, or `None` if nothing decoded.
    """
    import cv2
    import numpy as np

    capture = cv2.VideoCapture(str(video_path))
    scores: list[tuple[float, int]] = []
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if grey.shape == still.shape:
                difference = np.abs(still.astype(np.int16) - grey.astype(np.int16))
                scores.append(
                    (float((difference < _ALARM_FRAME_PIXEL_TOLERANCE).mean()), index)
                )
            index += 1
    finally:
        capture.release()

    if not scores:
        return None
    scores.sort(reverse=True)
    return scores[0][1], scores[0][0], (scores[1][0] if len(scores) > 1 else None)


def _locate_alarm_frame(
    *,
    clip: ClipRef,
    client,
    bucket: str,
    frames_dir: Path,
    records: list[FrameRecord],
    out_dir: Path,
    video_path: Path | None = None,
    site: str = UNKNOWN_FIELD,
    camera: str = UNKNOWN_FIELD,
    fps: float = 1.0,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    annotated: bytes | None = None,
) -> tuple[int | None, float | None, float | None]:
    """Find which CLEAN clip frame production CV fired on.

    THE SNAPSHOT STILL CANNOT BE USED DIRECTLY. It carries production's own
    bounding box and the literal word "violation" burned into the pixels, so
    feeding it to the judge would hand over the answer in text, showing it to a
    reviewer would defeat the blind, and running YOLO on it would put a
    high-contrast rectangle in front of a person detector.

    So the still is used only as a KEY. Its source frame is in `raw_video/`
    already, clean, and it is recoverable: the two images differ only by the
    drawn overlay, which is a few thousand pixels out of two million. Scoring
    each clip frame by the fraction of pixels within a tolerance of the still
    picks it out -- 0.95-0.99 for the true frame against 0.72-0.92 for a wrong
    one where the scene has moved.

    **The alarm frame is NOT frame 0.** Measured over 18 alarms it was frame 0
    nine times, and otherwise anywhere up to frame 32: the clip is a window
    around the alarm, not a clip that starts at it. Assuming frame 0 would
    mislabel half the corpus.

    Where several frames tie, they agree with each other to 98-99% too, so the
    choice does not matter; where it does matter the scene has moved and the
    gap is decisive. Below `ALARM_FRAME_MIN_AGREEMENT` we return `None` rather
    than guess.

    Returns `(frame_index, agreement, runner_up_agreement)`, or all-`None` when
    there is no still or no confident match. The annotated still is kept OUTSIDE the frames
    tree (`annotated/<alarm_id>.jpg`) so the evidence of what production drew
    survives, unreachable by the frame endpoint, which only ever resolves
    `frames/<alarm_id>/<name>`.

    `annotated` is the still's BYTES when the caller already holds them. The
    live feed reads the same object to recover the vendor's boxes, so it was
    paying for the same ~0.9MB object twice per alarm; handing them over costs
    nothing and the file is still written either way, because it is evidence.
    Empty or absent means "fetch it", which is also how a caller whose own
    fetch failed falls back onto this one and its failure handling.
    """
    import cv2
    import numpy as np

    key = clip.snapshot_key
    if annotated:
        body = annotated
    else:
        try:
            body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
        except Exception as exc:  # noqa: BLE001 - a missing still is not fatal
            logger.info("no snapshot for %s (%s): %s", clip.alarm_id, key, exc)
            return None, None, None

    annotated_dir = out_dir / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    (annotated_dir / f"{clip.alarm_id}.jpg").write_bytes(body)

    still = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if still is None:
        logger.warning("snapshot for %s did not decode", clip.alarm_id)
        return None, None, None

    scores: list[tuple[float, int]] = []
    for record in records:
        frame = cv2.imread(str(frames_dir / record.frame_name), cv2.IMREAD_GRAYSCALE)
        if frame is None or frame.shape != still.shape:
            continue
        difference = np.abs(still.astype(np.int16) - frame.astype(np.int16))
        scores.append((float((difference < _ALARM_FRAME_PIXEL_TOLERANCE).mean()), record.frame_index))

    if not scores:
        logger.warning("no comparable frames for %s", clip.alarm_id)
        return None, None, None

    scores.sort(reverse=True)
    best_agreement, best_index = scores[0]
    runner_up = scores[1][0] if len(scores) > 1 else None

    if best_agreement < ALARM_FRAME_MIN_AGREEMENT and video_path is not None:
        # THE SAMPLED FRAMES ARE A SIEVE, NOT THE VIDEO.
        #
        # The stride is 1.0s and the corpus median is 4.67 fps (only 6% of
        # clips are the ~1 fps the first twenty suggested; 17% are 15+), so a
        # sampled set is roughly one frame in five. The alarm frame is ONE
        # specific frame, so most of the time it simply is not in the sample --
        # measured, 31% of clips that HAVE a still matched nothing above 0.94.
        #
        # It is in the video though. Two failures checked by decoding every
        # frame scored 0.993 and 0.980 against 0.934 and 0.762 over the sample.
        # So when the sieve misses, scan the whole clip: the video is already
        # downloaded and open, the scan is local, and the alternative is
        # throwing away a third of the CV arm's population.
        scanned = _scan_video_for_still(video_path, still)
        if scanned is not None:
            index, agreement, runner = scanned
            if agreement >= ALARM_FRAME_MIN_AGREEMENT:
                logger.info(
                    "alarm frame for %s found by full scan at frame %d (%.3f); "
                    "the %.3f best in the sampled set would have been discarded",
                    clip.alarm_id, index, agreement, best_agreement,
                )
                return index, agreement, runner

    if best_agreement < ALARM_FRAME_MIN_AGREEMENT:
        logger.warning(
            "no confident alarm frame for %s (best agreement %.3f); it will score the VLM arm only",
            clip.alarm_id,
            best_agreement,
        )
        return None, None, None
    # The runner-up is RECORDED, not enforced. A margin requirement would be
    # wrong here: this footage is ~1fps and often static, so the two frames
    # either side of the alarm can legitimately tie at 0.98 — and when they do,
    # either is the right answer. The margin is only worth having so an auditor
    # can see whether a given match was clear-cut or a coin flip.
    return best_index, best_agreement, runner_up


def extract_clip(
    clip: ClipRef,
    *,
    client,
    bucket: str,
    out_dir: Path,
    site: str = UNKNOWN_FIELD,
    camera: str = UNKNOWN_FIELD,
    stride_seconds: float = DEFAULT_STRIDE_SECONDS,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    max_frames: int | None = None,
    annotated: bytes | None = None,
) -> ClipExtraction:
    """Extract one clip and write its sidecar.

    Never raises for a bad clip. A download or decode failure is recorded as
    a `failed` sidecar with its reason and the run continues — over 27k clips
    some proportion will be truncated or zero-length, and one of them must
    not take down a multi-hour pass. A failed sidecar still counts as done,
    so `--retry-failed` is what re-attempts them.

    `annotated` is passed through to `_locate_alarm_frame`: the annotated
    still, when the caller has already fetched it, so it is not fetched twice.
    """
    params = {
        "stride_seconds": stride_seconds,
        "jpeg_quality": jpeg_quality,
        "site": safe_field(site),
        "camera": safe_field(camera),
        "max_frames": max_frames,
    }
    extraction = ClipExtraction(
        schema=SCHEMA_VERSION,
        alarm_id=clip.alarm_id,
        clip_key=clip.key,
        violation_type=clip.violation_type,
        clip_start_utc=clip.start.isoformat().replace("+00:00", "Z"),
        clip_end_utc=clip.end.isoformat().replace("+00:00", "Z"),
        status="ok",
        params=params,
    )

    frames_dir = out_dir / "frames" / clip.alarm_id
    video_path: Path | None = None
    try:
        frames_dir.mkdir(parents=True, exist_ok=True)
        video_path = _download_to_temp(client, bucket, clip.key)
        records, fps, fps_source, decoded = _sample_frames(
            video_path,
            clip,
            frames_dir=frames_dir,
            site=site,
            camera=camera,
            stride_seconds=stride_seconds,
            jpeg_quality=jpeg_quality,
            max_frames=max_frames,
        )
        extraction.fps = fps
        extraction.fps_source = fps_source
        extraction.frames_decoded = decoded
        alarm_index, agreement, runner_up = _locate_alarm_frame(
            clip=clip, client=client, bucket=bucket, frames_dir=frames_dir,
            records=records, out_dir=out_dir, video_path=video_path,
            site=site, camera=camera, fps=fps, jpeg_quality=jpeg_quality,
            annotated=annotated,
        )
        if alarm_index is not None:
            known = {record.frame_index for record in records}
            if alarm_index not in known:
                # A full-scan hit. The frame is NOT in the sampled set, so
                # stamping a role over `records` would find nothing and leave
                # `alarm_frame_index` pointing at a frame that does not exist on
                # disk -- recorded as located, unusable in practice, and silent.
                # Decode and write it.
                extra = _write_frame_at_index(
                    video_path, alarm_index, clip=clip, frames_dir=frames_dir,
                    site=site, camera=camera, fps=fps, jpeg_quality=jpeg_quality,
                )
                if extra is None:
                    alarm_index = agreement = runner_up = None
                else:
                    records = [*records, extra]
            if alarm_index is not None:
                records = [
                    replace(record, frame_role=FRAME_ROLE_ALARM)
                    if record.frame_index == alarm_index
                    else record
                    for record in records
                ]
                records.sort(key=lambda record: record.frame_index)
        extraction.alarm_frame_index = alarm_index
        extraction.alarm_frame_agreement = agreement
        extraction.alarm_frame_runner_up = runner_up
        extraction.frames = [asdict(record) for record in records]
        if not records:
            extraction.status = "failed"
            extraction.error = "clip decoded to zero frames"
    except Exception as exc:  # noqa: BLE001 - one bad clip must not end the pass
        logger.warning("extraction failed for %s: %s", clip.key, exc)
        extraction.status = "failed"
        extraction.error = f"{type(exc).__name__}: {exc}"
    finally:
        if video_path is not None:
            video_path.unlink(missing_ok=True)

    _write_sidecar(out_dir, extraction)
    return extraction


def _write_sidecar(out_dir: Path, extraction: ClipExtraction) -> None:
    """Write the sidecar atomically. A run killed mid-write would otherwise
    leave truncated JSON that reads as "done" and is never retried."""
    path = sidecar_path(out_dir, extraction.alarm_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(asdict(extraction), indent=2), encoding="utf-8")
    os.replace(temp_path, path)


def extract_corpus(
    clips: Iterable[ClipRef],
    *,
    client,
    bucket: str,
    out_dir: Path,
    workers: int = DEFAULT_WORKERS,
    retry_failed: bool = False,
    re_extract: bool = False,
    progress_every: int = 25,
    **clip_kwargs,
) -> dict[str, int]:
    """Extract many clips concurrently, returning per-status counts.

    Threads rather than processes: the work is download-bound, and OpenCV
    releases the GIL across decode, so the pool scales without the pickling
    cost of shipping frames between processes.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {"ok": 0, "failed": 0, "skipped": 0}

    pending = []
    for clip in clips:
        # `re_extract` overrides the resume rule entirely. It exists because a
        # SCHEMA CHANGE makes an existing sidecar wrong rather than done: a v1
        # sidecar has no snapshot frame in it, and without this the only way to
        # pick one up was to delete files by hand.
        if not re_extract and is_extracted(out_dir, clip.alarm_id):
            if not retry_failed or _sidecar_status(out_dir, clip.alarm_id) == "ok":
                counts["skipped"] += 1
                continue
        pending.append(clip)

    if not pending:
        return counts

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                extract_clip, clip, client=client, bucket=bucket, out_dir=out_dir, **clip_kwargs
            ): clip
            for clip in pending
        }
        for done, future in enumerate(as_completed(futures), start=1):
            extraction = future.result()
            counts[extraction.status] = counts.get(extraction.status, 0) + 1
            if progress_every and done % progress_every == 0:
                logger.info(
                    "extracted %d/%d clips (ok=%d failed=%d)",
                    done,
                    len(pending),
                    counts["ok"],
                    counts["failed"],
                )

    return counts


def _sidecar_status(out_dir: Path, alarm_id: str) -> str | None:
    try:
        payload = json.loads(sidecar_path(out_dir, alarm_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload.get("status")


def iter_sidecars(out_dir: Path) -> Iterator[dict]:
    """Every sidecar in `out_dir`, alarm-id order. Unreadable ones are
    skipped with a warning rather than ending the walk."""
    index_dir = out_dir / "index"
    if not index_dir.is_dir():
        return
    for path in sorted(index_dir.glob("*.json")):
        try:
            yield json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("unreadable sidecar %s: %s", path, exc)


def build_manifest(out_dir: Path, manifest_path: Path | None = None) -> int:
    """Merge the per-clip sidecars into one JSONL of frames, returning the
    row count. One row per frame, so downstream reads a flat file instead of
    globbing tens of thousands of sidecars."""
    manifest_path = manifest_path or (out_dir / "frames.jsonl")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    temp_path = manifest_path.with_suffix(".jsonl.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        for sidecar in iter_sidecars(out_dir):
            if sidecar.get("status") != "ok":
                continue
            for record in sidecar.get("frames", ()):
                handle.write(json.dumps(record) + "\n")
                rows += 1
    os.replace(temp_path, manifest_path)
    return rows
