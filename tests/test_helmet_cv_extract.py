"""Tests for turning alarm clips into named, sampled frames.

Nothing here decodes video: `_sample_frames` and `_download_to_temp` are
stubbed, so OpenCV is never imported and no clip is ever fetched. What is
under test is the module's own logic -- the frame-rate fallback, the naming
scheme, the sidecar's double duty as manifest shard and resume marker, and
the atomic write that keeps a killed run from leaving a "done" marker
behind.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ppe_pipeline.helmet_cv import extract
from ppe_pipeline.helmet_cv.clips import ClipRef
from ppe_pipeline.helmet_cv.extract import (
    _ASSUMED_FPS,
    _resolve_fps,
    build_manifest,
    extract_clip,
    extract_corpus,
    frame_name,
    is_extracted,
    iter_sidecars,
    safe_field,
    sidecar_path,
)

_ALARM = "726f88cb-a3d0-4907-9d8d-111055ebc676"


def _clip(alarm_id: str = _ALARM) -> ClipRef:
    return ClipRef(
        key=f"raw_video/helmet_violation_2026-04-27_07-02-09_to_07-03-01_{alarm_id}.mp4",
        violation_type="helmet_violation",
        alarm_id=alarm_id,
        start=datetime(2026, 4, 27, 7, 2, 9, tzinfo=UTC),
        end=datetime(2026, 4, 27, 7, 3, 1, tzinfo=UTC),
    )


def _record(clip: ClipRef, index: int = 0) -> extract.FrameRecord:
    name = frame_name(
        site="unk", camera="unk", frame_utc=clip.start, alarm_id=clip.alarm_id, frame_index=index
    )
    return extract.FrameRecord(
        frame_name=name,
        relative_path=f"frames/{clip.alarm_id}/{name}",
        alarm_id=clip.alarm_id,
        clip_key=clip.key,
        violation_type=clip.violation_type,
        frame_role=extract.FRAME_ROLE_CLIP,
        site="unk",
        camera="unk",
        frame_index=index,
        offset_ms=0,
        frame_utc="2026-04-27T07:02:09Z",
        width=1920,
        height=1080,
    )


@pytest.fixture
def stubbed_decode(monkeypatch, tmp_path):
    """Replace download+decode with a canned result, so no clip is fetched
    and OpenCV is never imported."""
    state = {"records": None, "raises": None, "calls": 0}

    def fake_download(_client, _bucket, _key):
        path = tmp_path / "clip.mp4"
        path.write_bytes(b"")
        return path

    def fake_sample(_video_path, clip, **_kwargs):
        state["calls"] += 1
        if state["raises"] is not None:
            raise state["raises"]
        records = state["records"]
        if records is None:
            records = [_record(clip)]
        return records, 1.0, "container", len(records)

    monkeypatch.setattr(extract, "_download_to_temp", fake_download)
    monkeypatch.setattr(extract, "_sample_frames", fake_sample)
    return state


# --- locating the alarm frame ---------------------------------------------


def _jpeg_bytes(width: int = 640, height: int = 480, fill: int = 0) -> bytes:
    import cv2
    import numpy as np

    ok, buf = cv2.imencode(".jpg", np.full((height, width, 3), fill, dtype=np.uint8))
    assert ok
    return buf.tobytes()


class _StillClient:
    """An S3 stub serving one annotated still, or nothing."""

    def __init__(self, available: bool = True, body: bytes | None = None):
        self.available = available
        self.body = body
        self.requested: list[str] = []

    def get_object(self, Bucket: str, Key: str):  # noqa: N803 - boto3's own casing
        self.requested.append(Key)
        if not self.available:
            raise RuntimeError("NoSuchKey")
        return {"Body": _Body(self.body if self.body is not None else _jpeg_bytes())}


class _Body:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self) -> bytes:
        return self._payload


@pytest.fixture
def real_frames(monkeypatch, tmp_path):
    """Write two visibly different clip frames and let extract_clip find them.

    Real files rather than a stub, because the thing under test is a pixel
    comparison -- a fake that returns records without writing images would
    assert nothing about the matching.
    """
    import cv2
    import numpy as np

    def fake_download(_client, _bucket, _key):
        path = tmp_path / "clip.mp4"
        path.write_bytes(b"")
        return path

    def fake_sample(_video_path, clip, *, frames_dir, **_kwargs):
        records = []
        for index, fill in ((0, 0), (1, 255)):
            record = _record(clip, index)
            frames_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(frames_dir / record.frame_name),
                np.full((480, 640, 3), fill, dtype=np.uint8),
            )
            records.append(record)
        return records, 1.0, "container", len(records)

    monkeypatch.setattr(extract, "_download_to_temp", fake_download)
    monkeypatch.setattr(extract, "_sample_frames", fake_sample)


def test_the_alarm_frame_is_the_clean_clip_frame_the_still_matches(real_frames, tmp_path):
    """The still is a KEY, not a frame. It carries production's box and the word
    "violation" burned in, so it can never be judged or shown -- but its source
    frame is in raw_video already, clean, and recoverable by pixel agreement."""
    clip = _clip()
    # The still resembles the SECOND frame (white), not the first.
    client = _StillClient(body=_jpeg_bytes(640, 480, fill=255))

    extraction = extract.extract_clip(
        clip, client=client, bucket="b", out_dir=tmp_path / "out"
    )

    assert client.requested == [clip.snapshot_key]
    assert extraction.alarm_frame_index == 1
    assert extraction.alarm_frame_agreement > 0.9
    roles = {f["frame_index"]: f["frame_role"] for f in extraction.frames}
    assert roles == {0: "clip", 1: "alarm"}


def test_the_annotated_still_is_kept_outside_the_frames_tree(real_frames, tmp_path):
    """Evidence of what production drew is worth keeping, but the frame endpoint
    resolves `frames/<alarm>/<name>` and must never be able to serve it."""
    clip = _clip()
    out = tmp_path / "out"

    extract.extract_clip(clip, client=_StillClient(body=_jpeg_bytes(640, 480, fill=255)),
                         bucket="b", out_dir=out)

    assert (out / "annotated" / f"{clip.alarm_id}.jpg").is_file()
    assert not list((out / "frames").rglob("*annotated*"))


def test_a_weak_match_names_no_alarm_frame_rather_than_the_wrong_one(real_frames, tmp_path):
    """"Which frame did CV fire on" is the CV arm's entire scoring population,
    so a bad guess is worse than an absence: it would score CV against a frame
    it never saw."""
    still = _jpeg_bytes(640, 480, fill=128)  # resembles neither frame

    extraction = extract.extract_clip(
        _clip(), client=_StillClient(body=still), bucket="b", out_dir=tmp_path / "out"
    )

    assert extraction.status == "ok"
    assert extraction.alarm_frame_index is None
    assert {f["frame_role"] for f in extraction.frames} == {"clip"}


def test_the_still_is_not_fetched_again_when_the_caller_already_holds_it(real_frames, tmp_path):
    """The live feed reads the same object to recover the vendor's boxes, so
    one alarm was fetching the same ~0.9MB still twice. Handed the bytes it
    makes no request at all, and locates the same frame from them."""
    clip = _clip()
    # Any GET at all raises here, so a second fetch cannot pass unnoticed.
    client = _StillClient(available=False)

    extraction = extract.extract_clip(
        clip, client=client, bucket="b", out_dir=tmp_path / "out",
        annotated=_jpeg_bytes(640, 480, fill=255),
    )

    assert client.requested == []
    assert extraction.alarm_frame_index == 1
    # And the evidence of what production drew is still written out.
    assert (tmp_path / "out" / "annotated" / f"{clip.alarm_id}.jpg").is_file()


def test_a_missing_still_costs_the_alarm_its_cv_arm_not_its_frames(real_frames, tmp_path):
    """Some alarms have no still -- measured, 2 of 20. That alarm contributes to
    the VLM arm only and must not lose its frames as well."""
    extraction = extract.extract_clip(
        _clip(), client=_StillClient(available=False), bucket="b", out_dir=tmp_path / "out"
    )

    assert extraction.status == "ok"
    assert extraction.alarm_frame_index is None
    assert len(extraction.frames) == 2


# --- resolving the frame rate --------------------------------------------


def test_a_plausible_container_figure_is_believed():
    assert _resolve_fps(1.2, frame_count=60, duration_seconds=52) == (1.2, "container")


def test_a_broken_container_figure_falls_back_to_the_decoded_length():
    """A real measurement, not a guess -- it tracked the container's own
    figure to within 0.6% across six sampled clips."""
    fps, source = _resolve_fps(0.0, frame_count=52.0, duration_seconds=26.0)

    assert (fps, source) == (2.0, "derived")


def test_a_bare_constant_is_the_last_resort_only():
    """Reached only when the container reports nothing usable AND the frame
    count is unreadable too."""
    assert _resolve_fps(0.0, frame_count=0.0, duration_seconds=52.0) == (_ASSUMED_FPS, "assumed")
    assert _resolve_fps(float("nan"), frame_count=10.0, duration_seconds=0.0) == (
        _ASSUMED_FPS,
        "assumed",
    )


def test_a_sub_1fps_container_figure_is_believed_not_rejected():
    """This corpus is ~1fps CCTV export -- a 24fps floor would reject every
    real value in the bucket (0.974-1.855 observed)."""
    assert _resolve_fps(0.974, frame_count=35, duration_seconds=36)[1] == "container"


def test_an_absurd_container_figure_is_rejected():
    assert _resolve_fps(100000.0, frame_count=0.0, duration_seconds=0.0)[1] == "assumed"


def test_a_derived_figure_is_itself_sanity_checked():
    """A truncated clip can report thousands of frames over a 52s span; the
    fallback must not launder that into a believed rate."""
    assert _resolve_fps(0.0, frame_count=1_000_000.0, duration_seconds=1.0)[1] == "assumed"


# --- naming ---------------------------------------------------------------


def test_a_frame_name_carries_every_field_it_needs_to_be_traced_back():
    name = frame_name(
        site="unk",
        camera="unk",
        frame_utc=datetime(2026, 4, 27, 7, 2, 24, tzinfo=UTC),
        alarm_id=_ALARM,
        frame_index=450,
    )

    assert name == f"unk__unk__20260427T070224Z__{_ALARM}__f000450.jpg"
    assert name.split("__")[3] == _ALARM


def test_the_frame_index_is_in_the_name_as_well_as_the_timestamp():
    """A sub-second stride would otherwise put two frames in the same second
    under one name."""
    kwargs = {
        "site": "unk",
        "camera": "unk",
        "frame_utc": datetime(2026, 4, 27, 7, 2, 24, tzinfo=UTC),
        "alarm_id": _ALARM,
    }

    assert frame_name(**kwargs, frame_index=1) != frame_name(**kwargs, frame_index=2)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Berth 21", "Berth-21"),
        ("cam_03", "cam-03"),
        ("a/b\\c:d", "a-b-c-d"),
        ("!!!", "unk"),
        ("", "unk"),
    ],
)
def test_a_name_component_is_reduced_to_a_safe_charset(raw, expected):
    """Underscores fold to `-` as well: `_` is the field separator's own
    character, and one inside a site name would fake an extra field."""
    assert safe_field(raw) == expected


# --- the sidecar: manifest shard and resume marker ------------------------


def test_a_successful_clip_writes_a_sidecar_carrying_its_frames(tmp_path, stubbed_decode):
    clip = _clip()

    result = extract_clip(clip, client=None, bucket="b", out_dir=tmp_path)

    assert result.status == "ok"
    payload = json.loads(sidecar_path(tmp_path, clip.alarm_id).read_text(encoding="utf-8"))
    assert payload["clip_key"] == clip.key
    assert payload["fps_source"] == "container"
    assert len(payload["frames"]) == 1
    assert payload["params"]["stride_seconds"] == extract.DEFAULT_STRIDE_SECONDS


def test_a_decode_failure_is_recorded_rather_than_raised(tmp_path, stubbed_decode):
    """Over 27k clips some proportion is truncated; one of them must not take
    down a multi-hour pass."""
    stubbed_decode["raises"] = RuntimeError("OpenCV could not open it")

    result = extract_clip(_clip(), client=None, bucket="b", out_dir=tmp_path)

    assert result.status == "failed"
    assert "OpenCV could not open it" in result.error
    assert is_extracted(tmp_path, _ALARM)


def test_a_clip_that_decodes_to_nothing_counts_as_failed(tmp_path, stubbed_decode):
    stubbed_decode["records"] = []

    result = extract_clip(_clip(), client=None, bucket="b", out_dir=tmp_path)

    assert result.status == "failed"
    assert result.error == "clip decoded to zero frames"


def test_an_extracted_clip_is_skipped_so_a_killed_run_resumes(tmp_path, stubbed_decode):
    clip = _clip()

    extract_corpus([clip], client=None, bucket="b", out_dir=tmp_path)
    counts = extract_corpus([clip], client=None, bucket="b", out_dir=tmp_path)

    assert counts == {"ok": 0, "failed": 0, "skipped": 1}
    assert stubbed_decode["calls"] == 1


def test_a_failed_clip_still_counts_as_done_until_retry_failed(tmp_path, stubbed_decode):
    stubbed_decode["raises"] = RuntimeError("truncated")
    clip = _clip()
    extract_corpus([clip], client=None, bucket="b", out_dir=tmp_path)

    skipped = extract_corpus([clip], client=None, bucket="b", out_dir=tmp_path)
    stubbed_decode["raises"] = None
    retried = extract_corpus([clip], client=None, bucket="b", out_dir=tmp_path, retry_failed=True)

    assert skipped["skipped"] == 1
    assert retried["ok"] == 1


def test_retry_failed_leaves_successful_clips_alone(tmp_path, stubbed_decode):
    clip = _clip()
    extract_corpus([clip], client=None, bucket="b", out_dir=tmp_path)

    counts = extract_corpus([clip], client=None, bucket="b", out_dir=tmp_path, retry_failed=True)

    assert counts["skipped"] == 1


def test_re_extract_overrides_the_resume_rule(tmp_path, stubbed_decode):
    """`--retry-failed` cannot pick up a SCHEMA change, because the sidecar it
    would skip is `ok` rather than failed. A v1 sidecar holds no snapshot frame,
    which makes it wrong rather than done, and without this flag the only way
    to re-extract was to delete files by hand."""
    clip = _clip()
    extract_corpus([clip], client=None, bucket="b", out_dir=tmp_path)

    skipped = extract_corpus([clip], client=None, bucket="b", out_dir=tmp_path)
    forced = extract_corpus([clip], client=None, bucket="b", out_dir=tmp_path, re_extract=True)

    assert skipped["skipped"] == 1
    assert forced["ok"] == 1
    assert forced["skipped"] == 0


# --- atomic replace -------------------------------------------------------


def test_the_sidecar_is_written_through_a_temp_file_and_replaced(tmp_path, monkeypatch):
    """A run killed mid-write would otherwise leave truncated JSON that reads
    as "done" and is never retried."""
    seen = {}
    real_replace = extract.os.replace

    def watching_replace(src, dst):
        seen["src"] = Path(src)
        seen["dst"] = Path(dst)
        seen["destination_existed"] = Path(dst).exists()
        seen["temp_payload"] = json.loads(Path(src).read_text(encoding="utf-8"))
        real_replace(src, dst)

    monkeypatch.setattr(extract.os, "replace", watching_replace)

    extract._write_sidecar(
        tmp_path,
        extract.ClipExtraction(
            schema=extract.SCHEMA_VERSION,
            alarm_id=_ALARM,
            clip_key="k",
            violation_type="helmet_violation",
            clip_start_utc="2026-04-27T07:02:09Z",
            clip_end_utc="2026-04-27T07:03:01Z",
            status="ok",
        ),
    )

    assert seen["src"].suffixes[-2:] == [".json", ".tmp"]
    assert seen["dst"] == sidecar_path(tmp_path, _ALARM)
    assert seen["destination_existed"] is False
    assert seen["temp_payload"]["alarm_id"] == _ALARM
    assert not seen["src"].exists()


def test_a_crash_before_the_replace_leaves_no_resume_marker(tmp_path, monkeypatch, stubbed_decode):
    def exploding_replace(*_args):
        raise OSError("killed")

    monkeypatch.setattr(extract.os, "replace", exploding_replace)

    with pytest.raises(OSError):
        extract_clip(_clip(), client=None, bucket="b", out_dir=tmp_path)

    assert not is_extracted(tmp_path, _ALARM)


# --- merging the shards ---------------------------------------------------


def test_the_manifest_merges_every_ok_sidecar_into_one_row_per_frame(tmp_path, stubbed_decode):
    for alarm in ("aaa", "bbb"):
        clip = _clip(alarm)
        stubbed_decode["records"] = [_record(clip, 0), _record(clip, 1)]
        extract_clip(clip, client=None, bucket="b", out_dir=tmp_path)

    rows = build_manifest(tmp_path)

    lines = (tmp_path / "frames.jsonl").read_text(encoding="utf-8").splitlines()
    assert rows == 4
    assert len(lines) == 4
    assert {json.loads(line)["alarm_id"] for line in lines} == {"aaa", "bbb"}


def test_a_failed_sidecar_contributes_no_manifest_rows(tmp_path, stubbed_decode):
    stubbed_decode["raises"] = RuntimeError("truncated")
    extract_clip(_clip(), client=None, bucket="b", out_dir=tmp_path)

    assert build_manifest(tmp_path) == 0


def test_an_unreadable_sidecar_is_skipped_rather_than_ending_the_walk(tmp_path):
    (tmp_path / "index").mkdir(parents=True)
    (tmp_path / "index" / "aaa.json").write_text('{"status": "ok", "frames": []}', encoding="utf-8")
    (tmp_path / "index" / "bbb.json").write_text("{ truncated", encoding="utf-8")

    assert [s["status"] for s in iter_sidecars(tmp_path)] == ["ok"]


def test_the_manifest_is_replaced_atomically(tmp_path, stubbed_decode):
    extract_clip(_clip(), client=None, bucket="b", out_dir=tmp_path)
    build_manifest(tmp_path)

    build_manifest(tmp_path)

    assert not (tmp_path / "frames.jsonl.tmp").exists()
    assert len((tmp_path / "frames.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_the_threshold_sits_above_the_wrong_frame_band():
    """0.90 sat INSIDE the overlap and 0.94 does not.

    A wrong frame in a moving scene reaches 0.92; the true match scores
    0.95-0.99. At 0.90, six of 61 accepted matches on a 130-clip sample landed
    in 0.90-0.94 — 10% of the CV arm's whole population in the band where a
    wrong frame is possible, and a wrong alarm frame scores CV against a frame
    it never saw, invisibly.
    """
    assert extract.ALARM_FRAME_MIN_AGREEMENT > 0.92


def test_the_runner_up_is_recorded_so_a_match_can_be_audited(real_frames, tmp_path):
    """Recorded, NOT enforced. This footage is ~1fps and often static, so the
    frames either side of the alarm can legitimately tie — and when they do,
    either is right. A margin requirement would reject correct answers; the
    number is kept so an auditor can tell a clear-cut match from a coin flip."""
    extraction = extract.extract_clip(
        _clip(), client=_StillClient(body=_jpeg_bytes(640, 480, fill=255)),
        bucket="b", out_dir=tmp_path / "out",
    )

    assert extraction.alarm_frame_index == 1
    assert extraction.alarm_frame_agreement > extraction.alarm_frame_runner_up
    # The tie case is accepted, not rejected.
    assert extraction.alarm_frame_runner_up is not None
