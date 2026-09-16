"""Tests for the CV pass: candidate records, the per-alarm sidecar, resume,
atomic replace, and the merged manifest.

No detector at all: person boxes come from the vendor CV's own snapshot
and torch is never imported. OpenCV is used only to write and read back a
handful of tiny jpegs, which is the one thing the pass genuinely does to
disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from ppe_pipeline.config import PipelineConfig
from ppe_pipeline.helmet_cv import runner
from ppe_pipeline.helmet_cv.records import CvProvenance, candidate_id, execution_id, frame_stem
from ppe_pipeline.helmet_cv.runner import group_by_frame

ALARM = "726f88cb-a3d0-4907-9d8d-111055ebc676"
def _frame_name(index: int = 0) -> str:
    return f"unk__unk__20260427T07022{index}Z__{ALARM}__f{index:06d}.jpg"


def _frame_row(index: int = 0) -> dict:
    name = _frame_name(index)
    return {
        "frame_name": name,
        "relative_path": f"frames/{ALARM}/{name}",
        "alarm_id": ALARM,
        "clip_key": f"raw_video/helmet_violation_2026-04-27_07-02-09_to_07-03-01_{ALARM}.mp4",
        "violation_type": "helmet_violation",
        "frame_index": index,
        "offset_ms": index * 1000,
        "frame_utc": f"2026-04-27T07:02:2{index}Z",
        "width": 400,
        "height": 400,
    }


@pytest.fixture
def provenance() -> CvProvenance:
    return CvProvenance(cv_run_id="cv-test")


@pytest.fixture
def corpus(tmp_path) -> Path:
    """An extracted corpus of two frames for one alarm, on disk."""
    frames_dir = tmp_path / "frames" / ALARM
    frames_dir.mkdir(parents=True)
    for index in (0, 1):
        image = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.imwrite(str(frames_dir / _frame_name(index)), image)
    manifest = tmp_path / "frames.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(_frame_row(index)) for index in (0, 1)) + "\n", encoding="utf-8"
    )
    return tmp_path


# --- candidate identity ---------------------------------------------------


def test_a_candidate_is_one_person_in_one_frame():
    """Not a clip and not a frame: a 52-second clip holds frames where the
    worker has walked out of shot, and a frame can hold four people who are
    not all in the same state."""
    assert candidate_id(_frame_name(0), 0) == f"{frame_stem(_frame_name(0))}__p00"
    assert candidate_id(_frame_name(0), 11).endswith("__p11")


def test_a_candidate_id_stays_inside_the_safe_key_charset():
    """It is used verbatim as an `alarm_id`, which becomes a storage key."""
    import re

    assert re.fullmatch(r"[A-Za-z0-9._-]+", candidate_id(_frame_name(0), 3))


def test_execution_id_stays_inside_the_safe_storage_key_charset():
    import re

    from ppe_pipeline.helmet_cv.records import execution_id

    value = execution_id("height-constant", candidate_id(_frame_name(0), 3))
    assert re.fullmatch(r"[A-Za-z0-9._-]+", value)
    assert execution_id("a", "b__c") != execution_id("a__b", "c")


# --- one frame ------------------------------------------------------------


def test_provenance_names_no_detector_because_none_runs():
    """The fields stay so a row scored before 2 Sep is still readable, but a
    new pass leaves them empty rather than naming a checkpoint that never
    ran. The vendor publishes a box, not a score, so there is no confidence
    and no inference resolution to record either."""
    provenance = CvProvenance.from_config(PipelineConfig(), cv_run_id="cv-x")

    assert provenance.box_source == "vendor_cv"
    assert provenance.detector_weights == ""
    assert provenance.imgsz is None
    assert provenance.od_confidence == 0.0


# --- the pass, its sidecar, and resume -----------------------------------


def test_the_sidecar_is_written_through_a_temp_file_and_replaced(tmp_path, monkeypatch):
    """A run killed mid-write would otherwise leave truncated JSON that reads
    as "done" and is never retried."""
    seen = {}
    real_replace = runner.os.replace

    def watching_replace(src, dst):
        seen["src"] = Path(src)
        seen["destination_existed"] = Path(dst).exists()
        real_replace(src, dst)

    monkeypatch.setattr(runner.os, "replace", watching_replace)

    runner._write_sidecar(tmp_path, ALARM, {"alarm_id": ALARM, "candidates": []})

    assert seen["src"].suffixes[-2:] == [".json", ".tmp"]
    assert seen["destination_existed"] is False
    assert not seen["src"].exists()


def test_a_missing_frame_manifest_says_what_to_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="extract"):
        list(runner.iter_frames(tmp_path))


# --- the merged manifest --------------------------------------------------


def test_an_unreadable_sidecar_is_skipped_rather_than_ending_the_walk(tmp_path):
    (tmp_path / "cv").mkdir()
    (tmp_path / "cv" / "aaa.json").write_text('{"candidates": []}', encoding="utf-8")
    (tmp_path / "cv" / "bbb.json").write_text("{ truncated", encoding="utf-8")

    assert len(list(runner.iter_cv_sidecars(tmp_path))) == 1


def _candidate(person_index: int, frame_index: int = 0, person_box=(10.0, 10.0, 50.0, 200.0)) -> dict:
    name = _frame_name(frame_index)
    return {
        "candidate_id": candidate_id(name, person_index),
        "frame_name": name,
        "relative_path": f"frames/{ALARM}/{name}",
        "alarm_id": ALARM,
        "clip_key": "raw_video/clip.mp4",
        "frame_utc": "2026-04-27T07:02:24Z",
        "frame_role": "clip",
        "person_index": person_index,
        "person_box": list(person_box) if person_box else None,
        "person_confidence": 0.9,
        "people_in_frame": 1,
        "helmets_in_frame": 0,
        "provenance": {"cv_run_id": "cv-test"},
    }


class _RecordingJudge:
    """Stands in for `run_track1_judge_inference`, recording every call."""

    def __init__(self, fails_for: set[str] | None = None):
        self.calls = []
        self._fails_for = fails_for or set()
        self._lock = __import__("threading").Lock()

    def __call__(self, image_bytes, media_type, violation_type, person_box, alarm_id,
                 source_detection_id, **kwargs):
        with self._lock:
            self.calls.append(
                {
                    "alarm_id": alarm_id,
                    "person_box": person_box,
                    "image_bytes": image_bytes,
                    "kwargs": kwargs,
                }
            )
        if alarm_id in self._fails_for:
            raise RuntimeError("judge blew up")


def test_candidates_are_grouped_by_the_frame_they_came_from():
    grouped = group_by_frame([_candidate(1, 0), _candidate(0, 0), _candidate(0, 1)])

    assert list(grouped) == [_frame_name(0), _frame_name(1)]
    # Person order within a frame is restored, so a partial run judges
    # p00 before p01 rather than whatever order the manifest happened to hold.
    assert [c["person_index"] for c in grouped[_frame_name(0)]] == [0, 1]


def test_a_frame_is_read_from_disk_once_however_many_people_are_in_it(corpus, monkeypatch):
    """Four routed workers in one frame used to cost four reads of the same
    1920x1080 JPEG."""
    reads = {"count": 0}
    real = runner._frame_bytes

    def counting(frames_dir, candidate):
        reads["count"] += 1
        return real(frames_dir, candidate)

    monkeypatch.setattr(runner, "_frame_bytes", counting)
    judge = _RecordingJudge()

    counts = runner.judge_candidates(
        [_candidate(i) for i in range(4)],
        frames_dir=corpus,
        config=PipelineConfig(),
        judge=judge,
        max_workers=1,
    )

    assert counts["judged"] == 4
    assert reads["count"] == 1
    # All four saw the same bytes object -- one decode's worth of memory,
    # not four.
    assert len({id(call["image_bytes"]) for call in judge.calls}) == 1
    assert all(call["kwargs"]["prepared_crop"].candidate_id for call in judge.calls)
    assert len({id(call["kwargs"]["prepared_crop"].image_bytes) for call in judge.calls}) == 4


def test_each_person_still_gets_their_own_judge_call(corpus):
    """Grouping is about I/O, not about asking one question for four people.
    Batched multi-crop inference was measured and deferred."""
    judge = _RecordingJudge()

    runner.judge_candidates(
        [_candidate(0), _candidate(1)], frames_dir=corpus, config=PipelineConfig(),
        judge=judge, max_workers=1,
    )

    assert len(judge.calls) == 2
    assert {call["alarm_id"] for call in judge.calls} == {
        execution_id("cv-test", candidate_id(_frame_name(0), 0)),
        execution_id("cv-test", candidate_id(_frame_name(0), 1)),
    }


def test_the_cv_block_travels_with_each_call(corpus):
    judge = _RecordingJudge()

    runner.judge_candidates(
        [_candidate(0)], frames_dir=corpus, config=PipelineConfig(), judge=judge, max_workers=1
    )

    cv = judge.calls[0]["kwargs"]["cv_decision"]
    assert cv["candidate_id"].endswith("__p00")
    assert cv["ground_truth_id"] == cv["candidate_id"]
    assert cv["execution_id"] == execution_id("cv-test", cv["candidate_id"])
    assert cv["cv_run_id"] == "cv-test"


def test_a_frame_with_nobody_detected_is_not_judged_at_all(corpus, tmp_path):
    """REPLACES two tests that asserted the whole frame was judged instead.

    That behaviour asked the judge "is this person wearing a helmet" about a
    picture with no identified person. Measured on one live run: 7 of 99 rows
    were this case and 6 came back `CONFIRM_FP / helmet_worn` -- confidently
    wrong, and headed for the training export.
    """
    log = tmp_path / "log.json"
    judge = _RecordingJudge()

    counts = runner.judge_candidates(
        [_candidate(0, person_box=None)],
        frames_dir=corpus,
        config=PipelineConfig(),
        judge=judge,
        max_workers=1,
        audit_log_path=str(log),
        alarm_image_dir=str(tmp_path / "stored"),
    )

    assert judge.calls == []          # no judge call at all
    assert counts["no_person"] == 1
    assert counts["judged"] == 0

    rows = json.loads(log.read_text(encoding="utf-8"))
    assert len(rows) == 1
    row = rows[0]
    # No person means no person in violation.
    assert row["outcome"] == "CONFIRM_FP"
    # Its OWN reason, distinct from the gate's: "the detector found nobody"
    # and "the VLM looked and found nobody" are different findings.
    assert row["reason_code"] == "no_person_detected"
    assert row["judge_provider"] == "person_detector"
    # And the frame stays visible in the audit: a detector that misses people
    # is exactly what was found on 1 Sep, so a human must be able to overturn
    # this.
    assert row["image_key"]
    assert row["row_kind"] == "helmet_cv"


def test_frames_run_concurrently_up_to_max_workers(corpus):
    """Threads because the time goes on waiting for Bedrock. Asserted by
    observing overlap rather than by timing, which would flake."""
    import threading

    in_flight = {"now": 0, "peak": 0}
    lock = threading.Lock()
    released = threading.Event()

    def slow_judge(*args, **kwargs):
        with lock:
            in_flight["now"] += 1
            in_flight["peak"] = max(in_flight["peak"], in_flight["now"])
            if in_flight["peak"] >= 2:
                released.set()
        released.wait(timeout=2.0)
        with lock:
            in_flight["now"] -= 1

    runner.judge_candidates(
        [_candidate(0, 0), _candidate(0, 1)],
        frames_dir=corpus,
        config=PipelineConfig(),
        judge=slow_judge,
        max_workers=2,
    )

    assert in_flight["peak"] == 2


def test_one_worker_serialises_the_pass(corpus):
    judge = _RecordingJudge()

    runner.judge_candidates(
        [_candidate(0, 0), _candidate(0, 1)], frames_dir=corpus, config=PipelineConfig(),
        judge=judge, max_workers=1,
    )

    assert len(judge.calls) == 2


def test_the_worker_count_comes_from_config(corpus, monkeypatch):
    seen = {}

    class _Pool:
        def __init__(self, max_workers):
            seen["max_workers"] = max_workers

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def map(self, fn, items):
            return [fn(item) for item in items]

    monkeypatch.setattr(runner, "ThreadPoolExecutor", _Pool)
    config = PipelineConfig()
    config.helmet_cv.max_workers = 7

    runner.judge_candidates(
        [_candidate(0)], frames_dir=corpus, config=config, judge=_RecordingJudge()
    )

    assert seen["max_workers"] == 7


def test_an_already_judged_candidate_is_skipped(corpus):
    judge = _RecordingJudge()

    counts = runner.judge_candidates(
        [_candidate(0), _candidate(1)],
        frames_dir=corpus,
        config=PipelineConfig(),
        judge=judge,
        already_judged={candidate_id(_frame_name(0), 0)},
        max_workers=1,
    )

    assert counts == {"judged": 1, "skipped": 1, "failed": 0, "stopped": 0, "gated": 0, "gate_calls": 0, "tp": 0, "fp": 0, "escalated": 0, "no_person": 0}
    remaining = _candidate(1)
    assert judge.calls[0]["alarm_id"] == execution_id("cv-test", remaining["candidate_id"])


def test_a_fully_judged_frame_is_never_read_from_disk(corpus, monkeypatch):
    def exploding(*_args, **_kwargs):
        raise AssertionError("a fully judged frame must not be re-read")

    monkeypatch.setattr(runner, "_frame_bytes", exploding)

    counts = runner.judge_candidates(
        [_candidate(0)],
        frames_dir=corpus,
        config=PipelineConfig(),
        judge=_RecordingJudge(),
        already_judged={candidate_id(_frame_name(0), 0)},
        max_workers=1,
    )

    assert counts == {"judged": 0, "skipped": 1, "failed": 0, "stopped": 0, "gated": 0, "gate_calls": 0, "tp": 0, "fp": 0, "escalated": 0, "no_person": 0}


def test_one_failed_call_does_not_end_the_pass(corpus):
    failing = execution_id("cv-test", candidate_id(_frame_name(0), 0))
    judge = _RecordingJudge(fails_for={failing})

    counts = runner.judge_candidates(
        [_candidate(0), _candidate(1)], frames_dir=corpus, config=PipelineConfig(),
        judge=judge, max_workers=1,
    )

    assert counts == {"judged": 1, "skipped": 0, "failed": 1, "stopped": 0, "gated": 0, "gate_calls": 0, "tp": 0, "fp": 0, "escalated": 0, "no_person": 0}


def test_an_unreadable_frame_costs_only_its_own_candidates(corpus):
    missing = dict(_candidate(0), relative_path="frames/nope/missing.jpg")
    judge = _RecordingJudge()

    counts = runner.judge_candidates(
        [missing, _candidate(0, 1)], frames_dir=corpus, config=PipelineConfig(),
        judge=judge, max_workers=1,
    )

    assert counts["failed"] == 1
    assert counts["judged"] == 1


def test_the_limit_caps_judge_calls_deterministically(corpus):
    """Applied before the pool starts. A cap enforced by racing a counter
    inside the pool would depend on thread scheduling, and a cap that depends
    on scheduling is not a cap."""
    judge = _RecordingJudge()

    counts = runner.judge_candidates(
        [_candidate(0, 0), _candidate(1, 0), _candidate(0, 1), _candidate(0, 2)],
        frames_dir=corpus,
        config=PipelineConfig(),
        judge=judge,
        limit=2,
        max_workers=4,
    )

    assert counts["judged"] == 2
    assert len(judge.calls) == 2


def test_stop_is_honoured_between_people_not_only_between_frames(corpus):
    """"Stop immediately" must cost at most the call already in flight.

    Between FRAMES is not good enough here: the live feed judges one alarm
    frame at a time and 8.54 people stand in the average one, so a stop that
    waits for the frame is a stop that spends a frame.
    """
    judge = _RecordingJudge()
    stop = {"now": False}

    def should_stop():
        return stop["now"]

    def stopping(*args, **kwargs):
        judge(*args, **kwargs)
        stop["now"] = True

    # crop_workers=1 so the guarantee under test is the strict one: nothing
    # further is even dispatched. The concurrent guarantee is weaker by
    # exactly `crop_workers` and is asserted in the next test.
    config = PipelineConfig()
    config.helmet_cv.crop_workers = 1

    counts = runner.judge_candidates(
        [_candidate(0), _candidate(1), _candidate(2)],
        frames_dir=corpus,
        config=config,
        judge=stopping,
        max_workers=1,
        should_stop=should_stop,
    )

    assert counts == {"judged": 1, "skipped": 0, "failed": 0, "stopped": 2, "gated": 0, "gate_calls": 0, "tp": 0, "fp": 0, "escalated": 0, "no_person": 0}
    assert len(judge.calls) == 1


def test_crops_of_one_frame_are_judged_concurrently(corpus):
    """MEASURED on the deployed feed: 9 calls at ~25s each, sequential, was
    four minutes for one alarm -- while the download and the detector together
    took 5.1 seconds. `max_workers` never helped: it counts FRAMES in flight
    and a live alarm is one frame.

    Saves no tokens. Wall clock only.
    """
    import threading

    config = PipelineConfig()
    config.helmet_cv.crop_workers = 4
    live = {"now": 0, "peak": 0}
    lock = threading.Lock()
    barrier = threading.Barrier(4, timeout=5)

    def slow_judge(*args, **kwargs):
        with lock:
            live["now"] += 1
            live["peak"] = max(live["peak"], live["now"])
        # Every task must be inside the judge at once, or the pool is not
        # actually running them together.
        barrier.wait()
        with lock:
            live["now"] -= 1

    counts = runner.judge_candidates(
        [_candidate(i) for i in range(4)],
        frames_dir=corpus,
        config=config,
        judge=slow_judge,
        max_workers=1,
        should_stop=None,
    )

    assert counts["judged"] == 4
    assert live["peak"] == 4


def test_a_stop_under_concurrency_skips_what_was_not_dispatched(corpus):
    """The guarantee is weaker by exactly `crop_workers`, and that is the
    honest statement: a task already dispatched cannot be recalled. What must
    NOT happen is a skipped crop being recorded as a verdict."""
    config = PipelineConfig()
    config.helmet_cv.crop_workers = 2
    judge = _RecordingJudge()
    stop = {"now": False}

    def stopping(*args, **kwargs):
        judge(*args, **kwargs)
        stop["now"] = True

    counts = runner.judge_candidates(
        [_candidate(i) for i in range(8)],
        frames_dir=corpus,
        config=config,
        judge=stopping,
        max_workers=1,
        should_stop=lambda: stop["now"],
    )

    assert counts["judged"] + counts["stopped"] == 8
    # Bounded by the workers in flight, not by the whole frame.
    assert counts["judged"] <= 1 + config.helmet_cv.crop_workers
    assert counts["stopped"] >= 8 - 1 - config.helmet_cv.crop_workers



def test_a_frame_with_no_recoverable_box_settles_without_a_judge_call(corpus, tmp_path):
    """No rectangle could be read off the snapshot, so there is no box to crop
    and nothing to ask about. It settles as no-violation with its OWN reason
    code and the detector named as the answerer -- never `helmet_worn`, which
    is a claim about a person somebody actually looked at.

    It still writes a row. A settled crop with no row has no case, so no
    image, so it appeared in the feed's results and then 404'd when opened.
    """
    judge = _RecordingJudge()
    monkeypatched_rows = tmp_path / "audit.json"
    counts = runner.judge_candidates(
        [_candidate(0, person_box=None)],
        frames_dir=corpus,
        judge=judge,
        config=PipelineConfig(),
        audit_log_path=str(monkeypatched_rows),
        alarm_image_dir=str(tmp_path / "frames"),
    )

    assert counts["judged"] == 0
    assert counts["no_person"] == 1
    assert judge.calls == []

    rows = json.loads(monkeypatched_rows.read_text())
    assert [row["reason_code"] for row in rows] == ["no_person_detected"]
    assert rows[0]["judge_provider"] == "person_detector"




# --------------------------------------------------------------------------
# THE CROP PROGRESS COUNT THE LIVE FEED RENDERS AS "3 of 9".
# --------------------------------------------------------------------------


def _counter():
    """A drop-in for the live feed's `reporter.crop_judged`."""
    seen = []
    return seen, lambda: seen.append(1)


def _always_fails(*_args, **_kwargs):
    raise RuntimeError("judge blew up")


def test_every_crop_that_settles_is_counted_however_it_settled(corpus, tmp_path):
    """The caller's total is `len(candidates)`, so a crop answered WITHOUT a
    judge call still has to report or the screen's "3 of 9" stops short and
    reads as a hang. A frame the detector found nobody in is exactly that:
    measured at 7 of 99 rows on one live run, settled by reason code."""
    counted, on_crop_judged = _counter()

    counts = runner.judge_candidates(
        [_candidate(0), _candidate(1, person_box=None)],
        frames_dir=corpus,
        config=PipelineConfig(),
        judge=_RecordingJudge(),
        max_workers=1,
        audit_log_path=str(tmp_path / "log.json"),
        alarm_image_dir=str(tmp_path / "stored"),
        on_crop_judged=on_crop_judged,
    )

    assert counts["judged"] == 1 and counts["no_person"] == 1
    assert len(counted) == 2, "the box-less crop settled without being counted"


def test_a_failed_judge_call_still_counts_as_answered(corpus, tmp_path):
    """"Answered", not "successful". A total that stalled at 2 of 3 because one
    call errored would send an operator looking for a hang that is not there."""
    counted, on_crop_judged = _counter()

    counts = runner.judge_candidates(
        [_candidate(0), _candidate(1)],
        frames_dir=corpus,
        config=PipelineConfig(),
        judge=_always_fails,
        max_workers=1,
        audit_log_path=str(tmp_path / "log.json"),
        alarm_image_dir=str(tmp_path / "stored"),
        on_crop_judged=on_crop_judged,
    )

    assert counts["failed"] == 2
    assert len(counted) == 2


def test_an_unreadable_frame_counts_all_of_its_crops(tmp_path):
    """The frame never loads, so no crop is ever attempted -- but the caller
    already told the screen how many there were."""
    counted, on_crop_judged = _counter()

    counts = runner.judge_candidates(
        [_candidate(0), _candidate(1), _candidate(2)],
        frames_dir=tmp_path / "no-such-corpus",
        config=PipelineConfig(),
        judge=_RecordingJudge(),
        max_workers=1,
        audit_log_path=str(tmp_path / "log.json"),
        alarm_image_dir=str(tmp_path / "stored"),
        on_crop_judged=on_crop_judged,
    )

    assert counts["failed"] == 3
    assert len(counted) == 3


def test_a_crop_judged_on_an_earlier_pass_counts_too(corpus, tmp_path):
    """It is answered -- just not by this pass. Left out, a resumed alarm could
    never reach its own total."""
    counted, on_crop_judged = _counter()
    already = {candidate_id(_frame_name(0), 0)}

    counts = runner.judge_candidates(
        [_candidate(0), _candidate(1)],
        frames_dir=corpus,
        config=PipelineConfig(),
        judge=_RecordingJudge(),
        max_workers=1,
        already_judged=already,
        audit_log_path=str(tmp_path / "log.json"),
        alarm_image_dir=str(tmp_path / "stored"),
        on_crop_judged=on_crop_judged,
    )

    assert counts["skipped"] == 1 and counts["judged"] == 1
    assert len(counted) == 2


def test_a_crop_dropped_by_a_stop_is_not_counted(corpus, tmp_path):
    """It was never attempted, and the run is ending. A total that completed
    itself on the way out would claim work nobody did."""
    counted, on_crop_judged = _counter()

    counts = runner.judge_candidates(
        [_candidate(0), _candidate(1)],
        frames_dir=corpus,
        config=PipelineConfig(),
        judge=_RecordingJudge(),
        max_workers=1,
        should_stop=lambda: True,
        audit_log_path=str(tmp_path / "log.json"),
        alarm_image_dir=str(tmp_path / "stored"),
        on_crop_judged=on_crop_judged,
    )

    assert counts["stopped"] == 2
    assert counted == []
