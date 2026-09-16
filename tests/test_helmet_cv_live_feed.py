import json
import threading
import time
from datetime import UTC, datetime, timedelta
from itertools import count

import pytest

from ppe_pipeline.config import PipelineConfig
from ppe_pipeline.helmet_cv import live_feed
from ppe_pipeline.helmet_cv.snapshot_boxes import parse_snapshot_key
from ppe_pipeline.helmet_cv.live_feed import FeedCaps, new_state, poll_once


class FakePaginator:
    def __init__(self, keys):
        self.keys = keys
        self.calls = []

    def paginate(self, **kwargs):
        self.calls.append(kwargs)
        start = kwargs.get("StartAfter")
        yield {"Contents": [{"Key": key} for key in self.keys if start is None or key > start]}


class FakeS3:
    def __init__(self, keys):
        self.paginator = FakePaginator(keys)

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return self.paginator


def key(day):
    """A SNAPSHOT key. The feed walks `snapshot/`, not `raw_video/`: the
    boxes come from the vendor's annotated still, and one exists exactly
    where the vendor's CV fired."""
    return f"snapshot/helmet_violation_2026-04-{day:02d}_07-02-09_726f88cb-a3d0-4907-9d8d-111055ebc676.jpg"


def test_poll_is_idempotent_and_resumes_exclusively():
    client = FakeS3([key(1), key(2)])
    state = new_state(FeedCaps(10, 10), now=datetime(2026, 4, 1, tzinfo=UTC))
    seen = []
    processor = lambda clip, *_: seen.append(clip.key) or {"status": "judged", "calls": 2}
    poll_once(state, client=client, now=datetime(2026, 4, 1, tzinfo=UTC), processor=processor)
    poll_once(state, client=client, now=datetime(2026, 4, 1, tzinfo=UTC), processor=processor)
    assert seen == [key(1), key(2)]
    assert client.paginator.calls[-1]["StartAfter"] == key(2)


def test_budget_exhaustion_pauses_without_consuming_or_advancing():
    state = new_state(FeedCaps(1, 1), now=datetime(2026, 4, 1, tzinfo=UTC))
    state["calls"] = state["daily_calls"] = 1
    poll_once(state, client=FakeS3([key(1)]), now=datetime(2026, 4, 1, tzinfo=UTC), processor=lambda *_: None)
    assert state["processed_keys"] == {}
    assert state["watermark"] is None
    assert state["paused_reason"] == "budget"


def test_failure_is_recorded_and_retried_without_cursor_advance():
    state = new_state(FeedCaps(10, 10), now=datetime(2026, 4, 1, tzinfo=UTC))
    def fail(*_):
        raise RuntimeError("decode failed")
    poll_once(state, client=FakeS3([key(1)]), now=datetime(2026, 4, 1, tzinfo=UTC), processor=fail)
    assert state["failures"][0]["error"] == "RuntimeError: decode failed"
    assert state["failures"][0]["run_id"] == state["run_id"]
    assert state["watermark"] is None


def test_a_stopped_clip_is_not_consumed_and_holds_the_cursor():
    """Stopping mid-frame leaves the alarm half judged, so it must NOT count
    as processed: the cursor holds and a resumed feed judges it whole."""
    state = new_state(FeedCaps(10, 10), now=datetime(2026, 4, 1, tzinfo=UTC))
    poll_once(
        state,
        client=FakeS3([key(1), key(2)]),
        now=datetime(2026, 4, 1, tzinfo=UTC),
        processor=lambda *_: {"status": "stopped", "calls": 3, "stopped": 5},
    )

    assert state["processed_keys"] == {}
    assert state["watermark"] is None
    # The spend still happened and still counts against both caps.
    assert state["calls"] == 3
    assert state["partial_frames"] == [
        {"clip_key": key(1), "run_id": state["run_id"], "calls": 3, "unjudged_people": 5}
    ]


def _stub_live_snapshot(monkeypatch, judged, stopped):
    """Everything `process_live_snapshot` reaches for, replaced -- the point
    of the test is its SIGNATURE and its status mapping, not S3."""
    frame = {"frame_role": "alarm", "relative_path": "f.jpg", "frame_name": "f"}
    extracted = type("E", (), {"status": "ok", "alarm_frame_index": 0, "frames": [frame], "error": None})()
    monkeypatch.setattr(live_feed, "extract_clip", lambda *a, **k: extracted)
    monkeypatch.setattr(live_feed, "get_config", PipelineConfig)
    monkeypatch.setattr(live_feed.CvProvenance, "from_config", classmethod(lambda cls, *a, **k: {}))
    monkeypatch.setattr(live_feed, "extract_boxes", lambda _bytes: [(1.0, 2.0, 3.0, 4.0)])
    monkeypatch.setattr(live_feed, "clip_key_for", lambda *a, **k: CLIP_KEY)
    monkeypatch.setattr(live_feed, "records_from_boxes", lambda *a, **k: [])
    monkeypatch.setattr(
        live_feed,
        "judge_candidates",
        lambda *a, **k: {"judged": judged, "skipped": 0, "failed": 0, "stopped": stopped},
    )


CLIP_KEY = (
    "raw_video/helmet_violation_2026-04-01_07-02-09_to_07-03-01"
    "_726f88cb-a3d0-4907-9d8d-111055ebc676.mp4"
)

SNAPSHOT = parse_snapshot_key(key(1))


class _FakeBody:
    @staticmethod
    def read():
        return b"annotated-bytes"


class _FakeClient:
    @staticmethod
    def get_object(**_kwargs):
        return {"Body": _FakeBody()}


@pytest.mark.parametrize(
    ("judged", "stopped", "status"), [(2, 3, "stopped"), (5, 0, "judged")]
)
def test_process_live_snapshot_takes_the_stop_switch_the_router_hands_it(
    monkeypatch, judged, stopped, status
):
    """Called exactly as `routers/track1._run_live_feed` calls it. The router
    once passed `should_stop=` to a function that did not take it, and nothing
    caught it because no test ever called this one."""
    _stub_live_snapshot(monkeypatch, judged, stopped)

    result = live_feed.process_live_snapshot(
        SNAPSHOT,
        10,
        "live-1",
        client=_FakeClient(),
        audit_log_path="eval/track1_audit_log.json",
        alarm_image_dir="eval/alarm_images",
        should_stop=lambda: False,
        # Passed IN, exactly as the router passes it. Read from a relative
        # path inside the worker, a server started from anywhere but the repo
        # root turned every arriving alarm into its own durable
        # FileNotFoundError.
        few_shot_library=None,
    )

    assert result["status"] == status
    assert result["calls"] == judged


# --- where in the feed a run starts ---------------------------------------
#
# An operator wants alarms from now. A tester wants the corpus that already
# exists. The same button has to do both, and the default has to be the
# operator's -- but a run that silently judges nothing forever is the failure
# that cost an afternoon, so each mode is pinned here.


def test_from_now_ignores_everything_already_in_the_bucket():
    """The operator's feed. Historical alarms are not new alarms.

    AND DOES NOT PAGE THEM TO FIND THAT OUT (OPEN_TASKS #39): the start time
    names the key the listing begins at, so the older alarms are never sent.
    They used to be -- all 75,242 of them, 11.75s per poll, discarded.
    """
    client = FakeS3([key(1), key(2)])
    state = new_state(
        FeedCaps(10, 10),
        now=datetime(2026, 4, 5, tzinfo=UTC),
        start_on_or_after=datetime(2026, 4, 5, tzinfo=UTC),
    )
    seen = []
    poll_once(
        state,
        client=client,
        now=datetime(2026, 4, 5, tzinfo=UTC),
        processor=lambda clip, *_: seen.append(clip.key) or {"status": "judged", "calls": 1},
    )

    assert seen == []
    assert client.paginator.calls[-1]["StartAfter"] == "snapshot/helmet_violation_2026-04-05_00-00-00"
    assert state["activity"]["keys_examined"] == 0


def test_resuming_reads_the_corpus_that_is_already_there():
    """`oldest`: no start time at all, so the whole bucket is in scope. The
    only way to exercise the live path against alarms that already exist."""
    state = new_state(FeedCaps(10, 10), now=datetime(2026, 4, 5, tzinfo=UTC), resume=True)
    seen = []
    poll_once(
        state,
        client=FakeS3([key(1), key(2)]),
        now=datetime(2026, 4, 5, tzinfo=UTC),
        processor=lambda clip, *_: seen.append(clip.key) or {"status": "judged", "calls": 1},
    )

    assert seen == [key(1), key(2)]


def test_a_stale_watermark_cannot_outrank_a_start_time():
    """Carried over, it would silently win: `poll_once` only applies a start
    time when there is no watermark, so the run would resume mid-corpus while
    the screen said it was starting from today."""
    previous = {"watermark": key(1), "processed_keys": {key(1): {"status": "judged"}}}
    fresh = new_state(
        FeedCaps(10, 10),
        now=datetime(2026, 4, 5, tzinfo=UTC),
        previous=previous,
        start_on_or_after=datetime(2026, 4, 5, tzinfo=UTC),
    )

    assert fresh["watermark"] is None
    # What has already been judged still carries: that is what makes a re-run
    # idempotent, whichever end of the feed it starts from.
    assert fresh["processed_keys"] == previous["processed_keys"]

    resumed = new_state(
        FeedCaps(10, 10), now=datetime(2026, 4, 5, tzinfo=UTC), previous=previous, resume=True
    )
    assert resumed["watermark"] == key(1)


def test_a_cap_below_one_alarm_says_so_instead_of_looking_busy():
    """A per-run cap of 1 skips every alarm forever -- whole frames only. The
    screen said "running" and nothing else, which is the report that brought
    this here."""
    state = new_state(FeedCaps(1, 1), now=datetime(2026, 4, 1, tzinfo=UTC), resume=True)
    poll_once(
        state,
        client=FakeS3([key(1)]),
        now=datetime(2026, 4, 1, tzinfo=UTC),
        processor=lambda *_: {"status": "not_judged_budget", "calls": 0, "candidates": 3, "needed": 6},
    )

    assert state["paused_reason"] == "budget"
    assert "needs 6 calls" in state["paused_detail"]
    assert "3 people" in state["paused_detail"]
    assert state["watermark"] is None


def test_an_old_failure_is_not_attributed_to_a_new_run():
    """The list is carried forward on purpose -- an unjudged alarm stays on
    the record -- but reported as one number it made every fresh run look
    like it was failing, including runs that had made no calls at all."""
    previous = {
        "failures": [{"clip_key": key(1), "error": "FileNotFoundError: few_shot/manifest.json",
                      "run_id": "live-old"}],
    }
    state = new_state(FeedCaps(10, 10), now=datetime(2026, 4, 5, tzinfo=UTC), previous=previous)

    assert len(state["failures"]) == 1
    assert state["failures"][0]["run_id"] != state["run_id"]


def test_a_clip_that_can_never_be_extracted_stops_blocking_the_queue():
    """MEASURED ON A REAL RUN: 8 failure records, one clip. A failure holds
    the cursor so an unjudged alarm is never skipped silently -- but about 14%
    of this corpus never yields a clean alarm frame, and retrying one of those
    forever means every alarm behind it is never judged while the screen shows
    a rising failure count for a feed that has stopped."""
    state = new_state(FeedCaps(50, 50), now=datetime(2026, 4, 1, tzinfo=UTC), resume=True)
    seen = []

    def processor(clip, *_):
        seen.append(clip.key)
        if clip.key == key(1):
            return {"status": "failed", "calls": 0, "error": "RuntimeError: no clean alarm frame"}
        return {"status": "judged", "calls": 1}

    client = FakeS3([key(1), key(2)])
    for _ in range(4):
        poll_once(state, client=client, now=datetime(2026, 4, 1, tzinfo=UTC), processor=processor)

    # Three attempts on the broken clip, then past it and on to the next.
    assert seen == [key(1), key(1), key(1), key(2)]
    assert state["skipped_keys"][key(1)]["attempts"] == 3
    assert key(2) in state["processed_keys"]
    # ONE record for the one broken clip, counting up -- not one per attempt.
    assert len(state["failures"]) == 1
    assert state["failures"][0]["attempts"] == 3


def test_a_skipped_clip_is_not_retried_on_the_next_run():
    """A deterministic failure is still deterministic tomorrow. Three more
    attempts per run would put the same stall back."""
    previous = {"skipped_keys": {key(1): {"error": "boom", "attempts": 3}}}
    state = new_state(
        FeedCaps(50, 50), now=datetime(2026, 4, 1, tzinfo=UTC), previous=previous, resume=True
    )
    seen = []
    poll_once(
        state,
        client=FakeS3([key(1), key(2)]),
        now=datetime(2026, 4, 1, tzinfo=UTC),
        processor=lambda clip, *_: seen.append(clip.key) or {"status": "judged", "calls": 1},
    )

    assert seen == [key(2)]


def test_a_frame_with_nobody_in_it_reads_as_no_violation():
    """Every detection was a ship. No person means no person in violation,
    and the gated crops' own rows say exactly that -- so the feed's column
    must not read "not judged" while the audit case reads "no violation"."""
    from ppe_pipeline.helmet_cv.live_feed import frame_verdict

    assert frame_verdict({"tp": 0, "fp": 0, "escalated": 0, "gated": 12}) == "no_violation"
    assert frame_verdict({"tp": 0, "fp": 0, "escalated": 0, "gated": 0}) is None
    # A real answer always outranks the gate's.
    assert frame_verdict({"tp": 1, "fp": 0, "escalated": 0, "gated": 11}) == "violation"
    assert frame_verdict({"tp": 0, "fp": 0, "escalated": 1, "gated": 11}) == "cannot_determine"


def test_a_person_the_judge_could_not_read_leaves_the_frame_undecided():
    """TP > escalated > FP, the order every other site ORs in. A frame is not
    cleared because the judge cleared the people it COULD read."""
    from ppe_pipeline.helmet_cv.live_feed import frame_verdict

    assert frame_verdict({"tp": 0, "fp": 3, "escalated": 1}) == "cannot_determine"
    assert frame_verdict({"tp": 1, "fp": 3, "escalated": 1}) == "violation"
    assert frame_verdict({"tp": 0, "fp": 3, "escalated": 0}) == "no_violation"


def test_clearing_failures_keeps_the_cursor_and_the_paid_work():
    """The records go; the watermark and the processed-alarm list stay. Losing
    those would re-judge alarms already paid for."""
    from ppe_pipeline.helmet_cv.live_feed import clear_failures

    state = {
        "watermark": key(2),
        "processed_keys": {key(1): {"status": "judged"}},
        "failures": [{"clip_key": key(3), "error": "boom", "attempts": 3}],
        "skipped_keys": {key(3): {"error": "boom", "attempts": 3}},
        "attempts": {key(3): 3},
        "paused_reason": "budget",
        "paused_detail": "spent",
    }

    clear_failures(state)

    assert state["failures"] == []
    assert state["skipped_keys"] == {}
    assert state["attempts"] == {}
    assert "paused_reason" not in state
    assert state["watermark"] == key(2)
    assert state["processed_keys"] == {key(1): {"status": "judged"}}


def test_the_alarm_in_flight_is_recorded_before_the_work_starts():
    """A feed working through its first alarm looked identical to one doing
    nothing: `calls` only moves when a whole frame is done and `updated_at`
    only when a result is recorded, so the screen said "waiting on the first
    alarm" for minutes while it was judging one."""
    state = new_state(FeedCaps(10, 10), now=datetime(2026, 4, 1, tzinfo=UTC), resume=True)
    seen = []

    def processor(clip, *_):
        # What the screen would have to show while this runs.
        seen.append(dict(state.get("in_flight") or {}))
        return {"status": "judged", "calls": 1}

    poll_once(
        state,
        client=FakeS3([key(1)]),
        now=datetime(2026, 4, 1, tzinfo=UTC),
        processor=processor,
        checkpoint=lambda _s: None,
    )

    assert seen[0]["clip_key"] == key(1)
    assert seen[0]["alarm_id"]
    assert seen[0]["since"]
    # And it is gone once the alarm is done, so nothing stale is left on screen.
    assert state.get("in_flight") is None


def test_a_stopped_run_leaves_nothing_in_flight():
    state = new_state(FeedCaps(10, 10), now=datetime(2026, 4, 1, tzinfo=UTC), resume=True)
    state["in_flight"] = {"clip_key": key(1), "alarm_id": "x", "since": "2026-04-01T00:00:00+00:00"}
    state["status"] = "stopped"

    poll_once(state, client=FakeS3([key(1)]), now=datetime(2026, 4, 1, tzinfo=UTC),
              processor=lambda *_: {"status": "judged", "calls": 1})

    assert state.get("in_flight") is None


# --- the same alarm, twice, in the feed ----------------------------------


def test_an_image_already_judged_is_skipped_before_it_is_downloaded(monkeypatch):
    """The vendor re-serves one stale frame with a fresh alarm id: 6,818
    redundant alarms across the corpus, 21.8% of it. Caught on the ETag from
    the LISTING, so it costs no download and no judge call."""
    state = new_state(FeedCaps(10, 10), now=datetime(2026, 4, 1, tzinfo=UTC))
    state["processed_images"] = {"deadbeef": key(1)}

    class _WithEtags(FakeS3):
        def get_paginator(self, name):
            paginator = super().get_paginator(name)
            original = paginator.paginate

            def paginate(**kwargs):
                for page in original(**kwargs):
                    for obj in page["Contents"]:
                        obj["ETag"] = '"deadbeef"'
                    yield page

            paginator.paginate = paginate
            return paginator

    called = []
    poll_once(
        state,
        client=_WithEtags([key(2)]),
        now=datetime(2026, 4, 1, tzinfo=UTC),
        processor=lambda *a: called.append(a) or {"status": "judged", "calls": 2},
    )

    assert called == []  # never reached the processor
    assert state["duplicates_skipped"] == 1
    assert state["processed_keys"][key(2)]["status"] == "duplicate_image"
    assert state["calls"] == 0
    # The cursor still moves: a skip is a processed alarm, not a gap.
    assert state["watermark"] == key(2)


def test_a_judged_alarm_joins_the_recent_scene_ring():
    state = new_state(FeedCaps(10, 10), now=datetime(2026, 4, 1, tzinfo=UTC))

    poll_once(
        state,
        client=FakeS3([key(3)]),
        now=datetime(2026, 4, 1, tzinfo=UTC),
        processor=lambda *a: {
            "status": "judged", "calls": 1,
            "scene_hash": "0f0f0f0f0f0f0f0f", "boxes": [[1.0, 2.0, 3.0, 4.0]],
        },
    )

    assert [s["scene_hash"] for s in state["recent_scenes"]] == ["0f0f0f0f0f0f0f0f"]


def test_a_duplicate_does_not_join_the_ring():
    """It would chain the window forward off an alarm nobody judged."""
    state = new_state(FeedCaps(10, 10), now=datetime(2026, 4, 1, tzinfo=UTC))

    poll_once(
        state,
        client=FakeS3([key(4)]),
        now=datetime(2026, 4, 1, tzinfo=UTC),
        processor=lambda *a: {
            "status": "duplicate_event", "calls": 0,
            "scene_hash": "0f0f0f0f0f0f0f0f", "boxes": [],
        },
    )

    assert state.get("recent_scenes", []) == []


def test_the_same_camera_firing_twice_on_one_moment_is_not_judged_twice(monkeypatch):
    """The other repetition: the view is the same but the JPEG is not, so no
    equality test sees it. Both tests have to agree -- the scene AND the
    boxes."""
    _stub_live_snapshot(monkeypatch, judged=1, stopped=0)
    monkeypatch.setattr(live_feed, "scene_hash", lambda _b: "0f0f0f0f0f0f0f0f")
    monkeypatch.setattr(live_feed, "extract_boxes", lambda _b: [(10.0, 10.0, 60.0, 110.0)])

    recent = [{
        "key": key(1),
        "at": SNAPSHOT.start.isoformat(),
        "scene_hash": "0f0f0f0f0f0f0f0f",
        "boxes": [[10.0, 10.0, 60.0, 110.0]],
    }]
    result = live_feed.process_live_snapshot(
        SNAPSHOT, 10, "live-1", client=_FakeClient(), few_shot_library=None,
        # BOTH RINGS, carrying the same entries, which is what vendor mode
        # keeps: the image the judge is shown and the image that named the
        # alarm are the same still, so `poll_once` fills both from it.
        recent=recent, recent_annotated=recent,
    )

    assert result["status"] == "duplicate_event"
    assert result["calls"] == 0
    assert result["duplicate_of"] == key(1)


def test_the_same_view_with_different_boxes_IS_judged(monkeypatch):
    """13 of 17 same-scene pairs measured had boxes that had moved. Those are
    a different set of people flagged, and skipping them loses the finding."""
    _stub_live_snapshot(monkeypatch, judged=1, stopped=0)
    monkeypatch.setattr(live_feed, "scene_hash", lambda _b: "0f0f0f0f0f0f0f0f")
    monkeypatch.setattr(live_feed, "extract_boxes", lambda _b: [(10.0, 10.0, 60.0, 110.0)])

    recent = [{
        "key": key(1),
        "at": SNAPSHOT.start.isoformat(),
        "scene_hash": "0f0f0f0f0f0f0f0f",
        # Two people last time, one now.
        "boxes": [[10.0, 10.0, 60.0, 110.0], [400.0, 10.0, 450.0, 110.0]],
    }]
    result = live_feed.process_live_snapshot(
        SNAPSHOT, 10, "live-1", client=_FakeClient(), few_shot_library=None,
        recent=recent, recent_annotated=recent,
    )

    assert result["status"] == "judged"


def test_the_same_view_an_hour_later_IS_judged(monkeypatch):
    """Beyond the window the same view is a new moment, with the same people
    still standing there. Without it a quiet camera would have every alarm all
    day collapsed into its first."""
    _stub_live_snapshot(monkeypatch, judged=1, stopped=0)
    monkeypatch.setattr(live_feed, "scene_hash", lambda _b: "0f0f0f0f0f0f0f0f")
    monkeypatch.setattr(live_feed, "extract_boxes", lambda _b: [(10.0, 10.0, 60.0, 110.0)])

    stale = SNAPSHOT.start.replace(hour=(SNAPSHOT.start.hour + 2) % 24)
    recent = [{
        "key": key(1),
        "at": stale.isoformat(),
        "scene_hash": "0f0f0f0f0f0f0f0f",
        "boxes": [[10.0, 10.0, 60.0, 110.0]],
    }]
    result = live_feed.process_live_snapshot(
        SNAPSHOT, 10, "live-1", client=_FakeClient(), few_shot_library=None,
        recent=recent, recent_annotated=recent,
    )

    assert result["status"] == "judged"


def test_each_alarm_is_stamped_when_IT_finished(monkeypatch):
    """`now` is one reading taken once per poll, for the day boundary and the
    budget reset. A poll judges alarms until its budget runs out, so stamping
    every one of them with it gave a whole batch an identical time: measured on
    the live feed, 26 results carrying two distinct timestamps, 14 alarms
    sharing one of them."""
    # A CLOCK THAT NEVER REPEATS, rather than a fixed pair of readings: the
    # poll reads it for every checkpoint and every lease refresh as well as
    # for each result, and how many times it does that is not what this test
    # is about.
    ticks = count(1)

    class _Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 4, 1, 10, 0, tzinfo=UTC) + timedelta(seconds=next(ticks))

    monkeypatch.setattr(live_feed, "datetime", _Clock)
    state = new_state(FeedCaps(10, 10), now=datetime(2026, 4, 1, tzinfo=UTC))

    poll_once(
        state,
        client=FakeS3([key(1), key(2)]),
        now=datetime(2026, 4, 1, tzinfo=UTC),
        processor=lambda *a: {"status": "judged", "calls": 1},
    )

    stamps = [r["at"] for r in state["results"]]
    assert len(set(stamps)) == 2, "two alarms in one poll must not share a stamp"
    # And the alarm's own time, which is a different question entirely.
    assert [r["alarm_at"] for r in state["results"]] == [
        "2026-04-01T07:02:09", "2026-04-02T07:02:09"
    ]


def test_the_detector_option_finds_boxes_on_the_CLEAN_frame(monkeypatch, tmp_path):
    """Never on the snapshot: a detector run over a picture with red
    rectangles drawn on it is being handed the answer.

    THE SNAPSHOT IS FETCHED IN DETECTOR MODE NOW, so the old "not even
    fetched" clause is gone -- and it was never true anyway. `extract_clip`
    always GETs the still, in both modes, because the still is the key that
    names the clean alarm frame; the assertion only held here because the
    stub replaces `extract_clip` and hid that GET. What is load-bearing is
    what the DETECTOR is handed, and that is pinned below: the clean frame's
    bytes, never the annotated still's."""
    _stub_live_snapshot(monkeypatch, judged=2, stopped=0)
    seen: list[bytes] = []
    monkeypatch.setattr(
        live_feed,
        "get_person_detector",
        lambda _config: type("D", (), {"find": staticmethod(lambda b: seen.append(b) or [(1.0, 2.0, 3.0, 4.0)])})(),
    )
    monkeypatch.setattr(live_feed.Path, "read_bytes", lambda _self: b"clean-frame-bytes")

    result = live_feed.process_live_snapshot(
        SNAPSHOT, 10, "live-1", client=_FakeClient(), few_shot_library=None,
        box_source="detector",
    )

    assert result["status"] == "judged"
    assert seen == [b"clean-frame-bytes"]
    assert b"annotated-bytes" not in seen


def test_the_vendor_option_still_reads_the_snapshot(monkeypatch):
    _stub_live_snapshot(monkeypatch, judged=1, stopped=0)

    result = live_feed.process_live_snapshot(
        SNAPSHOT, 10, "live-1", client=_FakeClient(), few_shot_library=None,
        box_source="vendor_cv",
    )

    assert result["status"] == "judged"


# --- how many times one alarm fetches the annotated still -----------------
#
# ONE, and the count spans two modules. `extract_clip` -> `_locate_alarm_frame`
# has always fetched it unconditionally, in both modes, because the still is
# the KEY that names the clean alarm frame -- so the feed's own fetch made it
# two per alarm in vendor mode, of a ~0.9MB object. Neither module can be
# stubbed out of a test about that, so the fixture below lets the real
# `extract_clip` run over frames written to disk.


@pytest.fixture
def real_extraction(monkeypatch, tmp_path):
    """The real `extract_clip`, over two frames on disk, with no video.

    Only the download and the decode are replaced. `_locate_alarm_frame` runs
    for real, including its own fetch of the still -- which is the thing being
    counted.
    """
    import cv2
    import numpy as np

    from ppe_pipeline.helmet_cv import extract as extract_module

    def fake_download(_client, _bucket, _key):
        path = tmp_path / "clip.mp4"
        path.write_bytes(b"")
        return path

    def fake_sample(_video_path, clip, *, frames_dir, **_kwargs):
        records = []
        frames_dir.mkdir(parents=True, exist_ok=True)
        for index, fill in ((0, 0), (1, 255)):
            name = extract_module.frame_name(
                site="unk", camera="unk", frame_utc=clip.start,
                alarm_id=clip.alarm_id, frame_index=index,
            )
            cv2.imwrite(str(frames_dir / name), np.full((480, 640, 3), fill, dtype=np.uint8))
            records.append(
                extract_module.FrameRecord(
                    frame_name=name,
                    relative_path=f"frames/{clip.alarm_id}/{name}",
                    alarm_id=clip.alarm_id,
                    clip_key=clip.key,
                    violation_type=clip.violation_type,
                    frame_role=extract_module.FRAME_ROLE_CLIP,
                    site="unk",
                    camera="unk",
                    frame_index=index,
                    offset_ms=0,
                    frame_utc=clip.start.isoformat(),
                    width=640,
                    height=480,
                )
            )
        return records, 1.0, "container", len(records)

    monkeypatch.setattr(extract_module, "_download_to_temp", fake_download)
    monkeypatch.setattr(extract_module, "_sample_frames", fake_sample)


def _white_jpeg() -> bytes:
    """A still matching the second frame the fixture writes."""
    import cv2
    import numpy as np

    ok, buf = cv2.imencode(".jpg", np.full((480, 640, 3), 255, dtype=np.uint8))
    assert ok
    return buf.tobytes()


class _CountingClient:
    """Serves the still and counts every GET of it."""

    def __init__(self, body: bytes):
        self.body = body
        self.requested: list[str] = []

    def get_object(self, Bucket=None, Key=None, **_kwargs):  # noqa: N803 - boto3's own casing
        self.requested.append(Key)
        return {"Body": type("B", (), {"read": staticmethod(lambda: self.body)})()}


def test_vendor_cv_fetches_the_snapshot_exactly_once_per_alarm(
    monkeypatch, real_extraction, tmp_path
):
    """It was twice: once here for the vendor's boxes, and once inside
    `_locate_alarm_frame` for the same object. ~0.9MB per alarm, paid twice."""
    monkeypatch.setattr(live_feed, "get_config", PipelineConfig)
    monkeypatch.setattr(live_feed.CvProvenance, "from_config", classmethod(lambda cls, *a, **k: {}))
    monkeypatch.setattr(live_feed, "clip_key_for", lambda *a, **k: CLIP_KEY)
    monkeypatch.setattr(live_feed, "extract_boxes", lambda _bytes: [(1.0, 2.0, 3.0, 4.0)])
    monkeypatch.setattr(live_feed, "records_from_boxes", lambda *a, **k: [])
    monkeypatch.setattr(
        live_feed, "judge_candidates",
        lambda *a, **k: {"judged": 1, "skipped": 0, "failed": 0, "stopped": 0},
    )
    client = _CountingClient(_white_jpeg())

    result = live_feed.process_live_snapshot(
        SNAPSHOT, 10, "live-1", client=client, few_shot_library=None, box_source="vendor_cv"
    )

    assert result["status"] == "judged"
    assert client.requested == [SNAPSHOT.key]


class _NoSuchKey:
    """Every GET fails, the way a deleted key or a denied bucket does."""

    calls: list[str] = []

    @staticmethod
    def get_object(Bucket=None, Key=None, **_kwargs):  # noqa: N803 - boto3's own casing
        _NoSuchKey.calls.append(Key)
        raise RuntimeError("An error occurred (NoSuchKey) when calling the GetObject operation")


def test_a_vendor_alarm_with_no_still_fails_verbatim_before_it_spends_anything(
    monkeypatch, real_extraction, tmp_path
):
    """WHERE IT HAS ALWAYS FAILED, and with the S3 error unchanged.

    Vendor mode cannot proceed without this object -- the boxes it judges are
    drawn on it -- so the fetch raises from here, `poll_once` records
    `NoSuchKey` or `AccessDenied` verbatim, holds the cursor, and the alarm is
    retried having cost nothing.

    Swallowing it instead was worse than a lost diagnostic. An empty still
    recovers no boxes, no boxes is not an error downstream, and the alarm
    would be judged on the whole frame and marked done -- so a network blip
    silently cleared an alarm the vendor's CV had flagged. The listing below
    is wired to blow up because the cost matters too: master reached this
    conclusion before the clip was even listed."""
    monkeypatch.setattr(live_feed, "get_config", PipelineConfig)

    def _never_list(*_args, **_kwargs):
        raise AssertionError("a still-less vendor alarm must fail before its clip is listed")

    monkeypatch.setattr(live_feed, "clip_key_for", _never_list)

    with pytest.raises(RuntimeError, match="NoSuchKey"):
        live_feed.process_live_snapshot(
            SNAPSHOT, 10, "live-1", client=_NoSuchKey(), few_shot_library=None,
            box_source="vendor_cv",
        )


def test_a_transient_failure_on_the_still_never_quietly_judges_a_vendor_alarm(monkeypatch):
    """A THROTTLE IS NOT AN ANSWER. `SlowDown`, a read timeout, a reset
    mid-`.read()`: swallowed, each of them handed the judge one whole-frame
    candidate with no person box -- the 1920x1080 case measured at
    "helmet_worn" 6 times in 7 -- spent a call on it, and recorded the alarm
    as judged so it was never retried."""
    _stub_live_snapshot(monkeypatch, judged=1, stopped=0)
    judged = []
    monkeypatch.setattr(
        live_feed, "judge_candidates",
        lambda *a, **k: judged.append(a) or {"judged": 1, "skipped": 0, "failed": 0, "stopped": 0},
    )

    class _Flaky:
        """The still is there. This one GET is not answered."""

        @staticmethod
        def get_object(**_kwargs):
            raise RuntimeError("SlowDown: please reduce your request rate")

    with pytest.raises(RuntimeError, match="SlowDown"):
        live_feed.process_live_snapshot(
            SNAPSHOT, 10, "live-1", client=_Flaky(), few_shot_library=None,
            box_source="vendor_cv",
        )

    assert judged == [], "nothing may be judged from a still that was never read"


def test_a_detector_alarm_with_no_still_loses_its_alarm_frame_as_it_always_did(
    monkeypatch, real_extraction, tmp_path
):
    """The other half of the same condition. Detector mode never fetched the
    still here, so there is no failure of THIS fetch to preserve: the alarm
    dies one layer down in `_locate_alarm_frame`, which has always swallowed
    the error and named no frame. Raising the S3 error out of a fetch master
    never made would be a new failure mode for an unchanged condition."""
    monkeypatch.setattr(live_feed, "get_config", PipelineConfig)
    monkeypatch.setattr(live_feed, "clip_key_for", lambda *a, **k: CLIP_KEY)
    # And nothing tries to read boxes out of nothing: `cv2.imdecode` of empty
    # bytes raises an assertion that `extract_boxes` logs at ERROR with a
    # traceback, per still-less alarm, for an expected condition.
    monkeypatch.setattr(
        live_feed, "extract_boxes",
        lambda payload: (_ for _ in ()).throw(AssertionError("decoded an empty still"))
        if not payload else [],
    )

    with pytest.raises(RuntimeError, match="clean alarm frame could not be located"):
        live_feed.process_live_snapshot(
            SNAPSHOT, 10, "live-1", client=_NoSuchKey(), few_shot_library=None,
            box_source="detector",
        )


# --- the duplicate caught BEFORE the clip is downloaded --------------------
#
# In detector mode the cheap check was a structural no-op: nothing fetched the
# annotated still before extraction, so the fingerprint it compared was empty
# and it returned None for every alarm. The real check ran on the clean frame,
# AFTER a clip download, an ffmpeg decode and a detector pass -- which is the
# entire cost of a duplicate, paid on every one of them.
#
# The still is now read first, so the same question can be asked in snapshot
# space before any of that. What is NOT settled is whether the two spaces
# agree, so `PREFILTER_SKIPS_EXTRACTION` is False and the pre-filter only
# records its verdict for now. Both halves are pinned here: the shadow
# behaviour that ships, and the skipping behaviour the switch turns on.

_ANNOTATED_HASH = "0f0f0f0f0f0f0f0f"
_CLEAN_HASH = "f0f0f0f0f0f0f0f0"
#: What the stubbed `extract_boxes` recovers from the annotated still.
_ANNOTATED_BOXES = [[1.0, 2.0, 3.0, 4.0]]
#: What the stubbed detector finds on the clean frame. Deliberately nothing
#: like the boxes above: two pictures of one moment, described two ways.
_DETECTOR_BOXES = [(10.0, 10.0, 60.0, 110.0)]


def _two_spaces(monkeypatch):
    """Hash the annotated still and the clean frame to DIFFERENT values.

    A single stubbed hash would let a test pass while the code compared the
    clean frame against the snapshot ring, which is the mistake worth
    catching."""
    monkeypatch.setattr(
        live_feed,
        "scene_hash",
        lambda payload: _ANNOTATED_HASH if payload == b"annotated-bytes" else _CLEAN_HASH,
    )
    monkeypatch.setattr(
        live_feed,
        "get_person_detector",
        lambda _config: type("D", (), {"find": staticmethod(lambda _b: list(_DETECTOR_BOXES))})(),
    )
    monkeypatch.setattr(live_feed.Path, "read_bytes", lambda _self: b"clean-frame-bytes")


def _ring(scene_hash, boxes):
    return [{
        "key": key(1),
        "at": SNAPSHOT.start.isoformat(),
        "scene_hash": scene_hash,
        "boxes": boxes,
    }]


def test_in_shadow_mode_the_pre_filter_records_its_verdict_and_judges_anyway(monkeypatch):
    """The measurement `PREFILTER_SKIPS_EXTRACTION` was added for: acting on
    the snapshot-space verdict means the clean-frame one is never asked, so
    the alarm is extracted and judged and both answers land on it.

    The measurement is DONE and the switch now ships on, but this pins the
    shadow half so the 2x2 can be taken again -- against another camera,
    another box source, or a changed window -- without rebuilding it. Pinned
    by monkeypatch and not by reading the shipped default: a test that asserts
    which way a constant is set fails when someone flips it, which is a
    decision, not a regression."""
    monkeypatch.setattr(live_feed, "PREFILTER_SKIPS_EXTRACTION", False)
    _stub_live_snapshot(monkeypatch, judged=1, stopped=0)
    _two_spaces(monkeypatch)

    result = live_feed.process_live_snapshot(
        SNAPSHOT, 10, "live-1", client=_FakeClient(), few_shot_library=None,
        box_source="detector",
        recent_annotated=_ring(_ANNOTATED_HASH, _ANNOTATED_BOXES),
    )

    assert result["status"] == "judged"
    assert result["calls"] == 1
    assert result["pre_extraction_duplicate_of"] == key(1)
    # And both descriptions of the alarm, so the 2x2 can be read off the state.
    assert result["annotated_hash"] == _ANNOTATED_HASH
    assert result["scene_hash"] == _CLEAN_HASH


def test_with_the_switch_on_a_snapshot_space_duplicate_never_reaches_the_clip(monkeypatch):
    """What flipping the constant buys: no clip listing, no download, no
    decode, no detector pass. Both are wired to blow up here, so a duplicate
    that quietly still pays for them cannot pass."""
    monkeypatch.setattr(live_feed, "PREFILTER_SKIPS_EXTRACTION", True)
    _stub_live_snapshot(monkeypatch, judged=1, stopped=0)
    _two_spaces(monkeypatch)

    def _never_extract(*_args, **_kwargs):
        raise AssertionError("a pre-filtered duplicate must not be extracted")

    def _never_list(*_args, **_kwargs):
        # The listing is wasted work on a duplicate too, so the check sits
        # above it rather than merely above the download.
        raise AssertionError("a pre-filtered duplicate must not list for its clip")

    monkeypatch.setattr(live_feed, "extract_clip", _never_extract)
    monkeypatch.setattr(live_feed, "clip_key_for", _never_list)

    result = live_feed.process_live_snapshot(
        SNAPSHOT, 10, "live-1", client=_FakeClient(), few_shot_library=None,
        box_source="detector",
        recent_annotated=_ring(_ANNOTATED_HASH, _ANNOTATED_BOXES),
    )

    assert result["status"] == "duplicate_event"
    assert result["calls"] == 0
    assert result["duplicate_of"] == key(1)
    assert result["duplicate_stage"] == "pre_extraction"


def test_the_pre_filter_never_compares_the_clean_frame_ring_against_the_still(monkeypatch):
    """TWO RINGS, TWO SPACES. `recent_scenes` holds the clean frames the judge
    was shown; `recent_annotated` holds the vendor's stills. Reading the wrong
    one would dedupe an alarm against a hash of a different picture -- so this
    puts an entry matching the STILL into the clean ring and leaves the
    snapshot ring empty. The alarm is judged."""
    _stub_live_snapshot(monkeypatch, judged=1, stopped=0)
    _two_spaces(monkeypatch)

    result = live_feed.process_live_snapshot(
        SNAPSHOT, 10, "live-1", client=_FakeClient(), few_shot_library=None,
        box_source="detector",
        recent=_ring(_ANNOTATED_HASH, _ANNOTATED_BOXES),
        recent_annotated=[],
    )

    assert result["status"] == "judged"
    assert result["pre_extraction_duplicate_of"] is None


def test_the_check_after_extraction_still_catches_what_the_pre_filter_misses(monkeypatch):
    """The clean-frame check is not replaced by the cheap one. It is the only
    one that sees what the judge sees, and while the switch is off it is still
    the one that decides."""
    _stub_live_snapshot(monkeypatch, judged=1, stopped=0)
    _two_spaces(monkeypatch)

    result = live_feed.process_live_snapshot(
        SNAPSHOT, 10, "live-1", client=_FakeClient(), few_shot_library=None,
        box_source="detector",
        recent=_ring(_CLEAN_HASH, [list(b) for b in _DETECTOR_BOXES]),
        recent_annotated=[],
    )

    assert result["status"] == "duplicate_event"
    assert result["duplicate_of"] == key(1)
    assert result["duplicate_stage"] == "post_extraction"
    # The two are never conflated: this one cost the download and the decode.
    assert result["pre_extraction_duplicate_of"] is None


def test_a_pre_filtered_duplicate_joins_neither_ring():
    """Same reason as the clean-frame ring's: an alarm nobody judged would
    chain the window forward off itself. It has to be true of both."""
    state = new_state(FeedCaps(10, 10), now=datetime(2026, 4, 1, tzinfo=UTC))

    poll_once(
        state,
        client=FakeS3([key(4)]),
        now=datetime(2026, 4, 1, tzinfo=UTC),
        processor=lambda *a: {
            "status": "duplicate_event", "calls": 0,
            "scene_hash": _ANNOTATED_HASH, "boxes": _ANNOTATED_BOXES,
            "duplicate_of": key(1), "duplicate_stage": "pre_extraction",
        },
    )

    assert state.get("recent_scenes", []) == []
    assert state.get("recent_annotated", []) == []
    # And it is counted where the screen can see it -- separately from
    # `duplicates_skipped`, which is the free ETag catch.
    assert state["pre_extraction_duplicates"] == 1
    assert state.get("duplicates_skipped", 0) == 0


def test_a_judged_detector_alarm_joins_both_rings_with_different_hashes():
    """The clean frame in one, the vendor's still in the other. If they ever
    carry the same value the two spaces have been collapsed, and the
    pre-filter is comparing across them."""
    state = new_state(FeedCaps(10, 10), now=datetime(2026, 4, 1, tzinfo=UTC))

    poll_once(
        state,
        client=FakeS3([key(3)]),
        now=datetime(2026, 4, 1, tzinfo=UTC),
        processor=lambda *a: {
            "status": "judged", "calls": 1,
            "scene_hash": _CLEAN_HASH, "boxes": [list(b) for b in _DETECTOR_BOXES],
            "annotated_hash": _ANNOTATED_HASH, "annotated_boxes": _ANNOTATED_BOXES,
            "pre_extraction_duplicate_of": None,
        },
    )

    assert [s["scene_hash"] for s in state["recent_scenes"]] == [_CLEAN_HASH]
    assert [s["scene_hash"] for s in state["recent_annotated"]] == [_ANNOTATED_HASH]
    assert state["recent_scenes"][0]["scene_hash"] != state["recent_annotated"][0]["scene_hash"]
    assert state["recent_scenes"][0]["boxes"] == [list(b) for b in _DETECTOR_BOXES]
    assert state["recent_annotated"][0]["boxes"] == _ANNOTATED_BOXES
    assert state.get("pre_extraction_duplicates", 0) == 0


def test_a_new_run_starts_with_an_empty_snapshot_ring():
    """Like `recent_scenes`, and for the same reason: a run fixes its box
    source at the start, so a ring carried in from a run of the other kind
    would hold entries from a space this one never compares against."""
    previous = {
        "recent_scenes": [{"key": key(1), "scene_hash": _CLEAN_HASH, "boxes": []}],
        "recent_annotated": [{"key": key(1), "scene_hash": _ANNOTATED_HASH, "boxes": []}],
    }

    state = new_state(FeedCaps(10, 10), now=datetime(2026, 4, 5, tzinfo=UTC), previous=previous)

    assert "recent_annotated" not in state
    assert "recent_scenes" not in state


# --- what a result costs to store -----------------------------------------
#
# Every result dict is kept per alarm in `processed_keys`, which is unbounded,
# carried across runs, rewritten through `update_json` on every checkpoint and
# returned whole to the browser every two seconds. A field that repeats what
# the same dict already says is paid for on all four.


def test_a_vendor_result_carries_no_second_copy_of_its_own_scene(monkeypatch):
    """In vendor mode the still IS the image the alarm was judged from, so
    `annotated_hash`/`annotated_boxes` would be byte-identical copies of
    `scene_hash`/`boxes` in the same dict -- measured at 661 -> 977 bytes an
    entry, ~+9.9MB over the 31k-alarm corpus. And `pre_extraction_duplicate_of`
    is a measurement only where the verdict is NOT acted on, which is detector
    mode; here it would read None on every alarm forever."""
    _stub_live_snapshot(monkeypatch, judged=1, stopped=0)
    _two_spaces(monkeypatch)

    result = live_feed.process_live_snapshot(
        SNAPSHOT, 10, "live-1", client=_FakeClient(), few_shot_library=None,
        box_source="vendor_cv",
    )

    assert result["status"] == "judged"
    assert result["scene_hash"] == _ANNOTATED_HASH
    assert "annotated_hash" not in result
    assert "annotated_boxes" not in result
    assert "pre_extraction_duplicate_of" not in result


def test_a_vendor_alarm_still_joins_the_snapshot_ring_from_its_one_copy():
    """The pre-filter reads `recent_annotated` in both modes, so vendor mode's
    ring has to be filled -- from the single copy its result carries. Empty,
    the check that catches the same camera firing twice would never fire."""
    state = new_state(FeedCaps(10, 10), now=datetime(2026, 4, 1, tzinfo=UTC))

    poll_once(
        state,
        client=FakeS3([key(3)]),
        now=datetime(2026, 4, 1, tzinfo=UTC),
        processor=lambda *a: {
            "status": "judged", "calls": 1,
            "scene_hash": _ANNOTATED_HASH, "boxes": _ANNOTATED_BOXES,
        },
    )

    assert [s["scene_hash"] for s in state["recent_annotated"]] == [_ANNOTATED_HASH]
    assert state["recent_annotated"][0]["boxes"] == _ANNOTATED_BOXES
    # Both rings, from one copy: in vendor mode they describe the same image.
    assert [s["scene_hash"] for s in state["recent_scenes"]] == [_ANNOTATED_HASH]


def test_a_detector_alarm_that_could_not_read_its_still_does_not_fall_back(monkeypatch):
    """The fallback is by key PRESENCE, not truthiness. A detector result
    carries an empty `annotated_hash` when the still could not be read, and
    reading that as "absent" would put the CLEAN frame's hash into the
    snapshot ring -- the exact cross-space comparison the two rings exist to
    prevent."""
    state = new_state(FeedCaps(10, 10), now=datetime(2026, 4, 1, tzinfo=UTC))

    poll_once(
        state,
        client=FakeS3([key(3)]),
        now=datetime(2026, 4, 1, tzinfo=UTC),
        processor=lambda *a: {
            "status": "judged", "calls": 1,
            "scene_hash": _CLEAN_HASH, "boxes": [list(b) for b in _DETECTOR_BOXES],
            "annotated_hash": "", "annotated_boxes": [],
            "pre_extraction_duplicate_of": None,
        },
    )

    assert [s["scene_hash"] for s in state["recent_scenes"]] == [_CLEAN_HASH]
    assert state.get("recent_annotated", []) == []


def test_a_shadow_flagged_alarm_is_counted_and_kept_out_of_the_snapshot_ring():
    """THE SHADOW RING HAS TO BE THE RING A SKIPPING RUN WOULD HAVE.

    While `PREFILTER_SKIPS_EXTRACTION` is False the flagged alarm is judged,
    so its status is `judged` and it would join the snapshot ring -- but with
    the switch on it returns `duplicate_event` and does not. Left in, it flags
    a later alarm that a skipping run would have judged: a match needs the
    hash AND the boxes, so A->B->C chains form where C matches B but not A,
    and the shadow over-reports what flipping the constant saves. That number
    is the only thing the shadow is for.

    The clean-frame ring is the opposite case and keeps it: this run really
    did judge that alarm, and `recent_scenes` describes what this run did."""
    state = new_state(
        FeedCaps(10, 10), now=datetime(2026, 4, 1, tzinfo=UTC), box_source="detector"
    )

    poll_once(
        state,
        client=FakeS3([key(3)]),
        now=datetime(2026, 4, 1, tzinfo=UTC),
        processor=lambda *a: {
            "status": "judged", "calls": 1,
            "scene_hash": _CLEAN_HASH, "boxes": [list(b) for b in _DETECTOR_BOXES],
            "annotated_hash": _ANNOTATED_HASH, "annotated_boxes": _ANNOTATED_BOXES,
            "pre_extraction_duplicate_of": key(1),
        },
    )

    # Judged, and counted: the count is the measurement.
    assert state["processed_keys"][key(3)]["status"] == "judged"
    assert state["pre_extraction_duplicates"] == 1
    assert [s["scene_hash"] for s in state["recent_scenes"]] == [_CLEAN_HASH]
    assert state.get("recent_annotated", []) == []


def test_a_vendor_alarm_the_pre_filter_let_through_still_joins_the_ring():
    """The extra condition must not reach vendor mode. Nothing there carries
    `pre_extraction_duplicate_of` at all -- a flagged vendor alarm is a
    `duplicate_event` and was already excluded -- so an absent field has to
    read as "not flagged" rather than as a reason to drop the entry. Empty,
    the ring would leave the vendor pre-filter with nothing to compare."""
    state = new_state(FeedCaps(10, 10), now=datetime(2026, 4, 1, tzinfo=UTC))

    poll_once(
        state,
        client=FakeS3([key(3)]),
        now=datetime(2026, 4, 1, tzinfo=UTC),
        processor=lambda *a: {
            "status": "judged", "calls": 1,
            "scene_hash": _ANNOTATED_HASH, "boxes": _ANNOTATED_BOXES,
        },
    )

    assert [s["scene_hash"] for s in state["recent_annotated"]] == [_ANNOTATED_HASH]


# --- what the screen is told the pre-filter is doing -----------------------


@pytest.mark.parametrize(
    ("box_source", "skips", "acts"),
    [
        ("vendor_cv", False, True),
        ("vendor_cv", True, True),
        ("detector", False, False),
        ("detector", True, True),
    ],
)
def test_the_run_records_whether_a_pre_extraction_duplicate_is_actually_skipped(
    monkeypatch, box_source, skips, acts
):
    """RECORDED ON THE RUN, because the screen cannot re-derive it.

    From `box_source` alone the answer is right only while
    `PREFILTER_SKIPS_EXTRACTION` is False. The day it flips, a screen still
    inferring it would report "nothing was skipped" while alarms were being
    skipped and calls saved -- which is the one thing flipping it is for.

    The four rows are the gate in `process_live_snapshot`, from the outside."""
    monkeypatch.setattr(live_feed, "PREFILTER_SKIPS_EXTRACTION", skips)

    state = new_state(
        FeedCaps(10, 10), now=datetime(2026, 4, 1, tzinfo=UTC), box_source=box_source
    )

    assert state["box_source"] == box_source
    assert state["pre_extraction_acts"] is acts


def test_the_recorded_behaviour_matches_what_the_alarm_actually_does(monkeypatch):
    """The state's claim and the runtime gate are two copies of one condition,
    so they are checked against each other rather than each against itself: a
    run that says duplicates are skipped must skip one."""
    _stub_live_snapshot(monkeypatch, judged=1, stopped=0)
    _two_spaces(monkeypatch)
    monkeypatch.setattr(live_feed, "PREFILTER_SKIPS_EXTRACTION", True)

    state = new_state(
        FeedCaps(10, 10), now=datetime(2026, 4, 1, tzinfo=UTC), box_source="detector"
    )
    result = live_feed.process_live_snapshot(
        SNAPSHOT, 10, "live-1", client=_FakeClient(), few_shot_library=None,
        box_source="detector",
        recent_annotated=_ring(_ANNOTATED_HASH, _ANNOTATED_BOXES),
    )

    assert state["pre_extraction_acts"] is True
    assert result["status"] == "duplicate_event"
    assert result["duplicate_stage"] == "pre_extraction"


def test_every_processed_alarm_records_which_run_processed_it():
    """`processed_keys` is carried across runs forever and its entries said
    nothing about which run wrote them, so a second run merged into the first
    with no way to separate them afterwards. That is fatal to the shadow: two
    runs with `PREFILTER_SKIPS_EXTRACTION` in different positions would be one
    undifferentiated pile. Cheap to record now, unrecoverable later."""
    state = new_state(FeedCaps(10, 10), now=datetime(2026, 4, 1, tzinfo=UTC))
    state["processed_images"] = {"deadbeef": key(1)}

    class _WithEtags(FakeS3):
        def get_paginator(self, name):
            paginator = super().get_paginator(name)
            original = paginator.paginate

            def paginate(**kwargs):
                for page in original(**kwargs):
                    for obj in page["Contents"]:
                        obj["ETag"] = '"deadbeef"' if obj["Key"] == key(2) else '"fresh"'
                    yield page

            paginator.paginate = paginate
            return paginator

    poll_once(
        state,
        client=_WithEtags([key(2), key(3)]),
        now=datetime(2026, 4, 1, tzinfo=UTC),
        processor=lambda *a: {"status": "judged", "calls": 1, "scene_hash": "0f0f0f0f0f0f0f0f"},
    )

    # The judged alarm and the one skipped from the listing alike: both are
    # entries a later run will read, so both have to say whose they are.
    assert state["processed_keys"][key(3)]["run_id"] == state["run_id"]
    assert state["processed_keys"][key(2)]["status"] == "duplicate_image"
    assert state["processed_keys"][key(2)]["run_id"] == state["run_id"]


# --------------------------------------------------------------------------
# THE LEASE, WHICH IS WHAT STOPS TWO WORKERS SPENDING THE SAME BUDGET TWICE.
# --------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A storage root of its own, so nothing here can reach the real feed
    state at `eval/live_helmet_feed.json`. `LocalStorage` is rooted at the
    working directory and `DEFAULT_STATE_KEY` is a default argument, so the
    directory is the only honest place to isolate."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "eval").mkdir()
    return tmp_path


def _snapshots(n):
    return [parse_snapshot_key(key(day)) for day in range(1, n + 1)]


def _serve(monkeypatch, snapshots):
    """`iter_snapshots`, replaced, honouring `start_after_key` and `on_key`."""

    def fake(*_a, **kwargs):
        for snapshot in snapshots:
            after = kwargs.get("start_after_key")
            if kwargs.get("on_key") is not None:
                kwargs["on_key"]()
            if after is None or snapshot.key > after:
                yield snapshot

    monkeypatch.setattr(live_feed, "iter_snapshots", fake)


def test_a_rival_cannot_claim_the_lease_from_inside_one_long_alarm(store, monkeypatch):
    """The gap between the checkpoint that ANNOUNCES an alarm and the one that
    RECORDS it is the whole alarm -- about four minutes for a nine-crop frame,
    against `LEASE_SECONDS` of 90. Refreshing at those two edges leaves the
    lease expired for most of every alarm, and `claim_state` would hand a run
    that is spending money to a second process."""
    monkeypatch.setattr(live_feed, "LEASE_SECONDS", 0.3)
    _serve(monkeypatch, _snapshots(1))
    state = new_state(FeedCaps(100, 100))
    assert live_feed.claim_state(state)
    live_feed.save_state(state)

    stolen = []

    def processor(clip, budget, run_id, reporter):
        reporter.stage("judging", crops_total=9, crops_judged=0)
        for _ in range(9):                       # nine crops at ~25s each
            time.sleep(0.05)
            reporter.crop_judged()
            rival = new_state(FeedCaps(100, 100))
            if live_feed.claim_state(rival):
                stolen.append(rival["lease_owner"])
        return {"status": "judged", "calls": 0}

    poll_once(state, client=None, processor=processor, checkpoint=live_feed.save_state)

    assert stolen == [], "a rival took a lease the worker was actively holding"
    assert live_feed.stop_requested(state) is False


def test_a_checkpoint_does_not_resurrect_a_lease_this_worker_lost(store, monkeypatch):
    """`save_state` is a blind whole-object overwrite, so a worker that had
    already lost the lease put its own `lease_owner` back on every checkpoint.
    That inverted the rule: "whoever claimed it wins" became "whoever
    checkpointed last wins", and the rival was then told to stand down while
    the zombie kept the run."""
    monkeypatch.setattr(live_feed, "LEASE_SECONDS", 0.2)
    old = new_state(FeedCaps(100, 100))
    assert live_feed.claim_state(old)
    live_feed.save_state(old)
    time.sleep(0.3)                              # the old worker is mid-alarm

    rival = new_state(FeedCaps(100, 100))
    assert live_feed.claim_state(rival)

    live_feed.save_state(old)                    # the old worker's next checkpoint

    assert live_feed.load_state()["lease_owner"] == rival["lease_owner"]
    # And the loser is the one told to stand down, not the winner.
    assert live_feed.stop_requested(old) is True
    assert live_feed.stop_requested(rival) is False


def test_losing_the_lease_stops_the_run_rather_than_spending_on(store, monkeypatch):
    """Discovered inside the write, where the read is already paid for."""
    monkeypatch.setattr(live_feed, "LEASE_SECONDS", 0.2)
    old = new_state(FeedCaps(100, 100))
    live_feed.claim_state(old)
    live_feed.save_state(old)
    time.sleep(0.3)
    live_feed.claim_state(new_state(FeedCaps(100, 100)))

    live_feed.save_state(old)

    assert old["status"] == "stopped"
    assert "took over" in old["stopped_reason"]


def test_a_state_that_never_claimed_a_lease_still_writes(store):
    """Most callers, and every test that passes no checkpoint: a state with no
    `lease_owner` has lost nothing and must not be refused."""
    state = new_state(FeedCaps(10, 10))
    live_feed.save_state(state)

    assert live_feed.load_state()["run_id"] == state["run_id"]
    assert state["status"] == "running"


# --------------------------------------------------------------------------
# WHAT THE SCREEN IS TOLD.
# --------------------------------------------------------------------------


def test_keys_examined_counts_the_scan_that_yields_nothing(monkeypatch):
    """The reading exists to explain a scan that yields NOTHING, which is still
    what a resumed run does while it steps over keys it has already judged --
    and what any run does over keys the client-side time filter drops. Counted
    in the loop body it read 0 for the whole window it was added to describe.

    The paginator here ignores `StartAfter` on purpose: the derived cursor
    (OPEN_TASKS #39) is not what is under test, the counting is."""
    seen = []

    class _Paginator:
        def paginate(self, **_kwargs):
            yield {"Contents": [{"Key": key(day)} for day in range(1, 29)]}

    class _S3:
        def get_paginator(self, _name):
            return _Paginator()

    state = new_state(
        FeedCaps(10, 10),
        # Later than every key above, so every one is filtered out.
        start_on_or_after=datetime(2027, 1, 1, tzinfo=UTC),
    )

    def processor(*_a):                                   # never reached
        raise AssertionError("no key should have been yielded")

    original = live_feed.iter_snapshots

    def watching(*a, **k):
        inner = k.pop("on_key", None)

        def on_key():
            inner()
            seen.append(state["activity"]["keys_examined"])

        yield from original(*a, on_key=on_key, **k)

    monkeypatch.setattr(live_feed, "iter_snapshots", watching)
    poll_once(state, client=_S3(), processor=processor)

    assert seen == list(range(1, 29)), "the reading did not move through the scan"
    assert state["activity"]["stage"] == "listing"
    assert state["activity"]["keys_examined"] == 28


def test_in_flight_survives_the_one_exception_poll_once_raises(monkeypatch):
    """The budget guard is the only thing `poll_once` raises itself, so it is
    the case that reliably produces a dead run. Clearing `in_flight` first left
    the worker's handler nothing to name but `watermark` -- the PREVIOUS,
    successful alarm -- so the failure record blamed the wrong key."""
    snapshots = _snapshots(2)
    _serve(monkeypatch, snapshots)
    state = new_state(FeedCaps(100, 100))
    seen = count()

    def processor(clip, budget, run_id, reporter):
        if next(seen) == 0:
            return {"status": "judged", "calls": 1}
        reporter.stage("judging", crops_total=9, crops_judged=0)
        reporter.crop_judged()
        return {"status": "judged", "calls": budget + 1}      # overspends

    with pytest.raises(ValueError, match="granted call budget"):
        poll_once(state, client=None, processor=processor)

    assert state["in_flight"]["clip_key"] == snapshots[1].key
    assert state["in_flight"]["stage"] == "judging"
    assert state["in_flight"]["crops_judged"] == 1
    # The watermark is the alarm BEFORE it, which is why it cannot be the
    # thing a failure record names.
    assert state["watermark"] == snapshots[0].key


def test_activity_and_in_flight_are_never_both_set(monkeypatch):
    """One line on the screen, so one answer to "what is it doing". A reader
    that had to choose between two live fields would eventually choose wrong."""
    _serve(monkeypatch, _snapshots(3))
    state = new_state(FeedCaps(100, 100))
    both = []

    def look():
        if state.get("in_flight") is not None and state.get("activity") is not None:
            both.append((dict(state["in_flight"]), dict(state["activity"])))

    def processor(clip, budget, run_id, reporter):
        look()
        for stage in ("reading_snapshot", "finding_clip", "judging"):
            reporter.stage(stage)
            look()
        return {"status": "judged", "calls": 1}

    poll_once(state, client=None, processor=processor, checkpoint=lambda _s: look())

    assert both == []
    # And when the run is not running, neither describes it.
    state["status"] = "stopped"
    poll_once(state, client=None, processor=processor)
    assert state.get("in_flight") is None
    assert state.get("activity") is None


def test_every_moment_of_an_alarm_carries_a_stage(monkeypatch):
    """The point of the batch: no window where the worker is alive and the
    screen has nothing to say."""
    _serve(monkeypatch, _snapshots(1))
    state = new_state(FeedCaps(100, 100))
    stages = []

    def note():
        flight = state.get("in_flight")
        current = (
            flight.get("stage")
            if isinstance(flight, dict)
            else (state.get("activity") or {}).get("stage")
        )
        if not stages or stages[-1] != current:
            stages.append(current)

    def processor(clip, budget, run_id, reporter):
        note()
        for stage in ("reading_snapshot", "checking_duplicate", "finding_clip",
                      "extracting", "detecting_people"):
            reporter.stage(stage)
            note()
        reporter.stage("judging", crops_total=2, crops_judged=0)
        note()
        reporter.crop_judged()
        reporter.crop_judged()
        return {"status": "judged", "calls": 2}

    poll_once(state, client=None, processor=processor, checkpoint=lambda _s: note())

    assert None not in stages, f"an unnamed window: {stages}"
    # The first thing seen is the checkpoint that ANNOUNCES the alarm, and the
    # last is the one that RECORDS it. Everything between is the alarm itself,
    # in order, with no gap.
    assert stages == [
        "saving",
        "reading_snapshot",
        "checking_duplicate",
        "finding_clip",
        "extracting",
        "detecting_people",
        "judging",
        "saving",
    ]
    # The alarm is over, so `activity` describes the worker again -- and the
    # poll ends on the write that recorded it. `saving` is deliberately not
    # restored to `listing` here: the next thing to happen always names
    # itself, which off the end of a poll is the worker's own `waiting`.
    assert state.get("in_flight") is None
    assert state["activity"]["stage"] == "saving"


def test_progress_survives_the_status_serializer():
    """`GET /live-feed/status` returns `{**state}`, a shallow copy that SHARES
    the nested `in_flight`. Adding a key to that dict while the serializer
    walks it raises "dictionary changed size during iteration"; rebinding the
    top-level key changes no size and is safe under the GIL."""
    state = {"in_flight": {"clip_key": "k", "since": "t"}, "bulk": list(range(200))}
    reporter = live_feed.StageReporter(state)
    stop = threading.Event()
    errors = []

    def serialize():
        while not stop.is_set():
            try:
                json.dumps({**state})
            except Exception as exc:  # noqa: BLE001 - the failure under test
                errors.append(repr(exc))
                return

    def report():
        for _ in range(3000):
            reporter.crop_judged()
            reporter.stage("judging", crops_total=9)

    readers = [threading.Thread(target=serialize) for _ in range(3)]
    writers = [threading.Thread(target=report) for _ in range(4)]
    for thread in readers + writers:
        thread.start()
    for thread in writers:
        thread.join()
    stop.set()
    for thread in readers:
        thread.join()

    assert errors == []
    # And the count is exact, because the crop pool writes it from several
    # threads at once: it is a completion count, not a loop index.
    assert state["in_flight"]["crops_judged"] == 4 * 3000


# --------------------------------------------------------------------------
# THE READINGS THAT MUST STAY PINNED, NEXT TO THE ONES THAT MUST STAY FRESH.
# --------------------------------------------------------------------------


def test_a_poll_across_midnight_does_not_grant_a_second_daily_budget(monkeypatch):
    """`now` is read ONCE for the day boundary, and `remaining_calls` shares
    that same reading. A poll runs for minutes and can straddle UTC midnight,
    so a fresh reading in either place would zero `daily_calls` mid-poll and
    hand the run a whole second daily budget -- and then book the spend
    against the PREVIOUS day's counter."""
    start = datetime(2026, 4, 1, 23, 59, 50, tzinfo=UTC)
    ticks = count(1)

    class _Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            # Every reading is a few seconds later, so the poll crosses
            # midnight while it is working.
            return start + timedelta(seconds=next(ticks) * 3)

    monkeypatch.setattr(live_feed, "datetime", _Clock)
    _serve(monkeypatch, _snapshots(20))
    state = new_state(FeedCaps(1000, 10), now=start)     # daily cap of 10
    granted = []

    def processor(clip, budget, run_id, reporter):
        granted.append(budget)
        return {"status": "judged", "calls": 1}

    poll_once(state, client=None, processor=processor, now=start)

    assert state["day"] == "2026-04-01", "the day rolled over mid-poll"
    assert state["daily_calls"] == 10 and state["calls"] == 10
    # Exactly the one budget, counting down. A second grant would show as the
    # sequence restarting at 10.
    assert granted == list(range(10, 0, -1))
    assert state["paused_reason"] == "budget"
    # And the reading that must be FRESH is: this poll ran past midnight.
    assert state["updated_at"][:10] == "2026-04-02"


def test_the_next_poll_rolls_the_day_and_grants_the_new_budget(monkeypatch):
    """The other half: pinned WITHIN a poll, rolled BETWEEN polls."""
    _serve(monkeypatch, _snapshots(20))
    state = new_state(FeedCaps(1000, 10), now=datetime(2026, 4, 1, 23, 59, 50, tzinfo=UTC))
    state["daily_calls"] = 10

    poll_once(
        state,
        client=None,
        processor=lambda *a: {"status": "judged", "calls": 1},
        now=datetime(2026, 4, 2, 0, 0, 30, tzinfo=UTC),
    )

    assert state["day"] == "2026-04-02"
    assert state["daily_calls"] == 10       # a fresh day's worth, not eleven


def test_the_clock_cannot_move_a_duplicate_verdict(monkeypatch):
    """`_duplicate_of` compares the ALARM's own time on both sides -- the
    snapshot's start against the ring entry's -- so nothing about when the
    poll happened to run can change which alarms collapse. Stamping every
    alarm freshly must not have touched this."""
    snapshot = parse_snapshot_key(key(1))
    ring = [{
        "key": key(2),
        "at": snapshot.start.isoformat(),          # the same moment
        "scene_hash": "0f0f0f0f0f0f0f0f",
        "boxes": [[1.0, 2.0, 3.0, 4.0]],
    }]

    verdicts = set()
    for offset in (0, 3600, 86400, -86400):
        class _Clock(datetime):
            @classmethod
            def now(cls, tz=None, _o=offset):
                return datetime(2026, 4, 1, tzinfo=UTC) + timedelta(seconds=_o)

        monkeypatch.setattr(live_feed, "datetime", _Clock)
        verdicts.add(
            live_feed._duplicate_of(snapshot, "0f0f0f0f0f0f0f0f", [(1.0, 2.0, 3.0, 4.0)], ring)
        )

    assert verdicts == {key(2)}, "the wall clock moved a duplicate verdict"


def test_the_shadow_verdict_still_lands_on_a_judged_alarm(monkeypatch, tmp_path):
    """With `PREFILTER_SKIPS_EXTRACTION` off, the pre-filter RECORDS its
    verdict in detector mode and the alarm is judged anyway, so the
    clean-frame check can be compared against it. The progress work must not
    have disturbed that measurement.

    The switch ships ON since 4 Sep, so this pins the shadow half by
    monkeypatch rather than by reading the shipped default. The 2x2 has to be
    retakeable -- another camera, another box source, a changed window -- and
    a test that asserts which way a constant is set fails on the flip, which
    is a decision rather than a regression."""
    monkeypatch.setattr(live_feed, "PREFILTER_SKIPS_EXTRACTION", False)
    _stub_live_snapshot(monkeypatch, judged=1, stopped=0)
    monkeypatch.setattr(
        live_feed,
        "get_person_detector",
        lambda _config: type("D", (), {"find": staticmethod(lambda _b: [(1.0, 2.0, 3.0, 4.0)])})(),
    )
    monkeypatch.setattr(live_feed, "scene_hash", lambda _b: "0f0f0f0f0f0f0f0f")
    monkeypatch.setattr(live_feed.Path, "read_bytes", lambda _self: b"clean-frame-bytes")
    ring = [{
        "key": key(9),
        "at": SNAPSHOT.start.isoformat(),
        "scene_hash": "0f0f0f0f0f0f0f0f",
        "boxes": [[1.0, 2.0, 3.0, 4.0]],
    }]

    result = live_feed.process_live_snapshot(
        SNAPSHOT, 10, "run-1", live_feed.NO_REPORTER,
        client=_FakeClient(),
        box_source="detector",
        recent_annotated=ring,
        audit_log_path=str(tmp_path / "log.json"),
        alarm_image_dir=str(tmp_path / "stored"),
    )

    # Flagged by the pre-filter AND judged anyway, which is the whole shadow.
    assert result["pre_extraction_duplicate_of"] == key(9)
    assert result["status"] == "judged"
    assert live_feed.PREFILTER_SKIPS_EXTRACTION is False


def test_the_reporter_is_optional_so_the_shadow_path_is_unchanged(monkeypatch, tmp_path):
    """`process_live_snapshot` is called directly in places that have no state
    to report onto. A required reporter would have made those a TypeError."""
    _stub_live_snapshot(monkeypatch, judged=1, stopped=0)

    result = live_feed.process_live_snapshot(
        SNAPSHOT, 10, "run-1",
        client=_FakeClient(),
        audit_log_path=str(tmp_path / "log.json"),
        alarm_image_dir=str(tmp_path / "stored"),
    )

    assert result["status"] == "judged"
