from datetime import datetime, timedelta
from pathlib import Path

from ppe_pipeline.batching import ImageFrame, TemporalBatchAccumulator, batch_frames_by_gap
from ppe_pipeline.models import BoundingBox, Detection, ViolationType

T0 = datetime(2026, 7, 1, 12, 0, 0)
BOX = BoundingBox(0, 0, 10, 10)


def _frame(path: str, offset_seconds: float) -> ImageFrame:
    det = Detection(ViolationType.HELMET, person_box=BOX, source="t")
    return ImageFrame(path=Path(path), timestamp=T0 + timedelta(seconds=offset_seconds), detections=[det])


def test_frame_stamps_its_own_detections_with_frame_path():
    frame = _frame("a.jpg", 0)
    assert frame.detections[0].frame_path == Path("a.jpg")


def test_burst_within_gap_stays_in_one_batch():
    # Matches the worked example: 0.0 -> 1.2 -> 2.9 all within 2s gaps,
    # 6.0 is a 3.1s gap from 2.9 and starts a new batch.
    frames = [_frame(f"f{i}.jpg", t) for i, t in enumerate([0.0, 1.2, 2.9, 6.0])]
    batches = batch_frames_by_gap(frames, gap_seconds=2.0)

    assert len(batches) == 2
    assert [f.path.name for f in batches[0].frames] == ["f0.jpg", "f1.jpg", "f2.jpg"]
    assert [f.path.name for f in batches[1].frames] == ["f3.jpg"]


def test_batching_is_gap_based_not_anchored_to_the_first_frame():
    # Each consecutive gap is 1.9s (<=2s), so a long chain of frames stays
    # in one batch even though the first and last are 7.6s apart overall —
    # this is what distinguishes session-window from a tumbling window.
    offsets = [0.0, 1.9, 3.8, 5.7, 7.6]
    frames = [_frame(f"f{i}.jpg", t) for i, t in enumerate(offsets)]
    batches = batch_frames_by_gap(frames, gap_seconds=2.0)
    assert len(batches) == 1
    assert len(batches[0].frames) == 5


def test_out_of_order_input_is_sorted_before_batching():
    frames = [_frame("late.jpg", 5.0), _frame("early.jpg", 0.0)]
    batches = batch_frames_by_gap(frames, gap_seconds=2.0)
    assert len(batches) == 2
    assert batches[0].frames[0].path.name == "early.jpg"
    assert batches[1].frames[0].path.name == "late.jpg"


def test_all_detections_flattens_across_frames_in_a_batch():
    frames = [_frame("f0.jpg", 0.0), _frame("f1.jpg", 1.0)]
    batches = batch_frames_by_gap(frames, gap_seconds=2.0)
    assert len(batches[0].all_detections) == 2


def test_accumulator_holds_back_the_still_open_trailing_batch():
    acc = TemporalBatchAccumulator(gap_seconds=2.0)
    acc.add_frame(_frame("f0.jpg", 0.0))
    acc.add_frame(_frame("f1.jpg", 1.0))

    # Polling right after the last frame: still within the gap window, so
    # nothing is confirmed closed yet.
    ready = acc.poll_ready_batches(now=T0 + timedelta(seconds=1.5))
    assert ready == []

    # A new frame arrives close enough to join the same batch.
    acc.add_frame(_frame("f2.jpg", 2.5))
    ready = acc.poll_ready_batches(now=T0 + timedelta(seconds=2.6))
    assert ready == []

    # Enough time passes with no new frame: the batch is now confirmed closed.
    ready = acc.poll_ready_batches(now=T0 + timedelta(seconds=5.0))
    assert len(ready) == 1
    assert [f.path.name for f in ready[0].frames] == ["f0.jpg", "f1.jpg", "f2.jpg"]

    # The accumulator is empty again until new frames arrive.
    assert acc.poll_ready_batches(now=T0 + timedelta(seconds=10.0)) == []


def test_accumulator_releases_earlier_batches_while_holding_the_latest_open():
    acc = TemporalBatchAccumulator(gap_seconds=2.0)
    acc.add_frame(_frame("f0.jpg", 0.0))
    acc.add_frame(_frame("f1.jpg", 10.0))  # gap > 2s: separate, closed batch
    acc.add_frame(_frame("f2.jpg", 10.5))  # joins f1's batch, still "recent"

    ready = acc.poll_ready_batches(now=T0 + timedelta(seconds=11.0))

    assert len(ready) == 1
    assert [f.path.name for f in ready[0].frames] == ["f0.jpg"]
