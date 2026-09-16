"""Group per-frame detections from the same fixed camera into temporal
batches before merging, so near-simultaneous captures of one incident are
treated as a single event instead of judged frame by frame.

Session-window (gap-based) batching: frames are sorted by timestamp, and a
frame joins the current batch as long as the gap to the *previous* frame
already in that batch is within `gap_seconds`. A gap larger than that
closes the batch and starts the next one — a burst of captures can grow a
batch indefinitely as long as no single gap between consecutive frames
exceeds the threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .config import PipelineConfig, get_config
from .models import Detection


@dataclass
class ImageFrame:
    """One captured image, its timestamp, and its raw per-frame detections
    (from a single fixed camera, before any cross-frame merge).

    Detections that don't already carry a `frame_path` are stamped with
    this frame's path — callers building mock/real detections don't need
    to set it themselves.
    """

    path: Path
    timestamp: datetime
    detections: list[Detection]

    def __post_init__(self) -> None:
        for d in self.detections:
            if d.frame_path is None:
                d.frame_path = self.path


@dataclass
class FrameBatch:
    frames: list[ImageFrame]

    @property
    def start_time(self) -> datetime:
        return self.frames[0].timestamp

    @property
    def end_time(self) -> datetime:
        return self.frames[-1].timestamp

    @property
    def all_detections(self) -> list[Detection]:
        return [d for frame in self.frames for d in frame.detections]


def batch_frames_by_gap(
    frames: list[ImageFrame],
    gap_seconds: float | None = None,
    config: PipelineConfig | None = None,
) -> list[FrameBatch]:
    """Sort `frames` by timestamp and chain them into session-window batches.

    `gap_seconds` overrides `config.batching.gap_seconds` (config.toml /
    built-in default — see config.py).
    """
    config = config or get_config()
    threshold = gap_seconds if gap_seconds is not None else config.batching.gap_seconds

    ordered = sorted(frames, key=lambda f: f.timestamp)
    batches: list[list[ImageFrame]] = []

    for frame in ordered:
        if batches and (frame.timestamp - batches[-1][-1].timestamp).total_seconds() <= threshold:
            batches[-1].append(frame)
        else:
            batches.append([frame])

    return [FrameBatch(frames=b) for b in batches]


class TemporalBatchAccumulator:
    """Incremental wrapper for a periodic job: feed frames as they arrive,
    poll for the batches that are safe to process.

    A batch is only "ready" once the gap between its last frame and `now`
    already exceeds the threshold — meaning a later frame could no longer
    join it. The still-open trailing batch is held back and re-checked on
    the next poll, matching "images within the window merge, the rest wait
    for the next processing pass".
    """

    def __init__(self, gap_seconds: float | None = None, config: PipelineConfig | None = None):
        config = config or get_config()
        self._gap_seconds = gap_seconds if gap_seconds is not None else config.batching.gap_seconds
        self._pending: list[ImageFrame] = []

    def add_frame(self, frame: ImageFrame) -> None:
        self._pending.append(frame)

    def poll_ready_batches(self, now: datetime) -> list[FrameBatch]:
        if not self._pending:
            return []

        batches = batch_frames_by_gap(self._pending, gap_seconds=self._gap_seconds)

        ready: list[FrameBatch] = []
        still_open: list[FrameBatch] = []
        for batch in batches:
            if (now - batch.end_time).total_seconds() > self._gap_seconds:
                ready.append(batch)
            else:
                still_open.append(batch)

        self._pending = [frame for batch in still_open for frame in batch.frames]
        return ready
