"""Tests for recovering the vendor CV's boxes from an annotated snapshot.

The sample under test is a real snapshot committed at `test_bbox/`.

The synthetic cases below are the failure modes measured over 48 real
snapshots on 2 Sep, where a colour-and-hollowness rule produced 128 boxes that
were not annotations: rust panels on a ship's hull, and a rectangle carved out
of the `violation` label between its white letters. Each is drawn here rather
than committed as a fixture so the reason it is rejected stays readable.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from ppe_pipeline.helmet_cv.snapshot_boxes import (
    MAX_INTERIOR_FILL,
    MIN_EDGE_COVERAGE,
    SCENE_HASH_DISTANCE,
    same_boxes,
    scene_distance,
    scene_hash,
    clip_key_for,
    extract_boxes,
    iter_snapshots,
    parse_snapshot_key,
)

SAMPLE = Path("test_bbox/helmet_violation_2026-04-29_07-55-13_a8956350-f448-40b3-9549-7c5fdd861c25 (1).jpg")


@pytest.mark.skipif(not SAMPLE.is_file(), reason="the sample snapshot is not checked out")
def test_the_vendors_boxes_are_recovered_from_a_real_snapshot():
    """Measured against the same alarm's clean frame: our COCO detector finds
    ONE person there, the vendor marked seven. That gap is why this exists."""
    boxes = extract_boxes(SAMPLE.read_bytes())

    assert len(boxes) == 6
    for x1, y1, x2, y2 in boxes:
        assert x2 > x1 and y2 > y1
        assert 20 <= x2 - x1 <= 400
        assert 20 <= y2 - y1 <= 400
    # Reading order, so `person_index` is stable across runs rather than
    # following whatever order components were labelled in.
    tops = [round(b[1] / 50) for b in boxes]
    assert tops == sorted(tops)


@pytest.mark.skipif(not SAMPLE.is_file(), reason="the sample snapshot is not checked out")
def test_the_scenes_own_red_is_not_mistaken_for_an_annotation():
    """This frame is full of red -- painted pipe ends across its middle,
    cones, machinery. None of it was DRAWN as a rectangle."""
    boxes = extract_boxes(SAMPLE.read_bytes())

    # The pipe-end band runs across y 430-520 for most of the frame's width.
    # Nothing there is a person box.
    assert not [b for b in boxes if 430 < b[1] < 520 and b[0] > 900]
    assert MIN_EDGE_COVERAGE >= 0.7
    assert MAX_INTERIOR_FILL <= 0.3


def test_unreadable_bytes_yield_no_boxes_rather_than_raising():
    """The caller falls back to detection; a bad image must not end a pass."""
    assert extract_boxes(b"not an image") == []
    assert extract_boxes(b"") == []


def test_a_snapshot_key_is_parsed_into_its_alarm():
    ref = parse_snapshot_key(
        "snapshot/helmet_violation_2026-04-27_07-02-09_726f88cb-a3d0-4907-9d8d-111055ebc676.jpg"
    )

    assert ref is not None
    assert ref.violation_type == "helmet_violation"
    assert ref.alarm_id == "726f88cb-a3d0-4907-9d8d-111055ebc676"
    assert ref.start == datetime(2026, 4, 27, 7, 2, 9)
    # A clip carries BOTH timestamps and a snapshot only the start, so the
    # clip cannot be named outright -- this prefix finds it.
    assert ref.clip_prefix == "raw_video/helmet_violation_2026-04-27_07-02-09"


@pytest.mark.parametrize(
    "key",
    [
        "snapshot/notes.txt",
        "snapshot/helmet_violation_2026-04-27_07-02-09.jpg",
        "snapshot/helmet_violation_not-a-date_07-02-09_726f88cb-a3d0-4907-9d8d-111055ebc676.jpg",
    ],
)
def test_a_key_that_is_not_ours_is_skipped(key):
    assert parse_snapshot_key(key) is None


class _FakePaginator:
    def __init__(self, keys):
        self.keys = keys
        self.calls = []

    def paginate(self, **kwargs):
        self.calls.append(kwargs)
        start = kwargs.get("StartAfter")
        prefix = kwargs.get("Prefix", "")
        yield {
            "Contents": [
                {"Key": key}
                for key in self.keys
                if key.startswith(prefix) and (start is None or key > start)
            ]
        }


class _FakeS3:
    def __init__(self, keys):
        self.paginator = _FakePaginator(keys)

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return self.paginator


def _snapshot(day: int) -> str:
    return f"snapshot/helmet_violation_2026-04-{day:02d}_07-02-09_726f88cb-a3d0-4907-9d8d-111055ebc676.jpg"


def test_snapshots_are_walked_in_key_order_and_resume_from_the_cursor():
    client = _FakeS3([_snapshot(1), _snapshot(2), _snapshot(3)])

    seen = [s.key for s in iter_snapshots(client, bucket="b", start_after_key=_snapshot(1))]

    assert seen == [_snapshot(2), _snapshot(3)]
    # Server-side, not a scan: the prefix filters by violation type and
    # StartAfter skips everything before the cursor.
    assert client.paginator.calls[-1]["Prefix"] == "snapshot/helmet_violation_"
    assert client.paginator.calls[-1]["StartAfter"] == _snapshot(1)


def _snapshot_at(day: int, time: str = "07-02-09", uuid: str = "726f88cb-a3d0-4907-9d8d-111055ebc676") -> str:
    return f"snapshot/helmet_violation_2026-04-{day:02d}_{time}_{uuid}.jpg"


def test_a_start_time_begins_the_listing_server_side_rather_than_scanning_to_it():
    """OPEN_TASKS #39. The time used to be a client-side filter alone, so a run
    started "from now" -- which has no watermark -- paged the whole prefix and
    discarded every key: 75,242 keys, 76 pages, 11.75s, on every poll."""
    client = _FakeS3([_snapshot_at(day) for day in (1, 2, 3)])
    examined = []

    seen = [
        s.key
        for s in iter_snapshots(
            client,
            bucket="b",
            start_on_or_after=datetime(2026, 4, 3),
            on_key=lambda: examined.append(1),
        )
    ]

    assert seen == [_snapshot_at(3)]
    # The two older keys were never sent, let alone looked at.
    assert len(examined) == 1
    assert client.paginator.calls[-1]["StartAfter"] == "snapshot/helmet_violation_2026-04-03_00-00-00"


def test_the_boundary_second_is_kept_and_a_part_second_is_truncated_downwards():
    """`StartAfter` is exclusive and the derived key carries no uuid, so every
    real key at the boundary second sorts after it. Rounding the second UP
    instead would step over an alarm; truncating costs a few examined keys
    that the filter then drops."""
    key = _snapshot_at(2, "07-02-09")

    kept = _FakeS3([key])
    assert [s.key for s in iter_snapshots(kept, bucket="b", start_on_or_after=datetime(2026, 4, 2, 7, 2, 9))] == [key]

    part = _FakeS3([key])
    examined = []
    inside_the_second = datetime(2026, 4, 2, 7, 2, 9, 500_000)
    assert [
        s.key
        for s in iter_snapshots(
            part, bucket="b", start_on_or_after=inside_the_second, on_key=lambda: examined.append(1)
        )
    ] == []
    assert len(examined) == 1, "the key must reach the client-side filter to be dropped by it"


def test_a_watermark_and_a_start_time_begin_at_the_later_of_the_two():
    """Both are lower bounds and neither may be dropped: under-scanning loses
    an alarm nobody judged, over-scanning is only slow."""
    keys = [_snapshot_at(day) for day in (1, 2, 3, 4)]

    time_wins = _FakeS3(keys)
    assert [
        s.key
        for s in iter_snapshots(
            time_wins,
            bucket="b",
            start_after_key=_snapshot_at(1),
            start_on_or_after=datetime(2026, 4, 3),
        )
    ] == [_snapshot_at(3), _snapshot_at(4)]
    assert time_wins.paginator.calls[-1]["StartAfter"] == "snapshot/helmet_violation_2026-04-03_00-00-00"

    watermark_wins = _FakeS3(keys)
    assert [
        s.key
        for s in iter_snapshots(
            watermark_wins,
            bucket="b",
            start_after_key=_snapshot_at(3),
            start_on_or_after=datetime(2026, 4, 1),
        )
    ] == [_snapshot_at(4)]
    assert watermark_wins.paginator.calls[-1]["StartAfter"] == _snapshot_at(3)


def test_the_derived_cursor_yields_exactly_what_the_client_side_filter_alone_did():
    """The check that matters: a run must not skip an alarm it has not judged.
    Compared against the old behaviour -- no derived cursor, the time applied
    key by key -- over every combination of the two arguments."""
    keys = [
        _snapshot_at(day, time)
        for day in (1, 2, 3)
        for time in ("07-02-09", "07-02-10", "23-59-59")
    ]
    unparsable = "snapshot/helmet_violation_notes.txt"

    for watermark in [None] + keys:
        for start in [
            None,
            datetime(2026, 4, 1),
            datetime(2026, 4, 2, 7, 2, 10),
            datetime(2026, 4, 3, 23, 59, 59),
            datetime(2026, 4, 9),
        ]:
            client = _FakeS3(sorted(keys + [unparsable]))
            expected = [
                key
                for key in sorted(keys)
                if (watermark is None or key > watermark)
                and (start is None or parse_snapshot_key(key).start >= start)
            ]

            got = [
                s.key
                for s in iter_snapshots(
                    client, bucket="b", start_after_key=watermark, start_on_or_after=start
                )
            ]

            assert got == expected, (watermark, start)


def test_the_clip_is_found_by_prefix_and_checked_by_uuid():
    """Two alarms sharing a start second would otherwise pair a snapshot with
    the wrong clip's video, silently."""
    right = ("raw_video/helmet_violation_2026-04-27_07-02-09_to_07-03-01_"
             "726f88cb-a3d0-4907-9d8d-111055ebc676.mp4")
    wrong = ("raw_video/helmet_violation_2026-04-27_07-02-09_to_07-03-01_"
             "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee.mp4")
    ref = parse_snapshot_key(
        "snapshot/helmet_violation_2026-04-27_07-02-09_726f88cb-a3d0-4907-9d8d-111055ebc676.jpg"
    )

    assert clip_key_for(_FakeS3([wrong, right]), ref, bucket="b") == right
    assert clip_key_for(_FakeS3([wrong]), ref, bucket="b") is None


# --- the shapes that are NOT annotations ---------------------------------
#
# Every case here was produced by the colour-only rule on real snapshots.

ANNOTATION_RED = (40, 37, 204)  # BGR, measured on a real snapshot


def _canvas(width: int = 400, height: int = 400):
    """A grey frame to draw on, encoded the way a snapshot arrives."""
    import numpy as np

    return np.full((height, width, 3), 90, dtype=np.uint8)


def _encoded(image) -> bytes:
    import cv2

    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 100])
    assert ok
    return buffer.tobytes()


def test_a_drawn_rectangle_is_recovered():
    cv2 = pytest.importorskip("cv2")
    image = _canvas()
    cv2.rectangle(image, (100, 120), (160, 260), ANNOTATION_RED, 3)

    boxes = extract_boxes(_encoded(image))

    assert len(boxes) == 1
    x1, y1, x2, y2 = boxes[0]
    assert abs(x1 - 100) <= 4 and abs(x2 - 160) <= 4
    assert abs(y1 - 120) <= 4 and abs(y2 - 260) <= 4


def test_a_red_hose_lying_diagonally_is_not_a_box():
    """A diagonal line fills little of its own bounding box, so hollowness
    alone called it an annotation. Its sides are not drawn."""
    cv2 = pytest.importorskip("cv2")
    image = _canvas()
    cv2.line(image, (60, 60), (300, 320), ANNOTATION_RED, 5)

    assert extract_boxes(_encoded(image)) == []


def test_a_solid_red_object_is_not_a_box():
    cv2 = pytest.importorskip("cv2")
    image = _canvas()
    cv2.rectangle(image, (100, 100), (200, 240), ANNOTATION_RED, -1)

    assert extract_boxes(_encoded(image)) == []


def test_a_small_red_patch_with_a_red_middle_is_not_a_box():
    """The rust-panel case. Small, and its middle is red rather than a person
    -- measured, the seven false boxes with red middles all had a side of 34
    or less while the three real ones were 65x107, 84x63 and 66x143."""
    cv2 = pytest.importorskip("cv2")
    image = _canvas()
    cv2.rectangle(image, (100, 100), (128, 130), ANNOTATION_RED, 2)
    cv2.rectangle(image, (106, 106), (122, 124), (60, 55, 190), -1)

    assert extract_boxes(_encoded(image)) == []


def test_a_worker_against_a_red_hull_is_still_a_box():
    """The other half of that rule. A big box may be red inside because the
    person is standing against rust, and it is still a real annotation."""
    cv2 = pytest.importorskip("cv2")
    image = _canvas()
    cv2.rectangle(image, (100, 100), (190, 250), ANNOTATION_RED, 3)
    cv2.rectangle(image, (110, 110), (180, 180), (60, 55, 190), -1)

    assert len(extract_boxes(_encoded(image))) == 1


def test_a_label_as_wide_as_its_box_does_not_lose_the_box():
    """A label no wider than its box leaves both vertical edges running the
    same distance, so the rows they share include the label. Measured on a
    real snapshot the label ends at y=145 and its box starts at y=148, and
    that 3px gap is inside the tolerance that treats collinear runs as one
    edge -- so the label has to be excluded by its THICKNESS.

    The cost is a crop ~20px taller than the person it is meant to frame, on
    footage where a worker is often 80px tall. The label itself never reaches
    the judge: crops are taken from the clean frame, not the snapshot."""
    cv2 = pytest.importorskip("cv2")
    image = _canvas()
    cv2.rectangle(image, (100, 146), (160, 280), ANNOTATION_RED, 3)
    cv2.rectangle(image, (100, 120), (160, 140), ANNOTATION_RED, -1)

    boxes = extract_boxes(_encoded(image))

    assert len(boxes) == 1
    assert boxes[0][1] >= 141  # the box, not the label above it


def test_two_overlapping_annotations_split_into_two_boxes():
    """The colour rule returned their union as one box, cropping two people
    and the gap between them into a single image."""
    cv2 = pytest.importorskip("cv2")
    image = _canvas()
    cv2.rectangle(image, (100, 100), (170, 250), ANNOTATION_RED, 3)
    cv2.rectangle(image, (150, 130), (220, 280), ANNOTATION_RED, 3)

    boxes = extract_boxes(_encoded(image))

    assert len(boxes) == 2
    assert [round(b[0]) for b in boxes] == sorted(round(b[0]) for b in boxes)


# --- the same alarm, twice -----------------------------------------------
#
# Two different repetitions, and they need different tests.
#
# MEASURED over all 31,281 helmet snapshots: 24,463 distinct images, 3,120
# alarmed more than once, 6,818 redundant alarms (21.8%). One CAM01 frame was
# uploaded 38 times across 3.5 days. Only 139 of those groups fall within a
# minute, so most are a stale frame re-served, not burst firing.


def test_the_same_view_hashes_the_same_when_the_bytes_differ():
    """The case an equality test cannot see. Measured on 40 close pairs, only
    2 in 24 were byte-identical while 10 in 24 were the same scene."""
    cv2 = pytest.importorskip("cv2")
    image = _canvas(400, 400)
    cv2.rectangle(image, (100, 120), (160, 260), ANNOTATION_RED, 3)
    first = _encoded(image)
    # Re-encoded at a different quality: same view, different bytes.
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 70])
    assert ok
    second = buffer.tobytes()

    assert first != second
    assert scene_distance(scene_hash(first), scene_hash(second)) <= SCENE_HASH_DISTANCE


def test_a_different_view_does_not():
    cv2 = pytest.importorskip("cv2")
    left = _canvas(400, 400)
    cv2.rectangle(left, (20, 20), (120, 300), (200, 40, 40), -1)
    right = _canvas(400, 400)
    cv2.circle(right, (300, 300), 90, (30, 220, 30), -1)

    assert scene_distance(scene_hash(_encoded(left)), scene_hash(_encoded(right))) > SCENE_HASH_DISTANCE


def test_boxes_that_moved_are_not_the_same_finding():
    """THE SAFETY NET. Of 17 same-scene pairs measured, 13 had boxes that had
    moved -- counts going 2->1, 1->5, 0->2 -- which is a different set of
    people flagged. Treating those as duplicates discards real observations."""
    assert same_boxes([(0, 0, 50, 100)], [(0, 0, 50, 100)]) is True
    # One person left the frame.
    assert same_boxes([(0, 0, 50, 100), (200, 0, 250, 100)], [(0, 0, 50, 100)]) is False
    # The same count, in a different place.
    assert same_boxes([(0, 0, 50, 100)], [(300, 200, 350, 300)]) is False
    # Nothing marked on either side is not a match either: it says nothing.
    assert same_boxes([], []) is False


def test_a_missing_hash_never_matches():
    """OpenCV absent, or an unreadable image. A duplicate check that cannot
    see must not claim a duplicate."""
    assert scene_distance("", "abcd") == 64
    assert scene_hash(b"") == ""
    assert scene_hash(b"not an image") == ""
