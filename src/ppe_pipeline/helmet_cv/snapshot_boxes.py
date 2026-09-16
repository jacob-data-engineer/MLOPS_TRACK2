"""Recover the vendor CV's own person boxes from an annotated snapshot.

WHY THIS EXISTS

`snapshot/` holds one annotated still per alarm: the vendor's CV draws a red
rectangle around every person it flagged, with a filled `violation` banner
above each box. Those are the boxes that RAISED the alarm, so recovering them
means judging exactly what production flagged -- and it does not depend on our
own detector finding the same people.

It does not, and that is the point. Measured 2 Sep on alarm
`a8956350-f448-40b3-9549-7c5fdd861c25` (Ch1 J1, an overhead view of a ship's
hold): the snapshot marks SEVEN people, and our COCO detector finds ONE on the
clean frame of the same moment. The one it finds matches an annotation box
exactly; the other six are 60-100px workers seen from above, which that model
does not pick up. On yesterday's fisheye camera COCO did fine. Detector
quality swings by camera, and the vendor's boxes do not.

THE SNAPSHOT IS NEVER JUDGED. It is read for coordinates only; the crops go to
the judge from the CLEAN frame in the matching clip. Judging a picture with
`violation` written on it would hand the model the answer -- see
`HANDOFF_helmet_cv.md` §3.

HOW THE BOXES ARE FOUND

By reconstructing the RECTANGLE, not by trusting the colour. Colour alone was
the first implementation and it was wrong often enough to matter: measured
2 Sep over 48 snapshots spanning 27 Apr to 2 Sep, "red blob that is mostly
hollow" shipped 128 boxes that were not annotations at all, on 32 of the 48
images -- and on 7 images EVERY box it produced was junk. Rust panels on a
ship's hull are the same red as the annotation and are hollow enough to pass.

So each red region is asked whether an actual rectangle was drawn in it:

  1. Erode with a long thin kernel to find straight runs -- the drawn edges.
  2. Pair vertical runs and take the rows they SHARE (never their union: the
     `violation` label extends one edge upward past the box, and an occluding
     box cuts the other short).
  3. Confirm all four sides are really drawn and the middle is really empty.

This is the line-reconstruction idea from `test_bbox/bbox-extraction-pipeline.md`,
with morphological runs in place of Hough transforms and angle binning -- the
edges here are axis-aligned, so the angle machinery buys nothing.

Overlapping annotations SPLIT correctly, which the colour version could not
do: it returned their union as one box. The CEO was content to leave those
merged (2 Sep: "you can leave the merged boxes as is and vlm get judge it as a
whole no problem"), so splitting them is a bonus rather than the goal, and it
gives the judge one person per crop instead of a pair plus the gap between.

Measured on the same 48 snapshots: 85 rectangles where the colour version
found 60 real ones -- 59 of those 60 kept, 26 it never found (split merges,
and boxes its hollowness test rejected), and none of its 128 false ones.

WHEN NO RECTANGLE IS FOUND the frame settles as NO VIOLATION with no judge
call at all, and reads on the feed's results as "0 of 1 judged". It is not a
fallback to judging the whole frame -- that was tried and removed, because
asking "is this person wearing a helmet" about a 1920x1080 picture with no
identified person answered `helmet_worn` 6 times out of 7.

That is worth knowing before trusting the number: it converts OUR failure to
read an annotation into a cleared alarm. One snapshot in the 48 measured is
in this state, a box running off two edges of the frame where only two sides
were ever drawn. The row is still written, so the frame appears in the audit
with its own reason code (`no_person_detected`) and a human can overturn it.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Callable

logger = logging.getLogger(__name__)

SNAPSHOT_PREFIX = "snapshot/"

#: The annotation's red, as BGR bounds. Measured on a real snapshot: the
#: drawn pixels sit around (40, 37, 204), while the scene's own reds -- cones,
#: painted pipe ends, machinery -- are duller and less saturated. Only 1.4% of
#: the pixels in a band packed with red pipe ends fall inside this.
RED_LOWER = (0, 0, 150)
RED_UPPER = (90, 90, 255)

#: Smallest annotation worth trusting, in pixels. Below this a rectangle is
#: label text, a glyph, or noise -- not a person's box.
MIN_BOX_SIDE = 20

#: Shortest straight run counted as a drawn edge. Every real box is taller
#: than this, and it is short enough that an edge broken by an overlapping
#: box still leaves a usable piece on each side of the break.
EDGE_RUN = 15

#: How far from a side to look for that side's line, in pixels. The drawn
#: line measured 3px wide; 4 leaves a pixel of slack for the JPEG edge and
#: for the two-pixel drift a merged run's centre can pick up.
EDGE_BAND = 4

#: Collinear runs within this many pixels are the same edge. It is the line
#: width on purpose: wider, and a box's right edge fuses with the `violation`
#: label's right edge a few pixels away, which drags the reported edge off
#: the actual line and fails the box that owns it.
MERGE_TOLERANCE = 3

#: How much of each side must actually be red for a rectangle to count as
#: drawn. Real boxes sit at 0.94-1.00 even with a label over one corner; a
#: rectangle assembled from unrelated lines does not come close.
MIN_EDGE_COVERAGE = 0.70

#: Above this the middle is filled in, so the shape is a solid red object and
#: not an outline. Generous on purpose: a worker standing against a rust-red
#: hull measured 0.17-0.23 inside a perfectly real box.
MAX_INTERIOR_FILL = 0.30

#: A SMALL box with a red middle is a red artifact -- a rust patch, or a
#: rectangle carved out of the `violation` label between its white letters.
#: Neither half of this rule works alone and both were measured: of the ten
#: recovered boxes with interiors above 0.15, the three real ones were
#: 65x107, 84x63 and 66x143, and all seven false ones had a side of 34 or
#: less. People standing against red are big; red pretending to be a box is
#: small.
SUSPECT_MIN_SIDE = 35
SUSPECT_INTERIOR_FILL = 0.15

#: Thicker than this, a band of red is a filled label rather than a drawn
#: line. The `violation` banner sits directly above its own box, and a
#: rectangle that starts at the banner's top instead of the box's is ~20px
#: too tall -- a looser crop around the person, on a corpus where a worker is
#: often only 80px to begin with. No answer leaks either way: the judge sees
#: the CLEAN frame, and the label exists only on the snapshot.
THICK_BAND = 8

#: Two rectangles overlapping by more than this are the same annotation found
#: from two pairings; the larger is kept.
DUPLICATE_IOU = 0.6

#: A rectangle this far inside a larger one is not an annotation: it is the
#: overlap of two boxes, bounded by one box's right edge, the other's left,
#: and one horizontal edge from each. Real annotations do not nest.
CONTAINED = 0.9

#: `snapshot/helmet_violation_2026-04-27_07-02-09_<uuid>.jpg`
_SNAPSHOT_NAME = re.compile(
    r"^(?P<type>[a-z_]+)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}-\d{2}-\d{2})_(?P<uuid>[0-9a-fA-F-]{36})\.jpg$"
)


@dataclass(frozen=True)
class SnapshotRef:
    """One annotated still, identified by its key alone."""

    key: str
    violation_type: str
    alarm_id: str
    start: datetime
    #: THE IMAGE'S OWN IDENTITY, from the listing rather than the bytes.
    #:
    #: S3 returns an ETag per object, and for a single-part upload that is the
    #: md5 of the content. Every snapshot here is 0.5-0.9MB, well under the
    #: multipart threshold, and two pairs were confirmed by hashing the
    #: downloaded bytes.
    #:
    #: It is here because the vendor re-alarms on the SAME IMAGE. Measured
    #: over the 31,281 helmet snapshots the prefix held at the time: 24,463
    #: distinct images, 3,120 of them alarmed more than once, 6,818 redundant
    #: alarms -- 21.8% of the corpus. One CAM01 frame was uploaded 38 times
    #: across 3.5 days. Only 139 of those groups fall within a minute, so this
    #: is not burst firing; it is a stale frame re-served with a fresh alarm
    #: id. THE PREFIX NOW HOLDS 75,242 KEYS and the ratio has not been
    #: re-measured against it.
    #:
    #: Available WITHOUT downloading, which is the point: a re-alarm is
    #: skipped before its 0.9MB is fetched and before a judge call is spent.
    etag: str = ""

    @property
    def clip_prefix(self) -> str:
        """The `raw_video/` prefix that finds this alarm's clip.

        A clip key carries BOTH timestamps (`..._07-02-09_to_07-03-01_<uuid>`)
        while a snapshot carries only the start, so the clip cannot be named
        outright -- but the start stamp is unique enough to list on, and was
        measured to return exactly one object.
        """
        stamp = self.start.strftime("%Y-%m-%d_%H-%M-%S")
        return f"raw_video/{self.violation_type}_{stamp}"


def parse_snapshot_key(key: str) -> SnapshotRef | None:
    """A `SnapshotRef` from a key, or `None` if the name is not one of ours."""
    name = key.rsplit("/", 1)[-1]
    match = _SNAPSHOT_NAME.match(name)
    if match is None:
        return None
    try:
        start = datetime.strptime(
            f"{match.group('date')}_{match.group('time')}", "%Y-%m-%d_%H-%M-%S"
        )
    except ValueError:
        return None
    return SnapshotRef(
        key=key,
        violation_type=match.group("type"),
        alarm_id=match.group("uuid"),
        start=start,
    )


def start_after_key_for(start: datetime, *, violation_type: str = "helmet_violation") -> str:
    """A synthetic key sorting just BEFORE every snapshot at or after `start`,
    for `StartAfter` to begin a listing at.

    Snapshot names are fixed and zero-padded
    (`snapshot/<type>_<YYYY-MM-DD>_<HH-MM-SS>_<uuid>.jpg`), so comparing them
    as strings compares them as times. This is the name up to and including
    the second, with no uuid: every real key at that second is this string
    plus `_<uuid>.jpg`, so it sorts strictly after, and `StartAfter` -- which
    is exclusive -- keeps the boundary second rather than dropping it. That
    matches `iter_snapshots`'s own filter, which is `>=` too.

    The second is TRUNCATED, never rounded up. A start time inside a second
    lands the cursor at the top of that second and the client-side filter
    drops what is left, which costs a few keys; rounding up would step over
    an alarm.

    NAIVE, like the stamps it is compared against: `parse_snapshot_key` reads
    a key's time without a zone, and the filter in `iter_snapshots` drops the
    caller's zone the same way. The two must agree, or the derived cursor
    would skip what the filter would have kept.
    """
    return (
        f"{SNAPSHOT_PREFIX}{violation_type}_"
        f"{start.replace(tzinfo=None).strftime('%Y-%m-%d_%H-%M-%S')}"
    )


def iter_snapshots(
    client,
    *,
    bucket: str,
    violation_type: str = "helmet_violation",
    start_after_key: str | None = None,
    start_on_or_after: datetime | None = None,
    on_key: Callable[[], None] | None = None,
):
    """Annotated stills in key order, which is chronological.

    Same shape as `clips.iter_clips` and for the same reasons: the prefix
    filters by violation type server-side, and `StartAfter` resumes without
    paginating everything before the cursor.

    BOTH ARGUMENTS ARE LOWER BOUNDS, and the listing begins at the LATER of
    them. `start_after_key` is a resume point: everything at or before it has
    been dealt with. `start_on_or_after` is a time, and because keys sort
    chronologically it names a key too (`start_after_key_for`). Beginning at
    the later of the two skips only keys that would have been discarded
    anyway -- at or before the watermark, or earlier than the start time --
    so a run yields exactly what it yielded when the time was a client-side
    filter alone. What changes is that a run started "from now", which has no
    watermark, no longer pages the WHOLE prefix in order to discard it:
    measured at 75,242 keys, 76 pages and 11.75s on every poll
    (OPEN_TASKS #39).

    THE CLIENT-SIDE TIME FILTER BELOW STAYS, as the backstop the derivation
    needs: a key that does not parse, or one that does not sort the way the
    derivation assumes, is answered by reading its own stamp rather than by
    where it sat in the listing.

    `on_key` IS CALLED FOR EVERY KEY LOOKED AT, not for every one yielded. A
    resumed run still walks everything from its cursor forward, and a caller
    counting what it received rather than what it examined would report zero
    for that window and describe the longest silence in the poll as no work
    at all.
    """
    paginator = client.get_paginator("list_objects_v2")
    kwargs = {"Bucket": bucket, "Prefix": f"{SNAPSHOT_PREFIX}{violation_type}_"}
    cursors = [
        cursor
        for cursor in (
            start_after_key,
            None
            if start_on_or_after is None
            else start_after_key_for(start_on_or_after, violation_type=violation_type),
        )
        if cursor
    ]
    if cursors:
        kwargs["StartAfter"] = max(cursors)
    for page in paginator.paginate(**kwargs):
        for obj in page.get("Contents", ()):
            if on_key is not None:
                on_key()
            snapshot = parse_snapshot_key(obj["Key"])
            if snapshot is None:
                continue
            if start_on_or_after is not None and snapshot.start < start_on_or_after.replace(tzinfo=None):
                continue
            yield replace(snapshot, etag=str(obj.get("ETag") or "").strip('"'))


def clip_key_for(client, snapshot: SnapshotRef, *, bucket: str) -> str | None:
    """The clip holding this alarm's CLEAN frames, or `None` if it is gone.

    One prefix listing, not a scan: measured to return exactly one object.
    The uuid is checked anyway -- two alarms sharing a start second would
    otherwise silently pair a snapshot with the wrong clip's video.
    """
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=snapshot.clip_prefix):
        for obj in page.get("Contents", ()):
            if snapshot.alarm_id in obj["Key"]:
                return obj["Key"]
    return None


def _runs(mask, *, vertical: bool) -> list[tuple[int, int, int]]:
    """Straight runs of at least `EDGE_RUN` pixels, as (position, start, end).

    Eroding with a long thin kernel leaves only pixels that had a full run
    through them, which is exactly the set of drawn edges. It also SHORTENS
    every run by the kernel, so the pad puts the ends back where the ink is.
    """
    import cv2
    import numpy as np

    kernel = np.ones((EDGE_RUN, 1), np.uint8) if vertical else np.ones((1, EDGE_RUN), np.uint8)
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        cv2.erode(mask, kernel), connectivity=8
    )
    pad = EDGE_RUN // 2
    runs = []
    for index in range(1, count):
        x, y, width, height = (int(v) for v in stats[index][:4])
        if vertical:
            runs.append((x + width // 2, y - pad, y + height + pad))
        else:
            runs.append((y + height // 2, x - pad, x + width + pad))
    return runs


def _merge_collinear(runs: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    """One edge per line. Occlusion and JPEG noise break a single edge up."""
    merged: list[tuple[int, int, int]] = []
    for position, start, end in sorted(runs):
        for index, (at, lo, hi) in enumerate(merged):
            near = abs(at - position) <= MERGE_TOLERANCE
            touching = start <= hi + MERGE_TOLERANCE and end >= lo - MERGE_TOLERANCE
            if near and touching:
                merged[index] = ((at + position) // 2, min(lo, start), max(hi, end))
                break
        else:
            merged.append((position, start, end))
    return merged


def _is_drawn_rectangle(mask, x1: int, y1: int, x2: int, y2: int) -> bool:
    """All four sides drawn, middle empty, and not a small patch of red."""
    inside = mask[y1 : y2 + 1, x1 : x2 + 1] > 0
    height, width = inside.shape
    if height < MIN_BOX_SIDE or width < MIN_BOX_SIDE:
        return False
    band = min(EDGE_BAND, height // 3, width // 3)
    sides = (
        inside[:band, :].any(axis=0).mean(),
        inside[height - band :, :].any(axis=0).mean(),
        inside[:, :band].any(axis=1).mean(),
        inside[:, width - band :].any(axis=1).mean(),
    )
    if min(sides) < MIN_EDGE_COVERAGE:
        return False
    middle = inside[band * 2 : height - band * 2, band * 2 : width - band * 2]
    if middle.size == 0:
        return True
    filled = float(middle.mean())
    if filled > MAX_INTERIOR_FILL:
        return False
    return not (min(width, height) < SUSPECT_MIN_SIDE and filled >= SUSPECT_INTERIOR_FILL)


def _horizontal_edges(mask, x1: int, x2: int, top: int, bottom: int) -> list[tuple[int, int]]:
    """Row bands where red spans most of x1..x2 -- the drawn horizontal edges."""
    coverage = (mask[top : bottom + 1, x1 : x2 + 1] > 0).mean(axis=1)
    bands: list[tuple[int, int]] = []
    run: tuple[int, int] | None = None
    for offset, value in enumerate(coverage):
        if value >= MIN_EDGE_COVERAGE:
            run = (offset, offset) if run is None else (run[0], offset)
        elif run is not None:
            bands.append((top + run[0], top + run[1]))
            run = None
    if run is not None:
        bands.append((top + run[0], top + run[1]))
    return bands


def _fit(mask, x1: int, x2: int, top: int, bottom: int) -> tuple[int, int, int, int] | None:
    """The largest drawn rectangle these two vertical edges bound.

    Both ends of every band are candidates, not just the outermost rows. A
    `violation` label as wide as its own box merges with the box's top edge
    into one thick band, and the box's real top edge is that band's LAST row
    -- taking the first would enclose the solid label and fail the empty
    middle test, losing a box that is perfectly real.
    """
    bands = _horizontal_edges(mask, x1, x2, top, bottom)
    if len(bands) < 2:
        return None
    tops, bottoms = [], []
    for first, last in bands:
        if last - first > THICK_BAND:
            # A filled label. The box's own edge is where the label ENDS.
            tops.append(last)
            bottoms.append(first)
        else:
            tops.append(first)
            bottoms.append(last)
    pairs = [(a, b) for a in tops for b in bottoms if b - a >= MIN_BOX_SIDE]
    for y1, y2 in sorted(pairs, key=lambda pair: pair[0] - pair[1]):
        if _is_drawn_rectangle(mask, x1, y1, x2, y2):
            return (x1, y1, x2, y2)
    return None


def _overlaps(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / float(union) if union else 0.0


def _contained_in(inner: tuple[int, ...], outer: tuple[int, ...]) -> float:
    """How much of `inner` lies inside `outer`, as a fraction of `inner`."""
    left, top = max(inner[0], outer[0]), max(inner[1], outer[1])
    right, bottom = min(inner[2], outer[2]), min(inner[3], outer[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return intersection / float(area) if area else 0.0


def extract_boxes(image_bytes: bytes) -> list[tuple[float, float, float, float]]:
    """The vendor's person boxes, recovered from the drawn annotation.

    Returns `(x1, y1, x2, y2)` in the snapshot's own pixels -- which are the
    clean frame's pixels too, since both come from the same camera at the same
    resolution.

    Never raises: an unreadable image, or one with no recoverable rectangle,
    returns an empty list. The caller then settles that frame as no violation
    without a judge call -- see the module docstring, which is where the cost
    of that is written down.
    """
    try:
        import cv2
        import numpy as np
    except Exception:  # noqa: BLE001 - no OpenCV is a deployment fact, not an error here
        logger.warning("OpenCV unavailable; cannot recover snapshot boxes")
        return []

    try:
        buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            return []
        mask = cv2.inRange(
            image,
            np.array(RED_LOWER, dtype=np.uint8),
            np.array(RED_UPPER, dtype=np.uint8),
        )
        verticals = _merge_collinear(_runs(mask, vertical=True))
        found: list[tuple[int, int, int, int]] = []
        for first in range(len(verticals)):
            for second in range(first + 1, len(verticals)):
                (left, left_top, left_bottom) = verticals[first]
                (right, right_top, right_bottom) = verticals[second]
                x1, x2 = sorted((left, right))
                if x2 - x1 < MIN_BOX_SIDE:
                    continue
                # The rows the two edges SHARE, never their union. The
                # `violation` label extends one edge upward past the box, and
                # an occluding box cuts the other short.
                top, bottom = max(left_top, right_top), min(left_bottom, right_bottom)
                if bottom - top < MIN_BOX_SIDE:
                    continue
                box = _fit(mask, x1, x2, top, bottom)
                if box is not None:
                    found.append(box)
    except Exception:  # noqa: BLE001 - a bad image must not end a pass
        logger.exception("could not read the annotation from a snapshot")
        return []

    # Largest first, so when one annotation is found from two pairings the
    # box that holds the whole person wins over a piece of it.
    kept: list[tuple[int, int, int, int]] = []
    for box in sorted(found, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1])):
        if any(_overlaps(box, other) > DUPLICATE_IOU for other in kept):
            continue
        if any(_contained_in(box, other) > CONTAINED for other in kept):
            continue
        kept.append(box)

    # Left to right, top to bottom, so `person_index` is stable across runs of
    # the same snapshot rather than following component-labelling order.
    kept.sort(key=lambda b: (round(b[1] / 50), b[0]))
    return [(float(a), float(b), float(c), float(d)) for a, b, c, d in kept]


#: How different two scene hashes may be and still be the same view.
#:
#: MEASURED over 40 non-identical snapshot pairs within 30s of each other:
#: pairs of the same scene scored 1-8, pairs of different scenes 8-39. Six
#: sits inside the first range and below the second's minimum. The box match
#: below is the real safety net; this only decides "same view".
SCENE_HASH_DISTANCE = 6

#: How much two boxes must overlap to be the same person's box.
SAME_BOX_IOU = 0.5

#: How long a scene stays comparable.
#:
#: WAS 60, which assumed a camera re-fires within seconds and that beyond a
#: minute the same view is a new moment. The alarm bucket does not behave
#: that way: the vendor emits back-to-back 60s clips, so a static scene
#: re-alarms at 59-61s and lands ON the boundary. Measured over 40
#: consecutive alarms in `safetyai-cadit`, a pair scoring a scene distance
#: of 1 with identical boxes -- a duplicate by every content test here --
#: was kept only because its gap was 61s.
#:
#: 300 was chosen off that measurement: 60s deduped 8 of 40, 120s 13, 180s
#: 15, 300s 21, 600s 23. The curve flattens after 300.
#:
#: Widening is bounded by the CONTENT tests, not by this: a pair still has
#: to match on scene hash AND on `same_boxes` to collapse, so a longer
#: window can only catch alarms that are near-identical with nobody having
#: moved. `RECENT_SCENES` (40) spans ~40min at this cadence, so the ring
#: still reaches back further than the window.
DUPLICATE_WINDOW_SECONDS = 300


def scene_hash(image_bytes: bytes) -> str:
    """A 64-bit difference hash of the frame, as 16 hex characters.

    Small enough to keep a ring of them on the feed's stored state, which an
    image or a thumbnail is not.

    A difference hash and not the ETag, because the two catch different
    things. Identical BYTES are the vendor re-serving one stale frame, and
    `SnapshotRef.etag` catches those for free from the listing. This catches
    the same camera firing again a few seconds later: the view is the same but
    the JPEG is not, so no equality test can see it. Measured on 40 close
    pairs, only 2 in 24 were byte-identical while 10 in 24 were the same
    scene.
    """
    try:
        import cv2
        import numpy as np
    except Exception:  # noqa: BLE001 - no OpenCV is a deployment fact
        return ""
    if not image_bytes:
        return ""
    try:
        buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return ""
        small = cv2.resize(image, (9, 8))
        bits = small[:, 1:] > small[:, :-1]
        value = 0
        for bit in bits.flatten():
            value = (value << 1) | int(bit)
        return f"{value:016x}"
    except Exception:  # noqa: BLE001 - a bad image must not end a pass
        logger.exception("could not hash a snapshot's scene")
        return ""


def scene_distance(left: str, right: str) -> int:
    """Hamming distance between two scene hashes, or 64 if either is missing."""
    if not left or not right:
        return 64
    try:
        return bin(int(left, 16) ^ int(right, 16)).count("1")
    except ValueError:
        return 64


def _iou(a, b) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, right - left) * max(0.0, bottom - top)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def same_boxes(left: list, right: list) -> bool:
    """Whether two box sets mark the same people.

    THE SAFETY NET, and the reason a scene match alone is not enough. Of 17
    same-scene pairs measured, 13 had boxes that had MOVED -- counts going
    2->1, 1->5, 0->2, which is a different set of people flagged and so a
    different finding. Only 4 were the same view with the same boxes. Treating
    all 17 as duplicates would have discarded 13 real observations.
    """
    if not left or not right or len(left) != len(right):
        return False
    taken: set[int] = set()
    for box in left:
        best, index = 0.0, -1
        for other_index, other in enumerate(right):
            if other_index in taken:
                continue
            overlap = _iou(box, other)
            if overlap > best:
                best, index = overlap, other_index
        if best < SAME_BOX_IOU or index < 0:
            return False
        taken.add(index)
    return True
