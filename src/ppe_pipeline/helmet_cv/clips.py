"""The raw alarm-clip corpus in the UAT bucket: what a clip key means, and
how to enumerate the clips worth extracting frames from.

Every clip under `raw_video/` exists *because* production CV already fired
on it — the filename's leading token is that alarm's own violation type. So
this corpus measures CV precision and nothing else: frames CV missed are,
by construction, not in it. (`uncaptured_hazard/` is the only miss corpus in
the bucket, and it holds 5 files.)

WHAT A KEY CARRIES, AND WHAT IT DOESN'T

    raw_video/helmet_violation_2026-04-27_07-02-09_to_07-03-01_726f88cb-....mp4
              └── type ──────┘ └─ start ───────────┘    └ end ┘ └─ alarm id ─┘

There is no camera and no site anywhere in it. Not in the key, and not in
S3 user-metadata either (every object's `Metadata` is `{}`), so no mapping
can be recovered from the bucket alone — `extract` takes both as
caller-supplied values and defaults them to `unk` until an alarm-id → camera
mapping exists somewhere to join against.

The alarm UUID is the only durable identity, and it is worth keeping whole:
it joins a clip to its `annotated_video/` twin (same footage, production
CV's boxes burned in) and its `snapshot/` still (the single frame the alarm
fired on). At 27k helmet clips an 8-hex prefix would collide ~9% of the time.

TWO SHAPES, AND A TIMEZONE

The end timestamp comes in two forms — 12,688 helmet clips carry a bare
`_to_HH-MM-SS` and 14,478 carry a full `_to_YYYY-MM-DD_HH-MM-SS`. The short
form has no date, so a clip running through midnight has to infer one.

Times are UTC, not Singapore local. Verified against S3: the clip above
reports `LastModified: 2026-04-27T07:03:25+00:00` against a key ending
`07-03-01`. `aws s3 ls` renders local (+08) and shows 15:03 for the same
object, which is what makes this easy to get wrong by eight hours.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

DEFAULT_BUCKET = "safetyai-cadit"
DEFAULT_PREFIX = "raw_video/"

# The bucket lives in ap-southeast-1. This machine's ambient region resolves
# to ap-southeast-2, and addressing a bucket through the wrong regional
# endpoint costs a redirect round trip on *every* call — the same trap
# storage._get_shared_boto3_s3_client documents. Pinned rather than left to
# the environment, because the environment here is already wrong.
DEFAULT_REGION = "ap-southeast-1"

DEFAULT_VIOLATION_TYPE = "helmet_violation"

# Non-greedy type token: it is itself underscore-separated and of varying
# length ("helmet_violation", "fire_and_smoke_incident"), so it can only be
# delimited by the first thing that looks like a date.
_CLIP_RE = re.compile(
    r"^(?P<violation_type>.+?)_"
    r"(?P<start_date>\d{4}-\d{2}-\d{2})_(?P<start_time>\d{2}-\d{2}-\d{2})"
    r"_to_"
    r"(?:(?P<end_date>\d{4}-\d{2}-\d{2})_)?(?P<end_time>\d{2}-\d{2}-\d{2})_"
    r"(?P<alarm_id>[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})$"
)

_STAMP_FORMAT = "%Y-%m-%d %H-%M-%S"


@dataclass(frozen=True)
class ClipRef:
    """One alarm clip, as identified by its S3 key alone."""

    key: str
    violation_type: str
    alarm_id: str
    start: datetime
    end: datetime

    @property
    def duration_seconds(self) -> float:
        return (self.end - self.start).total_seconds()

    @property
    def snapshot_key(self) -> str:
        """The `snapshot/` still for this same alarm — the exact frame CV
        fired on. Built, not listed: the naming is `<type>_<start>_<uuid>`,
        dropping the `_to_<end>` span the clip carries."""
        stamp = self.start.strftime("%Y-%m-%d_%H-%M-%S")
        return f"snapshot/{self.violation_type}_{stamp}_{self.alarm_id}.jpg"


def _parse_stamp(date_part: str, time_part: str) -> datetime:
    return datetime.strptime(f"{date_part} {time_part}", _STAMP_FORMAT).replace(tzinfo=UTC)


def parse_clip_key(key: str) -> ClipRef | None:
    """A `ClipRef` for `key`, or `None` when the name doesn't match the
    alarm-clip convention.

    Returning `None` rather than raising is deliberate: the corpus is 56k
    keys written by a system we don't control, and one unparseable name
    should cost that clip, not the run.
    """
    stem = key.rsplit("/", 1)[-1]
    if stem.lower().endswith(".mp4"):
        stem = stem[: -len(".mp4")]

    match = _CLIP_RE.match(stem)
    if match is None:
        return None

    try:
        start = _parse_stamp(match["start_date"], match["start_time"])
        end = _parse_stamp(match["end_date"] or match["start_date"], match["end_time"])
    except ValueError:
        return None

    # Short-form names carry no end date. An end that lands before its start
    # is therefore the clip running through midnight, not bad data — the
    # only reading available, since the span is always minutes.
    if end < start and not match["end_date"]:
        end += timedelta(days=1)

    return ClipRef(
        key=key,
        violation_type=match["violation_type"],
        alarm_id=match["alarm_id"],
        start=start,
        end=end,
    )


def iter_clips(
    client,
    *,
    bucket: str = DEFAULT_BUCKET,
    prefix: str = DEFAULT_PREFIX,
    violation_type: str | None = DEFAULT_VIOLATION_TYPE,
    start_on_or_after: datetime | None = None,
    start_before: datetime | None = None,
    start_after_key: str | None = None,
) -> Iterator[ClipRef]:
    """Every parseable clip under `prefix`, oldest key first, filtered to
    `violation_type` when given.

    Paginated: the helmet slice alone is 27k keys, so a single
    `list_objects_v2` would silently truncate at 1000.

    The type filter is applied to the key before parsing, which is what
    makes a helmet-only pass cheap — 56k keys arrive either way, but only
    the matching ones are parsed.
    """
    paginator = client.get_paginator("list_objects_v2")
    key_prefix = f"{prefix}{violation_type}_" if violation_type else prefix

    paginate = {"Bucket": bucket, "Prefix": key_prefix}
    if start_after_key is not None:
        if not start_after_key.startswith(key_prefix):
            raise ValueError("start_after_key must be inside the selected clip prefix")
        paginate["StartAfter"] = start_after_key

    for page in paginator.paginate(**paginate):
        for entry in page.get("Contents", ()):
            key = entry["Key"]
            if not key.lower().endswith(".mp4"):
                continue
            clip = parse_clip_key(key)
            if clip is None:
                continue
            if start_on_or_after is not None and clip.start < start_on_or_after:
                continue
            if start_before is not None and clip.start >= start_before:
                continue
            yield clip


def snapshot_alarm_ids(
    client,
    *,
    bucket: str = DEFAULT_BUCKET,
    violation_type: str | None = DEFAULT_VIOLATION_TYPE,
) -> set[str]:
    """Alarm ids that HAVE a `snapshot/` still.

    Measured over 130 clips sampled across the corpus: only 52% had one, against
    90% in the oldest twenty. An alarm with no still has no locatable alarm
    frame, so it can never carry a CV claim and can never enter the CV arm —
    downloading and decoding its clip spends minutes of network and CPU to
    produce frames only the VLM arm could use.

    Listed ONCE and intersected, rather than a `head_object` per candidate:
    `snapshot/` holds ~63k objects, which is one paginated listing, where
    head-checking 27k clips is 27k round trips.

    The id is the last `_`-separated field of the key stem, matching
    `ClipRef.snapshot_key`'s construction (`<type>_<stamp>_<uuid>.jpg`).
    """
    paginator = client.get_paginator("list_objects_v2")
    prefix = f"snapshot/{violation_type}_" if violation_type else "snapshot/"
    ids: set[str] = set()
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for entry in page.get("Contents", ()):
            key = entry["Key"]
            if not key.lower().endswith(".jpg"):
                continue
            ids.add(key.rsplit("/", 1)[-1].rsplit(".", 1)[0].rsplit("_", 1)[-1])
    return ids


def get_s3_client(region_name: str = DEFAULT_REGION):
    """A boto3 S3 client pinned to the bucket's own region.

    Reuses `storage`'s shared-client cache so a process doing both eval I/O
    and frame extraction opens one client per region, not one per caller.
    """
    from .. import storage

    return storage._get_shared_boto3_s3_client(region_name)
