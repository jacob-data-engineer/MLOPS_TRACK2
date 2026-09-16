"""Tests for reading the raw alarm-clip corpus's key naming.

No S3 anywhere: `iter_clips` takes a client, so a fake paginator stands in
for the bucket. What's under test is this module's own reading of a key --
the two end-timestamp shapes, midnight rollover, and refusing to raise on a
name it does not understand.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ppe_pipeline.helmet_cv.clips import ClipRef, iter_clips, parse_clip_key

_ALARM = "726f88cb-a3d0-4907-9d8d-111055ebc676"


def _key(name: str) -> str:
    return f"raw_video/{name}"


class _FakePaginator:
    def __init__(self, pages):
        self._pages = pages
        self.calls = []

    def paginate(self, **kwargs):
        self.calls.append(kwargs)
        for page in self._pages:
            yield page


class _FakeS3:
    """Shaped like the slice of a boto3 S3 client `iter_clips` uses."""

    def __init__(self, keys, *, page_size=2):
        pages = [
            {"Contents": [{"Key": k} for k in keys[i : i + page_size]]}
            for i in range(0, len(keys), page_size)
        ] or [{}]
        self.paginator = _FakePaginator(pages)

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return self.paginator


# --- the two timestamp shapes --------------------------------------------


def test_the_long_form_end_timestamp_carries_its_own_date():
    """14,478 helmet clips are written this way."""
    clip = parse_clip_key(
        _key(f"helmet_violation_2026-04-27_07-02-09_to_2026-04-27_07-03-01_{_ALARM}.mp4")
    )

    assert clip == ClipRef(
        key=_key(f"helmet_violation_2026-04-27_07-02-09_to_2026-04-27_07-03-01_{_ALARM}.mp4"),
        violation_type="helmet_violation",
        alarm_id=_ALARM,
        start=datetime(2026, 4, 27, 7, 2, 9, tzinfo=UTC),
        end=datetime(2026, 4, 27, 7, 3, 1, tzinfo=UTC),
    )


def test_the_short_form_end_timestamp_borrows_the_start_date():
    """The other 12,688 carry a bare `_to_HH-MM-SS` with no date at all."""
    clip = parse_clip_key(_key(f"helmet_violation_2026-04-27_07-02-09_to_07-03-01_{_ALARM}.mp4"))

    assert clip.start == datetime(2026, 4, 27, 7, 2, 9, tzinfo=UTC)
    assert clip.end == datetime(2026, 4, 27, 7, 3, 1, tzinfo=UTC)
    assert clip.duration_seconds == 52.0


def test_a_short_form_clip_running_through_midnight_rolls_the_date_forward():
    """An end before its start is the only reading available -- the span is
    always minutes, so this is a rollover, not bad data."""
    clip = parse_clip_key(_key(f"helmet_violation_2026-04-27_23-59-40_to_00-00-31_{_ALARM}.mp4"))

    assert clip.end == datetime(2026, 4, 28, 0, 0, 31, tzinfo=UTC)
    assert clip.duration_seconds == 51.0


def test_a_long_form_end_before_its_start_is_left_alone():
    """It states its own date, so there is nothing to infer -- rolling it
    forward would invent a span the name explicitly denies."""
    clip = parse_clip_key(
        _key(f"helmet_violation_2026-04-27_23-59-40_to_2026-04-27_00-00-31_{_ALARM}.mp4")
    )

    assert clip.end == datetime(2026, 4, 27, 0, 0, 31, tzinfo=UTC)
    assert clip.duration_seconds < 0


def test_an_underscored_violation_type_survives_the_non_greedy_match():
    """The type token is itself underscore-separated and of varying length,
    so only the first date-shaped thing can delimit it."""
    clip = parse_clip_key(
        _key(f"fire_and_smoke_incident_2026-04-27_07-02-09_to_07-03-01_{_ALARM}.mp4")
    )

    assert clip.violation_type == "fire_and_smoke_incident"


def test_the_alarm_uuid_is_kept_whole():
    """At 27k helmet clips an 8-hex prefix collides ~9% of the time, and the
    uuid is the only thing joining a clip to its snapshot and annotated twin."""
    clip = parse_clip_key(_key(f"helmet_violation_2026-04-27_07-02-09_to_07-03-01_{_ALARM}.mp4"))

    assert clip.alarm_id == _ALARM
    assert clip.snapshot_key == f"snapshot/helmet_violation_2026-04-27_07-02-09_{_ALARM}.jpg"


def test_the_extension_is_optional():
    clip = parse_clip_key(f"helmet_violation_2026-04-27_07-02-09_to_07-03-01_{_ALARM}")

    assert clip is not None


# --- refusing to raise ----------------------------------------------------


def test_an_unparseable_name_costs_that_clip_not_the_run():
    """56k keys are written by a system we do not control."""
    assert parse_clip_key("raw_video/not-an-alarm-clip.mp4") is None
    assert parse_clip_key(_key(f"helmet_violation_2026-04-27_07-02-09_{_ALARM}.mp4")) is None
    assert parse_clip_key(_key("helmet_violation_2026-04-27_07-02-09_to_07-03-01_nope.mp4")) is None


def test_an_impossible_calendar_date_returns_none_rather_than_raising():
    """The regex accepts the shape; only strptime knows there is no 27th month."""
    assert parse_clip_key(_key(f"helmet_violation_2026-27-04_07-02-09_to_07-03-01_{_ALARM}.mp4")) is None


# --- enumerating ----------------------------------------------------------


def test_iter_clips_paginates_rather_than_truncating_at_1000():
    """The helmet slice alone is 27k keys."""
    keys = [
        _key(f"helmet_violation_2026-04-{day:02d}_07-02-09_to_07-03-01_{_ALARM}.mp4")
        for day in range(1, 8)
    ]
    client = _FakeS3(keys, page_size=2)

    found = list(iter_clips(client))

    assert len(found) == 7


def test_the_type_filter_is_pushed_into_the_key_prefix():
    """What makes a helmet-only pass cheap: 56k keys arrive either way, but
    only the matching ones are ever parsed."""
    client = _FakeS3([])

    list(iter_clips(client, violation_type="helmet_violation"))

    assert client.paginator.calls[0]["Prefix"] == "raw_video/helmet_violation_"


def test_no_type_filter_lists_the_whole_prefix():
    client = _FakeS3([])

    list(iter_clips(client, violation_type=None))

    assert client.paginator.calls[0]["Prefix"] == "raw_video/"


def test_exclusive_watermark_is_pushed_to_s3_start_after():
    client = _FakeS3([])
    watermark = _key(f"helmet_violation_2026-04-27_07-02-09_to_07-03-01_{_ALARM}.mp4")

    list(iter_clips(client, start_after_key=watermark))

    assert client.paginator.calls[0]["StartAfter"] == watermark


def test_watermark_outside_selected_prefix_is_rejected():
    with pytest.raises(ValueError, match="selected clip prefix"):
        list(iter_clips(_FakeS3([]), start_after_key="raw_video/glove_violation_bad.mp4"))


def test_non_mp4_and_unparseable_keys_are_skipped():
    keys = [
        _key(f"helmet_violation_2026-04-27_07-02-09_to_07-03-01_{_ALARM}.mp4"),
        _key("helmet_violation_something_else.jpg"),
        _key("helmet_violation_garbage.mp4"),
    ]

    found = list(iter_clips(_FakeS3(keys, page_size=3)))

    assert [c.alarm_id for c in found] == [_ALARM]


def test_the_date_window_filters_on_the_clip_start():
    keys = [
        _key(f"helmet_violation_2026-04-{day:02d}_07-02-09_to_07-03-01_{_ALARM}.mp4")
        for day in (1, 5, 10)
    ]

    found = list(
        iter_clips(
            _FakeS3(keys, page_size=3),
            start_on_or_after=datetime(2026, 4, 5, tzinfo=UTC),
            start_before=datetime(2026, 4, 10, tzinfo=UTC),
        )
    )

    assert [c.start.day for c in found] == [5]
