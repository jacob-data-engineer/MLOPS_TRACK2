"""Durable, capped ingestion of production helmet alarms."""

from __future__ import annotations

import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
import threading
from pathlib import Path
from typing import Callable

from .. import storage
from ..config import get_config
from .clips import DEFAULT_BUCKET, get_s3_client, parse_clip_key
from .snapshot_boxes import (
    DUPLICATE_WINDOW_SECONDS,
    SCENE_HASH_DISTANCE,
    clip_key_for,
    extract_boxes,
    iter_snapshots,
    same_boxes,
    scene_distance,
    scene_hash,
)
from .extract import extract_clip
from .records import FRAME_ROLE_ALARM, CvProvenance, frame_stem
from ..person_detector import get_person_detector
from .runner import judge_candidates, records_from_boxes

DEFAULT_STATE_KEY = "eval/live_helmet_feed.json"
#: How many times one clip may fail before the feed moves past it.
#:
#: A failure holds the cursor on purpose -- an alarm nobody judged must not be
#: skipped silently. But some failures are DETERMINISTIC: about 14% of this
#: corpus never yields a clean alarm frame (the pilot located 316 of 380), and
#: 13 of 380 clips fail to decode at all with hevc reference errors. Retrying
#: one of those forever means every alarm behind it is never judged, and the
#: screen shows a growing failure count for a feed that has silently stopped
#: making progress. Measured on a real run: 8 failure records, ONE clip.
#:
#: Three attempts, then it is recorded as skipped and the cursor moves on. The
#: alarm stays on the record as one that was never judged; what changes is
#: that it no longer blocks the queue behind it.
MAX_CLIP_ATTEMPTS = 3
POLL_SECONDS = 30
LEASE_SECONDS = 90
PROCESS_LOCK = threading.Lock()

#: Whether a duplicate caught BEFORE the clip is downloaded actually skips it.
#:
#: FALSE, ON PURPOSE, AND THIS IS A MEASUREMENT IN PROGRESS.
#:
#: In detector mode the two duplicate checks look at two different images.
#: The pre-filter reads the vendor's annotated still -- the vendor's own boxes
#: over the vendor's own scene -- and it can be asked before anything is
#: downloaded. The check after extraction reads the CLEAN frame with our
#: detector's boxes on it, which is the image the judge is actually shown, and
#: it cannot be asked any earlier than that. Nobody has measured how often the
#: two disagree, and acting on the first one means the second is never asked.
#:
#: MEASURED, 4 Sep, run `live-594f7d3e1b53`: 40 alarms, detector mode. The
#: 2x2 the shadow existed to produce:
#:
#:                       post: DUP   post: not-dup
#:     pre: DUP              20            4
#:     pre: not-dup           0           16
#:
#: The pre-filter never missed a duplicate the clean-frame check caught, so it
#: is a strict superset. The four it would skip that the clean-frame check
#: kept are NOT it being wrong: those four are the same view by the
#: CLEAN-FRAME hash itself -- pairwise distances 0 to 4 against a threshold of
#: 6, one pair identical -- so that check cannot have rejected them on the
#: picture. It rejected them on `same_boxes`, and in detector mode those are
#: OUR detector's boxes, which jitter. Five people, five boxes, five judged,
#: all five confirmed false positive, on all four. The vendor's drawn
#: rectangles do not jitter, which is why the pre-filter got them right.
#:
#: The two fingerprint spaces also turned out to be nearly one space: clean vs
#: annotated hash on those alarms measured 1, 0, 2 and 0 bits of 64. The
#: overlay barely moves a dHash, exactly as `scene_hash` predicted.
#:
#: WHAT THIS DOES NOT SETTLE. Forty alarms, one ~20 minute window, one camera,
#: five idle workers, which is a favourable case for dedupe. No alarm in the
#: risk cell was a violation, which is the case that would actually cost
#: something. And the residual danger is not this switch's: a worker removing
#: a helmet while everyone stands still moves neither the hash nor the boxes,
#: so the clean-frame check is equally blind to it. That belongs to dedupe
#: itself and to `DUPLICATE_WINDOW_SECONDS`.
#:
#: VENDOR MODE IS NOT SHADOWED BY THIS and never was: there the pre-filter
#: compares exactly what the old check compared, at the same point in the
#: alarm, and it is the only duplicate check that run has. See
#: `process_live_snapshot`.
PREFILTER_SKIPS_EXTRACTION = True


@dataclass(frozen=True)
class FeedCaps:
    per_run_calls: int
    daily_calls: int

    def __post_init__(self) -> None:
        if self.per_run_calls < 1 or self.daily_calls < 1:
            raise ValueError("live feed call caps must be positive")


#: Why a deployment might not be able to read the clip bucket, in the words
#: the operator needs.
#:
#: THE 1 SEP POSTURE IS SUPERSEDED, deliberately and on the record.
#:
#: CEO, 1 Sep: production footage does not go on the app server -- "dont push
#: on instance" -- so the deployed role was to have no access and the feed was
#: to run from a workstation whose own credential did. On 3 Sep the
#: `ClipBucketRead` inline policy on the `EC2_access` role was widened to
#: `safetyai-cadit` so the deployed Start button would work.
#:
#: That left the policy saying one thing and the decision another, so it was
#: put back to the CEO on 6 Sep with the consequence stated plainly: keeping
#: 1 Sep means this panel goes disabled and runs move to a workstation.
#: **CEO, 6 Sep: "i want to click start on the ui".** The instance keeps its
#: read on the production clip bucket, and 1 Sep no longer applies.
#:
#: So a deployed instance CAN read the clip bucket, by decision rather than by
#: drift. Anyone narrowing this policy back is reversing a call the CEO made
#: knowing what it costs -- take it to them first.
#:
#: The message stays because a deployment can still meet AccessDenied -- a
#: narrowed policy, a different instance, a local run with no credential --
#: and an operator meeting a raw AccessDenied will reasonably assume
#: something is broken and ask for it to be granted.
#: KEPT TERSE, and with no em-dash or `--`: it is on-screen copy, and the
#: house rule for on-screen copy is terse. What an operator needs is that it
#: may be deliberate rather than broken, where to run it instead, and the
#: error itself. The reasoning above is for whoever edits this, not for them.
NO_BUCKET_ACCESS_DETAIL = (
    "Cannot read the alarm clip bucket {bucket!r}. Run the live feed from a machine whose "
    "AWS credentials can read it. Everything else here works normally. ({error})"
)


def check_bucket_access(client, bucket: str = DEFAULT_BUCKET) -> None:
    """Prove this deployment can list the clip bucket. Raises if it cannot.

    One `list_objects_v2` for a single key -- cheap, and it fails the way the
    real poll would. Checked before a run starts because bucket access is a
    fact about the DEPLOYMENT, not about an alarm: discovered inside the
    worker it became a durable failure record per arriving clip for a
    condition no retry can fix.
    """
    client.list_objects_v2(Bucket=bucket, MaxKeys=1)


def bucket_readable(bucket: str = DEFAULT_BUCKET) -> tuple[bool, str]:
    """Whether this deployment can read the clip bucket, and why not.

    Answered ONCE per process and cached: it is a property of the machine's
    credentials, and the screen polls the feed's status every two seconds.
    A failure is cached too -- an operator who fixes the credential restarts
    the app, which is what a deployment change means here anyway.
    """
    global _BUCKET_READABLE
    if _BUCKET_READABLE is None:
        try:
            check_bucket_access(get_s3_client(), bucket)
            _BUCKET_READABLE = (True, "")
        except Exception as exc:  # noqa: BLE001 - any failure to read is a no
            _BUCKET_READABLE = (False, str(exc))
    return _BUCKET_READABLE


#: Cache for `bucket_readable`. `None` means "not asked yet".
_BUCKET_READABLE: tuple[bool, str] | None = None


def new_state(
    caps: FeedCaps,
    *,
    now: datetime | None = None,
    previous: dict | None = None,
    start_on_or_after: datetime | None = None,
    resume: bool = False,
    box_source: str = "vendor_cv",
) -> dict:
    """A fresh run's state, and WHERE IN THE FEED IT STARTS.

    Three starting points, because an operator and a tester want opposite
    things from the same button:

    * `resume=True` -- carry the previous run's watermark and read forward
      from it. "The oldest alarm nobody has judged yet", which for a first
      run is the beginning of the corpus. This is how the historical corpus
      gets replayed.
    * `start_on_or_after=<time>` -- ignore any old watermark and take alarms
      from that time on. Passing `now` is the operator's feed: alarms that
      arrive from here, nothing historical.
    * neither -- everything, from the beginning.

    The old watermark is DROPPED unless `resume`, because a carried watermark
    silently beats a start time: the run would resume mid-corpus while the
    screen said it was starting from today.

    `processed_keys` always carries. It is what makes a re-run idempotent,
    and that is true whichever end of the feed you start from.

    `box_source` is fixed for the life of the run and is taken HERE rather
    than written onto the state afterwards, because `pre_extraction_acts` is
    derived from it: set apart, the two could disagree, and the screen would
    describe behaviour the run does not have.
    """
    now = now or datetime.now(UTC)
    previous = previous or {}
    same_day = previous.get("day") == now.date().isoformat()
    return {
        "run_id": f"live-{uuid.uuid4().hex[:12]}",
        "started_at": now.isoformat(),
        "status": "running",
        "watermark": previous.get("watermark") if resume else None,
        "start_on_or_after": start_on_or_after.isoformat() if start_on_or_after else None,
        # Recorded on the RUN so a state written before 2 Sep, when the feed
        # could also be driven by our own detector over `raw_video/`, is not
        # mistaken for one of these. Its watermark points into a different key
        # space and would mean nothing here.
        "box_source": box_source,
        # WHETHER A PRE-EXTRACTION DUPLICATE IS ACTUALLY SKIPPED ON THIS RUN,
        # or only recorded and judged anyway.
        #
        # Recorded, not re-derived. The screen has to say which of those
        # happened, and the only honest source is the run itself: inferred
        # from `box_source` alone it is right only while
        # `PREFILTER_SKIPS_EXTRACTION` is False, and the day that flips the
        # screen would read "nothing was skipped" while alarms were being
        # skipped and calls saved.
        #
        # THIS EXPRESSION IS THE GATE in `process_live_snapshot`, written out
        # a second time because it has to be answerable before the first alarm
        # arrives. If one of them moves, the other moves with it.
        "pre_extraction_acts": PREFILTER_SKIPS_EXTRACTION or box_source != "detector",
        "processed_keys": dict(previous.get("processed_keys", {})),
        # CARRIED, like `processed_keys`, and for the same reason: the vendor
        # re-alarms on one image for days, so a run that forgot what it had
        # already judged would pay for the duplicate all over again.
        "processed_images": dict(previous.get("processed_images", {})),
        # CARRIED, because a failure is a durable record of an alarm that was
        # not judged and the next run has to be able to see it. Each entry
        # says which run produced it, so a screen can tell "this run failed
        # 18 times" from "eighteen failures happened once, days ago" -- read
        # as the former, an old list makes every new run look broken.
        "failures": list(previous.get("failures", [])),
        # Clips that failed `MAX_CLIP_ATTEMPTS` times and were stepped over.
        # Carried like `processed_keys`: a deterministic failure is still
        # deterministic on the next run, and re-trying it three more times per
        # run would put the same stall back.
        "skipped_keys": dict(previous.get("skipped_keys", {})),
        "attempts": dict(previous.get("attempts", {})),
        "calls": 0,
        "daily_calls": int(previous.get("daily_calls", 0)) if same_day else 0,
        "day": now.date().isoformat(),
        "per_run_cap": caps.per_run_calls,
        "daily_cap": caps.daily_calls,
        # WHEN THIS STATE WAS LAST WRITTEN. It used to mean "when a result was
        # last recorded", stamped at three of `poll_once`'s eight checkpoints
        # only, so a run stepping through duplicates or paused on budget wrote
        # its state repeatedly without ever moving this. Every checkpoint moves
        # it now -- see `poll_once.checkpoint_state`.
        #
        # NOT WHAT THE SCREEN SHOWS, and it never was. The field is not on the
        # `LiveFeedState` type and nothing under `web/src` reads it; the frozen
        # clock an operator actually saw was `in_flight.since`, fixed
        # separately. This is for anyone reading the stored object, and for
        # whoever next asks "did this run die or is it just slow".
        "updated_at": now.isoformat(),
    }


def remaining_calls(state: dict, *, today: date | None = None) -> int:
    today = today or datetime.now(UTC).date()
    daily_used = state.get("daily_calls", 0) if state.get("day") == today.isoformat() else 0
    return max(
        0,
        min(
            int(state["per_run_cap"]) - int(state.get("calls", 0)),
            int(state["daily_cap"]) - int(daily_used),
        ),
    )


class StageReporter:
    """WHAT THE ALARM IN FLIGHT IS DOING RIGHT NOW, in memory and only there.

    One alarm is minutes of work -- a clip download, an ffmpeg decode, a
    detector pass, then one judge call per person, measured at ~25s each. The
    screen polls every two seconds and had nothing to show for any of it, so a
    feed correctly working through a nine-person alarm was indistinguishable
    from one doing nothing.

    NOT CHECKPOINTED. `save_state` is a GET plus a PUT of the whole state
    object (212KB today, heading for ~53MB at the full corpus) through a
    compare-and-swap, and `poll_once` already pays for two of those per alarm.
    Writing one per stage would triple that to say "downloading the clip", and
    every extra CAS is another `ConcurrentModificationError` -- which, out of
    `save_state`, ends the run. This works on object identity instead: the
    session dict, the worker thread and `GET /live-feed/status` all hold the
    SAME state object, so a mutation here is on the next two-second poll at no
    storage cost. A stage does get persisted incidentally by the checkpoints
    that already exist, and a stale one can survive into the stored state --
    which is why the screen renders this only for the session that owns the run.

    `in_flight` IS REPLACED, NEVER MUTATED IN PLACE. `/live-feed/status`
    returns `{**state}`, a shallow copy that shares this nested dict, so adding
    a key to it while the serializer walks it raises "dictionary changed size
    during iteration". Rebinding the top-level key changes no size and is safe
    under the GIL.
    """

    def __init__(self, state: dict | None, heartbeat: Callable[[], None] | None = None):
        self._state = state
        # `crop_workers` defaults to 4, so the crop counter is a completion
        # count written from several threads, not a loop index.
        self._lock = threading.Lock()
        # PROGRESS IS ALSO PROOF OF LIFE. Every stage change and every finished
        # crop is a moment this worker can say it is still here, and inside a
        # four-minute alarm they are the ONLY such moments -- see `poll_once`.
        self._heartbeat = heartbeat

    def _beat(self) -> None:
        # OUTSIDE `self._lock`, always. The heartbeat writes to the store, and
        # holding the progress lock across a network round trip would stall
        # every other crop thread reporting its own completion.
        if self._heartbeat is not None:
            self._heartbeat()

    def stage(self, name: str, **fields) -> None:
        with self._lock:
            current = self._state.get("in_flight") if self._state is not None else None
            if isinstance(current, dict):
                self._state["in_flight"] = {**current, "stage": name, **fields}
        self._beat()

    def crop_judged(self) -> None:
        with self._lock:
            current = self._state.get("in_flight") if self._state is not None else None
            if isinstance(current, dict):
                self._state["in_flight"] = {
                    **current,
                    "crops_judged": int(current.get("crops_judged", 0)) + 1,
                }
        self._beat()


#: For a caller that has no state to report onto, which is most tests.
NO_REPORTER = StageReporter(None)


def mark_activity(state: dict, stage: str) -> None:
    """WHAT THE WORKER IS DOING WHEN NO ALARM IS IN FLIGHT.

    The counterpart to `StageReporter`, and in memory for the same reasons.
    `activity` and `in_flight` are mutually exclusive on purpose, so a reader
    never has to decide which of the two is the live one: an alarm is
    described by `in_flight`, and every gap between alarms by this.

    Rebinds the key rather than editing the dict under it -- `{**state}` is a
    shallow copy that shares it, and the argument is the one on
    `StageReporter`.
    """
    state["activity"] = {"stage": stage, "since": datetime.now(UTC).isoformat()}


def poll_once(
    state: dict,
    *,
    client,
    processor: Callable[[ClipRef, int, str, StageReporter], dict],
    checkpoint: Callable[[dict], None] | None = None,
    bucket: str = DEFAULT_BUCKET,
    now: datetime | None = None,
) -> dict:
    """Process new keys in order; only successful alarms advance the cursor."""
    now = now or datetime.now(UTC)
    if state.get("status") != "running":
        state.pop("in_flight", None)
        state.pop("activity", None)
        return state
    # PINNED, AND IT MUST STAY PINNED, unlike every other reading below.
    #
    # A poll runs for many minutes and can straddle UTC midnight. Re-reading
    # the clock here mid-poll would zero `daily_calls` and hand the run a
    # second full daily budget. The `remaining_calls` call below shares this
    # one reading for the same reason, and they must not diverge: with the day
    # rolled over there and not here, `remaining_calls` sees `state["day"]`
    # against a later date, reads `daily_used` as 0, grants the whole daily
    # budget, and the spend is then added to the PREVIOUS day's counter.
    if state.get("day") != now.date().isoformat():
        state["day"] = now.date().isoformat()
        state["daily_calls"] = 0

    def saving() -> None:
        """`saving` goes on whichever field is describing the worker.

        `in_flight` while an alarm is being worked on, `activity` otherwise.
        One label, two carriers, because the operator is reading one line and
        the answer to "what is it doing" is the same either way.

        NOT RESTORED afterwards, and it does not need to be: every path out of
        a checkpoint reaches something that sets its own stage -- the next
        listing bump, the processor's first stage, or the worker's `waiting`.
        """
        if isinstance(state.get("in_flight"), dict):
            reporter.stage("saving")
        else:
            mark_activity(state, "saving")

    def checkpoint_state() -> None:
        """Stamp the state as written, hold the lease, and store it.

        EVERY checkpoint goes through here. `updated_at` used to be stamped at
        three of the eight sites, from `now` -- one reading taken once per poll
        -- so the screen's clock did not move however many alarms completed,
        and the three sites that checkpoint without a result at all (the
        byte-identical duplicate skip, the budget pause, and an alarm too
        expensive to judge) never moved it even in principle. A run working
        through duplicates showed a frozen clock while making real progress.

        THE LEASE IS HELD HERE TOO, and this is the only place it can be. It
        is refreshed once per while-loop iteration in the worker, `LEASE_SECONDS`
        is 90, and one `poll_once` runs for many minutes -- so the lease was
        expired for most of a poll and `claim_state` would let another process
        take a feed that was busy spending money. `PROCESS_LOCK` covers the
        in-process case; this covers a second worker or a restart.
        """
        state["updated_at"] = datetime.now(UTC).isoformat()
        refresh_lease(state)
        if checkpoint is None:
            return
        # THE WRITE IS ITSELF A STAGE. `save_state` is a GET plus a PUT of the
        # whole state object through a compare-and-swap: 212KB today, where it
        # is imperceptible, and heading for ~53MB at the full corpus, where it
        # is not. Unnamed, the screen would hold whatever label it last had and
        # lie about where the time went, which is the complaint this batch
        # exists to answer. Read and write are ONE label deliberately: two
        # flickering past on a fast object is noise, and one that starts
        # mattering as the object grows is the signal.
        saving()
        checkpoint(state)

    beat_lock = threading.RLock()
    last_beat = [time.monotonic()]
    beating = [False]

    def heartbeat() -> None:
        """HOLD THE LEASE FROM INSIDE THE ALARM, not only at its edges.

        There is no checkpoint between the one that announces an alarm and the
        one that records its result, and that gap IS the alarm: measured at
        about four minutes for a nine-crop frame, against `LEASE_SECONDS` of
        90. So for most of every alarm the stored lease was expired and
        `claim_state` would hand a paid, in-flight run to a second process.
        Refreshing at the edges cannot fix that; the middle is the problem.

        DRIVEN BY PROGRESS, THROTTLED BY THE CLOCK. Every stage change and
        every finished crop offers a beat, but only one per `LEASE_SECONDS/3`
        is taken -- so the write rate is bounded by wall clock rather than by
        how many crops an alarm happens to have, and a nine-crop alarm costs
        about eight state writes instead of two. That is the price of a lease
        that means anything, and it is paid in the one place that was silent.

        `monotonic`, not the wall clock: a clock stepped backwards by NTP
        would stop the beats and drop the lease under a healthy run.
        """
        with beat_lock:
            if beating[0]:
                # The checkpoint below announces its own `saving` stage, which
                # offers a beat straight back into here. One beat, not a
                # recursion -- hence the flag and the reentrant lock.
                return
            moment = time.monotonic()
            if moment - last_beat[0] < LEASE_SECONDS / 3:
                return
            last_beat[0] = moment
            beating[0] = True
            try:
                checkpoint_state()
            finally:
                beating[0] = False

    reporter = StageReporter(state, heartbeat)

    # THE FEED IS DRIVEN BY `snapshot/`, the vendor's annotated stills, and
    # the boxes come from the annotation drawn on them. There are fewer
    # snapshots than clips, and that is correct rather than a shortfall: a
    # snapshot exists where the vendor's CV fired, which is the population
    # this arm is about.
    start_time = None
    if state.get("watermark") is None:
        # `started_at` is the fallback for a state written before the start
        # point was a choice: those runs meant "from now", and must keep
        # meaning it.
        stamp = state.get("start_on_or_after", state.get("started_at"))
        if stamp:
            start_time = datetime.fromisoformat(stamp)

    # WHAT HAPPENS BETWEEN ALARMS, which `in_flight` cannot describe because
    # there is no alarm in flight. Walking the listing is not free: a resumed
    # run reads forward from its cursor over a 75,242-key prefix, and until
    # this said so the screen went blank for that whole time. A run started
    # "from now" used to be the worst of it -- 76 pages and 11.75s per poll,
    # discarding every key -- and now begins at a cursor derived from its
    # start time (OPEN_TASKS #39). In memory only, like every stage -- see
    # `StageReporter`.
    listing_since = datetime.now(UTC).isoformat()
    # A one-slot list, because `listing` and `examined_one` both close over it
    # and one of them writes.
    examined = [0]

    def listing() -> None:
        # Its own writer rather than `mark_activity`, for the two things that
        # differ: `keys_examined`, and a `since` that is pinned to when the
        # scan resumed rather than re-read on every key.
        state["activity"] = {
            "stage": "listing",
            "since": listing_since,
            "keys_examined": examined[0],
        }

    def examined_one() -> None:
        # COUNTED WHERE THE KEY IS LOOKED AT, inside the listing, not where one
        # is yielded. `iter_snapshots` still drops keys client-side -- the time
        # filter it keeps as a backstop, and anything that does not parse -- so
        # counting the loop body would leave this frozen at 0 through a scan
        # that yields nothing.
        examined[0] += 1
        listing()

    listing()
    for clip in iter_snapshots(
        client,
        bucket=bucket,
        violation_type="helmet_violation",
        start_after_key=state.get("watermark"),
        start_on_or_after=start_time,
        on_key=examined_one,
    ):
        if state.get("status") != "running":
            break
        if clip.key in state["processed_keys"]:
            state["watermark"] = clip.key
            continue
        if clip.key in state.get("skipped_keys", {}):
            # Already given its three attempts. Stepping over it again is
            # free; retrying it is not.
            state["watermark"] = clip.key
            continue
        # THE SAME IMAGE, ALARMED AGAIN.
        #
        # The vendor re-serves a stale frame with a fresh alarm id. Measured
        # over the 31,281 helmet snapshots the prefix held at the time: 24,463
        # distinct images, 6,818 redundant alarms, 21.8% of the corpus, one
        # CAM01 frame uploaded 38 times across 3.5 days. Only 139 of those
        # groups fall within a minute, so this is not burst firing. THE PREFIX
        # NOW HOLDS 75,242 KEYS and the ratio has not been re-measured; the
        # 21.8% above describes the older, smaller sample.
        #
        # Judging one again would spend a call to re-answer a question, count
        # one event twice in every metric, and put an exact duplicate in the
        # training set. Caught on the ETag from the LISTING, so it costs
        # nothing: no download, no judge call.
        #
        # `processed_keys` still records the key, so the skip is visible as a
        # processed alarm rather than a gap in the cursor.
        seen = state.setdefault("processed_images", {})
        if clip.etag and clip.etag in seen:
            state["processed_keys"][clip.key] = {
                "status": "duplicate_image",
                "calls": 0,
                "first_seen": seen[clip.etag],
                "run_id": state["run_id"],
            }
            state["duplicates_skipped"] = int(state.get("duplicates_skipped", 0)) + 1
            state["watermark"] = clip.key
            checkpoint_state()
            continue
        remaining = remaining_calls(state, today=now.date())
        if remaining == 0:
            state["paused_reason"] = "budget"
            # WHAT HAPPENED, THE NUMBERS, THE NEXT MOVE, and nothing else:
            # on-screen copy is terse here.
            state["paused_detail"] = (
                f"Call budget spent: {state.get('calls', 0)} of {state['per_run_cap']} this "
                f"run, {state.get('daily_calls', 0)} of {state['daily_cap']} today. Raise the "
                f"caps to continue."
            )
            checkpoint_state()
            break
        else:
            # WHICH ALARM IS BEING WORKED ON, written before the work starts.
            #
            # One alarm is a clip download, an ffmpeg decode to find the clean
            # alarm frame, a detector pass, and then one judge call per person
            # -- minutes, during which nothing else in this state changes:
            # `calls` only moves when a whole frame is done. So a feed
            # correctly working through its first alarm looked identical to one
            # doing nothing, and the screen said "waiting on the first alarm"
            # for minutes while it was in fact judging one. `reporter` fills in
            # which of those minutes it is currently in.
            #
            # `since` IS READ FRESH, not from `now`. One poll works through
            # many alarms on a single reading of the clock, so every alarm's
            # start was stamped with the poll's start: measured on the live
            # feed, an `in_flight.since` 84 seconds BEFORE another alarm in the
            # same poll had already finished. Not stale, self-contradictory.
            state["in_flight"] = {
                "clip_key": clip.key,
                "alarm_id": clip.alarm_id,
                "since": datetime.now(UTC).isoformat(),
            }
            # THE TWO ARE MUTUALLY EXCLUSIVE, so a reader never has to decide
            # which of them is the live one: `activity` describes the gaps
            # between alarms, `in_flight` describes an alarm.
            state.pop("activity", None)
            checkpoint_state()
            try:
                result = processor(clip, remaining, state["run_id"], reporter)
            except Exception as exc:  # noqa: BLE001 - record and retry this same key next poll
                result = {"status": "failed", "calls": 0, "error": f"{type(exc).__name__}: {exc}"}
        calls = int(result.get("calls", 0))
        if calls < 0 or calls > remaining:
            # RAISED WITH `in_flight` STILL ON THE STATE, on purpose. This is
            # the one exception `poll_once` raises itself, and so the one that
            # reliably produces a dead run -- clearing first left the worker's
            # handler with nothing to name but `watermark`, which is the
            # PREVIOUS, successful alarm. The record would have blamed the
            # wrong key. See the handler in `_run_live_feed`.
            raise ValueError("live processor exceeded its granted call budget")
        state.pop("in_flight", None)
        listing_since = datetime.now(UTC).isoformat()
        listing()
        state["calls"] += calls
        state["daily_calls"] += calls
        if result.get("status") == "failed":
            error = result.get("error", "unknown failure")
            attempts = dict(state.get("attempts", {}))
            attempt = int(attempts.get(clip.key, 0)) + 1
            attempts[clip.key] = attempt
            state["attempts"] = attempts
            # ONE record per clip, counting up -- not one per attempt. Eight
            # entries for one clip read as eight broken alarms.
            record = {
                "clip_key": clip.key,
                "error": error,
                "run_id": state["run_id"],
                "at": datetime.now(UTC).isoformat(),
                "attempts": attempt,
            }
            for index, existing in enumerate(state["failures"]):
                if existing.get("clip_key") == clip.key and existing.get("run_id") == state["run_id"]:
                    state["failures"][index] = record
                    break
            else:
                state["failures"].append(record)
            if attempt >= MAX_CLIP_ATTEMPTS:
                # Deterministically broken. Recorded as never judged, and
                # stepped over so the alarms behind it are not held hostage.
                state["skipped_keys"][clip.key] = {"error": error, "attempts": attempt}
                state["watermark"] = clip.key
                checkpoint_state()
                continue
            checkpoint_state()
            break
        if result.get("status") == "stopped":
            state["partial_frames"] = list(state.get("partial_frames", [])) + [
                {
                    "clip_key": clip.key,
                    "run_id": state["run_id"],
                    "calls": calls,
                    "unjudged_people": int(result.get("stopped", 0)),
                }
            ]
            checkpoint_state()
            break
        if result.get("status") == "not_judged_budget":
            state["paused_reason"] = "budget"
            # WHY it stopped, in calls. Without this the screen says
            # "running" while nothing can ever happen: a cap below one
            # alarm's cost skips every alarm, forever, in silence.
            #
            # The reason whole frames are all-or-nothing (half a judged frame
            # reads as a frame nobody was in violation in) used to be in the
            # message itself. It is an explanation, not an instruction, and an
            # operator staring at a stalled feed needs the number and the next
            # move. On-screen copy is terse; the reasoning lives here.
            state["paused_detail"] = (
                f"This alarm needs {result.get('needed', 0)} calls for "
                f"{result.get('candidates', 0)} people; {remaining} left in the budget. "
                f"Raise the per-run cap and start again."
            )
            checkpoint_state()
            break
        # WHICH RUN JUDGED IT. `processed_keys` carries across runs forever
        # and had nothing on an entry to say which run wrote it, so a second
        # run merged indistinguishably into the first and no later reading
        # could separate them. That matters most for the shadow: two runs of
        # `PREFILTER_SKIPS_EXTRACTION` in different positions would be one
        # undifferentiated pile of results. It is a `run_id` per entry now,
        # like `failures` already carries; recovering it afterwards is not
        # possible at any price.
        state["processed_keys"][clip.key] = {**result, "run_id": state["run_id"]}
        # RECORDED ONLY ONCE JUDGED. Registering the image before the work
        # would make a failure look like a duplicate and skip the retry.
        if clip.etag:
            state.setdefault("processed_images", {}).setdefault(clip.etag, clip.key)
        # THE RING OF RECENT SCENES, for the near-duplicate check. Hashes and
        # boxes only: a thumbnail would not fit on a stored state rewritten
        # every checkpoint. A duplicate adds nothing -- it would chain the
        # window forward off an alarm nobody judged.
        if result.get("scene_hash") and result.get("status") != "duplicate_event":
            ring = list(state.get("recent_scenes", []))
            ring.append({
                "key": clip.key,
                "at": clip.start.isoformat(),
                "scene_hash": result["scene_hash"],
                "boxes": [list(b) for b in result.get("boxes", [])],
            })
            state["recent_scenes"] = ring[-RECENT_SCENES:]
        # AND THE SAME RING IN SNAPSHOT SPACE, under the same guard and for
        # the same reason. It is what the check BEFORE extraction compares
        # against, and it has to be kept separately: in detector mode the two
        # rings describe two different images of one moment, and an entry from
        # one compared against the other would be a hash of the clean frame
        # measured against a hash of the annotated still.
        #
        # FALLING BACK TO `scene_hash`/`boxes` when the result carries no
        # snapshot-space copy, which is vendor mode: there the still IS the
        # image the alarm was judged from, so one copy describes both rings
        # and storing a second per alarm cost ~9.9MB across the corpus to say
        # it twice. Read by key PRESENCE, not truthiness -- a detector alarm
        # whose still could not be read carries an empty hash on purpose, and
        # must not silently fall back to the clean frame's.
        #
        # AND AN ALARM THE PRE-FILTER FLAGGED STAYS OUT, which is the extra
        # condition this ring has and `recent_scenes` does not.
        #
        # In shadow mode a flagged alarm is judged anyway, so its status is
        # `judged` and it would join. With `PREFILTER_SKIPS_EXTRACTION` on,
        # the same alarm returns `duplicate_event` and does not -- so the ring
        # the shadow measures against would not be the ring the run it is
        # predicting would have. The error compounds rather than cancels:
        # a match needs the hash AND the boxes, so A->B->C chains form where C
        # matches B but not A, and every B left in the ring flags a C a
        # skipping run would have judged. The shadow would over-report what
        # flipping the constant saves, which is the one number it exists to
        # produce. `DUPLICATE_WINDOW_SECONDS` went to 300s the commit before
        # this ring existed, which is exactly the regime those chains form in.
        #
        # `recent_scenes` is right to have no such condition: it drives what
        # this run actually does, and this run really did judge that alarm on
        # that clean frame. This ring answers a counterfactual instead.
        #
        # In vendor mode nothing changes: a flagged alarm returns
        # `duplicate_event` there and was already out, and no vendor result
        # carries the field at all.
        annotated_hash = result.get("annotated_hash", result.get("scene_hash", ""))
        annotated_boxes = result.get("annotated_boxes", result.get("boxes", []))
        if (
            annotated_hash
            and result.get("status") != "duplicate_event"
            and result.get("pre_extraction_duplicate_of") is None
        ):
            ring = list(state.get("recent_annotated", []))
            ring.append({
                "key": clip.key,
                "at": clip.start.isoformat(),
                "scene_hash": annotated_hash,
                "boxes": [list(b) for b in annotated_boxes],
            })
            state["recent_annotated"] = ring[-RECENT_SCENES:]
        # WHAT THE PRE-FILTER CAUGHT, counted where the screen can see it.
        #
        # NOT folded into `duplicates_skipped`, which counts the ETag path
        # only. Those are free -- caught from the listing, no download. These
        # are the ones that would otherwise cost a clip download and a decode
        # before anything noticed, so the two numbers answer different
        # questions and one number would answer neither.
        #
        # Either shape counts: the alarm the pre-filter SKIPPED carries the
        # stage, and the alarm it merely flagged while
        # `PREFILTER_SKIPS_EXTRACTION` is False carries the shadow verdict and
        # was judged anyway. The second is the size of what turning the switch
        # on would save.
        if result.get("duplicate_stage") == "pre_extraction" or result.get(
            "pre_extraction_duplicate_of"
        ):
            state["pre_extraction_duplicates"] = int(state.get("pre_extraction_duplicates", 0)) + 1
        state["results"] = (
            list(state.get("results", []))
            + [
                {
                    "clip_key": clip.key,
                    "alarm_id": result.get("alarm_id") or clip.alarm_id,
                    "frame_key": result.get("frame_key"),
                    "case_id": result.get("case_id"),
                    "verdict": result.get("verdict"),
                    "people": result.get("people", 0),
                    "judged": result.get("judged", 0),
                    "gated": result.get("gated", 0),
                    "failed": result.get("failed", 0),
                    "calls": calls,
                    # WHEN THIS ALARM FINISHED, not when the poll began.
                    # `now` is one reading taken once per poll, for the day
                    # boundary and the budget reset; a poll judges alarms
                    # until its budget runs out, so stamping every one with it
                    # gave a whole batch an identical time. Measured on the
                    # live feed: 26 results carrying two distinct timestamps,
                    # 14 alarms sharing one of them.
                    "at": datetime.now(UTC).isoformat(),
                    # And WHEN THE ALARM ITSELF WAS RAISED, which is the other
                    # question this column was being read as. It comes from
                    # the snapshot key and has nothing to do with when we got
                    # round to it.
                    "alarm_at": clip.start.isoformat(),
                }
            ]
        )[-RESULT_TAIL:]
        state["watermark"] = clip.key
        state.pop("paused_reason", None)
        state.pop("paused_detail", None)
        checkpoint_state()
    return state


#: How many recent scenes the near-duplicate check compares against. Bounded
#: because the state is a stored object rewritten on every checkpoint, and a
#: minute of alarms across every camera is well inside this.
RECENT_SCENES = 40

#: How many judged alarms the state carries. The state is a stored object
#: rewritten on every checkpoint, so the results list is a tail, not a log --
#: the durable record of every alarm is the audit ledger, and the audit queue
#: is where they are reviewed.
RESULT_TAIL = 200


def frame_verdict(counts: dict) -> str | None:
    """One alarm's verdict, OR'd over the people in it.

    ANY person in violation makes the frame a violation; a person the judge
    declined to answer about leaves the frame undecided rather than cleared;
    it is cleared only when everyone in it was cleared. TP > escalated > FP,
    the same order as `audit.case_outcome`, `audit.vlm_frame_verdicts` and
    `rolling_metrics._IMAGE_OUTCOMES` -- one question, and it must have one
    answer wherever it is asked.

    THE POPULATION IS DELIBERATELY UNFILTERED BY `audit.included_people`,
    unlike every other site that ORs -- and that carve-out is about WHO IS IN
    THE FRAME, not about the order above, which moves with the others. This
    reports what a run JUST DID, before any human has seen the frame, so an
    auditor's strike-out cannot exist yet: there is nothing to exclude, and
    reading the audit sidecar here would only add a fetch per alarm to the
    poll loop. The frame's audit case picks the exclusions up the moment one
    is recorded. Do not "fix" the population to match the others.
    """
    if counts.get("tp"):
        return "violation"
    if counts.get("escalated"):
        return "cannot_determine"
    if counts.get("fp"):
        return "no_violation"
    if counts.get("gated") or counts.get("no_person"):
        # Nothing was judged, and either every detection was empty or the
        # detector found nobody at all: no person in the frame, so no person
        # in violation. The same answer the gated crops'
        # own rows carry, so the feed's column and the audit case agree --
        # reported as "not judged" here while the case read "no violation"
        # would be one frame with two verdicts.
        return "no_violation"
    return None


def stop_requested(state: dict, key: str = DEFAULT_STATE_KEY) -> bool:
    """Whether the STORED state says this run should stop.

    The durable stop switch, and the reason there is one: a running feed
    lives on a thread inside one process, holding a lease it refreshes every
    poll, while `POST /live-feed/stop` used to reach only the session that
    started it. After a restart -- or from a second browser -- nobody could
    stop it and nobody could take it over: "A live feed lease is active in
    another worker", refreshed every 30 seconds, indefinitely.

    So the worker asks the store on every cycle. Any session, and any other
    worker, can stop a feed by writing `status` to something other than
    `running`; the thread notices within one poll and lets its lease lapse.

    A read failure returns False. Losing the store for a moment must not kill
    a run that is spending money and making progress.
    """
    try:
        stored = load_state(key)
    except Exception:  # noqa: BLE001 - a transient read must not stop a paid run
        return False
    if not stored:
        return False
    if stored.get("status") != "running":
        return True
    # Somebody else claimed the lease. Two workers on one feed would double
    # every call, so the loser stands down.
    return lease_taken(state, stored)


def request_stop(key: str = DEFAULT_STATE_KEY) -> dict:
    """Write the stop into the store, whoever is running it.

    Returns the stored state as it now reads. Reachable from any session,
    because a feed that cannot be stopped from the screen in front of you is
    a feed that has to be stopped by restarting the server.
    """

    def mutate(current):
        current = current if isinstance(current, dict) else {}
        if not current:
            return current
        return {**current, "status": "stopped"}

    value = storage.get_storage().update_json(key, mutate, default={})
    return value if isinstance(value, dict) else {}


def clear_failures(state: dict) -> dict:
    """Drop the failure records, the skip list and the attempt counts.

    NOT the watermark and NOT `processed_keys`. Those are the cursor and the
    idempotency record: clearing them would re-judge alarms already paid for.
    What goes is the REPORT -- entries an operator has read and cannot act on,
    which otherwise accumulate across runs and make every new run look broken.

    A skipped clip becomes eligible again, which is deliberate: clearing the
    record is also the way to say "try those once more" after fixing whatever
    made them fail.
    """
    state["failures"] = []
    state["skipped_keys"] = {}
    state["attempts"] = {}
    state.pop("paused_reason", None)
    state.pop("paused_detail", None)
    return state


def clear_stored_failures(key: str = DEFAULT_STATE_KEY) -> dict:
    """Clear the failure records IN THE STORE, for a session that does not own
    the run.

    A read-modify-write through `update_json`, the same shape as
    `request_stop`, rather than `clear_failures` over a loaded copy followed by
    `save_state`. The copy route is how a non-owner ended up holding a whole
    frozen snapshot of somebody else's run -- see `clear_live_feed_failures`.

    Returns the state as the store now reads, or `{}` if there is none.
    """

    def mutate(current):
        current = current if isinstance(current, dict) else {}
        if not current:
            return current
        return clear_failures({**current})

    value = storage.get_storage().update_json(key, mutate, default={})
    return value if isinstance(value, dict) else {}


def _duplicate_of(snapshot, fingerprint: str, boxes: list, recent: list[dict] | None) -> str | None:
    """The key of a recently judged alarm this one repeats, or `None`.

    Recent in TIME, not merely recent in the ring: a camera re-firing on one
    moment does it within seconds, and beyond a minute the same view is a new
    moment with the same people still standing there. Without the window a
    quiet camera would have every alarm all day collapsed into its first.
    """
    if not fingerprint or not recent:
        return None
    for seen in reversed(recent):
        gap = abs((snapshot.start - datetime.fromisoformat(seen["at"])).total_seconds())
        if gap > DUPLICATE_WINDOW_SECONDS:
            continue
        if scene_distance(fingerprint, seen.get("scene_hash", "")) > SCENE_HASH_DISTANCE:
            continue
        if same_boxes(boxes, [tuple(b) for b in seen.get("boxes", [])]):
            return str(seen.get("key") or "")
    return None


def process_live_snapshot(
    snapshot,
    call_budget: int,
    run_id: str,
    reporter: StageReporter | None = None,
    *,
    client=None,
    bucket: str = DEFAULT_BUCKET,
    audit_log_path: str = "eval/track1_audit_log.json",
    alarm_image_dir: str = "eval/alarm_images",
    should_stop: Callable[[], bool] | None = None,
    few_shot_library=None,
    recent: list[dict] | None = None,
    recent_annotated: list[dict] | None = None,
    box_source: str = "vendor_cv",
) -> dict:
    """One alarm. Judged on the CLEAN frame from its clip, always.

    THE SNAPSHOT IS NEVER JUDGED -- it has `violation` written on it, and
    handing the model a picture already labelled with the answer is the trap
    `HANDOFF_helmet_cv.md` §3 records.

    WHERE THE BOXES COME FROM is the choice a run makes:

    * `vendor_cv` reads the annotated still for the rectangles the vendor's
      own CV drew. Those are the boxes that RAISED the alarm, so judging them
      measures the vendor's claims directly.
    * `detector` runs `person_detector` over the clean frame instead. A
      different model answering the same question, and which is better is a
      measurement nobody has taken.

    EITHER WAY THE SAME ALARMS ARE JUDGED, because the feed walks `snapshot/`
    in both cases. The detector option used to walk `raw_video/`, so the two
    arms saw different populations and their numbers could not be compared at
    all. Only the source of the boxes differs now.
    """
    client = client or get_s3_client()
    config = get_config()
    using_detector = box_source == "detector"
    # POSITIONAL, and it has to stay positional: `poll_once` calls its
    # `processor` with four positional arguments, which every test double
    # written as `lambda *a` absorbs unchanged. As a keyword it would break
    # them all.
    reporter = reporter or NO_REPORTER

    # NAMED BEFORE THE FETCH, not after. This is a ~0.9MB GET plus rectangle
    # reconstruction and a scene hash over the decoded image, and until it had
    # a stage of its own `in_flight` carried a clip key and no answer at all to
    # what was being done with it.
    reporter.stage("reading_snapshot")
    # THE SNAPSHOT IS FETCHED ONCE, IN BOTH MODES, AND THE BYTES ARE REUSED.
    #
    # It used to be fetched only in vendor mode, on the reasoning that
    # detector mode reads nothing in it and it is ~0.9MB per alarm. That was
    # wrong in both directions: `extract_clip` -> `_locate_alarm_frame` GETs
    # the same object unconditionally, in both modes, because the still is the
    # KEY that names the clean alarm frame. So vendor mode paid for it TWICE
    # per alarm, and detector mode paid for it anyway, one layer down and too
    # late to answer anything with.
    #
    # Fetched here and threaded into `extract_clip`, it costs the same one GET
    # it always cost -- and its bytes are available before the clip download,
    # which is what lets the duplicate check below run before the expensive
    # part rather than after it.
    #
    # A FAILED FETCH IS FATAL IN VENDOR MODE, AND ONLY THERE.
    #
    # Vendor mode cannot proceed without this object: the boxes it judges are
    # drawn on it. It has always raised from exactly here, and that is what
    # must not change -- `poll_once` records the failure with the S3 error
    # verbatim (`NoSuchKey`, `AccessDenied`, `SlowDown`), holds the cursor and
    # retries the alarm, having spent nothing.
    #
    # Swallowing it instead was a real regression, not a cosmetic one. An
    # empty still recovers NO boxes, and no boxes is not an error downstream:
    # `records_from_boxes` makes one whole-frame candidate, the judge answers
    # it (the 1920x1080 case `snapshot_boxes` measured at "helmet_worn" 6
    # times in 7), the alarm is recorded `judged`, and `processed_keys` means
    # it is never looked at again. A network blip would silently clear an
    # alarm the vendor's CV had flagged.
    #
    # Detector mode is the opposite case: it never fetched this object here
    # at all, so there is no failure of this fetch to preserve. A still it
    # cannot read costs it the alarm frame, one layer down, where
    # `_locate_alarm_frame` has always swallowed the same error and named no
    # frame -- and that failure still happens, unchanged, below.
    try:
        annotated = client.get_object(Bucket=bucket, Key=snapshot.key)["Body"].read()
    except Exception:  # noqa: BLE001 - re-raised below unless detector mode
        if not using_detector:
            raise
        annotated = b""
    # THE STILL'S OWN BOXES AND THE STILL'S OWN SCENE, in BOTH modes.
    #
    # In vendor mode these ARE `boxes` and `fingerprint`: the same numbers
    # under a second name. That waste is deliberate -- one pre-filter that
    # reads the same way whichever mode the run is in beats two that have to
    # be compared to be trusted, and the cost is one hash of an image already
    # in memory.
    #
    # In detector mode they are something the run has nothing else like: a
    # description of this alarm available BEFORE the clip is downloaded.
    # `boxes` and `fingerprint` stay empty here and are filled from the clean
    # frame after extraction, because that is the image the judge is shown.
    #
    # Named `annotated_*` and not `snapshot_*`: `snapshot_boxes` is a module
    # in this package, and a local by that name would read as one.
    #
    # GUARDED ON EMPTY, which only detector mode can reach: `cv2.imdecode` of
    # nothing raises an assertion that `extract_boxes` logs at ERROR with a
    # traceback, and a still-less alarm is an expected, documented condition
    # -- roughly 14% of this corpus -- not something to put thousands of
    # tracebacks in the log for.
    annotated_boxes = extract_boxes(annotated) if annotated else []
    annotated_hash = scene_hash(annotated) if annotated else ""
    boxes = [] if using_detector else annotated_boxes
    fingerprint = "" if using_detector else annotated_hash

    # THE SAME MOMENT, ALARMED AGAIN -- ASKED BEFORE ANYTHING IS DOWNLOADED.
    #
    # Byte-identical re-alarms are already gone by here, caught on the ETag
    # from the listing. This is the other half: the same camera firing again
    # seconds later, where the view is the same but the JPEG is not.
    #
    # BOTH tests, never just the scene. Measured over 40 close pairs, 17 were
    # the same scene and 13 of those had boxes that had MOVED (2->1, 1->5,
    # 0->2), which is a different set of people flagged and so a real second
    # observation. Only 4 were the same view AND the same boxes.
    #
    # ASKED HERE, against the snapshot-space ring, because everything below is
    # what a duplicate costs: a clip listing, a clip download, an ffmpeg
    # decode, and in detector mode a detector pass -- minutes, to re-answer a
    # question already answered. In detector mode this used to be structurally
    # impossible: `annotated` was empty, so the fingerprint was empty and this
    # check returned None for every alarm without ever comparing anything.
    reporter.stage("checking_duplicate")
    pre_duplicate_of = _duplicate_of(
        snapshot, annotated_hash, annotated_boxes, recent_annotated
    )
    # CARRIED ON THE RESULT ONLY WHERE IT SAYS SOMETHING NEW.
    #
    # In vendor mode `annotated_hash` and `annotated_boxes` are byte-identical
    # copies of `scene_hash` and `boxes`, in the same dict, and that dict is
    # stored per alarm in `processed_keys` -- which is unbounded, carried
    # across runs, rewritten through `update_json` on every checkpoint and
    # returned whole to the browser every two seconds. Measured: 661 -> 977
    # bytes an entry, ~+9.9MB over the 31k-alarm corpus, to say twice what it
    # already said once. `poll_once` reads `scene_hash`/`boxes` when these are
    # absent, which is what vendor mode's snapshot-space ring is built from.
    #
    # `pre_extraction_duplicate_of` is the shadow measurement, and it is only
    # a measurement in detector mode: in vendor mode the pre-filter DECIDES,
    # so a result that exists at all is one it let through, and the field
    # would be None on every alarm forever.
    snapshot_space = {
        "annotated_hash": annotated_hash,
        "annotated_boxes": [list(b) for b in annotated_boxes],
    } if using_detector else {}
    shadow = {"pre_extraction_duplicate_of": pre_duplicate_of} if using_detector else {}
    # WHETHER THAT VERDICT IS ACTED ON.
    #
    # In vendor mode, yes, always: the judge is shown crops of the vendor's
    # boxes, so snapshot space IS the space this run judges in, and this is
    # the check that has always run here.
    #
    # In detector mode it is the shadow measurement described on
    # `PREFILTER_SKIPS_EXTRACTION`: the verdict is recorded and the alarm is
    # judged anyway, so the clean-frame check below can be compared against it.
    #
    # `new_state` writes this same condition onto the run as
    # `pre_extraction_acts`, which is what the screen reads. If one of them
    # moves, the other moves with it.
    if pre_duplicate_of is not None and (PREFILTER_SKIPS_EXTRACTION or not using_detector):
        return {
            "status": "duplicate_event",
            "calls": 0,
            "alarm_id": snapshot.alarm_id,
            "scene_hash": fingerprint,
            "boxes_recovered": len(annotated_boxes),
            "duplicate_of": pre_duplicate_of,
            # WHICH CHECK CAUGHT IT. Also what `poll_once` counts on: with the
            # verdict acted on there is nothing to shadow, so `duplicate_of`
            # and the stage say all of it.
            "duplicate_stage": "pre_extraction",
            **snapshot_space,
        }

    reporter.stage("finding_clip")
    clip_key = clip_key_for(client, snapshot, bucket=bucket)
    if clip_key is None:
        raise RuntimeError(f"no clip in {bucket} for snapshot {snapshot.key}")
    clip = parse_clip_key(clip_key)
    if clip is None:
        raise RuntimeError(f"could not read the clip key {clip_key!r}")

    with tempfile.TemporaryDirectory(prefix="ppe-live-") as raw_dir:
        root = Path(raw_dir)
        # NOT SPLIT into download / decode / locate. Measured on the deployed
        # feed, 1 Sep: the clip download and the detector together were 5.1s
        # against roughly four minutes of judging on a nine-crop alarm. The
        # minutes are in judging, so finer stages here would cost a change for
        # a part of the wall clock nobody is waiting on.
        reporter.stage("extracting")
        extracted = extract_clip(
            clip, client=client, bucket=bucket, out_dir=root, annotated=annotated
        )
        if extracted.status != "ok" or extracted.alarm_frame_index is None:
            raise RuntimeError(extracted.error or "clean alarm frame could not be located")
        frame = next(row for row in extracted.frames if row.get("frame_role") == FRAME_ROLE_ALARM)

        if using_detector:
            # ON THE CLEAN FRAME, never the snapshot: a detector run over a
            # picture with red rectangles drawn on it is being handed the
            # answer. The dedupe fingerprint comes from here too, for the
            # same reason it exists at all -- it has to describe the image
            # the judge is about to be shown.
            detector = get_person_detector(config)
            if detector is None:
                raise RuntimeError("person detector weights are unavailable")
            clean = (root / frame["relative_path"]).read_bytes()
            reporter.stage("detecting_people")
            boxes = detector.find(clean)
            fingerprint = scene_hash(clean)
            duplicate_of = _duplicate_of(snapshot, fingerprint, boxes, recent)
            if duplicate_of is not None:
                return {
                    "status": "duplicate_event",
                    "calls": 0,
                    "alarm_id": snapshot.alarm_id,
                    "scene_hash": fingerprint,
                    "boxes_recovered": len(boxes),
                    "duplicate_of": duplicate_of,
                    # WHICH CHECK CAUGHT IT, never conflated with the other.
                    # This one cost the download, the decode and the detector
                    # pass; the pre-extraction one costs nothing. Read as one
                    # number they would say nothing about either.
                    "duplicate_stage": "post_extraction",
                    **snapshot_space,
                    **shadow,
                }

        provenance = CvProvenance.from_config(config, cv_run_id=run_id)
        candidates = [
            record.as_dict()
            for record in records_from_boxes(
                frame, boxes, provenance=provenance, box_source=box_source
            )
        ]
        # ONE CALL PER CROP. It was two while the first-pass person gate
        # existed, and the cap had to cover the worst case; the gate is gone
        # (2 Sep) and a crop costs exactly one judge call. Whole frames only,
        # as everywhere else: half a frame's people judged is a frame verdict
        # that looks complete and is not.
        needed = len(candidates)
        if needed > call_budget:
            return {
                "status": "not_judged_budget",
                "calls": 0,
                "candidates": len(candidates),
                "needed": needed,
            }
        # THE PART THE MINUTES ARE IN, so it reports its own progress: one
        # judge call per crop at ~25s each, four at a time.
        reporter.stage("judging", crops_total=needed, crops_judged=0)
        counts = judge_candidates(
            candidates,
            frames_dir=root,
            on_crop_judged=reporter.crop_judged,
            config=config,
            few_shot_library=few_shot_library,
            audit_log_path=audit_log_path,
            alarm_image_dir=alarm_image_dir,
            max_workers=1,
            should_stop=should_stop,
            lineage={
                "source": "live_cv",
                "run_id": run_id,
                "feed_key": snapshot.key,
                "watermark": snapshot.key,
                "frame_key": frame["frame_name"],
                "alarm_id": snapshot.alarm_id,
                "box_source": box_source,
                # THE IMAGE'S OWN IDENTITY, carried onto every row of it. The
                # alarm id is not it: the vendor re-alarms on one stale frame
                # with a fresh id, and 1,402 duplicate groups straddled train
                # and val when the split was keyed on the id.
                "snapshot_etag": snapshot.etag,
            },
        )
        if counts.get("stopped"):
            status = "stopped"
        elif counts["failed"]:
            status = "failed"
        else:
            status = "judged"
        calls = counts["judged"] + counts.get("gate_calls", 0)
        return {
            "status": status,
            "calls": calls,
            "alarm_id": snapshot.alarm_id,
            "frame_key": frame["frame_name"],
            "case_id": frame_stem(frame["frame_name"]),
            "people": len(candidates),
            "verdict": frame_verdict(counts),
            "boxes_recovered": len(boxes),
            # Carried out so `poll_once` can keep its rings without
            # re-reading or re-hashing either image. `scene_hash`/`boxes`
            # describe the image the judge was shown; `snapshot_space`, in
            # detector mode only, describes the vendor's still that named the
            # alarm. Two different pictures of one moment there, and
            # collapsing them would make the pre-filter compare across spaces.
            "scene_hash": fingerprint,
            "boxes": [list(b) for b in boxes],
            **snapshot_space,
            # WHAT THE PRE-FILTER WOULD HAVE DONE, in detector mode. This
            # alarm was judged despite it, on purpose -- see
            # `PREFILTER_SKIPS_EXTRACTION` -- and it is recorded so the two
            # answers can be read against each other afterwards.
            **shadow,
            **counts,
        }


def lease_taken(state: dict, stored: dict) -> bool:
    """Whether the STORE says a different worker owns this feed now.

    THE ONE RULE, asked from the two places that must not disagree:
    `stop_requested`, where the loser stands down, and `save_state`, where it
    refuses to write. Written twice they would drift, and the drift is a run
    that keeps spending while believing it stood down.

    A state with no `lease_owner` never lost anything -- that is most tests,
    and `clear_failures` on a feed nobody claimed.
    """
    owner = state.get("lease_owner")
    held = stored.get("lease_owner")
    return bool(owner and held and held != owner)


def save_state(state: dict, key: str = DEFAULT_STATE_KEY) -> dict:
    """Store the state, UNLESS another worker owns the feed now.

    THE WRITE IS A BLIND WHOLE-OBJECT OVERWRITE, and that is why this guard
    has to exist. `update_json` gives compare-and-swap against concurrent
    WRITERS, not against a lease: `lambda _current: state` ignores whatever it
    read, so a worker whose lease had already been claimed by somebody else
    would put its own `lease_owner` back on every checkpoint. Measured: the
    rival claims the lease, the store agrees it owns it, and the old worker's
    very next checkpoint takes it back -- after which `stop_requested` tells
    the RIVAL to stand down and the zombie keeps the run. Refreshing the lease
    on every checkpoint is what turned "whoever claimed it wins" into "whoever
    checkpointed last wins".

    So the write asks, inside the swap where the read is already paid for,
    whether this worker still owns what it is about to overwrite. If it does
    not, the store is left exactly as the new owner wrote it and this state is
    marked stopped -- `poll_once` breaks out of its loop on that, and the
    worker's next cycle ends.

    THE LOSER'S LAST ALARM IS NOT RECORDED HERE, and that is the intended
    trade. Its judged rows are already in the audit ledger under their own
    key; what is lost is this feed's cursor entry for it, which the winner
    will re-derive. The alternative is overwriting the winner's whole state,
    which is the bug above.
    """
    taken = False

    def write(current):
        nonlocal taken
        current = current if isinstance(current, dict) else {}
        if lease_taken(state, current):
            taken = True
            return current
        return state

    storage.get_storage().update_json(key, write, default={})
    if taken:
        state["status"] = "stopped"
        # ON-SCREEN COPY, so terse and with no em-dash: what happened, and
        # that it was not this operator's doing.
        state["stopped_reason"] = "Another worker took over this feed."
    return state


def claim_state(state: dict, key: str = DEFAULT_STATE_KEY, *, now: datetime | None = None) -> bool:
    now = now or datetime.now(UTC)
    owner = uuid.uuid4().hex
    state["lease_owner"] = owner
    state["lease_expires_at"] = (now + timedelta(seconds=LEASE_SECONDS)).isoformat()

    def claim(current):
        current = current if isinstance(current, dict) else {}
        expires = current.get("lease_expires_at")
        active = False
        if current.get("status") == "running" and expires:
            try:
                active = datetime.fromisoformat(expires) > now
            except ValueError:
                active = False
        return current if active else state

    value = storage.get_storage().update_json(key, claim, default={})
    return isinstance(value, dict) and value.get("lease_owner") == owner


def refresh_lease(state: dict, *, now: datetime | None = None) -> None:
    now = now or datetime.now(UTC)
    state["lease_expires_at"] = (now + timedelta(seconds=LEASE_SECONDS)).isoformat()


def load_state(key: str = DEFAULT_STATE_KEY) -> dict:
    value, _ = storage.get_storage().read_json(key, default={})
    return value if isinstance(value, dict) else {}
