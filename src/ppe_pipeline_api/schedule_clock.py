"""Pure schedule cadence shared by data sensors and training."""
import datetime as _dt
SGT = _dt.timezone(_dt.timedelta(hours=8))
LEGACY_EPOCH = _dt.date(1970, 1, 1)
def _now():
    return _dt.datetime.now(_dt.timezone.utc)

def _valid_time(text: str) -> bool:
    parts = str(text or "").split(":")
    if len(parts) != 2:
        return False
    try:
        hour, minute = int(parts[0]), int(parts[1])
    except ValueError:
        return False
    return 0 <= hour <= 23 and 0 <= minute <= 59


def _grid_start(plan: dict) -> _dt.datetime | None:
    """Slot zero of the grid, in SGT, or None when `at` is unusable.

    `from_date` (the operator's own "from this date") wins, then the
    stored `anchor`, then `LEGACY_EPOCH`. Never the current time: a grid
    counted from "now" moves with the clock, which is what let a weekly
    schedule fall due daily.
    """
    if not _valid_time(str(plan.get("at") or "")):
        return None
    hour, minute = (int(part) for part in str(plan["at"]).split(":"))
    day = LEGACY_EPOCH
    for key in ("from_date", "anchor"):
        try:
            day = _dt.date.fromisoformat(str(plan.get(key) or "").strip()[:10])
            break
        except ValueError:
            continue
    return _dt.datetime(day.year, day.month, day.day, hour, minute, tzinfo=SGT)


def fresh_anchor(plan: dict, *, now: _dt.datetime | None = None) -> str:
    """The anchor for a schedule being switched on or re-timed NOW.

    The date of the first `at` strictly after `now` -- today if the time
    has not come yet, else tomorrow -- so arming a schedule can never
    make a slot that is already due.
    """
    if not _valid_time(str(plan.get("at") or "")):
        return ""
    hour, minute = (int(part) for part in str(plan["at"]).split(":"))
    moment = (now or _now()).astimezone(SGT)
    first = moment.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if first <= moment:
        first += _dt.timedelta(days=1)
    return first.date().isoformat()


def with_anchor(before: dict, after: dict, *,
                now: _dt.datetime | None = None) -> dict:
    """`after`, with the anchor it should be saved with.

    Re-anchored (to `fresh_anchor`) only on a DELIBERATE reset: the
    schedule being switched on, or its interval or time changing. Any
    other save keeps the grid -- and a schedule with no anchor yet pins
    the one it was already showing (the `LEGACY_EPOCH` grid's next date),
    so upgrading moves no run. A disabled schedule is left as it is.
    """
    out = dict(after)
    if not out.get("enabled"):
        return out

    def every(plan):
        try:
            return max(1, int(plan.get("every_days") or 1))
        except (TypeError, ValueError):
            return 1

    reset = (not before.get("enabled")
             or every(before) != every(out)
             or str(before.get("at") or "") != str(out.get("at") or ""))
    if reset:
        out["anchor"] = fresh_anchor(out, now=now)
    elif not str(out.get("anchor") or "").strip():
        upcoming = next_runs({**out, "from_date": ""}, start=now, count=1)
        out["anchor"] = upcoming[0].astimezone(SGT).date().isoformat() if upcoming else ""
    return out


def next_runs(plan: dict, *, start: _dt.datetime | None = None,
              count: int = 4) -> list:
    """When the next few would fire, in UTC.

    Shown in the card because "every 7 days at 02:00" is not something
    anybody can turn into a date in their head, and the first question
    asked of a schedule is always "so when does it actually run".
    """
    # Singapore time, which is what the field says and what the EC2
    # schedule this system already runs on uses.
    #
    # ONE grid, fixed by `_grid_start` (from_date, else anchor, else the
    # legacy epoch) and never by `start`: this used to count from
    # whatever moment it was handed when no from_date was set, and
    # `_passed_slots` handed it "now minus an interval", so the grid slid
    # with the clock and a weekly schedule fell due every day (F2).
    grid = _grid_start(plan)
    if grid is None:
        return []
    every = max(1, int(plan.get("every_days") or 1))
    step = _dt.timedelta(days=every)
    moment = (start or _now()).astimezone(SGT)
    first = grid
    if first <= moment:
        first += step * (((moment - first) // step) + 1)
    return [(first + step * n).astimezone(_dt.timezone.utc)
            for n in range(max(0, int(count)))]
