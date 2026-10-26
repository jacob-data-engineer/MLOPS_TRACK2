"""Queue a retrain once a pull has finished, without starting it.

The unattended middle ground. Starting a run outright commits GPU hours
with nobody watching; doing nothing means somebody has to notice the pull
landed and rebuild the dataset by hand, which is the step that gets
skipped. Queueing does the whole chain -- rebuild the lists, rebuild the
yaml, freeze it -- and then STOPS, leaving an entry a person releases by
pressing Start Queue.

So the expensive, irreversible part stays a human decision and the
fiddly, forgettable part does not.

WHY THIS LIVES IN THE APP AND NOT THE PULL LAMBDA. The queue is a list in
this process (see track2_shared's `_training_queue`), advanced by
`queue_poll_loop`. A Lambda appending to it would be writing to its own
copy. What it can do instead is watch: every sweep writes an `event` row
to DynamoDB, so the app sees a pull that ran in the Lambda exactly as it
sees one that ran here -- and a single watcher covers both triggers
rather than one path per trigger.
"""

from __future__ import annotations

import datetime as _dt

from . import pull_control as pc

#: Where the marker lives. In the config beside everything else, written
#: through `write_config`, which merges -- so recording a run id cannot
#: clobber a setting the card saved a moment earlier.
KEY = "queue_after_pull"

DEFAULTS = {
    #: Off. An option that queues GPU work should be asked for, not
    #: arrived at by upgrading.
    "enabled": False,
    #: Queue after a sweep that moved nothing. Off by default: a pull
    #: that found no files leaves the corpus exactly as the last run
    #: already trained on, so the entry would be a duplicate of a model
    #: that exists.
    "even_if_nothing_moved": False,
    #: The last sweep acted on. The whole of the de-duplication: a run id
    #: is written only after the entry is queued, so a crash in between
    #: retries rather than silently skipping.
    "last_run_id": "",
    #: What happened last time, for the card to show. A background action
    #: with no visible outcome is one nobody trusts.
    "last_result": {},
    #: Which pull triggers may chain. `schedule` only, by default, and
    #: that default is deliberate: it is exactly what this did before the
    #: list existed, so turning the list on cannot change what an
    #: existing settings file does.
    #:
    #: `arrival` is now selectable, but read the guards below before
    #: ticking it. An arrival pull fires per upload burst, and on
    #: 2026-10-06 three fired in ten minutes -- one of them moving a
    #: single file -- each of which would be a GPU training run.
    #: `min_files` and `cooldown_minutes` are what make it safe, which is
    #: why they are here rather than being left to be remembered.
    #:
    #: `manual` is offered too, for a person who wants Pull now to carry
    #: straight through. It is off by default for the same reason.
    "triggers": ["schedule"],
    #: How many files a sweep must have moved to be worth a run. 1 keeps
    #: the old behaviour (any movement counts). Raise it for `arrival`:
    #: it is the difference between "somebody uploaded a photo" and "a
    #: batch landed".
    "min_files": 1,
    #: Shortest gap between two chains, in minutes. 0 keeps the old
    #: behaviour. The second guard for `arrival`, and the one that caps
    #: the spend: a burst of twenty uploads cannot become twenty runs,
    #: whatever the file counts say.
    "cooldown_minutes": 0,
}


def allowed_triggers(plan: dict) -> list:
    """The triggers this configuration lets chain, always as a list.

    Tolerates a string, and an empty value means the documented default
    rather than "nothing": a file written before this existed has no
    `triggers` key at all, and such a file must keep chaining on the
    schedule exactly as it did.
    """
    raw = plan.get("triggers")
    if isinstance(raw, str):
        raw = [raw]
    cleaned = [str(t).strip().lower() for t in (raw or []) if str(t).strip()]
    return cleaned or ["schedule"]


def settings(config: dict | None = None) -> dict:
    """The chain's settings -- with `enabled` and `triggers` DERIVED from
    the pull's own mode, not stored (2026-10-26).

    The PULL, BUILD, QUEUE, TRAIN strip is the one switch: "On a
    schedule" chains every scheduled pull, "When files arrive" every
    arrival pull (behind `min_files` and `cooldown_minutes`), "Off"
    nothing. The separate "Run after every scheduled pull" tick, its
    trigger boxes and step 6's "Enabled for scheduling" were three more
    answers to the same question, and the strip could read "On a
    schedule" while all of them said off. Stored values for these two
    keys are ignored; the guards and `even_if_nothing_moved` still are
    settings.
    """
    config = config if config is not None else pc.read_config()
    plan = {**DEFAULTS, **(config.get(KEY) or {})}
    mode = str(config.get("mode") or "off")
    plan["enabled"] = mode in ("schedule", "arrival")
    plan["triggers"] = [mode] if plan["enabled"] else ["schedule"]
    return plan


def _finished_pulls(limit: int = 10, plan: dict | None = None) -> list:
    """Completed sweeps, newest first.

    Reads the `run_state` rows, not the `event` rows. Both exist; they
    do not cover the same sweeps:

      * `_record_run` writes an `event` row, and DELIBERATELY SKIPS a
        scheduled sweep that moved nothing -- "a wall of empty runs" in
        the history. So the event rows cannot see the one case
        `even_if_nothing_moved` exists for, and that option could never
        take effect for the scheduled trigger.
      * `write_run_state` writes a `run_state` row for EVERY sweep,
        unconditionally, carrying `status` and `finished_at`.

    So this is both the honest completion signal and the only source
    that can answer "the pull ran and moved nothing".

    A sweep in flight has no `finished_at`; one that was stopped or
    failed says so in `status`. Neither should trigger a retrain -- the
    first because the week folders are still filling, the second because
    what landed is by definition not what was meant to.
    """
    allowed = allowed_triggers(plan if plan is not None else settings())
    out = []
    for run in pc.recent_run_states(limit):
        if not run.get("finished_at"):
            continue
        if run.get("aborted") or run.get("status") in ("stopped", "failed"):
            continue
        # Which triggers may chain is now a setting, defaulting to
        # `schedule` -- which is what this line used to hardcode.
        #
        # It was hardcoded because of a real incident: on 2026-10-06 an
        # ARRIVAL pull that moved ONE file started a full GPU training
        # run, and three arrival pulls fired in ten minutes that morning.
        # That is still the risk. What has changed is that `min_files`
        # and `cooldown_minutes` below can now express "a batch landed"
        # rather than "a file landed", so arrival can be ticked
        # deliberately instead of being refused outright.
        if str(run.get("trigger") or "") not in allowed:
            continue
        out.append(run)
    return out


def handled(run_id: str) -> bool:
    """Has this sweep already been chained?

    Reads the CLAIM ROW, not `last_run_id` on the config. The config is
    one JSON blob written read-modify-write with no condition, so every
    writer races every other and the loser's key silently reverts. That
    is not a theory: on 2026-10-06 `last_run_id` sat at the 09:27 sweep
    while four later sweeps had finished and been acted on, so `pending`
    answered "act" forever and the only thing preventing a chain per tick
    was the claim row.

    A claim is its own item with its own key, written conditionally. It
    cannot be clobbered by a writer that happens to hold a stale copy of
    the config, which is exactly the property the marker needed and the
    blob could never give it.

    `last_run_id` is still written, for the card to show. It is a
    DISPLAY value now, not the gate.
    """
    return bool(run_id) and pc.claim_exists("queue_after_pull", run_id)


def pending(config: dict | None = None) -> dict:
    """The sweep that would be acted on, and why it would or would not be.

    Separated from `check` so the card can say what is about to happen
    without doing it, and so a test can assert the decision without
    queueing anything.
    """
    config = config if config is not None else pc.read_config()
    plan = settings(config)
    if not plan["enabled"]:
        return {"act": False, "reason": "Not switched on.", "run": None}

    # The PULL is not the trigger when "Use these folders only" is on.
    #
    # That switch turns Pull Control, Auto Training and Retention off:
    # no sweep runs, every pull reports `moved: 0`, and the three guards
    # here -- `min_files`, `cooldown_minutes`, `even_if_nothing_moved` --
    # all exist to pace training against upload bursts that cannot
    # happen. `retrain_schedule.check_training_schedule` owns the clock
    # in that mode, so letting a no-op pull chain as well would be two
    # triggers for one GPU slot.
    from . import retrain_schedule as rs

    if not rs.follows_pull(config):
        return {"act": False, "run": None,
                "reason": "Training data is built by hand, so the pull is not "
                          "the trigger -- the retrain schedule's own clock is."}

    runs = _finished_pulls(plan=plan)
    if not runs:
        allowed = ", ".join(allowed_triggers(plan))
        return {"act": False,
                "reason": f"No completed pull yet from a trigger that may "
                          f"chain ({allowed}).",
                "run": None}

    newest = runs[0]
    run_id = str(newest.get("run_id") or "")
    if handled(run_id) or (run_id and run_id == plan["last_run_id"]):
        return {"act": False, "reason": "Already queued for this pull.", "run": newest}

    moved = int(newest.get("moved") or 0)
    if not moved and not plan["even_if_nothing_moved"]:
        return {
            "act": False,
            "reason": "The last pull moved no files, so the corpus is "
                      "unchanged and a run would repeat the last one.",
            "run": newest,
        }

    # The batch guard. `moved` of 1 is a person uploading a photo;
    # `min_files` is how this configuration says what a batch is. Skipped
    # when the sweep moved nothing and `even_if_nothing_moved` is on,
    # because that case is deliberately about the empty sweep.
    minimum = max(1, int(plan.get("min_files") or 1))
    if moved and moved < minimum:
        return {
            "act": False,
            "reason": f"The last pull moved {moved} file"
                      f"{'' if moved == 1 else 's'}, below the {minimum} "
                      f"this configuration treats as a batch worth training on.",
            "run": newest,
        }

    # The spend cap. Without it a burst of uploads is a burst of GPU
    # runs, each one passing every check above on its own merits.
    cooldown = max(0, int(plan.get("cooldown_minutes") or 0))
    if cooldown:
        waited = _minutes_since_last_chain(plan)
        if waited is not None and waited < cooldown:
            return {
                "act": False,
                "reason": f"Last chained {int(waited)} minute"
                          f"{'' if int(waited) == 1 else 's'} ago; this "
                          f"configuration waits {cooldown} between runs.",
                "run": newest,
            }
    return {"act": True, "reason": "", "run": newest}


def _minutes_since_last_chain(plan: dict):
    """Minutes since the last chain, or None if there has not been one.

    Read off `last_result.at`, which is written after every attempt,
    successful or not. Deliberately not off the claim rows: a failed
    chain still cost the preflight and still means "something just
    fired", which is what a cooldown is about.
    """
    stamp = str((plan.get("last_result") or {}).get("at") or "")
    if not stamp:
        return None
    try:
        then = _dt.datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    if not then.tzinfo:
        then = then.replace(tzinfo=_dt.timezone.utc)
    delta = _dt.datetime.now(_dt.timezone.utc) - then
    return max(0.0, delta.total_seconds() / 60.0)


def check(config: dict | None = None) -> dict:
    """Called on the app's poll tick. Queues at most one run.

    Every path returns rather than raises: this runs inside the loop that
    also advances the queue and reconciles audits, and an exception here
    must not stop those.
    """
    from . import retrain_schedule as rs

    config = config if config is not None else pc.read_config()
    decision = pending(config)
    if not decision["act"]:
        return {"queued": False, "reason": decision["reason"]}

    run = decision["run"] or {}
    run_id = str(run.get("run_id") or "")

    # One container only. Dev and stable share this table and run the
    # same tick, so both can read "not handled yet" before either writes
    # the marker -- and what follows is a GPU training run. The claim is
    # a conditional put, so exactly one caller wins.
    if run_id and not pc.claim_once("queue_after_pull", run_id):
        return {"queued": False, "reason": "Another instance is handling this pull."}

    try:
        outcome = rs.run_chain(
            # From the TICKED SOURCES, always: the week folders, the
            # golden set and the Assemble folders, each when its card is
            # ticked. A frozen snapshot picked in step 6 is for runs
            # queued by hand; an automatic run after a pull exists to
            # train on what the pull just brought in.
            config=rs.automatic(config),
            trigger="after-pull",
            submit="queue",
        )
    except Exception as exc:                                   # noqa: BLE001
        outcome = {"ok": False, "steps": [{"step": "chain", "ok": False,
                                           "detail": str(exc)}]}

    # The marker is written EITHER WAY. A chain that failed its preflight
    # will fail it again on the next tick 25 seconds later, and the tick
    # after that -- so not recording it turns one misconfiguration into a
    # log line every 25 seconds until somebody notices.
    result = {
        "run_id": run_id,
        "at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "ok": bool(outcome.get("ok")),
        "detail": _first_problem(outcome),
        "moved": int(run.get("moved") or 0),
    }
    # Best effort, and deliberately so: the claim row above is what
    # actually prevents a second chain. This is for the card to show, and
    # a blob write that loses a race must not turn into a repeated run.
    try:
        pc.write_config({KEY: {**settings(pc.read_config()), "last_run_id": run_id,
                               "last_result": result}})
    except Exception as exc:                                   # noqa: BLE001
        print(f"Could not record the queue-after-pull result: {exc}")
    return {"queued": bool(outcome.get("ok")), "result": result,
            "steps": outcome.get("steps", [])}


def _first_problem(outcome: dict) -> str:
    """The step that stopped it, or a short success line."""
    for step in outcome.get("steps") or []:
        if not step.get("ok"):
            return f"{step.get('step', 'step')}: {step.get('detail', '')}"[:400]
    return "Queued." if outcome.get("ok") else "Nothing happened."
