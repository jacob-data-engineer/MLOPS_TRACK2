"""The app's clock: every 25 seconds, everything that has to happen
without anybody pressing a button.

Moved out of `training/track2_shared.py` with the data/training split
(2026-10-28). The tick drives BOTH sides -- it advances the training
queue, runs the chain (pull -> dataset version -> queue), drains the pull
queue and runs the auto export -- so it belongs above them, here, rather
than inside one of them. The queue Lambda (`queue_runner.py`, beside this)
is the same clock for when the app is off.

Started from `main.lifespan` unless the container runs with
APP_BACKGROUND=off.
"""

from __future__ import annotations

import asyncio

from .. import training


async def queue_poll_loop() -> None:
    """Runs for the life of the process as an asyncio background task
    (started from main.py's lifespan). Uses asyncio.sleep (not time.sleep)
    since it shares the event loop with every request handler."""
    # Restored here rather than at import: this runs once the app is up, so
    # a DynamoDB outage delays the queue instead of preventing the process
    # from starting at all.
    try:
        training.restore_queue()
    except Exception as exc:
        print(f"Queue restore failed: {exc}")
    while True:
        await asyncio.sleep(25)
        try:
            # OFF the event loop. `try_advance_queue` is synchronous boto3
            # -- SageMaker, DynamoDB -- and calling it inline stalled every
            # request handler in this single-worker process for the length
            # of a submission. The `asyncio.sleep` above was chosen to
            # avoid exactly that, and then the work underneath it blocked
            # the loop anyway.
            await asyncio.to_thread(training.try_advance_queue)
        except Exception as exc:
            print(f"Queue poll failed: {exc}")
        # Same tick rather than its own loop: both want to know when a run
        # changed state, and one poll is one set of SageMaker calls.
        try:
            await asyncio.to_thread(training.reconcile_training_audits)
        except Exception as exc:
            print(f"Training audit reconcile failed: {exc}")
        # Queue a retrain if a pull finished since the last tick. On this
        # loop rather than its own, because it is one DynamoDB query and
        # it only ever does anything on the tick after a sweep ends.
        #
        # Watched rather than called by the pull: a SCHEDULED pull runs in
        # a Lambda, whose process has no access to this queue, so a hook
        # there would have worked for manual pulls only -- the ones least
        # in need of it.
        try:
            from . import queue_after_pull
            await asyncio.to_thread(queue_after_pull.check)
        except Exception as exc:
            print(f"Queue-after-pull check failed: {exc}")
        # Rebuild the Assemble card's dataset after a scheduled pull when
        # the card says "Scheduled" -- Data Preparation's own clock, no
        # training (2026-10-29). Here and not in the queue-runner Lambda,
        # which may only read S3.
        try:
            from . import assemble_after_pull
            built = await asyncio.to_thread(assemble_after_pull.check)
            if built.get("run_id"):
                print(f"Scheduled assemble after {built['run_id']}: "
                      f"{'built ' + built.get('folder', '') if built.get('built') else built.get('detail')}")
        except Exception as exc:
            print(f"Scheduled assemble check failed: {exc}")
        # ...and start a retrain if its OWN clock says one is due.
        #
        # On this tick rather than a schedule of its own in AWS: the
        # tick already runs, and `claim_once` inside the check is what
        # stops this container, the stable one and the two-minute Lambda
        # starting three runs for one slot. A pull is not involved at
        # all, which is the point -- "Use these folders only" switches
        # Pull Control off, and a schedule hung off a disabled mechanism
        # is the bug this replaces.
        try:
            from . import retrain_schedule
            await asyncio.to_thread(retrain_schedule.check_training_schedule)
        except Exception as exc:
            print(f"Training schedule check failed: {exc}")
        # Drain the PULL queue on the same tick.
        #
        # The router Lambda drains it too, on its own schedule, so a
        # queued sweep moves whether or not this container is up. Both
        # callers go through `advance_pull_queue`, which refuses to start
        # a second sweep while one is running -- so the two tickers
        # cannot race each other into the overlap this queue exists to
        # prevent.
        try:
            from .. import pull_control as _pc
            outcome = await asyncio.to_thread(_pc.advance_pull_queue, trigger="queue")
            if outcome.get("started"):
                print(f"Pull queue: started {outcome.get('run_id')} "
                      f"({outcome.get('remaining', 0)} still queued).")
        except Exception as exc:                                   # noqa: BLE001
            print(f"Pull queue tick failed: {exc}")
        # Export the best checkpoint of a watched Class when a new version
        # registers in it. Same tick and the same reasoning: a model
        # package appears because a PIPELINE registered it, so there is no
        # hook to hang this on -- only something to watch. Off unless
        # armed, and then one list_model_packages per tick; the expensive
        # half only runs on the tick after a version actually appears.
        try:
            await asyncio.to_thread(training.check_auto_export)
        except Exception as exc:
            print(f"Auto metrics export check failed: {exc}")
        # The safety net for a teardown that never ran because the process
        # died between the last job and it. Cheap when there is no endpoint:
        # one describe_endpoint that returns not-found.
        #
        # Off the event loop: when it does delete, it waits for the
        # inference component to go first, which took 150s in practice --
        # running that inline would stall every request in this
        # single-worker process for the duration.
        try:
            await asyncio.get_running_loop().run_in_executor(None, training.sweep_idle_endpoint)
        except Exception as exc:
            print(f"Idle endpoint sweep failed: {exc}")
