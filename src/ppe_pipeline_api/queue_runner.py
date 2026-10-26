"""Advances the training queue when the app container is not there to.

WHY THIS EXISTS. The queue itself has always been durable -- entries are
DynamoDB rows, FIFO order is restored on startup, and `try_advance_queue`
asks SageMaker whether anything is executing before starting the next, so
a restart can neither lose an entry nor submit a duplicate. What was
missing was the TICK. The poll loop is an asyncio task in the app's
lifespan, so while the container is down nothing advances: five queued
runs with the app off for an hour start an hour late, on a GPU that sat
idle the whole time.

WHY IT CALLS THE REAL FUNCTION rather than reimplementing it. The guard
that decides whether it is safe to start a run reads SageMaker pipeline
executions, custom runs in progress, and a pending-submission marker.
Writing a second version of "is the GPU free?" for a Lambda is how an
account with one training instance comes to have two runs submitted
against it -- the second fails outright with SAGEMAKER_RESOURCE_LIMIT and
no retry. So this imports `track2_shared` and calls the same
`try_advance_queue` the container calls, and there is exactly one
implementation of the decision.

WHAT IS IN THE ZIP. `ppe_pipeline_api/` plus PyYAML, assembled by
deploy.sh and zipped by Terraform. Lambda ships boto3 and botocore and
nothing else.

PyYAML is there because the advance really does need it: starting a run
validates the dataset, and validating reads its `data.yaml`. The lazy
`_yaml()` in track2_shared is still worth having -- it means importing
the module costs nothing and an unrelated Lambda path cannot die on a
missing dependency -- but it does not remove the need here, and an
earlier version of this comment claimed it did.

Only the `yaml/` package is copied, never the `_yaml` C extension beside
it: that one is built for the machine deploy.sh runs on, and PyYAML falls
back to its pure-Python parser when the extension will not import.

SAFE TO RUN BESIDE THE APP, but only because of the refresh below. Both
tickers call the same function, which re-checks SageMaker before popping
anything, and the entry is deleted from DynamoDB the moment it is popped
-- so the loser of a race finds the queue one shorter.

That argument assumes both tickers READ DynamoDB. This one did not: it
restored only into an empty queue, and a reused Lambda container is not
empty. It could therefore hold an entry the app had already started and
deleted, and start it again once that run finished. The refresh is what
makes the paragraph above true.
"""
from __future__ import annotations

import json


def lambda_handler(event=None, _context=None):
    """One advance per invocation. Returns what it did, for the log.

    Deliberately not a loop. Starting a run returns as soon as CodeBuild
    accepts the build, so a loop would submit the whole queue at once
    into a single GPU slot -- every one after the first failing on the
    resource limit. One per tick, and the queue drains at the rate runs
    actually finish.
    """
    from . import track2_shared as t2

    # REFRESHED, not merely restored. A cold container has an empty queue
    # and any restore fills it; a WARM one -- and the two-minute tick
    # means most are warm -- still holds the list it built last time.
    #
    # The app ticks the same queue. It pops an entry, starts it and
    # deletes the row, and this container hears none of that. Without a
    # refresh it goes on holding an entry that has already run, and once
    # that run finishes and the submission marker ages out, it starts it
    # a second time: a full unattended GPU run of a job that is done.
    #
    # The rows are the truth here. The in-memory copy is only the truth
    # inside the app, which is why `refresh` is opt-in rather than the
    # default.
    restored = t2.restore_queue(refresh=True)

    # The retrain clock, on the same tick. The app container runs this
    # too; `claim_once` inside it means exactly one of them acts on a
    # given slot. Here as well as there because the container can be
    # down -- and a schedule that only fires while somebody has the page
    # open is not a schedule.
    try:
        from . import retrain_schedule
        retrain_schedule.check_training_schedule()
    except Exception as exc:                                       # noqa: BLE001
        print(f"Training schedule check failed: {exc}")

    # Automatic export of the best checkpoint, on the same tick -- so it
    # still runs with the app off (2026-10-26). The app ticks it too; the
    # claim inside `check` means one of them acts per version.
    try:
        from . import metrics_export
        print(f"Auto export: {metrics_export.check()}")
    except Exception as exc:                                       # noqa: BLE001
        print(f"Auto export check failed: {exc}")

    before = t2.queue_snapshot() if hasattr(t2, "queue_snapshot") else None
    try:
        t2.try_advance_queue()
    except Exception as exc:                                       # noqa: BLE001
        # Never raise: a raising Lambda is retried by EventBridge, and a
        # retry of "start the next training run" is the one thing that
        # must not happen automatically.
        print(json.dumps({"queue_runner": "failed", "error": str(exc)[:400]}))
        return {"advanced": False, "restored": restored, "error": str(exc)[:400]}

    out = {"advanced": True, "restored": restored}
    if before is not None:
        out["before"] = before
    print(json.dumps({"queue_runner": "ticked", **out}))
    return out
