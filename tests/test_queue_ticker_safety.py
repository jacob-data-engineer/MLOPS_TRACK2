"""The two tickers must not start the same run, or block each other.

The training queue is advanced by two processes: the app's poll loop and
the `track2-app-queue-tick` Lambda. They share DynamoDB rows but not
memory, and that is where both of these bugs lived.
"""
from __future__ import annotations

import inspect

import pytest

from ppe_pipeline_api import queue_runner
from ppe_pipeline_api import track2_shared as t2


def _entry(entry_id: str, order: str) -> dict:
    return {
        "id": entry_id, "record_type": t2.QUEUE_RECORD_TYPE, "added_at": order,
        "generation": "11", "size": "s", "epochs": 1, "img_size": 640,
        "batch": 1, "seed": 1, "notes": "", "dataset_folder": "d",
        "base_model_folder": "",
    }


@pytest.fixture
def queue(monkeypatch):
    """A fake table, and a record of every run started."""
    rows: dict = {}
    started: list = []

    class Table:
        def scan(self, **kw):
            return {"Items": list(rows.values())}

        def put_item(self, Item):
            rows[Item["id"]] = Item

        def delete_item(self, Key):
            rows.pop(Key["id"], None)

        def get_item(self, Key):
            return {"Item": {"id": t2.QUEUE_STATE_ID, "started": True}}

    monkeypatch.setattr(t2, "training_audit_table", lambda: Table())
    monkeypatch.setattr(t2, "dataset_source_folders", lambda folder, split="": [])
    # The GPU is free and nothing was submitted recently: the state right
    # after a run finishes, which is when the duplicate used to fire.
    monkeypatch.setattr(t2, "training_already_running_reason", lambda progress: "")
    monkeypatch.setattr(t2, "get_training_progress", lambda: {})
    monkeypatch.setattr(t2, "start_training_run",
                        lambda *a, **k: started.append(a[0]) or "build-1")
    t2._training_queue[:] = []
    t2._queue_running = None
    t2._queue_starting = False
    yield {"rows": rows, "started": started}
    t2._training_queue[:] = []
    t2._queue_running = None
    t2._queue_starting = False


# --- the warm container ---------------------------------------------------------

def test_a_warm_lambda_does_not_restart_what_the_app_already_ran(queue):
    """The bug this file exists for.

    A Lambda container is reused between two-minute ticks, so its copy of
    the queue outlives the invocation that built it. The app pops an
    entry, starts it and deletes the row -- and the warm container hears
    none of that. It used to go on holding the entry and start it again
    once the run finished: a second unattended GPU run of a finished job.
    """
    queue["rows"]["A"] = _entry("A", "1")
    queue["rows"]["B"] = _entry("B", "2")

    queue_runner.lambda_handler()                 # cold: starts A
    assert queue["started"] == ["11"]
    assert [e["id"] for e in t2._training_queue] == ["B"]

    # The app takes B, starts it, and deletes its row.
    queue["rows"].pop("B")
    t2._queue_running = None

    queue_runner.lambda_handler()                 # warm: must NOT start B
    assert queue["started"] == ["11"], "the warm container started a finished run again"
    assert t2._training_queue == [], "the stale entry should have been dropped"


def test_the_lambda_asks_for_a_refresh(queue):
    """Read off the call rather than the behaviour, so the reason stays
    visible if somebody simplifies it back."""
    source = inspect.getsource(queue_runner.lambda_handler)
    assert "restore_queue(refresh=True)" in source


def test_the_app_still_keeps_its_own_queue(queue):
    """`refresh` is opt-in for a reason: inside the app the in-memory list
    is the live one and may hold entries not yet written. Replacing it
    from a scan would lose them."""
    t2._training_queue[:] = [_entry("local", "9")]
    queue["rows"]["other"] = _entry("other", "1")

    assert t2.restore_queue() == 0
    assert [e["id"] for e in t2._training_queue] == ["local"]


def test_a_refresh_also_re_reads_the_started_flag(queue, monkeypatch):
    """The early return skipped it, so a warm container never learned
    that Stop Queue had been pressed."""
    t2._training_queue[:] = [_entry("stale", "1")]
    t2._queue_started = True

    class Stopped:
        def scan(self, **kw):
            return {"Items": []}

        def get_item(self, Key):
            return {"Item": {"id": t2.QUEUE_STATE_ID, "started": False}}

        def delete_item(self, Key):
            pass

    monkeypatch.setattr(t2, "training_audit_table", lambda: Stopped())
    t2.restore_queue(refresh=True)
    assert t2._queue_started is False


# --- pagination -----------------------------------------------------------------

def test_the_scan_reads_every_page(monkeypatch):
    """A DynamoDB scan returns at most 1MB and sets `LastEvaluatedKey`.
    Reading one page meant that, once the shared audit table crossed that,
    queued runs would vanish on restart -- silently, and exactly the loss
    the durable queue exists to prevent."""
    pages = [
        {"Items": [_entry("one", "1")], "LastEvaluatedKey": {"id": "one"}},
        {"Items": [_entry("two", "2")]},
    ]
    seen: list = []

    class Paged:
        def scan(self, **kw):
            seen.append(kw.get("ExclusiveStartKey"))
            return pages[len(seen) - 1]

    monkeypatch.setattr(t2, "training_audit_table", lambda: Paged())
    rows = t2._scan_queue_rows()
    assert [r["id"] for r in rows] == ["one", "two"]
    assert seen == [None, {"id": "one"}], "the second page needs the key from the first"


# --- the lock -------------------------------------------------------------------

def test_the_submission_does_not_hold_the_queue_lock():
    """Holding `_queue_lock` across a SageMaker call blocked every other
    reader -- adding a run, removing one, and the card's 15-second poll."""
    source = inspect.getsource(t2.try_advance_queue)
    under_lock, _, _ = source.partition("# --- out of the lock from here")
    assert "start_custom_training_run(" not in under_lock
    assert "start_training_run(" not in under_lock
    assert "_forget_queue_entry(" not in under_lock


def test_a_second_ticker_cannot_pop_while_one_is_submitting():
    """Releasing the lock before the network call reopens the race it was
    preventing, so a flag has to close it."""
    source = inspect.getsource(t2.try_advance_queue)
    assert "if _queue_starting:" in source
    assert "_queue_starting = True" in source


def test_the_flag_is_always_cleared():
    """One failed submission must not stop the queue for the life of the
    process."""
    source = inspect.getsource(t2.try_advance_queue)
    tail = source.split("finally:")[-1]
    assert "_queue_starting = False" in tail


def test_a_failed_submission_clears_the_flag_and_skips_the_entry(queue, monkeypatch):
    def explode(*a, **k):
        raise RuntimeError("SageMaker said no")

    monkeypatch.setattr(t2, "start_training_run", explode)
    queue["rows"]["A"] = _entry("A", "1")
    t2.restore_queue(refresh=True)

    t2.try_advance_queue()
    assert t2._queue_starting is False, "a failure left the queue wedged"
    assert t2._training_queue == [], "a bad entry must not be retried forever"
    assert "failed to start" in t2._queue_last_error


# --- the poll loop ---------------------------------------------------------------

def test_the_poll_tick_runs_off_the_event_loop():
    """Every call in the tick is synchronous boto3. Inline, they stalled
    every request handler in this single-worker process -- which is the
    very thing the `asyncio.sleep` above them was chosen to avoid."""
    source = inspect.getsource(t2.queue_poll_loop)
    for call in ("try_advance_queue", "reconcile_training_audits",
                 "queue_after_pull.check", "metrics_export.check"):
        assert f"asyncio.to_thread({call}" in source, f"{call} still blocks the loop"


# --- a freeze that cannot freeze ------------------------------------------------

def test_a_snapshot_missing_a_split_is_refused(monkeypatch):
    """`create` used to `continue` past a failed copy, so the row was
    written and reported as successful while silently missing the file
    list that makes it a snapshot.

    Its own comment names the consequence: a restore then hands back a
    yaml naming `_retained/train.txt` whose contents have moved on,
    "which looks right and is not, and is the worst of the three possible
    outcomes".
    """
    import inspect

    from ppe_pipeline_api import dataset_snapshots as ds

    source = inspect.getsource(ds.create)
    # To the next statement rather than a fixed window: a character count
    # stopped short of the `raise` once the comment explaining it was
    # written.
    after = source.split("pc.s3.copy_object(")[1]
    copy_block = after[:after.index("lists[split] = {")]
    assert "continue" not in copy_block, (
        "a split that could not be copied must fail the freeze, not be skipped"
    )
    assert "raise ValueError(" in copy_block


def test_the_schedule_says_when_it_ignores_a_number():
    """A corrupt `epochs` used to be swallowed by a bare `pass`, so the
    schedule trained at the default while the card showed the bad value
    and nothing anywhere disagreed."""
    import inspect

    from ppe_pipeline_api import retrain_schedule as rs

    source = inspect.getsource(rs)
    block = source.split("plan[key] = int(shape[key])")[1][:500]
    assert "print(" in block, "ignoring a value silently is how the two disagree"
