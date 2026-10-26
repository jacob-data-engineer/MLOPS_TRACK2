"""The pull queue: sweeps lined up instead of racing each other.

Two sweeps at once do not corrupt the week structure -- the week is
arithmetic and `create_week` is a conditional put -- but they duplicate
every copy, race the delete so the loser records errors that never
happened, and write overlapping undo manifests. A pull MOVES, so undoing
either run then deletes a file nothing has a second copy of.

The guard used to be on the manual route only. The Lambda, which is the
one that runs unattended, had none at all.
"""
from __future__ import annotations

import pytest

from ppe_pipeline_api import pull_control as pc


@pytest.fixture
def table(monkeypatch):
    """A tiny in-memory stand-in for the DynamoDB table."""
    rows: dict = {}

    def put_item(**kw):
        item = kw["Item"]
        rows[(item["kind"]["S"], item["sort"]["S"])] = item
        return {}

    def query(**kw):
        kind = kw["ExpressionAttributeValues"][":k"]["S"]
        got = [v for (k, _s), v in rows.items() if k == kind]
        got.sort(key=lambda i: i["sort"]["S"],
                 reverse=not kw.get("ScanIndexForward", True))
        return {"Items": got}

    def delete_item(**kw):
        rows.pop((kw["Key"]["kind"]["S"], kw["Key"]["sort"]["S"]), None)
        return {}

    counter = {"value": 0}

    def update_item(**kw):
        """The atomic ADD the sequence depends on."""
        counter["value"] += 1
        return {"Attributes": {"value": {"N": str(counter["value"])}}}

    monkeypatch.setattr(pc.ddb, "update_item", update_item)
    monkeypatch.setattr(pc.ddb, "put_item", put_item)
    monkeypatch.setattr(pc.ddb, "query", query)
    monkeypatch.setattr(pc.ddb, "delete_item", delete_item)
    monkeypatch.setattr(pc, "read_config",
                        lambda: {"sources": [{"prefix": "drop"}], "root": "retraining"})
    return rows


# --- an entry is a piece of work, not a repeat ---------------------------------

def test_an_entry_carries_the_settings_it_was_queued_with(table):
    """A sweep reads the live config and moves whatever is waiting, so two
    entries that both meant 'sweep now' would be the same instruction
    twice. The snapshot is what makes an entry distinct work."""
    entry = pc.enqueue_pull(config={"sources": [{"prefix": "alpha"}], "root": "r1"})
    assert entry["config"]["sources"][0]["prefix"] == "alpha"

    pc.enqueue_pull(config={"sources": [{"prefix": "beta"}], "root": "r2"})
    queued = pc.list_pull_queue()
    assert [e["config"]["sources"][0]["prefix"] for e in queued] == ["alpha", "beta"]


def test_the_queue_is_fifo_by_the_key_not_by_sorting_afterwards(table):
    for _ in range(3):
        pc.enqueue_pull(config={"sources": [{"prefix": "x"}]})
    added = [e["added_at"] for e in pc.list_pull_queue()]
    assert added == sorted(added)


def test_an_entry_says_what_it_will_sweep(table):
    entry = pc.enqueue_pull(
        config={"sources": [{"prefix": "drop/a"}], "root": "retraining"})
    assert entry["summary"] == "drop/a into retraining"


# --- draining -------------------------------------------------------------------

def test_nothing_starts_while_a_sweep_is_running(table, monkeypatch):
    pc.enqueue_pull(config={"sources": [{"prefix": "a"}]})
    monkeypatch.setattr(pc, "active_run", lambda: {"status": "running", "run_id": "r1"})
    swept = []
    monkeypatch.setattr(pc, "route_pending", lambda **kw: swept.append(kw) or {})

    out = pc.advance_pull_queue()
    assert out["started"] is False
    assert swept == []
    assert len(pc.list_pull_queue()) == 1, "the entry must stay queued"


def test_a_lost_sweep_does_not_block_the_queue_for_ever(table, monkeypatch):
    """`active_run` reports anything quiet past RUN_STALE_SECONDS as
    `lost`. Treating that as 'running' would fix the overlap by
    introducing a queue that never moves again."""
    pc.enqueue_pull(config={"sources": [{"prefix": "a"}]})
    monkeypatch.setattr(pc, "active_run", lambda: {"status": "lost", "run_id": "r1"})
    monkeypatch.setattr(pc, "route_pending", lambda **kw: {"moved": 3})
    monkeypatch.setattr(pc, "write_run_state", lambda *a, **k: {})

    assert pc.advance_pull_queue()["started"] is True


def test_the_entry_runs_under_its_own_snapshot(table, monkeypatch):
    pc.enqueue_pull(config={"sources": [{"prefix": "alpha"}], "root": "r1"})
    monkeypatch.setattr(pc, "active_run", lambda: None)
    monkeypatch.setattr(pc, "write_run_state", lambda *a, **k: {})
    seen = {}
    monkeypatch.setattr(pc, "route_pending",
                        lambda **kw: seen.update(kw) or {"moved": 1})

    pc.advance_pull_queue()
    assert seen["config"]["sources"][0]["prefix"] == "alpha"


def test_an_entry_is_dropped_before_it_runs_not_after(table, monkeypatch):
    """A sweep can take fifteen minutes and the process can die. An entry
    still queued when that happens is picked up by the next tick and runs
    a second time -- which is what this queue exists to prevent. Losing
    one sweep is recoverable: the next moves the same waiting files."""
    pc.enqueue_pull(config={"sources": [{"prefix": "a"}]})
    monkeypatch.setattr(pc, "active_run", lambda: None)
    monkeypatch.setattr(pc, "write_run_state", lambda *a, **k: {})

    during = {}

    def sweep(**kw):
        during["queued"] = len(pc.list_pull_queue())
        return {"moved": 1}

    monkeypatch.setattr(pc, "route_pending", sweep)
    pc.advance_pull_queue()
    assert during["queued"] == 0


def test_a_refused_snapshot_is_recorded_not_swallowed(table, monkeypatch):
    pc.enqueue_pull(config={"sources": [{"prefix": "a"}]})
    monkeypatch.setattr(pc, "active_run", lambda: None)
    states = []
    monkeypatch.setattr(pc, "write_run_state",
                        lambda run_id, **kw: states.append(kw) or {})

    def refuse(**kw):
        raise ValueError("no label folder picked")

    monkeypatch.setattr(pc, "route_pending", refuse)
    out = pc.advance_pull_queue()
    assert out["started"] is False
    assert "no label folder" in out["reason"]
    assert states and states[0]["status"] == "failed"


# --- the handler no longer races ------------------------------------------------

def test_a_trigger_during_a_sweep_is_queued_not_run(table, monkeypatch):
    """The bug this fixes: the Lambda called route_pending with no check
    at all, so an arrival burst of 500 files was 500 concurrent sweeps."""
    monkeypatch.setattr(pc, "active_run", lambda: {"status": "running", "run_id": "r1"})
    swept = []
    monkeypatch.setattr(pc, "route_pending", lambda **kw: swept.append(kw) or {})

    out = pc.lambda_handler({"detail-type": "Scheduled Event"})
    assert out["queued"] is True
    assert swept == [], "it must not sweep beside the running one"
    assert len(pc.list_pull_queue()) == 1


def test_a_tick_drains_one_entry_and_never_sweeps_on_its_own(table, monkeypatch):
    """One per tick: a sweep can run the full fifteen minutes a Lambda
    gets, so draining two risks killing the second mid-copy."""
    pc.enqueue_pull(config={"sources": [{"prefix": "a"}]})
    pc.enqueue_pull(config={"sources": [{"prefix": "b"}]})
    monkeypatch.setattr(pc, "active_run", lambda: None)
    monkeypatch.setattr(pc, "write_run_state", lambda *a, **k: {})
    runs = []
    monkeypatch.setattr(
        pc, "route_pending",
        lambda **kw: runs.append(kw["config"]["sources"][0]["prefix"]) or {"moved": 1})

    pc.lambda_handler({"tick": True})
    assert runs == ["a"]
    assert len(pc.list_pull_queue()) == 1


def test_a_tick_with_an_empty_queue_does_nothing(table, monkeypatch):
    monkeypatch.setattr(pc, "active_run", lambda: None)
    swept = []
    monkeypatch.setattr(pc, "route_pending", lambda **kw: swept.append(kw) or {})

    out = pc.lambda_handler({"tick": True})
    assert out["started"] is False
    assert swept == []


# --- housekeeping ---------------------------------------------------------------

def test_removing_and_clearing(table):
    first = pc.enqueue_pull(config={"sources": [{"prefix": "a"}]})
    pc.enqueue_pull(config={"sources": [{"prefix": "b"}]})
    assert pc.remove_pull_queue_entry(first["id"]) is True
    assert pc.remove_pull_queue_entry(first["id"]) is False
    assert len(pc.list_pull_queue()) == 1
    assert pc.clear_pull_queue() == 1
    assert pc.list_pull_queue() == []


def test_entries_expire_so_a_stale_sweep_is_not_run_next_week(table):
    """A sweep queued a week ago was queued against folders something else
    has since pulled. Running it then is not catching up."""
    pc.enqueue_pull(config={"sources": [{"prefix": "a"}]})
    item = next(iter(table.values()))
    assert "expires_at" in item
    assert int(item["expires_at"]["N"]) > 0


def test_entries_queued_in_the_same_clock_tick_still_run_in_order(table):
    """The bug this had. The key was `added_at#entry_id` on the assumption
    that two entries never share a timestamp -- but the clock here has
    ~15ms granularity, so a burst got identical timestamps and the
    tiebreak fell to the RANDOM hex in the entry id. The queue called
    itself FIFO and was not.

    `time.time_ns()` was measured and is equally coarse, so a finer clock
    was never the answer; the tiebreak is a counter.
    """
    import datetime as dt

    # Every entry in the same tick, which is what the real clock does.
    frozen = dt.datetime(2026, 10, 14, 1, 0, 0, tzinfo=dt.timezone.utc)
    original = pc._now
    pc._now = lambda: frozen
    try:
        for name in ("first", "second", "third", "fourth", "fifth"):
            pc.enqueue_pull(config={"sources": [{"prefix": name}]})
    finally:
        pc._now = original

    order = [e["config"]["sources"][0]["prefix"] for e in pc.list_pull_queue()]
    assert order == ["first", "second", "third", "fourth", "fifth"]
    # And they really did collide, or this proves nothing.
    assert len({e["added_at"] for e in pc.list_pull_queue()}) == 1


def test_order_survives_a_burst_of_cold_starts(table, monkeypatch):
    """The hole the first fix left.

    `itertools.count()` restarts at zero in every Lambda execution
    environment, and an arrival burst is many invocations -- so five cold
    starts queuing inside one clock tick came back
    `second, fifth, fourth, third, first`. Scrambled in precisely the case
    the queue exists for.

    The sequence has to be shared, which means the table.
    """
    import datetime as dt
    import itertools

    frozen = dt.datetime(2026, 10, 14, 2, 0, 0, tzinfo=dt.timezone.utc)
    monkeypatch.setattr(pc, "_now", lambda: frozen)

    for name in ("first", "second", "third", "fourth", "fifth"):
        # Each enqueue from a fresh process, which is what a burst is.
        monkeypatch.setattr(pc, "_pull_queue_fallback_seq", itertools.count())
        pc.enqueue_pull(config={"sources": [{"prefix": name}]})

    order = [e["config"]["sources"][0]["prefix"] for e in pc.list_pull_queue()]
    assert order == ["first", "second", "third", "fourth", "fifth"]
    assert len({e["added_at"] for e in pc.list_pull_queue()}) == 1, "they must really collide"


def test_the_sequence_is_an_atomic_add_not_a_read_then_write(table, monkeypatch):
    """Read-then-write has a window in which two callers read the same
    number. `ADD` does not."""
    seen = {}
    monkeypatch.setattr(pc.ddb, "update_item",
                        lambda **kw: seen.update(kw)
                        or {"Attributes": {"value": {"N": "7"}}})
    assert pc._next_pull_queue_seq() == 7
    assert "ADD" in seen["UpdateExpression"]
    # And it must not expire: a counter that restarted at one would sort
    # new entries ahead of old ones.
    assert "expires_at" not in str(seen)


def test_an_unavailable_counter_still_lets_you_queue(table, monkeypatch):
    """Degraded, not broken. Refusing to queue because a counter row is
    unreachable would turn a tidy-ordering problem into a lost sweep."""
    from botocore.exceptions import ClientError

    def broken(**kw):
        raise ClientError({"Error": {"Code": "ProvisionedThroughputExceededException"}},
                          "UpdateItem")

    monkeypatch.setattr(pc.ddb, "update_item", broken)
    entry = pc.enqueue_pull(config={"sources": [{"prefix": "a"}]})
    assert entry["id"]
    assert len(pc.list_pull_queue()) == 1
