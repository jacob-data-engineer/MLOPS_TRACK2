"""A restored queue has to come back in the state it was left in.

The ENTRIES already survived a restart -- they are written to DynamoDB
when they are queued. The SWITCH did not: `_queue_started` is a module
global, so every deploy brought the queue back full and stopped. The log
said "Restored 1 queued training run(s) after restart" and then nothing
happened, with the GPU free and nothing on screen explaining why.

Persisted rather than defaulted either way. Defaulting to started would
spend GPU nobody asked for after a crash; defaulting to stopped is what
produced this.
"""
from __future__ import annotations

import importlib

import pytest

t2 = importlib.import_module("src.ppe_pipeline_api.track2_shared")


class FakeTable:
    def __init__(self, items=None):
        self.items = {i["id"]: i for i in (items or [])}

    def put_item(self, Item):
        self.items[Item["id"]] = Item
        return {}

    def get_item(self, Key):
        got = self.items.get(Key["id"])
        return {"Item": got} if got else {}

    def scan(self):
        return {"Items": list(self.items.values())}


@pytest.fixture
def table(monkeypatch):
    fake = FakeTable()
    monkeypatch.setattr(t2, "training_audit_table", lambda: fake)
    monkeypatch.setattr(t2, "_training_queue", [])
    monkeypatch.setattr(t2, "_queue_started", False)
    monkeypatch.setattr(t2, "_queue_running", None)
    return fake


def test_starting_the_queue_records_it(table):
    t2._training_queue.append({"id": "e1", "added_at": "2026-10-06T00:00:00"})
    # try_advance_queue would reach SageMaker; the flag is what is checked.
    import unittest.mock as mock
    with mock.patch.object(t2, "try_advance_queue", lambda: None):
        assert t2.queue_start() is True
    assert table.items[t2.QUEUE_STATE_ID]["started"] is True


def test_stopping_the_queue_records_it(table):
    t2.queue_stop()
    assert table.items[t2.QUEUE_STATE_ID]["started"] is False


def test_a_queue_that_was_running_resumes(table):
    """The regression. Entry restored, switch restored, so the next poll
    tick pops it instead of the GPU sitting idle indefinitely."""
    table.put_item(Item={"id": t2.QUEUE_STATE_ID,
                         "record_type": t2.QUEUE_STATE_RECORD_TYPE, "started": True})
    table.put_item(Item={"id": "e1", "record_type": t2.QUEUE_RECORD_TYPE,
                         "added_at": "2026-10-06T00:00:00"})
    assert t2.restore_queue() == 1
    assert t2._queue_started is True


def test_a_queue_that_was_stopped_stays_stopped(table):
    """Resuming one somebody had deliberately stopped would start GPU
    work they had just refused."""
    table.put_item(Item={"id": t2.QUEUE_STATE_ID,
                         "record_type": t2.QUEUE_STATE_RECORD_TYPE, "started": False})
    table.put_item(Item={"id": "e1", "record_type": t2.QUEUE_RECORD_TYPE,
                         "added_at": "2026-10-06T00:00:00"})
    t2.restore_queue()
    assert t2._queue_started is False


def test_no_recorded_state_means_stopped(table):
    """An older deployment never wrote the row. Stopped is the safe
    reading of silence -- it costs a button press, not GPU hours."""
    table.put_item(Item={"id": "e1", "record_type": t2.QUEUE_RECORD_TYPE,
                         "added_at": "2026-10-06T00:00:00"})
    t2.restore_queue()
    assert t2._queue_started is False


def test_the_state_row_is_not_restored_as_a_queue_entry(table):
    """It lives in the same table. Restoring it as an entry would queue a
    training run with no dataset and no detector."""
    table.put_item(Item={"id": t2.QUEUE_STATE_ID,
                         "record_type": t2.QUEUE_STATE_RECORD_TYPE, "started": True})
    assert t2.restore_queue() == 0
    assert t2._training_queue == []


def test_a_table_that_cannot_be_read_does_not_crash_the_startup(monkeypatch):
    """This runs in the app's lifespan. A DynamoDB outage should delay
    the queue, not stop the process from starting."""
    def boom():
        raise RuntimeError("no table")
    monkeypatch.setattr(t2, "training_audit_table", boom)
    monkeypatch.setattr(t2, "_training_queue", [])
    assert t2.restore_queue() == 0


def test_a_failed_persist_does_not_fail_the_start(monkeypatch, table):
    """A queue that runs but is not durable still beats one that refuses
    to start."""
    def boom():
        raise RuntimeError("throttled")
    monkeypatch.setattr(t2, "training_audit_table", boom)
    t2.queue_stop()      # must not raise
    assert t2._queue_started is False


def test_the_log_line_says_which_state_it_came_back_in(table, capsys):
    """"Restored 1 queued training run(s)" said nothing about whether
    anything would happen next, which is the only question that mattered."""
    table.put_item(Item={"id": t2.QUEUE_STATE_ID,
                         "record_type": t2.QUEUE_STATE_RECORD_TYPE, "started": False})
    table.put_item(Item={"id": "e1", "record_type": t2.QUEUE_RECORD_TYPE,
                         "added_at": "2026-10-06T00:00:00"})
    t2.restore_queue()
    assert "press Start Queue" in capsys.readouterr().out
