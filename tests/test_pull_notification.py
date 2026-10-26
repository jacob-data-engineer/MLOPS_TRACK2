"""A finished sweep tells the subscribers -- but not every sweep.

The notification goes in `route_pending`, not the router, because that is
the one line both triggers reach: a manual Pull now runs in the app
container, the schedule and arrival rules run the same function in the
Lambda. In the router it would have notified for exactly the pulls
somebody was already watching and stayed silent for every unattended one.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ppe_pipeline_api import pull_control as pc  # noqa: E402


class FakeSns:
    def __init__(self, raises: Exception | None = None):
        self.published: list[dict] = []
        self.raises = raises

    def publish(self, **kwargs):
        if self.raises:
            raise self.raises
        self.published.append(kwargs)
        return {"MessageId": "x"}


@pytest.fixture
def sns(monkeypatch):
    fake = FakeSns()
    monkeypatch.setattr(pc, "NOTIFY_TOPIC_ARN", "arn:aws:sns:ap-southeast-1:1:topic")
    monkeypatch.setattr(pc.boto3, "client", lambda *a, **k: fake)
    return fake


class TestWhenItSpeaks:
    def test_a_sweep_that_moved_files(self, sns):
        pc._notify_pull_finished({"moved": 12, "trigger": "schedule", "run_id": "r1"})
        assert len(sns.published) == 1
        assert "12 file(s) filed" in sns.published[0]["Subject"]
        assert "schedule" in sns.published[0]["Message"]

    def test_a_sweep_that_errored_even_with_nothing_moved(self, sns):
        pc._notify_pull_finished({"moved": 0, "errors": 3})
        assert len(sns.published) == 1
        assert "Errors:   3" in sns.published[0]["Message"]

    def test_a_failed_sweep(self, sns):
        pc._notify_pull_finished({"moved": 0}, aborted=True)
        assert "FAILED" in sns.published[0]["Subject"]

    def test_a_stopped_sweep(self, sns):
        pc._notify_pull_finished({"moved": 0}, stopped=True)
        assert "stopped" in sns.published[0]["Subject"]


class TestWhenItStaysQuiet:
    def test_a_schedule_that_found_nothing(self, sns):
        # The whole point. A daily "0 files" is how an address gets
        # filtered, and then the one that mattered goes unread too.
        pc._notify_pull_finished({"moved": 0, "errors": 0, "trigger": "schedule"})
        assert sns.published == []

    def test_no_topic_configured(self, monkeypatch):
        sent = []
        monkeypatch.setattr(pc, "NOTIFY_TOPIC_ARN", "")
        monkeypatch.setattr(pc.boto3, "client",
                            lambda *a, **k: sent.append(1) or FakeSns())
        pc._notify_pull_finished({"moved": 9})
        # Not even a client is constructed.
        assert sent == []


class TestItCannotBreakAPull:
    def test_an_sns_failure_is_swallowed(self, monkeypatch, capsys):
        # A pull that moved files has done its job. Failing it because SNS
        # was unreachable turns a notification problem into a data problem.
        monkeypatch.setattr(pc, "NOTIFY_TOPIC_ARN", "arn:aws:sns:x:1:t")
        monkeypatch.setattr(
            pc.boto3, "client",
            lambda *a, **k: FakeSns(raises=RuntimeError("AccessDenied")))
        pc._notify_pull_finished({"moved": 4})  # must not raise
        assert "notify_failed" in capsys.readouterr().out

    def test_the_failure_is_logged_not_silent(self, monkeypatch, capsys):
        monkeypatch.setattr(pc, "NOTIFY_TOPIC_ARN", "arn:aws:sns:x:1:t")
        monkeypatch.setattr(
            pc.boto3, "client",
            lambda *a, **k: FakeSns(raises=RuntimeError("boom")))
        pc._notify_pull_finished({"moved": 1})
        assert "boom" in capsys.readouterr().out


class TestItIsWiredIn:
    def test_route_pending_calls_it(self):
        # A notifier nothing calls is the failure this would ship as.
        import inspect

        assert "_notify_pull_finished(" in inspect.getsource(pc.route_pending)
