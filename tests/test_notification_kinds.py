"""Subscribers choose which kinds of email they get.

Stored as each SNS subscription's filter policy; messages carry an
`event` attribute. Untagged messages still reach everybody.
"""
import importlib.util
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppe_pipeline_api import notify  # noqa: E402
from ppe_pipeline_api import retrain_schedule as rs  # noqa: E402
from ppe_pipeline_api.routers import track2_notifications as tn  # noqa: E402


def test_every_policy_lets_untagged_mail_through():
    policy = json.loads(notify.filter_policy(["training_started"]))
    assert policy["event"] == ["training_started", {"exists": False}]
    assert json.loads(notify.filter_policy([]))["event"] == [{"exists": False}]


def test_a_policy_reads_back_as_its_choices():
    assert notify.events_from_policy(notify.filter_policy(["error", "training_ended"])) \
        == ["training_ended", "error"]
    # No policy means SNS filters nothing.
    assert notify.events_from_policy(None) == list(notify.EVENTS)


def test_evaluation_metrics_is_offered_but_kiv():
    assert "evaluation_metrics" in notify.EVENTS and "evaluation_metrics" in notify.KIV
    assert "evaluation_metrics" not in notify.DEFAULT_EVENTS


def test_publish_labels_the_message_and_never_raises():
    sent = []

    class Ok:
        def publish(self, **kw):
            sent.append(kw)

    class Broken:
        def publish(self, **kw):
            raise RuntimeError("AccessDenied")

    assert notify.publish("s", "m", ["error"], client=Ok()) is True
    assert json.loads(sent[0]["MessageAttributes"]["event"]["StringValue"]) == ["error"]
    assert sent[0]["MessageAttributes"]["event"]["DataType"] == "String.Array"
    assert notify.publish("s", "m", ["error"], client=Broken()) is False


class SNS:
    def __init__(self):
        self.calls = []

    def subscribe(self, **kw):
        self.calls.append(("subscribe", kw))

    def set_subscription_attributes(self, **kw):
        self.calls.append(("set", kw))


def test_a_new_subscriber_gets_their_choices_at_subscribe(monkeypatch):
    sns = SNS()
    monkeypatch.setattr(tn.t2, "sns", sns)
    monkeypatch.setattr(tn.t2, "reviewer_notifications_topic_arn", lambda: "arn:topic")
    tn.add_subscription(tn.AddSubscriptionRequest(email="a@b.c", events=["error"]), session=None)
    policy = json.loads(sns.calls[0][1]["Attributes"]["FilterPolicy"])
    assert policy["event"] == ["error", {"exists": False}]


def test_choices_are_changed_on_the_subscription(monkeypatch):
    sns = SNS()
    monkeypatch.setattr(tn.t2, "sns", sns)
    out = tn.set_events(tn.EventsRequest(subscription_arn="arn:topic:1",
                                         events=["training_started"]), session=None)
    assert out["events"] == ["training_started"]
    assert sns.calls[0][1]["AttributeName"] == "FilterPolicy"


def test_unknown_kinds_and_pending_subscribers_are_refused(monkeypatch):
    monkeypatch.setattr(tn.t2, "sns", SNS())
    with pytest.raises(Exception):
        tn.set_events(tn.EventsRequest(subscription_arn="arn:x", events=["nope"]), session=None)
    with pytest.raises(Exception):
        tn.set_events(tn.EventsRequest(subscription_arn="PendingConfirmation",
                                       events=["error"]), session=None)


def test_a_stopped_chain_emails_errors_and_a_queued_one_does_not(monkeypatch):
    sent = []
    monkeypatch.setattr(notify, "publish", lambda subject, message, events, **kw: sent.append((subject, events)))
    monkeypatch.setattr(rs, "_publish_progress", lambda **kw: None)
    monkeypatch.setattr(rs, "_run_chain_body", lambda **kw: {
        "ok": False, "steps": [{"step": "data.yaml", "ok": False, "detail": "no images"}]})
    rs.run_chain(trigger="schedule")
    assert sent == [("Retrain chain stopped at data.yaml", ["error"])]
    sent.clear()
    monkeypatch.setattr(rs, "_run_chain_body", lambda **kw: {
        "ok": True, "steps": [{"step": "start", "ok": False, "detail": "GPU busy, queued"}]})
    rs.run_chain(trigger="schedule")
    assert sent == []


def _lambda():
    path = ROOT / "terraform" / "platform" / "lambda_src" / "training_completed_notify.py"
    spec = importlib.util.spec_from_file_location("tcn", path)
    return path.read_text(encoding="utf-8"), spec


def test_the_training_lambda_labels_each_status():
    src, _ = _lambda()
    assert '"Executing": ["training_started"]' in src
    assert '"Failed": ["training_ended", "error"]' in src
    assert "MessageAttributes=attributes" in src


def test_the_rule_watches_the_apps_own_pipeline():
    tf = (ROOT / "terraform" / "platform" / "monitoring.tf").read_text(encoding="utf-8")
    assert "pipeline/track2-yolo-open-training-pipeline" in tf


def test_the_page_offers_the_ticks_and_marks_kiv():
    page = (ROOT / "web" / "src" / "routes" / "Track2NotificationsPage.tsx").read_text(encoding="utf-8")
    assert "setTrack2NotificationEvents" in page and "KIV" in page
