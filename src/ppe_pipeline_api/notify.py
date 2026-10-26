"""What kind of email a message is, and who chose to receive that kind.

Every human-facing email goes through ONE SNS topic,
`track2-yolo-reviewer-notifications`. Subscribers pick which kinds they
want on the Notifications page, and the choice is stored on their SNS
subscription itself as a FILTER POLICY, not in the app's config: SNS
then does the filtering, for every publisher -- this app, the pull
router, the queue runner and the platform Lambdas -- without any of them
needing to know who wants what.

How a message says what it is: a `event` message attribute, a
String.Array, so one message can be two kinds (a failed run is both
"training ended" and "an error").

Messages WITHOUT the attribute -- drift review alerts, weights approved,
the review host starting, an ordinary pull summary -- still reach every
subscriber. That is the `{"exists": false}` entry every policy carries:
a policy without it would silently stop every untagged email the moment
somebody ticked a box.
"""

from __future__ import annotations

import json
import os

#: The kinds a subscriber can choose, in the order the page shows them.
#: `evaluation_metrics` is offered but KIV: nothing publishes it yet.
EVENTS = {
    "training_started": "Training started",
    "training_ended": "Training ended (completed, stopped or failed)",
    "error": "Errors (a failed run, a failed pull, a retrain chain that stopped)",
    "evaluation_metrics": "Evaluation metrics",
}
#: Offered on the page, but not yet sent by anything.
KIV = {"evaluation_metrics"}
#: What a new subscriber gets unless they untick something.
DEFAULT_EVENTS = [e for e in EVENTS if e not in KIV]

ATTRIBUTE = "event"


def topic_arn() -> str:
    """The one notifications topic.

    From the environment where it is set (the Lambdas); otherwise built
    from the account, as `track2_shared` does for the page.
    """
    arn = (os.environ.get("NOTIFY_TOPIC_ARN")
           or os.environ.get("PULL_NOTIFY_TOPIC_ARN") or "").strip()
    if arn:
        return arn
    from . import track2_shared as t2

    return t2.reviewer_notifications_topic_arn()


def attributes(events: list) -> dict:
    """The MessageAttributes that label a message as `events`."""
    return {ATTRIBUTE: {"DataType": "String.Array",
                        "StringValue": json.dumps(sorted(set(events)))}}


def publish(subject: str, message: str, events: list, *, client=None) -> bool:
    """Send one email to whoever wants any of `events`. Never raises.

    A notification is never worth failing the thing it reports on: a
    chain that stopped would otherwise also lose its error report.
    """
    try:
        if client is None:
            import boto3

            client = boto3.client("sns", region_name=os.environ.get("AWS_REGION", "ap-southeast-1"))
        client.publish(TopicArn=topic_arn(), Subject=subject[:100], Message=message,
                       MessageAttributes=attributes(events))
        return True
    except Exception as exc:                                     # noqa: BLE001
        print(json.dumps({"notify": "publish_failed", "subject": subject[:100],
                          "error": str(exc)}))
        return False


def filter_policy(events: list) -> str:
    """The SNS filter policy for a subscriber who chose `events`.

    Always lets untagged messages through, so the emails that have no
    kind yet keep arriving exactly as before.
    """
    chosen = sorted({e for e in events if e in EVENTS})
    return json.dumps({ATTRIBUTE: [*chosen, {"exists": False}]})


def events_from_policy(policy: str | None) -> list:
    """Which kinds a subscription receives, read back from its policy.

    No policy at all means SNS filters nothing: every kind.
    """
    if not policy:
        return list(EVENTS)
    try:
        values = json.loads(policy).get(ATTRIBUTE) or []
    except (ValueError, AttributeError):
        return list(EVENTS)
    return [e for e in EVENTS if e in values]
