"""Track2 reviewer-notification email subscribers -- backs
`/track2/notifications`.

Each subscriber also chooses WHICH kinds of email they get. The choice
is stored on their SNS subscription as a filter policy -- see
`ppe_pipeline_api.notify` -- so SNS does the filtering for every
publisher, and there is no second copy in the app's config to drift.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from .. import notify
from .. import track2_shared as t2
from ..auth import require_session

router = APIRouter()


def _events_of(subscription_arn: str) -> list | None:
    """What a confirmed subscriber receives. None when it cannot be read.

    An unconfirmed subscriber has no ARN yet, and SNS will not show its
    attributes until it does.
    """
    if not subscription_arn.startswith("arn:"):
        return None
    try:
        attrs = t2.sns.get_subscription_attributes(
            SubscriptionArn=subscription_arn).get("Attributes") or {}
    except Exception as exc:                                     # noqa: BLE001
        print(f"Could not read subscription attributes: {exc}")
        return None
    return notify.events_from_policy(attrs.get("FilterPolicy"))


def _list_subscriptions() -> list[dict]:
    subscriptions = []
    kwargs: dict = {}
    while True:
        response = t2.sns.list_subscriptions_by_topic(TopicArn=t2.reviewer_notifications_topic_arn(), **kwargs)
        subscriptions.extend(response.get("Subscriptions", []))
        token = response.get("NextToken")
        if not token:
            break
        kwargs["NextToken"] = token
    return [
        {
            "subscription_arn": s["SubscriptionArn"],
            "email": s["Endpoint"],
            "confirmed": s["SubscriptionArn"] != "PendingConfirmation",
            "events": _events_of(s["SubscriptionArn"]),
        }
        for s in subscriptions
        if s["Protocol"] == "email"
    ]


@router.get("")
def get_notifications(session=Depends(require_session)) -> dict:
    return {
        "subscriptions": _list_subscriptions(),
        # The choices, from the one place that defines them, so the page
        # never offers a kind nothing knows how to filter on.
        "events": [{"key": k, "label": v, "kiv": k in notify.KIV}
                   for k, v in notify.EVENTS.items()],
    }


def _clean(events: list | None) -> list:
    if events is None:
        return list(notify.DEFAULT_EVENTS)
    unknown = [e for e in events if e not in notify.EVENTS]
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unknown notification kind: {', '.join(unknown)}.")
    return list(events)


class AddSubscriptionRequest(BaseModel):
    email: str
    # Which kinds to send. Omitted: everything that is sent today.
    events: list[str] | None = None


@router.post("/add")
def add_subscription(payload: AddSubscriptionRequest, session=Depends(require_session)) -> dict:
    email = payload.email.strip()
    if not email:
        raise HTTPException(status_code=422, detail="No email address given.")
    events = _clean(payload.events)
    # Set at subscribe time: an unconfirmed subscription has no ARN to
    # set it on afterwards, and the choice must already hold for the
    # first email after they confirm.
    t2.sns.subscribe(TopicArn=t2.reviewer_notifications_topic_arn(), Protocol="email",
                     Endpoint=email,
                     Attributes={"FilterPolicy": notify.filter_policy(events)})
    return {"email": email, "message": f"Confirmation email sent to {email} -- they must click the link before notifications start."}


class EventsRequest(BaseModel):
    subscription_arn: str
    events: list[str]


@router.post("/events")
def set_events(payload: EventsRequest, session=Depends(require_session)) -> dict:
    """Change which kinds of email one subscriber receives."""
    if not payload.subscription_arn.startswith("arn:"):
        raise HTTPException(
            status_code=422,
            detail="That subscriber has not confirmed their email yet, so AWS has no "
                   "subscription to change. Their choices were set when they were added.",
        )
    events = _clean(payload.events)
    try:
        t2.sns.set_subscription_attributes(
            SubscriptionArn=payload.subscription_arn,
            AttributeName="FilterPolicy",
            AttributeValue=notify.filter_policy(events),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not save: {exc}")
    return {"subscription_arn": payload.subscription_arn, "events": notify.events_from_policy(
        notify.filter_policy(events))}


class RemoveSubscriptionRequest(BaseModel):
    subscription_arn: str


@router.post("/remove")
def remove_subscription(payload: RemoveSubscriptionRequest, session=Depends(require_session)) -> dict:
    if not payload.subscription_arn:
        raise HTTPException(status_code=422, detail="No subscription specified.")
    # SNS reports an unconfirmed subscriber's ARN as the literal string
    # "PendingConfirmation" -- there is no ARN until the person clicks the
    # link in their email, and no API to delete a subscription that does
    # not have one. Posting that string to Unsubscribe raises
    # InvalidParameterException, which surfaced as a bare 500 and the
    # message "Remove failed." forever, with nothing saying why.
    #
    # AWS expires an unconfirmed subscription after three days on its own,
    # so there is nothing to do but say so.
    if not payload.subscription_arn.startswith("arn:"):
        raise HTTPException(
            status_code=422,
            detail="That subscriber has not confirmed their email yet, so there is "
                   "nothing to unsubscribe -- AWS has not issued a subscription for "
                   "them. It expires on its own three days after the invitation.",
        )
    t2.sns.unsubscribe(SubscriptionArn=payload.subscription_arn)
    return {"removed": True}
