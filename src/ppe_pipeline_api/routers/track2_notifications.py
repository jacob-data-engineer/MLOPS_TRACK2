"""Track2 reviewer-notification email subscribers -- backs
`/track2/notifications`."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from .. import track2_shared as t2
from ..auth import require_session

router = APIRouter()


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
        }
        for s in subscriptions
        if s["Protocol"] == "email"
    ]


@router.get("")
def get_notifications(session=Depends(require_session)) -> dict:
    return {"subscriptions": _list_subscriptions()}


class AddSubscriptionRequest(BaseModel):
    email: str


@router.post("/add")
def add_subscription(payload: AddSubscriptionRequest, session=Depends(require_session)) -> dict:
    email = payload.email.strip()
    if not email:
        raise HTTPException(status_code=422, detail="No email address given.")
    t2.sns.subscribe(TopicArn=t2.reviewer_notifications_topic_arn(), Protocol="email", Endpoint=email)
    return {"email": email, "message": f"Confirmation email sent to {email} -- they must click the link before notifications start."}


class RemoveSubscriptionRequest(BaseModel):
    subscription_arn: str


@router.post("/remove")
def remove_subscription(payload: RemoveSubscriptionRequest, session=Depends(require_session)) -> dict:
    if not payload.subscription_arn:
        raise HTTPException(status_code=422, detail="No subscription specified.")
    t2.sns.unsubscribe(SubscriptionArn=payload.subscription_arn)
    return {"removed": True}
