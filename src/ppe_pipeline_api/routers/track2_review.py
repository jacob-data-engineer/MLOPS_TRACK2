"""Track2 (YOLO/Track2 training pipeline) review queue + model registry
approve/reject/delete -- backs the `/track2/review` page. Ported from
`terraform/review-app/review_app.py`'s Review Queue tab.

Note: that tab's module docstring describes a richer per-flagged-image
review-card UI, but the actual rendering function only ever shows
aggregate queue counts plus the model registry sections -- this router
matches what the old app actually renders, not its stale docstring.
"""

from __future__ import annotations

from datetime import datetime, timezone

import botocore.exceptions
from fastapi import APIRouter, Depends, HTTPException

from .. import track2_shared as t2
from ..auth import require_session

router = APIRouter()


def _serialize_package(pkg: dict) -> dict:
    lineage = pkg.get("Lineage")
    return {
        "model_package_arn": pkg["ModelPackageArn"],
        "version": pkg["ModelPackageArn"].rsplit("/", 1)[-1],
        "approval_status": pkg["ModelApprovalStatus"],
        "description": pkg.get("Description", ""),
        "lineage": lineage,
        "creation_time": pkg["CreationTime"].isoformat() if pkg.get("CreationTime") else None,
    }


@router.get("")
def get_review(session=Depends(require_session)) -> dict:
    counts = t2.review_queue_counts()
    # Production groups and open-taxonomy groups in one listing, so a
    # custom-trained model is visible where models are reviewed. They stay
    # out of serialize_registries_brief, which the Deploy page reads.
    registries = {**t2.all_model_registries(), **t2.open_model_registries()}
    return {
        "counts": counts,
        "registries": {
            group: {
                "total": summary["total"],
                "approved": summary["approved"],
                "packages": [_serialize_package(p) for p in summary["packages"]],
            }
            for group, summary in registries.items()
        },
    }


@router.post("/{review_id}/approve")
def approve_review(review_id: str, session=Depends(require_session)) -> dict:
    return _decide_review(review_id, "approve")


@router.post("/{review_id}/reject")
def reject_review(review_id: str, session=Depends(require_session)) -> dict:
    return _decide_review(review_id, "reject")


def _decide_review(review_id: str, decision: str) -> dict:
    t2.table().update_item(
        Key={"review_id": review_id},
        UpdateExpression="SET #s = :status, reviewed_at = :ts",
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues={
            ":status": "approved" if decision == "approve" else "rejected",
            ":ts": datetime.now(timezone.utc).isoformat(),
        },
    )
    return {"review_id": review_id, "status": "approved" if decision == "approve" else "rejected"}


@router.post("/model/{model_package_arn:path}/approve")
def approve_model(model_package_arn: str, session=Depends(require_session)) -> dict:
    return _decide_model(model_package_arn, "Approved")


@router.post("/model/{model_package_arn:path}/reject")
def reject_model(model_package_arn: str, session=Depends(require_session)) -> dict:
    return _decide_model(model_package_arn, "Rejected")


def _decide_model(model_package_arn: str, status: str) -> dict:
    try:
        t2.sm.update_model_package(ModelPackageArn=model_package_arn,
                                   ModelApprovalStatus=status)
    except botocore.exceptions.ClientError as exc:
        # `delete_model` below has always caught this; approve and reject
        # did not, so any ClientError came straight out as a 500 with a
        # traceback behind it. The UAT pass found it with an ARN outside
        # the IAM grant -- AccessDeniedException, unhandled -- but a
        # throttle or a version deleted in another tab would have done
        # the same, and those are ordinary.
        raise HTTPException(
            status_code=400,
            detail=f"Could not mark it {status}: {exc}",
        ) from exc
    # The listing this status came out of is cached. Without this the
    # operator approves a version, the page reloads, and it still reads
    # Pending -- which looks like the approval did not take.
    t2.forget_registry_listing()
    return {"model_package_arn": model_package_arn, "approval_status": status}


@router.post("/model/{model_package_arn:path}/delete")
def delete_model(model_package_arn: str, session=Depends(require_session)) -> dict:
    try:
        # Also unlinks the run that produced it, so the history does not go
        # on claiming a version that is gone.
        return t2.delete_model_package_and_detach_audit(model_package_arn)
    except botocore.exceptions.ClientError as exc:
        raise HTTPException(status_code=400, detail=f"Delete failed: {exc}")
