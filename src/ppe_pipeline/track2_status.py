"""Read-only view onto Track2's SageMaker Model Registry, for the
`/models` screen. No model has ever actually been registered as of
this writing (see Track1's own gaps/TODO notes) — every
function here is expected to return an empty result today, not raise,
so the UI can show an honest "not yet available" state instead of a
traceback.
"""

from __future__ import annotations

from botocore.exceptions import ClientError, NoRegionError

from aiops_pipeline.config import get_config
from aiops_pipeline.models import ModelType
from aiops_pipeline.registry import DEPLOYMENT_STATUS_KEY, ModelRegistry, model_package_group_name


def list_model_versions(model_type: ModelType) -> list[dict]:
    """All Model Package versions registered in `model_type`'s Model
    Package Group, newest first — mirrors the list-then-describe pattern
    `registry.ModelRegistry.query_production_model` already uses.

    Each dict carries: `version` (SageMaker's `ModelPackageVersion`),
    `model_package_arn`, `deployment_status` (candidate/uat/production/
    rolled_back, or `None` if the custom metadata key was never set),
    `approval_status` (SageMaker's native Approved/Rejected/
    PendingManualApproval), `creation_time`, and `eval_metrics_s3_uri`
    (the `ModelQuality.Statistics.S3Uri` pointer, if present).

    Returns `[]` on any AWS error (`ClientError`, `NoRegionError`, or
    anything else) or if the Model Package Group doesn't exist yet —
    expected today, since infra/ hasn't been applied and nothing has ever
    been registered.
    """
    try:
        config = get_config()
        registry = ModelRegistry(region=config.aws.region)
        group_name = model_package_group_name(model_type)
        summaries = registry.client.list_model_packages(
            ModelPackageGroupName=group_name,
            SortBy="CreationTime",
            SortOrder="Descending",
        ).get("ModelPackageSummaryList", [])

        versions = []
        for summary in summaries:
            detail = registry.client.describe_model_package(ModelPackageName=summary["ModelPackageArn"])
            customer_metadata = detail.get("CustomerMetadataProperties", {})
            model_quality = detail.get("ModelMetrics", {}).get("ModelQuality", {}).get("Statistics", {})
            versions.append(
                {
                    "version": detail.get("ModelPackageVersion"),
                    "model_package_arn": detail.get("ModelPackageArn"),
                    "deployment_status": customer_metadata.get(DEPLOYMENT_STATUS_KEY),
                    "approval_status": detail.get("ModelApprovalStatus"),
                    "creation_time": detail.get("CreationTime"),
                    "eval_metrics_s3_uri": model_quality.get("S3Uri"),
                }
            )
        return versions
    except (ClientError, NoRegionError):
        return []
    except Exception:
        return []
