"""Deploys an approved Model Package to a SageMaker Asynchronous Inference
endpoint (instance-based, scale-to-zero after idle -- see infra/autoscaling.tf).

Why not Serverless Inference: SageMaker Serverless Inference rejects any
container image over 10GB (confirmed via a live CreateEndpointConfig
ValidationException: "Image size 14572534568 is greater than supported size
10737418240") -- this stack's serve-latest image is ~14.57GB, so serverless
is not usable here at all, regardless of concurrency/memory settings.

Why not a plain real-time endpoint: a classic ProductionVariant endpoint
cannot register an Application Auto Scaling target with min_capacity=0
(confirmed via a live API ValidationException on a sibling project) -- it's
floored at 1 always-running instance, billing continuously. Async Inference
is the one architecture that supports both a container over 10GB AND
scale-to-zero. The tradeoff: callers must upload the request to S3 and call
invoke_endpoint_async + poll for the result, instead of invoke_endpoint.

Data capture is NOT enabled here (unlike the original serverless attempt):
confidence_monitor.py's drift detection has nothing to read as a result.
Restoring that would mean either wiring DataCaptureConfig back in (Async
Inference DOES support it, unlike Serverless -- this could be revisited)
or an explicit log-to-S3 step wrapping every invocation.

This is deliberately a script, not a Terraform resource — it needs a real
model package ARN to exist first, which only happens after a training run
completes and (per docs/production_promotion.md-equivalent judgment) is
approved. See promote_and_deploy.py for the human-approval-gated wrapper
around this.

    uv run python deploy_endpoint.py <model-package-arn>
"""

import sys
from datetime import datetime, timezone

import boto3

from config import CONFIG

sm = boto3.client("sagemaker", region_name=CONFIG.region)

ENDPOINT_NAME = f"{CONFIG.project}-endpoint"
INSTANCE_TYPE = "ml.g4dn.xlarge"
ASYNC_OUTPUT_S3_URI = f"s3://{CONFIG.pipeline_bucket}/async-output/"
ASYNC_FAILURE_S3_URI = f"s3://{CONFIG.pipeline_bucket}/async-failure/"


def _timestamp_suffix() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def deploy(model_package_arn: str):
    desc = sm.describe_model_package(ModelPackageName=model_package_arn)
    if desc["ModelApprovalStatus"] != "Approved":
        raise RuntimeError(
            f"Refusing to deploy: ModelApprovalStatus is "
            f"{desc['ModelApprovalStatus']!r}, not Approved. "
            "A human must approve this package first — see "
            "docs/production_promotion.md."
        )

    # Timestamped, not fixed: SageMaker Models and EndpointConfigs are
    # immutable, so a fixed name collides with whatever the previous run
    # created (confirmed live: a stray Failed endpoint's EndpointConfig
    # blocked a plain re-run here) -- this makes every re-run safe.
    suffix = _timestamp_suffix()
    model_name = f"{CONFIG.project}-model-{suffix}"
    endpoint_config_name = f"{CONFIG.project}-endpoint-config-{suffix}"

    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={"ModelPackageName": model_package_arn},
        ExecutionRoleArn=CONFIG.sagemaker_execution_role_arn,
    )

    sm.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "VariantName": "primary",
                "ModelName": model_name,
                "InstanceType": INSTANCE_TYPE,
                "InitialInstanceCount": 1,
            }
        ],
        AsyncInferenceConfig={
            "OutputConfig": {
                "S3OutputPath": ASYNC_OUTPUT_S3_URI,
                "S3FailurePath": ASYNC_FAILURE_S3_URI,
            },
            "ClientConfig": {"MaxConcurrentInvocationsPerInstance": 1},
        },
    )

    # A Failed endpoint still counts as "existing" here -- UpdateEndpoint is
    # the correct recovery path for it (same as any healthy endpoint), not
    # something that needs deleting and recreating first.
    existing = sm.list_endpoints(NameContains=ENDPOINT_NAME)["Endpoints"]
    if any(e["EndpointName"] == ENDPOINT_NAME for e in existing):
        sm.update_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=endpoint_config_name)
        print(f"Updating existing endpoint {ENDPOINT_NAME} to {model_package_arn} (config: {endpoint_config_name})")
    else:
        sm.create_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=endpoint_config_name)
        print(f"Creating endpoint {ENDPOINT_NAME} serving {model_package_arn} (config: {endpoint_config_name})")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: deploy_endpoint.py <model-package-arn>")
        sys.exit(1)
    deploy(sys.argv[1])
