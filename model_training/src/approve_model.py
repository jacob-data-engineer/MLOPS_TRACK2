"""The human-approval gate itself. Deliberately a separate, manually-run
script from deploy_endpoint.py — same split Track2's own
docs/production_promotion.md enforces: nothing in pipeline_definition.py
ever calls this, and deploy_endpoint.py refuses to deploy anything that
isn't already Approved. Getting a model into production always requires
a person to run this explicitly, having actually looked at its
training/validation results first.

    uv run python approve_model.py <model-package-arn>
    uv run python approve_model.py --list          # find candidate ARNs, all sizes
    uv run python approve_model.py --list m        # find candidate ARNs, just size m
"""

import sys

import boto3

from config import CONFIG, SIZES, model_package_group_name

sm = boto3.client("sagemaker", region_name=CONFIG.region)


def list_candidates(size: str | None = None, generation: str = "11"):
    sizes = [size] if size else SIZES
    for s in sizes:
        packages = sm.list_model_packages(
            ModelPackageGroupName=model_package_group_name(generation, s),
            SortBy="CreationTime",
            SortOrder="Descending",
        )["ModelPackageSummaryList"]
        for pkg in packages:
            print(f"[{s}] {pkg['ModelPackageArn']}  status={pkg['ModelApprovalStatus']}  created={pkg['CreationTime']}")


def approve(model_package_arn: str):
    sm.update_model_package(
        ModelPackageArn=model_package_arn,
        ModelApprovalStatus="Approved",
    )
    print(f"Approved {model_package_arn}. Now run: uv run python deploy_endpoint.py {model_package_arn}")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        sys.exit(1)
    if sys.argv[1] == "--list":
        list_candidates(sys.argv[2] if len(sys.argv) == 3 else None)
    else:
        approve(sys.argv[1])
