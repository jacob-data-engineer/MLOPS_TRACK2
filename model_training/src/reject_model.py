"""The explicit-discard companion to approve_model.py. Same shape,
opposite status — keeps the parked list legible without deleting
history (see brainstorming.md's "park -> train -> evaluate -> keep").

    uv run python reject_model.py <model-package-arn>
    uv run python reject_model.py --list          # all sizes
    uv run python reject_model.py --list m        # just size m
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


def reject(model_package_arn: str):
    sm.update_model_package(
        ModelPackageArn=model_package_arn,
        ModelApprovalStatus="Rejected",
    )
    print(f"Rejected {model_package_arn}.")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        sys.exit(1)
    if sys.argv[1] == "--list":
        list_candidates(sys.argv[2] if len(sys.argv) == 3 else None)
    else:
        reject(sys.argv[1])
