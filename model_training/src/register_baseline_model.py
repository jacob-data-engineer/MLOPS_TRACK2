"""Registers a pretrained (COCO) Ultralytics checkpoint directly into the
Model Registry as a baseline — no training involved. Gives a real,
working (if generic-object, not hazard-specific) reference point to
compare fine-tuned candidates against, available immediately rather than
waiting on a training run. See brainstorming.md's "park -> train ->
evaluate -> keep" writeup for why this is worth having.

Registers via create_model_package directly (not through the Pipeline's
RegisterModel step — that's for pipeline-produced models specifically).

Size-aware: a brand-new generation/size's Model Package Group starts out
completely empty, and pipeline_definition.py refuses to train one with no
Approved package to fine-tune from at all -- so this is how it gets its
very first, un-trained starting point before training for it can happen
for the first time. Matches the Ultralytics checkpoint name to the size by
default (yolo11s.pt for size s, etc.) -- YOLO11 onwards only, same policy
as review_app.py's generation dropdown -- override with --checkpoint if
you want a different starting point.

Registerable sizes (REGISTERABLE_SIZES below) intentionally include Nano,
unlike config.py's SIZES (the four sizes with a live serving Inference
Component) and GENERATION_SIZES (the sizes selectable in the web page's
training form) -- a baseline can exist in the registry for reference/
future use without being deployable or trainable-via-the-form yet.

    uv run python register_baseline_model.py [--generation 11] [--size n|s|m|l|x] [--checkpoint <ultralytics-name>]
"""

import argparse
import tarfile
from datetime import datetime, timezone

import boto3
from ultralytics import YOLO

from config import CONFIG, model_package_group_name
from pipeline_definition import DYNAMODB_LINEAGE_TABLE

s3 = boto3.client("s3", region_name=CONFIG.region)
sm = boto3.client("sagemaker", region_name=CONFIG.region)
dynamodb = boto3.resource("dynamodb", region_name=CONFIG.region)

REGISTERABLE_SIZES = ("n", "s", "m", "l", "x")


def package_and_upload(generation: str, size: str, checkpoint: str) -> str:
    YOLO(checkpoint)  # auto-downloads to cwd if not cached

    tarball_path = f"{checkpoint.removesuffix('.pt')}-baseline.tar.gz"
    with tarfile.open(tarball_path, "w:gz") as tar:
        tar.add(checkpoint, arcname="model.pt")

    baseline_key = f"model-registry/{checkpoint.removesuffix('.pt')}-baseline-{generation}{size}/model.tar.gz"
    s3.upload_file(tarball_path, CONFIG.pipeline_bucket, baseline_key)
    return f"s3://{CONFIG.pipeline_bucket}/{baseline_key}"


def _write_baseline_lineage(generation: str, size: str, checkpoint: str, model_data_url: str, approved: bool) -> str:
    """Same DYNAMODB_LINEAGE_TABLE pipeline_definition.py's
    _write_run_metadata() writes to, so review_app.py's lineage lookup
    (package -> CustomerMetadataProperties.lineage_id -> get_item) works
    identically for a baseline as for a real training run -- "store all
    the information" means every registration path, not just pipeline
    ones. No S3 copy here (unlike the pipeline path): there's no free-text
    notes field or dataset to lose track of, just the checkpoint name
    already visible in package_and_upload()'s own S3 key."""
    lineage_id = f"baseline-yolo{generation}-{size}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    item = {
        "id": lineage_id,
        "record_type": "baseline",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "yolo": {
            "generation": generation,
            "size": size,
            "model_package_group_name": model_package_group_name(generation, size),
        },
        "checkpoint": checkpoint,
        "model_data_url": model_data_url,
        "approved_immediately": approved,
    }
    dynamodb.Table(DYNAMODB_LINEAGE_TABLE).put_item(Item=item)
    return lineage_id


def register(generation: str, size: str, checkpoint: str, model_data_url: str, approve: bool = False) -> str:
    lineage_id = _write_baseline_lineage(generation, size, checkpoint, model_data_url, approved=approve)
    response = sm.create_model_package(
        ModelPackageGroupName=model_package_group_name(generation, size),
        InferenceSpecification={
            "Containers": [{"Image": CONFIG.serving_image_uri, "ModelDataUrl": model_data_url}],
            "SupportedContentTypes": ["image/jpeg"],
            "SupportedResponseMIMETypes": ["application/json"],
        },
        ModelApprovalStatus="Approved" if approve else "PendingManualApproval",
        CustomerMetadataProperties={"source": "pretrained-coco-baseline", "lineage_id": lineage_id},
    )
    return response["ModelPackageArn"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation", default="11", help="YOLO generation (default: 11)")
    parser.add_argument("--size", choices=REGISTERABLE_SIZES, default="s")
    parser.add_argument("--checkpoint", default=None, help="Ultralytics checkpoint name, e.g. yolo11m.pt")
    parser.add_argument("--approve", action="store_true",
                         help="Register as Approved immediately instead of PendingManualApproval")
    args = parser.parse_args()
    checkpoint = args.checkpoint or f"yolo{args.generation}{args.size}.pt"

    uri = package_and_upload(args.generation, args.size, checkpoint)
    print(f"Uploaded baseline weights to {uri}")
    arn = register(args.generation, args.size, checkpoint, uri, approve=args.approve)
    status = "Approved" if args.approve else "PendingManualApproval"
    print(f"Registered ({status}): {arn}")
