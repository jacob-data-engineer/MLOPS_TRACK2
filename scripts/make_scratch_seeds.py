"""Build the random-init seeds that "train from scratch" starts from.

`make_scratch_checkpoint.py` has to run INSIDE `track2-yolo-train`, not on
a workstation: an Ultralytics .pt is a pickled live nn.Module, so the
writer's library versions must match the reader's. The training image has
ultralytics 8.4.x on torch 2.3/py3.11; a laptop pip-install has neither,
and the resulting file fails to unpickle in the job with a
ModuleNotFoundError that looks nothing like a version problem.

This launches it as a SageMaker Processing job, one per architecture. The
script runs on CPU and takes a couple of minutes, so it does not touch the
account's single GPU slot.

    .venv/Scripts/python.exe scripts/make_scratch_seeds.py --arch yolo26s
    .venv/Scripts/python.exe scripts/make_scratch_seeds.py --generation 26

Seeds are immutable on purpose. The script refuses to overwrite without
`--overwrite`: its sha256 is what makes two scratch runs months apart
provably identical, and replacing one silently would break that.
"""
from __future__ import annotations

import argparse
import sys
import time

import boto3

REGION = "ap-southeast-1"
BUCKET = "track2-s3"
IMAGE = "125192671932.dkr.ecr.ap-southeast-1.amazonaws.com/track2-yolo-train:train-latest"
ROLE = "arn:aws:iam::125192671932:role/track2-yolo-sagemaker-execution"
CODE_KEY = "custom/code/make_scratch_checkpoint.py"
SIZES = ("s", "m", "l", "x")


def already_there(s3, arch: str) -> bool:
    try:
        s3.head_object(Bucket=BUCKET, Key=f"external_models/scratch/{arch}/model.pt")
        return True
    except Exception:
        return False


def launch(sm, arch: str, overwrite: bool) -> str:
    name = f"scratch-seed-{arch}-{int(time.time())}"
    arguments = ["--arch", arch, "--bucket", BUCKET, "--region", REGION]
    if overwrite:
        arguments.append("--overwrite")
    sm.create_processing_job(
        ProcessingJobName=name,
        AppSpecification={
            "ImageUri": IMAGE,
            "ContainerEntrypoint": ["python3", "/opt/ml/processing/input/code/make_scratch_checkpoint.py"],
            "ContainerArguments": arguments,
        },
        RoleArn=ROLE,
        ProcessingResources={"ClusterConfig": {
            # CPU. Building a randomly-initialised model is a few seconds
            # of torch; the GPU slot is for training.
            "InstanceType": "ml.m5.xlarge", "InstanceCount": 1, "VolumeSizeInGB": 30,
        }},
        StoppingCondition={"MaxRuntimeInSeconds": 1800},
        ProcessingInputs=[{
            "InputName": "code",
            "S3Input": {
                "S3Uri": f"s3://{BUCKET}/{CODE_KEY}",
                "LocalPath": "/opt/ml/processing/input/code",
                "S3DataType": "S3Prefix", "S3InputMode": "File",
                "S3DataDistributionType": "FullyReplicated",
            },
        }],
    )
    return name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", action="append", default=[],
                        help="e.g. yolo26s. Repeatable.")
    parser.add_argument("--generation", help="Build every size of one generation, e.g. 26.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace an existing seed. Read the note above first.")
    parser.add_argument("--apply", action="store_true",
                        help="Actually launch. Without it this only reports.")
    args = parser.parse_args()

    archs = list(args.arch)
    if args.generation:
        archs += [f"yolo{args.generation}{size}" for size in SIZES]
    if not archs:
        parser.error("Pass --arch or --generation.")

    s3 = boto3.client("s3", region_name=REGION)
    sm = boto3.client("sagemaker", region_name=REGION)

    # The script the job runs has to be in S3 where the job can read it.
    s3.upload_file("model_training/custom_training/make_scratch_checkpoint.py",
                   BUCKET, CODE_KEY)
    print(f"uploaded s3://{BUCKET}/{CODE_KEY}")

    for arch in archs:
        if already_there(s3, arch) and not args.overwrite:
            print(f"  {arch}: seed already exists -- skipping")
            continue
        if not args.apply:
            print(f"  {arch}: WOULD launch a processing job")
            continue
        print(f"  {arch}: launched {launch(sm, arch, args.overwrite)}")

    if not args.apply:
        print("\nNothing launched. Re-run with --apply.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
