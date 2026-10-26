"""Builds a randomly-initialised YOLO checkpoint -- the seed for a
"train from scratch" run.

Why this exists at all: every training path in this estate fine-tunes.
`train.py` requires exactly one of --base-model-package-arn or
--base-model-s3-uri and always does `YOLO(checkpoint).train(...)`, so there
is no argument that means "start from nothing". Rather than change a
training script that lives in a digest-pinned image nothing in this repo
builds, scratch is expressed as a base model whose weights happen to be
random. The trainer then does exactly what it always does.

Run this INSIDE track2-yolo-train, as a SageMaker Processing job -- not on
a workstation. An Ultralytics .pt is a pickled live nn.Module, so the
writer's library versions must match the reader's. The training image has
ultralytics 8.3.x on torch 2.3/py3.11; a laptop pip-install has neither,
and the resulting file can fail to unpickle in the job with a
ModuleNotFoundError that looks nothing like a version problem.

The seed's own class names are set to the production pair deliberately.
They have no effect on the trained model -- the trainer rebuilds the head
for whatever the dataset declares -- but the digest-pinned image asserts
`model.names == {0: "person", 1: "helmet"}` on its base checkpoint, so a
seed carrying them is usable on BOTH pipelines instead of only the open
one. Ultralytics writes {0: "0", 1: "1", ...} for a model built from a
bare .yaml, which would fail that assertion.

    python3 make_scratch_checkpoint.py --arch yolo11s --bucket track2-s3
"""
import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone

import boto3

NAMES = {0: "person", 1: "helmet"}
PREFIX = "external_models/scratch"


def build(arch: str, out_path: str) -> dict:
    import torch
    import ultralytics
    import yaml as pyyaml
    from ultralytics import YOLO
    from ultralytics.nn.tasks import yaml_model_load

    # Strips the scale letter and finds the family config; no network.
    cfg_dict = yaml_model_load(f"{arch}.yaml")
    cfg_dict["nc"] = len(NAMES)
    cfg_path = os.path.join(os.path.dirname(out_path), f"{arch}-scratch.yaml")
    # The filename keeps the scale letter, which is how Ultralytics recovers
    # "s" from "yolo11s" when it re-reads this config.
    with open(cfg_path, "w") as handle:
        pyyaml.safe_dump(cfg_dict, handle)

    model = YOLO(cfg_path)
    model.model.names = dict(NAMES)
    model.save(out_path)

    # Reload and assert the exact comparison the pinned image performs,
    # here rather than 40 minutes into a training job.
    reloaded = YOLO(out_path)
    if reloaded.names != NAMES:
        raise SystemExit(f"seed reloaded with names {reloaded.names}, expected {NAMES}")

    digest = hashlib.sha256(open(out_path, "rb").read()).hexdigest()
    return {
        "architecture": arch,
        "nc": len(NAMES),
        "names": {str(k): v for k, v in NAMES.items()},
        "weights": "random-init",
        "ultralytics_version": ultralytics.__version__,
        "torch_version": torch.__version__,
        "sha256": digest,
        "size_bytes": os.path.getsize(out_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="yolo11s")
    parser.add_argument("--bucket", default="track2-s3")
    parser.add_argument("--region", default="ap-southeast-1")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace an existing seed. Off by default: a seed is an "
                             "immutable input to reproducible runs.")
    args = parser.parse_args()

    s3 = boto3.client("s3", region_name=args.region)
    key = f"{PREFIX}/{args.arch}/model.pt"
    if not args.overwrite:
        try:
            s3.head_object(Bucket=args.bucket, Key=key)
            raise SystemExit(f"s3://{args.bucket}/{key} already exists; pass --overwrite to replace it.")
        except s3.exceptions.ClientError:
            pass

    with tempfile.TemporaryDirectory() as work:
        out_path = os.path.join(work, "model.pt")
        manifest = build(args.arch, out_path)
        s3.upload_file(out_path, args.bucket, key)
        s3.put_object(
            Bucket=args.bucket, Key=f"{PREFIX}/{args.arch}/manifest.json",
            Body=(json.dumps(manifest, indent=2) + "\n").encode(),
            ContentType="application/json",
        )
    print(json.dumps({"uploaded": f"s3://{args.bucket}/{key}", **manifest}, indent=2))


if __name__ == "__main__":
    main()
