"""Push the SageMaker pipeline definitions and the processing code live.

`deploy.sh` does NOT do this, and cannot: `model_training/*` is excluded
from the deploy archive, so the JSON that the app reads in this repo and
the definition SageMaker actually executes are two separate things with
nothing reconciling them. A fix made only in the file is a silent no-op,
which is how the evaluation step went months without the class filter it
needed -- the repo said one thing and the running pipeline another, and
neither ever disagreed out loud.

Also uploads the processing scripts, which the pipeline reads from S3 by
key at step start. Same reason: they are not in the image either.

Idempotent. Run it after changing anything under
`model_training/custom_training/`:

    .venv/Scripts/python.exe scripts/deploy_pipelines.py          # show the diff
    .venv/Scripts/python.exe scripts/deploy_pipelines.py --apply  # land it
"""
from __future__ import annotations

import argparse
import difflib
import json
import pathlib
import sys

import boto3

REGION = "ap-southeast-1"
HERE = pathlib.Path(__file__).resolve().parents[1]
CUSTOM = HERE / "model_training" / "custom_training"

#: definition file -> live pipeline name.
PIPELINES = {
    CUSTOM / "pipeline_open.json": "track2-yolo-open-training-pipeline",
}

#: DRIFTED, and deliberately not synced. The live
#: `track2-yolo-custom-training-pipeline` differs from the file in ways
#: nobody here chose: it still entry-points `evaluate.py` rather than
#: `evaluate_any.py`, and two parameter defaults differ. Pushing the file
#: over it to land an unrelated change would swap its evaluation script
#: as a side effect. Same rule as the platform Terraform stack -- make
#: the targeted change, do not apply the whole drifted definition.
#: Pass --include-drifted once somebody has reconciled it by hand.
DRIFTED = {
    CUSTOM / "pipeline.json": "track2-yolo-custom-training-pipeline",
}

#: local file -> the exact S3 uri the pipeline definition names
CODE = {
    CUSTOM / "evaluate_any.py": "s3://track2-s3/custom/code/evaluate_any.py",
}


def _canonical(text: str) -> str:
    """Compared as parsed JSON, so indentation is never a 'change'."""
    return json.dumps(json.loads(text), indent=2, sort_keys=True)


def sync_pipelines(sm, apply: bool, targets: dict) -> int:
    changed = 0
    for path, name in targets.items():
        if not path.exists():
            print(f"  skip {name}: {path} is not in the tree")
            continue
        wanted = path.read_text(encoding="utf-8")
        try:
            live = sm.describe_pipeline(PipelineName=name)["PipelineDefinition"]
        except sm.exceptions.ResourceNotFound:
            print(f"  {name}: does not exist -- create it before syncing")
            continue
        if _canonical(live) == _canonical(wanted):
            print(f"  {name}: already matches")
            continue
        changed += 1
        print(f"  {name}: DIFFERS")
        for line in list(difflib.unified_diff(
                _canonical(live).splitlines(), _canonical(wanted).splitlines(),
                "live", str(path.name), lineterm=""))[:60]:
            print("    " + line)
        if apply:
            sm.update_pipeline(PipelineName=name, PipelineDefinition=wanted)
            print(f"  {name}: updated")
    return changed


def sync_code(s3, apply: bool) -> int:
    changed = 0
    for path, uri in CODE.items():
        bucket, key = uri.removeprefix("s3://").split("/", 1)
        wanted = path.read_bytes()
        try:
            live = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        except Exception:
            live = None
        if live == wanted:
            print(f"  {key}: already matches")
            continue
        changed += 1
        print(f"  {key}: DIFFERS ({len(live or b'')} -> {len(wanted)} bytes)")
        if apply:
            s3.put_object(Bucket=bucket, Key=key, Body=wanted)
            print(f"  {key}: uploaded")
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="actually write. Without it this only reports.")
    parser.add_argument("--include-drifted", action="store_true",
                        help="also sync pipelines listed in DRIFTED. Read why "
                             "they are listed there before using this.")
    args = parser.parse_args()
    targets = dict(PIPELINES)
    if args.include_drifted:
        targets.update(DRIFTED)
    else:
        for path, name in DRIFTED.items():
            print(f"  {name}: SKIPPED -- drifted from {path.name}; see DRIFTED in this file")

    sm = boto3.client("sagemaker", region_name=REGION)
    s3 = boto3.client("s3", region_name=REGION)

    print("Pipeline definitions:")
    changed = sync_pipelines(sm, args.apply, targets)
    print("Processing code:")
    changed += sync_code(s3, args.apply)

    if not changed:
        print("\nEverything already matches.")
    elif args.apply:
        print(f"\n{changed} item(s) updated.")
    else:
        print(f"\n{changed} item(s) differ. Re-run with --apply to land them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
