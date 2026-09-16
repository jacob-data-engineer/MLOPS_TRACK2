"""PPE & safety violation verification pipeline.

See README.md for the architecture this implements. `main()` is a demo CLI
that runs the pipeline against one image using mock detections (real
YOLOv8 annotations aren't wired in yet — see mock_data.py).
"""

import argparse
import sys
from pathlib import Path

from .config import load_config
from .few_shot import load_few_shot_library
from .judge_client import JudgeClient
from .mock_data import generate_mock_detections
from .pipeline import run_pipeline

__all__ = ["main"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the PPE verification pipeline on one image using mock detections."
    )
    parser.add_argument(
        "image",
        type=Path,
        help="Path to a hazard image, e.g. ../resources/hazard_images/H-260701-0002.jpg",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a config.toml overriding pipeline parameters (default: ./config.toml)",
    )
    parser.add_argument(
        "--few-shot-manifest",
        type=Path,
        default=None,
        help="Path to a few-shot example manifest.json (see few_shot.py); omit to run zero-shot",
    )
    args = parser.parse_args()

    if not args.image.is_file():
        print(f"error: image not found: {args.image}", file=sys.stderr)
        raise SystemExit(1)

    config = load_config(args.config)
    few_shot_library = load_few_shot_library(args.few_shot_manifest) if args.few_shot_manifest else None

    from PIL import Image

    with Image.open(args.image) as image:
        image_size = image.size

    detections = generate_mock_detections(image_size)
    judge_client = JudgeClient(few_shot_library=few_shot_library, config=config)
    result = run_pipeline(args.image, detections, judge_client, config=config)

    print(f"person -> violations: {result.person_to_violations}")
    print(f"violation -> people:  {result.violation_to_people}")
    print()

    for person in result.people:
        all_results = (
            person.processed.true_positives
            + person.processed.false_positives
            + person.processed.uncertain
        )
        for r in all_results:
            print(
                f"  person {person.cluster.person_id} [{r.violation_type.value}] "
                f"-> {r.verdict} ({r.reason}, {r.confidence}): {r.explanation}"
            )

    if result.review_queue:
        print(f"\n{len(result.review_queue)} entr(ies) sent to human review queue")
