"""CLI for the eval harness — hits the live Bedrock judge, so it needs
AWS_BEARER_TOKEN_BEDROCK set. Not covered by the test suite for that reason; the
underlying runner/metrics logic is (see tests/test_eval_runner.py).

    uv run python -m ppe_pipeline.eval.cli zero-shot-vs-few-shot dataset.json --few-shot-manifest manifest.json
    uv run python -m ppe_pipeline.eval.cli cropped-vs-uncropped dataset.json
    uv run python -m ppe_pipeline.eval.cli full-comparison dataset.json --few-shot-manifest manifest.json
    uv run python -m ppe_pipeline.eval.cli glove-crop-experiment dataset.json
    uv run python -m ppe_pipeline.eval.cli sync-gt-rationale eval/gold_annotations.json --eval-dir eval

--violation-type (repeatable) and --balance/--seed apply to every mode
except sync-gt-rationale, e.g.:

    uv run python -m ppe_pipeline.eval.cli zero-shot-vs-few-shot dataset.json \\
        --few-shot-manifest manifest.json --violation-type helmet --balance
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..config import load_config
from ..few_shot import load_few_shot_library
from ..judge_client import JudgeClient
from ..models import ViolationType
from .balance import balance_by_ground_truth
from .dataset import filter_by_violation_type, load_eval_dataset
from .rationale_sync import sync_ground_truth_rationale
from .report import (
    full_comparison_rows,
    full_comparison_token_savings,
    glove_crop_experiment_rows,
    print_report,
    token_savings_rows,
)
from .runner import (
    compare_cropped_vs_uncropped,
    compare_zero_shot_vs_few_shot,
    run_full_comparison,
    run_glove_crop_experiment,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PPE judge eval harness against ground truth.")
    parser.add_argument(
        "mode",
        choices=[
            "zero-shot-vs-few-shot",
            "cropped-vs-uncropped",
            "full-comparison",
            "glove-crop-experiment",
            "sync-gt-rationale",
        ],
        help="Which comparison to run. full-comparison runs all 4 combinations "
        "of {zero-shot, few-shot} x {cropped, uncropped} — 4x the judge calls. "
        "glove-crop-experiment runs only the dataset's glove cases through 3 crop-pad "
        "variants (tight/wide/tight+wide) — 3x the judge calls, glove cases only. "
        "sync-gt-rationale doesn't hit the judge at all — it just propagates "
        "resolved gold-set rationales into the eval dataset files.",
    )
    parser.add_argument(
        "dataset",
        type=Path,
        help="Path to an eval dataset manifest (see eval/dataset.py), or to "
        "gold_annotations.json for sync-gt-rationale",
    )
    parser.add_argument(
        "--few-shot-manifest", type=Path, default=None, help="Required for zero-shot-vs-few-shot and full-comparison"
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to a config.toml (default: ./config.toml)")
    parser.add_argument(
        "--eval-dir", type=Path, default=None, help="Directory of eval dataset *.json files to sync into "
        "(required for sync-gt-rationale)"
    )
    parser.add_argument(
        "--violation-type", action="append", dest="violation_types", choices=[v.value for v in ViolationType],
        default=None, help="Restrict the dataset to this violation type — repeatable to include several. "
        "Omit to include every violation type in the dataset (no filtering).",
    )
    parser.add_argument(
        "--balance", action="store_true",
        help="Auto-balance the (filtered) dataset's TP/FP split per violation type before running "
        "(see eval/balance.py::balance_by_ground_truth).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for --balance's sampling (default: 42).",
    )
    args = parser.parse_args()

    if not args.dataset.is_file():
        print(f"error: file not found: {args.dataset}", file=sys.stderr)
        raise SystemExit(1)

    if args.mode == "sync-gt-rationale":
        if args.eval_dir is None:
            print("error: --eval-dir is required for sync-gt-rationale", file=sys.stderr)
            raise SystemExit(1)
        updated = sync_ground_truth_rationale(args.dataset, args.eval_dir)
        if not updated:
            print("No files needed updating.")
        else:
            print(f"Updated {len(updated)} file(s):")
            for path in updated:
                print(f"  {path}")
        return

    needs_few_shot = args.mode in ("zero-shot-vs-few-shot", "full-comparison")
    if needs_few_shot and args.few_shot_manifest is None:
        print(f"error: --few-shot-manifest is required for {args.mode}", file=sys.stderr)
        raise SystemExit(1)

    config = load_config(args.config)
    dataset = load_eval_dataset(args.dataset)

    if args.violation_types:
        dataset = filter_by_violation_type(dataset, [ViolationType(v) for v in args.violation_types])
    if args.balance:
        dataset, balance_warnings = balance_by_ground_truth(dataset, seed=args.seed)
        for warning in balance_warnings:
            print(f"balance: {warning}")

    if args.mode == "zero-shot-vs-few-shot":
        library = load_few_shot_library(args.few_shot_manifest)
        zero_shot_client = JudgeClient(config=config)
        few_shot_client = JudgeClient(few_shot_library=library, config=config)
        zero_shot_report, few_shot_report = compare_zero_shot_vs_few_shot(
            dataset, zero_shot_client, few_shot_client, config=config
        )
        print_report(zero_shot_report)
        print_report(few_shot_report)

    elif args.mode == "cropped-vs-uncropped":
        judge_client = JudgeClient(config=config)
        cropped_report, uncropped_report = compare_cropped_vs_uncropped(dataset, judge_client, config=config)
        print_report(cropped_report)
        print_report(uncropped_report)
        print("\n=== token savings from cropping ===")
        for row in token_savings_rows(cropped_report, uncropped_report):
            print(row)

    elif args.mode == "glove-crop-experiment":
        judge_client = JudgeClient(config=config)
        results = run_glove_crop_experiment(dataset, judge_client, config=config)

        for report in results.values():
            print_report(report)

        print("\n=== combined table (tight / wide / tight+wide) ===")
        for row in glove_crop_experiment_rows(results):
            print(row)

    else:  # full-comparison
        library = load_few_shot_library(args.few_shot_manifest)
        zero_shot_client = JudgeClient(config=config)
        few_shot_client = JudgeClient(few_shot_library=library, config=config)
        matrix = run_full_comparison(dataset, zero_shot_client, few_shot_client, config=config)

        for report in matrix.as_dict().values():
            print_report(report)

        print("\n=== combined table (all 4 combinations) ===")
        for row in full_comparison_rows(matrix):
            print(row)

        print("\n=== token savings from cropping, per shot type ===")
        for row in full_comparison_token_savings(matrix):
            print(row)


if __name__ == "__main__":
    main()
