"""Ground-truth evaluation harness: judge verdicts vs human-labeled truth.

- dataset.py  — load a labeled EvalCase manifest
- metrics.py  — confusion-matrix counts and precision/recall/F1
- runner.py   — run_eval, compare_zero_shot_vs_few_shot, compare_cropped_vs_uncropped,
                run_full_comparison (all 4 combinations), run_six_method_comparison
                (the 6 named eval methods)
- report.py   — format an EvalReport / ComparisonMatrix / MethodComparisonMatrix as
                plain-dict rows for tables/UI
- replay.py   — rebuild a scored EvalReport from a streamed eval_runs/*.jsonl file
                (recovering a run that was interrupted partway through)
"""

from .dataset import EvalCase, load_eval_dataset
from .metrics import ConfusionCounts, classify
from .replay import load_report_from_jsonl
from .report import (
    full_comparison_rows,
    full_comparison_token_savings,
    method_comparison_rows,
    print_report,
    report_to_rows,
    token_savings_rows,
)
from .runner import (
    CaseOutcome,
    ComparisonMatrix,
    EvalReport,
    MethodComparisonMatrix,
    build_report,
    compare_cropped_vs_uncropped,
    compare_zero_shot_vs_few_shot,
    run_eval,
    run_full_comparison,
    run_six_method_comparison,
)

__all__ = [
    "EvalCase",
    "load_eval_dataset",
    "ConfusionCounts",
    "classify",
    "CaseOutcome",
    "EvalReport",
    "ComparisonMatrix",
    "MethodComparisonMatrix",
    "run_eval",
    "build_report",
    "compare_zero_shot_vs_few_shot",
    "compare_cropped_vs_uncropped",
    "run_full_comparison",
    "run_six_method_comparison",
    "print_report",
    "report_to_rows",
    "token_savings_rows",
    "full_comparison_rows",
    "full_comparison_token_savings",
    "method_comparison_rows",
    "load_report_from_jsonl",
]
