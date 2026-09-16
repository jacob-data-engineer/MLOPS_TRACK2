"""Auto-balance an eval dataset's TP/FP split per violation type, so a
skewed ground-truth mix (e.g. 71 no_violation vs. 36 violation helmet
cases) doesn't dominate precision/recall numbers with one verdict's own
base rate.
"""

from __future__ import annotations

import random

from ..models import ViolationType
from .dataset import EvalCase


def balance_by_ground_truth(
    dataset: list[EvalCase], seed: int = 42
) -> tuple[list[EvalCase], list[str]]:
    """Bucket `dataset` by `(violation_type, ground_truth_is_violation)` and
    cap each violation type's TP/FP pair to `min(len(TP_bucket),
    len(FP_bucket))`, so every violation type comes out with an equal TP/FP
    split.

    Cases with `ground_truth_is_violation is None` (no resolved ground
    truth to balance against) are dropped, with a warning. Sampling within
    each bucket is seeded and sorts by `case_id` first — same reproducibility
    idiom as `scripts/sample_gold_stratified.py` — so re-running with the
    same `seed` on the same input always drops the same cases.

    Returns `(balanced_dataset, warnings)`, where warnings are shaped like
    `"helmet: capped no_violation 71 -> 36"`.
    """
    rng = random.Random(seed)
    warnings: list[str] = []

    buckets: dict[tuple[ViolationType, bool], list[EvalCase]] = {}
    dropped = 0
    for case in dataset:
        if case.ground_truth_is_violation is None:
            dropped += 1
            continue
        buckets.setdefault((case.violation_type, case.ground_truth_is_violation), []).append(case)

    if dropped:
        warnings.append(
            f"dropped {dropped} case(s) with no resolved ground truth (ground_truth_is_violation is None)"
        )

    balanced: list[EvalCase] = []
    for violation_type in sorted({vt for vt, _ in buckets}, key=lambda v: v.value):
        tp_bucket = buckets.get((violation_type, True), [])
        fp_bucket = buckets.get((violation_type, False), [])
        target = min(len(tp_bucket), len(fp_bucket))

        for is_violation, bucket, label in (
            (True, tp_bucket, "violation"),
            (False, fp_bucket, "no_violation"),
        ):
            ordered = sorted(bucket, key=lambda c: c.case_id)
            if target < len(bucket):
                warnings.append(
                    f"{violation_type.value}: capped {label} {len(bucket)} -> {target}"
                )
            balanced.extend(rng.sample(ordered, target))

    return balanced, warnings
