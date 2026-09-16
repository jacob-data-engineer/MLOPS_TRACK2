"""Formats EvalReport objects as plain dict rows — reused by the CLI
(printed as a table) and the API (serialized straight to the UI's tables)."""

from __future__ import annotations

from dataclasses import dataclass

from ..models import ViolationType
from .metrics import ConfusionCounts
from .runner import CaseOutcome, ComparisonMatrix, EvalReport, MethodComparisonMatrix


def report_to_rows(report: EvalReport) -> list[dict]:
    """One row per violation type: label, sample size, metrics, and token
    usage — everything needed for a comparison table. Also includes each
    violation type's mean reasoning-alignment score (`reasoning_alignment_mean`)
    over cases with a resolved GT rationale, alongside how many of that
    violation type's cases were actually scored vs total — an unscored case
    (no resolved GT rationale) must never silently read as a 0 in the mean."""
    rows = []
    for violation_type in sorted(report.counts, key=lambda v: v.value):
        counts = report.counts[violation_type]
        usage = report.usage.get(violation_type)
        type_outcomes = [co for co in report.case_outcomes if co.case.violation_type == violation_type]
        replayed_scores = report.replayed_reasoning_scores.get(violation_type)
        rows.append(
            {
                "label": report.label,
                "violation_type": violation_type.value,
                "n": counts.total,
                "precision": round(counts.precision, 3),
                "recall": round(counts.recall, 3),
                "f1": round(counts.f1, 3),
                "accuracy": round(counts.accuracy, 3),
                "uncertain_rate": round(counts.uncertain_rate, 3),
                # How the judge actually answered, split by what the case
                # really was. `n` and the rate columns say how OFTEN it was
                # right; these say WHICH WAY it went when it was wrong --
                # a recall of 0.8 hides whether the missing 20% was cleared
                # outright or sent to a human, and those are very different
                # failures on a safety violation.
                "ground_truth_violation_n": counts.ground_truth_violation_total,
                "violation_distribution": counts.violation_verdict_distribution,
                "ground_truth_no_violation_n": counts.ground_truth_no_violation_total,
                "no_violation_distribution": counts.no_violation_verdict_distribution,
                "raw_cv_precision": round(report.raw_cv_precision.get(violation_type, 0.0), 3),
                "input_tokens": usage.input_tokens if usage else 0,
                "output_tokens": usage.output_tokens if usage else 0,
                "cache_read_tokens": usage.cache_read_input_tokens if usage else 0,
                **(
                    reasoning_alignment_summary(type_outcomes)
                    if type_outcomes or replayed_scores is None
                    else _reasoning_summary(replayed_scores)
                ),
            }
        )
    return rows


def _reasoning_summary(scores: list[float | None]) -> dict:
    """Shared shape for both sources of reasoning scores — the live run's
    `CaseOutcome`s and a replayed jsonl's bare score list.

    The mean is over cases that actually got scored (a resolved GT rationale
    existed), so an unscored case never drags the mean toward 0; it is still
    counted in the total, so it can never vanish silently either."""
    scored = [score for score in scores if score is not None]
    return {
        "reasoning_alignment_mean": round(sum(scored) / len(scored), 3) if scored else None,
        "reasoning_scored_count": len(scored),
        "reasoning_total_count": len(scores),
    }


def reasoning_alignment_summary(case_outcomes: list[CaseOutcome]) -> dict:
    """Mean `reasoning_score` plus scored/total counts over `case_outcomes`."""
    return _reasoning_summary([co.reasoning_score for co in case_outcomes])


def case_outcome_rows(case_outcomes: list[CaseOutcome]) -> list[dict]:
    """One row per case — case id, violation type, verdict/outcome, GT
    rationale, and the LLM's own explanation plus its reasoning-alignment
    score/critique, for a per-case detail table alongside the aggregate
    `report_to_rows` summary."""
    return [
        {
            "case_id": co.case.case_id,
            "violation_type": co.case.violation_type.value,
            "ground_truth": "TP" if co.case.ground_truth_is_violation else "FP",
            "verdict": co.verdict,
            "outcome": co.outcome,
            "confidence": co.confidence,
            "explanation": co.explanation,
            "ground_truth_rationale": co.case.ground_truth_rationale,
            "reasoning_score": co.reasoning_score,
            "reasoning_critique": co.reasoning_critique,
        }
        for co in case_outcomes
    ]


@dataclass
class ForkliftFilterStats:
    """Splits too_near_forklift cases into what the geometric proximity
    filter decided outright (`filter_counts`, `reached_judge=False`) versus
    what actually reached the judge (`judge_counts`, `reached_judge=True`)
    — so a bad end-to-end too_near_forklift number can be traced back to
    whichever stage is actually wrong, instead of one blended figure."""

    filter_counts: ConfusionCounts
    judge_counts: ConfusionCounts


def forklift_filter_stats(case_outcomes: list[CaseOutcome]) -> ForkliftFilterStats:
    """Build `ForkliftFilterStats` from a report's `case_outcomes`, over
    only its too_near_forklift cases (other violation types never set
    `reached_judge=False`, so they'd just land entirely in `judge_counts`
    anyway, but filtering to the right violation type keeps this useful
    when a report mixes violation types)."""
    filter_counts = ConfusionCounts()
    judge_counts = ConfusionCounts()
    for co in case_outcomes:
        if co.case.violation_type is not ViolationType.TOO_NEAR_FORKLIFT:
            continue
        if co.reached_judge:
            judge_counts.record(co.outcome, co.case.ground_truth_is_violation)
        else:
            filter_counts.record(co.outcome, co.case.ground_truth_is_violation)
    return ForkliftFilterStats(filter_counts=filter_counts, judge_counts=judge_counts)


def forklift_filter_stats_rows(stats: ForkliftFilterStats) -> list[dict]:
    """One row per stage (filter, judge) — sample size and confusion-matrix
    metrics for each, dataframe-ready like `report_to_rows`."""
    rows = []
    for stage, counts in (("filter", stats.filter_counts), ("judge", stats.judge_counts)):
        rows.append(
            {
                "stage": stage,
                "n": counts.total,
                "precision": round(counts.precision, 3),
                "recall": round(counts.recall, 3),
                "f1": round(counts.f1, 3),
                "accuracy": round(counts.accuracy, 3),
            }
        )
    return rows


def token_savings_rows(cropped: EvalReport, uncropped: EvalReport) -> list[dict]:
    """Per-violation-type input-token comparison between the cropped and
    uncropped runs of the same dataset — the basis for the "how many
    tokens does cropping save" figure."""
    rows = []
    for violation_type in sorted(cropped.usage, key=lambda v: v.value):
        cropped_usage = cropped.usage[violation_type]
        uncropped_usage = uncropped.usage.get(violation_type)
        if uncropped_usage is None or uncropped_usage.input_tokens == 0:
            saved_pct = 0.0
        else:
            saved_pct = 1 - (cropped_usage.input_tokens / uncropped_usage.input_tokens)
        rows.append(
            {
                "violation_type": violation_type.value,
                "cropped_input_tokens": cropped_usage.input_tokens,
                "uncropped_input_tokens": uncropped_usage.input_tokens if uncropped_usage else 0,
                "input_tokens_saved_pct": round(saved_pct * 100, 1),
            }
        )
    return rows


def full_comparison_rows(matrix: ComparisonMatrix) -> list[dict]:
    """One combined table across all 4 combinations, tagged by each
    report's own `label` — group/pivot by `label` and `violation_type` in
    the UI to slice however you like."""
    rows: list[dict] = []
    for report in matrix.as_dict().values():
        rows.extend(report_to_rows(report))
    return rows


def method_comparison_rows(matrix: MethodComparisonMatrix) -> list[dict]:
    """One combined table across all 6 named eval methods, tagged by each
    report's own `label` — mirrors `full_comparison_rows`'s flattening
    style. Skips token-savings reporting (not in scope for the 6-method
    bundle — see `full_comparison_token_savings` for the 4-combination
    equivalent)."""
    rows: list[dict] = []
    for report in matrix.as_dict().values():
        rows.extend(report_to_rows(report))
    return rows


def glove_crop_experiment_rows(results: dict[str, EvalReport]) -> list[dict]:
    """One combined table across the glove crop-tightness experiment's
    variants ("tight", "wide", "tight+wide"), tagged by `variant` — mirrors
    `full_comparison_rows`'s flattening style, but keyed by the experiment's
    plain `dict[str, EvalReport]` rather than a `ComparisonMatrix`."""
    rows: list[dict] = []
    for variant, report in results.items():
        for row in report_to_rows(report):
            rows.append({"variant": variant, **row})
    return rows


def full_comparison_token_savings(matrix: ComparisonMatrix) -> list[dict]:
    """Token savings from cropping, computed separately per shot type
    (zero-shot and few-shot each get their own cropped-vs-uncropped
    comparison, since few-shot examples add their own token overhead that
    shouldn't be conflated with the crop-vs-no-crop savings)."""
    rows = []
    for shot_type, cropped, uncropped in (
        ("zero-shot", matrix.zero_shot_cropped, matrix.zero_shot_uncropped),
        ("few-shot", matrix.few_shot_cropped, matrix.few_shot_uncropped),
    ):
        for row in token_savings_rows(cropped, uncropped):
            rows.append({"shot_type": shot_type, **row})
    return rows


def print_report(report: EvalReport) -> None:
    rows = report_to_rows(report)
    if not rows:
        print(f"[{report.label}] no cases")
        return
    print(f"\n=== {report.label} ===")
    header = list(rows[0].keys())
    print(" | ".join(header))
    for row in rows:
        print(" | ".join(str(row[col]) for col in header))

    overall = reasoning_alignment_summary(report.case_outcomes)
    print(
        f"overall reasoning_alignment_mean: {overall['reasoning_alignment_mean']} "
        f"({overall['reasoning_scored_count']}/{overall['reasoning_total_count']} cases scored)"
    )

    forklift_outcomes = [co for co in report.case_outcomes if co.case.violation_type is ViolationType.TOO_NEAR_FORKLIFT]
    if forklift_outcomes:
        print("too_near_forklift filter-vs-judge breakdown:")
        for row in forklift_filter_stats_rows(forklift_filter_stats(forklift_outcomes)):
            print(row)
