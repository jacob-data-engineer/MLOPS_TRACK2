"""Rebuild an EvalReport from a jsonl stream written by the UI's per-case
`on_case` hook — the "recoverable if this run is interrupted"
promise: a crash, an OOM, or running out of API credits partway through
still leaves every completed case scored on disk, this just turns those
raw rows back into the same metrics `report_to_rows`/`print_report` expect.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..judge_client import JudgeUsage
from ..models import ViolationType
from .metrics import ConfusionCounts, classify
from .runner import EvalReport


def iter_rows(path: Path):
    """Every parsed row of an eval_runs/*.jsonl, in file order.

    Public because the metrics API needs the raw per-case rows (verdict,
    confidence, explanation) to drill into one cell of a scored report, not
    just the aggregate the loaders below build.
    """
    yield from _iter_rows(path)


def rows_for_method(path: Path, method: str) -> list[dict]:
    """Rows belonging to one method label. A row with no `run` key is filed
    under the file's stem, matching `load_reports_from_jsonl`."""
    return [row for row in _iter_rows(path) if (row.get("run") or path.stem) == method]


def _iter_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _record(report: EvalReport, ground_truth_by_type: dict, row: dict) -> None:
    """Fold one jsonl row into `report`.

    Each line already carries its own `outcome` from the live run, but it's
    recomputed here via `classify` rather than trusted verbatim — cheap to
    redo and keeps this in sync with `metrics.classify` if that ever changes.
    """
    violation_type = ViolationType(row["violation_type"])
    is_violation = row["ground_truth"] == "TP"

    counts = report.counts.setdefault(violation_type, ConfusionCounts())
    counts.record(classify(is_violation, row["verdict"]), is_violation)

    usage = JudgeUsage(
        input_tokens=row.get("input_tokens", 0),
        output_tokens=row.get("output_tokens", 0),
    )
    report.usage[violation_type] = report.usage.get(violation_type, JudgeUsage()) + usage
    ground_truth_by_type.setdefault(violation_type, []).append(is_violation)

    # A row written before `reasoning_score` was recorded has no key at all,
    # which is NOT the same as a case that was scored as None: the first
    # means "this run never captured it", the second means "there was no
    # ground-truth rationale to score against". Only the latter belongs in
    # the total, so a file missing the key entirely contributes nothing and
    # reads as 0/0 rather than pretending every case went unscored.
    if "reasoning_score" in row:
        report.replayed_reasoning_scores.setdefault(violation_type, []).append(
            row["reasoning_score"]
        )


def _finalise(report: EvalReport, ground_truth_by_type: dict) -> EvalReport:
    """`raw_cv_precision` is computed over the same (partial) set of cases
    present in the file, not the full original dataset, since a truncated
    run doesn't know about cases it never reached."""
    for violation_type, flags in ground_truth_by_type.items():
        report.raw_cv_precision[violation_type] = sum(flags) / len(flags)
    return report


def load_reports_from_jsonl(path: Path) -> dict[str, EvalReport]:
    """Replay one eval_runs/*.jsonl into ONE REPORT PER METHOD, keyed by the
    method label the run wrote into each row's `run` field.

    A multi-method run streams every method's cases into a single file, so
    folding the whole file into one report blends methods that were run
    precisely to be compared: the result is a case-weighted average, which
    belongs to no method and hides the difference between them. Splitting on
    `run` is what makes the file mean what the Advanced tab's method
    multi-select implied — that is why `EVAL_METHODS`' labels are written
    verbatim into every row (see the ordering note above `EVAL_METHODS` in
    `ppe_pipeline_api/routers/eval.py`).

    A row with no `run` key (an older file, written before the field
    existed) is filed under the file's own stem, so such a file still
    replays as exactly one report rather than vanishing.

    Insertion-ordered by first appearance, so methods come back in the order
    the run executed them.
    """
    reports: dict[str, EvalReport] = {}
    ground_truth: dict[str, dict[ViolationType, list[bool]]] = {}

    for row in _iter_rows(path):
        method = row.get("run") or path.stem
        if method not in reports:
            reports[method] = EvalReport(label=method)
            ground_truth[method] = {}
        _record(reports[method], ground_truth[method], row)

    return {method: _finalise(report, ground_truth[method]) for method, report in reports.items()}


def load_report_from_jsonl(path: Path, label: str | None = None) -> EvalReport:
    """Replay a streamed eval_runs/*.jsonl file into ONE scored EvalReport,
    blending every method in the file together.

    Kept for callers that genuinely want the whole file as a single number.
    For a multi-method file this is an average across methods and is almost
    never what you want to read —
    prefer `load_reports_from_jsonl`.
    """
    report = EvalReport(label=label or path.stem)
    ground_truth_by_type: dict[ViolationType, list[bool]] = {}
    for row in _iter_rows(path):
        _record(report, ground_truth_by_type, row)
    return _finalise(report, ground_truth_by_type)
