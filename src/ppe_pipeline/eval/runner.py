"""Runs the judge over a ground-truth eval dataset and scores it — the
basis for the zero-shot-vs-few-shot and cropped-vs-uncropped comparisons.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from typing import Callable

from PIL import Image
from pydantic import ValidationError

from .. import storage
from ..config import PipelineConfig, get_config
from ..cropping import crop_box_for_violation, crop_bytes, full_image_bytes, judge_box_to_original
from ..forklift_rule import evaluate_too_near_forklift
from .. import helmet_detector
from ..judge_client import JudgeUsage
from ..judge_prompt import build_forklift_operator_context
from ..models import BoundingBox, ViolationType
from .dataset import EvalCase
from .metrics import ConfusionCounts, Outcome, classify


@dataclass
class CaseOutcome:
    case: EvalCase
    outcome: Outcome
    verdict: str
    confidence: str
    explanation: str
    usage: JudgeUsage
    # 1-5 reasoning-alignment rubric score + critique from JudgeClient.
    # score_reasoning, populated only when `case.ground_truth_rationale` is
    # not None — left None (never 0) when there's no resolved GT rationale
    # to score against, so an unscored case can't silently read as the worst
    # possible score in an aggregate mean.
    reasoning_score: int | None = None
    reasoning_critique: str | None = None
    # False only for too_near_forklift cases the geometric proximity filter
    # (forklift_rule.evaluate_too_near_forklift) dropped before the judge
    # ever saw them — matches production's own pre-filter (pipeline.py).
    # True for every other case, so nothing else needs updating.
    reached_judge: bool = True
    # The filter's own coverage_ratio / is_operator verdict, whenever the
    # filter ran (any too_near_forklift case with a forklift_box) —
    # regardless of whether it dropped the case or let it through. None for
    # cases the filter never applies to.
    filter_coverage_ratio: float | None = None
    filter_is_operator: bool | None = None
    # Where the judge said the helmet is, on a helmet case it cleared as
    # "helmet_worn" — normalized to the image that case's judge call was
    # actually given, which is the crop for a cropped run and the full
    # frame for an uncropped one (see judge_prompt.HELMET_BOX_SCHEMA).
    # None for every other case, including any the judge never saw.
    helmet_box: list[float] | None = None
    # The same box in the ORIGINAL image's pixel coordinates, inverted
    # through whatever region the judge was actually shown. Independent of
    # crop settings, so unlike `helmet_box` it stays meaningful once the
    # crop it was judged on is gone — which is what lets it be stored,
    # replayed, or drawn on the full frame.
    helmet_box_original: BoundingBox | None = None
    # The region of the original image this case's judge call was actually
    # shown, in original coordinates — the crop box for a cropped run, the
    # full frame for an uncropped one. Carried so a consumer can express the
    # judge's normalized box in the judged image's own pixels without
    # re-fetching the image to re-derive the crop.
    judged_region: BoundingBox | None = None
    # IoU between two independent localization asks, when the grounding call
    # from the stage-2 helmet detector (see helmet_detector.py). A box the
    # could actually see lands in the same place twice; one it guessed does
    # not. `None` when only one ask was made, or no box came back.
    helmet_box_confidence: float | None = None
    # "cv" (the stage-2 YOLO11s detector) or "vlm" (the judge's own inline
    # helmet_box, used when the detector is disabled, has no weights, cannot
    # load, or found nothing above threshold). None when there is no box at
    # all. Surfaced so a silent fall back to the looser box is visible
    # rather than looking like a detector result with no score.
    helmet_box_source: str | None = None


@dataclass
class EvalReport:
    label: str
    counts: dict[ViolationType, ConfusionCounts] = field(default_factory=dict)
    usage: dict[ViolationType, JudgeUsage] = field(default_factory=dict)
    raw_cv_precision: dict[ViolationType, float] = field(default_factory=dict)
    case_outcomes: list[CaseOutcome] = field(default_factory=list)
    # Reasoning-alignment scores per violation type, one entry per case,
    # None for a case that had no resolved ground-truth rationale to score
    # against. Only populated when a report is REPLAYED from an
    # eval_runs/*.jsonl (see eval.replay), where the CaseOutcome objects
    # `case_outcomes` would normally hold cannot be rebuilt -- the jsonl
    # carries a case's score but not its image path or boxes. A live run
    # leaves this empty and `report_to_rows` reads case_outcomes instead;
    # the two are never both populated, so there is one source of truth
    # per report.
    replayed_reasoning_scores: dict[ViolationType, list[float | None]] = field(default_factory=dict)
    # True when this report's run bypassed the too_near_forklift geometric
    # proximity pre-filter entirely (the "no filter" / base condition) —
    # False (default) matches today's shipped behavior, where the filter
    # runs as normal.
    filter_bypassed: bool = False

    @property
    def overall_counts(self) -> ConfusionCounts:
        total = ConfusionCounts()
        for counts in self.counts.values():
            total = total + counts
        return total

    @property
    def total_usage(self) -> JudgeUsage:
        total = JudgeUsage()
        for usage in self.usage.values():
            total = total + usage
        return total


def _image_size(image_bytes: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(image_bytes)) as image:
        return image.size


def _image_bytes_for_case(
    case: EvalCase,
    use_crop: bool,
    config: PipelineConfig,
    pad_override: float | None = None,
    force_person_box: bool = False,
) -> tuple[bytes, str, BoundingBox]:
    """Also returns the region of the original image the judge is being
    shown, in original coordinates — the crop box, or the full image bounds
    for an uncropped run. That's what makes a judge-reported normalized box
    invertible back onto the original frame (see
    `cropping.judge_box_to_original`); without it, the crop geometry is
    computed here and then lost.
    """
    # Fetch this case's image bytes exactly once — both the crop-box sizing
    # below and the final crop/full-image encoding work off the same bytes,
    # rather than each independently hitting storage.
    image_bytes = storage.get_storage().get_bytes(case.image_path.as_posix())
    if not use_crop:
        width, height = _image_size(image_bytes)
        full_bytes, media_type = full_image_bytes(image_bytes)
        return full_bytes, media_type, BoundingBox(x1=0, y1=0, x2=width, y2=height)
    box = crop_box_for_violation(
        case.violation_type,
        case.person_box,
        _image_size(image_bytes),
        region_box=None if force_person_box else case.region_box,
        forklift_box=case.forklift_box,
        config=config,
        pad_override=pad_override,
    )
    cropped_bytes, media_type = crop_bytes(image_bytes, box)
    return cropped_bytes, media_type, box


def _group_by_type(dataset: list[EvalCase]) -> dict[ViolationType, list[EvalCase]]:
    grouped: dict[ViolationType, list[EvalCase]] = {}
    for case in dataset:
        grouped.setdefault(case.violation_type, []).append(case)
    return grouped


OnCase = Callable[[str, int, int, CaseOutcome, bytes, str], None]


def run_eval(
    dataset: list[EvalCase],
    judge_client,
    label: str = "run",
    use_crop: bool = True,
    config: PipelineConfig | None = None,
    on_case: OnCase | None = None,
    pad_override: float | None = None,
    force_person_box: bool = False,
    bypass_forklift_filter: bool = False,
) -> EvalReport:
    """Judge every case in `dataset` and score against its ground truth.

    too_near_forklift cases are first run through production's own
    geometric proximity pre-filter (forklift_rule.evaluate_too_near_forklift)
    — the same check `run_pipeline` applies before a candidate ever reaches
    cropping or the judge. A case the filter finds not a violation (person
    and forklift too far apart) never hits the judge at all: it's scored
    directly via `classify()` against a synthetic "FP" model verdict, with
    `reached_judge=False` and zero `JudgeUsage()`, and reasoning-alignment
    scoring is skipped (there's no judge explanation to score against GT
    rationale). A case the filter instead flags as the forklift's own
    possible operator (high containment_ratio) still reaches the judge, with
    the containment ratio passed as `extra_context` so the judge can weigh
    in from the image. This keeps the eval harness's baseline honest about
    what production actually shows the judge, instead of scoring every case
    as if it always reached the judge.

    `use_crop=False` sends the full original image instead of the
    violation-specific crop — the basis for the cropped-vs-uncropped
    token/accuracy comparison. `judge_client` only needs a `.judge(...)`
    method returning a `JudgeCallResult` (see judge_client.SupportsJudge).

    If `judge_client.judge(...)` itself raises `json.JSONDecodeError` or
    `pydantic.ValidationError` (a model response that still fails to parse
    or validate after the judge client's own internal retries — see
    judge_client._call_and_validate_with_retry), that failure is isolated to
    this one case rather than aborting the whole run: it's scored as
    "uncertain" (routing to human review, like any other uncertain verdict),
    with the error message recorded in `explanation` and zero `JudgeUsage()`,
    matching the shape of a filter-dropped case above.

    Whenever `case.ground_truth_rationale` is not None (a resolved gold-set
    annotation exists for that case), `judge_client.score_reasoning(...)` is
    also called to rate how well the judge's explanation aligns with that
    rationale — always on, no separate flag, since every eval run should
    surface this alongside the verdict metrics. Cases with no resolved GT
    rationale are simply left unscored (`reasoning_score` stays `None`). If
    `score_reasoning` itself raises, that failure is caught so it can't
    abort the whole run — the case's already-obtained verdict/explanation
    is kept, `reasoning_score` stays `None`, and `reasoning_critique` notes
    the scoring call failed.

    `on_case`, if given, is called after every case as
    `on_case(label, cases_done, total_cases, case_outcome, image_bytes,
    media_type)` — a hook for callers (e.g. the API's background eval jobs)
    to report progress, including the actual crop just judged, on eval runs
    that can take a long time, without changing anything about how the run
    itself is scored.

    `pad_override`, if given, is passed through to `crop_box_for_violation`
    to replace `config.cropping.glove_helmet_pad` for GLOVE/HELMET cases —
    used by `run_glove_crop_experiment` to vary crop tightness without a
    config change. `None` (the default) preserves existing behavior exactly.

    `force_person_box`, if True, suppresses `case.region_box` when building
    the crop, so the GLOVE/HELMET branch of `crop_box_for_violation` always
    pads `person_box` regardless of whether the case has a `region_box` set
    — used by `run_glove_crop_experiment`, which is scoped to varying pad on
    `person_box` specifically. `False` (the default) preserves existing
    behavior exactly.

    `bypass_forklift_filter`, if True, skips the geometric proximity
    pre-filter entirely — every too_near_forklift case falls through to the
    exact same judge call path as a non-forklift case (`extra_context=None`,
    `reached_judge=True`, `filter_coverage_ratio=None`,
    `filter_is_operator=None`). This is the eval harness's "no filter" / base
    condition, for measuring what the judge does unaided by production's own
    pre-filter. `False` (the default) preserves existing behavior exactly.
    """
    config = config or get_config()
    report = EvalReport(label=label, filter_bypassed=bypass_forklift_filter)
    total = len(dataset)

    for i, case in enumerate(dataset, start=1):
        image_bytes, media_type, judged_region = _image_bytes_for_case(
            case, use_crop, config, pad_override=pad_override, force_person_box=force_person_box
        )

        filter_result = None
        if (
            not bypass_forklift_filter
            and case.violation_type is ViolationType.TOO_NEAR_FORKLIFT
            and case.forklift_box is not None
        ):
            filter_result = evaluate_too_near_forklift(case.person_box, case.forklift_box, config=config)

        if filter_result is not None and not filter_result.is_violation and not filter_result.is_operator:
            outcome = classify(case.ground_truth_is_violation, "FP")
            counts = report.counts.setdefault(case.violation_type, ConfusionCounts())
            counts.record(outcome, case.ground_truth_is_violation)
            report.usage[case.violation_type] = report.usage.get(case.violation_type, JudgeUsage()) + JudgeUsage()
            case_outcome = CaseOutcome(
                case=case,
                outcome=outcome,
                verdict="FP",
                confidence="high",
                explanation="Filtered before reaching the judge: geometric proximity check "
                "found the person and forklift not near enough.",
                usage=JudgeUsage(),
                reached_judge=False,
                filter_coverage_ratio=filter_result.coverage_ratio,
                filter_is_operator=filter_result.is_operator,
            )
            report.case_outcomes.append(case_outcome)
            if on_case is not None:
                on_case(label, i, total, case_outcome, image_bytes, media_type)
            continue

        extra_context = None
        if filter_result is not None and filter_result.is_operator:
            extra_context = build_forklift_operator_context(filter_result.containment_ratio)

        try:
            call = judge_client.judge(
                image_bytes=image_bytes,
                media_type=media_type,
                person_id=case.case_id,
                violation_types=[case.violation_type],
                extra_context=extra_context,
            )
        except (json.JSONDecodeError, ValidationError) as exc:
            # The judge client already retried the call+parse+validate unit
            # a bounded number of times (see judge_client._call_and_validate_
            # with_retry) — if it still failed, isolate this one case as
            # "uncertain" (routes to human review, per metrics.classify)
            # rather than letting it kill the rest of the eval run.
            outcome = classify(case.ground_truth_is_violation, "uncertain")
            counts = report.counts.setdefault(case.violation_type, ConfusionCounts())
            counts.record(outcome, case.ground_truth_is_violation)
            report.usage[case.violation_type] = report.usage.get(case.violation_type, JudgeUsage()) + JudgeUsage()
            case_outcome = CaseOutcome(
                case=case,
                outcome=outcome,
                verdict="uncertain",
                confidence="low",
                explanation=f"Judge call failed validation after retries exhausted: {exc}",
                usage=JudgeUsage(),
                reached_judge=True,
                filter_coverage_ratio=filter_result.coverage_ratio if filter_result else None,
                filter_is_operator=filter_result.is_operator if filter_result else None,
            )
            report.case_outcomes.append(case_outcome)
            if on_case is not None:
                on_case(label, i, total, case_outcome, image_bytes, media_type)
            continue

        result = call.response.results[0]
        outcome = classify(case.ground_truth_is_violation, result.verdict)

        # A cleared helmet gets a dedicated localization pass from the
        # stage-2 YOLO11s detector, whose boxes are far tighter than the
        # `helmet_box` field on the verdict itself (see helmet_detector.py).
        # Falls back to the inline field when the detector is disabled, has
        # no weights, or finds nothing above threshold — so a box is never
        # lost, only ever improved. It never changes the verdict.
        helmet_box = result.helmet_box
        helmet_box_confidence = None
        helmet_box_source = "vlm" if helmet_box is not None else None
        if (
            case.violation_type is ViolationType.HELMET
            and result.verdict == "FP"
            and result.reason == "helmet_worn"
        ):
            detector = helmet_detector.get_helmet_detector(config)
            located = detector.locate(image_bytes, media_type) if detector else None
            if located is not None:
                helmet_box = located.box
                helmet_box_confidence = located.confidence
                helmet_box_source = "cv"

        reasoning_score = reasoning_critique = None
        if case.ground_truth_rationale is not None:
            try:
                alignment = judge_client.score_reasoning(
                    case.violation_type, result.explanation, case.ground_truth_rationale
                )
            except Exception as e:
                reasoning_critique = f"Reasoning-alignment scoring failed: {e}"
            else:
                reasoning_score = alignment.score
                reasoning_critique = alignment.critique

        counts = report.counts.setdefault(case.violation_type, ConfusionCounts())
        counts.record(outcome, case.ground_truth_is_violation)
        report.usage[case.violation_type] = report.usage.get(case.violation_type, JudgeUsage()) + call.usage
        case_outcome = CaseOutcome(
            case=case,
            outcome=outcome,
            verdict=result.verdict,
            confidence=result.confidence,
            explanation=result.explanation,
            usage=call.usage,
            reasoning_score=reasoning_score,
            reasoning_critique=reasoning_critique,
            reached_judge=True,
            filter_coverage_ratio=filter_result.coverage_ratio if filter_result else None,
            filter_is_operator=filter_result.is_operator if filter_result else None,
            helmet_box=helmet_box,
            helmet_box_original=(
                judge_box_to_original(helmet_box, judged_region) if helmet_box is not None else None
            ),
            judged_region=judged_region,
            helmet_box_confidence=helmet_box_confidence,
            helmet_box_source=helmet_box_source,
        )
        report.case_outcomes.append(case_outcome)
        if on_case is not None:
            on_case(label, i, total, case_outcome, image_bytes, media_type)

    for violation_type, cases in _group_by_type(dataset).items():
        report.raw_cv_precision[violation_type] = sum(c.ground_truth_is_violation for c in cases) / len(cases)

    return report


def build_report(label: str, case_outcomes: list[CaseOutcome], filter_bypassed: bool = False) -> EvalReport:
    """Rebuild an EvalReport from already-judged CaseOutcomes, reclassifying
    each one against its case's *current* `ground_truth_is_violation` rather
    than trusting the `outcome` computed at judge time.

    This is the hook a human ground-truth review step uses: correct
    `case.ground_truth_is_violation` in place on any case that turned out to
    be mislabeled, then call this to rescore without re-judging anything.

    `filter_bypassed` should mirror whatever `bypass_forklift_filter` value
    the original `run_eval` call used to produce `case_outcomes` — it's not
    inferred from the outcomes themselves, since a bypassed run's cases look
    identical in shape to a non-forklift case."""
    report = EvalReport(label=label, filter_bypassed=filter_bypassed)
    for case_outcome in case_outcomes:
        case = case_outcome.case
        outcome = classify(case.ground_truth_is_violation, case_outcome.verdict)

        counts = report.counts.setdefault(case.violation_type, ConfusionCounts())
        counts.record(outcome, case.ground_truth_is_violation)
        report.usage[case.violation_type] = (
            report.usage.get(case.violation_type, JudgeUsage()) + case_outcome.usage
        )
        report.case_outcomes.append(
            CaseOutcome(
                case=case,
                outcome=outcome,
                verdict=case_outcome.verdict,
                confidence=case_outcome.confidence,
                explanation=case_outcome.explanation,
                usage=case_outcome.usage,
                reasoning_score=case_outcome.reasoning_score,
                reasoning_critique=case_outcome.reasoning_critique,
                reached_judge=case_outcome.reached_judge,
                filter_coverage_ratio=case_outcome.filter_coverage_ratio,
                filter_is_operator=case_outcome.filter_is_operator,
            )
        )

    for violation_type, cases in _group_by_type([co.case for co in case_outcomes]).items():
        report.raw_cv_precision[violation_type] = sum(c.ground_truth_is_violation for c in cases) / len(cases)

    return report


def compare_zero_shot_vs_few_shot(
    dataset: list[EvalCase],
    zero_shot_client,
    few_shot_client,
    config: PipelineConfig | None = None,
    on_case: OnCase | None = None,
) -> tuple[EvalReport, EvalReport]:
    """Run the same dataset through a zero-shot and a few-shot-primed judge
    client, returning (zero_shot_report, few_shot_report)."""
    zero_shot = run_eval(dataset, zero_shot_client, label="zero-shot", config=config, on_case=on_case)
    few_shot = run_eval(dataset, few_shot_client, label="few-shot", config=config, on_case=on_case)
    return zero_shot, few_shot


def compare_cropped_vs_uncropped(
    dataset: list[EvalCase],
    judge_client,
    config: PipelineConfig | None = None,
    on_case: OnCase | None = None,
) -> tuple[EvalReport, EvalReport]:
    """Run the same dataset cropped and uncropped through the same judge
    client, returning (cropped_report, uncropped_report) — compare
    `.total_usage` between the two for the token/cost savings, and
    `.counts` for whether cropping costs any accuracy."""
    cropped = run_eval(dataset, judge_client, label="cropped", use_crop=True, config=config, on_case=on_case)
    uncropped = run_eval(dataset, judge_client, label="uncropped", use_crop=False, config=config, on_case=on_case)
    return cropped, uncropped


@dataclass
class ComparisonMatrix:
    """All four combinations of {zero-shot, few-shot} x {cropped, uncropped}
    over the same dataset — the full picture of what few-shot examples and
    cropping each buy you, independently and together."""

    zero_shot_cropped: EvalReport
    zero_shot_uncropped: EvalReport
    few_shot_cropped: EvalReport
    few_shot_uncropped: EvalReport

    def as_dict(self) -> dict[str, EvalReport]:
        return {
            "zero-shot / cropped": self.zero_shot_cropped,
            "zero-shot / uncropped": self.zero_shot_uncropped,
            "few-shot / cropped": self.few_shot_cropped,
            "few-shot / uncropped": self.few_shot_uncropped,
        }


def run_full_comparison(
    dataset: list[EvalCase],
    zero_shot_client,
    few_shot_client,
    config: PipelineConfig | None = None,
    on_case: OnCase | None = None,
) -> ComparisonMatrix:
    """Run all 4 combinations over the same dataset. This is 4x the judge
    calls of a single `run_eval` — mind the cost on a large dataset."""
    return ComparisonMatrix(
        zero_shot_cropped=run_eval(
            dataset, zero_shot_client, label="zero-shot / cropped", use_crop=True, config=config, on_case=on_case
        ),
        zero_shot_uncropped=run_eval(
            dataset, zero_shot_client, label="zero-shot / uncropped", use_crop=False, config=config, on_case=on_case
        ),
        few_shot_cropped=run_eval(
            dataset, few_shot_client, label="few-shot / cropped", use_crop=True, config=config, on_case=on_case
        ),
        few_shot_uncropped=run_eval(
            dataset, few_shot_client, label="few-shot / uncropped", use_crop=False, config=config, on_case=on_case
        ),
    )


@dataclass
class MethodComparisonMatrix:
    """All 6 named eval methods over the same dataset: {zero-shot, few-shot}
    x {cropped, uncropped}, with zero-shot additionally split into
    {no filter (base), filter} conditions for the too_near_forklift
    geometric proximity pre-filter. Few-shot is always run with the filter
    active, matching production's own shipped behavior."""

    zero_shot_base_cropped: EvalReport
    zero_shot_base_uncropped: EvalReport
    zero_shot_filter_cropped: EvalReport
    zero_shot_filter_uncropped: EvalReport
    few_shot_filter_cropped: EvalReport
    few_shot_filter_uncropped: EvalReport

    def as_dict(self) -> dict[str, EvalReport]:
        return {
            "zero-shot (no filter) / cropped": self.zero_shot_base_cropped,
            "zero-shot (no filter) / uncropped": self.zero_shot_base_uncropped,
            "zero-shot (filter) / cropped": self.zero_shot_filter_cropped,
            "zero-shot (filter) / uncropped": self.zero_shot_filter_uncropped,
            "few-shot (filter) / cropped": self.few_shot_filter_cropped,
            "few-shot (filter) / uncropped": self.few_shot_filter_uncropped,
        }


def run_six_method_comparison(
    dataset: list[EvalCase],
    zero_shot_client,
    few_shot_client,
    config: PipelineConfig | None = None,
    on_case: OnCase | None = None,
) -> MethodComparisonMatrix:
    """Run all 6 named eval methods over the same dataset. This is 6x the
    judge calls of a single `run_eval` — mind the cost on a large dataset."""
    return MethodComparisonMatrix(
        zero_shot_base_cropped=run_eval(
            dataset,
            zero_shot_client,
            label="zero-shot (no filter) / cropped",
            use_crop=True,
            bypass_forklift_filter=True,
            config=config,
            on_case=on_case,
        ),
        zero_shot_base_uncropped=run_eval(
            dataset,
            zero_shot_client,
            label="zero-shot (no filter) / uncropped",
            use_crop=False,
            bypass_forklift_filter=True,
            config=config,
            on_case=on_case,
        ),
        zero_shot_filter_cropped=run_eval(
            dataset,
            zero_shot_client,
            label="zero-shot (filter) / cropped",
            use_crop=True,
            bypass_forklift_filter=False,
            config=config,
            on_case=on_case,
        ),
        zero_shot_filter_uncropped=run_eval(
            dataset,
            zero_shot_client,
            label="zero-shot (filter) / uncropped",
            use_crop=False,
            bypass_forklift_filter=False,
            config=config,
            on_case=on_case,
        ),
        few_shot_filter_cropped=run_eval(
            dataset,
            few_shot_client,
            label="few-shot (filter) / cropped",
            use_crop=True,
            bypass_forklift_filter=False,
            config=config,
            on_case=on_case,
        ),
        few_shot_filter_uncropped=run_eval(
            dataset,
            few_shot_client,
            label="few-shot (filter) / uncropped",
            use_crop=False,
            bypass_forklift_filter=False,
            config=config,
            on_case=on_case,
        ),
    )


# Pad values for the glove crop-tightness experiment: how much the GLOVE
# crop (on person_box — no eval dataset has region_box populated for glove
# cases yet) is expanded beyond its base box. "wide" matches today's
# config.toml default (glove_helmet_pad); "tight" and "tight+wide" are
# experiment-only variants, not proposed config changes.
GLOVE_CROP_PAD_VARIANTS: dict[str, float] = {
    "tight": 0.0,
    "wide": 0.3,
    "tight+wide": 0.6,
}


def run_glove_crop_experiment(
    dataset: list[EvalCase],
    judge_client,
    config: PipelineConfig | None = None,
    use_crop: bool = True,
    on_case: OnCase | None = None,
) -> dict[str, EvalReport]:
    """Run the dataset's glove cases three times, each with a different
    crop pad on `person_box` — "tight" (0.0), "wide" (0.3, today's default),
    and "tight+wide" (0.6) — to see how crop tightness trades off precision,
    recall, and abstention for glove violations.

    Non-glove cases are filtered out first, since only GLOVE/HELMET crops
    respect `pad_override` and this experiment is glove-scoped. "tight+wide"
    is an independent third `run_eval` pass, not a fused dual-image judge
    call — no change to `JudgeClient.judge()`'s single-image signature.

    `force_person_box=True` is always passed to `run_eval` here: this
    experiment is locked to varying pad on `person_box`, regardless of
    whether any case happens to have a `region_box` set (today, no glove
    case in any eval dataset does, but that's a fact about the data, not
    something this experiment should depend on).

    Returns a plain `dict[str, EvalReport]` keyed by variant name ("tight",
    "wide", "tight+wide") rather than a new dataclass, since there's no
    fixed shape to name fields after the way `ComparisonMatrix` does for
    its 4 combinations — just 3 named runs over one filtered dataset.
    """
    glove_cases = [case for case in dataset if case.violation_type is ViolationType.GLOVE]
    return {
        variant: run_eval(
            glove_cases,
            judge_client,
            label=variant,
            use_crop=use_crop,
            config=config,
            on_case=on_case,
            pad_override=pad,
            force_person_box=True,
        )
        for variant, pad in GLOVE_CROP_PAD_VARIANTS.items()
    }
