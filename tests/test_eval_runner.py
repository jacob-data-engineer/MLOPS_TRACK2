import io
import json
from pathlib import Path

from botocore.exceptions import ClientError
from PIL import Image

from ppe_pipeline.config import PipelineConfig
from ppe_pipeline.cropping import crop_box_for_violation, judge_box_to_original
from ppe_pipeline.eval.dataset import load_eval_dataset
from ppe_pipeline.eval.report import (
    full_comparison_rows,
    full_comparison_token_savings,
    glove_crop_experiment_rows,
    method_comparison_rows,
    report_to_rows,
    token_savings_rows,
)
from ppe_pipeline.eval import runner
from ppe_pipeline.eval.runner import (
    build_report,
    compare_cropped_vs_uncropped,
    compare_zero_shot_vs_few_shot,
    run_eval,
    run_full_comparison,
    run_glove_crop_experiment,
    run_six_method_comparison,
)
from ppe_pipeline.judge_client import JudgeCallResult, JudgeResponse, JudgeResult, JudgeUsage
from ppe_pipeline.models import ViolationType
from ppe_pipeline.storage import S3Storage


def _build_dataset(tmp_path: Path) -> list:
    Image.new("RGB", (1000, 1000), color="white").save(tmp_path / "scene.jpg")
    manifest = [
        {
            "case_id": "glove_tp",
            "image": "scene.jpg",
            "violation_type": "glove",
            "person_box": [100, 100, 300, 500],
            "region_box": [110, 400, 160, 450],
            "ground_truth": "TP",
        },
        {
            "case_id": "glove_fp",
            "image": "scene.jpg",
            "violation_type": "glove",
            "person_box": [400, 100, 600, 500],
            "region_box": [410, 400, 460, 450],
            "ground_truth": "FP",
        },
        {
            "case_id": "helmet_tp",
            "image": "scene.jpg",
            "violation_type": "helmet",
            "person_box": [650, 100, 850, 500],
            "region_box": [700, 100, 800, 160],
            "ground_truth": "TP",
        },
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return load_eval_dataset(manifest_path)


class _AlwaysVerdictClient:
    """Always returns the same verdict, regardless of the case — used to
    simulate a judge that gets everything wrong (or everything uncertain)."""

    def __init__(self, verdict: str):
        self._verdict = verdict

    def judge(self, image_bytes, media_type, person_id, violation_types, extra_context=None):
        response = JudgeResponse(
            person_id=person_id,
            results=[
                JudgeResult(
                    violation_type=v,
                    verdict=self._verdict,
                    reason="clear_bare_hand" if v.value == "glove" else "bare_head",
                    confidence="high",
                    explanation="fake",
                )
                for v in violation_types
            ],
        )
        return JudgeCallResult(response=response, usage=JudgeUsage(input_tokens=100, output_tokens=20))


class _OracleClient:
    """Answers correctly by peeking at the case_id convention used in
    _build_dataset ("*_tp" / "*_fp"). Usage scales with image size, so
    cropped vs uncropped calls can be told apart by token count."""

    def judge(self, image_bytes, media_type, person_id, violation_types, extra_context=None):
        verdict = "TP" if person_id.endswith("_tp") else "FP"
        response = JudgeResponse(
            person_id=person_id,
            results=[
                JudgeResult(
                    violation_type=v,
                    verdict=verdict,
                    reason="clear_bare_hand" if v.value == "glove" else "bare_head",
                    confidence="high",
                    explanation="fake oracle",
                )
                for v in violation_types
            ],
        )
        usage = JudgeUsage(input_tokens=len(image_bytes), output_tokens=20)
        return JudgeCallResult(response=response, usage=usage)


class _OracleClientWithFailingReasoningScore(_OracleClient):
    """Same verdicts as _OracleClient, but score_reasoning raises for one
    specific ground_truth_rationale (simulating a bad reasoning-scoring API
    call for a single case) and succeeds normally for any other."""

    def __init__(self, failing_rationale: str):
        self._failing_rationale = failing_rationale

    def score_reasoning(self, violation_type, llm_explanation, ground_truth_rationale):
        if ground_truth_rationale == self._failing_rationale:
            raise RuntimeError("fake reasoning-scoring API failure")
        from ppe_pipeline.judge_client import ReasoningAlignmentScore

        return ReasoningAlignmentScore(score=4, critique="looks fine")


# Same coordinate convention as tests/test_forklift_rule.py, so these boxes'
# near/far outcome is independent of whatever thresholds config.toml is
# tuned to at any given time (each test below passes an explicit
# PipelineConfig() with forklift_rule.py's own built-in defaults).
FORKLIFT_BOX = [0, 0, 300, 200]
NEARBY_PERSON_BOX = [280, 20, 340, 180]  # edge overlaps the forklift's danger zone
FAR_PERSON_BOX = [1000, 1000, 1100, 1200]  # nowhere near the forklift
OPERATOR_SHAPED_PERSON_BOX = [50, 20, 150, 180]  # almost entirely inside the forklift box


def _build_forklift_dataset(tmp_path: Path, entries: list[dict]) -> list:
    Image.new("RGB", (1500, 1500), color="white").save(tmp_path / "forklift_scene.jpg")
    manifest = [
        {
            "case_id": entry["case_id"],
            "image": "forklift_scene.jpg",
            "violation_type": "too_near_forklift",
            "person_box": entry["person_box"],
            "forklift_box": FORKLIFT_BOX,
            "ground_truth": entry["ground_truth"],
        }
        for entry in entries
    ]
    manifest_path = tmp_path / "forklift_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return load_eval_dataset(manifest_path)


class _SpyOracleClient(_OracleClient):
    """Same verdicts as _OracleClient, but records every person_id it was
    actually asked to judge — used to assert the geometric forklift filter
    kept a dropped case from ever reaching `.judge()`."""

    def __init__(self):
        self.judge_calls: list[str] = []
        self.extra_contexts: dict[str, str | None] = {}

    def judge(self, image_bytes, media_type, person_id, violation_types, extra_context=None):
        self.judge_calls.append(person_id)
        self.extra_contexts[person_id] = extra_context
        return super().judge(image_bytes, media_type, person_id, violation_types, extra_context)


def test_forklift_case_dropped_by_filter_with_gt_fp_is_true_negative_and_skips_the_judge(tmp_path):
    dataset = _build_forklift_dataset(
        tmp_path, [{"case_id": "forklift_far_fp", "person_box": FAR_PERSON_BOX, "ground_truth": "FP"}]
    )
    spy = _SpyOracleClient()
    report = run_eval(dataset, spy, label="forklift", config=PipelineConfig())

    assert spy.judge_calls == []  # the filter dropped it before the judge was ever asked
    outcome = report.case_outcomes[0]
    assert outcome.outcome == "true_negative"
    assert outcome.reached_judge is False
    assert outcome.usage == JudgeUsage()


def test_forklift_true_positive_dropped_by_filter_is_a_false_negative_not_silently_miscounted(tmp_path):
    dataset = _build_forklift_dataset(
        tmp_path, [{"case_id": "forklift_far_tp", "person_box": FAR_PERSON_BOX, "ground_truth": "TP"}]
    )
    spy = _SpyOracleClient()
    report = run_eval(dataset, spy, label="forklift", config=PipelineConfig())

    assert spy.judge_calls == []
    outcome = report.case_outcomes[0]
    assert outcome.outcome == "false_negative"
    assert outcome.reached_judge is False


def test_forklift_case_that_passes_the_filter_behaves_exactly_as_before(tmp_path):
    dataset = _build_forklift_dataset(
        tmp_path, [{"case_id": "forklift_near_tp", "person_box": NEARBY_PERSON_BOX, "ground_truth": "TP"}]
    )
    spy = _SpyOracleClient()
    report = run_eval(dataset, spy, label="forklift", config=PipelineConfig())

    assert spy.judge_calls == ["forklift_near_tp"]  # the judge was actually asked, same as before this change
    outcome = report.case_outcomes[0]
    assert outcome.outcome == "true_positive"
    assert outcome.reached_judge is True
    assert outcome.verdict == "TP"


def test_forklift_operator_shaped_case_now_reaches_the_judge_with_extra_context(tmp_path):
    dataset = _build_forklift_dataset(
        tmp_path, [{"case_id": "forklift_operator_tp", "person_box": OPERATOR_SHAPED_PERSON_BOX, "ground_truth": "TP"}]
    )
    spy = _SpyOracleClient()
    report = run_eval(dataset, spy, label="forklift", config=PipelineConfig())

    assert spy.judge_calls == ["forklift_operator_tp"]  # no longer auto-resolved to FP
    extra_context = spy.extra_contexts["forklift_operator_tp"]
    assert extra_context is not None
    assert "contained within the forklift's bounding box" in extra_context

    outcome = report.case_outcomes[0]
    assert outcome.reached_judge is True
    assert outcome.filter_is_operator is True


def test_bypass_forklift_filter_sends_far_apart_case_straight_to_the_judge(tmp_path):
    dataset = _build_forklift_dataset(
        tmp_path, [{"case_id": "forklift_far_tp", "person_box": FAR_PERSON_BOX, "ground_truth": "TP"}]
    )
    spy = _SpyOracleClient()
    report = run_eval(dataset, spy, label="forklift", config=PipelineConfig(), bypass_forklift_filter=True)

    # With the filter bypassed, even a far-apart case reaches the judge
    # instead of being auto-resolved to FP by the geometric filter.
    assert spy.judge_calls == ["forklift_far_tp"]
    assert spy.extra_contexts["forklift_far_tp"] is None
    outcome = report.case_outcomes[0]
    assert outcome.reached_judge is True
    assert outcome.filter_coverage_ratio is None
    assert outcome.filter_is_operator is None
    assert outcome.verdict == "TP"  # the oracle answers correctly since person_id ends "_tp"


def test_eval_report_filter_bypassed_flag_set_by_run_eval_and_build_report(tmp_path):
    dataset = _build_forklift_dataset(
        tmp_path, [{"case_id": "forklift_far_tp", "person_box": FAR_PERSON_BOX, "ground_truth": "TP"}]
    )
    spy = _SpyOracleClient()

    default_report = run_eval(dataset, spy, label="forklift", config=PipelineConfig())
    assert default_report.filter_bypassed is False

    bypassed_report = run_eval(dataset, spy, label="forklift", config=PipelineConfig(), bypass_forklift_filter=True)
    assert bypassed_report.filter_bypassed is True

    rebuilt_default = build_report("forklift", bypassed_report.case_outcomes)
    assert rebuilt_default.filter_bypassed is False

    rebuilt_bypassed = build_report("forklift", bypassed_report.case_outcomes, filter_bypassed=True)
    assert rebuilt_bypassed.filter_bypassed is True


class _CountingFakeS3Client:
    """Minimal fake `get_object`, recording how many times each key is
    fetched — proves `_image_bytes_for_case` reads a case's image through
    `storage.get_storage()` and fetches it exactly once per call, not once
    for sizing and again for the crop/full-image encode."""

    def __init__(self, objects: dict[str, bytes]):
        self._objects = objects
        self.calls: dict[str, int] = {}

    def get_object(self, Bucket, Key):
        self.calls[Key] = self.calls.get(Key, 0) + 1
        if Key not in self._objects:
            raise ClientError({"Error": {"Code": "NoSuchKey", "Message": "Not Found"}}, "GetObject")
        return {"Body": io.BytesIO(self._objects[Key]), "ETag": '"etag"'}


def test_run_eval_reads_case_images_through_storage_exactly_once_per_case(tmp_path, monkeypatch):
    # _build_dataset's 3 cases all share the same underlying "scene.jpg", so
    # asserting per-key call counts would hide a per-case double-fetch —
    # assert the total call count instead, one fetch per case in the dataset.
    dataset = _build_dataset(tmp_path)
    image_bytes = (tmp_path / "scene.jpg").read_bytes()
    fake_client = _CountingFakeS3Client({case.image_path.as_posix(): image_bytes for case in dataset})
    fake_backend = S3Storage(bucket="fake-bucket", client=fake_client)
    monkeypatch.setattr(runner.storage, "get_storage", lambda: fake_backend)

    run_eval(dataset, _OracleClient(), label="oracle")

    assert sum(fake_client.calls.values()) == len(dataset)


def test_run_eval_scores_against_ground_truth(tmp_path):
    dataset = _build_dataset(tmp_path)
    report = run_eval(dataset, _OracleClient(), label="oracle")

    from ppe_pipeline.models import ViolationType

    glove_counts = report.counts[ViolationType.GLOVE]
    assert glove_counts.true_positive == 1
    assert glove_counts.true_negative == 1
    assert glove_counts.precision == 1.0
    assert glove_counts.recall == 1.0

    helmet_counts = report.counts[ViolationType.HELMET]
    assert helmet_counts.true_positive == 1

    # raw_cv_precision is just "fraction of cases that were real violations"
    assert report.raw_cv_precision[ViolationType.GLOVE] == 0.5


def test_run_eval_on_case_callback_fires_once_per_case_with_running_progress(tmp_path):
    dataset = _build_dataset(tmp_path)
    calls = []
    run_eval(dataset, _OracleClient(), label="oracle", on_case=lambda *args: calls.append(args))

    assert [c[1:3] for c in calls] == [(1, 3), (2, 3), (3, 3)]  # (done, total) per case, in order
    assert all(c[0] == "oracle" for c in calls)
    assert {c[3].case.case_id for c in calls} == {"glove_tp", "glove_fp", "helmet_tp"}
    # (label, done, total, case_outcome, image_bytes, media_type) — the crop actually judged
    assert all(isinstance(c[4], bytes) and len(c[4]) > 0 for c in calls)
    assert all(c[5] == "image/jpeg" for c in calls)


def test_run_eval_survives_a_score_reasoning_failure_on_one_case(tmp_path):
    dataset = _build_dataset(tmp_path)
    for case in dataset:
        case.ground_truth_rationale = "failing rationale" if case.case_id == "glove_tp" else "good rationale"

    client = _OracleClientWithFailingReasoningScore(failing_rationale="failing rationale")
    report = run_eval(dataset, client, label="oracle")

    outcomes = {co.case.case_id: co for co in report.case_outcomes}
    assert len(outcomes) == 3  # the run completed for every case, including the one after the failure

    failed = outcomes["glove_tp"]
    assert failed.reasoning_score is None
    assert failed.reasoning_critique is not None
    assert "failed" in failed.reasoning_critique.lower()
    # the case's actual judge verdict must still be kept
    assert failed.verdict == "TP"

    for case_id in ("glove_fp", "helmet_tp"):
        unaffected = outcomes[case_id]
        assert unaffected.reasoning_score == 4
        assert unaffected.reasoning_critique == "looks fine"


class _OracleClientThatFailsJudgeForOneCase(_OracleClient):
    """Same verdicts as _OracleClient, but `.judge(...)` itself raises for
    one specific person_id (simulating a model response that still fails
    JudgeResponse validation after the judge client's own internal retries
    are exhausted) and behaves normally for every other case."""

    def __init__(self, failing_person_id: str):
        self._failing_person_id = failing_person_id

    def judge(self, image_bytes, media_type, person_id, violation_types, extra_context=None):
        if person_id == self._failing_person_id:
            from ppe_pipeline.judge_client import JudgeResponse

            JudgeResponse.model_validate({})  # raises pydantic.ValidationError: missing required fields
        return super().judge(image_bytes, media_type, person_id, violation_types, extra_context)


def test_run_eval_survives_a_judge_call_that_fails_validation_after_retries(tmp_path):
    dataset = _build_dataset(tmp_path)
    client = _OracleClientThatFailsJudgeForOneCase(failing_person_id="glove_tp")
    report = run_eval(dataset, client, label="oracle")

    outcomes = {co.case.case_id: co for co in report.case_outcomes}
    assert len(outcomes) == 3  # the run completed for every case, including the one after the failure

    failed = outcomes["glove_tp"]
    assert failed.verdict == "uncertain"
    assert failed.outcome == "uncertain"
    assert failed.usage == JudgeUsage()
    assert failed.reached_judge is True
    assert "failed" in failed.explanation.lower()

    assert outcomes["glove_fp"].verdict == "FP"
    assert outcomes["helmet_tp"].verdict == "TP"


def test_a_judge_that_always_says_fp_produces_false_negatives(tmp_path):
    dataset = _build_dataset(tmp_path)
    report = run_eval(dataset, _AlwaysVerdictClient("FP"), label="always-fp")

    from ppe_pipeline.models import ViolationType

    glove_counts = report.counts[ViolationType.GLOVE]
    assert glove_counts.false_negative == 1  # the glove_tp case got wrongly suppressed
    assert glove_counts.true_negative == 1  # the glove_fp case was correctly suppressed


def test_compare_zero_shot_vs_few_shot_shows_the_better_client_scoring_higher(tmp_path):
    dataset = _build_dataset(tmp_path)
    zero_shot_report, few_shot_report = compare_zero_shot_vs_few_shot(
        dataset, zero_shot_client=_AlwaysVerdictClient("uncertain"), few_shot_client=_OracleClient()
    )

    assert zero_shot_report.overall_counts.uncertain == 3
    assert few_shot_report.overall_counts.true_positive == 2
    assert few_shot_report.overall_counts.true_negative == 1


def test_compare_cropped_vs_uncropped_shows_token_savings(tmp_path):
    dataset = _build_dataset(tmp_path)
    cropped_report, uncropped_report = compare_cropped_vs_uncropped(dataset, _OracleClient())

    # Same oracle answers both ways, so accuracy is identical...
    assert cropped_report.overall_counts == uncropped_report.overall_counts
    # ...but the crop is a small fraction of the full 1000x1000 frame, so it
    # must cost meaningfully fewer input tokens.
    assert cropped_report.total_usage.input_tokens < uncropped_report.total_usage.input_tokens

    savings = token_savings_rows(cropped_report, uncropped_report)
    assert all(row["input_tokens_saved_pct"] > 50 for row in savings)


def test_report_to_rows_produces_one_row_per_violation_type(tmp_path):
    dataset = _build_dataset(tmp_path)
    report = run_eval(dataset, _OracleClient())
    rows = report_to_rows(report)
    assert {row["violation_type"] for row in rows} == {"glove", "helmet"}
    assert all("precision" in row and "input_tokens" in row for row in rows)


def test_run_full_comparison_produces_all_four_combinations(tmp_path):
    dataset = _build_dataset(tmp_path)
    matrix = run_full_comparison(
        dataset,
        zero_shot_client=_AlwaysVerdictClient("uncertain"),
        few_shot_client=_OracleClient(),
    )

    # zero-shot (always uncertain) is identical whether cropped or not —
    # the oracle's answers differ from it, but crop/no-crop shouldn't
    # change what a fixed-verdict client says.
    assert matrix.zero_shot_cropped.overall_counts == matrix.zero_shot_uncropped.overall_counts
    assert matrix.zero_shot_cropped.overall_counts.uncertain == 3

    # few-shot (oracle) gets everything right in both crop modes...
    assert matrix.few_shot_cropped.overall_counts == matrix.few_shot_uncropped.overall_counts
    assert matrix.few_shot_cropped.overall_counts.true_positive == 2
    # ...but uncropped costs more tokens for the same accuracy.
    assert matrix.few_shot_cropped.total_usage.input_tokens < matrix.few_shot_uncropped.total_usage.input_tokens


def test_full_comparison_rows_covers_all_four_labels(tmp_path):
    dataset = _build_dataset(tmp_path)
    matrix = run_full_comparison(
        dataset,
        zero_shot_client=_AlwaysVerdictClient("uncertain"),
        few_shot_client=_OracleClient(),
    )
    rows = full_comparison_rows(matrix)

    labels = {row["label"] for row in rows}
    assert labels == {
        "zero-shot / cropped",
        "zero-shot / uncropped",
        "few-shot / cropped",
        "few-shot / uncropped",
    }
    # 2 violation types x 4 combinations
    assert len(rows) == 8


def test_full_comparison_token_savings_reports_both_shot_types_separately(tmp_path):
    dataset = _build_dataset(tmp_path)
    matrix = run_full_comparison(
        dataset,
        zero_shot_client=_AlwaysVerdictClient("uncertain"),
        few_shot_client=_OracleClient(),
    )
    savings = full_comparison_token_savings(matrix)

    shot_types = {row["shot_type"] for row in savings}
    assert shot_types == {"zero-shot", "few-shot"}
    for row in savings:
        assert "input_tokens_saved_pct" in row


def test_run_six_method_comparison_produces_all_six_methods(tmp_path):
    dataset = _build_dataset(tmp_path)
    matrix = run_six_method_comparison(
        dataset,
        zero_shot_client=_AlwaysVerdictClient("uncertain"),
        few_shot_client=_OracleClient(),
    )

    # zero-shot (always uncertain) is identical across the no-filter and
    # filter conditions for a dataset with no too_near_forklift cases —
    # bypassing a filter that never runs anyway shouldn't change what a
    # fixed-verdict client says.
    assert matrix.zero_shot_base_cropped.overall_counts == matrix.zero_shot_filter_cropped.overall_counts
    assert matrix.zero_shot_base_uncropped.overall_counts == matrix.zero_shot_filter_uncropped.overall_counts
    assert matrix.zero_shot_base_cropped.overall_counts.uncertain == 3

    # few-shot (oracle) gets everything right in both crop modes...
    assert matrix.few_shot_filter_cropped.overall_counts == matrix.few_shot_filter_uncropped.overall_counts
    assert matrix.few_shot_filter_cropped.overall_counts.true_positive == 2
    # ...but uncropped costs more tokens for the same accuracy.
    assert (
        matrix.few_shot_filter_cropped.total_usage.input_tokens
        < matrix.few_shot_filter_uncropped.total_usage.input_tokens
    )

    assert matrix.zero_shot_base_cropped.filter_bypassed is True
    assert matrix.zero_shot_filter_cropped.filter_bypassed is False
    assert matrix.few_shot_filter_cropped.filter_bypassed is False


def test_method_comparison_rows_covers_all_six_labels(tmp_path):
    dataset = _build_dataset(tmp_path)
    matrix = run_six_method_comparison(
        dataset,
        zero_shot_client=_AlwaysVerdictClient("uncertain"),
        few_shot_client=_OracleClient(),
    )
    rows = method_comparison_rows(matrix)

    labels = {row["label"] for row in rows}
    assert labels == {
        "zero-shot (no filter) / cropped",
        "zero-shot (no filter) / uncropped",
        "zero-shot (filter) / cropped",
        "zero-shot (filter) / uncropped",
        "few-shot (filter) / cropped",
        "few-shot (filter) / uncropped",
    }
    # 2 violation types x 6 methods
    assert len(rows) == 12


def test_run_glove_crop_experiment_filters_to_glove_cases_and_runs_all_three_variants(tmp_path):
    dataset = _build_dataset(tmp_path)  # glove_tp, glove_fp, helmet_tp
    client = _OracleClient()  # usage scales with len(image_bytes), so pad differences are visible
    results = run_glove_crop_experiment(dataset, client, config=PipelineConfig())

    from ppe_pipeline.models import ViolationType

    assert set(results.keys()) == {"tight", "wide", "tight+wide"}
    for variant, report in results.items():
        assert report.label == variant
        assert set(report.counts.keys()) == {ViolationType.GLOVE}
        assert {co.case.case_id for co in report.case_outcomes} == {"glove_tp", "glove_fp"}

    # A wider pad crops a strictly bigger box, so it must cost more input
    # tokens (the oracle's usage scales with image byte size) — this is
    # the signal that each variant actually used different crop geometry.
    tokens = {variant: report.total_usage.input_tokens for variant, report in results.items()}
    assert tokens["tight"] < tokens["wide"] < tokens["tight+wide"]


def test_glove_crop_experiment_forces_person_box_even_when_region_box_is_set(tmp_path):
    # Tester's exact repro: person_box area 80000 (200x400), region_box area
    # 6000 (100x60) — well inside the 2500-12100 range Tester flagged.
    # run_glove_crop_experiment must always pad person_box for its own
    # crop-building, never silently fall back to region_box just because no
    # real dataset happens to populate it today.
    Image.new("RGB", (2000, 2000), color="white").save(tmp_path / "scene.jpg")
    manifest = [
        {
            "case_id": "glove_with_region",
            "image": "scene.jpg",
            "violation_type": "glove",
            "person_box": [500, 500, 700, 900],
            "region_box": [550, 500, 650, 560],
            "ground_truth": "TP",
        }
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    dataset = load_eval_dataset(manifest_path)

    crop_sizes = {}

    def on_case(label, done, total, case_outcome, image_bytes, media_type):
        with Image.open(io.BytesIO(image_bytes)) as img:
            crop_sizes[label] = img.size

    run_glove_crop_experiment(dataset, _OracleClient(), config=PipelineConfig(), on_case=on_case)

    # "tight" (pad=0.0) must equal person_box's own size (200x400) exactly,
    # not region_box's (100x60) — confirming the crop tracks person_box's
    # scale, not region_box's, even though region_box was set on the case.
    assert crop_sizes["tight"] == (200, 400)
    assert crop_sizes["tight"] != (100, 60)


def test_glove_crop_experiment_rows_tags_each_row_with_its_variant(tmp_path):
    dataset = _build_dataset(tmp_path)
    results = run_glove_crop_experiment(dataset, _OracleClient(), config=PipelineConfig())
    rows = glove_crop_experiment_rows(results)

    assert {row["variant"] for row in rows} == {"tight", "wide", "tight+wide"}
    assert all(row["violation_type"] == "glove" for row in rows)
    assert len(rows) == 3  # one violation type (glove) x 3 variants


class _HelmetClearingClient:
    """Clears every helmet case as "helmet_worn" and reports where the
    helmet is — the one verdict that carries a helmet_box."""

    def judge(self, image_bytes, media_type, person_id, violation_types, extra_context=None):
        response = JudgeResponse(
            person_id=person_id,
            results=[
                JudgeResult(
                    violation_type=v,
                    verdict="FP",
                    reason="helmet_worn",
                    confidence="high",
                    confidence_score=0.95,
                    helmet_box=[0.2, 0.1, 0.6, 0.4],
                    explanation="Hard hat with a domed shell.",
                )
                for v in violation_types
            ],
        )
        return JudgeCallResult(response=response, usage=JudgeUsage(input_tokens=100, output_tokens=20))


def _helmet_dataset(tmp_path: Path) -> list:
    Image.new("RGB", (1000, 1000), color="white").save(tmp_path / "scene.jpg")
    manifest = [
        {
            "case_id": "helmet_fp",
            "image": "scene.jpg",
            "violation_type": "helmet",
            "person_box": [100, 100, 300, 500],
            "region_box": [150, 110, 250, 200],
            "ground_truth": "FP",
        }
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return load_eval_dataset(manifest_path)


def test_a_cleared_helmet_carries_the_judges_box_onto_the_case_outcome(tmp_path):
    dataset = _helmet_dataset(tmp_path)

    report = run_eval(dataset, _HelmetClearingClient(), label="helmet")

    outcome = report.case_outcomes[0]
    assert outcome.verdict == "FP"
    assert outcome.helmet_box == [0.2, 0.1, 0.6, 0.4]


def test_a_case_the_judge_never_saw_has_no_helmet_box(tmp_path):
    """The geometric proximity filter short-circuits too_near_forklift cases
    before any judge call, so there is no box to report."""
    dataset = _build_dataset(tmp_path)
    forklift_cases = [c for c in dataset if c.violation_type is ViolationType.TOO_NEAR_FORKLIFT]

    report = run_eval(forklift_cases, _OracleClient(), label="forklift")

    assert all(o.helmet_box is None for o in report.case_outcomes)


def test_a_cleared_helmet_also_carries_the_box_in_original_image_coordinates(tmp_path):
    """The crop-relative box is only meaningful next to the crop it was
    judged on; the original-image one survives without it."""
    dataset = _helmet_dataset(tmp_path)

    report = run_eval(dataset, _HelmetClearingClient(), label="helmet")

    outcome = report.case_outcomes[0]
    crop = crop_box_for_violation(
        outcome.case.violation_type,
        outcome.case.person_box,
        (1000, 1000),
        region_box=outcome.case.region_box,
        config=PipelineConfig(),
    )
    assert outcome.helmet_box_original.as_tuple() == judge_box_to_original(
        outcome.helmet_box, crop
    ).as_tuple()
    # It lands inside the crop the judge was actually shown.
    assert crop.containment_ratio(outcome.helmet_box_original) == 1.0


def test_an_uncropped_run_maps_the_box_straight_onto_the_full_frame(tmp_path):
    """With use_crop=False the judge sees the whole image, so its normalized
    box scales onto the full frame with no offset."""
    dataset = _helmet_dataset(tmp_path)

    report = run_eval(dataset, _HelmetClearingClient(), label="helmet", use_crop=False)

    outcome = report.case_outcomes[0]
    assert outcome.helmet_box == [0.2, 0.1, 0.6, 0.4]
    assert outcome.helmet_box_original.as_tuple() == (200, 100, 600, 400)
