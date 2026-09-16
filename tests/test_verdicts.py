from ppe_pipeline.judge_client import JudgeResponse, JudgeResult
from ppe_pipeline.models import ViolationType
from ppe_pipeline.verdicts import is_flag_still_open, process_verdicts


def _result(violation_type, verdict, reason="x", confidence="high", explanation="e"):
    return JudgeResult(
        violation_type=violation_type,
        verdict=verdict,
        reason=reason,
        confidence=confidence,
        explanation=explanation,
    )


def test_process_verdicts_buckets_by_outcome():
    response = JudgeResponse(
        person_id="1",
        results=[
            _result(ViolationType.HELMET, "TP"),
            _result(ViolationType.GLOVE, "FP"),
            _result(ViolationType.DANGEROUS_PLANE, "uncertain"),
        ],
    )
    processed = process_verdicts(response, crop_reference="img.jpg:person_1")

    assert [r.violation_type for r in processed.true_positives] == [ViolationType.HELMET]
    assert [r.violation_type for r in processed.false_positives] == [ViolationType.GLOVE]
    assert [r.violation_type for r in processed.uncertain] == [ViolationType.DANGEROUS_PLANE]


def test_review_queue_only_contains_uncertain_results():
    response = JudgeResponse(
        person_id="1",
        results=[
            _result(ViolationType.HELMET, "TP"),
            _result(ViolationType.GLOVE, "FP"),
            _result(ViolationType.DANGEROUS_PLANE, "uncertain", reason="height_ambiguous", explanation="hard to tell"),
        ],
    )
    processed = process_verdicts(response, crop_reference="img.jpg:person_1")

    assert len(processed.review_queue) == 1
    entry = processed.review_queue[0]
    assert entry.violation_type is ViolationType.DANGEROUS_PLANE
    assert entry.reason == "height_ambiguous"
    assert entry.crop_reference == "img.jpg:person_1"


def test_false_positive_closes_the_flag_but_tp_and_uncertain_stay_open():
    assert is_flag_still_open(_result(ViolationType.HELMET, "TP")) is True
    assert is_flag_still_open(_result(ViolationType.HELMET, "uncertain")) is True
    assert is_flag_still_open(_result(ViolationType.HELMET, "FP")) is False
