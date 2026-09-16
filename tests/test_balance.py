from pathlib import Path

from ppe_pipeline.eval.balance import balance_by_ground_truth
from ppe_pipeline.eval.dataset import EvalCase
from ppe_pipeline.models import BoundingBox, ViolationType


def _case(case_id: str, violation_type: ViolationType, ground_truth_is_violation: bool | None) -> EvalCase:
    return EvalCase(
        case_id=case_id,
        image_path=Path(f"{case_id}.jpg"),
        violation_type=violation_type,
        person_box=BoundingBox(0, 0, 10, 10),
        ground_truth_is_violation=ground_truth_is_violation,
    )


def test_balance_caps_larger_bucket_down_to_smaller():
    cases = [
        _case("h_tp_1", ViolationType.HELMET, True),
        _case("h_tp_2", ViolationType.HELMET, True),
        _case("h_fp_1", ViolationType.HELMET, False),
        _case("h_fp_2", ViolationType.HELMET, False),
        _case("h_fp_3", ViolationType.HELMET, False),
        _case("h_fp_4", ViolationType.HELMET, False),
    ]

    balanced, warnings = balance_by_ground_truth(cases, seed=42)

    tp_count = sum(1 for c in balanced if c.ground_truth_is_violation)
    fp_count = sum(1 for c in balanced if not c.ground_truth_is_violation)
    assert tp_count == 2
    assert fp_count == 2
    assert any("capped no_violation 4 -> 2" in w for w in warnings)


def test_balance_is_independent_per_violation_type():
    cases = [
        _case("h_tp_1", ViolationType.HELMET, True),
        _case("h_fp_1", ViolationType.HELMET, False),
        _case("g_tp_1", ViolationType.GLOVE, True),
        _case("g_tp_2", ViolationType.GLOVE, True),
        _case("g_fp_1", ViolationType.GLOVE, False),
    ]

    balanced, warnings = balance_by_ground_truth(cases, seed=42)

    helmet_cases = [c for c in balanced if c.violation_type is ViolationType.HELMET]
    glove_cases = [c for c in balanced if c.violation_type is ViolationType.GLOVE]
    assert len(helmet_cases) == 2
    assert len(glove_cases) == 2
    assert any("glove: capped violation 2 -> 1" in w for w in warnings)


def test_balance_drops_cases_with_no_resolved_ground_truth():
    cases = [
        _case("h_tp_1", ViolationType.HELMET, True),
        _case("h_fp_1", ViolationType.HELMET, False),
        _case("h_none_1", ViolationType.HELMET, None),
    ]

    balanced, warnings = balance_by_ground_truth(cases, seed=42)

    assert {c.case_id for c in balanced} == {"h_tp_1", "h_fp_1"}
    assert any("dropped 1 case" in w for w in warnings)


def test_balance_no_warning_when_already_balanced():
    cases = [
        _case("h_tp_1", ViolationType.HELMET, True),
        _case("h_fp_1", ViolationType.HELMET, False),
    ]

    balanced, warnings = balance_by_ground_truth(cases, seed=42)

    assert len(balanced) == 2
    assert warnings == []


def test_balance_is_reproducible_for_same_seed():
    cases = [
        _case(f"h_tp_{i}", ViolationType.HELMET, True) for i in range(5)
    ] + [
        _case("h_fp_1", ViolationType.HELMET, False),
    ]

    first, _ = balance_by_ground_truth(cases, seed=42)
    second, _ = balance_by_ground_truth(cases, seed=42)

    assert [c.case_id for c in first] == [c.case_id for c in second]


def test_balance_empty_dataset_returns_empty():
    balanced, warnings = balance_by_ground_truth([], seed=42)

    assert balanced == []
    assert warnings == []
