import pytest

from ppe_pipeline.eval.metrics import ConfusionCounts, classify


@pytest.mark.parametrize(
    "ground_truth_is_violation, verdict, expected",
    [
        (True, "TP", "true_positive"),
        (True, "FP", "false_negative"),  # judge wrongly suppressed a real violation
        (False, "TP", "false_positive"),  # judge wrongly confirmed a false alarm
        (False, "FP", "true_negative"),
        (True, "uncertain", "uncertain"),
        (False, "uncertain", "uncertain"),
    ],
)
def test_classify_all_branches(ground_truth_is_violation, verdict, expected):
    assert classify(ground_truth_is_violation, verdict) == expected


def test_confusion_counts_metrics():
    counts = ConfusionCounts()
    for _ in range(8):
        counts.record("true_positive", True)
    for _ in range(2):
        counts.record("false_positive", False)
    for _ in range(1):
        counts.record("false_negative", True)
    for _ in range(5):
        counts.record("true_negative", False)
    # Two of the three hedges were on cases that really were violations.
    for _ in range(2):
        counts.record("uncertain", True)
    counts.record("uncertain", False)

    assert counts.total == 19
    assert counts.decided_total == 16
    assert counts.precision == pytest.approx(8 / 10)
    assert counts.recall == pytest.approx(8 / 9)
    assert counts.accuracy == pytest.approx(13 / 16)
    assert counts.uncertain_rate == pytest.approx(3 / 19)
    assert counts.f1 == pytest.approx(2 * counts.precision * counts.recall / (counts.precision + counts.recall))


def test_uncertain_splits_by_ground_truth_without_inflating_totals():
    counts = ConfusionCounts()
    for _ in range(8):
        counts.record("true_positive", True)
    for _ in range(1):
        counts.record("false_negative", True)
    for _ in range(2):
        counts.record("uncertain", True)
    for _ in range(5):
        counts.record("true_negative", False)
    counts.record("uncertain", False)

    # The split must never leak into the headline totals.
    assert counts.total == 17
    assert counts.uncertain == 3
    assert counts.uncertain_violation + counts.uncertain_no_violation == counts.uncertain

    assert counts.ground_truth_violation_total == 11
    assert counts.violation_verdict_distribution == {"hold": 8, "cleared": 1, "escalate": 2}
    assert sum(counts.violation_verdict_distribution.values()) == counts.ground_truth_violation_total

    assert counts.ground_truth_no_violation_total == 6
    assert counts.no_violation_verdict_distribution == {"hold": 0, "cleared": 5, "escalate": 1}
    assert sum(counts.no_violation_verdict_distribution.values()) == counts.ground_truth_no_violation_total


def test_adding_counts_carries_the_uncertain_split():
    a, b = ConfusionCounts(), ConfusionCounts()
    a.record("uncertain", True)
    b.record("uncertain", False)
    total = a + b
    assert total.uncertain == 2
    assert total.uncertain_violation == 1
    assert total.uncertain_no_violation == 1
    assert total.total == 2


def test_confusion_counts_are_additive():
    a = ConfusionCounts(true_positive=2, false_positive=1)
    b = ConfusionCounts(true_positive=3, uncertain=1)
    combined = a + b
    assert combined == ConfusionCounts(true_positive=5, false_positive=1, uncertain=1)


def test_empty_counts_do_not_divide_by_zero():
    counts = ConfusionCounts()
    assert counts.precision == 0.0
    assert counts.recall == 0.0
    assert counts.f1 == 0.0
    assert counts.accuracy == 0.0
    assert counts.uncertain_rate == 0.0
