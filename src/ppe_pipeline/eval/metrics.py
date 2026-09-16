"""Confusion-matrix metrics for judge verdicts against ground truth.

Ground truth is binary (a real violation, or a CV false alarm); the
judge's "uncertain" verdict is tracked as its own bucket rather than folded
into either outcome — per the architecture doc, uncertain goes to human
review and isn't counted as TP or FP in metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

Outcome = str  # "true_positive" | "false_negative" | "false_positive" | "true_negative" | "uncertain"


def classify(ground_truth_is_violation: bool, verdict: str) -> Outcome:
    """Bucket one judge verdict against its ground truth.

    Naming is from the judge's perspective, not the CV detector's: a
    `false_negative` here means the judge wrongly suppressed a genuine
    violation — the failure mode the doc calls worse than the false
    positives the judge exists to remove.
    """
    if verdict == "uncertain":
        return "uncertain"

    judge_confirms = verdict == "TP"
    if ground_truth_is_violation and judge_confirms:
        return "true_positive"
    if ground_truth_is_violation and not judge_confirms:
        return "false_negative"
    if not ground_truth_is_violation and judge_confirms:
        return "false_positive"
    return "true_negative"


_OUTCOME_FIELDS = ("true_positive", "false_negative", "false_positive", "true_negative", "uncertain")

# Counted alongside `uncertain`, not inside `_OUTCOME_FIELDS`: they split
# that one bucket by ground truth and would double-count every total built
# from the fields above.
_UNCERTAIN_SPLIT_FIELDS = ("uncertain_violation", "uncertain_no_violation")


@dataclass
class ConfusionCounts:
    true_positive: int = 0
    false_negative: int = 0
    false_positive: int = 0
    true_negative: int = 0
    uncertain: int = 0
    # `uncertain` alone cannot say which ground truth the judge hedged on,
    # and "how did the model answer the cases that really were violations"
    # is unanswerable without that: the other four outcomes each imply their
    # ground truth from the name, uncertain is the only ambiguous one. These
    # two always sum to `uncertain`.
    uncertain_violation: int = 0
    uncertain_no_violation: int = 0

    def record(self, outcome: Outcome, ground_truth_is_violation: bool) -> None:
        """`ground_truth_is_violation` is required rather than optional
        precisely because it is only strictly needed for `uncertain` — an
        optional argument would let a caller silently skip it there and
        leave the split under-counted with no error to notice."""
        setattr(self, outcome, getattr(self, outcome) + 1)
        if outcome == "uncertain":
            field = "uncertain_violation" if ground_truth_is_violation else "uncertain_no_violation"
            setattr(self, field, getattr(self, field) + 1)

    def __add__(self, other: ConfusionCounts) -> ConfusionCounts:
        return ConfusionCounts(
            **{
                field: getattr(self, field) + getattr(other, field)
                for field in (*_OUTCOME_FIELDS, *_UNCERTAIN_SPLIT_FIELDS)
            }
        )

    @property
    def total(self) -> int:
        return sum(getattr(self, field) for field in _OUTCOME_FIELDS)

    @property
    def ground_truth_violation_total(self) -> int:
        """Cases that really were violations, however the judge answered."""
        return self.true_positive + self.false_negative + self.uncertain_violation

    @property
    def ground_truth_no_violation_total(self) -> int:
        """Cases that really were false alarms, however the judge answered."""
        return self.true_negative + self.false_positive + self.uncertain_no_violation

    @property
    def violation_verdict_distribution(self) -> dict[str, int]:
        """How the judge answered the cases that really were violations, in
        the UI's own stamp words (see web VerdictStamp: TP is HOLD, FP is
        CLEARED, uncertain is ESCALATE). Sums to
        `ground_truth_violation_total`."""
        return {
            "hold": self.true_positive,
            "cleared": self.false_negative,
            "escalate": self.uncertain_violation,
        }

    @property
    def no_violation_verdict_distribution(self) -> dict[str, int]:
        """The same for cases that really were false alarms. Sums to
        `ground_truth_no_violation_total`."""
        return {
            "hold": self.false_positive,
            "cleared": self.true_negative,
            "escalate": self.uncertain_no_violation,
        }

    @property
    def decided_total(self) -> int:
        """Total excluding uncertain — the denominator for accuracy."""
        return self.total - self.uncertain

    @property
    def precision(self) -> float:
        """Of the violations the judge confirmed, how many were real."""
        denom = self.true_positive + self.false_positive
        return self.true_positive / denom if denom else 0.0

    @property
    def recall(self) -> float:
        """Of the real violations, how many did the judge confirm."""
        denom = self.true_positive + self.false_negative
        return self.true_positive / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def accuracy(self) -> float:
        denom = self.decided_total
        return (self.true_positive + self.true_negative) / denom if denom else 0.0

    @property
    def uncertain_rate(self) -> float:
        return self.uncertain / self.total if self.total else 0.0
