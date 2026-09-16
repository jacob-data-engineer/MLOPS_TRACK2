"""Bucket judge verdicts into true positive / false positive / uncertain,
and build human-review queue entries for the uncertain bucket.

Each bucket is handled differently, not as two outcomes plus an edge case:
TP flows into the normal violation process, FP is suppressed, and uncertain
goes to a human review queue while the original flag stays open (a missed
real violation is worse than someone spending a few seconds on an
ambiguous crop).
"""

from __future__ import annotations

from dataclasses import dataclass

from .judge_client import JudgeResponse, JudgeResult
from .models import ViolationType


@dataclass
class ReviewQueueEntry:
    person_id: str
    violation_type: ViolationType
    crop_reference: str
    reason: str
    explanation: str


@dataclass
class ProcessedVerdicts:
    true_positives: list[JudgeResult]
    false_positives: list[JudgeResult]
    uncertain: list[JudgeResult]
    review_queue: list[ReviewQueueEntry]


def process_verdicts(response: JudgeResponse, crop_reference: str) -> ProcessedVerdicts:
    true_positives = [r for r in response.results if r.verdict == "TP"]
    false_positives = [r for r in response.results if r.verdict == "FP"]
    uncertain = [r for r in response.results if r.verdict == "uncertain"]

    review_queue = [
        ReviewQueueEntry(
            person_id=response.person_id,
            violation_type=r.violation_type,
            crop_reference=crop_reference,
            reason=r.reason,
            explanation=r.explanation,
        )
        for r in uncertain
    ]

    return ProcessedVerdicts(
        true_positives=true_positives,
        false_positives=false_positives,
        uncertain=uncertain,
        review_queue=review_queue,
    )


def is_flag_still_open(result: JudgeResult) -> bool:
    """False positives are suppressed; TP and uncertain keep the original
    flag open (uncertain defaults to open until a human resolves it)."""
    return result.verdict != "FP"
