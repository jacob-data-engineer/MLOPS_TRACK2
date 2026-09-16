"""Tests for `ppe_pipeline.track1_adapter` — `_resolve_outcome`'s truth
table and `run_track1_judge_inference`'s end-to-end behavior (with a fake
`JudgeClient.judge` standing in for a real model call)."""

from __future__ import annotations

import json

import pytest
from PIL import Image

from ppe_pipeline.cropping import crops_for_frame, judge_box_to_original
from ppe_pipeline.judge_client import JudgeCallResult, JudgeResponse, JudgeResult, JudgeUsage
from ppe_pipeline.models import BoundingBox
from ppe_pipeline.models import ViolationType as VT
from ppe_pipeline.track1_adapter import (
    VIOLATION_TYPE_TO_HAZARD_LABEL,
    _resolve_outcome,
    run_track1_judge_inference,
)


# --- _resolve_outcome ---------------------------------------------------


def test_resolve_outcome_fp_always_confirms_regardless_of_confidence():
    assert _resolve_outcome("FP", 0.01) == "CONFIRM_FP"
    assert _resolve_outcome("FP", 0.99) == "CONFIRM_FP"


def test_resolve_outcome_tp_confirms_above_threshold():
    assert _resolve_outcome("TP", 0.9) == "CONFIRM_TP"


def test_resolve_outcome_tp_at_exactly_half_confirms():
    # CEO decision: a confidence_score of exactly 0.5 must confirm, not
    # escalate.
    assert _resolve_outcome("TP", 0.5) == "CONFIRM_TP"


def test_resolve_outcome_tp_below_threshold_escalates():
    assert _resolve_outcome("TP", 0.3) == "ESCALATE"


def test_resolve_outcome_uncertain_always_escalates():
    assert _resolve_outcome("uncertain", 0.99) == "ESCALATE"


def test_resolve_outcome_tp_with_missing_confidence_score_escalates():
    # Not every provider's structured output reliably populates
    # confidence_score (Bedrock's JSON mode is not a strict-schema
    # guarantee) — a missing score on a real TP must escalate to a human,
    # not crash comparing None to a float.
    assert _resolve_outcome("TP", None) == "ESCALATE"


def test_resolve_outcome_fp_with_missing_confidence_score_still_confirms():
    assert _resolve_outcome("FP", None) == "CONFIRM_FP"


# --- run_track1_judge_inference -----------------------------------------


def _make_image_bytes(width=200, height=200) -> bytes:
    import io

    buf = io.BytesIO()
    Image.new("RGB", (width, height), color="white").save(buf, format="JPEG")
    return buf.getvalue()


class _FakeJudgeClient:
    """Stands in for `JudgeClient.judge` — used via monkeypatching
    `ppe_pipeline.track1_adapter.JudgeClient` to a factory returning this,
    so `run_track1_judge_inference`'s own `JudgeClient(...)` construction
    doesn't need real provider credentials."""

    last_call_kwargs: dict | None = None

    def __init__(self, verdict="TP", confidence="high", confidence_score=0.9, reason="bare_head", explanation="x"):
        self._verdict = verdict
        self._confidence = confidence
        self._confidence_score = confidence_score
        self._reason = reason
        self._explanation = explanation

    def judge(self, image_bytes, media_type, person_id, violation_types, extra_context=None):
        type(self).last_call_kwargs = {
            "image_bytes": image_bytes,
            "media_type": media_type,
            "person_id": person_id,
            "violation_types": violation_types,
            "extra_context": extra_context,
        }
        result = JudgeResult(
            violation_type=violation_types[0],
            verdict=self._verdict,
            reason=self._reason,
            confidence=self._confidence,
            confidence_score=self._confidence_score,
            explanation=self._explanation,
        )
        return JudgeCallResult(
            response=JudgeResponse(person_id=person_id, results=[result]), usage=JudgeUsage()
        )


def _patch_judge_client(monkeypatch, fake_client: _FakeJudgeClient) -> None:
    monkeypatch.setattr(
        "ppe_pipeline.track1_adapter.JudgeClient",
        lambda **kwargs: fake_client,
    )


def test_run_track1_judge_inference_confirms_high_confidence_tp(monkeypatch, tmp_path):
    fake_client = _FakeJudgeClient(verdict="TP", confidence="high", confidence_score=0.92)
    _patch_judge_client(monkeypatch, fake_client)

    audit_log_path = tmp_path / "audit_log.json"
    result = run_track1_judge_inference(
        image_bytes=_make_image_bytes(),
        media_type="image/jpeg",
        violation_type=VT.HELMET,
        person_box=BoundingBox(x1=10, y1=10, x2=100, y2=150),
        alarm_id="alarm-xyz",
        source_detection_id="detection-xyz",
        audit_log_path=audit_log_path,
    )

    assert result.alarm_id == "alarm-xyz"
    assert result.hazard_type == VIOLATION_TYPE_TO_HAZARD_LABEL[VT.HELMET]
    assert result.violation_type == "helmet"
    assert result.verdict == "TP"
    assert result.confidence_level == "high"
    assert result.confidence_score == 0.92
    assert result.outcome == "CONFIRM_TP"

    rows = json.loads(audit_log_path.read_text())
    assert len(rows) == 1
    row = rows[0]
    assert row["alarm_id"] == "alarm-xyz"
    assert row["source_detection_id"] == "detection-xyz"
    assert row["outcome"] == "CONFIRM_TP"
    assert row["confidence"] == 0.92
    assert row["confidence_level"] == "high"
    assert row["hazard_type"] == "no_helmet"
    assert "vehicle_box" not in row


def test_run_track1_judge_inference_low_confidence_tp_escalates(monkeypatch, tmp_path):
    fake_client = _FakeJudgeClient(verdict="TP", confidence="low", confidence_score=0.3)
    _patch_judge_client(monkeypatch, fake_client)

    result = run_track1_judge_inference(
        image_bytes=_make_image_bytes(),
        media_type="image/jpeg",
        violation_type=VT.HELMET,
        person_box=BoundingBox(x1=10, y1=10, x2=100, y2=150),
        alarm_id="alarm-1",
        source_detection_id="detection-1",
        audit_log_path=tmp_path / "audit_log.json",
    )

    assert result.outcome == "ESCALATE"


def test_run_track1_judge_inference_appends_across_calls(monkeypatch, tmp_path):
    fake_client = _FakeJudgeClient()
    _patch_judge_client(monkeypatch, fake_client)
    audit_log_path = tmp_path / "audit_log.json"

    for alarm_id in ("alarm-1", "alarm-2"):
        run_track1_judge_inference(
            image_bytes=_make_image_bytes(),
            media_type="image/jpeg",
            violation_type=VT.HELMET,
            person_box=BoundingBox(x1=10, y1=10, x2=100, y2=150),
            alarm_id=alarm_id,
            source_detection_id="d",
            audit_log_path=audit_log_path,
        )

    rows = json.loads(audit_log_path.read_text())
    assert [row["alarm_id"] for row in rows] == ["alarm-1", "alarm-2"]


def test_run_track1_judge_inference_forklift_filter_drop_skips_judge(monkeypatch, tmp_path):
    fake_client = _FakeJudgeClient()
    _patch_judge_client(monkeypatch, fake_client)
    _FakeJudgeClient.last_call_kwargs = None

    result = run_track1_judge_inference(
        image_bytes=_make_image_bytes(width=1000, height=1000),
        media_type="image/jpeg",
        violation_type=VT.TOO_NEAR_FORKLIFT,
        # Person and forklift boxes far enough apart that the geometric
        # proximity filter drops this before it ever reaches the judge.
        person_box=BoundingBox(x1=0, y1=0, x2=20, y2=20),
        forklift_box=BoundingBox(x1=900, y1=900, x2=990, y2=990),
        alarm_id="alarm-forklift",
        source_detection_id="d",
        audit_log_path=tmp_path / "audit_log.json",
    )

    assert result.verdict == "FP"
    assert result.confidence_level == "high"
    assert result.outcome == "CONFIRM_FP"
    assert result.reason == "geometric_proximity_filter"
    assert "Filtered before reaching the judge" in result.explanation
    # The judge was never called for this case.
    assert _FakeJudgeClient.last_call_kwargs is None


def test_run_track1_judge_inference_forklift_operator_context_reaches_judge(monkeypatch, tmp_path):
    fake_client = _FakeJudgeClient(verdict="FP", confidence="high", confidence_score=0.9)
    _patch_judge_client(monkeypatch, fake_client)
    _FakeJudgeClient.last_call_kwargs = None

    result = run_track1_judge_inference(
        image_bytes=_make_image_bytes(width=200, height=200),
        media_type="image/jpeg",
        violation_type=VT.TOO_NEAR_FORKLIFT,
        # Person box almost entirely contained within the forklift's own
        # box -- flagged "possible operator", still reaches the judge.
        person_box=BoundingBox(x1=10, y1=10, x2=30, y2=30),
        forklift_box=BoundingBox(x1=0, y1=0, x2=40, y2=40),
        alarm_id="alarm-operator",
        source_detection_id="d",
        audit_log_path=tmp_path / "audit_log.json",
    )

    assert result.verdict == "FP"
    assert _FakeJudgeClient.last_call_kwargs is not None
    assert "contained within the forklift" in _FakeJudgeClient.last_call_kwargs["extra_context"]


def test_run_track1_judge_inference_includes_vehicle_box_when_present(monkeypatch, tmp_path):
    fake_client = _FakeJudgeClient(verdict="TP", confidence="high", confidence_score=0.9)
    _patch_judge_client(monkeypatch, fake_client)
    audit_log_path = tmp_path / "audit_log.json"

    run_track1_judge_inference(
        image_bytes=_make_image_bytes(width=200, height=200),
        media_type="image/jpeg",
        violation_type=VT.TOO_NEAR_FORKLIFT,
        person_box=BoundingBox(x1=10, y1=10, x2=30, y2=30),
        forklift_box=BoundingBox(x1=20, y1=20, x2=60, y2=60),
        alarm_id="alarm-vehicle",
        source_detection_id="d",
        audit_log_path=audit_log_path,
    )

    rows = json.loads(audit_log_path.read_text())
    assert rows[0]["vehicle_box"] == {"x1": 20.0, "y1": 20.0, "x2": 60.0, "y2": 60.0}


def test_prepared_crop_does_not_reopen_the_original_frame(monkeypatch, tmp_path):
    image_bytes = _make_image_bytes(width=200, height=100)
    prepared = crops_for_frame(
        image_bytes,
        [{"candidate_id": "candidate", "person_index": 0, "person_box": [10, 10, 80, 90]}],
        VT.HELMET,
    )[0]
    fake_client = _FakeJudgeClient(verdict="TP", confidence="high", confidence_score=0.9)
    _patch_judge_client(monkeypatch, fake_client)
    monkeypatch.setattr("ppe_pipeline.track1_adapter.Image.open", lambda *_: pytest.fail("re-decoded"))

    result = run_track1_judge_inference(
        image_bytes=image_bytes,
        media_type="image/jpeg",
        violation_type=VT.HELMET,
        person_box=BoundingBox(10, 10, 80, 90),
        alarm_id="safe_execution_id",
        source_detection_id="safe_execution_id",
        audit_log_path=tmp_path / "audit.json",
        prepared_crop=prepared,
    )

    assert result.crop_box == prepared.crop_box
    assert result.judged_image_bytes == prepared.image_bytes


class _HelmetClearingJudgeClient(_FakeJudgeClient):
    """Clears the alarm as "helmet_worn" and reports where the helmet is."""

    def __init__(self, helmet_box):
        super().__init__(verdict="FP", confidence="high", confidence_score=0.95, reason="helmet_worn")
        self._helmet_box = helmet_box

    def judge(self, image_bytes, media_type, person_id, violation_types, extra_context=None):
        call = super().judge(image_bytes, media_type, person_id, violation_types, extra_context)
        result = call.response.results[0].model_copy(update={"helmet_box": self._helmet_box})
        return JudgeCallResult(
            response=call.response.model_copy(update={"results": [result]}), usage=call.usage
        )


def test_run_track1_judge_inference_returns_the_helmet_box_in_both_spaces(monkeypatch, tmp_path):
    # Held onto so `last_call_kwargs` is read off the class that actually
    # set it — `_FakeJudgeClient.last_call_kwargs` would find a stale value
    # left by an earlier test, since the subclass sets its own.
    fake_client = _HelmetClearingJudgeClient([0.25, 0.15, 0.7, 0.55])
    _patch_judge_client(monkeypatch, fake_client)
    person_box = BoundingBox(x1=10, y1=10, x2=100, y2=150)

    result = run_track1_judge_inference(
        image_bytes=_make_image_bytes(),
        media_type="image/jpeg",
        violation_type=VT.HELMET,
        person_box=person_box,
        alarm_id="alarm-helmet",
        source_detection_id="detection-helmet",
        audit_log_path=tmp_path / "audit_log.json",
        # This test is about coordinate spaces, so it asks for the box source
        # that always produces one. In cv mode a detector that finds nothing
        # now records no box at all.
        helmet_box_mode="vlm",
    )

    assert result.helmet_box == [0.25, 0.15, 0.7, 0.55]
    # Mapped back through the crop the judge was actually shown.
    assert result.helmet_box_original == judge_box_to_original(result.helmet_box, result.crop_box)
    assert result.crop_box.containment_ratio(result.helmet_box_original) == 1.0
    # The bytes the judge saw come back, so a UI renders what was judged.
    assert result.judged_image_bytes == type(fake_client).last_call_kwargs["image_bytes"]


def test_the_audit_row_stores_the_helmet_box_in_original_coordinates(monkeypatch, tmp_path):
    """Crop-relative coordinates would go stale the moment cropping is
    retuned, since the row can't reproduce the crop."""
    _patch_judge_client(monkeypatch, _HelmetClearingJudgeClient([0.25, 0.15, 0.7, 0.55]))
    audit_log_path = tmp_path / "audit_log.json"

    result = run_track1_judge_inference(
        image_bytes=_make_image_bytes(),
        media_type="image/jpeg",
        violation_type=VT.HELMET,
        person_box=BoundingBox(x1=10, y1=10, x2=100, y2=150),
        alarm_id="alarm-helmet",
        source_detection_id="detection-helmet",
        audit_log_path=audit_log_path,
        helmet_box_mode="vlm",
    )

    row = json.loads(audit_log_path.read_text())[0]
    assert row["helmet_box"] == {
        "x1": result.helmet_box_original.x1,
        "y1": result.helmet_box_original.y1,
        "x2": result.helmet_box_original.x2,
        "y2": result.helmet_box_original.y2,
    }


def test_a_non_helmet_verdict_writes_no_helmet_box(monkeypatch, tmp_path):
    _patch_judge_client(monkeypatch, _FakeJudgeClient(verdict="TP", reason="bare_head"))
    audit_log_path = tmp_path / "audit_log.json"

    result = run_track1_judge_inference(
        image_bytes=_make_image_bytes(),
        media_type="image/jpeg",
        violation_type=VT.HELMET,
        person_box=BoundingBox(x1=10, y1=10, x2=100, y2=150),
        alarm_id="alarm-bare",
        source_detection_id="detection-bare",
        audit_log_path=audit_log_path,
    )

    assert result.helmet_box is None
    assert result.helmet_box_original is None
    assert "helmet_box" not in json.loads(audit_log_path.read_text())[0]


class _FakeHelmetDetector:
    """Stands in for the YOLO11s stage-2 detector so tests never load real
    weights. `located` of None means "found nothing above threshold"."""

    def __init__(self, located):
        self._located = located
        self.locate_calls = 0

    def locate(self, image_bytes, media_type=None):
        self.locate_calls += 1
        return self._located


def _patch_helmet_detector(monkeypatch, detector):
    """Overrides conftest's autouse fixture, which disables the detector for
    the rest of the suite. Patched on the module because `track1_adapter`
    reaches it as `helmet_detector.get_helmet_detector(...)`."""
    from ppe_pipeline import helmet_detector

    monkeypatch.setattr(helmet_detector, "get_helmet_detector", lambda *a, **kw: detector)
    return detector


def test_run_pipeline_prefers_the_detector_box_over_the_inline_one(monkeypatch, tmp_path):
    """The inline box is too loose to label from, so a cleared helmet must
    go through the stage-2 detector here exactly as it does in an eval run —
    otherwise the same case gets a different box on the two screens."""
    from ppe_pipeline.helmet_detector import HelmetLocation

    _patch_judge_client(monkeypatch, _HelmetClearingJudgeClient([0.1, 0.1, 0.9, 0.9]))
    detector = _patch_helmet_detector(
        monkeypatch,
        _FakeHelmetDetector(HelmetLocation(box=[0.3, 0.2, 0.5, 0.4], confidence=0.8)),
    )

    result = run_track1_judge_inference(
        image_bytes=_make_image_bytes(),
        media_type="image/jpeg",
        violation_type=VT.HELMET,
        person_box=BoundingBox(x1=10, y1=10, x2=100, y2=150),
        alarm_id="alarm-locate",
        source_detection_id="detection-locate",
        audit_log_path=tmp_path / "audit_log.json",
    )

    assert detector.locate_calls == 1
    assert result.helmet_box == [0.3, 0.2, 0.5, 0.4]
    assert result.helmet_box_confidence == 0.8


def test_in_cv_mode_a_failed_localization_records_no_box_at_all(monkeypatch, tmp_path):
    """The detector is the chosen source, so its silence is the answer.

    The judge's inline box is deliberately NOT a fallback here. Keeping it
    would record a box nobody asked for, draw it on the review crop, and let
    a reviewer accept it in one press while believing they were endorsing
    YOLO11s. The verdict is unaffected either way; only the box is.
    """
    _patch_judge_client(monkeypatch, _HelmetClearingJudgeClient([0.1, 0.1, 0.9, 0.9]))
    _patch_helmet_detector(monkeypatch, _FakeHelmetDetector(None))

    result = run_track1_judge_inference(
        image_bytes=_make_image_bytes(),
        media_type="image/jpeg",
        violation_type=VT.HELMET,
        person_box=BoundingBox(x1=10, y1=10, x2=100, y2=150),
        alarm_id="alarm-fallback",
        source_detection_id="detection-fallback",
        audit_log_path=tmp_path / "audit_log.json",
    )

    assert result.helmet_box is None
    assert result.helmet_box_original is None
    assert result.helmet_box_source is None
    assert result.helmet_box_confidence is None
    # The verdict still stands; only the box is withheld.
    assert result.verdict == "FP"


def test_vlm_mode_still_uses_the_inline_box(monkeypatch, tmp_path):
    """The deliberate way to produce loose boxes for testing the annotation
    tool. Choosing the VLM explicitly must keep working."""
    _patch_judge_client(monkeypatch, _HelmetClearingJudgeClient([0.1, 0.1, 0.9, 0.9]))
    _patch_helmet_detector(monkeypatch, _FakeHelmetDetector(None))

    result = run_track1_judge_inference(
        image_bytes=_make_image_bytes(),
        media_type="image/jpeg",
        violation_type=VT.HELMET,
        person_box=BoundingBox(x1=10, y1=10, x2=100, y2=150),
        alarm_id="alarm-vlm-mode",
        source_detection_id="detection-vlm-mode",
        audit_log_path=tmp_path / "audit_log.json",
        helmet_box_mode="vlm",
    )

    assert result.helmet_box == [0.1, 0.1, 0.9, 0.9]
    assert result.helmet_box_source == "vlm"


def test_a_non_cleared_helmet_makes_no_localization_call(monkeypatch, tmp_path):
    """Asking "where is it" only makes sense once the verdict says there is
    one — every other verdict must not pay for the call."""

    _patch_judge_client(monkeypatch, _FakeJudgeClient(verdict="TP", reason="bare_head"))
    detector = _patch_helmet_detector(monkeypatch, _FakeHelmetDetector(None))

    run_track1_judge_inference(
        image_bytes=_make_image_bytes(),
        media_type="image/jpeg",
        violation_type=VT.HELMET,
        person_box=BoundingBox(x1=10, y1=10, x2=100, y2=150),
        alarm_id="alarm-tp",
        source_detection_id="detection-tp",
        audit_log_path=tmp_path / "audit_log.json",
    )

    assert detector.locate_calls == 0


def test_a_detector_box_is_labelled_cv(monkeypatch, tmp_path):
    """A reviewer has to be able to tell a detector box from the fallback:
    only the detector reports a confidence, and its boxes are much tighter."""
    from ppe_pipeline.helmet_detector import HelmetLocation

    _patch_judge_client(monkeypatch, _HelmetClearingJudgeClient([0.1, 0.1, 0.9, 0.9]))
    _patch_helmet_detector(
        monkeypatch, _FakeHelmetDetector(HelmetLocation(box=[0.3, 0.2, 0.5, 0.4], confidence=0.8))
    )

    result = run_track1_judge_inference(
        image_bytes=_make_image_bytes(),
        media_type="image/jpeg",
        violation_type=VT.HELMET,
        person_box=BoundingBox(x1=10, y1=10, x2=100, y2=150),
        alarm_id="alarm-cv",
        source_detection_id="det-cv",
        audit_log_path=tmp_path / "audit_log.json",
    )

    assert result.helmet_box_source == "cv"
    assert result.helmet_box_confidence == 0.8
def test_no_box_at_all_has_no_source(monkeypatch, tmp_path):
    _patch_judge_client(monkeypatch, _HelmetClearingJudgeClient(None))
    _patch_helmet_detector(monkeypatch, _FakeHelmetDetector(None))

    result = run_track1_judge_inference(
        image_bytes=_make_image_bytes(),
        media_type="image/jpeg",
        violation_type=VT.HELMET,
        person_box=BoundingBox(x1=10, y1=10, x2=100, y2=150),
        alarm_id="alarm-nobox",
        source_detection_id="det-nobox",
        audit_log_path=tmp_path / "audit_log.json",
    )

    assert result.helmet_box_source is None
    assert result.helmet_box is None


def test_the_audit_row_records_which_model_drew_the_box(monkeypatch, tmp_path):
    """Provenance has to survive into the audit log, or a later retraining
    pass cannot tell detector labels from VLM guesses."""
    import json

    from ppe_pipeline.helmet_detector import HelmetLocation

    audit_log_path = tmp_path / "audit_log.json"
    _patch_judge_client(monkeypatch, _HelmetClearingJudgeClient([0.1, 0.1, 0.9, 0.9]))
    _patch_helmet_detector(
        monkeypatch, _FakeHelmetDetector(HelmetLocation(box=[0.3, 0.2, 0.5, 0.4], confidence=0.66))
    )

    run_track1_judge_inference(
        image_bytes=_make_image_bytes(),
        media_type="image/jpeg",
        violation_type=VT.HELMET,
        person_box=BoundingBox(x1=10, y1=10, x2=100, y2=150),
        alarm_id="alarm-audit-src",
        source_detection_id="det-audit-src",
        audit_log_path=audit_log_path,
    )

    row = json.loads(audit_log_path.read_text(encoding="utf-8"))[-1]
    assert row["helmet_box_source"] == "cv"
    assert row["helmet_box_confidence"] == 0.66


def test_helmet_box_mode_vlm_skips_the_detector(monkeypatch, tmp_path):
    """A deliberate way to produce the loose fallback box the review queue
    exists to catch. The detector clears most cases well above the review
    threshold, so bad examples are otherwise hard to come by."""
    from ppe_pipeline.helmet_detector import HelmetLocation

    _patch_judge_client(monkeypatch, _HelmetClearingJudgeClient([0.1, 0.1, 0.9, 0.9]))
    detector = _patch_helmet_detector(
        monkeypatch, _FakeHelmetDetector(HelmetLocation(box=[0.3, 0.2, 0.5, 0.4], confidence=0.9))
    )

    result = run_track1_judge_inference(
        image_bytes=_make_image_bytes(),
        media_type="image/jpeg",
        violation_type=VT.HELMET,
        person_box=BoundingBox(x1=10, y1=10, x2=100, y2=150),
        alarm_id="alarm-vlm-mode",
        source_detection_id="det-vlm-mode",
        audit_log_path=tmp_path / "audit_log.json",
        helmet_box_mode="vlm",
    )

    assert detector.locate_calls == 0
    assert result.helmet_box_source == "vlm"
    assert result.helmet_box_confidence is None
    assert result.helmet_box == [0.1, 0.1, 0.9, 0.9]


def test_helmet_box_mode_defaults_to_the_detector(monkeypatch, tmp_path):
    from ppe_pipeline.helmet_detector import HelmetLocation

    _patch_judge_client(monkeypatch, _HelmetClearingJudgeClient([0.1, 0.1, 0.9, 0.9]))
    detector = _patch_helmet_detector(
        monkeypatch, _FakeHelmetDetector(HelmetLocation(box=[0.3, 0.2, 0.5, 0.4], confidence=0.9))
    )

    result = run_track1_judge_inference(
        image_bytes=_make_image_bytes(),
        media_type="image/jpeg",
        violation_type=VT.HELMET,
        person_box=BoundingBox(x1=10, y1=10, x2=100, y2=150),
        alarm_id="alarm-cv-default",
        source_detection_id="det-cv-default",
        audit_log_path=tmp_path / "audit_log.json",
    )

    assert detector.locate_calls == 1
    assert result.helmet_box_source == "cv"


def test_the_audit_row_records_the_judged_crop_region(monkeypatch, tmp_path):
    """Recorded rather than recomputed at review time: the crop depends on
    config pad factors AND on region_box, which is not stored, so a later
    config change would reconstruct a different crop and a box drawn on it
    would map back to the wrong place."""
    import json

    audit_log_path = tmp_path / "audit_log.json"
    _patch_judge_client(monkeypatch, _HelmetClearingJudgeClient([0.1, 0.1, 0.9, 0.9]))

    result = run_track1_judge_inference(
        image_bytes=_make_image_bytes(),
        media_type="image/jpeg",
        violation_type=VT.HELMET,
        person_box=BoundingBox(x1=10, y1=10, x2=100, y2=150),
        alarm_id="alarm-crop",
        source_detection_id="det-crop",
        audit_log_path=audit_log_path,
    )

    row = json.loads(audit_log_path.read_text(encoding="utf-8"))[-1]
    assert set(row["crop_box"]) == {"x1", "y1", "x2", "y2"}
    assert result.crop_box is not None
    assert row["crop_box"]["x1"] == result.crop_box.x1


# --- the golden row -------------------------------------------------------


def test_the_single_path_audit_row_is_exactly_this(monkeypatch, tmp_path):
    """A golden row for `run_track1_judge_inference`.

    `POST /track1/run`, `/track1/run-job`, `/track1/batch-run`, the eval
    stepper in `routers/eval.py` and `eval/runner.run_eval` all call this one
    function, and `rolling_metrics` reads the rows back. Its shape is
    therefore load-bearing in five places that do not test each other, and
    the helmet-CV pilot now calls it too.

    Pinned as an EXACT key set and exact values (bar the timestamp, which is
    wall-clock) rather than as spot checks, because the failure this guards
    against is not a wrong value -- the other tests here catch those -- it is
    a field quietly appearing or disappearing while a shared helper is
    factored out underneath. `row_kind` and `cv` in particular must be absent
    from a non-CV row: their presence is what tells a pilot row from a live
    alarm, and a live alarm that grew one would be counted as pilot data.
    """
    _patch_judge_client(monkeypatch, _FakeJudgeClient(verdict="TP", confidence="high", confidence_score=0.92))
    audit_log_path = tmp_path / "audit_log.json"

    run_track1_judge_inference(
        image_bytes=_make_image_bytes(width=200, height=200),
        media_type="image/jpeg",
        violation_type=VT.HELMET,
        person_box=BoundingBox(x1=10, y1=10, x2=100, y2=150),
        alarm_id="alarm-golden",
        source_detection_id="detection-golden",
        audit_log_path=audit_log_path,
        alarm_image_dir=tmp_path / "frames",
    )

    row = json.loads(audit_log_path.read_text())[0]
    timestamp = row.pop("timestamp")

    assert row == {
        "alarm_id": "alarm-golden",
        "hazard_type": "no_helmet",
        "outcome": "CONFIRM_TP",
        "reason_code": "bare_head",
        "confidence": 0.92,
        "confidence_level": "high",
        "reasoning": "x",
        "source_detection_id": "detection-golden",
        "person_box": {"x1": 10, "y1": 10, "x2": 100, "y2": 150},
        "image_width": 200,
        "image_height": 200,
        "crop_box": {"x1": 0, "y1": 0, "x2": 127, "y2": 192},
        "judge_provider": "bedrock",
        "judge_model": "qwen.qwen3-vl-235b-a22b-instruct",
        "image_key": f"{(tmp_path / 'frames').as_posix()}/alarm-golden.jpg",
    }
    assert timestamp.endswith("+00:00")


def test_a_non_cv_row_carries_no_pilot_marker(monkeypatch, tmp_path):
    """The other half of the golden row, called out on its own because it is
    a correctness boundary rather than a shape: an untagged CV row would be
    counted as a live alarm by `rolling_metrics`, and a tagged live alarm
    would be counted as pilot data."""
    _patch_judge_client(monkeypatch, _FakeJudgeClient())
    audit_log_path = tmp_path / "audit_log.json"

    run_track1_judge_inference(
        image_bytes=_make_image_bytes(),
        media_type="image/jpeg",
        violation_type=VT.HELMET,
        person_box=BoundingBox(x1=10, y1=10, x2=100, y2=150),
        alarm_id="alarm-plain",
        source_detection_id="d",
        audit_log_path=audit_log_path,
        alarm_image_dir=tmp_path / "frames",
    )

    row = json.loads(audit_log_path.read_text())[0]

    assert "row_kind" not in row
    assert "cv" not in row


def test_the_cv_decision_only_adds_the_nested_block_and_the_marker(monkeypatch, tmp_path):
    """Passing `cv_decision` must change the row in exactly two ways. Anything
    else would mean the pilot path and the live path have diverged."""
    _patch_judge_client(monkeypatch, _FakeJudgeClient())
    plain_log = tmp_path / "plain.json"
    cv_log = tmp_path / "cv.json"
    kwargs = {
        "image_bytes": _make_image_bytes(),
        "media_type": "image/jpeg",
        "violation_type": VT.HELMET,
        "person_box": BoundingBox(x1=10, y1=10, x2=100, y2=150),
        "alarm_id": "alarm-same",
        "source_detection_id": "d",
        "alarm_image_dir": tmp_path / "frames",
    }

    run_track1_judge_inference(**kwargs, audit_log_path=plain_log)
    run_track1_judge_inference(**kwargs, audit_log_path=cv_log, cv_decision={"cv_run_id": "cv-1"})

    plain = json.loads(plain_log.read_text())[0]
    with_cv = json.loads(cv_log.read_text())[0]
    plain.pop("timestamp")
    with_cv.pop("timestamp")

    assert set(with_cv) - set(plain) == {"row_kind", "cv"}
    assert with_cv["row_kind"] == "helmet_cv"
    assert with_cv["cv"] == {"cv_run_id": "cv-1"}
    assert {k: v for k, v in with_cv.items() if k in plain} == plain
