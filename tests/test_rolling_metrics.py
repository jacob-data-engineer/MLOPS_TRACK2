"""Tests for `ppe_pipeline.rolling_metrics`, including regressions for two
bugs an adversarial test pass found:

- Bug 1: a non-string `timestamp` (e.g. a JSON number) made
  `datetime.fromisoformat` raise `TypeError`, which the old `except
  ValueError:` didn't catch.
- Bug 2: a corrupt/malformed (or 0-byte) JSON audit-log/gold-annotations
  file made `json.loads` raise `json.JSONDecodeError` uncaught, breaking
  the module's own documented "degrades to zeroed/empty results rather
  than raising" contract.
"""

from __future__ import annotations

import json

import pytest

from ppe_pipeline.rolling_metrics import compute_shadow_mode_agreement, compute_track1_metrics


def _row(alarm_id, outcome, hazard_type="no_helmet", confidence=0.9, timestamp="2026-08-14T00:00:00+00:00"):
    return {
        "alarm_id": alarm_id,
        "hazard_type": hazard_type,
        "outcome": outcome,
        "confidence": confidence,
        "timestamp": timestamp,
    }


# --- compute_track1_metrics: baseline behavior --------------------------


def test_compute_track1_metrics_missing_file_returns_zeroed(tmp_path):
    result = compute_track1_metrics(tmp_path / "does_not_exist.json")

    assert result["total_alarms"] == 0
    assert result["outcome_counts"] == {"CONFIRM_TP": 0, "CONFIRM_FP": 0, "ESCALATE": 0}
    assert result["auto_annotation_rate"] == 0.0
    assert result["label_volume_this_week"] == 0
    assert result["flag_miss_count"] == 0
    assert "no_helmet" in result["per_hazard_type"]


def test_compute_track1_metrics_basic_counts(tmp_path):
    path = tmp_path / "audit_log.json"
    path.write_text(
        json.dumps(
            [
                _row("a1", "CONFIRM_TP", confidence=0.9),
                _row("a2", "CONFIRM_FP", confidence=0.5),
                _row("a3", "ESCALATE", hazard_type="hatch_fall_zone", confidence=0.4),
            ]
        )
    )

    result = compute_track1_metrics(path)

    assert result["total_alarms"] == 3
    assert result["outcome_counts"] == {"CONFIRM_TP": 1, "CONFIRM_FP": 1, "ESCALATE": 1}
    # auto_annotation_rate is derived from `outcome` (CONFIRM_TP/CONFIRM_FP
    # vs. ESCALATE), not from each row's own `confidence` value -- so it's
    # 2/3 here (a1 CONFIRM_TP + a2 CONFIRM_FP) regardless of a2's low
    # confidence=0.5, and independent of whatever confidence threshold
    # `_resolve_outcome` used when the row was originally written.
    assert result["auto_annotation_rate"] == pytest.approx(2 / 3)
    assert result["per_hazard_type"]["no_helmet"]["total"] == 2
    assert result["per_hazard_type"]["hatch_fall_zone"]["total"] == 1


# --- Bug 1 regression: non-string timestamp ------------------------------


def test_compute_track1_metrics_non_string_timestamp_does_not_raise(tmp_path):
    path = tmp_path / "audit_log.json"
    path.write_text(json.dumps([_row("a1", "CONFIRM_TP", timestamp=12345)]))

    result = compute_track1_metrics(path)

    assert result["total_alarms"] == 1
    # The bad timestamp can't be parsed, so it just doesn't count toward
    # the weekly volume -- the important thing is nothing raised.
    assert result["label_volume_this_week"] == 0


def test_compute_track1_metrics_unparseable_string_timestamp_does_not_raise(tmp_path):
    path = tmp_path / "audit_log.json"
    path.write_text(json.dumps([_row("a1", "CONFIRM_TP", timestamp="not-a-timestamp")]))

    result = compute_track1_metrics(path)

    assert result["total_alarms"] == 1
    assert result["label_volume_this_week"] == 0


# --- Bug 2 regression: corrupt/empty audit-log file ----------------------


def test_compute_track1_metrics_corrupt_json_degrades_to_zeroed(tmp_path):
    path = tmp_path / "audit_log.json"
    path.write_text("not valid json {{{")

    with pytest.warns(RuntimeWarning):
        result = compute_track1_metrics(path)

    assert result["total_alarms"] == 0
    assert result["outcome_counts"] == {"CONFIRM_TP": 0, "CONFIRM_FP": 0, "ESCALATE": 0}


def test_compute_track1_metrics_empty_file_degrades_to_zeroed(tmp_path):
    path = tmp_path / "audit_log.json"
    path.write_text("")

    with pytest.warns(RuntimeWarning):
        result = compute_track1_metrics(path)

    assert result["total_alarms"] == 0


def test_compute_track1_metrics_non_list_json_degrades_to_zeroed(tmp_path):
    path = tmp_path / "audit_log.json"
    path.write_text(json.dumps({"not": "a list"}))

    with pytest.warns(RuntimeWarning):
        result = compute_track1_metrics(path)

    assert result["total_alarms"] == 0


# --- compute_shadow_mode_agreement ---------------------------------------


def test_compute_shadow_mode_agreement_insufficient_data_below_threshold(tmp_path):
    audit_path = tmp_path / "audit_log.json"
    gold_path = tmp_path / "gold_annotations.json"
    audit_path.write_text(json.dumps([_row("case-1", "CONFIRM_TP")]))
    gold_path.write_text(json.dumps({"case-1": {"final_label": "violation"}}))

    result = compute_shadow_mode_agreement(audit_path, gold_path)

    assert result == {"insufficient_data": True, "matched_count": 1}


def test_compute_shadow_mode_agreement_computes_rate_once_threshold_met(tmp_path):
    audit_path = tmp_path / "audit_log.json"
    gold_path = tmp_path / "gold_annotations.json"

    rows = []
    gold = {}
    for i in range(10):
        outcome = "CONFIRM_TP" if i < 8 else "CONFIRM_FP"
        rows.append(_row(f"case-{i}", outcome))
        # case-0..7 (CONFIRM_TP) and case-9 (CONFIRM_FP) all agree with
        # gold; case-8 (CONFIRM_FP vs. a real violation) disagrees.
        if i < 8:
            gold[f"case-{i}"] = {"final_label": "violation"}
        elif i == 8:
            gold[f"case-{i}"] = {"final_label": "violation"}
        else:
            gold[f"case-{i}"] = {"final_label": "no_violation"}

    audit_path.write_text(json.dumps(rows))
    gold_path.write_text(json.dumps(gold))

    result = compute_shadow_mode_agreement(audit_path, gold_path)

    assert result["insufficient_data"] is False
    assert result["matched_count"] == 10
    assert result["agreement_count"] == 9
    assert result["agreement_rate"] == pytest.approx(0.9)


def test_compute_shadow_mode_agreement_missing_files_returns_insufficient_data(tmp_path):
    result = compute_shadow_mode_agreement(
        tmp_path / "no_audit_log.json", tmp_path / "no_gold_annotations.json"
    )

    assert result == {"insufficient_data": True, "matched_count": 0}


# --- Bug 2 regression: corrupt audit log / gold annotations --------------


def test_compute_shadow_mode_agreement_corrupt_audit_log_degrades_gracefully(tmp_path):
    audit_path = tmp_path / "audit_log.json"
    gold_path = tmp_path / "gold_annotations.json"
    audit_path.write_text("{{{not json")
    gold_path.write_text(json.dumps({"case-1": {"final_label": "violation"}}))

    with pytest.warns(RuntimeWarning):
        result = compute_shadow_mode_agreement(audit_path, gold_path)

    assert result == {"insufficient_data": True, "matched_count": 0}


def test_compute_shadow_mode_agreement_corrupt_gold_annotations_degrades_gracefully(tmp_path):
    audit_path = tmp_path / "audit_log.json"
    gold_path = tmp_path / "gold_annotations.json"
    audit_path.write_text(json.dumps([_row("case-1", "CONFIRM_TP")]))
    gold_path.write_text("")

    with pytest.warns(RuntimeWarning):
        result = compute_shadow_mode_agreement(audit_path, gold_path)

    assert result == {"insufficient_data": True, "matched_count": 0}


# --- what auto_annotation_rate is a claim about ---------------------------


def _cv_row(person_index: int, outcome: str, frame="frame-1"):
    """One judged crop of `frame`, the shape the live feed writes."""
    return {
        **_row(f"exec__{frame}__p{person_index:02d}", outcome),
        "row_kind": "helmet_cv",
        "cv": {
            "candidate_id": f"{frame}__p{person_index:02d}",
            "frame_name": f"{frame}.jpg",
            "person_index": person_index,
        },
    }


def test_a_frame_the_judge_both_fired_on_and_hedged_about_is_not_auto_annotated(tmp_path):
    """Its own question, and not readable off the outcome: TP outranks
    ESCALATE there, so this frame reads CONFIRM_TP while still owing a human
    an answer about the crop the judge could not read. Counting it claimed a
    frame a reviewer is queued to look at."""
    path = tmp_path / "audit_log.json"
    path.write_text(json.dumps([_cv_row(0, "CONFIRM_TP"), _cv_row(1, "ESCALATE")]))

    result = compute_track1_metrics(path)

    assert result["total_alarms"] == 1
    assert result["outcome_counts"] == {"CONFIRM_TP": 1, "CONFIRM_FP": 0, "ESCALATE": 0}
    assert result["auto_annotation_rate"] == 0.0


def test_a_frame_with_no_hedge_in_it_is_auto_annotated(tmp_path):
    """The numerator moved; it did not shrink to nothing."""
    path = tmp_path / "audit_log.json"
    path.write_text(json.dumps([_cv_row(0, "CONFIRM_TP"), _cv_row(1, "CONFIRM_FP")]))

    result = compute_track1_metrics(path)

    assert result["auto_annotation_rate"] == 1.0


def test_a_hedge_outranks_a_clearance_in_an_images_outcome(tmp_path):
    """The same order as `audit.case_outcome` and `live_feed.frame_verdict`:
    an image the judge declined to answer about is not one it cleared."""
    path = tmp_path / "audit_log.json"
    path.write_text(json.dumps([_cv_row(0, "CONFIRM_FP"), _cv_row(1, "ESCALATE")]))

    result = compute_track1_metrics(path)

    assert result["outcome_counts"] == {"CONFIRM_TP": 0, "CONFIRM_FP": 0, "ESCALATE": 1}
    assert result["auto_annotation_rate"] == 0.0
