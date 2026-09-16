"""Tests for few-shot examples drawn from audited cases.

The rule that matters is BALANCE, and what happens when there is not enough
of one side. Measured on the live corpus, 2 Sep: 71 audited clearances and 3
audited violations, which is exactly the case the CEO named.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from PIL import Image

from ppe_pipeline import few_shot_sampling as sampling
from ppe_pipeline import storage
from ppe_pipeline.few_shot import load_few_shot_library
from ppe_pipeline.models import ViolationType

FRAME = "unk__unk__20260427T070224Z__aaaa__f000450"


def _row(index: int, outcome: str, reason: str) -> dict:
    candidate = f"{FRAME}{index}__p00"
    return {
        "alarm_id": f"exec__{candidate}",
        "hazard_type": "no_helmet",
        "outcome": outcome,
        "reason_code": reason,
        "reasoning": f"reasoning for {candidate}",
        "timestamp": "2026-09-02T00:00:00+00:00",
        "image_key": "eval/alarm_images/frame.jpg",
        "crop_box": {"x1": 0, "y1": 0, "x2": 200, "y2": 400},
        "row_kind": "helmet_cv",
        "cv": {"frame_name": f"{FRAME}{index}.jpg", "candidate_id": candidate,
               "person_index": 0, "people_in_frame": 1},
    }


def _audit(index: int, verdict: str) -> tuple[str, dict]:
    return f"{FRAME}{index}", {
        "decision": "agree",
        "audited_verdict": verdict,
        "ground_truth_verdict": verdict,
    }


def _corpus(violations: int, clearances: int) -> tuple[list[dict], dict]:
    rows, audits = [], {}
    for i in range(violations):
        rows.append(_row(i, "CONFIRM_TP", "bare_head"))
        key, entry = _audit(i, "violation")
        audits[key] = entry
    for j in range(violations, violations + clearances):
        rows.append(_row(j, "CONFIRM_FP", "helmet_worn"))
        key, entry = _audit(j, "no_violation")
        audits[key] = entry
    return rows, audits


FIXED = Path("few_shot/manifest.json")


def test_a_balanced_sample_when_both_sides_are_plentiful():
    rows, audits = _corpus(violations=40, clearances=40)

    plan = sampling.build_sample(rows, audits, fixed_manifest=FIXED, per_side=36, seed=1)

    for verdict in ("TP", "FP"):
        assert plan["sides"][verdict]["audited_used"] == 36
        assert plan["sides"][verdict]["fixed_used"] == 0


def test_a_short_side_is_topped_up_from_the_fixed_library():
    """The CEO's own example: "if sample chosen but only 8 confirm violations
    in the audited results, use the 8 and randomly replace the ones that are
    fix"."""
    rows, audits = _corpus(violations=8, clearances=40)

    plan = sampling.build_sample(rows, audits, fixed_manifest=FIXED, per_side=36, seed=1)

    assert plan["sides"]["TP"]["audited_used"] == 8
    assert plan["sides"]["TP"]["fixed_used"] == 28
    assert plan["sides"]["FP"]["audited_used"] == 36
    assert plan["sides"]["FP"]["fixed_used"] == 0
    # Topped up from the SAME side, never from the plentiful one: a short side
    # padded with its opposite is not balanced, it is differently biased.
    assert all(e["verdict"] == "TP" for e in plan["sides"]["TP"]["fixed"])


def test_a_crop_the_human_overturned_is_not_used():
    """The example carries an explanation and the only one on file is the
    judge's, so on a case a human overturned it argues for the opposite of the
    label it would be paired with."""
    rows, audits = _corpus(violations=2, clearances=0)
    audits[f"{FRAME}0"] = {
        "decision": "disagree",
        "corrected_verdict": "no_violation",
        "audited_verdict": "violation",
    }

    usable = sampling.candidates(rows, audits)

    assert [c.row["alarm_id"] for c in usable] == [rows[1]["alarm_id"]]


def test_a_reason_the_judge_never_emits_is_not_an_example():
    """Rows settled without a judge call carry their own codes, which are not
    helmet reasons and would fail `FewShotExample`'s own validation."""
    rows, audits = _corpus(violations=0, clearances=1)
    rows[0]["reason_code"] = "no_person_detected"

    assert sampling.candidates(rows, audits) == []


def test_the_sample_is_reproducible_from_its_seed():
    rows, audits = _corpus(violations=40, clearances=40)

    one = sampling.build_sample(rows, audits, fixed_manifest=FIXED, per_side=8, seed=7)
    two = sampling.build_sample(rows, audits, fixed_manifest=FIXED, per_side=8, seed=7)

    def ids(plan):
        return [c.row["alarm_id"] for c in plan["sides"]["TP"]["audited"]]

    assert ids(one) == ids(two)


def test_the_written_sample_loads_as_a_library(tmp_path, monkeypatch):
    """End to end: what a sampled run actually judges against.

    The fixed half is REFERENCED by storage key rather than copied, so this
    also pins that `load_few_shot_library` resolves it. A relative
    `../few_shot/...` cannot do that job: `validate_key` refuses upward
    traversal in a key.
    """
    backend = storage.LocalStorage(tmp_path)
    monkeypatch.setattr(storage, "get_storage", lambda: backend)

    buffer = io.BytesIO()
    Image.new("RGB", (1920, 1080), (90, 90, 90)).save(buffer, format="JPEG")
    backend.put_bytes("eval/alarm_images/frame.jpg", buffer.getvalue(), content_type="image/jpeg")

    # A committed library of one example, in its own directory.
    backend.put_bytes("fixed/crops/a.jpg", buffer.getvalue(), content_type="image/jpeg")
    backend.put_bytes(
        "fixed/manifest.json",
        json.dumps({"helmet": [{"image": "crops/a.jpg", "verdict": "TP",
                                "reason": "bare_head", "confidence": "high",
                                "explanation": "no helmet"}]}).encode(),
        content_type="application/json",
    )

    rows, audits = _corpus(violations=1, clearances=1)
    plan = sampling.build_sample(
        rows, audits, fixed_manifest=Path("fixed/manifest.json"), per_side=2, seed=3
    )
    report = sampling.write_sample(plan, fixed_manifest=Path("fixed/manifest.json"))

    assert report["audited_used"] == {"TP": 1, "FP": 1}
    assert report["fixed_used"]["TP"] == 1  # topped up

    library = load_few_shot_library(Path(report["manifest"]))
    examples = library[ViolationType.HELMET]

    assert len(examples) == 3
    # Every example resolves to bytes the judge can actually be shown.
    for example in examples:
        assert backend.get_bytes(example.image_path.as_posix())
