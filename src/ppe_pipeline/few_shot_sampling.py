"""Few-shot examples drawn from cases a human has already audited.

WHY

The committed library under `few_shot/` was hand-picked once, from footage
that is not this site. Every audited case, by contrast, is this site's own
camera, this site's own workers, and a verdict a person here stood behind.
Using them as the judge's prior turns is the cheapest way to make it argue in
the terms this yard actually uses.

WHAT IT DOES NOT DO

It never touches `few_shot/`. That library is the OTHER option on the run
screen, so editing it would destroy the thing a run is being compared
against. Samples are written to their own directory, whole, every time
(CEO, 2 Sep: "at all times the fixed one should not be edited coz it is an
option to run, so maybe a different folder for this sample").

BALANCE, AND WHAT HAPPENS WHEN THERE IS NOT ENOUGH

The sample is balanced by verdict: half examples of a violation, half of a
clearance. An unbalanced prior is a thumb on the scale -- a judge shown nine
violations and one clearance has been told what the answer usually is.

Audited data is rarely balanced, so a short side is TOPPED UP from the fixed
library rather than left short or padded from the long side. With eight
audited violations and a target of 36 a side, the sample carries those eight
plus 28 of the committed violation examples, picked at random against a
recorded seed. That is exactly the CEO's instruction ("if sample chosen but
only 8 confirm violations in the audited results, use the 8 and randomly
replace the ones that are fix").

WHICH CROPS QUALIFY

Only crops whose human verdict AGREES with the judge's own. The example
carries an explanation, and the only explanation on file is the judge's --
so on a case a human overturned, the recorded reasoning argues for the
opposite of the label it would be paired with. Teaching the model with an
argument against its own label is worse than one example fewer.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

from . import storage
from .cropping import crop_bytes
from .models import BoundingBox, ViolationType

logger = logging.getLogger(__name__)

#: Where a sample is written. NOT `few_shot/`, which is the fixed library and
#: the other half of the choice being offered.
SAMPLE_DIR = "few_shot_sampled"

#: Examples per side. The committed helmet library holds 37 violations and 35
#: clearances, so this matches the prior a fixed run gets.
DEFAULT_PER_SIDE = 36

#: What a row's outcome means as a few-shot verdict.
_OUTCOME_TO_VERDICT = {"CONFIRM_TP": "TP", "CONFIRM_FP": "FP"}

#: The human verdict each of those has to agree with to be usable.
_VERDICT_TO_HUMAN = {"TP": "violation", "FP": "no_violation"}


@dataclass(frozen=True)
class Candidate:
    """One audited crop, ready to become a few-shot example."""

    row: dict
    verdict: str
    reason: str
    explanation: str

    @property
    def frame_key(self) -> str:
        return str(self.row.get("image_key") or "")

    @property
    def crop_box(self) -> list[float] | None:
        box = self.row.get("crop_box")
        if not isinstance(box, dict):
            return None
        return [float(box[k]) for k in ("x1", "y1", "x2", "y2")]


def candidates(
    rows: list[dict],
    audits: dict[str, dict],
    *,
    violation_type: ViolationType = ViolationType.HELMET,
) -> list[Candidate]:
    """Every audited crop whose human verdict backs the judge's own words."""
    from .audit import AUDIT_DECISIONS, is_cv_row, reviewer_verdict
    from .escalations import _entry_for
    from .judge_prompt import REASON_CODES
    from .training_export import _case_key_of

    usable: list[Candidate] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        verdict = _OUTCOME_TO_VERDICT.get(str(row.get("outcome") or ""))
        if verdict is None:
            continue
        key = _case_key_of(row) if is_cv_row(row) else row.get("alarm_id")
        entry = _entry_for(audits, str(key or ""))
        if entry.get("decision") not in AUDIT_DECISIONS:
            continue
        # The human's own words, recovered by the function that owns the
        # agree/disagree encoding rather than decoded a second time here.
        if reviewer_verdict(entry) != _VERDICT_TO_HUMAN[verdict]:
            continue
        reason, explanation = row.get("reason_code"), row.get("reasoning")
        if not reason or not explanation or not row.get("image_key"):
            continue
        # A REASON THE JUDGE COULD HAVE GIVEN. Rows settled without a judge
        # call carry their own codes (`no_person_detected`), which are not
        # helmet reasons and would fail `FewShotExample`'s own validation --
        # and an example whose reason the model never emits teaches nothing
        # about how to answer.
        if str(reason) not in REASON_CODES[violation_type]:
            continue
        usable.append(Candidate(row=row, verdict=verdict, reason=str(reason),
                                explanation=str(explanation)))
    return usable


def _fixed_examples(manifest_path: Path, violation_type: ViolationType) -> list[dict]:
    """The committed library's own entries for this type, read not copied."""
    raw, _ = storage.get_storage().read_json(manifest_path.as_posix(), default={})
    return [e for e in (raw or {}).get(violation_type.value, []) if isinstance(e, dict)]


def build_sample(
    rows: list[dict],
    audits: dict[str, dict],
    *,
    fixed_manifest: Path,
    violation_type: ViolationType = ViolationType.HELMET,
    per_side: int = DEFAULT_PER_SIDE,
    seed: int | None = None,
) -> dict:
    """Decide what the sample contains, without writing anything.

    Returns the plan: the audited crops chosen per side, and the committed
    examples topping each side up. Separated from the write so what a run
    WOULD use can be shown before a single crop is cut.
    """
    rng = random.Random(seed)
    pool = candidates(rows, audits, violation_type=violation_type)
    fixed = _fixed_examples(fixed_manifest, violation_type)

    plan: dict = {"violation_type": violation_type.value, "per_side": per_side,
                  "seed": seed, "sides": {}}
    for verdict in ("TP", "FP"):
        audited = [c for c in pool if c.verdict == verdict]
        rng.shuffle(audited)
        taken = audited[:per_side]
        # TOPPED UP FROM THE COMMITTED LIBRARY, not from the other side. A
        # short side padded with its opposite is not a balanced prior, it is
        # a differently biased one.
        spare = [e for e in fixed if e.get("verdict") == verdict]
        rng.shuffle(spare)
        fill = spare[: max(0, per_side - len(taken))]
        plan["sides"][verdict] = {
            "audited_available": len(audited),
            "audited_used": len(taken),
            "fixed_used": len(fill),
            "audited": taken,
            "fixed": fill,
        }
    return plan


def write_sample(plan: dict, *, fixed_manifest: Path, out_dir: str = SAMPLE_DIR) -> dict:
    """Write the sample: its crops, and a manifest `load_few_shot_library` reads.

    THE MANIFEST IS AUTHORITATIVE. It is rewritten whole and names only what
    this sample chose, so nothing reads a crop it does not name. Crops are
    named after their case, so rebuilding overwrites rather than accumulates;
    a crop dropped by a later sample stays on disk, unreferenced and inert,
    and the set is bounded by the number of distinct audited crops.

    A committed example is REFERENCED, by a storage key back into
    `few_shot/`, not copied. Copying would move ~33 images on every build,
    which is ten seconds of round trips at this bucket's latency, and would
    make the copy the thing somebody edits one day.
    """
    backend = storage.get_storage()
    entries: list[dict] = []
    written = 0
    for verdict in ("TP", "FP"):
        side = plan["sides"][verdict]
        for candidate in side["audited"]:
            image_key = _write_crop(backend, candidate, out_dir)
            if image_key is None:
                continue
            written += 1
            entries.append({
                "image": image_key,
                "verdict": candidate.verdict,
                "reason": candidate.reason,
                # The judge's own confidence is a float on the row and this
                # field is a three-way word. `high` is honest for an example a
                # human confirmed: the point of an audited example is that
                # somebody stood behind it.
                "confidence": "high",
                "explanation": candidate.explanation,
                "source": "audited",
                "alarm_id": candidate.row.get("alarm_id"),
            })
        for example in side["fixed"]:
            entries.append({
                **example,
                # A storage key back into the committed library, not a copy
                # of it and not a relative path: `validate_key` refuses the
                # `..` a relative one would need.
                "image_key": (fixed_manifest.parent / example["image"]).as_posix(),
                "source": "fixed",
            })

    manifest = {plan["violation_type"]: entries}
    backend.put_bytes(
        f"{out_dir}/manifest.json",
        json.dumps(manifest, indent=2).encode("utf-8"),
        content_type="application/json",
    )
    return {
        "manifest": f"{out_dir}/manifest.json",
        "examples": len(entries),
        "crops_written": written,
        "audited_used": {v: plan["sides"][v]["audited_used"] for v in ("TP", "FP")},
        "fixed_used": {v: plan["sides"][v]["fixed_used"] for v in ("TP", "FP")},
        "seed": plan["seed"],
    }


def _write_crop(backend, candidate: Candidate, out_dir: str) -> str | None:
    """One audited crop, cut from its frame and written into the sample.

    Returns the manifest-relative path, or `None` when the frame is gone --
    a missing frame costs one example, never the sample.
    """
    box = candidate.crop_box
    if box is None:
        return None
    try:
        frame = backend.get_bytes(candidate.frame_key)
        cropped, _ = crop_bytes(frame, BoundingBox(*box))
    except Exception:  # noqa: BLE001 - one lost example must not end the build
        logger.warning("could not cut the crop for %s", candidate.row.get("alarm_id"))
        return None
    name = f"crops/{_safe(str(candidate.row.get('alarm_id') or 'case'))}.jpg"
    backend.put_bytes(f"{out_dir}/{name}", cropped, content_type="image/jpeg")
    return name


def _safe(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in value)[:120]
