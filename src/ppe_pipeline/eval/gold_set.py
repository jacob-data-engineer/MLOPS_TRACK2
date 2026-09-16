"""Gold Standard Evaluation Set: dual human review of eval cases, frozen
into a scoring-ready dataset — see roadmap step 1, "Annotation policy and
Gold Standard Evaluation Set".

`gold_annotations.json` is a pure sidecar (keyed by `case_id`) recording two
reviewers' labels plus an adjudicated final label, mirroring how
`few_shot/annotations.json` sidecars `few_shot/manifest.json` (see
`few_shot.py`). `freeze_gold_set` turns that sidecar plus a base case list
into `gold_set_v1.json` — the frozen file eval runs actually score against —
by resolving each case's `final_label` to a `ground_truth` of "TP"/"FP" and
dropping cases with no annotation yet or a `cannot_determine` verdict.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from .. import storage

logger = logging.getLogger(__name__)

GOLD_LABELS = ["violation", "no_violation", "cannot_determine"]
EVIDENCE_SUFFICIENCY = ["sufficient", "insufficient"]
DIFFICULTY = ["easy", "medium", "hard"]
ADJUDICATION_STATUS = ["not_needed", "pending", "adjudicated"]
GOLD_SET_REVIEWERS = ["Aidil", "Jacob", "Dom", "Brendon", "Han"]

# Bug 4 safety net: the only per-role fields `update_case_annotation` will
# accept when a `role` is passed in — a lightweight guard against a caller
# handing it a mismatched-role key set, not a full permission framework.
ROLE_ALLOWED_FIELDS = {
    "reviewer_1": {"reviewer_a_label", "reviewer_a_rationale", "reviewer_a_name"},
    "reviewer_2": {"reviewer_b_label", "reviewer_b_rationale", "reviewer_b_name"},
    "adjudicator": {
        "adjudication_status", "final_label", "final_rationale",
        "reason_code", "evidence_sufficiency", "difficulty",
    },
}


def load_gold_annotations(path: Path) -> dict[str, dict]:
    value, _ = storage.get_storage().read_json(path.as_posix(), default={})
    return value


def save_gold_annotations(path: Path, annotations: dict) -> None:
    storage.get_storage().update_json(path.as_posix(), lambda _current: annotations, default={})


def update_case_annotation(
    path: Path, case_id: str, updates: dict, role: str | None = None
) -> dict[str, dict]:
    """Concurrency-safe partial update of one case's annotation entry.

    Re-reads `path` fresh (never trusts an in-memory copy from earlier in
    the caller's session, since another reviewer may have written to it
    since), merges only the keys in `updates` into that
    case's entry, runs `apply_auto_resolution` on the merged entry, and
    writes the result back via `storage.update_json` — which retries the
    whole read-merge-write cycle if another reviewer's save raced it
    (local: a `filelock.FileLock`-guarded critical section; S3: ETag-based
    compare-and-swap), so agreement detection/resolution is atomic with the
    write rather than a separate step a race could land between.

    If `role` is given, `updates` may only contain keys in
    `ROLE_ALLOWED_FIELDS[role]` — anything else raises `ValueError`. If
    `role` is `None` (the intentional "no identity selected, everything
    editable" mode), no validation is performed.

    Raises `storage.ConcurrentModificationError` if a real reviewer conflict
    survives every retry — this is expected to be rare (a handful of named
    reviewers, and a save is one fast round trip), but it is not caught
    here. Callers should treat it as "someone else just saved this case —
    reload and retry" rather than an unhandled error.
    """
    if role is not None:
        allowed = ROLE_ALLOWED_FIELDS.get(role, set())
        disallowed = set(updates) - allowed
        if disallowed:
            raise ValueError(f"role {role!r} cannot update fields: {sorted(disallowed)}")

    def mutate(fresh: dict) -> dict:
        entry = fresh.setdefault(case_id, {})
        entry.update(updates)
        fresh[case_id] = apply_auto_resolution(entry)
        return fresh

    return storage.get_storage().update_json(path.as_posix(), mutate, default={})


def compute_agreement(entry: dict) -> str:
    label_a = entry.get("reviewer_a_label")
    label_b = entry.get("reviewer_b_label")
    if not label_a or not label_b:
        return "single_review"
    return "agree" if label_a == label_b else "disagree"


def reviewers_done(entry: dict) -> bool:
    return compute_agreement(entry) != "single_review"


def _merge_reviewer_rationales(entry: dict) -> str | None:
    """Concatenate both reviewers' rationale into a single `final_rationale`
    for an agreed case, since there's no adjudicator around to write one.
    Returns `None` if either reviewer's rationale is missing/blank, so a
    caller never fabricates ground-truth reasoning out of half the story.
    """
    rationale_a = (entry.get("reviewer_a_rationale") or "").strip()
    rationale_b = (entry.get("reviewer_b_rationale") or "").strip()
    if not rationale_a or not rationale_b:
        return None

    def _with_period(text: str) -> str:
        return text if text[-1] in ".!?" else f"{text}."

    return f"Reviewer A: {_with_period(rationale_a)} Reviewer B: {_with_period(rationale_b)}"


def apply_auto_resolution(entry: dict) -> dict:
    """Auto-resolve a case the moment both reviewers agree, so agreement
    alone (no adjudicator) is sufficient to reach a `final_label` — per the
    "adjudication only on disagreement" policy. Mutates `entry` in place
    (and returns it too, for clarity at call sites).

    An agreed case also has no adjudicator to write a `final_rationale`, so
    one is synthesized from both reviewers' individual rationales (via
    `_merge_reviewer_rationales`) whenever `final_rationale` isn't already
    set — this never overwrites a rationale a human adjudicator wrote
    (which only happens on disagreement, but a case can in principle be
    re-run through this function after already carrying one). An agreed
    `"cannot_determine"` verdict is excluded from synthesis: both reviewers
    agreeing they can't determine the case isn't a real verdict with
    reasoning to capture, and `rationale_sync.sync_ground_truth_rationale`
    already refuses to propagate a `cannot_determine` case's rationale
    downstream — synthesizing one here would just fabricate ground-truth
    text that never gets used but still pollutes the sidecar.
    """
    agreement = compute_agreement(entry)
    if agreement == "agree":
        entry["final_label"] = entry["reviewer_a_label"]
        entry["adjudication_status"] = "not_needed"
        if (
            entry["final_label"] != "cannot_determine"
            and (not entry.get("final_rationale") or not entry["final_rationale"].strip())
        ):
            merged_rationale = _merge_reviewer_rationales(entry)
            if merged_rationale:
                entry["final_rationale"] = merged_rationale
    elif agreement == "disagree" and not entry.get("final_label"):
        entry["adjudication_status"] = "pending"
    return entry


def case_rotation(case_id: str, people: list[str] = GOLD_SET_REVIEWERS) -> list[str]:
    """The full round-robin ordering for `case_id`, rotated to start at a
    position derived from a fixed hash of `case_id` itself — not its
    position in any surrounding case list — so a case's rotation (and
    therefore its role assignment) is permanently stable regardless of what
    other cases exist in the dataset, past or future. `hashlib.md5` (not
    Python's built-in `hash()`, which is randomized per-process for
    strings) is used so the result is deterministic across runs/machines.
    """
    n = len(people)
    i = int(hashlib.md5(case_id.encode()).hexdigest(), 16) % n
    return [people[(i + k) % n] for k in range(n)]


def role_for(
    person: str,
    case_id: str,
    people: list[str] = GOLD_SET_REVIEWERS,
) -> str | None:
    """Deterministic round-robin role assignment, computed per case rather
    than stored in an assignment file."""
    rotated = case_rotation(case_id, people)
    if person == rotated[0]:
        return "reviewer_1"
    if person == rotated[1]:
        return "reviewer_2"
    if person == rotated[2]:
        return "adjudicator"
    return None


def to_ground_truth(label: str) -> str | None:
    if label == "violation":
        return "TP"
    if label == "no_violation":
        return "FP"
    if label == "cannot_determine":
        return None
    raise ValueError(f"invalid gold label: {label!r}")


def freeze_gold_set(
    base_cases: list[dict], annotations: dict[str, dict], few_shot_manifest: dict
) -> tuple[list[dict], list[str]]:
    """Build the frozen gold-set scoring file from `base_cases` (any case
    list shaped like a `combined_eval_dataset.json`/`stratified_80.json`
    entry — dicts, not `EvalCase` objects) plus their `gold_annotations.json`
    entries. Cases with no annotation yet, or with `final_label ==
    "cannot_determine"`, are excluded entirely (per CEO decision — tracked
    only in the sidecar, never scored).

    Also flags any included case whose image filename also appears
    somewhere in the few-shot manifest, so a reviewer/CEO can see the
    overlap rather than have it silently contaminate the eval set.
    """
    few_shot_images: set[str] = set()
    for entries in few_shot_manifest.values():
        for entry in entries:
            few_shot_images.add(Path(entry["image"]).name)

    output_cases: list[dict] = []
    warnings: list[str] = []

    for case in base_cases:
        entry = annotations.get(case["case_id"])
        if not entry:
            continue
        final_label = entry.get("final_label")
        if not final_label or final_label == "cannot_determine":
            continue

        output_cases.append(
            {
                **case,
                "ground_truth": to_ground_truth(final_label),
                "reason_code": entry.get("reason_code"),
                "evidence_sufficiency": entry.get("evidence_sufficiency"),
                "difficulty": entry.get("difficulty"),
                "final_label": final_label,
                "final_rationale": entry.get("final_rationale"),
            }
        )

        image_name = Path(case["image"]).name
        if image_name in few_shot_images:
            warnings.append(
                f"Case {case['case_id']} image also appears in the few-shot library ({image_name})"
            )

        missing_fields = [
            field
            for field in ("reason_code", "evidence_sufficiency", "difficulty")
            if not entry.get(field)
        ]
        if missing_fields:
            warnings.append(
                f"Case {case['case_id']} is missing: {', '.join(missing_fields)}"
            )

    return output_cases, warnings


def rewrite_case_rationales(
    path: Path,
    case_id: str,
    violation_type=None,
    config=None,
) -> dict | None:
    """Best-effort pass that replaces one case's rationale text with the
    professionalised paragraphs from `rationale_rewrite`.

    Call this AFTER `update_case_annotation` has already persisted the
    reviewer's own words, never instead of it. Returns the refreshed
    annotations dict when something was rewritten, or `None` when there was
    nothing to do or the rewrite failed — it never raises, so a slow or down
    API costs a reviewer nothing but the polish.

    The API calls deliberately happen OUTSIDE `storage.update_json`'s
    critical section. Doing them inside would hold the local filelock (or
    burn S3 CAS retries) across a multi-second model call while other
    reviewers wait, and every CAS retry would re-issue the calls. Instead
    this reads the entry, rewrites, then takes the lock only to write the
    result back — and a field whose stored text changed while the rewrite
    was in flight is dropped rather than overwriting whatever the human just
    typed.
    """
    from ..config import get_config
    from .. import rationale_rewrite

    config = config or get_config()
    if not config.rewriter.rewrite_on_save:
        return None

    backend = storage.get_storage()
    try:
        current, _ = backend.read_json(path.as_posix(), default={})
        before = current.get(case_id)
        if not before:
            return None

        after, rewritten, failed = rationale_rewrite.rewrite_entry_fields(
            before, violation_type=violation_type, config=config
        )
        if failed:
            logger.warning(
                "rationale rewrite incomplete for %s; left as written: %s",
                case_id, ", ".join(failed),
            )
        if not rewritten:
            return None

        def mutate(fresh: dict) -> dict:
            entry = fresh.get(case_id)
            if not entry:
                return fresh
            hashes = dict(entry.get(rationale_rewrite.HASHES_FIELD) or {})
            changed = False
            for field in rewritten:
                # Only replace text that is still what we rewrote. If a human
                # saved over it in the meantime, theirs wins and the next save
                # will rewrite the new text.
                if entry.get(field) != before.get(field):
                    continue
                entry[field] = after[field]
                hashes[field] = after[rationale_rewrite.HASHES_FIELD][field]
                changed = True
            if not changed:
                return fresh
            entry[rationale_rewrite.HASHES_FIELD] = hashes
            fresh[case_id] = entry
            return fresh

        return backend.update_json(path.as_posix(), mutate, default={})
    except Exception:
        logger.exception("rationale rewrite pass failed for %s; annotation is unaffected", case_id)
        return None
