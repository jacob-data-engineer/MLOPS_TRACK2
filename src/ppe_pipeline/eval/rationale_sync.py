"""Propagate a resolved gold-set reviewer rationale to every eval dataset
file that already has an entry for that case_id — the rationale-sync
counterpart to `ground_truth_sync.sync_ground_truth`.

The same labeled case (by `case_id`) is deliberately duplicated across
`eval/known_cases.json`, `eval/validated_cases.json`,
`eval/combined_eval_dataset.json`, and stratified samples like
`eval/stratified_80.json`. Without this, `ground_truth_rationale` would only
ever exist in `eval/gold_annotations.json` itself, and the reasoning-
alignment scorer (which reads `EvalCase.ground_truth_rationale`, populated
from the eval dataset file at load time) would have nothing to score
against no matter which of those files an eval run pointed at.
"""

from __future__ import annotations

import json
from pathlib import Path

from .. import storage


def sync_ground_truth_rationale(gold_annotations_path: Path, eval_dir: Path) -> list[Path]:
    """Update `ground_truth_rationale` for every case in every case-list
    *.json file directly under `eval_dir` that has a matching `case_id`
    entry in `gold_annotations_path`. Returns the paths actually rewritten
    (empty if nothing needed changing).

    Only annotations with a *resolved* `final_label` (present and not
    `"cannot_determine"`) and a non-blank `final_rationale` (whitespace-only
    text doesn't count) are propagated — an unresolved or `cannot_determine`
    case, or one with no real rationale text yet, has no ground-truth
    reasoning to score against, and must never have one fabricated. Files
    that aren't a JSON list of case dicts are skipped rather than raising, since
    `eval_dir` may hold other JSON (few-shot manifests, etc.), matching
    `ground_truth_sync.sync_ground_truth`'s behavior.

    Each file is updated via `storage.update_json`, so two reviewers
    syncing overlapping files concurrently retry rather than clobber one
    another (see `ppe_pipeline.storage`). A file that still conflicts after
    those retries are exhausted (`storage.ConcurrentModificationError`) is
    skipped like an unparseable file, rather than aborting the sync of the
    remaining files in `eval_dir`.
    """
    backend = storage.get_storage()
    annotations, _ = backend.read_json(gold_annotations_path.as_posix())

    rationales: dict[str, str] = {}
    for case_id, entry in annotations.items():
        final_label = entry.get("final_label")
        final_rationale = entry.get("final_rationale")
        if not final_label or final_label == "cannot_determine":
            continue
        if not final_rationale or not final_rationale.strip():
            continue
        rationales[case_id] = final_rationale

    updated = []
    for key in backend.list_keys(eval_dir.as_posix()):

        def mutate(cases):
            if not isinstance(cases, list):
                return cases
            for entry in cases:
                if not isinstance(entry, dict):
                    continue
                case_id = entry.get("case_id")
                if case_id in rationales and entry.get("ground_truth_rationale") != rationales[case_id]:
                    entry["ground_truth_rationale"] = rationales[case_id]
            return cases

        try:
            _, changed = backend.update_json(key, mutate, return_changed=True)
        except (json.JSONDecodeError, OSError, storage.ConcurrentModificationError):
            continue

        if changed:
            updated.append(Path(key))

    return updated
