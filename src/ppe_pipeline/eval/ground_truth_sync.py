"""Propagate a human-confirmed ground-truth correction to every eval
dataset file that already has an entry for that case_id — not just the
one file a review session happened to load.

The same labeled case (by `case_id`) is deliberately duplicated across
`eval/known_cases.json`, `eval/validated_cases.json`,
`eval/combined_eval_dataset.json`, and stratified samples like
`eval/stratified_80.json`. Without this, correcting a wrong label while
reviewing one of them silently leaves the others stale — the next eval run
against a different dataset file scores against the old, wrong label again.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from .. import storage

GroundTruth = Literal["TP", "FP"]


def sync_ground_truth(case_id: str, ground_truth: GroundTruth, eval_dir: Path) -> list[Path]:
    """Update `ground_truth` for `case_id` in every case-list *.json file
    directly under `eval_dir` that already has an entry for it. Returns the
    paths actually rewritten (empty if nothing needed changing).

    Only entries that already carry a `ground_truth` key are touched — a
    case still sitting in `pending_validation.json` (not yet labeled) is
    left alone; this syncs corrections, it doesn't invent labels. Files
    that aren't a JSON list of case dicts are skipped rather than raising,
    since `eval_dir` may hold other JSON (few-shot manifests, etc.).

    Each file is updated via `storage.update_json`, so two reviewers
    syncing overlapping files concurrently retry rather than clobber one
    another (see `ppe_pipeline.storage`). A file that still conflicts after
    those retries are exhausted (`storage.ConcurrentModificationError`) is
    skipped like an unparseable file, rather than aborting the sync of the
    remaining files in `eval_dir`.
    """
    backend = storage.get_storage()
    updated = []
    for key in backend.list_keys(eval_dir.as_posix()):

        def mutate(cases):
            if not isinstance(cases, list):
                return cases
            for entry in cases:
                if not isinstance(entry, dict):
                    continue
                if (
                    entry.get("case_id") == case_id
                    and "ground_truth" in entry
                    and entry["ground_truth"] != ground_truth
                ):
                    entry["ground_truth"] = ground_truth
            return cases

        try:
            _, changed = backend.update_json(key, mutate, return_changed=True)
        except (json.JSONDecodeError, OSError, storage.ConcurrentModificationError):
            continue

        if changed:
            updated.append(Path(key))

    return updated
