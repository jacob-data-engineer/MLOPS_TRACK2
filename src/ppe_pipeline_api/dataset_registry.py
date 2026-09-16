"""Named-label decoration for the eval-dataset picker dropdowns shared by
Track1's "Run pipeline" screen, "Evaluate against ground truth", and
`GoldsetPage`'s dataset-shaped fields — promotes the two canonical,
human-named gold sets to the top of the list labeled by name, with every
other real eval-dataset file listed underneath by its own filename,
alphabetically.
"""

from __future__ import annotations

from pathlib import Path

# Ordered so a plain alphabetical sort of the label text ("Gold Set v1" <
# "Gold Set v2") already puts v1 before v2 — no separate ordering rule needed.
NAMED_DATASET_LABELS: dict[str, str] = {
    "eval/gold_set_v1.json": "Gold Set v1",
    "eval/gold_set_v2.json": "Gold Set v2",
}


def decorate_dataset_keys(keys: list[str]) -> list[dict]:
    """`keys` (real, valid eval-dataset storage keys — see
    `ppe_pipeline.eval.dataset.list_valid_eval_dataset_paths`) ->
    `[{"key": ..., "label": ..., "is_named": ...}, ...]`, with the named
    gold sets first (v1 then v2) and everything else after, sorted
    alphabetically by its own filename."""
    named_keys = sorted((k for k in keys if k in NAMED_DATASET_LABELS), key=lambda k: NAMED_DATASET_LABELS[k])
    other_keys = sorted((k for k in keys if k not in NAMED_DATASET_LABELS), key=lambda k: Path(k).name)

    return [
        {"key": key, "label": NAMED_DATASET_LABELS[key], "is_named": True} for key in named_keys
    ] + [
        {"key": key, "label": Path(key).name, "is_named": False} for key in other_keys
    ]
