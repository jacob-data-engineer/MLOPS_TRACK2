"""Patching a helper that lives on more than one side (2026-10-28).

Before the data/training split, `assemble_dataset`, `dataset_classes`,
`_list_keys`, the S3 client and friends all lived in `track2_shared`, so
`monkeypatch.setattr(t2, "dataset_classes", ...)` replaced the one copy
every caller used. They now live in `data/assemble.py` and `storage.py`,
and `track2_shared` keeps the same names as the data side's front door.
A replacement set on one module is invisible to code in the other.

`patch_everywhere` sets it on each module that has the name -- training,
storage and the data side -- so a test still means what it meant.
"""

from __future__ import annotations

import importlib

_HOMES = ("training.track2_shared", "storage", "data.assemble")


def patch_everywhere(monkeypatch, name: str, value) -> None:
    for prefix in ("ppe_pipeline_api", "src.ppe_pipeline_api"):
        for home in _HOMES:
            try:
                module = importlib.import_module(f"{prefix}.{home}")
            except ImportError:
                continue
            if hasattr(module, name):
                monkeypatch.setattr(module, name, value)
