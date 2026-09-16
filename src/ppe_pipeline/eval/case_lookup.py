"""Find an `EvalCase` by `case_id` across the eval dataset files.

Needed because `eval_runs/*.jsonl` records a case by id but not which
dataset the run was pointed at, so a row on its own cannot be turned back
into an image: the id is there, the image path and boxes are not. Rebuilding
a case for display therefore means looking it up across the dataset files
that exist now.

That is a lossy reconstruction and it can legitimately fail. A dataset the
run used may have been re-frozen, renamed or filtered since, in which case
the id is simply gone. Callers must handle `None` rather than treat a miss
as an error: an eval run from three weeks ago is still a valid measurement
even if one of its cases can no longer be rendered.

Cached with a short TTL, for the same reason `dataset.list_valid_eval_dataset_paths`
is: on S3 a full scan is one GetObject per dataset file, sequentially, and a
drill-down panel that renders twelve thumbnails must not pay for twelve
scans.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from .dataset import EvalCase, list_valid_eval_dataset_paths, load_eval_dataset

# Matches _DATASET_LIST_CACHE_TTL_SECONDS in dataset.py: long enough that one
# panel's worth of image requests share a single scan, short enough that a
# newly written dataset shows up without a restart.
_CACHE_TTL_SECONDS = 30.0

_index_cache: dict[str, tuple[float, dict[str, EvalCase]]] = {}
_index_cache_lock = threading.Lock()


def _build_index(eval_dir: str) -> dict[str, EvalCase]:
    """`{case_id: EvalCase}` across every loadable dataset under `eval_dir`.

    First file wins on a duplicate id. The same case is deliberately
    duplicated across known_cases / validated_cases / combined_eval_dataset
    and the stratified samples (see `ground_truth_sync`), and those copies
    describe the same image and boxes, so which one answers does not matter
    for rendering. A file that fails to load is skipped rather than raising,
    matching `list_valid_eval_dataset_paths`.
    """
    index: dict[str, EvalCase] = {}
    for key in list_valid_eval_dataset_paths(eval_dir):
        try:
            cases = load_eval_dataset(Path(key))
        except Exception:
            continue
        for case in cases:
            index.setdefault(case.case_id, case)
    return index


def find_case(case_id: str, eval_dir: str) -> EvalCase | None:
    """The case with this id, or None if no current dataset still holds it."""
    with _index_cache_lock:
        now = time.monotonic()
        cached = _index_cache.get(eval_dir)
        if cached is None or now - cached[0] >= _CACHE_TTL_SECONDS:
            cached = (now, _build_index(eval_dir))
            _index_cache[eval_dir] = cached
        return cached[1].get(case_id)
