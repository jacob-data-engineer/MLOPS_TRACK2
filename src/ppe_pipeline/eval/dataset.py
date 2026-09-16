"""Ground-truth eval dataset: labeled crops with a known real-violation /
false-alarm outcome, used to measure judge precision/recall against human
verification — see the architecture doc's "Evaluating it before we trust
it" section.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .. import storage
from ..models import BoundingBox, ViolationType


@dataclass
class EvalCase:
    case_id: str
    image_path: Path
    violation_type: ViolationType
    person_box: BoundingBox
    # True: this was a real violation (a "TP" in the doc's human-labeling
    # sense). False: the CV detector's original flag was a false alarm.
    # None: this case never had an upstream CV/detector flag to begin with
    # (e.g. a fresh manually bbox-annotated frame going straight into Gold
    # Set dual review) — there is no prior verdict to show a reviewer.
    ground_truth_is_violation: bool | None
    region_box: BoundingBox | None = None
    forklift_box: BoundingBox | None = None
    # The CV detector's own confidence on its original (pre-judge) flag —
    # optional, only used to report the raw-CV-precision baseline.
    cv_confidence: float | None = None
    # A resolved gold-set reviewer's rationale for this case's ground truth,
    # synced in by rationale_sync.sync_ground_truth_rationale. None means no
    # resolved gold annotation exists yet for this case — reasoning-alignment
    # scoring must skip it rather than fabricate a comparison.
    ground_truth_rationale: str | None = None


def _box(coords: list[float] | None) -> BoundingBox | None:
    return BoundingBox(*coords) if coords else None


def _resolve_image_key(base_dir: Path, image_ref: str) -> Path:
    """`image_ref` values shaped like `resources/...` are already a
    top-level storage key (S3 has no cwd and no `..` to walk against), so
    they're returned as-is rather than joined to `base_dir` — unlike
    few-shot crops, which are still genuinely relative to the manifest's
    own directory."""
    if image_ref.startswith("resources/"):
        return Path(image_ref)
    return base_dir / image_ref


def load_eval_dataset(manifest_path: Path) -> list[EvalCase]:
    """Load a manifest shaped like:

        [
          {
            "case_id": "glove_001",
            "image": "images/glove_001.jpg",
            "violation_type": "glove",
            "person_box": [x1, y1, x2, y2],
            "region_box": [x1, y1, x2, y2],
            "ground_truth": "TP",
            "cv_confidence": 0.87
          },
          ...
        ]

    Image paths resolve relative to the manifest's own directory, except
    for `"resources/..."` paths, which are already top-level storage keys
    (see `_resolve_image_key`) — the convention every shipped eval manifest
    actually uses today, since those images live under the repo's own
    `resources/` tree, not next to the manifest.
    `ground_truth` is "TP" (real violation, confirmed by a human) or "FP"
    (the CV detector's original flag was a false alarm), or may be absent
    entirely for a case with no prior CV/detector flag (a fresh manually
    bbox-annotated frame going straight into Gold Set dual review) — such
    a case's `ground_truth_is_violation` resolves to `None` rather than a
    fabricated verdict. `region_box` only
    applies to glove/helmet cases; `forklift_box` only to
    too_near_forklift cases. `ground_truth_rationale` (synced in by
    rationale_sync.sync_ground_truth_rationale) is read directly if present,
    falling back to `final_rationale` (gold_set_v1.json's own frozen field
    name) so a frozen gold set doesn't need this field added separately.
    """
    raw, _ = storage.get_storage().read_json(manifest_path.as_posix())
    base_dir = manifest_path.parent

    return [
        EvalCase(
            case_id=entry["case_id"],
            image_path=_resolve_image_key(base_dir, entry["image"]),
            violation_type=ViolationType(entry["violation_type"]),
            person_box=_box(entry["person_box"]),
            region_box=_box(entry.get("region_box")),
            forklift_box=_box(entry.get("forklift_box")),
            ground_truth_is_violation=(
                None if entry.get("ground_truth") is None else entry["ground_truth"] == "TP"
            ),
            cv_confidence=entry.get("cv_confidence"),
            ground_truth_rationale=entry.get("ground_truth_rationale") or entry.get("final_rationale"),
        )
        for entry in raw
    ]


def filter_by_violation_type(
    dataset: list[EvalCase], violation_types: list[ViolationType]
) -> list[EvalCase]:
    """Keep only cases whose `violation_type` is in `violation_types`.

    An empty or `None` list means "no filtering" — returns `dataset`
    unchanged, rather than an empty list, so this can always be called
    unconditionally by a caller that may or may not have a real filter
    selected.
    """
    if not violation_types:
        return dataset
    allowed = set(violation_types)
    return [case for case in dataset if case.violation_type in allowed]


_REQUIRED_CASE_KEYS = ("case_id", "image", "violation_type", "person_box")

# Every call below reads and fully parses every `.json` file under
# `directory` just to check its shape — on S3, that's one full GetObject
# per file, sequentially, on every single page load of every screen with
# a dataset picker (Run pipeline, Evaluate against ground truth, Gold
# Set). Fine as a one-off; noticeably slow (and, under several reviewers
# hitting it at once, contended enough to look like it never finishes)
# against real production data. A short TTL cache means only the first
# request per window actually touches S3 — a newly frozen gold set or
# added dataset file can take up to this long to show up in a picker,
# an acceptable tradeoff for a ~7-person internal tool.
_DATASET_LIST_CACHE_TTL_SECONDS = 30.0
_dataset_list_cache: dict[str, tuple[float, list[str]]] = {}
# Guards the check-then-refill below so concurrent requests that all miss
# the cache at once (e.g. several reviewers' page loads landing right as
# a 30s window expires) serialize on *one* of them actually doing the
# S3 scan, instead of each independently racing to do their own full
# sequential per-file scan at the same time and fighting over the same
# boto3 connection pool -- confirmed in production that this is what
# turns an already-slow cold scan into one that looks like it never
# finishes under real concurrent load.
_dataset_list_cache_lock = threading.Lock()


def list_valid_eval_dataset_paths(directory: str | Path) -> list[str]:
    """The `.json` storage keys directly under `directory` that are real,
    loadable eval-dataset case lists — i.e. `read_json` returns a JSON
    list whose first element has every key in `_REQUIRED_CASE_KEYS`.

    This excludes files that are shaped completely differently, like
    `eval/gold_annotations.json` (a dict keyed by case_id, not a list) and
    `eval/track1_audit_log.json` (a list, but of audit rows with none of
    the required keys) — both would otherwise show up as spurious entries
    in a dataset picker built from a raw directory listing.

    Any per-file exception (invalid JSON, wrong top-level type, an empty
    list, a first element missing a required key) excludes just that one
    file rather than raising and breaking the whole listing — a single
    malformed or unrelated file under the eval data directory shouldn't
    take down every dataset picker.

    Cached per `directory` for `_DATASET_LIST_CACHE_TTL_SECONDS` — see the
    module-level comment above for why. The whole check-then-refill is
    under `_dataset_list_cache_lock`, so only one caller per expired
    window actually pays for the scan; everyone else just waits for it
    and then reads the same freshly-cached result.
    """
    directory_key = Path(directory).as_posix()
    with _dataset_list_cache_lock:
        now = time.monotonic()
        cached = _dataset_list_cache.get(directory_key)
        if cached is not None and now - cached[0] < _DATASET_LIST_CACHE_TTL_SECONDS:
            return cached[1]
        return _rescan_eval_dataset_directory(directory_key, now)


def _is_valid_eval_dataset_file(key: str) -> str | None:
    """`key` if it's a real, loadable eval-dataset case list, else `None`
    — a per-file check run in its own thread by `_rescan_eval_dataset_directory`
    below, so the wall-clock cost of scanning N files is close to one
    file's S3 round trip instead of N of them end to end."""
    try:
        raw, _ = storage.get_storage().read_json(key)
        first = raw[0]
        if not all(required_key in first for required_key in _REQUIRED_CASE_KEYS):
            return None
    except Exception:
        return None
    return key


def _rescan_eval_dataset_directory(directory_key: str, now: float) -> list[str]:
    """The actual (slow, if run sequentially) scan — only ever called
    from inside `_dataset_list_cache_lock`, by whichever caller wins the
    race to refill an expired/missing cache entry. Each file's
    read+validate is its own blocking S3 round trip with nothing to
    compute in between, so running them concurrently (not sequentially)
    turns "N files times one round trip each" into roughly one round
    trip total — confirmed in production this is what actually mattered:
    the cache means this only runs once per TTL window regardless of
    concurrent requests, but that one run still has to pay the full
    per-file cost itself, and real S3 latency didn't drop enough on its
    own to make a sequential scan feel fast.
    """
    keys = storage.get_storage().list_keys(directory_key, suffixes=(".json",))
    with ThreadPoolExecutor(max_workers=min(len(keys), 16) or 1) as pool:
        results = list(pool.map(_is_valid_eval_dataset_file, keys))
    valid = [key for key in results if key is not None]
    _dataset_list_cache[directory_key] = (now, valid)
    return valid


def remove_cases_by_id(path: Path, case_ids: set[str]) -> bool:
    """Drop every case in the `*.json` case-list file at `path` whose
    `case_id` is in `case_ids` — concurrency-safe via
    `storage.get_storage().update_json`, same pattern as
    `rationale_sync.sync_ground_truth_rationale`. Returns whether the file
    actually changed (False if none of `case_ids` were present, or `path`
    doesn't exist).
    """
    if not case_ids:
        return False

    def mutate(current):
        if not isinstance(current, list):
            return current
        return [entry for entry in current if entry.get("case_id") not in case_ids]

    _, changed = storage.get_storage().update_json(
        path.as_posix(), mutate, default=[], return_changed=True
    )
    return changed
