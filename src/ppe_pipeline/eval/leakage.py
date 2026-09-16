"""Keep few-shot example images out of the ground-truth eval set, and keep
duplicate images out of the few-shot library itself.

A few-shot method hands the judge N labeled crops as prior turns, complete
with the verdict, the reason code and the explanation. If one of those exact
images is ALSO a case in the eval dataset, the judge is being graded on an
answer it was just given — that case scores as a success no matter how good
the model is, and every precision/recall number it feeds is inflated. The
same image can reach both sides innocently: `scripts/promote_*_to_fewshot.py`
promotes gold cases into the few-shot library by copying the crop, so a
promoted case keeps existing in whichever eval dataset it came from.

Matching uses SHA-256 of the image BYTES as its primary signal: it cannot be
fooled by a rename in either direction, and it catches the same picture
reaching both sides under different names. It is backed by a second signal —
a few-shot crop named `{case_id}.jpg` for a case_id in the dataset — because
the promote scripts CROP the frame on their way into the library, so a
promoted case's crop is a different file from the gold set's full frame and
will never hash equal to it. Hashing alone would be blind to exactly the leak
that is easiest to create.

Two entry points. The eval-run API that used to call them
(`ppe_pipeline_api.routers.eval`) has since been removed along with the
rest of the PPE judge surface; these remain part of the library:

  `exclude_few_shot_leakage`  drops the leaked cases from ONE RUN's dataset.
      Non-destructive by design: it never touches `gold_annotations.json` or
      the eval case files. Leakage is a property of a case *relative to a
      particular few-shot manifest*, not a defect in the case, so the same
      case stays perfectly good ground truth for a zero-shot run or against
      a different manifest. Re-checked from scratch every run.

  `deduplicate_few_shot_manifest`  removes duplicate images from the
      few-shot library ON DISK, keeping the first occurrence. Unlike the
      above this does persist, because a duplicated example is simply waste
      with no upside: it costs its image's tokens in every single judge call
      that loads that violation type, and it skews the library toward
      whatever it happens to duplicate.

Hashes are memoised per storage key for the life of the process. Crop images
are written once under a uuid- or case_id-derived name and never edited in
place, so a key is treated as immutable; a genuinely rewritten image would
need a new process (or a new key) to be seen. That assumption is what makes
this affordable to run unconditionally on every eval — without it, each run
would re-read every case image from S3 purely to hash it.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .. import storage
from .dataset import EvalCase

logger = logging.getLogger(__name__)

# Mirrors `_rescan_eval_dataset_directory`'s reasoning in dataset.py: the
# wall-clock cost of hashing N images should be close to one image's S3
# round trip rather than N of them end to end.
_HASH_WORKERS = 16

_hash_cache: dict[str, str | None] = {}
_hash_cache_lock = threading.Lock()


def image_hash(key: str) -> str | None:
    """SHA-256 of the image at storage `key`, or None if it cannot be read.

    An unreadable image is not an error here. A dataset that references a
    missing crop is already broken in a way the eval run itself will report
    far more clearly, and a leakage check is the wrong place to be the first
    thing that raises — a case whose bytes we cannot see simply cannot be
    proven to collide, so it is kept.
    """
    with _hash_cache_lock:
        if key in _hash_cache:
            return _hash_cache[key]

    try:
        digest = hashlib.sha256(storage.get_storage().get_bytes(key)).hexdigest()
    except Exception:
        logger.warning("could not hash %s for the few-shot leakage check; keeping the case", key)
        digest = None

    with _hash_cache_lock:
        _hash_cache[key] = digest
    return digest


def hash_keys(keys: list[str]) -> dict[str, str | None]:
    """`{key: sha256}` for every key, hashed in parallel and memoised."""
    uncached = []
    with _hash_cache_lock:
        uncached = [k for k in dict.fromkeys(keys) if k not in _hash_cache]
    if uncached:
        with ThreadPoolExecutor(max_workers=_HASH_WORKERS) as pool:
            list(pool.map(image_hash, uncached))
    with _hash_cache_lock:
        return {key: _hash_cache.get(key) for key in keys}


def few_shot_image_keys(manifest_path: Path) -> list[str]:
    """Every image storage key the few-shot manifest references.

    Paths resolve against the manifest's own directory, exactly as
    `few_shot.load_few_shot_library` resolves them — the two must agree or
    this would hash a different set of files than the judge is actually shown.
    """
    raw, _ = storage.get_storage().read_json(manifest_path.as_posix(), default={})
    base_dir = manifest_path.parent
    return [
        (base_dir / entry["image"]).as_posix()
        for entries in raw.values()
        for entry in entries
        if entry.get("image")
    ]


def exclude_few_shot_leakage(
    dataset: list[EvalCase], manifest_path: Path
) -> tuple[list[EvalCase], list[str], list[dict]]:
    """Drop cases the judge has already been shown as a few-shot example.

    Returns `(kept_cases, warnings, collisions)`. `collisions` holds one
    `{"case_id", "case_image", "few_shot_image", "match", "sha256"}` dict per
    dropped case, so a caller can report exactly what was removed and why
    rather than just a count.

    Applied to the WHOLE run, not only its few-shot methods. Scoring a
    few-shot method on 78 cases and a zero-shot method on 80 would make the
    two columns of the comparison table quietly incomparable, which defeats
    the point of running them together; every method sees the same cases.

    TWO signals, because neither alone is sufficient:

      `sha256`   the case image and a few-shot image are byte-identical.
                 Catches an image that reached both sides as the same file
                 under any name — a copy, a re-export, a rename.

      `case_id`  a few-shot image is named `{case_id}.jpg` for a case_id in
                 this dataset. `scripts/promote_*_to_fewshot.py` CROPS the
                 frame on its way into the library (see its docstring), so a
                 promoted case's stored crop is a different, smaller file
                 than the gold set's full frame and can never hash equal to
                 it. Content hashing structurally cannot see the most likely
                 leak; the naming convention the promote scripts guarantee
                 can. Promotion already removes the ids it promotes, so this
                 fires only when a case came back — restored from an older
                 freeze, re-added by hand, or added through the UI's "Add
                 example" button, which does not do the removal.

    Each collision records which signal fired in its `match` field.
    """
    if not dataset:
        return dataset, [], []

    few_shot_keys = few_shot_image_keys(manifest_path)
    if not few_shot_keys:
        return dataset, [], []

    few_shot_hashes = hash_keys(few_shot_keys)
    # First key wins, so the reported collision is stable across runs.
    by_hash: dict[str, str] = {}
    for key, digest in few_shot_hashes.items():
        if digest and digest not in by_hash:
            by_hash[digest] = key
    by_stem: dict[str, str] = {}
    for key in few_shot_keys:
        by_stem.setdefault(Path(key).stem, key)

    case_hashes = hash_keys([case.image_path.as_posix() for case in dataset])

    kept: list[EvalCase] = []
    collisions: list[dict] = []
    for case in dataset:
        key = case.image_path.as_posix()
        digest = case_hashes.get(key)

        few_shot_key = by_hash.get(digest) if digest else None
        match = "sha256"
        if few_shot_key is None:
            few_shot_key = by_stem.get(case.case_id)
            match = "case_id"

        if few_shot_key is None:
            kept.append(case)
            continue
        collisions.append(
            {
                "case_id": case.case_id,
                "case_image": key,
                "few_shot_image": few_shot_key,
                "match": match,
                "sha256": digest,
            }
        )

    warnings: list[str] = []
    if collisions:
        by_signal = ", ".join(
            f"{sum(1 for c in collisions if c['match'] == signal)} by {signal}"
            for signal in ("sha256", "case_id")
            if any(c["match"] == signal for c in collisions)
        )
        shown = ", ".join(c["case_id"] for c in collisions[:5])
        more = f" (+{len(collisions) - 5} more)" if len(collisions) > 5 else ""
        warnings.append(
            f"excluded {len(collisions)} case(s) already shown to the judge as a "
            f"few-shot example ({by_signal}), to keep them out of the score: "
            f"{shown}{more}. The gold set on disk is unchanged."
        )
    return kept, warnings, collisions


def deduplicate_few_shot_manifest(
    manifest_path: Path,
    annotations_path: Path | None = None,
    *,
    write: bool = True,
    backup_path: Path | None = None,
) -> tuple[list[dict], list[str]]:
    """Remove duplicate images from the few-shot library, keeping the first.

    Duplicates are found by image bytes, per violation type AND across types
    — the same crop filed under two violation types is still the same picture
    shown twice, and the judge pays for it in both prompts.

    Returns `(removed, warnings)`, where `removed` holds one
    `{"violation_type", "image", "kept_image", "sha256"}` dict per dropped
    entry. With `write=False` nothing is persisted and the return value is
    the dry-run answer.

    Before the first write the current manifest is snapshotted to
    `backup_path` (default: `<manifest dir>/backups/manifest.pre_dedupe.json`)
    unless that key already exists — the same
    write-the-original-once contract `scripts/backfill_rationale_rewrites.py`
    uses, so the very first copy is the one preserved rather than the last.

    The manifest is the source of truth; `annotations.json` is a sidecar
    keyed by image FILENAME. An entry is pruned from the sidecar only when no
    surviving manifest entry still uses that filename — two different
    directories can hold the same basename, and dropping the sidecar row for
    a filename another entry still needs would strip that example's reason
    and explanation.
    """
    backend = storage.get_storage()
    raw, _ = backend.read_json(manifest_path.as_posix(), default={})
    base_dir = manifest_path.parent

    keys = [
        (base_dir / entry["image"]).as_posix()
        for entries in raw.values()
        for entry in entries
        if entry.get("image")
    ]
    hashes = hash_keys(keys)

    seen: dict[str, str] = {}
    removed: list[dict] = []
    survivors: dict[str, list[dict]] = {}

    for violation_type, entries in raw.items():
        kept_entries = []
        for entry in entries:
            image = entry.get("image")
            digest = hashes.get((base_dir / image).as_posix()) if image else None
            # An unhashable image is kept: we cannot prove it is a duplicate,
            # and silently dropping an example we merely failed to read would
            # shrink the library for the wrong reason.
            if digest is None or digest not in seen:
                if digest is not None:
                    seen[digest] = image
                kept_entries.append(entry)
                continue
            removed.append(
                {
                    "violation_type": violation_type,
                    "image": image,
                    "kept_image": seen[digest],
                    "sha256": digest,
                }
            )
        survivors[violation_type] = kept_entries

    warnings: list[str] = []
    if not removed:
        return removed, warnings

    surviving_filenames = {
        Path(entry["image"]).name for entries in survivors.values() for entry in entries
    }
    dropped_filenames = {
        Path(item["image"]).name for item in removed
    } - surviving_filenames

    shown = ", ".join(item["image"] for item in removed[:5])
    more = f" (+{len(removed) - 5} more)" if len(removed) > 5 else ""
    warnings.append(
        f"{'removed' if write else 'would remove'} {len(removed)} duplicate few-shot "
        f"example(s), keeping one copy of each image: {shown}{more}"
    )
    if not write:
        return removed, warnings

    backup_key = (backup_path or base_dir / "backups" / "manifest.pre_dedupe.json").as_posix()
    existing, _ = backend.read_json(backup_key, default=None)
    if existing is None:
        backend.update_json(backup_key, lambda _current: raw, default={})
        warnings.append(f"backed up the pre-dedupe manifest to {backup_key}")

    def mutate_manifest(fresh: dict) -> dict:
        for violation_type, entries in survivors.items():
            # Only prune what we actually saw; an entry added since the read
            # above is left alone rather than clobbered by a stale snapshot.
            keep = {entry["image"] for entry in entries}
            dropped = {item["image"] for item in removed if item["violation_type"] == violation_type}
            fresh[violation_type] = [
                entry
                for entry in fresh.get(violation_type, [])
                if entry.get("image") in keep or entry.get("image") not in dropped
            ]
        return fresh

    backend.update_json(manifest_path.as_posix(), mutate_manifest, default={})

    if annotations_path is not None and dropped_filenames:
        def mutate_annotations(fresh: dict) -> dict:
            for filename in dropped_filenames:
                fresh.pop(filename, None)
            return fresh

        backend.update_json(annotations_path.as_posix(), mutate_annotations, default={})

    return removed, warnings
