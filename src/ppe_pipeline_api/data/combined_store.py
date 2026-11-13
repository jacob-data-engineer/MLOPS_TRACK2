"""Combine every data source into the central store, then decay a copy of
it for training (2026-11-13).

    <store>/                    the central data store -- ONLY EVER ADDED TO
      weeks/week_N/images/{train,val}/...   + labels/   (Pull Control's weeks)
      golden/images/{train,val}/...         + labels/   (the Golden Set card)
      folders/<slug>/images/{train,val}/... + labels/   (Assemble's folders)
      objects/...                                       (frozen versions; central_store.py)

    <store>_decay/              rebuilt from <store> before every run, deleted after it
      weeks/week_N/images/train/...   only what the retention policy keeps
      golden/...                      in full -- it has no age
      folders/<slug>/...              in full -- picked by hand, no age

The store is never pruned and never overwritten: a key that is already
there is skipped, so a file revised upstream keeps its first copy. That is
the operator's rule ("the central store must never be touched, it just
keeps growing"), and it is what lets the decayed copy be thrown away after
each run -- everything in it can be made again from the store.

Validation is never decayed. A week's val side is not copied into the
decayed folder because the training chain has never trained or scored on
it (the Assemble card's val folders and the golden set's val are the
evaluation set); copying it would change nothing but the bill.

Each source's own `data.yaml` is copied beside its files, once: it is the
only record of what that source's label numbers mean, and the yaml
builder reads it from there exactly as it would from the original.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from concurrent.futures import ThreadPoolExecutor

from .. import pull_control as pc
from .. import storage
from . import central_store
from . import retained_lists

#: Copies in flight at once. Server-side copies: nothing is downloaded.
WORKERS = 16
#: Appended to the store's folder for the decayed, per-run copy.
DECAY_SUFFIX = "_decay"
#: Written inside the decayed folder: what built it and when.
MARKER = "_decay.json"
LABEL_EXTENSIONS = (".txt",)
YAML_NAME = "data.yaml"


class Stopped(Exception):
    """The operator pressed Stop. Copies already made stay; none is half
    made -- each S3 copy is whole or absent."""


def _check(stop) -> None:
    if stop is not None and stop():
        raise Stopped("Stopped by hand.")


def location() -> tuple[str, str, str]:
    """`(bucket, store, decayed)` from the Central data store card.

    The decayed copy is what a run downloads as its one prefix, and a run
    reads its dataset from this app's bucket -- so the store must be in it.
    Said now, not discovered when a run cannot find its images.
    """
    bucket, store = central_store.location()
    if bucket != storage.BUCKET:
        raise ValueError(
            f"The central store is in {bucket}; combining for training needs it in "
            f"{storage.BUCKET}, the bucket runs read their dataset from.")
    return bucket, store, store + DECAY_SUFFIX


def _split_uri(prefix: str, default_bucket: str) -> tuple[str, str]:
    prefix = (prefix or "").strip()
    if prefix.startswith("s3://"):
        bucket, _, key = prefix[5:].partition("/")
        return bucket, key.strip("/")
    return default_bucket, prefix.strip("/")


def _keys(bucket: str, prefix: str) -> list[str]:
    """Every object under `prefix/`, folder markers left out."""
    out, token = [], None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix.rstrip("/") + "/"}
        if token:
            kwargs["ContinuationToken"] = token
        page = storage.s3.list_objects_v2(**kwargs)
        out += [o["Key"] for o in page.get("Contents", []) if not o["Key"].endswith("/")]
        token = page.get("NextContinuationToken")
        if not page.get("IsTruncated"):
            return out


def _is_image(key: str) -> bool:
    return key.lower().endswith(storage.IMAGE_EXTENSIONS)


def _is_label(key: str) -> bool:
    return key.lower().endswith(LABEL_EXTENSIONS) and "/labels/" in f"/{key}"


def _is_data(key: str) -> bool:
    return _is_image(key) or _is_label(key)


def slug_of(images_prefix: str) -> str:
    """A stable folder name for an Assemble source: its dataset root (the
    part before `/images`), bucket included when it is not ours. Stable,
    because the store is append-only: the same source must land in the
    same place every time, or each run would copy it again."""
    bucket, key = _split_uri(images_prefix, storage.BUCKET)
    root = re.split(r"/images(?:/|$)", key, maxsplit=1)[0] or key
    name = root if bucket == storage.BUCKET else f"{bucket}/{root}"
    return re.sub(r"[^A-Za-z0-9._-]+", "__", name).strip("_") or "source"


def _split_of(images_prefix: str, declared: str) -> str:
    """`train` or `val` for an Assemble entry: the list it was put in."""
    return "val" if declared == "val" else "train"


def _copy(pairs: list, dest_bucket: str, *, skip_existing: set, stop=None) -> dict:
    """Server-side copies of `(src_bucket, src_key, dest_key)`. A dest key
    already in `skip_existing` is left alone -- never overwritten.

    `stop`, when given, is asked before every copy; once it answers True
    the copies not yet started are skipped and `Stopped` is raised."""
    todo = [p for p in pairs if p[2] not in skip_existing]
    skipped = object()

    def one(pair):
        if stop is not None and stop():
            return skipped
        src_bucket, src_key, dest_key = pair
        try:
            storage.s3.copy_object(Bucket=dest_bucket, Key=dest_key,
                                   CopySource={"Bucket": src_bucket, "Key": src_key})
            return None
        except Exception as exc:                                    # noqa: BLE001
            return f"{src_key}: {exc}"

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        outcomes = list(pool.map(one, todo))
    if any(o is skipped for o in outcomes):
        done = sum(1 for o in outcomes if o is None)
        raise Stopped(f"Stopped by hand after {done:,} of {len(todo):,} copies.")
    errors = [e for e in outcomes if e]
    return {"copied": len(todo) - len(errors), "already": len(pairs) - len(todo),
            "failed": len(errors), "errors": errors[:10]}


# --------------------------------------------------------------------------
# 1. combine: sources -> the store (append only)
# --------------------------------------------------------------------------

def combine(*, weeks_root: str = "", weeks_bucket: str = "", week_prefix: str = "week_",
            golden_folder: str = "", golden_bucket: str = "",
            folders: list | None = None, stop=None) -> dict:
    """Copy every file of every ticked source into the store. Adds only.

    `folders` are the Assemble card's entries, each `{"images", "labels",
    "split"}`. Returns per-source counts and the slugs the folders went to.
    """
    bucket, store, _ = location()
    existing = set(_keys(bucket, store))
    pairs: list = []
    report: dict = {}

    def add(name: str, new: list):
        pairs.extend(new)
        report[name] = len(new)

    if weeks_root:
        wb, root = _split_uri(weeks_root, weeks_bucket or storage.BUCKET)
        new = []
        for key in _keys(wb, root):
            rel = key[len(root) + 1:]
            first = rel.split("/", 1)[0]
            if (first.startswith(week_prefix) and _is_data(key)) or rel == YAML_NAME:
                new.append((wb, key, f"{store}/weeks/{rel}"))
        add("weeks", new)

    if golden_folder:
        gb, gf = _split_uri(golden_folder, golden_bucket or storage.BUCKET)
        new = [(gb, key, f"{store}/golden/{key[len(gf) + 1:]}")
               for key in _keys(gb, gf)
               if _is_data(key) or key == f"{gf}/{YAML_NAME}"]
        add("golden", new)

    slugs: dict = {}
    for entry in folders or []:
        images = str(entry.get("images") or "").strip()
        if not images:
            continue
        split = _split_of(images, str(entry.get("split") or "train"))
        ib, ik = _split_uri(images, storage.BUCKET)
        lb, lk = _split_uri(str(entry.get("labels") or "") or
                            re.sub(r"(^|/)images(/|$)", r"\1labels\2", ik, count=1), ib)
        slug = slug_of(images)
        slugs.setdefault(slug, set()).add(split)
        base = f"{store}/folders/{slug}"
        new = [(ib, k, f"{base}/images/{split}/{k[len(ik) + 1:]}")
               for k in _keys(ib, ik) if _is_image(k)]
        new += [(lb, k, f"{base}/labels/{split}/{k[len(lk) + 1:]}")
                for k in _keys(lb, lk) if k.lower().endswith(LABEL_EXTENSIONS)]
        # The source's own numbering, when it has one.
        root = re.split(r"/images(?:/|$)", ik, maxsplit=1)[0]
        if root and root != ik:
            try:
                storage.s3.head_object(Bucket=ib, Key=f"{root}/{YAML_NAME}")
                new.append((ib, f"{root}/{YAML_NAME}", f"{base}/{YAML_NAME}"))
            except Exception:                                       # noqa: BLE001
                pass
        add(f"folders/{slug}/{split}", new)

    _check(stop)
    result = _copy(pairs, bucket, skip_existing=existing, stop=stop)
    return {"bucket": bucket, "store": store, "sources": report,
            "slugs": {k: sorted(v) for k, v in slugs.items()}, **result}


# --------------------------------------------------------------------------
# 2. decay: the store -> <store>_decay (rebuilt every time)
# --------------------------------------------------------------------------

def read_marker() -> dict | None:
    bucket, _, decayed = location()
    try:
        body = storage.s3.get_object(Bucket=bucket, Key=f"{decayed}/{MARKER}")["Body"].read()
        return json.loads(body)
    except Exception:                                               # noqa: BLE001
        return None


def _write_marker(bucket: str, decayed: str, **fields) -> None:
    storage.s3.put_object(Bucket=bucket, Key=f"{decayed}/{MARKER}",
                          Body=json.dumps(fields, indent=1).encode(),
                          ContentType="application/json")


def delete_decayed() -> int:
    """Remove the decayed copy, every key of it. The store is not touched."""
    bucket, store, decayed = location()
    assert decayed != store and decayed.endswith(DECAY_SUFFIX)
    keys = _keys(bucket, decayed)
    for i in range(0, len(keys), 1000):
        storage.s3.delete_objects(Bucket=bucket, Delete={
            "Objects": [{"Key": k} for k in keys[i:i + 1000]], "Quiet": True})
    return len(keys)


def rebuild_decayed(*, retention: dict, use_weeks: bool, use_golden: bool,
                    golden_split: bool = True, slugs: dict | None = None,
                    for_run: bool = False, stop=None) -> dict:
    """Empty `<store>_decay`, then copy into it what a run trains on.

    Weeks through the retention policy (`retained_lists.select`, the same
    selection the card's bars and the old path-list used); the golden set
    and the Assemble folders whole. Returns the yaml entries that name it.
    """
    bucket, store, decayed = location()
    now = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    delete_decayed()
    _write_marker(bucket, decayed, status="building", started_at=now, for_run=for_run)

    pairs: list = []
    train: list = []
    val: list = []
    weeks_report: list = []

    if use_weeks:
        chosen = retained_lists.select(root=f"{store}/weeks", config=retention, bucket=bucket)
        kept_weeks: set = set()
        for rel in chosen["train"]:
            rel = rel[2:] if rel.startswith("./") else rel
            src = f"{store}/weeks/{rel}"
            pairs.append((bucket, src, f"{decayed}/weeks/{rel}"))
            label = storage.label_path_for(src)
            if label:
                pairs.append((bucket, label, f"{decayed}/weeks/{label[len(store) + 7:]}"))
            kept_weeks.add(rel.split("/", 1)[0])
        if f"{store}/weeks/{YAML_NAME}" in set(_keys(bucket, f"{store}/weeks")):
            pairs.append((bucket, f"{store}/weeks/{YAML_NAME}", f"{decayed}/weeks/{YAML_NAME}"))
        train += [{"images": f"{decayed}/weeks/{w}/images/train", "labels": ""}
                  for w in sorted(kept_weeks, key=lambda n: int(re.sub(r"\D", "", n) or 0))]
        weeks_report = chosen.get("weeks") or []

    def whole(sub: str):
        for key in _keys(bucket, f"{store}/{sub}"):
            pairs.append((bucket, key, f"{decayed}/{key[len(store) + 1:]}"))

    if use_golden:
        whole("golden")
        if golden_split:
            train.append({"images": f"{decayed}/golden/images/train", "labels": ""})
            val.append({"images": f"{decayed}/golden/images/val", "labels": ""})
        else:
            train.append({"images": f"{decayed}/golden/images", "labels": ""})

    for slug, splits in (slugs or {}).items():
        for split in splits:
            whole(f"folders/{slug}/images/{split}")
            whole(f"folders/{slug}/labels/{split}")
            (val if split == "val" else train).append(
                {"images": f"{decayed}/folders/{slug}/images/{split}", "labels": ""})
        try:
            storage.s3.head_object(Bucket=bucket, Key=f"{store}/folders/{slug}/{YAML_NAME}")
            pairs.append((bucket, f"{store}/folders/{slug}/{YAML_NAME}",
                          f"{decayed}/folders/{slug}/{YAML_NAME}"))
        except Exception:                                           # noqa: BLE001
            pass

    try:
        _check(stop)
        result = _copy(pairs, bucket, skip_existing=set(), stop=stop)
    except Stopped:
        # A half-built decayed copy is a dataset nobody chose. Remove it.
        delete_decayed()
        raise
    _write_marker(bucket, decayed, status="built", started_at=now,
                  built_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                  for_run=for_run, files=result["copied"], failed=result["failed"])
    if result["failed"]:
        raise ValueError(f"{result['failed']} file(s) could not be copied into {decayed}: "
                         + "; ".join(result["errors"][:3]))
    return {"bucket": bucket, "folder": decayed, "train": train, "val": val,
            "weeks": weeks_report, **result}
