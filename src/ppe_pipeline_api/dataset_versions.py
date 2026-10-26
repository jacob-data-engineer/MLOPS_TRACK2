"""Dataset versions: every queued run trains on a dataset nobody can change.

THE PROBLEM THIS SOLVES (2026-10-26). A queued run used to name a folder
and a yaml that live in SHARED, overwritable places -- `<folder>/data.yaml`,
`_generated/<hash>/data.yaml`, and the evaluation yaml written into a val
folder. The run read them when it STARTED, so anything built in between --
another Add to Queue, a schedule, the Generate button -- changed what it
trained on. A golden-set-only run failed that way: its training yaml had
been replaced by a composed run's evaluation yaml.

WHAT A VERSION IS. At the moment a run is queued, the yaml it would train
on is resolved to the exact image keys, and everything the run needs is
written ONCE under `_datasets/<version_id>/` in the app's bucket:

    data.yaml          the training yaml; its train/val are the two lists
    train/train.txt    the training images, one `./<path>` per line
    train/val.txt      the validation images
    eval/data.yaml     the evaluation yaml (evaluator layout)
    eval/val.txt       the evaluation images
    manifest.json      every image as s3 key, the sources, the classes

and one row is put in the snapshot table (`record_type = dataset_version`).
Nothing under `_datasets/` is ever written twice.

HOW THE RUN READS IT -- no trainer, evaluator or pipeline change:

  * training: `TrainingDatasetUri` = the data prefixes PLUS
    `_datasets/<id>/train/`, and `DataYamlUri` = `_datasets/<id>/data.yaml`.
    The trainer lays several prefixes out by full key (`local_layout`), so
    the list lines are written relative to where the list itself lands.
    Ultralytics reads a `.txt` split as a list of image paths, replacing a
    leading `./` with the list's own folder; the trainer's label filter and
    preflight read lists the same way.
  * evaluation: `EvaluationDatasetUri` = `_datasets/<id>/eval/` (one
    prefix, as the evaluator requires) and `EvalExtraUris` = the folders
    holding the evaluation images and their labels, which the evaluator
    fetches to `<data>/<key>` -- or `<data>/<bucket>/<key>` from another
    bucket.

The folders are still DOWNLOADED whole; the lists decide what is READ. So
images that land in a week folder after the run was queued are on the
disk and are not trained on.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import posixpath

import yaml

from . import dataset_yaml as dy
from . import pull_control as pc

PREFIX = "_datasets"
RECORD_TYPE = "dataset_version"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _uri(u: str) -> tuple:
    u = str(u).strip()
    b, _, k = u[5:].partition("/")
    return b, k.strip("/")


def _doc(text: str) -> dict:
    return yaml.safe_load(text or "") or {}


def _entries(doc: dict, split: str) -> list:
    v = doc.get(split)
    return [str(x) for x in (v if isinstance(v, list) else [v] if v else [])]


def _read(bucket: str, key: str) -> str:
    return pc.s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8", "replace")


def _put(bucket: str, key: str, body: str, content_type: str = "text/plain") -> None:
    pc.s3.put_object(Bucket=bucket, Key=key, Body=body.encode("utf-8"),
                     ContentType=content_type)


class _Layout:
    """`local_layout` and its inverse, for one list of download prefixes."""

    def __init__(self, prefixes: list):
        self.parsed = [_uri(u) for u in prefixes]
        self.many = len(self.parsed) > 1
        self.multi = len({b for b, _ in self.parsed}) > 1
        self.local = dy.local_layout(list(prefixes))

    def key(self, local: str) -> tuple:
        """Where a path on the trainer's disk came from: (bucket, key)."""
        local = local.strip("/")
        if self.multi:
            b, _, k = local.partition("/")
            return b, k
        b, base = self.parsed[0]
        if self.many:
            return b, local
        if local in ("", "."):
            return b, base
        return b, f"{base}/{local}" if base else local


def _images_under(bucket: str, key: str) -> list:
    return sorted(o["Key"] for o in pc._list_objects(key.rstrip("/") + "/", bucket)
                  if o["Key"].lower().endswith(IMAGE_EXTS))


def resolve(yaml_text: str, prefixes: list, split: str) -> list:
    """Every image a split names, as (bucket, key), sorted and unique.

    A folder entry is listed now; a `.txt` entry is read and each
    `./<path>` line resolved against the list's own folder -- the rule
    Ultralytics uses.
    """
    layout = _Layout(prefixes)
    found: set = set()
    for entry in _entries(_doc(yaml_text), split):
        b, k = layout.key(entry)
        if k.lower().endswith(".txt"):
            folder = posixpath.dirname(entry.strip("/"))
            for line in _read(b, k).splitlines():
                line = line.strip()
                if not line:
                    continue
                rel = line[2:] if line.startswith("./") else line
                lb, lk = layout.key(posixpath.normpath(posixpath.join(folder, rel)))
                if lk.lower().endswith(IMAGE_EXTS):
                    found.add((lb, lk))
            continue
        found.update((b, img) for img in _images_under(b, k))
    return sorted(found)


def _labels_dir(images_dir: str) -> str:
    parts = images_dir.split("/")
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            return "/".join(parts[:i] + ["labels"] + parts[i + 1:])
    return ""


def source_folders(images: list, home: str) -> list:
    """The distinct folders a list of (bucket, key) images sits in -- a key
    in the app's bucket, `s3://bucket/key` in any other."""
    out = set()
    for b, k in images:
        d = posixpath.dirname(k)
        out.add(d if b == home else f"s3://{b}/{d}")
    return sorted(out)


def _from_manifest(version: dict) -> dict:
    """Folders and counts for a version made before they were recorded."""
    m = json.loads(_read(version["bucket"], f"{version['folder']}/manifest.json"))
    home = version["bucket"]

    def pairs(names):
        return [tuple(n.split("/", 1)) for n in names if "/" in n]
    train, val = pairs(m.get("train") or []), pairs(m.get("val") or [])
    return {"train_folders": source_folders(train, home), "val_folders": source_folders(val, home),
            "train_images": len(train), "val_images": len(val)}


def _version_id(train: list, val: list, names: dict) -> str:
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%S")
    digest = hashlib.sha1(json.dumps([train, val, sorted(names.items())],
                                     default=str).encode("utf-8")).hexdigest()[:8]
    return f"dv-{stamp}-{digest}"


def _rel_line(from_dir: str, to_path: str) -> str:
    """`./<path>` from a list in `from_dir` to `to_path`, both trainer-local."""
    return "./" + posixpath.relpath(to_path, from_dir or ".")


def create(*, home: str, yaml_text: str, prefixes: list,
           eval_yaml_text: str = "", eval_prefixes: list | None = None,
           trigger: str = "", sources: list | None = None,
           drop_undeclared: bool = False, note: str = "") -> dict:
    """Freeze the dataset a run is about to be queued with. Returns the
    version, including the five values the run is started with."""
    home = home or pc.destination_bucket()
    doc = _doc(yaml_text)
    names = {int(k): str(v) for k, v in (doc.get("names") or {}).items()}
    if not names:
        raise ValueError("The yaml declares no classes, so there is nothing to version.")

    train = resolve(yaml_text, prefixes, "train")
    val = resolve(yaml_text, prefixes, "val")
    # An evaluation yaml of its own only matters when the training yaml has
    # no held-out side of its own (older single-folder builds).
    if eval_yaml_text and (not val or set(val) == set(train)):
        val = resolve(eval_yaml_text, eval_prefixes or prefixes, "val")
    if not train:
        raise ValueError("The yaml names no training images -- nothing would be trained.")
    if not val:
        raise ValueError("The yaml names no validation images -- the run would have "
                         "nothing to validate or be scored on.")
    both = set(train) & set(val)
    if both:
        b, k = sorted(both)[0]
        raise ValueError(f"{len(both)} image(s) are on both the training and the validation "
                         f"side, e.g. s3://{b}/{k}. A validation score on training images "
                         "is not a score.")

    vid = _version_id(train, val, names)
    folder = f"{PREFIX}/{vid}"
    train_folders = source_folders(train, home)
    val_folders = source_folders(val, home)

    # ---- training: the data prefixes plus this version's list folder ----
    run_prefixes = [p if p.endswith("/") else p + "/" for p in prefixes]
    run_prefixes.append(f"s3://{home}/{folder}/train/")
    layout = dy.local_layout(run_prefixes)
    list_dir = posixpath.dirname(layout(home, f"{folder}/train/train.txt"))
    train_lines = [_rel_line(list_dir, layout(b, k)) for b, k in train]
    val_lines = [_rel_line(list_dir, layout(b, k)) for b, k in val]
    old = _Layout(prefixes)
    remap = {}
    for local_dir, mapping in (doc.get("x_label_remap") or {}).items():
        b, k = old.key(str(local_dir))
        remap[layout(b, k)] = {int(s): int(t) for s, t in (mapping or {}).items()}
    train_doc = {
        "path": f"s3://{home}/{folder}",
        "train": [layout(home, f"{folder}/train/train.txt")],
        "val": [layout(home, f"{folder}/train/val.txt")],
        "nc": len(names),
        "names": {i: names[i] for i in sorted(names)},
    }
    if remap:
        train_doc["x_label_remap"] = remap

    # ---- evaluation: one prefix (this folder's eval/) plus the folders ----
    def elocal(b: str, k: str) -> str:
        return k if b == home else f"{b}/{k}"
    eval_lines = ["./" + elocal(b, k) for b, k in val]
    extra: list = []
    for b, k in val:
        d = posixpath.dirname(k)
        for f in (d, _labels_dir(d)):
            if f and f"s3://{b}/{f}/" not in extra:
                extra.append(f"s3://{b}/{f}/")
    eval_remap = {}
    for local_dir, mapping in (doc.get("x_label_remap") or {}).items():
        b, k = old.key(str(local_dir))
        eval_remap[elocal(b, k)] = {int(s): int(t) for s, t in (mapping or {}).items()}
    eval_doc = {
        "path": f"s3://{home}/{folder}/eval",
        "train": ["val.txt"],
        "val": ["val.txt"],
        "nc": len(names),
        "names": {i: names[i] for i in sorted(names)},
    }
    if eval_remap:
        eval_doc["x_label_remap"] = eval_remap

    header = [dy.IN_PLACE_MARKER,
              f"# Dataset version {vid} -- written once, never changed.",
              f"# {len(train)} training and {len(val)} validation images, listed by name."]
    _put(home, f"{folder}/train/train.txt", "\n".join(train_lines) + "\n")
    _put(home, f"{folder}/train/val.txt", "\n".join(val_lines) + "\n")
    _put(home, f"{folder}/eval/val.txt", "\n".join(eval_lines) + "\n")
    _put(home, f"{folder}/eval/data.yaml",
         "\n".join(header + ["# EVALUATION ONLY -- `train:` repeats `val:`.", "",
                             dy._dump(eval_doc)]), "application/x-yaml")
    _put(home, f"{folder}/data.yaml",
         "\n".join(header + ["", dy._dump(train_doc)]), "application/x-yaml")
    manifest = {
        "version_id": vid, "created_at": _now(), "trigger": trigger,
        "sources": list(sources or []), "names": {str(i): n for i, n in train_doc["names"].items()},
        "drop_undeclared": bool(drop_undeclared), "prefixes": list(prefixes),
        "train": [f"{b}/{k}" for b, k in train], "val": [f"{b}/{k}" for b, k in val],
    }
    _put(home, f"{folder}/manifest.json", json.dumps(manifest, indent=1), "application/json")

    version = {
        "version_id": vid,
        "folder": folder,
        "bucket": home,
        "data_yaml_uri": f"s3://{home}/{folder}/data.yaml",
        "train_data_uris": ",".join(run_prefixes),
        # A KEY in the app's bucket, not an s3:// URI: the start-up check
        # (`validate_dataset_folder`) reads it as a key in that bucket, and
        # an s3:// string made it look for a key that cannot exist -- the
        # queue skipped the first versioned run on 2026-10-26.
        "eval_dataset_folder": f"{folder}/eval",
        "eval_extra_uris": ",".join(extra),
        "manifest_uri": f"s3://{home}/{folder}/manifest.json",
        "train_images": len(train),
        "val_images": len(val),
        # Where the images COME FROM, for the queue card: the version's own
        # folder holds only lists, and naming it says nothing.
        "train_folders": train_folders,
        "val_folders": val_folders,
        "classes": list(train_doc["names"].values()),
        "sources": list(sources or []),
        "drop_undeclared": bool(drop_undeclared),
        "trigger": trigger,
        "note": note,
        "created_at": manifest["created_at"],
        "state": "created",
    }
    _row_put(version)
    return version


# --- the row ----------------------------------------------------------------------

def _table():
    from . import dataset_snapshots as ds
    return ds._table()


def _row_put(version: dict) -> None:
    """One row in the snapshot table, beside the snapshots it extends."""
    _table().put_item(Item={"snapshot_id": version["version_id"],
                            "record_type": RECORD_TYPE,
                            **{k: _for_dynamo(v) for k, v in version.items()
                               if k != "version_id" and v not in (None, "")}})


def get(version_id: str) -> dict:
    row = _table().get_item(Key={"snapshot_id": version_id}).get("Item")
    if not row or row.get("record_type") != RECORD_TYPE:
        raise ValueError(f"No dataset version called {version_id}.")
    return _plain({**row, "version_id": row["snapshot_id"]})


def listing(limit: int = 50) -> list:
    rows, kwargs = [], {"FilterExpression": "record_type = :t",
                        "ExpressionAttributeValues": {":t": RECORD_TYPE}}
    while True:
        resp = _table().scan(**kwargs)
        rows += resp.get("Items") or []
        if "LastEvaluatedKey" not in resp:
            break
        kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
    rows.sort(key=lambda r: str(r.get("created_at") or ""), reverse=True)
    return [_plain({**r, "version_id": r["snapshot_id"]}) for r in rows[:max(1, int(limit))]]


def mark(version_id: str, **fields) -> None:
    """Record what happened to a version: queued, started, which run."""
    if not version_id or not fields:
        return
    # Read, merge, write -- not UpdateItem: the app's role on this table
    # is Scan/Put/Get/Delete (terraform/app/audit.tf), and a version is
    # marked by one caller at a time (the queue, or the chain that made it).
    try:
        row = _table().get_item(Key={"snapshot_id": version_id}).get("Item")
        if not row:
            return
        row.update({k: _for_dynamo(v) for k, v in fields.items() if v not in (None, "")})
        _table().put_item(Item=row)
    except Exception as exc:                                        # noqa: BLE001
        print(json.dumps({"dataset_version": "mark_failed", "id": version_id, "error": str(exc)}))


def _for_dynamo(value):
    """DynamoDB takes no floats; the queue entry template carries some."""
    from decimal import Decimal
    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, dict):
        return {k: _for_dynamo(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_for_dynamo(v) for v in value]
    return value


def _plain(value):
    from decimal import Decimal
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if isinstance(value, Decimal):
        return int(value) if value == int(value) else float(value)
    return value


def yaml_text(version_id: str) -> dict:
    """The two yamls of a version, for reading on screen."""
    v = get(version_id)
    return {"version_id": version_id,
            "data_yaml": _read(v["bucket"], f"{v['folder']}/data.yaml"),
            "eval_yaml": _read(v["bucket"], f"{v['folder']}/eval/data.yaml")}


def files(version_id: str) -> dict:
    v = get(version_id)
    m = json.loads(_read(v["bucket"], f"{v['folder']}/manifest.json"))
    return {"train": m.get("train") or [], "val": m.get("val") or []}


# --- for a run --------------------------------------------------------------------

def for_run(*, home: str, root: str, train_data_uris: str = "", data_yaml_uri: str = "",
            eval_root: str = "", trigger: str = "", sources: list | None = None,
            drop_undeclared: bool = False, note: str = "") -> dict:
    """A version from whatever a run was about to be given.

    `train_data_uris` + `data_yaml_uri` for a composed run; otherwise the
    one folder `root`, whose data.yaml sits at its top. `eval_root` is the
    separate evaluation folder an older single-folder build may have.
    """
    home = home or pc.destination_bucket()
    if train_data_uris and data_yaml_uri:
        prefixes = [u.strip() for u in train_data_uris.split(",") if u.strip()]
        yb, yk = _uri(data_yaml_uri)
        text = _read(yb, yk)
    else:
        rb, rk = _uri(root) if str(root).startswith("s3://") else (home, str(root).strip("/"))
        prefixes = [f"s3://{rb}/{rk}/"]
        text = _read(rb, f"{rk}/data.yaml")
    eval_text, eval_prefixes = "", None
    if eval_root and eval_root.strip("/") != str(root).strip("/"):
        eb, ek = _uri(eval_root) if eval_root.startswith("s3://") else (home, eval_root.strip("/"))
        try:
            eval_text = _read(eb, f"{ek}/data.yaml")
            eval_prefixes = [f"s3://{eb}/{ek}/"]
        except Exception:                                           # noqa: BLE001
            eval_text = ""
    return create(home=home, yaml_text=text, prefixes=prefixes,
                  eval_yaml_text=eval_text, eval_prefixes=eval_prefixes,
                  trigger=trigger, sources=sources, drop_undeclared=drop_undeclared,
                  note=note)


def run_fields(version: dict) -> dict:
    """The queue-entry / start fields that point a run at a version."""
    eval_folder = str(version["eval_dataset_folder"])
    home = f"s3://{version.get('bucket') or pc.destination_bucket()}/"
    if eval_folder.startswith(home):
        # Versions made before the fix above stored the URI; a key in the
        # app's bucket is what the run's start-up check reads.
        eval_folder = eval_folder[len(home):]
    return {
        "dataset_folder": version["folder"],
        "train_data_uris": version["train_data_uris"],
        "data_yaml_uri": version["data_yaml_uri"],
        "eval_dataset_folder": eval_folder.strip("/"),
        "eval_extra_uris": version["eval_extra_uris"],
        "dataset_version_id": version["version_id"],
        # For the queue card: where the images come from, and how many.
        "dataset_source_folders": list(version.get("train_folders") or []),
        "eval_source_folders": list(version.get("val_folders") or []),
        "dataset_version_train_images": int(version.get("train_images") or 0),
        "dataset_version_val_images": int(version.get("val_images") or 0),
    }


_card_cache: dict = {}


def card_fields(version_id: str) -> dict:
    """The queue-card fields of a version, for an entry queued before they
    were carried on it. Cached: the queue card polls every 15 seconds and
    a version never changes."""
    if version_id not in _card_cache:
        v = get(version_id)
        if not v.get("train_folders"):
            v = {**v, **_from_manifest(v)}
        f = run_fields(v)
        _card_cache[version_id] = {k: f[k] for k in (
            "dataset_source_folders", "eval_source_folders",
            "dataset_version_train_images", "dataset_version_val_images")}
    return _card_cache[version_id]
