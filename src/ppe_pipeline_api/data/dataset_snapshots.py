"""A dataset definition, frozen hard enough to run again.

A generated `data.yaml` names week folders in place -- deliberately, so a
training set is not a second copy of the corpus. The cost is that it does
not pin anything: the same yaml, run in June and again in September,
names the same folders and trains on different images, because the pull
keeps filling them.

That is fine for "what would we train on now" and useless for "run 47
again". A snapshot closes the gap by recording three things together:

  * the `data.yaml` text, byte for byte
  * a MANIFEST -- every image and label key the yaml selected, at the
    moment it was taken
  * per-week counts, so drift is visible without reading the manifest

The manifest is what makes a run reproducible. `data.yaml` can only name
DIRECTORIES; it has no way to say "these files and not the others", so
something downstream has to do the filtering. `train.py` takes the
manifest URI and downloads exactly those keys -- which is both exact and
faster than fetching a folder that has since grown.

Nothing here deletes or copies image data. The manifest is a list of
names; the images stay where the pull put them, which is what makes any
of this recoverable at all.

Layout, outside the training prefix on purpose -- under it, every run
would download the whole snapshot history before training:

    _dataset_snapshots/<root>/<id>/data.yaml
    _dataset_snapshots/<root>/<id>/manifest.json
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os

from . import dataset_yaml
from .. import pull_control as pc

#: Where snapshots live. Deliberately not under the week-folder root:
#: `train.py` downloads every object beneath the prefix it is given, so a
#: history kept inside it would be fetched on every single run.
PREFIX = "_dataset_snapshots"

TABLE_NAME = os.environ.get("DATASET_SNAPSHOT_TABLE", "track2-yolo-dataset-snapshots")

#: A listing guard. The table is small by nature -- one row per saved
#: definition -- but a scan with no ceiling is how a page hangs.
MAX_ROWS = 200


def _table():
    import boto3

    return boto3.resource(
        "dynamodb", region_name=os.environ.get("AWS_REGION", "ap-southeast-1"),
    ).Table(TABLE_NAME)


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def composed_local_paths(roots: list, home: str, files_by_root: dict) -> list:
    """Every file of a composed run, named where the trainer puts it.

    `roots` are the run's download prefixes (a bare key in `home`, or
    `s3://bucket/key`); `files_by_root` gives each one's files relative to
    it. Named by `dataset_yaml.local_layout` -- the trainer's own rule --
    so the manifest, the yaml and the disk agree: one prefix stripped,
    several in one bucket full keys, several buckets <bucket>/<key> for
    every file. Code review 2026-10-26, F5: this recorded only FOREIGN
    files with a bucket, and kept full keys for a single prefix.
    """
    def uri(root: str) -> tuple:
        if root.startswith("s3://"):
            b, _, k = root[5:].partition("/")
            return b, k.strip("/")
        return home, root.strip("/")

    parsed = [uri(r) for r in roots]
    local = dataset_yaml.local_layout([f"s3://{b}/{k}" for b, k in parsed])
    out: list = []
    for root, (b, k) in zip(roots, parsed):
        out += [local(b, f"{k}/{rel}") for rel in files_by_root.get(root) or []]
    return sorted(set(out))


def collect_files(root: str, *, bucket: str = "", images_subpath: str = "images",
                  labels_subpath: str = "labels") -> dict:
    """Every image and label key under a root, relative to it.

    Relative, not absolute: `train.py` downloads a prefix and joins each
    key to a local directory, so a manifest of paths relative to that
    prefix is the form it can check against without knowing the bucket.

    Sorted, so the same corpus always produces the same manifest bytes
    and therefore the same hash. An unsorted listing would make every
    snapshot of an unchanged dataset look like a new one.
    """
    root = (root or "").strip().strip("/")
    if not root:
        raise ValueError("Pick the folder that holds the week folders.")
    bucket = bucket or pc.destination_bucket()
    images_subpath = (images_subpath or "images").strip("/")
    labels_subpath = (labels_subpath or "labels").strip("/")

    files: list = []
    per_week: dict = {}
    for obj in pc._list_objects(root + "/", bucket):
        key = obj["Key"]
        rest = key[len(root) + 1:]
        parts = rest.split("/")
        # WHERE the images/labels segment sits decides the layout, rather
        # than assuming it is always second.
        #
        #   <week>/images/f.jpg   -- week folders, what the pull builds
        #   images/train/f.jpg    -- the ordinary Ultralytics layout, and
        #                            what "Assemble a dataset" names
        #
        # This used to read `parts[0]` as the week and require the REST to
        # start with `images/`. On the second layout that makes the week
        # "images" and the rest "train/f.jpg", which starts with neither
        # -- so every file was skipped and the snapshot froze ZERO files.
        # A freeze is the record that makes a run repeatable, so an empty
        # one is worse than no freeze: it reports success and records
        # nothing.
        # A key ending in "/" is a prefix marker, not a file. Counting it
        # would put a name in the manifest the trainer cannot download.
        if not parts[-1]:
            continue
        at = next((i for i, part in enumerate(parts[:-1])
                   if part in (images_subpath, labels_subpath)), -1)
        if at < 0:
            continue
        kind = "images" if parts[at] == images_subpath else "labels"
        # The grouping the weeks table shows: whatever the folder sits
        # under, or -- when images/ IS the top level -- the split beneath
        # it, which is the only meaningful division there.
        week = parts[at - 1] if at > 0 else (parts[at + 1] if len(parts) > at + 2 else "all")
        files.append(rest)
        row = per_week.setdefault(week, {"week": week, "images": 0, "labels": 0})
        row[kind] += 1

    files.sort()
    weeks = sorted(per_week.values(),
                   key=lambda r: (dataset_yaml._digits(r["week"]), r["week"]))
    return {"root": root, "bucket": bucket, "files": files, "weeks": weeks}


def _read_text(bucket: str, key: str) -> str:
    return pc.s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")


def _manifest_files(row: dict) -> list:
    """The file names a snapshot archived, or an empty list.

    Empty rather than raising: a restore that cannot read its manifest
    should fall back to copying the yaml, which is what it always did,
    not fail outright and leave the live slot untouched with an error
    nobody can act on.
    """
    try:
        body = _read_text(row["bucket"], row["manifest_key"])
        return list((json.loads(body) or {}).get("files") or [])
    except Exception:                                          # noqa: BLE001
        return []


def lists_from_manifest(files: list, *, images_subpath: str = "images") -> dict:
    """The manifest's images, split into train and val lines.

    This is what makes a snapshot of a FOLDER-NAMING yaml actually
    frozen. Such a yaml says `train: [week_1/images/train, ...]`, so
    restoring it hands the trainer whatever those folders hold TODAY --
    which for a corpus that is still being pulled into is not the dataset
    that was frozen at all. The manifest already lists every file; it was
    simply never turned back into something a run could read.

    Lines are relative to the ROOT and carry the `./` Ultralytics expands
    against THE LIST FILE'S OWN DIRECTORY -- not against `path:`. That is
    why the lists belong at the root beside data.yaml and not in a
    subfolder: a list at `<root>/_retained/train.txt` holding
    `./week_1/...` resolves to `<root>/_retained/week_1/...`, which does
    not exist. `retained_lists.FOLDER` is "" for the same reason, and it
    cost a run to learn.
    """
    images_subpath = (images_subpath or "images").strip("/")
    out: dict = {"train": [], "val": []}
    for rest in files or []:
        parts = str(rest).split("/")
        # week / images / split / name
        if len(parts) < 4 or parts[1] != images_subpath:
            continue
        if parts[2] not in out:
            continue
        out[parts[2]].append("./" + rest.lstrip("/"))
    out["train"].sort()
    out["val"].sort()
    return out


def _yaml_naming_lists(yaml_text: str, train_entry: str, val_entry: str) -> str:
    """The same yaml, with its splits pointed at two list files.

    Re-dumped rather than patched line by line: the splits are a LIST in
    a folder-naming yaml, and editing a multi-line block with string
    surgery is how a half-replaced `train:` ends up naming both a list
    and three week folders.

    The header comments do not survive, which is the honest trade: a
    restored yaml describes the snapshot it came from, and the snapshot's
    own archived copy still carries the original wording.
    """
    import yaml as _yaml

    document = _yaml.safe_load(yaml_text) or {}
    if not isinstance(document, dict):
        document = {}
    document["train"] = train_entry
    document["val"] = val_entry
    header = [
        "# Written by a snapshot restore -- every image pinned by name.",
        "# `train:`/`val:` name list files rather than week folders, so",
        "# this trains on exactly what the snapshot froze, whatever the",
        "# week folders hold now.",
    ]
    return "\n".join(header) + "\n" + _yaml.safe_dump(document, sort_keys=False)


def _identity(yaml_text: str, files: list) -> str:
    """The snapshot's id: a hash over the definition AND the data.

    Both, because either alone lies. The same yaml over a grown corpus is
    a different dataset; the same corpus under a re-ordered class list
    trains a different model. Hashing the pair means two snapshots share
    an id only when a run against either would be the same run.

    `files` is the SELECTION where the yaml points at file lists, and the
    whole corpus otherwise. That is the same rule the audit follows, and
    for the same reason: when a list decides what a run reads, the corpus
    around it can grow without changing the run at all.
    """
    digest = hashlib.sha256()
    digest.update(yaml_text.encode("utf-8"))
    digest.update(b"\0")
    for name in files:
        digest.update(name.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()[:16]


def _yaml_doc(text: str) -> dict:
    """The yaml as a mapping, or {} if it will not parse.

    Never raises: a snapshot that refuses to be taken because the live
    yaml is malformed loses the record of a run that is about to happen
    anyway.
    """
    try:
        import yaml as _yaml

        doc = _yaml.safe_load(text or "") or {}
    except Exception:                                          # noqa: BLE001
        return {}
    return doc if isinstance(doc, dict) else {}


def _yaml_names(text: str) -> dict:
    """`names` out of a data.yaml, as {index: name}.

    Class names exist ONLY in the yaml -- a label line is
    `2 0.51 0.22 0.26 0.08`, so scanning labels gives indices and counts
    and never names. If this is not read here, the snapshot has no
    record of what the indices meant.
    """
    names = _yaml_doc(text).get("names")
    if isinstance(names, dict):
        pairs = names.items()
    elif isinstance(names, list):
        pairs = enumerate(names)
    else:
        return {}
    out: dict = {}
    for index, name in pairs:
        try:
            out[int(index)] = str(name)
        except (TypeError, ValueError):
            continue
    return out


def _count_declared(text: str, split: str, files: list, images_subpath: str) -> int:
    """How many images sit under the folders this yaml names for a split.

    `train:` and `val:` each take a directory, a list of directories, or
    a .txt of image paths. The .txt case is already counted from the
    lists; this is the directory case, which was counted as zero.
    """
    entry = _yaml_doc(text).get(split)
    if isinstance(entry, str):
        entries = [entry]
    elif isinstance(entry, list):
        entries = [str(e) for e in entry]
    else:
        return 0
    wanted = [e.strip().strip("/") for e in entries if str(e).strip()]
    # A path list, not a folder -- the lists already answered this.
    wanted = [w for w in wanted if not w.lower().endswith(".txt")]
    if not wanted:
        return 0
    marker = f"/{images_subpath}/"
    count = 0
    for name in files:
        if not any(name == w or name.startswith(w + "/") for w in wanted):
            continue
        if name.startswith(images_subpath + "/") or marker in name:
            count += 1
    return count


def create(*, root: str, bucket: str = "", source_bucket: str = "",
           images_subpath: str = "images", labels_subpath: str = "labels",
           note: str = "", eval_root: str = "",
           extra_roots=()) -> dict:
    """Freeze the current definition of `root` and record it.

    Writes two small objects and one row. No image is copied and no
    folder is created -- the manifest is a list of names.

    `eval_root` is the SECOND download, when the evaluation set lives
    under a prefix of its own. Training and evaluation are separate
    downloads -- the training job takes TRAIN_DATA_S3_URI, the evaluation
    step takes EVAL_DATA_S3_URI and runs in a different container -- so
    the validation set can sit anywhere in the bucket. A snapshot that
    walked only `root` recorded none of it: the manifest held the
    training files, the row's counts described the training files, and
    the images the model was SCORED on had no record at all. Every
    number a registered version is approved on comes from that side.

    Ignored when it sits under `root`, which is the ordinary case -- one
    prefix covers both splits and one manifest already has them.

    `extra_roots` is the COMPOSED case: a dataset built from folders
    under unrelated roots, where `root` holds the generated yaml and
    nothing else. Walking it alone would freeze a manifest of one
    file and call it the dataset. The images are listed under their
    own keys, which is also the form `train.py` writes them to disk
    in once it is downloading more than one prefix -- so the manifest
    and the download agree without either converting.
    """
    from . import selection as _selection

    collected = collect_files(root, bucket=bucket, images_subpath=images_subpath,
                              labels_subpath=labels_subpath)
    target = collected["bucket"]

    # The composed case: every prefix the run will fetch, listed as
    # FULL keys. `prefix` then has to be empty, because there is no
    # one prefix these are relative to.
    composed = [str(r).strip().strip("/") for r in (extra_roots or ()) if str(r).strip()]
    key_prefix = collected["root"] + "/"
    if composed:
        by_root: dict = {}
        weeks: list = []
        for extra in composed:
            # `s3://bucket/key` for a root in another bucket.
            xb, key = target, extra
            if extra.startswith("s3://"):
                xb, _, key = extra[5:].partition("/")
            side = collect_files(key, bucket=xb,
                                 images_subpath=images_subpath,
                                 labels_subpath=labels_subpath)
            by_root[extra] = side["files"]
            # Named for the prefix they came from, so a week number
            # that appears under two roots stays two rows.
            weeks += [{**row, "week": f"{side['root']}/{row['week']}"}
                      for row in side["weeks"]]
        # Where the trainer lays each file out -- see composed_local_paths.
        merged = composed_local_paths(composed, target, by_root)
        collected = {**collected, "files": merged, "weeks": weeks}
        key_prefix = ""

    # The LIVE yaml where there is one, rebuilt only as a fallback. A
    # snapshot is a record of what is configured right now, and the live
    # slot is what a run would read -- rebuilding instead would freeze a
    # yaml that has never been anywhere, and could differ from the one
    # sitting in the bucket.
    live = _selection.read_selection(collected["root"], bucket=target)
    if live["yaml"]:
        yaml_text = live["yaml"]
        # Read OUT of the live yaml, not left empty.
        #
        # These two were filled only from the file lists, so a yaml that
        # names FOLDERS -- which is every dataset the Assemble card
        # writes -- froze as "0 train, 0 val images, no classes" beside a
        # manifest of 16,910 names. The class list is the thing that
        # makes an old run explicable after somebody reorders a taxonomy,
        # and it was the field most reliably absent.
        declared = _yaml_names(yaml_text)
        listed_train = len(live["lists"].get("train", {}).get("lines") or [])
        listed_val = len(live["lists"].get("val", {}).get("lines") or [])
        built = {
            "download_prefix": f"s3://{target}/{collected['root']}/",
            "write_to": f"{collected['root']}/data.yaml",
            # A pinned list where there is one; otherwise count what the
            # folders the yaml names actually hold, which is what a run
            # reading those folders would train on.
            "train_images": listed_train or _count_declared(
                yaml_text, "train", collected["files"], images_subpath),
            "val_images": listed_val or _count_declared(
                yaml_text, "val", collected["files"], images_subpath),
            "classes": declared, "problems": [], "ready": True,
        }
    else:
        built = dataset_yaml.build(
            root=root, bucket=bucket, source_bucket=source_bucket,
            images_subpath=images_subpath, labels_subpath=labels_subpath,
        )
        yaml_text = built["yaml"]

    # The second download, when there is one.
    eval_root = (eval_root or "").strip().strip("/")
    separate_eval = bool(eval_root) and not (
        eval_root == collected["root"]
        or eval_root.startswith(collected["root"] + "/")
    )
    eval_collected: dict = {}
    eval_yaml = ""
    if separate_eval:
        eval_collected = collect_files(eval_root, bucket=target,
                                       images_subpath=images_subpath,
                                       labels_subpath=labels_subpath)
        eval_live = _selection.read_selection(eval_collected["root"], bucket=target)
        eval_yaml = eval_live["yaml"] or ""

    # Hashed over what a RUN would read: the selection when the yaml
    # names file lists, the whole corpus when it names folders.
    #
    # The evaluation side is in the hash too. Two runs over the same
    # training images and DIFFERENT held-out sets are not the same
    # dataset, and giving them one id would file two sets of metrics
    # under one record.
    selection = sorted(live["images"]) if live["images"] else collected["files"]
    snapshot_id = _identity(yaml_text + eval_yaml,
                            selection + (eval_collected.get("files") or []))
    folder = f"{PREFIX}/{collected['root']}/{snapshot_id}"

    manifest = {
        "snapshot_id": snapshot_id,
        "root": collected["root"],
        # The prefix the keys are relative to, so the trainer can check a
        # downloaded file without re-deriving it. Empty for a composed
        # dataset, whose names are already full keys.
        "prefix": key_prefix,
        "created_at": _now(),
        "weeks": collected["weeks"],
        "files": collected["files"],
    }

    pc.s3.put_object(
        Bucket=target, Key=f"{folder}/data.yaml",
        Body=yaml_text.encode("utf-8"), ContentType="text/yaml",
    )
    pc.s3.put_object(
        Bucket=target, Key=f"{folder}/manifest.json",
        Body=json.dumps(manifest, indent=1).encode("utf-8"),
        ContentType="application/json",
    )

    # The evaluation side gets its own pair, beside the first and named
    # for what it is. Not merged into one manifest: the two are separate
    # downloads with separate prefixes, and a single list of relative
    # names could not say which prefix each belonged to.
    eval_block: dict = {}
    if separate_eval:
        eval_manifest = {
            "snapshot_id": snapshot_id,
            "root": eval_collected["root"],
            "prefix": eval_collected["root"] + "/",
            "created_at": _now(),
            "weeks": eval_collected["weeks"],
            "files": eval_collected["files"],
        }
        pc.s3.put_object(
            Bucket=target, Key=f"{folder}/eval-manifest.json",
            Body=json.dumps(eval_manifest, indent=1).encode("utf-8"),
            ContentType="application/json",
        )
        if eval_yaml:
            pc.s3.put_object(
                Bucket=target, Key=f"{folder}/eval-data.yaml",
                Body=eval_yaml.encode("utf-8"), ContentType="text/yaml",
            )
        eval_block = {
            "root": eval_collected["root"],
            "manifest_key": f"{folder}/eval-manifest.json",
            "yaml_key": f"{folder}/eval-data.yaml" if eval_yaml else "",
            "download_prefix": f"s3://{target}/{eval_collected['root']}/",
            "file_count": len(eval_collected["files"]),
            "images": _count_declared(eval_yaml, "val", eval_collected["files"],
                                      images_subpath)
            or _count_declared(eval_yaml, "train", eval_collected["files"],
                               images_subpath),
            "weeks": eval_collected["weeks"],
        }

    # The two file lists, if the yaml points at any. Without these a
    # restore hands back a yaml naming `_retained/train.txt` whose
    # contents have moved on since -- which looks right and is not, and
    # is the worst of the three possible outcomes.
    lists: dict = {}
    for split, entry in (live.get("lists") or {}).items():
        key = f"{folder}/{split}.txt"
        try:
            pc.s3.copy_object(
                Bucket=target, Key=key,
                CopySource={"Bucket": target, "Key": entry["key"]},
                ContentType="text/plain", MetadataDirective="REPLACE",
            )
        except Exception as exc:                               # noqa: BLE001
            # NOT skipped. The comment above says why: without the lists a
            # restore hands back a yaml naming `_retained/train.txt` whose
            # contents have moved on, "which looks right and is not, and
            # is the worst of the three possible outcomes".
            #
            # Continuing produced exactly that -- a snapshot written,
            # recorded and reported as successful, silently missing the
            # one thing that makes it a snapshot. A freeze that cannot
            # freeze has to fail, and `run_chain` already treats a failed
            # freeze as fatal rather than training something there is no
            # record of.
            raise ValueError(
                f"The {split} file list could not be copied into the "
                f"snapshot ({exc}). Nothing was recorded: a snapshot "
                f"missing a split restores a dataset that is not the one "
                f"it froze."
            )
        lists[split] = {
            "key": key,
            # Where it goes back to on a restore. Kept per snapshot
            # because the yaml's entry is what decides it, and an older
            # yaml may name a different path.
            "live_key": entry["key"],
            "lines": len(entry["lines"]),
        }

    row = {
        "snapshot_id": snapshot_id,
        "created_at": _now(),
        "root": collected["root"],
        "bucket": target,
        "note": (note or "").strip()[:400],
        "yaml_key": f"{folder}/data.yaml",
        "manifest_key": f"{folder}/manifest.json",
        # The selection, archived. `lists` is empty for a folder-list
        # yaml, which trains on whole weeks and has nothing to pin.
        "lists": lists,
        "selected": len(selection),
        "names_from": "list" if live["images"] else "folder",
        "manifest_uri": f"s3://{target}/{folder}/manifest.json",
        # Where a run must be pointed, and where a restore puts the yaml.
        "download_prefix": built["download_prefix"],
        "live_key": built["write_to"],
        # Small enough for the row, and the whole point of it: drift is
        # readable without fetching a manifest of a hundred thousand
        # names.
        "weeks": collected["weeks"],
        "file_count": len(collected["files"]),
        "train_images": built["train_images"],
        "val_images": built["val_images"],
        # In index order. This is what makes an old run explicable after
        # somebody reorders a taxonomy.
        "classes": [built["classes"][k] for k in sorted(built["classes"], key=int)]
        if built["classes"] else [],
        "problems": built["problems"],
        "ready": built["ready"],
        # The second download. Empty for the ordinary single-prefix case,
        # where one manifest already covers both splits.
        "eval": eval_block,
    }
    try:
        _table().put_item(Item=_decimalise(row))
    except Exception as exc:                                   # noqa: BLE001
        raise ValueError(
            f"The snapshot was built but could not be recorded: {exc}. "
            f"If the table '{TABLE_NAME}' does not exist yet, apply terraform/app "
            f"(aws_dynamodb_table.dataset_snapshots)."
        )
    row["yaml"] = yaml_text
    return row


def _decimalise(value):
    """Floats are not a DynamoDB number. Everything here is a count or a
    string, so rounding is not in question -- but `problems` carries
    whatever `build` put in it."""
    if isinstance(value, float):
        import decimal

        return decimal.Decimal(str(value))
    if isinstance(value, dict):
        return {k: _decimalise(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decimalise(v) for v in value]
    return value


def _plain(value):
    """The inverse, for sending a row to the browser."""
    import decimal

    if isinstance(value, decimal.Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain(v) for v in value]
    return value


def table_ready() -> bool:
    """Whether the snapshot table can be read.

    Asked separately so an empty list can mean "nothing frozen" and only
    that. Swallowing the missing-table error into `[]` would have the
    card say "nothing frozen yet" about a table that cannot be written
    to, which is the wrong thing to go and check.

    A one-row Scan, not `table_status`. Reading the status is
    `dynamodb:DescribeTable`, which this app is not granted and does not
    otherwise need -- so the probe reported the table unreachable while
    every operation the feature actually performs was working. Test the
    permission the feature needs, not a neighbouring one.
    """
    try:
        _table().scan(Limit=1)
        return True
    except Exception:                                          # noqa: BLE001
        return False


def listing(*, root: str = "", limit: int = 50) -> list:
    """Snapshots, newest first. Scanned, not queried: the table holds one
    row per saved definition, which is tens, not millions."""
    limit = max(1, min(int(limit or 50), MAX_ROWS))
    try:
        rows = _table().scan(Limit=MAX_ROWS).get("Items", [])
    except Exception:                                          # noqa: BLE001
        return []
    # Dataset versions share this table (dataset_versions.py) and are not
    # snapshots anybody picked; they are listed on their own.
    rows = [_plain(r) for r in rows if r.get("record_type") != "dataset_version"]
    if root:
        wanted = root.strip("/")
        rows = [r for r in rows if r.get("root") == wanted]
    rows.sort(key=lambda r: str(r.get("created_at") or ""), reverse=True)
    return rows[:limit]


def get(snapshot_id: str, *, with_yaml: bool = True) -> dict:
    row = _table().get_item(Key={"snapshot_id": snapshot_id}).get("Item")
    if not row:
        raise ValueError(f"No snapshot {snapshot_id}.")
    row = _plain(row)
    if with_yaml:
        try:
            body = pc.s3.get_object(Bucket=row["bucket"], Key=row["yaml_key"])["Body"].read()
            row["yaml"] = body.decode("utf-8", "replace")
        except Exception:                                      # noqa: BLE001
            row["yaml"] = ""
    return row


def drift(snapshot_id: str) -> dict:
    """What has changed under the snapshot since it was taken.

    Per week rather than in total, because "the corpus grew by 400" does
    not say whether an OLD week was backfilled -- and a backfilled old
    week is the case where a re-run silently trains on more than the
    original did.
    """
    row = get(snapshot_id, with_yaml=False)
    now = collect_files(row["root"], bucket=row["bucket"])
    then = {w["week"]: w for w in row.get("weeks") or []}
    current = {w["week"]: w for w in now["weeks"]}

    changes = []
    for week in sorted(set(then) | set(current),
                       key=lambda n: (dataset_yaml._digits(n), n)):
        was = then.get(week, {"images": 0, "labels": 0})
        has = current.get(week, {"images": 0, "labels": 0})
        if int(was["images"]) != int(has["images"]):
            changes.append({
                "week": week,
                "was": int(was["images"]),
                "now": int(has["images"]),
                "new_week": week not in then,
            })
    return {
        "snapshot_id": snapshot_id,
        "root": row["root"],
        "file_count_then": int(row.get("file_count") or 0),
        "file_count_now": len(now["files"]),
        "changes": changes,
        # The snapshot is still runnable either way -- that is what the
        # manifest is for. This says whether a re-run WITHOUT it would
        # have differed.
        "unchanged": not changes,
    }


def remove(snapshot_id: str) -> dict:
    """Delete a frozen dataset: its row and its three artifacts.

    The IMAGES are untouched, and that is the whole point -- a snapshot
    is a list of names beside a copy of a yaml, never a copy of the
    corpus. Deleting one throws away the record of a selection, not the
    data it selected, so the only thing lost is the ability to repeat
    that exact run.

    Which is still a real loss, and the reason this is deliberate rather
    than tidy-up: a training audit that points at a snapshot nobody can
    read any more is a run that can no longer be explained. Delete the
    ones that were experiments, keep the ones a model was shipped from.
    """
    row = get(snapshot_id, with_yaml=False)
    keys = [row.get("yaml_key"), row.get("manifest_key")]
    keys += [entry.get("key") for entry in (row.get("lists") or {}).values()]

    removed = []
    for key in [k for k in keys if k]:
        try:
            pc.s3.delete_object(Bucket=row["bucket"], Key=key)
            removed.append(key)
        except Exception:                                      # noqa: BLE001
            # A missing object is already in the state being asked for.
            continue

    # The row LAST. Deleting it first and then failing on the objects
    # would leave artifacts nothing points at -- unreachable, and
    # invisible to a listing that reads the table.
    _table().delete_item(Key={"snapshot_id": snapshot_id})
    return {"snapshot_id": snapshot_id, "removed": removed,
            "root": row.get("root", "")}


def restore(snapshot_id: str) -> dict:
    """Put a snapshot's yaml back in the one place the trainer reads.

    There is exactly one live slot per root -- `<parent>/data.yaml` --
    because `train.py` requires the yaml at the root of the prefix it
    downloads, and only that parent contains every week folder the yaml
    names. So restoring is a copy INTO that slot, replacing what is
    there.

    Nothing is lost by it: every snapshot keeps its own copy, so the
    definition being replaced is still readable under its own id.
    """
    row = get(snapshot_id, with_yaml=False)

    # The LISTS first, then the yaml. The other order leaves a window
    # where the restored yaml names lists that still hold the previous
    # selection, and a run started in it trains on the wrong images
    # while looking entirely correct.
    restored = []
    for split, entry in (row.get("lists") or {}).items():
        pc.s3.copy_object(
            Bucket=row["bucket"], Key=entry["live_key"],
            CopySource={"Bucket": row["bucket"], "Key": entry["key"]},
            ContentType="text/plain", MetadataDirective="REPLACE",
        )
        restored.append(entry["live_key"])

    # No lists to put back means the snapshot froze a FOLDER-NAMING yaml,
    # and restoring that yaml alone would train on whatever those week
    # folders hold today. For a corpus that is still being pulled into,
    # that is not the dataset anybody froze -- which is exactly what
    # "Train on this" did, and why it looked broken while reporting
    # success.
    #
    # The manifest has always held every file. Turning it back into the
    # two lists a run can read is all that was missing.
    pinned = 0
    if not (row.get("lists") or {}):
        lists = lists_from_manifest(_manifest_files(row))
        if lists["train"] or lists["val"]:
            from . import retained_lists as rl

            entries = {}
            for split, name in (("train", rl.TRAIN_NAME), ("val", rl.VAL_NAME)):
                key = f"{row['root']}/{name}"
                pc.s3.put_object(
                    Bucket=row["bucket"], Key=key,
                    Body=rl.render(lists[split]).encode("utf-8"),
                    ContentType="text/plain",
                )
                restored.append(key)
                entries[split] = name
                pinned += len(lists[split])

            yaml_text = _read_text(row["bucket"], row["yaml_key"])
            pc.s3.put_object(
                Bucket=row["bucket"], Key=row["live_key"],
                Body=_yaml_naming_lists(
                    yaml_text, entries["train"], entries["val"],
                ).encode("utf-8"),
                ContentType="text/yaml",
            )
            restored.append(row["live_key"])
            return {
                "snapshot_id": snapshot_id,
                "root": row["root"],
                "written_to": f"s3://{row['bucket']}/{row['live_key']}",
                "restored": restored,
                # WHEN IT WAS FROZEN, not when it was put back. The card
                # showed the restore time, so a snapshot taken on 30
                # September read as "at 2026-10-06T06:59" -- the one fact
                # that tells one frozen dataset from another, replaced by
                # the time you clicked.
                "created_at": str(row.get("created_at") or ""),
                "note": str(row.get("note") or ""),
                "download_prefix": row["download_prefix"],
                "manifest_uri": row["manifest_uri"],
                # What a run will now actually train on, which is the
                # number worth reporting -- not the row's own `selected`,
                # which for a folder yaml counted the whole corpus.
                "selected": pinned,
                "names_from": "list",
                "pinned_from_manifest": True,
            }

    pc.s3.copy_object(
        Bucket=row["bucket"], Key=row["live_key"],
        CopySource={"Bucket": row["bucket"], "Key": row["yaml_key"]},
        ContentType="text/yaml", MetadataDirective="REPLACE",
    )
    restored.append(row["live_key"])

    return {
        "snapshot_id": snapshot_id,
        "root": row["root"],
        "written_to": f"s3://{row['bucket']}/{row['live_key']}",
        "restored": restored,
        # The same, for the unpinned path.
        "created_at": str(row.get("created_at") or ""),
        "note": str(row.get("note") or ""),
        "download_prefix": row["download_prefix"],
        "manifest_uri": row["manifest_uri"],
        "selected": int(row.get("selected") or 0),
        "names_from": row.get("names_from") or "folder",
        "pinned_from_manifest": False,
    }
