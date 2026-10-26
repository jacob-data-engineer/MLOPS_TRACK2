"""Turn the week folders into a `data.yaml`, without moving a file.

The pull writes weeks; this points a trainer at them. Nothing is copied,
consolidated or rewritten -- an Ultralytics `train:` accepts a LIST of
image directories, so every week can be named in place and the folders
stay exactly where the pull left them.

    train:
      - retraining_training/week_1/images
      - retraining_training/week_2/images
    val:
      - retraining_evaluation/week_1/images

Labels are never mentioned, and must not be: Ultralytics finds them by
swapping `/images/` for `/labels/` in each image path. That is why the
pull's layout has to be `<week>/images/` beside `<week>/labels/` -- name
the label folders here and they would be ignored, silently, and every
image would train as a background example with no annotations.
"""

from __future__ import annotations

import datetime as _dt
import concurrent.futures
import hashlib
import json

import yaml

from . import pull_control as pc

#: Where a `data.yaml` is looked for, relative to a source folder: the
#: folder itself, then its parents. A pulled dataset usually keeps its
#: yaml beside `images/`, which is one or two levels above the folder
#: that was actually picked (`.../images/train`).
CLASS_SEARCH_DEPTH = 3


def _digits(name: str) -> int:
    """The number in `week_10` / `day_3`, for ordering."""
    found = "".join(ch for ch in name if ch.isdigit())
    return int(found) if found else 0


def scan_root(root: str, *, bucket: str = "", images_subpath: str = "images",
              labels_subpath: str = "labels", split: str = "") -> list:
    """Every week folder under a root, oldest first, with what is in it.

    Ordered by the NUMBER in the folder name, not alphabetically, so
    `week_10` follows `week_9` instead of sitting between `week_1` and
    `week_2`. The prefix itself is not assumed: `day_3` and `batch-3`
    order the same way, because only the digits are read.

    `split` reads one side of a single-root layout -- `images/train` or
    `images/val` under each week. Blank reads the week's `images/` whole,
    which is what a root dedicated to one split looks like.
    """
    root = (root or "").strip().strip("/")
    if not root:
        return []
    bucket = bucket or pc.destination_bucket()
    images_subpath = (images_subpath or "images").strip("/")
    labels_subpath = (labels_subpath or "labels").strip("/")

    split = (split or "").strip("/")
    # What a file of each kind looks like, relative to the week folder.
    # Matched as a PREFIX rather than a substring: with a split in play,
    # `images/train/` and `images/` are different places, and `in` would
    # count every val image as a train one.
    image_under = f"{images_subpath}/{split}/" if split else f"{images_subpath}/"
    label_under = f"{labels_subpath}/{split}/" if split else f"{labels_subpath}/"

    counts: dict = {}
    for obj in pc._list_objects(root + "/", bucket):
        rest = obj["Key"][len(root) + 1:]
        parts = rest.split("/")
        if len(parts) < 2:
            continue
        week, inside = parts[0], "/".join(parts[1:])
        row = counts.setdefault(week, {"week": week, "images": 0, "labels": 0, "bytes": 0})
        if inside.startswith(image_under):
            row["images"] += 1
            row["bytes"] += obj.get("Size", 0)
        elif inside.startswith(label_under):
            row["labels"] += 1
            row["bytes"] += obj.get("Size", 0)

    # A week the split never reached is not a week of this split.
    counts = {name: row for name, row in counts.items()
              if row["images"] or row["labels"]}

    weeks = sorted(counts.values(), key=lambda r: (_digits(r["week"]), r["week"]))
    for row in weeks:
        tail = f"/{split}" if split else ""
        row["images_path"] = f"{root}/{row['week']}/{images_subpath}{tail}"
        row["labels_path"] = f"{root}/{row['week']}/{labels_subpath}{tail}"
        # A week holding images but no labels trains every one of them as
        # a background example. That is a legitimate thing to want and a
        # catastrophic thing to do by accident, so it is reported rather
        # than decided.
        row["unlabelled"] = bool(row["images"]) and not row["labels"]
    return weeks


def find_classes(candidates, *, bucket: str = "") -> dict:
    """The class names to write, and where they were read from.

    Read from a `data.yaml` that already sits beside the source images,
    because the indices in the label files came from that file. Any other
    source -- a detector's list, a typed one -- can disagree with it in
    ORDER, and a disagreement in order is silent: every box trains under
    the wrong class and nothing errors.
    """
    bucket = bucket or pc.destination_bucket()
    tried: list = []
    for candidate in candidates:
        prefix = str(candidate or "").strip().strip("/")
        if not prefix:
            continue
        parts = prefix.split("/")
        # The folder, then its parents. `.../images/train` keeps its yaml
        # two levels up.
        for depth in range(min(CLASS_SEARCH_DEPTH, len(parts)) + 1):
            base = "/".join(parts[:len(parts) - depth]) if depth else prefix
            if not base:
                continue
            key = f"{base}/data.yaml"
            if key in tried:
                continue
            tried.append(key)
            try:
                body = pc.s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            except Exception:
                continue
            names = pc._classes_from_yaml(body.decode("utf-8", "replace"))
            if names:
                return {"names": {int(k): v for k, v in names.items()},
                        "source": f"s3://{bucket}/{key}", "looked_in": tried}
    return {"names": {}, "source": "", "looked_in": tried}


def _common_parent(*roots: str) -> str:
    """The deepest prefix that contains all of them.

    It matters because the trainer downloads ONE prefix and reads the
    yaml from its root, so the yaml can only reference paths inside that
    prefix. Two top-level roots have only the bucket in common, which
    would mean downloading the whole bucket -- worth saying out loud
    rather than letting the run discover it.
    """
    parts = [r.strip("/").split("/") for r in roots if r and r.strip("/")]
    if not parts:
        return ""
    shared: list = []
    for pieces in zip(*parts):
        if len(set(pieces)) != 1:
            break
        shared.append(pieces[0])
    return "/".join(shared)


def build(*, training_root: str = "", eval_root: str = "", root: str = "",
          bucket: str = "", source_bucket: str = "", images_subpath: str = "images",
          labels_subpath: str = "labels", class_sources=(),
          names: dict | None = None,
          retained_train: str = "", retained_val: str = "") -> dict:
    """The yaml, the paths in it, and everything wrong with it.

    Writes nothing. The card shows the result and only then offers to
    save it, because a generated `data.yaml` is a training decision and
    should be read before it is one.

    `bucket` holds the week folders; `source_bucket` holds the datasets
    the pull reads, which is where the class names are found. They are
    usually different buckets, and conflating them is why the names came
    back empty the first time this ran.
    """
    bucket = bucket or pc.destination_bucket()
    root = (root or "").strip().strip("/")
    training_root = (training_root or "").strip().strip("/")
    eval_root = (eval_root or "").strip().strip("/")

    if root:
        # One root holding both splits: each week has images/train and
        # images/val under it, so the two sides are read from the same
        # tree rather than from two roots.
        training_root = eval_root = root
        train_weeks = scan_root(root, bucket=bucket, split="train",
                                images_subpath=images_subpath,
                                labels_subpath=labels_subpath)
        val_weeks = scan_root(root, bucket=bucket, split="val",
                              images_subpath=images_subpath,
                              labels_subpath=labels_subpath)
    else:
        if not training_root:
            raise ValueError("Pick the root folder.")
        train_weeks = scan_root(training_root, bucket=bucket,
                                images_subpath=images_subpath,
                                labels_subpath=labels_subpath)
        val_weeks = scan_root(eval_root, bucket=bucket,
                              images_subpath=images_subpath,
                              labels_subpath=labels_subpath) if eval_root else []

    found = {"names": dict(names or {}), "source": "given", "looked_in": []}
    if not found["names"]:
        # The SOURCE bucket, not the destination. The class sources are
        # the folders the pull reads from, and a dataset's data.yaml sits
        # beside its images -- looking for it in the bucket the weeks are
        # written to finds nothing and reports "no names found" while the
        # file is sitting there in the other bucket.
        found = find_classes(list(class_sources), bucket=source_bucket or bucket)

    if not found["names"] and training_root:
        # Last resort: the yaml this root already has, in the DESTINATION
        # bucket. A previous build wrote it, and it is the record of the
        # names the label indices in these week folders were written
        # against -- which is exactly the question being asked.
        #
        # Without this the unattended path could not build at all on a
        # root whose sources carry no data.yaml of their own: the UI
        # supplies `names` from its editable field, and nothing supplies
        # them when there is no operator to type into it.
        found = find_classes([training_root], bucket=bucket)
        if found["names"]:
            found["source"] = found["source"] + " (this root's own)"

    # Paths are written relative to the prefix that will be downloaded,
    # so the yaml reads the same locally as it does in the bucket.
    base = _common_parent(training_root, eval_root) if eval_root else training_root
    def relative(path: str) -> str:
        return path[len(base) + 1:] if base and path.startswith(base + "/") else path

    document: dict = {"path": f"s3://{bucket}/{base}" if base else f"s3://{bucket}"}
    # Two forms, and Ultralytics reads both (`data/base.py`,
    # `get_img_files`): a directory is globbed whole, a FILE is read as a
    # list of image paths. The folder list trains on everything in each
    # week; the list form trains on exactly what the retention policy
    # chose, which is the only way the policy reaches the run at all.
    if retained_train:
        document["train"] = retained_train
        if retained_val:
            document["val"] = retained_val
    else:
        document["train"] = [relative(w["images_path"]) for w in train_weeks if w["images"]]
        if val_weeks:
            document["val"] = [relative(w["images_path"]) for w in val_weeks if w["images"]]
    if found["names"]:
        document["names"] = {int(k): v for k, v in sorted(found["names"].items())}

    problems: list = []
    if not document["train"]:
        problems.append({
            "level": "blocking",
            "text": f"No week folder under {training_root} holds any images.",
        })
    if not eval_root:
        problems.append({
            "level": "blocking",
            "text": "No evaluation root, so the yaml has no val split and "
                    "training cannot measure itself.",
        })
    elif not document.get("val"):
        where = (f"{root}/<week>/{images_subpath}/val" if root
                 else f"a week folder under {eval_root}")
        problems.append({
            "level": "blocking",
            "text": f"Nothing in {where}, so there is no held-out set and "
                    f"training cannot measure itself.",
        })
    if not found["names"]:
        problems.append({
            "level": "blocking",
            "text": "No data.yaml was found to read class names from. A label "
                    "line is only an index, so without names the dataset has "
                    "no meaning.",
        })
    for week in train_weeks + val_weeks:
        if week["unlabelled"]:
            problems.append({
                "level": "warn",
                "text": f"{week['week']} has {week['images']} images and no labels -- "
                        f"every one of them would train as a background example.",
            })
    # The trainer downloads one prefix. Saying which, and when that is
    # the whole bucket, is the difference between this being usable and
    # being a file that looks right and cannot be read.
    if eval_root and not base:
        problems.append({
            "level": "warn",
            "text": "The two roots share no parent folder, so the prefix holding "
                    "both is the whole bucket. Put them under one parent (for "
                    "example retraining/training and retraining/evaluation) to "
                    "make this a single small download.",
        })

    text = _render(document, training_root, eval_root, train_weeks, val_weeks, found)
    return {
        "yaml": text,
        "document": json.loads(json.dumps(document)),
        "download_prefix": f"s3://{bucket}/{base}/" if base else f"s3://{bucket}/",
        "write_to": f"{base}/data.yaml" if base else "data.yaml",
        "bucket": bucket,
        "train_weeks": train_weeks,
        "val_weeks": val_weeks,
        "train_images": sum(w["images"] for w in train_weeks),
        "val_images": sum(w["images"] for w in val_weeks),
        "classes": found,
        "problems": problems,
        "ready": not any(p["level"] == "blocking" for p in problems),
    }


def _describe(entry, weeks: list) -> str:
    """What a split says it is, in the header.

    A list of folders and a path to a file list are different claims: the
    first trains on everything in those folders, the second on exactly
    the images named. Printing "N week folders" for both would describe
    the wrong one half the time.
    """
    total = sum(w["images"] for w in weeks)
    if isinstance(entry, str) and entry:
        return f"file list at {entry}, chosen from {total} images"
    return f"{len(entry or [])} week folders, {total} images"


def _render(document: dict, training_root: str, eval_root: str,
            train_weeks: list, val_weeks: list, found: dict) -> str:
    """The yaml as text, with a header saying where it came from.

    Generated files get edited by hand eventually. The header is what
    tells whoever opens it in six weeks that editing it is pointless
    because the next build overwrites it.
    """
    body = _dump(document)
    stamp = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    header = [
        "# Generated by pull control. Regenerated on demand -- edits are lost.",
        f"# built     {stamp}",
        f"# train     {training_root}  ({_describe(document.get('train'), train_weeks)})",
    ]
    if eval_root:
        header.append(
            f"# val       {eval_root}  ({_describe(document.get('val'), val_weeks)})"
        )
    if found.get("source"):
        header.append(f"# names     read from {found['source']}")
    header.append(
        "#"
    )
    header.append(
        "# Labels are not listed: Ultralytics finds them by swapping"
    )
    header.append(
        "# /images/ for /labels/ in each path above -- per image, so a"
    )
    header.append(
        "# list may cross week folders and each entry still finds its own."
    )
    header.append(
        "# `path` is the S3 origin. The training container rewrites it to"
    )
    header.append(
        "# the local download directory, and resolves the lists against it."
    )
    return "\n".join(header) + "\n" + body


def save(*, key: str, text: str, bucket: str = "") -> dict:
    """Writes the yaml. The only call here that changes anything."""
    bucket = bucket or pc.destination_bucket()
    key = (key or "").strip().strip("/")
    if not key.endswith(".yaml") and not key.endswith(".yml"):
        raise ValueError("The file has to be a .yaml.")
    pc.s3.put_object(Bucket=bucket, Key=key, Body=text.encode("utf-8"),
                     ContentType="text/yaml")
    return {"bucket": bucket, "key": key, "uri": f"s3://{bucket}/{key}"}


# --- a yaml over folders picked by hand ---------------------------------------
#
# `build` above turns the PULL's week folders into a yaml. This does the
# same job for folders somebody chose themselves on the Assemble card:
# name them in place, copy nothing, write one data.yaml.
#
# The constraint that shapes all of it: Ultralytics cannot read s3://, so
# the trainer downloads ONE prefix to local disk and only then opens the
# yaml (see train.py's download_dataset_with_progress). Every folder the
# yaml names therefore has to live under that one prefix -- which is the
# common parent of everything picked. Folders under unrelated top-level
# prefixes cannot be expressed this way at all, and this says so rather
# than writing a yaml that fails fifteen minutes into a GPU run.

#: Written into the generated file so a later build can tell its own
#: output from a data.yaml somebody wrote by hand. Overwriting the former
#: is housekeeping; overwriting the latter is destroying their work.
IN_PLACE_MARKER = "# Generated by Assemble a dataset (yaml only, nothing copied)."


class _NoAliasDumper(yaml.SafeDumper):
    """Never write YAML anchors. Two keys holding equal lists were dumped
    as `train: &id001` / `val: *id001`, which reads as validation BEING
    training even where -- as in the evaluation-only yaml -- nothing trains
    from the file."""

    def ignore_aliases(self, data):
        return True


def _dump(document) -> str:
    return yaml.dump(document, Dumper=_NoAliasDumper, sort_keys=False,
                     default_flow_style=False)


def _is_path_list(entry: str) -> bool:
    """Is this yaml entry a .txt of image paths rather than a folder?

    Ultralytics takes either. `get_img_files` tests `is_dir()` and
    `is_file()` per entry, so one `train:` list may hold both -- which
    is what a composed dataset needs: picked folders named directly,
    and retention's selection named as the list it already writes.
    """
    return entry.strip().lower().endswith(".txt")


def _download_dir_for(entry: str) -> str:
    """The directory that has to be FETCHED for this entry to resolve.

    A folder is its own. A path list is the directory it SITS IN,
    because its lines are written `./<path>` and Ultralytics resolves
    those against the file's parent -- fetching the list alone would
    arrive with every line pointing at nothing.
    """
    entry = entry.strip().strip("/")
    if _is_path_list(entry):
        return entry.rsplit("/", 1)[0] if "/" in entry else ""
    return entry


def label_indices_used(prefix: str, *, bucket: str = "",
                       sample: int | None = 200) -> dict:
    """Which class indices the label files under `prefix` actually use.

    Read from the FILES, because they are the only place the answer
    exists. A data.yaml says what the indices were MEANT to be; the
    labels say what was written, and the two disagreed on this bucket
    for weeks without anything erroring: a yaml declaring `{0: helmet}`
    sat beside label files using 0, 1, 2 and 3.

    `sample` keeps a card preview to a couple of hundred GETs. The
    unattended chain passes None and reads them all -- it has the time,
    and a run that is about to spend GPU on the answer should not be
    guessing from a sample.
    """
    bucket = bucket or pc.destination_bucket()
    prefix = prefix.strip("/")
    keys = [o["Key"] for o in pc._list_objects(prefix + "/", bucket)
            if o["Key"].lower().endswith(".txt") and "/labels/" in f"/{o['Key']}"]
    total = len(keys)
    if sample and total > sample:
        step = total / sample
        keys = [keys[int(i * step)] for i in range(sample)]
    # IN PARALLEL. Each label file is a few hundred bytes behind one GET,
    # so the time is all round trips: read one after another, the
    # chain's full scan of this bucket's 8,455 files took 751 seconds --
    # twelve and a half minutes added to every unattended run. Sixteen
    # at a time brings that to under a minute. boto3 clients are
    # thread-safe; the per-file work is a handful of int() calls.
    def _indices_in(key: str):
        try:
            body = pc.s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        except Exception:                                      # noqa: BLE001
            return None
        found: set = set()
        for line in body.decode("utf-8", "replace").splitlines():
            parts = line.split()
            if not parts:
                continue
            try:
                found.add(int(float(parts[0])))
            except ValueError:
                continue
        return found

    used: set = set()
    unreadable = 0
    if keys:
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            for found in pool.map(_indices_in, keys):
                if found is None:
                    unreadable += 1
                else:
                    used |= found
    return {"indices": used, "sampled": len(keys), "total": total,
            "unreadable": unreadable}


def _labels_beyond(entries: list, names: dict, bucket: str, sample) -> list:
    """One sentence per source whose labels use a class the yaml would
    not declare.

    This is the check that was missing when `golden_set/images/val`
    arrived with labels 0-3 under a yaml naming one class: Ultralytics
    marked every one of its 20 images corrupt and refused to start --
    thirty minutes in, after the download. The same mismatch on the
    training side does not even error: the images are silently dropped
    and the run trains on whatever is left.

    Unknown is not broken. A listing that fails -- a bucket this process
    cannot read, a stubbed client in a test -- reports nothing rather
    than refusing every build.
    """
    declared = set(int(k) for k in names)
    problems: list = []  # one dict per offending source: entry, indices, text
    for entry in entries:
        where = (_download_dir_for(entry) if _is_path_list(entry)
                 else _labels_dir_for(entry))
        if not where:
            continue
        try:
            found = label_indices_used(where, bucket=bucket, sample=sample)
        except Exception:                                      # noqa: BLE001
            continue
        beyond = sorted(i for i in found["indices"] if i not in declared)
        if not beyond:
            continue
        scope = ("" if found["sampled"] >= found["total"]
                 else f" (in {found['sampled']} of {found['total']} label files read)")
        problems.append({
            "entry": entry,
            "indices": beyond,
            "sampled": found["sampled"],
            "total": found["total"],
            "text": (
                f"'{entry}': its label files use class "
                f"{', '.join(str(i) for i in beyond)}{scope}, and the yaml would "
                f"declare only {', '.join(f'{i}: {names[i]}' for i in sorted(names))}. "
                "Ultralytics drops EVERY image holding a box outside the declared "
                "range -- the whole image, not the box -- so most of this folder "
                "would never be trained on, and a validation folder like this "
                "fails outright. Either name the rest in Label classes on the "
                "Assemble card, or tick 'Filter undeclared classes' so the "
                "training container drops those boxes before Ultralytics reads "
                "them, leaving the label files in S3 unchanged."
            ),
        })
    return problems


def _overlaps(train_dirs: list, val_dirs: list) -> list:
    """Pairs where a validation folder IS, contains, or sits inside a
    training folder -- the same images on both sides. Folder-level, which
    is exact for directories; a path list is compared as the folder it
    resolves against."""
    def norm(d):
        return (_download_dir_for(d) if _is_path_list(d) else d).strip("/")
    pairs = []
    for t in train_dirs:
        for v in val_dirs:
            a, b = norm(t), norm(v)
            if a == b or a.startswith(b + "/") or b.startswith(a + "/"):
                pairs.append((t, v))
    return pairs


def _bk(d: str, home: str) -> tuple:
    """(bucket, key) for a folder named either as a key in the home bucket
    or as `s3://bucket/key` in any other."""
    d = str(d).strip()
    if d.startswith("s3://"):
        b, _, k = d[5:].partition("/")
        return b, k.strip("/")
    return home, d.strip("/")


def _qualify(d: str, home: str) -> str:
    """A folder as this module keys it: a bare key in the home bucket,
    `s3://bucket/key` anywhere else -- so the same folder is one string."""
    b, k = _bk(d, home)
    return k if b == home else f"s3://{b}/{k}"


def _remap_block(remap: dict, home: str, local) -> dict:
    """`x_label_remap` for a yaml: {local labels dir: {src index: dst}}.

    `local(bucket, key)` turns a folder into the path this yaml's reader
    will have it at, so the trainer and evaluator can apply the map to
    their downloaded copies without knowing how the yaml was built."""
    out = {}
    for d, mapping in (remap or {}).items():
        b, k = _bk(d, home)
        labels = _labels_dir_for(k)
        if labels:
            out[local(b, labels)] = {int(s): int(t) for s, t in mapping.items()}
    return out


#: Beside a source's data.yaml, a verbatim copy of it made the first time
#: Assemble REPLACES it (`write_in_place` with "replace it" ticked). The
#: source's own yaml is the only record of what its label numbers mean;
#: once a generated yaml stands in its place, that record would otherwise
#: be gone, and every later build would read the label files with the
#: wrong class names -- valid indices, so nothing would ever error.
#: Code review 2026-10-26, F1.
SOURCE_TAXONOMY_FILE = "data.source.yaml"


def _read_text(bucket: str, key: str) -> str | None:
    """An object's text, or None when it is not there."""
    try:
        return pc.s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8", "replace")
    except Exception:                                              # noqa: BLE001
        return None


def _names_in(text: str | None) -> dict | None:
    """`names:` of a yaml as {index: name}, or None when it has none."""
    if not text:
        return None
    try:
        raw = (yaml.safe_load(text) or {}).get("names") or {}
    except Exception:                                              # noqa: BLE001
        return None
    if isinstance(raw, (list, tuple)):
        raw = dict(enumerate(raw))
    return {int(i): str(n) for i, n in raw.items()} or None


def _list(names: dict) -> str:
    return ", ".join(f"{i} {names[i]}" for i in sorted(names)) or "nothing"


def _source_taxonomy(bucket: str, root: str, names: dict) -> tuple:
    """What a source's label numbers MEAN: ({index: name or None}, problem).

    In order of evidence:
      1. its own data.yaml, when this module did not write it;
      2. the copy kept beside it when this module replaced it
         (SOURCE_TAXONOMY_FILE);
      3. a generated yaml from before copies were kept: its own
         `x_label_remap` says how source numbers were translated onto its
         `names:`, which recovers the original -- a class that remap
         dropped (-1) has no recoverable name and is None;
      4. a generated yaml with no remap: the labels were taken to be in
         its own `names:` order. That is fine while the requested order is
         the same; if it changed, the meaning of the numbers is unknown,
         and that is REFUSED rather than guessed.
    ({}, "") when there is no data.yaml at all -- nothing to remap, as
    before (week folders, a dataset that arrived without one).
    """
    text = _read_text(bucket, f"{root}/data.yaml")
    if text is None:
        return {}, ""
    if IN_PLACE_MARKER not in text:
        return _names_in(text) or {}, ""
    kept = _names_in(_read_text(bucket, f"{root}/{SOURCE_TAXONOMY_FILE}"))
    if kept:
        return kept, ""
    try:
        doc = yaml.safe_load(text) or {}
    except Exception:                                              # noqa: BLE001
        doc = {}
    ours = _names_in(_dump({"names": doc.get("names") or {}})) or {}
    maps = {tuple(sorted((int(a), int(b)) for a, b in (m or {}).items()))
            for m in (doc.get("x_label_remap") or {}).values()}
    if len(maps) == 1:
        (only,) = maps
        return {src: (ours.get(dst) if dst >= 0 else None) for src, dst in only}, ""
    if not maps and ours == names:
        return {}, ""
    return None, (
        f"'{root}/data.yaml' was written by Assemble, and the source's own class list "
        "was not kept, so this cannot tell what that dataset's label numbers mean "
        + ("(it remapped its folders in more than one way). " if maps else
           f"(it named {_list(ours)}, and the classes asked for now are {_list(names)}). ")
        + f"Restore the dataset's original data.yaml at '{root}/data.yaml', or, if the "
        "label files really are numbered in the order asked for now, delete that "
        "generated data.yaml and build again.")


def _source_remaps(dirs: list, names: dict, home: str, drop: bool) -> tuple:
    """Per-source class-ID maps, for sources that arrived with their own
    numbering.

    A dataset whose labels are numbered in another order (see
    `_source_taxonomy` for where that is read from) is REMAPPED by name
    onto the declared list, in the run's downloaded copy only -- S3 is
    never touched. A class it has that the declared list lacks is dropped
    when the filter is ticked, and refused otherwise, naming the class and
    the two fixes. Returns ({qualified folder: {src: dst}}, [problems])."""
    remaps, problems, seen = {}, [], {}
    by_name = {v: k for k, v in names.items()}
    for d in dirs:
        b, k = _bk(d, home)
        parts = k.split("/")
        if "images" not in parts:
            continue
        root = "/".join(parts[:len(parts) - 1 - parts[::-1].index("images")])
        if not root:
            continue
        if (b, root) not in seen:
            seen[(b, root)] = _source_taxonomy(b, root, names)
            if seen[(b, root)][1]:
                problems.append(seen[(b, root)][1])
        theirs, trouble = seen[(b, root)]
        if trouble or not theirs:
            continue
        mapping, missing = {}, []
        for i, n in theirs.items():
            if n is not None and n in by_name:
                mapping[i] = by_name[n]
            elif drop:
                mapping[i] = -1
            else:
                missing.append(f"{i} '{n}'" if n is not None
                               else f"{i} (dropped by an earlier build; its name was not kept)")
        if missing:
            problems.append(
                f"'{root}/data.yaml' numbers its classes differently and has "
                f"{', '.join(missing)}, which the yaml does not name. Add "
                f"{'it' if len(missing) == 1 else 'them'} to Label classes, or tick "
                "Filter undeclared classes to drop those boxes.")
            continue
        if any(i != j for i, j in mapping.items()):
            remaps[_qualify(d, home)] = mapping
    return remaps, problems


def _foreign_taxonomy_conflicts(dirs: list, names: dict, bucket: str) -> list:
    """A source whose OWN data.yaml numbers its classes differently.

    Every source shares one `names:` list, so label index 1 must mean the
    same thing in every folder. A dataset that arrived with its own
    data.yaml says what its indices mean; if that disagrees with the
    declared list at any shared index, merging them would train boxes
    under the wrong class with nothing erroring. Yaml files this module
    wrote (they carry IN_PLACE_MARKER) are skipped -- they are derived
    from the declared list, not evidence about the labels.
    """
    problems = []
    seen = set()
    for d in dirs:
        parts = d.strip("/").split("/")
        if "images" not in parts:
            continue
        root = "/".join(parts[:len(parts) - 1 - parts[::-1].index("images")])
        if not root or root in seen:
            continue
        seen.add(root)
        try:
            text = pc.s3.get_object(Bucket=bucket, Key=f"{root}/data.yaml")["Body"].read().decode("utf-8", "replace")
        except Exception:                                          # noqa: BLE001
            continue
        if IN_PLACE_MARKER in text:
            continue
        try:
            theirs = (yaml.safe_load(text) or {}).get("names") or {}
        except Exception:                                          # noqa: BLE001
            continue
        if isinstance(theirs, (list, tuple)):
            theirs = dict(enumerate(theirs))
        theirs = {int(k): str(v) for k, v in theirs.items()}
        clash = [i for i in theirs if i in names and theirs[i] != names[i]]
        if clash:
            problems.append(
                f"'{root}/data.yaml' numbers its classes differently: "
                + ", ".join(f"{i} is '{theirs[i]}' there but '{names[i]}' here" for i in clash)
                + ". Merged as-is, those boxes would train under the wrong class. "
                "Set Label classes to that dataset's own order, or assemble it separately."
            )
    return problems


def _labels_dir_for(images_dir: str) -> str:
    """Where Ultralytics will look for the labels of images in this folder.

    Not a guess. Ultralytics finds a label by swapping the LAST `/images/`
    in the image's path for `/labels/` -- so the label folder is not a
    choice, it is a consequence of where the images are. A folder picked
    as "labels" that is not this one would be ignored in silence, and
    every image would train as a background example with no annotations:
    a run that succeeds, reports plausible-looking metrics and has
    learned nothing.
    """
    parts = images_dir.strip("/").split("/")
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            return "/".join(parts[:i] + ["labels"] + parts[i + 1:])
    return ""


def _parent_of_all(keys: list) -> str:
    """The deepest prefix every one of these shares, on a / boundary.

    Deliberately NOT `_common_parent` above, which this first shadowed by
    accident: that one takes varargs and returns the shared prefix as-is.
    This one also steps UP when every source is the same folder, because
    the yaml has to sit at the root of what is downloaded and so cannot
    sit inside the images directory it is describing.
    """
    if not keys:
        return ""
    split = [k.strip("/").split("/") for k in keys]
    shared: list = []
    for parts in zip(*split):
        if len(set(parts)) != 1:
            break
        shared.append(parts[0])
    # Never return a folder as the parent of itself: if every source IS
    # the same folder, the parent is one level up, so the yaml sits beside
    # it rather than inside the images directory.
    if shared and len(shared) == min(len(p) for p in split):
        shared = shared[:-1]
    return "/".join(shared)


def _existing_yaml(bucket: str, key: str) -> dict:
    """Whether a data.yaml is already there, and whether we wrote it."""
    try:
        body = pc.s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception:                                              # noqa: BLE001
        return {}
    text = body.decode("utf-8", "replace")
    return {"bytes": len(body), "ours": IN_PLACE_MARKER in text}


def _document(base: str, *, train: list, val: list, names: dict,
              bucket: str, role: str, remap: dict | None = None) -> dict:
    """One data.yaml for one downloaded prefix.

    `train` and `val` are ABSOLUTE keys; they are made relative to `base`
    here, because the trainer sets `path` to wherever it put the download
    and Ultralytics resolves a relative list against that.
    """
    rel_train = [k[len(base) + 1:] for k in train]
    rel_val = [k[len(base) + 1:] for k in val] if val else rel_train
    block = _remap_block(remap, bucket, lambda b, k: k[len(base) + 1:])
    document: dict = {
        "path": f"s3://{bucket}/{base}",
        "train": rel_train,
        # Ultralytics needs a `val` key. When the real validation set is
        # under a DIFFERENT prefix it is not on this container's disk, so
        # naming it here would fail the run -- the training job validates
        # against its own images and the held-out numbers come from the
        # evaluation step, which is where the registered metrics come
        # from anyway.
        "val": rel_val,
        "nc": len(names),
        "names": {i: names[i] for i in sorted(names)},
    }
    if block:
        # Read by train.py / evaluate_any.py, ignored by Ultralytics.
        document["x_label_remap"] = block
    stamp = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    text = "\n".join([
        IN_PLACE_MARKER,
        f"# {role} -- nothing copied, these folders are named where they are",
        f"# built     {stamp}",
        f"# download  s3://{bucket}/{base}/",
        "",
        _dump(document),
    ])
    key = f"{base}/data.yaml"
    return {"folder": base, "key": key, "uri": f"s3://{bucket}/{key}",
            "train": rel_train, "val": rel_val, "text": text,
            "existing": _existing_yaml(bucket, key)}


_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def _image_counts(train_dirs: list, val_dirs: list, home: str) -> dict:
    """Images per folder the yaml names, per split -- what the card shows.

    Counted by image extension, so a folder's label files are not added in
    (a "55 objects" that was 55 images, beside a validation set that was
    really 1,748, is what this replaced). A path list counts its lines.
    Unknown is shown as None rather than as zero."""
    out = {}
    for split, dirs in (("train", train_dirs), ("val", val_dirs)):
        rows = []
        for d in dirs:
            b, k = _bk(d, home)
            try:
                if _is_path_list(k):
                    body = pc.s3.get_object(Bucket=b, Key=k)["Body"].read().decode("utf-8", "replace")
                    n = sum(1 for line in body.splitlines() if line.strip())
                else:
                    n = sum(1 for o in pc._list_objects(k.rstrip("/") + "/", b)
                            if o["Key"].lower().endswith(_IMAGE_EXTS))
            except Exception:                                      # noqa: BLE001
                n = None
            rows.append({"folder": d, "images": n})
        out[split] = {"folders": rows,
                      "total": None if any(r["images"] is None for r in rows)
                      else sum(r["images"] for r in rows)}
    return out


def _download_cost(base: str, absolute_dirs: list, bucket: str) -> dict:
    """How much of that prefix the run pays for and does not train on.

    Takes ABSOLUTE keys. Passing the relative ones compared `a/images/`
    against `jp/a/images/1.jpg` and counted every object as waste.
    """
    wanted = {d.rstrip("/") + "/" for d in absolute_dirs}
    inside = outside = 0
    inside_bytes = outside_bytes = 0
    try:
        for obj in pc._list_objects(base + "/", bucket):
            size = int(obj.get("Size") or 0)
            if any(obj["Key"].startswith(d) for d in wanted):
                inside += 1
                inside_bytes += size
            else:
                outside += 1
                outside_bytes += size
    except Exception as exc:                                       # noqa: BLE001
        print(f"Could not size s3://{bucket}/{base}/: {exc}")
    return {"prefix": base, "in_named_folders": inside,
            "in_named_bytes": inside_bytes, "extra_objects": outside,
            "extra_bytes": outside_bytes}


def _images_and_labels(source) -> tuple:
    """One picked folder pair, however the caller spells it.

    `routers/track2_models._splits_of` is the ONE place the card's
    request is turned into splits, and it builds `(images, labels)`
    tuples because `assemble_dataset` and `inspect_dataset_layout` --
    the two older consumers -- take tuples. This one took dicts, so
    every press of "Preview the yaml" and "Write it" in multi-source
    mode died on `'tuple' object has no attribute 'get'` and came back
    as a 400 the operator read as the button doing nothing.

    Taking both here rather than changing `_splits_of` keeps one
    producer: a second splits-builder for this consumer alone is how the
    two drift apart again.
    """
    if isinstance(source, dict):
        images, labels = source.get("images"), source.get("labels")
    else:
        images, labels = (list(source) + ["", ""])[:2]
    return (str(images or "").strip().strip("/"),
            str(labels or "").strip().strip("/"))


def _many_roots_document(*, train_dirs: list, val_dirs: list, bases: list,
                         names: dict, bucket: str, overwrite: bool,
                         remap: dict | None = None) -> dict:
    """One yaml for a run that fetches SEVERAL prefixes -- in one bucket
    or several.

    Entries are full key paths, which is how `train.py` lays files out
    once it fetches more than one prefix. Across SEVERAL BUCKETS the same
    key could exist twice, so every entry is also prefixed with its bucket
    (`bucket/key`) and the trainer lays files out the same way.

    The yaml lives under `_generated/<hash>/` in the home bucket. The
    evaluation step takes one prefix, so a second, evaluation-only yaml is
    written at the val folders' primary prefix, naming the others for the
    evaluator to fetch itself.

    The evaluation variables are named apart from the training ones. They
    once shared `text`, and the training yaml returned was the EVALUATION
    yaml -- `train:` = the val folders -- whenever val spanned two roots.
    """
    home = bucket
    tq = [_bk(d, home) for d in train_dirs]
    vq = [_bk(d, home) for d in val_dirs]
    multi = len({b for b, _ in tq + vq}) > 1

    groups: dict = {}
    for b, k in tq + vq:
        groups.setdefault(b, []).append(k)
    prefixes = [f"s3://{b}/{base}/" for b in groups for base in _bases_for(groups[b])]
    # Entries in the layout the trainer will ACTUALLY produce for these
    # downloads -- see `local_layout`.
    local = local_layout(prefixes)

    digest = hashlib.sha1(
        "\n".join(sorted(map(str, train_dirs)) + ["--"] + sorted(map(str, val_dirs))).encode("utf-8")
    ).hexdigest()[:12]
    folder = f"_generated/{digest}"
    train_local = [local(b, k) for b, k in tq]
    val_local = [local(b, k) for b, k in vq]
    document = {
        "path": f"s3://{home}",
        "train": list(train_local),
        "val": list(val_local) if val_local else list(train_local),
        "nc": len(names),
        "names": {index: names[index] for index in sorted(names)},
    }
    block = _remap_block(remap, home, local)
    if block:
        document["x_label_remap"] = block
    stamp = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    text = "\n".join([
        IN_PLACE_MARKER,
        "# several prefixes -- nothing copied, these folders are named where they are",
        *(["# several buckets -- every entry is <bucket>/<key>"] if multi else
          ["# one prefix -- entries are relative to it"] if len(prefixes) == 1 else []),
        f"# built     {stamp}",
        "# download  " + ", ".join(prefixes),
        "",
        _dump(document),
    ])
    key = f"{folder}/data.yaml"
    existing = _existing_yaml(home, key)

    # ---- the evaluation side: one prefix plus what the evaluator fetches ----
    # THE EVALUATION STEP TAKES ONE PREFIX. A SageMaker ProcessingInput is
    # one S3 URI; given the trainer's comma-separated list it failed at
    # CreateProcessingJob with "No S3 objects found" after a full run.
    eval_side = None
    if vq:
        vgroups: dict = {}
        for b, k in vq:
            vgroups.setdefault(b, []).append(k)
        candidates = [(b, base) for b in vgroups for base in _bases_for(vgroups[b])]
        pb, pbase = next(((b, base) for b, base in candidates
                          if any(b == vb and (vk == base or vk.startswith(base + "/"))
                                 and "golden" in vk for vb, vk in vq)),
                         candidates[0])
        inside = [(b, k) for b, k in vq if b == pb and (k == pbase or k.startswith(pbase + "/"))]
        outside = [(b, k) for b, k in vq if (b, k) not in inside]

        def elocal(b, k):
            if b == pb and (k == pbase or k.startswith(pbase + "/")):
                return k[len(pbase) + 1:]
            return k if b == pb else f"{b}/{k}"

        entries = [elocal(b, k) for b, k in inside + outside]
        edoc = {
            "path": f"s3://{pb}/{pbase}",
            "train": list(entries),
            "val": list(entries),
            "nc": len(names),
            "names": {i: names[i] for i in sorted(names)},
        }
        eblock = _remap_block(remap, home, elocal)
        if eblock:
            edoc["x_label_remap"] = eblock
        etext = "\n".join([
            IN_PLACE_MARKER,
            "# EVALUATION ONLY -- nothing trains from this file. `train:` is required",
            "# by Ultralytics and repeats `val:`; it is not a training set.",
            "# `path:` is replaced by the evaluator's local download directory.",
            "# Entries under it are relative; other folders are fetched by the",
            "# evaluator to <local dir>/<key>, or <local dir>/<bucket>/<key> when",
            "# they are in another bucket.",
            f"# built     {stamp}",
            f"# download  s3://{pb}/{pbase}/" + (" + " + ", ".join(f"s3://{b}/{k}" for b, k in outside) if outside else ""),
            "",
            _dump(edoc),
        ])
        extra = []
        for b, k in outside:
            extra.append(f"s3://{b}/{k}/")
            lab = _labels_dir_for(k)
            if lab:
                extra.append(f"s3://{b}/{lab}/")
        ekey = f"{pbase}/data.yaml"
        eval_side = {
            # A folder in another bucket travels as a full URI.
            "folder": pbase if pb == home else f"s3://{pb}/{pbase}",
            "bucket": pb, "key": ekey, "uri": f"s3://{pb}/{ekey}",
            "train": entries, "val": entries, "text": etext,
            "existing": _existing_yaml(pb, ekey),
            "download": _download_cost(pbase, [k for _, k in inside], pb),
            "extra_uris": extra,
        }

    needs = [x for x in (existing, (eval_side or {}).get("existing"))
             if x and not x.get("ours")]
    return {
        "folder": folder,
        "key": key,
        "uri": f"s3://{home}/{key}",
        "bucket": home,
        "download_prefixes": prefixes,
        "download_prefix": ",".join(prefixes),
        "train": train_local,
        "val": val_local if val_local else train_local,
        "names": {index: names[index] for index in sorted(names)},
        "text": text,
        "existing": existing,
        "eval": eval_side,
        "download": {"prefix": ", ".join(prefixes), "in_named_folders": 0,
                     "in_named_bytes": 0, "extra_objects": 0, "extra_bytes": 0},
        "needs_overwrite": bool(needs and not overwrite),
        "many_roots": True,
        "multi_bucket": multi,
    }


def local_layout(prefixes: list):
    """`(bucket, key) -> path` on the trainer's disk, for these downloads.

    THE rule `train.py download_dataset_with_progress` lays files out by,
    stated once so the yaml, the remap block and the snapshot agree with it:

      * ONE prefix  -- stripped: `<base>/images/train/x.jpg` lands at
                       `images/train/x.jpg`;
      * several prefixes in ONE bucket -- full keys;
      * several BUCKETS -- `<bucket>/<key>`, for every file, home included.

    Decided by the download list, never by whether a folder happens to be
    in another bucket: one foreign prefix is still one prefix, and the
    trainer strips it (code review 2026-10-26, F5).
    """
    parsed = []
    for u in prefixes:
        u = str(u)
        if u.startswith("s3://"):
            b, _, k = u[5:].partition("/")
        else:
            b, k = "", u
        parsed.append((b, k.strip("/")))
    many = len(parsed) > 1
    multi = len({b for b, _ in parsed}) > 1

    def local(b: str, k: str) -> str:
        k = k.strip("/")
        if multi:
            return f"{b}/{k}"
        if many:
            return k
        base = parsed[0][1] if parsed else ""
        if not base:
            return k
        if k == base:
            return "."
        return k[len(base) + 1:] if k.startswith(base + "/") else k
    return local


def _bases_for(dirs: list) -> list:
    """The minimal set of prefixes that covers these folders AND their
    labels siblings.

    One entry for the ordinary case -- folders under a single dataset
    root -- and that is the path everything already takes. Several when
    the folders live under unrelated roots: a dataset under
    `Testing_Dataset/`, a golden set under `golden_set/`, week folders
    under `retraining/`. There is no common parent short of the bucket,
    and `_base_or_refuse` used to refuse exactly that.

    Refusing was right while `train.py` downloaded ONE prefix. It takes
    a comma-separated list now, so the yaml can name folders anywhere
    and this returns what the run has to fetch.
    """
    if not dirs:
        return []
    # A path list resolves against the folder it sits in, so that
    # folder is what has to be downloaded, not the file.
    dirs = [d for d in (_download_dir_for(i) for i in dirs) if d]
    if not dirs:
        return []

    # THE COMMON PARENT FIRST. `jp/a/images` and `jp/b/images` share
    # `jp`, and that is one download -- taking each folder's own base
    # would split a perfectly ordinary dataset into two prefixes and
    # write the yaml in the scattered form for no reason.
    labels_too = [d for d in (_labels_dir_for(i) for i in dirs) if d]
    shared = _parent_of_all(list(dirs) + labels_too)
    if shared:
        return [shared]

    bases: list = []
    for images in dirs:
        labels = _labels_dir_for(images)
        base = _parent_of_all([images] + ([labels] if labels else []))
        if not base:
            # Nothing above it but the bucket. Use the folder itself
            # rather than downloading everything.
            base = images.strip("/")
        bases.append(base)

    # Drop any base another one already covers, so a dataset root and a
    # folder inside it do not become two overlapping downloads.
    minimal: list = []
    for base in sorted(set(bases), key=len):
        if not any(base == kept or base.startswith(kept + "/") for kept in minimal):
            minimal.append(base)
    return minimal


def _base_or_refuse(dirs: list, role: str) -> str:
    # The labels siblings count toward the base too. This prefix is
    # downloaded WHOLE and nothing outside it reaches the container, so a
    # base covering `<root>/images/train` but not `<root>/labels/train`
    # names folders whose annotations never arrive -- every image then
    # trains as a background example, and the run succeeds and reports
    # plausible metrics having learned nothing.
    #
    # Picking `images/train` and `images/val` off one dataset root -- the
    # ordinary Ultralytics layout, and the first thing anybody tries --
    # did exactly that: both sat under `<root>/images`, so `<root>/labels`
    # fell outside the download.
    dirs = [d for d in (_download_dir_for(k) for k in dirs) if d]
    labels = [d for d in (_labels_dir_for(k) for k in dirs) if d]
    base = _parent_of_all(dirs + labels)
    if base:
        return base
    roots = sorted({k.split("/")[0] for k in dirs})
    raise ValueError(
        f"The {role} folders share no parent in the bucket -- "
        + ", ".join(f"'{r}'" for r in roots)
        + ". Each side is downloaded as ONE prefix, so the folders within a side "
        "have to sit under a common one. Training and validation do NOT have to "
        "be near each other -- those are separate downloads."
    )


def build_in_place(*, splits: dict, classes: dict, bucket: str = "",
                   overwrite: bool = False, label_scan: int | None = 200,
                   drop_undeclared: bool = False) -> dict:
    """A data.yaml per DOWNLOAD, naming folders where they already are.

    Training and validation are two separate downloads -- the training
    job takes `TRAIN_DATA_S3_URI`, the evaluation step takes
    `EVAL_DATA_S3_URI` and runs in a different container. So the
    validation set can live anywhere in the bucket. An earlier version of
    this demanded both sides share one parent, which nothing required.

    What DOES have to share a parent is the folders WITHIN one side,
    because that side is one prefix and the trainer downloads it whole.

    When both sides land on the same prefix -- the ordinary case of one
    root split train/val -- there is one download and one file, with both
    splits in it.

    Writes nothing. `write_in_place` is what puts it in S3.
    """
    bucket = bucket or pc.destination_bucket()
    names = {int(k): str(v) for k, v in (classes or {}).items()}
    if not names:
        raise ValueError("No class names, so the yaml would declare nothing.")

    chosen: dict = {}
    problems: list = []
    for split in ("train", "val"):
        dirs = []
        for source in splits.get(split) or []:
            images, given = _images_and_labels(source)
            if not images:
                continue
            if _is_path_list(images):
                # A path list names its images line by line, and each
                # line's labels are derived from that line. There is no
                # folder here to swap `/images/` in, and nothing to
                # check -- `retained_lists.write` already counts its
                # own lines back against what the policy chose.
                dirs.append(images)
                continue
            expected = _labels_dir_for(images)
            if not expected:
                problems.append(
                    f"'{images}' has no 'images' folder in its path, so Ultralytics "
                    "has no way to find its labels -- it finds them by swapping "
                    "'/images/' for '/labels/'."
                )
                continue
            if given and given != expected:
                problems.append(
                    f"labels for '{images}' would be read from '{expected}', not the "
                    f"'{given}' that was picked. Ultralytics derives the label path "
                    "from the image path and cannot be told otherwise."
                )
                continue
            dirs.append(images)
        chosen[split] = dirs

    # Problems FIRST. A rejected source is also a missing one, so checking
    # for "no training folders" ahead of this answered every bad labels
    # path with "there is nothing to name" -- true, unhelpful, and it hid
    # the sentence that said what to fix.
    if problems:
        raise ValueError(" ".join(problems))
    if not chosen.get("train"):
        raise ValueError("No training folders, so there is nothing to name.")

    # One spelling per folder: a bare key in the home bucket, s3://... for
    # any other. Several buckets go the several-prefix way below.
    chosen = {k: [_qualify(d, bucket) for d in v] for k, v in chosen.items()}
    train_dirs, val_dirs = chosen["train"], chosen["val"]
    multi_bucket = any(str(d).startswith("s3://") for d in train_dirs + val_dirs)

    # TRAIN AND VAL MUST NOT OVERLAP. Refused on preview and write alike:
    # a validation score on training images is not a score.
    overlap = _overlaps(train_dirs, val_dirs)
    if overlap:
        raise ValueError(
            "Training and validation overlap -- the same images would be on "
            "both sides: " + "; ".join(f"'{t}' and '{v}'" for t, v in overlap)
            + ". Pick validation folders that are separate from training."
        )
    # A source's own data.yaml numbering its classes differently.
    # Sources that number their classes differently are REMAPPED by name
    # (in the run's copy); a class the declared list lacks is dropped when
    # the filter is on, refused otherwise.
    remap, conflicts = _source_remaps(train_dirs + val_dirs, names, bucket, drop_undeclared)
    if conflicts:
        raise ValueError(" ".join(conflicts))
    # NO VALIDATION FOLDER is reported here and refused by the write --
    # the builder used to reuse the training folders as val.
    no_val = not val_dirs

    # THE LABELS HAVE TO FIT THE NAMES, and the names used to be the Model
    # Name's list whatever the label files said. `label_scan` is how many
    # label files per source to read: a sample for a card preview, None
    # for the chain, 0 to skip.
    #
    # Reported, not raised, from here: the card needs the indices to offer
    # "name the rest", and a preview that died with a sentence could not
    # carry them. `write_in_place` is what refuses -- unless the operator
    # has ticked the filter, in which case the training container drops
    # those boxes itself and the mismatch is a fact to record, not a fault.
    undeclared: list = []
    if label_scan != 0:
        # Per bucket, and not for folders being remapped -- their indices
        # are the source's own and are translated before Ultralytics reads.
        groups: dict = {}
        for d in train_dirs + val_dirs:
            if d in remap:
                continue
            b, k = _bk(d, bucket)
            groups.setdefault(b, []).append(k)
        for b, keys in groups.items():
            found = _labels_beyond(keys, names, b, label_scan)
            for item in found:
                if b != bucket:
                    item["entry"] = f"s3://{b}/{item['entry']}"
            undeclared += found

    # SEVERAL UNRELATED ROOTS, or the ordinary single one.
    #
    # `train.py` takes a comma-separated list of prefixes now, so the
    # yaml may name folders that share no parent -- a dataset under
    # `Testing_Dataset/`, a golden set under `golden_set/`, week folders
    # under `retraining/`. When that happens the entries are written as
    # FULL key paths, because the downloader keeps full paths in that
    # mode; stripping would collide the trees.
    #
    # The single-root path below is untouched. Every data.yaml this
    # system has written says `images/train`, and that only resolves
    # against a stripped single prefix.
    train_bases = [] if multi_bucket else _bases_for(train_dirs)
    val_bases = [] if multi_bucket or not val_dirs else _bases_for(val_dirs)
    # Train and val under DIFFERENT prefixes go the several-prefix way
    # too. The single-root builder used to write the training yaml with
    # `val:` = the training folders (the real val was a separate download
    # the trainer never saw), so best.pt was picked on training images.
    # Downloading both lets training validate on the real val folders.
    split_apart = (not multi_bucket and bool(val_dirs)
                   and len(_bases_for(train_dirs + val_dirs)) > 1)
    if multi_bucket or len(train_bases) > 1 or len(val_bases) > 1 or split_apart:
        many = _many_roots_document(
            train_dirs=train_dirs, val_dirs=val_dirs,
            bases=[] if multi_bucket else _bases_for(train_dirs + val_dirs),
            names=names, bucket=bucket, overwrite=overwrite, remap=remap)
        many["undeclared"] = undeclared
        many["drop_undeclared"] = bool(drop_undeclared)
        many["no_val"] = no_val
        many["image_counts"] = _image_counts(train_dirs, val_dirs, bucket)
        return many

    # ONE download covers both sides here -- `split_apart` above sent every
    # case needing two to the several-prefix builder. So the yaml is
    # rendered against THAT prefix, both splits relative to it.
    #
    # It used to compare each side's own base for containment, which
    # missed SIBLINGS of one ancestor: train under `jp/a`, val under
    # `jp/b`, neither covering the other although `jp` covers both. That
    # fell through to a training yaml with no val, which `_document` then
    # filled with the TRAINING folders -- the trainer refused the overlap,
    # after the download (code review 2026-10-26, F4). Rendering against
    # the shared prefix keeps the real validation folders, and the remap.
    if val_dirs:
        combined = _bases_for(train_dirs + val_dirs)
        if len(combined) != 1:
            # Unreachable while `split_apart` routes these above. Refused
            # rather than falling back to validating on training images.
            raise ValueError(
                "Training and validation need more than one download, and this "
                "path takes one. Nothing was built.")
        main = _document(combined[0], train=train_dirs, val=val_dirs,
                         names=names, bucket=bucket,
                         role="training and validation", remap=remap)
    else:
        main = _document(_base_or_refuse(train_dirs, "training"), train=train_dirs,
                         val=[], names=names, bucket=bucket, role="training",
                         remap=remap)
    eval_side = None

    sides = [s for s in (main, eval_side) if s]
    needs = [s for s in sides if s["existing"] and not s["existing"].get("ours")]
    return {
        "folder": main["folder"],
        "key": main["key"],
        "uri": main["uri"],
        "bucket": bucket,
        "download_prefix": f"s3://{bucket}/{main['folder']}/",
        "train": main["train"],
        "val": main["val"],
        "names": {i: names[i] for i in sorted(names)},
        "text": main["text"],
        "existing": main["existing"],
        # The second download, when validation lives under another prefix.
        # None when one prefix covers both.
        "eval": None,
        "download": _download_cost(main["folder"], train_dirs + val_dirs, bucket),
        "needs_overwrite": bool(needs and not overwrite),
        # Sources whose label files use a class the yaml does not name,
        # and whether the run has been told to drop those boxes itself.
        "undeclared": undeclared,
        "drop_undeclared": bool(drop_undeclared),
        "no_val": no_val,
        "image_counts": _image_counts(train_dirs, val_dirs, bucket),
    }


def write_in_place(*, splits: dict, classes: dict, bucket: str = "",
                   overwrite: bool = False, label_scan: int | None = 200,
                   drop_undeclared: bool = False) -> dict:
    """Build both sides, then write them.

    Rebuilt here rather than taking the text the card is showing: a yaml
    saved from a stale preview would name folders that have since
    changed, which is the same trap `save_dataset_yaml` above avoids.
    """
    out = build_in_place(splits=splits, classes=classes, bucket=bucket,
                         overwrite=overwrite, label_scan=label_scan,
                         drop_undeclared=drop_undeclared)
    # Labels the yaml cannot name stop the WRITE -- the preview already
    # showed them -- unless the filter is on, in which case the training
    # container drops them and the files in S3 stay as they are.
    if out.get("undeclared") and not drop_undeclared:
        raise ValueError(" ".join(p["text"] for p in out["undeclared"]))
    if out["needs_overwrite"]:
        raise ValueError(
            "A data.yaml already exists that this card did not write. "
            "Overwriting it would destroy one somebody wrote by hand. "
            "Tick 'replace it' if that is what you mean to do."
        )
    # No validation folder: refused, not filled in with the training
    # folders. Decided 2026-10-20 -- reuse was train/val overlap.
    if out.get("no_val"):
        raise ValueError(
            "No validation folder is selected, so the run would validate on "
            "the images it trains on. Pick evaluation folders on the "
            "Assemble card, or tick Use Pick Golden Set / Edge Cases."
        )

    # KEEP THE SOURCE'S OWN YAML before it is replaced: it is the only
    # record of what that dataset's label numbers mean (F1). Copied
    # verbatim, once -- an existing copy is never overwritten, because the
    # yaml standing there now may already be a generated one. All copies
    # first, so a failure here leaves every original in place.
    for side in (out, out.get("eval")):
        if not side:
            continue
        b = side.get("bucket") or out["bucket"]
        existing = side.get("existing") or {}
        if existing and not existing.get("ours") and side["key"].endswith("/data.yaml"):
            keep = side["key"][:-len("data.yaml")] + SOURCE_TAXONOMY_FILE
            if _read_text(b, keep) is None:
                original = pc.s3.get_object(Bucket=b, Key=side["key"])["Body"].read()
                pc.s3.put_object(Bucket=b, Key=keep, Body=original,
                                 ContentType="application/x-yaml")

    written = []
    for side in (out, out.get("eval")):
        if not side:
            continue
        pc.s3.put_object(Bucket=side.get("bucket") or out["bucket"], Key=side["key"],
                         Body=side["text"].encode("utf-8"),
                         ContentType="application/x-yaml")
        written.append(side["uri"])
    out["written"] = True
    out["written_uris"] = written
    return out
