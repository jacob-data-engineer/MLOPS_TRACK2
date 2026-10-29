"""Datasets out of folders: assembling, the source map, class filtering
and what a data.yaml declares (2026-10-28).

Moved out of `training/track2_shared.py` with the data/training split.
Nothing here knows about detectors, runs or the registry: a detector
arrives as an argument (`class_filter_for`) and the answers go back as
plain data. The training side reaches all of it through
`ppe_pipeline_api.data`.
"""

from __future__ import annotations

import concurrent.futures
import json
from datetime import datetime, timezone

from ..storage import BUCKET, IMAGE_EXTENSIONS, _list_keys, _yaml, list_s3_folders, s3


FILTERED_DATASET_MARKER = ".filtered-from.json"


ASSEMBLED_DATASET_PREFIX = "data/assembled"


def _basenames(prefix: str, extensions: tuple) -> dict:
    """{stem: key} for files under a prefix with one of the extensions.

    Keyed on the stem because that is what pairs an image with its label:
    `img001.jpg` belongs to `img001.txt`, wherever each of them lives.
    """
    out = {}
    for key in _list_keys(prefix.strip("/") + "/"):
        name = key.rsplit("/", 1)[-1]
        if not name or not name.lower().endswith(extensions):
            continue
        out[name.rsplit(".", 1)[0]] = key
    return out


def discover_dataset_sources(
    *, root: str, images_subpath: str = "images", labels_subpath: str = "labels",
) -> dict:
    """Every child folder of `root`, expanded into (images, labels) pairs.

        retraining_evaluation/
          week_1/  images/  labels/
          week2/   images/  labels/

    Pick the root and show it the shape of ONE week; every sibling is
    found from there. Adding week_5 next month needs no new configuration
    -- it simply appears, which is the whole point of discovering rather
    than listing folders by hand.

    Each child is reported with its own counts, so a week that was
    exported without labels is visible before anything is assembled
    rather than after a run trains on it.
    """
    # Whitespace first: `strip("/")` leaves a blank field blank-but-truthy,
    # which sent an empty prefix to S3 instead of asking for a folder.
    root = (root or "").strip().strip("/")
    if not root:
        raise ValueError("Pick the main folder that holds the week folders.")
    images_subpath = (images_subpath or "").strip().strip("/")
    labels_subpath = (labels_subpath or "").strip().strip("/")

    children = []
    for name in sorted(list_s3_folders(root)):
        child = f"{root}/{name}"
        images_prefix = f"{child}/{images_subpath}" if images_subpath else child
        labels_prefix = f"{child}/{labels_subpath}" if labels_subpath else ""
        images = _basenames(images_prefix, IMAGE_EXTENSIONS)
        labels = _basenames(labels_prefix, (".txt",)) if labels_prefix else {}
        children.append({
            "name": name,
            "images_prefix": images_prefix,
            "labels_prefix": labels_prefix,
            "image_count": len(images),
            "label_count": len(labels),
            "paired": len(set(images) & set(labels)),
        })
    return {
        "root": root,
        "images_subpath": images_subpath,
        "labels_subpath": labels_subpath,
        "children": children,
        # A child holding nothing is the common case for a partially
        # uploaded week, and is worth separating from the rest rather
        # than being silently included as zero images.
        "usable": [c["name"] for c in children if c["image_count"]],
    }


def _as_sources(value) -> list:
    """A split's sources, whether one pair was given or several.

    A split used to be exactly one (images, labels) pair. It can now be a
    list of them, so a dataset scattered across several folders assembles
    in one go. Both shapes are accepted so the single-source callers did
    not all have to change at once.
    """
    if not value:
        return []
    if isinstance(value, (tuple, list)) and len(value) == 2 and all(
        isinstance(part, str) for part in value
    ):
        return [tuple(value)]
    return [tuple(pair) for pair in value if pair]


def inspect_dataset_layout(*, splits: dict) -> dict:
    """Pairs each split's image folder with its label folder and reports it.

    `splits` is {"train": (images_prefix, labels_prefix), "val": (...)}.

    Exists because a dataset in this bucket does not always arrive in the
    layout Ultralytics demands. It derives label paths from image paths by
    swapping `/images/` for `/labels/`, so labels sitting anywhere else are
    invisible to it -- the run trains on images with no annotations and
    reports a confident zero. Pairing is therefore checked BEFORE anything
    is built, on stems, so `img001.jpg` finds `img001.txt` wherever it is.
    """
    report = {"splits": [], "warnings": [], "total_paired": 0}
    for split, sources in splits.items():
        for index, (images_prefix, labels_prefix) in enumerate(_as_sources(sources)):
            if not images_prefix:
                continue
            images = _basenames(images_prefix, IMAGE_EXTENSIONS)
            labels = _basenames(labels_prefix, (".txt",)) if labels_prefix else {}
            paired = sorted(set(images) & set(labels))
            unlabelled = sorted(set(images) - set(labels))
            orphans = sorted(set(labels) - set(images))
            report["splits"].append({
                "split": split,
                # Which source within the split, so a report over several
                # folders can be read row by row rather than as one total
                # that hides an empty folder among full ones.
                "source_index": index,
                "images_prefix": images_prefix.strip("/"),
                "labels_prefix": (labels_prefix or "").strip("/"),
                "image_count": len(images),
                "label_count": len(labels),
                "paired": len(paired),
                # Not an error: an image with no label is a background
                # example, and a few are wanted. A LOT of them usually
                # means the wrong label folder was picked, which is why
                # the number is shown.
                "unlabelled": len(unlabelled),
                "orphan_labels": len(orphans),
                "sample": [f"{name}" for name in (paired or unlabelled)[:4]],
            })
            report["total_paired"] += len(paired)
            if not images:
                report["warnings"].append(f"{split}: no images under {images_prefix}.")
            elif labels_prefix and not paired:
                report["warnings"].append(
                    f"{split}: none of the {len(images)} images match a label file in "
                    f"{labels_prefix}. That is usually the wrong label folder."
                )
            elif unlabelled and len(unlabelled) == len(images):
                report["warnings"].append(f"{split}: no image has a label.")
    if not report["splits"]:
        report["warnings"].append("Pick a training images folder to begin.")
    return report


def assemble_dataset(*, splits: dict, classes: dict, destination: str = "",
                     keep_stems: dict | None = None) -> dict:
    """Builds an Ultralytics-shaped dataset from folders laid out any way.

        <dest>/data.yaml
        <dest>/images/train/...   <dest>/labels/train/...
        <dest>/images/val/...     <dest>/labels/val/...

    The images are copied server-side and the labels copied verbatim --
    this only moves things into the shape the trainer requires, it does
    not rewrite class indices. Use the detector filter for that.

    Only PAIRED files are copied, plus unlabelled images with an empty
    label written for them, so every image in the result has a label file
    and none of them is silently missing one.

    `keep_stems` narrows a source to the stems it lists -- `{images_prefix:
    {stem, ...}}`. Recency weighting uses it to take a thinned slice of an
    older week without copying the week somewhere else first: the week
    folder stays whole and only the assembled dataset is smaller.
    """
    keep_stems = keep_stems or {}
    if not classes:
        raise ValueError("Give the dataset a class list, in index order.")
    dest = (destination or "").strip("/") or (
        f"{ASSEMBLED_DATASET_PREFIX}/{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    )
    copied = {"images": 0, "labels": 0, "backgrounds": 0}
    # Where every copied image CAME FROM, keyed by its path inside the
    # assembled folder. Without this the trail stops at the copy: the
    # audit can say "data/assembled/20260922-102139/images/train/x.jpg"
    # and nothing at all about the folder someone actually chose, which
    # is the only part a human recognises.
    sources: dict = {}

    def copy(src_key: str, dest_key: str) -> None:
        s3.copy_object(
            Bucket=BUCKET, CopySource={"Bucket": BUCKET, "Key": src_key}, Key=dest_key,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
        futures = []
        for split, split_sources in splits.items():
            pairs = _as_sources(split_sources)
            # Stems only have to be unique within a split, and two folders
            # merged into one split routinely both hold "img001.jpg".
            # Left alone, the second copy would silently overwrite the
            # first and the dataset would be short of images with no error
            # anywhere. `taken` makes the collision visible and resolves
            # it by suffixing, rather than losing data.
            taken: set = set()
            for source_index, (images_prefix, labels_prefix) in enumerate(pairs):
                if not images_prefix:
                    continue
                images = _basenames(images_prefix, IMAGE_EXTENSIONS)
                labels = _basenames(labels_prefix, (".txt",)) if labels_prefix else {}
                wanted = keep_stems.get(images_prefix)
                if wanted is not None:
                    images = {stem: key for stem, key in images.items() if stem in wanted}
                for stem, image_key in sorted(images.items()):
                    extension = image_key.rsplit(".", 1)[-1]
                    unique = stem
                    if unique in taken:
                        # Suffixed with the source it came from, so the
                        # name still says where it is from rather than
                        # becoming img001-2.jpg with no explanation.
                        unique = f"{stem}__s{source_index}"
                        counter = 2
                        while unique in taken:
                            unique = f"{stem}__s{source_index}-{counter}"
                            counter += 1
                    taken.add(unique)
                    futures.append(pool.submit(
                        copy, image_key, f"{dest}/images/{split}/{unique}.{extension}",
                    ))
                    sources[f"images/{split}/{unique}.{extension}"] = image_key
                    copied["images"] += 1
                    label_key = labels.get(stem)
                    if label_key:
                        futures.append(pool.submit(
                            copy, label_key, f"{dest}/labels/{split}/{unique}.txt",
                        ))
                        copied["labels"] += 1
                    else:
                        # An explicit empty label, not a missing file: that
                        # is what marks the image as a background example
                        # rather than one whose annotations went astray.
                        futures.append(pool.submit(
                            s3.put_object, Bucket=BUCKET,
                            Key=f"{dest}/labels/{split}/{unique}.txt", Body=b"",
                        ))
                        copied["backgrounds"] += 1
        for future in concurrent.futures.as_completed(futures):
            future.result()

    if not copied["images"]:
        raise ValueError("Nothing was copied -- the image folders held no images.")

    # Any val source with an images prefix. Indexing [0] here broke once a
    # split became a LIST of pairs: it read the first pair rather than its
    # images prefix, and an empty list raised.
    has_val = any(images for images, _labels in _as_sources(splits.get("val")))
    manifest = {
        "names": {int(index): name for index, name in sorted(
            ((int(k), v) for k, v in classes.items()))},
        "train": "images/train",
        # Ultralytics needs a val key; without a separate set it evaluates
        # on the training images, which is what the pipeline already does.
        "val": "images/val" if has_val else "images/train",
    }
    s3.put_object(
        Bucket=BUCKET, Key=f"{dest}/data.yaml",
        Body=_yaml().safe_dump(manifest, sort_keys=False).encode(),
    )
    # One object, not an attribute per file: the audit reads it once and
    # resolves the whole listing, and a dataset of 24,000 images would
    # not fit anywhere smaller.
    s3.put_object(
        Bucket=BUCKET, Key=f"{dest}/{SOURCES_FILE}",
        Body=json.dumps({"sources": sources}, indent=2).encode(),
        ContentType="application/json",
    )
    print(f"Assembled {dest}: {copied}")
    return {"folder": dest, **copied, "classes": manifest["names"]}


SOURCES_FILE = "sources.json"


def dataset_source_map(folder: str) -> dict:
    """{path inside the assembled folder: the S3 key it was copied from}.

    Empty for anything not produced by assemble_dataset -- a hand-built
    folder, or one assembled before this was recorded -- so every caller
    must fall back to the assembled path rather than treat absence as an
    error.
    """
    if not folder:
        return {}
    try:
        body = s3.get_object(
            Bucket=BUCKET, Key=f"{folder.strip('/')}/{SOURCES_FILE}",
        )["Body"].read()
    except Exception:  # noqa: BLE001 - absent is the common case
        return {}
    try:
        return dict(json.loads(body).get("sources") or {})
    except (ValueError, AttributeError):
        return {}


def class_filter_for(source_folder: str, detector: dict) -> dict:
    """How to train `detector` from a dataset labelled with more.

    Three answers, and which one applies is a fact about the two
    taxonomies rather than a preference:

    `{}` -- nothing to do. The dataset already declares exactly this
    detector's classes at the same indices.

    `{"classes": [...], "single_cls": True}` -- Ultralytics can do it at
    load time. A ONE-class detector whose class exists in the dataset
    needs the other boxes dropped and its own renumbered to 0, and that
    is precisely what `classes=` plus `single_cls=` do: the first filters
    the boxes, the second renumbers them and sizes the head to one
    output. Nothing in S3 is touched, which matters because the week
    folders are the record and a filtered copy is a second corpus to
    store, keep in step and explain.

    `{"derive": True}` -- neither applies, so a rewritten copy is the
    only way. A detector wanting SEVERAL of the dataset's classes needs
    them renumbered to 0..n-1, and Ultralytics has no mechanism for that:
    `classes=` keeps their original indices, and `single_cls=` would
    collapse them all into one, which is a different model from the one
    asked for.
    """
    wanted = {int(k): v for k, v in detector["classes"].items()}
    available = dataset_classes(source_folder)
    if available and available == wanted:
        return {}

    by_name = {name: index for index, name in available.items()}
    if len(wanted) == 1:
        only = next(iter(wanted.values()))
        if only in by_name:
            return {"classes": [by_name[only]], "single_cls": True}
    return {"derive": True}


def derive_filtered_dataset(source_folder: str, detector: dict, *, split_names=("train", "val")) -> str:
    """Writes a copy of a dataset holding only the detector's classes.

    A site dataset is usually labelled once with everything in it --
    helmet, gloves, person, boots. A helmet detector must be trained on
    helmet alone, or its single output class learns from six kinds of box.
    The training image is digest-pinned and takes no class filter, so the
    filtering happens here: the trainer is handed an ordinary
    single-purpose Ultralytics dataset and needs to know nothing.

    Images with none of the wanted classes are KEPT, with an empty label
    file. They are negative examples, and dropping them is what makes a
    single-class model fire on everything that looks vaguely like the
    thing. Ultralytics recommends roughly 0-10% backgrounds.

    Idempotent: a marker records what was derived from what, so the same
    pairing is built once and reused. Returns the destination folder.
    """
    source = source_folder.strip("/")
    wanted = {name: index for index, name in detector["classes"].items()}
    dest = f"{source}__{detector['slug']}"

    available = dataset_classes(source)          # {source_index: name}

    # Already exactly this detector's taxonomy: same names AT THE SAME
    # INDICES. There is no filtering to do and no remap to perform, so
    # deriving would copy the dataset to a second location to change
    # nothing.
    #
    # It also unblocks the shape the yaml workflow produces. A derive
    # reads `<source>/images/train` directly, which only exists when a
    # dataset was ASSEMBLED into one folder. A root of week folders --
    # what a generated data.yaml names in place -- has no such path, so
    # this used to fail with "no images under images/train" on a dataset
    # that was entirely fine and needed nothing done to it.
    if available and available == {int(k): v for k, v in detector["classes"].items()}:
        print(f"'{source_folder}' already declares exactly "
              f"{detector['slug']}'s classes; nothing to filter.")
        return source

    by_name = {name: index for index, name in available.items()}
    missing = [name for name in wanted if name not in by_name]
    if missing:
        # NOT "has no labels for". Both sides of this comparison come
        # from `dataset_classes`, which reads data.yaml and nothing else
        # -- the .txt files are never opened here. Saying "no labels"
        # sent somebody to check a dataset that was completely intact,
        # when what had happened was the yaml being overwritten by the
        # Assemble card with a different Model Name selected.
        raise ValueError(
            f"'{source_folder}' does not declare {', '.join(missing)} in its "
            f"data.yaml, so a '{detector['slug']}' model cannot be trained "
            f"from it. The yaml declares "
            f"{', '.join(available.values()) or 'nothing'}. "
            + _YAML_IS_THE_ONLY_SOURCE_OF_NAMES
        )
    if available == detector["classes"]:
        # Already exactly right -- nothing to derive, and copying it would
        # only make a second identical dataset to keep track of.
        return source

    # source class index -> the detector's index for that same name.
    remap = {by_name[name]: index for name, index in wanted.items()}
    fingerprint = json.dumps(
        {"source": source, "detector": detector["slug"],
         "classes": {str(k): v for k, v in sorted(detector["classes"].items())}},
        sort_keys=True,
    )
    marker_key = f"{dest}/{FILTERED_DATASET_MARKER}"
    try:
        existing = s3.get_object(Bucket=BUCKET, Key=marker_key)["Body"].read().decode()
        if existing == fingerprint:
            print(f"Reusing filtered dataset {dest}.")
            return dest
    except Exception:
        pass

    print(f"Deriving {dest} from {source}: keeping {', '.join(wanted)}.")
    kept_boxes = 0
    kept_images = 0

    def filter_label(key: str) -> int:
        body = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read().decode("utf-8", "replace")
        lines = []
        for line in body.splitlines():
            parts = line.split()
            if not parts:
                continue
            try:
                source_index = int(float(parts[0]))
            except ValueError:
                continue
            if source_index in remap:
                lines.append(" ".join([str(remap[source_index]), *parts[1:]]))
        # Written even when empty: that is what makes the image a negative
        # example rather than an image with missing labels.
        s3.put_object(
            Bucket=BUCKET,
            Key=f"{dest}/{key[len(source) + 1:]}",
            Body=("\n".join(lines) + ("\n" if lines else "")).encode(),
        )
        return len(lines)

    def copy_image(key: str) -> None:
        # Server-side: the image bytes never come through this container.
        s3.copy_object(
            Bucket=BUCKET,
            CopySource={"Bucket": BUCKET, "Key": key},
            Key=f"{dest}/{key[len(source) + 1:]}",
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
        for split in split_names:
            labels = _list_keys(f"{source}/labels/{split}/")
            images = _list_keys(f"{source}/images/{split}/")
            if not images:
                continue
            kept_images += len(images)
            for count in pool.map(filter_label, [k for k in labels if k.endswith(".txt")]):
                kept_boxes += count
            list(pool.map(copy_image, images))

    if not kept_images:
        # Distinguish "empty dataset" from "a shape this cannot read".
        # A root of week folders holds its images at
        # `<root>/<week>/images/train`, and pointing the derive at the
        # root finds nothing -- which is true, and says nothing useful
        # about what to do next.
        weeks = [name for name in list_s3_folders(source)
                 if _list_keys(f"{source}/{name}/images/")]
        if weeks:
            raise ValueError(
                f"'{source_folder}' holds its images inside week folders "
                f"({', '.join(weeks[:3])}{'...' if len(weeks) > 3 else ''}), not "
                f"directly under images/train. Filtering to "
                f"'{detector['slug']}' copies a dataset laid out as one folder, "
                f"which this is not. Pick a detector whose classes already match "
                f"the dataset's, or assemble the weeks into a single folder first."
            )
        raise ValueError(
            f"'{source_folder}' has no images under images/train or images/val, "
            "so there is nothing to filter."
        )

    s3.put_object(
        Bucket=BUCKET,
        Key=f"{dest}/data.yaml",
        Body=_yaml().safe_dump({
            "names": {index: name for index, name in sorted(detector["classes"].items())},
            "train": "images/train",
            "val": "images/val",
        }, sort_keys=False).encode(),
    )
    # Written last, so an interrupted derive is not mistaken for a finished
    # one on the next run.
    s3.put_object(Bucket=BUCKET, Key=marker_key, Body=fingerprint.encode())
    print(f"Derived {dest}: {kept_images} images, {kept_boxes} boxes kept.")
    return dest


def _evaluates_on_its_training_images(folder: str, separate_folder_chosen: bool) -> bool:
    """Whether the reported metrics are measured on the training images.

    Not the same question as "was a separate evaluation folder chosen".
    An assembled dataset holds train and val as splits of ONE folder, so
    both fields name the same path while the evaluation is genuinely held
    out -- the evaluator runs model.val(split="val"). Reporting that as
    "same as training -- not held out" was wrong, and wrong in the
    direction that makes good metrics look untrustworthy.

    What actually decides it is the folder's own data.yaml: if its `val`
    resolves to the same images as `train`, the evaluation is on training
    data whatever the folders are called.
    """
    prefix = (folder or "").strip("/")
    if not prefix:
        return True
    try:
        body = s3.get_object(Bucket=BUCKET, Key=f"{prefix}/data.yaml")["Body"].read()
        manifest = _yaml().safe_load(body) or {}
    except Exception:
        # Unreadable: fall back to the old proxy rather than claiming a
        # held-out evaluation that cannot be shown.
        return not separate_folder_chosen
    train = str(manifest.get("train") or "").strip("/")
    val = str(manifest.get("val") or manifest.get("valid") or "").strip("/")
    if not val:
        return True
    return val == train


#: Appended to every "this dataset does not declare X" refusal.
#:
#: Both of those refusals compare a detector's class list against
#: `dataset_classes`, which reads data.yaml and opens no .txt file at
#: all. They used to say "has no labels for X", which sent somebody to
#: check a dataset that was completely intact -- the labels had every
#: box; the YAML had been overwritten by the Assemble card while a
#: different Model Name was selected, and a one-class yaml landed on
#: six-class data.
_YAML_IS_THE_ONLY_SOURCE_OF_NAMES = (
    "Class names live ONLY in data.yaml -- a label line is "
    "'2 0.51 0.22 0.26 0.08', indices and no names -- so this says nothing "
    "about whether those boxes exist. If they do, the yaml was last written "
    "with a different Model Name selected; rewrite it in 'Assemble a "
    "dataset' with this one chosen."
)


def dataset_classes(folder: str) -> dict:
    """The taxonomy a folder's data.yaml declares, as {index: name}."""
    prefix = folder.strip("/")
    body = s3.get_object(Bucket=BUCKET, Key=f"{prefix}/data.yaml")["Body"].read()
    manifest = _yaml().safe_load(body) or {}
    names = manifest.get("names") or {}
    # `names: [person, helmet]` is as standard as the mapping form -- it is
    # what a Roboflow export writes -- and treating it as a dict raised
    # AttributeError, which surfaced as "could not read data.yaml".
    if isinstance(names, (list, tuple)):
        names = {index: value for index, value in enumerate(names)}
    return {int(k): str(v) for k, v in names.items()}
