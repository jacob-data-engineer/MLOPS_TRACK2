"""What a run actually consumed, kept where it cannot move afterwards.

The training audit records a run's files by listing the dataset folder
recursively. That was exactly right while `assemble_dataset` copied the
chosen images into a folder of their own: the folder WAS the selection,
so listing it and reading the selection were the same act.

It stopped being right when the yaml started naming week folders in
place. `dataset_folder` is now the whole root, a recursive listing finds
every image in every week, and a run that trained on 587 of them would
be recorded as having used all 748. An audit that overstates what a run
saw is worse than none: it reads as precise.

So when the yaml points at FILE LISTS, those lists are the selection and
this reads them instead. And all three artifacts -- the yaml and the two
lists -- are copied to a folder of the run's own at submit time:

    _runs/<audit_id>/data.yaml
    _runs/<audit_id>/train.txt
    _runs/<audit_id>/val.txt

Copied, not referenced. The live slots are overwritten by the next Save,
so a pointer to them answers "what does that path hold now", which is a
different question from "what did this run train on" and is the wrong
one.

Outside the training prefix, for the same reason snapshots are: the
trainer downloads every object beneath the prefix it is given, so a
history of runs kept inside it would be fetched by every later run.
"""

from __future__ import annotations

import posixpath

import yaml as _yaml

from . import pull_control as pc

#: One folder per run. Not under the week-folder root -- see the module
#: docstring.
PREFIX = "_runs"

#: A guard on reading a list. These are image paths, one per line; a file
#: far past this is not a training selection and reading it whole into a
#: request would be the wrong way to find that out.
MAX_LINES = 500_000


def _read(key: str, bucket: str) -> str:
    body = pc.s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    return body.decode("utf-8", "replace")


def _split_entry(document: dict, name: str) -> str:
    """A split's entry only when it is a FILE LIST.

    A list of week folders is not one, and neither is a bare directory:
    both mean "everything under here", which is what the recursive
    listing already describes.
    """
    entry = document.get(name)
    return entry.strip() if isinstance(entry, str) and entry.strip() else ""


def read_selection(folder: str, *, bucket: str = "") -> dict:
    """The yaml a dataset folder holds, and the lists it points at.

    Returns empty-handed rather than raising when there is no yaml, no
    list form, or nothing readable: this runs inside an audit write, and
    losing the audit because its extra detail could not be gathered would
    trade something whole for something better.
    """
    folder = (folder or "").strip("/")
    bucket = bucket or pc.destination_bucket()
    blank = {"yaml": "", "yaml_key": "", "lists": {}, "images": [], "labels": []}
    if not folder:
        return blank

    yaml_key = f"{folder}/data.yaml"
    try:
        text = _read(yaml_key, bucket)
        document = _yaml.safe_load(text) or {}
    except Exception:                                          # noqa: BLE001
        return blank
    if not isinstance(document, dict):
        return blank

    out = {"yaml": text, "yaml_key": yaml_key, "lists": {},
           "images": [], "labels": []}
    for split in ("train", "val"):
        entry = _split_entry(document, split)
        if not entry:
            continue
        key = posixpath.normpath(f"{folder}/{entry}").replace("\\", "/")
        try:
            body = _read(key, bucket)
        except Exception:                                      # noqa: BLE001
            continue
        lines = [line.strip() for line in body.splitlines()[:MAX_LINES]
                 if line.strip()]
        out["lists"][split] = {"key": key, "entry": entry, "lines": lines}

    # The names as the audit records them: relative to the dataset folder,
    # because the by-file-name index is keyed on that shape and a change
    # here would orphan every row already written.
    for split, found in out["lists"].items():
        for line in found["lines"]:
            name = line[2:] if line.startswith("./") else line.lstrip("/")
            out["images"].append(name)
            # The label that supervises it. Same swap Ultralytics makes,
            # so the audit names the file the run actually learned from
            # rather than one inferred differently.
            if "/images/" in name:
                stem = name.rsplit(".", 1)[0]
                out["labels"].append(stem.replace("/images/", "/labels/", 1) + ".txt")
    return out


def capture(audit_id: str, folder: str, *, bucket: str = "") -> dict:
    """Copy this run's yaml and lists into a folder of its own.

    Never raises. An audit write failing must not take down the training
    submission it describes, and that applies doubly to the part of it
    that is only for reading later.
    """
    if not audit_id:
        return {}
    bucket = bucket or pc.destination_bucket()
    found = read_selection(folder, bucket=bucket)
    if not found["yaml_key"]:
        return {}

    base = f"{PREFIX}/{audit_id}"
    written: dict = {}
    try:
        pc.s3.copy_object(
            Bucket=bucket, Key=f"{base}/data.yaml",
            CopySource={"Bucket": bucket, "Key": found["yaml_key"]},
            ContentType="text/yaml", MetadataDirective="REPLACE",
        )
        written["yaml_key"] = f"{base}/data.yaml"
        for split, entry in found["lists"].items():
            name = f"{split}.txt"
            pc.s3.copy_object(
                Bucket=bucket, Key=f"{base}/{name}",
                CopySource={"Bucket": bucket, "Key": entry["key"]},
                ContentType="text/plain", MetadataDirective="REPLACE",
            )
            written[f"{split}_list_key"] = f"{base}/{name}"
            written[f"{split}_list_lines"] = len(entry["lines"])
    except Exception as exc:                                   # noqa: BLE001
        print(f"Could not capture run artifacts for {audit_id}: {exc}")
        return written

    written["folder"] = base
    written["s3_uri"] = f"s3://{bucket}/{base}/"
    return written
