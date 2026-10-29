"""What a dataset SELECTS: its data.yaml and the file lists it names.

Moved out of `training/run_artifacts.py` with the data/training split
(2026-10-28). Reading a selection is a question about a dataset -- the
snapshot freeze asks it too -- so it is the data side's. Keeping a run's
own copy of it (`run_artifacts.capture`) stays with training.
"""

from __future__ import annotations

import posixpath

import yaml as _yaml

from .. import pull_control as pc


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
