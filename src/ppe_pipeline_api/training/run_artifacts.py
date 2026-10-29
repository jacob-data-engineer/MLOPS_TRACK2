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

from .. import pull_control as pc
from .. import data as _data

#: What a dataset selects is a data-side question (`data/selection.py`
#: since the split); a run's copy of it is this module's.
read_selection = _data.read_selection

#: One folder per run. Not under the week-folder root -- see the module
#: docstring.
PREFIX = "_runs"


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
