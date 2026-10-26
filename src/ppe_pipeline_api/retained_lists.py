"""The retention policy, written where the trainer will actually obey it.

Until now the decay was a number on a card. A generated `data.yaml` names
week FOLDERS, so a run trained on every image in them whatever the policy
said -- the card could report "keep 301 of 310" while the run used all
310, and nothing in the system disagreed.

Ultralytics resolves a split entry two ways (`data/base.py`,
`get_img_files`): a directory is globbed, and a FILE is read as a list of
image paths, one per line. That second form is the hook. Writing the
selected images to a `.txt` and pointing `train:` at it makes the policy
real, and needs nothing added to the trainer.

    train: _retained/train.txt
    val:   _retained/val.txt

Labels are still found per image by swapping `/images/` for `/labels/`,
so a list may cross week folders freely -- each entry resolves its own
label in its own week. That is the property the whole idea rests on.

THE TRAIN SIDE IS DECAYED; THE VAL SIDE IS NOT. An evaluation set that
shrinks week by week makes every run's numbers incomparable with the
last, and an improvement you cannot measure is not one you can ship.
"""

from __future__ import annotations

from . import pull_control as pc
from . import retention as rt

#: Where the live lists sit: the ROOT of the downloaded prefix, beside
#: data.yaml. Not a subfolder, and that is correctness rather than
#: tidiness.
#:
#: `get_img_files` expands a leading `./` against THE LIST FILE'S OWN
#: DIRECTORY, not against `path:`:
#:
#:     parent = str(p.parent) + os.sep
#:     f += [x.replace("./", parent, 1) if x.startswith("./") else x ...]
#:
#: So a list at `<root>/_retained/train.txt` holding `./week_1/...`
#: resolves to `<root>/_retained/week_1/...`, which does not exist. That
#: is exactly how it failed -- "No valid images found in
#: .../_retained/week_1/labels/train.cache". The lines are relative to
#: the root, so the list belongs at the root.
FOLDER = ""
TRAIN_NAME = "train.txt"
VAL_NAME = "val.txt"


def select(*, root: str, config: dict | None = None, bucket: str = "",
           images_subpath: str = "images") -> dict:
    """Which images each side of a run should actually train on.

    Returns paths RELATIVE to `root`, which is the prefix the trainer
    downloads -- so the same list reads correctly in the bucket and in
    the container.

    The decay is applied through `rt.plan`, not reimplemented: the same
    ranking, the same floors, the same skipping of empty weeks. A second
    implementation would drift from the card that describes it, and the
    first anybody would know is a run that trained on something other
    than the grid promised.
    """
    root = (root or "").strip().strip("/")
    if not root:
        raise ValueError("Pick the folder that holds the week folders.")
    bucket = bucket or pc.destination_bucket()
    config = {**rt.DEFAULTS, **(config or {})}

    # The SAME inventory the card's bars and the schedule grid count
    # from: weeks only (not `_retained/`, which this very module writes
    # into the root), in the same order, the train side and the val side
    # read apart. A second reading here is how the card came to say
    # "100 of 200" over a week this list took 50 of 100 from.
    inventory = rt.read_inventory(root, images_subpath=images_subpath,
                                  week_prefix=config.get("week_prefix") or "week_")
    names = [w["name"] for w in inventory]
    train_keys = {w["name"]: w["train"] for w in inventory}
    val_keys = {w["name"]: w["val"] for w in inventory}

    # The train side, through the policy.
    planned = rt.plan_inventory(inventory, config)
    kept_by_week = {row["week"]: row for row in planned["weeks"]}

    train: list = []
    rows: list = []
    for week in names:
        row = kept_by_week.get(week) or {}
        kept = row.get("stems") or []
        train += [_relative(train_keys[week][stem], root)
                  for stem in kept if stem in train_keys[week]]
        rows.append({
            "week": week,
            "available": len(train_keys[week]),
            "keep": len(kept),
            "age_weeks": row.get("age_weeks", 0),
            "fraction": row.get("fraction", 1.0),
            "val": len(val_keys[week]),
        })

    # The val side, whole.
    val: list = []
    for week in names:
        val += [_relative(key, root) for key in sorted(val_keys[week].values())]

    train.sort()
    val.sort()
    return {
        "root": root,
        "bucket": bucket,
        "train": train,
        "val": val,
        "weeks": rows,
        "train_available": sum(r["available"] for r in rows),
        "val_available": sum(r["val"] for r in rows),
        "enabled": bool(config.get("enabled")),
    }


def _relative(key: str, root: str) -> str:
    """A key as the trainer will see it: relative to the download root,
    with the `./` Ultralytics resolves against the list's own folder."""
    inside = key[len(root) + 1:] if key.startswith(root + "/") else key
    return "./" + inside.lstrip("/")


def render(paths: list) -> str:
    """The list as Ultralytics reads it: one path per line.

    No comments and no blank lines. A line whose extension is not an
    image format is dropped SILENTLY by `get_img_files` -- it filters on
    the extension before anything checks the file exists -- so a header
    comment here would not be an annotation, it would be an invisible
    reduction of the dataset.
    """
    return "\n".join(paths) + "\n" if paths else ""


def write(*, root: str, config: dict | None = None, bucket: str = "",
          images_subpath: str = "images", folder: str = "") -> dict:
    """Write the two lists where the yaml will point at them.

    Checked after writing, not assumed: `get_img_files` drops a line with
    a non-image extension without a word, so a typo is a quietly smaller
    training set. Counting the lines back against what the policy chose
    is the only thing that catches it.
    """
    chosen = select(root=root, config=config, bucket=bucket,
                    images_subpath=images_subpath)
    base = (folder or f"{chosen['root']}/{FOLDER}").strip("/")

    written = {}
    for name, paths in ((TRAIN_NAME, chosen["train"]), (VAL_NAME, chosen["val"])):
        key = f"{base}/{name}"
        body = render(paths)
        pc.s3.put_object(Bucket=chosen["bucket"], Key=key,
                         Body=body.encode("utf-8"), ContentType="text/plain")
        lines = [line for line in body.splitlines() if line.strip()]
        if len(lines) != len(paths):
            raise ValueError(
                f"{key} was written with {len(lines)} lines for {len(paths)} "
                f"selected images."
            )
        written[name] = {"key": key, "lines": len(lines)}

    chosen["written"] = written
    # What the yaml's `train:` and `val:` must say, relative to the root.
    # The entry as the yaml writes it: relative to the root, which is
    # where the list itself sits, so `./` in its lines resolves there.
    inside = base[len(chosen["root"]) + 1:] if base.startswith(
        chosen["root"] + "/") else ""
    chosen["train_entry"] = f"{inside}/{TRAIN_NAME}" if inside else TRAIN_NAME
    chosen["val_entry"] = f"{inside}/{VAL_NAME}" if inside else VAL_NAME
    return chosen
