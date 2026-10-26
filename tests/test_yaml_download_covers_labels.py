"""The prefix the run downloads has to contain the labels.

A data.yaml names only image folders -- Ultralytics finds the labels by
swapping the last `/images/` for `/labels/`. So the labels are never
written down anywhere, and nothing about the yaml itself says whether
they will be on disk when the run opens it.

They will be only if they sit under the ONE prefix `train.py` downloads,
and that prefix is derived from the image folders alone. The ordinary
Ultralytics layout breaks that: `<root>/images/train` and
`<root>/images/val` share `<root>/images`, which excludes `<root>/labels`
entirely. The run then succeeds, trains every image as a background
example with no boxes, and reports metrics for a model that learned
nothing.
"""
from __future__ import annotations

import pytest

from ppe_pipeline_api import dataset_yaml as dy

ROOT = "Testing_Dataset/object_detection_dataset"


def _covers(base: str, key: str) -> bool:
    return key == base or key.startswith(base.rstrip("/") + "/")


# --- the case that was broken ----------------------------------------------------

@pytest.mark.parametrize("split", ["train", "val"])
def test_the_standard_layout_downloads_its_labels(split):
    images = f"{ROOT}/images/{split}"
    base = dy._base_or_refuse([images], "training")
    assert _covers(base, dy._labels_dir_for(images)), (
        f"base {base!r} excludes the labels for {images!r}"
    )


def test_train_and_val_off_one_root_land_on_the_dataset_root():
    """Not on `<root>/images`, which was the whole bug."""
    assert dy._base_or_refuse([f"{ROOT}/images/train"], "training") == ROOT
    assert dy._base_or_refuse([f"{ROOT}/images/val"], "validation") == ROOT


def test_the_relative_paths_still_resolve_from_that_base():
    """Stepping the base up must not strand the folders it names: the
    trainer sets `path` to the download and Ultralytics joins from there.
    """
    base = dy._base_or_refuse([f"{ROOT}/images/train"], "training")
    assert f"{ROOT}/images/train"[len(base) + 1:] == "images/train"


# --- every shape, not just the one that bit ---------------------------------------

@pytest.mark.parametrize("dirs", [
    [f"{ROOT}/images/train"],
    [f"{ROOT}/images/train", f"{ROOT}/images/val"],
    ["jp/a/images", "jp/b/images"],
    ["jp/site-a/images"],
    ["a/x/images/train", "a/y/images/train"],
])
def test_no_arrangement_leaves_its_labels_outside(dirs):
    base = dy._base_or_refuse(dirs, "training")
    for d in dirs:
        assert _covers(base, dy._labels_dir_for(d)), (d, base)


# --- what must not have changed ---------------------------------------------------

def test_the_bases_that_were_already_right_are_untouched():
    assert dy._base_or_refuse(["jp/a/images", "jp/b/images"], "training") == "jp"
    assert dy._base_or_refuse(["jp/site-a/images"], "training") == "jp/site-a"


def test_folders_with_no_common_root_are_still_refused():
    with pytest.raises(ValueError, match="share no parent"):
        dy._base_or_refuse(["alpha/images", "beta/images"], "training")


def test_the_refusal_still_names_the_image_roots_not_the_labels():
    """The operator picked image folders; naming `labels` roots they
    never chose would send them looking for something that is not on
    their screen."""
    with pytest.raises(ValueError) as exc:
        dy._base_or_refuse(["alpha/images", "beta/images"], "training")
    assert "'alpha'" in str(exc.value) and "'beta'" in str(exc.value)
