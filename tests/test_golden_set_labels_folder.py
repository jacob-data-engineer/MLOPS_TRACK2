"""The golden set can take its labels from a second folder.

It always read ONE prefix and sorted what fell out by extension, which
covers two layouts -- `<source>/images/ + labels/`, and a flat folder
with both in it -- and cannot reach a third: images under one prefix and
labels under another, with no parent in common.

`labels_source` is that third shape. Empty, nothing changes.
"""
from __future__ import annotations

import inspect
import pathlib

import pytest

from ppe_pipeline_api import golden_set as gs

PANEL = (pathlib.Path(__file__).resolve().parents[1]
         / "web" / "src" / "components" / "Track2GoldenSetPanel.tsx")


@pytest.fixture
def bucket(monkeypatch):
    """Two folders with no parent in common."""
    objects = {
        "curated/images": [
            {"Key": "curated/images/a.jpg", "Size": 10},
            {"Key": "curated/images/b.jpg", "Size": 10},
            # A stray label sitting with the images. It must NOT be read
            # as the label for `a` when a labels folder was named.
            {"Key": "curated/images/a.txt", "Size": 1},
        ],
        "elsewhere/labels": [
            {"Key": "elsewhere/labels/a.txt", "Size": 2},
            {"Key": "elsewhere/labels/b.txt", "Size": 2},
        ],
        "together": [
            {"Key": "together/images/a.jpg", "Size": 10},
            {"Key": "together/labels/a.txt", "Size": 2},
        ],
    }
    monkeypatch.setattr(gs.pc, "_list_objects",
                        lambda prefix, b: objects.get(prefix.rstrip("/"), []))
    monkeypatch.setattr(gs.pc, "destination_bucket", lambda: "track2-s3")
    return objects


# --- the new shape ---------------------------------------------------------------

def test_labels_can_come_from_another_folder(bucket):
    found = gs.read_folder("curated/images", labels_source="elsewhere/labels")
    # A SET: `paired` is ordered by a hash of the stem, deliberately, so
    # that the same folder always splits the same way. Asserting an
    # alphabetical order here would be pinning something the code does
    # not promise.
    assert set(found["paired"]) == {"a", "b"}
    assert found["layout"] == "two folders"
    assert found["labels_source"] == "elsewhere/labels"


def test_the_pairing_order_stays_deterministic(bucket):
    """It is what makes a build repeatable, so the second folder must not
    make it depend on listing order."""
    first = gs.read_folder("curated/images", labels_source="elsewhere/labels")
    again = gs.read_folder("curated/images", labels_source="elsewhere/labels")
    assert first["paired"] == again["paired"]


def test_the_labels_folder_wins_over_a_stray_label_beside_the_images(bucket):
    """`curated/images/a.txt` is in the images prefix. Reading labels
    from both prefixes would pair whichever listed last, which is a
    silent coin toss over which annotation trains."""
    found = gs.read_folder("curated/images", labels_source="elsewhere/labels")
    assert found["_label_keys"]["a"]["key"] == "elsewhere/labels/a.txt"


def test_an_image_prefix_with_no_matching_labels_pairs_nothing(bucket):
    found = gs.read_folder("curated/images", labels_source="together/labels")
    assert found["paired"] == []


# --- the shapes that already worked ----------------------------------------------

def test_one_folder_holding_images_and_labels_is_unchanged(bucket):
    found = gs.read_folder("together")
    assert found["paired"] == ["a"]
    assert found["layout"] == "images/ + labels/"
    assert found["labels_source"] == ""


def test_a_flat_folder_still_pairs_what_is_beside_it(bucket):
    """`curated/images` holds a.jpg, b.jpg and a.txt. Read as ONE folder
    that is the flat layout, and a pairs."""
    found = gs.read_folder("curated/images")
    assert found["paired"] == ["a"]
    assert found["layout"] == "flat"


def test_the_source_is_still_required(bucket):
    with pytest.raises(ValueError, match="Pick the folder"):
        gs.read_folder("", labels_source="elsewhere/labels")


# --- it reaches the whole way through --------------------------------------------

@pytest.mark.parametrize("fn", [gs.read_folder, gs.plan, gs.build])
def test_every_step_accepts_it(fn):
    assert "labels_source" in inspect.signature(fn).parameters


def test_the_route_accepts_it_or_it_is_dropped_in_silence():
    from ppe_pipeline_api.routers.track2_pull_control import SplitRequest

    assert "labels_source" in SplitRequest.model_fields
    assert SplitRequest().labels_source == "", "empty must stay the default"


def test_both_handlers_pass_it_on():
    from ppe_pipeline_api.routers import track2_pull_control as r

    for handler in (r.split_preview, r.split_build):
        assert "labels_source=payload.labels_source" in inspect.getsource(handler)


# --- the card --------------------------------------------------------------------

def _panel() -> str:
    return "\n".join(
        line for line in PANEL.read_text(encoding="utf-8").split("\n")
        if not line.strip().startswith(("//", "*", "/*"))
    )


def test_the_card_offers_the_second_picker():
    """Rendered from a two-entry array now, in the Assemble card's pair
    shape, rather than two hand-written always-open fields."""
    code = _panel()
    assert '["images", "Images"' in code
    assert '["labels", "Labels"' in code


def test_both_pickers_are_the_shared_component():
    """Not a second implementation of browsing a bucket.

    The pair shares ONE `Track2S3FolderBrowser`, opened under whichever
    field was clicked -- the Assemble card's arrangement. The target
    folder below keeps its own always-open `S3FolderField`, because it
    is a single folder and the only one here that may be created.
    """
    code = _panel()
    assert code.count("<Track2S3FolderBrowser") == 1
    assert code.count("<S3FolderField") == 1


def test_the_second_folder_is_saved_with_the_rest():
    code = _panel()
    assert "labels_source: labelsSource" in code
    assert "if (g.labels_source) setLabelsSource(g.labels_source);" in code


def test_it_is_sent_with_the_split_request():
    code = _panel()
    block = code.split("const request = useCallback")[1][:400]
    assert "labels_source: labelsSource" in block
