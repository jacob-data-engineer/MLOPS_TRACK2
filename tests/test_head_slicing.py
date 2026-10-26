"""Cutting a YOLO detect head down to a subset of its classes.

The point of this over the old behaviour: picking a subset used to
register a SEED -- the registry said three classes, the weights still
predicted eighty, and only a training run reconciled them. That is right
when the subset is a starting point for new data and wrong when the
weights are already good and the extra classes are simply unwanted, which
is the common case. Cutting the head gives the same model minus some
outputs, deployable immediately.

The claim that has to hold is exact, not approximate: the kept classes
must predict what they predicted before. These tests check that on a real
network rather than on a mock, because the thing that could be wrong is
the tensor indexing, and a mock cannot be wrong about it.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="head slicing needs torch")
ultralytics = pytest.importorskip("ultralytics", reason="head slicing needs ultralytics")

from ppe_pipeline_api import track2_shared as t2  # noqa: E402


@pytest.fixture(scope="module")
def net():
    """A real YOLO network, built from its yaml -- no weights downloaded."""
    from ultralytics import YOLO

    model = YOLO("yolo11n.yaml")
    model.model.eval()
    return model


def test_the_kept_classes_predict_exactly_what_they_did_before(net):
    """The whole justification. If this is approximate, slicing is a
    different model and the feature is a lie."""
    from ultralytics import YOLO

    fresh = YOLO("yolo11n.yaml")
    fresh.model.eval()
    # Same weights in both, so a difference can only come from the cut.
    fresh.model.load_state_dict(net.model.state_dict())

    image = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        before = fresh.model(image)
    before = before[0] if isinstance(before, (list, tuple)) else before

    keep = [0, 2, 7]
    t2._slice_detect_head(fresh.model, keep)
    fresh.model.eval()
    with torch.no_grad():
        after = fresh.model(image)
    after = after[0] if isinstance(after, (list, tuple)) else after

    box_rows = 4
    kept_before = before[:, box_rows:, :][:, keep, :]
    kept_after = after[:, box_rows:, :]
    assert torch.equal(kept_before, kept_after), "class scores must be identical"
    assert torch.equal(before[:, :box_rows, :], after[:, :box_rows, :]), (
        "boxes are predicted class-agnostically; cutting a class must not "
        "move any geometry"
    )


def test_the_output_shrinks_by_exactly_the_dropped_classes(net):
    from ultralytics import YOLO

    fresh = YOLO("yolo11n.yaml")
    fresh.model.eval()
    fresh.model.load_state_dict(net.model.state_dict())
    head = fresh.model.model[-1]
    before_nc = head.nc

    t2._slice_detect_head(fresh.model, [1, 3])

    assert head.nc == 2
    assert head.no == 2 + head.reg_max * 4, "outputs per anchor must follow nc"
    assert before_nc != head.nc
    with torch.no_grad():
        out = fresh.model(torch.randn(1, 3, 640, 640))
    out = out[0] if isinstance(out, (list, tuple)) else out
    assert out.shape[1] == 4 + 2


def test_the_names_are_renumbered_from_zero_keeping_their_meaning(net):
    from ultralytics import YOLO

    fresh = YOLO("yolo11n.yaml")
    fresh.model.load_state_dict(net.model.state_dict())
    original = dict(fresh.model.names)

    names = t2._slice_detect_head(fresh.model, [2, 7])

    # Renumbered from zero, in the order asked for. (That the order is the
    # CALLER'S, not numeric, is pinned separately below -- this one is
    # about the numbering starting at 0 and the names travelling with the
    # weights into every place that reads them.)
    assert names == {0: original[2], 1: original[7]}
    assert fresh.model.names == names
    assert fresh.model.yaml["nc"] == 2


def test_asking_for_a_class_the_file_does_not_have_is_refused(net):
    from ultralytics import YOLO

    fresh = YOLO("yolo11n.yaml")
    fresh.model.load_state_dict(net.model.state_dict())
    with pytest.raises(ValueError, match="not in this checkpoint"):
        t2._slice_detect_head(fresh.model, [0, 9999])


def test_asking_for_nothing_is_refused(net):
    from ultralytics import YOLO

    fresh = YOLO("yolo11n.yaml")
    fresh.model.load_state_dict(net.model.state_dict())
    with pytest.raises(ValueError, match="nothing to detect"):
        t2._slice_detect_head(fresh.model, [])


def test_an_end_to_end_head_is_refused_rather_than_half_cut(net, monkeypatch):
    """v10-style heads carry a second one2one branch. Cutting one branch
    and not the other leaves a head predicting two taxonomies."""
    from ultralytics import YOLO

    fresh = YOLO("yolo11n.yaml")
    fresh.model.load_state_dict(net.model.state_dict())
    # `end2end` is a read-only PROPERTY on Detect, not a plain attribute:
    # assigning to the instance silently does nothing and reads back
    # False. Patch the class, or this tests nothing.
    head = fresh.model.model[-1]
    monkeypatch.setattr(type(head), "end2end", property(lambda self: True))
    assert head.end2end is True, "the patch itself has to take"
    with pytest.raises(ValueError, match="end-to-end head"):
        t2._slice_detect_head(fresh.model, [0, 1])


def test_the_callers_order_is_the_new_numbering(net):
    """When somebody types their own class names, their FIRST LINE is
    class 0. Keeping [7, 2] has to mean {0: <7>, 1: <2>}.

    Sorting here would file every label under the wrong name and nothing
    would error -- the model would simply be wrong about what it detects.
    """
    from ultralytics import YOLO

    fresh = YOLO("yolo11n.yaml")
    fresh.model.load_state_dict(net.model.state_dict())
    original = dict(fresh.model.names)

    names = t2._slice_detect_head(fresh.model, [7, 2])
    assert names == {0: original[7], 1: original[2]}


def test_the_weights_follow_the_callers_order_too(net):
    """Not just the names. If the rows were taken in numeric order while
    the names were reordered, every score would be attached to the wrong
    label -- the worst version of this bug, because it looks right."""
    from ultralytics import YOLO

    fresh = YOLO("yolo11n.yaml")
    fresh.model.eval()
    fresh.model.load_state_dict(net.model.state_dict())
    image = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        before = fresh.model(image)
    before = before[0] if isinstance(before, (list, tuple)) else before

    t2._slice_detect_head(fresh.model, [7, 2])
    fresh.model.eval()
    with torch.no_grad():
        after = fresh.model(image)
    after = after[0] if isinstance(after, (list, tuple)) else after

    box_rows = 4
    assert torch.equal(before[:, box_rows:, :][:, 7, :], after[:, box_rows:, :][:, 0, :])
    assert torch.equal(before[:, box_rows:, :][:, 2, :], after[:, box_rows:, :][:, 1, :])


def test_duplicates_are_dropped_without_disturbing_the_order(net):
    from ultralytics import YOLO

    fresh = YOLO("yolo11n.yaml")
    fresh.model.load_state_dict(net.model.state_dict())
    original = dict(fresh.model.names)
    names = t2._slice_detect_head(fresh.model, [5, 1, 5])
    assert names == {0: original[5], 1: original[1]}
