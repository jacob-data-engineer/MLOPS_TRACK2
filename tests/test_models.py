import pytest

from ppe_pipeline.models import BoundingBox


def test_iou_of_identical_boxes_is_one():
    box = BoundingBox(0, 0, 10, 10)
    assert box.iou(box) == pytest.approx(1.0)


def test_iou_of_disjoint_boxes_is_zero():
    a = BoundingBox(0, 0, 10, 10)
    b = BoundingBox(20, 20, 30, 30)
    assert a.iou(b) == 0.0


def test_containment_ratio_high_when_small_box_nested_in_large_box():
    # A hand box fully inside a much bigger person box: IoU is tiny, but
    # containment_ratio (the metric merge.py actually clusters on) is 1.0.
    person = BoundingBox(0, 0, 200, 400)
    hand = BoundingBox(10, 300, 60, 360)
    assert person.iou(hand) < 0.1
    assert hand.containment_ratio(person) == pytest.approx(1.0)


def test_expand_grows_symmetrically_and_clips():
    box = BoundingBox(10, 10, 20, 20)
    expanded = box.expand(1.0)  # +100% of own size each side
    assert expanded == BoundingBox(0, 0, 30, 30)

    clipped = box.expand(1.0, clip_to=(15, 15))
    assert clipped.x2 == 15
    assert clipped.y2 == 15


def test_union_box_covers_both_inputs():
    a = BoundingBox(0, 0, 10, 10)
    b = BoundingBox(5, 5, 20, 30)
    u = a.union_box(b)
    assert u == BoundingBox(0, 0, 20, 30)


def test_degenerate_box_rejected():
    with pytest.raises(ValueError):
        BoundingBox(10, 10, 10, 20)
