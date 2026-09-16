"""Hand-verified tests for the IoU/average-precision math backing Track2's
evaluation queue -- these are the numbers a reviewer judges a model by, so
they're checked against manually-computed expected values, not just
"does it run"."""

from __future__ import annotations

from ppe_pipeline_api import track2_shared as t2


def test_iou_identical_boxes_is_one():
    box = (0.0, 0.0, 1.0, 1.0)
    assert t2._box_iou(box, box) == 1.0


def test_iou_partial_overlap():
    box_a = (0.0, 0.0, 1.0, 1.0)
    box_b = (0.5, 0.0, 1.5, 1.0)  # two unit squares shifted by 0.5 -> intersection 0.5, union 1.5
    assert abs(t2._box_iou(box_a, box_b) - (0.5 / 1.5)) < 1e-9


def test_iou_no_overlap_is_zero():
    assert t2._box_iou((0.0, 0.0, 1.0, 1.0), (2.0, 2.0, 3.0, 3.0)) == 0.0


def test_average_precision_single_perfect_hit():
    assert t2._average_precision([(0.9, True)], num_ground_truth=1) == 1.0


def test_average_precision_extra_fp_after_recall_saturates_does_not_reduce_ap():
    # Once recall reaches 1.0, further false positives at lower confidence
    # add no additional area under the (recall on x-axis) PR curve.
    ap = t2._average_precision([(0.9, True), (0.5, False)], num_ground_truth=1)
    assert ap == 1.0


def test_average_precision_no_predictions_is_zero():
    assert t2._average_precision([], num_ground_truth=1) == 0.0


def test_average_precision_classic_one_hit_one_miss():
    # 2 ground truth boxes, 2 predictions: the higher-confidence one hits,
    # the second is a false positive (no ground truth left to claim).
    # PR points: (precision=1.0, recall=0.5), (precision=2/3, recall=0.5).
    # Precision envelope (non-increasing from the recall=1 end): [1.0, 2/3].
    # AP = 1.0*(0.5-0) + (2/3)*(0.5-0.5) = 0.5.
    ap = t2._average_precision([(0.9, True), (0.8, False)], num_ground_truth=2)
    assert abs(ap - 0.5) < 1e-9


def test_match_class_predictions_greedy_by_confidence_not_by_list_order():
    # The higher-confidence prediction has zero overlap and must NOT steal
    # the ground-truth box away from the lower-confidence prediction that
    # actually matches it.
    per_image = {
        "img1": {
            "gt": [(0, (0.0, 0.0, 1.0, 1.0))],
            "pred": [
                (0.9, 0, (0.0, 0.0, 1.0, 1.0)),
                (0.95, 0, (5.0, 5.0, 6.0, 6.0)),
            ],
        }
    }
    hits, num_gt = t2._match_class_predictions(per_image, class_id=0, iou_threshold=0.5)
    assert num_gt == 1
    assert hits == [(0.95, False), (0.9, True)]


def test_match_class_predictions_below_iou_threshold_is_a_miss():
    per_image = {
        "img1": {
            "gt": [(0, (0.0, 0.0, 1.0, 1.0))],
            "pred": [(0.9, 0, (0.5, 0.0, 1.5, 1.0))],  # IoU = 1/3, below the 0.5 threshold
        }
    }
    hits, num_gt = t2._match_class_predictions(per_image, class_id=0, iou_threshold=0.5)
    assert num_gt == 1
    assert hits == [(0.9, False)]


def test_match_class_predictions_ignores_other_classes():
    per_image = {
        "img1": {
            "gt": [(0, (0.0, 0.0, 1.0, 1.0)), (1, (0.0, 0.0, 1.0, 1.0))],
            "pred": [(0.9, 1, (0.0, 0.0, 1.0, 1.0))],
        }
    }
    hits, num_gt = t2._match_class_predictions(per_image, class_id=0, iou_threshold=0.5)
    assert num_gt == 1
    assert hits == []  # the only prediction is class 1, not class 0
