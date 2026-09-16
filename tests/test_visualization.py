from PIL import Image

from ppe_pipeline.models import BoundingBox, Detection, ViolationType
from ppe_pipeline.visualization import (
    draw_detections,
    draw_judge_helmet_box,
    normalized_box_to_pixels,
)


def test_draw_detections_returns_same_size_rgb_image_without_mutating_input():
    image = Image.new("L", (200, 200))  # grayscale, to check RGB conversion too
    detections = [
        Detection(
            ViolationType.HELMET,
            person_box=BoundingBox(10, 10, 100, 190),
            region_box=BoundingBox(20, 10, 80, 50),
            source="t",
        )
    ]

    annotated = draw_detections(image, detections)

    assert annotated.size == image.size
    assert annotated.mode == "RGB"
    assert image.mode == "L"  # original untouched


def test_draw_detections_handles_empty_list():
    image = Image.new("RGB", (50, 50), color="white")
    annotated = draw_detections(image, [])
    assert annotated.size == (50, 50)


def test_draw_detections_handles_forklift_pair():
    image = Image.new("RGB", (500, 500), color="white")
    detections = [
        Detection(
            ViolationType.TOO_NEAR_FORKLIFT,
            person_box=BoundingBox(50, 50, 150, 250),
            forklift_box=BoundingBox(140, 50, 400, 250),
            source="t",
        )
    ]
    annotated = draw_detections(image, detections)
    assert annotated.size == (500, 500)


def test_draw_judge_helmet_box_outlines_the_reported_region():
    image = Image.new("L", (200, 100))  # grayscale, to check RGB conversion too

    annotated = draw_judge_helmet_box(image, [0.1, 0.2, 0.5, 0.6])

    assert annotated.size == image.size
    assert annotated.mode == "RGB"
    assert image.mode == "L"  # original untouched
    # Something was actually drawn inside the reported region.
    assert annotated.crop((20, 20, 100, 60)).getcolors() != [(80 * 40, (0, 0, 0))]


def test_draw_judge_helmet_box_without_a_box_returns_the_image_unchanged():
    image = Image.new("RGB", (50, 50), color="white")

    annotated = draw_judge_helmet_box(image, None)

    assert annotated.size == (50, 50)
    assert annotated.getcolors() == [(50 * 50, (255, 255, 255))]


def test_normalized_box_to_pixels_scales_to_the_judged_image():
    # The judge normalizes against whatever image it was shown, so the same
    # normalized box lands differently on a crop than on a full frame.
    assert normalized_box_to_pixels([0.0, 0.0, 0.5, 0.5], (200, 100)).as_tuple() == (0, 0, 100, 50)
    assert normalized_box_to_pixels([0.0, 0.0, 0.5, 0.5], (1000, 1000)).as_tuple() == (0, 0, 500, 500)
