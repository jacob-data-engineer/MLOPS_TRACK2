"""Draw detection boxes onto a copy of the source image, for the UI — lets
you see exactly what a person_box/region_box/forklift_box lines up with
before spending a judge call on it.
"""

from __future__ import annotations

from typing import Sequence

from PIL import Image, ImageDraw

from .config import PipelineConfig, get_config
from .forklift_rule import evaluate_too_near_forklift
from .models import BoundingBox, Detection, ViolationType

_COLORS = {
    "glove": "#e74c3c",
    "helmet": "#f39c12",
    "dangerous_plane": "#8e44ad",
    "too_near_forklift": "#2980b9",
}
_DEFAULT_COLOR = "#2ecc71"

_FORKLIFT_COLORS = {
    "person": "#2980b9",
    "forklift": "#16a085",
    "person_expanded": "#f1c40f",
    "forklift_expanded": "#e91e63",
}


def draw_detections(
    image: Image.Image, detections: list[Detection], config: PipelineConfig | None = None
) -> Image.Image:
    """Return a new RGB image with each detection's boxes outlined:
    person_box (thick), plus region_box / forklift_box (thin) where
    present, color-coded by violation type, labeled with the violation
    type name. For too_near_forklift detections, draws both the original
    and gate-expanded person/forklift boxes (4 boxes total) plus a summary
    line with the gate's coverage ratio and operator/violation call."""
    config = config or get_config()
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)

    for det in detections:
        if det.violation_type is ViolationType.TOO_NEAR_FORKLIFT and det.forklift_box is not None:
            _draw_forklift_pair(draw, det, config)
            continue

        color = _COLORS.get(det.violation_type.value, _DEFAULT_COLOR)
        draw.rectangle(det.person_box.as_tuple(), outline=color, width=3)
        if det.region_box is not None:
            draw.rectangle(det.region_box.as_tuple(), outline=color, width=2)
        if det.forklift_box is not None:
            draw.rectangle(det.forklift_box.as_tuple(), outline=color, width=2)

        label_y = max(0, det.person_box.y1 - 14)
        draw.text((det.person_box.x1, label_y), det.violation_type.value, fill=color)

    return annotated


def _draw_forklift_pair(draw: ImageDraw.ImageDraw, det: Detection, config: PipelineConfig) -> None:
    """Draw both the original and expanded person/forklift boxes for a
    too_near_forklift detection, plus a summary line with the gate's own
    coverage ratio and operator/violation call — so a reviewer can see
    exactly what the geometric proximity gate reasoned over, not just its
    final verdict."""
    result = evaluate_too_near_forklift(det.person_box, det.forklift_box, config=config)
    expanded_person = det.person_box.expand(config.forklift.expand_pct)
    expanded_forklift = det.forklift_box.expand(config.forklift.expand_pct)

    boxes = [
        (det.person_box, "person", _FORKLIFT_COLORS["person"], 3),
        (det.forklift_box, "vehicle", _FORKLIFT_COLORS["forklift"], 2),
        (expanded_person, "person (expanded)", _FORKLIFT_COLORS["person_expanded"], 2),
        (expanded_forklift, "vehicle (expanded)", _FORKLIFT_COLORS["forklift_expanded"], 2),
    ]
    for box, label, color, width in boxes:
        draw.rectangle(box.as_tuple(), outline=color, width=width)
        label_y = max(0, box.y1 - 14)
        draw.text((box.x1, label_y), label, fill=color)

    if result.is_operator:
        containment = det.person_box.containment_ratio(det.forklift_box)
        status_text = f"OPERATOR (ambiguous — sent to judge) — containment={containment:.2f}"
    elif result.is_violation:
        status_text = f"VIOLATION — coverage={result.coverage_ratio:.2f}"
    else:
        status_text = f"no violation — coverage={result.coverage_ratio:.2f}"
    top_y = max(0, min(b.y1 for b, *_ in boxes) - 14)
    draw.text(
        (min(b.x1 for b, *_ in boxes), top_y),
        f"too_near_forklift — {status_text}",
        fill=_FORKLIFT_COLORS["person"],
    )


# The judge's own helmet box, drawn in a colour used by nothing else here:
# green reads as "cleared", and it must never be mistaken for the orange
# `_COLORS["helmet"]` box, which is the CV detector's head region, not the
# judge's finding.
_JUDGE_HELMET_COLOR = "#00e676"


def normalized_box_to_pixels(box: Sequence[float], size: tuple[int, int]) -> BoundingBox:
    """[x1, y1, x2, y2] in 0-1, relative to an image of `size`, as pixels.

    The judge normalizes against whatever image it was shown, so the caller
    passes the size of that same image — the crop for a cropped run, the
    full frame for an uncropped one — and gets back coordinates in that
    image's own space.
    """
    width, height = size
    x1, y1, x2, y2 = box
    return BoundingBox(x1=x1 * width, y1=y1 * height, x2=x2 * width, y2=y2 * height)


def draw_judge_helmet_box(
    image: Image.Image, helmet_box: Sequence[float] | None, label: str = "helmet (judge)"
) -> Image.Image:
    """Return a new RGB image with the judge's reported helmet box outlined.

    `helmet_box` is normalized to `image` itself (see
    `normalized_box_to_pixels`). `None` — every verdict that isn't a helmet
    cleared as "helmet_worn" — returns the image unchanged, so callers can
    pass a judge result through unconditionally.
    """
    annotated = image.convert("RGB")
    if helmet_box is None:
        return annotated

    annotated = annotated.copy()
    box = normalized_box_to_pixels(helmet_box, annotated.size)
    draw = ImageDraw.Draw(annotated)
    draw.rectangle(box.as_tuple(), outline=_JUDGE_HELMET_COLOR, width=3)

    # Helmet crops are narrow — a label left-aligned on the box runs off the
    # right edge often enough to be worth clamping, unlike draw_detections'
    # labels, which sit on person-width boxes.
    width, height = annotated.size
    label_x = max(0, min(box.x1, width - draw.textlength(label)))
    label_y = max(0, min(box.y1 - 14, height - 14))
    draw.text((label_x, label_y), label, fill=_JUDGE_HELMET_COLOR)
    return annotated
