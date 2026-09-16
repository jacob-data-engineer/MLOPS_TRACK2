"""Per-violation crop strategy, and the batching rule that picks one crop
per person when multiple violations apply.

| Violation          | Pad factor (of own bbox)          |
|--------------------|------------------------------------|
| Glove / Helmet     | config.cropping.glove_helmet_pad   |
| DangerousPlane     | config.cropping.dangerous_plane_pad|
| SuspendedLoad      | config.cropping.suspended_load_pad |
| TooNearForklift    | union(person, forklift) + config.cropping.forklift_union_pad |

Pad factors live in config.py / config.toml, not as module constants —
see PipelineConfig.cropping.

ONE PERSON, OR SEVERAL

`crop_box_for_violation` / `crop_bytes` / `determine_crop` are the
single-person path and every existing caller uses them. `crops_for_frame` at
the bottom is the multi-person one: same crop boxes, but the frame is decoded
ONCE for N people, and each crop comes back as a `PersonCrop` carrying the
provenance needed to map a verdict back to the person, box, frame and source
video it came from. The two are kept separate rather than merged because
generalising the single-person path would make every caller that has exactly
one person pay for a loop it does not need.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from PIL import Image

from .config import PipelineConfig, get_config
from .models import BoundingBox, PersonCluster, ViolationType


@dataclass
class CropCandidate:
    violation_type: ViolationType
    box: BoundingBox
    # Which image this crop should come from — set from the driving
    # detection's frame_path for multi-frame batches; None for the
    # single-image pipeline, which falls back to its own image path.
    image_path: Path | None = None


def crop_box_for_violation(
    violation_type: ViolationType,
    person_box: BoundingBox,
    image_size: tuple[int, int],
    region_box: BoundingBox | None = None,
    forklift_box: BoundingBox | None = None,
    config: PipelineConfig | None = None,
    pad_override: float | None = None,
) -> BoundingBox:
    """The padded crop box for one violation type, independent of any
    PersonCluster — shared by the pipeline's per-person crop selection and
    the eval harness, which judges single labeled cases directly.

    `pad_override`, if given, replaces `cropping_config.glove_helmet_pad`
    for the GLOVE/HELMET branch specifically — used by the glove crop-
    tightness experiment (eval/runner.run_glove_crop_experiment) to vary
    pad without touching config.toml. Left `None` everywhere else, this is
    a no-op and every existing call site behaves exactly as before."""
    config = config or get_config()
    cropping_config = config.cropping

    if violation_type in (ViolationType.GLOVE, ViolationType.HELMET):
        base = region_box or person_box
        pad = cropping_config.glove_helmet_pad if pad_override is None else pad_override
        return base.expand(pad, clip_to=image_size)
    elif violation_type is ViolationType.DANGEROUS_PLANE:
        return person_box.expand(cropping_config.dangerous_plane_pad, clip_to=image_size)
    elif violation_type is ViolationType.SUSPENDED_LOAD:
        return person_box.expand(cropping_config.suspended_load_pad, clip_to=image_size)
    elif violation_type is ViolationType.TOO_NEAR_FORKLIFT:
        if forklift_box is None:
            raise ValueError("too_near_forklift requires a forklift_box")
        union = person_box.union_box(forklift_box)
        return union.expand(cropping_config.forklift_union_pad, clip_to=image_size)
    else:  # pragma: no cover - exhaustive over ViolationType
        raise ValueError(f"unhandled violation type: {violation_type}")


def determine_crop(
    cluster: PersonCluster,
    image_size: tuple[int, int],
    config: PipelineConfig | None = None,
) -> CropCandidate:
    """Pick the single crop to send to the judge for this person.

    Batching rule: when multiple violations apply, use the largest crop
    required among them — not the smallest. A wider crop still shows enough
    of the smaller region to judge it too, at a bit lower relative
    resolution, which is an acceptable tradeoff for one call instead of two.
    """
    config = config or get_config()
    candidates = []
    for violation_type in sorted(cluster.violations, key=lambda v: v.value):
        if violation_type is ViolationType.TOO_NEAR_FORKLIFT:
            det = cluster.detection_for_forklift_context()
        else:
            det = cluster.detection_for(violation_type)
        box = crop_box_for_violation(
            violation_type,
            det.person_box,
            image_size,
            region_box=det.region_box,
            forklift_box=det.forklift_box,
            config=config,
        )
        candidates.append(CropCandidate(violation_type=violation_type, box=box, image_path=det.frame_path))
    return max(candidates, key=lambda c: c.box.area)


def crop_bytes(image_bytes: bytes, box: BoundingBox, format: str = "JPEG") -> tuple[bytes, str]:
    """Crop `box` out of an already-loaded image and return (bytes, media_type)."""
    x1, y1, x2, y2 = box.as_tuple()
    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"crop box rounds to a zero-size region: {box} rounds to "
            f"({x1}, {y1}, {x2}, {y2}), width={x2 - x1}px, height={y2 - y1}px. This "
            "usually means a sub-pixel-width box was expanded with too small a pad "
            "(e.g. pad_override=0.0)."
        )
    with Image.open(io.BytesIO(image_bytes)) as image:
        cropped = image.convert("RGB").crop((x1, y1, x2, y2))
        buf = io.BytesIO()
        cropped.save(buf, format=format)
        return buf.getvalue(), f"image/{format.lower()}"


def full_image_bytes(image_bytes: bytes, format: str = "JPEG") -> tuple[bytes, str]:
    """No-crop baseline: the whole original image, for the cropped-vs-
    uncropped token/accuracy comparison in the eval harness."""
    with Image.open(io.BytesIO(image_bytes)) as image:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format=format)
        return buf.getvalue(), f"image/{format.lower()}"


def crop_image_to_bytes(image_path: Path, box: BoundingBox, format: str = "JPEG") -> tuple[bytes, str]:
    """Crop `box` out of the image at `image_path` and return (bytes, media_type).

    Thin wrapper around `crop_bytes` — kept for `pipeline.py`'s live
    single-image-upload call sites, which are genuinely local and never S3."""
    return crop_bytes(image_path.read_bytes(), box, format=format)


def full_image_to_bytes(image_path: Path, format: str = "JPEG") -> tuple[bytes, str]:
    """No-crop baseline: the whole original image, for the cropped-vs-
    uncropped token/accuracy comparison in the eval harness.

    Thin wrapper around `full_image_bytes` — kept for `pipeline.py`'s live
    single-image-upload call sites, which are genuinely local and never S3."""
    return full_image_bytes(image_path.read_bytes(), format=format)


def judge_box_to_original(box: Sequence[float], judged_region: BoundingBox) -> BoundingBox:
    """Map a judge-reported normalized box back into original-image pixels.

    The judge normalizes against the image it was handed, so its `[x1, y1,
    x2, y2]` in 0-1 means nothing without knowing which region of the
    original that image was. `judged_region` is that region in original
    coordinates — the crop box for a cropped call, the full image bounds
    for an uncropped one — which makes this the exact inverse of the crop
    and lets a box judged on a crop be drawn on, or stored against, the
    original frame.

    Inverts against `judged_region.as_tuple()`, not its raw floats:
    `crop_bytes` cuts on the rounded tuple, so the judged image's real
    pixel extent is the rounded one. Using the unrounded width here would
    offset every box by up to a pixel, in a direction that varies per case.
    """
    x1, y1, x2, y2 = judged_region.as_tuple()
    width, height = x2 - x1, y2 - y1
    nx1, ny1, nx2, ny2 = box
    return BoundingBox(
        x1=x1 + nx1 * width,
        y1=y1 + ny1 * height,
        x2=x1 + nx2 * width,
        y2=y1 + ny2 * height,
    )


# --- multi-person crops -----------------------------------------------------
#
# The spec's section 5: a frame holds several people, each needs their own
# crop, and every crop has to be traceable back to the person, the box, the
# frame and the source video it came from. What is NOT here is batched
# inference -- see `CropBatch`.


@dataclass(frozen=True)
class PersonCrop:
    """One person's crop, carrying everything needed to map a verdict back to
    where it came from.

    The provenance fields are the point. A crop on its own is an anonymous
    JPEG; the moment several of them exist per frame, "which answer belongs to
    which worker" stops being obvious and starts being a bug you find weeks
    later in a training set. `crop_label` is the human-facing name for one
    person within a frame ("Person 1"), `candidate_id` is the durable key, and
    `source_video`/`source_frame` close the chain back to the clip.
    """

    crop_label: str
    candidate_id: str
    person_index: int
    #: In ORIGINAL FRAME PIXELS, like every other box in this codebase, so a
    #: box the judge returns can be mapped back with `judge_box_to_original`.
    crop_box: BoundingBox
    image_bytes: bytes
    media_type: str
    original_width: int
    original_height: int
    #: Where the frame came from. `None` for callers that have no clip behind
    #: them (a manual upload), rather than a placeholder that reads as real.
    source_video: str | None = None
    source_frame: str | None = None


@dataclass(frozen=True)
class CropBatch:
    """Several crops from ONE frame, kept together.

    Today this is a grouping, not a batched model call: `helmet_cv.runner`
    uses it to decode the frame once and store it once for N people, and each
    crop is still judged by its own `run_track1_judge_inference` call.

    Batched VLM inference was measured and deferred rather than forgotten.
    At pilot scale the whole token bill is ~$0.40-$4.00 and batching saves
    ~$0.30, and because `helmet_cv.sample.is_sampled` keys on `candidate_id`,
    two people in one frame are independent draws -- so a "batch" would be
    size 1 almost every time. This type exists so that when the economics
    change, the association between a response and its person is already
    modelled and does not have to be retrofitted.
    """

    source_frame: str | None
    source_video: str | None
    crops: list[PersonCrop] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.crops)


def crops_for_frame(
    image_bytes: bytes,
    people: Sequence[dict],
    violation_type: ViolationType,
    config: PipelineConfig | None = None,
    *,
    format: str = "JPEG",
    full_frame_for_missing: bool = False,
) -> list[PersonCrop]:
    """One crop per person in one frame, decoding the frame ONCE.

    `people` is a sequence of dicts, each carrying at least `person_box` (four
    original-frame pixel values) and `candidate_id`; `person_index`,
    `source_video` and `source_frame` are read when present. A dict rather
    than a typed record because the callers are `helmet_cv`'s candidate rows,
    which are already plain JSON on disk.

    The single decode is the whole reason this exists. `crop_bytes` reopens
    the image for every call, which is correct for one crop and wasteful for
    four -- a 1920x1080 JPEG decoded four times to cut four small rectangles
    out of it. `crop_box_for_violation` and `crop_bytes` are deliberately
    untouched: every existing caller has exactly one person and would pay for
    a generalisation it does not need.

    A person whose box will not crop is SKIPPED, not fatal. One degenerate
    box out of a frame of four must not cost the other three, and the caller
    records the loss.
    """
    config = config or get_config()

    with Image.open(io.BytesIO(image_bytes)) as image:
        frame = image.convert("RGB")
        size = frame.size

        crops: list[PersonCrop] = []
        for order, person in enumerate(people):
            box_values = person.get("person_box")
            if not box_values and full_frame_for_missing:
                # No box means the detector found nobody, and the whole frame
                # is the only thing left to ask about. Not padded, because
                # there is nothing outside it to pad into.
                crop_box = BoundingBox(0, 0, size[0], size[1])
            elif not box_values:
                continue
            else:
                try:
                    person_box = BoundingBox(*(float(v) for v in box_values))
                    crop_box = crop_box_for_violation(
                        violation_type, person_box, size, config=config
                    )
                except ValueError:
                    # `BoundingBox` refuses a degenerate box outright, which
                    # is right for a single-person call where the caller
                    # wants to know. Here it would cost the other three
                    # people in the frame, so it costs only its own.
                    continue
            x1, y1, x2, y2 = crop_box.as_tuple()
            if x2 <= x1 or y2 <= y1:
                continue

            buf = io.BytesIO()
            frame.crop((x1, y1, x2, y2)).save(buf, format=format)
            index = person.get("person_index", order)
            crops.append(
                PersonCrop(
                    # 1-based: this is the label a reviewer reads, and
                    # "Person 0" is a sentence only a programmer writes.
                    crop_label=f"Person {index + 1}",
                    candidate_id=str(person.get("candidate_id") or ""),
                    person_index=index,
                    crop_box=crop_box,
                    image_bytes=buf.getvalue(),
                    media_type=f"image/{format.lower()}",
                    original_width=size[0],
                    original_height=size[1],
                    source_video=person.get("clip_key"),
                    source_frame=person.get("frame_name"),
                )
            )
    return crops
