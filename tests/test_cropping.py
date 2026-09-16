import io

import pytest
from PIL import Image

from ppe_pipeline.config import PipelineConfig
from ppe_pipeline.cropping import (
    CropBatch,
    crop_box_for_violation,
    crop_bytes,
    crop_image_to_bytes,
    crops_for_frame,
    determine_crop,
    full_image_bytes,
    full_image_to_bytes,
    judge_box_to_original,
)
from ppe_pipeline.models import BoundingBox, Detection, PersonCluster, ViolationType

IMAGE_SIZE = (2000, 2000)
PERSON = BoundingBox(500, 500, 700, 900)
HEAD = BoundingBox(550, 500, 650, 560)
FORKLIFT = BoundingBox(680, 500, 1000, 900)


def test_glove_helmet_crop_pads_the_region_box_not_the_person_box():
    cluster = PersonCluster(
        person_id="1",
        box=PERSON,
        detections=[Detection(ViolationType.HELMET, person_box=PERSON, region_box=HEAD, source="t")],
    )
    candidate = determine_crop(cluster, IMAGE_SIZE)
    expected = HEAD.expand(0.3)
    assert candidate.violation_type is ViolationType.HELMET
    assert candidate.box == expected


def test_dangerous_plane_crop_is_much_wider_than_the_person_box():
    cluster = PersonCluster(
        person_id="1",
        box=PERSON,
        detections=[Detection(ViolationType.DANGEROUS_PLANE, person_box=PERSON, source="t")],
    )
    candidate = determine_crop(cluster, IMAGE_SIZE)
    assert candidate.box.area > PERSON.area * 4


def test_suspended_load_crop_is_much_wider_than_the_person_box():
    cluster = PersonCluster(
        person_id="1",
        box=PERSON,
        detections=[Detection(ViolationType.SUSPENDED_LOAD, person_box=PERSON, source="t")],
    )
    candidate = determine_crop(cluster, IMAGE_SIZE)
    assert candidate.violation_type is ViolationType.SUSPENDED_LOAD
    assert candidate.box.area > PERSON.area * 4


def test_forklift_crop_is_union_of_person_and_forklift_padded():
    cluster = PersonCluster(
        person_id="1",
        box=PERSON,
        detections=[
            Detection(ViolationType.TOO_NEAR_FORKLIFT, person_box=PERSON, forklift_box=FORKLIFT, source="t")
        ],
    )
    candidate = determine_crop(cluster, IMAGE_SIZE)
    union = PERSON.union_box(FORKLIFT)
    assert candidate.box == union.expand(0.25)


def test_forklift_crop_uses_the_genuine_violation_forklift_not_a_first_listed_ambiguous_operator():
    # Same person flagged too_near_forklift against two different forklifts:
    # an ambiguous-operator one (listed first) and a genuine near-violation
    # one. The crop must be built from the genuine forklift's box, matching
    # cluster.detection_for_forklift_context() — not plain list order.
    ambiguous_forklift = BoundingBox(0, 0, 750, 950)  # operator-shaped: mostly contains PERSON
    detections = [
        Detection(
            ViolationType.TOO_NEAR_FORKLIFT,
            person_box=PERSON,
            forklift_box=ambiguous_forklift,
            source="t",
            forklift_is_ambiguous_operator=True,
            forklift_containment_ratio=1.0,
        ),
        Detection(ViolationType.TOO_NEAR_FORKLIFT, person_box=PERSON, forklift_box=FORKLIFT, source="t"),
    ]
    cluster = PersonCluster(person_id="1", box=PERSON, detections=detections)

    candidate = determine_crop(cluster, IMAGE_SIZE)

    genuine_union = PERSON.union_box(FORKLIFT)
    assert candidate.box == genuine_union.expand(0.25)
    wrong_union = PERSON.union_box(ambiguous_forklift)
    assert candidate.box != wrong_union.expand(0.25)


def test_batching_rule_picks_the_largest_crop_when_multiple_violations_apply():
    # Helmet crop (region-based, ~1.6x a small head box) is much smaller
    # than the too-near-forklift union crop — the union crop must win.
    cluster = PersonCluster(
        person_id="4",
        box=PERSON,
        detections=[
            Detection(ViolationType.HELMET, person_box=PERSON, region_box=HEAD, source="t"),
            Detection(ViolationType.TOO_NEAR_FORKLIFT, person_box=PERSON, forklift_box=FORKLIFT, source="t"),
        ],
    )
    candidate = determine_crop(cluster, IMAGE_SIZE)
    assert candidate.violation_type is ViolationType.TOO_NEAR_FORKLIFT


def test_pad_override_replaces_glove_helmet_pad():
    box = crop_box_for_violation(
        ViolationType.GLOVE, PERSON, IMAGE_SIZE, region_box=HEAD, pad_override=0.0
    )
    assert box == HEAD.expand(0.0)
    assert box != HEAD.expand(0.3)


def test_omitting_pad_override_preserves_existing_behavior():
    default = crop_box_for_violation(ViolationType.GLOVE, PERSON, IMAGE_SIZE, region_box=HEAD)
    explicit_none = crop_box_for_violation(
        ViolationType.GLOVE, PERSON, IMAGE_SIZE, region_box=HEAD, pad_override=None
    )
    assert default == HEAD.expand(0.3) == explicit_none


def test_pad_override_only_applies_to_glove_helmet_branch():
    # A pad_override passed for a dangerous_plane case must be ignored —
    # that branch only ever reads config.cropping.dangerous_plane_pad.
    with_override = crop_box_for_violation(
        ViolationType.DANGEROUS_PLANE, PERSON, IMAGE_SIZE, pad_override=0.0
    )
    without_override = crop_box_for_violation(ViolationType.DANGEROUS_PLANE, PERSON, IMAGE_SIZE)
    assert with_override == without_override


def test_crop_image_to_bytes_returns_jpeg_bytes(tmp_path):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (400, 400), color="red").save(image_path)

    box = BoundingBox(10, 10, 100, 100)
    data, media_type = crop_image_to_bytes(image_path, box)

    assert media_type == "image/jpeg"
    assert data[:2] == b"\xff\xd8"  # JPEG magic bytes


def test_crop_image_to_bytes_raises_a_clear_error_for_a_zero_size_rounded_crop(tmp_path):
    # Reproduces Tester's repro: pad_override=0.0 (the new "tight" glove
    # crop variant) on a sub-pixel-width box rounds both x-endpoints to the
    # same integer, which would otherwise fail deep inside PIL's .save()
    # with a cryptic "cannot write empty image" ValueError.
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (400, 400), color="red").save(image_path)

    tiny_region = BoundingBox(10.0, 10.0, 10.3, 20.0)  # sub-pixel width
    box = crop_box_for_violation(
        ViolationType.GLOVE, PERSON, IMAGE_SIZE, region_box=tiny_region, pad_override=0.0
    )
    assert box.width < 1  # still sub-pixel after the (no-op) pad

    with pytest.raises(ValueError, match="zero-size"):
        crop_image_to_bytes(image_path, box)


def _sample_image_bytes(size=(400, 400), color="red") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="PNG")
    return buf.getvalue()


def test_crop_bytes_returns_jpeg_bytes():
    box = BoundingBox(10, 10, 100, 100)
    data, media_type = crop_bytes(_sample_image_bytes(), box)

    assert media_type == "image/jpeg"
    assert data[:2] == b"\xff\xd8"  # JPEG magic bytes


def test_crop_bytes_raises_a_clear_error_for_a_zero_size_rounded_crop():
    tiny_region = BoundingBox(10.0, 10.0, 10.3, 20.0)  # sub-pixel width
    box = crop_box_for_violation(
        ViolationType.GLOVE, PERSON, IMAGE_SIZE, region_box=tiny_region, pad_override=0.0
    )
    assert box.width < 1  # still sub-pixel after the (no-op) pad

    with pytest.raises(ValueError, match="zero-size"):
        crop_bytes(_sample_image_bytes(), box)


def test_full_image_bytes_returns_jpeg_bytes():
    data, media_type = full_image_bytes(_sample_image_bytes())

    assert media_type == "image/jpeg"
    assert data[:2] == b"\xff\xd8"  # JPEG magic bytes


def test_crop_image_to_bytes_is_a_thin_wrapper_around_crop_bytes(tmp_path):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (400, 400), color="red").save(image_path)
    box = BoundingBox(10, 10, 100, 100)

    via_path = crop_image_to_bytes(image_path, box)
    via_bytes = crop_bytes(image_path.read_bytes(), box)

    assert via_path == via_bytes


def test_full_image_to_bytes_is_a_thin_wrapper_around_full_image_bytes(tmp_path):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (400, 400), color="red").save(image_path)

    via_path = full_image_to_bytes(image_path)
    via_bytes = full_image_bytes(image_path.read_bytes())

    assert via_path == via_bytes


def test_judge_box_to_original_maps_a_crop_relative_box_onto_the_full_frame():
    crop = BoundingBox(100, 50, 300, 250)  # a 200x200 crop at (100, 50)

    assert judge_box_to_original([0.25, 0.25, 0.75, 0.75], crop).as_tuple() == (150, 100, 250, 200)
    # The whole crop maps back to exactly the crop box.
    assert judge_box_to_original([0.0, 0.0, 1.0, 1.0], crop).as_tuple() == crop.as_tuple()


def test_judge_box_to_original_is_the_identity_for_an_uncropped_run():
    """An uncropped call's judged region is the whole image, so normalized
    coordinates scale straight onto it with no offset."""
    full = BoundingBox(0, 0, 1000, 800)

    assert judge_box_to_original([0.1, 0.2, 0.3, 0.4], full).as_tuple() == (100, 160, 300, 320)


def test_judge_box_to_original_inverts_against_the_crop_actually_cut():
    """`crop_bytes` cuts on the rounded tuple, so a crop box with fractional
    edges produces an image whose real extent is the rounded one. Inverting
    against the raw floats would offset the result."""
    image_bytes = _sample_image_bytes((1000, 1000))
    crop = BoundingBox(100.4, 50.6, 300.4, 250.6)  # rounds to (100, 51, 300, 251)

    cropped, _ = crop_bytes(image_bytes, crop)
    with Image.open(io.BytesIO(cropped)) as cropped_image:
        cropped_size = cropped_image.size

    mapped = judge_box_to_original([0.0, 0.0, 1.0, 1.0], crop)

    assert cropped_size == (200, 200)
    assert mapped.as_tuple() == (100, 51, 300, 251)
    assert (mapped.width, mapped.height) == cropped_size


def test_a_box_round_trips_through_a_real_crop():
    """The full loop: crop a region out, have the judge report a box
    normalized to that crop, and land back on the original pixels."""
    image_bytes = _sample_image_bytes((1000, 1000))
    crop = crop_box_for_violation(
        ViolationType.HELMET,
        BoundingBox(400, 300, 600, 700),
        (1000, 1000),
        region_box=BoundingBox(450, 310, 550, 400),
        config=PipelineConfig(),
    )
    cropped, _ = crop_bytes(image_bytes, crop)
    with Image.open(io.BytesIO(cropped)) as cropped_image:
        crop_width, crop_height = cropped_image.size

    # A helmet occupying the middle of the crop, expressed the way the judge
    # expresses it.
    original = judge_box_to_original([0.25, 0.25, 0.75, 0.75], crop)

    x1, y1, _, _ = crop.as_tuple()
    assert original.as_tuple() == (
        round(x1 + 0.25 * crop_width),
        round(y1 + 0.25 * crop_height),
        round(x1 + 0.75 * crop_width),
        round(y1 + 0.75 * crop_height),
    )
    # And it lands inside the crop it came from.
    assert crop.containment_ratio(original) == 1.0


# --- multi-person crops ---------------------------------------------------


def _frame_bytes(width=2000, height=2000) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color="white").save(buf, format="JPEG")
    return buf.getvalue()


def _person(candidate_id, box, index, **extra) -> dict:
    person = {
        "candidate_id": candidate_id,
        "person_index": index,
        "person_box": list(box),
        "clip_key": "raw_video/helmet_violation_2026-04-27_07-02-09_to_07-03-01_abc.mp4",
        "frame_name": "unk__unk__20260427T070224Z__abc__f000450.jpg",
    }
    person.update(extra)
    return person


def test_one_crop_per_person():
    crops = crops_for_frame(
        _frame_bytes(),
        [
            _person("c__p00", (500, 500, 700, 900), 0),
            _person("c__p01", (1000, 500, 1200, 900), 1),
        ],
        ViolationType.HELMET,
        PipelineConfig(),
    )

    assert [c.candidate_id for c in crops] == ["c__p00", "c__p01"]
    assert all(c.image_bytes for c in crops)
    # Different regions of the frame, not two copies of one crop.
    assert crops[0].crop_box != crops[1].crop_box
    assert crops[0].crop_box.x1 < crops[1].crop_box.x1


def test_a_crop_matches_the_single_person_path_exactly():
    """The multi-person path must not become a second cropping rule. Same
    boxes, same bytes -- the only difference is how many times the frame was
    decoded to produce them."""
    frame = _frame_bytes()
    config = PipelineConfig()
    expected_box = crop_box_for_violation(ViolationType.HELMET, PERSON, IMAGE_SIZE, config=config)
    expected_bytes, expected_media = crop_bytes(frame, expected_box)

    crop = crops_for_frame(frame, [_person("c__p00", PERSON.as_tuple(), 0)], ViolationType.HELMET, config)[0]

    assert crop.crop_box == expected_box
    assert crop.image_bytes == expected_bytes
    assert crop.media_type == expected_media


def test_the_frame_is_decoded_once_for_several_people(monkeypatch):
    """The whole reason this exists. `crop_bytes` reopens the image per call,
    which is right for one person and wasteful for four -- a 1920x1080 JPEG
    decoded four times to cut four small rectangles out of it."""
    import ppe_pipeline.cropping as cropping_mod

    opens = {"count": 0}
    real_open = cropping_mod.Image.open

    def counting_open(*args, **kwargs):
        opens["count"] += 1
        return real_open(*args, **kwargs)

    monkeypatch.setattr(cropping_mod.Image, "open", counting_open)

    crops_for_frame(
        _frame_bytes(),
        [_person(f"c__p{i:02d}", (500 + i * 100, 500, 600 + i * 100, 900), i) for i in range(4)],
        ViolationType.HELMET,
        PipelineConfig(),
    )

    assert opens["count"] == 1


def test_a_crop_carries_the_provenance_back_to_its_source():
    """A crop on its own is an anonymous JPEG. The moment several exist per
    frame, "which answer belongs to which worker" stops being obvious and
    starts being a bug found weeks later in a training set."""
    crop = crops_for_frame(
        _frame_bytes(), [_person("c__p02", PERSON.as_tuple(), 2)], ViolationType.HELMET, PipelineConfig()
    )[0]

    assert crop.candidate_id == "c__p02"
    assert crop.person_index == 2
    assert crop.crop_label == "Person 3"
    assert crop.source_video.startswith("raw_video/")
    assert crop.source_frame.endswith("__f000450.jpg")


def test_the_crop_label_is_one_based():
    """It is what a reviewer reads, and "Person 0" is a sentence only a
    programmer writes."""
    crops = crops_for_frame(
        _frame_bytes(),
        [_person("a", (500, 500, 700, 900), 0), _person("b", (900, 500, 1100, 900), 1)],
        ViolationType.HELMET,
        PipelineConfig(),
    )

    assert [c.crop_label for c in crops] == ["Person 1", "Person 2"]


def test_a_person_with_no_box_is_skipped_not_fatal():
    """One degenerate box out of a frame of three must not cost the other
    two."""
    crops = crops_for_frame(
        _frame_bytes(),
        [
            _person("a", (500, 500, 700, 900), 0),
            {"candidate_id": "b", "person_index": 1, "person_box": None},
            _person("c", (900, 500, 1100, 900), 2),
        ],
        ViolationType.HELMET,
        PipelineConfig(),
    )

    assert [c.candidate_id for c in crops] == ["a", "c"]


def test_a_box_that_rounds_to_nothing_is_skipped_rather_than_raising():
    """`crop_bytes` raises here, which is right for a single-person call
    where the caller wants to know. In a frame of four it would cost the
    other three."""
    crops = crops_for_frame(
        _frame_bytes(),
        [_person("a", (500.0, 500.0, 500.0, 500.0), 0), _person("b", (900, 500, 1100, 900), 1)],
        ViolationType.HELMET,
        PipelineConfig(),
    )

    assert [c.candidate_id for c in crops] == ["b"]


def test_no_people_yields_no_crops():
    assert crops_for_frame(_frame_bytes(), [], ViolationType.HELMET, PipelineConfig()) == []


def test_a_crop_batch_holds_a_frames_crops_together():
    """A grouping today, not a batched model call -- batching was measured
    and deferred. The type exists so the response-to-person association is
    already modelled when the economics change."""
    crops = crops_for_frame(
        _frame_bytes(), [_person("a", PERSON.as_tuple(), 0)], ViolationType.HELMET, PipelineConfig()
    )
    batch = CropBatch(source_frame="f.jpg", source_video="raw_video/x.mp4", crops=crops)

    assert len(batch) == 1
    assert batch.crops[0].candidate_id == "a"


def test_an_empty_crop_batch_has_no_crops():
    assert len(CropBatch(source_frame=None, source_video=None)) == 0
