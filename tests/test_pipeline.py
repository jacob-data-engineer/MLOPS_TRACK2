import io
from datetime import datetime, timedelta

from PIL import Image, ImageDraw

from ppe_pipeline.batching import ImageFrame
from ppe_pipeline.judge_client import JudgeCallResult, JudgeResponse, JudgeResult, JudgeUsage
from ppe_pipeline.mock_data import generate_mock_detections
from ppe_pipeline.models import BoundingBox, Detection, PersonCluster, ViolationType
from ppe_pipeline.pipeline import run_pipeline, run_pipeline_for_frames


class FakeJudgeClient:
    """Stands in for JudgeClient so pipeline tests never hit the network."""

    def __init__(self):
        self.calls = []

    def judge(self, image_bytes, media_type, person_id, violation_types, extra_context=None):
        self.calls.append((person_id, violation_types, extra_context))
        response = JudgeResponse(
            person_id=person_id,
            results=[
                JudgeResult(
                    violation_type=v,
                    verdict="TP",
                    reason="clear_bare_hand" if v.value == "glove" else "bare_head",
                    confidence="high",
                    explanation="fake judge always confirms",
                )
                for v in violation_types
            ],
        )
        usage = JudgeUsage(input_tokens=len(image_bytes) // 10 + 100, output_tokens=50)
        return JudgeCallResult(response=response, usage=usage)


def test_run_pipeline_end_to_end_with_mock_detections(tmp_path):
    image_path = tmp_path / "scene.png"
    Image.new("RGB", (1200, 600), color="gray").save(image_path)

    detections = generate_mock_detections((1200, 600))
    fake_judge = FakeJudgeClient()

    result = run_pipeline(image_path, detections, fake_judge)

    assert len(result.people) == 4
    assert result.person_to_violations["2"] == ["too_near_forklift"]
    assert sorted(result.person_to_violations["1"]) == ["glove", "helmet"]
    assert sorted(result.person_to_violations["4"]) == ["helmet", "too_near_forklift"]

    assert result.violation_to_people["helmet"] == ["1", "4"]

    # Every call asked the judge about exactly that person's applicable violations.
    assert len(fake_judge.calls) == 4

    # Fake judge returns TP for everything, so nothing lands in review.
    assert result.review_queue == []
    total_tp = sum(len(p.processed.true_positives) for p in result.people)
    assert total_tp == 6  # 2 + 1 + 1 + 2 violations across the four people

    # Usage is tracked per person and aggregated across the whole pipeline run.
    assert result.total_usage.output_tokens == 4 * 50
    assert result.total_usage == sum((p.usage for p in result.people), JudgeUsage())


def test_run_pipeline_drops_too_near_forklift_candidates_that_fail_the_proximity_check(tmp_path):
    # Person and forklift boxes are nowhere near each other — this should
    # never reach the judge, regardless of what a detector labeled it.
    image_path = tmp_path / "scene.png"
    Image.new("RGB", (1000, 1000), color="gray").save(image_path)

    person = BoundingBox(0, 0, 100, 100)
    forklift = BoundingBox(800, 800, 950, 950)
    detections = [
        Detection(ViolationType.TOO_NEAR_FORKLIFT, person_box=person, forklift_box=forklift, source="t")
    ]
    fake_judge = FakeJudgeClient()

    result = run_pipeline(image_path, detections, fake_judge)

    assert result.people == []
    assert fake_judge.calls == []


def test_run_pipeline_sends_operator_shaped_forklift_candidates_to_the_judge(tmp_path):
    # Person box mostly inside the forklift box — geometry alone can't tell
    # a driver from someone standing dangerously on the vehicle, so this
    # should reach the judge instead of being auto-rejected.
    image_path = tmp_path / "scene.png"
    Image.new("RGB", (1000, 1000), color="gray").save(image_path)

    forklift = BoundingBox(0, 0, 300, 200)
    operator = BoundingBox(50, 20, 150, 180)  # almost entirely inside the forklift box
    detections = [
        Detection(ViolationType.TOO_NEAR_FORKLIFT, person_box=operator, forklift_box=forklift, source="t")
    ]
    fake_judge = FakeJudgeClient()

    result = run_pipeline(image_path, detections, fake_judge)

    assert len(result.people) == 1
    assert len(fake_judge.calls) == 1
    _, _, extra_context = fake_judge.calls[0]
    assert extra_context is not None
    assert "contained within the forklift's bounding box" in extra_context


def test_detection_for_forklift_context_prefers_genuine_over_ambiguous_operator():
    # Same person flagged too_near_forklift against two different forklifts:
    # one ambiguous-operator (high containment), one genuine near-violation.
    # List order puts the ambiguous one first — it must not win by default.
    person_box = BoundingBox(50, 20, 150, 180)
    ambiguous = Detection(
        ViolationType.TOO_NEAR_FORKLIFT,
        person_box=person_box,
        forklift_box=BoundingBox(0, 0, 300, 200),
        source="t",
        forklift_is_ambiguous_operator=True,
        forklift_containment_ratio=1.0,
    )
    genuine = Detection(
        ViolationType.TOO_NEAR_FORKLIFT,
        person_box=person_box,
        forklift_box=BoundingBox(160, 20, 260, 180),
        source="t",
    )
    cluster = PersonCluster(person_id="1", box=person_box, detections=[ambiguous, genuine])

    assert cluster.detection_for_forklift_context() is genuine


def test_detection_for_forklift_context_falls_back_to_ambiguous_operator_when_no_genuine_exists():
    person_box = BoundingBox(50, 20, 150, 180)
    ambiguous = Detection(
        ViolationType.TOO_NEAR_FORKLIFT,
        person_box=person_box,
        forklift_box=BoundingBox(0, 0, 300, 200),
        source="t",
        forklift_is_ambiguous_operator=True,
        forklift_containment_ratio=1.0,
    )
    cluster = PersonCluster(person_id="1", box=person_box, detections=[ambiguous])

    assert cluster.detection_for_forklift_context() is ambiguous


def test_run_pipeline_forklift_context_ignores_ambiguous_operator_when_genuine_violation_also_present(tmp_path):
    # Person is a probable operator of forklift A (high containment) but
    # also genuinely too near a second forklift B — the judge should get no
    # operator-context sentence, since a real violation exists in the same
    # cluster and must not be hidden by list order. The crop actually sent
    # to the judge must also be built from forklift B's box (the genuine
    # violation), not forklift A's — otherwise the real violation could end
    # up barely visible in the image the judge sees, with no context text
    # to compensate.
    image_path = tmp_path / "scene.png"
    image = Image.new("RGB", (2000, 2000), color="gray")
    draw = ImageDraw.Draw(image)
    person_box = BoundingBox(900, 900, 1100, 1300)
    forklift_a = BoundingBox(850, 850, 1150, 1350)  # operator-shaped: contains person_box
    forklift_b = BoundingBox(1150, 900, 1350, 1300)  # near but not containing: genuine violation
    draw.rectangle(forklift_a.as_tuple(), fill=(0, 0, 255))  # blue
    draw.rectangle(forklift_b.as_tuple(), fill=(255, 0, 0))  # red
    image.save(image_path)

    detections = [
        Detection(ViolationType.TOO_NEAR_FORKLIFT, person_box=person_box, forklift_box=forklift_a, source="t"),
        Detection(ViolationType.TOO_NEAR_FORKLIFT, person_box=person_box, forklift_box=forklift_b, source="t"),
    ]
    fake_judge = FakeJudgeClient()

    result = run_pipeline(image_path, detections, fake_judge)

    assert len(result.people) == 1
    assert len(fake_judge.calls) == 1
    _, _, extra_context = fake_judge.calls[0]
    assert extra_context is None

    # Geometry check: the crop must match forklift B's (genuine) union box,
    # not forklift A's (ambiguous, listed first) — the two differ enough in
    # size/position to tell apart.
    genuine_crop_box = person_box.union_box(forklift_b).expand(0.25)
    x1, y1, x2, y2 = genuine_crop_box.as_tuple()
    crop = Image.open(io.BytesIO(result.people[0].crop_bytes))
    assert crop.size == (x2 - x1, y2 - y1)

    # Content check: a point that's inside forklift B's painted region and
    # inside the genuine crop, but would fall outside the crop the old
    # (buggy) forklift-A-first selection would have produced, must show
    # forklift B's red paint — not gray background or forklift A's blue.
    point_x, point_y = 1300, 1100
    local_x, local_y = point_x - x1, point_y - y1
    r, g, b = crop.convert("RGB").getpixel((local_x, local_y))
    assert r > 180 and g < 80 and b < 80


def test_run_pipeline_for_frames_merges_the_same_person_across_a_camera_burst(tmp_path):
    # Same fixed camera, two near-simultaneous frames: helmet detected in
    # the first, glove detected in the second, boxes overlapping enough to
    # be the same physical person once merged across frames.
    frame1_path = tmp_path / "frame1.png"
    frame2_path = tmp_path / "frame2.png"
    Image.new("RGB", (1000, 1000), color="gray").save(frame1_path)
    Image.new("RGB", (1000, 1000), color="gray").save(frame2_path)

    person_box_1 = BoundingBox(100, 100, 300, 500)
    head_1 = BoundingBox(150, 100, 250, 160)
    person_box_2 = BoundingBox(105, 100, 305, 500)  # same person, slight jitter
    hand_2 = BoundingBox(110, 400, 160, 450)

    frame1 = ImageFrame(
        path=frame1_path,
        timestamp=datetime(2026, 7, 1, 12, 0, 0),
        detections=[
            Detection(ViolationType.HELMET, person_box=person_box_1, region_box=head_1, source="helmet_clf")
        ],
    )
    frame2 = ImageFrame(
        path=frame2_path,
        timestamp=datetime(2026, 7, 1, 12, 0, 1),
        detections=[
            Detection(ViolationType.GLOVE, person_box=person_box_2, region_box=hand_2, source="glove_clf")
        ],
    )

    fake_judge = FakeJudgeClient()
    result = run_pipeline_for_frames([frame1, frame2], fake_judge)

    assert len(result.people) == 1
    person = result.people[0]
    assert sorted(v.value for v in person.cluster.violations) == ["glove", "helmet"]

    # Each contributing detection still remembers which frame it came from.
    frame_names = {d.frame_path.name for d in person.cluster.detections}
    assert frame_names == {"frame1.png", "frame2.png"}
