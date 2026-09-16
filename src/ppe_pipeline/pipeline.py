"""End-to-end orchestration: merge -> dictionary -> crop -> judge -> verdicts.

Repeats the crop-and-judge step for every canonical person, per the
architecture doc's pipeline overview. `run_pipeline` covers the
single-image case; `run_pipeline_for_frames` / `run_pipeline_for_batch`
cover multiple near-simultaneous captures from one fixed camera (see
batching.py) — the same camera means cross-frame detections can be merged
spatially just like within one image.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .batching import FrameBatch, ImageFrame
from .config import PipelineConfig, get_config
from .cropping import crop_image_to_bytes, determine_crop
from .dictionary import build_person_violation_dict, invert_violations
from .forklift_rule import filter_too_near_forklift_detections
from .judge_client import JudgeClient, JudgeUsage
from .judge_prompt import build_forklift_operator_context
from .merge import merge_detections
from .models import Detection, PersonCluster, ViolationType
from .verdicts import ProcessedVerdicts, ReviewQueueEntry, process_verdicts


@dataclass
class PersonResult:
    cluster: PersonCluster
    processed: ProcessedVerdicts
    usage: JudgeUsage
    # The exact crop bytes sent to the judge for this person — lets a UI
    # show what the judge actually saw, not just its verdict.
    crop_bytes: bytes
    crop_media_type: str


@dataclass
class PipelineResult:
    person_to_violations: dict[str, list[str]]
    violation_to_people: dict[str, list[str]]
    people: list[PersonResult]
    review_queue: list[ReviewQueueEntry]
    total_usage: JudgeUsage


def run_pipeline(
    image_path: Path,
    detections: list[Detection],
    judge_client: JudgeClient,
    config: PipelineConfig | None = None,
) -> PipelineResult:
    with Image.open(image_path) as image:
        image_size = image.size
    detections = filter_too_near_forklift_detections(detections, config=config)
    clusters = merge_detections(detections, config=config)
    return _process_clusters(clusters, image_size, judge_client, default_image_path=image_path, config=config)


def run_pipeline_for_frames(
    frames: list[ImageFrame],
    judge_client: JudgeClient,
    config: PipelineConfig | None = None,
) -> PipelineResult:
    """Same as `run_pipeline`, but for several near-simultaneous frames from
    one fixed camera (e.g. one `FrameBatch` from batching.py). Detections
    across all frames are merged together, and each person's crop is taken
    from whichever frame produced the detection that drove the crop choice.
    """
    if not frames:
        raise ValueError("run_pipeline_for_frames requires at least one frame")

    with Image.open(frames[0].path) as image:
        image_size = image.size

    all_detections = [d for frame in frames for d in frame.detections]
    all_detections = filter_too_near_forklift_detections(all_detections, config=config)
    clusters = merge_detections(all_detections, config=config)
    return _process_clusters(
        clusters, image_size, judge_client, default_image_path=frames[0].path, config=config
    )


def run_pipeline_for_batch(
    batch: FrameBatch,
    judge_client: JudgeClient,
    config: PipelineConfig | None = None,
) -> PipelineResult:
    return run_pipeline_for_frames(batch.frames, judge_client, config=config)


def _process_clusters(
    clusters: list[PersonCluster],
    image_size: tuple[int, int],
    judge_client: JudgeClient,
    default_image_path: Path,
    config: PipelineConfig | None,
) -> PipelineResult:
    config = config or get_config()
    person_to_violations = build_person_violation_dict(clusters)
    violation_to_people = invert_violations(person_to_violations)

    people: list[PersonResult] = []
    review_queue: list[ReviewQueueEntry] = []
    total_usage = JudgeUsage()

    for cluster in clusters:
        candidate = determine_crop(cluster, image_size, config=config)
        crop_source = candidate.image_path or default_image_path
        crop_bytes, media_type = crop_image_to_bytes(crop_source, candidate.box)

        violation_types = sorted(cluster.violations, key=lambda v: v.value)
        extra_context = None
        if ViolationType.TOO_NEAR_FORKLIFT in cluster.violations:
            forklift_det = cluster.detection_for_forklift_context()
            if forklift_det.forklift_is_ambiguous_operator:
                extra_context = build_forklift_operator_context(forklift_det.forklift_containment_ratio)

        call = judge_client.judge(
            image_bytes=crop_bytes,
            media_type=media_type,
            person_id=cluster.person_id,
            violation_types=violation_types,
            extra_context=extra_context,
        )

        crop_reference = f"{crop_source.name}:person_{cluster.person_id}:{candidate.box.as_tuple()}"
        processed = process_verdicts(call.response, crop_reference)

        people.append(
            PersonResult(
                cluster=cluster,
                processed=processed,
                usage=call.usage,
                crop_bytes=crop_bytes,
                crop_media_type=media_type,
            )
        )
        review_queue.extend(processed.review_queue)
        total_usage = total_usage + call.usage

    return PipelineResult(
        person_to_violations=person_to_violations,
        violation_to_people=violation_to_people,
        people=people,
        review_queue=review_queue,
        total_usage=total_usage,
    )
