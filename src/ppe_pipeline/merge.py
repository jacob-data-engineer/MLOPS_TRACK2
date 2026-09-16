"""Merge per-detector detections into canonical persons by spatial overlap.

Every detector numbers people independently, so detections are clustered by
overlap rather than by index. Plain NMS would keep one box per cluster and
silently drop the violation labels attached to the discarded boxes; this
merge unions every label in a cluster before picking a representative box.
"""

from __future__ import annotations

from .config import PipelineConfig, get_config
from .models import BoundingBox, Detection, PersonCluster


def _overlaps(a: BoundingBox, b: BoundingBox, threshold: float) -> bool:
    # containment_ratio (not IoU) so a hand/head box nested inside a much
    # larger full-body box still counts as the same person.
    return a.containment_ratio(b) >= threshold


def merge_detections(
    detections: list[Detection],
    overlap_threshold: float | None = None,
    config: PipelineConfig | None = None,
) -> list[PersonCluster]:
    """Cluster detections whose person boxes overlap into canonical persons.

    Clustering is greedy union-find: each detection joins the first existing
    cluster whose current representative box overlaps it, else starts a new
    cluster. The representative box is the largest person_box in the
    cluster, since a bigger box is more likely to be a full-body box than a
    hand/head sub-region.

    `overlap_threshold` overrides `config.merge.overlap_threshold` (which
    itself defaults to config.toml / built-in defaults — see config.py).
    """
    config = config or get_config()
    threshold = overlap_threshold if overlap_threshold is not None else config.merge.overlap_threshold

    clusters: list[list[Detection]] = []

    for det in detections:
        target = None
        for cluster in clusters:
            rep = _representative_box(cluster)
            if _overlaps(det.person_box, rep, threshold):
                target = cluster
                break
        if target is None:
            clusters.append([det])
        else:
            target.append(det)

    return [
        PersonCluster(
            person_id=str(i + 1),
            box=_representative_box(cluster),
            detections=cluster,
        )
        for i, cluster in enumerate(clusters)
    ]


def _representative_box(cluster: list[Detection]) -> BoundingBox:
    return max((d.person_box for d in cluster), key=lambda b: b.area)
