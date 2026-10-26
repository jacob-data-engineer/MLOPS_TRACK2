"""Bucket/region/role/instance-type tunables — mirrors Track2's own
config.py/config.toml pattern. Values here match the Terraform outputs
from ../../terraform/platform; re-run `terraform output` there if any of
these drift.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    region: str = "ap-southeast-1"
    account_id: str = "125192671932"
    project: str = "track2-yolo"

    pipeline_bucket: str = "track2-s3"
    source_bucket: str = "safety-model-dataset-dev"
    source_images_prefix: str = "resources/hazard_images/Annotated/testing_set/images/"

    sagemaker_execution_role_arn: str = (
        "arn:aws:iam::125192671932:role/track2-yolo-sagemaker-execution"
    )
    training_image_uri: str = (
        "125192671932.dkr.ecr.ap-southeast-1.amazonaws.com/track2-yolo-train:train-latest"
    )
    serving_image_uri: str = (
        "125192671932.dkr.ecr.ap-southeast-1.amazonaws.com/track2-yolo-train:serve-latest"
    )
    training_instance_type: str = "ml.g4dn.xlarge"

    # The one shared endpoint hosting all four sizes as separate Inference
    # Components -- see model_training/src/deploy_inference_component.py.
    # Fully separate from the single-model track2-yolo-endpoint
    # (Asynchronous Inference) that deploy_endpoint.py manages.
    multi_size_endpoint_name: str = "track2-yolo-multi-size-endpoint"
    multi_size_endpoint_config_name: str = "track2-yolo-multi-size-config"
    multi_size_variant_name: str = "AllSizes"
    multi_size_instance_type: str = "ml.m5.2xlarge"

    data_quality_monitor_schedule_name: str = "track2-yolo-data-quality-monitor"

    # filename category tag -> YOLO class id, matching Track1/JP_UI's own
    # hazard taxonomy (HazardType/ViolationType)
    class_map: dict = None

    def __post_init__(self):
        if self.class_map is None:
            object.__setattr__(
                self,
                "class_map",
                {"Helmet": 0, "Gloves": 1, "Vehicles": 2, "FallZone": 3},
            )


# The four sizes with a live serving Inference Component (see
# deploy_inference_component.py / review_app.py's COMPUTE_REQUIREMENTS) --
# narrower than "sizes with a registered baseline", which also includes
# Nano (see register_baseline_model.py's own REGISTERABLE_SIZES).
SIZES = ("s", "m", "l", "x")

# YOLO11 and YOLO26. YOLOv8/9/10/12 are deliberately excluded, not just
# "not yet added" (YOLO12 has no registered baseline and is not wanted
# here regardless). YOLO26 was excluded on the same footing until
# 2026-09-30 and is now in. Nano also
# dropped from every generation's size options -- by policy, not because
# those combinations are invalid Ultralytics releases. Must match
# terraform/review-app/review_app.py's GENERATION_SIZES exactly (that app
# doesn't import this module -- its own deployed source is just the one
# file, see terraform/scripts/deploy_review_app.sh -- so the two copies
# are kept in sync by convention).
GENERATION_SIZES = {
    # YOLO26 added 2026-09-30. Needs ultralytics >= 8.4 in the training
    # image -- 8.3 has no yolo26 cfg at all, so a run would die at model
    # construction. See training-container/Dockerfile.
    "11": ("s", "m", "l", "x"),
    "26": ("s", "m", "l", "x"),
}


def model_package_group_name(generation: str, size: str) -> str:
    """Uniform Model Package Group naming for every generation+size:
    yolo<generation><size>-detector. Must match
    infra/sagemaker.tf's yolo_size_group_names exactly for generation "11"
    (the only generation Terraform pre-declares groups for; every other
    generation gets this same pattern created on demand -- see
    pipeline_definition.py's _resolve_model_package_group_name). There used
    to be an irregular exception here (generation 11 size "s" kept a
    pre-existing "track2-yolo11" name with no size suffix, since Model
    Package Group names are immutable and renaming would have deleted its
    real registered history) -- that history was deliberately cleared and
    everything re-registered under this uniform scheme instead, so the
    exception is gone."""
    return f"yolo{generation}{size}-detector"


CLASS_NAMES = ["helmet", "gloves", "vehicle", "fall_zone"]

CONFIG = Config()
