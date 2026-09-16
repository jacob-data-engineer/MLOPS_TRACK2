# --- Model Package Groups: one per YOLO11 size variant, where
# RegisterModel puts each trained candidate for that size. Distinct from
# Track2's real track2-*-models groups. -----------------------------------

locals {
  # Uniform yolo11<size>-detector naming (must match model_training/src/
  # config.py's model_package_group_name exactly). "s" used to keep an
  # irregular pre-existing name ("track2-yolo11", no size suffix) because
  # Model Package Group names are immutable and renaming means
  # delete-and-recreate, losing every package already registered in it --
  # that history was deliberately cleared and everything (now including
  # "n", never onboarded before) re-registered fresh under this scheme.
  # No "n": the nano group was never created in AWS and nothing uses it.
  # The app serves s/m/l/x only (track2_shared.py's COMPUTE_REQUIREMENTS /
  # REGISTRY_SIZES), so declaring it here just made every plan want to
  # create an empty group, and widened the IAM/event-rule members below to
  # a package group that does not exist.
  yolo_size_group_names = {
    s = "yolo11s-detector"
    m = "yolo11m-detector"
    l = "yolo11l-detector"
    x = "yolo11x-detector"
  }

  # model_package_group_description is ForceNew (confirmed via `terraform
  # plan`: changing it showed "must be replaced", deleting every
  # already-registered package in that group) -- consistent wording across
  # all four served sizes.
  yolo_size_group_descriptions = {
    s = "YOLO11-s hazard-detection candidates for the Track2 demo pipeline (curation -> train -> register -> Model Monitor -> A2I -> endpoint)"
    m = "YOLO11-m hazard-detection candidates for the Track2 demo pipeline (curation -> train -> register -> Model Monitor -> A2I -> endpoint)"
    l = "YOLO11-l hazard-detection candidates for the Track2 demo pipeline (curation -> train -> register -> Model Monitor -> A2I -> endpoint)"
    x = "YOLO11-x hazard-detection candidates for the Track2 demo pipeline (curation -> train -> register -> Model Monitor -> A2I -> endpoint)"
  }
}

moved {
  from = aws_sagemaker_model_package_group.yolo
  to   = aws_sagemaker_model_package_group.yolo["s"]
}

resource "aws_sagemaker_model_package_group" "yolo" {
  for_each = local.yolo_size_group_names

  model_package_group_name        = each.value
  model_package_group_description = local.yolo_size_group_descriptions[each.key]
}

# NOTE: this file originally also created an Amazon A2I human-review loop
# (aws_sagemaker_workforce/workteam/human_task_ui/flow_definition). Both
# SageMaker private-workforce management AND Amazon A2I have stopped
# accepting new customers account-wide (confirmed via AWS's own docs,
# and via a real CreateWorkforce/CreateHumanTaskUi 400 ValidationException
# on this account) — not fixable by retrying or reconfiguring. Replaced
# with a custom review-queue + notification + small web app; see
# review_app.tf and monitoring.tf.
