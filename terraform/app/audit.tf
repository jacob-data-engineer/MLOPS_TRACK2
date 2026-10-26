# The training audit table, and the app's access to it.
#
# Both live here rather than in the stack that owns the EC2 role: this is
# a new table created for a feature of this app, and attaching its own
# policy to the existing role adds a resource instead of editing one that
# stack manages. Role policy attachments are not exhaustive, so the other
# stack stays authoritative for everything it already grants.

# Keyed by its own generated id, not by the CodeBuild build id or the
# pipeline execution arn: neither exists yet at the moment the row is
# written (the execution is created later, from inside the build), and a
# submission that never becomes an execution still has to be auditable.
resource "aws_dynamodb_table" "training_audit" {
  name         = var.training_audit_table
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "id"

  attribute {
    name = "id"
    type = "S"
  }
}

# One row per FILE a run used, rather than the file list living on the
# run's own row. A DynamoDB item is capped at 400KB, and at ~101 bytes per
# filename a 24,000-image dataset needs 2.4MB -- six times over. Exceeding
# it does not truncate the list, it rejects the whole write, so the audit
# row would be lost because its file list was too long.
#
# Per-file rows have no such ceiling, and the GSI below is the reason to
# prefer this over chunking: it answers "which runs used this image?",
# which a blob of names cannot.
resource "aws_dynamodb_table" "training_files" {
  name         = var.training_files_table
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "audit_id"
  range_key    = "file_key"

  attribute {
    name = "audit_id"
    type = "S"
  }
  # "<role>#<zero-padded index>", e.g. train#000042. Padded so the sort
  # order is the listing order, and prefixed so one Query can fetch just
  # the training files or just the evaluation ones.
  attribute {
    name = "file_key"
    type = "S"
  }
  attribute {
    name = "name"
    type = "S"
  }

  # Reverse lineage. KEYS_ONLY because the answer wanted is which runs,
  # and the keys already carry the audit id -- projecting more would
  # duplicate every filename a second time for nothing.
  global_secondary_index {
    name            = "by-file-name"
    hash_key        = "name"
    range_key       = "audit_id"
    projection_type = "KEYS_ONLY"
  }
}

# One row per FROZEN dataset definition. A generated data.yaml names week
# folders in place, so it pins nothing -- the same yaml trains on more
# images next month, because the pull keeps filling those folders. A
# snapshot records the yaml, the per-week counts at that moment, and a
# pointer to an S3 manifest naming every file it selected; the manifest is
# what makes a run reproducible, because data.yaml can only name
# DIRECTORIES and has no way to say "these files and not the others".
#
# No range key: the hash is over the yaml text AND the sorted file list
# together, so two snapshots share an id only when a run against either
# would be the same run. Re-freezing an unchanged dataset therefore
# overwrites its own row rather than growing the table.
resource "aws_dynamodb_table" "dataset_snapshots" {
  name         = var.dataset_snapshot_table
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "snapshot_id"

  attribute {
    name = "snapshot_id"
    type = "S"
  }
}

data "aws_iam_policy_document" "training_audit" {
  statement {
    sid    = "TrainingAuditReadWrite"
    effect = "Allow"
    actions = [
      "dynamodb:PutItem",
      "dynamodb:GetItem",
      "dynamodb:UpdateItem",
      "dynamodb:Scan",
      "dynamodb:Query",
      "dynamodb:DeleteItem",
    ]
    resources = [aws_dynamodb_table.training_audit.arn]
  }
  statement {
    sid    = "TrainingFilesReadWrite"
    effect = "Allow"
    actions = [
      "dynamodb:BatchWriteItem",
      "dynamodb:PutItem",
      "dynamodb:Query",
      "dynamodb:DeleteItem",
    ]
    # The index needs its own grant: a Query against a GSI is authorised
    # on the index ARN, not the table's.
    resources = [
      aws_dynamodb_table.training_files.arn,
      "${aws_dynamodb_table.training_files.arn}/index/*",
    ]
  }
  statement {
    sid    = "DatasetSnapshotsReadWrite"
    effect = "Allow"
    actions = [
      "dynamodb:PutItem",
      "dynamodb:GetItem",
      "dynamodb:Scan",
      "dynamodb:DeleteItem",
    ]
    # Scanned rather than queried: one row per saved definition is tens
    # of rows, not millions, and a GSI to sort by date would cost more
    # to keep than the scan it saves.
    resources = [aws_dynamodb_table.dataset_snapshots.arn]
  }
}

resource "aws_iam_policy" "training_audit" {
  name   = "${var.app_name}-training-audit"
  policy = data.aws_iam_policy_document.training_audit.json
}

resource "aws_iam_role_policy_attachment" "training_audit" {
  role       = data.aws_iam_role.review_host.name
  policy_arn = aws_iam_policy.training_audit.arn
}

# Two grants the host role never had, both found by the UI failing:
#
#   Stop Training returned AccessDenied on StopPipelineExecution -- the
#   button existed and called the API, but nothing allowed the action, so
#   a run could be started from here and not stopped from here.
#
#   Deleting a run from the training history deletes the model package it
#   produced, which is the point of the delete -- a history entry whose
#   model outlived it would be a record of something that no longer
#   matches what is in the registry.
data "aws_iam_policy_document" "training_control" {
  statement {
    sid    = "ControlOwnPipelineExecutions"
    effect = "Allow"
    actions = [
      "sagemaker:StopPipelineExecution",
      "sagemaker:DescribePipelineExecution",
      "sagemaker:ListPipelineExecutionSteps",
      # Only the custom pipeline is started from the app. Production runs
      # go through the replay coordinator, which owns the snapshot
      # bookkeeping -- starting that one from here would bypass it.
      "sagemaker:StartPipelineExecution",
    ]
    resources = [
      "arn:aws:sagemaker:${var.region}:${local.account_id}:pipeline/${var.training_pipeline_name}",
      "arn:aws:sagemaker:${var.region}:${local.account_id}:pipeline/${var.training_pipeline_name}/execution/*",
      "arn:aws:sagemaker:${var.region}:${local.account_id}:pipeline/${var.custom_pipeline_name}",
      "arn:aws:sagemaker:${var.region}:${local.account_id}:pipeline/${var.custom_pipeline_name}/execution/*",
      # The pipeline custom-class runs are actually submitted to. Adding a
      # pipeline the app starts without adding it here fails at
      # StartPipelineExecution with AccessDenied, which is how this one was
      # found -- the grant and the code that needs it live in different
      # stacks, so they have to be changed together.
      "arn:aws:sagemaker:${var.region}:${local.account_id}:pipeline/${var.open_pipeline_name}",
      "arn:aws:sagemaker:${var.region}:${local.account_id}:pipeline/${var.open_pipeline_name}/execution/*",
    ]
  }
  # "Stop Training" can end a GPU job this app did not start.
  #
  # The account has ONE ml.g4dn.xlarge on-demand slot, and the single-GPU
  # guard refuses to start a run while anything holds it -- including a
  # job from another system. Without this grant the app could SEE that
  # job, name it in the status bar, and do nothing about it, so the slot
  # could stay blocked for hours with no control anywhere on the page.
  #
  # WHY THE RESOURCE IS A WILDCARD, when every other statement here is
  # scoped: a foreign job is by definition one this app did not name, so
  # there is no prefix to scope to. IAM cannot condition on instance
  # type either. The narrowing that does exist is in the code --
  # `stop_foreign_gpu_jobs` only ever stops jobs on GPU_INSTANCE_TYPE --
  # and in the UI, which names each job and asks before sending
  # `include_foreign`. That is a weaker boundary than an ARN scope, and
  # worth knowing when deciding whether to apply this.
  statement {
    sid    = "StopAnyGpuTrainingJob"
    effect = "Allow"
    actions = [
      "sagemaker:StopTrainingJob",
      # Already used by get_gpu_resource_usage to read instance type and
      # EnableManagedSpotTraining; listed here so the statement stands on
      # its own if the broader describe grant is ever narrowed.
      "sagemaker:DescribeTrainingJob",
    ]
    resources = [
      "arn:aws:sagemaker:${var.region}:${local.account_id}:training-job/*",
    ]
  }
  statement {
    sid    = "DeleteRegisteredModelPackage"
    effect = "Allow"
    actions = [
      "sagemaker:DeleteModelPackage",
      "sagemaker:DescribeModelPackage",
    ]
    # Scoped to this project's own YOLO detector groups -- a wildcard over
    # every model package in the account is far more than a delete button
    # on this page should be able to reach.
    resources = [
      "arn:aws:sagemaker:${var.region}:${local.account_id}:model-package/yolo*-detector/*",
      "arn:aws:sagemaker:${var.region}:${local.account_id}:model-package/track2-yolo-custom-detector/*",
      # The full-frame replay group. The training history lists runs that
      # registered here, and reading a run's metrics, classes or training
      # data all go through DescribeModelPackage -- without this the app
      # could see the run but not describe the version it produced.
      "arn:aws:sagemaker:${var.region}:${local.account_id}:model-package/track2-ppe-fullframe-replay/*",
    ]
  }
  statement {
    sid    = "CreateDetectorRegistryGroups"
    effect = "Allow"
    actions = [
      "sagemaker:CreateModelPackageGroup",
      "sagemaker:DescribeModelPackageGroup",
    ]
    # Adding a detector in the UI has to create its per-size registry
    # groups. Registering does NOT create them implicitly -- the pipeline's
    # RegisterModel step fails outright on a missing group, at the very end
    # of the run, after the GPU hours are spent.
    #
    # Note the resource TYPE: `model-package-group/`, not `model-package/`.
    # The delete grant above is on the packages; this is on the containers,
    # and a group ARN has no trailing version segment.
    resources = [
      "arn:aws:sagemaker:${var.region}:${local.account_id}:model-package-group/yolo*-detector",
    ]
  }
  statement {
    sid    = "RegisterImportedCheckpoint"
    effect = "Allow"
    actions = [
      "sagemaker:CreateModelPackage",
    ]
    # Importing a pretrained .pt as a version. Registration otherwise only
    # ever happens inside the pipeline, under the SageMaker execution role
    # -- this is the one path where the app itself creates a version.
    #
    # Deliberately NOT granted on the four production `yolo11{s,m,l,x}-
    # detector` groups: approving a version there fires the rule that
    # swaps the live inference weights, so an import must never be able to
    # land in one. register_uploaded_base_model refuses the built-in
    # detector for the same reason; this is the grant that makes that
    # refusal enforceable rather than merely polite.
    resources = [
      "arn:aws:sagemaker:${var.region}:${local.account_id}:model-package/yolo*-*-detector/*",
    ]
  }
  statement {
    sid    = "ReadServeImageForRegistration"
    effect = "Allow"
    actions = [
      "ecr:BatchGetImage",
      "ecr:DescribeImages",
      "ecr:GetDownloadUrlForLayer",
      "ecr:BatchCheckLayerAvailability",
    ]
    # CreateModelPackage validates the InferenceSpecification's container
    # image using the CALLER's credentials, not SageMaker's own. Without
    # this the call fails with "Access denied for registry ID ... please
    # check if your ECR image exists and has proper pull permissions",
    # which reads like a missing image and is not one -- serve-latest is
    # present; the role simply could not see it.
    #
    # Read-only and scoped to the one repository whose image these
    # packages reference.
    resources = [
      "arn:aws:ecr:${var.region}:${local.account_id}:repository/track2-yolo-train",
    ]
  }
  statement {
    sid    = "RestartEmptyRegistryNumbering"
    effect = "Allow"
    actions = [
      "sagemaker:DeleteModelPackageGroup",
    ]
    # Deleting an EMPTY group and letting it be recreated is the only way
    # to put a registry back to v1: SageMaker version numbers are
    # monotonic per group and never reused, so a registry emptied of
    # v1..v4 shows its next import as v5. The code only calls this when
    # the group holds nothing, so it can destroy no versions.
    #
    # Scoped with TWO hyphens -- `yolo*-*-detector` matches
    # yolo11s-helmet-detector but cannot match yolo11s-detector, so the
    # four production groups are outside this grant entirely. The create
    # grant above is deliberately wider; a delete must not be.
    resources = [
      "arn:aws:sagemaker:${var.region}:${local.account_id}:model-package-group/yolo*-*-detector",
    ]
  }
  statement {
    sid     = "EcrAuthToken"
    effect  = "Allow"
    actions = ["ecr:GetAuthorizationToken"]
    # The one ECR action that takes no resource -- it only mints a token,
    # and reading anything still requires the repository grant above.
    resources = ["*"]
  }
}

resource "aws_iam_policy" "training_control" {
  name   = "${var.app_name}-training-control"
  policy = data.aws_iam_policy_document.training_control.json
}

resource "aws_iam_role_policy_attachment" "training_control" {
  role       = data.aws_iam_role.review_host.name
  policy_arn = aws_iam_policy.training_control.arn
}
