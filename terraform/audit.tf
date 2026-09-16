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
    ]
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
