# Weekly, automatic retrigger of the training pipeline — Sunday nights,
# Asia/Singapore time.
#
# THIRD approach tried for this, and the one that stuck — see
# brainstorming.md for the full trail:
#   1. EventBridge Scheduler's templated SageMaker target
#      (SageMakerPipelineParameters) — has a documented Terraform
#      provider bug creating a target with no pipeline parameters, and
#      pipeline_definition.py defines none.
#   2. EventBridge Scheduler's universal target
#      (arn:aws:scheduler:::aws-sdk:sagemaker:startPipelineExecution) —
#      the raw StartPipelineExecution API requires a ClientRequestToken
#      for idempotency; a static one from Terraform config would make
#      every week's invocation look like a duplicate of the first,
#      silently stopping retraining after week one.
#   3. This: a tiny Lambda (boto3 auto-generates a fresh idempotency
#      token per call) + a classic EventBridge rule — the exact same
#      mechanism confidence_monitor.py already uses successfully
#      elsewhere in this stack (monitoring.tf), not the newer EventBridge
#      Scheduler service.

# The full-frame replay coordinator: three modules, not one
# (trigger_weekly_retrain imports prepare_replay_dataset, which imports
# replay_policy). source_dir, not source_file -- this was source_file
# against a stale single-module stub in lambda_src/, so applying it would
# have published a one-module zip and the function would have failed on
# import at its next invocation.
data "archive_file" "trigger_weekly_retrain" {
  type        = "zip"
  source_dir  = "${path.module}/../../model_training/replay_lambda"
  excludes    = ["README.md"]
  output_path = "${path.module}/lambda_src/trigger_weekly_retrain.zip"
}

resource "aws_iam_role" "weekly_retrain_lambda" {
  name               = "${var.project}-weekly-retrain-lambda"
  assume_role_policy = data.aws_iam_policy_document.lambda_trust.json
}

locals {
  training_pipeline_arn = "arn:aws:sagemaker:${var.region}:${var.account_id}:pipeline/track2-yolo-training-pipeline"
}

data "aws_iam_policy_document" "weekly_retrain_lambda_permissions" {
  statement {
    sid       = "Logs"
    effect    = "Allow"
    actions   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["arn:aws:logs:${var.region}:${var.account_id}:*"]
  }
  statement {
    sid       = "StartPipeline"
    effect    = "Allow"
    actions   = ["sagemaker:StartPipelineExecution"]
    resources = [local.training_pipeline_arn]
  }
}

resource "aws_iam_policy" "weekly_retrain_lambda" {
  name   = "${var.project}-weekly-retrain-lambda"
  policy = data.aws_iam_policy_document.weekly_retrain_lambda_permissions.json
}

resource "aws_iam_role_policy_attachment" "weekly_retrain_lambda" {
  role       = aws_iam_role.weekly_retrain_lambda.name
  policy_arn = aws_iam_policy.weekly_retrain_lambda.arn
}

resource "aws_lambda_function" "weekly_retrain" {
  function_name = "${var.project}-weekly-retrain"
  role          = aws_iam_role.weekly_retrain_lambda.arn
  handler       = "trigger_weekly_retrain.handler"
  runtime       = "python3.12"
  # Matches what is deployed, and what the work needs: this function
  # prepares the replay dataset in-process (prepare_replay_dataset copies
  # and rewrites thousands of objects) rather than just starting a
  # pipeline. The 30s/128MB this used to declare were from when it did
  # only the latter -- applying those would have timed the function out.
  timeout     = 900
  memory_size = 2048
  # The coordinator is written around being serialized -- it reads
  # pending.json, supersedes any queued request and writes it back with no
  # locking of its own, so two concurrent invocations would race over that
  # state. Declared here because it is deployed, and leaving it out makes
  # Terraform remove the reservation.
  reserved_concurrent_executions = 1
  filename                       = data.archive_file.trigger_weekly_retrain.output_path
  source_code_hash               = data.archive_file.trigger_weekly_retrain.output_base64sha256
}

resource "aws_cloudwatch_event_rule" "weekly_retrain_schedule" {
  name                = "${var.project}-weekly-retrain-schedule"
  schedule_expression = "cron(0 15 ? * SUN *)" # Sunday 23:00 Asia/Singapore (UTC+8) = Sunday 15:00 UTC
  # Note: classic EventBridge cron expressions are UTC-only, unlike
  # EventBridge Scheduler's schedule_expression_timezone (used in
  # ec2_schedule.tf) — the hour above is pre-converted from SGT, not a
  # timezone parameter. Re-convert by hand if the target local time
  # changes: Asia/Singapore is UTC+8 year-round (no DST).

  lifecycle {
    # The Training Control tab (review-app) lets a person change this
    # rule's cron and enabled/disabled state live via the EventBridge
    # API (events:PutRule/EnableRule/DisableRule) — without this,
    # `terraform apply` would silently revert whatever schedule they set
    # back to the Sunday-23:00 default baked into schedule_expression
    # above. This value is the *initial* schedule only; ownership of the
    # live value passes to the UI after first apply.
    ignore_changes = [schedule_expression, state]
  }
}

resource "aws_cloudwatch_event_target" "weekly_retrain" {
  rule = aws_cloudwatch_event_rule.weekly_retrain_schedule.name
  arn  = aws_lambda_function.weekly_retrain.arn
}

resource "aws_lambda_permission" "allow_weekly_retrain_eventbridge" {
  statement_id  = "AllowEventBridgeInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.weekly_retrain.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.weekly_retrain_schedule.arn
}

output "weekly_retrain_lambda_name" {
  value = aws_lambda_function.weekly_retrain.function_name
}
