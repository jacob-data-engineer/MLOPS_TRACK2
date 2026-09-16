# Model Monitor -> review trigger chain:
#   confidence_monitor Lambda (scheduled every 15min) reads the endpoint's
#   captured input/output, computes mean prediction confidence, compares
#   to a stored baseline, publishes a CloudWatch metric -> alarm fires on
#   drift -> SNS -> trigger_review Lambda writes each flagged example to
#   the review-queue DynamoDB table and republishes to the same SNS topic
#   so reviewer_emails get an email pointing at the review app.
#
# NOT Amazon A2I: both A2I and SageMaker private-workforce management are
# closed to new AWS customers account-wide (confirmed via a real
# CreateWorkforce/CreateHumanTaskUi 400 on this account) — see
# sagemaker.tf's note. trigger_review.py + review_queue.tf + review_app.py
# (deployed to the EC2 instance) are the replacement.
#
# NOT AWS's built-in SageMaker Data Quality Model Monitor either: that
# construct expects tabular/JSON feature data captured from the endpoint,
# and doesn't apply directly to a CV endpoint whose captured requests are
# raw image bytes. confidence_monitor.py is a small purpose-built
# replacement that achieves the same "flag drift -> human review" outcome
# correctly for this model type.

data "archive_file" "trigger_review" {
  type        = "zip"
  source_file = "${path.module}/lambda_src/trigger_review.py"
  output_path = "${path.module}/lambda_src/trigger_review.zip"
}

resource "aws_lambda_function" "trigger_review" {
  function_name    = "${var.project}-trigger-review"
  role             = aws_iam_role.review_trigger_lambda.arn
  handler          = "trigger_review.handler"
  runtime          = "python3.12"
  timeout          = 60
  filename         = data.archive_file.trigger_review.output_path
  source_code_hash = data.archive_file.trigger_review.output_base64sha256

  environment {
    variables = {
      PIPELINE_BUCKET    = aws_s3_bucket.pipeline.bucket
      REVIEW_QUEUE_TABLE = aws_dynamodb_table.review_queue.name
      NOTIFY_TOPIC_ARN   = aws_sns_topic.reviewer_notifications.arn
      REVIEW_APP_URL     = "http://${aws_instance.review_host.public_ip}:${var.review_app_port}/"
    }
  }
}

# Two separate topics, deliberately: monitor_drift is the alarm -> Lambda
# trigger channel; reviewer_notifications is what the Lambda itself
# publishes to. Reusing one topic for both would mean trigger_review's own
# publish() re-triggers itself (it's subscribed to monitor_drift) — an
# infinite loop.

resource "aws_sns_topic" "monitor_drift" {
  name = "${var.project}-monitor-drift"
}

resource "aws_sns_topic_subscription" "trigger_review_lambda" {
  topic_arn = aws_sns_topic.monitor_drift.arn
  protocol  = "lambda"
  endpoint  = aws_lambda_function.trigger_review.arn
}

resource "aws_sns_topic" "reviewer_notifications" {
  name = "${var.project}-reviewer-notifications"
}

# Replaces A2I's own task-assignment notification, since there's no
# workforce/portal to assign tasks in. SNS emails require the recipient
# to click a confirmation link once.
resource "aws_sns_topic_subscription" "reviewer_email" {
  for_each  = toset(var.reviewer_emails)
  topic_arn = aws_sns_topic.reviewer_notifications.arn
  protocol  = "email"
  endpoint  = each.value
}

resource "aws_lambda_permission" "allow_sns" {
  statement_id  = "AllowSNSInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.trigger_review.function_name
  principal     = "sns.amazonaws.com"
  source_arn    = aws_sns_topic.monitor_drift.arn
}

# confidence_monitor.py publishes this metric every run; the alarm fires
# on any positive drift value (i.e. current-window confidence measurably
# below baseline). Tune the threshold once a real baseline exists and the
# false-positive rate under real traffic is known.
resource "aws_cloudwatch_metric_alarm" "confidence_drift" {
  alarm_name          = "${var.project}-confidence-drift"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "feature_baseline_drift_confidence"
  namespace           = "Track2Yolo/ModelMonitor"
  period              = 900
  statistic           = "Maximum"
  threshold           = 0
  treat_missing_data  = "notBreaching"
  alarm_description   = "Fires when the Track2 YOLO endpoint's mean prediction confidence drifts below baseline"
  alarm_actions       = [aws_sns_topic.monitor_drift.arn]
}

# --- confidence_monitor Lambda, run every 15 minutes ---------------------

data "archive_file" "confidence_monitor" {
  type        = "zip"
  source_file = "${path.module}/lambda_src/confidence_monitor.py"
  output_path = "${path.module}/lambda_src/confidence_monitor.zip"
}

resource "aws_iam_role" "confidence_monitor_lambda" {
  name               = "${var.project}-confidence-monitor-lambda"
  assume_role_policy = data.aws_iam_policy_document.lambda_trust.json
}

data "aws_iam_policy_document" "confidence_monitor_lambda_permissions" {
  statement {
    sid       = "Logs"
    effect    = "Allow"
    actions   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["arn:aws:logs:${var.region}:${var.account_id}:*"]
  }
  statement {
    sid       = "PipelineBucket"
    effect    = "Allow"
    actions   = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
    resources = [aws_s3_bucket.pipeline.arn, "${aws_s3_bucket.pipeline.arn}/*"]
  }
  statement {
    sid       = "Metrics"
    effect    = "Allow"
    actions   = ["cloudwatch:PutMetricData"]
    resources = ["*"]
  }
}

resource "aws_iam_policy" "confidence_monitor_lambda" {
  name   = "${var.project}-confidence-monitor-lambda"
  policy = data.aws_iam_policy_document.confidence_monitor_lambda_permissions.json
}

resource "aws_iam_role_policy_attachment" "confidence_monitor_lambda" {
  role       = aws_iam_role.confidence_monitor_lambda.name
  policy_arn = aws_iam_policy.confidence_monitor_lambda.arn
}

resource "aws_lambda_function" "confidence_monitor" {
  function_name    = "${var.project}-confidence-monitor"
  role             = aws_iam_role.confidence_monitor_lambda.arn
  handler          = "confidence_monitor.handler"
  runtime          = "python3.12"
  timeout          = 120
  filename         = data.archive_file.confidence_monitor.output_path
  source_code_hash = data.archive_file.confidence_monitor.output_base64sha256

  environment {
    variables = {
      PIPELINE_BUCKET = aws_s3_bucket.pipeline.bucket
    }
  }
}

resource "aws_cloudwatch_event_rule" "confidence_monitor_schedule" {
  name                = "${var.project}-confidence-monitor-schedule"
  schedule_expression = "rate(15 minutes)"
}

resource "aws_cloudwatch_event_target" "confidence_monitor" {
  rule = aws_cloudwatch_event_rule.confidence_monitor_schedule.name
  arn  = aws_lambda_function.confidence_monitor.arn
}

resource "aws_lambda_permission" "allow_eventbridge" {
  statement_id  = "AllowEventBridgeInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.confidence_monitor.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.confidence_monitor_schedule.arn
}

# --- dump_approved_weights Lambda: on every model-package approval, lands
# the weights at a fixed "latest" S3 location (archiving whatever was there
# before) and emails stakeholders. Triggered by SageMaker's own built-in
# EventBridge event -- no polling, fires natively whenever
# UpdateModelPackage (what approve_model.py calls) changes the status. -----

data "archive_file" "dump_approved_weights" {
  type        = "zip"
  source_file = "${path.module}/lambda_src/dump_approved_weights.py"
  output_path = "${path.module}/lambda_src/dump_approved_weights.zip"
}

resource "aws_iam_role" "dump_approved_weights_lambda" {
  name               = "${var.project}-dump-approved-weights-lambda"
  assume_role_policy = data.aws_iam_policy_document.lambda_trust.json
}

data "aws_iam_policy_document" "dump_approved_weights_lambda_permissions" {
  statement {
    sid       = "Logs"
    effect    = "Allow"
    actions   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["arn:aws:logs:${var.region}:${var.account_id}:*"]
  }
  statement {
    sid       = "PipelineBucketReadWrite"
    effect    = "Allow"
    actions   = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
    resources = [aws_s3_bucket.pipeline.arn, "${aws_s3_bucket.pipeline.arn}/*"]
  }
  statement {
    sid     = "DescribeApprovedModelPackage"
    effect  = "Allow"
    actions = ["sagemaker:DescribeModelPackage"]
    # Covers all five YOLO11 size groups (sagemaker.tf's
    # yolo_size_group_names) -- this Lambda fires on an Approved
    # transition in any of them, not just one fixed group.
    resources = [for name in values(local.yolo_size_group_names) : "arn:aws:sagemaker:${var.region}:${var.account_id}:model-package/${name}/*"]
  }
  statement {
    sid       = "NotifyStakeholders"
    effect    = "Allow"
    actions   = ["sns:Publish"]
    resources = [aws_sns_topic.reviewer_notifications.arn]
  }
  statement {
    # Lets this Lambda deploy the newly-approved package straight to that
    # size's Inference Component (see dump_approved_weights.py's
    # _deploy_to_inference_component) -- CreateModel/{Create,Update,
    # Describe}InferenceComponent don't support resource-level ARN
    # scoping the way DescribeModelPackage above does, so these stay
    # Resource "*", same reasoning as ModelRegistryList in iam.tf.
    sid    = "DeployToInferenceComponent"
    effect = "Allow"
    actions = [
      "sagemaker:CreateModel",
      "sagemaker:CreateInferenceComponent",
      "sagemaker:UpdateInferenceComponent",
      "sagemaker:DescribeInferenceComponent",
    ]
    resources = ["*"]
  }
  statement {
    # CreateModel needs to hand this role to SageMaker for the Inference
    # Component's container to run as -- same PassRole requirement as
    # codebuild_run_pipeline's own CreateTrainingJob.
    sid       = "PassSageMakerExecutionRole"
    effect    = "Allow"
    actions   = ["iam:PassRole"]
    resources = [aws_iam_role.sagemaker_execution.arn]
  }
}

resource "aws_iam_policy" "dump_approved_weights_lambda" {
  name   = "${var.project}-dump-approved-weights-lambda"
  policy = data.aws_iam_policy_document.dump_approved_weights_lambda_permissions.json
}

resource "aws_iam_role_policy_attachment" "dump_approved_weights_lambda" {
  role       = aws_iam_role.dump_approved_weights_lambda.name
  policy_arn = aws_iam_policy.dump_approved_weights_lambda.arn
}

resource "aws_lambda_function" "dump_approved_weights" {
  function_name    = "${var.project}-dump-approved-weights"
  role             = aws_iam_role.dump_approved_weights_lambda.arn
  handler          = "dump_approved_weights.handler"
  runtime          = "python3.12"
  timeout          = 120
  filename         = data.archive_file.dump_approved_weights.output_path
  source_code_hash = data.archive_file.dump_approved_weights.output_base64sha256

  environment {
    variables = {
      PIPELINE_BUCKET              = aws_s3_bucket.pipeline.bucket
      NOTIFY_TOPIC_ARN             = aws_sns_topic.reviewer_notifications.arn
      GROUP_TO_SIZE_JSON           = jsonencode({ for size, name in local.yolo_size_group_names : name => size })
      MULTI_SIZE_ENDPOINT_NAME     = "${var.project}-multi-size-endpoint"
      MULTI_SIZE_VARIANT_NAME      = "AllSizes"
      SAGEMAKER_EXECUTION_ROLE_ARN = aws_iam_role.sagemaker_execution.arn
      PROJECT                      = var.project
    }
  }
}

# SageMaker's own built-in event -- fires on every ModelApprovalStatus
# transition for every model package in this account; filtered here to
# just Approved transitions within this project's own model package
# groups (all four YOLO11 sizes), so approvals in other groups (e.g.
# Track2's real, separate infra) don't trigger this.
resource "aws_cloudwatch_event_rule" "model_package_approved" {
  name = "${var.project}-model-package-approved"
  event_pattern = jsonencode({
    source      = ["aws.sagemaker"]
    detail-type = ["SageMaker Model Package State Change"]
    detail = {
      ModelApprovalStatus   = ["Approved"]
      ModelPackageGroupName = values(local.yolo_size_group_names)
    }
  })
}

resource "aws_cloudwatch_event_target" "dump_approved_weights" {
  rule = aws_cloudwatch_event_rule.model_package_approved.name
  arn  = aws_lambda_function.dump_approved_weights.arn
}

resource "aws_lambda_permission" "allow_eventbridge_model_approval" {
  statement_id  = "AllowEventBridgeInvokeModelApproval"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.dump_approved_weights.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.model_package_approved.arn
}

# --- training_completed_notify Lambda: emails stakeholders whenever a
# track2-yolo-training-pipeline execution reaches a terminal state
# (Succeeded/Failed/Stopped), not just on approval -- closes the gap where
# training could finish (or fail) with nobody notified until someone
# happened to check the console. Triggered by SageMaker's own built-in
# "SageMaker Model Building Pipeline Execution Status Change" EventBridge
# event -- schema confirmed via `aws schemas describe-schema --registry-
# name aws.events --schema-name aws.sagemaker@SageMakerModelBuilding
# PipelineExecutionStatusChange` rather than guessed. ------------------

data "archive_file" "training_completed_notify" {
  type        = "zip"
  source_file = "${path.module}/lambda_src/training_completed_notify.py"
  output_path = "${path.module}/lambda_src/training_completed_notify.zip"
}

resource "aws_iam_role" "training_completed_notify_lambda" {
  name               = "${var.project}-training-completed-notify-lambda"
  assume_role_policy = data.aws_iam_policy_document.lambda_trust.json
}

data "aws_iam_policy_document" "training_completed_notify_lambda_permissions" {
  statement {
    sid       = "Logs"
    effect    = "Allow"
    actions   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["arn:aws:logs:${var.region}:${var.account_id}:*"]
  }
  statement {
    sid       = "NotifyStakeholders"
    effect    = "Allow"
    actions   = ["sns:Publish"]
    resources = [aws_sns_topic.reviewer_notifications.arn]
  }
  statement {
    # Best-effort lookup of the newly-registered candidate on a Succeeded
    # run -- List-type action, needs Resource "*" (same lesson as
    # review_app_access's ModelRegistryList statement).
    sid       = "ListModelPackages"
    effect    = "Allow"
    actions   = ["sagemaker:ListModelPackages"]
    resources = ["*"]
  }
}

resource "aws_iam_policy" "training_completed_notify_lambda" {
  name   = "${var.project}-training-completed-notify-lambda"
  policy = data.aws_iam_policy_document.training_completed_notify_lambda_permissions.json
}

resource "aws_iam_role_policy_attachment" "training_completed_notify_lambda" {
  role       = aws_iam_role.training_completed_notify_lambda.name
  policy_arn = aws_iam_policy.training_completed_notify_lambda.arn
}

resource "aws_lambda_function" "training_completed_notify" {
  function_name    = "${var.project}-training-completed-notify"
  role             = aws_iam_role.training_completed_notify_lambda.arn
  handler          = "training_completed_notify.handler"
  runtime          = "python3.12"
  timeout          = 30
  filename         = data.archive_file.training_completed_notify.output_path
  source_code_hash = data.archive_file.training_completed_notify.output_base64sha256

  environment {
    variables = {
      NOTIFY_TOPIC_ARN          = aws_sns_topic.reviewer_notifications.arn
      MODEL_PACKAGE_GROUPS_JSON = jsonencode(values(local.yolo_size_group_names))
    }
  }
}

resource "aws_cloudwatch_event_rule" "training_pipeline_completed" {
  name = "${var.project}-training-pipeline-completed"
  event_pattern = jsonencode({
    source      = ["aws.sagemaker"]
    detail-type = ["SageMaker Model Building Pipeline Execution Status Change"]
    detail = {
      pipelineArn = [local.training_pipeline_arn]
      # Executing: fires once, at the transition into this status -- i.e.
      # pipeline start, not every step transition (this is the execution-
      # level status-change event, not a step-level one).
      currentPipelineExecutionStatus = ["Executing", "Succeeded", "Failed", "Stopped"]
    }
  })
}

resource "aws_cloudwatch_event_target" "training_completed_notify" {
  rule = aws_cloudwatch_event_rule.training_pipeline_completed.name
  arn  = aws_lambda_function.training_completed_notify.arn
}

resource "aws_lambda_permission" "allow_eventbridge_training_completed" {
  statement_id  = "AllowEventBridgeInvokeTrainingCompleted"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.training_completed_notify.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.training_pipeline_completed.arn
}

# --- pull_training_data Lambda: on-demand, reads
# data/track2-datasets/track2_pull/latest.json and copies the pull it
# points at (via its "snapshot" field) into a fixed
# data/track2-datasets/sagemaker_training/ prefix, ready for the training
# pipeline to read. Not triggered by anything -- invoke manually:
#   aws lambda invoke --function-name track2-sagemaker-pull-training --region ap-southeast-1 /dev/stdout
# -----------------------------------------------------------------------

data "archive_file" "pull_training_data" {
  type        = "zip"
  source_file = "${path.module}/lambda_src/pull_training_data.py"
  output_path = "${path.module}/lambda_src/pull_training_data.zip"
}

resource "aws_iam_role" "pull_training_data_lambda" {
  name               = "${var.project}-pull-training-data-lambda"
  assume_role_policy = data.aws_iam_policy_document.lambda_trust.json
}

data "aws_iam_policy_document" "pull_training_data_lambda_permissions" {
  statement {
    sid       = "Logs"
    effect    = "Allow"
    actions   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["arn:aws:logs:${var.region}:${var.account_id}:*"]
  }
  statement {
    # Source (data/track2-datasets/track2_pull/latest.json + the pull
    # snapshot it points at) and destination (sagemaker_training/) are both
    # inside this one bucket -- a single broad statement covers reading
    # latest.json, reading the snapshot's objects, and writing the copies.
    sid       = "PipelineBucketReadWrite"
    effect    = "Allow"
    actions   = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
    resources = [aws_s3_bucket.pipeline.arn, "${aws_s3_bucket.pipeline.arn}/*"]
  }
}

resource "aws_iam_policy" "pull_training_data_lambda" {
  name   = "${var.project}-pull-training-data-lambda"
  policy = data.aws_iam_policy_document.pull_training_data_lambda_permissions.json
}

resource "aws_iam_role_policy_attachment" "pull_training_data_lambda" {
  role       = aws_iam_role.pull_training_data_lambda.name
  policy_arn = aws_iam_policy.pull_training_data_lambda.arn
}

resource "aws_lambda_function" "pull_training_data" {
  function_name    = "track2-sagemaker-pull-training"
  role             = aws_iam_role.pull_training_data_lambda.arn
  handler          = "pull_training_data.handler"
  runtime          = "python3.12"
  timeout          = 300 # copying ~467 objects one-by-one via copy_object; generous headroom
  filename         = data.archive_file.pull_training_data.output_path
  source_code_hash = data.archive_file.pull_training_data.output_base64sha256
}

resource "aws_lambda_permission" "allow_s3_latest_json" {
  statement_id  = "AllowS3InvokeLatestJson"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.pull_training_data.function_name
  principal     = "s3.amazonaws.com"
  source_arn    = aws_s3_bucket.pipeline.arn
}

# Fires pull_training_data automatically whenever the external pull
# pipeline (re)writes data/track2-datasets/track2_pull/latest.json --
# S3 treats overwriting an existing key as an ObjectCreated:Put event too,
# so this catches every new completed pull, not just the first one.
#
# CAUTION: aws_s3_bucket_notification is a full-replace resource -- only
# one may exist per bucket in Terraform, and applying it overwrites
# whatever notification configuration currently exists on the live bucket.
# Confirmed via `aws s3api get-bucket-notification-configuration` that
# track2-s3 has none today, so this is safe now -- but if anything else
# (this external pipeline included) ever configures its own bucket
# notification directly on track2-s3 outside this Terraform, applying this
# resource again would silently clobber it. Re-check before ever changing
# this block.
resource "aws_s3_bucket_notification" "track2_aws_s3_bucket_notification_latest_json" {
  bucket = aws_s3_bucket.pipeline.id

  lambda_function {
    lambda_function_arn = aws_lambda_function.pull_training_data.arn
    events              = ["s3:ObjectCreated:*"]
    filter_prefix       = "data/track2-datasets/track2_pull/"
    filter_suffix       = "latest.json"
  }

  depends_on = [aws_lambda_permission.allow_s3_latest_json]
}
