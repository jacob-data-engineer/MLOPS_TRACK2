# Pull control: the drop folders, the router that empties them, and the
# two triggers that can run it.
#
# Files arrive in a flat drop folder and have to end up under
# <root>/<week>/{images,labels}. A Lambda does the moving, and either an
# arrival event or a schedule invokes it -- configurable, because a site
# that uploads continuously wants one and a site that uploads a batch
# overnight wants the other.
#
# It lives in this stack rather than the platform one because it is this
# app's own feature: a new table, a new function and two new rules, none
# of which the platform stack knows about.

variable "pull_control_table" {
  description = "Settings, week definitions and routing history for pull control."
  type        = string
  default     = "track2-pull-control"
}

# The escalation datasets live in their own bucket, separate from this
# app's. Read-only on purpose: pull control copies OUT of them and writes
# week folders into the app's bucket, so a fault here cannot damage the
# dataset it was pointed at. Both the dev and the live bucket are granted
# because the card lets the source be switched without a redeploy.
variable "dataset_source_buckets" {
  description = "Buckets pull control may READ datasets from."
  type        = list(string)
  default     = ["safety-model-dataset-dev", "safety-model-dataset"]
}

# Week folders may be written to any of these. Deliberately a list and
# not "*": the card offers a destination bucket, and a picker that can
# name a bucket nothing is allowed to write to is a choice that fails
# later instead of being refused now.
variable "dataset_write_buckets" {
  description = "Buckets pull control may WRITE week folders into."
  type        = list(string)
  default     = ["track2-s3", "safety-model-dataset-dev", "safety-model-dataset"]
}

locals {
  dataset_source_bucket_arns = [
    for name in var.dataset_source_buckets : "arn:aws:s3:::${name}"
  ]
  dataset_source_object_arns = [
    for name in var.dataset_source_buckets : "arn:aws:s3:::${name}/*"
  ]
  dataset_write_bucket_arns = [
    for name in var.dataset_write_buckets : "arn:aws:s3:::${name}"
  ]
  dataset_write_object_arns = [
    for name in var.dataset_write_buckets : "arn:aws:s3:::${name}/*"
  ]
}

# One table, three kinds of row, told apart by the partition key:
#
#   kind=config sort=v1                    the settings, as one JSON blob
#   kind=week   sort=<starts_at>#<name>    a week and when it began
#   kind=event  sort=<iso>#<rand>          one row per sweep, TTL'd
#
# Weeks are rows rather than a list inside the config so that two
# triggers creating the same week at once race harmlessly on a
# conditional put, instead of one overwriting the other's read.
resource "aws_dynamodb_table" "pull_control" {
  name         = var.pull_control_table
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "kind"
  range_key    = "sort"

  attribute {
    name = "kind"
    type = "S"
  }
  attribute {
    name = "sort"
    type = "S"
  }

  # Routing history answers "what happened last night", which stops being
  # a question long before the rows stop costing anything to keep.
  ttl {
    attribute_name = "expires_at"
    enabled        = true
  }
}

# The router is the API's own module, zipped as-is. Terraform points at
# the file in src/ rather than a copy under lambda_src/ deliberately: the
# rule that decides which week a file belongs to has to be the same one
# the UI previews, and two files drift.
data "archive_file" "pull_control_router" {
  type        = "zip"
  source_file = "${path.module}/../../src/ppe_pipeline_api/pull_control.py"
  output_path = "${path.module}/.build/pull_control_router.zip"
}

data "aws_iam_policy_document" "pull_control_lambda_trust" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "pull_control_router" {
  name               = "${var.app_name}-pull-control-router"
  assume_role_policy = data.aws_iam_policy_document.pull_control_lambda_trust.json
}

data "aws_iam_policy_document" "pull_control_router" {
  statement {
    sid       = "Logs"
    effect    = "Allow"
    actions   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["arn:aws:logs:${var.region}:${local.account_id}:*"]
  }

  # A move is a copy then a delete, so the delete is not optional. Scoped
  # to this bucket; the router has no business anywhere else.
  statement {
    sid    = "MoveArrivingFiles"
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
    ]
    resources = ["${data.aws_s3_bucket.pipeline.arn}/*"]
  }
  statement {
    sid       = "FindArrivingFiles"
    effect    = "Allow"
    actions   = ["s3:ListBucket"]
    resources = [data.aws_s3_bucket.pipeline.arn]
  }

  statement {
    sid    = "ReadSettingsAndRecordRuns"
    effect = "Allow"
    actions = [
      "dynamodb:GetItem",
      "dynamodb:PutItem",
      "dynamodb:Query",
      # The queue: a trigger arriving during a sweep enqueues itself
      # rather than racing, a tick removes the entry it is about to run,
      # and UpdateItem is the atomic ADD behind the queue's ordering.
      #
      # Without UpdateItem the counter fails closed to a per-process one
      # -- which is the bug it was added to fix, and it would fail that
      # way ONLY in the Lambda, where bursts actually happen. A permission
      # missing here looks like working software everywhere it is tested.
      "dynamodb:UpdateItem",
      "dynamodb:DeleteItem",
    ]
    resources = [aws_dynamodb_table.pull_control.arn]
  }

  # Writing week folders, and clearing one when a week is deleted. The
  # delete is what `purge_week` needs; nothing else here removes anything.
  statement {
    sid    = "WriteWeekFolders"
    effect = "Allow"
    actions = [
      "s3:PutObject",
      "s3:DeleteObject",
    ]
    resources = local.dataset_write_object_arns
  }
  statement {
    sid       = "ListWeekFolders"
    effect    = "Allow"
    actions   = ["s3:ListBucket"]
    resources = local.dataset_write_bucket_arns
  }
  # Read-only on the SOURCE datasets: a copy needs Get on the objects and
  # List on the bucket to find them. Separate from the write grant above
  # so a source bucket that is not also a destination stays untouchable.
  statement {
    sid       = "ReadSourceDatasets"
    effect    = "Allow"
    actions   = ["s3:GetObject"]
    resources = local.dataset_source_object_arns
  }
  statement {
    sid       = "ListSourceDatasets"
    effect    = "Allow"
    actions   = ["s3:ListBucket"]
    resources = local.dataset_source_bucket_arns
  }
  # Telling the subscribers a sweep has ended. Scoped to the one topic
  # every human-facing notification already goes through -- the router
  # has no business publishing anywhere else. Without this the publish
  # fails AccessDenied, which `_notify_pull_finished` swallows by design,
  # so the symptom would be silence rather than an error.
  statement {
    sid       = "NotifyPullFinished"
    effect    = "Allow"
    actions   = ["sns:Publish"]
    resources = ["arn:aws:sns:${var.region}:${local.account_id}:track2-yolo-reviewer-notifications"]
  }
}

resource "aws_iam_policy" "pull_control_router" {
  name   = "${var.app_name}-pull-control-router"
  policy = data.aws_iam_policy_document.pull_control_router.json
}

resource "aws_iam_role_policy_attachment" "pull_control_router" {
  role       = aws_iam_role.pull_control_router.name
  policy_arn = aws_iam_policy.pull_control_router.arn
}

resource "aws_lambda_function" "pull_control_router" {
  function_name = "${var.app_name}-pull-control-router"
  role          = aws_iam_role.pull_control_router.arn
  handler       = "pull_control.lambda_handler"
  runtime       = "python3.12"
  # A sweep is thousands of small copies, one at a time. Fifteen minutes
  # is the ceiling Lambda allows and the limit the router's own `limit`
  # is there to stay inside.
  timeout          = 900
  memory_size      = 512
  filename         = data.archive_file.pull_control_router.output_path
  source_code_hash = data.archive_file.pull_control_router.output_base64sha256

  environment {
    variables = {
      PIPELINE_BUCKET    = var.pipeline_bucket
      PULL_CONTROL_TABLE = aws_dynamodb_table.pull_control.name
      # Named rather than derived, so this file keeps importing nothing
      # beyond boto3 and the standard library -- Terraform zips it as the
      # handler, and the module the other callers get this ARN from does
      # not resolve there. Empty means "do not notify", which is what a
      # deployment that has not set it gets.
      PULL_NOTIFY_TOPIC_ARN = "arn:aws:sns:${var.region}:${local.account_id}:track2-yolo-reviewer-notifications"
      # The hash of the ROUTING SOURCE this function was built from. The
      # app computes the same hash over its own copy of the file and
      # compares, so a router running older code is visible in the card
      # instead of being discovered by files landing in the wrong shape.
      #
      # It has gone stale three times in one day: the container deploys,
      # the Lambda does not, and both halves are self-consistent so
      # nothing complains. deploy.sh now updates it; this is what catches
      # the case where that step was skipped or failed.
      ROUTER_SOURCE_SHA = filesha256("${path.module}/../../src/ppe_pipeline_api/pull_control.py")
    }
  }
}

# Both rules start disabled. Creating them switched on would begin moving
# files the moment this applies, before anyone has said which folders to
# watch -- and the drop folder is empty of configuration, not of files.
#
# The UI enables exactly one of them, which is what "configurable: on a
# schedule or on arrival" means in practice.

resource "aws_cloudwatch_event_rule" "pull_control_arrival" {
  name        = "${var.app_name}-pull-control-arrival"
  description = "A file landed in a watched drop folder -- file it into its week"
  state       = "DISABLED"

  # No prefix filter here. The drop prefixes are settings that change
  # from the UI, and a rule pattern that had to be rewritten to match
  # would be a second place to keep them in step; the router re-reads
  # them from the table on every invocation instead.
  event_pattern = jsonencode({
    source      = ["aws.s3"]
    detail-type = ["Object Created"]
    detail = {
      bucket = {
        name = [var.pipeline_bucket]
      }
    }
  })

  # Which trigger is on is an operating decision made from the card, not
  # a property of the stack. Without this, the next apply of an unrelated
  # change quietly switches routing back off.
  lifecycle {
    ignore_changes = [state]
  }
}

resource "aws_cloudwatch_event_target" "pull_control_arrival" {
  rule = aws_cloudwatch_event_rule.pull_control_arrival.name
  arn  = aws_lambda_function.pull_control_router.arn
}

resource "aws_lambda_permission" "pull_control_arrival" {
  statement_id  = "AllowEventBridgeArrival"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.pull_control_router.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.pull_control_arrival.arn
}

resource "aws_cloudwatch_event_rule" "pull_control_schedule" {
  name                = "${var.app_name}-pull-control-schedule"
  description         = "Sweep the watched drop folders on a timer"
  state               = "DISABLED"
  schedule_expression = "rate(1 day)"

  # The expression is set from the UI. Terraform owns the default; a
  # change made there is deliberate and should not be reverted by the
  # next apply of an unrelated change.
  lifecycle {
    ignore_changes = [schedule_expression, state]
  }
}

resource "aws_cloudwatch_event_target" "pull_control_schedule" {
  rule = aws_cloudwatch_event_rule.pull_control_schedule.name
  arn  = aws_lambda_function.pull_control_router.arn
}

resource "aws_lambda_permission" "pull_control_schedule" {
  statement_id  = "AllowEventBridgeSchedule"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.pull_control_router.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.pull_control_schedule.arn
}

# What the app itself needs: the settings table, the ability to switch
# the two rules on and off, and to run a sweep on demand from the card.
data "aws_iam_policy_document" "pull_control_app" {
  statement {
    sid    = "PullControlSettings"
    effect = "Allow"
    actions = [
      "dynamodb:GetItem",
      "dynamodb:PutItem",
      "dynamodb:DeleteItem",
      "dynamodb:Query",
      # Same atomic ADD: the card queues sweeps too.
      "dynamodb:UpdateItem",
    ]
    resources = [aws_dynamodb_table.pull_control.arn]
  }

  # "Run now" sweeps from the app, under the review-host role, not under
  # the Lambda's. Granting the delete to the Lambda alone was not enough:
  # the host could copy but not delete, so every file it filed stayed in
  # the drop folder as well -- a duplicate rather than a move, ~3,000 of
  # them before anyone noticed. The router now stops at the first such
  # failure, and this is the grant that stops it happening at all.
  statement {
    sid       = "RemoveFilesItHasFiled"
    effect    = "Allow"
    actions   = ["s3:DeleteObject"]
    resources = ["${data.aws_s3_bucket.pipeline.arn}/*"]
  }


  # Writing week folders, and clearing one when a week is deleted. The
  # delete is what `purge_week` needs; nothing else here removes anything.
  statement {
    sid    = "WriteWeekFolders"
    effect = "Allow"
    actions = [
      "s3:PutObject",
      "s3:DeleteObject",
    ]
    resources = local.dataset_write_object_arns
  }
  statement {
    sid       = "ListWeekFolders"
    effect    = "Allow"
    actions   = ["s3:ListBucket"]
    resources = local.dataset_write_bucket_arns
  }
  # Read-only on the SOURCE datasets: a copy needs Get on the objects and
  # List on the bucket to find them. Separate from the write grant above
  # so a source bucket that is not also a destination stays untouchable.
  statement {
    sid       = "ReadSourceDatasets"
    effect    = "Allow"
    actions   = ["s3:GetObject"]
    resources = local.dataset_source_object_arns
  }
  statement {
    sid       = "ListSourceDatasets"
    effect    = "Allow"
    actions   = ["s3:ListBucket"]
    resources = local.dataset_source_bucket_arns
  }

  # Bucket NAMES only, for the picker in the card. ListAllMyBuckets has no
  # resource-level scoping -- it is "*" or nothing -- but it reveals only
  # the names, never any contents. Reading objects stays restricted to the
  # buckets granted above, so a bucket can appear in the list and still be
  # unreadable; the card probes and says so rather than offering a choice
  # that fails on the next click.
  statement {
    sid       = "ListBucketNamesForThePicker"
    effect    = "Allow"
    actions   = ["s3:ListAllMyBuckets"]
    resources = ["*"]
  }

  # Reading the router's configuration to compare its source hash with
  # the app's. No invoke, no modify -- just enough to notice drift.
  statement {
    sid       = "ReadRouterVersion"
    effect    = "Allow"
    actions   = ["lambda:GetFunctionConfiguration"]
    resources = [aws_lambda_function.pull_control_router.arn]
  }

  statement {
    sid    = "SwitchTriggers"
    effect = "Allow"
    actions = [
      "events:DescribeRule",
      "events:EnableRule",
      "events:DisableRule",
      "events:PutRule",
      "events:ListTargetsByRule",
    ]
    resources = [
      aws_cloudwatch_event_rule.pull_control_arrival.arn,
      aws_cloudwatch_event_rule.pull_control_schedule.arn,
    ]
  }

  # The timer itself. Read and update only -- the schedule is created by
  # terraform, so the app can change when it fires and whether it is on,
  # but cannot invent a schedule pointing somewhere else.
  statement {
    sid    = "SetThePullSchedule"
    effect = "Allow"
    actions = [
      "scheduler:GetSchedule",
      "scheduler:UpdateSchedule",
    ]
    resources = [aws_scheduler_schedule.pull_control.arn]
  }

  # UpdateSchedule re-sends the target, and a target carries a role, so
  # the call is a PassRole whether or not the role is changing. Scoped to
  # the one role, and only when Scheduler is the service being passed to.
  statement {
    sid       = "PassTheSchedulerRole"
    effect    = "Allow"
    actions   = ["iam:PassRole"]
    resources = [aws_iam_role.pull_control_scheduler.arn]
    condition {
      test     = "StringEquals"
      variable = "iam:PassedToService"
      values   = ["scheduler.amazonaws.com"]
    }
  }
}

# --- the pull timer -----------------------------------------------------
#
# EventBridge Scheduler rather than the `pull_control_schedule` rule above.
# A classic rule cannot express "first pull at 09:00 on the 26th, then
# every fortnight": its `rate()` is anchored to whenever the rule was last
# written, and its cron has no start date and no timezone. Scheduler has
# both, which is what choosing a pull time actually requires.
#
# The old rule is kept but is no longer driven -- `set_mode` disables it
# on every call, so a schedule armed before this change cannot go on
# firing next to the new one. It can be removed once no environment is
# still carrying an enabled copy of it.

data "aws_iam_policy_document" "pull_control_scheduler_trust" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["scheduler.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "pull_control_scheduler" {
  name               = "${var.app_name}-pull-control-scheduler"
  assume_role_policy = data.aws_iam_policy_document.pull_control_scheduler_trust.json
}

data "aws_iam_policy_document" "pull_control_scheduler" {
  statement {
    effect    = "Allow"
    actions   = ["lambda:InvokeFunction"]
    resources = [aws_lambda_function.pull_control_router.arn]
  }
}

resource "aws_iam_policy" "pull_control_scheduler" {
  name   = "${var.app_name}-pull-control-scheduler"
  policy = data.aws_iam_policy_document.pull_control_scheduler.json
}

resource "aws_iam_role_policy_attachment" "pull_control_scheduler" {
  role       = aws_iam_role.pull_control_scheduler.name
  policy_arn = aws_iam_policy.pull_control_scheduler.arn
}

resource "aws_scheduler_schedule" "pull_control" {
  name = "${var.app_name}-pull-control"

  # Created switched off and left that way by terraform. Turning it on is
  # a decision made in the card, against a configured interval and start
  # date; a schedule that armed itself the moment the stack was applied
  # would start moving real files on its own.
  state = "DISABLED"

  # Both are placeholders. `set_mode` overwrites them from the settings,
  # and `ignore_changes` below stops a later apply reverting the interval
  # someone chose -- the same reasoning as the two rules above.
  schedule_expression          = "rate(7 days)"
  schedule_expression_timezone = "Asia/Singapore"

  # OFF: a pull that may start up to fifteen minutes late makes "which
  # files had arrived by then" a different question each run. The window
  # costs nothing to keep shut.
  flexible_time_window {
    mode = "OFF"
  }

  target {
    arn      = aws_lambda_function.pull_control_router.arn
    role_arn = aws_iam_role.pull_control_scheduler.arn

    # The router reads its own settings; this only says who woke it, so
    # a scheduled sweep is distinguishable from an arrival in the run log.
    input = jsonencode({ trigger = "schedule" })

    retry_policy {
      maximum_retry_attempts = 2
    }
  }

  lifecycle {
    ignore_changes = [schedule_expression, state, start_date]
  }
}

resource "aws_iam_policy" "pull_control_app" {
  name   = "${var.app_name}-pull-control"
  policy = data.aws_iam_policy_document.pull_control_app.json
}

resource "aws_iam_role_policy_attachment" "pull_control_app" {
  role       = data.aws_iam_role.review_host.name
  policy_arn = aws_iam_policy.pull_control_app.arn
}

output "pull_control_table" {
  value = aws_dynamodb_table.pull_control.name
}

output "pull_control_arrival_rule" {
  value = aws_cloudwatch_event_rule.pull_control_arrival.name
}

output "pull_control_schedule_rule" {
  value = aws_cloudwatch_event_rule.pull_control_schedule.name
}

output "pull_control_schedule" {
  value = aws_scheduler_schedule.pull_control.name
}

# --------------------------------------------------------------------------
# The queue tick
# --------------------------------------------------------------------------
#
# Drains the pull queue every couple of minutes. Separate from the
# schedule rule above because they answer different questions: the
# schedule asks "is it time to pull again?" once a day, the tick asks "is
# anything queued and is the slot free?".
#
# This is what makes the queue survive the app container going away. The
# app ticks the same queue on its own 25-second loop, but that loop stops
# with the container -- during a deploy, or whenever the box is off. Both
# callers go through `advance_pull_queue`, which refuses to start a second
# sweep while one is running, so the two tickers cannot race.
#
# Created by hand first and written here to match, per CLAUDE.md: a
# targeted AWS change plus the .tf that agrees with it, rather than an
# apply that might carry other drift with it.
resource "aws_cloudwatch_event_rule" "queue_tick" {
  name                = "${var.app_name}-queue-tick"
  description         = "Drains the pull queue so a queued sweep runs whether or not the app container is up."
  schedule_expression = "rate(2 minutes)"
  state               = "ENABLED"
}

resource "aws_cloudwatch_event_target" "queue_tick" {
  rule = aws_cloudwatch_event_rule.queue_tick.name
  # Named, so this matches the target created by hand rather than adding
  # a second one beside it.
  target_id = "tick"
  arn       = aws_lambda_function.pull_control_router.arn

  # A tick NEVER sweeps on its own account -- it only drains what is
  # already queued. The handler branches on this, so the same function
  # serves both without a second deployment artefact.
  input = jsonencode({ tick = true })
}

resource "aws_lambda_permission" "queue_tick" {
  statement_id  = "queue-tick-invoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.pull_control_router.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.queue_tick.arn
}

# --------------------------------------------------------------------------
# The queue runner
# --------------------------------------------------------------------------
#
# Advances the TRAINING queue when the app container is not there to.
#
# The queue has always been durable -- entries are DynamoDB rows, FIFO
# order is restored on startup, and `try_advance_queue` asks SageMaker
# before starting so a restart cannot double-submit. What was missing was
# the tick: the poll loop is an asyncio task in the app's lifespan, so
# while the container is down nothing advances and the GPU sits idle.
#
# It imports `track2_shared` and calls the SAME `try_advance_queue` the
# container calls. A second implementation of "is the GPU free?" is how an
# account with one training instance comes to have two runs submitted
# against it.
#
# The zip is assembled by deploy.sh into .build/queue_runner: the
# `ppe_pipeline_api` package plus PyYAML, which the advance needs because
# validating a dataset reads its data.yaml. Terraform only zips what is
# already there, so a plan run without deploy.sh having assembled it will
# fail loudly rather than ship an empty function.
data "archive_file" "queue_runner" {
  type        = "zip"
  source_dir  = "${path.module}/.build/queue_runner"
  output_path = "${path.module}/.build/queue_runner.zip"
}

resource "aws_lambda_function" "queue_runner" {
  function_name = "${var.app_name}-queue-runner"
  role          = aws_iam_role.queue_runner.arn
  handler       = "ppe_pipeline_api.queue_runner.lambda_handler"
  runtime       = "python3.12"
  # One advance per invocation, and an advance is a few API calls plus a
  # dataset validation. It does not wait for the run.
  timeout          = 120
  memory_size      = 512
  filename         = data.archive_file.queue_runner.output_path
  source_code_hash = data.archive_file.queue_runner.output_base64sha256

  environment {
    variables = {
      PIPELINE_BUCKET    = var.pipeline_bucket
      PULL_CONTROL_TABLE = aws_dynamodb_table.pull_control.name
      # Where a stopped chain's error email goes; see notify.py.
      PULL_NOTIFY_TOPIC_ARN = "arn:aws:sns:${var.region}:${local.account_id}:track2-yolo-reviewer-notifications"
    }
  }
}

resource "aws_iam_role" "queue_runner" {
  name               = "${var.app_name}-queue-runner"
  assume_role_policy = data.aws_iam_policy_document.pull_control_lambda_trust.json
}

resource "aws_iam_role_policy_attachment" "queue_runner_logs" {
  role       = aws_iam_role.queue_runner.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# Everything an advance touches: the queue rows and the submission marker
# in the audit table, the dataset it validates, the build it starts, and
# the SageMaker state it reads to decide the slot is free.
data "aws_iam_policy_document" "queue_runner" {
  # A scheduled retrain chain that stops short emails the subscribers who
  # ticked "Errors" (notify.py). Only this one topic.
  statement {
    sid       = "NotifyChainStopped"
    effect    = "Allow"
    actions   = ["sns:Publish"]
    resources = ["arn:aws:sns:${var.region}:${local.account_id}:track2-yolo-reviewer-notifications"]
  }
  statement {
    sid    = "TheQueueAndTheMarker"
    effect = "Allow"
    actions = [
      "dynamodb:GetItem",
      "dynamodb:PutItem",
      "dynamodb:DeleteItem",
      "dynamodb:Query",
      "dynamodb:Scan",
      "dynamodb:UpdateItem",
    ]
    resources = ["*"]
  }

  statement {
    sid       = "ReadTheDatasetItValidates"
    effect    = "Allow"
    actions   = ["s3:GetObject", "s3:ListBucket"]
    resources = [data.aws_s3_bucket.pipeline.arn, "${data.aws_s3_bucket.pipeline.arn}/*"]
  }

  statement {
    sid       = "StartTheRun"
    effect    = "Allow"
    actions   = ["codebuild:StartBuild", "codebuild:BatchGetBuilds"]
    resources = ["*"]
  }

  # Read-only on SageMaker: this decides whether the slot is free, it does
  # not start pipelines itself. The custom path, which does, is left to
  # the container -- a Lambda that could start a pipeline execution is a
  # wider grant than the job needs.
  statement {
    sid    = "IsTheGpuFree"
    effect = "Allow"
    actions = [
      "sagemaker:ListPipelineExecutions",
      "sagemaker:DescribePipelineExecution",
      "sagemaker:ListPipelineExecutionSteps",
      "sagemaker:ListTrainingJobs",
      "sagemaker:DescribeTrainingJob",
      "sagemaker:ListProcessingJobs",
    ]
    resources = ["*"]
  }
}

resource "aws_iam_policy" "queue_runner" {
  name   = "${var.app_name}-queue-runner"
  policy = data.aws_iam_policy_document.queue_runner.json
}

resource "aws_iam_role_policy_attachment" "queue_runner" {
  role       = aws_iam_role.queue_runner.name
  policy_arn = aws_iam_policy.queue_runner.arn
}

# Same tick as the pull queue. One rule, two targets: both queues are
# asking the same question -- is anything waiting and is the slot free.
resource "aws_cloudwatch_event_target" "queue_runner" {
  rule      = aws_cloudwatch_event_rule.queue_tick.name
  target_id = "runner"
  arn       = aws_lambda_function.queue_runner.arn
}

resource "aws_lambda_permission" "queue_runner_tick" {
  statement_id  = "queue-tick-invoke-runner"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.queue_runner.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.queue_tick.arn
}
