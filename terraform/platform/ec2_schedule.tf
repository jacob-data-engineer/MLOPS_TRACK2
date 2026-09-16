# Auto-stop the review-host instance daily at 22:00, auto-start it at
# 09:00 Mon-Fri only — it stays off entirely over the weekend, since
# nothing starts it Sat/Sun. Meaningful saving given ec2.tf's note that
# this is the one continuously-billed resource in this stack: ~13
# hours/weekday + 0 on weekends instead of 24/7 cuts it to roughly a
# quarter of the always-on cost.
#
# EventBridge Scheduler targets the EC2 API directly (a "universal
# target") rather than going through a Lambda — nothing to write for
# something this simple.

variable "schedule_timezone" {
  description = "IANA timezone for the stop/start cron expressions below"
  type        = string
  default     = "Asia/Singapore"
}

data "aws_iam_policy_document" "scheduler_trust" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["scheduler.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "review_host_scheduler" {
  name               = "${var.project}-review-host-scheduler"
  assume_role_policy = data.aws_iam_policy_document.scheduler_trust.json
}

data "aws_iam_policy_document" "review_host_scheduler_permissions" {
  statement {
    effect    = "Allow"
    actions   = ["ec2:StopInstances", "ec2:StartInstances"]
    resources = ["arn:aws:ec2:${var.region}:${var.account_id}:instance/${aws_instance.review_host.id}"]
  }
}

resource "aws_iam_policy" "review_host_scheduler" {
  name   = "${var.project}-review-host-scheduler"
  policy = data.aws_iam_policy_document.review_host_scheduler_permissions.json
}

resource "aws_iam_role_policy_attachment" "review_host_scheduler" {
  role       = aws_iam_role.review_host_scheduler.name
  policy_arn = aws_iam_policy.review_host_scheduler.arn
}

resource "aws_scheduler_schedule" "review_host_stop" {
  name                         = "${var.project}-review-host-stop"
  schedule_expression          = "cron(0 22 * * ? *)" # every day, 22:00
  schedule_expression_timezone = var.schedule_timezone

  flexible_time_window {
    mode = "OFF"
  }

  target {
    arn      = "arn:aws:scheduler:::aws-sdk:ec2:stopInstances"
    role_arn = aws_iam_role.review_host_scheduler.arn

    input = jsonencode({
      InstanceIds = [aws_instance.review_host.id]
    })
  }
}

resource "aws_scheduler_schedule" "review_host_start" {
  name                         = "${var.project}-review-host-start"
  schedule_expression          = "cron(0 9 ? * MON-FRI *)" # weekdays only, 09:00
  schedule_expression_timezone = var.schedule_timezone

  flexible_time_window {
    mode = "OFF"
  }

  target {
    arn      = "arn:aws:scheduler:::aws-sdk:ec2:startInstances"
    role_arn = aws_iam_role.review_host_scheduler.arn

    input = jsonencode({
      InstanceIds = [aws_instance.review_host.id]
    })
  }
}

# --- announce the new URL whenever the host starts ----------------------
#
# The host has no Elastic IP, and a stopped instance releases its public
# IPv4 outright -- so every start above hands it a different address and
# yesterday's bookmark is dead. Rather than pay for an EIP to keep one
# address reserved across ~14 idle hours a day plus weekends, the new URL
# is emailed out on each start.
#
# Fires off EC2's own state-change event rather than off the scheduler, so
# a manual start is announced identically. Publishes to the topic every
# other human-facing notification already uses, so existing subscribers
# (see the app's Notifications tab) get this with nothing new to opt into.

variable "notify_extra_ports" {
  description = "Extra ports to include in the review-app URL email, alongside review_app_port (8501 is the track2_ui test container)"
  type        = list(number)
  default     = [8501]
}

data "archive_file" "review_host_started_notify" {
  type        = "zip"
  source_file = "${path.module}/lambda_src/review_host_started_notify.py"
  output_path = "${path.module}/lambda_src/review_host_started_notify.zip"
}

resource "aws_iam_role" "review_host_started_notify_lambda" {
  name               = "${var.project}-review-host-started-notify-lambda"
  assume_role_policy = data.aws_iam_policy_document.lambda_trust.json
}

data "aws_iam_policy_document" "review_host_started_notify_permissions" {
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
    # DescribeInstances has no resource-level scoping (same lesson as the
    # List-type actions elsewhere in this stack), so "*" it is.
    sid       = "ReadInstanceAddress"
    effect    = "Allow"
    actions   = ["ec2:DescribeInstances"]
    resources = ["*"]
  }
}

resource "aws_iam_policy" "review_host_started_notify" {
  name   = "${var.project}-review-host-started-notify"
  policy = data.aws_iam_policy_document.review_host_started_notify_permissions.json
}

resource "aws_iam_role_policy_attachment" "review_host_started_notify" {
  role       = aws_iam_role.review_host_started_notify_lambda.name
  policy_arn = aws_iam_policy.review_host_started_notify.arn
}

resource "aws_lambda_function" "review_host_started_notify" {
  function_name = "${var.project}-review-host-started-notify"
  role          = aws_iam_role.review_host_started_notify_lambda.arn
  handler       = "review_host_started_notify.handler"
  runtime       = "python3.12"
  # Long enough to cover the IP-lookup retries in the handler (6 x 5s)
  # with room to spare.
  timeout          = 60
  filename         = data.archive_file.review_host_started_notify.output_path
  source_code_hash = data.archive_file.review_host_started_notify.output_base64sha256

  environment {
    variables = {
      NOTIFY_TOPIC_ARN      = aws_sns_topic.reviewer_notifications.arn
      REVIEW_APP_PORTS_JSON = jsonencode(distinct(concat([var.review_app_port], var.notify_extra_ports)))
    }
  }
}

resource "aws_cloudwatch_event_rule" "review_host_running" {
  name        = "${var.project}-review-host-running"
  description = "Review host finished starting -- announce its new public IP"
  event_pattern = jsonencode({
    source      = ["aws.ec2"]
    detail-type = ["EC2 Instance State-change Notification"]
    detail = {
      state       = ["running"]
      instance-id = [aws_instance.review_host.id]
    }
  })
}

resource "aws_cloudwatch_event_target" "review_host_running" {
  rule = aws_cloudwatch_event_rule.review_host_running.name
  arn  = aws_lambda_function.review_host_started_notify.arn
}

resource "aws_lambda_permission" "allow_eventbridge_review_host_running" {
  statement_id  = "AllowEventBridgeInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.review_host_started_notify.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.review_host_running.arn
}
