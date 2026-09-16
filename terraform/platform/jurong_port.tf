# Jurong Port's SageMaker BYOC serving scaffold, merged into this stack's
# infra/ so both projects are managed from one Terraform root + state
# (same AWS account, ap-southeast-1). Everything here is prefixed
# jurong_port_ specifically to stay unambiguous now that it shares a
# module with Track2's own resources -- e.g. both projects originally had
# their own aws_iam_role.sagemaker_execution; this one was renamed to
# aws_iam_role.jurong_port_sagemaker_execution (via `terraform state mv`,
# not recreated) to resolve that exact collision.
#
# Currently unused: no model is deployed under Jurong Port right now
# (InstructIR/SwinIR were both decommissioned, Real-ESRGAN lived in a
# separate project). This is reusable scaffolding for whatever model
# gets added next -- see JURONG PORT's own README.md's "Adding a model"
# section (that repo, not this one, is where a new model's own
# .tf/Dockerfile/inference.py would be authored).

data "aws_region" "current" {}

variable "jurong_port_project_name" {
  description = "Prefix applied to Jurong Port's resource names/tags."
  type        = string
  default     = "jurong-port"
}

variable "jurong_port_environment" {
  description = "Jurong Port deployment environment tag (dev/staging/prod)."
  type        = string
  default     = "dev"
}

variable "jurong_port_model_artifact_bucket_name" {
  description = "S3 bucket name (not ARN/URI) holding Jurong Port's model artifact, scoped for the execution role's read policy."
  type        = string
  default     = "gp-jurong-port-s3-jl"
}

# There is deliberately no aws_s3_bucket for Jurong Port model artifacts.
# The bucket this stack used to point at (gp-jurong-port-s3-jl) was deleted
# outside Terraform, so the resource no longer described anything real --
# it only ever offered to create a brand new empty bucket under that name.
# It sat un-applied for exactly that reason, which meant every plan carried
# a create nobody intended to run. Removed outright rather than left as a
# trap: reinstate it (plus its bucket-name variable) only if Jurong Port
# artifacts actually need a home again.

variable "jurong_port_tags" {
  description = "Common tags applied to every Jurong Port resource."
  type        = map(string)
  default     = { Owner = "mlops-team" }
}

locals {
  jurong_port_common_tags = merge(
    {
      Project     = var.jurong_port_project_name
      Environment = var.jurong_port_environment
      ManagedBy   = "terraform"
    },
    var.jurong_port_tags,
  )
}


data "aws_iam_policy_document" "jurong_port_sagemaker_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "jurong_port_sagemaker_execution" {
  name               = "${var.jurong_port_project_name}-sagemaker-execution-role-jl"
  assume_role_policy = data.aws_iam_policy_document.jurong_port_sagemaker_assume_role.json
  tags               = local.jurong_port_common_tags
}

# --- Strict, read-only access to the model artifact bucket only ---
data "aws_iam_policy_document" "jurong_port_s3_model_read" {
  statement {
    sid       = "ListModelArtifactBucket"
    actions   = ["s3:ListBucket"]
    resources = ["arn:aws:s3:::${var.jurong_port_model_artifact_bucket_name}"]
  }

  statement {
    sid       = "ReadModelArtifacts"
    actions   = ["s3:GetObject"]
    resources = ["arn:aws:s3:::${var.jurong_port_model_artifact_bucket_name}/*"]
  }
}

resource "aws_iam_role_policy" "jurong_port_s3_model_read" {
  name   = "${var.jurong_port_project_name}-s3-model-read-jl"
  role   = aws_iam_role.jurong_port_sagemaker_execution.id
  policy = data.aws_iam_policy_document.jurong_port_s3_model_read.json
}

# --- CloudWatch Logs: SageMaker writes container stdout/stderr + metrics here ---
data "aws_iam_policy_document" "jurong_port_cloudwatch_logging" {
  statement {
    sid = "SageMakerLogGroups"
    actions = [
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:PutLogEvents",
      "logs:DescribeLogStreams",
    ]
    resources = [
      "arn:aws:logs:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:log-group:/aws/sagemaker/*",
    ]
  }

  statement {
    sid       = "CloudWatchMetrics"
    actions   = ["cloudwatch:PutMetricData"]
    resources = ["*"] # PutMetricData does not support resource-level restriction
    condition {
      test     = "StringEquals"
      variable = "cloudwatch:namespace"
      values   = ["/aws/sagemaker/Endpoints", "AWS/SageMaker"]
    }
  }
}

resource "aws_iam_role_policy" "jurong_port_cloudwatch_logging" {
  name   = "${var.jurong_port_project_name}-cloudwatch-logging-jl"
  role   = aws_iam_role.jurong_port_sagemaker_execution.id
  policy = data.aws_iam_policy_document.jurong_port_cloudwatch_logging.json
}

output "jurong_port_sagemaker_execution_role_arn" {
  value = aws_iam_role.jurong_port_sagemaker_execution.arn
}
