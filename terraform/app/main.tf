# Build infrastructure for the Track2 app. Self-contained: its own state,
# its own IAM role, its own ECR repo. Nothing here is defined by, or
# depends on the state of, the jurong-port stack.
#
# What it does NOT own is the things that already exist and are shared --
# the pipeline S3 bucket and the review host. Those are read through data
# sources, so this stack can see them without being able to change or
# destroy them, and applying it can never disturb the stack that does own
# them.
#
#   cd MLOPS_TRACK2/terraform && terraform init && terraform apply
#
# One-time, before the first build (the app installs aiops-pipeline from
# the private SNAIC-JP/Track2 repo):
#
#   aws ssm put-parameter --name /track2-yolo/github-token \
#     --type SecureString --value ghp_xxx --region ap-southeast-1

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region
}

data "aws_caller_identity" "current" {}

# Read-only handles on what the other stack owns.
data "aws_s3_bucket" "pipeline" {
  bucket = var.pipeline_bucket
}

data "aws_instances" "review_host" {
  instance_tags = {
    Name = var.review_host_name
  }
  instance_state_names = ["running", "stopped"]
}

locals {
  account_id = data.aws_caller_identity.current.account_id
}

# --- where built images land --------------------------------------------

resource "aws_ecr_repository" "app" {
  name = var.app_name

  image_scanning_configuration {
    scan_on_push = true
  }
}

# Each build is a multi-GB image and the deploy always runs the tag it just
# produced, so without an expiry this repo grows without limit for the sake
# of tags nobody will roll back to.
resource "aws_ecr_lifecycle_policy" "app" {
  repository = aws_ecr_repository.app.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep the 10 most recent images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 10
      }
      action = { type = "expire" }
    }]
  })
}

# --- the build role ------------------------------------------------------

data "aws_iam_policy_document" "codebuild_trust" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["codebuild.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "codebuild" {
  name               = "${var.app_name}-codebuild"
  assume_role_policy = data.aws_iam_policy_document.codebuild_trust.json
}

data "aws_iam_policy_document" "codebuild" {
  statement {
    sid       = "Logs"
    effect    = "Allow"
    actions   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["arn:aws:logs:${var.region}:${local.account_id}:log-group:/aws/codebuild/${var.app_name}*"]
  }
  statement {
    sid       = "ReadSource"
    effect    = "Allow"
    actions   = ["s3:GetObject", "s3:GetObjectVersion"]
    resources = ["${data.aws_s3_bucket.pipeline.arn}/${var.source_prefix}/*"]
  }
  statement {
    sid       = "EcrAuth"
    effect    = "Allow"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }
  statement {
    # Pulling a base image from a registry that isn't ours needs these
    # unrestricted: the resource being read is not addressable from this
    # side, so it cannot be scoped to an ARN.
    sid       = "EcrPullAnyRegistry"
    effect    = "Allow"
    actions   = ["ecr:BatchCheckLayerAvailability", "ecr:GetDownloadUrlForLayer", "ecr:BatchGetImage"]
    resources = ["*"]
  }
  statement {
    sid       = "EcrPushAppRepo"
    effect    = "Allow"
    actions   = ["ecr:PutImage", "ecr:InitiateLayerUpload", "ecr:UploadLayerPart", "ecr:CompleteLayerUpload"]
    resources = [aws_ecr_repository.app.arn]
  }
  statement {
    sid       = "ReadGithubToken"
    effect    = "Allow"
    actions   = ["ssm:GetParameters"]
    resources = ["arn:aws:ssm:${var.region}:${local.account_id}:parameter${var.github_token_parameter}"]
  }
}

resource "aws_iam_policy" "codebuild" {
  name   = "${var.app_name}-codebuild"
  policy = data.aws_iam_policy_document.codebuild.json
}

resource "aws_iam_role_policy_attachment" "codebuild" {
  role       = aws_iam_role.codebuild.name
  policy_arn = aws_iam_policy.codebuild.arn
}

# --- the build itself ----------------------------------------------------

resource "aws_codebuild_project" "app" {
  name         = var.app_name
  description  = "Builds + pushes the Track2 app (FastAPI + React) to ECR, off any local machine"
  service_role = aws_iam_role.codebuild.arn
  # A node stage plus a torch/ultralytics install on a cold cache runs well
  # past the 30 minutes that is enough for the smaller training images.
  build_timeout = 60

  source {
    type      = "S3"
    location  = "${data.aws_s3_bucket.pipeline.bucket}/${var.source_prefix}/app.zip"
    buildspec = <<-EOT
      version: 0.2
      env:
        variables:
          REPO_URI: "${aws_ecr_repository.app.repository_url}"
        parameter-store:
          TRACK_REPOS_TOKEN: "${var.github_token_parameter}"
      phases:
        pre_build:
          commands:
            - aws ecr get-login-password --region $AWS_DEFAULT_REGION | docker login --username AWS --password-stdin $REPO_URI
            # IMAGE_TAG comes from the deploy script, so the tag the
            # instance is told to run is exactly the one built here. A
            # fixed "latest" cannot tell two overlapping deploys apart.
            - echo "Building $REPO_URI:$IMAGE_TAG"
        build:
          commands:
            # The token reaches docker as a BuildKit secret, never as an
            # ARG or ENV -- those survive into image layers and history.
            - printf '%s' "$TRACK_REPOS_TOKEN" > /tmp/token
            - DOCKER_BUILDKIT=1 docker build --secret id=track_repos_token,src=/tmp/token -t $REPO_URI:$IMAGE_TAG .
            - rm -f /tmp/token
        post_build:
          commands:
            - docker push $REPO_URI:$IMAGE_TAG
    EOT
  }

  artifacts {
    type = "NO_ARTIFACTS"
  }

  environment {
    type                        = "LINUX_CONTAINER"
    compute_type                = "BUILD_GENERAL1_LARGE"
    image                       = "aws/codebuild/standard:7.0"
    image_pull_credentials_type = "CODEBUILD"
    privileged_mode             = true # required for docker build/push inside the build
  }
}

# --- letting the instance pull what was built ----------------------------

# The review host's role is granted ECR pull by the stack that owns the
# instance, scoped to the repos that existed when that was written -- so
# pulling this one came back 403 Forbidden even though `docker login`
# succeeded. The grant is attached here, to the repository this stack
# owns, rather than by widening the other stack's IAM: the fix then lives
# with the thing being accessed, and applying it cannot disturb the stack
# that owns the instance.
#
# Same account, so a repository policy alone authorizes the pull. The one
# account-level action (ecr:GetAuthorizationToken) is already in the
# instance role -- the login in the deploy log proves it.
data "aws_iam_role" "review_host" {
  name = var.review_host_role_name
}

resource "aws_ecr_repository_policy" "app_pull" {
  repository = aws_ecr_repository.app.name
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid       = "AllowReviewHostPull"
      Effect    = "Allow"
      Principal = { AWS = data.aws_iam_role.review_host.arn }
      Action = [
        "ecr:BatchCheckLayerAvailability",
        "ecr:BatchGetImage",
        "ecr:GetDownloadUrlForLayer",
      ]
    }]
  })
}
