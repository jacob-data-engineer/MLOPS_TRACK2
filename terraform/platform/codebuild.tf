# Builds and pushes the training/serving containers via AWS CodeBuild
# instead of locally — the local machine hit a real disk-space wall
# (7.7GB free, 99% full) partway through pulling the multi-GB SageMaker
# GPU base images. CodeBuild runs on AWS's own disk, so this sidesteps
# that entirely and is more resilient for future rebuilds too.
#
# Source is S3 (a zip of model_training/training-container/), not GitHub
# directly — the GitHub repo is private, and wiring CodeBuild up to a
# private repo needs a CodeStar Connection that requires a manual
# one-time OAuth approval in the console. S3 avoids that; the zip must be
# (re)built and dropped at model_training/training-container-src.zip
# whenever model_training/training-container/ changes.

resource "aws_s3_object" "codebuild_source" {
  bucket = aws_s3_bucket.pipeline.id
  key    = "codebuild-source/training-container.zip"
  source = "${path.module}/../../model_training/training-container-src.zip"
  etag   = filemd5("${path.module}/../../model_training/training-container-src.zip")
}

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
  name               = "${var.project}-codebuild"
  assume_role_policy = data.aws_iam_policy_document.codebuild_trust.json
}

data "aws_iam_policy_document" "codebuild_permissions" {
  statement {
    sid       = "Logs"
    effect    = "Allow"
    actions   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["arn:aws:logs:${var.region}:${var.account_id}:log-group:/aws/codebuild/${var.project}-train-images*"]
  }
  statement {
    sid       = "ReadSource"
    effect    = "Allow"
    actions   = ["s3:GetObject", "s3:GetObjectVersion"]
    resources = ["${aws_s3_bucket.pipeline.arn}/codebuild-source/*"]
  }
  statement {
    sid       = "EcrAuth"
    effect    = "Allow"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }
  statement {
    # Pulling the SageMaker DLC base images (account 763104351884, not
    # ours) needs these READ actions unrestricted — cross-account ECR
    # pull relies on the OTHER account's repo policy being permissive
    # plus this side allowing the action on Resource "*"; you can't
    # reference another account's specific ARN meaningfully here. This
    # is what a 403 Forbidden on the base image's manifest HEAD request
    # means: not an auth failure (login succeeded), an IAM one.
    sid       = "EcrPullAnyRegistry"
    effect    = "Allow"
    actions   = ["ecr:BatchCheckLayerAvailability", "ecr:GetDownloadUrlForLayer", "ecr:BatchGetImage"]
    resources = ["*"]
  }
  statement {
    sid       = "EcrPushOwnRepo"
    effect    = "Allow"
    actions   = ["ecr:PutImage", "ecr:InitiateLayerUpload", "ecr:UploadLayerPart", "ecr:CompleteLayerUpload"]
    resources = [aws_ecr_repository.yolo_train.arn]
  }
}

resource "aws_iam_policy" "codebuild" {
  name   = "${var.project}-codebuild"
  policy = data.aws_iam_policy_document.codebuild_permissions.json
}

resource "aws_iam_role_policy_attachment" "codebuild" {
  role       = aws_iam_role.codebuild.name
  policy_arn = aws_iam_policy.codebuild.arn
}

resource "aws_codebuild_project" "train_images" {
  name          = "${var.project}-train-images"
  description   = "Builds + pushes the YOLO training/serving containers to ECR, off the local machine's disk"
  service_role  = aws_iam_role.codebuild.arn
  build_timeout = 30

  source {
    type      = "S3"
    location  = "${aws_s3_bucket.pipeline.bucket}/${aws_s3_object.codebuild_source.key}"
    buildspec = <<-EOT
      version: 0.2
      env:
        variables:
          REPO_URI: "${aws_ecr_repository.yolo_train.repository_url}"
          # Resolved once via the SageMaker SDK on the account/region this
          # stack actually runs in — see training-container/build_and_push.sh's
          # own comment for why this isn't hand-guessed.
          TRAIN_BASE: "763104351884.dkr.ecr.${var.region}.amazonaws.com/pytorch-training:2.3-gpu-py311"
          SERVE_BASE: "763104351884.dkr.ecr.${var.region}.amazonaws.com/pytorch-inference:2.3-gpu-py311"
      phases:
        pre_build:
          commands:
            - aws ecr get-login-password --region $AWS_DEFAULT_REGION | docker login --username AWS --password-stdin $REPO_URI
            - aws ecr get-login-password --region $AWS_DEFAULT_REGION | docker login --username AWS --password-stdin 763104351884.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com
        build:
          commands:
            - docker build --build-arg BASE_IMAGE=$TRAIN_BASE -f Dockerfile -t $REPO_URI:train-latest .
            - docker build --build-arg BASE_IMAGE=$SERVE_BASE -f Dockerfile.serve -t $REPO_URI:serve-latest .
        post_build:
          commands:
            - docker push $REPO_URI:train-latest
            - docker push $REPO_URI:serve-latest
    EOT
  }

  artifacts {
    type = "NO_ARTIFACTS"
  }

  environment {
    type                        = "LINUX_CONTAINER"
    compute_type                = "BUILD_GENERAL1_LARGE" # generous disk/CPU margin for multi-GB image layers
    image                       = "aws/codebuild/standard:7.0"
    image_pull_credentials_type = "CODEBUILD"
    privileged_mode             = true # required for `docker build`/`docker push` inside the build
  }
}
