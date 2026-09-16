# Runs src/run_pipeline.py entirely on AWS (via CodeBuild), instead of
# from whichever machine happens to invoke it. The actual YOLO training
# always ran on SageMaker's own managed compute regardless of where
# run_pipeline.py was invoked from — this closes the remaining gap,
# where the lightweight "define + start + wait on the pipeline" API
# calls were still made from a local machine. Same S3-source pattern as
# codebuild.tf, for the same reason (private GitHub repo, no CodeStar
# Connection OAuth step needed).

resource "aws_s3_object" "run_pipeline_source" {
  bucket = aws_s3_bucket.pipeline.id
  key    = "codebuild-source/run-pipeline-src.zip"
  source = "${path.module}/../../model_training/run-pipeline-src.zip"
  etag   = filemd5("${path.module}/../../model_training/run-pipeline-src.zip")
}

resource "aws_iam_role" "codebuild_run_pipeline" {
  name               = "${var.project}-codebuild-run-pipeline"
  assume_role_policy = data.aws_iam_policy_document.codebuild_trust.json
}

data "aws_iam_policy_document" "codebuild_run_pipeline_permissions" {
  statement {
    sid       = "Logs"
    effect    = "Allow"
    actions   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["arn:aws:logs:${var.region}:${var.account_id}:log-group:/aws/codebuild/${var.project}-run-pipeline*"]
  }
  statement {
    sid       = "ReadSource"
    effect    = "Allow"
    actions   = ["s3:GetObject", "s3:GetObjectVersion"]
    resources = ["${aws_s3_bucket.pipeline.arn}/codebuild-source/*"]
  }
  statement {
    # pipeline_definition.py's _write_run_metadata() writes one JSON
    # record per run here (dataset/YOLO/base-model/notes) -- runs under
    # this CodeBuild role at build_pipeline() time, same reasoning as
    # ManagePipeline below (not the sagemaker_execution role).
    sid       = "WriteRunMetadata"
    effect    = "Allow"
    actions   = ["s3:PutObject"]
    resources = ["${aws_s3_bucket.pipeline.arn}/training-runs/*"]
  }
  statement {
    # Same _write_run_metadata() call also writes the same record into
    # track2-dynamoDB (reused existing empty table, hash key "id") for
    # fast indexed lookup -- see review_app_access's matching GetItem
    # grant for the read side. S3 write above stays too, as the durable
    # full-fidelity copy.
    sid       = "WriteRunLineageToDynamoDB"
    effect    = "Allow"
    actions   = ["dynamodb:PutItem"]
    resources = [aws_dynamodb_table.track2_dynamodb.arn]
  }
  statement {
    # Pipeline ARNs don't exist yet at plan time (created by this job
    # itself on first run), so these stay Resource "*" rather than
    # referencing a not-yet-real ARN.
    #
    # ListModelPackages: pipeline_definition.py's
    # _resolve_latest_approved_base_model() calls this directly (via its
    # own boto3 client) at build_pipeline() time -- runs under whichever
    # identity invokes run_pipeline.py, i.e. this CodeBuild role, not the
    # sagemaker_execution role the pipeline itself later assumes. Without
    # it, the very first call inside build_pipeline() fails before the
    # pipeline definition is even constructed. Same lesson as
    # review_app_access's ModelRegistryList statement: List-type actions
    # need Resource "*", scoped ARNs don't satisfy the resource-level
    # permission check for them.
    sid    = "ManagePipeline"
    effect = "Allow"
    actions = [
      "sagemaker:CreatePipeline", "sagemaker:UpdatePipeline", "sagemaker:DescribePipeline",
      "sagemaker:StartPipelineExecution", "sagemaker:DescribePipelineExecution",
      "sagemaker:ListPipelineExecutionSteps", "sagemaker:ListPipelines",
      "sagemaker:CreateModelPackage", "sagemaker:DescribeModelPackageGroup", "sagemaker:CreateModelPackageGroup",
      "sagemaker:ListModelPackages",
      "sagemaker:CreateTrainingJob", "sagemaker:DescribeTrainingJob",
      "sagemaker:CreateModel", "sagemaker:DescribeModel", "sagemaker:AddTags",
    ]
    resources = ["*"]
  }
  statement {
    # The pipeline needs to hand this role to SageMaker when it creates
    # training jobs on our behalf — without PassRole, CreateTrainingJob
    # fails even though the caller (this CodeBuild role) never assumes
    # sagemaker_execution itself.
    sid       = "PassSageMakerExecutionRole"
    effect    = "Allow"
    actions   = ["iam:PassRole"]
    resources = [aws_iam_role.sagemaker_execution.arn]
  }
  statement {
    # sagemaker.Session()/PipelineSession() always checks permissions on
    # the SDK's own default bucket (sagemaker-<region>-<account>, separate
    # from our track2-s3 pipeline bucket) as part of construction --
    # confirmed live: build_pipeline() failed with "403 Forbidden" on
    # HeadBucket before it ever got to defining the pipeline, since this
    # role had zero S3 access to that bucket. Not hit when run locally
    # because local credentials have broader S3 access.
    sid    = "SageMakerDefaultBucket"
    effect = "Allow"
    actions = [
      "s3:ListBucket", "s3:GetBucketLocation", "s3:CreateBucket",
      "s3:GetObject", "s3:PutObject",
    ]
    resources = [
      "arn:aws:s3:::sagemaker-${var.region}-${var.account_id}",
      "arn:aws:s3:::sagemaker-${var.region}-${var.account_id}/*",
    ]
  }
}

resource "aws_iam_policy" "codebuild_run_pipeline" {
  name   = "${var.project}-codebuild-run-pipeline"
  policy = data.aws_iam_policy_document.codebuild_run_pipeline_permissions.json
}

resource "aws_iam_role_policy_attachment" "codebuild_run_pipeline" {
  role       = aws_iam_role.codebuild_run_pipeline.name
  policy_arn = aws_iam_policy.codebuild_run_pipeline.arn
}

resource "aws_codebuild_project" "run_pipeline" {
  name          = "${var.project}-run-pipeline"
  description   = "Runs the YOLO training pipeline end-to-end from AWS, not a local machine"
  service_role  = aws_iam_role.codebuild_run_pipeline.arn
  build_timeout = 60 # generous: waits for the actual training job to finish, not just triggers it

  source {
    type      = "S3"
    location  = "${aws_s3_bucket.pipeline.bucket}/${aws_s3_object.run_pipeline_source.key}"
    buildspec = <<-EOT
      version: 0.2
      phases:
        install:
          runtime-versions:
            python: 3.12
          commands:
            - pip install --quiet -r requirements.txt
        build:
          commands:
            - python run_pipeline.py
    EOT
  }

  artifacts {
    type = "NO_ARTIFACTS"
  }

  environment {
    type                        = "LINUX_CONTAINER"
    compute_type                = "BUILD_GENERAL1_SMALL" # this job just calls the SageMaker API and polls; the real work runs elsewhere
    image                       = "aws/codebuild/standard:7.0"
    image_pull_credentials_type = "CODEBUILD"
  }
}

output "run_pipeline_codebuild_project" {
  value = aws_codebuild_project.run_pipeline.name
}
