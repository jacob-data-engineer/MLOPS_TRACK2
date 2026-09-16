# One bucket, prefix-organized — mirrors how Track2's real infra already
# organizes safety-model-dataset-dev under a track2/ prefix rather than
# spinning up a bucket per pipeline stage.
#
# images/raw/          copied from the real hazard images in
#                       safety-model-dataset-dev (275 images, no boxes)
# images/labels/        synthetic YOLO .txt boxes derived from filename
#                       category tags — proof-of-mechanics only, see
#                       scripts/seed_training_data.py
# training-output/      SageMaker TrainingStep output (weights, logs)
# model-registry/       packaged artifacts for approved model versions
# data-capture/         real-time endpoint inference capture — required
#                       for the Data Quality Model Monitor to have
#                       anything to compare against the training baseline
# monitor/baseline/     Model Monitor's baseline statistics + constraints
# monitor/reports/      per-execution monitoring violation reports
# a2i/output/           Amazon A2I human loop output (reviewer decisions)

resource "aws_s3_bucket" "pipeline" {
  # Hardcoded rather than "${var.project}-pipeline-${var.account_id}": renamed
  # at explicit request, decoupled from the project-wide naming convention
  # the rest of this stack's resources use. This is a full delete+recreate,
  # not an in-place rename (S3 has no rename API) -- all data in the old
  # bucket was deleted first (explicitly confirmed), so this create is clean.
  bucket = "track2-s3"
}

resource "aws_s3_bucket_versioning" "pipeline" {
  bucket = aws_s3_bucket.pipeline.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_public_access_block" "pipeline" {
  bucket                  = aws_s3_bucket.pipeline.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "pipeline" {
  bucket = aws_s3_bucket.pipeline.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}
