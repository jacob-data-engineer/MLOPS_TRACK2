# Track2 YOLO retraining + serving pipeline — static scaffolding only.
#
# Deliberately separate from Track2's existing applied infra (the real
# Track2SageMakerExecutionRole + 4 Model Package Groups in
# SNAIC-JP/Track2's own infra/) — nothing here touches those, per explicit
# choice: this is a fully parallel stack so the new YOLO/Monitor/A2I/
# Endpoint work can't collide with or destabilize what Track2 already has
# applied for its (placeholder-model) curation->train->register->UAT
# pipeline.
#
# Split in two layers, matching Track1/Track2's own convention:
#   - Terraform (here): S3, ECR, IAM, the Model Package Group, the A2I
#     workteam/human-task-UI/flow-definition, and the CloudWatch alarm +
#     Lambda that turns a Model Monitor violation into a human review.
#   - Python/SageMaker SDK (../scripts, ../../model_training/src, written next): the actual
#     Pipeline definition (curate -> train YOLO -> register), the
#     serverless endpoint deployment, and the Data Quality monitoring
#     schedule — none of these can be a static Terraform resource, since
#     they all need a real trained model's artifact to point at.
#
# Same AWS account as Track1/Track2/JP_UI (125192671932). Region
# ap-southeast-1, matching where the real hazard-image S3 data actually
# lives (safety-model-dataset-dev) and where Track2's own infra already
# runs — verified via aws s3api get-bucket-location.

terraform {
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
