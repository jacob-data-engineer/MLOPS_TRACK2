# Custom training container (ultralytics + boto3 + the SageMaker training
# toolkit), same "bring your own image, push to ECR" pattern JP_UI's
# deploy.yml already uses for ppe-pipeline — avoids a fragile pip-install
# of ultralytics inside SageMaker's stock PyTorch container at job start.
# Dockerfile + build/push script land in ../../model_training/training-container/
# (next step, alongside the pipeline_definition.py-equivalent Python).

resource "aws_ecr_repository" "yolo_train" {
  name                 = "${var.project}-train"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}
