# Read the exact pipeline revision before recording a repeatable run in SQL.
resource "aws_iam_role_policy" "prepared_training_versions" {
  name = "${var.app_name}-prepared-training-versions"
  role = data.aws_iam_role.review_host.name
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["sagemaker:DescribePipeline", "sagemaker:ListPipelineVersions", "sagemaker:UpdatePipeline"]
      Resource = "arn:aws:sagemaker:${var.region}:${local.account_id}:pipeline/${var.open_pipeline_name}"
    }, {
      Effect   = "Allow"
      Action   = ["ecr:DescribeImages"]
      Resource = "arn:aws:ecr:${var.region}:${local.account_id}:repository/track2-yolo-train"
    }]
  })
}
