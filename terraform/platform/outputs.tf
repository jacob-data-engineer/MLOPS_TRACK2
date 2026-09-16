output "pipeline_bucket" {
  value = aws_s3_bucket.pipeline.bucket
}

output "training_ecr_repository_uri" {
  value = aws_ecr_repository.yolo_train.repository_url
}

output "sagemaker_execution_role_arn" {
  value = aws_iam_role.sagemaker_execution.arn
}

output "model_package_group_names" {
  description = "Map of size (s/m/l/x) -> its Model Package Group name."
  value       = { for size, g in aws_sagemaker_model_package_group.yolo : size => g.model_package_group_name }
}

output "review_queue_table" {
  value = aws_dynamodb_table.review_queue.name
}

output "review_host_instance_id" {
  value = aws_instance.review_host.id
}

output "review_app_url" {
  value = "http://${aws_instance.review_host.public_ip}:${var.review_app_port}/"
}

output "monitor_drift_sns_topic_arn" {
  value = aws_sns_topic.monitor_drift.arn
}

output "reviewer_notifications_topic_arn" {
  value = aws_sns_topic.reviewer_notifications.arn
}
