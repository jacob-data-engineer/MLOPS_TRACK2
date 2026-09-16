output "ecr_repository_uri" {
  value       = aws_ecr_repository.app.repository_url
  description = "Where built app images land."
}

output "codebuild_project" {
  value       = aws_codebuild_project.app.name
  description = "CodeBuild project that builds the app image."
}

output "source_bucket" {
  value       = data.aws_s3_bucket.pipeline.bucket
  description = "Bucket the deploy script uploads build source to."
}

output "source_prefix" {
  value       = var.source_prefix
  description = "Key prefix within source_bucket for the zipped build source."
}

output "review_host_instance_id" {
  # Looked up by Name tag rather than stored, so it cannot drift from
  # reality if the instance is ever replaced.
  value       = try(data.aws_instances.review_host.ids[0], "")
  description = "Instance the app is deployed onto."
}
