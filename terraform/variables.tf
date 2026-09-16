variable "region" {
  description = "AWS region everything here lives in"
  type        = string
  default     = "ap-southeast-1"
}

variable "app_name" {
  description = "Names the ECR repo, the CodeBuild project and the build role"
  type        = string
  default     = "track2-app"
}

variable "pipeline_bucket" {
  description = "Existing S3 bucket used to hand build source to CodeBuild. Owned by the jurong-port stack; read-only here."
  type        = string
  default     = "track2-s3"
}

variable "source_prefix" {
  description = "Key prefix within pipeline_bucket for the zipped build source"
  type        = string
  default     = "track2-app-source"
}

variable "review_host_name" {
  description = "Name tag of the EC2 instance the app is deployed onto. Looked up rather than hardcoded so this stack does not carry an instance id that can silently go stale."
  type        = string
  default     = "track2-yolo-review-host"
}

variable "github_token_parameter" {
  description = "SSM SecureString parameter holding a GitHub PAT with read access to SNAIC-JP/Track2, which the build needs to install aiops-pipeline. Created out of band -- see main.tf's header -- so its value never enters terraform state."
  type        = string
  default     = "/track2-yolo/github-token"
}
