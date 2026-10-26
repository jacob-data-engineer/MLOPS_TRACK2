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
  description = "Existing S3 bucket used to hand build source to CodeBuild. Owned by the platform stack; read-only here."
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

variable "review_host_role_name" {
  description = "IAM role the review host runs as. Granted pull access to this stack's ECR repo, since the stack that owns the role only grants it the repos that existed when it was written."
  type        = string
  default     = "track2-yolo-review-host"
}

variable "training_audit_table" {
  description = "DynamoDB table holding one row per submitted training run: both dataset folders, every parameter, the notes, the registered model version, and start/end times. The file names live in training_files_table -- a run's list of them does not fit in a single 400KB item."
  type        = string
  default     = "track2-yolo-training-audit"
}

variable "training_files_table" {
  description = "DynamoDB table holding one row per file a training run used, keyed (audit_id, file_key). Separate from the audit row because a 24,000-image dataset's names are ~2.4MB and a DynamoDB item is capped at 400KB. Its by-file-name index answers which runs used a given image."
  type        = string
  default     = "track2-yolo-training-files"
}

variable "dataset_snapshot_table" {
  description = "DynamoDB table holding one row per frozen dataset definition: the data.yaml's hash, the per-week image counts at the moment it was taken, and a pointer to the S3 manifest listing every file it selected. The manifest itself is in S3, not here -- a 24,000-image dataset's names are ~2.4MB against a 400KB item cap -- and the counts are on the row so drift is readable without fetching it."
  type        = string
  default     = "track2-yolo-dataset-snapshots"
}

variable "training_pipeline_name" {
  description = "SageMaker pipeline the Training page starts and stops runs on."
  type        = string
  default     = "track2-yolo-training-pipeline"
}

variable "custom_pipeline_name" {
  description = "Separate pipeline for models trained on a dataset's own classes. Kept apart from the production one so a model whose classes are not person/helmet cannot enter the replay lineage."
  type        = string
  default     = "track2-yolo-custom-training-pipeline"
}

variable "open_pipeline_name" {
  description = "Where custom-class runs actually go. The custom pipeline above runs the digest-pinned recovery image, whose wrapper asserts the BASE checkpoint's classes are person/helmet -- so it can only fine-tune from a production checkpoint. This one runs track2-yolo-train (no such assertion) and is what the scratch / registry / uploaded base-model options need."
  type        = string
  default     = "track2-yolo-open-training-pipeline"
}
