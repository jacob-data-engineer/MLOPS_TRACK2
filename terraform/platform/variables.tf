variable "region" {
  description = "AWS region for all Track2 YOLO pipeline infra"
  type        = string
  default     = "ap-southeast-1"
}

variable "account_id" {
  description = "AWS account ID (used to namespace globally-unique resource names)"
  type        = string
  default     = "125192671932"
}

variable "project" {
  description = "Short name prefixing every resource this stack creates"
  type        = string
  default     = "track2-yolo"
}

variable "reviewer_emails" {
  description = "Reviewer emails, subscribed to the reviewer_notifications SNS topic (each must confirm via the email SNS sends once). Add more later by adding to this list and re-applying."
  type        = list(string)
  default     = ["2570475@sit.singaporetech.edu.sg"]
}

variable "training_instance_type" {
  description = "GPU instance type for the YOLO TrainingStep"
  type        = string
  default     = "ml.g4dn.xlarge"
}

variable "review_app_port" {
  description = "Port the review-app container listens on"
  type        = number
  default     = 8090
}

variable "operator_ips" {
  description = "Public IPs (bare, no /32) allowed to reach the review app -- add to this list as more people/networks need access, rather than replacing the existing entries. Ignored if review_app_public is true."
  type        = list(string)
}

variable "review_app_public" {
  description = "DANGER: if true, opens the review app to the entire internet (0.0.0.0/0) instead of just operator_ips. The app has NO authentication of its own -- it relies entirely on this network restriction. With this true, anyone with the URL can trigger real paid SageMaker training jobs, approve/reject/delete registered model versions, and change the retrain schedule, with no login at all. Explicitly confirmed with the user (2026-09-07) before ever setting this true. Default false -- must be passed explicitly (-var='review_app_public=true') to open it."
  type        = bool
  default     = false
}

