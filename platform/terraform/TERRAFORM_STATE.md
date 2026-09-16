# Track2 Terraform State — tracked snapshot

Not a source of truth by itself — always re-verify with `terraform state list`
/ `terraform plan` before trusting this after a gap in time. Updated by
Claude whenever asked to "keep track of terraform state" or after any
apply/destroy/import/state mv in this project.

Last captured: 2026-09-01, from `c:\SNAIC\terraformer_track_2\infra` (`terraform state list`).

## Recent change
The 5 endpoint autoscaling resources (`aws_appautoscaling_target.yolo_endpoint`,
`aws_appautoscaling_policy.yolo_endpoint_scale_to_zero`,
`aws_appautoscaling_policy.yolo_endpoint_scale_out_from_zero`,
`aws_cloudwatch_metric_alarm.yolo_endpoint_idle`,
`aws_cloudwatch_metric_alarm.yolo_endpoint_backlog_without_capacity`) were
destroyed — confirmed gone from state as of this capture. Reason: nothing in
the project ever calls `invoke_endpoint`/`invoke_endpoint_async` on
`track2-yolo-endpoint`, so the endpoint (and its autoscaling) was unused.
Still pending: delete the live endpoint + endpoint config itself (not
Terraform-managed — created by `src/deploy_endpoint.py` via boto3), and strip
the now-dead resource blocks out of `infra/autoscaling.tf` +
`infra/variables.tf`.

## Full resource list (66 resources)

### Data sources
- data.archive_file.confidence_monitor
- data.archive_file.dump_approved_weights
- data.archive_file.pull_training_data
- data.archive_file.trigger_review
- data.archive_file.trigger_weekly_retrain
- data.aws_ami.ubuntu
- data.aws_caller_identity.current
- data.aws_iam_policy_document.codebuild_permissions
- data.aws_iam_policy_document.codebuild_run_pipeline_permissions
- data.aws_iam_policy_document.codebuild_trust
- data.aws_iam_policy_document.confidence_monitor_lambda_permissions
- data.aws_iam_policy_document.dump_approved_weights_lambda_permissions
- data.aws_iam_policy_document.ec2_trust
- data.aws_iam_policy_document.lambda_trust
- data.aws_iam_policy_document.pull_training_data_lambda_permissions
- data.aws_iam_policy_document.review_app_access
- data.aws_iam_policy_document.review_host_scheduler_permissions
- data.aws_iam_policy_document.review_trigger_lambda_permissions
- data.aws_iam_policy_document.sagemaker_s3_scoped
- data.aws_iam_policy_document.sagemaker_trust
- data.aws_iam_policy_document.scheduler_trust
- data.aws_iam_policy_document.weekly_retrain_lambda_permissions
- data.aws_subnets.default
- data.aws_vpc.default

### EventBridge / scheduling
- aws_cloudwatch_event_rule.confidence_monitor_schedule
- aws_cloudwatch_event_rule.model_package_approved
- aws_cloudwatch_event_rule.weekly_retrain_schedule
- aws_cloudwatch_event_target.confidence_monitor
- aws_cloudwatch_event_target.dump_approved_weights
- aws_cloudwatch_event_target.weekly_retrain
- aws_scheduler_schedule.review_host_start
- aws_scheduler_schedule.review_host_stop

### Monitoring
- aws_cloudwatch_metric_alarm.confidence_drift

### CodeBuild
- aws_codebuild_project.run_pipeline
- aws_codebuild_project.train_images

### Storage
- aws_dynamodb_table.review_queue
- aws_dynamodb_table.track2_dynamodb
- aws_ecr_repository.yolo_train
- aws_s3_bucket.pipeline
- aws_s3_bucket_notification.track2_aws_s3_bucket_notification_latest_json
- aws_s3_bucket_public_access_block.pipeline
- aws_s3_bucket_server_side_encryption_configuration.pipeline
- aws_s3_bucket_versioning.pipeline
- aws_s3_object.codebuild_source
- aws_s3_object.run_pipeline_source

### IAM
- aws_iam_instance_profile.review_host
- aws_iam_policy.codebuild
- aws_iam_policy.codebuild_run_pipeline
- aws_iam_policy.confidence_monitor_lambda
- aws_iam_policy.dump_approved_weights_lambda
- aws_iam_policy.pull_training_data_lambda
- aws_iam_policy.review_app_access
- aws_iam_policy.review_host_scheduler
- aws_iam_policy.review_trigger_lambda
- aws_iam_policy.sagemaker_s3_scoped
- aws_iam_policy.weekly_retrain_lambda
- aws_iam_role.codebuild
- aws_iam_role.codebuild_run_pipeline
- aws_iam_role.confidence_monitor_lambda
- aws_iam_role.dump_approved_weights_lambda
- aws_iam_role.pull_training_data_lambda
- aws_iam_role.review_host
- aws_iam_role.review_host_scheduler
- aws_iam_role.review_trigger_lambda
- aws_iam_role.sagemaker_execution
- aws_iam_role.weekly_retrain_lambda
- aws_iam_role_policy_attachment.codebuild
- aws_iam_role_policy_attachment.codebuild_run_pipeline
- aws_iam_role_policy_attachment.confidence_monitor_lambda
- aws_iam_role_policy_attachment.dump_approved_weights_lambda
- aws_iam_role_policy_attachment.pull_training_data_lambda
- aws_iam_role_policy_attachment.review_host_access
- aws_iam_role_policy_attachment.review_host_scheduler
- aws_iam_role_policy_attachment.review_host_ssm
- aws_iam_role_policy_attachment.review_trigger_lambda
- aws_iam_role_policy_attachment.sagemaker_execution_managed
- aws_iam_role_policy_attachment.sagemaker_execution_scoped
- aws_iam_role_policy_attachment.weekly_retrain_lambda

### Compute / networking
- aws_instance.review_host
- aws_security_group.review_host

### Lambda
- aws_lambda_function.confidence_monitor
- aws_lambda_function.dump_approved_weights
- aws_lambda_function.pull_training_data
- aws_lambda_function.trigger_review
- aws_lambda_function.weekly_retrain
- aws_lambda_permission.allow_eventbridge
- aws_lambda_permission.allow_eventbridge_model_approval
- aws_lambda_permission.allow_s3_latest_json
- aws_lambda_permission.allow_sns
- aws_lambda_permission.allow_weekly_retrain_eventbridge

### SageMaker / SNS
- aws_sagemaker_model_package_group.yolo
- aws_sns_topic.monitor_drift
- aws_sns_topic.reviewer_notifications
- aws_sns_topic_subscription.reviewer_email["2570475@sit.singaporetech.edu.sg"]
- aws_sns_topic_subscription.trigger_review_lambda

## Not Terraform-managed (created via boto3 scripts, not tracked here)
- SageMaker Endpoint `track2-yolo-endpoint` + its EndpointConfig (`src/deploy_endpoint.py`) — pending deletion
- SageMaker Model resource `track2-yolo11-endpoint` (backs the endpoint config)
- SageMaker Model Packages inside `track2-yolo11` group (registry entries, e.g. `track2-yolo11/1`, `/2`)
