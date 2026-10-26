# Distinct roles per trust boundary — deliberately not reusing Track2's
# real Track2SageMakerExecutionRole (this stack is fully separate, per
# explicit choice), and not collapsing the SageMaker execution role, the
# EC2 instance's review-app access, and the Lambda role into one
# over-broad role even though they all touch this same S3 bucket.

# --- SageMaker execution role: runs curation/training/registration jobs
# and hosts the serverless endpoint --------------------------------------

data "aws_iam_policy_document" "sagemaker_trust" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "sagemaker_execution" {
  name               = "${var.project}-sagemaker-execution"
  assume_role_policy = data.aws_iam_policy_document.sagemaker_trust.json
}

resource "aws_iam_role_policy_attachment" "sagemaker_execution_managed" {
  role       = aws_iam_role.sagemaker_execution.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

data "aws_iam_policy_document" "sagemaker_s3_scoped" {
  statement {
    sid    = "PipelineBucketReadWrite"
    effect = "Allow"
    actions = [
      "s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket",
    ]
    resources = [
      aws_s3_bucket.pipeline.arn,
      "${aws_s3_bucket.pipeline.arn}/*",
    ]
  }
  statement {
    sid       = "ECRPullTrainingImage"
    effect    = "Allow"
    actions   = ["ecr:GetDownloadUrlForLayer", "ecr:BatchGetImage", "ecr:BatchCheckLayerAvailability"]
    resources = [aws_ecr_repository.yolo_train.arn]
  }
  statement {
    sid       = "ECRAuth"
    effect    = "Allow"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }
}

resource "aws_iam_policy" "sagemaker_s3_scoped" {
  name   = "${var.project}-sagemaker-s3-ecr"
  policy = data.aws_iam_policy_document.sagemaker_s3_scoped.json
}

resource "aws_iam_role_policy_attachment" "sagemaker_execution_scoped" {
  role       = aws_iam_role.sagemaker_execution.name
  policy_arn = aws_iam_policy.sagemaker_s3_scoped.arn
}

# --- Review-app access policy: attached to the dedicated review-host
# EC2 instance's own role (ec2.tf), not the shared prototype_validation_
# server's EC2_access — grants read/write on the review-queue table,
# read on the pipeline bucket's flagged images, and (for the Training
# Control tab) starting a manual pipeline run + editing the retrain
# schedule's cron/enabled state ------------------------------------------

data "aws_iam_policy_document" "review_app_access" {
  statement {
    sid       = "ReviewQueueReadWrite"
    effect    = "Allow"
    actions   = ["dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:UpdateItem", "dynamodb:Scan", "dynamodb:Query"]
    resources = [aws_dynamodb_table.review_queue.arn]
  }
  statement {
    # Read-only: shows each registry row's full lineage (dataset, base
    # model, hyperparameters, notes) via the "lineage_id" cross-linked
    # onto the package's own CustomerMetadataProperties at registration
    # time -- see pipeline_definition.py's _write_run_metadata() and
    # register_baseline_model.py's matching write.
    # dynamodb:Scan added for the review app's read-only "Lineage" tab,
    # which lists every training_run/baseline record in the table (not
    # just a single id lookup) -- see review_app.py's _scan_all_lineage.
    sid       = "ReadRunLineage"
    effect    = "Allow"
    actions   = ["dynamodb:GetItem", "dynamodb:Scan"]
    resources = [aws_dynamodb_table.track2_dynamodb.arn]
  }
  statement {
    # The eval-run and inference-job queues. Both moved from in-memory
    # lists to DynamoDB so a queued job survives an app restart, but the
    # tables were created outside this stack and no policy ever granted
    # access to them -- so GET /track2/deploy died on the queue Scan with
    # AccessDenied, i.e. the whole Deploy & Inference page returned 500.
    #
    # Addressed by constructed ARN rather than an aws_dynamodb_table
    # reference because neither table is Terraform-managed, the same
    # approach InvokeMultiSizeEndpoint below takes for the endpoint.
    # DeleteItem is needed for the queue history's own delete action.
    sid    = "InferenceLineageReadWrite"
    effect = "Allow"
    actions = [
      "dynamodb:GetItem",
      "dynamodb:PutItem",
      "dynamodb:Scan",
      "dynamodb:Query",
    ]
    resources = [aws_dynamodb_table.inference_lineage.arn]
  }
  statement {
    sid    = "QueueTablesReadWrite"
    effect = "Allow"
    actions = [
      "dynamodb:GetItem",
      "dynamodb:PutItem",
      "dynamodb:UpdateItem",
      "dynamodb:DeleteItem",
      "dynamodb:Scan",
      "dynamodb:Query",
    ]
    resources = [
      "arn:aws:dynamodb:${var.region}:${var.account_id}:table/track2-yolo-inference-queue",
      "arn:aws:dynamodb:${var.region}:${var.account_id}:table/track2-yolo-eval-runs",
    ]
  }
  statement {
    # GetObject: flagged review images. ListBucket: the Models tab's S3
    # folder pickers (list_objects_v2 with Delimiter needs it, not just
    # GetObject). PutObject: repackaging a raw model.pt into model.tar.gz
    # when deploying from the weights/ S3 picker instead of the registry.
    sid       = "FlaggedImageAndWeightsFolderAccess"
    effect    = "Allow"
    actions   = ["s3:GetObject", "s3:ListBucket", "s3:PutObject"]
    resources = [aws_s3_bucket.pipeline.arn, "${aws_s3_bucket.pipeline.arn}/*"]
  }
  statement {
    # Lets the Models tab deploy a selected weights version (or raw S3
    # folder) straight to that size's Inference Component -- see
    # review_app.py's _deploy_inference_component, which mirrors
    # model_training/src/deploy_inference_component.py. Resource "*" for
    # the same reason as monitoring.tf's DeployToInferenceComponent
    # statement (these actions don't support resource-level ARN scoping).
    sid    = "DeployToInferenceComponent"
    effect = "Allow"
    actions = [
      "sagemaker:CreateModel",
      "sagemaker:CreateInferenceComponent",
      "sagemaker:UpdateInferenceComponent",
      "sagemaker:DescribeInferenceComponent",
      "sagemaker:ListInferenceComponents",
      # Tearing the endpoint down needs the component removed first --
      # SageMaker rejects DeleteEndpoint while one is still attached.
      "sagemaker:DeleteInferenceComponent",
    ]
    resources = ["*"]
  }
  statement {
    # Backs the Cost tab. Cost Explorer has no resource-level scoping at
    # all (its ARNs aren't addressable), so "*" is the only valid form.
    # Read-only: GetCostAndUsage cannot change anything, and billing data
    # is account-wide by nature -- there is no SageMaker-only variant of
    # the permission, the service filter is applied in the query instead.
    #
    # pricing:GetProducts: the published on-demand rate, so today's spend
    # can be MEASURED rather than waited for. Cost Explorer lags roughly
    # a day, so on the afternoon of a day spent training it answers
    # $0.00 -- which reads as "this is free" rather than "ask again
    # tomorrow". Multiplying each job's BillableTimeInSeconds by the
    # rate agrees with the bill and answers on the day. Read-only, free,
    # and account-independent -- the price list is public.
    sid       = "ReadCostExplorer"
    effect    = "Allow"
    actions   = ["ce:GetCostAndUsage", "pricing:GetProducts"]
    resources = ["*"]
  }
  statement {
    # ensure_scale_to_zero registers a 0-1 scalable target on the active
    # Inference Component and suspends scale-IN for the duration of each
    # deploy (a scale-in firing mid-deploy collides with the component
    # update). Every deploy calls it, and the role had no
    # application-autoscaling permission whatsoever -- so the call died
    # with AccessDenied on RegisterScalableTarget and failed the job after
    # the endpoint had already been built, which is the worst possible
    # place to stop: paying for the instance, with nothing served.
    #
    # Target-tracking policies create and delete CloudWatch alarms on the
    # caller's behalf, so the alarm actions below are part of the same
    # grant, not a separate feature. None of these support resource-level
    # scoping in a form that works for a scalable-target id.
    sid    = "ManageInferenceComponentAutoscaling"
    effect = "Allow"
    actions = [
      "application-autoscaling:RegisterScalableTarget",
      "application-autoscaling:DeregisterScalableTarget",
      "application-autoscaling:DescribeScalableTargets",
      "application-autoscaling:PutScalingPolicy",
      "application-autoscaling:DeleteScalingPolicy",
      "application-autoscaling:DescribeScalingPolicies",
      "application-autoscaling:DescribeScalingActivities",
      "cloudwatch:PutMetricAlarm",
      "cloudwatch:DeleteAlarms",
      "cloudwatch:DescribeAlarms",
      # Not a SageMaker call this app makes -- it is what Application Auto
      # Scaling will do on the caller's behalf (changing the component's
      # copy count), and RegisterScalableTarget refuses up front unless
      # the caller already holds it: "User is missing the following
      # permissions: sagemaker:UpdateInferenceComponentRuntimeConfig".
      # Grouped here rather than with the other sagemaker actions because
      # autoscaling is the only reason it is needed.
      "sagemaker:UpdateInferenceComponentRuntimeConfig",
    ]
    resources = ["*"]
  }
  statement {
    # The endpoint is created on demand and deleted once every queue has
    # drained, rather than left standing: a real-time endpoint bills its
    # instance for every hour it EXISTS, and the Inference Component's
    # scale-to-zero does NOT stop that (it frees the compute slot inside
    # the instance, not the instance) -- confirmed against the bill, days
    # with zero invocations and scale-in actions still billed 24
    # instance-hours. So the app needs the endpoint lifecycle actions, not
    # just the Inference Component ones above. Resource "*" because
    # CreateEndpointConfig doesn't support resource-level scoping, and the
    # endpoint isn't Terraform-managed so there's no ARN to reference.
    sid    = "ManageMultiSizeEndpointLifecycle"
    effect = "Allow"
    actions = [
      "sagemaker:CreateEndpoint",
      "sagemaker:CreateEndpointConfig",
      "sagemaker:DeleteEndpoint",
      "sagemaker:DescribeEndpoint",
      "sagemaker:DescribeEndpointConfig",
    ]
    resources = ["*"]
  }
  statement {
    sid       = "PassSageMakerExecutionRoleForModelsTab"
    effect    = "Allow"
    actions   = ["iam:PassRole"]
    resources = [aws_iam_role.sagemaker_execution.arn]
  }
  statement {
    # The Models tab's "Start Training" button triggers this CodeBuild
    # project with a TRAIN_SIZE/TRAIN_DATA_S3_URI override, instead of
    # calling StartPipelineExecution directly -- the pipeline definition
    # only picks up a different size/dataset at *build* time
    # (pipeline_definition.py reads those env vars when run_pipeline.py
    # calls build_pipeline()), so the rebuild has to happen first.
    sid       = "TriggerTrainingBuild"
    effect    = "Allow"
    actions   = ["codebuild:StartBuild"]
    resources = [aws_codebuild_project.run_pipeline.arn]
  }
  statement {
    # Lets the Models tab show real status ("Submitting...", "Finalizing...",
    # or a real failure reason) for the gap between "codebuild.start_build
    # succeeded" and "a new Pipeline Execution actually shows up" --
    # see review_app.py's _pending_submission_overlay/_get_build_status.
    # Scoped to every build of this one project (build IDs are only known
    # at runtime, can't be enumerated ahead of time).
    sid       = "CheckTrainingBuildStatus"
    effect    = "Allow"
    actions   = ["codebuild:BatchGetBuilds"]
    resources = ["${replace(aws_codebuild_project.run_pipeline.arn, ":project/", ":build/")}:*"]
  }
  statement {
    sid       = "ManualTrainingTrigger"
    effect    = "Allow"
    actions   = ["sagemaker:StartPipelineExecution", "sagemaker:DescribePipelineExecution"]
    resources = [local.training_pipeline_arn]
  }
  statement {
    # ListPipelineExecutions/ListPipelineExecutionSteps: the Models tab's
    # training-progress panel (3-stage Train/Evaluate/Register bar) --
    # List-type actions, Resource "*" for the same reason as
    # ModelRegistryList above (scoped ARNs don't satisfy the
    # resource-level check for List* calls). DescribeTrainingJob: reads
    # the current run's own "epochs" hyperparameter, to compute the
    # epoch sub-progress denominator.
    # ListTrainingJobs: account-wide (not scoped to this app's own
    # pipeline) -- the live "SageMaker GPU Resource" indicator, which
    # exists specifically because a completely different system in this
    # same account (an experiment harness unrelated to this app) can and
    # does occupy the account's one ml.g4dn.xlarge training slot, causing
    # this app's own StartTrainingJob calls to fail with
    # SAGEMAKER_RESOURCE_LIMIT even though _get_training_progress()
    # (which only looks at THIS pipeline's own executions) shows nothing
    # running. Confirmed live, see summary.md.
    #
    # ListProcessingJobs/DescribeProcessingJob: the CPU half of the same
    # indicator. A run is not one machine -- TrainYolo holds the GPU,
    # then EvaluateYolo runs for seven or eight minutes on an
    # ml.m5.xlarge -- so without these the bar reported "GPU free" while
    # a run was plainly still going, which was asked about three times in
    # one morning before anyone looked at the logs and found the deny.
    # Account-wide for the same List* reason as above.
    sid       = "TrainingProgressPanel"
    effect    = "Allow"
    actions   = ["sagemaker:ListPipelineExecutions", "sagemaker:ListPipelineExecutionSteps", "sagemaker:DescribeTrainingJob", "sagemaker:ListTrainingJobs", "sagemaker:ListProcessingJobs", "sagemaker:DescribeProcessingJob"]
    resources = ["*"]
  }
  statement {
    # CloudWatch metric this same panel polls for live "epoch X/Y"
    # progress -- see pipeline_definition.py's metric_definitions
    # (Estimator scrapes Ultralytics' own per-epoch log line into a
    # "train:epoch" metric under namespace /aws/sagemaker/TrainingJobs).
    sid       = "TrainingProgressMetric"
    effect    = "Allow"
    actions   = ["cloudwatch:GetMetricStatistics"]
    resources = ["*"]
  }
  statement {
    # Model Registry list, for the Review Queue tab's registry panel —
    # missing this crashed every single page load (list_model_packages
    # is called unconditionally in _render_page, regardless of which tab
    # is active), not just the schedule/training actions.
    #
    # ListModelPackages specifically needs Resource "*" — it's a List-type
    # action that queries across a collection rather than acting on one
    # named ARN, so scoping it to a specific resource ARN (as originally
    # tried) doesn't satisfy the resource-level permission check even
    # though the policy looks correct on paper. First attempt at this
    # (scoped to the group/package ARNs) still got AccessDeniedException.
    sid       = "ModelRegistryList"
    effect    = "Allow"
    actions   = ["sagemaker:ListModelPackages"]
    resources = ["*"]
  }
  statement {
    # UpdateModelPackage: lets the review app's own Approve/Reject buttons
    # on the Model Registry list flip ModelApprovalStatus directly (same
    # single API call src/approve_model.py makes) -- so approving a
    # candidate no longer requires running that script from a local
    # checkout. DeleteModelPackage: the review app's own Delete button,
    # for cleaning up a bad/duplicate/test registration -- irreversible,
    # gated behind a confirm() dialog client-side, but the IAM grant itself
    # doesn't distinguish "irreversible" from "reversible", same as a
    # human running `aws sagemaker delete-model-package` directly would
    # have. Scoped to this group's packages only, not "*".
    sid    = "ModelRegistryDescribeApproveAndDelete"
    effect = "Allow"
    actions = [
      "sagemaker:DescribeModelPackage",
      "sagemaker:DescribeModelPackageGroup",
      "sagemaker:UpdateModelPackage",
      "sagemaker:DeleteModelPackage",
    ]
    # One size's group + packages per entry -- covers all five YOLO11
    # sizes (see sagemaker.tf's yolo_size_group_names), not just the
    # original single group.
    resources = flatten([
      for name in values(local.yolo_size_group_names) : [
        "arn:aws:sagemaker:${var.region}:${var.account_id}:model-package-group/${name}",
        "arn:aws:sagemaker:${var.region}:${var.account_id}:model-package/${name}/*",
      ]
    ])
  }
  statement {
    sid       = "EditRetrainSchedule"
    effect    = "Allow"
    actions   = ["events:PutRule", "events:DescribeRule", "events:EnableRule", "events:DisableRule"]
    resources = [aws_cloudwatch_event_rule.weekly_retrain_schedule.arn]
  }
  statement {
    # New Notifications tab: lists, adds, and removes email subscribers
    # on the single SNS topic every human-facing notification already
    # goes through (training pipeline status, drift-triggered review
    # needed, weights approved -- see monitoring.tf's
    # aws_sns_topic.reviewer_notifications and every lambda_src/*.py
    # that publishes to it). Subscribe/ListSubscriptionsByTopic act on
    # the topic itself; Unsubscribe acts on a subscription ARN instead
    # (shape "<topic-arn>:<uuid>", assigned once confirmed) -- both
    # resource shapes included, scoped to this one topic specifically,
    # not "*".
    sid     = "ManageReviewerNotificationSubscriptions"
    effect  = "Allow"
    # Get/SetSubscriptionAttributes: each subscriber's choice of which
    # kinds of email they get is their subscription's FilterPolicy.
    # Publish: the app emails "Errors" subscribers when a retrain chain
    # it ran stops short (src/ppe_pipeline_api/notify.py).
    actions = ["sns:Subscribe", "sns:Unsubscribe", "sns:ListSubscriptionsByTopic",
    "sns:GetSubscriptionAttributes", "sns:SetSubscriptionAttributes", "sns:Publish"]
    resources = [
      aws_sns_topic.reviewer_notifications.arn,
      "${aws_sns_topic.reviewer_notifications.arn}:*",
    ]
  }
  statement {
    # New Inference tab: lets a person actually invoke the shared
    # multi-size endpoint's currently-deployed model for a chosen size,
    # against a real image, instead of only being able to deploy one
    # (the Deploy tab) with no way to try it. The boto3 client for this
    # is "sagemaker-runtime", but the IAM action itself is still under
    # the "sagemaker" prefix, not "sagemaker-runtime" -- confirmed live
    # via a real AccessDeniedException naming sagemaker:InvokeEndpoint
    # as the actual required action. The resource-level check when
    # InferenceComponentName is set (as every call here does) is against
    # the *inference component's own* ARN, not the shared endpoint's --
    # also confirmed live, a second attempt scoped to the endpoint ARN
    # still got denied, naming the inference-component ARN instead. Not
    # Terraform-managed (see deploy_inference_component.py's own
    # comment: created via boto3, no aws_sagemaker_endpoint/
    # aws_sagemaker_inference_component resource exists), so both ARNs
    # are built by hand from the same fixed names every other deploy
    # path already hardcodes; wildcarded across all four sizes rather
    # than listing each track2-yolo-{s,m,l,x} literally.
    sid     = "InvokeMultiSizeEndpoint"
    effect  = "Allow"
    actions = ["sagemaker:InvokeEndpoint"]
    resources = [
      "arn:aws:sagemaker:${var.region}:${var.account_id}:endpoint/track2-yolo-multi-size-endpoint",
      "arn:aws:sagemaker:${var.region}:${var.account_id}:inference-component/track2-yolo-*",
    ]
  }
}

resource "aws_iam_policy" "review_app_access" {
  name   = "${var.project}-review-app-access"
  policy = data.aws_iam_policy_document.review_app_access.json
}

# --- Lambda role: reads Model Monitor's violation report, writes flagged
# items to the review-queue table, and publishes an SNS notification ------

data "aws_iam_policy_document" "lambda_trust" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "review_trigger_lambda" {
  name               = "${var.project}-review-trigger-lambda"
  assume_role_policy = data.aws_iam_policy_document.lambda_trust.json
}

data "aws_iam_policy_document" "review_trigger_lambda_permissions" {
  statement {
    sid       = "Logs"
    effect    = "Allow"
    actions   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["arn:aws:logs:${var.region}:${var.account_id}:*"]
  }
  statement {
    sid       = "ReadMonitorReports"
    effect    = "Allow"
    actions   = ["s3:GetObject", "s3:ListBucket"]
    resources = [aws_s3_bucket.pipeline.arn, "${aws_s3_bucket.pipeline.arn}/*"]
  }
  statement {
    sid       = "WriteReviewQueue"
    effect    = "Allow"
    actions   = ["dynamodb:PutItem"]
    resources = [aws_dynamodb_table.review_queue.arn]
  }
  statement {
    sid       = "NotifyReviewers"
    effect    = "Allow"
    actions   = ["sns:Publish"]
    resources = [aws_sns_topic.reviewer_notifications.arn]
  }
}

resource "aws_iam_policy" "review_trigger_lambda" {
  name   = "${var.project}-review-trigger-lambda"
  policy = data.aws_iam_policy_document.review_trigger_lambda_permissions.json
}

resource "aws_iam_role_policy_attachment" "review_trigger_lambda" {
  role       = aws_iam_role.review_trigger_lambda.name
  policy_arn = aws_iam_policy.review_trigger_lambda.arn
}
