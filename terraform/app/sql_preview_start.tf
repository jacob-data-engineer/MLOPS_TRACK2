# Keep the preview database's Start button permission with the app deployment.
# Separate from the one-time SQL setup policy: no database setup or stop action
# is needed to grant this permission.
data "aws_iam_policy_document" "sql_preview_start" {
  statement {
    sid       = "StartSqlPreviewDatabase"
    effect    = "Allow"
    actions   = ["rds:StartDBInstance"]
    resources = ["arn:aws:rds:${var.region}:${local.account_id}:db:jpui-preview-crop-authority"]
  }
}

resource "aws_iam_role_policy" "sql_preview_start" {
  name   = "track2-sql-preview-start"
  role   = data.aws_iam_role.review_host.name
  policy = data.aws_iam_policy_document.sql_preview_start.json
}
