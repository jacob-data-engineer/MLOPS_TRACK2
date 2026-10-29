# --- the central data catalog (2026-10-29) -------------------------------
#
# A small Postgres OWNED BY THIS PLATFORM, recording every file in the
# central data store: where it came from, its labels and their versions,
# every dataset version built from it (with that run's own split and the
# decay/retention decision for each file), and which training run used
# which version. Files are never deleted from the store; a version leaves
# a file out and says why. Training, recreating an old run and exporting a
# zip all read a version's members from here.
#
# Its own database, not a schema in violation-review-postgres: that one
# belongs to the review app, and this platform's user there is read-only
# on purpose.
#
# Private, in the review host's own VPC (the default one), reachable only
# from the review host's security group -- the app connects directly, no
# query Lambda needed. The master password is managed by RDS in Secrets
# Manager and rotated by it; the app reads it on connect.
#
# Applied on its own with `-target`: this stack has unrelated pending
# changes (CodeBuild, training_control, the queue tick) that a full apply
# would push.

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

data "aws_security_group" "review_host" {
  name   = "track2-yolo-review-host"
  vpc_id = data.aws_vpc.default.id
}

resource "aws_security_group" "catalog" {
  name        = "track2-catalog-db"
  description = "Track2 central data catalog: Postgres from the review host only"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description     = "Postgres from the review host"
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [data.aws_security_group.review_host.id]
  }

  tags = { Project = "track2" }
}

resource "aws_db_subnet_group" "catalog" {
  name       = "track2-catalog"
  subnet_ids = data.aws_subnets.default.ids
  tags       = { Project = "track2" }
}

resource "aws_db_instance" "catalog" {
  identifier     = "track2-catalog"
  engine         = "postgres"
  engine_version = "16"
  instance_class = "db.t4g.micro"

  allocated_storage     = 20
  max_allocated_storage = 100
  storage_type          = "gp3"
  storage_encrypted     = true

  db_name                     = "track2_catalog"
  username                    = "catalog_admin"
  manage_master_user_password = true

  db_subnet_group_name   = aws_db_subnet_group.catalog.name
  vpc_security_group_ids = [aws_security_group.catalog.id]
  publicly_accessible    = false

  # The catalog is the only record of which files trained which model.
  # Losing it would not lose a file, but would lose why any of them is
  # anywhere -- so it is backed up and cannot be deleted by accident.
  backup_retention_period    = 7
  deletion_protection        = true
  skip_final_snapshot        = false
  final_snapshot_identifier  = "track2-catalog-final"
  copy_tags_to_snapshot      = true
  auto_minor_version_upgrade = true

  tags = { Project = "track2" }
}

data "aws_iam_policy_document" "catalog_access" {
  statement {
    sid       = "ReadCatalogPassword"
    effect    = "Allow"
    actions   = ["secretsmanager:GetSecretValue"]
    resources = [aws_db_instance.catalog.master_user_secret[0].secret_arn]
  }
  statement {
    # The same Start / Stop the SQL tab offers for the review databases.
    sid       = "StartStopCatalog"
    effect    = "Allow"
    actions   = ["rds:StartDBInstance", "rds:StopDBInstance", "rds:DescribeDBInstances"]
    resources = [aws_db_instance.catalog.arn]
  }
}

resource "aws_iam_role_policy" "catalog_access" {
  name   = "track2-catalog-access"
  role   = data.aws_iam_role.review_host.name
  policy = data.aws_iam_policy_document.catalog_access.json
}

output "catalog_endpoint" {
  value = aws_db_instance.catalog.address
}

output "catalog_secret_arn" {
  value = aws_db_instance.catalog.master_user_secret[0].secret_arn
}
