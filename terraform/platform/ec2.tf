# Dedicated EC2 instance for the review app, replicating
# prototype_validation_server's (i-05d03999bbde05c29, ap-southeast-2)
# settings — same instance type, AMI family, volume, EBS optimization,
# and IMDSv2 enforcement. NOT the same VPC/subnet/AMI ID/security group:
# those are region-specific and this stack lives in ap-southeast-1 (see
# main.tf's note on why), so they're resolved dynamically instead of
# copied literally. Also NOT the same IAM role: the original's EC2_access
# is broad (predates this stack, carries unrelated Bedrock/S3 grants) —
# this gets its own narrowly-scoped role, consistent with every other
# role in this stack.
#
# Cost note: unlike the rest of this stack (S3/Lambda/DynamoDB — mostly
# free-tier or pay-per-use), this is the one continuously-billed
# resource. c5d.large replicates the source instance faithfully but is
# oversized for just running the review-app's tiny Python process —
# override training_... no, override review_host_instance_type (below)
# to something smaller (e.g. t3.micro) if you don't need the parity.
#
# No SSH: replicates the source instance's own "no SSH, SSM only" design
# (see Track1's aws_deployment_setup.md) — no key_name, access via
# `aws ssm start-session --target <instance-id> --region ap-southeast-1`.

variable "review_host_instance_type" {
  description = "Instance type for the review-app host. Defaults to an exact replica of prototype_validation_server (c5d.large, ~$70/mo at 24/7) — override to something smaller (e.g. t3.micro) if you don't need that parity, since this instance only runs a small Python web app, not any heavy compute."
  type        = string
  default     = "c5d.large"
}

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd-gp3/ubuntu-*-26.04-amd64-server-*"]
  }
  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

resource "aws_security_group" "review_host" {
  name        = "${var.project}-review-host"
  description = "Track2 YOLO review-app host - SSM access only, no SSH, review-app port restricted to operator IP"
  vpc_id      = data.aws_vpc.default.id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    # See variables.tf's review_app_public for why this can go fully
    # public and what that actually exposes (no auth of its own on the
    # app side -- this network rule is the only gate).
    description = var.review_app_public ? "review app, OPEN TO THE INTERNET (review_app_public=true, no auth of its own)" : "review app, restricted to operator IPs"
    from_port   = var.review_app_port
    to_port     = var.review_app_port
    protocol    = "tcp"
    cidr_blocks = var.review_app_public ? ["0.0.0.0/0"] : [for ip in var.operator_ips : "${ip}/32"]
  }

  dynamic "ingress" {
    # The JP_UI test container (.github/workflows/deploy-track2-test.yml
    # deploys the track2_ui branch here as a second container, alongside
    # the review app on review_app_port). This rule was originally opened
    # by hand against the live group, so it existed in AWS but not in this
    # config -- which meant the first plan after state was rebuilt wanted
    # to DELETE it and close the port the test UI is served on. Declared
    # here so it survives, and so "what is reachable on this host" is
    # answerable from the config alone.
    for_each = toset(var.notify_extra_ports)
    content {
      description = "JP_UI track2_ui test deploy"
      from_port   = ingress.value
      to_port     = ingress.value
      protocol    = "tcp"
      cidr_blocks = ["0.0.0.0/0"]
    }
  }

  ingress {
    # The refactor slot (2026-10-28): a third app container,
    # track2-app-refactor, serving the data/training split while it is
    # built, with APP_BACKGROUND=off so it only shows pages. Opened by
    # hand against the live group (this stack has drift -- see CLAUDE.md)
    # and declared here so a future apply keeps it. Close it once the
    # split is promoted.
    description = "track2-app-refactor (data/training split, background off)"
    from_port   = 8092
    to_port     = 8092
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

data "aws_iam_policy_document" "ec2_trust" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "review_host" {
  name               = "${var.project}-review-host"
  assume_role_policy = data.aws_iam_policy_document.ec2_trust.json
}

# Same AWS-managed policy the source instance's EC2_access role has —
# required for the SSM Agent to register/communicate (no SSH needed).
resource "aws_iam_role_policy_attachment" "review_host_ssm" {
  role       = aws_iam_role.review_host.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_role_policy_attachment" "review_host_access" {
  role       = aws_iam_role.review_host.name
  policy_arn = aws_iam_policy.review_app_access.arn
}

resource "aws_iam_instance_profile" "review_host" {
  name = "${var.project}-review-host"
  role = aws_iam_role.review_host.name
}

resource "aws_instance" "review_host" {
  ami                         = data.aws_ami.ubuntu.id
  instance_type               = var.review_host_instance_type
  subnet_id                   = data.aws_subnets.default.ids[0]
  vpc_security_group_ids      = [aws_security_group.review_host.id]
  iam_instance_profile        = aws_iam_instance_profile.review_host.name
  ebs_optimized               = true
  associate_public_ip_address = true # relied on downstream (public_ip output, Lambda's REVIEW_APP_URL)

  root_block_device {
    volume_size = 30
    volume_type = "gp3"
    iops        = 3000
  }

  metadata_options {
    http_tokens                 = "required" # IMDSv2 enforced, same as source
    http_put_response_hop_limit = 2
  }

  monitoring = false # matches source (basic monitoring, not detailed)

  # Bootstraps Docker + AWS CLI v2 automatically — the source instance
  # had these installed manually post-launch (see
  # aws_deployment_setup.md's troubleshooting notes); doing it at launch
  # here instead since Terraform is what's creating the instance.
  user_data = <<-EOF
    #!/bin/bash
    apt-get update -y
    apt-get install -y docker.io unzip
    systemctl enable --now docker
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip -q awscliv2.zip
    ./aws/install
  EOF

  tags = {
    Name = "track2-yolo-review-host"
  }

  lifecycle {
    # associate_public_ip_address reads back as false while the instance
    # is stopped (no public IP currently associated, since it's not
    # running) even though it's true in config and gets a fresh public IP
    # on every start regardless. Without this, any `plan` run while the
    # instance happens to be stopped falsely shows a forced replacement.
    #
    # ami: data.aws_ami.ubuntu (most_recent = true) resolves to whatever
    # the newest Ubuntu AMI is at plan time -- this drifts on its own as
    # AWS publishes new AMIs, with no config change on our end. ami is a
    # ForceNew attribute, so without ignoring it here, any plan/apply that
    # happens to run after a new AMI is published destroys and recreates
    # this live instance as a side effect -- confirmed as a near-miss once
    # already (a targeted apply for an unrelated IAM policy pulled this in
    # via the dependency graph). The instance was provisioned once from
    # whatever AMI was current then; it doesn't need to track new AMI
    # releases automatically.
    ignore_changes = [associate_public_ip_address, ami]
  }
}
