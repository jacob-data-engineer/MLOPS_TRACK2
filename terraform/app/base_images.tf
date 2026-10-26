# Base images, mirrored into this account's own registry.
#
# The Dockerfile used to pull `node:20-slim` from Docker Hub and
# `ghcr.io/astral-sh/uv` from GitHub. Both are outside AWS, and fetching
# them is the single thing that decides how long a deploy takes:
#
#   17 Sep   12 builds   3.9 - 6.9 min, almost all 4.0
#   24 Sep   same code   4.0 - 18.3 min, bimodal
#
# The fast builds found the layers already on the build host; the slow
# ones fetched them over the public internet, once at 20 KB/s. Nothing
# about the app changed between those two days.
#
# Mirroring them here removes the variable. Every build then pulls from
# ECR in the same region, which is also where the app image already goes,
# so a deploy stops depending on Docker Hub being reachable or willing.
#
# Refreshing is deliberate, not automatic: run the mirror build again when
# a newer base image is actually wanted. Pinning the base of a production
# image is the behaviour you want anyway -- a silent upgrade underneath a
# build is how a deploy breaks for reasons nobody changed.
resource "aws_ecr_repository" "base" {
  name                 = "${var.app_name}-base"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = false
  }
}

# Base images are few and large; keeping every historical pull would cost
# more than it could ever be worth. Ten is enough to roll back.
resource "aws_ecr_lifecycle_policy" "base" {
  repository = aws_ecr_repository.base.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep the ten most recent base images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 10
      }
      action = { type = "expire" }
    }]
  })
}

output "base_image_repository" {
  value = aws_ecr_repository.base.repository_url
}
