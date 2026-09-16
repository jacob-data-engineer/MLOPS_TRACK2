# Replaces Amazon A2I's human-loop tracking, which this account can't
# use (see sagemaker.tf's note — both A2I and SageMaker private-workforce
# management are closed to new customers). One item per flagged
# prediction; review_app.py (deployed to the EC2 instance, same pattern
# as the Grafana demo) reads/writes this directly.

resource "aws_dynamodb_table" "review_queue" {
  name         = "${var.project}-review-queue"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "review_id"

  attribute {
    name = "review_id"
    type = "S"
  }
}

# Created manually outside Terraform (2026-09-01), brought under management
# via `terraform import` -- name/schema below mirrors its real live
# configuration exactly (PAY_PER_REQUEST, hash key "id"/String, no
# GSIs/streams/TTL). Purpose not yet documented by whoever created it.
resource "aws_dynamodb_table" "track2_dynamodb" {
  name         = "track2-dynamoDB"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "id"

  attribute {
    name = "id"
    type = "S"
  }
}

# Inference-run lineage: what produced each set of results -- the source
# folder and exact files read, the model size/version behind them, and the
# inference window.
#
# Deliberately NOT the track2-dynamoDB table above. That one holds
# training-run and baseline records, and scan_all_lineage() returns it
# wholesale to the Lineage tab with no record_type filter, so inference
# rows written there would surface as malformed training rows. Different
# entity, different shape, own table.
#
# Keyed by the queue item's own id: a run is the thing being described,
# and two runs can legitimately write to the same output folder, which
# keying on the folder would silently collapse into one record.
resource "aws_dynamodb_table" "inference_lineage" {
  name         = "${var.project}-inference-lineage"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "id"

  attribute {
    name = "id"
    type = "S"
  }
}
