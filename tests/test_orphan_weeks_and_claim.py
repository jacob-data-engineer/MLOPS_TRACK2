"""Two quiet failures: data nobody trains on, and a race nobody sees.

`orphan_week_folders` -- the data.yaml is built from the week DEFINITIONS
in DynamoDB, so a week FOLDER in S3 with no matching definition holds
images that were pulled, are paying for storage, and will never be
learned from. Every component is behaving correctly off a list the folder
is not on, so nothing reports it. Seen live: retraining/week_4 held a
filed image while the definitions listed weeks 1, 2, 3, 6 and 7.

`claim_once` -- dev and stable share this table and run the same poll
tick. "Has this pull been handled?" answered by read-then-write has a
window in which both read "no", and what follows is a GPU training run.
"""
from __future__ import annotations

import importlib

import pytest
from botocore.exceptions import ClientError

pc = importlib.import_module("src.ppe_pipeline_api.pull_control")


# ----------------------------------------------------------- orphan weeks
CONFIG = {"root": "retraining", "week_prefix": "week_",
          "destination_bucket": "track2-s3"}


def _objects(keys):
    return [{"Key": k, "Size": 1, "ETag": '"x"'} for k in keys]


@pytest.fixture
def listed(monkeypatch):
    def use(keys, weeks):
        monkeypatch.setattr(pc, "_list_objects", lambda prefix, bucket="", **kw: iter(_objects(keys)))
        monkeypatch.setattr(pc, "list_weeks", lambda: [{"name": w, "starts_at": ""} for w in weeks])
        monkeypatch.setattr(pc, "destination_bucket", lambda config=None: "track2-s3")
    return use


def test_a_week_folder_with_no_definition_is_reported(listed):
    listed([
        "retraining/week_1/images/train/a.jpg",
        "retraining/week_4/images/train/b.jpg",
    ], ["week_1"])
    got = pc.orphan_week_folders(CONFIG)
    assert [o["name"] for o in got] == ["week_4"]
    assert got[0]["images"] == 1


def test_a_defined_week_is_not_an_orphan(listed):
    listed(["retraining/week_1/images/train/a.jpg"], ["week_1"])
    assert pc.orphan_week_folders(CONFIG) == []


def test_labels_do_not_count_as_images(listed):
    """Counting a .txt would report an orphan for a folder holding only
    labels, which is a different (and real) problem with a different fix."""
    listed(["retraining/week_4/labels/train/b.txt"], ["week_1"])
    assert pc.orphan_week_folders(CONFIG) == []


def test_non_week_folders_are_ignored(listed):
    """`data.yaml`, `_snapshots`, anything that is not a week."""
    listed(["retraining/data.yaml", "retraining/_runs/x.json"], ["week_1"])
    assert pc.orphan_week_folders(CONFIG) == []


def test_several_orphans_come_back_newest_looking_first(listed):
    listed([
        "retraining/week_4/images/train/a.jpg",
        "retraining/week_9/images/val/b.jpg",
        "retraining/week_9/images/val/c.jpg",
    ], [])
    got = pc.orphan_week_folders(CONFIG)
    assert [o["name"] for o in got] == ["week_9", "week_4"]
    assert got[0]["images"] == 2


def test_no_root_means_nothing_to_check(listed):
    listed(["retraining/week_4/images/train/a.jpg"], [])
    assert pc.orphan_week_folders({"root": ""}) == []


def test_preflight_warns_rather_than_blocks(monkeypatch):
    """A warning: the run is still correct, it is just training on less
    than was pulled. Blocking would stop a chain over a stray folder."""
    rs = importlib.import_module("src.ppe_pipeline_api.retrain_schedule")
    monkeypatch.setattr(rs.pc, "orphan_week_folders",
                        lambda config=None: [{"name": "week_4", "images": 7}])
    monkeypatch.setattr(rs.pc, "read_config", lambda: {
        "root": "retraining", "mode": "schedule",
        "picked": {"train_images": ["x"]},
        "retrain_schedule": {"detector_slug": "helmet", "follow_pull": True},
    })
    problems = rs.preflight()["problems"]
    orphan = [p for p in problems if "week_4" in p["text"]]
    assert len(orphan) == 1
    assert orphan[0]["level"] == "warn"
    assert "never trained on" in orphan[0]["text"]


# ------------------------------------------------------------------ claim
def test_the_first_caller_wins(monkeypatch):
    monkeypatch.setattr(pc.ddb, "put_item", lambda **kw: {})
    assert pc.claim_once("queue_after_pull", "run-1") is True


def test_the_second_caller_loses(monkeypatch):
    def already(**kw):
        raise ClientError({"Error": {"Code": "ConditionalCheckFailedException"}}, "PutItem")
    monkeypatch.setattr(pc.ddb, "put_item", already)
    assert pc.claim_once("queue_after_pull", "run-1") is False


def test_a_broken_table_does_NOT_grant_the_claim(monkeypatch):
    """Returning True on error looks like the safe default and is the
    wrong one: it turns an unavailable table into two training runs."""
    def broken(**kw):
        raise ClientError({"Error": {"Code": "ProvisionedThroughputExceededException"}}, "PutItem")
    monkeypatch.setattr(pc.ddb, "put_item", broken)
    assert pc.claim_once("queue_after_pull", "run-1") is False


def test_the_claim_is_conditional_not_a_plain_put(monkeypatch):
    seen = {}
    monkeypatch.setattr(pc.ddb, "put_item", lambda **kw: seen.update(kw) or {})
    pc.claim_once("queue_after_pull", "run-1")
    assert "attribute_not_exists" in seen["ConditionExpression"]
    # And it expires -- a race guard, not a record.
    assert "expires_at" in seen["Item"]


def test_queue_after_pull_claims_before_it_chains():
    import inspect
    qap = importlib.import_module("src.ppe_pipeline_api.queue_after_pull")
    source = inspect.getsource(qap.check)
    before = source.split("rs.run_chain")[0]
    assert "claim_once" in before, "the claim must come BEFORE the GPU work"
