"""Deleting the week folders and everything in them.

There is no second copy. The pull MOVES files out of the drop folders
rather than copying them -- `undo_run` exists because a move cannot be
undone from the destination -- so every object this removes is the only
one. These tests exist because the two ways it could go quietly wrong
both look like success: deleting the first 1000 of 1496 and reporting
"done", or deleting more than the person was shown.
"""

from __future__ import annotations

import pytest

from ppe_pipeline_api import pull_control as pc


CONFIG = {
    "root": "retraining",
    "sources": [
        {"destination": "retraining", "split": "train"},
        {"destination": "retraining", "eval_destination": "retraining_eval",
         "split": "val"},
    ],
}


class FakeS3:
    """Pages at 1000 keys, like the real one."""

    def __init__(self, keys):
        self.keys = list(keys)
        self.deleted: list = []

    def list_objects_v2(self, **kw):
        prefix = kw["Prefix"]
        matching = [k for k in self.keys if k.startswith(prefix)]

        # Delimiter mode: the folder listing, as CommonPrefixes. This is
        # how the week folders are discovered -- from the bucket, not
        # from the week rows.
        if kw.get("Delimiter") == "/":
            commons = sorted({
                prefix + rest.split("/", 1)[0] + "/"
                for rest in (k[len(prefix):] for k in matching)
                if "/" in rest
            })
            return {"CommonPrefixes": [{"Prefix": c} for c in commons],
                    "Contents": [], "IsTruncated": False}

        token = int(kw.get("ContinuationToken") or 0)
        page = matching[token:token + 1000]
        more = len(matching) > token + 1000
        return {
            "Contents": [{"Key": k, "Size": 10} for k in page],
            "IsTruncated": more,
            "NextContinuationToken": str(token + 1000) if more else None,
        }

    def delete_objects(self, **kw):
        batch = [o["Key"] for o in kw["Delete"]["Objects"]]
        assert len(batch) <= 1000, "delete_objects takes at most 1000 keys"
        self.deleted += batch
        self.keys = [k for k in self.keys if k not in set(batch)]
        return {}


@pytest.fixture
def s3(monkeypatch):
    keys = (
        [f"retraining/week_1/images/train/img_{i:05d}.jpg" for i in range(1200)]
        + [f"retraining/week_1/labels/train/img_{i:05d}.txt" for i in range(1200)]
        + ["retraining_eval/week_1/images/val/a.jpg"]
        # NOT under a week: must survive.
        + ["retraining/data.yaml", "retraining/train.txt",
           "retraining/_dataset_snapshots/abc/manifest.json"]
    )
    fake = FakeS3(keys)
    monkeypatch.setattr(pc, "s3", fake)
    monkeypatch.setattr(pc, "destination_bucket", lambda config=None: "track2-s3")
    monkeypatch.setattr(pc, "read_config", lambda: CONFIG)
    # Deliberately EMPTY: the folders must be found without any week
    # rows at all. That is the exact state the bulk delete leaves behind,
    # and the state in which the folders used to become undeletable.
    monkeypatch.setattr(pc, "list_weeks", lambda: [])
    return fake


def test_a_week_is_every_root_it_lives_in(s3):
    """A source that files `val` into its own eval root gives the same
    week a second home. Deleting only the training one leaves half of it
    behind, and the half left behind still trains."""
    prefixes = pc.week_folder_prefixes(CONFIG, ["week_1"])
    assert "retraining/week_1/" in prefixes
    assert "retraining_eval/week_1/" in prefixes


def test_the_count_is_the_warning(s3):
    """"Delete the week folders" and "delete 2,401 files" are not the
    same sentence. The count is shown before anything is pressed."""
    out = pc.count_week_objects(CONFIG, ["week_1"])
    assert out["objects"] == 2401          # 1200 + 1200 + 1
    assert out["bytes"] == 24010


def test_it_deletes_past_the_first_page(s3):
    """`list_objects_v2` caps at 1000 and `delete_objects` takes 1000 at
    a time. An unpaginated pass would delete the first thousand, report
    success, and leave the rest -- which reads as "it worked" and is the
    worst of the three outcomes."""
    out = pc.purge_week_folders(CONFIG, ["week_1"])
    assert out["deleted"] == 2401
    assert not [k for k in s3.keys if "/week_1/" in k]


def test_it_touches_nothing_outside_the_week_folders(s3):
    """data.yaml, the retention lists and the snapshots live at the root
    beside the weeks. Losing those loses the ability to reproduce every
    run ever made from this corpus."""
    pc.purge_week_folders(CONFIG, ["week_1"])
    assert "retraining/data.yaml" in s3.keys
    assert "retraining/train.txt" in s3.keys
    assert "retraining/_dataset_snapshots/abc/manifest.json" in s3.keys


def test_purge_week_folders_itself_touches_no_rows(s3, monkeypatch):
    """The S3 half is its own function. The ROUTE forgets the weeks
    afterwards -- deleting the files alone left every row on screen
    reading "0 img -- empty", which looks like the delete failed."""
    called: list = []
    monkeypatch.setattr(pc, "delete_week", lambda **kw: called.append(kw))
    pc.purge_week_folders(CONFIG, ["week_1"])
    assert called == []


def test_the_bulk_route_does_not_squat_on_the_single_week_path():
    """`POST /weeks/purge` already belongs to the single-week delete,
    which sends {name} and no confirm_objects. Two handlers on one path
    means FastAPI takes the first, and the row button starts 422ing --
    a working control broken by an addition that never touched it."""
    from ppe_pipeline_api.routers import track2_pull_control as r

    paths = [(route.path, tuple(sorted(route.methods)))
             for route in r.router.routes]
    assert paths.count(("/weeks/purge", ("POST",))) == 1
    assert ("/weeks/purge-all", ("POST",)) in paths


def test_the_folders_are_found_with_no_week_rows_at_all(s3):
    """The bug this came from. `purge_week` and the bulk delete both
    FORGET the week rows, and the folders outlive them in the bucket.
    Deriving the list from the rows meant that once they were gone the
    folders were invisible here and could never be deleted -- while the
    card went on showing them, because the card reads the bucket.

    Two sources of truth for one question, and the delete had the wrong
    one."""
    assert pc.list_weeks() == []
    prefixes = pc.week_folder_prefixes(CONFIG)
    assert "retraining/week_1/" in prefixes
    assert pc.count_week_objects(CONFIG)["objects"] == 2401


def test_only_folders_matching_the_week_prefix(s3):
    """`_dataset_snapshots/` and anything else living beside the weeks
    must not be swept up by a delete aimed at week folders."""
    s3.keys.append("retraining/_dataset_snapshots/abc/data.yaml")
    s3.keys.append("retraining/scratch_notes/readme.txt")
    prefixes = pc.week_folder_prefixes(CONFIG)
    assert all("/week_" in p for p in prefixes), prefixes


# --- the single-week purge had no confirmation at all -------------------------
#
# Found by the destructive-path UAT: `POST /weeks/purge` with a name and a
# fabricated confirmation answered 200. It was not ignoring a bad number --
# it had no number to ignore. `purge-all` has always required the exact
# object count and re-counted it server-side; this route, the one actually
# reachable from a row in the table and so the one more likely to be
# pressed, took a name and deleted the week on one request.

def test_purging_one_week_now_demands_the_count():
    import pydantic

    from ppe_pipeline_api.routers.track2_pull_control import PurgeWeekRequest

    with pytest.raises(pydantic.ValidationError):
        PurgeWeekRequest(name="week_1")
    assert PurgeWeekRequest(name="week_1", confirm_objects=7).confirm_objects == 7


@pytest.fixture
def api(monkeypatch):
    """An authenticated client. This file had none -- its other tests work
    on `pull_control` directly -- and the guard being added lives in the
    ROUTE, so it has to be exercised through one."""
    from fastapi.testclient import TestClient

    from ppe_pipeline_api.main import app

    monkeypatch.setenv("PPE_APP_PASSWORD", "uat-week-purge")
    c = TestClient(app)
    token = c.post("/auth/login", json={"password": "uat-week-purge"}).json()["token"]
    c.headers.update({"Authorization": f"Bearer {token}"})
    return c


def test_a_count_that_no_longer_matches_is_refused(api, monkeypatch):
    """Re-counted server-side, not trusted. Between looking and pressing a
    sweep may have landed files, and deleting more than was shown is the
    surprise the confirmation exists to prevent."""
    from ppe_pipeline_api.routers import track2_pull_control as router

    monkeypatch.setattr(router.pc, "count_week_objects",
                        lambda names=None: {"objects": 42, "bytes": 1, "prefixes": {}})
    deleted = []
    monkeypatch.setattr(router.pc, "purge_week",
                        lambda name: deleted.append(name) or {"week": name, "deleted": 0, "roots": []})

    res = api.post("/track2/pull-control/weeks/purge",
                      json={"name": "week_1", "confirm_objects": 7})
    assert res.status_code == 409
    assert "42" in res.json()["detail"]
    assert deleted == [], "nothing may be deleted when the count disagrees"


def test_the_matching_count_goes_through(api, monkeypatch):
    from ppe_pipeline_api.routers import track2_pull_control as router

    monkeypatch.setattr(router.pc, "count_week_objects",
                        lambda names=None: {"objects": 42, "bytes": 1, "prefixes": {}})
    monkeypatch.setattr(router.pc, "purge_week",
                        lambda name: {"week": name, "deleted": 42, "roots": []})

    res = api.post("/track2/pull-control/weeks/purge",
                      json={"name": "week_1", "confirm_objects": 42})
    assert res.status_code == 200
    assert res.json()["deleted"] == 42
