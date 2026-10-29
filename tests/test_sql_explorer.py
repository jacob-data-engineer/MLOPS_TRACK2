"""The SQL tab (2026-10-28): read-only Postgres queries, answered as images."""

import importlib.util
import io
import json
import pathlib

import pytest

from ppe_pipeline_api.data import sql_explorer as sx

ROOT = pathlib.Path(__file__).resolve().parents[1]


# --- which values are images ---------------------------------------------------

@pytest.mark.parametrize("value, bucket, key", [
    ("s3://track2-s3/data/week_1/images/a.jpg", "track2-s3", "data/week_1/images/a.jpg"),
    ("https://track2-s3.s3.ap-southeast-1.amazonaws.com/x/b%20c.png", "track2-s3", "x/b c.png"),
    ("https://s3.ap-southeast-1.amazonaws.com/other/y/c.JPEG?X-Amz-Signature=1", "other", "y/c.JPEG"),
    ("site/cam1/frame_001.jpg", "", "site/cam1/frame_001.jpg"),
])
def test_an_image_is_recognised_from_its_value(value, bucket, key):
    assert sx.as_image(value) == {"bucket": bucket, "key": key}


@pytest.mark.parametrize("value", [None, 12, "helmet", "s3://b/labels/a.txt",
                                   "https://example.com/a.jpg", "not a path.jpg"])
def test_other_values_are_not(value):
    assert sx.as_image(value) is None


def test_the_bucket_and_folder_narrow_the_list():
    cols = ["id", "path", "thumb"]
    rows = [
        [1, "s3://track2-s3/data/week_1/images/a.jpg", "s3://track2-s3/thumbs/a.jpg"],
        [2, "s3://other/data/week_1/images/b.jpg", None],
        [3, "data/week_1/images/c.jpg", None],   # bare: read as a key in the chosen bucket
        [4, "s3://track2-s3/data/week_1/images/a.jpg", None],  # duplicate
    ]
    every = sx.images_from(cols, rows)
    assert [i["uri"] for i in every] == [
        "s3://track2-s3/data/week_1/images/a.jpg", "s3://track2-s3/thumbs/a.jpg",
        "s3://other/data/week_1/images/b.jpg", "data/week_1/images/c.jpg"]
    narrowed = sx.images_from(cols, rows, bucket="track2-s3", folder="data/week_1/")
    assert [i["uri"] for i in narrowed] == [
        "s3://track2-s3/data/week_1/images/a.jpg", "s3://track2-s3/data/week_1/images/c.jpg"]
    assert narrowed[0]["column"] == "path" and narrowed[1]["row"] == 2


# --- talking to the query function ---------------------------------------------

class _Lambda:
    def __init__(self, body, error=None):
        self.body, self.error, self.calls = body, error, []

    def invoke(self, **kw):
        self.calls.append(kw)
        out = {"Payload": io.BytesIO(json.dumps(self.body).encode())}
        if self.error:
            out["FunctionError"] = self.error
        return out


def test_a_query_comes_back_as_rows_and_images(monkeypatch):
    fake = _Lambda({"ok": True, "columns": ["key"], "rows": [["s3://b/f/a.jpg"]], "truncated": False})
    monkeypatch.setattr(sx, "_client", lambda kind: fake)
    out = sx.run("violation-review", "select key from t", bucket="b", folder="f")
    assert out["images"][0]["uri"] == "s3://b/f/a.jpg"
    sent = json.loads(fake.calls[0]["Payload"])
    assert sent == {"action": "query", "sql": "select key from t"}
    assert fake.calls[0]["FunctionName"] == "track2-pg-query-violation-review"


def test_the_databases_own_words_reach_the_screen(monkeypatch):
    fake = _Lambda({"ok": False, "error": 'DatabaseError: column "nope" does not exist'})
    monkeypatch.setattr(sx, "_client", lambda kind: fake)
    with pytest.raises(ValueError, match='column "nope" does not exist'):
        sx.run("violation-review", "select nope from t")


def test_a_stopped_database_says_so_rather_than_timing_out_silently(monkeypatch):
    fake = _Lambda({"errorMessage": "Task timed out after 60.00 seconds"}, error="Unhandled")
    monkeypatch.setattr(sx, "_client", lambda kind: fake)
    with pytest.raises(ValueError, match="may be stopped"):
        sx.run("jpui-preview", "select 1")


def test_an_unknown_database_is_refused():
    with pytest.raises(ValueError):
        sx.run("somewhere-else", "select 1")


# --- the function itself cannot write -------------------------------------------

def _function():
    spec = importlib.util.spec_from_file_location(
        "pg_query", ROOT / "terraform" / "app" / "lambda_src" / "pg_query.py")
    source = spec.loader.get_source("pg_query")
    return source


def test_it_runs_read_only_with_a_time_limit():
    code = _function()
    assert 'cur.execute("BEGIN READ ONLY")' in code
    assert "SET LOCAL statement_timeout" in code
    assert 'cur.execute("ROLLBACK")' in code
    assert "default_transaction_read_only = on" in code


def test_bootstrap_is_off_unless_switched_on():
    code = _function()
    assert 'if os.environ.get("ALLOW_BOOTSTRAP") != "1":' in code


def test_one_statement_at_a_time():
    """Loaded without pg8000: the guard is plain Python."""
    code = _function()
    start = code.index("def _one_statement")
    end = code.index("\ndef ", start + 1)
    namespace: dict = {}
    exec(code[start:end], namespace)  # noqa: S102 -- our own source, one pure function
    one = namespace["_one_statement"]
    assert one("select 1;") == "select 1"
    assert one("select ';' as x") == "select ';' as x"
    for bad in ("select 1; delete from t", "SET transaction_read_only = off; DELETE FROM t", ";"):
        with pytest.raises(ValueError):
            one(bad)


# --- the tab --------------------------------------------------------------------

def test_the_tab_is_routed_offered_and_saves_its_settings():
    web = ROOT / "web" / "src"
    app = (web / "App.tsx").read_text(encoding="utf-8")
    nav = (web / "components" / "Layout.tsx").read_text(encoding="utf-8")
    page = (web / "routes" / "Track2SqlPage.tsx").read_text(encoding="utf-8")
    assert 'path="/track2/sql" element={<Track2SqlPage />}' in app
    assert '"Track2 SQL Database Query"' in nav
    assert "void saveTrack2PullConfig({ sql_explorer: {" in page
    assert "pull_bucket: pullBucket, pull_folder: pullFolder, train_percent: trainPercent," in page
    from ppe_pipeline_api import pull_control as pc
    from ppe_pipeline_api.routers.track2_pull_control import ConfigRequest
    assert "sql_explorer" in pc.DEFAULTS and "sql_explorer" in ConfigRequest.model_fields


def test_the_routes_exist():
    from ppe_pipeline_api.main import app
    paths = app.openapi()["paths"]
    for path in ("/track2/sql/databases", "/track2/sql/schema", "/track2/sql/query",
                 "/track2/sql/pull"):
        assert path in paths


def test_a_refused_status_says_setup_has_not_run(monkeypatch):
    """Before setup_sql_tab.sh the app may not describe the databases;
    that showed as "unknown (ClientError)", which says nothing useful."""
    class Denied:
        def describe_db_instances(self, **kw):
            raise Exception("An error occurred (AccessDenied) when calling DescribeDBInstances: "
                            "User is not authorized to perform: rds:DescribeDBInstances")
    monkeypatch.setattr(sx, "_client", lambda kind: Denied())
    for db in sx.databases():
        assert db["status"].startswith("not set up") and not db["available"]


# --- pulling the images into a training folder ----------------------------------

class _S3:
    def __init__(self, objects):
        self.objects = dict(objects)          # (bucket, key) -> bytes
        self.copies = []

    def head_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise Exception("404")
        return {}

    def copy_object(self, Bucket, Key, CopySource):
        self.objects[(Bucket, Key)] = self.objects[(CopySource["Bucket"], CopySource["Key"])]
        self.copies.append(Key)

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise Exception("404")
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    def put_object(self, Bucket, Key, Body, ContentType=""):
        self.objects[(Bucket, Key)] = Body


def _pull_setup(monkeypatch, rows, objects):
    fake = _Lambda({"ok": True, "columns": ["image"], "rows": rows, "truncated": False})
    monkeypatch.setattr(sx, "_client", lambda kind: fake)
    s3 = _S3(objects)
    monkeypatch.setattr(sx.storage, "s3", s3)
    return s3


def test_a_pull_copies_images_and_their_labels_into_the_training_layout(monkeypatch):
    s3 = _pull_setup(monkeypatch, [["s3://src/set/images/train/a.jpg"], ["s3://src/frames/b.jpg"]], {
        ("src", "set/images/train/a.jpg"): b"A", ("src", "set/labels/train/a.txt"): b"0 .5 .5 .1 .1",
        ("src", "frames/b.jpg"): b"B",
    })
    out = sx.pull("violation-review", "select 1", dest_bucket="track2-s3",
                  dest_folder="sql_pulls/helmet/", train_percent=100)
    assert out["copied"] == 2 and out["with_label"] == 1 and out["train"] == 2
    assert s3.objects[("track2-s3", "sql_pulls/helmet/images/train/a.jpg")] == b"A"
    assert s3.objects[("track2-s3", "sql_pulls/helmet/labels/train/a.txt")] == b"0 .5 .5 .1 .1"
    assert ("track2-s3", "sql_pulls/helmet/images/train/b.jpg") in s3.objects
    sources = json.loads(s3.objects[("track2-s3", "sql_pulls/helmet/sources.json")])
    assert sources["sql_pulls/helmet/images/train/a.jpg"] == "s3://src/set/images/train/a.jpg"
    record = json.loads(s3.objects[("track2-s3", "sql_pulls/helmet/query.json")])
    assert record["sql"] == "select 1" and record["database"] == "violation-review"


def test_a_repeated_pull_skips_what_is_already_there(monkeypatch):
    s3 = _pull_setup(monkeypatch, [["s3://src/f/a.jpg"]], {("src", "f/a.jpg"): b"A"})
    sx.pull("violation-review", "q", dest_bucket="d", dest_folder="p")
    again = sx.pull("violation-review", "q", dest_bucket="d", dest_folder="p")
    assert again["copied"] == 0 and again["skipped"] == 1
    assert s3.copies == ["p/images/train/a.jpg"]


def test_two_sources_with_one_name_do_not_overwrite_each_other(monkeypatch):
    s3 = _pull_setup(monkeypatch, [["s3://src/x/a.jpg"], ["s3://src/y/a.jpg"]],
                     {("src", "x/a.jpg"): b"X", ("src", "y/a.jpg"): b"Y"})
    out = sx.pull("violation-review", "q", dest_bucket="d", dest_folder="p")
    assert out["copied"] == 2
    copied = sorted(k for (b, k) in s3.objects if b == "d" and "/images/" in k)
    assert len(copied) == 2 and copied[0] != copied[1]


def test_the_split_is_fixed_per_image():
    uris = [f"s3://b/i/{n}.jpg" for n in range(400)]
    sides = [sx.split_for(u, 80) for u in uris]
    assert sides == [sx.split_for(u, 80) for u in uris]
    assert 0.7 < sides.count("train") / len(sides) < 0.9
    assert sx.split_for(uris[0], 100) == "train" and sx.split_for(uris[0], 0) == "val"


def test_the_label_is_looked_for_where_ultralytics_would():
    assert sx.label_key_for("set/images/train/a.jpg") == "set/labels/train/a.txt"
    assert sx.label_key_for("images/a.png") == "labels/a.txt"
    assert sx.label_key_for("frames/a.jpg") == ""


@pytest.mark.parametrize("folder", ["", "custom", "custom/code", "track2-yolo-weights/x"])
def test_a_pull_needs_a_folder_and_never_writes_into_code_or_weights(folder):
    with pytest.raises(ValueError):
        sx.pull("violation-review", "q", dest_bucket="track2-s3", dest_folder=folder)
