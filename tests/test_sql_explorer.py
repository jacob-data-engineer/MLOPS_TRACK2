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
                 "/track2/sql/classes", "/track2/sql/preset", "/track2/sql/previews",
                 "/track2/sql/databases/{key}/start", "/track2/sql/databases/{key}/stop",
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


# --- human-verified frames, by class --------------------------------------------

def test_the_classes_come_from_the_database(monkeypatch):
    fake = _Lambda({"ok": True, "columns": ["class", "frames"],
                    "rows": [["no_helmet", 5633], ["worker_on_edge", 1708]], "truncated": False})
    monkeypatch.setattr(sx, "_client", lambda kind: fake)
    got = sx.classes("violation-review")
    assert got == [{"name": "no_helmet", "frames": 5633, "has_boxes": True},
                   {"name": "worker_on_edge", "frames": 1708, "has_boxes": False}]


def test_the_preset_keeps_only_human_answers():
    sql = sx.preset_sql("no_helmet", "resolved")
    assert "data->>'hazard_type' = 'no_helmet'" in sql
    assert "<> 'similarity_prefill'" in sql and "<> 'true'" in sql
    assert "j.answered = j.kept_people" in sql
    # a frame "no violation" cannot cover a person somebody called a violation
    assert "fv.verdict = 'no_violation' AND j.violations = 0" in sql
    assert "AS image" in sql


def test_ready_and_test_narrow_the_helmet_class_only():
    assert "WHERE helmet_box <> 'missing'" in sx.preset_sql("no_helmet", "ready")
    assert "WHERE helmet_box = 'human (test-grade)'" in sx.preset_sql("no_helmet", "test")
    with pytest.raises(ValueError):
        sx.preset_sql("worker_on_edge", "ready")


@pytest.mark.parametrize("hazard", ["", "x'; drop table cases; --", "No Helmet"])
def test_a_class_name_is_never_written_into_sql_unchecked(hazard):
    with pytest.raises(ValueError):
        sx.preset_sql(hazard, "resolved")


def test_an_image_column_counts_without_an_extension():
    key = "eval/suspended_cargo/evidence/abc123/source"
    found = sx.images_from(["image", "note"], [[f"s3://b/{key}", f"s3://b/{key}"]])
    assert [i["column"] for i in found] == ["image"]


def test_an_extensionless_image_is_named_after_its_folder(monkeypatch):
    key = "eval/suspended_cargo/evidence/abc123/source"
    s3 = _pull_setup(monkeypatch, [[f"s3://src/{key}"]], {("src", key): b"\xff\xd8"})
    sx.pull("violation-review", "q", dest_bucket="d", dest_folder="p")
    assert ("d", "p/images/train/abc123.jpg") in s3.objects


# --- point and select, with a preview ------------------------------------------

def test_previews_sign_s3_images_only_and_serve_extensionless_as_jpeg(monkeypatch):
    signed = []

    class Signer:
        def generate_presigned_url(self, op, Params, ExpiresIn):
            signed.append((op, Params, ExpiresIn))
            return "https://signed/" + Params["Key"]

    monkeypatch.setattr(sx.storage, "s3", Signer())
    got = sx.previews(["s3://b/a.jpg", "frames/b.jpg", "s3://b/ev/1/source"])
    assert [g["uri"] for g in got] == ["s3://b/a.jpg", "s3://b/ev/1/source"]
    assert all(op == "get_object" and secs <= 900 for op, _, secs in signed)
    assert "ResponseContentType" not in signed[0][1]
    assert signed[1][1]["ResponseContentType"] == "image/jpeg"


def test_previews_are_capped_at_a_screenful(monkeypatch):
    class Signer:
        def generate_presigned_url(self, op, Params, ExpiresIn):
            return "u"
    monkeypatch.setattr(sx.storage, "s3", Signer())
    assert len(sx.previews([f"s3://b/{n}.jpg" for n in range(500)])) == sx.PREVIEW_LIMIT


def test_the_page_runs_on_select_and_shows_thumbnails():
    page = (ROOT / "web" / "src" / "routes" / "Track2SqlPage.tsx").read_text(encoding="utf-8")
    # no button between choosing a class and seeing it
    assert "void showClass(pickedClass.name, scope);" in page
    assert "getTrack2SqlPreviews(page)" in page
    assert '<img src={p.url}' in page
    # the editor is still there, folded away
    assert "Advanced: write your own SQL" in page


# --- labels from the reviewed boxes ---------------------------------------------

_ROW = {
    "width": 1000, "height": 500,
    "people": [
        {"person_box": {"x1": 100, "y1": 100, "x2": 200, "y2": 400}, "wearing": True,
         "auto": {"x1": 120, "y1": 100, "x2": 180, "y2": 140}},
        {"person_box": {"x1": 500, "y1": 50, "x2": 600, "y2": 450}, "wearing": False, "auto": None},
    ],
    "drawn_box": None,
}


def test_every_person_gets_a_box_and_each_wearer_a_helmet():
    labels = sx.frame_labels(_ROW)
    assert labels["complete"] is True
    assert sorted(b["name"] for b in labels["boxes"]) == ["helmet", "person", "person"]
    helmet = next(b for b in labels["boxes"] if b["cls"] == sx.HELMET)
    assert (helmet["x1"], helmet["y1"], helmet["x2"], helmet["y2"]) == (0.12, 0.2, 0.18, 0.28)


def test_a_drawn_box_wins_for_the_one_wearer():
    labels = sx.frame_labels({**_ROW, "drawn_box": [110, 90, 190, 150]})
    helmet = next(b for b in labels["boxes"] if b["cls"] == sx.HELMET)
    assert helmet["x1"] == 0.11 and helmet["y2"] == 0.3


def test_a_wearer_without_a_helmet_box_makes_the_frame_incomplete():
    row = {**_ROW, "people": [{**_ROW["people"][0], "auto": None}]}
    labels = sx.frame_labels(row)
    assert labels["complete"] is False and "missing" in labels["reason"]


def test_the_yolo_line_is_centre_and_size():
    text = sx.yolo_text({"boxes": [{"cls": 2, "x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.6}]})
    assert text == "2 0.200000 0.400000 0.200000 0.400000\n"


def test_the_class_numbers_match_the_curated_sets():
    assert sx.LABEL_NAMES[0] == "helmet" and sx.LABEL_NAMES[2] == "person"
    assert "0: helmet" in sx.data_yaml_text() and "2: person" in sx.data_yaml_text()


def test_a_labelled_pull_writes_labels_and_yaml_and_leaves_out_incomplete(monkeypatch):
    columns = ["image", "width", "height", "people", "drawn_box"]
    good = ["s3://src/f/a.jpg", 1000, 500, _ROW["people"], None]
    bad = ["s3://src/f/b.jpg", 1000, 500, [{**_ROW["people"][0], "auto": None}], None]
    fake = _Lambda({"ok": True, "columns": columns, "rows": [good, bad], "truncated": False})
    monkeypatch.setattr(sx, "_client", lambda kind: fake)
    s3 = _S3({("src", "f/a.jpg"): b"A", ("src", "f/b.jpg"): b"B"})
    monkeypatch.setattr(sx.storage, "s3", s3)
    out = sx.pull("violation-review", "q", dest_bucket="d", dest_folder="p", with_labels=True)
    assert out["copied"] == 1 and out["left_out_incomplete"] == 1
    assert out["labels_from"] == "database" and out["with_label"] == 1
    label = s3.objects[("d", "p/labels/train/a.txt")].decode()
    assert label.count("\n") == 3 and label.startswith("2 ")
    assert ("d", "p/data.yaml") in s3.objects
    assert ("d", "p/images/train/b.jpg") not in s3.objects


def test_without_labels_only_images_are_pulled(monkeypatch):
    columns = ["image", "width", "height", "people", "drawn_box"]
    bad = ["s3://src/f/b.jpg", 1000, 500, [{**_ROW["people"][0], "auto": None}], None]
    fake = _Lambda({"ok": True, "columns": columns, "rows": [bad], "truncated": False})
    monkeypatch.setattr(sx, "_client", lambda kind: fake)
    s3 = _S3({("src", "f/b.jpg"): b"B"})
    monkeypatch.setattr(sx.storage, "s3", s3)
    out = sx.pull("violation-review", "q", dest_bucket="d", dest_folder="p", with_labels=False)
    assert out["copied"] == 1 and out["labels_from"] == "none"
    assert not any(k.startswith("p/labels/") for (_, k) in s3.objects)
    assert ("d", "p/data.yaml") not in s3.objects


# --- starting and stopping a database from the tab --------------------------------

class _Rds:
    def __init__(self, status):
        self.status, self.calls = status, []

    def describe_db_instances(self, DBInstanceIdentifier):
        return {"DBInstances": [{"DBInstanceStatus": self.status}]}

    def start_db_instance(self, DBInstanceIdentifier):
        self.calls.append(("start", DBInstanceIdentifier))

    def stop_db_instance(self, DBInstanceIdentifier):
        self.calls.append(("stop", DBInstanceIdentifier))


def test_a_stopped_database_is_started_by_its_own_instance_name(monkeypatch):
    rds = _Rds("stopped")
    monkeypatch.setattr(sx, "_client", lambda kind: rds)
    out = sx.set_running("jpui-preview", True)
    assert out == {"key": "jpui-preview", "status": "starting", "changed": True}
    assert rds.calls == [("start", "jpui-preview-crop-authority")]


def test_starting_one_already_running_changes_nothing(monkeypatch):
    rds = _Rds("available")
    monkeypatch.setattr(sx, "_client", lambda kind: rds)
    assert sx.set_running("violation-review", True)["changed"] is False
    assert rds.calls == []


def test_only_databases_on_offer_can_be_started():
    with pytest.raises(ValueError):
        sx.set_running("some-other-instance", True)


def test_the_start_button_sits_under_the_stopped_database():
    page = (ROOT / "web" / "src" / "routes" / "Track2SqlPage.tsx").read_text(encoding="utf-8")
    assert 'onClick={() => switchPower(true)}' in page
    assert "runId.current += 1;      // abandon a search against the old one" in page
