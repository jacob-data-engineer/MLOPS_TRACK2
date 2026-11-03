"""The central data catalog, step 1 (2026-10-29): its tables and routes."""
from __future__ import annotations

import pathlib

from ppe_pipeline_api.data import catalog

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_training_history_uses_catalog_numbers_not_row_order(monkeypatch):
    from ppe_pipeline_api.routers import track2_models as router
    monkeypatch.setattr(router.t2, "scan_training_audits", lambda **kwargs: [{"id": "new"}, {"id": "old"}, {"id": "unsynced"}])
    monkeypatch.setattr(router.catalog_view, "run_numbers", lambda ids: {"old": 7, "new": 58})
    result = router.training_audit(session={})["audits"]
    assert [a["run_number"] for a in result] == [58, 7, None]


def test_training_history_still_loads_when_catalog_is_unavailable(monkeypatch):
    from ppe_pipeline_api.routers import track2_models as router
    monkeypatch.setattr(router.t2, "scan_training_audits", lambda **kwargs: [{"id": "run"}])
    def unavailable(ids):
        raise RuntimeError("catalog unavailable")
    monkeypatch.setattr(router.catalog_view, "run_numbers", unavailable)
    assert router.training_audit(session={})["audits"] == [{"id": "run", "run_number": None}]


def test_nothing_in_the_schema_drops_or_deletes():
    text = " ".join(catalog.SCHEMA).upper()
    for word in ("DROP ", "DELETE ", "TRUNCATE"):
        assert word not in text
    assert all("IF NOT EXISTS" in s for s in catalog.SCHEMA)


def test_the_split_and_decay_live_on_the_version_not_the_file():
    members = next(s for s in catalog.SCHEMA if "dataset_members" in s and "CREATE TABLE" in s)
    for column in ("split", "included", "reason", "decay_weight", "label_id"):
        assert column in members
    files = next(s for s in catalog.SCHEMA if "TABLE IF NOT EXISTS files" in s)
    assert "split" not in files


def test_labels_are_versioned_not_overwritten():
    labels = next(s for s in catalog.SCHEMA if "TABLE IF NOT EXISTS labels" in s)
    assert "UNIQUE (sha256, version)" in labels and "class_names" in labels


def test_unconfigured_says_so_rather_than_failing(monkeypatch):
    monkeypatch.setattr(catalog, "HOST", "")
    assert catalog.status() == {"configured": False, "reachable": False, "tables": {}}


def test_the_routes_and_the_deploy_wiring_exist():
    from ppe_pipeline_api.main import app
    paths = app.openapi()["paths"]
    assert "/track2/catalog/status" in paths and "/track2/catalog/schema" in paths
    deploy = (ROOT / "deploy.sh").read_text(encoding="utf-8")
    assert "-e CATALOG_HOST='$CATALOG_HOST'" in deploy
    tf = (ROOT / "terraform" / "app" / "catalog.tf").read_text(encoding="utf-8")
    assert "deletion_protection        = true" in tf and "publicly_accessible    = false" in tf


def test_the_connection_is_verified_against_the_shipped_rds_roots():
    import inspect
    assert catalog.CA_BUNDLE.exists()
    assert catalog.CA_BUNDLE.read_text().count("BEGIN CERTIFICATE") > 50
    source = inspect.getsource(catalog.connect)
    assert "cafile=str(CA_BUNDLE)" in source and "CERT_NONE" not in source


# --- records moving from DynamoDB (schema version 2) ------------------------------

from ppe_pipeline_api.data import catalog_sync as cs  # noqa: E402


def test_the_migration_only_adds():
    text = " ".join(s for steps in catalog.MIGRATIONS.values() for s in steps).upper()
    for word in ("DROP TABLE", "DROP COLUMN", "DELETE ", "TRUNCATE"):
        assert word not in text
    assert "RUN_NUMBER BIGSERIAL" in text and "VERSION_NUMBER BIGSERIAL" in text


def test_a_version_manifest_becomes_one_member_per_image(monkeypatch):
    import io, json as _json
    body = {"version_id": "dv-1", "train": ["b/set/images/train/a.jpg"],
            "val": ["b/set/images/val/c.jpg"]}
    monkeypatch.setattr(cs.storage, "s3", type("S", (), {
        "get_object": lambda self, Bucket, Key: {"Body": io.BytesIO(_json.dumps(body).encode())}})())
    got = cs.members_of({"manifest_uri": "s3://b/_datasets/dv-1/manifest.json"})
    assert got == [("s3://b/set/images/train/a.jpg", "s3://b/set/labels/train/a.txt", "train", ""),
                   ("s3://b/set/images/val/c.jpg", "s3://b/set/labels/val/c.txt", "val", "")]


def test_a_snapshot_manifest_keeps_images_only_with_their_week(monkeypatch):
    import io, json as _json
    body = {"prefix": "retraining", "files": ["week_1/images/train/a.jpg",
                                              "week_1/labels/train/a.txt",
                                              "week_2/images/val/b.png"]}
    monkeypatch.setattr(cs.storage, "s3", type("S", (), {
        "get_object": lambda self, Bucket, Key: {"Body": io.BytesIO(_json.dumps(body).encode())}})())
    got = cs.members_of({"bucket": "b", "manifest_key": "m.json"})
    assert [(g[0], g[2], g[3]) for g in got] == [
        ("s3://b/retraining/week_1/images/train/a.jpg", "train", "week_1"),
        ("s3://b/retraining/week_2/images/val/b.png", "val", "week_2")]


def test_long_file_lists_are_kept_as_counts_not_copied_into_the_record():
    record = {"train_dataset": {"file_names": [str(n) for n in range(1000)], "file_count": 1000}}
    assert cs._slim(record) == {"train_dataset": {"file_names": {"_count": 1000}, "file_count": 1000}}


def test_one_sync_per_five_minute_slot():
    import datetime as dt
    a = cs.due(dt.datetime(2026, 10, 29, 9, 41, 30, tzinfo=dt.timezone.utc))
    b = cs.due(dt.datetime(2026, 10, 29, 9, 44, 59, tzinfo=dt.timezone.utc))
    c = cs.due(dt.datetime(2026, 10, 29, 9, 45, 0, tzinfo=dt.timezone.utc))
    assert a == b != c


def test_the_tick_and_the_route_start_it():
    ticker = (ROOT / "src" / "ppe_pipeline_api" / "chain" / "ticker.py").read_text(encoding="utf-8")
    assert '_pcc.claim_once, "catalog_sync", _data.catalog_sync_slot()' in ticker
    from ppe_pipeline_api.main import app
    assert "/track2/catalog/sync" in app.openapi()["paths"]


# --- the Data Catalog page and its exports ----------------------------------------

from ppe_pipeline_api.data import catalog_view as cv  # noqa: E402


class _S3Zip:
    """Enough of S3 for an export: reads by key, a multipart upload in parts."""

    def __init__(self, objects):
        self.objects, self.parts = dict(objects), {}

    def get_object(self, Bucket, Key):
        import io
        if (Bucket, Key) not in self.objects:
            from botocore.exceptions import ClientError
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    def create_multipart_upload(self, Bucket, Key, ContentType):
        self.parts[Key] = []
        return {"UploadId": "u"}

    def upload_part(self, Bucket, Key, UploadId, PartNumber, Body):
        self.parts[Key].append(Body)
        return {"ETag": f"e{PartNumber}"}

    def complete_multipart_upload(self, Bucket, Key, UploadId, MultipartUpload):
        self.objects[(Bucket, Key)] = b"".join(self.parts[Key])

    def put_object(self, Bucket, Key, Body):
        self.objects[(Bucket, Key)] = Body

    def abort_multipart_upload(self, Bucket, Key, UploadId):
        self.parts.pop(Key, None)

    def delete_object(self, Bucket, Key):
        self.objects.pop((Bucket, Key), None)


def test_an_export_is_a_valid_zip_in_the_training_layout(monkeypatch):
    import io, zipfile
    s3 = _S3Zip({("b", "set/images/train/a.jpg"): b"A", ("b", "set/labels/train/a.txt"): b"0 .5 .5 .1 .1"})
    monkeypatch.setattr(cv.storage, "s3", s3)
    monkeypatch.setattr(cv, "PART", 10)            # several parts, even for tiny files
    sink = cv._MultipartSink("out", "x.zip")
    with zipfile.ZipFile(sink, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
        zf.writestr("images/train/a.jpg", s3.get_object("b", "set/images/train/a.jpg")["Body"].read())
        zf.writestr("labels/train/a.txt", b"0 .5 .5 .1 .1")
    sink.finish()
    with zipfile.ZipFile(io.BytesIO(s3.objects[("out", "x.zip")])) as back:
        assert back.read("images/train/a.jpg") == b"A"
        assert back.namelist() == ["images/train/a.jpg", "labels/train/a.txt"]
    assert len(s3.parts["x.zip"]) > 1


def test_a_file_that_cannot_be_read_is_reported_missing_not_fatal(monkeypatch):
    monkeypatch.setattr(cv.storage, "s3", _S3Zip({}))
    assert cv._fetch("s3://b/gone.jpg") is None
    assert cv._fetch("not-s3") is None


def test_the_page_is_routed_and_offered():
    web = ROOT / "web" / "src"
    assert 'path="/track2/catalog" element={<Track2CatalogPage />}' in (web / "App.tsx").read_text(encoding="utf-8")
    assert '"Track2 Data Catalog"' in (web / "components" / "Layout.tsx").read_text(encoding="utf-8")
    card = (web / "components" / "Track2CentralStorePanel.tsx").read_text(encoding="utf-8")
    assert "formatDayTime(catalog.last_sync.finished_at)" in card and "UTC`" not in card
    from ppe_pipeline_api.main import app
    paths = app.openapi()["paths"]
    for path in ("/track2/catalog/runs", "/track2/catalog/versions", "/track2/catalog/exports",
                 "/track2/catalog/exports/{export_id}/download"):
        assert path in paths


def test_class_names_are_read_whichever_way_they_were_recorded():
    """Versions store a list; others a numbered map. An export failed on the list."""
    assert cv._numbered(["helmet", "gloves"]) == {0: "helmet", 1: "gloves"}
    assert cv._numbered({"0": "helmet", "2": "person"}) == {0: "helmet", 2: "person"}
    assert cv._numbered('["helmet"]') == {0: "helmet"}
    assert cv._numbered(None) == {}


# --- fixes from the 2026-11-02 review ----------------------------------------------

def test_the_label_is_where_ultralytics_looks_last_images_folder():
    from ppe_pipeline_api import storage
    assert storage.label_path_for("s3://b/images/w40/images/train/a.jpg") == "s3://b/images/w40/labels/train/a.txt"
    assert storage.label_path_for("set/images/train/a.jpg") == "set/labels/train/a.txt"
    # no extension: only the file name is touched, never the bucket
    assert storage.label_path_for("s3://track2.bucket/x/images/source") == "s3://track2.bucket/x/labels/source.txt"
    assert storage.label_path_for("frames/a.jpg") == ""


def test_an_empty_snapshot_prefix_is_respected(monkeypatch):
    import io, json as _json
    body = {"prefix": "", "root": "x", "files": ["other/images/train/a.jpg"]}
    monkeypatch.setattr(cs.storage, "s3", type("S", (), {
        "get_object": lambda self, Bucket, Key: {"Body": io.BytesIO(_json.dumps(body).encode())}})())
    got = cs.members_of({"bucket": "b", "manifest_key": "m.json", "root": "x"})
    assert got[0][0] == "s3://b/other/images/train/a.jpg"


def test_a_bad_timestamp_is_dropped_not_fatal():
    assert cs._when("2026-10-16T04:24:57+00:00") == "2026-10-16T04:24:57+00:00"
    assert cs._when("None") is None and cs._when("1697000000") is None
    assert cs._int("12") == 12 and cs._int("n/a") == 0


def test_a_renamed_image_and_its_label_keep_the_same_suffix():
    import inspect
    src = inspect.getsource(cv._plan)
    assert "pair_key = image_uri.rsplit" in src


def test_the_export_always_frees_its_slot_and_records_on_a_fresh_connection():
    import inspect
    build = inspect.getsource(cv._build)
    assert "catalog.connect()" not in build.split("try:", 1)[0]
    assert "_building.release()" in build.split("finally:")[-1]
    assert "catalog.connect()" in inspect.getsource(cv._finish)


def test_the_run_is_filed_under_the_version_it_trained_on():
    import inspect
    src = inspect.getsource(cs._one_run)
    assert src.index('get("train_dataset")') < src.index('get("snapshot")')


def test_the_repair_cuts_by_position_not_by_pattern():
    """pg8000 sends a bare %s as text, and substring(x from '40') is a
    regex search -- every repaired address came out as s3://bucket/40."""
    import inspect
    src = inspect.getsource(cs._repair_composed)
    assert src.count("from %s::int)") == 2


def test_exports_write_real_bytes_to_both_destinations(monkeypatch):
    monkeypatch.setattr(cv, "_rows", lambda *a: [{"state": "building"}])
    import io
    import threading
    import zipfile

    for destination, export_format in (("local", "zip"), ("s3", "folder"), ("s3", "zip")):
        s3 = _S3Zip({("source", "a.jpg"): b"actual-image", ("source", "a.txt"): b"0 .5 .5 .1 .1"})
        monkeypatch.setattr(cv.storage, "s3", s3)
        slots = threading.BoundedSemaphore(1)
        slots.acquire()
        monkeypatch.setattr(cv, "_building", slots)
        results = []
        monkeypatch.setattr(cv, "_finish", lambda *args: results.append(args))
        entries = [(path, uri, {"path": path, "source": uri}) for path, uri in (
            ("images/train/a.jpg", "s3://source/a.jpg"),
            ("labels/train/a.txt", "s3://source/a.txt"),
            ("images/val/gone.jpg", "s3://source/gone.jpg"))]
        cv._build("export1", "run", "r", "run-1", entries, ["helmet"], destination, "chosen-bucket", "chosen/folder", export_format)
        root = "chosen/folder/run-1-export1/"
        if export_format == "zip":
            with zipfile.ZipFile(io.BytesIO(s3.objects[("chosen-bucket", root + "run-1.zip")])) as archive:
                output = {p: archive.read(p) for p in archive.namelist()}
        else:
            output = {k[len(root):]: v for (b, k), v in s3.objects.items() if b == "chosen-bucket"}
        assert output["images/train/a.jpg"] == b"actual-image"
        assert output["labels/train/a.txt"] == b"0 .5 .5 .1 .1"
        assert b"helmet" in output["data.yaml"]
        assert b"gone.jpg" in output["manifest.csv"]
        assert "images/val/gone.jpg" not in output
        params = results[-1][2]
        assert params[0] == "partial"
        assert params[3:5] == (2, 1)
        assert slots.acquire(blocking=False)


def test_access_denied_is_not_misreported_as_a_missing_file(monkeypatch):
    import pytest
    from botocore.exceptions import ClientError

    class Denied:
        def get_object(self, **kwargs):
            raise ClientError({"Error": {"Code": "AccessDenied"}}, "GetObject")
    monkeypatch.setattr(cv.storage, "s3", Denied())
    with pytest.raises(ClientError, match="AccessDenied"):
        cv._fetch("s3://source/a.jpg")


def test_export_destination_validation():
    import pytest
    assert cv._destination("s3", " chosen-bucket ", "/exports/helmet/") == ("chosen-bucket", "exports/helmet")
    for destination, bucket, prefix in (("email", "bucket", ""), ("s3", "", ""),
                                      ("s3", "s3://bucket", ""), ("s3", "bucket", "../images")):
        with pytest.raises(ValueError):
            cv._destination(destination, bucket, prefix)


def test_export_route_forwards_destination_and_old_requests_default_to_local(monkeypatch):
    from ppe_pipeline_api.routers import track2_catalog as router
    got = []
    monkeypatch.setattr(cv, "start_export", lambda *args, **kwargs: got.append((args, kwargs)) or {"state": "building"})
    router.start_export(router.ExportRequest(kind="run", id="r", destination="s3", bucket="chosen-bucket", prefix="exports"), session={})
    assert got[-1] == (("run", "r"), {"destination": "s3", "bucket": "chosen-bucket", "prefix": "exports", "format": None,
                                      "include_metadata": False, "include_images": True, "include_model": False})
    router.start_export(router.ExportRequest(kind="version", id="v"), session={})
    assert got[-1][1]["destination"] == "local"


def test_partial_zip_remains_downloadable(monkeypatch):
    monkeypatch.setattr(cv, "_rows", lambda *args: [{"zip_uri": "s3://output/run.zip", "label": "run-1", "state": "partial"}])
    class Signer:
        def generate_presigned_url(self, operation, **kwargs):
            assert kwargs["Params"]["Key"] == "run.zip"
            return "signed-url"
    monkeypatch.setattr(cv.storage, "s3", Signer())
    assert cv.download_url("export1") == "signed-url"


def test_s3_write_failure_is_recorded_and_releases_slot(monkeypatch):
    monkeypatch.setattr(cv, "_rows", lambda *a: [{"state": "building"}])
    import threading
    from botocore.exceptions import ClientError

    class Denied(_S3Zip):
        def put_object(self, **kwargs):
            raise ClientError({"Error": {"Code": "AccessDenied"}}, "PutObject")
    monkeypatch.setattr(cv.storage, "s3", Denied({("source", "a.jpg"): b"image"}))
    slots = threading.BoundedSemaphore(1)
    slots.acquire()
    monkeypatch.setattr(cv, "_building", slots)
    results = []
    monkeypatch.setattr(cv, "_finish", lambda *args: results.append(args))
    cv._build("export2", "run", "r", "run-1", [("images/train/a.jpg", "s3://source/a.jpg", {})],
              None, "s3", "destination", "exports")
    assert "state = 'failed'" in results[-1][1]
    assert "AccessDenied" in results[-1][2][0]
    assert slots.acquire(blocking=False)


def test_stop_during_export_cleans_output_and_releases_slot(monkeypatch):
    import threading
    for fmt in ('zip', 'folder'):
        s3 = _S3Zip({('source', 'a.jpg'): b'image'})
        monkeypatch.setattr(cv.storage, 's3', s3)
        monkeypatch.setattr(cv, '_rows', lambda *a: [{'state': 'building'}])
        event = threading.Event()
        cv._stop_events['cancel-test'] = event
        original_fetch = cv._fetch
        def fetch(uri):
            body = original_fetch(uri)
            event.set()
            return body
        with monkeypatch.context() as patch:
            patch.setattr(cv, '_fetch', fetch)
            slots = threading.BoundedSemaphore(1)
            slots.acquire()
            patch.setattr(cv, '_building', slots)
            results = []
            patch.setattr(cv, '_finish', lambda *a: results.append(a))
            cv._build('cancel-test', 'run', 'r', 'run-1', [('images/train/a.jpg', 's3://source/a.jpg', {})], None, 's3', 'dest', 'exports', fmt)
            assert "state = 'cancelled'" in results[-1][1]
            assert not s3.parts
            assert not any(bucket == 'dest' for bucket, key in s3.objects)
            assert slots.acquire(blocking=False)
            assert 'cancel-test' not in cv._stop_events


def test_stop_wins_final_publication_race(monkeypatch):
    import threading
    s3 = _S3Zip({('source', 'a.jpg'): b'image'})
    monkeypatch.setattr(cv.storage, 's3', s3)
    monkeypatch.setattr(cv, '_rows', lambda *a: [{'state': 'building'}])
    slots = threading.BoundedSemaphore(1)
    slots.acquire()
    monkeypatch.setattr(cv, '_building', slots)
    results = []
    def finish(*args):
        results.append(args)
        if 'AND state' in args[1]:
            return False
    monkeypatch.setattr(cv, '_finish', finish)
    cv._build('race', 'run', 'r', 'run-1', [('images/train/a.jpg', 's3://source/a.jpg', {})], None, 'local', 'dest', 'exports', 'zip')
    assert "state = 'cancelled'" in results[-1][1]
    assert not any(bucket == 'dest' for bucket, key in s3.objects)


def test_cancel_cleanup_failure_is_visible(monkeypatch):
    import threading
    class DeniedAbort(_S3Zip):
        def abort_multipart_upload(self, **kwargs):
            raise RuntimeError('abort denied')
    s3 = DeniedAbort({})
    monkeypatch.setattr(cv.storage, 's3', s3)
    monkeypatch.setattr(cv, '_rows', lambda *a: [{'state': 'building'}])
    event = threading.Event()
    cv._stop_events['denied-stop'] = event
    def fetch(uri):
        event.set()
        return b'image'
    monkeypatch.setattr(cv, '_fetch', fetch)
    slots = threading.BoundedSemaphore(1)
    slots.acquire()
    monkeypatch.setattr(cv, '_building', slots)
    results = []
    monkeypatch.setattr(cv, '_finish', lambda *a: results.append(a))
    cv._build('denied-stop','run','r','run-1',[('images/a.jpg','s3://source/a.jpg',{})],None,'local','dest','exports','zip')
    assert "state = 'failed'" in results[-1][1]
    assert 'cleanup failed: abort denied' in results[-1][2][0]
    assert slots.acquire(blocking=False)


def test_stop_before_build_creates_no_upload(monkeypatch):
    import threading
    s3 = _S3Zip({})
    monkeypatch.setattr(cv.storage, 's3', s3)
    monkeypatch.setattr(cv, '_rows', lambda *a: [{'state': 'stopping'}])
    slots = threading.BoundedSemaphore(1)
    slots.acquire()
    monkeypatch.setattr(cv, '_building', slots)
    results=[]
    monkeypatch.setattr(cv, '_finish', lambda *a: results.append(a))
    cv._build('early-stop','run','r','run-1',[],None,'local','dest','exports','zip')
    assert "state = 'cancelled'" in results[-1][1]
    assert not s3.parts
    assert slots.acquire(blocking=False)


def test_export_run_weights_uses_exact_checkpoint(monkeypatch):
    class S3:
        def __init__(self): self.copies=[]
        def head_object(self, **kw): assert kw == {'Bucket':'source','Key':'models/best.pt'}
        def generate_presigned_url(self, operation, **kw):
            assert operation == 'get_object'
            assert kw['Params']['Key'] == 'models/best.pt'
            assert kw['Params']['ResponseContentDisposition'] == 'attachment; filename="run-58.pt"'
            return 'signed-url'
        def copy(self, source, bucket, key): self.copies.append((source,bucket,key))
    s3=S3()
    monkeypatch.setattr(cv.storage,'s3',s3)
    monkeypatch.setattr(cv,'_rows',lambda *a: [{'run_number':58,'record':{'weights_uri':'s3://source/models/best.pt'}}])
    assert cv.export_run_weights('audit58') == {'url':'signed-url'}
    first=cv.export_run_weights('audit58',destination='s3',bucket='chosen-bucket',prefix='new/folder')
    second=cv.export_run_weights('audit58',destination='s3',bucket='chosen-bucket',prefix='new/folder')
    assert first != second
    assert first['uri'].startswith('s3://chosen-bucket/new/folder/run-58-')
    assert first['uri'].endswith('/run-58.pt')
    assert s3.copies[0][0] == {'Bucket':'source','Key':'models/best.pt'}
    assert s3.copies[0][1] == 'chosen-bucket'
    import pytest
    with pytest.raises(ValueError): cv.export_run_weights('audit58',destination='s3',bucket='chosen-bucket',prefix='../bad')


def test_export_run_weights_missing_checkpoint(monkeypatch):
    import pytest
    monkeypatch.setattr(cv,'_rows',lambda *a: [])
    with pytest.raises(ValueError,match='Run not found'): cv.export_run_weights('gone')
    monkeypatch.setattr(cv,'_rows',lambda *a: [{'run_number':1,'record':{}}])
    with pytest.raises(ValueError,match='No .pt checkpoint'): cv.export_run_weights('pending')


def test_complete_bundle_contains_enriched_model_and_checksums(monkeypatch):
    import io,zipfile,threading,json,csv,hashlib
    from ppe_pipeline_api.data import catalog_metadata as cm
    checkpoint=io.BytesIO()
    with zipfile.ZipFile(checkpoint,'w') as z: z.writestr('archive/data.pkl',b'opaque-model')
    for fmt in ('zip','folder'):
        s3=_S3Zip({('source','a.jpg'):b'image',('source','model.pt'):checkpoint.getvalue()})
        monkeypatch.setattr(cv.storage,'s3',s3)
        monkeypatch.setattr(cv,'_rows',lambda *a:[{'state':'building'}])
        slots=threading.BoundedSemaphore(1);slots.acquire();monkeypatch.setattr(cv,'_building',slots)
        results=[];monkeypatch.setattr(cv,'_finish',lambda *a:results.append(a))
        metadata={'schema':cm.SCHEMA,'label':'run-1','groups':{'Evaluation':{'mAP50':0.8},'Training history':{}},'records':{}}
        entries=[(path,'s3://source/'+key,{'path':path,'source':'s3://source/'+key}) for path,key in [('images/train/a.jpg','a.jpg'),('models/run-1.pt','model.pt')]]
        entries.insert(0, ('', 's3://source/excluded.jpg', {'path':'images/val/excluded.jpg','source':'s3://source/excluded.jpg','included':False}))
        cv._build('bundle','run','r','run-1',entries,['helmet'],'s3','dest','exports',fmt,metadata)
        root='exports/run-1-bundle/'
        if fmt=='zip':
            z=zipfile.ZipFile(io.BytesIO(s3.objects[('dest',root+'run-1.zip')]))
            files={n:z.read(n) for n in z.namelist()}
        else: files={key[len(root):]:value for (bucket,key),value in s3.objects.items() if bucket=='dest'}
        assert {'metadata.json','metrics.json','training-history.json','data.yaml','manifest.csv','models/run-1.pt','images/train/a.jpg'} <= files.keys()
        assert json.loads(files['metadata.json'])['export_result']['missing_files']==0
        embedded=cm.inspect_file(io.BytesIO(files['models/run-1.pt']),'model.pt')
        assert embedded['groups']['Evaluation']['mAP50']==0.8
        assert 'images/train/a.jpg' in embedded['file_manifest']
        assert 'excluded.jpg' in embedded['file_manifest']
        for row in csv.DictReader(io.StringIO(files['manifest.csv'].decode())):
            if row['included']=='False': continue
            assert hashlib.sha256(files[row['path']]).hexdigest()==row['sha256']
        assert results[-1][2][0]=='ready'
