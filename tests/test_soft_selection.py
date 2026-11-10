"""Selection behaviour with a SQLite SQL adapter and in-memory S3.
Real PostgreSQL/S3 integration is checked separately before deployment.
"""
import copy, datetime as dt, hashlib, io, json, sqlite3
import pytest
from botocore.exceptions import ClientError
from ppe_pipeline_api.data import soft_selection as soft, selection_catalog as db
from ppe_pipeline_api.data import selection_definition as definitions, selection_jobs as jobs
from ppe_pipeline_api import pull_control as pc

class Cursor:
    def __init__(self,c):self.c=c
    def execute(self,sql,args=()):
        if "pg_advisory" in sql:return self
        self.c.execute(sql.replace("%s","?").replace("::jsonb","").replace("now()","CURRENT_TIMESTAMP").replace(" FOR UPDATE",""),args)
        return self
    def fetchall(self):return self.c.fetchall()
    def fetchone(self):return self.c.fetchone()
    @property
    def description(self):return self.c.description
    @property
    def rowcount(self):return self.c.rowcount
class Conn:
    def __init__(self,path):
        self.c=sqlite3.connect(path,detect_types=sqlite3.PARSE_DECLTYPES)
        self.c.execute("PRAGMA foreign_keys=ON")
    def cursor(self):return Cursor(self.c.cursor())
    def commit(self):self.c.commit()
    def rollback(self):self.c.rollback()
    def close(self):self.c.close()
class S3:
    def __init__(self):self.objects={};self.copies=[]
    def copy_object(self,**kw):
        src=kw["CopySource"];self.objects[(kw["Bucket"],kw["Key"])]=self.objects[(src["Bucket"],src["Key"])];self.copies.append(kw)
    def put_object(self,**kw):self.objects[(kw["Bucket"],kw["Key"])]=kw["Body"]
    def get_object(self,**kw):
        key=(kw["Bucket"],kw["Key"])
        if key not in self.objects:raise ClientError({"Error":{"Code":"NoSuchKey"}},"GetObject")
        body=self.objects[key]
        if kw.get("IfMatch"):assert kw["IfMatch"].strip('"')==hashlib.md5(body).hexdigest()
        return {"Body":io.BytesIO(body)}

@pytest.fixture
def env(tmp_path,monkeypatch):
    sqlite3.register_converter("jsonb",lambda v:json.loads(v))
    path=tmp_path/"test.db";conn=Conn(path)
    for statement in [
        "CREATE TABLE files(sha256 TEXT PRIMARY KEY,store_bucket TEXT,store_key TEXT,bytes INTEGER)",
        "CREATE TABLE file_sources(sha256 TEXT,source_uri TEXT,source_kind TEXT,detail jsonb, UNIQUE(sha256,source_uri))",
        "CREATE TABLE labels(id INTEGER PRIMARY KEY AUTOINCREMENT,sha256 TEXT REFERENCES files(sha256),version INTEGER,class_names jsonb,yolo TEXT,origin TEXT,source_uri TEXT)",
        "CREATE TABLE dataset_versions(id TEXT PRIMARY KEY,purpose TEXT,class_names jsonb,settings jsonb,kind TEXT,state TEXT,train_images INTEGER,val_images INTEGER,record jsonb,members_synced BOOLEAN,version_number INTEGER DEFAULT 1)",
        "CREATE TABLE dataset_members(version_id TEXT,sha256 TEXT,label_id INTEGER,split TEXT,included BOOLEAN,reason TEXT,week TEXT,image_uri TEXT,label_uri TEXT,stored_image_uri TEXT,stored_label_uri TEXT,image_etag TEXT,UNIQUE(version_id,image_uri))",
    ]:conn.cursor().execute(statement)
    conn.commit();conn.close()
    monkeypatch.setattr(db.catalog,"ensure_schema_once",lambda:None)
    monkeypatch.setattr(db.catalog,"connect",lambda:Conn(path))
    s3=S3();monkeypatch.setattr(soft.storage,"s3",s3)
    monkeypatch.setattr(soft.central_store,"location",lambda:("store","shared"))
    monkeypatch.setattr(soft.central_store,"_etag",lambda b,k:"exists" if (b,k) in s3.objects else None)
    def listing(prefix,bucket):
        return [{"Key":k,"ETag":hashlib.md5(v).hexdigest(),"Size":len(v),"LastModified":dt.datetime.now(dt.timezone.utc)} for (b,k),v in list(s3.objects.items()) if b==bucket and k.startswith(prefix)]
    monkeypatch.setattr(pc,"_list_objects",listing)
    s3.objects[("src","data/data.yaml")]=b"names: {0: helmet, 2: person}"
    add(s3,"a","train");add(s3,"v","val")
    return s3

def add(env,name,split,content=None,label="2 0.5 0.5 0.2 0.2\n"):
    env.objects[("src",f"data/images/{split}/{name}.jpg")]=content if content is not None else name.encode()
    env.objects[("src",f"data/labels/{split}/{name}.txt")]=label.encode()

def config(sid):
    definition=definitions.Definition(sources=[definitions.Source(id=split,name=split,bucket="src",images="data/images/"+split,labels="data/labels/"+split,kind="collected",split=split) for split in ("train","val")],class_ids={0:"helmet",2:"person"})
    return {"soft_selection":{"enabled":True,"id":sid},"data_selection":definition.model_dump(),"retention":{"enabled":False}}

def test_independent_selections_share_canonical_assets(env):
    a=config(soft.create());b=config(soft.create())
    soft.sync(a);count=len(env.copies);soft.sync(b)
    assert len(env.copies)==count==2
    soft.exclude_week(a,"week_1");soft.sync(a)
    assert len(soft.members(a))==0 and len(soft.members(b))==2
    c=db.connection();cur=c.cursor();cur.execute("SELECT count(*) FROM files");assert cur.fetchone()[0]==2;c.close()

def test_annotation_revision_does_not_change_clone(env):
    a=config(soft.create());soft.sync(a);b=config(soft.create(a["soft_selection"]["id"]))
    add(env,"a","train",label="0 0.5 0.5 0.1 0.1\n");soft.sync(a)
    assert next(m for m in soft.members(a) if m["split"]=="train")["yolo"].startswith("0 ")
    assert all(m["yolo"].startswith("2 ") for m in soft.members(b))

def test_retention_ages_nonempty_collections_and_keeps_val(env):
    cfg=config(soft.create());soft.sync(cfg);assert soft.sync(cfg)["moved"]==0
    add(env,"b","train");soft.sync(cfg)
    cfg["retention"]={"enabled":True,"weekly_rate":1,"full_weeks":1,"use_floor":False}
    assert {m["source_uri"].rsplit("/",1)[-1] for m in soft.select(cfg) if m["included"]}=={"b.jpg","v.jpg"}
    assert len(soft.members(cfg))==3

def test_source_validation_is_read_only_and_reports_missing_labels(env):
    cfg=config(soft.create());del env.objects[("src","data/labels/train/a.txt")]
    before=dict(env.objects);report=soft.scan(cfg)
    assert report["eligible"]==1 and len(report["excluded"])==1
    assert env.objects==before and not soft.members(cfg)

def test_missing_source_blocks_refresh_without_removing_members(env):
    cfg=config(soft.create());soft.sync(cfg)
    cfg["data_selection"]["sources"][0]["images"]="missing"
    with pytest.raises(ValueError,match="no images"):soft.sync(cfg)
    assert len(soft.members(cfg))==2

def test_same_content_cannot_cross_splits(env):
    cfg=config(soft.create());add(env,"v","val",content=b"a")
    with pytest.raises(ValueError,match="both training and validation"):soft.sync(cfg)
    assert not soft.members(cfg)


def test_existing_golden_validation_membership_survives_migration(env):
    cfg=config(soft.create())
    cfg['data_selection']['sources']=cfg['data_selection']['sources'][:1]
    source=cfg['data_selection']['sources'][0]
    source.update(kind='golden',split='split',legacy_split_manifest='s3://src/golden/split.json')
    add(env,'b','train');add(env,'c','train')
    env.objects[('src','golden/split.json')]=json.dumps({'source':'s3://src/data/images/train','train':['a'],'val':['b']}).encode()
    soft.sync(cfg)
    assert {m['source_uri'].rsplit('/',1)[-1] for m in soft.members(cfg) if m['split']=='val'}=={'b.jpg'}
    del env.objects[('src','data/labels/train/b.txt')]
    with pytest.raises(ValueError,match='frozen validation'):soft.sync(cfg)
    assert len(soft.members(cfg))==3

def test_split_stays_frozen_and_new_golden_pairs_train(env):
    cfg=config(soft.create());cfg["data_selection"]["sources"]=cfg["data_selection"]["sources"][:1]
    cfg["data_selection"]["sources"][0].update(split="split",kind="golden",train_percent=50)
    add(env,"b","train");soft.sync(cfg)
    initial={m["source_uri"]:m["split"] for m in soft.members(cfg)}
    assert set(initial.values())=={"train","val"}
    add(env,"c","train");soft.sync(cfg)
    assert all(m["split"]==initial.get(m["source_uri"],"train") for m in soft.members(cfg))
    cfg["data_selection"]["sources"][0]["train_percent"]=80
    with pytest.raises(ValueError,match="already frozen"):soft.sync(cfg)

def test_source_original_yaml_wins_over_output_yaml(env):
    cfg=config(soft.create());env.objects[("src","data/data.source.yaml")]=env.objects[("src","data/data.yaml")]
    env.objects[("src","data/data.yaml")]=b"names: {0: person, 1: helmet}"
    assert soft.discover(cfg)[0]["classes"]=={"0":"helmet","2":"person"}

def test_cancel_preserves_previous_members(env):
    cfg=config(soft.create());soft.sync(cfg)
    def stop(*a):raise soft.SelectionStopped("stop")
    with pytest.raises(soft.SelectionStopped):soft.sync(cfg,progress=stop)
    assert len(soft.members(cfg))==2

@pytest.mark.parametrize("text",["7 0.5 0.5 0.2 0.2","0 nan 0.5 0.2 0.2","0 0.5 0.5 0 0.2","0 1 2"])
def test_bad_label(text):
    with pytest.raises(ValueError):soft.validate_label(text,{"0":"helmet"})

def test_golden_migration_uses_original_not_blank_retention_root():
    cfg={"retention":{"use_golden":True,"golden_root":""},"golden_set":{"source":"raw/images","labels_source":"raw/labels","target_folder":"unused","train_percent":50}}
    value,notes=definitions.migrate(cfg)
    assert value["sources"][0]["images"]=="raw/images"
    assert value["sources"][0]["split"]=="split"
    assert cfg["retention"]["golden_root"]=="" and notes

def test_snapshot_reuses_catalog_version_and_freezes_labels(env,monkeypatch):
    from ppe_pipeline_api.data import dataset_yaml,dataset_versions
    cfg=config(soft.create());soft.sync(cfg)
    cfg["data_selection"]["class_ids"]={0:"person",1:"helmet"}
    monkeypatch.setattr(dataset_yaml,"write_in_place",lambda **kw:{"text":"yaml"})
    calls=[]
    def freeze(**kw):
        assert kw["preserve"] is False
        calls.append(kw)
        return {"version_id":"dv-test","folder":"_datasets/dv-test","train_images":1,"val_images":1}
    monkeypatch.setattr(dataset_versions,"create",freeze)
    built=soft.materialize(cfg)
    assert soft.materialize(cfg)["id"]==built["id"] and len(calls)==1
    labels={k:v for k,v in env.objects.items() if "training-cache" in k[1] and "/labels/" in k[1]}
    assert all(v.startswith(b"0 ") for v in labels.values())
    add(env,"a","train",label="0 0.5 0.5 0.1 0.1\n");soft.sync(cfg)
    assert all(env.objects[k]==v for k,v in labels.items())
    c=db.connection();cur=c.cursor();cur.execute("SELECT count(*) FROM dataset_members WHERE version_id='dv-test'");assert cur.fetchone()[0]==2;c.close()
    actual=soft.training_config(cfg,built)
    assert actual["_selection_version"]["version_id"]=="dv-test" and actual["data_source"]=="snapshot"

def test_filter_to_model_is_preserved(env):
    cfg=config(soft.create());soft.sync(cfg)
    cfg["data_selection"].update(filter_to_model=True,model_classes={0:"helmet"})
    assert soft.output_classes(cfg,soft.members(cfg))=={0:"helmet"}

def test_save_as_clones_members_and_keeps_definition_schedule(env,monkeypatch,tmp_path):
    cfg=config(soft.create());soft.sync(cfg)
    cfg["data_selection"]["schedule"].update(mode="schedule",at="00:00")
    monkeypatch.setattr(pc,"SETTINGS_DIR",str(tmp_path));monkeypatch.setattr(pc,"write_config",lambda p:p)
    result=pc.save_settings_file(name="copy",config=cfg,scope="dp")
    saved=json.loads(__import__("pathlib").Path(result["path"]).read_text())["config"]
    assert saved["soft_selection"]["id"]!=cfg["soft_selection"]["id"]
    assert len(soft.members(saved))==2
    assert db.state(saved["soft_selection"]["id"])["config"]["data_selection"]["schedule"]["at"]=="00:00"

def test_jobs_are_durable_and_one_active_job_per_selection(env,monkeypatch):
    cfg=config(soft.create());monkeypatch.setattr(jobs,"launch",lambda jid:None)
    first=jobs.start(cfg,"scan");second=jobs.start(cfg,"refresh")
    assert first["job_id"]==second["job_id"]
    jobs.work(first["job_id"])
    assert jobs.history(cfg["soft_selection"]["id"])[0]["status"]=="complete"

def test_schedule_time_is_singapore():
    now=dt.datetime(2026,11,10,12,tzinfo=dt.timezone.utc)
    assert jobs.next_time({"mode":"schedule","every_days":1,"at":"00:00","from_date":""},now)==dt.datetime(2026,11,10,16,tzinfo=dt.timezone.utc)
