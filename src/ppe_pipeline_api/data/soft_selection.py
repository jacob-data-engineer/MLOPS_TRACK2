"""Database-owned selections; source files never moved or deleted."""
from __future__ import annotations
import copy
import datetime as dt
import hashlib
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from . import catalog, central_store, retention
from .. import pull_control as pc, storage

from . import selection_catalog as db, selection_definition as definitions

def enabled(config):
    return bool((config.get("soft_selection") or {}).get("enabled"))

def ident(config):
    value = str((config.get("soft_selection") or {}).get("id") or "")
    if not value or not all(c.isalnum() or c in "-_" for c in value):
        raise ValueError("Create a database selection on Data Preparation first.")
    return value

connection = db.connection
rows = db.rows
create = db.create


def split_uri(uri, bucket):
    if uri.startswith("s3://"):
        b, _, k = uri[5:].partition("/")
        return b, k.strip("/")
    return bucket, uri.strip("/")

def sources(config):
    out=[]
    for source in definitions.effective(config)["sources"]:
        if not source["enabled"]: continue
        bucket=source["bucket"] or storage.BUCKET
        b,k=split_uri(source["images"],bucket)
        lb,lk=split_uri(source["labels"],b)
        out.append({**source,"bucket":b,"images":k,"label_bucket":lb,"labels":lk,
                    "decay":source["kind"]=="collected"})
    if not out: raise ValueError("Enable at least one data source.")
    return out


def taxonomy(source, config):
    """Read source numbering, never infer it from a desired output mapping."""
    import yaml
    if source.get("classes"):
        return {str(k):v for k,v in source["classes"].items()}
    b=source["label_bucket"]; key=source["labels"]
    for _ in range(12):
        key=key.rsplit("/",1)[0] if "/" in key else ""
        for filename in ("data.source.yaml", "data.yaml"):
            path=(key+"/" if key else "")+filename
            try:
                doc=yaml.safe_load(storage.s3.get_object(Bucket=b,Key=path)["Body"].read()) or {}
            except Exception as exc:
                code=getattr(exc,"response",{}).get("Error",{}).get("Code")
                if code not in ("404","NoSuchKey","NotFound"): raise
                continue
            names=doc.get("names")
            if isinstance(names,list): names=dict(enumerate(names))
            if isinstance(names,dict) and names:
                result={str(int(i)):str(n) for i,n in names.items()}
                if any(int(i)<0 or not n.strip() for i,n in result.items()) or len(set(result.values())) != len(result):
                    raise ValueError("Invalid or ambiguous source class mapping: "+path)
                return result
        if not key: break
    raise ValueError("No source data.yaml class mapping for "+source["labels"]+". Supply the source label numbering before indexing.")


def validate_label(text, names):
    for line in text.splitlines():
        if not line.strip(): continue
        fields=line.split()
        if len(fields)!=5: raise ValueError("Expected YOLO detection labels with five columns.")
        cid=int(fields[0]); values=list(map(float,fields[1:]))
        if str(cid) not in names: raise ValueError("Label class ID "+str(cid)+" is absent from data.yaml.")
        if any(not 0 <= v <= 1 for v in values) or values[2]<=0 or values[3]<=0:
            raise ValueError("Invalid normalized bounding box in label file.")

class SelectionStopped(RuntimeError):
    pass


def discover(config, progress=None, report=None):
    report=report if report is not None else {}
    report.update(sources=[],excluded=[],blockers=[])
    found={}
    for source in sources(config):
        summary={"id":source["id"],"name":source["name"],"images":0,"eligible":0,"excluded":0,"classes":{}}
        report["sources"].append(summary)
        try:
            names=taxonomy(source,config)
            summary["classes"]=names
            legacy = None
            if source.get("legacy_split_manifest"):
                mb,mk=split_uri(source["legacy_split_manifest"],source["bucket"])
                try:
                    legacy=json.loads(storage.s3.get_object(Bucket=mb,Key=mk)["Body"].read())
                except Exception as exc:
                    if getattr(exc,"response",{}).get("Error",{}).get("Code") not in ("404","NoSuchKey","NotFound"):raise
                if legacy is not None:
                    if not all(isinstance(legacy.get(side),list) for side in ("train","val")) or set(legacy["train"]) & set(legacy["val"]):
                        raise ValueError("Invalid existing golden split manifest; repair it before migration.")
                    expected="s3://"+source["bucket"]+"/"+source["images"]
                    if str(legacy.get("source","")).rstrip("/")!=expected.rstrip("/"):
                        raise ValueError("Golden split manifest points to a different source; review the original source before migration.")
            objects=list(pc._list_objects(source["images"].rstrip("/")+"/",source["bucket"]))
        except Exception as exc:
            report["blockers"].append(source["name"]+": "+str(exc));continue
        prefix=source["images"].rstrip("/")+"/"
        for obj in objects:
            key=obj["Key"]
            if not key.lower().endswith(tuple(storage.IMAGE_EXTENSIONS)):continue
            summary["images"]+=1
            uri="s3://"+source["bucket"]+"/"+key
            rel=key[len(prefix):]
            label_key=source["labels"].rstrip("/")+"/"+rel.rsplit(".",1)[0]+".txt"
            label_uri="s3://"+source["label_bucket"]+"/"+label_key
            stem=rel.rsplit(".",1)[0]
            try:
                label_response=storage.s3.get_object(Bucket=source["label_bucket"],Key=label_key)
                body=label_response["Body"].read().decode("utf-8")
                if legacy and stem in legacy["val"]:
                    frozen=(legacy.get("validation_objects") or {}).get(stem,{})
                    for kind,etag in (("image",obj.get("ETag","")),("label",label_response.get("ETag",""))):
                        before=(frozen.get(kind) or {}).get("etag","")
                        if before and before.strip('"')!=etag.strip('"'):
                            report["blockers"].append("Frozen golden validation "+kind+" changed: "+uri)
                validate_label(body,names)
            except Exception as exc:
                summary["excluded"]+=1
                report["excluded"].append({"source_uri":uri,"label_uri":label_uri,"reason":str(exc)})
                continue
            item=dict(source_uri=uri,source_id=source["id"],kind=source["kind"],split=source["split"],
                      percent=source["train_percent"],week="fixed" if not source["decay"] else "",
                      etag=obj["ETag"].strip('"'),bytes=obj.get("Size",0),label_uri=label_uri,yolo=body,classes=names)
            if legacy is not None:
                item["legacy_split"]="val" if stem in legacy["val"] else "train"
            if uri in found:
                previous=found[uri]
                if previous["split"]!=item["split"] or previous["yolo"]!=body or previous["classes"]!=names:
                    report["blockers"].append("Conflicting split or labels for "+uri)
                elif item["kind"]=="golden":found[uri]=item
            else:found[uri]=item
            summary["eligible"]+=1
            if progress and sum(r["images"] for r in report["sources"])%25==0:progress(len(found),0)
        if legacy is not None:
            present={m["source_uri"][len("s3://"+source["bucket"]+"/"+prefix):].rsplit(".",1)[0] for m in found.values() if m["source_id"]==source["id"]}
            if set(legacy["val"])-present:report["blockers"].append(source["name"]+": frozen validation images or labels are missing.")
        if not summary["images"]:
            report["blockers"].append(source["name"]+": no images found at s3://"+source["bucket"]+"/"+source["images"])
    return list(found.values())


def assign_splits(items, previous, policies):
    """Freeze validation membership; subsequent arrivals are training only."""
    grouped={}
    for item in items: grouped.setdefault(item["source_id"],[]).append(item)
    for source_id, group in grouped.items():
        policy=json.dumps([group[0]["split"],group[0]["percent"]])
        if source_id in policies and policies[source_id]!=policy:
            raise ValueError("A source split is already frozen. Create a new independent definition before changing its split policy.")
        if group[0]["split"]=="split":
            ordered=sorted(group,key=lambda m:hashlib.sha256(m["source_uri"].encode()).hexdigest())
            n=max(1,min(len(group)-1,round(len(group)*group[0]["percent"]/100))) if len(group)>1 else len(group)
            initial_train={m["source_uri"] for m in ordered[:n]}
            for item in group:
                old=previous.get(item["source_uri"])
                item["split"]=old["split"] if old else item.get("legacy_split", "train" if source_id in policies or item["source_uri"] in initial_train else "val")
        else:
            for item in group:
                old=previous.get(item["source_uri"])
                if old and old["split"]!=item["split"]:
                    raise ValueError("An indexed image cannot change split within the same definition: "+item["source_uri"])
        policies[source_id]=policy


def scan(config):
    report={}
    items=discover(config,report=report)
    conn=connection()
    try:
        cur=conn.cursor();previous={m["source_uri"]:m for m in db.member_rows(cur,ident(config),False)}
        cur.execute("SELECT source_id,policy FROM dataset_selection_splits WHERE selection_id=%s",(ident(config),))
        policies=dict(cur.fetchall())
        try:assign_splits(items,previous,policies)
        except ValueError as exc:report["blockers"].append(str(exc))
    finally:conn.close()
    report.update(eligible=len(items),images=sum(r["images"] for r in report["sources"]),
                  train=sum(m["split"]=="train" for m in items),val=sum(m["split"]=="val" for m in items),checked_at=pc._now().isoformat())
    if not items:report["blockers"].append("No eligible image/label pairs were found.")
    report["ready"]=not report["blockers"]
    return report


def source_fingerprint(config):
    return hashlib.sha256(json.dumps(definitions.effective(config)["sources"],sort_keys=True).encode()).hexdigest()


def sync(config, progress=None, **unused):
    sid=ident(config);conn=connection();report={}
    try:
        c=conn.cursor();c.execute("SELECT pg_advisory_xact_lock(hashtext(%s))",(sid,));db.ensure(c,sid)
        if progress:progress(0,0)
        items=discover(config,progress=progress,report=report)
        if report["blockers"]:raise ValueError("; ".join(report["blockers"]))
        if not items:raise ValueError("No eligible paired images; previous selection was retained.")
        previous={m["source_uri"]:m for m in db.member_rows(c,sid,False)}
        c.execute("SELECT source_id,policy FROM dataset_selection_splits WHERE selection_id=%s",(sid,))
        policies=dict(c.fetchall());assign_splits(items,previous,policies)
        c.execute("SELECT collection FROM dataset_selections WHERE id=%s",(sid,));batch=int(c.fetchone()[0])+1
        store_bucket,store_folder=central_store.location()
        def preserve(item):
            old=previous.get(item["source_uri"])
            if old and old["source_etag"]==item["etag"]:
                return {**item,"asset_id":old["asset_id"],"stored_uri":old["stored_uri"]}
            b,k=split_uri(item["source_uri"],"")
            # A real content hash lets different sources share the same asset.
            body=storage.s3.get_object(Bucket=b,Key=k,IfMatch=item["etag"])["Body"]
            digest=hashlib.sha256()
            try:
                for chunk in iter(lambda:body.read(1024*1024),b""):digest.update(chunk)
            finally:body.close()
            aid=digest.hexdigest();ext="."+k.rsplit(".",1)[-1].lower()
            target=store_folder+"/objects/"+aid+ext
            if central_store._etag(store_bucket,target) is None:
                storage.s3.copy_object(Bucket=store_bucket,Key=target,CopySource={"Bucket":b,"Key":k},CopySourceIfMatch=item["etag"])
            return {**item,"asset_id":aid,"stored_uri":"s3://"+store_bucket+"/"+target}
        kept=[]
        with ThreadPoolExecutor(max_workers=8) as pool:
            for item in pool.map(preserve,items):
                kept.append(item)
                if progress and len(kept)%25==0:progress(len(kept),len(items))
        items=kept
        content_splits={}
        for item in items:
            if item["asset_id"] in content_splits and content_splits[item["asset_id"]]!=item["split"]:
                raise ValueError("Identical image content is in both training and validation: "+item["source_uri"])
            content_splits[item["asset_id"]]=item["split"]
        if progress:progress(len(items),len(items))
        c.execute("UPDATE dataset_selection_members SET active=false,reason='source no longer selected' WHERE selection_id=%s AND reason <> 'week removed'",(sid,))
        changed=0
        for item in sorted(items,key=lambda i:i["asset_id"]):
            old=previous.get(item["source_uri"])
            if old and old["reason"]=="week removed":continue
            b,k=split_uri(item["stored_uri"],"")
            c.execute("INSERT INTO files(sha256,store_bucket,store_key,bytes) VALUES(%s,%s,%s,%s) ON CONFLICT DO NOTHING",(item["asset_id"],b,k,item["bytes"]))
            c.execute("INSERT INTO file_sources(sha256,source_uri,source_kind,detail) VALUES(%s,%s,'import',%s::jsonb) ON CONFLICT DO NOTHING",(item["asset_id"],item["source_uri"],json.dumps({"etag":item["etag"]})))
            lid=db.annotation(c,item)
            week="fixed" if item["kind"]!="collected" else (old["week"] if old and old["week"]!="fixed" else "week_"+str(batch))
            changed+=not old or not old["active"] or (old["asset_id"],old["label_id"],old["kind"])!=(item["asset_id"],lid,item["kind"])
            c.execute("""INSERT INTO dataset_selection_members(selection_id,source_uri,sha256,label_id,source_id,kind,source_etag,split,week)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(selection_id,source_uri) DO UPDATE SET
                sha256=excluded.sha256,label_id=excluded.label_id,source_id=excluded.source_id,kind=excluded.kind,
                source_etag=excluded.source_etag,split=excluded.split,week=excluded.week,active=true,reason=''""",
                (sid,item["source_uri"],item["asset_id"],lid,item["source_id"],item["kind"],item["etag"],item["split"],week))
        for source_id,policy in policies.items():
            c.execute("INSERT INTO dataset_selection_splits VALUES(%s,%s,%s) ON CONFLICT DO NOTHING",(sid,source_id,policy))
        removed=len({u for u,m in previous.items() if m["active"]}-{m["source_uri"] for m in items})
        selected=len([i for i in items if previous.get(i["source_uri"],{}).get("reason")!="week removed"])
        report.update(status="complete",source_fingerprint=source_fingerprint(config),selection_id=sid,moved=changed+removed,selected=selected,skipped=selected-changed,
                      errors=0,mode="database",removed=removed,checked_at=pc._now().isoformat())
        c.execute("UPDATE dataset_selections SET updated_at=now(),collection=%s,last_result=%s::jsonb WHERE id=%s",
                  (batch if any(i["kind"]=="collected" and i["source_uri"] not in previous for i in items) else batch-1,json.dumps(report,default=str),sid))
        conn.commit();return report
    except Exception as exc:
        conn.rollback()
        db.result(sid,{**report,"status":"failed","error":str(exc),"checked_at":pc._now().isoformat()})
        raise
    finally:conn.close()


def members(config):
    conn=connection()
    try:return db.member_rows(conn.cursor(),ident(config))
    finally:conn.close()


def select(config, all_members=None):
    all_members=members(config) if all_members is None else all_members
    inventory={}
    for m in all_members:
        if m["week"]=="fixed": continue
        row=inventory.setdefault(m["week"],{"name":m["week"],"train":{},"val":{}})
        row[m["split"]][m["asset_id"]]=m["source_uri"]
    ordered=sorted(inventory.values(),key=lambda w:int(''.join(c for c in w["name"] if c.isdigit()) or 0))
    plan=retention.plan_inventory(ordered,config.get("retention") or {})
    keep={r["week"]:set(r.get("stems",[])) for r in plan["weeks"]}
    excluded=set((config.get("auto_training") or {}).get("excluded") or [])
    selected=[]
    seen={}
    for m in all_members:
        included=m["week"]=="fixed" or (m["week"] not in excluded and (m["split"]=="val" or m["asset_id"] in keep.get(m["week"],set())))
        reason="" if included else ("week excluded" if m["week"] in excluded else "retention")
        if m["asset_id"] in seen:
            other=seen[m["asset_id"]]
            if (m["split"],m["yolo"],m["classes"])!=(other["split"],other["yolo"],other["classes"]):
                raise ValueError("Conflicting annotations or splits for identical image content: "+m["source_uri"])
            included=False;reason="duplicate content"
        elif included:seen[m["asset_id"]]=m
        selected.append({**m,"included":included,"reason":reason})
    return selected

def status(config):
    if not (config.get("soft_selection") or {}).get("id"):
        return {"enabled":False,"id":"","available":0,"selected":0,"train":0,"val":0,"weeks":[]}
    selection=select(config)
    return {"enabled":enabled(config),"id":ident(config),"available":len(selection),
            "selected":sum(m["included"] for m in selection),
            "train":sum(m["included"] and m["split"]=="train" for m in selection),
            "val":sum(m["included"] and m["split"]=="val" for m in selection),
            "excluded":sum(not m["included"] for m in selection),
            "weeks":sorted({m["week"] for m in selection}),"retention":config.get("retention") or {},
            "excluded_weeks":(config.get("auto_training") or {}).get("excluded") or []}

def output_classes(config, selected):
    definition=definitions.effective(config)
    classes={int(k):str(v) for k,v in (definition["class_ids"] or (selected[0]["classes"] if selected else {})).items()}
    if definition["filter_to_model"]:
        wanted=set(definition["model_classes"].values())
        if not wanted:raise ValueError("Choose a model with declared classes before filtering labels to that model.")
        classes={k:v for k,v in classes.items() if v in wanted}
        missing=wanted-set(classes.values())
        if missing:raise ValueError("Output mapping is missing model classes: "+", ".join(sorted(missing)))
    else:
        for name in sorted({str(v) for m in selected for v in m["classes"].values()}):
            if name not in classes.values():
                index=0
                while index in classes:index+=1
                classes[index]=name
    if not classes:raise ValueError("No class mapping is available.")
    for index in range(max(classes)+1):
        if index not in classes:
            placeholder="unused_"+str(index)
            while placeholder in classes.values():placeholder="_"+placeholder
            classes[index]=placeholder
    return classes


def yaml_preview(config):
    """Describe the next frozen dataset without writing source or snapshot files."""
    import yaml
    selected=[m for m in select(config) if m["included"]]
    if not selected:
        raise ValueError("Refresh database selection before previewing the training dataset.")
    classes=output_classes(config,selected)
    b,base=central_store.location()
    folder=base+"/training-cache/<content-hash>"
    uri="s3://"+b+"/"+folder
    return {"folder":folder,"key":folder+"/data.yaml","uri":uri+"/data.yaml","bucket":b,
        "download_prefix":uri+"/","names":classes,"train":["images/train"],"val":["images/val"],
        "text":yaml.safe_dump({"path":".","train":"images/train","val":"images/val","names":classes},sort_keys=False),
        "download":{"in_named_folders":len(selected)*2,"in_named_bytes":sum(m["bytes"] for m in selected),"extra_objects":0,"extra_bytes":0},
        "existing":None,"eval":None,"many_roots":True,"needs_overwrite":False,"undeclared":[],"drop_undeclared":False,
        "no_val":not any(m["split"]=="val" for m in selected),
        "image_counts":{split:{"total":sum(m["split"]==split for m in selected),"folders":[{"folder":"SQL selection: "+ident(config),"images":sum(m["split"]==split for m in selected)}]} for split in ("train","val")}}


def materialize(config):
    """Prepare a cached YOLO bundle and publish one canonical catalog version."""
    from . import dataset_yaml, dataset_versions
    if db.state(ident(config)).get("last_result",{}).get("source_fingerprint")!=source_fingerprint(config):
        raise ValueError("Sources changed or have not been indexed. Refresh the selection before freezing it.")
    all_members=select(config)
    selected=[m for m in all_members if m["included"]]
    if not any(m["split"]=="train" for m in selected) or not any(m["split"]=="val" for m in selected):
        raise ValueError("Selection needs nonempty, separate train and validation splits.")
    classes=output_classes(config,selected);target={name:i for i,name in classes.items()}
    digest=hashlib.sha256(json.dumps([[(m["source_uri"],m["asset_id"],m["annotation_id"],m["split"],m["included"],m["reason"]) for m in all_members],classes],sort_keys=True).encode()).hexdigest()
    sid=ident(config);conn=connection()
    try:
        c=conn.cursor();c.execute("SELECT pg_advisory_xact_lock(hashtext(%s))",("freeze-"+sid,))
        c.execute("SELECT record FROM dataset_selection_versions WHERE selection_id=%s AND content_key=%s",(sid,digest))
        row=c.fetchone()
        if row:return row[0]
        b,base=central_store.location();folder=base+"/training-cache/"+digest
        def one(m):
            sb,sk=split_uri(m["stored_uri"],"");name=m["asset_id"]+"."+sk.rsplit(".",1)[-1]
            destination=folder+"/images/"+m["split"]+"/"+name
            if central_store._etag(b,destination) is None:
                storage.s3.copy_object(Bucket=b,Key=destination,CopySource={"Bucket":sb,"Key":sk})
            lines=[]
            for line in m["yolo"].splitlines():
                if line.strip():
                    cid,coords=line.split(maxsplit=1);label=m["classes"][str(int(cid))]
                    if label in target:lines.append(str(target[label])+" "+coords)
            lk=folder+"/labels/"+m["split"]+"/"+m["asset_id"]+".txt"
            storage.s3.put_object(Bucket=b,Key=lk,Body=("\n".join(lines)+"\n" if lines else "").encode())
            return m["source_uri"],"s3://"+b+"/"+destination,"s3://"+b+"/"+lk
        with ThreadPoolExecutor(max_workers=8) as pool:prepared={r[0]:r[1:] for r in pool.map(one,selected)}
        yaml=dataset_yaml.write_in_place(splits={"train":[(folder+"/images/train",folder+"/labels/train")],
             "val":[(folder+"/images/val",folder+"/labels/val")]},classes=classes,bucket=b,label_scan=None,overwrite=True)
        version=dataset_versions.create(home=b,yaml_text=yaml["text"],prefixes=["s3://"+b+"/"+folder+"/"],
            sources=["SQL selection "+sid],trigger="data-preparation",note="Frozen dataset definition",preserve=False)
        vid=version["version_id"]
        # Use the SAME version and members that training and Data Catalog read.
        c.execute("""INSERT INTO dataset_versions(id,purpose,class_names,settings,kind,state,train_images,val_images,record,members_synced)
            VALUES(%s,'training',%s::jsonb,%s::jsonb,'version','ready',%s,%s,%s::jsonb,true)
            ON CONFLICT(id) DO UPDATE SET settings=excluded.settings,members_synced=true""",
            (vid,json.dumps(classes),json.dumps(config,default=str),version["train_images"],version["val_images"],json.dumps(version)))
        for m in all_members:
            image,label=prepared.get(m["source_uri"],(m["stored_uri"],m["label_uri"]))
            c.execute("""INSERT INTO dataset_members(version_id,sha256,label_id,split,included,reason,week,image_uri,label_uri,stored_image_uri,stored_label_uri,image_etag)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(version_id,image_uri) DO NOTHING""",
                (vid,m["asset_id"],m["label_id"],m["split"],m["included"],m["reason"],m["week"],image,label,
                 image if m["included"] else m["stored_uri"],label if m["included"] else None,m["source_etag"]))
        result={"id":vid,"bucket":b,"folder":folder,"classes":classes,"train":version["train_images"],"val":version["val_images"],"version":version}
        c.execute("INSERT INTO dataset_selection_versions(selection_id,version_id,content_key,record) VALUES(%s,%s,%s,%s::jsonb)",(sid,vid,digest,json.dumps(result)))
        conn.commit();return result
    finally:conn.close()


def training_config(config, built):
    result=copy.deepcopy(config)
    result["soft_selection"]["enabled"]=False
    result["data_source"]="assembled"
    result["retention"]={**result.get("retention",{}),"enabled":False,"use_golden":False}
    result["golden_set"]={}
    old=result.get("dataset_assemble") or {}
    uri="s3://"+built["bucket"]+"/"+built["folder"]
    result["dataset_assemble"]={**old,"use":True,"only":True,"bucket":built["bucket"],
        "label_classes":built["classes"],"detector_slug":definitions.effective(config).get("detector_slug") or old.get("detector_slug") or (config.get("default_detector") or {}).get("slug", ""), "drop_undeclared_labels":False,
        "train_sources":[{"images":uri+"/images/train","labels":uri+"/labels/train"}],
        "eval_sources":[{"images":uri+"/images/val","labels":uri+"/labels/val"}]}
    if built.get("version"):
        result["data_source"]="snapshot"
        result["default_dataset"]={"folder":"s3://"+built["bucket"]+"/"+built["folder"],"snapshot_id":built["id"],"files":built["train"]+built["val"]}
        result["_selection_version"]=built["version"]
    return result

def exclude_week(config, name):
    conn=connection()
    try:
        c=conn.cursor()
        c.execute("UPDATE dataset_selection_members SET active=false,reason='week removed' WHERE selection_id=%s AND week=%s",(ident(config),name))
        count=c.rowcount
        conn.commit()
        return {"week":name,"removed_members":count,"deleted":0,"mode":"database"}
    finally: conn.close()

def preview_config(config):
    available=members(config)
    names=sorted({str(v) for m in available for v in m["classes"].values()})
    # Before the first refresh, taxonomy can be resolved from source declarations.
    if not names:
        names=sorted({str(v) for source in sources(config) for v in taxonomy(source,config).values()})
    return training_config(config,{"bucket":storage.BUCKET,"folder":"soft-preview/"+ident(config),"classes":dict(enumerate(names))})


def week_contents(config):
    available=members(config)
    weeks={}
    for m in available:
        row=weeks.setdefault(m["week"],{"week":m["week"],"images":0,"labels":0,"bytes":0,"roots":[],"defined":True})
        row["images"]+=1; row["labels"]+=1; row["bytes"]+=m["bytes"]
    return {"weeks":sorted(weeks.values(),key=lambda r:r["week"]),"roots":[],"mode":"database"}

def inventory(config):
    grouped={}
    for m in members(config):
        if m["week"]=="fixed": continue
        row=grouped.setdefault(m["week"],{"name":m["week"],"train":{},"val":{}})
        row[m["split"]][m["asset_id"]]=m["stored_uri"]
    return sorted(grouped.values(),key=lambda r:int(''.join(c for c in r["name"] if c.isdigit()) or 0))
