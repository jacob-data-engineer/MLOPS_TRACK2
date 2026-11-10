"""One Data Preparation definition: sources, policy, preview, versions and jobs."""
from __future__ import annotations

from collections import Counter, defaultdict

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..auth import require_session
from .. import pull_control as pc, training
from ..data import soft_selection as soft, selection_catalog as db
from ..data import selection_definition as definition, selection_jobs as jobs

router=APIRouter()


def enrich(value):
    value=definition.Definition.model_validate(value).model_dump()
    if value["detector_slug"]:
        detector=training.get_detector(value["detector_slug"])
        value["model_classes"]={int(k):str(v) for k,v in (detector.get("classes") or {}).items()}
    elif value["filter_to_model"]:
        raise ValueError("Choose a model name before enabling the model-class filter.")
    return value


def current():
    config=pc.read_config()
    proposal,notes=definition.migrate(config)
    try:proposal=enrich(proposal)
    except Exception as exc:notes.append(str(exc))
    return config,proposal,notes


@router.get("")
def status(session=Depends(require_session)):
    try:
        config,proposal,notes=current()
        sid=(config.get("soft_selection") or {}).get("id")
        state=db.state(sid) if sid else {}
        selection=soft.select(config) if sid else []
        groups={};coverage=defaultdict(lambda:Counter(available=0,selected=0))
        reasons=Counter()
        for row in selection:
            group=groups.setdefault(row["source_id"],dict(id=row["source_id"],available=0,selected=0,train=0,val=0,classes=row["classes"]))
            group["available"]+=1
            if row["included"]:
                group["selected"]+=1;group[row["split"]]+=1
            else:reasons[row["reason"]]+=1
            labels={row["classes"][str(int(line.split()[0]))] for line in row["yolo"].splitlines() if line.strip()}
            for label in labels:
                coverage[(row["source_id"],label)]["available"]+=1
                coverage[(row["source_id"],label)]["selected"]+=int(row["included"])
        threshold=int((config.get("retention") or {}).get("min_class_images") or 5)
        counts={split:sum(m["included"] and m["split"]==split for m in selection) for split in ("train","val")}
        cached=state.get("last_result") or {}
        return {"definition":proposal,"migration_required":not bool(config.get("data_selection")),"migration_notes":notes,
                "selection_id":sid or "","retention":{**soft.retention.DEFAULTS,**(config.get("retention") or {})},
                "excluded_weeks":(config.get("auto_training") or {}).get("excluded") or [],
                "weeks":sorted({m["week"] for m in selection if m["week"]!="fixed"}),
                "counts":{**counts,"available":len(selection),"selected":sum(m["included"] for m in selection),"excluded":dict(reasons)},
                "collections":[{"week":w,"available":sum(m["week"]==w for m in selection),"selected":sum(m["week"]==w and m["included"] for m in selection)} for w in sorted({m["week"] for m in selection})],
                "sources":list(groups.values()),"coverage":[{"source_id":sid_,"class_name":name,**counts_,"warning":counts_["selected"]<threshold} for (sid_,name),counts_ in sorted(coverage.items())],
                "last_refresh":cached,"jobs":jobs.history(sid) if sid else [],"versions":db.versions(sid) if sid else [],
                "saved_name":state.get("name") or "","next_at":state.get("next_at"),
                "schedule_saved":(state.get("config") or {}).get("data_selection",{}).get("schedule")==proposal["schedule"],
                "sources_stale":cached.get("source_fingerprint")!=soft.source_fingerprint({**config,"data_selection":proposal})}
    except Exception as exc:
        raise HTTPException(status_code=400,detail=str(exc))


@router.post("/definition")
def save_definition(payload:definition.Definition,session=Depends(require_session)):
    try:
        value=enrich(payload.model_dump())
        config=pc.read_config()
        sid=(config.get("soft_selection") or {}).get("id") or soft.create()
        pc.write_config({"data_selection":value,"soft_selection":{"enabled":True,"id":sid}})
        return {"definition":value,"selection_id":sid}
    except Exception as exc:
        raise HTTPException(status_code=400,detail=str(exc))


class JobRequest(BaseModel):
    kind:str


@router.post("/jobs")
def run_job(payload:JobRequest,session=Depends(require_session)):
    try:
        config=pc.read_config()
        if not config.get("data_selection"):
            raise ValueError("Apply the migrated definition before scanning or indexing it.")
        return jobs.start(config,payload.kind)
    except Exception as exc:
        raise HTTPException(status_code=400,detail=str(exc))


@router.post("/new-selection")
def new_selection(session=Depends(require_session)):
    """A fresh split policy is deliberate; no membership is moved between splits."""
    sid=soft.create()
    pc.write_config({"soft_selection":{"enabled":True,"id":sid}})
    return {"selection_id":sid}


@router.get("/members")
def preview_members(page:int=1,session=Depends(require_session)):
    from .. import storage
    config=pc.read_config()
    selection=soft.select(config) if (config.get("soft_selection") or {}).get("id") else []
    page=max(1,page);batch=selection[(page-1)*24:page*24];result=[]
    for row in batch:
        bucket,key=soft.split_uri(row["stored_uri"],"")
        boxes=[]
        for line in row["yolo"].splitlines():
            if line.strip():
                cid,x,y,w,h=line.split()
                boxes.append({"name":row["classes"][str(int(cid))],"id":int(cid),"x":float(x),"y":float(y),"w":float(w),"h":float(h)})
        result.append({"source_uri":row["source_uri"],"split":row["split"],"included":row["included"],"reason":row["reason"],"boxes":boxes,
            "url":storage.s3.generate_presigned_url("get_object",Params={"Bucket":bucket,"Key":key},ExpiresIn=600)})
    return {"items":result,"page":page,"total":len(selection),"pages":max(1,(len(selection)+23)//24)}
