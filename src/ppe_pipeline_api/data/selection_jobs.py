"""Durable Data Preparation jobs and per-saved-definition schedules on EC2."""
from __future__ import annotations

import datetime as dt
import json
import threading
import uuid

from . import selection_catalog as db, selection_definition as definitions
from .. import pull_control as pc, schedule_clock

_running = set()
_lock = threading.Lock()


def enqueue(cur, config, kind):
    from .soft_selection import ident
    sid=ident(config);db.ensure(cur,sid)
    cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))",("job-"+sid,))
    cur.execute("SELECT id FROM dataset_selection_jobs WHERE selection_id=%s AND status IN ('queued','running')",(sid,))
    existing=cur.fetchone()
    if existing:return existing[0]
    job_id="dp-"+uuid.uuid4().hex
    cur.execute("INSERT INTO dataset_selection_jobs(id,selection_id,kind,config) VALUES(%s,%s,%s,%s::jsonb)",
                (job_id,sid,kind,json.dumps(config,default=str)))
    return job_id


def start(config, kind):
    if kind not in ("scan","refresh","freeze"):
        raise ValueError("Unknown preparation job.")
    conn=db.connection()
    try:
        jid=enqueue(conn.cursor(),config,kind);conn.commit()
    finally:conn.close()
    launch(jid)
    return {"job_id":jid,"status":"queued"}


def launch(jid):
    with _lock:
        if jid in _running:return
        _running.add(jid)
    threading.Thread(target=work,args=(jid,),name=jid,daemon=True).start()


def work(jid):
    from . import soft_selection as soft
    conn=db.connection();stop=threading.Event()
    try:
        c=conn.cursor()
        c.execute("UPDATE dataset_selection_jobs SET status='running',updated_at=now() WHERE id=%s AND status='queued' RETURNING config,kind",(jid,))
        row=c.fetchone();conn.commit()
        if not row:return
        config,kind=row
        def heartbeat():
            while not stop.wait(20):
                try:update(jid)
                except Exception:pass  # The next heartbeat or terminal result retries.
        threading.Thread(target=heartbeat,daemon=True).start()
        progress=lambda done,total:update(jid,done=done,total=total)
        if kind=="scan":result=soft.scan(config)
        elif kind=="refresh":result=soft.sync(config,progress=progress)
        else:result=soft.materialize(config)
        stop.set()
        update(jid,status="complete",result=result)
    except Exception as exc:
        stop.set()
        update(jid,status="failed",error=str(exc))
    finally:
        stop.set();conn.close()
        with _lock:_running.discard(jid)


def update(jid, **fields):
    conn=db.connection()
    try:
        assignments=["updated_at=now()"]
        values=[]
        for key,value in fields.items():
            if key not in ("done","total","status","result","error"):raise ValueError(key)
            assignments.append(key+"=%s"+("::jsonb" if key=="result" else ""))
            values.append(json.dumps(value,default=str) if key=="result" else value)
        values.append(jid)
        conn.cursor().execute("UPDATE dataset_selection_jobs SET "+",".join(assignments)+" WHERE id=%s",tuple(values))
        conn.commit()
    finally:conn.close()


def history(sid):
    conn=db.connection()
    try:
        c=conn.cursor()
        c.execute("SELECT id,kind,status,created_at,updated_at,done,total,result,error FROM dataset_selection_jobs WHERE selection_id=%s ORDER BY created_at DESC LIMIT 10",(sid,))
        return db.rows(c)
    finally:conn.close()


def next_time(schedule, now):
    if schedule["mode"]=="off":return None
    if schedule["mode"]=="arrival":return now+dt.timedelta(minutes=1)
    plan={**schedule,"anchor":now.astimezone(schedule_clock.SGT).date().isoformat()}
    return schedule_clock.next_runs(plan,start=now,count=1)[0]


def save(config, name):
    """Only the main Save action arms the saved schedule, never draft edits."""
    from .soft_selection import ident
    definition=definitions.effective(config);sid=ident(config)
    conn=db.connection()
    try:
        cur=conn.cursor();db.ensure(cur,sid)
        cur.execute("SELECT config,next_at FROM dataset_selections WHERE id=%s FOR UPDATE",(sid,))
        old,upcoming=cur.fetchone()
        old_schedule=(old.get("data_selection") or {}).get("schedule")
        if old_schedule!=definition["schedule"] or not upcoming:
            upcoming=next_time(definition["schedule"],pc._now())
        if definition["schedule"]["mode"]=="off":upcoming=None
        saved={**config,"data_selection":definition}
        cur.execute("UPDATE dataset_selections SET name=%s,config=%s::jsonb,next_at=%s,updated_at=now() WHERE id=%s",
                    (name,json.dumps(saved,default=str),upcoming,sid))
        conn.commit()
    finally:conn.close()


def disable_file(name):
    conn=db.connection()
    try:
        conn.cursor().execute("UPDATE dataset_selections SET next_at=NULL,name='' WHERE name=%s",(name,));conn.commit()
    finally:conn.close()


def tick():
    """Queue due work atomically; recover interrupted jobs after three minutes."""
    conn=db.connection();queued=[]
    try:
        c=conn.cursor();now=pc._now()
        c.execute("SELECT pg_advisory_xact_lock(hashtext('dataset_selection_tick'))")
        c.execute("UPDATE dataset_selection_jobs SET status='queued',error='Resuming after worker interruption' WHERE status='running' AND updated_at < %s",(now-dt.timedelta(minutes=3),))
        c.execute("SELECT id,config,next_at FROM dataset_selections WHERE next_at<=%s AND name<>'' FOR UPDATE",(now,))
        for sid,config,previous in c.fetchall():
            queued.append(enqueue(c,config,"refresh"))
            schedule=definitions.effective(config)["schedule"]
            if schedule["mode"]=="schedule":
                interval=dt.timedelta(days=schedule["every_days"])
                following=previous+interval*((now-previous)//interval+1)
            else:following=next_time(schedule,now)
            c.execute("UPDATE dataset_selections SET next_at=%s WHERE id=%s",(following,sid))
        c.execute("SELECT id FROM dataset_selection_jobs WHERE status='queued' ORDER BY created_at LIMIT 4")
        queued.extend(row[0] for row in c.fetchall());conn.commit()
    finally:conn.close()
    for jid in set(queued):launch(jid)
