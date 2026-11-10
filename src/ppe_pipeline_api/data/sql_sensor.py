"""EC2-local SQL pull sensor. Uses the auto-training clock and manual pull path.

One sensor per app's persistent /settings mount. The app has one supported
worker; its RLock serializes schedule edits and run claims. Long checks run in
a separate thread, not on the shared training ticker. No training is launched.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from .. import pull_control
from ..chain import retrain_schedule as clock
from . import sql_explorer as sx
from .sql_settings import SqlSettings
from .sql_training_export import exists

_lock = threading.RLock()
_running = False
_recovered = False


class SensorConfig(BaseModel):
    enabled: bool = False
    every_days: int = Field(default=1, ge=1, le=365)
    at: str = "00:00"
    from_date: str = ""
    settings: SqlSettings = Field(default_factory=SqlSettings)

    @field_validator("at")
    @classmethod
    def time_valid(cls, value):
        if not clock._valid_time(value):
            raise ValueError("Use HH:MM in Singapore time.")
        return value

    @field_validator("from_date")
    @classmethod
    def date_valid(cls, value):
        if value:
            return dt.date.fromisoformat(value).isoformat()
        return ""


def now():
    return dt.datetime.now(dt.timezone.utc)


def path():
    return Path(pull_control.SETTINGS_DIR) / "sql-sensor" / "state.json"


def read():
    try:
        return json.loads(path().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"config": SensorConfig().model_dump(), "next_at": None, "history": [], "seen": []}


def write(state):
    target = path()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent, delete=False) as out:
            temporary = out.name
            json.dump(state, out)
        os.replace(temporary, target)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def status():
    with _lock:
        state = read()
        return {k: v for k, v in state.items() if k != "seen"} | {"running": _running, "timezone": "Asia/Singapore"}


def validate_selection(config):
    settings = config.settings
    if not settings.database or not settings.preset_class:
        raise ValueError("Choose a database and hazard before saving the sensor.")
    folder = settings.pull_folder.strip().strip("/")
    if not settings.pull_bucket or not folder:
        raise ValueError("Choose the destination bucket and folder first.")
    if any(folder == p or folder.startswith(p + "/") for p in sx.PROTECTED):
        raise ValueError("That destination is protected.")
    if settings.with_labels:
        ids = list(settings.class_ids.values())
        if not ids or len(set(ids)) != len(ids) or any(type(n) is not int or n < 0 or n > 99 for n in ids):
            raise ValueError("Save distinct class IDs from 0 to 99 for the selected labels.")


def save(config: SensorConfig, moment=None):
    if config.enabled:
        validate_selection(config)
    if config.enabled and os.environ.get("APP_BACKGROUND", "on").lower() == "off":
        raise ValueError("Background scheduling is disabled in this app instance.")
    moment = moment or now()
    with _lock:
        if _running:
            raise ValueError("A sensor check is running. Wait for it to finish before changing its settings.")
        state = read()
        before = state["config"]
        after = config.model_dump()
        after["settings"]["pull_folder"] = after["settings"]["pull_folder"].strip().strip("/")
        previous_plan = {**before, "anchor": state.get("anchor", "")}
        plan = clock.with_anchor(previous_plan, {**after, "anchor": state.get("anchor", "")}, now=moment)
        if before.get("settings") != after["settings"]:
            state["seen"] = []
        state.update(config=after, anchor=plan.get("anchor", ""), saved_at=moment.isoformat())
        state["next_at"] = clock.next_runs(plan, start=moment, count=1)[0].isoformat() if config.enabled else None
        write(state)
        return status()


def recover():
    """Mark a prior process's unfinished run explicitly; never pretend it finished."""
    global _recovered
    with _lock:
        if _recovered:
            return
        _recovered = True
        state = read()
        if (state.get("last_run") or {}).get("status") == "running":
            state["last_run"].update(status="interrupted", error="The app restarted during this check.", finished_at=now().isoformat())
            state["history"] = [state["last_run"], *state.get("history", [])][:20]
            if state["config"].get("enabled"):
                state["next_at"] = now().isoformat()
            write(state)


def start(*, manual=False, moment=None):
    global _running
    moment = moment or now()
    with _lock:
        recover()
        if _running:
            return False
        state = read()
        config = SensorConfig.model_validate(state["config"])
        if not manual:
            due = state.get("next_at")
            if not config.enabled or not due or dt.datetime.fromisoformat(due) > moment:
                return False
        validate_selection(config)
        # Claim the slot before launching. On downtime, catch up once, not once
        # for every missed day. Manual runs do not shift the scheduled grid.
        if not manual:
            plan = {**config.model_dump(), "anchor": state.get("anchor", "")}
            state["next_at"] = clock.next_runs(plan, start=moment, count=1)[0].isoformat()
        state["last_run"] = {"id": uuid.uuid4().hex, "status": "running", "started_at": moment.isoformat(),
                             "trigger": "manual" if manual else "schedule", "settings": config.settings.model_dump()}
        write(state)
        _running = True
        threading.Thread(target=execute, args=(config, state["last_run"]["id"]), daemon=True, name="sql-file-sensor").start()
        return True


def execute(config, run_id):
    global _running
    outcome = {}
    try:
        settings = config.settings
        sql = sx.preset_sql(settings.preset_class, settings.preset_scope,
                            crop_verdicts=sx.has_crop_verdicts(settings.database))
        result = sx.run(settings.database, sql)
        if result.get("truncated"):
            raise ValueError("Query was truncated; narrow the saved selection before scheduling a pull.")
        eligible = [i for i in result["images"] if i.get("bucket") and
                    (not settings.with_labels or not i.get("labels") or i["labels"].get("boxes"))]
        if len(eligible) > sx.PULL_LIMIT:
            raise ValueError("Too many eligible frames for one pull; narrow the saved selection.")
        destination = settings.pull_folder.strip().strip("/")
        if exists(settings.pull_bucket, destination + "/export-contract.json"):
            raise ValueError("This destination uses a different export format. Choose a regular frame-pull destination.")
        def pending(entry):
            image, side, name = entry
            image_key = f"{destination}/images/{side}/{name}"
            label_key = f"{destination}/labels/{side}/{name.rsplit('.', 1)[0]}.txt"
            return not exists(settings.pull_bucket, image_key) or (settings.with_labels and not exists(settings.pull_bucket, label_key))
        with ThreadPoolExecutor(max_workers=8) as pool:
            pending_count = sum(pool.map(pending, sx.copy_plan(eligible, settings.train_percent)))
        current = sorted({i["uri"] for i in eligible})
        with _lock:
            seen = set(read().get("seen", []))
        outcome.update(found=len(result["images"]), eligible=len(eligible), pending=pending_count,
                       newly_eligible=len(set(current) - seen), excluded=len(result["images"]) - len(eligible))
        if pending_count:
            # Use the same query snapshot, labels, names and destination checks
            # as the manual button. Existing image/label objects are skipped.
            outcome["summary"] = sx.pull(settings.database, sql, dest_bucket=settings.pull_bucket,
                dest_folder=destination, train_percent=settings.train_percent, with_labels=settings.with_labels,
                class_ids=settings.class_ids if settings.with_labels else None, query_result=result,
                progress=lambda event: update_progress(run_id, event))
            outcome["status"] = "partial" if outcome["summary"]["failed"] else "complete"
        else:
            outcome["status"] = "unchanged"
        with _lock:
            state = read()
            state["seen"] = current
            write(state)
    except Exception as exc:
        outcome.update(status="error", error=str(exc))
    finally:
        with _lock:
            try:
                state = read()
                state["last_run"].update(outcome, finished_at=now().isoformat())
                state["history"] = [state["last_run"], *state.get("history", [])][:20]
                write(state)
            finally:
                _running = False


def update_progress(run_id, event):
    # Persist only milestones, not thousands of per-file events.
    if event.get("completed", 0) % 50 and event.get("phase") not in ("finalizing",):
        return
    with _lock:
        state = read()
        if (state.get("last_run") or {}).get("id") == run_id:
            state["last_run"]["progress"] = event
            write(state)


def tick():
    if os.environ.get("APP_BACKGROUND", "on").lower() != "off":
        start()
