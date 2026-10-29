"""Assemble a dataset on Data Preparation's own schedule (2026-10-29).

The Assemble card has a radio: MANUAL or SCHEDULED. Scheduled means the
dataset is rebuilt by itself after each SCHEDULED pull on Data
Preparation's PULL SCHEDULE -- the same build the training chain does
(`retrain_schedule.run_chain`), stopped before anything trains
(`submit="build"`). One clock per tab: the data side follows the pull,
the training side keeps its own (Multi-Auto Training's toolbar).

Watched from the app's poll tick, like `queue_after_pull`: a scheduled
pull runs in a Lambda, which has no hook into this process. NOT from the
queue-runner Lambda -- it may only read S3, and a build writes.

`claim_once` per pull run means the stable and development containers,
which both tick, build once between them.
"""

from __future__ import annotations

from .. import pull_control as pc
from . import queue_after_pull as qap
from . import retrain_schedule as rs

#: Where the radio is stored, inside the Assemble card's own settings.
MODES = ("manual", "scheduled")
#: The claim that makes one build per pull, across every container.
CLAIM = "assemble_after_pull"
#: Where the last outcome is kept. A result, not a setting, so no settings
#: file carries it (it is in neither tab's SCOPES).
LAST = "assemble_schedule_last"


def mode(config: dict | None = None) -> str:
    config = config if config is not None else pc.read_config()
    value = str((config.get("dataset_assemble") or {}).get("run") or "manual")
    return value if value in MODES else "manual"


def pending(config: dict | None = None) -> dict:
    """Would a build start now, and if not, why not. Acts on nothing."""
    config = config if config is not None else pc.read_config()
    if mode(config) != "scheduled":
        return {"act": False, "reason": "Assemble is set to manual.", "run": None}
    if str(config.get("mode") or "off") != "schedule":
        return {"act": False, "run": None,
                "reason": "Data Preparation's pull schedule is not \"On a schedule\", "
                          "so there is no scheduled pull to build after."}
    # When training follows the pull, the training chain rebuilds the same
    # dataset itself on the same pull; building it here too would be two
    # writers of one data.yaml racing each other.
    if rs.training_trigger(config) == "pull":
        return {"act": False, "run": None,
                "reason": "Training follows each pull, and that chain builds this "
                          "dataset itself."}
    runs = qap._finished_pulls(plan={"triggers": ["schedule"]})
    if not runs:
        return {"act": False, "reason": "No scheduled pull has finished yet.", "run": None}
    newest = runs[0]
    run_id = str(newest.get("run_id") or "")
    if not run_id or pc.claim_exists(CLAIM, run_id):
        return {"act": False, "reason": "Already built for the last pull.", "run": newest}
    if not int(newest.get("moved") or 0):
        return {"act": False, "run": newest,
                "reason": "The last pull moved no files, so the dataset is unchanged."}
    return {"act": True, "reason": "", "run": newest}


def check(config: dict | None = None) -> dict:
    """Called on the app's poll tick. Builds at most once per pull. Never
    raises: a tick that crashed would stop the queue behind it."""
    try:
        config = config if config is not None else pc.read_config()
        decision = pending(config)
        if not decision["act"]:
            return {"built": False, "reason": decision["reason"]}
        run_id = str(decision["run"].get("run_id") or "")
        if not pc.claim_once(CLAIM, run_id):
            return {"built": False, "reason": "Another container has this pull."}
        out = rs.run_chain(config=rs.automatic(config), trigger="data schedule",
                           submit="build")
        last = {"run_id": run_id, "ok": bool(out.get("ok")), "at": out.get("at", ""),
                "folder": out.get("folder", ""),
                "snapshot_id": out.get("snapshot_id", ""),
                "detail": next((s.get("detail", "") for s in reversed(out.get("steps") or [])
                                if not s.get("ok")), "") if not out.get("ok") else ""}
        pc.write_config({LAST: last})
        return {"built": bool(out.get("ok")), **last}
    except Exception as exc:                                       # noqa: BLE001
        return {"built": False, "reason": f"{type(exc).__name__}: {exc}"}
