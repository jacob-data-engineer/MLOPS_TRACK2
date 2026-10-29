"""TRAINING SIDE: models, the registry, the queue, runs, their records
and what happens to a trained model afterwards.

Rule: it reaches the data side only through `ppe_pipeline_api.data`
(the front door), never a module inside it. See
tests/test_side_boundaries.py.
"""

from .._door import through as _through

#: What the training side offers. Everything else in training/ is free to change.
__all__ = [
    # models and the registry
    "get_detector", "detector_group_name", "split_architecture",
    "model_registry_summary", "scratch_refusal",
    # the queue and runs -- each trains on a data-side dataset version
    "queue_add", "queue_start", "start_custom_training_run",
    # what the clock (chain/ticker.py, chain/queue_runner.py) drives
    "restore_queue", "queue_snapshot", "try_advance_queue",
    "reconcile_training_audits", "sweep_idle_endpoint", "check_auto_export",
]

get_detector = _through(__name__, "track2_shared", "get_detector")
detector_group_name = _through(__name__, "track2_shared", "detector_group_name")
split_architecture = _through(__name__, "track2_shared", "split_architecture")
model_registry_summary = _through(__name__, "track2_shared", "model_registry_summary")
scratch_refusal = _through(__name__, "track2_shared", "scratch_refusal")
queue_add = _through(__name__, "track2_shared", "queue_add")
queue_start = _through(__name__, "track2_shared", "queue_start")
start_custom_training_run = _through(__name__, "track2_shared", "start_custom_training_run")
restore_queue = _through(__name__, "track2_shared", "restore_queue")
queue_snapshot = _through(__name__, "track2_shared", "queue_snapshot")
try_advance_queue = _through(__name__, "track2_shared", "try_advance_queue")
reconcile_training_audits = _through(__name__, "track2_shared", "reconcile_training_audits")
sweep_idle_endpoint = _through(__name__, "track2_shared", "sweep_idle_endpoint")
check_auto_export = _through(__name__, "metrics_export", "check")
