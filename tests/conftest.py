"""Shared fixtures.

Everything here used to be split between the PPE judge and Track2. The
judge is gone, so what remains is the one guard that matters to this
suite: tests must not write to live AWS.
"""
import pytest


# --- the AWS write guard -------------------------------------------------------
#
# A test once reached live AWS for real. It called `claim_once` because
# the fixture mocked everything except that, and left a claim keyed `r2`
# in production. Harmless that time. The same gap with `write_config`
# would have overwritten the live configuration, and with
# `delete_objects` it would have removed a week of images.
#
# So: make the damaging calls impossible rather than remembering to mock
# each one.
#
# WRITES ONLY, deliberately. A great many tests in this suite read live
# AWS, and converting all of that to fixtures in one go would be a large
# change to tests that are currently honest about what they check. A read
# cannot corrupt anything; a write can. Blocking the half that does the
# damage is the fix that matches the incident. A test that genuinely
# means to exercise a write monkeypatches that method itself, which still
# works.
_DESTRUCTIVE = {
    "put_item", "update_item", "delete_item", "batch_write_item",
    "transact_write_items",
    "put_object", "delete_object", "delete_objects", "copy_object",
    "upload_file", "upload_fileobj",
}


class _NoLiveWrites:
    """Wraps a boto3 client, refusing the calls that change things."""

    def __init__(self, inner, which: str) -> None:
        self._inner = inner
        self._which = which

    def __getattr__(self, call: str):
        if call in _DESTRUCTIVE:
            def refuse(*args, **kwargs):
                raise AssertionError(
                    f"A test called {self._which}.{call}() for real, against live "
                    f"AWS. Monkeypatch it. See the AWS guard in conftest.py."
                )
            return refuse
        return getattr(self._inner, call)


def _loaded(name: str) -> list:
    """Every loaded copy of `ppe_pipeline_api.<name>`.

    There can be two. Some test files import the package as
    `src.ppe_pipeline_api` (by file path), which Python treats as a
    DIFFERENT module with its own `ddb` -- and a guard installed on only
    one copy left the other writing to live AWS. On 2026-10-26 that wrote
    a fake armed watcher and a claim into the live pull-control table.
    """
    import importlib
    import sys

    found = []
    for prefix in ("ppe_pipeline_api", "src.ppe_pipeline_api"):
        module = sys.modules.get(f"{prefix}.{name}")
        if module is None and prefix == "ppe_pipeline_api":
            module = importlib.import_module(f"{prefix}.{name}")
        if module is not None:
            found.append(module)
    return found


@pytest.fixture(autouse=True)
def _auto_export_record_in_memory(monkeypatch):
    """The automatic export's shared record, kept in memory. READS of it
    would otherwise see the live armed watcher and make tests depend on
    whatever the card is set to today."""
    store: dict = {}
    for _mx in _loaded("metrics_export"):
        monkeypatch.setattr(_mx, "_shared_get", lambda: store.get("plan"))
        monkeypatch.setattr(_mx, "_shared_put", lambda plan: store.__setitem__("plan", dict(plan)))
        monkeypatch.setattr(_mx, "_file_settings", lambda: None)


@pytest.fixture(autouse=True)
def _no_live_aws_writes(monkeypatch):
    for _pc in _loaded("pull_control"):
        monkeypatch.setattr(_pc, "ddb", _NoLiveWrites(_pc.ddb, "pull_control.ddb"), raising=False)
        monkeypatch.setattr(_pc, "s3", _NoLiveWrites(_pc.s3, "pull_control.s3"), raising=False)
    # The shared bucket client (2026-10-28). Since the data/training split
    # one `storage.s3` is bound under three names -- storage, the data
    # side's assemble module and the training side's track2_shared. ONE
    # guard, given to all three, so a fake method set on `t2.s3` is seen by
    # the data side too, and no copy of the client is left unguarded.
    for prefix in ("ppe_pipeline_api", "src.ppe_pipeline_api"):
        homes = []
        for name in ("storage", "data.assemble", "training.track2_shared"):
            homes.extend(m for m in _loaded(name) if m.__name__.startswith(prefix + "."))
        if not homes:
            continue
        guard = _NoLiveWrites(homes[0].s3, "storage.s3")
        for module in homes:
            monkeypatch.setattr(module, "s3", guard)
