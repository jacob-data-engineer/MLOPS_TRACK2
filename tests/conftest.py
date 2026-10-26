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


@pytest.fixture(autouse=True)
def _no_live_aws_writes(monkeypatch):
    from ppe_pipeline_api import pull_control as _pc

    monkeypatch.setattr(_pc, "ddb", _NoLiveWrites(_pc.ddb, "pull_control.ddb"), raising=False)
    monkeypatch.setattr(_pc, "s3", _NoLiveWrites(_pc.s3, "pull_control.s3"), raising=False)
