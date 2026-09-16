import os

# BEFORE any `ppe_pipeline` import, and deliberately not a fixture.
#
# `config.py` calls `load_dotenv()` at import time, and this developer's
# `.env` sets `PPE_STORAGE_BACKEND=s3` with the real dev bucket. The autouse
# fixture below pins the backend to `local` for the duration of each test --
# but several endpoints judge on a DAEMON THREAD, and a thread that outlives
# the test that started it runs after monkeypatch has restored `s3`.
#
# That is not theoretical. One row in the real bucket's audit log reads
# `image_key: C:/.../pytest-of-shali/pytest-502/test_cancelling_stops_after_
# th0/alarm_images/case0.jpg` -- written on 2026-08-31 by
# `test_cancelling_stops_after_the_case_in_flight`, whose batch thread was
# still judging when the test returned. It surfaces months later as "Image
# not found" on the escalations screen, pointing at a temp directory that no
# longer exists.
#
# Setting it here makes it stick for the whole session, so a late thread
# cannot reach S3 no matter when it lands. `load_dotenv()` does not override
# a variable that is already set, so this survives the import too.
os.environ["PPE_STORAGE_BACKEND"] = "local"
os.environ.pop("PPE_S3_BUCKET", None)

import pytest  # noqa: E402 - must follow the environment pin above

from ppe_pipeline import helmet_detector, storage, track1_adapter  # noqa: E402
from ppe_pipeline_api import paths as api_paths  # noqa: E402


@pytest.fixture(autouse=True)
def _dummy_bedrock_credential(monkeypatch):
    """Give every test a placeholder `AWS_BEARER_TOKEN_BEDROCK`.

    `JudgeClient.__init__` builds an `openai.OpenAI` client for Bedrock, and
    that SDK raises `OpenAIError: Missing credentials` at *construction*
    time when no key is present — before any request is made. So a test
    that stubs `JudgeClient.judge` but lets the real `JudgeClient` be
    constructed (as the API routers do) still blows up with no credential
    in the environment.

    This is the gap that let CI and local runs disagree: a developer `.env`
    supplies a real token, GitHub Actions has none. Pinning a fake value
    here makes the suite behave identically in both, and — combined with
    `_no_live_helmet_localization` below — means a stray real call fails
    with a 401 rather than quietly succeeding against production Bedrock.

    Tests that care about the credential itself (see
    `test_judge_client_bedrock_provider_builds_base_url_from_region_config`)
    set their own value; their `monkeypatch.setenv` runs after this and wins.
    """
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "test-dummy-token")


@pytest.fixture(autouse=True)
def _no_helmet_detector(monkeypatch):
    """Keep the stage-2 helmet detector out of the suite by default.

    Loading the YOLO11s weights costs seconds and pulls torch into every
    test process, for a box that is decorative to almost every test here.
    Returning `None` is a supported state -- it means "no weights / disabled",
    which callers already handle by falling back to the judge's own inline
    `helmet_box` -- so a test asserting a box is present still gets one.

    Tests that exercise localization patch
    `ppe_pipeline.helmet_detector.get_helmet_detector` themselves; because
    both call sites reach it through the module (not a `from` import), that
    patch reaches them.
    """
    monkeypatch.setattr(helmet_detector, "get_helmet_detector", lambda *a, **kw: None)


@pytest.fixture(autouse=True)
def _isolate_alarm_image_dir(monkeypatch, tmp_path):
    """Keep persisted alarm frames out of the repo.

    `run_track1_judge_inference` writes each judged frame through
    `storage`, and the local backend is rooted at the working directory --
    so the default `eval/alarm_images` key lands real .jpg files in the
    source tree during a test run. Redirect every test to its own tmp dir.
    """
    target = str(tmp_path / "alarm_images")
    monkeypatch.setattr(track1_adapter, "DEFAULT_ALARM_IMAGE_DIR", target)
    # The API routers pass `alarm_image_dir=paths.ALARM_IMAGE_DIR` explicitly,
    # so patching only the module default above misses every test that goes
    # through HTTP -- which is how binary test output reached the repo once
    # already. Patch both.
    monkeypatch.setattr(api_paths, "ALARM_IMAGE_DIR", target)


@pytest.fixture(autouse=True)
def _isolate_storage_backend(monkeypatch):
    """Force `PPE_STORAGE_BACKEND` to the safe "local" default for every
    test, so a test can never accidentally make `storage.get_storage()`
    reach out to the real S3 bucket just because the ambient environment
    happens to have `PPE_STORAGE_BACKEND=s3` set (e.g. a developer's shell,
    or CI).

    Tests that want to exercise `S3Storage` should construct it directly
    with an injected fake/mock client (see `tests/test_storage.py`) rather
    than going through `get_storage()`'s env-based backend selection — this
    fixture doesn't interfere with that, since it never touches `S3Storage`
    itself.
    """
    monkeypatch.setenv("PPE_STORAGE_BACKEND", "local")
    monkeypatch.delenv("PPE_S3_BUCKET", raising=False)


@pytest.fixture(autouse=True)
def _empty_process_caches():
    """Start every test with both process-wide byte caches empty.

    `storage._corpus_cache` is keyed by (bucket, full key) and the S3 tests
    all use the same handful of both, so without this one test's fake objects
    would answer another's conditional GET. Against real S3 the ETag would
    settle it; two hand-built fakes can agree on an ETag by accident.

    `storage._frame_cache` is keyed by storage key alone, and several tests
    write their own bytes at the same tidy `eval/alarm_images/case0.jpg`. It
    revalidates -- a re-run overwrites an alarm frame at the same key, and
    production does exactly that -- so this is belt and braces rather than the
    freshness rule: a tmp file two tests write in the same filesystem tick,
    to the same length, would otherwise be proved current by a token that
    cannot tell them apart.
    """
    storage._corpus_cache.clear()
    storage._frame_cache.clear()
    yield
    storage._corpus_cache.clear()
    storage._frame_cache.clear()


@pytest.fixture(autouse=True)
def _no_real_bucket_after_the_test(monkeypatch):
    """The same pin, restored to `local` rather than to whatever was there.

    Belt and braces with the module-level pin above: a test that legitimately
    sets `PPE_STORAGE_BACKEND=s3` itself must not leave it set for a daemon
    thread that outlives it.
    """
    yield
    os.environ["PPE_STORAGE_BACKEND"] = "local"
    os.environ.pop("PPE_S3_BUCKET", None)
