"""Tests for `ppe_pipeline.storage`'s two backends.

`S3Storage` is exercised against a hand-rolled fake shaped like the real
boto3 S3 client surface (`get_object`/`put_object`/`get_paginator`), rather
than `moto` — `moto` isn't already a dependency here and pulls in a fairly
heavy transitive footprint (flask, werkzeug, ...) for a narrow surface we
fully control via `S3Storage`'s injectable `client` param; the codebase
already prefers this hand-rolled-fake style for other SDKs (see
`tests/test_judge_client.py`'s `FakeAnthropicBlock` etc.). No real AWS
credentials or network access are used, matching the precedent set by
commit 1e0cfe8 for the Bedrock provider test.
"""

from __future__ import annotations

import hashlib
import io
import json
import pathlib
from concurrent.futures import ThreadPoolExecutor

import pytest
from botocore.exceptions import ClientError

from ppe_pipeline import storage as storage_mod
from ppe_pipeline.storage import (
    ConcurrentModificationError,
    LocalStorage,
    S3Storage,
    Storage,
    get_storage,
)


# --- LocalStorage ------------------------------------------------------


def test_local_read_json_returns_default_when_missing(tmp_path):
    store = LocalStorage(root=tmp_path)
    value, token = store.read_json("gold_annotations.json", default={})
    assert value == {}
    assert token is None


def test_local_read_json_raises_when_missing_and_no_default(tmp_path):
    store = LocalStorage(root=tmp_path)
    with pytest.raises(FileNotFoundError):
        store.read_json("gold_annotations.json")


def test_local_write_json_cas_and_read_json_roundtrip(tmp_path):
    store = LocalStorage(root=tmp_path)
    store.write_json_cas("gold_annotations.json", {"abc": {"final_label": "violation"}}, expected_token=None)

    value, _ = store.read_json("gold_annotations.json")
    assert value == {"abc": {"final_label": "violation"}}


def test_local_update_json_creates_parent_dirs(tmp_path):
    store = LocalStorage(root=tmp_path)
    store.update_json("nested/dir/gold_annotations.json", lambda _current: {"a": 1}, default={})

    assert (tmp_path / "nested" / "dir" / "gold_annotations.json").is_file()


def test_local_list_keys_matches_direct_children_only(tmp_path):
    (tmp_path / "known_cases.json").write_text("[]")
    (tmp_path / "combined_eval_dataset.json").write_text("[]")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "nested.json").write_text("[]")

    store = LocalStorage(root=".")
    keys = store.list_keys(str(tmp_path))

    names = {key.split("/")[-1] for key in keys}
    assert names == {"known_cases.json", "combined_eval_dataset.json"}


def test_local_get_bytes_put_bytes_roundtrip(tmp_path):
    store = LocalStorage(root=tmp_path)
    store.put_bytes("crops/glove_01.jpg", b"fake-jpeg-bytes")

    assert store.get_bytes("crops/glove_01.jpg") == b"fake-jpeg-bytes"


def test_local_get_bytes_raises_when_missing(tmp_path):
    store = LocalStorage(root=tmp_path)
    with pytest.raises(FileNotFoundError):
        store.get_bytes("crops/missing.jpg")


def test_local_put_bytes_silently_overwrites_existing_key(tmp_path):
    store = LocalStorage(root=tmp_path)
    store.put_bytes("crops/glove_01.jpg", b"first")
    store.put_bytes("crops/glove_01.jpg", b"second")

    assert store.get_bytes("crops/glove_01.jpg") == b"second"


def test_local_list_keys_filters_by_custom_suffixes(tmp_path):
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "glove_01.jpg").write_bytes(b"a")
    (tmp_path / "glove_02.jpeg").write_bytes(b"b")
    (tmp_path / "notes.txt").write_text("x")

    store = LocalStorage(root=tmp_path)
    keys = store.list_keys(".", suffixes=(".jpg", ".jpeg", ".png"))

    names = {key.split("/")[-1] for key in keys}
    assert names == {"glove_01.jpg", "glove_02.jpeg"}


def test_local_list_keys_custom_suffix_matching_is_case_insensitive(tmp_path):
    (tmp_path / "IMAGE.JPG").write_bytes(b"a")
    (tmp_path / "photo.jpg").write_bytes(b"b")

    store = LocalStorage(root=tmp_path)

    names_lower_suffix = {key.split("/")[-1] for key in store.list_keys(".", suffixes=(".jpg",))}
    assert names_lower_suffix == {"IMAGE.JPG", "photo.jpg"}

    names_upper_suffix = {key.split("/")[-1] for key in store.list_keys(".", suffixes=(".JPG",))}
    assert names_upper_suffix == {"IMAGE.JPG", "photo.jpg"}


def test_local_update_json_rejects_non_positive_retries(tmp_path):
    store = LocalStorage(root=tmp_path)
    store.write_json_cas("counter.json", {"count": 0}, expected_token=None)

    with pytest.raises(ValueError):
        store.update_json("counter.json", lambda cur: {"count": cur["count"] + 1}, retries=0)


def test_local_update_json_serializes_concurrent_writers_via_filelock(tmp_path):
    """Many threads incrementing the same counter through `update_json`
    must not lose an update — this is the local-mode analogue of the
    concurrency problem `S3Storage`'s CAS logic solves for S3."""
    store = LocalStorage(root=tmp_path)
    store.write_json_cas("counter.json", {"count": 0}, expected_token=None)

    def increment(_):
        store.update_json("counter.json", lambda cur: {"count": cur["count"] + 1})

    with ThreadPoolExecutor(max_workers=10) as executor:
        list(executor.map(increment, range(50)))

    value, _ = store.read_json("counter.json")
    assert value["count"] == 50


# --- get_storage() backend selection --------------------------------------


def test_get_storage_defaults_to_local_under_test_isolation_fixture():
    """Guards against the autouse fixture in `tests/conftest.py` silently
    failing to isolate tests from the ambient `PPE_STORAGE_BACKEND` — if it
    stopped working, this would only select S3 (and try real network
    access) when a developer's shell/CI happens to have that env var set,
    which is exactly the flaky-by-environment failure mode it exists to
    prevent."""
    assert isinstance(get_storage(), LocalStorage)


def test_get_storage_raises_when_s3_backend_selected_without_bucket_set(monkeypatch):
    """SCRUM-11 follow-up: `PPE_S3_BUCKET` must never silently fall back to
    the real production bucket (`safety-model-dataset`, which this dev
    machine has live credentials for) — someone flipping on
    `PPE_STORAGE_BACKEND=s3` for casual local testing without also setting
    `PPE_S3_BUCKET` should get a clear error, not real network calls
    against production."""
    monkeypatch.setenv("PPE_STORAGE_BACKEND", "s3")
    monkeypatch.delenv("PPE_S3_BUCKET", raising=False)

    with pytest.raises(RuntimeError, match="PPE_S3_BUCKET"):
        get_storage()


def test_get_storage_uses_s3_when_backend_and_bucket_are_both_set(monkeypatch):
    monkeypatch.setenv("PPE_STORAGE_BACKEND", "s3")
    monkeypatch.setenv("PPE_S3_BUCKET", "some-explicit-bucket")

    store = get_storage()

    assert isinstance(store, S3Storage)
    assert store.bucket == "some-explicit-bucket"


# --- S3Storage -----------------------------------------------------------


class FakeS3Client:
    """Minimal in-memory stand-in for boto3's S3 client, just enough of the
    real surface (get_object/put_object/list_objects_v2 conditional-write
    semantics, including botocore's ClientError shape) for S3Storage's own
    logic to be tested deterministically."""

    def __init__(self):
        self._objects: dict[str, tuple[bytes, str]] = {}
        self.get_calls: list[tuple[str, str | None]] = []

    @staticmethod
    def _etag(body: bytes) -> str:
        return f'"{hashlib.md5(body).hexdigest()}"'

    def get_object(self, Bucket, Key, IfNoneMatch=None):
        self.get_calls.append((Key, IfNoneMatch))
        if Key not in self._objects:
            raise ClientError({"Error": {"Code": "NoSuchKey", "Message": "Not Found"}}, "GetObject")
        body, etag = self._objects[Key]
        if IfNoneMatch is not None and IfNoneMatch == etag:
            # S3 answers a matching If-None-Match with a bodiless 304, which
            # botocore surfaces as a ClientError rather than a response.
            raise ClientError(
                {
                    "Error": {"Code": "304", "Message": "Not Modified"},
                    "ResponseMetadata": {"HTTPStatusCode": 304},
                },
                "GetObject",
            )
        return {"Body": io.BytesIO(body), "ETag": etag}

    def put_object(self, Bucket, Key, Body, ContentType=None, IfMatch=None, IfNoneMatch=None):
        existing = self._objects.get(Key)
        if IfNoneMatch == "*" and existing is not None:
            raise ClientError({"Error": {"Code": "PreconditionFailed", "Message": "at least one condition failed"}}, "PutObject")
        if IfMatch is not None and (existing is None or existing[1] != IfMatch):
            raise ClientError({"Error": {"Code": "PreconditionFailed", "Message": "at least one condition failed"}}, "PutObject")

        etag = self._etag(Body)
        self._objects[Key] = (Body, etag)
        return {"ETag": etag}

    def head_object(self, Bucket, Key):
        if Key not in self._objects:
            raise ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject")
        return {"ETag": self._objects[Key][1]}

    def delete_object(self, Bucket, Key):
        self._objects.pop(Key, None)
        return {}

    def get_paginator(self, operation_name):
        assert operation_name == "list_objects_v2"
        client = self

        class _Paginator:
            def paginate(self, Bucket, Prefix=""):
                contents = [{"Key": key} for key in sorted(client._objects) if key.startswith(Prefix)]
                yield {"Contents": contents}

        return _Paginator()


class AlwaysConflictingFakeS3Client(FakeS3Client):
    """Simulates permanent contention: every put_object loses the race, no
    matter how many times the caller retries."""

    def put_object(self, **kwargs):
        raise ClientError({"Error": {"Code": "PreconditionFailed", "Message": "at least one condition failed"}}, "PutObject")


class ConflictOnceThenAlreadyAppliedFakeS3Client(FakeS3Client):
    """Simulates a racing writer that wins the first attempt and happens to
    write the exact same effective value this call was about to write —
    e.g. two reviewers independently correcting the same case_id to the
    same ground_truth at nearly the same time. The first `put_object` call
    loses the CAS race (as a real racing writer's commit would cause), and
    that racing write is applied directly to the backing store so a
    subsequent read reflects it."""

    def __init__(self, winning_value: dict):
        super().__init__()
        self._winning_body = json.dumps(winning_value, indent=2).encode("utf-8")
        self.put_calls = 0

    def put_object(self, Bucket, Key, Body, ContentType=None, IfMatch=None, IfNoneMatch=None):
        self.put_calls += 1
        if self.put_calls == 1:
            self._objects[Key] = (self._winning_body, self._etag(self._winning_body))
            raise ClientError({"Error": {"Code": "PreconditionFailed", "Message": "at least one condition failed"}}, "PutObject")
        return super().put_object(Bucket=Bucket, Key=Key, Body=Body, ContentType=ContentType, IfMatch=IfMatch, IfNoneMatch=IfNoneMatch)


def test_s3_update_json_rejects_non_positive_retries():
    store = S3Storage(bucket="safety-model-dataset", client=FakeS3Client())

    with pytest.raises(ValueError):
        store.update_json("counter.json", lambda cur: {"count": cur["count"] + 1}, default={"count": 0}, retries=0)


def test_s3_update_json_return_changed_is_false_when_retry_finds_value_already_matches():
    """Issue 2 regression: a call whose first write attempt conflicts, but
    whose retry re-read shows the value already matches what this call
    would have written, must report `changed=False` — it made zero
    successful writes of its own."""
    client = ConflictOnceThenAlreadyAppliedFakeS3Client(winning_value={"count": 1})
    client._objects["counter.json"] = (b'{"count": 0}', '"seed-etag"')
    store = S3Storage(bucket="safety-model-dataset", client=client)

    new_value, changed = store.update_json(
        "counter.json", lambda cur: {"count": 1}, retries=3, return_changed=True
    )

    assert new_value == {"count": 1}
    assert changed is False
    assert client.put_calls == 1


def test_s3_first_write_of_a_key_uses_if_none_match(tmp_path=None):
    store = S3Storage(bucket="safety-model-dataset", client=FakeS3Client())

    store.write_json_cas("gold_annotations.json", {"abc": {}}, expected_token=None)

    value, token = store.read_json("gold_annotations.json")
    assert value == {"abc": {}}
    assert token is not None


def test_s3_first_write_fails_if_key_already_exists():
    store = S3Storage(bucket="safety-model-dataset", client=FakeS3Client())
    store.write_json_cas("gold_annotations.json", {"abc": {}}, expected_token=None)

    with pytest.raises(ConcurrentModificationError):
        store.write_json_cas("gold_annotations.json", {"xyz": {}}, expected_token=None)


def test_s3_write_json_cas_succeeds_when_etag_matches():
    store = S3Storage(bucket="safety-model-dataset", client=FakeS3Client())
    store.write_json_cas("gold_annotations.json", {"count": 1}, expected_token=None)
    _, token = store.read_json("gold_annotations.json")

    new_token = store.write_json_cas("gold_annotations.json", {"count": 2}, expected_token=token)

    assert new_token is not None
    value, _ = store.read_json("gold_annotations.json")
    assert value == {"count": 2}


def test_s3_write_json_cas_raises_on_stale_etag():
    store = S3Storage(bucket="safety-model-dataset", client=FakeS3Client())
    store.write_json_cas("gold_annotations.json", {"count": 1}, expected_token=None)

    with pytest.raises(ConcurrentModificationError):
        store.write_json_cas("gold_annotations.json", {"count": 2}, expected_token='"stale-etag"')


def test_s3_update_json_raises_concurrent_modification_after_exhausting_retries():
    store = S3Storage(bucket="safety-model-dataset", client=AlwaysConflictingFakeS3Client())
    store.client._objects["counter.json"] = (b'{"count": 0}', '"seed-etag"')

    with pytest.raises(ConcurrentModificationError):
        store.update_json("counter.json", lambda cur: {"count": cur["count"] + 1}, retries=3)


# --- the revalidating corpus cache ---------------------------------------
#
# THE PIN FOR ALL OF IT is the first test here: something else changes the
# object underneath a cached copy, and the next read must see the change.
# Everything below it is detail; if that one fails the cache is a bug.


def test_revalidating_read_sees_a_write_made_underneath_it():
    client = FakeS3Client()
    store = S3Storage(bucket="safety-model-dataset", client=client)
    client._objects["eval/audit_reviews.json"] = (b'{"a": 1}', '"one"')
    assert store.read_json("eval/audit_reviews.json", revalidate=True)[0] == {"a": 1}

    # Not through `store` -- a different writer entirely, which is the only
    # case a revalidating cache has to survive.
    client._objects["eval/audit_reviews.json"] = (b'{"a": 2}', '"two"')

    value, token = store.read_json("eval/audit_reviews.json", revalidate=True)
    assert value == {"a": 2}
    assert token == '"two"'


def test_revalidating_read_reuses_the_body_when_the_etag_still_matches():
    client = FakeS3Client()
    store = S3Storage(bucket="safety-model-dataset", client=client)
    client._objects["eval/audit_reviews.json"] = (b'{"a": 1}', '"one"')

    first = store.read_json("eval/audit_reviews.json", revalidate=True)
    second = store.read_json("eval/audit_reviews.json", revalidate=True)

    assert first == second == ({"a": 1}, '"one"')
    # The second read still went to S3 -- that is what proves freshness --
    # but it went conditionally, so S3 sent no body.
    assert client.get_calls == [
        ("eval/audit_reviews.json", None),
        ("eval/audit_reviews.json", '"one"'),
    ]


def test_read_without_revalidate_is_never_cached():
    client = FakeS3Client()
    store = S3Storage(bucket="safety-model-dataset", client=client)
    client._objects["eval/audit_reviews.json"] = (b'{"a": 1}', '"one"')

    store.read_json("eval/audit_reviews.json")
    store.read_json("eval/audit_reviews.json")

    assert [etag for _, etag in client.get_calls] == [None, None]


def test_revalidating_read_hands_each_caller_its_own_object():
    """A cached body is re-parsed per read, so one caller mutating what it
    got cannot corrupt what the next caller is given -- `update_json`'s
    `mutate_fn` is free to work in place, and several here do."""
    client = FakeS3Client()
    store = S3Storage(bucket="safety-model-dataset", client=client)
    client._objects["eval/audit_reviews.json"] = (b'{"a": {"n": 1}}', '"one"')

    first, _ = store.read_json("eval/audit_reviews.json", revalidate=True)
    first["a"]["n"] = 99
    first["b"] = {}

    second, _ = store.read_json("eval/audit_reviews.json", revalidate=True)
    assert second == {"a": {"n": 1}}


def test_a_write_refreshes_the_cached_copy_rather_than_leaving_it_stale():
    """A saved decision is read back by the same request that made it -- every
    save here re-reads the corpus after writing it. Correctness does not turn
    on this (the conditional GET would catch the change either way, which is
    the point of revalidating rather than expiring); the transfer does. The
    put already carries both the new bytes and the new ETag, so paying for
    them again is pure waste."""
    client = FakeS3Client()
    store = S3Storage(bucket="safety-model-dataset", client=client)
    client._objects["eval/audit_reviews.json"] = (b'{"a": 1}', '"one"')
    store.read_json("eval/audit_reviews.json", revalidate=True)

    store.update_json("eval/audit_reviews.json", lambda cur: {**cur, "b": 2})
    written_etag = client._objects["eval/audit_reviews.json"][1]

    value, token = store.read_json("eval/audit_reviews.json", revalidate=True)

    assert value == {"a": 1, "b": 2}
    assert token == written_etag
    # The read-back went out conditional on what was JUST written, so it came
    # home a 304 rather than another copy of the file.
    assert client.get_calls[-1] == ("eval/audit_reviews.json", written_etag)


def test_an_update_to_a_cached_key_does_not_read_it_again():
    """`update_json` is a GET and a PUT, and against this bucket the GET on an
    8 KB sidecar is a bare 177 ms round trip -- latency, not transfer, so
    revalidating it would save nothing and skipping it saves the whole trip."""
    client = FakeS3Client()
    store = S3Storage(bucket="safety-model-dataset", client=client)
    client._objects["eval/audit_reviews.json"] = (b'{"a": 1}', '"one"')
    store.read_json("eval/audit_reviews.json", revalidate=True)
    client.get_calls.clear()

    store.update_json("eval/audit_reviews.json", lambda cur: {**cur, "b": 2})

    assert client.get_calls == []
    assert json.loads(client._objects["eval/audit_reviews.json"][0]) == {"a": 1, "b": 2}


def test_a_stale_cached_copy_cannot_lose_another_writers_update():
    """THE CORRECTNESS ARGUMENT for skipping that read. The put goes out under
    the cached copy's own ETag, so a copy that is no longer current fails the
    same precondition a racing writer would fail, and the retry reads for
    real. What must never happen is the other writer's key vanishing."""
    client = FakeS3Client()
    store = S3Storage(bucket="safety-model-dataset", client=client)
    client._objects["eval/audit_reviews.json"] = (b'{"a": 1}', '"one"')
    store.read_json("eval/audit_reviews.json", revalidate=True)

    somebody_else = json.dumps({"a": 1, "theirs": 2}).encode()
    client._objects["eval/audit_reviews.json"] = (somebody_else, '"two"')

    result = store.update_json("eval/audit_reviews.json", lambda cur: {**cur, "mine": 3})

    assert result == {"a": 1, "theirs": 2, "mine": 3}


def test_a_cached_copy_cannot_answer_that_nothing_changed():
    """"No change" is a claim about the CURRENT value, which an unproven copy
    is not. So a mutation that is a no-op against the cached copy re-reads
    before answering, and finds the write it would otherwise have skipped."""
    client = FakeS3Client()
    store = S3Storage(bucket="safety-model-dataset", client=client)
    client._objects["eval/audit_reviews.json"] = (b'{"a": 1}', '"one"')
    store.read_json("eval/audit_reviews.json", revalidate=True)

    client._objects["eval/audit_reviews.json"] = (b'{"a": 2}', '"two"')
    client.get_calls.clear()

    value, changed = store.update_json(
        "eval/audit_reviews.json",
        lambda cur: {**cur, "a": 1},
        return_changed=True,
    )

    assert client.get_calls != []
    assert (value, changed) == ({"a": 1}, True)


def test_an_uncached_key_is_still_read_before_it_is_written():
    client = FakeS3Client()
    store = S3Storage(bucket="safety-model-dataset", client=client)
    client._objects["eval/audit_reviews.json"] = (b'{"a": 1}', '"one"')

    store.update_json("eval/audit_reviews.json", lambda cur: {**cur, "b": 2})

    assert client.get_calls == [("eval/audit_reviews.json", None)]


def test_deleting_a_key_drops_its_cached_copy():
    client = FakeS3Client()
    store = S3Storage(bucket="safety-model-dataset", client=client)
    client._objects["eval/audit_reviews.json"] = (b'{"a": 1}', '"one"')
    store.read_json("eval/audit_reviews.json", revalidate=True)

    store.delete("eval/audit_reviews.json")

    assert store.read_json("eval/audit_reviews.json", default={}) == ({}, None)
    assert client.get_calls[-1][1] is None


def test_a_key_that_disappears_is_not_served_from_cache():
    client = FakeS3Client()
    store = S3Storage(bucket="safety-model-dataset", client=client)
    client._objects["eval/audit_reviews.json"] = (b'{"a": 1}', '"one"')
    store.read_json("eval/audit_reviews.json", revalidate=True)

    client._objects.pop("eval/audit_reviews.json")

    assert store.read_json("eval/audit_reviews.json", default={}, revalidate=True) == ({}, None)


def test_the_cache_is_per_bucket_and_per_full_key():
    """Two buckets, or one bucket under two prefixes, are two objects. The
    fake gives identical bodies the same ETag, so a key collision here would
    be invisible rather than loud."""
    client = FakeS3Client()
    client._objects["a/eval/audit_reviews.json"] = (b'{"which": "a"}', '"a"')
    client._objects["b/eval/audit_reviews.json"] = (b'{"which": "b"}', '"b"')
    one = S3Storage(bucket="safety-model-dataset", prefix="a/", client=client)
    two = S3Storage(bucket="safety-model-dataset", prefix="b/", client=client)

    assert one.read_json("eval/audit_reviews.json", revalidate=True)[0] == {"which": "a"}
    assert two.read_json("eval/audit_reviews.json", revalidate=True)[0] == {"which": "b"}
    assert one.read_json("eval/audit_reviews.json", revalidate=True)[0] == {"which": "a"}


def test_the_cache_is_bounded():
    client = FakeS3Client()
    store = S3Storage(bucket="safety-model-dataset", client=client)
    for n in range(storage_mod.CORPUS_CACHE_MAX_ENTRIES + 3):
        client._objects[f"eval/f{n}.json"] = (b"{}", f'"e{n}"')
        store.read_json(f"eval/f{n}.json", revalidate=True)

    assert len(storage_mod._corpus_cache) == storage_mod.CORPUS_CACHE_MAX_ENTRIES


def test_local_read_json_accepts_revalidate_and_ignores_it(tmp_path):
    """The keyword is part of `Storage`'s contract, so callers do not have to
    know which backend they are on. A local file read has no ETag and nothing
    to save by not doing it."""
    store = LocalStorage(root=tmp_path)
    (tmp_path / "audit_reviews.json").write_text('{"a": 1}')

    assert store.read_json("audit_reviews.json", revalidate=True) == ({"a": 1}, None)


def test_s3_list_keys_matches_only_direct_children_with_json_suffix():
    client = FakeS3Client()
    for key in ["eval/known_cases.json", "eval/combined_eval_dataset.json", "eval/sub/nested.json", "eval/manifest.txt"]:
        client.put_object(Bucket="b", Key=key, Body=b"{}")

    store = S3Storage(bucket="b", client=client)
    keys = store.list_keys("eval")

    assert keys == ["eval/combined_eval_dataset.json", "eval/known_cases.json"]


def test_s3_list_keys_custom_suffix_matching_is_case_insensitive():
    client = FakeS3Client()
    for key in ["eval/IMAGE.JPG", "eval/photo.jpg"]:
        client.put_object(Bucket="b", Key=key, Body=b"x")

    store = S3Storage(bucket="b", client=client)

    assert store.list_keys("eval", suffixes=(".jpg",)) == ["eval/IMAGE.JPG", "eval/photo.jpg"]
    assert store.list_keys("eval", suffixes=(".JPG",)) == ["eval/IMAGE.JPG", "eval/photo.jpg"]


def test_s3_get_bytes_put_bytes_roundtrip():
    store = S3Storage(bucket="b", client=FakeS3Client())
    store.put_bytes("crops/glove_01.jpg", b"fake-jpeg-bytes", content_type="image/jpeg")

    assert store.get_bytes("crops/glove_01.jpg") == b"fake-jpeg-bytes"


def test_s3_get_bytes_raises_filenotfounderror_when_missing():
    store = S3Storage(bucket="b", client=FakeS3Client())
    with pytest.raises(FileNotFoundError):
        store.get_bytes("crops/missing.jpg")


def test_s3_put_bytes_silently_overwrites_existing_key():
    store = S3Storage(bucket="b", client=FakeS3Client())
    store.put_bytes("crops/glove_01.jpg", b"first")
    store.put_bytes("crops/glove_01.jpg", b"second")

    assert store.get_bytes("crops/glove_01.jpg") == b"second"


def test_s3_list_keys_filters_by_custom_suffixes():
    client = FakeS3Client()
    for key in ["eval/manifest.json", "eval/glove_01.jpg", "eval/glove_02.jpeg", "eval/notes.txt"]:
        client.put_object(Bucket="b", Key=key, Body=b"x")

    store = S3Storage(bucket="b", client=client)
    keys = store.list_keys("eval", suffixes=(".jpg", ".jpeg", ".png"))

    assert keys == ["eval/glove_01.jpg", "eval/glove_02.jpeg"]


# --- the frame cache ---------------------------------------------------
#
# AN ALARM FRAME IS NOT WRITE-ONCE, which is what this cache was built
# believing. `POST /track1/run` takes the alarm id from the client,
# `track1_adapter` derives the frame key from it, and both the queue builder
# and the Track1 router say a re-run supersedes the earlier row and overwrites
# the frame at the same key. So the reviewer was served the pre-re-run picture
# for as long as the process lived.


def test_a_local_frame_is_served_from_memory_until_it_changes(tmp_path):
    key = str(tmp_path / "alarm.jpg")
    pathlib.Path(key).write_bytes(b"first")
    store = LocalStorage()
    reads: list[str] = []
    real = store.get_bytes
    store.get_bytes = lambda k: (reads.append(k), real(k))[1]

    assert storage_mod.frame_bytes(key, store) == b"first"
    assert storage_mod.frame_bytes(key, store) == b"first"
    assert reads == [key]

    pathlib.Path(key).write_bytes(b"second-and-longer")

    assert storage_mod.frame_bytes(key, store) == b"second-and-longer"
    assert reads == [key, key]


def test_a_deleted_frame_is_not_still_served(tmp_path):
    key = str(tmp_path / "alarm.jpg")
    pathlib.Path(key).write_bytes(b"first")
    store = LocalStorage()
    storage_mod.frame_bytes(key, store)

    assert store.delete(key) is True
    with pytest.raises(FileNotFoundError):
        storage_mod.frame_bytes(key, store)


def test_deleting_a_frame_lets_go_of_its_bytes(tmp_path):
    """A reset deletes every alarm frame in the corpus. Correctness comes from
    revalidating, not from this -- what this saves is ~0.9MB a frame of bytes
    nothing can be served from again."""
    key = str(tmp_path / "alarm.jpg")
    pathlib.Path(key).write_bytes(b"first")
    store = LocalStorage()
    storage_mod.frame_bytes(key, store)

    store.delete(key)

    assert key not in storage_mod._frame_cache


def test_an_s3_frame_costs_a_round_trip_and_not_a_transfer():
    client = FakeS3Client()
    store = S3Storage(bucket="b", client=client)
    store.put_bytes("eval/alarm_images/a.jpg", b"first")

    assert storage_mod.frame_bytes("eval/alarm_images/a.jpg", store) == b"first"
    assert storage_mod.frame_bytes("eval/alarm_images/a.jpg", store) == b"first"

    # The second read still asked -- conditionally, which is what S3 answers
    # with a bodiless 304.
    assert [call[1] for call in client.get_calls] == [None, client._etag(b"first")]


def test_an_s3_frame_overwritten_by_a_re_run_is_re_fetched():
    client = FakeS3Client()
    store = S3Storage(bucket="b", client=client)
    store.put_bytes("eval/alarm_images/a.jpg", b"first")
    storage_mod.frame_bytes("eval/alarm_images/a.jpg", store)

    store.put_bytes("eval/alarm_images/a.jpg", b"second")

    assert storage_mod.frame_bytes("eval/alarm_images/a.jpg", store) == b"second"


def test_a_backend_that_names_no_version_is_not_cached(tmp_path):
    """`Storage.get_bytes_if_changed`'s default. Cacheable would mean stale,
    so it simply is not cached -- the same rule `_RevalidatingCache.put`
    applies to an object with no ETag."""

    class Versionless(LocalStorage):
        def get_bytes_if_changed(self, key, token):
            return Storage.get_bytes_if_changed(self, key, token)

    key = str(tmp_path / "alarm.jpg")
    pathlib.Path(key).write_bytes(b"first")
    store = Versionless()

    assert storage_mod.frame_bytes(key, store) == b"first"
    assert storage_mod._frame_cache == {}
