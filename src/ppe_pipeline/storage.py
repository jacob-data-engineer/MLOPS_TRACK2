"""Storage abstraction for the small, live-edited JSON state the eval/gold-set
tooling reads and writes (`gold_annotations.json`, `known_cases.json`,
`combined_eval_dataset.json`, few-shot sidecars, ...) — pluggable between
local filesystem (today's behavior, unchanged) and S3 (durable,
multi-reviewer-safe concurrent writes), selected via `PPE_STORAGE_BACKEND`.

S3 has no shared filesystem and no flock equivalent, so `LocalStorage`'s
`filelock.FileLock`-based locking doesn't carry over. `S3Storage` instead
uses ETag-based optimistic concurrency (compare-and-swap) on `put_object`'s
`IfMatch`/`IfNoneMatch` parameters — a single reviewer's Save click is one
fast round trip, not a long-held transaction, and contention among a
handful of named reviewers is rare, so a full lock service is unwarranted.

Callers should go through `update_json` (or the `Storage.update_json`
method) rather than juggling version tokens themselves: it reads the
current value, applies the caller's `mutate_fn`, and retries the whole
read-mutate-write cycle (bounded) if another writer raced it, raising
`ConcurrentModificationError` only once retries are exhausted.
"""

from __future__ import annotations

import copy
import json
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Union

import filelock

JsonValue = Union[dict, list]

def validate_key(key: str) -> str:
    """Reject a storage key that could escape its root.

    Keys are frequently built by interpolating caller-supplied values (an
    alarm id, a case id, a dataset name). Without this, `..` in any of them
    reaches outside the data directory -- on the local backend that is an
    arbitrary file write, and read-back is then an arbitrary file read.
    Enforced here, at the one place every backend funnels through, rather
    than at each of the many call sites that build a key.

    Only upward traversal is refused. Absolute keys are deliberately
    allowed: production sets PPE_EVAL_DATA_DIR to an absolute /data path and
    the tests pass absolute tmp paths, both of which rely on
    `LocalStorage._path`'s `root / key` returning the key itself. Absolute
    keys are also not the injection risk -- a caller-supplied value always
    lands *after* a prefix, so it can never make the key absolute.
    """
    text = str(key)
    if not text:
        raise ValueError("storage key must not be empty")
    normalized = text.replace("\\", "/")
    if any(part == ".." for part in normalized.split("/")):
        raise ValueError(f"storage key must not traverse upwards, got {key!r}")
    return text


_MISSING = object()


#: How many objects the cache below will hold at once. Four keys use it today
#: (the Track1 audit log, the escalation reviews, the audit reviews, the crop
#: export index); the bound is set a little above that so adding a fifth does
#: not silently start thrashing, and low enough that this can never become a
#: general-purpose object cache by accident. The audit log is ~2.6 MB, so the
#: worst case is tens of megabytes, not hundreds.
CORPUS_CACHE_MAX_ENTRIES = 8


class _RevalidatingCache:
    """Object BYTES kept beside the ETag they arrived with, re-proved on every
    read via a conditional GET.

    NOT A TTL CACHE, and deliberately not the thing `CONTRACT.md` refuses when
    it says "a cached aggregate is one more thing to invalidate wrongly". That
    decision stands. Nothing derived is stored here: this holds the exact
    bytes of an S3 object, and every read still goes to S3 with
    `IfNoneMatch`, so a 304 is S3's own proof that the copy is current. There
    is no invalidation rule to get wrong, because there is no window in which
    a stale copy can be served -- the request that would serve it is the same
    request that learns the object changed.

    What it buys is transfer, not round trips: a 304 on the 2.6 MB audit log
    is a bare round trip instead of 2.6 MB, and re-parsing the cached bytes
    costs about half what handing out a deep copy of a parsed value would.
    Re-parsing rather than sharing the parsed object also means no caller can
    corrupt another's -- `update_json`'s `mutate_fn` may work in place.

    Racing threads (the live-feed worker is one, in this same process) cannot
    make it wrong either. The worst interleaving installs an older body under
    its own older ETag; the next conditional GET then fails to match and S3
    sends the current object. A stale entry costs a transfer, never an answer.
    """

    def __init__(self, max_entries: int):
        self._max_entries = max_entries
        self._lock = threading.Lock()
        self._entries: OrderedDict[tuple[str, str], tuple[str, bytes]] = OrderedDict()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def get(self, key: tuple[str, str]) -> tuple[str, bytes] | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
            return entry

    def put(self, key: tuple[str, str], etag: str | None, body: bytes) -> None:
        if not etag:
            # Nothing to revalidate against. A backend that does not report
            # ETags simply does not get cached.
            self.drop(key)
            return
        with self._lock:
            self._entries[key] = (etag, body)
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)

    def refresh(self, key: tuple[str, str], etag: str | None, body: bytes) -> None:
        """Bring an ALREADY-CACHED key up to what was just written to it.

        AN OPTIMISATION, NOT THE INVALIDATION RULE. Every save here re-reads
        the corpus after writing it -- deliberately, so the response is
        computed from the new state -- and that read would come back correct
        without this: the entry is stale, the conditional GET misses, S3 sends
        the current object. Being a revalidating cache rather than an expiring
        one is what makes that true, and it is why there is no window here to
        get wrong. What this saves is the transfer, by noticing that the put
        already told us both the new bytes and the new ETag.

        Only refreshes what is already there. A write is not a reason to start
        caching a key nobody asked to revalidate.
        """
        with self._lock:
            if key not in self._entries:
                return
        self.put(key, etag, body)

    def drop(self, key: tuple[str, str]) -> None:
        with self._lock:
            self._entries.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


_corpus_cache = _RevalidatingCache(CORPUS_CACHE_MAX_ENTRIES)


#: Alarm frames held in memory by storage key, beside the version token they
#: arrived with, most-recently-used last.
#:
#: HERE RATHER THAN IN THE ESCALATIONS ROUTER, where this cache used to live,
#: because the router is not the only reader of a frame. Saving a verdict
#: publishes the crop dataset and the full-frame dataset on two threads, and
#: both fetch the SAME frame -- measured at 0.94MB and 828ms of the save,
#: which the router's copy could not reach because `training_export` sits
#: below the API and cannot import it. `storage` is the module both already
#: import, and it is where the other process-wide byte cache lives, so there
#: is one place to look for what bytes this process is holding.
#:
#: AN ALARM FRAME IS NOT WRITE-ONCE, and this cache was built believing it
#: was. The key is derived from the alarm id (`track1_adapter.image_key_for`),
#: `alarm_id` is a client-supplied field on `POST /track1/run`, and both
#: `escalations.build_queue` and `routers/track1` say in as many words that a
#: re-run supersedes the earlier row and overwrites the frame at the same key.
#: A reviewer was served the pre-re-run picture, and the router's `ETag` then
#: pinned it in their browser for five minutes on top. So this revalidates
#: like `_corpus_cache` does: every read still asks the backend whether the
#: object it holds is current, and the transfer -- not the round trip -- is
#: what is saved.
#:
#: Twelve, at ~0.9MB a frame, is ~11MB held at worst. Sized to cover the
#: biggest audit case in the corpus (nine crops of one frame) with headroom,
#: and bounded rather than unbounded because "every frame ever asked for" is
#: 640k objects. A full re-publish walks hundreds of distinct frames and
#: simply churns it, which costs nothing but the eviction: the working set
#: that matters is one case's frame, held across the two datasets that both
#: want it.
FRAME_CACHE_MAX_ENTRIES = 12
_frame_cache: OrderedDict[str, tuple[str, bytes]] = OrderedDict()
_frame_cache_lock = threading.Lock()


def drop_frame(key: str) -> None:
    """Forget any cached copy of `key`."""
    with _frame_cache_lock:
        _frame_cache.pop(key, None)


def frame_bytes(key: str, backend: Storage | None = None) -> bytes:
    """One alarm frame, transferred at most once per version of it.

    Keyed by the storage key alone, which within a process that talks to one
    bucket names the object -- and the version token beside it names its
    CONTENTS, which is what a re-run changes. A read whose token still matches
    is answered from memory; one whose token does not gets the new bytes in
    the same round trip that discovered they were new, so there is no window
    in which a stale frame can be served and no invalidation rule to get
    wrong. The same shape as `_RevalidatingCache`, for the same reason.
    """
    backend = backend or get_storage()
    with _frame_cache_lock:
        cached = _frame_cache.get(key)
        if cached is not None:
            _frame_cache.move_to_end(key)
    token = cached[0] if cached is not None else None
    try:
        data, current = backend.get_bytes_if_changed(key, token)
    except FileNotFoundError:
        # Deleted under us -- a reset does exactly this. Holding the bytes of
        # an object that no longer exists is the stale read by another door.
        drop_frame(key)
        raise
    if data is None:
        # Only reachable with a token, which only exists with a cached copy.
        return cached[1] if cached is not None else backend.get_bytes(key)
    if not current:
        # A backend with no version to prove: cacheable would mean stale, so
        # it simply is not cached. Same rule as `_RevalidatingCache.put`.
        drop_frame(key)
        return data
    with _frame_cache_lock:
        _frame_cache[key] = (current, data)
        _frame_cache.move_to_end(key)
        while len(_frame_cache) > FRAME_CACHE_MAX_ENTRIES:
            _frame_cache.popitem(last=False)
    return data


class ConcurrentModificationError(Exception):
    """Raised by `write_json_cas`/`update_json` when a key was modified by
    another writer between read and write, and retries were exhausted."""


class Storage:
    def read_json(
        self, key: str, default: JsonValue = _MISSING, *, revalidate: bool = False
    ) -> tuple[JsonValue, str | None]:
        """Returns `(value, version_token)`. If `key` doesn't exist and
        `default` was given, returns `(default, None)` instead of raising.

        `revalidate=True` marks a key as one of the few whole-corpus files
        every screen re-reads, and asks the backend to keep its bytes and
        re-prove them with a conditional GET rather than transfer them again.
        It changes cost, never the value — see `_RevalidatingCache`. Backends
        with no version token to condition on ignore it."""
        raise NotImplementedError

    def write_json_cas(self, key: str, data: JsonValue, expected_token: str | None) -> str | None:
        """Writes `data` to `key`, conditioned on `key` currently being at
        `expected_token` (or not existing yet, if `expected_token is None`).
        Raises `ConcurrentModificationError` if that condition doesn't hold.
        Returns the new version token."""
        raise NotImplementedError

    def list_keys(self, prefix: str, suffixes: tuple[str, ...] = (".json",)) -> list[str]:
        """The keys directly under `prefix` (not recursive) whose filename
        ends with one of `suffixes` (default: `.json`, matching today's
        behavior)."""
        raise NotImplementedError

    def get_bytes(self, key: str) -> bytes:
        """Raises `FileNotFoundError` if `key` doesn't exist."""
        raise NotImplementedError

    def get_bytes_if_changed(
        self, key: str, token: str | None
    ) -> tuple[bytes | None, str | None]:
        """`(bytes, version_token)` for `key`, or `(None, token)` when `key`
        is still exactly the object `token` names.

        The freshness proof `frame_bytes` holds its copies against, in one
        round trip rather than a HEAD followed by a GET. A backend with no
        version to report answers `(bytes, None)`, which reads as "changed"
        and keeps the object out of that cache -- the same rule
        `_RevalidatingCache.put` applies to an object with no ETag.

        Raises `FileNotFoundError` if `key` doesn't exist, like `get_bytes`.
        """
        return self.get_bytes(key), None

    def delete(self, key: str) -> bool:
        """Remove `key`. Returns whether it existed.

        Added for the training export's retraction path: an audit can prove
        an exported label wrong, and leaving a knowingly-incorrect sample in
        the dataset is worse than having no sample. Absent is the desired end
        state either way, so this does not raise on a missing key.

        IT ALSO DROPS THIS PROCESS'S CACHED COPIES of the key. Not for
        correctness -- both caches revalidate, so a deleted key answers with
        `FileNotFoundError` rather than with what it used to hold -- but
        because `reset.py` deletes every alarm frame in the corpus, and a
        megabyte a frame of bytes nothing can ever be served again is worth
        more than the two lines it costs to let go of.
        """
        raise NotImplementedError

    def put_bytes(self, key: str, data: bytes, content_type: str | None = None) -> None:
        """Writes `data` to `key`, silently overwriting if it already
        exists. No CAS/locking — a binary file here is one writer's whole
        object, never a value two writers edit in place.

        THAT IS NOT "WRITTEN ONCE". A re-run of an alarm re-publishes its
        frame and its crops under the same names, deliberately, so the
        newest run is what a reviewer sees. Anything holding these bytes
        must revalidate rather than assume — see `frame_bytes`."""
        raise NotImplementedError

    def _cached_json(self, key: str) -> tuple[JsonValue, str] | None:
        """An UNPROVEN copy of `key` and the token it arrived with, if this
        backend is holding one, else `None`.

        Deliberately not part of `read_json`: nothing may be answered from
        here without a round trip proving it current. The one caller is
        `update_json`, which can prove it a cheaper way — see there."""
        return None

    def update_json(
        self,
        key: str,
        mutate_fn: Callable[[JsonValue], JsonValue],
        default: JsonValue = _MISSING,
        retries: int = 3,
        return_changed: bool = False,
    ) -> JsonValue | tuple[JsonValue, bool]:
        """`retries` is the total number of read-mutate-write attempts (not
        "retries after the first"), so it must be >= 1.

        If `return_changed` is True, returns `(new_value, changed)` instead
        of just `new_value` — `changed` reflects only the attempt that
        actually decided the outcome (either the one that wrote, or the one
        whose freshly-read value already matched, in which case no write
        happened and `changed` is `False`). Callers that need to know
        whether *this call* caused a write (e.g. to report "N files
        updated") should use this instead of tracking it themselves via a
        mutate_fn closure, since a naive closure-based flag would
        incorrectly persist `True` from an earlier failed attempt across
        retries.

        THE FIRST ATTEMPT MAY SKIP THE READ. This is a GET and a PUT, and
        against a bucket a region away the GET is a bare 177 ms round trip on
        files of 8 and 35 KB -- latency, not transfer, so revalidating them
        would save nothing. When `_cached_json` is holding a copy, the copy is
        mutated and written under its own token instead, and the PUT's
        `IfMatch` is what proves it was current: a stale copy loses the
        compare-and-swap exactly as a racing writer would, and the retry below
        re-reads for real. A lost update cannot get through, because the
        condition that rejects one is the same condition either way.

        A speculative attempt does not consume a retry, and cannot answer
        "nothing changed" -- against an unproven copy that claim is not
        knowable, so a no-op mutation falls back to a real read rather than
        returning.
        """
        if retries <= 0:
            raise ValueError("retries must be >= 1")
        last_error: ConcurrentModificationError | None = None
        speculative = self._cached_json(key)
        attempts = 0
        while attempts < retries:
            if speculative is not None:
                current, token = speculative
                speculative = None
                proven = False
            else:
                attempts += 1
                current, token = self.read_json(key, default=default)
                proven = True
            before = copy.deepcopy(current)
            new_value = mutate_fn(current)
            changed = new_value != before
            if not changed:
                if not proven:
                    continue
                return (new_value, False) if return_changed else new_value
            try:
                self.write_json_cas(key, new_value, token)
                return (new_value, True) if return_changed else new_value
            except ConcurrentModificationError as exc:
                last_error = exc
        raise last_error


class LocalStorage(Storage):
    """Today's behavior: plain files on local/mounted disk, guarded by
    `filelock.FileLock` rather than any version-token check."""

    def __init__(self, root: str | Path = "."):
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        validate_key(key)
        return self.root / key

    def read_json(self, key, default=_MISSING, *, revalidate=False):
        path = self._path(key)
        if not path.is_file():
            if default is _MISSING:
                raise FileNotFoundError(key)
            return default, None
        return json.loads(path.read_text()), None

    def write_json_cas(self, key, data, expected_token):
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = filelock.FileLock(f"{path}.lock")
        with lock:
            path.write_text(json.dumps(data, indent=2))
        return None

    def update_json(self, key, mutate_fn, default=_MISSING, retries=3, return_changed=False):
        if retries <= 0:
            raise ValueError("retries must be >= 1")
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = filelock.FileLock(f"{path}.lock")
        with lock:
            if path.is_file():
                current = json.loads(path.read_text())
            elif default is not _MISSING:
                current = default
            else:
                raise FileNotFoundError(key)
            before = copy.deepcopy(current)
            new_value = mutate_fn(current)
            changed = new_value != before
            if changed:
                path.write_text(json.dumps(new_value, indent=2))
            return (new_value, changed) if return_changed else new_value

    def list_keys(self, prefix, suffixes=(".json",)):
        suffixes = tuple(s.lower() for s in suffixes)
        base = self._path(prefix)
        return sorted(
            p.as_posix() for p in base.glob("*") if p.suffix.lower() in suffixes
        )

    def get_bytes(self, key):
        path = self._path(key)
        if not path.is_file():
            raise FileNotFoundError(key)
        return path.read_bytes()

    def get_bytes_if_changed(self, key, token):
        path = self._path(key)
        if not path.is_file():
            raise FileNotFoundError(key)
        stat = path.stat()
        # Modification time AND size. Either alone repeats: a rewrite inside
        # the filesystem's timestamp granularity keeps the mtime, and a re-run
        # that re-encodes the same scene keeps the size.
        current = f"{stat.st_mtime_ns}:{stat.st_size}"
        if token is not None and token == current:
            return None, current
        return self.get_bytes(key), current

    def put_bytes(self, key, data, content_type=None):
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def delete(self, key):
        path = self._path(key)
        drop_frame(key)
        if not path.is_file():
            return False
        path.unlink()
        return True


# `get_storage()` constructs a fresh `S3Storage` wrapper on every call
# (existing, relied-on behavior -- see tests/test_storage.py), but a real
# boto3 client carries its own connection pool, and building a brand new
# one per wrapper means every S3 call pays a fresh TCP+TLS handshake
# instead of reusing a warm connection -- confirmed in production that
# even calls run concurrently via a thread pool each independently paid
# this, since none of them shared a pool to reuse. Sharing one real
# client per region (region, not bucket, since a client isn't bucket-
# specific) lets every `S3Storage` instance that didn't inject its own
# fake/mock client reuse the same warm connections.
_shared_s3_clients: dict[str | None, object] = {}
_shared_s3_clients_lock = threading.Lock()


def _get_shared_boto3_s3_client(region_name: str | None):
    with _shared_s3_clients_lock:
        client = _shared_s3_clients.get(region_name)
        if client is None:
            import boto3

            # Without an explicit region, boto3 falls back to whatever
            # the ambient environment/instance resolves to, which is not
            # necessarily where the target bucket actually lives --
            # addressing a bucket via the wrong region's regional
            # endpoint costs a redirect round trip *every single call*,
            # on top of real cross-region latency if the two differ.
            # Confirmed in production: the app's EC2 instance is in
            # ap-southeast-2, the real bucket is in ap-southeast-1, and
            # this was a large, persistent, hard-to-diagnose part of "why
            # does every S3 call feel slow" (a VPC S3 endpoint in the
            # instance's own region did not fix it, for exactly this
            # reason) -- see `get_storage()`'s `region` for where the
            # correct value actually comes from.
            client = boto3.client("s3", region_name=region_name)
            _shared_s3_clients[region_name] = client
        return client


class S3Storage(Storage):
    """ETag-based compare-and-swap on S3, via `put_object`'s
    `IfMatch`/`IfNoneMatch` parameters. `client` is injectable for testing
    (a fake, or a `moto`-mocked one) without requiring real AWS credentials
    or network access."""

    def __init__(self, bucket: str, prefix: str = "", client=None, region_name: str | None = None):
        self.bucket = bucket
        self.prefix = prefix
        self._client = client
        self._region_name = region_name

    @property
    def client(self):
        if self._client is not None:
            # Explicitly injected (a fake/mock in tests, or a caller with
            # its own client) -- always instance-specific, never touches
            # the shared cache below.
            return self._client
        return _get_shared_boto3_s3_client(self._region_name)

    def _full_key(self, key: str) -> str:
        validate_key(key)
        return f"{self.prefix}{key}" if self.prefix else key

    def _cache_key(self, full_key: str) -> tuple[str, str]:
        return (self.bucket, full_key)

    def _cached_json(self, key):
        entry = _corpus_cache.get(self._cache_key(self._full_key(key)))
        if entry is None:
            return None
        etag, body = entry
        try:
            return json.loads(body), etag
        except json.JSONDecodeError:
            return None

    def read_json(self, key, default=_MISSING, *, revalidate=False):
        from botocore.exceptions import ClientError

        full_key = self._full_key(key)
        cache_key = self._cache_key(full_key)
        cached = _corpus_cache.get(cache_key) if revalidate else None

        kwargs = {"Bucket": self.bucket, "Key": full_key}
        if cached is not None:
            kwargs["IfNoneMatch"] = cached[0]
        try:
            response = self.client.get_object(**kwargs)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if cached is not None and (code == "304" or status == 304):
                # S3 has just certified the cached bytes as current. This is
                # the whole point: freshness proved, transfer skipped.
                return json.loads(cached[1]), cached[0]
            if code in ("NoSuchKey", "404"):
                _corpus_cache.drop(cache_key)
                if default is _MISSING:
                    raise FileNotFoundError(key) from exc
                return default, None
            raise
        body = response["Body"].read()
        if revalidate:
            _corpus_cache.put(cache_key, response.get("ETag"), body)
        return json.loads(body), response["ETag"]

    def write_json_cas(self, key, data, expected_token):
        from botocore.exceptions import ClientError

        full_key = self._full_key(key)
        body = json.dumps(data, indent=2).encode("utf-8")
        kwargs = {"Bucket": self.bucket, "Key": full_key, "Body": body, "ContentType": "application/json"}
        if expected_token is None:
            kwargs["IfNoneMatch"] = "*"
        else:
            kwargs["IfMatch"] = expected_token
        try:
            response = self.client.put_object(**kwargs)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            if code in ("PreconditionFailed", "412", "ConditionalRequestConflict"):
                raise ConcurrentModificationError(f"{key!r} was modified concurrently") from exc
            raise
        etag = response.get("ETag")
        _corpus_cache.refresh(self._cache_key(full_key), etag, body)
        return etag

    def list_keys(self, prefix, suffixes=(".json",)):
        suffixes = tuple(s.lower() for s in suffixes)
        full_prefix = self._full_key(prefix)
        if full_prefix and not full_prefix.endswith("/"):
            full_prefix = f"{full_prefix}/"

        keys = []
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
            for obj in page.get("Contents", []):
                full_key = obj["Key"]
                if not full_key.lower().endswith(suffixes):
                    continue
                if "/" in full_key[len(full_prefix):]:
                    continue
                key = full_key[len(self.prefix):] if self.prefix else full_key
                keys.append(key)
        return sorted(keys)

    def get_bytes(self, key):
        from botocore.exceptions import ClientError

        full_key = self._full_key(key)
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=full_key)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            if code in ("NoSuchKey", "404"):
                raise FileNotFoundError(key) from exc
            raise
        return response["Body"].read()

    def get_bytes_if_changed(self, key, token):
        from botocore.exceptions import ClientError

        full_key = self._full_key(key)
        kwargs = {"Bucket": self.bucket, "Key": full_key}
        if token:
            kwargs["IfNoneMatch"] = token
        try:
            response = self.client.get_object(**kwargs)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if token and (code == "304" or status == 304):
                # S3 has just certified the held bytes as current.
                return None, token
            if code in ("NoSuchKey", "404"):
                raise FileNotFoundError(key) from exc
            raise
        return response["Body"].read(), response.get("ETag")

    def put_bytes(self, key, data, content_type=None):
        full_key = self._full_key(key)
        kwargs = {"Bucket": self.bucket, "Key": full_key, "Body": data}
        if content_type is not None:
            kwargs["ContentType"] = content_type
        self.client.put_object(**kwargs)

    def delete(self, key):
        full_key = self._full_key(key)
        # head_object first so the return value means what it says. S3's
        # delete_object succeeds on a key that never existed, which would
        # make "did this remove anything" unanswerable.
        try:
            self.client.head_object(Bucket=self.bucket, Key=full_key)
        except Exception:  # noqa: BLE001 - any failure to find it means absent
            _corpus_cache.drop(self._cache_key(full_key))
            drop_frame(key)
            return False
        self.client.delete_object(Bucket=self.bucket, Key=full_key)
        _corpus_cache.drop(self._cache_key(full_key))
        drop_frame(key)
        return True


def get_storage() -> Storage:
    backend = os.environ.get("PPE_STORAGE_BACKEND", "local")
    if backend == "s3":
        bucket = os.environ.get("PPE_S3_BUCKET")
        if not bucket:
            raise RuntimeError(
                "PPE_STORAGE_BACKEND=s3 requires PPE_S3_BUCKET to be set "
                "explicitly. No default bucket is used, to avoid "
                "accidentally hitting production from a dev machine."
            )
        # `safety-model-dataset-dev` (the only real bucket this project
        # uses today) actually lives in ap-southeast-1 -- see S3Storage's
        # `client` property for why getting this right matters far more
        # than it looks like it should. Overridable via PPE_S3_REGION in
        # case a future bucket lives somewhere else.
        region = os.environ.get("PPE_S3_REGION", "ap-southeast-1")
        return S3Storage(bucket=bucket, region_name=region)
    return LocalStorage()


def update_json(
    key: str,
    mutate_fn: Callable[[JsonValue], JsonValue],
    default: JsonValue = _MISSING,
    retries: int = 3,
    return_changed: bool = False,
) -> JsonValue | tuple[JsonValue, bool]:
    """Module-level convenience wrapper around `get_storage().update_json`."""
    return get_storage().update_json(key, mutate_fn, default=default, retries=retries, return_changed=return_changed)
