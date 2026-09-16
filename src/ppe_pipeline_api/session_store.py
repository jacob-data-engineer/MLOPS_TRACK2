"""In-memory per-session state, keyed by the token issued at `/auth/login`.

Holds the eval one-by-one stepper's live `JudgeClient` + dataset + index +
pending judge-call state; the currently-loaded gold-set dataset +
annotations; the few-shot library currently open for editing; and Track1's
rolling per-session results. Per the architect's decision, this is
deliberately simple in-memory state in this one FastAPI process — no
external session store, no horizontal scaling — acceptable for a ~7-person
internal tool. This ties the deployment to a single process: running more
than one worker/replica would give each its own, disjoint set of sessions.

Each session's dict has no fixed schema; routers get/set whichever keys
they own (`rolling_results`, `session_usage`, `eval_stepper`,
`eval_bulk_report`, `fewshot`, `goldset`, ...) — one key namespace per
screen.
"""

from __future__ import annotations

import threading
import uuid


class SessionStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, dict] = {}
        self._session_locks: dict[str, threading.Lock] = {}

    def create(self) -> str:
        token = uuid.uuid4().hex
        with self._lock:
            self._sessions[token] = {}
            self._session_locks[token] = threading.Lock()
        return token

    def get(self, token: str) -> dict | None:
        with self._lock:
            return self._sessions.get(token)

    def all_sessions(self) -> list[dict]:
        """Every live session's dict, as a snapshot list.

        For the rare thing that belongs to the PROCESS rather than to one
        session. The live feed is the only one today: it holds a process-wide
        lock so at most one runs, any session has to be able to see and act on
        it, and the object the worker thread is holding lives in whichever
        session happened to press Start.
        """
        with self._lock:
            return list(self._sessions.values())

    def lock_for(self, token: str) -> threading.Lock:
        """A per-session lock for critical sections that mutate a session's
        state across several steps (e.g. the eval stepper's judge-then-store
        sequence) — guards against two requests for the *same* session
        token (e.g. a double-click) interleaving, not a substitute for the
        storage layer's own CAS-based concurrency control across sessions."""
        with self._lock:
            return self._session_locks.setdefault(token, threading.Lock())


SESSIONS = SessionStore()
