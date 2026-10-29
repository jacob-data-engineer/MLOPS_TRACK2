"""FastAPI app for the React frontend: serves both the API and the built
SPA in `web/dist` from one process.

    uv run uvicorn ppe_pipeline_api.main:app --reload

Session state (the eval stepper, the currently-loaded gold set/few-shot
library, Track1's rolling results) lives in this one process's memory —
see `session_store.py`. That means exactly one FastAPI process/worker is
supported today; running more than one (e.g. `uvicorn --workers 2`, or
multiple replicas behind a load balancer) would silently split a user's
session across processes that don't share state. Fine for this ~7-person
internal tool per the architect's decision; revisit if that changes.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import FileResponse, Response

from . import auth, track2_shared
from .chain import ticker
from .routers import (
    track2_cost,
    track2_deploy,
    track2_eval,
    track2_export,
    track2_lineage,
    track2_metrics,
    track2_models,
    track2_notifications,
    track2_pull_control,
    track2_retention,
    track2_review,
    track2_schedule,
    track2_sql,
)


def background_enabled() -> bool:
    """False in a container started with APP_BACKGROUND=off.

    Every container shares production's tables, bucket and queue, so its
    background work acts on real data: the startup recovery fails any
    in-flight inference/eval row it finds, and the poll loop starts queued
    runs, chains retrains and drains pulls. A container showing code still
    being built (the refactor slot on 8092) must only LOOK -- with this
    off it serves pages and leaves all of that to the stable container,
    dev and the queue Lambda. Unset means on, as before.
    """
    return os.environ.get("APP_BACKGROUND", "on").strip().lower() != "off"


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not background_enabled():
        print("APP_BACKGROUND=off: no startup recovery, no queue poll loop.")
    # Any inference/eval job still marked in-flight belongs to a process
    # that no longer exists -- this one just started, and its workers are
    # created below. Left alone, such a row reads as unfinished work
    # forever, which blocks the endpoint teardown and bills until someone
    # notices. Runs before the workers so it can never race one.
    if background_enabled():
        try:
            track2_shared.recover_interrupted_jobs()
        except Exception as exc:  # never block startup on this
            print(f"Could not recover interrupted jobs: {exc}")
    # Advances the ad-hoc Track2 training queue in the background -- see
    # track2_shared.py's module docstring on why this state is in-memory,
    # single-process only, same as session_store.py.
    poll_task = (asyncio.create_task(ticker.queue_poll_loop())
                 if background_enabled() else None)
    # Fill the registry cache before anybody asks for it.
    #
    # Review & Metrics lists every model package group and describes every
    # version in them. Cached, that is instant -- but the FIRST load after
    # a deploy pays the whole 9-12 seconds, and a deploy ends every
    # session, so the first person back in always pays it. Doing it here
    # means they do not.
    #
    # In a thread, not awaited: it is blocking boto3, and the point is
    # that startup does not wait for it. Failure is ignored on purpose --
    # this is a warm-up, and the routes all still work cold.
    # Once now, then again just inside the cache's lifetime for as long as
    # the app is up. The one-off warm-up alone was not enough: the entries
    # aged out before anybody arrived and the first request paid the full
    # price regardless -- 7.6 seconds, measured over HTTP against the
    # warmed process.
    warm_task = asyncio.create_task(asyncio.to_thread(track2_shared.warm_registry_cache))
    refresh_task = asyncio.create_task(track2_shared.registry_warm_loop())
    try:
        yield
    finally:
        if poll_task is not None:
            poll_task.cancel()
        warm_task.cancel()
        refresh_task.cancel()


app = FastAPI(title="Track2 MLOps API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    # Vite's default dev server origin -- the production deploy serves the
    # built SPA from this same app/port (see below), so those requests are
    # same-origin and never hit CORS at all; this only matters for local
    # `npm run dev` against a separately-run backend.
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.post("/auth/login", response_model=auth.LoginResponse, tags=["auth"])(auth.login)
# Probed by the browser whenever any other call 401s, to tell a dead
# session from a one-off failure before signing anyone out.
app.get("/auth/session", tags=["auth"])(auth.check_session)

# Track2 only. The original PPE judge routers (track1, metrics, eval,
# fewshot, goldset, models, escalations, helmet-cv) were removed along with
# the pages that called them. The ppe_pipeline library they were built on
# stays -- the container's own smoke test imports it, and it is what the
# judge itself runs on.
app.include_router(track2_review.router, prefix="/track2/review", tags=["track2-review"])
app.include_router(track2_models.router, prefix="/track2/models", tags=["track2-models"])
app.include_router(track2_schedule.router, prefix="/track2/schedule", tags=["track2-schedule"])
app.include_router(track2_lineage.router, prefix="/track2/lineage", tags=["track2-lineage"])
app.include_router(track2_metrics.router, prefix="/track2/metrics", tags=["track2-metrics"])
app.include_router(track2_deploy.router, prefix="/track2/deploy", tags=["track2-deploy"])
app.include_router(track2_notifications.router, prefix="/track2/notifications", tags=["track2-notifications"])
app.include_router(track2_export.router, prefix="/track2/export", tags=["track2-export"])
app.include_router(track2_cost.router, prefix="/track2/cost", tags=["track2-cost"])
app.include_router(track2_eval.router, prefix="/track2/eval", tags=["track2-eval"])
app.include_router(track2_pull_control.router, prefix="/track2/pull-control", tags=["track2-pull-control"])
app.include_router(track2_retention.router, prefix="/track2/retention", tags=["track2-retention"])
app.include_router(track2_sql.router, prefix="/track2/sql", tags=["track2-sql"])


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/build")
def build() -> dict:
    """Which image this container is running, and what it is for.

    Unauthenticated on purpose: it carries no data, and the one moment
    you most need to know which build you are looking at is before you
    have logged into it.

    Two containers may run side by side -- a stable one and a development
    one, on different ports, against the same tables. That is only safe
    while the difference is visible, which is what this is for.
    """
    import os as _os

    return {
        "tag": _os.environ.get("APP_BUILD_TAG", ""),
        "role": _os.environ.get("APP_ROLE", ""),
        "port": _os.environ.get("APP_PORT", ""),
    }


# Serve the built React SPA (Dockerfile's frontend-builder stage) --
# present only in the production image; local dev serves the frontend via
# `npm run dev` on its own port instead, so this mount is simply absent
# then. Registered last so every API route above still wins; any path
# that isn't a real file under `web/dist` (a React Router route like
# `/eval`, a deep link, a reload) falls through to `index.html`, which is
# what a single-page app needs to handle client-side routing itself.
_WEB_DIST = Path(__file__).resolve().parent.parent.parent / "web" / "dist"

if _WEB_DIST.is_dir():

    @app.exception_handler(HTTPException)
    async def _page_request_gets_the_app(request: Request, exc: HTTPException) -> Response:
        """A browser asking for a PAGE gets the app, not an API error.

        Ten of the SPA's routes are spelled exactly like the API routers
        that back them -- `/track2/models` is both the page and the data
        behind it. The routers are registered first and FastAPI takes the
        first match, so the catch-all below never runs for those paths:
        typing the URL, reloading, or opening a bookmark returned
        `{"detail": "Missing or malformed Authorization header."}` where
        the page should have been. Clicking through the nav always worked,
        because React Router never asks the server -- which is why this
        survived so long.

        Discriminated on Accept. A top-level browser navigation sends
        `text/html`; `fetch` with no Accept set -- which is every call the
        SPA makes -- sends `*/*`. So the app's own requests still get
        their JSON 401 and its session-recovery still runs, and only a
        human pointing a browser at a page gets the page.

        Registered inside this block on purpose: with no built frontend
        there is nothing to serve, and an API-only deployment should keep
        answering 401 in JSON.
        """
        if exc.status_code in (401, 403) and "text/html" in (
            request.headers.get("accept") or ""
        ):
            return FileResponse(
                _WEB_DIST / "index.html", headers={"Cache-Control": "no-cache"}
            )
        return await http_exception_handler(request, exc)

    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_spa(full_path: str) -> FileResponse:
        candidate = (_WEB_DIST / full_path).resolve()
        # Vite content-hashes every file under assets/ (a changed file gets
        # a new filename), so those can be cached forever -- but index.html
        # is what points at the current hash, and was being served with NO
        # Cache-Control header at all, letting browsers apply their own
        # (often generous) heuristic and keep serving a stale build after a
        # redeploy with no way to tell short of a hard refresh. `no-cache`
        # still lets the browser cache it, just forces a cheap revalidation
        # (a 304, via the ETag FileResponse already sends) on every load.
        if candidate.is_file() and _WEB_DIST in candidate.parents:
            if candidate.parent == _WEB_DIST / "assets":
                return FileResponse(candidate, headers={"Cache-Control": "public, max-age=31536000, immutable"})
            return FileResponse(candidate, headers={"Cache-Control": "no-cache"})
        return FileResponse(_WEB_DIST / "index.html", headers={"Cache-Control": "no-cache"})
