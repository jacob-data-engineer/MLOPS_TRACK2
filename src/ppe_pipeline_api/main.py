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
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from . import auth, track2_shared
from .routers import (
    track2_cost,
    track2_deploy,
    track2_eval,
    track2_export,
    track2_lineage,
    track2_metrics,
    track2_models,
    track2_notifications,
    track2_review,
    track2_schedule,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Advances the ad-hoc Track2 training queue in the background -- see
    # track2_shared.py's module docstring on why this state is in-memory,
    # single-process only, same as session_store.py.
    poll_task = asyncio.create_task(track2_shared.queue_poll_loop())
    try:
        yield
    finally:
        poll_task.cancel()


app = FastAPI(title="PPE Pipeline API", lifespan=lifespan)

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


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# Serve the built React SPA (Dockerfile's frontend-builder stage) --
# present only in the production image; local dev serves the frontend via
# `npm run dev` on its own port instead, so this mount is simply absent
# then. Registered last so every API route above still wins; any path
# that isn't a real file under `web/dist` (a React Router route like
# `/eval`, a deep link, a reload) falls through to `index.html`, which is
# what a single-page app needs to handle client-side routing itself.
_WEB_DIST = Path(__file__).resolve().parent.parent.parent / "web" / "dist"

if _WEB_DIST.is_dir():

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
