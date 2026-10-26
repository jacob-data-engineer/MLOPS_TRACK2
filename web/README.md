# Track2 — web frontend

React + TypeScript + Vite SPA for the Track2 retraining platform. This is
the only UI, and **it is what the deployed container serves** (see the
root `README.md` and `Dockerfile`).

Presentation layer only. It talks to the FastAPI backend in
`src/ppe_pipeline_api/` — see `CONTRACT.md` there for what the API
offers, and `/docs` on a running container for the exact shapes.

The PPE judge this frontend was originally built for has been retired.
Its screens (Run pipeline, Few-shot, Gold Set, Evaluate, Escalations,
Helmet CV) and their client functions are gone, not moved.

## Running it locally

```bash
# 1. FastAPI backend, from the repo root
PPE_APP_PASSWORD=<pick-a-dev-password> uv run uvicorn ppe_pipeline_api.main:app --reload --app-dir src

# 2. this frontend
cd web
npm install
npm run dev
```

Open `http://localhost:5173` and log in with whatever you set
`PPE_APP_PASSWORD` to. The backend fails closed (503) if it is unset.

### There is no local sandbox. Read this before running the backend.

A locally-run backend talks to **the real DynamoDB tables and the real
S3 bucket**. There is no local storage mode: the `PPE_STORAGE_BACKEND`
switch that used to provide one belonged to the judge and no longer
exists anywhere in this repo.

So a local server is a third window onto production data, alongside the
stable and development containers. Pressing a button that pulls, purges,
trains or deploys moves real files and starts real jobs. Read-only
browsing is fine; treat every write as live, because it is.

The test suite has its own guard for this — `tests/conftest.py` wraps
boto3 so a test cannot call `put_item`, `delete_objects` and friends for
real. That guard covers tests, not a dev server.

## API base URL

In dev the frontend defaults to `http://localhost:8000`. Override it with
`web/.env.local`:

```
VITE_API_BASE_URL=http://your-api-host:8000
```

**Production does not use this.** `VITE_API_BASE_URL` is a Vite
build-time value baked into the JS at `npm run build`, and the
`Dockerfile` builds with `VITE_API_BASE_URL=""` — an empty string, not
unset — so the built SPA calls relative paths and hits whichever origin
served it. `main.py` serves the API and the built SPA from one process on
one port, so those requests are same-origin and never reach CORS. The
backend's CORS allowlist (`http://localhost:5173`) exists only for
`npm run dev` against a separately-run backend.

## The screens

Five are offered in the sidebar (`src/components/Layout.tsx`):

| Nav entry | Route | Component |
|---|---|---|
| Track2 Multi-Auto Training | `/track2/multi-auto-training` | `Track2ModelsPage` (`variant="multi"`) |
| Track2 Review & Metrics | `/track2/metrics` | `Track2MetricsPage` |
| Track2 Deploy & Inference | `/track2/deploy` | `Track2DeployPage` |
| Track2 Notifications | `/track2/notifications` | `Track2NotificationsPage` |
| Track2 Help | `/track2/help` | `Track2HelpPage` |

Several routes are **deliberately reachable but not offered**: Training
(the same component as Multi-Auto Training), Schedule, Lineage, Export,
Cost, Evaluation and Synthetic. Dropping a nav entry keeps the route, so
a bookmark or a link in somebody's notes still resolves instead of
404ing. `Layout.tsx` records why each one left the sidebar — Schedule is
the instructive case: it drove a *second*, unrelated retrain, and having
both set meant two runs with the single-GPU guard refusing whichever
lost.

Note that an unmatched path does **not** 404 — the SPA catch-all returns
`index.html` with a 200. That is what makes the kept routes work, and it
is also why a 200 from this API proves nothing on its own.

## Structure

```
src/
  lib/
    api.ts             fetch wrapper plus every request/response type.
                       One file, deliberately: the shapes and the calls
                       that use them stay in step.
    format.ts          dates, sizes, durations
    slug.ts            detector-name slugging, mirrored from the backend
    pendingChanges.ts  "you have unsaved edits" bookkeeping
  context/
    AuthContext.tsx     session token, held in sessionStorage and in an
                        in-memory var api.ts attaches as Authorization
    NavGuardContext.tsx lets a page say "I have unsaved changes" so
                        Layout.tsx's nav links can warn first
  components/          Card, Badge, StatTile, ProgressBar, RequireAuth,
                       the error boundaries (RouteBoundary/CardBoundary),
                       the shared S3 pickers (S3FolderField,
                       Track2S3FolderBrowser) and one panel per card
  routes/              one file per screen, listed above
```

### Two conventions worth knowing

**Every S3 folder field uses the same picker.** `S3FolderField` wraps
`Track2S3FolderBrowser` — the always-open browser from the "Import a
pretrained COCO or an already-trained model" card. Any new field that
takes a bucket path uses it rather than a free-text input, so that
choosing a folder behaves identically everywhere.

**Cards fail alone.** `CardBoundary` wraps each panel and `RouteBoundary`
each route, so one card throwing leaves the rest of the page usable
instead of blanking it.

## Adding a screen

1. Add the request/response types and a fetch function to `src/lib/api.ts`.
   Check them against the running API (`/docs`), not against the Python
   source alone.
2. Build it in `src/routes/`, composing the existing
   `Card`/`Badge`/`StatTile`/`ProgressBar` rather than one-off markup.
3. Register the route in `src/App.tsx`, and add a nav entry in
   `Layout.tsx` only if it should be offered.
4. Wrap anything that can fail independently in `CardBoundary`.
