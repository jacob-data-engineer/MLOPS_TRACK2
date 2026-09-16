# PPE Safety Judge — web frontend

React + TypeScript + Vite frontend for the PPE Safety Violation Judge
tool. This is the only UI, and **it is what the production deployment
serves** (see the root `README.md`'s "Deployment" section, and
`Dockerfile`).

Presentation layer only. It talks to the FastAPI backend in
`src/ppe_pipeline_api/` (see `CONTRACT.md` there for the endpoint list),
which in turn calls into `ppe_pipeline` — no judge/eval/storage logic
lives in this directory.

## Running it locally

You need both processes running:

```bash
# 1. FastAPI backend (from the repo root)
PPE_STORAGE_BACKEND=local PPE_APP_PASSWORD=<pick-a-dev-password> uv run uvicorn ppe_pipeline_api.main:app --reload

# 2. This frontend
cd web
npm install
npm run dev
```

Open `http://localhost:5173` and log in with whatever password you set for
`PPE_APP_PASSWORD`. The backend must have that env var set — it fails
closed (503) if it's unset.

**Always set `PPE_STORAGE_BACKEND=local` when running this locally.** The
repo's `.env` sets `PPE_STORAGE_BACKEND=s3` for production, and `config.py`
loads `.env` automatically — an un-overridden local server silently reads
*and writes* the real production S3 bucket. This already happened once
during this build (a local Track1 test run appended fake entries to the
real production audit log; caught and cleaned up, but don't repeat it):
any mutating endpoint you exercise locally (`POST /track1/run`,
`PUT/DELETE /fewshot/entries/...`, gold-set annotate/freeze, etc.) writes
for real unless the backend is explicitly local.

The frontend's API base URL defaults to `http://localhost:8000` in dev.
Override it by creating `web/.env.local`:

```
VITE_API_BASE_URL=http://your-api-host:8000
```

**Production doesn't use this at all.** `VITE_API_BASE_URL` is a Vite
build-time value, baked into the JS at `npm run build` — the production
`Dockerfile` builds with `VITE_API_BASE_URL=""` (empty string, not unset)
so the built SPA calls relative paths and hits whichever origin/port
served it, since `main.py` serves both the API and the built SPA from one
process on one port (see root `README.md`'s "Deployment"). The FastAPI
app's CORS config only allows `http://localhost:5173` (see
`src/ppe_pipeline_api/main.py`) — that's for local dev only (`npm run dev`
on a separate port than the backend); production's same-origin requests
never hit CORS at all, so this doesn't need to change for deployment.

## What's built so far

All six nav sections (`src/components/Layout.tsx`) have a real
implementation:

| Section | Status |
|---|---|
| Run pipeline | **built** — `src/routes/Track1Page.tsx` (dataset-based alarm source only; no upload-manually/mock-detection flow yet, see below) |
| Rolling metrics | **built** — `src/routes/MetricsPage.tsx` |
| Evaluate against ground truth | **built** — `src/routes/EvalPage.tsx` |
| Few-shot library review | **built** — `src/routes/FewshotPage.tsx` |
| Gold Set review | **built** — `src/routes/GoldsetPage.tsx` |
| Model deployment status | **built** — `src/routes/ModelsPage.tsx` |

The first two (Metrics, Models) were picked as the walking skeleton because
they're read-only, single-request, no multi-step session state. Few-shot
and Track1 were the first screens with real session state and mutations
(add/edit/remove, a judge-calling action). Eval and Gold Set (this pass)
were the most stateful: the bulk eval stepper reflects
`session["eval_stepper"]`'s server-held index/live judge client/pending
result exactly (`GET /eval/stepper/state` is polled, never re-derived
client-side), and Gold Set's dual-review reflects each case's
server-computed role/rotation/owned-fields (`gold_set.role_for`) rather
than reimplementing that assignment logic in the browser.

Track1's "Run pipeline" page intentionally covers only the "pick an alarm
from an eval dataset" source, not a manual-upload branch —
there's no live camera feed anywhere in this codebase to build a richer
manual-upload/mock-detection flow against yet, and the dataset-based path
is what reviewers actually spot-check with day to day. `POST
/track1/detections/parse` and `POST /track1/detections/mock` are already
in the API contract if that flow gets scoped in later.

**Run pipeline's judge**: `POST /track1/run-job` uses `ppe_pipeline`'s own
"Few-shot / filter / cropped" judge (few-shot examples, cropping, the
too-near-forklift geometric filter) — the same approach Eval's harness
uses — not the separate `hazard_validation` (real Track1 worker) package.
That package's own code is untouched and still what the real live
worker runs; only what this review app's "Run pipeline" screen calls
changed. See SCRUM-28 for the (not yet started) follow-up to port this
into the live worker itself.

**Resumable background jobs + state persistence.** Track1's run and both
Eval modes (Simple and Advanced) execute as server-tracked background
jobs (`POST /track1/run-job`, `POST /eval/bulk-run`, `POST
/eval/advanced-run` — see `CONTRACT.md`), polled every ~2s from
`BulkRunPanel`/`SimpleBulkRunPanel`, and every route's form
selections/loaded state persist via `hooks/useSessionState.ts`
(sessionStorage-backed). All of this exists specifically so navigating to
another tab and back — which remounts the route — resumes an in-progress
or just-finished job and restores what was selected, instead of silently
losing it (a real problem this session hit against production: an
orphaned blocking request kept running server-side with nowhere for the
result to go once its component unmounted). Gold Set additionally warns
before losing an unsaved annotation draft to sidebar navigation
(`context/NavGuardContext.tsx`, wired into `Layout.tsx`'s nav links) and
offers to restore it if you navigate away and back.

## Structure

```
src/
  lib/api.ts            fetch wrapper + typed request/response shapes,
                         kept in step with CONTRACT.md
  context/
    AuthContext.tsx      session token, held in sessionStorage + an
                         in-memory var api.ts attaches as Authorization
    NavGuardContext.tsx   lets a page register "do I have unsaved
                         changes?" so Layout.tsx's sidebar links can warn
                         before navigating away -- Gold Set is the one
                         consumer today
  hooks/
    useApiImage.ts        loads an authenticated `.../image` endpoint as a
                          blob object URL (plain `<img src>` can't carry
                          the Authorization header those need)
    useSessionState.ts    sessionStorage-backed useState -- every route's
                          form selections/loaded state use this so
                          navigating away and back (which remounts the
                          route) restores rather than resets them; every
                          restored value is re-validated against a fresh
                          server fetch before being trusted
  components/
    Layout.tsx           persistent sidebar nav (the six domains); nav
                         links consult NavGuardContext before navigating
    RequireAuth.tsx       route guard, redirects to /login
    Card.tsx, Badge.tsx, StatTile.tsx, OutcomeBarChart.tsx
                          the shared building blocks every screen composes
    DetectionImage.tsx    image + SVG box overlay for Track1's alarm preview
                          (also reused, boxes-less, for Gold Set's already-
                          annotated full/crop images and the eval stepper's
                          already-annotated crop)
    FewshotEntryForm.tsx  verdict/reason/confidence/explanation fields,
                          shared by "edit an example" and "add an example"
    CaseStepper.tsx        case i-of-n position readout, with or without
                          Previous/Next -- the eval stepper (forward-only)
                          and Gold Set's case nav (free Previous/Next) share it
    ReviewerRoleBadge.tsx  Gold Set: current identity's role on this case,
                          the full round-robin rotation, and the agreement Badge
    GoldsetReviewerPanel.tsx  Gold Set: one reviewer slot (A or B) -- locked
                          name + label/rationale, editable only in your own slot
    AdjudicationPanel.tsx  Gold Set: the six adjudicator-owned fields that
                          resolve a disagreement to a final label
    DatasetPicker.tsx      named-dataset dropdown (Gold Set v1/v2 first,
                          then other real files) + a free-text "Other"
                          escape hatch -- shared by every screen that
                          used to have a raw path field
    VerdictStamp.tsx       the one place a judge/adjudicated verdict
                          renders anywhere (a literal stamped word, not a
                          colored pill) -- see "Design direction" below
    BulkRunPanel.tsx       generic polling/progress/per-case-image-list UI
                          for a background job (start, poll, show rows as
                          they complete, cancel, done/error/cancelled
                          states) -- parameterized by which job's
                          endpoints to call
    SimpleBulkRunPanel.tsx  a thin wrapper over BulkRunPanel for Eval's
                          Simple mode specifically (auto-finalizes with no
                          correction step); Advanced mode's job uses
                          BulkRunPanel directly since it still needs the
                          manual ground-truth review step for a
                          single-method run
    EvalReportTable.tsx    the precision/recall/F1/accuracy table, used by
                          both Eval modes' "done" states
  routes/
    LoginPage.tsx
    MetricsPage.tsx, ModelsPage.tsx        the read-only walking skeleton
    FewshotPage.tsx, Track1Page.tsx        session state + mutations
    EvalPage.tsx, GoldsetPage.tsx          the two most stateful screens
```

## Design direction

Customs case-file & manifest stamps — a committed visual identity, not the
restrained blue-accent/status-triad dashboard this frontend shipped in its
first pass. Full rationale lives as a comment block at the top of
`src/App.tsx` (THESIS/OWN-WORLD/STORY/FIRST VIEWPORT/FORM/FINISH).

- **One real saturated color in the whole system**: alarm-red ink
  (`--color-critical` in `src/index.css`), spent on stamped verdicts
  (`VerdictStamp.tsx` — TP/FP/uncertain, Track1's
  CONFIRM_TP/CONFIRM_FP/ESCALATE) and on "needs a human's eyes now"
  feedback (form/API errors), which shares that register. Everything else
  — page background, cards, hairlines, primary-action buttons, nav — stays
  in manila/kraft/carbon-paper neutrals. There is no green/amber status
  triad anymore and no second (blue) accent; `Badge.tsx`'s non-verdict
  pills (deployment/approval status, pass/fail) are ink-weight (solid /
  outline / muted), never color-coded.
- **Verdicts are stamped, not pilled.** A verdict renders as a literal
  stamped word — CLEARED / HOLD / ESCALATE — in `VerdictStamp.tsx`, in the
  one alarm-red ink, rotated and bordered like a rubber stamp. The moment
  a verdict is *confirmed* (a Track1 run completing, a few-shot verdict
  save), it plays a landing motion (`stamp-landing` in `src/index.css`):
  oversized and rotated, overshoots, settles. Historical rows in a ledger
  table render the same stamp at rest (`landed={false}`) — no replay on
  every re-render.
- **Typewriter-monospace for data, plain sans for prose.** `.tabular`
  (already applied to every case ID, count, and tabular-nums figure) now
  also sets `font-family: var(--font-mono)` — a real monospace stack
  (`ui-monospace`/IBM Plex Mono/system fallbacks), so this propagated to
  every existing call site for free. Body copy and labels stay on the one
  workhorse sans (`system-ui` stack) — no display face.
- **Ledger-row tables, not card grids or colored bar charts**, for every
  dense list: few-shot entries, eval-run rows, model version history,
  Track1's rolling results. `OutcomeBarChart.tsx` (an aggregate
  distribution, not a single case's verdict) renders its three rows in
  one neutral ink tone labeled with the same stamp words, rather than
  reintroducing a green/red/amber triad for a chart.
- **Two type roles, Tailwind's default size scale.** No custom `--text-*`
  tokens are defined — body copy/labels and data both use Tailwind v4's
  built-in `text-xs`…`text-xl` utilities, split only by font (plain sans
  vs. `.tabular`'s monospace), not by a bespoke scale.

Full detail (colors, type, shape, elevation, component-by-component rules,
do's/don'ts) lives in `../DESIGN.md`. Tokens live in `src/index.css` as
Tailwind v4 `@theme` custom properties (`--color-*`, `--font-*`). The old
restrained palette was replaced outright, not kept alongside this one.

## Adding a future screen

All six nav sections have a real implementation here. If a new domain is
ever added to the API, follow the pattern the six built screens
establish:

1. Add typed request/response shapes + a fetch function to `src/lib/api.ts`
   (check them against `CONTRACT.md`, then hit the real running API to
   confirm the shape — don't guess from the Python alone).
2. Build the screen in `src/routes/`, composing `Card`/`Badge`/`StatTile`/
   `OutcomeBarChart`/`VerdictStamp`/`CaseStepper` rather than one-off markup
   — add a new shared component there only if a genuinely new pattern is
   needed (see `ReviewerRoleBadge.tsx`/`GoldsetReviewerPanel.tsx`/
   `AdjudicationPanel.tsx` for what that looked like for Gold Set's
   round-robin dual review).
3. Add its route in `src/App.tsx`.
4. If the screen needs multi-step session state (like the bulk eval
   stepper or Gold Set's dual review), read how `session_store.py` and the
   relevant router already model that server-side before designing the
   client state — don't invent a second source of truth for stepper
   position, per-case annotations, or role assignments; poll the server's
   own state instead.
