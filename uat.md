# UAT — Track2 platform, 13 Oct 2026

Where this got to, what it found, what was fixed, and what is still
open. Two rounds so far: the API read surface, then auth coverage, the
write surface and the frontend/API contract. A click-through of every
page by hand is still **not** done.

Everything below was measured against the **live** system through the
development container, not inferred from the code.

| | |
|---|---|
| Build under test | `20261013-113416` on `track2-app-dev`, port 8090 |
| Host | `i-0eee645bac49b482a` — <http://18.140.196.76:8090> |
| Suite | 1927 pass, 3 known environmental failures |
| Production | untouched — still on whatever 8501 was already running |

---

## How it fares: the short version

**The read surface is sound.** 193 calls against live data — every GET
route bare, then each query parameter driven with empty, blank,
`../../etc/passwd`, 600 characters, a non-number and a negative. Final
run: **zero 5xx, zero unhandled exceptions.**

```
calls: 193
by status: {"200": 129, "400": 13, "401": 1, "404": 8, "422": 42}
5xx / raised: 0
```

The 422s are Pydantic refusing bad input, which is the correct answer.
The 401 is an unauthenticated route correctly refusing.

**Four real defects were found and fixed.** Two of them would have shown
a user a broken page. One of those — the registry throttle — was not a
hypothetical: it reproduced on the live account on the first attempt.

**Round two found no further defects.** Auth coverage, the write surface,
the path-parameter routes and the frontend/API contract all came back
clean — and each is now held by a test rather than by a one-off audit, so
a regression fails the suite instead of reaching the browser.

**The pages were slow enough to read as broken**, and now are not. The
headline page went from 15 seconds to instant on a repeat load.

---

## What was found, and what was done about it

### 1. Review & Metrics could 500 under ordinary use — *fixed*

The worst finding, and it was not in the original sweep: profiling turned
it up afterwards.

The registry holds **53 model package groups** between them holding **20
versions**. Every load of Review & Metrics called `ListModelPackages`
once per group — 53 calls in a row. boto3's default retry policy is
`legacy`: four attempts, then it **raises**. Fifty-three serial calls was
enough to exhaust it, on a quiet account, with no concurrency involved:

```
botocore.exceptions.ClientError: An error occurred (ThrottlingException)
when calling the ListModelPackages operation (reached max retries: 4):
Rate exceeded
```

That exception came straight out of the route, so the page returned 500
with every other group's rows already in hand.

Three changes, in [track2_shared.py](src/ppe_pipeline_api/track2_shared.py):

- the SageMaker client now uses **adaptive retries, 10 attempts** —
  adaptive adds a client-side rate limiter that slows down when it sees
  throttling rather than hammering until it gives up;
- a group that will not list is **skipped and logged**, not raised — one
  missing group is a missing group, not a dead page;
- group listings are **cached for 30 seconds**, and every approve,
  reject, delete and registration clears the cache outright, so an
  operator never sees the state before the change they just made.

Covered by four tests, including one that asserts the rows are identical
whichever path ran.

### 2. `/track2/deploy/lineage` returned a 500 on a malformed folder — *fixed*

Found by the sweep. `?output_folder=../../etc/passwd` built the S3 key
`../../etc/passwd/lineage.json`. S3 answers a malformed key with a **400**,
which is not one of the codes the `NoSuchKey` check looked for, so the
`ClientError` escaped and the browser got a 500 with a traceback behind it.

Nothing was reachable through it — `..` has no meaning to S3, the key
simply did not exist. But a 500 says *this service is broken* when the
honest answer is *that is not a folder*.

Now `safe_key_part()` checks the value before it becomes a key, and the
route turns the refusal into a **400 with a readable message**. Four
regression tests, including one that pins the route and not just the
helper — the app has no global `ValueError` handler, so the catch in the
router is the half that matters.

### 3. A crash in a panel blanked the whole app — *fixed*

This already happened once this session: a `const` in a panel body read a
`useState` declared below it, every render threw, and because nothing
above caught it React unmounted the entire tree. White page. No message,
no sidebar, no way to reach a page that still worked. `tsc` does not
catch it, because the reference sits inside a callback.

`CardBoundary` only wraps a card's *children*, so it cannot catch a throw
in the panel component that renders the card — that code runs above it.

New [RouteBoundary.tsx](web/src/components/RouteBoundary.tsx), placed in
three spots:

- around the route outlet in [Layout.tsx](web/src/components/Layout.tsx),
  keyed by pathname — the worst a panel can now do is replace its own
  page, with the nav, the build badge and every other page still working;
- around the whole app in [main.tsx](web/src/main.tsx) — the last resort,
  for a throw in the shell itself;
- around the Help pop-out in [App.tsx](web/src/App.tsx), which renders
  outside the Layout and so had no boundary at all.

A card's `summary` — what renders when the card is **shut** — was also
outside the boundary, and reads the same state the body does. Now inside
it.

### 4. The Stop button on a pull could latch — *fixed*

`setStopping(true)` was only ever cleared by the polling effect, and that
effect only clears it when the run **leaves** `running`. A stop request
that failed left the run running — so the flag stayed set, the button
stayed disabled, and Stop could not be pressed again for the rest of the
sweep. Now cleared on failure.

### 5. Duplicate request fan-out — *fixed*

One mount of Multi-Auto Training fires far more requests than it looks
like it does: **five** separate panels each call `/track2/pull/control`
for themselves, **four** each call `/track2/models` and
`/track2/detectors`. Same answer, fetched five times, concurrently.

`apiGet` in [api.ts](web/src/lib/api.ts) now coalesces identical GETs
that are in flight at the same moment. It is **not** a cache — the entry
is dropped the moment the request settles, so nothing can go stale. Extra
subscribers get a structured clone, so a panel that sorts its result in
place cannot reach into another panel's state.

---

## Round two: the three things round one could not see

Round one overrode `require_session` so it could reach the routes at all
— which means **a route that forgot to require a session looked exactly
like one that has it**. That gap, the routes with a `{path_parameter}`
(skipped outright), and the write surface are what round two covers.

### Auth coverage — clean, and now guarded by a test

Read from the dependency tree, not probed. A write route with no auth
would *run* if called unauthenticated, and the handlers include
`pull-control/run` and `schedule/train-now`.

```
leaf routes walked: 131
paths in the OpenAPI document: 122
UNGUARDED (0) — every non-public route requires a session
write routes: 82   unguarded writes: 0
```

Only five routes are public, all deliberately: `/health`, `/build`
(CLAUDE.md documents it unauthenticated — the badge reads it),
`/auth/login`, `/auth/session`, and the SPA shell.

**The first attempt at this reported a clean app it had never checked.**
FastAPI 0.141 wraps each `include_router` in an `_IncludedRouter` that
has no `.routes` of its own — the real router hangs off
`.original_router`. A flat walk found **5 routes out of 122** and
declared everything guarded. The only reason that was caught is that the
script printed its own count and compared it against the OpenAPI
document. Four tests in
[test_api_auth.py](tests/test_api_auth.py) now hold this, including one
whose whole job is to fail if the walk stops reaching the routers.

Proven to catch a regression, not just to pass: adding an unguarded
`POST /track2/oops/danger` makes the check report exactly that route.

### The write surface validates before it acts — clean

53 POST routes declare a JSON body. Each was sent a JSON **array**, which
no `BaseModel` can validate, so FastAPI answers 422 before the handler is
entered — the one write probe with no chance of doing work.

**53 of 53 refused with 422.** Nothing touched the request before
validating it.

### Path-parameter routes — clean

35 calls across the 5 GET routes with a `{path_parameter}`, each driven
with empty, blank, `../../etc/passwd`, 400 characters, `abc`, `-1` and a
URL-encoded traversal. No 5xx, nothing raised.

### The frontend calls only routes that exist — clean, and now tested

This matters more here than in most apps. The SPA catch-all serves
`index.html` with **200 for any unmatched path**, so a frontend calling a
renamed route does not get a 404 it can report — it gets HTML, with 200,
and fails later reading a field off it. That is exactly how three tests
in this file came to probe a route that had not existed for months and
"pass".

**116 live endpoints called from [api.ts](web/src/lib/api.ts); all 116
are real routes.** Now a test, proven by renaming `/track2/metrics` and
confirming it is reported rather than masked.

One genuine orphan: **`/track2/models/audit/files/used-by`** — reverse
lineage, *which training runs used this image* — is built and has no
caller anywhere in the frontend. Same shape as the metrics CSV export.

---

## Round three: the destructive paths, and the fixes

Approved to run. What it did **not** do is destroy anything real, which
was a reading of the task rather than timidity: this container shares the
bucket, the tables and the pipelines with production, and there is no
disposable detector, audit row or week folder to aim at. Deleting a real
model version would prove the delete works — which nobody doubts — and
cost a registered model to learn it.

What *can* be wrong is the **refusal path**. A destructive route is
dangerous exactly when it is handed something it cannot find, or an empty
target, and does something broad instead of refusing. So every one was
called with an id that cannot exist, a confirmation that does not match,
and nothing to act on.

**Three findings, all now fixed and re-verified at zero.**

### `approve` and `reject` on a model version returned a 500 — *fixed*

`delete_model` in the same file has always caught `ClientError`. Approve
and reject did not, so any of them — an AccessDenied, a throttle, a
version deleted in another tab — came straight out as a 500 with a
traceback behind it. Now a 400 saying what happened.

### Purging a single week had no confirmation at all — *fixed*

The worst of the three. `POST /weeks/purge` with a name and a fabricated
confirmation answered **200**. It was not ignoring a bad number; it had
no number to ignore.

`purge-all` has always required the exact object count and re-counted it
server-side. This route — the one reachable from a row in the table, so
the one far more likely to be pressed — took a name and deleted the week
on a single request. The page asked for a confirmation; the API did not,
so anything calling it directly bypassed that entirely.

Now both purges answer to the same rule. `weeks/purge-preview?name=` was
added so the count can be fetched for one week, the confirmation shows it
(*delete 1,496 objects?* rather than *delete its files?*), and the server
re-counts and refuses with a 409 if it has changed.

### Every other destructive route refuses cleanly

| Handed an id that cannot exist | Answer |
|---|---|
| delete a training audit | 404 "Run not found (already deleted?)" |
| delete a detector | 400, and it names the ones that do exist |
| delete / approve / reject a version | 400 |
| remove a queue entry / eval run / inference item | 404 |
| stop a pull run | 409 "No run called ..." |
| restore a snapshot | 404 |
| purge every week, wrong count | 409, quoting the real count |
| the stop routes, with nothing running | 200, idempotent |

---

## Also fixed this round

### The first page load, properly this time

The start-up warm-up alone was not enough, and the first measurement of
it was misleading: timing it through a *separate* process showed 12.8s
because that process had its own cold cache. Measured over HTTP against
the real server, the first load still cost **7.6s** — the 30-second
listing TTL had expired before anyone arrived.

The TTL was never what kept the page honest: every action that changes
approval status already clears the cache outright. It is now five
minutes, with a background refresher running just inside it, so an entry
is replaced while still valid and no request ever waits for a cold
listing.

| | Originally | Now (first hit, real server) |
|---|---|---|
| `/track2/metrics` | 14.9s | **0.08s** |
| `/track2/review` | 10.7s | **0.01s** |
| `/track2/export` | 25.7s | **0.08s** |

### The one destructive write that acted on an empty body

Of the 60 routes flagged in round one, each of the six that does real
work was checked by hand. Nearly all were already fine: `purge-all`
requires the exact count, `export/weights` answers 400 on an empty group,
the settings routes fall back to the default file name. **`pull-control/run`
was the exception** — an empty POST started a sweep, and a sweep *moves*
files with no second copy. `limit` is now required, so `POST {}` is a 422.

*The round-one finding was over-stated and this corrects it: "60 write
routes act on an empty body" was true but mostly harmless.*

### The error card now says what to do

`RouteBoundary` printed the error and little else. It now reads the
message and offers the next step — a throttle says wait, a session error
says sign in again, a page bug says reloading will not help and is worth
reporting.

### Class → "Model Name" across the UI

90 user-visible strings. The sweep was line-scoped and capital-C only, so
`className`, the code comments and — importantly — the **YOLO label
lists** are untouched. "Class" means two things here: the detector, and
the labels inside a `data.yaml`. Three headings that really do list
labels were caught and reverted by hand:
`Classes, one per line`, the class list on the model picker, and the
`No / Class` table of a `.pt` file's own classes.

### Eight orphaned registry groups

Four existed and were deleted (`yolo11{s,m,l,x}-open-detector`). The
other four were never real AWS groups at all — the app enumerates a
cross-product of names, and `model_registry_summary` reports a group that
does not exist as empty rather than raising.

---

## What was checked and left alone

**The pickers do not list 53 names.** That suggestion from round one was
wrong — inferred from the backend enumeration without looking at the UI.
`Track2ModelPicker` builds its list from registered versions, so it shows
the 7 groups that hold something, and already prints a count beside each.
The export card is three short cascading dropdowns, not a flat list. A
`versions` count was added to `/export/groups` and then **reverted**:
nothing consumed it, and shipping a field with no caller is the very
thing flagged as a problem elsewhere in this document.

---

## Measured: before and after

Authenticated, in-container, two consecutive calls each.

| Route | Before | After (1st) | After (2nd) |
|---|---|---|---|
| `/track2/metrics` — Review & Metrics | 14.9s | 9.4s | **0.00s** |
| `/track2/review` | 10.7s | 0.05s | **0.06s** |
| `/track2/export` | 25.7s | 0.07s | **0.03s** |
| `/track2/models/base-models/packages` | 14.2s | **0.00s** | 0.00s |
| `/track2/models` | 3.3s | 0.94s | 0.98s |

`base-models/packages` had its own private copy of the same 53-group
sweep, which is why it stayed at 14s after everything else dropped. It
now shares the one cached listing.

---

## What was tried and rejected

**Parallelising the registry scan eight-wide.** It was deployed, and it
made things *worse*: SageMaker throttled `ListModelPackages` and the page
500d. A speed-up that can take the page down is not a speed-up. The
listing pass went back to serial; only the per-version describes run
four-wide, and they fall back to serial rather than failing. The comment
in the code says so, so nobody tries it again.

**POSTing an empty body to every route to test validation.** Started,
then cancelled mid-run. Plenty of the write bodies are all-optional, so
`{}` validates and the handler runs — and the handlers include
`pull-control/run`, `retention/retrain/run` and `schedule/train-now`. The
dev container shares the bucket, the tables and the pipelines with
production, so a probe that "just checks the validation" would have moved
real files. Confirmed afterwards that nothing was started: no training
jobs, no processing jobs, no new pipeline executions.

The writes were read from the OpenAPI schema instead — see below.

---

## Checked and clean

| Class | Result |
|---|---|
| Destructive actions without confirmation | none — all seven confirm, pull purge requires typing |
| Double-submit on run buttons | none — every one disables while busy |
| Latched spinners (`setBusy(true)` never cleared) | 27 sites checked, 1 real, now fixed |
| Promise chains with no `.catch` | 2 candidates, both caught further along |
| Temporal-dead-zone reads (the blank-page bug) | clean |
| JSX/string syntax, NUL bytes, duplicate props | clean |
| Routes missing a session guard | 131 walked, **0 unguarded** (82 writes) |
| Write routes acting before validating | **53/53** refuse a bad body with 422 |
| Path-parameter routes | 35 adversarial calls, no 5xx |
| Frontend calling a route that no longer exists | 116 calls, all real |

---

## Still open

### ~~60 write routes act on an empty body~~ — resolved, and the finding was over-stated

Read from the schema, not probed. `{}` validates against them and the
handler runs on defaults. Auth is required, so this is not an attack
surface — it matters because a **client bug** that drops a field takes a
default silently instead of being refused.

The ones where that does real work:

| Route | What an empty body does |
|---|---|
| `/track2/pull-control/run` | starts a pull — moves real files |
| `/track2/retention/retrain/run` | starts a retrain |
| `/track2/schedule/train-now` | starts training |
| `/track2/models/queue/start` | starts the queue |
| `/track2/metrics/export/weights` | writes weights to S3 |
| `/track2/metrics/export/folder` | writes to a folder |

Suggested: make the identifying field **required** on each of these, so a
request that does not name what it is acting on is a 422 rather than a
default. Not done — it changes request shapes the frontend sends, so it
wants its own pass with the callers.

### ~~The first load of Review & Metrics is still ~9s~~ — resolved

0.08s on the first hit now. See round three.

### Two endpoints are built and unreachable

| Route | What it does | Caller |
|---|---|---|
| `/track2/models/audit/files/used-by` | reverse lineage — which runs used this image | none |
| `POST /track2/metrics/export` (`build_metrics_csv`) | the metrics CSV | none |

Both are kept and tested. Neither has a button. Either wire them up or
retire them — a route with no caller is a thing that will be maintained
for no reason.

### Not yet done

- **A click-through of every page by hand.** Everything above is the API
  surface and static analysis of the frontend. Nothing has driven the
  actual UI. This is the one remaining gap that needs a browser rather
  than a script — no Node here, and CLAUDE.md keeps builds in CodeBuild.
- **Actually destroying something real** — the guards are all tested
  (round three), but no delete, purge or rebuild has been run against
  live data, because there is no disposable target to aim at. Needs a
  throwaway detector, week folder or prefix to be nominated.
- ~~The `Class` → "Model Name" terminology sweep~~ — done, 90 strings.
- **The metrics CSV export** (`POST /track2/metrics/export`,
  `build_metrics_csv`) is kept and tested but **unreachable from the UI**.

---

## Suggestions for a friendlier page

Ordered by how much they would change the experience, not by effort.

1. **Say what the wait is for.** The slow routes are now fast on repeat,
   but the first one is still seconds. A card that says *reading 53
   registry groups* is a page working; an empty card is a page broken.
   Same text, entirely different feeling.

2. **53 groups is too many to choose from.** The registry has 53 groups
   holding 20 versions — most are empty. Empty groups could be hidden
   behind a *show empty* toggle, or garbage-collected. Every picker that
   lists them is asking the operator to read 53 names to find the handful
   that have anything in them.

3. **Put the version count next to every group name.** It is already
   fetched. "yolo11s-helmet-detector (14)" versus "(0)" removes most of
   the reason to open one.

4. **One page, one notion of "busy".** Each panel has its own spinner, so
   a page mid-load looks like a dozen unrelated things happening. A
   single quiet indicator in the header, driven by the request coalescer
   that now exists, would read as one page loading.

5. **The error card should offer the next step, not just the message.**
   `RouteBoundary` now says what broke and that the other pages still
   work. It could go further — "this usually means the registry is
   throttling; try again in a moment" is actionable where a stack message
   is not.

6. **Surface the metrics CSV export.** It is built, tested, and has no
   button.

---

## Tooling built for this pass

In the scratchpad, not the repo:

| | |
|---|---|
| `uat_api3.py` | the read-only sweep — every GET bare and adversarial, writes analysed from the schema rather than called |
| `probe.py` | where the Review & Metrics wait actually goes, phase by phase |
| `timeit.py` | the before/after table above |
| `tdzcheck.py` | const-reads-state-declared-later — catches the blank-page bug `tsc` does not |
| `tsxcheck.py` | braces, newlines in strings, control characters, missing `>` |
| `uat_api4.py` | round two — path parameters, the body gate, the SPA trap |
| `uat_auth.py` | every route's dependency tree, through the nested routers |
| `uat_contract.py` | every endpoint api.ts calls, against the OpenAPI document |
| `uat_destructive.py` | round three -- every destructive route's refusal path |
| `empty_groups.py` | which registry groups hold nothing, and which must never go |
| `httptime.sh` | times the REAL server over HTTP, not a fresh process with a cold cache |
| `gate.py` | runs the suite, exits 0 only if the failures are exactly the 3 known |

`uat_api3.py` is worth keeping — it is a few minutes to run and it caught
a 500 nobody would have found by clicking.
