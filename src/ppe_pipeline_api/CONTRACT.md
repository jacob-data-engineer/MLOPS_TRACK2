# API contract

FastAPI backend for the React SPA in `web/`, served by the same process
on the same port. Everything here is **Track2**: the retraining
platform. The PPE judge and its Track1 screens were retired and this
API no longer calls into them -- if you are looking for
`/track1/...` or an eval/gold-set endpoint, it is gone, not moved.

137 operations across 14 routers, listed below.

## Where the authoritative detail lives

**`GET /openapi.json`, and `/docs` for the rendered version.** Those are
generated from the code, so they cannot drift; request and response
shapes, field names and defaults belong there and are deliberately not
copied into this file. What this file is for is the part a schema
cannot tell you: what each group of endpoints is *for*, and the four
conventions below that hold across all of them.

## Conventions

### A 200 from this API proves nothing on its own

The SPA catch-all serves `index.html` with **200** for any unmatched
path, because a browser deep-link to `/track2/metrics` has to return
the app. So a request to a route that does not exist -- a typo, or an
endpoint that was removed -- comes back *successful*, with HTML.

This has already cost real time: three tests probed a deleted endpoint,
two failed for reasons that looked like auth bugs, and the third
asserted 200 and **passed** without ever reaching an authenticated
route. Before believing a route answered, check the content-type, or
that the path is in `app.openapi()["paths"]`.

### Auth

`POST /auth/login` takes the password and returns a bearer token;
everything except `/health`, `/build` and the login itself goes through
`require_session`, which 401s on a missing, malformed or expired
`Authorization` header. Login is rate-limited to
`MAX_LOGIN_ATTEMPTS` before it starts 429ing.

There is no per-route permission model: a session can do anything the
UI can do.

### Writes refuse a bad body rather than guessing

Every POST validates through a Pydantic model, so a missing or
wrong-typed field is a 422 before any handler runs. Two fields that
look optional are deliberately **required** -- `limit` on a pull run and
`confirm_objects` on a week purge -- because defaulting them meant an
empty body could pull or delete everything.

### Long work is queued, not awaited

Pulls, training runs, evaluations and endpoint deploys all return as
soon as the work is **accepted**, not when it finishes. Each has a
queue with the same shape (`/queue`, `/queue/add`, `/queue/remove/{id}`,
`/start`, `/stop`), and both the pull queue and the training queue are
drained by an EventBridge rule hitting a Lambda -- **not** by this
process. They keep moving while the app is down.

## The routers

### `root` (2)

Unauthenticated on purpose. `/health` for the load check, `/build` so you can ask a container which image it is actually running without logging in -- the same thing the badge under the wordmark shows.

- `GET /build`
- `GET /health`

### `auth` (2)

Password in, bearer token out. Every other router depends on `require_session`, so there is exactly one way in.

- `POST /auth/login`
- `GET /auth/session`

### `track2-models` (32)

The main screen, and the biggest surface here. Detectors (the named things you train), the base-model library (upload or import a `.pt`, optionally slicing its head down to the classes you want), dataset assembly and `data.yaml` authoring, starting a run, the training queue, and the audit trail of what was trained on what.

- `GET /track2/models`
- `GET /track2/models/audit`
- `GET /track2/models/audit/files/used-by`
- `POST /track2/models/audit/{audit_id}/delete`
- `GET /track2/models/audit/{audit_id}/export.csv`
- `GET /track2/models/audit/{audit_id}/weights-url`
- `GET /track2/models/base-models/architectures`
- `GET /track2/models/base-models/library`
- `GET /track2/models/base-models/packages`
- `POST /track2/models/base-models/register`
- `POST /track2/models/base-models/upload`
- `POST /track2/models/base-models/upload-url`
- `POST /track2/models/base-models/verify`
- `POST /track2/models/dataset-assemble`
- `GET /track2/models/dataset-classes`
- `POST /track2/models/dataset-classes/from-text`
- `POST /track2/models/dataset-discover`
- `POST /track2/models/dataset-layout`
- `POST /track2/models/dataset-yaml-preview`
- `POST /track2/models/dataset-yaml-write`
- `GET /track2/models/detectors`
- `POST /track2/models/detectors`
- `POST /track2/models/detectors/{slug}/delete`
- `GET /track2/models/live`
- `POST /track2/models/queue/add`
- `POST /track2/models/queue/remove/{entry_id}`
- `POST /track2/models/queue/start`
- `POST /track2/models/queue/stop`
- `GET /track2/models/scratch-check`
- `POST /track2/models/train/custom`
- `POST /track2/models/train/dataset`
- `POST /track2/models/train/stop`

### `track2-pull-control` (37)

Getting images out of the source bucket and into weekly folders: schedule, preview, run, and the pull queue that serialises overlapping pulls. Also owns week contents, gap filling, train/val splits and purging.

- `GET /track2/pull-control`
- `GET /track2/pull-control/buckets`
- `POST /track2/pull-control/config`
- `POST /track2/pull-control/folders`
- `GET /track2/pull-control/forecast`
- `GET /track2/pull-control/last-pull`
- `POST /track2/pull-control/mode`
- `POST /track2/pull-control/pairing`
- `GET /track2/pull-control/pending`
- `POST /track2/pull-control/preview`
- `GET /track2/pull-control/queue`
- `POST /track2/pull-control/queue/add`
- `POST /track2/pull-control/queue/advance`
- `POST /track2/pull-control/queue/clear`
- `POST /track2/pull-control/queue/remove/{entry_id}`
- `GET /track2/pull-control/run`
- `POST /track2/pull-control/run`
- `GET /track2/pull-control/run/{run_id}`
- `POST /track2/pull-control/run/{run_id}/stop`
- `GET /track2/pull-control/schedule`
- `GET /track2/pull-control/settings`
- `POST /track2/pull-control/settings/delete`
- `POST /track2/pull-control/settings/load`
- `POST /track2/pull-control/settings/new`
- `POST /track2/pull-control/settings/save`
- `POST /track2/pull-control/split/build`
- `POST /track2/pull-control/split/preview`
- `POST /track2/pull-control/units`
- `POST /track2/pull-control/weeks`
- `GET /track2/pull-control/weeks/contents`
- `POST /track2/pull-control/weeks/delete`
- `POST /track2/pull-control/weeks/fill-gaps`
- `POST /track2/pull-control/weeks/purge`
- `POST /track2/pull-control/weeks/purge-all`
- `GET /track2/pull-control/weeks/purge-preview`
- `POST /track2/pull-control/yaml-classes`
- `POST /track2/pull-control/yaml-files`

### `track2-retention` (21)

How much history a retrain is allowed to see. The retention curve and matrix, dataset snapshots (and restoring or training from one), and the pull-then-retrain chain.

- `GET /track2/retention`
- `POST /track2/retention/build`
- `GET /track2/retention/chain-status`
- `POST /track2/retention/config`
- `POST /track2/retention/curve`
- `POST /track2/retention/dataset-yaml`
- `POST /track2/retention/dataset-yaml/save`
- `POST /track2/retention/matrix`
- `POST /track2/retention/preview`
- `GET /track2/retention/queue-after-pull`
- `POST /track2/retention/queue-after-pull`
- `GET /track2/retention/retrain`
- `POST /track2/retention/retrain`
- `POST /track2/retention/retrain/run`
- `GET /track2/retention/snapshots`
- `POST /track2/retention/snapshots`
- `DELETE /track2/retention/snapshots/{snapshot_id}`
- `GET /track2/retention/snapshots/{snapshot_id}`
- `GET /track2/retention/snapshots/{snapshot_id}/drift`
- `POST /track2/retention/snapshots/{snapshot_id}/restore`
- `POST /track2/retention/snapshots/{snapshot_id}/train`

### `track2-metrics` (13)

Per-run metrics, and the exporter that writes them out to a folder on a schedule.

- `GET /track2/metrics`
- `POST /track2/metrics/export`
- `GET /track2/metrics/export/auto`
- `POST /track2/metrics/export/auto/arm`
- `POST /track2/metrics/export/auto/disarm`
- `POST /track2/metrics/export/folder`
- `GET /track2/metrics/export/groups`
- `POST /track2/metrics/export/preview`
- `GET /track2/metrics/export/settings`
- `POST /track2/metrics/export/settings/delete`
- `POST /track2/metrics/export/settings/load`
- `POST /track2/metrics/export/settings/save`
- `POST /track2/metrics/export/weights`

### `track2-deploy` (8)

Inference endpoints: what is live, the deploy queue, and tearing one down.

- `GET /track2/deploy`
- `GET /track2/deploy/browse`
- `POST /track2/deploy/endpoint/delete`
- `POST /track2/deploy/inference-queue/add`
- `POST /track2/deploy/inference-queue/remove/{item_id}`
- `POST /track2/deploy/inference-queue/start`
- `POST /track2/deploy/inference-queue/stop`
- `GET /track2/deploy/lineage`

### `track2-eval` (5)

Evaluation runs, queued the same way training is.

- `GET /track2/eval`
- `POST /track2/eval/add`
- `POST /track2/eval/remove/{run_id}`
- `POST /track2/eval/start`
- `POST /track2/eval/stop`

### `track2-review` (6)

Approve or reject a model package before it can become the live one.

- `GET /track2/review`
- `POST /track2/review/model/{model_package_arn}/approve`
- `POST /track2/review/model/{model_package_arn}/delete`
- `POST /track2/review/model/{model_package_arn}/reject`
- `POST /track2/review/{review_id}/approve`
- `POST /track2/review/{review_id}/reject`

### `track2-schedule` (3)

The weekly retrain schedule, plus a manual trigger.

- `GET /track2/schedule`
- `POST /track2/schedule`
- `POST /track2/schedule/train-now`

### `track2-lineage` (1)

What produced what: run to dataset to weights.

- `GET /track2/lineage`

### `track2-export` (2)

Dataset export.

- `GET /track2/export`
- `POST /track2/export`

### `track2-notifications` (3)

Who gets told when something finishes or fails.

- `GET /track2/notifications`
- `POST /track2/notifications/add`
- `POST /track2/notifications/remove`

### `track2-cost` (2)

Spend, overall and today.

- `GET /track2/cost`
- `GET /track2/cost/today`

