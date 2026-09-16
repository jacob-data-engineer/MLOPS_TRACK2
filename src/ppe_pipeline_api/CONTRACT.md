# PPE Pipeline API — contract

FastAPI backend for the React frontend in `web/`. Calls straight into the
`ppe_pipeline` package and exposes it over HTTP. This
document is meant to stay accurate to what's actually implemented in
`src/ppe_pipeline_api/` — if it drifts, fix the doc or the code, not just
one of them.

## Conventions

- All endpoints except `POST /auth/login` and `GET /health` require
  `Authorization: Bearer <token>` (see Auth below). Missing/invalid token
  -> `401`.
- Request/response bodies are JSON unless noted (file uploads are
  `multipart/form-data`; image-fetch endpoints return raw image bytes with
  the appropriate `Content-Type`).
- A storage write that loses a concurrent-modification race (two people
  saving the same file at once) returns `409` with a JSON body
  `{"detail": "..."}` describing the conflict — the frontend should treat
  this as "someone else just saved this, reload and retry."
- Every "path" field the frontend sends (`dataset_path`,
  `annotations_path`, `few_shot_manifest_path`, ...) is a storage key, in
  the free-text sense the UI's own path field allows (including a path
  that doesn't exist yet, for an output file) — the API does not restrict
  these to a fixed directory.
- Enum-like string fields (`violation_type`, `verdict`, `confidence`,
  `hazard_type`, gold-set labels, ...) are the same string values
  `ppe_pipeline` already uses (e.g. `ViolationType.value`), not
  reinvented names.

## Session state (in-memory, single-process)

The eval one-by-one stepper, the currently-loaded gold set, the
currently-loaded few-shot library, and Track1's rolling per-session
results all live in an in-memory dict keyed by the session token issued
at login (see `session_store.py`). This is deliberately simple — no
external session store, no horizontal scaling — per the architect's
decision for a ~7-person internal tool. **This ties the deployment to a
single FastAPI process/worker**: running more than one would split a
user's session across processes that don't share state.

The UI's gold-set "you have unsaved changes — save or discard before
continuing" dirty-check guard is **not** reproduced server-side — it's a
client-side concern, not one a request/response API has: the frontend
already holds its own form state and can do that check locally before
calling `annotate`.
Role-based field ownership, which *is* a real authorization boundary, is
still validated server-side (see Gold Set below).

## Auth

### `POST /auth/login`
Body: `{"password": "..."}`. Compares against `PPE_APP_PASSWORD`
(`hmac.compare_digest`, fails closed — `503` if the env var isn't set).
`401` on a wrong password (rate-limited per client host, `429` after 5
failed attempts — keyed by client host rather than by session, since no
session exists before login).
Response: `{"token": "<opaque token>"}`. Send this token as
`Authorization: Bearer <token>` on every other request — it doubles as
the session-state key described above.

### `GET /health`
No auth. `{"status": "ok"}`.

## Track1 alarm review (`/track1`) — backs the `/run` screen

As of this build, `POST /track1/run`'s single-case inference runs through
`ppe_pipeline`'s own "Few-shot / filter / cropped" judge method
(`track1_adapter.run_track1_judge_inference`) — few-shot examples,
`crop_box_for_violation` cropping, and the geometric too-near-forklift
pre-filter — the same method "Evaluate against ground truth"'s stepper
uses, **not** the separate `hazard_validation` package's zero-shot,
uncropped Track1 worker this endpoint called previously. (The live
`hazard_validation` worker itself not using few-shot is tracked separately,
Jira SCRUM-28, cross-repo — out of scope here.)

- `GET /track1/hazard-types` -> `{hazard_types: [...5 ViolationType values...]}`. **Deprecated** — kept only for compatibility with any caller still expecting this shape; prefer `GET /eval/violation-types`, which this now just mirrors (`POST /track1/run` itself takes `violation_type`, not a Track1-specific hazard type string).
- `GET /track1/config` -> `{provider, model, model_options}` — `ppe_pipeline`'s own config default, for an "effective judge" caption. `model_options` is `config.judge.model_options`: `{provider: [model, ...], ...}`, the same starter list `POST /eval/run`'s provider/model dropdowns can source from.
- `GET /track1/eval-datasets` -> `{datasets: [{key, label, is_named}, ...]}` — real, valid eval-dataset case-list files under `PPE_EVAL_DATA_DIR` (see "Eval-dataset listing" below for the shape and filtering rule). Same filtering/shape as `GET /eval/datasets`, kept as a separate endpoint per the existing pattern of two independent path pickers.
- `GET /track1/eval-datasets/cases?path=...` -> `{cases: [{case_id, violation_type}, ...], counts: {violation_type: N, ...}}`. `counts` is per violation type over the whole file, so the batch-run control can say how many cases a selection covers *before* spending a judge call on any of them. Malformed entries (not a dict, or missing `case_id`/`violation_type`) are skipped rather than crashing the listing.
- `GET /track1/eval-datasets/case?path=...&case_id=...` -> `{case_id, media_type, detections: [one detection dict]}`.
- `GET /track1/eval-datasets/case-image?path=...&case_id=...` -> raw image bytes.
- `POST /track1/detections/parse` (multipart `detections_file`) -> `{detections: [...]}`, parsed via `parse_detections`.
- `POST /track1/detections/mock` body `{width, height}` -> `{detections: [...]}` via `generate_mock_detections`.
- `POST /track1/run` (multipart form): `image` (file), `media_type`, `violation_type` (one of `ViolationType`'s 5 values — renamed from `hazard_type`), `person_box` (JSON array string), `forklift_box` (optional JSON array string; only for `too_near_forklift`), `region_box` (optional JSON array string; only for `glove`/`helmet`), `alarm_id`/`source_detection_id` (optional, generated if omitted), `using_mock_detections`/`acknowledge_mock` (bool; a mock detection is rejected with `400` unless acknowledged), `override_provider`/`override_model` (optional; `"bedrock"` is the only accepted `override_provider` — anything else is rejected with `400`), `helmet_box_mode` (`"cv"` default, or `"vlm"`; `cv` runs the stage-2 YOLO11s detector, `vlm` skips it so the judge's own inline `helmet_box` stands and `helmet_box_source` comes back `"vlm"` — a way to deliberately produce the loose boxes the review queue exists to catch, which are otherwise scarce since the detector clears most cases well above the review threshold). Loads the canonical few-shot manifest (`<PPE_FEWSHOT_DATA_DIR>/manifest.json`) automatically — not caller-configurable. Calls `run_track1_judge_inference`, appends to this session's rolling results, and returns `{alarm_id, hazard_type, violation_type, outcome, verdict, confidence_level, confidence_score, reason, explanation}` — both `confidence_level` (the categorical word) and `confidence_score` (the model's own real 0.0-1.0 probability estimate for this call) are returned, so the frontend can render e.g. "0.9 (high)". Judge/inference failure (including a missing few-shot manifest) -> `502`. **Superseded by `POST /track1/run-job` below** (left in place, unused by the current frontend — nothing else in this codebase calls it).
- `GET /track1/session` -> `{rolling_results: [...], calls: N}` — each rolling result is `{alarm_id, hazard_type, violation_type, outcome, confidence_level, confidence_score, reason}`.
- `POST /track1/session/clear` -> resets both, returns the cleared state.

### Track1 "Run pipeline" background job — `/track1/run-job`, `/track1/run-job/progress`

A single judge call only takes seconds — this isn't fixing a real
orphaned-work risk the way the eval jobs below are. It exists purely so
"Run pipeline" follows the same job-poll shape as every other long-running
action in this app, per the CEO's consistency call. State lives in its own
session key, `session["track1_job"]`, independent of every eval job slot.

- `POST /track1/run-job` — identical multipart body/validation to `POST /track1/run` (same fields, same 400s for a bad `violation_type`/`person_box`/`forklift_box`/`region_box`, same mock-acknowledgement 400). `409` ("A Track1 job is already running for this session.") if this session already has a `track1_job` with `status == "running"`. On success, starts a background thread and returns immediately: `{"job_id": "...", "total": 1}`.
- `GET /track1/run-job/progress` -> always `200`, never `404`. `{"status": "none"}` if no job has ever been started this session. Otherwise: `{job_id, status: "running"|"done"|"error", total: 1, processed: 0|1, result: <the exact same dict POST /track1/run returns> | null, error: string | null}`. `rolling_results`/`session_usage` are updated the same way `POST /track1/run` updates them, once the job reaches `"done"` — poll `GET /track1/session` separately for that.

### Track1 batch run — `/track1/batch-run`, `/track1/batch-run/progress`, `/track1/batch-run/cancel`

Runs **every case of the chosen violation types** in an eval dataset through
the real judge, one call at a time, on a background thread. Same shape as
the eval screen's multi-select, but writing to the audit log the escalation
and audit queues read — clicking through cases one at a time is not a route
to the few hundred labels retraining needs. State lives in
`session["track1_batch_job"]`, independent of `session["track1_job"]`.

- `POST /track1/batch-run` body `{dataset_path, violation_types: [...], helmet_box_mode: "cv"|"vlm", limit: 50, using_mock_detections, acknowledge_mock}`.
  - `400` for an empty `violation_types`, a value that isn't one of `ViolationType`'s, a `helmet_box_mode` other than `"cv"`/`"vlm"`, or a selection that matches no case in the file (the detail names the dataset and the types).
  - `404` if `dataset_path` doesn't exist; `409` if this session already has a batch with `status == "running"` — two batches sharing one session's rolling results would interleave them.
  - `502` ("Few-shot library unavailable: ...") if `<PPE_FEWSHOT_DATA_DIR>/manifest.json` won't load. Loaded **once here**, before the thread starts, so a broken manifest is one failure at start rather than N identical per-case errors on a run the operator is watching tick up.
  - `limit` caps cases per run (default 50): a judge call costs money and takes seconds, so a mis-clicked "all violations" on a 1848-case dataset must not be able to spend hours before anyone notices. Cases are taken in file order.
  - On success -> `{job_id, total}` immediately, where `total` is the post-cap count.
- `GET /track1/batch-run/progress` -> always `200`. `{"status": "none"}` before any batch this session. Otherwise `{job_id, status: "running"|"done"|"cancelled"|"error", total, processed, helmet_box_mode, rows: [{case_id, alarm_id, violation_type, outcome, confidence_score, confidence_level, reason, helmet_box_source, helmet_box_confidence}, ...], errors: [{case_id, error}, ...]}`. `confidence_score`/`confidence_level` are the judge's confidence in the VERDICT, which is a different question from `helmet_box_confidence`, the detector's confidence in the box. `processed` counts attempts, so `processed == total` at the end whether or not every case produced a row; `rows + errors` accounts for all of them. Each case is independent — a failure is recorded against that case and the run continues, because abandoning 49 good cases over a malformed 50th would waste judge calls already paid for. Every successful case is appended to this session's `rolling_results` as it lands, in the same narrowed 7-key shape the single-case paths use (not the full row -- `GET /track1/session` is polled, and fifty full rows would carry fifty judge explanations in every poll), and each case increments `session_usage.calls`, since each one is a real judge call. All session and job mutations happen under `SESSIONS.lock_for(token)`; the judge call itself does not, so a batch never stalls this operator's own progress polling.
- `POST /track1/batch-run/cancel` -> `{status, processed, total}`; `404` if no batch has been started. Cancellation is **cooperative**: the worker checks between cases, so the call in flight finishes and its result is kept — it has already been paid for.

### One judged case, in full — `/track1/judged/{alarm_id}`

The batch's progress rows are deliberately lean, so the detail comes from
the AUDIT LOG rather than from the session job. That costs no memory per
case, survives a reload, and works for any case ever judged — not only the
ones from the batch still in this session. Inlining an annotated PNG per
case in `/batch-run/progress` instead would ship tens of megabytes every
two seconds on a fifty-case run.

- `GET /track1/judged/{alarm_id}` -> the newest audit-log row for that alarm, as `{alarm_id, hazard_type, timestamp, outcome, reason, reasoning, explanation, confidence_score, confidence_level, person_box, crop_box, helmet_box, helmet_box_source, helmet_box_confidence, image_width, image_height, judge_model, judge_provider, helmet_detector_weights, frame_available, crop_available}`. `404` if nothing was ever judged under that id. **Newest**, because an alarm id identifies the image: re-judging appends a row and supersedes the earlier one, so a detail view must never show a superseded verdict. `reasoning` and `explanation` are the same string under both names — the log calls it one thing and the run response the other. **Every box is in ORIGINAL FRAME PIXELS**, alongside the `crop_box` they were measured in, so a caller drawing over the cropped image subtracts the crop origin itself rather than being handed one space and left to guess.
- `GET /track1/judged/{alarm_id}/image?crop=true` -> raw bytes, **unannotated**, cropped to what the judge saw. Cropped by default for the same reason the escalation queue crops: a worker is often ~70px tall in a 1920x1080 frame. `crop_bytes` does not resize, so a pixel here is a pixel of the original offset by the crop origin. A case with no recorded crop region serves the whole frame rather than nothing — `crop_available` on the detail response says which you got. `404` for a legacy row whose frame was never stored (a bare filename, no `/`).

## Rolling metrics (`/metrics`) — backs the `/metrics` screen

- `GET /metrics/rolling` -> `{track1_metrics, shadow_mode_agreement, shadow_mode_agreement_source, gold_set_agreement, shadow_mode_agreement_target, label_volume_weekly_target, track2_retraining_done, audit}`.
  **`shadow_mode_agreement` is measured against AUDITS, not the gold set** (`shadow_mode_agreement_source: "audit"`). The Run pipeline path exists to rehearse the live feed, and a live alarm has no gold label, so scoring this path against a gold set would measure something the real thing can never reproduce. Gold stays where a gold label genuinely exists: scoring an eval run against ground truth, on the Eval screen.
  The DEFINITION is unchanged — of the cases the judge committed to a verdict on, how often it matched the human; escalations excluded from numerator and denominator, since deferring is not a verdict to agree with. Only the source of the human answer changed, which is what keeps the number comparable to what the tile reported before. Withheld as `{insufficient_data: true, matched_count}` below `audit.min_samples`.
  It is `audit.confusion.overall`'s accuracy read out loud, deliberately, rather than a second statistic beside it — if the two diverged the page would be making two claims about one thing.
  `gold_set_agreement` is the same judge measured the old way, against reviewers' gold labels. Still returned, but for reference rather than as the benchmark: where both exist the GAP is the interesting part, since an auditor sees the pipeline's answer before agreeing with it and a gold labeller did not.
  `audit` carries the agreement rate plus `audit.confusion`; its full shape is documented under "Audit" below rather than repeated here. `track1_metrics`/`shadow_mode_agreement` are `compute_track1_metrics`/`compute_shadow_mode_agreement`'s own dict shapes verbatim.
  `track1_metrics.label_volume_this_week` and `label_volume_breakdown` count what was DELIVERED, so they run over the crops an auditor KEPT: a struck crop is retracted from both datasets, and counting it reported a label the export had pulled. The direction is over-report and it grows with how much the strike-out feature is used, so it reads correct until auditors start using it — and this figure is graded against the 400-a-week target. Every count in that block is on the same population; a new one belongs there too.
  `track1_metrics.no_valid_crops` counts images an auditor struck every crop out of. They stay in `total_alarms` — they are still images — and are held out of **both** sides of `auto_annotation_rate`: an auditor striking crops says nothing about whether the pipeline needed a human for that frame, and leaving them below the line turned a frame that had auto-annotated at 1.0 into a rate of 0.0. `audit_summary` pulls the same case out of both sides of its own rate, so the two screens agree.
  `track1_metrics.no_valid_crops` counts images an auditor struck every crop out of. They stay in `total_alarms` — they are still images — and are held out of **both** sides of `auto_annotation_rate`: an auditor striking crops says nothing about whether the pipeline needed a human for that frame, and leaving them below the line turned a frame that had auto-annotated at 1.0 into a rate of 0.0. `audit_summary` pulls the same case out of both sides of its own rate, so the two screens agree.
  `track1_metrics.auto_annotation_rate`'s NUMERATOR is images with no unanswered hedge in them, read off the crops rather than off the image's outcome. The outcome is an OR and puts CONFIRM_TP above ESCALATE, so a frame the judge fired on AND hedged about reads CONFIRM_TP — and it is still queued for a human, so counting it here claimed a frame a reviewer is about to open. The two are different questions: what the judge concluded, and whether the frame needed a person. Measured on the live corpus, 6 Sep: 0.9839 -> 0.8172, the whole difference being 34 frames carrying 62 hedged crops.
- `GET /metrics/eval-runs` -> `{runs: [filename, ...]}` (newest first) under `PPE_EVAL_RUNS_DIR`.
- `GET /metrics/reset/preview` -> `{judged_cases, escalation_reviews, audit_decisions, alarm_images, exported_files, confirmation, preserved: [...]}`. Counts what a reset would destroy without touching any of it, so the confirmation can name real numbers instead of asking someone to approve an abstraction. `confirmation` is the word `POST /metrics/reset` requires, reported here so the UI never hardcodes it.
- `POST /metrics/reset` body `{confirm: "RESET"}` -> `{cleared: {...}, failures: [{key, error}]}`. **Destructive.** Clears everything one offline test run produced: the audit log, escalation reviews, audit decisions, persisted alarm frames, and the whole exported YOLO dataset. They go together on purpose — clearing the metrics alone would leave reviews and a training set pointing at cases that no longer exist, which is worse than either extreme. A typed word rather than a boolean, because `{"confirm": true}` is what a stray client sends by accident; anything else is a `400` and nothing is touched. **Preserved:** reviewers' gold annotations, the eval datasets, the few-shot library, and saved eval-run reports — a reset destroys a test run's output, never a person's input. Idempotent: a second reset reports zeros. Deletion continues past an individual failure and reports it in `failures`, because aborting halfway would leave exactly the mixed state this exists to prevent.
- `POST /metrics/eval-runs/combine` body `{filenames: [...]}` -> `{rows: [...]}` (`report_to_rows` over the combined `EvalReport`). `404` if a filename doesn't exist.

## Bulk eval + one-by-one stepper (`/eval`) — backs the `/eval` screen

### Eval-dataset listing
`GET /eval/datasets` and `GET /track1/eval-datasets` both list only real,
valid eval-dataset case-list files under `PPE_EVAL_DATA_DIR`
(`ppe_pipeline.eval.dataset.list_valid_eval_dataset_paths`): a `.json` key
whose contents parse to a JSON list whose first element has `case_id`,
`image`, `violation_type`, and `person_box` keys. This deliberately
excludes `eval/gold_annotations.json` (a dict, not a case list) and
`eval/track1_audit_log.json` (a list, but of audit rows with none of those
keys) — any per-file exception (bad JSON, wrong shape, missing keys)
excludes just that one file rather than breaking the whole listing.

Both endpoints return `{"datasets": [{"key": ..., "label": ..., "is_named": ...}, ...]}`
(`ppe_pipeline_api.dataset_registry.decorate_dataset_keys`): `eval/gold_set_v1.json`
and `eval/gold_set_v2.json` are labeled "Gold Set v1"/"Gold Set v2" with
`is_named: true` and sorted first (v1 then v2); every other file is
labeled with its own filename, `is_named: false`, sorted alphabetically
after. `GoldsetPage`'s dataset-shaped fields (merge base/add, the Load
section's dataset path) reuse `GET /eval/datasets` too — the same
filtering is correct there, since gold-set inputs are also real case-list
files.

- `GET /eval/datasets` -> `{datasets: [{key, label, is_named}, ...]}` (see "Eval-dataset listing" above).
- `GET /eval/methods` -> `{methods: {<method name>: {shot, use_crop, bypass_forklift_filter, label}, ...}}` — the 6 named methods, verbatim from `EVAL_METHODS` in `routers/eval.py`.
- `GET /eval/violation-types` -> `{violation_types: [...]}`.
- `GET /eval/rule-text` -> `{rule_text: {violation_type: markdown}}`.
- `GET /eval/runs` -> `{runs: [filename, ...]}` (newest first).
- `GET /eval/runs/{filename}` -> `{rows: [...]}` (parsed jsonl).

### `POST /eval/run`
Body: `{dataset_path, methods: [...method names...], also_run_glove_crop, violation_types (omit/empty = all), auto_balance, balance_seed, few_shot_manifest_path (required if any method is few-shot), provider, model, config_path (optional)}`.

Runs every selected method (and/or the glove crop experiment) over the
filtered/balanced dataset, **blocking until the whole run finishes** —
there is no incremental per-case progress/image
streamed back; this is a plain request/response call. (A live progress
bar + "latest crop" preview + streaming table would need SSE/WebSocket;
a future iteration could add that if the frontend needs it.)
Every judged case is still streamed to `PPE_EVAL_RUNS_DIR/<...>.jsonl` as
it completes, so a run is still recoverable from disk if
the request itself is interrupted.

Response, if exactly one method was selected and `also_run_glove_crop` is
false: `{"mode": "bulk_review_pending", "review_rows": [...], "warnings": [...], "results_path": "..."}` — the report is stashed in this session's state; call `/eval/bulk-review/finalize` next.

Response otherwise: `{"mode": "final", "reports": {<method name>: [rows...]}, "combined_rows": [...] (if >1 method), "glove_crop_rows": [...] (if also_run_glove_crop), "warnings": [...], "results_path": "..."}`.

`404` if `dataset_path`/`few_shot_manifest_path` doesn't exist. `400` for
an empty method selection, an unknown method name, or a missing few-shot
manifest path when required.

**Superseded by `POST /eval/advanced-run` below** (left in place, unused
by the current frontend — existing tests cover it).

### Advanced-mode background job — `/eval/advanced-run`, `/eval/advanced-run/progress`, `/eval/advanced-run/cancel`, `/eval/advanced-run/case-image`

Runs the exact same thing `POST /eval/run` runs (any combination of the 6
named methods and/or the glove crop experiment), but on a background
`threading.Thread(daemon=True)` that returns immediately — the CEO's
original explicit ask for observable Advanced-mode progress. State lives
in its own session key, `session["eval_advanced_job"]`, which is entirely
independent of Simple mode's `session["eval_bulk_job"]` (see above) — the
two may genuinely run concurrently in one session, with no cross-check
between them, and cancelling/finishing one has no effect on the other.

- `POST /eval/advanced-run` — same `RunEvalRequest` body/validation as `POST /eval/run` (methods list, also_run_glove_crop, violation_types, auto_balance, balance_seed, few_shot_manifest_path, provider, model, config_path; same 400s for an empty/unknown method selection or a missing few-shot manifest path when required; same `404` if `dataset_path` doesn't exist — a missing few-shot manifest *file*, unlike `dataset_path`, is only caught once the job actually reaches that method, same as `POST /eval/run` itself, and surfaces as this job's `error` rather than an immediate `404`). `total` in the response **is not simply `len(dataset) * len(methods)`** whenever `also_run_glove_crop` is set — the glove crop experiment (`run_glove_crop_experiment`) only runs over glove-violation cases, 3 passes each, so `total` adds `<glove case count> * 3` on top of the per-method contribution instead of `<dataset size> * 3`. `409` ("An Advanced-mode run is already in progress for this session.") if *this session's own* `eval_advanced_job` is `"running"` — `eval_bulk_job`'s status is never checked here. On success, starts the background thread and returns immediately: `{"job_id": "...", "total": <int>}`.
- `GET /eval/advanced-run/progress` -> always `200`, never `404`. `{"status": "none"}` if no job exists yet. Otherwise: `{job_id, status: "running"|"done"|"error"|"cancelled", current_method: string | null, methods_done: [...method names, plus "glove_crop" once that experiment finishes...], total, processed, rows: [...], final: {...} | null, error: string | null, results_path_name}`. `rows` is the same per-case shape `GET /eval/bulk-review`'s `review_rows`/Simple mode's `rows` use. Once `status == "done"`: if more than one method ran and/or `also_run_glove_crop` was set, `final` is populated with the exact same shape `POST /eval/run`'s "final" branch returns (`{"mode": "final", "reports": {...}, "combined_rows": [...] (if >1 method), "glove_crop_rows": [...] (if glove crop ran)}`). **If exactly one method ran and `also_run_glove_crop` was false, `final` stays `null`** — instead, this job populates `session["eval_bulk_report"]`/`session["eval_bulk_dataset_dir"]` exactly like `POST /eval/run`'s single-method branch does, so the existing `GET /eval/bulk-review` / `POST /eval/bulk-review/finalize` / `POST /eval/bulk-review/discard` manual-review flow works unchanged against this job's result — poll `GET /eval/advanced-run/progress` until `status == "done"` with `final == null` and exactly one method, then call `GET /eval/bulk-review` as usual.
- `POST /eval/advanced-run/cancel` -> `404` if no job exists for this session; otherwise the same cooperative-cancel semantics as `POST /eval/bulk-review/cancel` (checked once per case; idempotent once the job has already reached a terminal status). Returns `{"job_id": "...", "status": "..."}`.
- `GET /eval/advanced-run/case-image?case_id=...` -> that case's crop with its judged box(es) drawn, PNG bytes — same recompute-on-demand pattern as `GET /eval/bulk-review/case-image`, but reads from `session["eval_advanced_job"]["case_outcomes"]` specifically (a separate endpoint, not shared, so the two job slots' state stays fully independent). `404` if no job exists for this session, or `case_id` hasn't been processed yet.

`session["eval_advanced_job"]` has no `warnings` field for `auto_balance`'s
own warnings (unlike `POST /eval/run`'s response) — a gap worth flagging:
if the frontend needs to surface an auto-balance warning for an
Advanced-mode job run, there is currently no field carrying it back.

### Ground-truth review — bulk ("after all predictions")
- `GET /eval/bulk-review` -> `{review_rows: [...]}` or `404` if nothing pending.
- `POST /eval/bulk-review/finalize` body `{corrections: {case_id: "TP"|"FP", ...}}` — **only case_ids whose label changed need to be included**; anything omitted keeps its original dataset label (resubmitting every row would have the same net effect on `sync_ground_truth`, since it's a no-op for an unchanged value — this just keeps the payload small). Applies corrections, calls `sync_ground_truth` per changed case, rescoring via `build_report`, clears the session's pending review. Returns `{rows: [...], synced_files: [...]}`.
- `POST /eval/bulk-review/discard` -> clears the pending review without scoring. `{"discarded": true}`.

### Simple-mode bulk run (observable) — `/eval/bulk-run`, `/eval/bulk-review/progress`, `/eval/bulk-review/cancel`, `/eval/bulk-review/case-image`

The web UI's "Evaluate against ground truth" screen has a Simple/Advanced
toggle. Advanced mode is `POST /eval/run` above, unchanged. Simple mode
instead runs the fixed "Few-shot / filter / cropped" method on a raw
`threading.Thread(daemon=True)`, started from `POST /eval/bulk-run`, which
returns immediately — the frontend polls `GET /eval/bulk-review/progress`
(recommended: every ~2s) for progress instead of blocking on the request.
This state lives in its own session key, `session["eval_bulk_job"]`,
entirely separate from Advanced mode's `session["eval_bulk_report"]`/
`session["eval_bulk_dataset_dir"]` — the two modes never interact, and
starting/finishing/cancelling a Simple-mode run has no effect on any
pending Advanced-mode `/eval/bulk-review` state (and vice versa).

Critically, **the job keeps running to completion even if the client that
started it disconnects entirely** (closed tab, closed browser) — that's
the whole point of moving Simple mode to a background job. It just sits at
`status == "done"`/`"error"`/`"cancelled"` until next polled, however long
that takes.

- `POST /eval/bulk-run` body `{dataset_path, few_shot_manifest_path, provider, model, config_path (optional)}`. Always runs "Few-shot / filter / cropped" (`use_crop=True, bypass_forklift_filter=False`) over the *unfiltered, unbalanced* dataset in full (no `violation_types`/`auto_balance` options — Simple mode is deliberately the plain "run everything" path). `404` if `dataset_path`/`few_shot_manifest_path` doesn't exist. `409` ("A run is already in progress for this session.") if this session already has a job with `status == "running"` — only one Simple-mode run at a time per session (no cap across sessions). On success, starts the background thread and returns immediately: `{"job_id": "...", "total": <case count>}`.
- `GET /eval/bulk-review/progress` -> always `200`, never `404` (safe to call unconditionally, including when nothing has ever run this session). If no job exists yet: `{"status": "none"}`. Otherwise the full job state: `{job_id, status: "running"|"done"|"error"|"cancelled", label, total, processed, rows: [...], final_rows: [...] | null, error: string | null, results_path_name}`. `rows` grows one entry per case as the run proceeds (same per-case shape `GET /eval/bulk-review`'s `review_rows` uses: `{case_id, label, violation_type, ground_truth, verdict, confidence, explanation, ground_truth_rationale, reasoning_score}`); `final_rows` (the `report_to_rows`-shaped aggregate table) is only populated once `status == "done"`. There is one shape for both the idle-check and the active-poll case — no separate "peek"/"full" response.
- `POST /eval/bulk-review/cancel` -> `404` if no job exists for this session. Otherwise sets the job's internal cancel flag if (and only if) it's still `"running"`, and is a no-op — but still `200` — if the job has already reached `"done"`/`"error"`/`"cancelled"` (idempotent, safe to call more than once or after the fact). The job doesn't stop instantly: the flag is checked once per case, so the case in flight when cancel is called still finishes and is recorded before the run stops (`processed` will be less than `total`, not necessarily `processed - 1`). Returns `{"job_id": "...", "status": "..."}` — the status *at the time of the call*, so a `"running"` response here doesn't mean cancellation was rejected, just that the job hadn't yet observed the flag.
- `GET /eval/bulk-review/case-image?case_id=...` -> that case's crop, with its judged box(es) drawn — PNG bytes, recomputed on demand from the case's already-known `CaseOutcome` (no re-judging, and nothing decoded here is cached in the session — a 320-case run could mean hundreds of images in memory otherwise). `404` if no job exists for this session, or if `case_id` doesn't match any case that has been processed yet (either it's not in the dataset, or the run simply hasn't reached it yet — same `404`, no distinction).

### Ground-truth review — one by one (the stepper)
Only ever runs the "Few-shot / filter / cropped" method, matching
The restriction that "One by one" review is offered only when
that's the sole selected method.

- `POST /eval/stepper/start` body `{dataset_path, few_shot_manifest_path, violation_types, auto_balance, balance_seed, provider, model, config_path}` -> loads+filters+balances the dataset, builds a live `JudgeClient`, and stores it all in this session's state. Returns `{total, idx: 0, warnings}`.
- `GET /eval/stepper/state` -> if the current case has no judge result yet, **calls the judge now** (a side effect on a GET — lazy-on-first-fetch, documented rather than hidden) and caches it in session state; repeat calls for the same index are free. Once every case is confirmed (`idx >= total`): `{"done": true, idx, total, report_rows: [...]}`. Otherwise: `{"done": false, idx, total, confirmed_count, case_id, violation_type, current_ground_truth, ground_truth_rationale, judge_verdict, judge_confidence, judge_explanation, reasoning_score, reasoning_critique}`. `404` if no stepper session exists.
- `GET /eval/stepper/image` -> the current case's crop, with its judged box(es) drawn — PNG bytes. `404` if there's no pending case yet (call `/state` first).
- `POST /eval/stepper/confirm` body `{ground_truth: "TP"|"FP"}` -> records the (possibly corrected) ground truth, scores the outcome, syncs it via `sync_ground_truth`, appends the row to the run's jsonl file, and advances to the next case. Returns `{idx, total, done, synced_files}`. `409` if called before `/state` has ever computed a pending result for this index.
- `POST /eval/stepper/cancel` -> discards the stepper session. `{"cancelled": true}`.

## Few-shot library (`/fewshot`) — backs the `/fewshot` screen

- `GET /fewshot/paths/defaults` -> `{manifest_path, annotations_path}` defaults under `PPE_FEWSHOT_DATA_DIR`.
- `POST /fewshot/load` body `{manifest_path, annotations_path}` -> loads both into session state. Returns `{manifest_path, annotations_path, counts: {violation_type: n, ...}}`.
- `GET /fewshot/entries?violation_type=...` -> `{entries: [{index, image, verdict, reason, confidence, explanation}, ...], tp_count, fp_count, uncertain_count, reason_options}`.
- `GET /fewshot/entries/{violation_type}/{index}/image` -> raw image bytes.
- `PUT /fewshot/entries/{violation_type}/{index}` body `{verdict, reason, confidence, explanation}` -> validates against `VERDICTS`/`CONFIDENCE_LEVELS`/`REASON_CODES[violation_type]`, saves to both `manifest.json` and `annotations.json`. Returns the updated entry.
- `DELETE /fewshot/entries/{violation_type}/{index}` -> removes the entry from both files. Returns `{removed: filename}`.
- `GET /fewshot/crops/available?violation_type=...` -> unused crop filenames under `<manifest dir>/crops/` (not already referenced by any violation type's entries).
- `POST /fewshot/entries/from-existing-crop` body `{violation_type, existing_filename, verdict, reason, confidence, explanation}` -> adds an example pointing at an existing crop file.
- `POST /fewshot/entries/upload` (multipart `image` + query/form fields `violation_type, verdict, reason, confidence, explanation`) -> uploads a new crop into `<manifest dir>/crops/`, then adds it as an example.

All of the above except `/paths/defaults`, `/load`, and `/status` require
a library already loaded this session (`404` otherwise).

- `GET /fewshot/status` -> never errors. `{"loaded": false, "manifest_path": null, "annotations_path": null, "counts": null}` if nothing has been loaded this session yet; otherwise `{"loaded": true, "manifest_path": "...", "annotations_path": "...", "counts": {violation_type: n, ...}}` — the same `counts` shape `POST /fewshot/load` itself returns, reflecting `session["fewshot"]`. For a frontend to check "is a library already loaded?" on page load without needing a `404` round-trip first.

## Gold Set dual review (`/goldset`) — backs the `/goldset` screen

- `GET /goldset/options` -> `{reviewers, gold_labels, evidence_sufficiency, difficulty, adjudication_status}` — static option lists for the review form.
- `POST /goldset/merge` body `{base_path, add_path, output_path}` -> concatenates (no dedup) and writes. `{total, base_count, added_count}`.
- `POST /goldset/load` body `{dataset_path, annotations_path}` -> loads both into session state. `{count, completion: [{violation_type, done, total}, ...]}`.
- `GET /goldset/cases?violation_type=All&adjudication_status=all&reviewer=&queue_mode=unfinished` -> `{case_ids: [...], count, completion: [...]}`. `adjudication_status` is `all`/`disagreements`/`pending`; `queue_mode` (`unfinished`/`all`) only applies once `reviewer` is set to one of `gold_set.GOLD_SET_REVIEWERS`.
- `GET /goldset/cases/{case_id}?reviewer=` -> full case detail: `{case_id, violation_type, current_ground_truth, agreement, role, owned_fields, adjudicator_ready, rotation, rule_text, reason_options, entry: {...all sidecar fields...}}`. `role`/`owned_fields`/`rotation` are `null`/`[]` when `reviewer` is omitted ("no identity selected -> everything editable" mode).
- `GET /goldset/cases/{case_id}/image?crop=false` -> the full image (or, with `crop=true`, the crop sent to the judge), with its box(es) drawn — PNG bytes.
- `POST /goldset/cases/{case_id}/annotate` body `{reviewer: "" | one of GOLD_SET_REVIEWERS, updates: {field: value, ...}}` — **server-side role gating**: `reviewer`'s role on this case is computed via `gold_set.role_for` (never trusted from the client), and any field in `updates` outside that role's owned fields is rejected with `403` — including that an adjudicator's save is rejected (`409`) until both reviewers have completed theirs. A real save conflict (another reviewer saved first) -> `409`. On success, if the case's `final_label` resolved to a real verdict, ground truth is synced via `sync_ground_truth`. Returns `{entry, role, synced_files}`.
- `POST /goldset/sync-rationale` -> propagates resolved `final_rationale`s into every eval dataset file in the loaded dataset's directory. `{updated_files: [...]}`.
- `POST /goldset/freeze-v1` body `{few_shot_manifest_path}` -> freezes the currently-loaded cases into `<eval_dir>/gold_set_v1.json`. `{count, output_path, warnings}`.
- `POST /goldset/freeze-v2` body `{few_shot_manifest_path}` -> re-reads `<eval_dir>/testing_dataset_batch_new.json` fresh from disk and freezes into `<eval_dir>/gold_set_v2.json`. `404` if that file doesn't exist. `{count, output_path, warnings}`.

All endpoints except `/options`, `/merge`, and `/status` require a gold
set already loaded this session (`404` otherwise).

- `GET /goldset/status` -> never errors. `{"loaded": false, "dataset_path": null, "annotations_path": null, "count": null, "completion": null}` if nothing has been loaded this session yet; otherwise `{"loaded": true, "dataset_path": "...", "annotations_path": "...", "count": N, "completion": [...]}` — `count`/`completion` are the same shape `POST /goldset/load` itself returns, reflecting `session["goldset"]`. For a frontend to check "is a gold set already loaded?" on page load without needing a `404` round-trip first.

## Model deployment status (`/models`) — backs the `/models` screen

- `GET /models/status` -> `{model_type: "ppe", latest: {...} | null, versions: [...]}` — the single `ModelType.PPE` card the `/models` screen shows.
- `GET /models/versions/{model_type}` -> `{model_type, versions: [...]}` for any `ModelType` value (`ppe`/`plane`/`load`/`vehicle`) — kept queryable for history, same as `list_model_versions` itself isn't restricted to PPE. `404` for an unknown `model_type`.

`track2_retraining_done` (the "at least one full retraining cycle
completed, for at least one model" benchmark) lives under
`GET /metrics/rolling`, not here — the same split the UI makes, where that
check appears in the Rolling metrics tab, not the Models tab.

## Evaluate escalations (`/escalations`) — backs the `/escalations` screen

Two queues, both **views over the Track1 audit log** rather than their own
work list, so a case cannot be lost to an ended session and a reviewed case
simply stops matching. Human decisions live in a separate sidecar
(`<PPE_EVAL_DATA_DIR>/escalation_reviews.json`) keyed by `alarm_id`; the
audit log is append-only history of what the *models* decided and is never
rewritten.

There is deliberately **no reviewer rotation, no role gating and no
adjudication** here (unlike `/goldset`). This review exists to produce CV
retraining labels, not to grade the VLM's reasoning, so it collects only a
verdict or a box — no rationale, reason code, evidence rating or difficulty.
`reviewer` is recorded for provenance only and is never validated against a
roster.

- `GET /escalations/queue?queue=verdict|box&include_reviewed=false&include_unreviewable=false` ->
  `{queue, items: [...], counts: {verdict: {pending, total, unreviewable}, box: {...}}, verdict_options, exportable_verdicts, box_status_options, review_confidence}`.
  `verdict_options` is `["violation", "no_violation", "cannot_determine"]`;
  `exportable_verdicts` is the subset that yields a training label (the first
  two). Sent rather than left for the client to hardcode, since the server
  decides what gets exported.
  A case whose frame was never stored is **excluded by default** and reported
  in `counts.*.unreviewable` instead: nothing a reviewer does can settle it,
  so listing it made an entirely unworkable queue look like a backlog. Every
  row written before frame persistence is in this state. `include_unreviewable=true`
  brings them back. Each item carries `frame_available` for the same reason —
  note that a row having an `image_key` does NOT mean it has a frame, because
  the old writer recorded a bare filename it never stored bytes at.
  **A helmet-CV frame is one item, not one per person**, keyed by the frame --
  the same id an audit decision about that picture is filed under. Its crops
  come through the collapse on `candidates`, ONE PER PERSON -- a re-judged
  frame has two rows for each of its people and the newest wins, the same rule
  every other reader of the audit log applies:
  `[{alarm_id, candidate_id, person_index, person_box, crop_box,
  judge_outcome, judge_reason, judge_reasoning, excluded}]`, in the order the
  judge saw them. `alarm_id` there is the CROP's own id (what
  `GET /escalations/cases/{alarm_id}/image?crop=true` serves), while the
  item's `alarm_id` is the frame. `excluded` is whether the crop has already
  been struck out. Empty for a live alarm, which is one crop and has no strip.
  Carried so a reviewer can strike a crop out while resolving the escalation;
  the DECISION is still one per image.
  `queue` outside `verdict|box` -> `422`. Items are newest-first. Default
  hides decided cases, which is what makes a save advance the queue;
  `include_reviewed=true` keeps them so a reviewer can revisit their own call.
  - **verdict** queue = images with a crop the judge hedged on that no human
    has answered. Off the CROPS, not off the image's outcome: the outcome is
    an OR that puts CONFIRM_TP above ESCALATE, so a frame the judge fired on
    AND hedged about reads CONFIRM_TP and still owes an answer.
  - **box** queue = rows with `outcome == "CONFIRM_FP"`, `reason_code == "helmet_worn"`,
    `hazard_type == "no_helmet"`, whose helmet box is not trainable: either
    `helmet_box_source != "cv"` (the VLM fallback was used) or the detector
    scored it below `helmet_box.review_confidence`. An escalation a human has
    since resolved to `no_violation` is the same thing said by a person, and
    joins it — **whether they said it about the frame or about that one
    person**. A crop answered `no_violation` on its own reached neither queue
    and shipped with a person box and no helmet box, which teaches the
    detector that a helmeted head is background. WHICH FRAMES are here is
    otherwise unchanged and still asked of the image: asking it per crop in
    general would queue every frame holding any weakly-boxed cleared worker
    (measured, pending 16 -> 94), which is a workload decision nobody has
    taken.
  - **A box item's crop is the crop that OWES the box**, which is rarely the
    crop that decided the frame's verdict: `crop_box`, `person_box`,
    `helmet_box`, `helmet_box_source`/`_confidence`, `judge_outcome`,
    `judge_reason` and `judge_reasoning` all come from it. `alarm_id` is still
    the frame — one box decision per image, filed against the picture.
    `box_crop_id` names that crop by candidate id (null for a live alarm,
    which is one crop), because a box decision can be an answer ABOUT that
    person: see `no_helmet_present`.
  - The two queues are still disjoint AS WORK: a frame reaches the box queue
    by being cleared, either by the judge or by the human who answered its
    escalation, so nothing is owed both at once. One image does appear in both
    LISTINGS once `include_reviewed=true` shows the answered verdict — on two
    different crops of itself, which is what the image endpoint's `queue`
    exists for.
- `GET /escalations/cases/{alarm_id}?queue=` -> one item, same shape as a
  queue entry. `404` if that alarm is not in that queue.
- `GET /escalations/cases/{alarm_id}/image?crop=true` -> the judged frame,
  **unannotated** (the box screen draws its own overlay; a server-burned
  rectangle would be indistinguishable from the one being placed), cropped by
  default to the exact region the judge saw. A full 1920x1080 frame is not
  reviewable — the worker is often ~70px tall in it — so the reviewer works
  the crop and `crop=false` is context only.
  **`crop=true` on a row with no recorded `crop_box` serves the WHOLE FRAME**,
  because no image at all is worse than one shown too small: asking for a crop
  is not proof of getting one, so check `crop_box` on the item. The dangerous
  consequence is closed server-side — the box POST refuses such a case.
  Searches both queues, so the caller need not know which holds it.
  **Optional `queue=verdict|box` names WHICH crop to serve.** One image can be
  in both queues on two different crops — the verdict question is about the
  crop that decided the frame, the box question about the crop that owes the
  helmet box — and a box drawn on one and stored against the other is a
  rectangle on the wrong head. Without `queue` the crop of whichever question
  is still OPEN is served (both open or both answered: the verdict's), which is
  right for every flow today; the box screen sending `queue=box` makes it
  exact.
  `404` with an explicit "No frame was stored" message for rows written
  before frame persistence existed.
- `POST /escalations/cases/{alarm_id}/verdict` body `{verdict, reviewer?, excluded_people?}` —
  `verdict` must be one of `verdict_options` (`422` otherwise). `404` if the
  alarm is not in the verdict queue.
  **`excluded_people` strikes crops out from the escalation screen**, exactly
  as the audit screen does and into the SAME store: resolving an escalation
  already writes an audit entry, so there is no second exclusion sidecar and
  `audit.excluded_people` stays the one reader. Ids not belonging to this case
  are dropped. Omitting it means "not touching them" and `[]` clears them,
  the same distinction the audit POST draws. An image whose every crop is
  struck has nothing left to ask a human, so it leaves both queues — but it
  stays **reachable by id here**, so the strike can be undone on the screen
  where it was made by resolving again with `excluded_people: []`. The crops to strike arrive on the queue item as `candidates`
  (see `GET /escalations/queue`), because the queue collapses a frame's crops
  to one row and a reviewer cannot strike out a crop they were never shown.
  `cannot_determine` settles the case and clears the queue but yields **no
  training label**. It exists because a case is escalated precisely when the
  evidence is insufficient (`head_occluded` is the commonest reason): with
  only two options a reviewer who genuinely cannot tell is forced to guess,
  and that guess becomes a CV training label. It mirrors `no_helmet_visible`
  on the box side; `escalations.is_exportable` is what enforces the exclusion.
- **Box statuses** are `drawn` | `accepted_cv` | `no_helmet_present` | `no_helmet_visible`.
- **`no_helmet_present` means "there is no helmet in this image", and it is a VERDICT finding, not a box finding.** A case only reaches this queue because something concluded a helmet is worn, so a reviewer finding none has overturned the clearance: a violation the judge cleared, which is a false negative and the worst error this pipeline makes. Saving it does three things in one act, because the reviewer has already done the looking: writes `verdict: "violation"` on the review **and against the person the box question was about** (`box_crop_id`, into `crop_verdicts`) — a per-crop answer outranks the frame's everywhere, so without that an earlier `no_violation` on that person silently reverted the finding and the case went on settling and exporting as a clearance while the audit recorded the correction beside it — records an **audit** (`decision: "disagree"`, `corrected_verdict: "violation"`, `source: "box_review"`, returned as `audit` on the response) so the finding reaches `/metrics/rolling` and scores as a false negative against the judge, and exports the case as a person with no helmet box, the negative a helmet detector needs. The model's own helmet box is never exported for it: a reviewer just said there is no helmet, so shipping one would train on the hallucination they rejected. A `box_review` audit is exempt from the staleness rule, since it IS the reason the settled verdict changed and would otherwise retire itself the instant it was written.
- `no_helmet_visible` means the opposite thing about the verdict: **a helmet IS worn but cannot be boxed** (occluded, cut off by the crop). The verdict stands, no audit is recorded, and the case produces no label. Conflating the two threw the false negative away silently and left the case counted as a confident correct clearance in the auto-annotation rate. `accepted_cv` keeps the DETECTOR's own box with a human's endorsement behind it, and takes no `helmet_box` payload: redrawing a box that is already right is wasted work and worse data, since a hand-drawn copy of a good detector box is less accurate than the box it replaces. The queue asks for a human because the detector was *unsure*, not because it was wrong. It is a label (it exports), unlike `no_helmet_visible`, and the export records its source as **`cv_accepted`** rather than `human` or `cv` — more trustworthy than an unreviewed pseudo-label, less precise than a hand-drawn box, and a retraining run can only weight those differently if the export says which it is. Accepting on a case with no detector box is a `400`: it would mark the case labelled and export it with no helmet box at all, a sample that silently teaches "no helmet here" on a case where one is worn.
- `POST /escalations/cases/{alarm_id}/box` body `{status, helmet_box?, reviewer?}`.
  `status` is `drawn` or `no_helmet_visible`. `helmet_box` is
  `[x1, y1, x2, y2]` in **CROP pixels** — the natural pixels of the image
  `GET .../image?crop=true` served, so the client sends exactly what it
  measured. The server maps it back with `judge_box_to_original` and stores
  the frame-pixel box (plus the crop-pixel one it was given): the crop is a
  review-time rendering decision, so a box stored only against it would go
  stale the moment crop padding is retuned.
  `drawn` without a box -> `400`; a box supplied with any other status ->
  `400`; a box extending past the crop -> `400` (out-of-crop coordinates
  normalize outside 0-1, which YOLO accepts silently and trains on as
  garbage); a case with no recorded `crop_box` -> `400`, since the box could
  not be mapped back; a box under 2px in either axis, or inverted, negative
  or wrong-length -> `422`.
  `no_helmet_visible` is **not** a verdict change: it records that no helmet
  could be found to draw, so the case is excluded from training data rather
  than exported with a guessed rectangle.

- `GET /escalations/audit/readiness` ->
  `{ready, blockers: [{code, message}], counts: {auto_settled, settled_by_human, labelled_by_human, excluded_by_reviewer, exportable, exported, audited}, sample_rate, audit_target}`.
  Backs a third tab that **exists before the audit does**: "what has to happen
  before I can audit" is a more useful answer than a hidden or disabled tab,
  and these are the same counts the export will consume.
  `auto_settled` is cases the pipeline settled with no human involved — the
  ones with the least oversight, and so the ones most worth auditing.
  `excluded_by_reviewer` counts `cannot_determine` / `no_helmet_visible`:
  finished work that deliberately yields no label, which is why it is not in
  `exportable`. `blockers` currently always includes `no_export`, since
  nothing has been exported yet.
  `sample_rate` is the deterministic share of exportable cases the audit will
  sample, keyed on alarm id so a case is permanently in or out — an agreement
  *rate* only means something if its denominator is not drifting as the queue
  grows.

- `GET /escalations/export/preview` ->
  `{batch_id, prefix, case_count, cases: [{alarm_id, image_name, classes, box_sources, settled_by}], skipped: [{alarm_id, reason}]}`.
  What an export WOULD write, without writing it — separate from the POST so
  the decision is inspectable before a run mutates a shared bucket, and so
  "why is this only 3 of 13 settled cases" is answerable without exporting
  first.
- `POST /escalations/export` -> the same shape plus `exported_total` and
  `metadata_case_count`. Writes an Ultralytics YOLO dataset (YOLO11-compatible
  — the format is one spec across v5/v8/v11) to its own prefix, currently
  `track2/escalation_dataset`:
  `data.yaml`, `images/train/<alarm_id>.jpg` (the person CROP, not the frame),
  `labels/train/<alarm_id>.txt` (`class cx cy w h`, crop-normalized),
  `metadata.json` (cumulative full record per case), `batches/<batch_id>.json`
  (what this batch held) and `index.json` (a cheap "was this exported" lookup).
  Additive — exporting twice tops the dataset up. `400` when nothing is
  exportable, listing the reasons. The prefix is server-side only: a
  client-supplied destination would be a way to write anywhere in the bucket.

Both endpoints merge into one sidecar entry per alarm, so a box decision
never wipes a verdict recorded earlier on the same case.

Items also carry `crop_box` (the judged region, frame pixels) plus
`person_box_in_crop` / `helmet_box_in_crop` — the same boxes translated into
crop pixels, so the review UI never does coordinate arithmetic. `crop_bytes`
does not resize, so that translation is an offset by the crop origin, not a
scale.

## Audit (`/escalations/audit/*`) — backs the `/escalations/audit` screen

Audits SETTLED cases. Its output is the ground truth for live data, which has
none: an alarm off a camera cannot be scored against a gold label, so an
auditor's agreement IS the measurement. `/metrics/rolling` carries it.

- `GET /escalations/audit/queue?settled_by=all|pipeline|human&include_audited=false`
  -> `{settled_by, cases: [...], items: [...], summary: {...}}`.

  **`cases` is the queue, and its unit is one IMAGE** (CEO, 1 Sep: "all the
  crops from 1 image should be part of 1 audit"). Each case is
  `{case_id, people: [...], settled_by, hazard_type, timestamp,
  settled_verdict, judge_outcome, source, complete, people_expected,
  people_gated, people_awaiting, people_excluded, exported, audited,
  stale_audit, audit}`, with `people` holding
  one ordinary queue item per person the judge was asked about — each carrying
  **its own** `judge_outcome`, `judge_reason` and `judge_reasoning`. The frame
  verdict is an OR, so the crop that decided it is usually not the first one;
  a single rationale beside a frame-level stamp was showing person 0's words
  against a verdict some other person fired.

  `settled_verdict` and `judge_outcome` on the case are the OR: **any one
  person in violation makes the image a violation**, it is cleared only when
  everyone was cleared, and it hedges only when nothing was decided.

  **The OR runs over the crops the auditor KEPT** (CEO, 3 Sep: "same logic if
  after remove, as long as there is one violation, then the frame is
  violation"). A crop named in `audit.excluded_people` leaves the population,
  and the precedence above then runs over what is left -- so striking the only
  violating crop settles the image as `no_violation`. `people_excluded` counts
  them. They stay in `people`: the strike-out is reversible and the auditor
  has to see what they struck.

  A person awaiting only a HELMET BOX still contributes their verdict to
  that OR: the judge answered them and what is owed is a rectangle, which is
  a labelling gap and not a verdict gap. Only a person the judge HEDGED on
  (`awaiting: "verdict"`) contributes nothing, and an unanswered one also
  BLOCKS the image from settling unless it is already a `violation`.

  **A frame-level verdict does not reach every crop.** `no_violation` and
  `cannot_determine` are statements about the whole picture and every crop of
  it inherits them; `violation` is a statement about SOMEBODY, so only the
  crops the judge fired on inherit it -- a worker standing beside a violator
  is not thereby a violator. A crop the judge hedged on is answered by the
  frame's verdict when the crop is not NEWER than the answer -- the reviewer is
  shown every crop of the frame, so one save covers the people they were shown,
  and a crop a later re-run added is not one of them. What a frame-level answer
  cannot express is which of two hedged people was the violator; per-crop
  verdict storage is what lifts that, and until it exists both carry the
  frame's answer (the export is unexposed: a non-firing crop of a violation
  frame gets no label).

  **`settled_verdict` is `"no_valid_crops"` when every crop was struck out.**
  Its own outcome, not `no_violation` -- "none of these boxes is worth looking
  at" is a different finding from "nobody here was in violation", and folding
  the first into the second would put a frame carrying no evidence into the
  cleared bucket. It is out of every denominator in `confusion` and counted in
  its own `no_valid_crops` bucket in `summary`, beside `box_rejected` and
  `self_settled` -- decided, but not part of any rate. `judge_outcome` is
  `null` for it, since no crop is left to have produced one.

  `complete` is false when the image has fewer judged people than the detector
  found and the gate did not account for — a failed call, or a live feed
  stopped part way. Those images are held out of the scored population rather
  than read as "nobody else was in violation".

  `source` is the arm: `live_cv`, `offline_gold`, or `""` for a row written
  before lineage was recorded. **Every arm lands in this one queue**; the live
  feed, the helmet-CV pilot and the offline gold run are not three processes.

  `items` is the same queue flattened back to one entry per crop, for a caller
  reading the ledger rather than reviewing. It is NOT a second queue: a
  decision is recorded against a `case_id`, never against a person.

  Oldest first, capped at
  `audit.AUDIT_LIMIT` (400). A **workload cap, not a sample** — audited cases
  drop out so finishing some lets more in, and oldest-first means a new case
  joins the end rather than displacing one someone is part-way through.
  `settled_by` separates cases the pipeline settled alone (no human ever
  looked — the least-overseen slice, and the reason the audit exists) from
  ones a reviewer decided (auditing those audits the reviewer, not the model).
  A case with no frame, or settled deliberately without a label
  (`cannot_determine` / `no_helmet_visible`), is not auditable: there is
  nothing to look at, or nothing in the dataset to check it against.
- `POST /escalations/audit/cases/{case_id}/preview` body `{excluded_people: [...]}` -> `{case_id, settled_verdict, judge_outcome, deciding_person, people_excluded, excluded_people, people_awaiting_verdict}`. **Writes nothing.**
  **This is the authority for what the crop strip displays.** A reviewer ticking a crop has to see the verdict their ticks would produce before saving, and the server cannot be asked about ticks that have not been posted — so it is asked about hypothetical ones instead. **Do not reintroduce a client-side copy of the precedence.** It is written out at five sites server-side and `audit.included_people` exists so those cannot drift; a sixth copy in the browser is one no test on either side of the wire could catch drifting, which is exactly what this endpoint replaced.
  Answered through `audit.case_outcome` — the same call the queue and the audit POST both go through — so a preview and the save that follows it cannot give different answers. `excluded_people` on the response is the ids that would ACTUALLY apply: one naming a crop of another picture is dropped here exactly as the save drops it, so the preview cannot promise something the save will not do.
  `settled_verdict` carries both empty cases and they are not the same: `"no_valid_crops"` when every crop was struck, and `null` when the crops still in the image are all ones the judge hedged on. `people_awaiting_verdict` counts the second, so a screen can render it as **no verdict yet** rather than as a clearance.
  **The strike-outs already recorded play no part** — the request's list is the whole question, or "what if I un-tick this" would be unanswerable.
  `404` for a case id that no longer resolves, matching `GET /audit/cases/{case_id}` and the audit POST: the client is looking at something re-run or audited away, and inventing a verdict for it would show a confident answer about a case that is not there.
  Cheap on purpose, because the client calls it on every checkbox toggle (debounced): it reads the audit log and the review sidecar and nothing else. The audit sidecar is not fetched (the strike-outs come from the request) and neither is the export index (`exported` plays no part in a verdict).
- `POST /escalations/audit/cases/{alarm_id}` body
  `{decision, corrected_verdict?, auditor?, note?, excluded_people?}`. The path parameter is a
  **`case_id`** — an image, not a person. For a live alarm and a gold case
  that is the alarm id exactly as before (one person per image), which is what
  keeps every decision ever recorded attached to its case; for a helmet-CV
  image it is the frame stem the review screen already files decisions under.
  `decision` is `agree`|`disagree`|`box_wrong`.
  **`box_wrong` is "the verdict is right, the helmet box is not"** — the finding neither of the others can express. A `disagree` would blame the judge for a verdict it got right and corrupt the verdict metrics; an `agree` would leave a bad label sitting in the training set. It takes no `corrected_verdict` (`400` if one is sent), endorses the verdict for `audit_confusion` (so the judge is still scored as correct), counts in its own `box_rejected` bucket in `summary` rather than as agreement or disagreement, and **retracts the case and re-queues it under Helmet boxes**. Saving a new box puts it straight back in the dataset. A `disagree` REQUIRES `corrected_verdict`
  (`400` otherwise) — "wrong" alone gives an error rate, while the correction
  also gives a usable label; `agree` must not carry one (`400`).
  **A disagreement acts on the training set**, since an audit that only
  recorded a number would leave a label a human just called wrong sitting in
  it:
  - corrected to `no_violation` → the helmet IS worn, so the case needs a box
    it does not have. Retracted from the dataset and **re-queued under Helmet
    boxes**; drawing the box publishes it again.
  - corrected to `violation` or `cannot_determine` → neither can produce a
    helmet box, so the wrong sample is retracted and does not return.
  **`excluded_people` is the crops struck out of this image**, by
  `candidate_id`. Independent of the verdict: the verdict says what the
  PICTURE was, this says which of its boxes are not worth training on.
  **Omitting it (or sending `null`) means "I am not touching the strike-outs";
  `[]` means "clear them".** Every save merges over the previous entry, so the
  two cannot share a default — a client that says nothing about the field
  would otherwise erase an auditor's work, which is exactly what the box
  review and the helmet-CV frame review did until this distinction existed.
  Ids that do not belong to this case are dropped silently, so one case's
  decision cannot suppress another's.
  A strike-out keeps the crop out of both datasets AND **pulls it from the
  crop dataset if it was already published**, then rebuilds the frame. It is
  reversible in the dataset as well as on screen: un-striking **republishes**
  the crop's label rather than merely stopping the next export from skipping
  it. Both directions are reported on the response as
  `dataset.excluded_crops`, alongside `frame`. Exclusion after export is
  allowed and rewrites the export; no metric is snapshotted anywhere, so
  there is nothing to invalidate.

  Each decision also records `audited_verdict` — the verdict it was judging.
  **An audit retires itself once that verdict changes.** An alarm id is the
  IMAGE's identity (an eval case id is its file stem, and `/track1/batch-run`
  uses it), so re-running a case supersedes the earlier row rather than
  duplicating it — but an `agree` recorded against the old verdict would then
  silently read as agreeing with the new, possibly opposite one. A mismatch
  makes the entry stale: it stops counting in `summary`, is reported there as
  `stale`, and the case returns to the queue carrying `stale_audit: true` so
  an auditor seeing it twice knows why. Entries written before the field
  existed have no `audited_verdict` and are taken at face value — there is no
  way to know what they agreed with. Reviewer verdicts and drawn boxes are
  deliberately NOT retired this way: those are facts about the photograph, and
  the person box comes from the dataset, so the crop is identical on a re-run.
  **`audited_verdict` is the verdict THIS request's `excluded_people`
  produce**, not the one the case had before them -- otherwise an auditor who
  struck out the only violating crop would flip the verdict and the audit that
  caused the flip would read stale immediately, bouncing back into the queue
  it had just left.
  **A `null` `audited_verdict` is a recorded value, not an absent one.** An
  entry with no `audited_verdict` KEY predates the field and is taken at face
  value; one recorded as `null` was recorded against an image that had no
  settled verdict (every crop still in it is one the judge hedged on), and it
  retires as soon as a real verdict appears. Recording it is allowed — the
  strike-out is a real act — but it is not agreement: `summary` counts it in
  its own `no_verdict` bucket and out of `agreement_rate`, and `confusion`
  has no ground truth to score it with.
`audit` also carries **`confusion`** — precision/recall/F1/accuracy for the
judge, scored against audit decisions instead of a gold set. This is the
whole point of auditing live data: an offline eval scores the judge against
gold labels, a live alarm has none, so an auditor's confirmation becomes the
ground truth and the same confusion matrix falls out of it. Deliberately the
same `ConfusionCounts` the offline eval uses, so the two cannot drift into
two definitions of precision. Shape: `{overall: {...}, per_hazard_type:
{hazard: {...}}, min_samples, unit: "images", per_crop: {...}}`, each block
being `{unit, n, sample_is_thin,
counts: {true_positive, false_negative, false_positive, true_negative,
uncertain}, precision, recall, f1, accuracy, decided_total, uncertain_rate,
ground_truth_violation_total, ground_truth_no_violation_total,
violation_verdict_distribution, no_violation_verdict_distribution}`. Ground
truth per case is the settled verdict when the auditor agreed, the correction
when they disagreed, and nothing at all when either is `cannot_determine` —
an auditor finding the evidence never supported a verdict is a real finding
but not a label. An `ESCALATE` counts as `uncertain`, split by what the case
turned out to be, never as a miss: the judge declining to answer is a
different failure from answering wrongly, and accuracy is over `decided_total`
so hedging cannot drag it down. Stale audits contribute nothing. An image
whose crops were ALL struck out contributes nothing either -- see
`no_valid_crops` above.

**`confusion.per_crop` is a SECOND block in a SECOND UNIT, and the two are
never summed.** Every block carries its own `unit`, on the response rather
than in a reader's head. `overall` is in IMAGES and is the block
`shadow_mode_agreement` is read off; `per_crop` is in CROPS, is ADDITIVE, and
is never the headline.

It is **attribution, not a derived precision.** There is no per-crop human
ground truth: an auditor records ONE verdict per frame, and on a violation
image only the crop that FIRED inherits the label while every other crop
inherits nothing -- pushing the frame's verdict down onto its crops would
label every bystander in every violation frame. So the block answers what the
crops can answer: **did the judge fire on a crop the auditor kept, or on one
they struck out?** Shape: `{unit: "crops", n, excluded_crops, sample_is_thin,
min_samples, counts: {fired_on_kept, fired_on_struck, cleared_on_kept,
cleared_on_struck, escalated, escalated_on_kept, escalated_on_struck}, fired_total,
fired_kept_rate}`. The counts are
named for what they are rather than TP/FP/FN/TN: "the judge cleared a crop the
auditor kept" is the ordinary case and not an error, which is also why there
is deliberately no accuracy here. `escalated` is reported with BOTH halves of
its split beside it: the total alone cannot attribute a hedge, so a card
carrying it had to rule escalations off below the table. The accounting
closes: `escalated_on_kept + escalated_on_struck == escalated`,
`excluded_crops` is exactly the three `_on_struck` counts summed, and a crop
with no readable judge answer is counted on neither side.
**Only decisions taken with the crop strip in front of the person are
scored.** The marker is the presence of the `excluded_people` key on the audit
entry, which `save_audit` writes only for a caller that manages strike-outs --
so a legacy audit, a box review and a helmet-CV frame review are all out.
Without that, six legacy frames carrying a plain `{"decision": "agree"}` read
as `fired_kept_rate: 1.0`, and at ~8.54 crops a frame legacy audits alone
clear `min_samples`, so the thin-sample gate would not have caught it. `fired_kept_rate` is the only rate, over
`fired_total`, which is a real population. `min_samples` is
`AUDIT_CROP_MIN_SAMPLES` (40), four times the frame floor: the same human
decisions produce several times as many crops as frames, so a crop `n` would
clear a frame-tuned floor on identical evidence.

- `GET /metrics/rolling` gains `audit`, the same `summary` shape, recomputed
  from the sidecar on every request rather than stored: an audit is one small
  write, and a cached aggregate is one more thing to invalidate wrongly.
  `agreement_rate` is `null` when nothing has been audited — never `0`, which
  is a different and much worse claim. Four kinds of audited case are counted
  and held out of it, each in its own bucket: `box_rejected`, `self_settled`,
  `no_valid_crops` and `no_verdict`. `blocked` is a fifth bucket and a
  different kind: it is not audited at all, it is the part of `pending` a
  human still owes a verdict on, which `GET /escalations/audit/queue` holds
  out of the list. **`pending - blocked` is the number of cases that queue
  returns** — the badge and the list beneath it have to agree.
  `sample_is_thin` is true below
  `min_samples` (10), because at n=3 one disagreement moves the rate 33
  points; the UI renders that `n` in red, a sanctioned exception to
  DESIGN.md's One Red Rule.

**The exported dataset has a real train/val split.** `data.yaml` points
`val` at `images/val`, not at `images/train` as it once did (which made any
val metric a measurement of the model's own training images). Cases are held
out **by scene**, not by frame: `training_export.split_for` hashes the clip
name recovered from the case id (`Helmet_Hatch_21_frame_0099` ->
`Helmet_Hatch_21`) and sends ~20% of scenes to `val`. Two properties this
buys, both of which a per-frame random split lacks. Consecutive frames of one
clip are near-duplicates, so splitting per frame puts the same worker in both
halves and the val score reads highest exactly when the model has generalised
least. And the assignment is a pure function of the case, which matters
because cases are published ONE AT A TIME as they settle: there is no moment
when the whole set is in hand to divide, so it must be stable across runs and
insensitive to order. Each case's split is recorded in `metadata.json`, and
publishing a case deletes any copy of it under the other split, so a re-sync
migrates the pre-split backlog instead of leaving it in both.

**Export is a consequence of settling, not a publishing step.** Saving a
verdict or a box syncs the training set immediately (`dataset` in the
response says what happened), so `POST /escalations/export` is a **background job** (`{job_id, total}`
immediately; poll `GET /escalations/export/progress` for
`{status: "none"|"running"|"done"|"error", total, processed, summary, error}`;
`409` if one is already running for the session). A full publish is ~3 storage
round trips per case, so 130 cases runs for minutes and the synchronous
version gave the operator a spinner indistinguishable from a hang. Per-case
publishing on settle stays synchronous, since that is one case. It is only a manual
backstop. A sync failure never loses the decision that triggered it — the
decision is already saved, and the next sync picks the case up.

## Helmet-CV pilot (`/helmet-cv`) — backs the `/helmet-cv` screen

Helmet-CV execution data is run-scoped. New artifacts live under
`cv_runs/<cv_run_id>/`; CLI commands after `cv` require `--cv-run-id`, and
API alarms, frames, review, and summary requests require `cv_run_id` to select
one run. `GET /helmet-cv/runs` returns
`{runs: [{cv_run_id, imgsz, detector_weights, created_at, legacy}],
default_cv_run_id: null}`. The source-frame image endpoint is run-independent.
Rows without a run id remain readable only through an explicit
`legacy-unscoped` selection and are never rewritten.

Each candidate carries a stable `ground_truth_id` and a run-scoped
`execution_id`. Human labels are keyed by `ground_truth_id`, may be revised
(with `revision` and `revises_audited_at` metadata), and the newest revision
scores every run. Summary responses can take `compare_run_id`; the
`paired_intersection` block compares only identities present in both runs,
while each run's ordinary metrics remain full-population.

`cv-judge` prints the number of pending paid calls and refuses to invoke
Bedrock unless `--confirm-paid` is supplied. Multi-person frames are decoded
once through `crops_for_frame` and passed to Track1 through `prepared_crop`;
the legacy Track1 path is unchanged when no prepared crop is supplied.

Run IDs are restricted to `[A-Za-z0-9][A-Za-z0-9._-]{0,127}` and execution
IDs are `<cv_run_id>__<ground_truth_id>`, safe for storage keys and filenames.
Immutable `cv_runs/<cv_run_id>/run.json` records creation time and complete
provenance; a configuration mismatch must use a new run ID.

Scores **production CV** and the **judge** against one human decision per
frame. This router is the read side plus the one write a human makes.

**THE CASCADE IS GONE.** This used to score an offline emulation of
`pj.jpsai.postprocessing_logic`'s `postprocess/PPE/Helmet/v1_0` gate. That
emulation ran, and could not operate on this footage: 69.9% of its candidates
reached `no_pose_for_person` because the workers are ~83px tall, and the gate
fired on 0.38% of them. So there is no CV state, no `gate_fired`, and no gate
reason anywhere in this router. The detector now supplies person crops, the
judge decides, and a human scores both.

**Two arms, deliberately different shapes.** `cv` is production CV, and it has
PRECISION ONLY: every alarm in this corpus exists because production fired, so
CV claimed a violation on all of them and cleared none — recall has no
denominator. `vlm` has the full four. There is no third composed arm; with no
gate in front of it, the pipeline IS the VLM.

**Two units, and this is the part to hold on to.**

* A **candidate** is one person in one frame. `candidate_id` is
  `<frame stem>__pNN`, `NN` being the detector's output order (an index, not an
  identity — there is no tracker). Crops are judged one call each, per
  candidate.
* A **human decision is keyed by the FRAME** (`audit.ground_truth_key`). The
  reviewer is asked "is this frame a violation?", the judge's per-crop answers
  are OR'd up to the same unit, and the crop that fired travels alongside as
  attribution. So `POST /helmet-cv/frames/{frame_name}/review` no longer 400s
  on a multi-person frame: several people in a frame is the normal case, not an
  ambiguity to reject.

**`frame_role` splits the corpus.** `alarm` is the frame production actually
fired on, and the only frame CV ever claimed anything about — so the only frame
the `cv` arm is scored on. `clip` frames were decoded at our own stride;
production said nothing about them, and they feed the `vlm` arm alone. That
asymmetry is exactly what gives the VLM a recall the CV arm cannot have.

**The alarm frame comes from `raw_video/`, never from `snapshot/`.** The
`snapshot/` still is ANNOTATED: production's own red box, the literal word
"violation" in red text, plus the camera's burned-in clock and label. Sending it
to the judge would hand over the answer in words, showing it to a reviewer would
defeat the blind, and running a person detector on it would put a
high-contrast rectangle in front of the model. So the still is used only as a
KEY: `extract._locate_alarm_frame` finds the clip frame it was drawn on by
pixel agreement (0.95-0.99 for the true frame, 0.72-0.92 for a wrong one) and
stamps THAT frame `alarm`. The still is kept under `annotated/<alarm_id>.jpg`,
outside the frames tree, where the frame endpoint cannot reach it — it only
ever resolves `frames/<alarm_id>/<name>`.

**The alarm frame is not frame 0.** Measured over 18 alarms: frame 0 nine
times, otherwise up to frame 32. The clip is a window around the alarm. An
alarm whose still is missing or matches nothing above 0.90 gets NO alarm frame
and contributes to the `vlm` arm only; ~10% of alarms are in that state.

**One ceiling, stated once.** The detector supplies every crop, so a worker it
never found is a worker the judge was never shown, and no arm here can see
that miss. Two parameters move that ceiling and both are on every run's
provenance: `imgsz` (640 found nobody at all in 26.8% of frames; 1920 in 5.5%)
and `person_confidence_threshold`. The reference's 0.5 person filter was
measured to discard 34-50% of the people the detector found and is no longer
applied — the filter follows `od_confidence` (0.25), because a weak person box
is a crop the VLM can rule on while a discarded one is invisible to every arm.

### Blindness is SERVER-SIDE

A frame's `cv` and `judge` blocks are `null` until a review has been posted
for that frame. Not hidden in the client, not greyed out — **not in the
payload at all**. No head box and no crop are sent pre-review either.

This is deliberately stronger than the Audit screen, where the frontend
withholds the settled verdict at render time while the API sends it. That
version depends on every future component author remembering not to render a
field they were handed; this one cannot be undone from the frontend at all.

**`candidate_id`, `person_index` and `person_box` DO cross the blind**, and
have to. "Is THIS person wearing a helmet?" cannot be asked without saying
which person, and a frame holding several people needs the reviewer to see
which crop the judge was asked about.

`person_box` is therefore whitelisted rather than withheld, and that carries
one obligation: **the field must be uninformative about the model's answer.**
It very nearly was not. The no-person sentinel records `person_box: null`, and
nothing else does, so an absent box identified that case exactly — and under
the old cascade, that the gate had not fired. A tell by omission is the same
leak as a tell by value. The rule outlived the cascade and still applies to
every field added here.

So the no-person sentinel is sent the WHOLE FRAME as its `person_box`
(`_blind_person_box`). Every blinded row then carries a box, and the whole
frame is not a fiction: it is exactly what `runner._judge_one_frame` crops
for that candidate, so the box drawn is the box judged. The only row that
still sends `null` is one whose frame dimensions are unknown, where there is
no whole-frame box to fall back to.

Note that the string-scan guard (`test_the_blinded_payload_leaks_no_gate_answer_anywhere`)
cannot catch this class of leak — a structural tell has no string to scan
for. Any new pre-review field needs the same argument made by hand: could a
reviewer infer the gate's answer from its value, or from its absence?

### Correctness is computed SERVER-SIDE

`cv_outcome` and `judge_outcome` come from `ppe_pipeline.audit`'s own
`audited_ground_truth` — the same function the live-alarm audit uses. **The
frontend must never derive correctness itself**: two definitions of precision
in one app is how a screen ends up disagreeing with the metrics tab about the
same case.

Both are one of exactly four strings: `correct` | `incorrect` | `not_scored`
| `uncertain`.

- `not_scored` — nobody has reviewed it, the reviewer said `cannot_determine`,
  or (for `cv_outcome` only) CV found nobody in the frame and so made no
  claim.
- `uncertain` — **judge only**. An `ESCALATE` is the judge declining to
  answer, which is neither right nor wrong. CV has no equivalent: its
  `"Unknown"` is an abstention that scores as a negative (see below).

### There is no gate, and no CV state

Both were real and both are deleted. `gate_fired` was true for `"No Helmet"`
and nothing else, `"Unknown"` was production declining to claim a violation,
and `cv.state` carried five values on the wire. None of it survives, because
the emulation that produced it could not run on this footage — see the
measurement at the top of this section.

What replaced them on the `cv` block is geometry and context and nothing else:
`candidate_id`, `frame_role`, `person_index`, `person_box`,
`person_confidence`, `people_in_frame`, `helmets_in_frame`, `provenance`.
`helmets_in_frame` is a COUNT and not a claim: nothing pairs a helmet to a head
any more, so it says how many helmets were in shot and never whose. **`null`
means NOT MEASURED** — the shipped person detector is stock COCO, which has no
helmet class, so it cannot see helmets at all; `0` still means the detector
looked and found none. The helmet box a verdict rests on never came from here:
it comes from the stage-2 crop model, per person.

The reference implementation and its documented defects are still in the git
history of `helmet_cv/cascade.py`, `pose.py`, `color.py` and `geometry.py`, and
in `HANDOFF_helmet_cv.md`.

### Endpoints

- `GET /helmet-cv/alarms?reviewed=true|false&limit=200`
  -> `{alarms: [...]}`.
  Each alarm row is `{alarm_id, clip_key, clip_start_utc, frames_total,
  frames_reviewed, people_found}`.
  `reviewed` omitted keeps everything; `true` keeps alarms with at least one
  reviewed frame, `false` keeps alarms with none. No `site` or `camera`: the
  bucket carries no camera mapping, so both would always be `unk` (the camera
  IS burned into the pixels — `CAM W2` — but nothing reads it).

  **THERE IS NO FILTER OVER FRAME ROLE, AND THERE CANNOT BE ONE.** The ledger
  filtered on CV's gate, then CV's colour state, then briefly on frame role.
  `audit.cv_verdict` is literally `return "TP" if frame_role == "alarm"`, so
  frame role IS the CV arm's verdict — a filter over it discloses CV's claim on
  exactly the frames the CV precision measurement rests on, before the reviewer
  has decided. Restricting it to reviewed frames (which is what it did) also
  made it useless precisely when a reviewer wants it: under `reviewed=false`
  every row's role set is empty, so `role=alarm` plus "not yet reviewed"
  always returned nothing. And it was worthless even so — every alarm holds
  exactly one alarm frame plus its clip frames by construction, so an
  unrestricted role set is `["alarm","clip"]` on every row. Which frame a row
  is stays visible per frame, post-review, inside the withheld `cv` block.

  **`frames_reviewed` counts DISTINCT FRAMES**, matching `GET /helmet-cv/summary`.
  One verdict settles a frame however many people are in it, so counting
  candidates reported a crowded frame as several reviews and overstated
  progress against `frames_total`.

  **Every detector-derived aggregate here covers REVIEWED frames only.**
  `people_found` counts reviewed frames and nothing else — the blindness rule
  applied one level up, since per-frame blindness holds perfectly and an alarm
  row still leaks if it aggregates over the whole clip. It reads the frame's
  own `people_in_frame` ONCE PER FRAME; summing over candidates computed
  `sum(people_in_frame^2)`, measured 6.56x corpus-wide. `frames_total` and
  `frames_reviewed` are deliberately unrestricted: neither discloses a model
  answer, and the difference is the pending count that makes the ledger usable
  for triage.

- `GET /helmet-cv/alarms/{alarm_id}/frames` -> `{alarm_id, frames: [...]}`,
  in clip order. Each frame row is exactly
  `{candidates, people, frame_name, frame_index, offset_ms, frame_utc, width,
  height, cv, judge, review}` — **ONE ROW PER FRAME**, with its people nested
  under `candidates` as `{candidate_id, person_index, person_box, crop_box}`.
  `crop_box` is the exact original-frame rectangle used to produce that
  candidate's judge image. The other three
  cross the blind on purpose (see **Blindness is SERVER-SIDE**); `cv` and
  `judge` are `null` until reviewed. `404` for an alarm with no scored frames.

  It used to be one row per CROP, which is the practical face of a half-applied
  unit: on the busiest frame of the corpus the reviewer was shown the same
  frame, the same image and the same question **18 times**, every row carrying
  an identical `frame_index`, and all 18 unblinding together when one was
  answered.

  Once reviewed, `judge` carries `frame_verdict` (the OR across the frame's
  crops), `fired_candidates` (which crops the judge called a violation —
  ATTRIBUTION, not the answer) and `crops` (per-crop detail). Read
  `frame_verdict`, not `outcome`: the latter is one crop's raw answer.
- `GET /helmet-cv/frames/{frame_name}/image` -> raw bytes, `image/jpeg`. Same
  convention as `/track1/judged/{alarm_id}/image` **except that it is not
  cropped**: there, the crop is what the judge was shown; here, cropping to
  the box CV drew would hand the reviewer CV's answer to the question they
  are being asked. `400` for a name that is not a well-formed frame name,
  `404` when the file is not on disk.
- `GET /helmet-cv/runs/{cv_run_id}/frames/{frame_name}/candidates/{candidate_id}/image`
  -> the per-person JPEG crop sent to the judge. All three identifiers must
  match one audit item, and the source is resolved only through the clean
  helmet-CV frame path. `404` for a mismatched identity, missing frame, or
  absent crop geometry; `422` for crop geometry or bytes that cannot render.
- `POST /helmet-cv/frames/{frame_name}/review` body
  `{verdict, reviewer?, note?, candidate_id?}`. `verdict` is
  `violation` | `no_violation` | `cannot_determine` (the same
  `VERDICT_OPTIONS` the escalation review uses — nothing downstream learns a
  second vocabulary). **A verdict, not an agreement**: the reviewer was never
  shown anything to agree with, so the server encodes agree/disagree against
  the settled verdict itself, which is what lets the pilot reuse
  `audited_ground_truth` and the existing confusion matrix unchanged instead
  of growing a second definition of ground truth. Each decision records
  `audited_verdict` and therefore **retires itself if the case is re-judged**,
  exactly as an Audit-screen decision does. Returns the now-unblinded frame
  row. `400` when the frame has several scored people and no `candidate_id`
  (the error names the available ids); `404` for an unknown frame or
  candidate; `422` for an unknown verdict.
- `GET /helmet-cv/summary` -> exactly:

  ```json
  {
    "cv_run_id": "...",
    "frames_reviewed": 0,
    "alarms_total": 0,
    "cv":  {"precision": null, "recall": null, "f1": null, "accuracy": null, "n": 0},
    "vlm": {"precision": null, "recall": null, "f1": null, "accuracy": null, "n": 0},
    "cv_vs_vlm": {
      "decided_total": 0, "confirmed": 0, "suppressed": 0,
      "judge_escalated": 0, "suppression_rate": null,
      "incomplete_frames": 0, "population": "..."
    },
    "population": "..."
  }
  ```

  **`incomplete_frames` counts alarm frames the judge never finished** — a
  failed call, or a live feed stopped between people. Those frames are held
  out of `decided_total` and out of both arms' `n`, because OR'ing a frame
  that is short of its `people_in_frame` reads as "no violation" and is
  indistinguishable from a complete frame afterwards. Reported rather than
  dropped in silence, so a short population is visible as one.

  **`cv.recall`, `cv.f1` and `cv.accuracy` are ALWAYS `null`, and that is not
  missing data.** Every alarm in this corpus exists because production CV
  fired, so CV claimed a violation on all of them and cleared none: recall has
  no denominator and accuracy no true negatives. The server sends `null`
  rather than the real-looking floats `ConfusionCounts` computes over empty
  denominators, because a real-looking float is how a meaningless recall ends
  up quoted. Render them as the dash any unmeasured figure gets. Never
  substitute a zero, and never compute one client-side.

  `cv_vs_vlm.suppressed` is the alarms production raised that the judge would
  clear — the figure this whole feature exists to produce. **There is no
  `judge_only` cell**: it would mean CV cleared something and the judge held
  it, and CV cleared nothing here. A cell that can never be populated is worse
  than an absent one, because a reader takes its zero for a finding.

  Identically-shaped arm blocks. **There is deliberately no second
  spelling of any figure**: no flat `cv_precision` beside `cv.precision`,
  because two spellings of one number is precisely how a second definition of
  precision gets born.

### Reading the summary

Three arms, all scored with the same `audited_ground_truth`, the same
`metrics.classify` and the same `ConfusionCounts` the offline eval and the
live-alarm audit use. Two arms reported side by side have to be two
measurements of one thing.

- **`cv`** — production CV's own claim. It fired, which is why the alarm
  exists; the human says whether it was right. **Precision only** — see the
  summary endpoint above for why the other three are `null`.
- **`vlm`** — the judge's answer over the same frames, OR'd from per-crop
  calls to a frame verdict in `audit.vlm_frame_verdicts`:

  ```
  frame_violation = ANY(crop is a violation)
  ```

  Any person in violation makes the frame one. That is the production question
  — "should this frame have raised an alarm?" — and a frame is not cleared
  because three of the four people in it are wearing helmets. `uncertain` does
  not win over a violation: a frame with one confident violation and one
  uncertain crop is a violation, and `uncertain` survives only when nothing in
  the frame was a confident answer either way.

  **The crop that fired is carried, not scored.** `fired_candidates` names
  which crops the judge called a violation. The human's verdict is about the
  frame, so a frame both call a violation counts as agreement even if the judge
  pointed at the wrong worker. That gap is real, and it is the reason crops are
  judged ONE CALL EACH rather than batched: with per-crop calls the attribution
  is at least trustworthy, whereas a batched call can swap content between
  crops and mislabel which worker fired while the frame answer stays right —
  a failure mode that is undetectable in-band.

  A hedge stays in the same `uncertain` bucket `metrics.classify` already
  uses; it is **not** silently resolved. `config.judge.uncertain_as_fp` is the
  project's existing auto-resolve rule, it lists `glove` only, and it is
  applied upstream in `judge_client` before a verdict reaches a row — so for
  helmet there is nothing to apply, and a second rule here would be inventing
  policy.

**`cv_vs_vlm.suppressed` is the number this whole exercise exists to
produce** — the alarms production raised that the judge would have cleared.
It replaced a `pipeline` minus `cv` subtraction, which had a composed arm to
subtract from. There is no composed arm now: with no gate, the pipeline is the
VLM, and the honest form of "what does the stage buy" is a count of alarms it
would remove rather than a difference between two precisions.

**Every rate is `null`, never `0.0`, until its own denominator exists** —
"nothing audited yet" and "got everything wrong" are opposite claims and
`ConfusionCounts` returns `0.0` for both. Each is nulled against the
denominator it actually rests on: precision needs TP+FP, recall needs TP+FN,
accuracy needs `decided_total`, and F1 is null if either parent is — a
corpus of nothing but true negatives has a real accuracy and no precision at
all. `n` is the audited candidate count the whole block rests on.

**Recall belongs to the VLM arm only, and `population` says exactly what it is
recall over.** The judge is asked about every frame in the sample, including
frames whose crops it clears, so a human verdict of `violation` on a frame the
judge cleared is a real false negative and recall has a denominator. The CV arm
has no equivalent: production fired on every alarm here, so it never asserted
a negative anywhere in this corpus.

There is one recall nothing here can measure, and it is worth stating in the
same breath: **the detector's own misses.** It supplies every crop, so a worker
it never found is a worker the judge was never shown and no arm can be marked
down for it. That is what `imgsz` in the run provenance is for — at
ultralytics' default 640 the detector found nobody at all in 41% of frames of
this corpus.

### The pilot's rows share the live audit log

By CEO decision the judge rows land in `eval/track1_audit_log.json` alongside
live alarms, tagged `row_kind: "helmet_cv"` at the top level with the CV
decision nested under `row["cv"]` (never flattened — the row already has
`confidence`, `helmet_box`, `helmet_box_source` and `helmet_box_confidence`
of its own, all meaning the *judge's*). Every audit query selects one kind or
the other, and a CV row can never reach the live audit queue: it is a frame
from a clip somebody already alarmed on, not an alarm, and counting it would
put a synthetic case in the shadow-mode agreement rate the 85% target is read
off.

**`POST /metrics/reset` will wipe the comparison corpus along with everything
else.** It clears produced state, and by this decision the pilot's rows are
produced state living inside that file. `reset` has no concept of row kind
and has not been taught one. Copy the audit log before resetting during a
pilot.

## Known gaps and deliberate simplifications (flagged, not papered over)

1. **No incremental progress during `POST /eval/run` (Advanced mode).**
   A live progress bar, results table, and latest judged crop as the run
   proceeds would be the ideal. This one endpoint still blocks and
   returns once the whole run is done — streaming would need SSE/
   WebSocket, which the plan didn't call for here, so flagging in case the
   frontend actually needs it for Advanced mode specifically. **This is
   now resolved for Simple mode**: `POST /eval/bulk-run` returns
   immediately and runs on a background thread, polled via
   `GET /eval/bulk-review/progress` — see "Simple-mode bulk run
   (observable)" above. **This is now also resolved for Advanced mode**:
   `POST /eval/advanced-run` runs the same thing on its own background
   thread, polled via `GET /eval/advanced-run/progress` — see "Advanced-mode
   background job" above. `POST /eval/run` itself is untouched and still
   blocks, kept only as a superseded-but-functional endpoint.
2. **Gold-set "unsaved changes" dirty-check/pending-nav guard is
   frontend-only**, not server state (see "Session state" above) — a
   deliberate simplification, not an oversight.
3. **Login lockout is keyed by client host, not by browser session** —
   there is no session before a token exists, so a per-session
   `MAX_LOGIN_ATTEMPTS` throttle has nothing to key on; client host is
   the closest available analogue.
4. `GET /eval/stepper/state` has a side effect (it calls the judge on
   first fetch for a new index) — flagged in its own section above,
   since a GET with a side effect is unusual, but it's the natural
   translation of the "compute pending on first render" logic into
   a request/response shape.
5. **`session["eval_advanced_job"]` has no field for `auto_balance`'s own
   warnings** (unlike `POST /eval/run`'s `warnings` response field) — not
   in the plan's enumerated job-state shape, so not added; flagging in
   case the frontend needs to surface an auto-balance warning for a job
   run specifically.
6. **A missing few-shot manifest *file* for `POST /eval/advanced-run`
   surfaces as the job's `error` field, not an immediate `404`** —
   `POST /eval/run` only discovers a missing few-shot manifest once its
   own (blocking) loop reaches that method too, so this isn't a new
   failure mode, but for a background job the caller only finds out by
   polling, one request-cycle later than the synchronous version's `404`
   would. `dataset_path` itself *is* still validated synchronously (an
   immediate `404`), since `POST /eval/run` already validates it
   synchronously and the dataset has to be loaded up front anyway to
   compute `total`.
7. **Multi-person batched VLM inference is deferred by decision, and the
   reason changed.** The crop side is built: `cropping.PersonCrop`,
   `cropping.CropBatch` and `cropping.crops_for_frame` produce one crop per
   person from a frame decoded **once**, each carrying the provenance the spec
   asks for (person, box, frame, source video). `helmet_cv.runner` groups work
   by frame and runs frames on a bounded pool (`helmet_cv.max_workers`,
   default 4). What is NOT built is a multi-crop model call: still one
   `run_track1_judge_inference` per person.

   The original argument was economic, and the measurement has since REVERSED
   it. Density on the pilot run (`pilot-v3`, 316 alarm frames / 2699
   candidates, `scripts/helmet_cv_density.py`) is **8.54 people per alarm
   frame**, 91.1% of frames hold more than one and the busiest holds **17**.
   Batching would save about 88% of the calls — and with no caching it saves
   the 11,427-token few-shot prefix too, not just the call. The cost case for
   batching is now large. (This figure has been restated twice as the corpus
   grew — 1.52, then 3.86. Re-derive it from the run you are actually
   quoting.)

   The argument that actually keeps it out is **attribution**. A batched call
   can return correct labels attached to the wrong crop, which leaves the
   frame-level verdict right and the "which worker" answer silently wrong.
   With OR aggregation the frame answer would look fine while
   `fired_candidates` pointed at the wrong person, and nothing in-band would
   reveal it. Per-crop calls make the attribution trustworthy by construction.
   Batching must not ship without a measured agreement study. `CropBatch`
   exists so the response-to-person association is already modelled when it
   does.
8. **The judge pass writes one copy of the frame per CANDIDATE, not per
   frame.** `run_track1_judge_inference` builds `image_key` from the
   `alarm_id`, which in the pilot IS the candidate id, so four people in one
   frame produce four identical ~300KB copies of the same 1080p JPEG
   under four keys. The disk READ is now once per frame, but this write is
   not, and it cannot be fixed from `helmet_cv.runner`: eliminating it needs
   either a new parameter on `run_track1_judge_inference` (a signature change)
   or a change to what it does with an existing one (a behaviour change), and
   both are ruled out while `POST /track1/run`, `/track1/run-job`,
   `/track1/batch-run`, the eval stepper and `eval/runner.run_eval` all depend
   on it unchanged. Flagged for a decision rather than worked around.

## Track2 (YOLO training pipeline) — backs the `/track2/*` screens

Ported from the standalone `terraform/review-app/review_app.py` app
(plain stdlib HTTP server, no auth, its own EC2 instance) into this API,
gated by the same session auth as everything else. Business logic lives
in `track2_shared.py`; each router below is thin parameter parsing +
response shaping over it. Nav labels are prefixed "Track2 " to avoid
colliding with the existing PPE `/models` and `/metrics` screens.

- `GET /track2/review` -> `{counts: {pending, approved, rejected}, registries: {group_name: {total, approved, packages: [{model_package_arn, version, approval_status, description, lineage, creation_time}]}}}`.
- `POST /track2/review/{review_id}/approve` / `/reject` -> `{review_id, status}`.
- `POST /track2/review/model/{model_package_arn}/approve` / `/reject` -> `{model_package_arn, approval_status}`. `model_package_arn` is a full ARN (contains `/`), captured via a path converter.
- `POST /track2/review/model/{model_package_arn}/delete` -> `{model_package_arn, deleted: true}`. `400` on a SageMaker error (e.g. package still referenced elsewhere).

- `GET /track2/models?browse_prefix=&weights_browse_prefix=` -> `{registries, latest_approved, generation_sizes, generation_labels, size_labels, default_dataset, training_progress, pending_message, gpu_jobs, queue, automated_queues, dataset_browser: {prefix, folders}, weights_browser: {prefix, folders}}`. `training_progress` is `null` if no pipeline execution has ever run; `pending_message`, when non-null, means "show this instead of the normal training-progress panel" (bridges the CodeBuild-submission-to-visible-execution gap). `browse_prefix` omitted entirely -> `dataset_browser` lands on the nice default subfolder (`default_dataset`'s parent); `browse_prefix=` (present but empty) -> the bucket root, listing everything. `weights_browse_prefix` has no such default -- omitted or empty both mean the bucket root.
- `POST /track2/models/train/dataset` body `{generation, size, epochs=50, img_size=640, batch=16, seed=42, dataset_folder, base_model_folder="", notes=""}` -> `{message, build_id}`. `409` if a run is already in progress anywhere in the account (only 1 GPU slot exists).
- `POST /track2/models/train/stop` -> `{message}`. `409` if nothing is currently Executing. Calls `sagemaker:StopPipelineExecution` -- a stopped execution cannot be resumed, a fresh run has to be started instead.
- `POST /track2/models/queue/add` (same body as above) -> `{message, entry}`. `400` if the queue is full (max 10) or params are invalid.
- `POST /track2/models/queue/remove/{entry_id}` -> `{removed: true}`. `404` if already started/removed.
- `POST /track2/models/queue/start` -> `{message}`. `400` if the queue is empty.
- `POST /track2/models/queue/stop` -> `{message}` — stops auto-advancing; a run already handed to CodeBuild keeps running.
- `POST /track2/models/automated-queue/add` body `{generation, size, epochs, img_size, batch, seed, dataset_folder, base_model_folder, notes, mode: "daily"|"weekly"|"monthly", daily_hour?, weekly_day?, weekly_hour?, monthly_day?, monthly_hour?}` -> `{message, queue}` — creates the DynamoDB record and its per-queue EventBridge rule.
- `POST /track2/models/automated-queue/update/{queue_id}` (same body) -> `{message, queue}`. `400` if `queue_id` doesn't exist.
- `POST /track2/models/automated-queue/delete/{queue_id}` -> `{deleted: true}`.

- `GET /track2/schedule` -> `{schedule: {enabled, expression}, training_progress, automated_queues}` — the single global retrain schedule, distinct from the per-recipe Automated Queues above (which this page only lists; add/edit/delete go through `/track2/models`).
- `POST /track2/schedule/train-now` -> `{message}`. `409` if a run is already in progress.
- `POST /track2/schedule` body `{mode: "manual"|"daily"|"weekly"|"monthly", ...same hour/day fields as Automated Queue}` -> `{message, expression?}`. `mode: "manual"` disables the EventBridge rule instead of setting an expression.

- `GET /track2/lineage` -> `{records: [...]}` — every DynamoDB lineage record (training runs + baselines), newest first. Read-only.

- `GET /track2/metrics` -> `{rows: [{group_name, size, version, model_package_arn, status, metrics: {mAP50, "mAP50-95", precision, recall, accuracy} | null, description, lineage}]}`. Bar-chart rendering is frontend-only (the old app rendered an inline SVG server-side); preserve its exact 8-color CVD-safe palette (`#3987e5 #d95926 #199e70 #c98500 #d55181 #008300 #9085e9 #e66767`) for continuity.

Only ONE model is ever live at a time, by explicit product decision -- there is a single shared active Inference Component (`track2-yolo-active`), not one per size. Deploying any size/version replaces whatever was there before, regardless of what size that was; there's no way to have two sizes both "Ready for inference" simultaneously. This was a deliberate simplification over an earlier per-size design (4 independent, always-visible slots), which read as confusing next to a single "what's live right now" answer.

- `GET /track2/deploy?browse_prefix=` -> `{registries, deployed: {active_size, status, model_name, last_modified, source, version, current_copy_count, desired_copy_count, running_inference} | null, folders, weight_files, inference_queue: [...], inference_queue_status: {started, worker_alive}}`. `deployed` is `null` if nothing's ever been deployed. `active_size` (`"s"`/`"m"`/`"l"`/`"x"`) is read back off the deployed Model resource's own name, not tracked separately. `source` is a friendly label ("YOLO11-S v3" or "S3: weights/s/model.pt") and `version` the bare registry version number ("12", `null` for an S3-sourced deploy) -- both read back off tags (`track2-source`/`track2-version`) on the SageMaker Model resource (set at deploy time), durable across an app restart. Both `null` only for a component deployed before this tagging existed, or if `sagemaker:AddTags` isn't granted (deploy still succeeds either way, just without the labels -- see `_create_model_tagged`). `current_copy_count: 0` means scaled to zero (idle 30+ min) -- there's no manual "spin up" endpoint; queuing (and starting) any deploy brings it back, since every deploy always requests `CopyCount: 1` regardless of the count beforehand. `running_inference` is `true` while this app itself has an `invoke_inference` call in flight (in practice, mid-way through an eval run or an inference-queue job below) -- it can't see inference traffic from anywhere else. `folders` is an arbitrary bucket-relative folder listing under `browse_prefix` (same convention as the training weights-folder browser, i.e. not scoped under `weights/{size}/`); `weight_files` is `.pt`/`.tar.gz` files directly under that same `browse_prefix`, for picking one specific checkpoint file to deploy.

Inference queue: the only way to deploy -- there's no separate one-shot deploy endpoint. Queue up several (model, images folder, output folder) jobs (up to `MAX_QUEUE_SIZE` = 10 still-`queued` at once), run one at a time -- each job deploys its model (an Approved registry version, or your own S3 weights -- exactly one of `weights_folder`, an arbitrary bucket-relative folder assumed to hold `model.pt`/`model.tar.gz`, or `weights_key`, an exact `.pt`/`.tar.gz` object of any filename, repackaged into a tarball named after that file's own stem so picking several different checkpoints from the same folder over time doesn't clobber each other's copy; either source deploy repackages onto `code/inference.py` if the tarball doesn't already have it, since the PyTorch serving container otherwise silently falls back to a default handler that rejects a plain YOLO checkpoint), swapping out whatever's currently active, waits for it to become `InService`, runs `run_batch_inference` over its own `images_folder` (writing `<output_folder>/<image_stem>.result.json` per image, a per-image failure landing in the job's own `result.errors` list rather than failing the whole job, plus one `<output_folder>/lineage.json` -- `{bucket, images_folder, output_folder, files: [...], model: {size, weights_kind, weights_label, version, model_package_arn, weights_folder, weights_key, notes}, generated_at}` -- so the folder is legible on its own without cross-referencing the in-memory queue history), then moves to the next queued job. Redeploying the exact same model/version already `InService` with a running copy is skipped entirely (no new Model resource, no reload) -- only a genuine version/size switch actually redeploys. DynamoDB-backed (`track2-yolo-inference-queue`, same shape as the eval queue's own table) -- every job's full history (status, errors, result) survives an app restart; only the in-memory started/worker-alive flags reset.

- `POST /track2/deploy/inference-queue/add` body `{size, weights_kind: "registry"|"s3", images_folder, output_folder, model_package_arn?, weights_folder?, weights_key?, notes?, comment?}` -> `{message, item}`. `model_package_arn` required for `weights_kind="registry"` (must already be Approved -- checked when the job actually deploys, not at queue time). For `"s3"`, give exactly one of `weights_folder`/`weights_key`; `notes` is a free-text description of the checkpoint, only meaningful for `"s3"` (a registry version already has its own recorded metadata). `comment` is a free-text note about the run itself, regardless of `weights_kind`. `422` if the wrong weights combination is given (neither, or both, or an unknown size/kind), if `images_folder`/`output_folder` is empty, or if 10 items are already `queued`. `item` is `{id, status, size, weights_kind, weights_label, model_package_arn, weights_folder, weights_key, notes, comment, images_folder, output_folder, created_at, started_at, completed_at, result, error}`; `status` is `queued` -> `deploying` -> `running` -> `done`|`failed`. `result` (once `done`) is `{processed, total, output_folder, errors: [...]}`, same shape `run_batch_inference` returns.
- `POST /track2/deploy/inference-queue/remove/{item_id}` -> `{removed: true}`. Despite the name, deletes the record outright at any status -- `queued`, `done`, or `failed` -- refusing only one still actively `deploying`/`running` (`404`, same as a nonexistent id).
- `POST /track2/deploy/inference-queue/start` -> `{message}`. `400` if nothing is queued. Starts (or confirms already-running) the background worker thread.
- `POST /track2/deploy/inference-queue/stop` -> `{message}`. Stops picking up further queued items once the current one (if any) finishes -- no clean way to abort a job already mid-run.

- `GET /track2/notifications` -> `{subscriptions: [{subscription_arn, email, confirmed}]}` — `confirmed: false` means SNS's own `PendingConfirmation` placeholder (expires in 3 days, no API to cancel it early).
- `POST /track2/notifications/add` body `{email}` -> `{email, message}`.
- `POST /track2/notifications/remove` body `{subscription_arn}` -> `{removed: true}`.

- `GET /track2/export?browse_prefix=` -> `{registries, folders}`.
- `POST /track2/export` body `{model_package_arn, destination_folder}` -> `{message, new_key, archived_key}` — writes to `<destination_folder>/yolo11-<size>-v<version>-<YYYYMMDD>.pt` (size/version parsed from `model_package_arn`'s own `yolo11<size>-detector/<version>` naming); archives any existing file at that exact key first (server-side S3 copy) rather than clobbering it.

No `/track2/help` endpoint — the Help page is static content (dataset
folder/label-format documentation), rendered directly in React with no
backend call.

## Track2 evaluation queue (`/track2/eval`) — backs the `/track2/eval` screen

Runs a registry version OR your own S3 `.pt` weights against a chosen
dataset folder and computes real detection metrics — persisted in
DynamoDB (`track2-yolo-eval-runs`) so results survive a restart. Ground
truth format matches the training dataset convention documented on the
Help page: `<data_folder>/images/*.jpg` + `<data_folder>/labels/*.txt`
(YOLO: `class_id cx cy w h`, normalized 0-1) — an image with no matching
label file is treated as zero ground-truth objects, not an error.

Evaluating deploys the requested weights first (the same path a real
deploy already uses — `track2_shared.deploy_inference_component`), so a
run replaces whatever was on the one shared active Inference Component
the same way a manual deploy would — leaving that run's weights deployed
afterward too, same as any deploy. Jobs run one at a time on a background thread (not
the asyncio event loop — each job makes one `invoke_inference` HTTP call
per image plus a wait for the deploy to become `InService`, which would
otherwise stall every other request in this single-worker process for as
long as the job takes).

- `GET /track2/eval` -> `{runs: [...], queue: {started, worker_alive}, registries, generation_sizes, size_labels}`. `runs` is every queued/running/completed/failed run, newest first; each has `{id, status, size, data_folder, weights_kind, weights_label, model_package_arn, s3_folder, created_at, started_at, completed_at, metrics, error}`. `status` is `queued` -> `deploying` -> `running` -> `completed`|`failed`. `metrics` (once `completed`) is `{mAP50, "mAP50-95", precision, recall, accuracy, num_images, num_classes}` — `accuracy` here is `TP / (TP+FP+FN)` at IoU 0.5, the closest single-number proxy object detection has (reported alongside precision/recall, not instead of them).
- `POST /track2/eval/add` body `{size, data_folder, weights_kind: "registry"|"s3", model_package_arn?, s3_folder?}` -> `{message, run}`. `model_package_arn` required for `weights_kind="registry"`, `s3_folder` required for `"s3"` (an arbitrary bucket-relative folder holding `model.pt`/`model.tar.gz`, same convention as Deploy's own raw-S3 weights folder) — `422` if the wrong one (or neither) is given, or `data_folder` is empty.
- `POST /track2/eval/remove/{run_id}` -> `{removed: true}`. Only removes a run still `queued` — `404` if it's already started, finished, or doesn't exist; once started, the record stays as history.
- `POST /track2/eval/start` -> `{message}`. `400` if nothing is queued. Starts (or confirms already-running) the background worker thread, which processes the oldest queued run, then the next, until none are queued or `/stop` is called.
- `POST /track2/eval/stop` -> `{message}`. Stops picking up further queued runs once the current one (if any) finishes — no clean way to abort a run already mid-evaluation.
