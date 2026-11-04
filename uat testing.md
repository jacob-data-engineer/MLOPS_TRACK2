# UAT testing — Track2 Retraining Platform

## Outcome

**UAT executed on development; not accepted for unrestricted release.** Authentication was completed after the operator supplied the password. This report supersedes the earlier login-blocked draft.

Test window: **4 November 2026, approximately 17:30–18:10 SGT**. Build: **`20261103-115240`**. `/build` reported `dev`, port `8090`, record backend `postgres`. Browser: isolated headless Google Chrome driven through Playwright, 1440 × 1000 viewport. Application: http://18.140.196.76:8090.

Of the 29 original cases: **23 PASS, 6 FAIL, 0 BLOCKED**. One additional schedule/cleanup case failed. Multiple failed cases share a cause; these are not seven independent defects. Results describe the observed development build, not an assurance about production or all datasets.

Screens were used for user actions. Authenticated API reads, S3 object inspection and ZIP inspection corroborated the browser results. Source inspection supports proposed fixes; it is not counted as live execution. No application code was changed, deployed or committed.

## Independent verification — 4 November 2026, 19:25–19:36 SGT

**Verdict: the fixes are deployed, but not all defects can be closed.** This section is the latest independent result. The Remediation section below records the implementer's earlier claims; the original FAIL table is historical evidence, not the current retest outcome.

Verified live development build **`20261104-111726`**, PostgreSQL backend, port **8090**, before and after testing. Local source HEAD was `43b6169`. No production access, training, schedule Apply, database start, new dataset freeze, or deletion of source images was performed. The independent test reused the already-created smallest nonempty dataset, Version 83. Application code was not changed.

| Defect | Independent result | Evidence |
|---|---|---|
| UAT-02: stopped database query | **PASS for original reproduction** | Loaded no helmet from the running database, switched to stopped preview, observed 25 seconds: zero SQL requests after switching, no spinner, no HTTP errors or console errors, Start button shown. [Recorded result](uat-evidence/2026-11-04/independent-retest/04-stopped-db.json), [screen](uat-evidence/2026-11-04/independent-retest/04-stopped-db.png). |
| UAT-03: saved card restoration | **PASS** | Saved default-open Assemble as `uat_independent_DP`, clicked Hide, loaded saved file: Assemble reopened. Final operational configuration matches the baseline. [Restored screen](uat-evidence/2026-11-04/independent-retest/03-card-restored.png), [load state](uat-evidence/2026-11-04/independent-retest/03-after-load.json). |
| UAT-04: PostgreSQL status/refresh | **PASS in PostgreSQL mode** | Card says records are written straight to PostgreSQL. Refresh returned 200 and showed `Counts read 04 Nov 19:28 SGT.` No obsolete DynamoDB message on this card. [Screen](uat-evidence/2026-11-04/independent-retest/02-central-refresh.png), [text](uat-evidence/2026-11-04/independent-retest/02-dp-refreshed.txt). DynamoDB-mode polling was not live-tested. |
| UAT-05: deleted file remains loaded | **PARTIAL — UI reproduction fixed; durability needs investigation** | DP and AT deletion responses returned the matching cleared reference. Scheduled showed no deleted name. However, a later configuration read again contained `settings_file_dp: uat_independent_DP` although the file was absent. A second controlled save/delete cleared it and final checks passed. [First deletion](uat-evidence/2026-11-04/independent-retest/03-dp-deleted.json), [AT deletion](uat-evidence/2026-11-04/independent-retest/05-at-deleted.json), [stale reference after deletion](uat-evidence/2026-11-04/independent-retest/09-cleanup.json), [final cleanup](uat-evidence/2026-11-04/independent-retest/10-final-cleanup.json). |
| UAT-01: preserved-file exports | **PARTIAL — deleted-original version export passes; reproducibility gaps remain** | Fresh Version 83 export `cba96aff0145`: ready, 144 files, zero missing. ZIP contains 72 valid images, 72 matching labels, 144 manifest rows, data.yaml, matching SHA-256 values. Scratch originals are absent and sampled preserved objects exist. Overwritten originals, run exports, and label completeness still fail isolated checks below. |

### Successful export evidence

- [Fresh export record](uat-evidence/2026-11-04/independent-retest/06-exports.json), [browser screen](uat-evidence/2026-11-04/independent-retest/06-export-ready.png).
- [Downloaded ZIP validation](uat-evidence/2026-11-04/independent-retest/08-zip-validation.json): all 72 images decoded, every image/label pair matched, no missing entries.
- [Read-only source verification](uat-evidence/2026-11-04/independent-retest/07-source-verification.json): scratch prefix has zero objects; sampled original image and label HEAD return 404; corresponding preserved image and label exist.
- Retained artifact: `s3://track2-s3/label-curation/validate_v1/export/_exports/version-83-cba96aff0145/version-83.zip`.
- Historical Version 46 is still incomplete; this was explicitly excluded from repair and is not a regression in the new preservation feature.

### Remaining issues found by independent verification

**R1 — P1: overwritten originals silently change a frozen export.**

Reproduced entirely in memory using the actual export builder with mocked S3/database access: original address contains `CHANGED-AFTER-FREEZE`; preserved address contains `ORIGINAL-AT-FREEZE`. Export returned **ready** and contained **CHANGED-AFTER-FREEZE**. No live source was overwritten.

Cause: `src/ppe_pipeline_api/data/catalog_view.py:392` fetches the original first and consults the stored copy only if the original is absent. This protects deletion but not overwriting, so the claim that frozen versions remain reproducible is too broad.

Fix direction: prefer the frozen stored object when available; if an original must be used, verify it against the recorded frozen identity and report mismatch. Add a behavioral regression test for an overwritten image and an overwritten label, not merely a source-string assertion. [Reproduction result](uat-evidence/2026-11-04/independent-retest/preservation-edge-cases.json).

**R2 — P1: run-based exports do not use preserved copies.**

An isolated run linked to a preserved version was exported with its original missing and preserved bytes present. Actual result: **partial, zero files, one missing**. The `_plan('run', ...)` query reads run_files and uses the version only for class names; it never attaches the stored mapping that version exports receive. This affects the user's Files in Run export path even when its dataset version has been preserved.

Fix direction: resolve each audited run file to the matching preserved member using the linked version and original URI; preserve the audited role/split/layout. Do not assume that every run file exists in the version. Test version-vs-run exports with deleted and changed originals. [Isolated reproduction and executed queries](uat-evidence/2026-11-04/independent-retest/08-run-preservation-gap.json). This is a code-level behavioral reproduction, not a destructive live run test.

**R3 — P2: preservation summary ignores missing labels.**

With an image present and its expected label absent, the actual preservation routine returned `files: 1, kept: 1, missing: 0` while `stored_label` was empty. `central_store.py:106–107` counts preserved images only. `pending()` at line 195 only selects versions with `preserved_at IS NULL`, so completed-but-incomplete preservation is not selected by that retry condition.

Fix direction: distinguish image counts, expected labels, missing labels, and intentionally unlabeled data; record partial preservation explicitly and retry recoverable incomplete copies. This finding concerns preservation bookkeeping; it does **not** claim the ZIP exporter reports a missing label as successfully included. [Reproduction result](uat-evidence/2026-11-04/independent-retest/preservation-edge-cases.json).

**R4 — P2: deleted DP reference reappeared after a successful clearing response.**

The first DP delete returned `cleared: ["settings_file_dp"]`, but a later authenticated read showed the deleted `uat_independent_DP` again while the file list contained no such file. This was observed live. The exact interleaving is not established: a stale concurrent configuration write is a hypothesis, not a proven cause. `pull_control.write_config()` at line 442 performs read/merge/write of the entire configuration without an atomic patch across that sequence, which merits investigation.

Fix direction: trace concurrent configuration writes and make updates atomic or revision-checked; test delete racing with card autosave/background writes. Keep the Scheduled missing-file guard. Recreating only the same UAT file and deleting it again cleared the reference; final baseline comparison has **zero differences**. Do not mark durable reference cleanup fully accepted from the response alone.

### Tests, cleanup and limits

- Focused existing regression suite: **121 passed**, one dependency deprecation warning. Command: `.venv/Scripts/python.exe -m pytest tests/test_central_store.py tests/test_settings_split_by_tab.py tests/test_catalog.py tests/test_sql_explorer.py -q`.
- New isolated edge checks above deliberately exercise cases missing from that passing suite. They use in-memory S3/database doubles and do not modify live objects or application source.
- UI help correction for Assemble and PostgreSQL status was observed live. Broader wording/documentation changes are present in source; this retest does not claim every caption or every guide paragraph was independently exercised.
- `uat_independent_DP` and `uat_independent_AT` were removed. The DP name was recreated once solely to recheck/clear the stale reference, then deleted again.
- Final server configuration matches the newly captured baseline exactly, including schedules, model/data settings, card states and loaded-file references. [Final verification](uat-evidence/2026-11-04/independent-retest/10-final-cleanup.json).
- One later read encountered a connection reset; a read-only retry succeeded on the same build. This did not prove an additional application defect. Temporary browser authentication state was removed at completion.
- Fresh Version 83 export and local evidence are retained. No source images, labels, historical records, or pre-existing settings files were deleted.
- This was a targeted defect retest, not another full 29-case run. Production, actual training, schedules firing, database-start permission, large exports, model/PT metadata compatibility and stop/cancel remain outside this verification.

**Next action:** fix and retest R1–R4 before treating the preservation feature and durable settings cleanup as fully accepted. No application fixes or Git commits were made during this independent verification.


## Remediation (4 November 2026, 18:30–19:20 SGT)

Every defect and confusing-text item below was fixed on branch `export`, deployed to development (8090), **retested live before commit**, and committed and pushed one fix per commit. Production (8501, `20261026-124922`) was not touched. Development now runs **`20261104-111726`** (record backend `postgres`, schema `track2_dev`, shared Lambdas skipped, as `deploy-dev.ps1` sets). Automated suite: **2,563 passed, 0 failed**.

| Defect | Commit | Fix | Live retest (development) | Result |
|---|---|---|---|---|
| UAT-02 (E2/G2) | `f49bcb4` | Class list tied to the database it came from; late answers for a previous database dropped; no query while the chosen database is not available | Build `20261104-103413`: 60 thumbnails shown from violation-review, switched to stopped preview, watched 25 s — no class/preset/query request, no HTTP ≥400, no console errors, no spinner; stopped status and Start button shown | **PASS** |
| UAT-03 (B3) | `ca1c5bc` | On load, the loaded tab's card states are cleared before the file's are applied, so an unsaved default returns; other tab's cards kept | Build `20261104-105113`: Assemble open at default → saved `uat03_DP` → Hide → closed → load → **open**; training settings unchanged; file deleted, card states restored | **PASS** |
| UAT-04 (D2/D3) | `f41f49a` | PostgreSQL mode: card says records are written straight to PostgreSQL with **Refresh** ("Counts read … SGT"); `/sync` answers `started:false, direct:true` instead of "Copying"; DynamoDB mode watches the copy until done (5 s × up to 5 min) and reports failure; status carries `checked_at` | Build `20261104-105844`: `/sync` → direct; no DynamoDB wording or Sync now on the card; Refresh → "Counts read 04 Nov 19:02 SGT" | **PASS** |
| UAT-05 (H1) | `ae16a6b` | Deleting a file clears only matching `settings_file` / `_dp` / `_at`; Scheduled treats a missing file as unsaved live settings | Build `20261104-110319`: Scheduled already showed live settings for stale `uat_AT`; saved+deleted `uat05_AT` and `uat05_DP` → each loaded name cleared; settings/schedules unchanged; Scheduled names no deleted file. Stale `uat_AT` and `uat03_DP` references cleared | **PASS** |
| Confusing text + `documentation.md` | `9a15a5b` | Pull control links to Multi-Auto Training for the schedule; Assemble caption says nothing is copied; SQL explains approximate split and what Test-grade does not promise; user guide updated to this baseline; `uat-evidence/` excluded from deploy, Docker and git | Build `20261104-111157`: new caption shown, old gone; shipped bundle has the link and both explanations; no `uat-evidence` in the image | **PASS** |
| UAT-01 (F5) | `43b6169` | At freeze, every image + label is copied server-side into the central data store (`objects/<etag><ext>`, stored once, never deleted), recorded in `_datasets/<id>/preserved.json` and the catalog (schema step 8); exports fall back to the stored copy; tick backstop for interrupted copies. Versions frozen before 2026-11-04 are **not** back-filled | Build `20261104-111726`: froze **Version 83** from a scratch copy of the UAT pull; 72/72 preserved; **deleted all 144 scratch originals**; export → ready, 144 files, **0 missing**, valid zip (72 images + 72 labels) | **PASS** |

Retests were run with an isolated headless Chromium (Playwright) and authenticated API/in-container checks. They cover the failed cases only; the cases this report already left untested (training execution, schedules firing/Apply, Start permission, production, large exports, model/metadata export, stop/cancel) remain untested.

## Baseline and safety

The supplied checklist predates the user's accepted changes. Tests use these current requirements:

- Sidebar: SQL, Data Preparation, Multi-Auto Training, Data Catalog, then remaining sections.
- Both pull and training schedules belong to Multi-Auto Training. Data Preparation has file controls, without the GPU/schedule strip, Assemble Manual/Scheduled radio, or Pull Queue.
- PostgreSQL deployments must identify PostgreSQL as their record backend.
- Scheduled Prototype displays Multi-Auto Training timing and `_AT` previews only.
- Names normalize a suffix once: `uat_dp` becomes `uat_DP`, and `uat_at` becomes `uat_AT`.

Port 8501 was not accessed. No Pull now/Run now, training submission, schedule Apply, schedule edit committed, database start/stop, or historical-record repair was performed. The SQL pull used only the permitted UAT destination. Only UAT-created settings files were deleted. The export used the smallest nonempty dataset version, Version 46 (100 images); the zero-member snapshot was not exported.

## Results

Evidence links are relative to this report. Counts describe observations at test time; the application contains live data.

| ID | Result | Steps and observed evidence |
|---|---|---|
| A1 | PASS | Inspected sidebar: SQL → Data Preparation → Multi-Auto Training → Data Catalog. Data Preparation/training green; Scheduled/Review yellow. [Screen](uat-evidence/2026-11-04/13-cargo-settled.png). |
| A2 | PASS | Deploy & Inference absent from sidebar; direct `/track2/deploy` opens endpoint/inference page. No deployment action used. [Screen](uat-evidence/2026-11-04/06-deploy-route.png). |
| A3 | PASS | Build badge reads `dev · 20261103-115240`. [Preflight](uat-evidence/2026-11-04/00-preflight.json). |
| B1 | PASS | Saved `uat_dp`; API 200 named `uat_DP`, 14,507 bytes. Pinned DP toolbar lists DP files only. [Save result](uat-evidence/2026-11-04/04-dp-save-result.png). |
| B2 | PASS | Saved `uat_at`; UI displayed Saved/Loaded `uat_AT`; AT toolbar excludes DP files. [Screen](uat-evidence/2026-11-04/05-at-save.png). |
| B3 | FAIL | Saved DP with Assemble open, closed it, loaded saved DP. Assemble stayed closed. Model/training settings and clocks remained unchanged: isolation passes, card restoration fails. [Isolation](uat-evidence/2026-11-04/07-settings-isolation.json), [saved recipe](uat-evidence/2026-11-04/25-saved-dp.json). |
| B4 | PASS | Each Delete required a Yes confirmation. Both UAT files ultimately disappeared; final settings list contains only the three pre-existing DP files. [Confirmation](uat-evidence/2026-11-04/25-confirm-uat_DP.png), [verified list](uat-evidence/2026-11-04/26-settings-cleanup.json). Separate dangling-reference defect below. |
| C1 | PASS | Corrected requirement: DP has no pull schedule/GPU strip; Multi-Auto Training has both schedules. [DP](uat-evidence/2026-11-04/23-dp-final.png), [training](uat-evidence/2026-11-04/28-after-cleanup-at.png). |
| C2 | PASS | Selected arrival/follow-pull context without Apply; guards showed at least 1 file and at most one run every 0 min, with GPU warning. Reload discarded draft; final schedules remain off. [Guards](uat-evidence/2026-11-04/05-arrival-guards.png). |
| D1 | PASS | Corrected requirement: Assemble Manual/Scheduled radio absent. [DP screen](uat-evidence/2026-11-04/23-dp-final.png). |
| D2 | FAIL | Central store has bucket/folder and table counts, but says `Not copied from DynamoDB yet.` while backend is PostgreSQL. [Screen text](uat-evidence/2026-11-04/23-dp-final.txt), [status](uat-evidence/2026-11-04/final-catalog-status.json). |
| D3 | FAIL | Clicked Sync now; 200 returned `Copying in the background.` Revisited many minutes later: same message, `last_sync: null`. No updated timestamp. This is also the wrong workflow wording for direct SQL recording. [Initial screen](uat-evidence/2026-11-04/07-sync.png), [later status](uat-evidence/2026-11-04/final-catalog-status.json). |
| E1 | PASS | Database choices include running violation-review-postgres and stopped jpui-preview-crop-authority. [SQL screen](uat-evidence/2026-11-04/08-sql-initial.png). |
| E2 | FAIL | Switched running → stopped preview database: Start button appeared, but Finding the frames remained while a query attempted the stopped host and returned HTTP 400 timeout. Start was deliberately not pressed. [Screen](uat-evidence/2026-11-04/13-stopped-settled.png). |
| E3 | PASS | No helmet listed 5,649 judged frames; choosing it automatically displayed thumbnails. Count differs from the checklist's approximate older value. [SQL](uat-evidence/2026-11-04/27-reloaded-sql.png). |
| E4 | PASS | Test-grade: 110; Ready: 1,270. For cargo, Ready/Test-grade options were disabled. [Test-grade](uat-evidence/2026-11-04/10-sql-test-grade.txt), [Ready](uat-evidence/2026-11-04/11-sql-ready.txt), [cargo](uat-evidence/2026-11-04/13-cargo-settled.txt). |
| E5 | PASS | Sampled visible frames show blue person and green helmet boxes and red missing-box borders. UI reports 38 missing-box frames. This verifies rendering, not annotation accuracy across every image. [Overlay screen](uat-evidence/2026-11-04/27-label-overlay.png). |
| E6 | PASS | Cargo images displayed, including the extensionless-image class. Some images were still lazy loading in screenshots; that alone was not classified as a failed image request. [Settled cargo screen](uat-evidence/2026-11-04/13-cargo-settled.png). |
| E7 | PASS | Pulled no helmet/Test-grade into `track2-s3/uat/2026-11-04`, 80% train setting, labels enabled: found/copied 72, labels 72, train 54, val 18, incomplete left out 38, failures 0. S3 pairs and metadata verified. [Result](uat-evidence/2026-11-04/16-pull-1.json), [S3 check](uat-evidence/2026-11-04/16-s3-verification.json). |
| E8 | PASS | Repeated same SQL pull: copied 0, skipped 72. [Result](uat-evidence/2026-11-04/16-pull-2.json). |
| E9 | PASS | Advanced SQL initially folded. Opened editor; SELECT 1 returned one row. DELETE FROM cases WHERE false was refused by read-only transaction. Table-loading control observed; full schema enumeration was not separately verified. [Read](uat-evidence/2026-11-04/14-select1.png), [refusal](uat-evidence/2026-11-04/14-write-refused.png). |
| F1 | PASS | 58 runs, descending 58/57/56…, submitted SGT, status/model/version/images. Run 58: succeeded, helmet/yolo11s, Version 78, 9,744 images. [Catalog](uat-evidence/2026-11-04/17-catalog.png). |
| F2 | PASS | 82 versions, version/snapshot kinds; Version 78 Used by includes Run 56 and Run 58. [Rows](uat-evidence/2026-11-04/19-versions.json), [screen](uat-evidence/2026-11-04/19-versions.png). |
| F3 | PASS | Run 58 files paged 1–100 → 101–200 → back. Split and use filters changed results; train showed 7,996 images; thumbnails displayed. [Filters](uat-evidence/2026-11-04/20-catalog-filters.png). |
| F4 | PASS | Downloaded Run 58 CSV: 19,488 image/label rows; headers include s3_uri, role, split, kind, folder. [CSV](uat-evidence/2026-11-04/18-run58-files.csv). |
| F5 | FAIL | Smallest nonempty version, 46: building → Incomplete, 120 files and 80 missing, 27.0 MB. Download works; 60 images decode, 60 labels pair, data.yaml/manifest exist. Full reconstruction fails because sources are unavailable; UI correctly reports it. [Export screen](uat-evidence/2026-11-04/22-export-result.png), [ZIP verification](uat-evidence/2026-11-04/22-zip-verification.json). |
| G1 | PASS | Reloaded all six visited functional pages. SQL retained the selected database/class/scope/destination; Catalog, Scheduled and Deploy control values matched. Settled DP reload restored the original file and open Assemble card; training restored vendor-reproduction-20261030/yolo26l. This covers exercised settings, not every possible combination. [Reload checks](uat-evidence/2026-11-04/29-reload-checks.json), [settled checks](uat-evidence/2026-11-04/30-settled-reload.json). |
| G2 | FAIL | Unexpected SQL database-switch HTTP 400 and console error (E2). Intentional read-only DELETE also generated expected HTTP 400. No uncaught JavaScript exception observed in the recorded runs. |
| G3 | PASS | Both UAT settings files deleted and verified absent. No database was started. S3 UAT objects retained as instructed. Display/SQL/export preferences restored; schedules/model/data configuration unchanged. One stale AT filename reference remains as a defect. [Final comparison](uat-evidence/2026-11-04/28-final-verification.json). |
| H1 (additional) | FAIL | Scheduled Prototype initially showed only `uat_AT`, SGT columns, no events because schedules off. After that file was deleted, it still displays `uat_AT — loaded`; training toolbar instead says no file loaded. [After deletion](uat-evidence/2026-11-04/28-after-cleanup-scheduled.png). |

## Defects and recommended fixes

### UAT-01 — P1 for dataset reproducibility: missing export source objects (F5)

**Status: FIXED for versions frozen from 2026-11-04 — commit `43b6169`.** Version 46 itself is not repaired (historical repair out of scope); its export still reports the 80 missing entries truthfully.

**Reproduce:** Catalog → Dataset versions → Version 46 → dataset ZIP → prepare → download incomplete ZIP.

**Expected:** All 100 recorded images and corresponding labels can be reconstructed. **Actual:** 60 images + 60 labels present; 80 of 200 manifest entries marked missing. All 60 present images decoded and every present image had a label. This is not evidence that the ZIP writer corrupts files.

A direct S3 HEAD on one missing source returned **404 Not Found**:
`s3://track2-s3/golden_set/images/train/10092025_Gloves_Hatch_01_02_Gloves_Hatch_01_frame_0139.jpg`.

Evidence: [source check](uat-evidence/2026-11-04/22-source-check.json), [manifest](uat-evidence/2026-11-04/22-manifest.csv), [ZIP checks](uat-evidence/2026-11-04/22-zip-verification.json).

**Fix direction:** Preserve the Incomplete state and missing manifest. For future runs, retain immutable/versioned source objects and record exact object identity at dataset freeze; validate availability before claiming a complete export. Investigate historical mapping/restoration separately only if authorized: the user previously excluded historical repairs. Do not create substitute labels or claim this dataset is reproducible. Regression: all-available export, image missing, label missing, and expired/moved source all produce truthful counts and status.

### UAT-02 — P2: switching to a stopped SQL database launches a query (E2/G2)

**Status: FIXED — commit `f49bcb4`.**

**Reproduce:** On SQL, load a class from violation-review; switch database to stopped preview.

**Expected:** Cancel old request/state, show stopped/start information, no frame query until available. **Actual:** spinner and HTTP 400, with:

> InterfaceError: Can't create a connection to host jpui-preview-crop-authority.cf08ywocgcwq.ap-southeast-1.rds.amazonaws.com and port 5432 (timeout is 15 and source_address is None).

Likely code path: `web/src/routes/Track2SqlPage.tsx`, class/preset effects around lines 180–215. Preset effect checks database/pickedClass but not database availability; a previous class can survive the switch render. **Fix:** gate by availability and class data belonging to the selected database; invalidate/cancel pending requests and clear busy state on switch. Regression: running→stopped, rapid A→B→A, late old response, failed query, stopped→available.

### UAT-03 — P2: saved DP file does not restore default-open card state (B3)

**Status: FIXED — commit `ca1c5bc`.**

**Reproduce:** Save with Assemble open by default; close Assemble; load the saved DP file.

**Expected:** Assemble opens as saved, without changing AT settings. **Actual:** Assemble stays closed. Saved recipe omits `assemble`; live record has `assemble: false`. AT isolation check passed.

Likely cause: `web/src/lib/cardState.ts` records explicit toggles, while `src/ppe_pipeline_api/pull_control.py` (`scope_slice`, `load_settings_file`, around lines 3057 and 3422) merges saved card keys onto live keys. An omitted default cannot overwrite a later toggle. **Fix:** save effective scoped defaults or replace/reset only the loaded scope's card keys before merging saved values. Preserve other-tab keys. Regression: default-open and explicit-open/closed round trips in both scopes, while proving other-tab settings unchanged.

### UAT-04 — P2: PostgreSQL Central Data Store presents obsolete DynamoDB sync (D2/D3)

**Status: FIXED — commit `f41f49a`.**

**Reproduce:** Development DP → Central data store → Sync now; revisit after more than a minute.

**Expected:** PostgreSQL status accurately describes direct writes; any refresh completes visibly. **Actual:** `Not copied from DynamoDB yet.` remains; Sync answers `Copying in the background.`; later status reports `record_backend: postgres`, `last_sync: null`. Catalog itself correctly says PostgreSQL.

**Fix:** branch presentation and action by backend in `web/src/components/Track2CentralStorePanel.tsx` (hardcoded message around lines 167–170). For SQL, show direct recording and a Refresh action with completion/error state; reserve copying timestamps for a real DynamoDB projection. For actual background jobs poll until terminal status instead of a single delayed refresh. Regression: SQL mode, Dynamo projection success/failure, slow completion, unavailable database.

### UAT-05 — P2: deleted settings file remains the loaded schedule identity (H1)

**Status: FIXED — commit `ae16a6b`.**

**Reproduce:** Save AT settings as `uat_AT`; delete with confirmation; reload training and Scheduled Prototype.

**Expected:** No page claims a deleted file is loaded; live settings remain unchanged. **Actual:** file list excludes `uat_AT`, training toolbar says `Loaded: no file loaded`, but Scheduled says `uat_AT — loaded`. Final config still contains `settings_file_at: uat_AT` (baseline empty).

**Fix:** in `src/ppe_pipeline_api/pull_control.py:3753`, deleting a file should conditionally clear matching loaded-file references while preserving configuration and schedules. Scheduled should also tolerate a reference to a nonexistent file and label live unsaved settings honestly. Test deleting the loaded AT/DP file and a nonloaded file; ensure no schedule changes. This report leaves the stale reference visible as evidence rather than invoking a full settings reset to hide it.

## Confusing text and coverage limits

**Status: the four wording items and the documentation item are FIXED — commit `9a15a5b`.** The coverage limits below are unchanged.

- DP Pull Control still directs users to the Pull row in the toolbar at the top of that page, although the schedule moved to Multi-Auto Training. Link to the correct page.
- Assemble introduction says it copies into images/labels layout, while later instructions say nothing is copied and existing folders are used. Explain the actual operation consistently.
- SQL Test-grade displays 110 frames but excludes 38 from the pull for missing helmet boxes. Export reports this correctly; clarify before clicking that Test-grade is not a guarantee every frame has complete exportable labels.
- 80% train is a deterministic per-address split, not an exact small-batch ratio: observed 54/72 (75%). Explain approximation; repeat-pull stability passed.
- `documentation.md` still describes older schedule/card placement. Update user and UAT instructions with the accepted baseline above.
- Deliberately untested: training execution, schedules firing/Apply, database Start permission, production, large exports, model/PT metadata export and stop/cancel controls (not exercised by this supplied checklist). The new stop/model features therefore do not receive a pass from this run.
- SQL write rejection was expected: `cannot execute DELETE in a read-only transaction` (PostgreSQL code `25006`). Do not count it as an additional malfunction.
- Screenshots captured lazy-loading images; blank tiles during loading were not independently established as persistent defects.
- G1 passed for exercised settings. Early six-second snapshots of DP/Training differed because settings were still loading; settled assertions confirmed restoration. Exhaustive combinations of every card remain outside this run. No assertion of production readiness or complete regression coverage.

## Artifacts, cleanup and retained state

| Item | Created/result | Cleanup |
|---|---|---|
| `uat_DP` settings | Saved for DP round trip | Deleted; absent from final server list. |
| `uat_AT` settings | Saved for AT isolation/calendar test | Deleted; absent from final server list. Stale loaded-file reference remains (UAT-05). |
| SQL pull | `s3://track2-s3/uat/2026-11-04/` | Retained as instructed: 54 train + 18 val images, matching labels, data.yaml, sources.json, query.json. Operator decides retention. |
| Dataset export | ID `fe925ef13119`, Version 46, Incomplete | Retained export row and normal export artifact: `s3://track2-s3/label-curation/validate_v1/export/_exports/version-46-fe925ef13119/version-46.zip`. No source object removed. |
| Local evidence | `uat-evidence/2026-11-04/`, including CSV, downloaded ZIP, screenshots and verification JSON | Retained for reproduction. ZIP contains real dataset images; keep within the project access boundary. |
| Browser testing tools | Temporary isolated Playwright installation | No application dependency changes. Authenticated temporary session state removed after testing. |
| Display/SQL/export settings | Four preference fields changed during UAT | Restored to captured baseline: card_state, sql_explorer, catalog_export, catalog_export_destination. |
| Models/data/schedules | Compared final config to baseline | All checked values unchanged; only remaining config difference is `settings_file_at`. No database started. |

| Remediation test data | `s3://track2-s3/uat/uat01-scratch/` copied from the UAT pull, used to freeze Version 83 (`dv-20261104T111932-f5850b0d`) | Scratch originals **deleted** on purpose to prove preservation (0 left). Version 83, its 72 preserved objects under `s3://track2-s3/label-curation/validate_v1/export/objects/` and export `5b081abbe386` are retained as evidence. |
| Remediation settings files | `uat03_DP`, `uat05_AT`, `uat05_DP` | Each deleted within its test; loaded-name references cleared. |
| ECR image | A deploy at ~19:09 packaged `uat-evidence/`; stopping its CodeBuild run was blocked | One image containing the evidence remains in the private `track2-app` ECR repository, not running. Development was redeployed clean (`20261104-111157` onward). Deleting that image is the operator's decision. |

[Final verification](uat-evidence/2026-11-04/28-final-verification.json) records the exact comparison. Schedule, training and data fields were not included in the cleanup write. The stale AT reference was not cleared by resetting settings because that could affect schedules outside UAT authorization.

## Fix order and retest

**Status:** items 1–4 done (see Remediation). Item 5 — an independent re-run of the failed cases and a wider G1 matrix on the next build — is still to do.

1. Treat missing sources as a blocker for claiming full reproducibility; implement future source-retention/validation safeguards. Historical repair stays out of scope.
2. Fix stopped-database query gating and scoped card restoration.
3. Correct SQL backend status/sync behavior and deleted-file identity handling.
4. Update moved-control help text/documentation.
5. Repeat failed cases and extend the G1 persistence matrix on the next development build. Separately authorize any training/schedule execution or broader feature acceptance test.
