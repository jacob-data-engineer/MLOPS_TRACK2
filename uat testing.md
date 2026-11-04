# UAT testing — Track2 Retraining Platform

## Outcome

**UAT executed on development; not accepted for unrestricted release.** Authentication was completed after the operator supplied the password. This report supersedes the earlier login-blocked draft.

Test window: **4 November 2026, approximately 17:30–18:10 SGT**. Build: **`20261103-115240`**. `/build` reported `dev`, port `8090`, record backend `postgres`. Browser: isolated headless Google Chrome driven through Playwright, 1440 × 1000 viewport. Application: http://18.140.196.76:8090.

Of the 29 original cases: **23 PASS, 6 FAIL, 0 BLOCKED**. One additional schedule/cleanup case failed. Multiple failed cases share a cause; these are not seven independent defects. Results describe the observed development build, not an assurance about production or all datasets.

Screens were used for user actions. Authenticated API reads, S3 object inspection and ZIP inspection corroborated the browser results. Source inspection supports proposed fixes; it is not counted as live execution. No application code was changed, deployed or committed.

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

**Reproduce:** Catalog → Dataset versions → Version 46 → dataset ZIP → prepare → download incomplete ZIP.

**Expected:** All 100 recorded images and corresponding labels can be reconstructed. **Actual:** 60 images + 60 labels present; 80 of 200 manifest entries marked missing. All 60 present images decoded and every present image had a label. This is not evidence that the ZIP writer corrupts files.

A direct S3 HEAD on one missing source returned **404 Not Found**:
`s3://track2-s3/golden_set/images/train/10092025_Gloves_Hatch_01_02_Gloves_Hatch_01_frame_0139.jpg`.

Evidence: [source check](uat-evidence/2026-11-04/22-source-check.json), [manifest](uat-evidence/2026-11-04/22-manifest.csv), [ZIP checks](uat-evidence/2026-11-04/22-zip-verification.json).

**Fix direction:** Preserve the Incomplete state and missing manifest. For future runs, retain immutable/versioned source objects and record exact object identity at dataset freeze; validate availability before claiming a complete export. Investigate historical mapping/restoration separately only if authorized: the user previously excluded historical repairs. Do not create substitute labels or claim this dataset is reproducible. Regression: all-available export, image missing, label missing, and expired/moved source all produce truthful counts and status.

### UAT-02 — P2: switching to a stopped SQL database launches a query (E2/G2)

**Reproduce:** On SQL, load a class from violation-review; switch database to stopped preview.

**Expected:** Cancel old request/state, show stopped/start information, no frame query until available. **Actual:** spinner and HTTP 400, with:

> InterfaceError: Can't create a connection to host jpui-preview-crop-authority.cf08ywocgcwq.ap-southeast-1.rds.amazonaws.com and port 5432 (timeout is 15 and source_address is None).

Likely code path: `web/src/routes/Track2SqlPage.tsx`, class/preset effects around lines 180–215. Preset effect checks database/pickedClass but not database availability; a previous class can survive the switch render. **Fix:** gate by availability and class data belonging to the selected database; invalidate/cancel pending requests and clear busy state on switch. Regression: running→stopped, rapid A→B→A, late old response, failed query, stopped→available.

### UAT-03 — P2: saved DP file does not restore default-open card state (B3)

**Reproduce:** Save with Assemble open by default; close Assemble; load the saved DP file.

**Expected:** Assemble opens as saved, without changing AT settings. **Actual:** Assemble stays closed. Saved recipe omits `assemble`; live record has `assemble: false`. AT isolation check passed.

Likely cause: `web/src/lib/cardState.ts` records explicit toggles, while `src/ppe_pipeline_api/pull_control.py` (`scope_slice`, `load_settings_file`, around lines 3057 and 3422) merges saved card keys onto live keys. An omitted default cannot overwrite a later toggle. **Fix:** save effective scoped defaults or replace/reset only the loaded scope's card keys before merging saved values. Preserve other-tab keys. Regression: default-open and explicit-open/closed round trips in both scopes, while proving other-tab settings unchanged.

### UAT-04 — P2: PostgreSQL Central Data Store presents obsolete DynamoDB sync (D2/D3)

**Reproduce:** Development DP → Central data store → Sync now; revisit after more than a minute.

**Expected:** PostgreSQL status accurately describes direct writes; any refresh completes visibly. **Actual:** `Not copied from DynamoDB yet.` remains; Sync answers `Copying in the background.`; later status reports `record_backend: postgres`, `last_sync: null`. Catalog itself correctly says PostgreSQL.

**Fix:** branch presentation and action by backend in `web/src/components/Track2CentralStorePanel.tsx` (hardcoded message around lines 167–170). For SQL, show direct recording and a Refresh action with completion/error state; reserve copying timestamps for a real DynamoDB projection. For actual background jobs poll until terminal status instead of a single delayed refresh. Regression: SQL mode, Dynamo projection success/failure, slow completion, unavailable database.

### UAT-05 — P2: deleted settings file remains the loaded schedule identity (H1)

**Reproduce:** Save AT settings as `uat_AT`; delete with confirmation; reload training and Scheduled Prototype.

**Expected:** No page claims a deleted file is loaded; live settings remain unchanged. **Actual:** file list excludes `uat_AT`, training toolbar says `Loaded: no file loaded`, but Scheduled says `uat_AT — loaded`. Final config still contains `settings_file_at: uat_AT` (baseline empty).

**Fix:** in `src/ppe_pipeline_api/pull_control.py:3753`, deleting a file should conditionally clear matching loaded-file references while preserving configuration and schedules. Scheduled should also tolerate a reference to a nonexistent file and label live unsaved settings honestly. Test deleting the loaded AT/DP file and a nonloaded file; ensure no schedule changes. This report leaves the stale reference visible as evidence rather than invoking a full settings reset to hide it.

## Confusing text and coverage limits

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

[Final verification](uat-evidence/2026-11-04/28-final-verification.json) records the exact comparison. Schedule, training and data fields were not included in the cleanup write. The stale AT reference was not cleared by resetting settings because that could affect schedules outside UAT authorization.

## Fix order and retest

1. Treat missing sources as a blocker for claiming full reproducibility; implement future source-retention/validation safeguards. Historical repair stays out of scope.
2. Fix stopped-database query gating and scoped card restoration.
3. Correct SQL backend status/sync behavior and deleted-file identity handling.
4. Update moved-control help text/documentation.
5. Repeat failed cases and extend the G1 persistence matrix on the next development build. Separately authorize any training/schedule execution or broader feature acceptance test.
