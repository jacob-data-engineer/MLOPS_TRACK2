# For Codex GPT — paused work handoff

Recorded: 4 November 2026 (Asia/Singapore).
Workspace: `C:\Projects\MLOPS_TRACK2`.

## Pause state

The user explicitly requested to pause here and record what was done. **Do not start training or resume testing until the user resumes the task.** No end-to-end training job has been started or queued by this verification work.

The next intended task is an end-to-end training test to close the remaining live acceptance gap for exporting a real run that uses preserved dataset files.

## Current repository state

- Branch: `export`.
- HEAD at handoff: `32e84c3` — `UAT report: record the R1-R4 fixes and their live retests`.
- `uat testing.md` has an uncommitted update containing the latest independent verification.
- This handoff file is newly created and uncommitted.
- No application code was changed during the independent verification, and that verification report update has not been committed or pushed.
- Earlier checkpoint `0161641` (`checkpoint SQL migration and UAT findings`) was committed and pushed to `origin/export` during this conversation. Later fixes were applied separately and are now in the local history.
- Test evidence is under `uat-evidence/2026-11-04/`, excluded from Git/deployment. Do not indiscriminately stage or package downloaded images, ZIPs, or test evidence.

## What was completed

1. Executed the original development UAT and documented reproduction steps, findings, priorities and evidence in [uat testing.md](uat%20testing.md).
2. Independently retested the initial fixes on development build `20261104-111726`. Confirmed UI fixes but found four additional issues:
   - R1: changed originals overrode preserved bytes during export.
   - R2: run exports did not resolve preserved dataset copies.
   - R3: preservation counts omitted missing labels and needed failed-copy retry handling.
   - R4: a deleted DP settings filename reappeared in configuration.
3. Read and independently verified the subsequent R1–R4 fixes on development build **`20261104-120224`**, port **8090**, PostgreSQL backend. Last verified during approximately **20:25–20:33 SGT on 4 November 2026**. Recheck build and state when resuming; these observations are not a claim about a future deployment.
4. Updated the top of `uat testing.md` with the latest independent results; historical failures remain below for traceability.

## Latest verified results

- **R1 passed within tested scope:** the original isolated overwritten-source reproduction now exports the frozen bytes. Existing Version 84 ZIP was downloaded through the browser; its three images and three labels exactly match preserved S3 copies and manifest hashes.
- **R2 passed isolated behavioral testing:** actual run planner and ZIP builder use preserved image/label copies when originals are absent. **A real training-run export from a preserved dataset has not yet been tested end to end.**
- **R3 passed:** missing labels are counted and preservation is incomplete when appropriate. Fresh Version 85 export correctly reports three included files and one missing label. Failed-copy/retry tests passed.
- **R4 passed the live development race checks:** three rounds, each deleting a temporary DP settings file while 18 identical card-state saves ran through six workers. All **54 saves** succeeded. Deleted names remained cleared, card state was unchanged, and Scheduled displayed no deleted names.
- **165 focused automated tests passed**, zero failed. One dependency deprecation warning in the first command:

```text
.venv/Scripts/python.exe -m pytest tests/test_central_store.py tests/test_settings_split_by_tab.py tests/test_catalog.py tests/test_sql_explorer.py -q
# 130 passed

.venv/Scripts/python.exe -m pytest tests/test_settings_roundtrip.py tests/test_sql_record_store.py -q
# 35 passed
```

This was targeted verification, not an independent rerun of the implementer's reported full 2,572-test suite or all original UAT cases.

## Evidence and cleanup

Latest evidence directory: `uat-evidence/2026-11-04/independent-retest-2/`.

Useful files:

- `build.json`, `baseline.json` — tested deployment and starting settings.
- `preservation-edge-cases.json` — independent R1/R3 reproductions.
- `03-run-export-reproduction.json` — isolated R2 plan/output.
- `02-delete-races.json` — three live R4 races.
- `04-preserved-84.json`, `04-preserved-85.json` — existing preservation records.
- `05-exports.json`, `05-exports.png` — live export outcome.
- `06-zip-byte-verification.json` — downloaded ZIP bytes/hashes versus preserved S3 copies.
- `07-final-cleanup.json` — final configuration matched the captured baseline exactly; no temporary test files remained.

Temporary files `uat_verify2_0_DP`, `uat_verify2_1_DP`, and `uat_verify2_2_DP` were deleted. Export preferences were restored. Temporary browser authentication state was removed. No training, schedule Apply, database start, new dataset freeze, production access, or source-file deletion/overwrite occurred in the latest verification.

Retained new export: Version 85, ID `46db64c87382`:
`s3://track2-s3/label-curation/validate_v1/export/_exports/version-85-46db64c87382/version-85.zip`.
It is intentionally incomplete because the test dataset lacks one label. Do not interpret that result as a failed missing-label reporting fix.

## Training test requested but not started

The user asked: “can u start a training test end to end”. Starting a training test was explicitly requested, but then the user paused before answering the proposed scope questions:

1. Smallest preserved dataset for a quick test, or current full training configuration?
2. Allow model-version registration, or stop after training and export verification?

The assistant recommended a small preserved dataset, **one epoch**, model registration allowed, and no production deployment. **These recommendations were not selected or approved by the user before the pause.** Preserve that distinction when resuming.

Before choosing a dataset, inspect its integrity and label coverage. **Version 85 is the smallest but intentionally has a missing label.** Version 84 has three complete image/label pairs; Version 83 has 72. Those facts do not by themselves establish suitability for training. Validate model/class compatibility, training and validation splits, preserved-file resolution, and current pipeline behavior first.

After the user resumes and scope is settled, the intended acceptance flow is to run the small development training test, monitor it to completion, verify its recorded run/dataset lineage and model artifact, export that actual run, and validate exported images/labels/metadata against the preserved dataset. Record actual results and any failures in `uat testing.md`. Do not manufacture a successful run record or claim isolated tests prove live end-to-end behavior.

## Continuing constraints and limitations

- Development URL: `http://18.140.196.76:8090`; production port 8501 was excluded and remains outside this task.
- Do not apply pull/training schedules or deploy inference merely to run a one-off test.
- User previously excluded historical repairs. Old missing source files, including Version 46, were not repaired.
- Older production/Lambda writers were not independently inspected or promoted; concurrency acceptance is for the tested development writers.
- Large exports, actual training, scheduled execution, model/PT compatibility, and stop/cancel were not covered by the latest targeted retest.
- Browser UAT used isolated headless Chrome via Playwright because native browser-control tooling was unavailable. Temporary runner previously existed at `%TEMP%\track2-uat-run.py`; verify availability rather than assuming it persists. Authentication state was deliberately removed; do not store passwords or session tokens in this handoff.
- User requested a completion beep; it was sounded after the completed verification and should be sounded after this handoff is saved.
