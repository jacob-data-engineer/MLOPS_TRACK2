# One-run training/comparison/export supervisor

Run `watch_training_export.py /state/job.json` inside the existing development
application image, with the same database/AWS environment and a persistent state
mount. It does not submit training. The normal background worker must continue
running preparation, training reconciliation and baseline comparison.

Required JSON fields: `run_id`, `bucket`, `prefix`. Use a unique S3 prefix for each
workflow. Optionally supply `preparation_key` (the preparation record's `kind` and
`sort`) to surface failures before a training audit exists. Never run two
supervisors for the same workflow with different state files.

The supervisor waits for both training `Succeeded` and comparison `Completed`.
It exports the exact resulting registered model, all saved images/labels and all
metadata as a ZIP, using the existing catalog exporter. It also saves the full
comparison audit as `comparison.json` next to the package directory. Export
progress remains in the normal SQL exports table. State and errors are written
atomically to the supplied JSON file. Missing files or a partial export are not
reported as success.

The browser and chat are not involved after submission. Transient SQL/AWS errors
retry in the live process. A restart before exporting resumes from the saved run;
a restart after the ZIP is ready recovers the export and writes the report. An
interruption during ZIP construction is marked failed and requires operator
review before retrying; the script does not claim crash-proof multipart recovery.
The filesystem lock prevents simultaneous supervisors using the same state file.

Development deployment on 24 November 2026:

- Container: `track2-dev-20epoch-export`, restart policy `on-failure:5`.
- Shared mount: `/settings/autonomous-20epoch-20261124/` in the dev app, worker
  and supervisor. Host: `/opt/track2-app/settings/track2-app-dev/`.
- State: `job.json`; executable: `watch_training_export.py`.
- Existing dev app/worker images remain unchanged; no active worker is restarted.
- Run: `run-42394c5538ff4c6ea2703637db07e173`, test-model YOLO26-S,
  starting checkpoint v10, 20 epochs, patience 0, baseline v9 (baseline registry v1).
- S3 prefix: `s3://track2-s3/helmet_yolo_pt_delete/v10-20epochs-20261124-42394c5538ff4c6ea2703637db07e173/`.

Inspect the state file and `docker logs track2-dev-20epoch-export` for progress.
Do not resubmit training when checking status. A completed/failed state exits
normally and is not restarted by the container's on-failure policy.
