# DynamoDB to PostgreSQL migration

Prepared 2026-11-03 on branch `export`.

## Development application cutover

**Deployed and verified:** build `20261103-070655`, development port 8090,
background processing on, schema `track2_dev`. Production remains build
`20261026-124922` on port 8501. Final regression suite: **2,530 passed**.
Live HTTP verification: SSM `8d84157d-1759-4130-ae5e-872f541ed4ac`.
Live SQL read/write and resource-isolation proof with DynamoDB blocked:
SSM `b21f31d1-b03e-402e-ba54-73b08156b943`.
Actual PostgreSQL and S3 ZIP/worker integration verification:
SSM `49e9bd31-d90c-4ed3-9a39-7e8bd54bbdc5`.

The development application now uses `TRACK2_RECORD_BACKEND=postgres` and
`CATALOG_SCHEMA=track2_dev`. Production remains on its original backend.
The operational SQL tables are `records_training_audits`,
`records_training_files`, `records_dataset_definitions`,
`records_pull_control`, `records_training_lineage`,
`records_inference_lineage`, `records_evaluation_runs`,
`records_inference_queue`, and `records_review_queue`.

- `record_store.py` implements the application's key reads, indexed queries,
  paginated scans, batch writes, conditional inserts, updates, deletes and
  atomic counters using PostgreSQL transactions. It does not fall back to
  DynamoDB when SQL fails. Unknown mappings/expressions are rejected.
- Transactional advisory locks serialize claims, counter increments and
  queue advancement. SQL expiry is checked on reads and cleaned by the worker.
- `data/sql_projection.py` updates catalog runs, file rows and dataset
  membership in the same transaction as the operational record. The old
  five-minute DynamoDB sync is disabled in SQL mode. Catalog screens now
  state that their records are stored directly in PostgreSQL.
- `data/sql_cutover.py` initializes the separate development schema once,
  preserves public catalog run/version numbers and verifies imported counts.
  It refuses to overwrite an initialized schema on a rerun. Startup requires
  its completion marker. Existing public tables and DynamoDB are retained.
- Copied pending training/pull jobs, submission markers and claims are not
  replayed. Copied automatic schedules are off until explicitly enabled in
  development. Their original records remain in migration staging.
- Development scheduling uses `chain/sql_worker.py` in the app's background
  loop, including SQL-backed pulls, expiry and worker-message ingestion.
  Arrival detection polls the configured sources. Development schedule edits
  do not modify production EventBridge rules or Scheduler schedules.
- Legacy training-lineage/baseline and review producers have an opt-in SQL
  mode that writes immutable messages under `sql-outbox/<schema>/` in S3.
  The development worker imports them transactionally, records receipts,
  and never overwrites a later review edit. This avoids opening private RDS
  access to remote producers. Existing production producers remain unchanged.
- Development uses separate endpoint/config/component names ending in
  `-dev`; its SQL queues cannot determine whether production's endpoint is
  idle. No new SageMaker endpoint is created merely by deploying the app.
- `deploy-dev.ps1` selects SQL explicitly. `deploy.sh` defaults development
  on 8090 to SQL, restricts SQL deployment to that slot, and skips shared
  production Lambda updates. Production defaults remain DynamoDB/public.

Verification performed against actual PostgreSQL: concurrent claims (one
winner), atomic counters, expiry/reclaim, descending pagination, nested typed
values, reverse file lookup, immediate run/file/dataset projections, and
idempotent worker-message import. A real S3 ZIP contained the exact test image
and label bytes, `data.yaml`, and `manifest.csv`. The disposable SQL schema
and S3 test objects were removed. No paid training or inference job was
submitted by these tests.

Authenticated live development API checks returned 58 runs, 82 dataset
versions, 58 numbered training-history records and 9,744 images for Run 58.
A separate live-process check blocked every DynamoDB API call and still
read the records and completed a SQL write/read/delete round trip.

Development SQL scheduling runs while the development application is up.
Production's always-on Lambda tickers still serve production; they are not
silently repointed at the development schema. Deploy separate SQL workers
if development must process schedules while its app container is stopped.

## Verified execution status

The ten staging tables exist in the live catalog's `migration_stage`
schema. Final verification command: SSM
`6c8be40a-bfd8-4ae7-8d80-145b705c251f`.

- Verified exact payloads, hashes and counts: 66 training-audit records,
  82 dataset definitions, 14 training-lineage records, 2 inference-lineage
  records, 3 inference-queue records. Evaluation and review were empty and
  verified as such. Total: 167 records across seven source tables.
- Training files and pull control initially returned `AccessDeniedException`
  for `dynamodb:Scan`. After the approved temporary grant, both imported:
  **818,172 training-file records and 329 pull-control records**, with
  matching counts, hashes and exact payloads and zero mismatches.
  Together with the seven sources above, **818,668 records across all
  nine sources** are verified. Existing catalog tables were not cleared
  or altered.
- The user explicitly approved temporary read-only access on 2026-11-03.
  The actual grant was narrowed to the two blocked sources, using role
  policy `Track2SqlMigrationReadOnly20261103`. Import command
  `c7d145cf-1ed3-4882-8a42-c29be5c0e98e` succeeded. Its cleanup monitor
  removed the policy after verification. The first immediate attempt
  failed during IAM propagation; the retry completed successfully.
- The user selected development first. Production cutover is not
  authorized. Staging itself does not switch readers or execute queues;
  the separate development cutover above now supplies the live SQL backend.
- Staging-phase regression suite: 2,521 passed. The stricter payload check was
  then verified live and with all nine importer tests passing.

Earlier attempts remain in separate batch IDs for auditability; the
record totals are source records in the final verified batches, not the sum
of every staging row from repeated imports.

## Implemented: isolated, verified import

`src/ppe_pipeline_api/data/dynamo_migration.py` creates schema
`migration_stage`, nine source tables, and a `batches` audit table. Each
copy has a unique batch ID. No application reads are redirected by this
tool, no source data is modified, and no jobs are dispatched.

| DynamoDB source | SQL staging table | Records and code owners |
| --- | --- | --- |
| track2-yolo-training-audit | training_audits | Training audits, detector definitions, queue entries, queue state and submission marker; training/track2_shared.py |
| track2-yolo-training-files | training_files | Image and label rows, role, split, file name and S3 address; training/track2_shared.py |
| track2-yolo-dataset-snapshots | dataset_definitions | Snapshots and dataset versions; data/dataset_snapshots.py and data/dataset_versions.py |
| track2-pull-control | pull_control | Config, weeks, claims, counters, pull queue, progress, events and metrics export settings; pull_control.py and training/metrics_export.py |
| track2-dynamoDB | training_lineage | Training and baseline lineage; model_training/src/pipeline_definition.py and register_baseline_model.py |
| track2-yolo-inference-lineage | inference_lineage | Inference input/output/model relationships; training/track2_shared.py |
| track2-yolo-eval-runs | evaluation_runs | Evaluation job records; training/track2_shared.py |
| track2-yolo-inference-queue | inference_queue | Queued inference work; training/track2_shared.py |
| track2-yolo-review-queue | review_queue | Review items; terraform/platform/lambda_src/trigger_review.py and application readers |

The CLI honors existing table-name environment overrides. The unrelated
Track1 table is excluded.

Each table contains `batch_id`, the complete composite `source_key`,
`record_type`, the complete typed DynamoDB `payload` as JSONB, and a SHA-256
digest. DynamoDB numeric strings are not converted to floats. Binary is
base64 encoded; sets are canonicalized. All original attributes remain
available, including attributes not understood by today's application.
These are **staging tables**. The development operational tables are separate
and are described above.

Existing catalog tables (`training_runs`, `run_files`, `dataset_versions`,
`dataset_members`, `exports`, etc.) remain unchanged. Historical repair
commands are not run as part of this migration.

### Commands

Run in an environment with private catalog access and AWS read access to
the nine sources. The application container has the PostgreSQL driver.

```powershell
.venv/Scripts/python.exe -m ppe_pipeline_api.data.dynamo_migration
.venv/Scripts/python.exe -m ppe_pipeline_api.data.dynamo_migration --apply --schema-only
.venv/Scripts/python.exe -m ppe_pipeline_api.data.dynamo_migration --apply
# Retry only a failed/changed source; this creates a separate new batch:
.venv/Scripts/python.exe -m ppe_pipeline_api.data.dynamo_migration --apply --source training_files
```

Without `--apply`, the command only prints the plan. Imports use paginated,
strongly consistent reads and bounded SQL inserts, then a fresh source
scan comparing keys, counts and content hashes. A difference gives state
`changed` and nonzero exit status. Failure preserves completed batches and
marks the current batch failed. Logs report counts, not record contents.

`verified` means the imported records matched a subsequent scan; a
DynamoDB scan is **not an atomic table snapshot**. Writes after that scan
remain possible. This is not evidence that live cutover is safe yet.

## Production promotion checklist (not performed)

1. Confirm development-only versus production scope. Both containers and
   shared Lambdas currently coordinate through the same queues/claims.
   Separate dev/prod job namespaces if only development is switched.
2. Implement SQL repositories for every read/write path above. Retain
   existing catalog run/version numbers. Map detector and queue records
   separately from training audits; they share one source table today.
3. Implement conditional claims atomically (unique keys and transactions),
   atomic queue counters and queue acquisition, and explicit TTL cleanup.
   A select followed by insert is not sufficient to prevent duplicate jobs.
4. Package PostgreSQL dependencies and the RDS CA bundle for the standalone
   pull Lambda, queue Lambda, review Lambda, training builds and baseline
   registration. Give each writer private network access and scoped secret
   access. The current database security group admits the EC2 host only.
5. Test SQL repositories against actual PostgreSQL: concurrent claims,
   FIFO ordering, restart recovery, deleted records, expiry, file reverse
   lookup, and export completeness, in addition to ordinary unit tests.
6. Pause or isolate writers for the final import; verify every source,
   reconcile writes/deletes since staging, then switch all affected
   readers/writers together. Do not run two independent claim stores for
   the same training jobs.
7. Disable the old catalog sync only after direct SQL writes cover its
   responsibilities. Verify training, pulls, evaluation, inference, review
   and ZIP/folder exports end to end.
8. Retain DynamoDB during the rollback window. Reverting to it after new
   SQL writes requires replaying those writes; an environment toggle alone
   is not a lossless rollback. Do not delete DynamoDB tables in this phase.

The standalone staging importer never switches application backends. The
development cutover is separate from production promotion. Images, labels,
models and ZIPs remain in S3 throughout. A rollback after development SQL
writes requires exporting/replaying those writes, not just changing an
environment flag back to DynamoDB.
