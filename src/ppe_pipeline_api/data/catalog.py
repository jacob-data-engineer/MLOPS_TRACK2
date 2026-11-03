"""The central data catalog (2026-10-29).

A Postgres owned by this platform (terraform/app/catalog.tf) recording
everything about the files in the central data store:

  files            one row per image, by content hash -- stored once,
                   however many places it arrived from
  file_sources     every place a file was seen (a pull, the golden set,
                   a SQL pull, an import, the backfill)
  labels           every version of a file's labels; an edit is a new
                   version, never an overwrite, so an old dataset can
                   always be rebuilt
  dataset_versions one per built dataset, with the settings that built it
  dataset_members  every file a version considered: its split FOR THAT
                   VERSION (re-drawn per run), whether it was included,
                   why not (decayed, source unticked, ...), its decay
                   weight and the label version used
  training_runs    which version each run trained on, and what it made
  exports          every zip made of a version

NOTHING IS DELETED. A file left out of a version is a member row with
`included = false` and a reason; it stays in the store and the next
version may take it again. Training, recreating an old run and exporting
a zip all read a version's members from here.

The app connects directly -- the database is private, in the review
host's VPC, open to the host's security group only. The password is the
RDS-managed master secret, read from Secrets Manager on connect (RDS
rotates it, so it is never cached across a failed login).
"""

from __future__ import annotations

import json
import os
import pathlib

import boto3

from ..storage import REGION

HOST = os.environ.get("CATALOG_HOST", "")
SECRET_ARN = os.environ.get("CATALOG_SECRET_ARN", "")
DATABASE = "track2_catalog"
INSTANCE = "track2-catalog"
#: Amazon's RDS root certificates, for verifying the connection.
CA_BUNDLE = pathlib.Path(__file__).with_name("rds-ca-bundle.pem")

#: The schema's version: SCHEMA is version 1, and each entry in MIGRATIONS
#: takes it one step further. Recorded in `schema_version`.
SCHEMA_VERSION = 7

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS schema_version (
         version    integer PRIMARY KEY,
         applied_at timestamptz NOT NULL DEFAULT now())""",
    """CREATE TABLE IF NOT EXISTS files (
         sha256       char(64) PRIMARY KEY,
         store_key    text NOT NULL,          -- key in the central store
         store_bucket text NOT NULL,
         bytes        bigint,
         width        integer,
         height       integer,
         content_type text,
         hazard       text,                   -- the class it was judged for, if any
         captured_at  timestamptz,            -- when the frame was taken, if known
         first_seen   timestamptz NOT NULL DEFAULT now())""",
    """CREATE TABLE IF NOT EXISTS file_sources (
         id          bigserial PRIMARY KEY,
         sha256      char(64) NOT NULL REFERENCES files(sha256),
         source_uri  text NOT NULL,           -- where it was found
         source_kind text NOT NULL,           -- pull | golden | sql | import | backfill
         seen_at     timestamptz NOT NULL DEFAULT now(),
         detail      jsonb NOT NULL DEFAULT '{}'::jsonb,
         UNIQUE (sha256, source_uri))""",
    """CREATE TABLE IF NOT EXISTS labels (
         id          bigserial PRIMARY KEY,
         sha256      char(64) NOT NULL REFERENCES files(sha256),
         version     integer NOT NULL,        -- 1, 2, 3 ... per file
         class_names jsonb NOT NULL,          -- {"0": "helmet", ...}: what the numbers mean
         yolo        text NOT NULL,           -- the label file, verbatim
         origin      text NOT NULL,           -- database | human | detector | import
         source_uri  text,
         created_at  timestamptz NOT NULL DEFAULT now(),
         UNIQUE (sha256, version))""",
    """CREATE TABLE IF NOT EXISTS dataset_versions (
         id          text PRIMARY KEY,
         created_at  timestamptz NOT NULL DEFAULT now(),
         created_by  text,
         purpose     text NOT NULL DEFAULT 'training',   -- training | export | recreate
         class_names jsonb,
         settings    jsonb NOT NULL DEFAULT '{}'::jsonb, -- sources ticked, retention curve,
                                                         -- split ratio and seed
         parent_id   text REFERENCES dataset_versions(id),
         note        text)""",
    """CREATE TABLE IF NOT EXISTS dataset_members (
         version_id   text NOT NULL REFERENCES dataset_versions(id),
         sha256       char(64) NOT NULL REFERENCES files(sha256),
         label_id     bigint REFERENCES labels(id),
         split        text CHECK (split IN ('train', 'val', 'test')),
         included     boolean NOT NULL,
         reason       text,                   -- why left out: decayed, unticked, ...
         decay_weight real,                   -- the retention weight it was given
         week         text,
         group_key    text,                   -- frames that must share a split
         PRIMARY KEY (version_id, sha256))""",
    """CREATE INDEX IF NOT EXISTS dataset_members_file ON dataset_members (sha256)""",
    """CREATE TABLE IF NOT EXISTS training_runs (
         run_id             text PRIMARY KEY,
         version_id         text NOT NULL REFERENCES dataset_versions(id),
         started_at         timestamptz NOT NULL DEFAULT now(),
         detector           text,
         execution_arn      text,
         model_package_arn  text,
         detail             jsonb NOT NULL DEFAULT '{}'::jsonb)""",
    """CREATE TABLE IF NOT EXISTS exports (
         id         text PRIMARY KEY,
         version_id text NOT NULL REFERENCES dataset_versions(id),
         created_at timestamptz NOT NULL DEFAULT now(),
         zip_uri    text,
         bytes      bigint,
         files      integer)""",
]

#: Version 2 (2026-10-29): the RECORDS move here from DynamoDB -- every
#: snapshot and dataset version with its files, every training run with
#: a run number and the files it used. Applied once, in order, on top of
#: version 1. Additive only: columns are added, constraints relaxed, and
#: no table or row is removed.
MIGRATIONS = {
    2: [
        # Snapshots and dataset versions, numbered 1, 2, 3 ... in order.
        "ALTER TABLE dataset_versions ADD COLUMN IF NOT EXISTS kind text NOT NULL DEFAULT 'version'",
        "ALTER TABLE dataset_versions ADD COLUMN IF NOT EXISTS version_number bigserial",
        "CREATE UNIQUE INDEX IF NOT EXISTS dataset_versions_number ON dataset_versions (version_number)",
        "ALTER TABLE dataset_versions ADD COLUMN IF NOT EXISTS state text",
        "ALTER TABLE dataset_versions ADD COLUMN IF NOT EXISTS run_id text",
        "ALTER TABLE dataset_versions ADD COLUMN IF NOT EXISTS train_images integer",
        "ALTER TABLE dataset_versions ADD COLUMN IF NOT EXISTS val_images integer",
        # The DynamoDB row as it was, so nothing it held is lost in the move.
        "ALTER TABLE dataset_versions ADD COLUMN IF NOT EXISTS record jsonb NOT NULL DEFAULT '{}'::jsonb",
        # Members by ADDRESS. The content hash is filled in once the file
        # is in the central store; until then the address is the identity.
        "ALTER TABLE dataset_members ADD COLUMN IF NOT EXISTS image_uri text",
        "ALTER TABLE dataset_members ADD COLUMN IF NOT EXISTS label_uri text",
        "ALTER TABLE dataset_members DROP CONSTRAINT IF EXISTS dataset_members_pkey",
        "ALTER TABLE dataset_members ALTER COLUMN sha256 DROP NOT NULL",
        "CREATE UNIQUE INDEX IF NOT EXISTS dataset_members_uri ON dataset_members (version_id, image_uri)",
        # Training runs, numbered 1, 2, 3 ... A run from before snapshots
        # existed has no version, so the link may be empty.
        "ALTER TABLE training_runs ADD COLUMN IF NOT EXISTS run_number bigserial",
        "CREATE UNIQUE INDEX IF NOT EXISTS training_runs_number ON training_runs (run_number)",
        "ALTER TABLE training_runs ALTER COLUMN version_id DROP NOT NULL",
        "ALTER TABLE training_runs DROP CONSTRAINT IF EXISTS training_runs_version_id_fkey",
        "ALTER TABLE training_runs ADD COLUMN IF NOT EXISTS status text",
        "ALTER TABLE training_runs ADD COLUMN IF NOT EXISTS submitted_at timestamptz",
        "ALTER TABLE training_runs ADD COLUMN IF NOT EXISTS ended_at timestamptz",
        "ALTER TABLE training_runs ADD COLUMN IF NOT EXISTS architecture text",
        "ALTER TABLE training_runs ADD COLUMN IF NOT EXISTS files_synced boolean NOT NULL DEFAULT false",
        "ALTER TABLE training_runs ADD COLUMN IF NOT EXISTS record jsonb NOT NULL DEFAULT '{}'::jsonb",
        # Every file a run used, as the training audit recorded it.
        """CREATE TABLE IF NOT EXISTS run_files (
             run_id    text NOT NULL REFERENCES training_runs(run_id),
             file_key  text NOT NULL,            -- role#kind#index, as recorded
             s3_uri    text NOT NULL,
             role      text,                     -- train | eval
             split     text,
             kind      text,                     -- image | label
             folder    text,
             PRIMARY KEY (run_id, file_key))""",
        "CREATE INDEX IF NOT EXISTS run_files_uri ON run_files (s3_uri)",
        # Each sync from DynamoDB, so "is the catalog up to date" has an answer.
        """CREATE TABLE IF NOT EXISTS sync_runs (
             id          bigserial PRIMARY KEY,
             started_at  timestamptz NOT NULL DEFAULT now(),
             finished_at timestamptz,
             ok          boolean,
             counts      jsonb NOT NULL DEFAULT '{}'::jsonb,
             error       text)""",
    ],
    # Version 3 (2026-10-29): an export is of a dataset version OR a run
    # (a run from before snapshots has no version), built in the
    # background, so it has a state.
    3: [
        "ALTER TABLE exports ALTER COLUMN version_id DROP NOT NULL",
        "ALTER TABLE exports DROP CONSTRAINT IF EXISTS exports_version_id_fkey",
        "ALTER TABLE exports ADD COLUMN IF NOT EXISTS run_id text",
        "ALTER TABLE exports ADD COLUMN IF NOT EXISTS label text",
        "ALTER TABLE exports ADD COLUMN IF NOT EXISTS state text NOT NULL DEFAULT 'building'",
        "ALTER TABLE exports ADD COLUMN IF NOT EXISTS finished_at timestamptz",
        "ALTER TABLE exports ADD COLUMN IF NOT EXISTS missing integer NOT NULL DEFAULT 0",
        "ALTER TABLE exports ADD COLUMN IF NOT EXISTS error text",
    ],
    # Version 4 (2026-11-02): a version's manifest is read once, and says
    # so -- an empty or label-only one was otherwise fetched every sync.
    4: [
        "ALTER TABLE dataset_versions ADD COLUMN IF NOT EXISTS members_synced boolean NOT NULL DEFAULT false",
        # Versions' members were copied correctly. Snapshots are left to be
        # read once more: composed ones were copied with the root wrongly
        # put in front of every address, and the sync repairs them.
        "UPDATE dataset_versions SET members_synced = true WHERE kind = 'version' AND EXISTS"
        " (SELECT 1 FROM dataset_members m WHERE m.version_id = dataset_versions.id)",
    ],
    5: [
        "ALTER TABLE exports ADD COLUMN IF NOT EXISTS destination text NOT NULL DEFAULT 'local'",
        "ALTER TABLE exports ADD COLUMN IF NOT EXISTS destination_uri text",
    ],
    7: [
        "CREATE INDEX IF NOT EXISTS run_files_image_count ON run_files (run_id) WHERE kind = 'image'",
    ],
    6: [
        "ALTER TABLE exports ADD COLUMN IF NOT EXISTS format text NOT NULL DEFAULT 'zip'",
        "UPDATE exports SET format = 'folder' WHERE destination = 's3' AND zip_uri IS NULL",
    ],
}

TABLES = ("files", "file_sources", "labels", "dataset_versions",
          "dataset_members", "training_runs", "run_files", "exports")


def configured() -> bool:
    return bool(HOST and SECRET_ARN)


def _password() -> tuple[str, str]:
    secret = boto3.client("secretsmanager", region_name=REGION).get_secret_value(
        SecretId=SECRET_ARN)["SecretString"]
    body = json.loads(secret)
    return body["username"], body["password"]


def connect():
    """A new connection. The caller closes it."""
    if not configured():
        raise ValueError("The catalog is not set up on this container "
                         "(CATALOG_HOST / CATALOG_SECRET_ARN are empty).")
    import ssl

    import pg8000.dbapi

    user, password = _password()
    # VERIFIED TLS. RDS signs with Amazon's own roots, which no system
    # trust store carries, so the published bundle ships beside this file
    # (https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem).
    # The read-only query Lambdas skip the check; this is the database the
    # app WRITES to, so it does not.
    context = ssl.create_default_context(cafile=str(CA_BUNDLE))
    return pg8000.dbapi.connect(host=HOST, port=5432, database=DATABASE, user=user,
                                password=password, ssl_context=context, timeout=15)


def ensure_schema(report: bool = True) -> dict:
    """Create any table that is missing. Safe to run any number of times:
    every statement is IF NOT EXISTS, and nothing is ever dropped."""
    conn = connect()
    try:
        cur = conn.cursor()
        for statement in SCHEMA:
            cur.execute(statement)
        cur.execute("INSERT INTO schema_version (version) VALUES (1) "
                    "ON CONFLICT (version) DO NOTHING")
        conn.commit()
        cur.execute("SELECT max(version) FROM schema_version")
        current = int(cur.fetchone()[0] or 1)
        for version in sorted(MIGRATIONS):
            if version <= current:
                continue
            # One transaction per step: a step half-applied would leave a
            # schema matching no version.
            for statement in MIGRATIONS[version]:
                cur.execute(statement)
            cur.execute("INSERT INTO schema_version (version) VALUES (%s)", (version,))
            conn.commit()
    finally:
        conn.close()
    # The counts are for a person asking; the sync does not need them.
    return status() if report else {}


def status() -> dict:
    """Is the catalog reachable, and how much is in each table."""
    if not configured():
        return {"configured": False, "reachable": False, "tables": {}}
    try:
        conn = connect()
    except Exception as exc:                                       # noqa: BLE001
        return {"configured": True, "reachable": False,
                "error": f"{type(exc).__name__}: {exc}", "tables": {}}
    try:
        cur = conn.cursor()
        cur.execute("SELECT max(version) FROM schema_version")
        version = cur.fetchone()[0]
        counts = {}
        for table in TABLES:
            if table == "run_files" and int(version or 0) < 2:
                continue
            cur.execute(f"SELECT count(*) FROM {table}")       # noqa: S608 -- fixed names
            counts[table] = int(cur.fetchone()[0])
        last = None
        if int(version or 0) >= 2:
            cur.execute("SELECT started_at, finished_at, ok, counts, error FROM sync_runs "
                        "ORDER BY id DESC LIMIT 1")
            row = cur.fetchone()
            if row:
                last = {"started_at": row[0].isoformat() if row[0] else None,
                        "finished_at": row[1].isoformat() if row[1] else None,
                        "ok": row[2], "counts": row[3], "error": row[4]}
        return {"configured": True, "reachable": True, "schema_version": version,
                "tables": counts, "last_sync": last}
    except Exception as exc:                                       # noqa: BLE001
        return {"configured": True, "reachable": True,
                "error": f"{type(exc).__name__}: {exc}", "tables": {}}
    finally:
        conn.close()
