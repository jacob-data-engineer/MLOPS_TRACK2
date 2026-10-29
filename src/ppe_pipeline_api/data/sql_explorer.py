"""The SQL tab: read-only queries against a Postgres database, answered as
a list of images (2026-10-28).

The databases are not this app's. They are private, each in its own VPC,
so the app cannot open a connection to them; it invokes a small query
Lambda placed inside each one's VPC (terraform/app/lambda_src/pg_query.py)
that logs in as `track2_readonly` and runs one statement in a read-only
transaction. This module only knows which function serves which database
and how to turn rows into images.

WHICH CELLS ARE IMAGES is decided from the values, not the column names:
the schema is somebody else's and can change. A cell is an image when it
is an `s3://` URI, an S3 https URL, or a path ending in an image
extension. The bucket and folder chosen on the tab then narrow that list.
"""

from __future__ import annotations

import json
import re
from urllib.parse import unquote, urlparse

import boto3

from ..storage import IMAGE_EXTENSIONS, REGION

#: The databases on offer. `function` is the query Lambda inside that
#: database's VPC; `instance` is the RDS instance, for its status.
DATABASES = {
    "violation-review": {
        "label": "violation-review-postgres (violation_review)",
        "instance": "violation-review-postgres",
        "function": "track2-pg-query-violation-review",
    },
    "jpui-preview": {
        "label": "jpui-preview-crop-authority (violation_review_preview)",
        "instance": "jpui-preview-crop-authority",
        "function": "track2-pg-query-jpui-preview",
    },
}

_lambda = None
_rds = None


def _client(kind: str):
    global _lambda, _rds
    if kind == "lambda":
        _lambda = _lambda or boto3.client("lambda", region_name=REGION)
        return _lambda
    _rds = _rds or boto3.client("rds", region_name=REGION)
    return _rds


def _database(key: str) -> dict:
    if key not in DATABASES:
        raise ValueError(f"Unknown database {key!r}.")
    return DATABASES[key]


def databases() -> list[dict]:
    """Every database on offer, with whether it is running right now. A
    stopped one is still listed: it says so instead of vanishing."""
    out = []
    for key, db in DATABASES.items():
        try:
            status = _client("rds").describe_db_instances(
                DBInstanceIdentifier=db["instance"])["DBInstances"][0]["DBInstanceStatus"]
        except Exception as exc:                                    # noqa: BLE001
            status = f"unknown ({type(exc).__name__})"
        out.append({"key": key, "label": db["label"], "status": status,
                    "available": status == "available"})
    return out


def _invoke(key: str, payload: dict) -> dict:
    db = _database(key)
    try:
        response = _client("lambda").invoke(
            FunctionName=db["function"], Payload=json.dumps(payload).encode())
    except Exception as exc:                                        # noqa: BLE001
        name = type(exc).__name__
        if "ResourceNotFound" in name or "ResourceNotFound" in str(exc):
            raise ValueError(f"The query function for {db['label']} is not set up yet.") from exc
        if "AccessDenied" in str(exc):
            raise ValueError("This app is not yet allowed to run the query function.") from exc
        raise ValueError(f"Could not reach the query function: {exc}") from exc
    body = json.loads(response["Payload"].read() or b"{}")
    if response.get("FunctionError"):
        message = body.get("errorMessage") or "the function failed"
        if "timed out" in message.lower():
            message = (f"No answer from {db['label']} -- it may be stopped or still "
                       "starting. Check its status in the list.")
        raise ValueError(message)
    if not body.get("ok"):
        raise ValueError(body.get("error") or "The query failed.")
    return body


def schema(key: str) -> list[dict]:
    """The tables and their columns, for the panel beside the editor."""
    return _invoke(key, {"action": "schema"})["tables"]


_S3_HTTPS = re.compile(
    r"^https?://(?:(?P<vhost>[^./]+)\.s3[.-][^/]*amazonaws\.com|s3[.-][^/]*amazonaws\.com/(?P<path>[^/]+))/(?P<key>.+)$")


def as_image(value) -> dict | None:
    """{bucket, key} when `value` names an image, else None. `bucket` is
    "" for a bare path, which carries no bucket of its own."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text.lower().split("?", 1)[0].endswith(IMAGE_EXTENSIONS):
        return None
    if text.startswith("s3://"):
        parsed = urlparse(text)
        return {"bucket": parsed.netloc, "key": parsed.path.lstrip("/")}
    match = _S3_HTTPS.match(text.split("?", 1)[0])
    if match:
        return {"bucket": match.group("vhost") or match.group("path"),
                "key": unquote(match.group("key"))}
    if "://" in text or " " in text:
        return None
    return {"bucket": "", "key": text.lstrip("/")}


def images_from(columns: list[str], rows: list[list], *, bucket: str = "",
                folder: str = "") -> list[dict]:
    """The distinct images named anywhere in the rows, in row order,
    narrowed to `bucket` and `folder` when those are chosen.

    A bare path has no bucket of its own, so a chosen bucket keeps it --
    it is read as a key in that bucket. A chosen folder keeps keys
    beneath it."""
    folder = folder.strip("/")
    seen, found = set(), []
    for index, row in enumerate(rows):
        for column, value in zip(columns, row):
            image = as_image(value)
            if not image:
                continue
            if bucket and image["bucket"] and image["bucket"] != bucket:
                continue
            if folder and not image["key"].startswith(folder + "/"):
                continue
            where = image["bucket"] or bucket
            uri = f"s3://{where}/{image['key']}" if where else image["key"]
            if uri in seen:
                continue
            seen.add(uri)
            found.append({"uri": uri, "bucket": where, "key": image["key"],
                          "column": column, "row": index})
    return found


def run(key: str, sql: str, *, bucket: str = "", folder: str = "") -> dict:
    out = _invoke(key, {"action": "query", "sql": sql})
    images = images_from(out["columns"], out["rows"], bucket=bucket, folder=folder)
    return {"columns": out["columns"], "rows": out["rows"],
            "truncated": out.get("truncated", False), "images": images}
