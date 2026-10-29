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

import datetime as _dt
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import unquote, urlparse

import boto3

from .. import storage
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
            # A refusal is not an unknown state: it means the SQL tab's
            # one-time setup (scripts/setup_sql_tab.sh) has not granted
            # this app `rds:DescribeDBInstances` yet. Say that.
            if "AccessDenied" in str(exc) or "not authorized" in str(exc):
                status = "not set up (scripts/setup_sql_tab.sh has not been run)"
            else:
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


# --- pulling the images a query names into a training folder (2026-10-29) ---

#: Most images one pull copies. Each is a server-side S3 copy, done in
#: parallel inside the request; past this a pull would outlast it.
PULL_LIMIT = 5000
#: Copies in flight at once.
PULL_WORKERS = 16
#: Never written into: the code and weights the pipelines run from.
PROTECTED = ("custom", "codebuild-source", "track2-app-source", "track2-yolo-weights")


def label_key_for(image_key: str) -> str:
    """Where Ultralytics would look for this image's label: `/images/`
    swapped for `/labels/`, extension for `.txt`. "" when the image is
    not under an `images/` folder, so has no label by that rule."""
    if "/images/" not in "/" + image_key:
        return ""
    swapped = ("/" + image_key).replace("/images/", "/labels/", 1)[1:]
    return swapped.rsplit(".", 1)[0] + ".txt"


def split_for(uri: str, train_percent: int) -> str:
    """"train" or "val", fixed per image: the same image lands on the same
    side every time a pull is repeated, so re-pulling cannot move an
    image from val into train."""
    if train_percent >= 100:
        return "train"
    if train_percent <= 0:
        return "val"
    bucket_of = int(hashlib.sha1(uri.encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket_of < train_percent else "val"


def _exists(bucket: str, key: str) -> bool:
    try:
        storage.s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:                                               # noqa: BLE001
        return False


def pull(key: str, sql: str, *, bucket: str = "", folder: str = "",
         dest_bucket: str, dest_folder: str, train_percent: int = 100) -> dict:
    """Run the query, then copy every image it names into
    `dest_bucket/dest_folder` in the layout a training run reads:

        <dest>/images/train/<name>   <dest>/labels/train/<stem>.txt
        <dest>/images/val/<name>     <dest>/labels/val/<stem>.txt

    A label is copied when one exists where Ultralytics would look for it
    beside the source image. Images without one are still copied -- the
    summary says how many -- because a frame can be labelled later, and
    refusing it would hide that it was found.

    Nothing is overwritten or deleted: a name already at the destination
    is skipped, so a pull can be repeated after the query finds more.
    `sources.json` maps each copied file back to where it came from, and
    `query.json` records the query that chose them -- the only record of
    why these images are in this folder.
    """
    dest_bucket = (dest_bucket or "").strip()
    dest_folder = (dest_folder or "").strip().strip("/")
    if not dest_bucket or not dest_folder:
        raise ValueError("Choose a bucket and a folder to pull the images into.")
    if any(dest_folder == p or dest_folder.startswith(p + "/") for p in PROTECTED):
        raise ValueError(f"{dest_folder} holds code or weights the pipelines run from; "
                         "pull into another folder.")
    train_percent = max(0, min(100, int(train_percent)))

    result = run(key, sql, bucket=bucket, folder=folder)
    images = [i for i in result["images"] if i.get("bucket")]
    bare = len(result["images"]) - len(images)
    if not images:
        raise ValueError("The query names no images with a bucket to copy from"
                         + (f" ({bare} bare paths: choose the bucket they are in)" if bare else "")
                         + ".")
    if len(images) > PULL_LIMIT:
        raise ValueError(f"The query names {len(images):,} images; one pull copies at most "
                         f"{PULL_LIMIT:,}. Add a LIMIT, or pull in parts.")

    # One file name per image. Two sources sharing a name would otherwise
    # overwrite each other's label, so the second gets a short tag.
    plan, taken = [], set()
    for image in images:
        name = image["key"].rsplit("/", 1)[-1]
        if name in taken:
            stem, _, ext = name.rpartition(".")
            name = f"{stem}_{hashlib.sha1(image['uri'].encode()).hexdigest()[:8]}.{ext}"
        taken.add(name)
        side = split_for(image["uri"], train_percent)
        plan.append((image, side, name))

    def copy(entry):
        image, side, name = entry
        target = f"{dest_folder}/images/{side}/{name}"
        out = {"image": image["uri"], "target": target, "side": side,
               "copied": False, "label": "", "error": ""}
        try:
            if _exists(dest_bucket, target):
                out["skipped"] = True
            else:
                storage.s3.copy_object(
                    Bucket=dest_bucket, Key=target,
                    CopySource={"Bucket": image["bucket"], "Key": image["key"]})
                out["copied"] = True
            source_label = label_key_for(image["key"])
            if source_label and _exists(image["bucket"], source_label):
                label_target = f"{dest_folder}/labels/{side}/{name.rsplit('.', 1)[0]}.txt"
                if not _exists(dest_bucket, label_target):
                    storage.s3.copy_object(
                        Bucket=dest_bucket, Key=label_target,
                        CopySource={"Bucket": image["bucket"], "Key": source_label})
                out["label"] = f"s3://{image['bucket']}/{source_label}"
        except Exception as exc:                                    # noqa: BLE001
            out["error"] = str(exc)
        return out

    with ThreadPoolExecutor(max_workers=PULL_WORKERS) as pool:
        done = list(pool.map(copy, plan))

    ok = [d for d in done if not d["error"]]
    sources_key = f"{dest_folder}/sources.json"
    try:
        sources = json.loads(storage.s3.get_object(
            Bucket=dest_bucket, Key=sources_key)["Body"].read())
    except Exception:                                               # noqa: BLE001
        sources = {}
    for d in ok:
        sources[d["target"]] = d["image"]
        if d["label"]:
            sources[d["target"].replace("/images/", "/labels/", 1).rsplit(".", 1)[0]
                    + ".txt"] = d["label"]
    summary = {
        "bucket": dest_bucket, "folder": dest_folder,
        "found": len(images), "copied": sum(d["copied"] for d in ok),
        "skipped": sum(bool(d.get("skipped")) for d in ok),
        "with_label": sum(bool(d["label"]) for d in ok),
        "train": sum(d["side"] == "train" for d in ok),
        "val": sum(d["side"] == "val" for d in ok),
        "bare_paths_left_out": bare,
        "failed": len(done) - len(ok),
        "errors": [f"{d['image']}: {d['error']}" for d in done if d["error"]][:10],
    }
    storage.s3.put_object(Bucket=dest_bucket, Key=sources_key,
                          Body=json.dumps(sources, indent=1).encode(),
                          ContentType="application/json")
    storage.s3.put_object(
        Bucket=dest_bucket, Key=f"{dest_folder}/query.json",
        Body=json.dumps({"database": key, "sql": sql, "bucket": bucket, "folder": folder,
                         "train_percent": train_percent,
                         "at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                         "summary": summary}, indent=1).encode(),
        ContentType="application/json")
    return summary
