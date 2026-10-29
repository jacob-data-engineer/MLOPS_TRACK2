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


def set_running(key: str, run: bool) -> dict:
    """Start or stop one of the databases on offer, for the tab's buttons.

    Only the instances in DATABASES -- the name comes from that table,
    never from the request. Asking for the state it is already in, or
    already heading to, is answered rather than refused."""
    db = _database(key)
    rds = _client("rds")
    status = rds.describe_db_instances(
        DBInstanceIdentifier=db["instance"])["DBInstances"][0]["DBInstanceStatus"]
    if run and status in ("available", "starting"):
        return {"key": key, "status": status, "changed": False}
    if not run and status in ("stopped", "stopping"):
        return {"key": key, "status": status, "changed": False}
    if run and status != "stopped":
        raise ValueError(f"{db['label']} is {status}; it can only be started from stopped.")
    if not run and status != "available":
        raise ValueError(f"{db['label']} is {status}; it can only be stopped once available.")
    try:
        if run:
            rds.start_db_instance(DBInstanceIdentifier=db["instance"])
        else:
            rds.stop_db_instance(DBInstanceIdentifier=db["instance"])
    except Exception as exc:                                        # noqa: BLE001
        if "AccessDenied" in str(exc) or "not authorized" in str(exc):
            raise ValueError("This app is not allowed to start or stop the databases yet "
                             "(rds:StartDBInstance / rds:StopDBInstance).") from exc
        raise ValueError(f"Could not {'start' if run else 'stop'} it: {exc}") from exc
    return {"key": key, "status": "starting" if run else "stopping", "changed": True}


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


#: A column with this name holds images by declaration: any `s3://` value
#: in it counts, extension or not. The cargo evidence frames are JPEGs
#: stored as `.../<id>/source`, which no extension test would find.
IMAGE_COLUMN = "image"


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
            if (not image and column == IMAGE_COLUMN and isinstance(value, str)
                    and value.startswith("s3://")):
                parsed = urlparse(value.strip())
                image = {"bucket": parsed.netloc, "key": parsed.path.lstrip("/")}
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
    # A query that returns the boxes (the human-verified preset does) gets
    # its labels worked out once, here, for both the overlay and the pull.
    if all(c in out["columns"] for c in ("people", "width", "height")):
        for image in images:
            image["labels"] = frame_labels(dict(zip(out["columns"], out["rows"][image["row"]])))
    return {"columns": out["columns"], "rows": out["rows"],
            "truncated": out.get("truncated", False), "images": images}


# --- labels, from the boxes the review database holds (2026-10-29) ----------

#: The class numbering labels are written in: the same as the curated sets
#: under `label-curation/pipeline_v1`, so a pull can sit beside them in one
#: dataset without renumbering. Only helmet (0) and person (2) are drawn.
LABEL_NAMES = {0: "helmet", 1: "gloves", 2: "person", 3: "working_boots",
               4: "man_cage", 5: "safe_area"}
HELMET, PERSON = 0, 2


def _box(value) -> tuple | None:
    """(x1, y1, x2, y2) from `{x1, y1, x2, y2}` or `[x1, y1, x2, y2]`."""
    try:
        if isinstance(value, dict):
            box = tuple(float(value[k]) for k in ("x1", "y1", "x2", "y2"))
        elif isinstance(value, (list, tuple)) and len(value) == 4:
            box = tuple(float(v) for v in value)
        else:
            return None
    except (KeyError, TypeError, ValueError):
        return None
    x1, y1, x2, y2 = box
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def frame_labels(row: dict) -> dict:
    """One frame's boxes, normalised to 0..1, and whether they are complete.

    Every kept person gets a `person` box. Each helmet-wearer (helmet class
    only) gets a `helmet` box: the one a person drew or accepted when the
    frame has one wearer, else the per-person correction or the detector's
    box at >= 0.5 (the query picks those as `auto`). A wearer left with no
    helmet box makes the frame INCOMPLETE: trained on, it would teach that
    the helmet is background, so a labelled pull leaves such frames out.
    """
    try:
        width, height = float(row.get("width") or 0), float(row.get("height") or 0)
    except (TypeError, ValueError):
        width = height = 0.0
    people = row.get("people") or []
    if isinstance(people, str):
        try:
            people = json.loads(people)
        except ValueError:
            people = []
    if width <= 0 or height <= 0:
        return {"boxes": [], "complete": False, "reason": "no frame size recorded"}
    drawn = _box(row.get("drawn_box"))
    wearers = [p for p in people if isinstance(p, dict) and p.get("wearing")]
    boxes, missing = [], 0
    for person in people:
        if not isinstance(person, dict):
            continue
        body = _box(person.get("person_box"))
        if body:
            boxes.append((PERSON, body))
        if person.get("wearing"):
            helmet = drawn if (drawn and len(wearers) == 1) else _box(person.get("auto"))
            if helmet:
                boxes.append((HELMET, helmet))
            else:
                missing += 1
    out = []
    for cls, (x1, y1, x2, y2) in boxes:
        x1, x2 = max(0.0, x1 / width), min(1.0, x2 / width)
        y1, y2 = max(0.0, y1 / height), min(1.0, y2 / height)
        if x2 > x1 and y2 > y1:
            out.append({"cls": cls, "name": LABEL_NAMES[cls],
                        "x1": round(x1, 6), "y1": round(y1, 6),
                        "x2": round(x2, 6), "y2": round(y2, 6)})
    reason = f"{missing} helmet box(es) missing" if missing else ""
    return {"boxes": out, "complete": not missing, "reason": reason,
            "width": int(width), "height": int(height)}


def yolo_text(labels: dict) -> str:
    """The boxes as a YOLO label file: `cls cx cy w h`, one per line."""
    lines = []
    for b in labels.get("boxes") or []:
        cx, cy = (b["x1"] + b["x2"]) / 2, (b["y1"] + b["y2"]) / 2
        lines.append(f"{b['cls']} {cx:.6f} {cy:.6f} {b['x2'] - b['x1']:.6f} {b['y2'] - b['y1']:.6f}")
    return "\n".join(lines) + ("\n" if lines else "")


def data_yaml_text() -> str:
    """The folder's data.yaml: the only record of what the label numbers
    mean once the files leave this tab."""
    names = "\n".join(f"  {k}: {v}" for k, v in LABEL_NAMES.items())
    return ("# Pulled from the review database by the SQL tab. Class numbers match\n"
            "# label-curation/pipeline_v1; only helmet (0) and person (2) are labelled.\n"
            "path: .\ntrain: images/train\nval: images/val\nnames:\n" + names + "\n")


# --- previews: short-lived links the page can show as thumbnails ------------

#: Thumbnails signed per request. The page asks for one screenful at a time.
PREVIEW_LIMIT = 60
#: Long enough to look through a screenful, short enough not to be a share link.
PREVIEW_SECONDS = 900


def previews(uris: list[str]) -> list[dict]:
    """A signed GET link for each `s3://` image, for an <img>. Signing is
    local -- no S3 call -- so a screenful costs nothing.

    An image stored without an extension (the cargo `.../source` frames)
    is served as JPEG, which is what those are; S3 holds them as
    octet-stream."""
    out = []
    for uri in (uris or [])[:PREVIEW_LIMIT]:
        if not isinstance(uri, str) or not uri.startswith("s3://"):
            continue
        parsed = urlparse(uri)
        key = parsed.path.lstrip("/")
        params = {"Bucket": parsed.netloc, "Key": key}
        if "." not in key.rsplit("/", 1)[-1]:
            params["ResponseContentType"] = "image/jpeg"
        out.append({"uri": uri, "url": storage.s3.generate_presigned_url(
            "get_object", Params=params, ExpiresIn=PREVIEW_SECONDS)})
    return out


# --- ready-made: human-verified frames of one class (2026-10-29) -------------
#
# The review database's own rule for what may be trained on (the Retraining
# Readiness Flow): a frame passes only when every person kept in it has a
# HUMAN `violation` / `no_violation`. VLM-only verdicts, similarity prefills
# and auto-settled answers do not count. Built here rather than pasted, so
# the 80-line query cannot arrive cut short.

#: Where the review app copies the frames it judges; `image_key` is a key
#: in this bucket.
REVIEW_IMAGE_BUCKET = "safety-model-dataset-dev"
#: Which frames of a class: everything a human resolved, only those with a
#: usable helmet box, or only those whose box a person drew or accepted.
PRESET_SCOPES = ("resolved", "ready", "test")
#: The only class with a helmet-box step; the others stop at "resolved".
BOX_CLASS = "no_helmet"
_CLASS_NAME = re.compile(r"^[a-z0-9_]{1,64}$")


def classes(key: str) -> list[dict]:
    """Every class (hazard type) the review database has judged frames for,
    with how many frames each -- read live, so a new class appears without
    a change here."""
    out = _invoke(key, {"action": "query", "sql": (
        "SELECT data->>'hazard_type' AS class, count(DISTINCT case_key) AS frames "
        "FROM audit_rows WHERE data->>'hazard_type' IS NOT NULL "
        "GROUP BY 1 ORDER BY 2 DESC")})
    return [{"name": str(r[0]), "frames": int(r[1] or 0),
             "has_boxes": str(r[0]) == BOX_CLASS}
            for r in out["rows"] if r and r[0]]


def preset_sql(hazard: str, scope: str = "resolved") -> str:
    """The human-verified frames of one class, with an `image` column.

    `scope` "ready" keeps frames whose helmet box is usable for training
    (drawn or accepted by a person, or a detector box >= 0.5 on every
    helmet-wearer); "test" keeps only person-drawn boxes, the test-grade
    ones. Both apply to the helmet class only -- the others have no box
    step yet.
    """
    if not _CLASS_NAME.match(hazard or ""):
        raise ValueError(f"Unknown class {hazard!r}.")
    if scope not in PRESET_SCOPES:
        raise ValueError(f"Unknown choice {scope!r}.")
    if scope != "resolved" and hazard != BOX_CLASS:
        raise ValueError(f"{hazard} has no helmet-box step; choose all human-resolved frames.")
    box = (f"CASE WHEN hb.case_key IS NOT NULL AND j.wearing <= 1 THEN 'human (test-grade)' "
           f"WHEN j.wearing_boxed = j.wearing THEN 'detector >= 0.5 (train only)' "
           f"ELSE 'missing' END") if hazard == BOX_CLASS else "'n/a'"
    keep = {"resolved": "", "ready": "WHERE helmet_box <> 'missing'",
            "test": "WHERE helmet_box = 'human (test-grade)'"}[scope]
    wearing = ("coalesce(an.verdict, 'no_violation') = 'no_violation'"
               if hazard == BOX_CLASS else "false")
    human_review = ("coalesce(r.data->>'review_origin', 'human') <> 'similarity_prefill' "
                    "AND coalesce(r.data->>'auto_settled', 'false') <> 'true'")
    return f"""SELECT * FROM (
WITH frames AS (
  SELECT a.case_key, min(a.data->>'image_key') AS image_key,
         array_agg(DISTINCT a.candidate_id) AS people
  FROM audit_rows a WHERE a.data->>'hazard_type' = '{hazard}'
  GROUP BY a.case_key
),
detector_box AS (
  SELECT case_key, candidate_id AS person,
         max((data->>'helmet_box_confidence')::float) AS confidence
  FROM audit_rows WHERE data->>'hazard_type' = '{hazard}'
  GROUP BY case_key, candidate_id
),
answers AS (
  SELECT DISTINCT ON (case_key, person) case_key, person, verdict
  FROM (
    SELECT h.case_key, h.candidate_id AS person, e.current_verdict AS verdict, 0 AS rank
    FROM crop_verdict_heads h JOIN crop_verdict_events e ON e.event_id = h.current_event_id
    WHERE e.actor_subject IS NOT NULL
    UNION ALL
    SELECT c.case_key, v.key, v.value, 1
    FROM corrections c, jsonb_each_text(coalesce(c.data->'crop_verdicts', '{{}}')) v
    UNION ALL
    SELECT r.case_key, v.key, v.value, 2
    FROM reviews r, jsonb_each_text(coalesce(r.data->'crop_verdicts', '{{}}')) v
    WHERE {human_review}
  ) x
  ORDER BY case_key, person, rank
),
struck AS (
  SELECT c.case_key, e AS person
  FROM corrections c, jsonb_array_elements_text(coalesce(c.data->'excluded_people', '[]')) e
  UNION
  SELECT case_key, person FROM answers
  WHERE verdict IN ('not_person', 'not_cargo', 'invalid_crop')
),
frame_verdict AS (
  SELECT case_key, coalesce(data->>'corrected_verdict', data->>'audited_verdict') AS verdict
  FROM corrections
  UNION ALL
  SELECT r.case_key, r.data->>'verdict' FROM reviews r WHERE {human_review}
),
human_box AS (
  SELECT DISTINCT case_key FROM reviews
  WHERE data->>'helmet_box_status' IN ('drawn', 'accepted_cv')
),
person_row AS (             -- each person's boxes, from their best-boxed judge row
  SELECT DISTINCT ON (case_key, candidate_id) case_key, candidate_id AS person,
         data->'person_box' AS person_box, data->'helmet_box' AS helmet_box,
         (data->>'helmet_box_confidence')::float AS helmet_conf,
         (data->>'image_width')::int AS width, (data->>'image_height')::int AS height
  FROM audit_rows WHERE data->>'hazard_type' = '{hazard}'
  ORDER BY case_key, candidate_id,
           (data->>'helmet_box_confidence')::float DESC NULLS LAST, judged_at DESC
),
box_override AS (           -- a helmet box corrected for one person on review
  SELECT r.case_key, o.key AS person, o.value->'helmet_box' AS box,
         o.value->>'helmet_box_source' AS source,
         (o.value->>'helmet_box_confidence')::float AS conf
  FROM reviews r, jsonb_each(coalesce(r.data->'helmet_box_overrides', '{{}}')) o
),
drawn_box AS (              -- a helmet box a person drew or accepted: [x1, y1, x2, y2]
  SELECT DISTINCT ON (case_key) case_key, data->'helmet_box' AS box
  FROM reviews
  WHERE data->>'helmet_box_status' IN ('drawn', 'accepted_cv')
    AND jsonb_typeof(data->'helmet_box') = 'array'
  ORDER BY case_key, data->>'reviewed_at' DESC NULLS LAST
),
judged AS (
  SELECT f.case_key, f.image_key,
         max(pr.width) AS width, max(pr.height) AS height,
         -- every kept person's boxes, for labels: `wearing` is a helmet
         -- answer, `auto` the best non-hand-drawn helmet box for them
         jsonb_agg(jsonb_build_object(
           'person_box', pr.person_box,
           'wearing', {wearing},
           'auto', CASE WHEN bo.box IS NOT NULL
                         AND (coalesce(bo.source, 'cv') <> 'cv' OR bo.conf >= 0.5) THEN bo.box
                        WHEN pr.helmet_conf >= 0.5 THEN pr.helmet_box END)) AS people,
         count(*) AS kept_people,
         count(*) FILTER (WHERE coalesce(an.verdict, 'no_violation') = 'no_violation') AS wearing,
         count(*) FILTER (WHERE coalesce(an.verdict, 'no_violation') = 'no_violation'
                            AND db.confidence >= 0.5) AS wearing_boxed,
         count(*) FILTER (WHERE an.verdict IN ('violation', 'no_violation')) AS answered,
         count(*) FILTER (WHERE an.verdict = 'violation') AS violations,
         count(*) FILTER (WHERE an.verdict = 'cannot_determine') AS undecided
  FROM frames f
  CROSS JOIN LATERAL unnest(f.people) AS p(person)
  LEFT JOIN answers an ON an.case_key = f.case_key AND an.person = p.person
  LEFT JOIN detector_box db ON db.case_key = f.case_key AND db.person = p.person
  LEFT JOIN person_row pr ON pr.case_key = f.case_key AND pr.person = p.person
  LEFT JOIN box_override bo ON bo.case_key = f.case_key AND bo.person = p.person
  WHERE NOT EXISTS (SELECT 1 FROM struck s WHERE s.case_key = f.case_key AND s.person = p.person)
  GROUP BY f.case_key, f.image_key
)
SELECT 's3://{REVIEW_IMAGE_BUCKET}/' || j.image_key AS image,
       j.case_key, j.kept_people, j.violations,
       {box} AS helmet_box,
       j.width, j.height, j.people, dbx.box AS drawn_box
FROM judged j
LEFT JOIN human_box hb ON hb.case_key = j.case_key
LEFT JOIN drawn_box dbx ON dbx.case_key = j.case_key
WHERE j.kept_people > 0
  AND (j.answered = j.kept_people
   OR (j.undecided = 0 AND EXISTS (
         SELECT 1 FROM frame_verdict fv
         WHERE fv.case_key = j.case_key
           AND ((fv.verdict = 'no_violation' AND j.violations = 0)
                OR (fv.verdict = 'violation' AND j.kept_people = 1)))))
) t
{keep}
ORDER BY helmet_box, case_key"""


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
         dest_bucket: str, dest_folder: str, train_percent: int = 100,
         with_labels: bool = True) -> dict:
    """Run the query, then copy every image it names into
    `dest_bucket/dest_folder` in the layout a training run reads:

        <dest>/images/train/<name>   <dest>/labels/train/<stem>.txt
        <dest>/images/val/<name>     <dest>/labels/val/<stem>.txt

    With `with_labels`, each image gets a label: written from the boxes in
    the database when the query returned them (see `frame_labels`; frames
    with a helmet box missing are left out, and counted), otherwise copied
    from where Ultralytics would look beside the source image. A data.yaml
    naming the classes is written when labels were generated. Without it,
    images only.

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
    incomplete = 0
    if with_labels:
        kept = []
        for image in images:
            if image.get("labels") and not image["labels"].get("complete"):
                incomplete += 1
                continue
            kept.append(image)
        images = kept
    generated = with_labels and any(i.get("labels") for i in images)

    plan, taken = [], set()
    for image in images:
        name = image["key"].rsplit("/", 1)[-1]
        if "." not in name:
            # `.../<id>/source`: the folder is the identity, and these are
            # JPEGs (the cargo evidence frames).
            parent = image["key"].rsplit("/", 2)[-2] if image["key"].count("/") else name
            name = f"{parent}.jpg"
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
            label_target = f"{dest_folder}/labels/{side}/{name.rsplit('.', 1)[0]}.txt"
            if not with_labels:
                return out
            if image.get("labels"):
                if not _exists(dest_bucket, label_target):
                    storage.s3.put_object(Bucket=dest_bucket, Key=label_target,
                                          Body=yolo_text(image["labels"]).encode(),
                                          ContentType="text/plain")
                out["label"] = "database"
                return out
            source_label = label_key_for(image["key"])
            if source_label and _exists(image["bucket"], source_label):
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
        if d["label"] and d["label"] != "database":
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
        "left_out_incomplete": incomplete,
        "labels_from": "database" if generated else ("source" if with_labels else "none"),
        "failed": len(done) - len(ok),
        "errors": [f"{d['image']}: {d['error']}" for d in done if d["error"]][:10],
    }
    storage.s3.put_object(Bucket=dest_bucket, Key=sources_key,
                          Body=json.dumps(sources, indent=1).encode(),
                          ContentType="application/json")
    if generated:
        storage.s3.put_object(Bucket=dest_bucket, Key=f"{dest_folder}/data.yaml",
                              Body=data_yaml_text().encode(), ContentType="text/yaml")
    storage.s3.put_object(
        Bucket=dest_bucket, Key=f"{dest_folder}/query.json",
        Body=json.dumps({"database": key, "sql": sql, "bucket": bucket, "folder": folder,
                         "train_percent": train_percent, "with_labels": with_labels,
                         "at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                         "summary": summary}, indent=1).encode(),
        ContentType="application/json")
    return summary
