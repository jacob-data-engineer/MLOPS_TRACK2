"""Independent, persisted best-model exports on a Singapore calendar."""
from __future__ import annotations

import datetime as dt
import json
import re
import uuid
import os
from botocore.exceptions import ClientError
from . import metrics_export as mx
from . import track2_shared as t2

SCOPES = ("comparison", "model_name")
KIND = "metrics_export_schedule"
SGT = dt.timezone(dt.timedelta(hours=8))


def now():
    return dt.datetime.now(dt.timezone.utc)


def _get(scope):
    from .. import pull_control as pc
    item = pc.ddb.get_item(TableName=pc.TABLE_NAME,
        Key={"kind": {"S": KIND}, "sort": {"S": scope}}, ConsistentRead=True).get("Item")
    return json.loads(item["body"]["S"]) if item else {"scope": scope, "enabled": False}


def _put(plan, previous=None):
    from .. import pull_control as pc
    options = {}
    if previous is not None:
        options = {"ConditionExpression": "#body = :previous",
                   "ExpressionAttributeNames": {"#body": "body"},
                   "ExpressionAttributeValues": {":previous": {"S": json.dumps(previous)}}}
    try:
        pc.ddb.put_item(TableName=pc.TABLE_NAME, Item={"kind": {"S": KIND},
            "sort": {"S": plan["scope"]}, "body": {"S": json.dumps(plan)}}, **options)
        return True
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException":
            return False
        raise


def list_schedules():
    plans = [_get(scope) for scope in SCOPES]
    for plan in plans:
        result = plan.get('last_result') or {}
        if result.get('export_id'):
            from ..data import catalog_view as cv
            jobs = cv.exports(run_id=result['run_id'])
            job = next((j for j in jobs if j['id'] == result['export_id']), None)
            if job:
                plan['last_result'] = {**result, 'state': job['state'], 'detail': job.get('error') or job.get('stage'),
                                       'written_to': job.get('destination_uri'), 'export': job}
    return plans


def package_options(plan):
    options = plan.get('package_options')
    if options is None:
        return None  # Existing saved schedules keep their checkpoint-only behavior.
    if not any(options.get(k, True) for k in ('include_model', 'include_images', 'include_metadata')):
        raise ValueError('Select at least one package item to export.')
    if options.get('include_metadata', True) and options.get('metadata_groups') == []:
        raise ValueError('Select at least one metadata category.')
    if options.get('format', 'zip') not in ('zip', 'folder'):
        raise ValueError('Choose ZIP or folder format.')
    return options


def candidates(scope, model_package_arns=None, model_name=""):
    if scope not in SCOPES:
        raise ValueError("Choose compared models or a Model Name.")
    if scope == "comparison":
        wanted = set(model_package_arns or [])
        if not wanted:
            raise ValueError("Select models in Comparison first.")
        if any(not arn.startswith("arn:") or ":model-package/" not in arn or len(arn.rsplit("/", 2)) != 3 for arn in wanted):
            raise ValueError("The compared models must be registered model package identifiers.")
        groups = {arn.rsplit("/", 2)[-2] for arn in wanted}
        rows = [row for group in sorted(groups) for row in mx.group_versions(group, strict=True)]
        rows = [row for row in rows if row.get("model_package_arn") in wanted]
        missing = wanted - {row.get("model_package_arn") for row in rows}
        if missing:
            raise ValueError("Some selected models are unavailable. Refresh the comparison and update the selection.")
    else:
        if not model_name:
            raise ValueError("Choose one Model Name.")
        groups = {group["group_name"] for group in t2.exportable_groups() if group.get("registry") == model_name}
        if not groups:
            raise ValueError("This Model Name is no longer available.")
        rows = [row for group in sorted(groups) for row in mx.group_versions(group, strict=True)]
    return sorted(rows, key=lambda row: row.get("model_package_arn", ""))


def winner(plan):
    if plan.get("criterion") not in mx.CRITERIA:
        raise ValueError("Choose a supported ranking criterion.")
    rows = candidates(plan["scope"], plan.get("model_package_arns"), plan.get("model_name", ""))
    best, reason = mx.pick_best(rows, plan["criterion"])
    return best, reason, len(rows)


def preview(plan):
    best, reason, count = winner(plan)
    return {"best": mx.describe_pick(best, plan["criterion"]) if best else None,
            "reason": reason, "considered": count}


def export_now(plan, expected_arn=""):
    options = package_options(plan)
    best, reason, count = winner(plan)
    if not best:
        raise ValueError(reason)
    if expected_arn and best.get("model_package_arn") != expected_arn:
        raise ValueError("The best model changed. Preview again before exporting.")
    if options is not None:
        from ..data import catalog_view as cv
        arn = best['model_package_arn']
        found = cv._rows('SELECT run_id FROM training_runs WHERE model_package_arn = %s', (arn,))
        if not found:
            raise ValueError('The winning model has no recorded training run for a package export. Choose a trained version.')
        run_id = found[0]['run_id']
        job = cv.start_export('run', run_id, destination=plan.get('destination', 's3'),
            bucket=plan.get('bucket', ''), prefix=plan.get('prefix', ''),
            format='zip' if plan.get('destination') == 'local' else options.get('format', 'zip'),
            include_model=options.get('include_model', True), include_images=options.get('include_images', True),
            include_metadata=options.get('include_metadata', True), metadata_groups=options.get('metadata_groups'),
            model_package_arn=arn)
        return {'picked': mx.describe_pick(best, plan['criterion']), 'considered': count,
                'state': job['state'], 'export_id': job['id'], 'run_id': run_id, 'export': job,
                'url': '', 'filename': '', 'written_to': ''}
    placed = mx.export_checkpoint(best, destination=plan.get("destination", "s3"),
        bucket=plan.get("bucket", ""), prefix=plan.get("prefix", ""), retention="keep_all")
    return {"picked": mx.describe_pick(best, plan["criterion"]), "considered": count, **placed}


def next_time(start_date, time_sgt, every_days, after):
    if not 1 <= every_days <= 365:
        raise ValueError("Repeat every 1 to 365 days.")
    if not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", time_sgt):
        raise ValueError("Choose a valid Singapore time (HH:mm).")
    day = dt.date.fromisoformat(start_date) if start_date else after.astimezone(SGT).date()
    hour, minute = map(int, time_sgt.split(":"))
    first = dt.datetime.combine(day, dt.time(hour, minute), tzinfo=SGT)
    interval = dt.timedelta(days=every_days)
    if first <= after:
        first += interval * (int((after - first) // interval) + 1)
    return first.astimezone(dt.timezone.utc).isoformat()


def save(scope, payload):
    if scope not in SCOPES:
        raise ValueError("Unknown export schedule.")
    if not payload.get("enabled"):
        # Preserve the saved scope and destination when stopping a schedule.
        plan = {**_get(scope), "enabled": False, "revision": uuid.uuid4().hex}
    else:
        package_options(payload)
        bucket = str(payload.get("bucket", "")).strip()
        if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", bucket):
            raise ValueError("A scheduled export needs a valid S3 bucket name.")
        prefix = str(payload.get("prefix", "")).strip("/")
        if ".." in prefix.split("/"):
            raise ValueError("The S3 prefix cannot contain parent path segments.")
        plan = {**payload, "scope": scope, "bucket": bucket, "prefix": prefix,
                "destination": "s3", "revision": uuid.uuid4().hex, "last_result": {}}
        # Validate exact model identities, but a named model may have no versions yet.
        winner(plan)
        plan["model_package_arns"] = list(dict.fromkeys(payload.get("model_package_arns", []))) if scope == "comparison" else []
        plan["model_name"] = payload.get("model_name", "") if scope == "model_name" else ""
        plan["next_run_at"] = next_time(plan.get("start_date", ""), plan["time_sgt"], plan["every_days"], now())
    _put(plan)
    return plan


def check():
    outcomes = []
    for scope in SCOPES:
        try:
            plan = _get(scope)
            # Package builds run in the persistent EC2 worker, not a short-lived Lambda invocation.
            if plan.get('package_options') is not None and os.getenv('AWS_LAMBDA_FUNCTION_NAME'):
                continue
            if not plan.get("enabled") or dt.datetime.fromisoformat(plan["next_run_at"]) > now():
                continue
            started = now()
            # Advancing the due slot atomically claims it across app/worker processes.
            # A missed interval produces one export, never a burst of historical exports.
            advance = {**plan, "next_run_at": next_time(
                dt.datetime.fromisoformat(plan["next_run_at"]).astimezone(SGT).date().isoformat(),
                plan["time_sgt"], plan["every_days"], started),
                "last_result": {"state": "running", "at": started.isoformat()}}
            if not _put(advance, plan):
                continue
            # An edit/stop before export takes precedence over this claimed slot.
            if _get(scope) != advance:
                continue
            try:
                # Separate scopes and slots even if both choose the same model in one second.
                slot = f"scheduled-{scope}/{plan['revision']}-{int(dt.datetime.fromisoformat(plan['next_run_at']).timestamp())}"
                run_plan = {**plan, "prefix": "/".join(p for p in (plan.get("prefix", ""), slot) if p)}
                result = {"state": "ready", "at": started.isoformat(), **export_now(run_plan)}
            except Exception as exc:
                result = {"state": "failed", "at": started.isoformat(), "detail": str(exc)[:500]}
            _put({**advance, "last_result": result}, advance)
            outcomes.append({"scope": scope, **result})
        except Exception as exc:
            outcomes.append({"scope": scope, "state": "failed", "detail": str(exc)[:500]})
    return outcomes
