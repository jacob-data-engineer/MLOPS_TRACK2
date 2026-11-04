"""PostgreSQL operational records, with the small Dynamo API used by Track2.

The adapter preserves existing call sites and typed values, but every SQL-mode
read and mutation goes to PostgreSQL. Unsupported expressions fail explicitly.
Each mutation locks its key transactionally; counters and conditional claims
are atomic across processes, including inserts of previously absent keys.
"""
from __future__ import annotations

import base64
import contextlib
import json
import os
import re
import functools
from decimal import Decimal

from boto3.dynamodb.types import TypeDeserializer, TypeSerializer
from botocore.exceptions import ClientError


def enabled() -> bool:
    mode = os.getenv("TRACK2_RECORD_BACKEND", "dynamodb")
    if mode not in ("dynamodb", "postgres"):
        raise ValueError("Unknown TRACK2_RECORD_BACKEND")
    if mode == "postgres" and not re.fullmatch(r"track2_dev(?:_[a-z0-9_]+)?", os.getenv("CATALOG_SCHEMA", "")):
        raise ValueError("Postgres development records require an isolated track2_dev schema")
    return mode == "postgres"


def inventory():
    from .data.dynamo_migration import sources
    return {s.table: s for s in sources()}


def connection():
    from .data import catalog
    return catalog.connect()


def verify_ready():
    if not enabled():
        return
    with contextlib.closing(connection()) as conn:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM backend_migrations WHERE name='dynamo_to_sql'")
        if not cur.fetchone():
            raise RuntimeError("SQL cutover has not been initialized; refusing to start")


def exclusive(name):
    """Serialize queue ownership across SQL workers; release on process death."""
    def decorate(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            if not enabled():
                return fn(*args, **kwargs)
            with contextlib.closing(connection()) as conn:
                cur = conn.cursor()
                key = os.environ["CATALOG_SCHEMA"] + ':' + name
                cur.execute("SELECT pg_try_advisory_lock(hashtextextended(%s,0))", (key,))
                if not cur.fetchone()[0]:
                    return {"started": False, "reason": "Another SQL worker owns this queue."}
                try:
                    return fn(*args, **kwargs)
                finally:
                    cur.execute("SELECT pg_advisory_unlock(hashtextextended(%s,0))", (key,))
        return wrapped
    return decorate


def _numbers(value):
    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, dict):
        return {k: _numbers(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_numbers(v) for v in value]
    return value


def typed(item):
    serial = TypeSerializer()
    return {k: serial.serialize(_numbers(v)) for k, v in item.items()}


def _binary(value):
    kind, body = next(iter(value.items()))
    if kind == "B" and isinstance(body, str):
        body = base64.b64decode(body)
    elif kind == "BS":
        body = [base64.b64decode(v) if isinstance(v, str) else v for v in body]
    elif kind == "M":
        body = {k: _binary(v) for k, v in body.items()}
    elif kind == "L":
        body = [_binary(v) for v in body]
    return {kind: body}


def plain(item):
    if isinstance(item, str):
        item = json.loads(item)
    deserial = TypeDeserializer()
    return {k: deserial.deserialize(_binary(v)) for k, v in item.items()}


def encode(item):
    from .data.dynamo_migration import canonical
    return canonical(typed(item))


def _name(value, names):
    value = names.get(value, value)
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"Unsupported record attribute: {value}")
    return value


def expression_tree(expr, names, values):
    if hasattr(expr, "get_expression"):
        e = expr.get_expression()
        args = e["values"]
        if e["operator"] in ("AND", "OR"):
            return (e["operator"], *(expression_tree(v, names, values) for v in args))
        return (e["operator"], args[0].name, *args[1:])
    expr = str(expr).strip()
    if " AND " in expr:
        a, b = expr.split(" AND ", 1)
        return ("AND", expression_tree(a, names, values), expression_tree(b, names, values))
    match = re.fullmatch(r"(attribute_not_exists|attribute_exists)\(([^)]+)\)", expr)
    if match:
        return match[1], _name(match[2].strip(), names)
    match = re.fullmatch(r"begins_with\(([^,]+),\s*(:\w+)\)", expr)
    if match:
        return "begins_with", _name(match[1].strip(), names), values[match[2]]
    match = re.fullmatch(r"([#\w]+)\s*=\s*(:\w+)", expr)
    if match:
        return "=", _name(match[1], names), values[match[2]]
    raise ValueError(f"Unsupported record expression: {expr}")


def matches(tree, item):
    op, *args = tree
    if op == "AND":
        return all(matches(v, item) for v in args)
    if op == "OR":
        return any(matches(v, item) for v in args)
    name = args[0]
    if op == "attribute_not_exists":
        return name not in item
    if op == "attribute_exists":
        return name in item
    if op == "=":
        return name in item and item[name] == args[1]
    if op == "begins_with":
        return str(item.get(name, "")).startswith(str(args[1]))
    raise ValueError(f"Unsupported condition: {op}")


class Table:
    def __init__(self, name):
        try:
            self.source = inventory()[name]
        except KeyError:
            raise ValueError(f"No SQL mapping for table {name}") from None
        self.name = name
        self.sql = "records_" + self.source.target

    def key(self, item):
        values = [str(item[k]) for k in self.source.keys]
        return values[0], values[1] if len(values) > 1 else ""

    def _condition(self, kwargs):
        expr = kwargs.get("ConditionExpression")
        return expression_tree(expr, kwargs.get("ExpressionAttributeNames", {}),
                               kwargs.get("ExpressionAttributeValues", {})) if expr else None

    def _lock(self, cur, key):
        lock = json.dumps([self.sql, *key], separators=(",", ":"))
        cur.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s,0))", (lock,))

    def _read(self, cur, key):
        cur.execute(f"SELECT payload FROM {self.sql} WHERE pk=%s AND sk=%s "
                    "AND (expires_at IS NULL OR expires_at>extract(epoch from now()))", key)
        row = cur.fetchone()
        return plain(row[0]) if row else {}

    def _save(self, cur, item):
        pk, sk = self.key(item)
        cur.execute(f"INSERT INTO {self.sql} (pk,sk,payload,record_type,file_name,expires_at) "
                    "VALUES (%s,%s,%s::jsonb,%s,%s,%s) ON CONFLICT (pk,sk) DO UPDATE SET "
                    "payload=EXCLUDED.payload,record_type=EXCLUDED.record_type,"
                    "file_name=EXCLUDED.file_name,expires_at=EXCLUDED.expires_at",
                    (pk, sk, encode(item), str(item.get("record_type", item.get("kind", ""))),
                     str(item.get("name", "")), item.get("expires_at")))
        from .data.sql_projection import project
        project(cur, self.source.target, item)

    def get_item(self, *, Key, **kwargs):
        with contextlib.closing(connection()) as conn:
            item = self._read(conn.cursor(), self.key(Key))
            return {"Item": item} if item else {}

    def put_item(self, *, Item, **kwargs):
        with contextlib.closing(connection()) as conn:
            cur = conn.cursor()
            self._lock(cur, self.key(Item))
            old = self._read(cur, self.key(Item))
            condition = self._condition(kwargs)
            if condition and not matches(condition, old):
                raise ClientError({"Error": {"Code": "ConditionalCheckFailedException"}}, "PutItem")
            self._save(cur, Item)
            conn.commit()
            return {"Attributes": old} if kwargs.get("ReturnValues") == "ALL_OLD" else {}

    def delete_item(self, *, Key, **kwargs):
        with contextlib.closing(connection()) as conn:
            cur = conn.cursor()
            key = self.key(Key)
            self._lock(cur, key)
            old = self._read(cur, key)
            condition = self._condition(kwargs)
            if condition and not matches(condition, old):
                raise ClientError({"Error": {"Code": "ConditionalCheckFailedException"}}, "DeleteItem")
            cur.execute(f"DELETE FROM {self.sql} WHERE pk=%s AND sk=%s", key)
            from .data.sql_projection import remove
            remove(cur, self.source.target, old)
            conn.commit()
            return {"Attributes": old} if kwargs.get("ReturnValues") == "ALL_OLD" else {}

    def update_item(self, *, Key, UpdateExpression, **kwargs):
        with contextlib.closing(connection()) as conn:
            cur = conn.cursor()
            self._lock(cur, self.key(Key))
            old = self._read(cur, self.key(Key))
            condition = self._condition(kwargs)
            if condition and not matches(condition, old):
                raise ClientError({"Error": {"Code": "ConditionalCheckFailedException"}}, "UpdateItem")
            item = {**Key, **old}
            names, values = kwargs.get("ExpressionAttributeNames", {}), kwargs.get("ExpressionAttributeValues", {})
            changed = {}
            if UpdateExpression.startswith("SET "):
                for assignment in UpdateExpression[4:].split(","):
                    name, token = (s.strip() for s in assignment.split("=", 1))
                    changed[_name(name, names)] = values[token]
            elif UpdateExpression.startswith("ADD "):
                name, token = UpdateExpression[4:].split()
                name = _name(name, names)
                changed[name] = item.get(name, 0) + values[token]
            else:
                raise ValueError("Unsupported update expression")
            if any(k in changed for k in self.source.keys):
                raise ValueError("Cannot change record keys")
            item.update(changed)
            self._save(cur, item)
            conn.commit()
            mode = kwargs.get("ReturnValues", "NONE")
            return {} if mode == "NONE" else {"Attributes": {"UPDATED_NEW": changed, "ALL_NEW": item, "ALL_OLD": old}[mode]}

    def _sql_condition(self, tree):
        op, *args = tree
        if op in ("AND", "OR"):
            parts = [self._sql_condition(t) for t in args]
            return "(" + f" {op} ".join(p[0] for p in parts) + ")", tuple(v for p in parts for v in p[1])
        name = _name(args[0], {})
        columns = {self.source.keys[0]: "pk", "record_type": "record_type", "name": "file_name"}
        if len(self.source.keys) > 1:
            columns[self.source.keys[1]] = "sk"
        if op == "=" and name in columns and isinstance(args[1], str):
            return columns[name] + "=%s", (args[1],)
        if op == "begins_with" and name in columns:
            prefix = str(args[1]).replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            return columns[name] + " LIKE %s", (prefix + "%",)
        if op == "=":
            return "payload -> %s = %s::jsonb", (name, json.dumps(json.loads(encode({name: args[1]}))[name]))
        if op in ("attribute_exists", "attribute_not_exists"):
            return ("NOT " if op == "attribute_not_exists" else "") + "(payload ? %s)", (name,)
        raise ValueError(f"Unsupported query operation: {op}")

    def _list(self, **kwargs):
        index = kwargs.get("IndexName")
        if index and (index != "by-file-name" or self.source.target != "training_files"):
            raise ValueError("Unknown record index")
        clauses = ["(expires_at IS NULL OR expires_at>extract(epoch from now()))"]
        params = []
        for field in ("KeyConditionExpression", "FilterExpression"):
            if kwargs.get(field) is not None:
                tree = expression_tree(kwargs[field], kwargs.get("ExpressionAttributeNames", {}), kwargs.get("ExpressionAttributeValues", {}))
                clause, values = self._sql_condition(tree)
                clauses.append(clause); params.extend(values)
        forward = kwargs.get("ScanIndexForward", True)
        last = kwargs.get("ExclusiveStartKey")
        if last:
            clauses.append("(pk,sk) " + (">" if forward else "<") + " (%s,%s)")
            params.extend(self.key(last))
        limit = min(max(int(kwargs.get("Limit", 1000)), 1), 10000)
        order = "ASC" if forward else "DESC"
        with contextlib.closing(connection()) as conn:
            cur = conn.cursor()
            cur.execute(f"SELECT payload FROM {self.sql} WHERE " + " AND ".join(clauses) +
                        f" ORDER BY pk {order},sk {order} LIMIT %s", tuple(params) + (limit + 1,))
            items = [plain(row[0]) for row in cur.fetchall()]
        more = len(items) > limit
        items = items[:limit]
        result = {"Items": items, "Count": len(items), "ScannedCount": len(items)}
        if more:
            result["LastEvaluatedKey"] = {k: items[-1][k] for k in self.source.keys}
        projection = kwargs.get("ProjectionExpression")
        if projection:
            fields = [_name(s.strip(), kwargs.get("ExpressionAttributeNames", {})) for s in projection.split(",")]
            result["Items"] = [{k: item[k] for k in fields if k in item} for item in items]
        return result

    scan = _list
    query = _list

    def batch_writer(self, **kwargs):
        return Batch(self)


class Batch:
    def __init__(self, table):
        self.table = table
        self.pending = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *args):
        if exc_type is None:
            self.flush()

    def put_item(self, Item):
        self.pending.append(("put", Item))
        if len(self.pending) >= 200:
            self.flush()

    def delete_item(self, Key):
        self.pending.append(("delete", Key))
        if len(self.pending) >= 200:
            self.flush()

    def flush(self):
        if not self.pending:
            return
        with contextlib.closing(connection()) as conn:
            cur = conn.cursor()
            for op, item in sorted(self.pending, key=lambda p: self.table.key(p[1])):
                key = self.table.key(item)
                self.table._lock(cur, key)
                if op == "put":
                    self.table._save(cur, item)
                else:
                    old = self.table._read(cur, key)
                    cur.execute(f"DELETE FROM {self.table.sql} WHERE pk=%s AND sk=%s", key)
                    from .data.sql_projection import remove
                    remove(cur, self.table.source.target, old)
            conn.commit()
        self.pending.clear()


class Resource:
    Table = staticmethod(Table)


class Client:
    """Low-level typed interface for the pull-control module."""
    def __getattr__(self, method):
        if method not in ("get_item", "put_item", "update_item", "delete_item", "query", "scan"):
            raise AttributeError(method)

        def call(*, TableName, **kwargs):
            for k in ("Item", "Key", "ExpressionAttributeValues", "ExclusiveStartKey"):
                if k in kwargs:
                    kwargs[k] = plain(kwargs[k])
            result = getattr(Table(TableName), method)(**kwargs)
            for k in ("Item", "Attributes", "LastEvaluatedKey"):
                if k in result:
                    result[k] = typed(result[k])
            if "Items" in result:
                result["Items"] = [typed(v) for v in result["Items"]]
            return result
        return call
