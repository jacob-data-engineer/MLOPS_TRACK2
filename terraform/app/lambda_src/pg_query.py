"""Read-only SQL against one Postgres database, for the Track2 SQL tab.

WHY A LAMBDA. The databases are private and live in their own VPCs, which
have no route to the review host and no NAT or VPC endpoints. A function
placed INSIDE the database's VPC can reach it on 5432 and nothing else;
the app invokes it through the Lambda API. One deployment per database,
same code -- a function sits in one VPC.

WHY IT CANNOT CHANGE ANYTHING, three ways over:
  1. It logs in as `track2_readonly`, a role granted SELECT and nothing
     else (created once by `bootstrap`, below).
  2. Every query runs inside `BEGIN READ ONLY`, so even a grant added
     later by somebody else cannot be used to write from here.
  3. One statement per call. `SET transaction_read_only = off; DELETE ...`
     is two, and is refused before it reaches the server.
Plus a statement timeout, so an accidental cross join ends instead of
holding a connection on somebody else's database.

The password arrives in the environment (encrypted at rest by Lambda):
the VPC has no route to Secrets Manager.
"""

from __future__ import annotations

import datetime
import decimal
import json
import os
import ssl
import uuid

import pg8000.dbapi

#: Lambda's synchronous response limit is 6 MB. Rows stop a little short
#: of it and the reply says so, rather than the whole call failing.
MAX_RESPONSE_BYTES = 5_500_000
DEFAULT_TIMEOUT_MS = 30_000


def _connect(user: str, password: str):
    context = ssl.create_default_context()
    # RDS's certificate chain is not in Lambda's default trust store; the
    # connection is still encrypted, and it never leaves the VPC.
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return pg8000.dbapi.connect(
        host=os.environ["PGHOST"], port=int(os.environ.get("PGPORT", "5432")),
        database=os.environ["PGDATABASE"], user=user, password=password,
        ssl_context=context, timeout=15,
    )


def _plain(value):
    """A JSON-safe copy of one cell."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        return value.isoformat()
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<{len(bytes(value))} bytes>"
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    return str(value)


def _one_statement(sql: str) -> str:
    """The SQL with any trailing semicolon removed, or ValueError when it
    holds more than one statement. Quotes and comments are respected, so a
    `;` inside a string literal does not count."""
    text = sql.strip()
    out, i, quote = [], 0, ""
    while i < len(text):
        ch = text[i]
        if quote:
            out.append(ch)
            if ch == quote:
                quote = ""
        elif ch in ("'", '"'):
            quote = ch
            out.append(ch)
        elif text.startswith("--", i):
            end = text.find("\n", i)
            i = len(text) if end < 0 else end
            continue
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = len(text) if end < 0 else end + 2
            continue
        elif ch == ";":
            if text[i + 1:].strip(" \t\r\n;"):
                raise ValueError("One statement at a time -- remove everything after the first ';'.")
            break
        else:
            out.append(ch)
        i += 1
    body = "".join(out).strip()
    if not body:
        raise ValueError("Type a query first.")
    return body


def _query(event: dict) -> dict:
    sql = _one_statement(str(event.get("sql") or ""))
    timeout = int(event.get("timeout_ms") or DEFAULT_TIMEOUT_MS)
    conn = _connect(os.environ["PGUSER"], os.environ["PGPASSWORD"])
    try:
        cur = conn.cursor()
        cur.execute("BEGIN READ ONLY")
        cur.execute(f"SET LOCAL statement_timeout = {max(1000, min(timeout, 120_000))}")
        cur.execute(sql)
        columns = [d[0] for d in (cur.description or [])]
        rows, size, truncated = [], 2, False
        if columns:
            for raw in cur:
                row = [_plain(v) for v in raw]
                size += len(json.dumps(row, default=str)) + 1
                if size > MAX_RESPONSE_BYTES:
                    truncated = True
                    break
                rows.append(row)
        cur.execute("ROLLBACK")
        return {"ok": True, "columns": columns, "rows": rows, "truncated": truncated}
    finally:
        conn.close()


SCHEMA_SQL = """
SELECT c.table_schema, c.table_name, c.column_name, c.data_type
FROM information_schema.columns c
JOIN information_schema.tables t
  ON t.table_schema = c.table_schema AND t.table_name = c.table_name
WHERE c.table_schema NOT IN ('pg_catalog', 'information_schema')
  AND c.table_schema NOT LIKE 'pg_toast%'
ORDER BY c.table_schema, c.table_name, c.ordinal_position
"""


def _schema(_event: dict) -> dict:
    out = _query({"sql": SCHEMA_SQL})
    tables: dict = {}
    for schema, table, column, data_type in out["rows"]:
        tables.setdefault(f"{schema}.{table}", []).append({"name": column, "type": data_type})
    return {"ok": True, "tables": [{"name": k, "columns": v} for k, v in tables.items()]}


def _bootstrap(event: dict) -> dict:
    """Creates (or re-passwords) `track2_readonly` with SELECT on every
    table, current and future. Needs the admin login in the EVENT -- it is
    never stored here -- and is switched off unless ALLOW_BOOTSTRAP=1, set
    only for the setup and removed after."""
    if os.environ.get("ALLOW_BOOTSTRAP") != "1":
        raise ValueError("Bootstrap is switched off on this function.")
    user, password = os.environ["PGUSER"], os.environ["PGPASSWORD"]
    conn = _connect(event["admin_user"], event["admin_password"])
    conn.autocommit = True
    try:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (user,))
        verb = "ALTER" if cur.fetchone() else "CREATE"
        # Identifiers and the password cannot be bound parameters in DDL;
        # both come from this function's own environment, not a caller.
        quoted = password.replace("'", "''")
        cur.execute(f'{verb} ROLE "{user}" LOGIN PASSWORD \'{quoted}\' '
                    "NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION")
        cur.execute(f'ALTER ROLE "{user}" SET default_transaction_read_only = on')
        cur.execute(f'GRANT CONNECT ON DATABASE "{os.environ["PGDATABASE"]}" TO "{user}"')
        cur.execute("SELECT nspname FROM pg_namespace WHERE nspname NOT LIKE 'pg_%' "
                    "AND nspname <> 'information_schema'")
        schemas = [r[0] for r in cur.fetchall()]
        for schema in schemas:
            cur.execute(f'GRANT USAGE ON SCHEMA "{schema}" TO "{user}"')
            cur.execute(f'GRANT SELECT ON ALL TABLES IN SCHEMA "{schema}" TO "{user}"')
            cur.execute(f'ALTER DEFAULT PRIVILEGES IN SCHEMA "{schema}" GRANT SELECT ON TABLES TO "{user}"')
        return {"ok": True, "role": user, "action": verb.lower(), "schemas": schemas}
    finally:
        conn.close()


def lambda_handler(event, _context=None):
    action = (event or {}).get("action", "query")
    try:
        if action == "query":
            return _query(event)
        if action == "schema":
            return _schema(event)
        if action == "bootstrap":
            return _bootstrap(event)
        return {"ok": False, "error": f"Unknown action {action!r}."}
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:                                        # noqa: BLE001
        # The database's own words: a syntax error or a missing column is
        # the answer the person typing needs.
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:2000]}
