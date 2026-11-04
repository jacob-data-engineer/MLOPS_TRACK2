from decimal import Decimal
import pytest
from boto3.dynamodb.conditions import Key, Attr
from ppe_pipeline_api import record_store as rs


def test_sql_requires_isolated_schema(monkeypatch):
    monkeypatch.setenv("TRACK2_RECORD_BACKEND", "postgres")
    monkeypatch.setenv("CATALOG_SCHEMA", "public")
    with pytest.raises(ValueError):
        rs.enabled()
    monkeypatch.setenv("CATALOG_SCHEMA", "track2_dev")
    assert rs.enabled()


@pytest.mark.parametrize("initialized", [True, False])
def test_sql_startup_requires_completed_cutover(monkeypatch, initialized):
    class Connection:
        def cursor(self): return self
        def execute(self, sql): assert 'backend_migrations' in sql
        def fetchone(self): return (1,) if initialized else None
        def close(self): pass
    monkeypatch.setattr(rs, "enabled", lambda: True)
    monkeypatch.setattr(rs, "connection", Connection)
    if initialized:
        rs.verify_ready()
    else:
        with pytest.raises(RuntimeError, match="not been initialized"):
            rs.verify_ready()


def test_typed_roundtrip_is_lossless():
    item = {"n": Decimal("12345678901234567890.123456789"), "tags": {"x", "y"},
            "blob": b"\x00\xff", "nested": [False, None, {"a": 3}], "f": 0.125}
    result = rs.plain(rs.encode(item))
    assert result["n"] == item["n"]
    assert result["tags"] == item["tags"]
    assert bytes(result["blob"]) == item["blob"]
    assert result["nested"] == item["nested"]
    assert result["f"] == Decimal("0.125")


def test_boto_and_string_conditions_match():
    tree = rs.expression_tree(Key("audit_id").eq("run") & Key("file_key").begins_with("train#"), {}, {})
    assert rs.matches(tree, {"audit_id": "run", "file_key": "train#1"})
    assert not rs.matches(tree, {"audit_id": "other", "file_key": "train#1"})
    assert rs.matches(rs.expression_tree("attribute_not_exists(#k)", {"#k": "id"}, {}), {})
    assert rs.matches(rs.expression_tree(Attr("record_type").eq("detector"), {}, {}), {"record_type": "detector"})


def test_unknown_expressions_and_identifiers_fail():
    with pytest.raises(ValueError):
        rs.expression_tree("id > :a", {}, {":a": "x"})
    with pytest.raises(ValueError):
        rs.Table("unknown; DROP TABLE files")


def test_query_prefix_escapes_wildcards():
    table = rs.Table("track2-yolo-training-files")
    sql, params = table._sql_condition(("begins_with", "file_key", "a_%"))
    assert sql == "sk LIKE %s"
    assert params == ("a\\_\\%%",)


def test_catalog_sync_never_reads_dynamo_in_sql_mode(monkeypatch):
    from ppe_pipeline_api.data import catalog_sync
    monkeypatch.setenv("TRACK2_RECORD_BACKEND", "postgres")
    monkeypatch.setenv("CATALOG_SCHEMA", "track2_dev")
    monkeypatch.setattr(catalog_sync.boto3, "resource", lambda *a, **k: pytest.fail("DynamoDB called"))
    assert catalog_sync.sync()["started"] is False


def test_sql_schedule_status_does_not_reach_production_scheduler(monkeypatch):
    from ppe_pipeline_api import pull_control as pc
    monkeypatch.setenv("TRACK2_RECORD_BACKEND", "postgres")
    monkeypatch.setattr(pc, "read_config", lambda: {"mode": "schedule", "schedule_expression": "rate(7 days)"})
    monkeypatch.setattr(pc, "scheduler", lambda: pytest.fail("Production scheduler called"))
    monkeypatch.setattr(pc, "events", lambda: pytest.fail("Production events called"))
    assert pc.schedule_status()["enabled"]
    assert pc.trigger_status()["schedule"]["enabled"]
