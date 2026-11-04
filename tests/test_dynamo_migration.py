import json

import pytest

from ppe_pipeline_api.data import dynamo_migration as dm


def test_preserves_nested_binary_decimal_sets_and_unknown_fields():
    item = {"id": {"S": "r"}, "extra": {"M": {
        "amount": {"N": "12345678901234567890.123456789"},
        "blob": {"B": b"\x00\xff"}, "names": {"SS": ["b", "a"]},
        "values": {"L": [{"NULL": True}, {"BOOL": False}]},
    }}}
    result = json.loads(dm.canonical(item))
    assert result["extra"]["M"]["amount"] == item["extra"]["M"]["amount"]
    assert result["extra"]["M"]["blob"] == {"B": "AP8="}
    assert result["extra"]["M"]["names"] == {"SS": ["a", "b"]}
    assert result["extra"]["M"]["values"] == item["extra"]["M"]["values"]


def test_hash_is_stable_for_dictionary_and_set_order_but_detects_changes():
    a = {"id": {"S": "a"}, "tags": {"SS": ["a", "b"]}}
    b = {"tags": {"SS": ["b", "a"]}, "id": {"S": "a"}}
    source = dm.sources()[0]
    assert dm.record(source, a) == dm.record(source, b)
    b["tags"] = {"SS": ["c"]}
    assert dm.record(source, a)[3] != dm.record(source, b)[3]


def test_composite_keys_do_not_collapse_files():
    source = dm.sources()[1]
    a = {"audit_id": {"S": "a"}, "file_key": {"S": "train#1"}}
    b = {**a, "file_key": {"S": "train#2"}}
    assert dm.record(source, a)[0] != dm.record(source, b)[0]
    with pytest.raises(KeyError):
        dm.record(source, {"audit_id": {"S": "a"}})


class Client:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def scan(self, **kwargs):
        self.calls.append(kwargs)
        result = next(self.responses)
        if isinstance(result, Exception):
            raise result
        return result


def test_scan_follows_empty_pages_and_uses_strong_reads():
    client = Client([{"Items": [], "LastEvaluatedKey": {"id": {"S": "a"}}},
                     {"Items": [{"id": {"S": "b"}}]}])
    assert len(list(dm.pages(client, dm.sources()[0]))) == 2
    assert client.calls[1]["ExclusiveStartKey"] == {"id": {"S": "a"}}
    assert all(c["ConsistentRead"] for c in client.calls)


class Connection:
    def __init__(self, mismatch=0):
        self.statements = []
        self.stored = 0
        self.mismatch = mismatch
        self.rollbacks = 0

    def cursor(self):
        return self

    def execute(self, sql, params=()):
        self.statements.append((sql, params))
        if sql.startswith("INSERT INTO migration_stage.training_audits"):
            self.stored += len(params) // 5

    def fetchone(self):
        return (self.mismatch if "LEFT JOIN" in self.statements[-1][0] else self.stored,)

    def commit(self):
        pass

    def rollback(self):
        self.rollbacks += 1


@pytest.mark.parametrize("mismatch,second,state", [(0, 1, "verified"), (1, 1, "changed"), (0, 0, "changed")])
def test_verification_detects_changes_and_deletions(mismatch, second, state):
    item = {"id": {"S": "a"}}
    client = Client([{"Items": [item]}, {"Items": [item] if second else []}])
    db = Connection(mismatch)
    result = dm.import_source(db, client, dm.sources()[0])
    assert result["state"] == state
    assert result["copied"] == 1
    assert result["checked"] == second


def test_failed_scan_records_failed_batch_and_raises():
    db = Connection()
    with pytest.raises(RuntimeError):
        dm.import_source(db, Client([RuntimeError("unavailable")]), dm.sources()[0])
    assert db.rollbacks == 1
    assert "state='failed'" in db.statements[-1][0]


def test_rejects_arbitrary_sql_identifiers_before_writing():
    db = Connection()
    with pytest.raises(ValueError):
        dm.import_source(db, None, dm.Source("x", "injected", ("id",)))
    assert db.statements == []
