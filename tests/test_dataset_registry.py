from ppe_pipeline_api.dataset_registry import decorate_dataset_keys


def test_decorate_dataset_keys_puts_named_gold_sets_first_v1_then_v2():
    keys = ["eval/zzz_other.json", "eval/gold_set_v2.json", "eval/gold_set_v1.json", "eval/aaa_other.json"]

    decorated = decorate_dataset_keys(keys)

    assert decorated == [
        {"key": "eval/gold_set_v1.json", "label": "Gold Set v1", "is_named": True},
        {"key": "eval/gold_set_v2.json", "label": "Gold Set v2", "is_named": True},
        {"key": "eval/aaa_other.json", "label": "aaa_other.json", "is_named": False},
        {"key": "eval/zzz_other.json", "label": "zzz_other.json", "is_named": False},
    ]


def test_decorate_dataset_keys_with_no_named_gold_sets():
    keys = ["eval/b.json", "eval/a.json"]

    decorated = decorate_dataset_keys(keys)

    assert decorated == [
        {"key": "eval/a.json", "label": "a.json", "is_named": False},
        {"key": "eval/b.json", "label": "b.json", "is_named": False},
    ]


def test_decorate_dataset_keys_empty_list():
    assert decorate_dataset_keys([]) == []
