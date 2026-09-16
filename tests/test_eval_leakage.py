"""Tests for the SHA-256 few-shot hygiene checks in `eval.leakage`.

Every case here writes real bytes to a tmp LocalStorage root, because the
whole point of the module is that it matches on CONTENT rather than on paths
— a test that stubbed the hashing would not be testing anything.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ppe_pipeline import storage
from ppe_pipeline.eval import leakage
from ppe_pipeline.eval.dataset import EvalCase
from ppe_pipeline.models import BoundingBox, ViolationType


@pytest.fixture(autouse=True)
def local_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "get_storage", lambda: storage.LocalStorage(tmp_path))
    # The hash memo is process-wide and keyed by storage key; two tests using
    # the same relative key under different tmp roots would otherwise see each
    # other's digests.
    leakage._hash_cache.clear()
    yield tmp_path
    leakage._hash_cache.clear()


def write_image(root: Path, key: str, content: bytes) -> str:
    path = root / key
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return key


def make_case(case_id: str, image_key: str, violation=ViolationType.HELMET, is_violation=True) -> EvalCase:
    return EvalCase(
        case_id=case_id,
        image_path=Path(image_key),
        violation_type=violation,
        person_box=BoundingBox(0, 0, 10, 10),
        ground_truth_is_violation=is_violation,
    )


def write_manifest(root: Path, entries_by_type: dict) -> Path:
    manifest_path = Path("few_shot/manifest.json")
    (root / manifest_path).parent.mkdir(parents=True, exist_ok=True)
    (root / manifest_path).write_text(json.dumps(entries_by_type), encoding="utf-8")
    return manifest_path


def fewshot_entry(image: str) -> dict:
    return {"image": image, "verdict": "TP", "reason": "bare_head",
            "confidence": "high", "explanation": "x"}


# --------------------------------------------------------------------------
# exclude_few_shot_leakage
# --------------------------------------------------------------------------


def test_case_sharing_bytes_with_a_few_shot_example_is_excluded(local_storage):
    root = local_storage
    write_image(root, "few_shot/crops/example.jpg", b"identical-bytes")
    # Same content, completely different path and filename -- the case a
    # filename check would miss.
    write_image(root, "resources/gold/unrelated-name.jpg", b"identical-bytes")
    write_image(root, "resources/gold/clean.jpg", b"different-bytes")
    manifest = write_manifest(root, {"helmet": [fewshot_entry("crops/example.jpg")]})

    dataset = [
        make_case("leaked", "resources/gold/unrelated-name.jpg"),
        make_case("clean", "resources/gold/clean.jpg"),
    ]
    kept, warnings, collisions = leakage.exclude_few_shot_leakage(dataset, manifest)

    assert [c.case_id for c in kept] == ["clean"]
    assert len(collisions) == 1
    assert collisions[0]["case_id"] == "leaked"
    assert collisions[0]["few_shot_image"] == "few_shot/crops/example.jpg"
    assert "excluded 1 case" in warnings[0]


def test_exclusion_does_not_touch_anything_on_disk(local_storage):
    """The whole contract of this check: it filters ONE RUN, never the gold
    set. A case is only leaked relative to a particular manifest."""
    root = local_storage
    write_image(root, "few_shot/crops/example.jpg", b"same")
    write_image(root, "resources/gold/a.jpg", b"same")
    manifest = write_manifest(root, {"helmet": [fewshot_entry("crops/example.jpg")]})
    dataset_file = root / "eval" / "cases.json"
    dataset_file.parent.mkdir(parents=True, exist_ok=True)
    dataset_file.write_text(json.dumps([{"case_id": "a"}]), encoding="utf-8")
    before = dataset_file.read_text(encoding="utf-8")

    leakage.exclude_few_shot_leakage([make_case("a", "resources/gold/a.jpg")], manifest)

    assert dataset_file.read_text(encoding="utf-8") == before
    assert (root / "resources/gold/a.jpg").read_bytes() == b"same"


def test_unreadable_case_image_is_kept_not_dropped(local_storage):
    """A missing crop is the eval runner's error to report clearly; a hygiene
    check must not silently delete cases it merely failed to read."""
    root = local_storage
    write_image(root, "few_shot/crops/example.jpg", b"same")
    manifest = write_manifest(root, {"helmet": [fewshot_entry("crops/example.jpg")]})

    kept, warnings, collisions = leakage.exclude_few_shot_leakage(
        [make_case("ghost", "resources/gold/does-not-exist.jpg")], manifest
    )

    assert [c.case_id for c in kept] == ["ghost"]
    assert collisions == []


def test_empty_manifest_changes_nothing(local_storage):
    root = local_storage
    write_image(root, "resources/gold/a.jpg", b"a")
    manifest = write_manifest(root, {})
    dataset = [make_case("a", "resources/gold/a.jpg")]

    kept, warnings, collisions = leakage.exclude_few_shot_leakage(dataset, manifest)

    assert kept == dataset
    assert warnings == []


def test_exclusion_applies_across_violation_types(local_storage):
    """A crop filed under 'glove' in the library still leaks a 'helmet' case
    that shares its bytes -- the judge saw the picture either way."""
    root = local_storage
    write_image(root, "few_shot/crops/g.jpg", b"shared")
    write_image(root, "resources/gold/h.jpg", b"shared")
    manifest = write_manifest(root, {"glove": [fewshot_entry("crops/g.jpg")]})

    kept, _warnings, collisions = leakage.exclude_few_shot_leakage(
        [make_case("h", "resources/gold/h.jpg", violation=ViolationType.HELMET)], manifest
    )

    assert kept == []
    assert len(collisions) == 1


# --------------------------------------------------------------------------
# deduplicate_few_shot_manifest
# --------------------------------------------------------------------------


def test_duplicate_images_are_removed_keeping_the_first(local_storage):
    root = local_storage
    write_image(root, "few_shot/crops/first.jpg", b"dupe")
    write_image(root, "few_shot/crops/second.jpg", b"dupe")
    write_image(root, "few_shot/crops/other.jpg", b"unique")
    manifest = write_manifest(root, {
        "helmet": [
            fewshot_entry("crops/first.jpg"),
            fewshot_entry("crops/second.jpg"),
            fewshot_entry("crops/other.jpg"),
        ]
    })

    removed, warnings = leakage.deduplicate_few_shot_manifest(manifest)

    assert [item["image"] for item in removed] == ["crops/second.jpg"]
    assert removed[0]["kept_image"] == "crops/first.jpg"
    written = json.loads((root / manifest).read_text(encoding="utf-8"))
    assert [e["image"] for e in written["helmet"]] == ["crops/first.jpg", "crops/other.jpg"]


def test_dedupe_spans_violation_types(local_storage):
    root = local_storage
    write_image(root, "few_shot/crops/a.jpg", b"same-picture")
    write_image(root, "few_shot/crops/b.jpg", b"same-picture")
    manifest = write_manifest(root, {
        "helmet": [fewshot_entry("crops/a.jpg")],
        "glove": [fewshot_entry("crops/b.jpg")],
    })

    removed, _warnings = leakage.deduplicate_few_shot_manifest(manifest)

    written = json.loads((root / manifest).read_text(encoding="utf-8"))
    assert len(removed) == 1
    assert written["helmet"] and written["glove"] == []


def test_dedupe_backs_up_the_original_manifest_once(local_storage):
    root = local_storage
    write_image(root, "few_shot/crops/a.jpg", b"same")
    write_image(root, "few_shot/crops/b.jpg", b"same")
    write_image(root, "few_shot/crops/c.jpg", b"same")
    manifest = write_manifest(root, {
        "helmet": [fewshot_entry("crops/a.jpg"), fewshot_entry("crops/b.jpg"),
                   fewshot_entry("crops/c.jpg")]
    })

    leakage.deduplicate_few_shot_manifest(manifest)
    backup = root / "few_shot/backups/manifest.pre_dedupe.json"
    first_backup = backup.read_text(encoding="utf-8")
    assert len(json.loads(first_backup)["helmet"]) == 3

    # A second pass must not overwrite the only copy of the original.
    write_image(root, "few_shot/crops/d.jpg", b"same")
    manifest_data = json.loads((root / manifest).read_text(encoding="utf-8"))
    manifest_data["helmet"].append(fewshot_entry("crops/d.jpg"))
    (root / manifest).write_text(json.dumps(manifest_data), encoding="utf-8")
    leakage.deduplicate_few_shot_manifest(manifest)
    assert backup.read_text(encoding="utf-8") == first_backup


def test_dedupe_dry_run_writes_nothing(local_storage):
    root = local_storage
    write_image(root, "few_shot/crops/a.jpg", b"same")
    write_image(root, "few_shot/crops/b.jpg", b"same")
    manifest = write_manifest(root, {
        "helmet": [fewshot_entry("crops/a.jpg"), fewshot_entry("crops/b.jpg")]
    })
    before = (root / manifest).read_text(encoding="utf-8")

    removed, warnings = leakage.deduplicate_few_shot_manifest(manifest, write=False)

    assert len(removed) == 1
    assert "would remove" in warnings[0]
    assert (root / manifest).read_text(encoding="utf-8") == before
    assert not (root / "few_shot/backups/manifest.pre_dedupe.json").exists()


def test_dedupe_prunes_the_annotations_sidecar(local_storage):
    root = local_storage
    write_image(root, "few_shot/crops/a.jpg", b"same")
    write_image(root, "few_shot/crops/b.jpg", b"same")
    manifest = write_manifest(root, {
        "helmet": [fewshot_entry("crops/a.jpg"), fewshot_entry("crops/b.jpg")]
    })
    annotations = Path("few_shot/annotations.json")
    (root / annotations).write_text(
        json.dumps({"a.jpg": {"reason": "x"}, "b.jpg": {"reason": "y"}}), encoding="utf-8"
    )

    leakage.deduplicate_few_shot_manifest(manifest, annotations)

    written = json.loads((root / annotations).read_text(encoding="utf-8"))
    assert "a.jpg" in written and "b.jpg" not in written


def test_dedupe_keeps_a_sidecar_row_another_entry_still_uses(local_storage):
    """Two directories can hold the same basename. Dropping the sidecar row
    for a filename a surviving entry still needs would strip that example's
    reason and explanation."""
    root = local_storage
    write_image(root, "few_shot/crops/shared.jpg", b"same")
    write_image(root, "few_shot/other/shared.jpg", b"same")
    manifest = write_manifest(root, {
        "helmet": [fewshot_entry("crops/shared.jpg"), fewshot_entry("other/shared.jpg")]
    })
    annotations = Path("few_shot/annotations.json")
    (root / annotations).write_text(json.dumps({"shared.jpg": {"reason": "x"}}), encoding="utf-8")

    removed, _warnings = leakage.deduplicate_few_shot_manifest(manifest, annotations)

    assert len(removed) == 1
    written = json.loads((root / annotations).read_text(encoding="utf-8"))
    assert "shared.jpg" in written


def test_dedupe_leaves_a_clean_library_untouched(local_storage):
    root = local_storage
    write_image(root, "few_shot/crops/a.jpg", b"one")
    write_image(root, "few_shot/crops/b.jpg", b"two")
    manifest = write_manifest(root, {
        "helmet": [fewshot_entry("crops/a.jpg"), fewshot_entry("crops/b.jpg")]
    })
    before = (root / manifest).read_text(encoding="utf-8")

    removed, warnings = leakage.deduplicate_few_shot_manifest(manifest)

    assert removed == [] and warnings == []
    assert (root / manifest).read_text(encoding="utf-8") == before
    assert not (root / "few_shot/backups/manifest.pre_dedupe.json").exists()


def test_promoted_case_is_caught_by_case_id_when_bytes_differ(local_storage):
    """The leak content hashing structurally cannot see: promotion CROPS the
    frame into few_shot/crops/{case_id}.jpg, so the stored crop is a
    different, smaller file than the gold set's full frame."""
    root = local_storage
    write_image(root, "few_shot/crops/helmet_042.jpg", b"a-small-crop")
    write_image(root, "resources/gold/frame.jpg", b"a-much-larger-full-frame")
    manifest = write_manifest(root, {"helmet": [fewshot_entry("crops/helmet_042.jpg")]})

    kept, warnings, collisions = leakage.exclude_few_shot_leakage(
        [make_case("helmet_042", "resources/gold/frame.jpg"),
         make_case("helmet_999", "resources/gold/frame.jpg")],
        manifest,
    )

    assert [c.case_id for c in kept] == ["helmet_999"]
    assert collisions[0]["match"] == "case_id"
    assert collisions[0]["few_shot_image"] == "few_shot/crops/helmet_042.jpg"
    assert "1 by case_id" in warnings[0]


def test_collision_records_which_signal_fired(local_storage):
    root = local_storage
    write_image(root, "few_shot/crops/byte_twin.jpg", b"identical")
    write_image(root, "few_shot/crops/named_case.jpg", b"a-crop")
    write_image(root, "resources/gold/twin.jpg", b"identical")
    write_image(root, "resources/gold/other.jpg", b"a-full-frame")
    manifest = write_manifest(root, {
        "helmet": [fewshot_entry("crops/byte_twin.jpg"), fewshot_entry("crops/named_case.jpg")]
    })

    _kept, warnings, collisions = leakage.exclude_few_shot_leakage(
        [make_case("byte_leak", "resources/gold/twin.jpg"),
         make_case("named_case", "resources/gold/other.jpg")],
        manifest,
    )

    by_case = {c["case_id"]: c["match"] for c in collisions}
    assert by_case == {"byte_leak": "sha256", "named_case": "case_id"}
    assert "1 by sha256" in warnings[0] and "1 by case_id" in warnings[0]
