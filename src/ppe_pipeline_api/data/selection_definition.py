"""The saved dataset definition and lossless translation of older DP settings."""
from __future__ import annotations

import copy
import hashlib
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .. import schedule_clock, storage


class Source(BaseModel):
    id: str
    name: str = ""
    enabled: bool = True
    bucket: str = ""
    images: str
    labels: str
    kind: Literal["fixed", "collected", "golden"] = "fixed"
    split: Literal["train", "val", "split"] = "train"
    train_percent: int = Field(default=80, ge=1, le=99)
    # Explicit source numbering is only needed when no original YAML exists.
    classes: dict[int, str] = Field(default_factory=dict)
    legacy_split_manifest: str = ""

    @field_validator("id", "images", "labels")
    @classmethod
    def required(cls, value):
        if not value.strip():
            raise ValueError("Source ID, image location and label location are required.")
        return value.strip()


class Schedule(BaseModel):
    mode: Literal["off", "schedule", "arrival"] = "off"
    every_days: int = Field(default=1, ge=1, le=365)
    at: str = "00:00"
    from_date: str = ""

    @field_validator("at")
    @classmethod
    def valid_time(cls, value):
        if not schedule_clock._valid_time(value):
            raise ValueError("Use HH:MM in Singapore time.")
        return value

    @field_validator("from_date")
    @classmethod
    def valid_date(cls, value):
        import datetime
        return datetime.date.fromisoformat(value).isoformat() if value else ""


class Definition(BaseModel):
    version: Literal[2] = 2
    sources: list[Source] = Field(default_factory=list)
    class_ids: dict[int, str] = Field(default_factory=dict)
    filter_to_model: bool = False
    detector_slug: str = ""
    model_classes: dict[int, str] = Field(default_factory=dict)
    schedule: Schedule = Field(default_factory=Schedule)

    @model_validator(mode="after")
    def unique(self):
        ids = [s.id for s in self.sources]
        if len(ids) != len(set(ids)):
            raise ValueError("Each source must have a unique ID.")
        for names in [self.class_ids, *[s.classes for s in self.sources]]:
            if any(k < 0 or k > 10000 or not v.strip() for k, v in names.items()):
                raise ValueError("Class IDs must be between 0 and 10000 and names cannot be blank.")
            if len(set(names.values())) != len(names):
                raise ValueError("Class names must be unique within a mapping.")
        return self


def migrate(config: dict) -> tuple[dict, list[str]]:
    """Return a proposal only. Never save, copy files, or enable a schedule."""
    if config.get("data_selection"):
        return Definition.model_validate(config["data_selection"]).model_dump(), []
    assemble = config.get("dataset_assemble") or {}
    base = str(config.get("source_bucket") or storage.BUCKET)
    rows = []
    notes = []

    def add(images, labels, bucket, kind, split, enabled=True, percent=80):
        identity = "|".join((bucket, images, labels, kind, split))
        rows.append(dict(id=hashlib.sha256(identity.encode()).hexdigest()[:16],
                         name=f"{kind}: {images}", images=images, labels=labels,
                         bucket=bucket, kind=kind, split=split, enabled=enabled,
                         train_percent=percent))

    for split, field in (("train", "train_sources"), ("val", "eval_sources")):
        for row in assemble.get(field) or []:
            if row.get("images"):
                images = row["images"]
                add(images, row.get("labels") or images.replace("/images/", "/labels/"),
                    assemble.get("bucket") or base, "fixed", split, assemble.get("use", True))
    picked = config.get("picked") or {}
    for split in ("train", "val"):
        images, labels = picked.get(split+"_images") or [], picked.get(split+"_labels") or []
        if len(images) != len(labels):
            raise ValueError(f"Pair the {split} image and label folders before migrating.")
        for image, label in zip(images, labels):
            add(image, label, base, "collected", split, not assemble.get("only", False))
    golden = config.get("golden_set") or {}
    policy = config.get("retention") or {}
    use_golden = bool(policy.get("use_golden"))
    if golden.get("source"):
        images = golden["source"]
        labels = golden.get("labels_source") or images.replace("/images", "/labels")
        add(images, labels, golden.get("bucket") or base, "golden",
            "split" if golden.get("split", True) else "train", use_golden,
            min(99, max(1, int(golden.get("train_percent") or 80))))
        if golden.get("target_folder") and golden.get("split", True):
            target = golden["target_folder"].strip("/")
            rows[-1]["legacy_split_manifest"] = target+"/split.json" if target.startswith("s3://") else "s3://"+(golden.get("target_bucket") or golden.get("bucket") or base)+"/"+target+"/split.json"
        notes.append("Golden-set membership will use the original source; no golden destination folder is required.")
    elif policy.get("golden_root"):
        root = str(policy["golden_root"]).strip("/")
        for split in ("train", "val"):
            add(root+"/images/"+split, root+"/labels/"+split,
                policy.get("golden_bucket") or base, "golden", split, use_golden)
    elif use_golden:
        raise ValueError("Golden-set inclusion is enabled but no original source or existing golden dataset is configured.")
    notes += ["Existing source choices, label mapping, retention and excluded weeks are preserved.",
              "Source images remain in S3. Collections and train/validation membership are recorded in SQL.",
              "Save stores this definition and its refresh schedule together. Training stays on Multi-Auto Training."]
    result = Definition(sources=rows, class_ids=assemble.get("label_classes") or {},
                        filter_to_model=bool(assemble.get("drop_undeclared_labels")),
                        detector_slug=assemble.get("detector_slug") or (config.get("default_detector") or {}).get("slug", ""),
                        schedule=Schedule(mode=config.get("mode") if config.get("mode") in ("schedule", "arrival") else "off",
                                          every_days=int(config.get("schedule_every_days") or 1),
                                          at=config.get("schedule_time") or "00:00",
                                          from_date=config.get("schedule_start_date") or ""))
    return result.model_dump(), notes


def effective(config):
    return migrate(config)[0]


def migrated_config(config):
    result = copy.deepcopy(config)
    result["data_selection"], _ = migrate(config)
    return result
