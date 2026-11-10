"""Named SQL-page presets on the same persistent EC2 mount as DP settings."""
import datetime
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from .. import pull_control


class SqlSettings(BaseModel):
    database: str = "violation-review"
    preset_class: str = ""
    preset_scope: Literal["resolved", "ready", "test"] = "resolved"
    pull_bucket: str = "track2-s3"
    pull_folder: str = ""
    train_percent: int = Field(default=80, ge=0, le=100)
    show_labels: bool = True
    with_labels: bool = True
    exploration_hazard: str = "all"
    exploration_open: bool = True
    frames_open: bool = True


def directory() -> Path:
    return Path(pull_control.SETTINGS_DIR) / "sql"


def preset_path(name: str) -> Path:
    name = name.strip()
    if name.lower().endswith("_sql"):
        name = name[:-4]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _-]{0,99}", name):
        raise ValueError("Use a name starting with a letter or number, followed by letters, numbers, spaces, hyphens or underscores (up to 100 characters).")
    return directory() / f"{name}_sql.json"


def list_files() -> dict:
    root = directory()
    files = [{"name": p.stem, "modified": p.stat().st_mtime}
             for p in root.glob("*_sql.json") if p.is_file() and not p.is_symlink()]
    return {"directory": str(root), "files": sorted(files, key=lambda f: f["modified"], reverse=True)}


def save(name: str, settings: SqlSettings) -> dict:
    path = preset_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {"name": path.stem, "saved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "settings": settings.model_dump()}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         suffix=".tmp", delete=False) as handle:
            temporary = handle.name
            json.dump(body, handle, indent=2)
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
    return {**body, "path": str(path)}


def load(name: str) -> dict:
    path = preset_path(name)
    if path.is_symlink():
        raise ValueError("Settings files must not be symbolic links.")
    body = json.loads(path.read_text(encoding="utf-8"))
    return {"name": path.stem, "saved_at": body.get("saved_at", ""),
            "settings": SqlSettings(**body["settings"]).model_dump()}


def delete(name: str) -> dict:
    path = preset_path(name)
    path.unlink()
    return {"name": path.stem, "deleted": True}
