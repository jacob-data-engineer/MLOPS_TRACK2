"""What the help assistant may answer from: the generated application guide
plus documentation an operator uploads on the Help page (2026-11-20).

Uploaded documents live in the app's bucket under `_help/docs/`, as plain
text or Markdown. They are DATA for the assistant -- never instructions --
and the chat prompt says so. Retrieval is deliberately simple and local:
the current page's and the asked card's sections always go in, then the
sections that share the most words with the question, inside a fixed
character budget. Nothing is embedded or sent anywhere to be indexed.
"""
from __future__ import annotations

import datetime as _dt
import json
import pathlib
import re
import time

from .. import storage

GUIDE = pathlib.Path(__file__).with_name("app_guide.json")
#: Where uploaded documentation lives in the app's bucket.
DOCS_PREFIX = "_help/docs/"
#: Largest uploaded document, and the most kept.
MAX_DOC_BYTES = 200_000
MAX_DOCS = 50
ALLOWED_SUFFIXES = (".md", ".txt")
#: How much documentation one question may carry to the model.
CONTEXT_BUDGET = 24_000

_cache: dict = {}


def guide() -> dict:
    if "guide" not in _cache:
        _cache["guide"] = json.loads(GUIDE.read_text(encoding="utf-8"))
    return _cache["guide"]


# --- uploaded documentation -------------------------------------------------------

def _safe_name(name: str) -> str:
    base = pathlib.PurePosixPath(str(name or "").replace("\\", "/")).name
    if not base.lower().endswith(ALLOWED_SUFFIXES):
        raise ValueError("Upload a .md or .txt file.")
    stem = re.sub(r"[^A-Za-z0-9 ._-]", "", base[: base.rfind(".")]).strip(" .")
    if not stem:
        raise ValueError("Give the file a name made of letters, numbers, spaces, dots, hyphens or underscores.")
    return f"{stem[:80]}{base[base.rfind('.'):].lower()}"


def list_docs() -> list[dict]:
    out = []
    resp = storage.s3.list_objects_v2(Bucket=storage.BUCKET, Prefix=DOCS_PREFIX)
    for obj in resp.get("Contents", []) or []:
        name = obj["Key"][len(DOCS_PREFIX):]
        if name and "/" not in name:
            modified = obj.get("LastModified")
            out.append({"name": name, "bytes": obj.get("Size", 0),
                        "updated_at": modified.isoformat() if hasattr(modified, "isoformat") else str(modified or "")})
    return sorted(out, key=lambda d: d["name"].lower())


def save_doc(name: str, body: bytes) -> dict:
    name = _safe_name(name)
    if len(body) > MAX_DOC_BYTES:
        raise ValueError(f"That file is {len(body):,} bytes; the limit is {MAX_DOC_BYTES:,}.")
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("The file is not UTF-8 text.") from None
    if not text.strip():
        raise ValueError("The file is empty.")
    existing = {d["name"] for d in list_docs()}
    if name not in existing and len(existing) >= MAX_DOCS:
        raise ValueError(f"There are already {MAX_DOCS} documents; delete one first.")
    storage.s3.put_object(Bucket=storage.BUCKET, Key=DOCS_PREFIX + name, Body=text.encode("utf-8"),
                          ContentType="text/markdown" if name.endswith(".md") else "text/plain")
    _cache.pop("docs", None)
    return {"name": name, "bytes": len(body), "replaced": name in existing}


def delete_doc(name: str) -> dict:
    name = _safe_name(name)
    if name not in {d["name"] for d in list_docs()}:
        raise ValueError(f"There is no document called {name}.")
    storage.s3.delete_object(Bucket=storage.BUCKET, Key=DOCS_PREFIX + name)
    _cache.pop("docs", None)
    return {"name": name, "deleted": True}


def _uploaded_sections() -> list[dict]:
    """Uploaded documents split at headings, cached for a minute."""
    hit = _cache.get("docs")
    if hit and time.monotonic() - hit[0] < 60:
        return hit[1]
    sections = []
    for doc in list_docs():
        try:
            body = storage.s3.get_object(Bucket=storage.BUCKET, Key=DOCS_PREFIX + doc["name"])["Body"].read()
            text = body.decode("utf-8", errors="replace")
        except Exception:                                           # noqa: BLE001
            continue
        parts = re.split(r"(?m)^(?=#{1,3} )", text)
        for i, part in enumerate(p for p in parts if p.strip()):
            heading = part.strip().splitlines()[0].lstrip("# ").strip()
            sections.append({"id": f"upload:{doc['name']}#{i}", "file": doc["name"], "pages": [],
                             "cards": [], "fields": [], "title": heading[:120],
                             "text": part.strip()[:6000], "uploaded": True})
    _cache["docs"] = (time.monotonic(), sections)
    return sections


# --- retrieval --------------------------------------------------------------------

def _words(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]{3,}", text.lower())
            if w not in {"the", "and", "for", "this", "that", "with", "what", "does", "how", "are", "you"}}


def select(question: str, *, page: str = "", card: str = "", budget: int = CONTEXT_BUDGET,
           include_uploads: bool = True) -> list[dict]:
    """The documentation sections for one question, most relevant first."""
    g = guide()
    sections = list(g["sections"]) + (_uploaded_sections() if include_uploads else [])
    want = _words(question + " " + card)
    card_l = card.strip().lower()

    def score(s):
        title_hit = card_l and any(c["title"].lower() == card_l for c in s["cards"])
        on_page = page and page in s["pages"]
        overlap = len(want & _words(s["text"][:4000] + " " + s.get("title", "")))
        return (2 if title_hit else 0, overlap + (3 if on_page else 0))

    chosen, used = [], 0
    for s in sorted(sections, key=score, reverse=True):
        if score(s) == (0, 0):
            break
        size = len(s["text"]) + 200
        if used + size > budget:
            continue
        chosen.append(s)
        used += size
    return chosen


def pages_summary() -> str:
    return "\n".join(f"- {p['name']} ({p['path']})" for p in guide()["pages"])


def stamp() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
