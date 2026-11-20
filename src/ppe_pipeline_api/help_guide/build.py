"""Build the help assistant's application guide FROM THE CODE (2026-11-20).

The chatbot answers only from documentation, and the documentation here is
derived from the frontend source itself -- page routes and their nav names,
the cards each page renders, every card's title, caption and help text,
field labels, options, placeholders and button text. So when a card
changes, rebuilding this changes what the assistant knows; nothing is
written by hand to drift from the screen.

    .venv/Scripts/python.exe -m ppe_pipeline_api.help_guide.build

writes `app_guide.json` beside this file. `tests/test_help_assistant.py`
fails when the checked-in guide no longer matches the source, so it cannot
silently fall behind.
"""
from __future__ import annotations

import html
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[3]
WEB = ROOT / "web" / "src"
OUT = pathlib.Path(__file__).with_name("app_guide.json")

#: Per file, at most this much text: a component is described, not reproduced.
MAX_TEXT = 9000

_ENTITY = {"&apos;": "'", "&rsquo;": "'", "&lsquo;": "'", "&ldquo;": '"', "&rdquo;": '"',
           "&mdash;": " - ", "&ndash;": "-", "&middot;": "·", "&nbsp;": " ", "&amp;": "&",
           "&hellip;": "...", "&rarr;": "->", "&times;": "x"}


def _clean(text: str) -> str:
    for k, v in _ENTITY.items():
        text = text.replace(k, v)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _strip_comments(src: str) -> str:
    # JSX comments and code comments are developer notes, not what users see.
    src = re.sub(r"\{/\*.*?\*/\}", " ", src, flags=re.S)
    src = re.sub(r"/\*.*?\*/", " ", src, flags=re.S)
    return re.sub(r"(?m)^\s*//.*$", " ", src)


def _useful(text: str) -> bool:
    if len(text) < 3 or not re.search(r"[A-Za-z]{2}", text):
        return False
    # Code fragments that leak through the text-node pattern.
    return not re.search(r"(=>|===|!==|&&|\|\||\bconst\b|\breturn\b|\(\)|className|\?\.|\)\s*:|\?\s*\(|^\)|\($)", text)


def describe(path: pathlib.Path) -> dict:
    """Cards, fields and visible text of one component file."""
    src = _strip_comments(path.read_text(encoding="utf-8"))
    cards = []
    for m in re.finditer(r"<Card\b([^>]*?)>", src, flags=re.S):
        attrs = m.group(1)
        title = re.search(r'\btitle=(?:"([^"]+)"|\{"([^"]+)"\})', attrs)
        caption = re.search(r'\bcaption=(?:"([^"]+)"|\{"([^"]+)"\})', attrs)
        if title:
            cards.append({"title": _clean(title.group(1) or title.group(2)),
                          "caption": _clean((caption.group(1) or caption.group(2)) if caption else "")})
    fields = []
    for pattern in (r'aria-label="([^"]+)"', r'placeholder="([^"]+)"', r'\blabel:\s*"([^"]+)"',
                    r'\btitle="([^"]+)"', r'\bhint:\s*"([^"]+)"'):
        fields += [_clean(x) for x in re.findall(pattern, src)]
    texts = [_clean(t) for t in re.findall(r">\s*([^<>{}]+?)\s*<", src)]
    texts += [_clean(t) for t in re.findall(r'\{"([^"{}]{12,})"\}', src)]
    # Labels chosen in code -- `{busy ? "Combining..." : "Combine data sources"}`,
    # `label: "..."` lists -- are string literals that start with a capital
    # and read as words. Class names and keys are lower-case, so they stay out.
    texts += [_clean(t) for t in re.findall(r'"([A-Z][A-Za-z0-9 ,.\'()/&:%?!-]{3,160})"', src)
              if " " in t or t.istitle()]
    seen, ordered = set(), []
    for t in [c["title"] for c in cards] + fields + texts:
        if t and _useful(t) and t not in seen:
            seen.add(t)
            ordered.append(t)
    text = "\n".join(ordered)
    return {"cards": cards, "fields": sorted(set(f for f in fields if _useful(f))),
            "text": text[:MAX_TEXT]}


def _imports(path: pathlib.Path) -> list[str]:
    src = path.read_text(encoding="utf-8")
    names = []
    for spec in re.findall(r'from\s+"(\.{1,2}/[^"]+)"', src):
        target = (path.parent / spec).resolve()
        for candidate in (target.with_suffix(".tsx"), target / "index.tsx"):
            if candidate.exists() and candidate.is_relative_to(WEB):
                names.append(candidate.relative_to(WEB).as_posix())
    return names


def _pages() -> list[dict]:
    app = (WEB / "App.tsx").read_text(encoding="utf-8")
    layout = (WEB / "components" / "Layout.tsx").read_text(encoding="utf-8")
    labels = dict(re.findall(r'\{\s*to:\s*"([^"]+)",\s*label:\s*"([^"]+)"', layout))
    pages = []
    for path, element in re.findall(r'path="([^"]+)"\s*element=\{<(\w+)', app):
        if element in ("Navigate", "LoginPage"):
            continue
        file = next((p for p in (WEB / "routes").glob("*.tsx") if p.stem == element), None)
        if file:
            pages.append({"path": path, "name": labels.get(path, element.replace("Track2", "Track2 ")
                                                            .replace("Page", "").strip()),
                          "file": file.relative_to(WEB).as_posix()})
    # The multi-auto route renders the models page through a multi-line element.
    m = re.search(r'path="(/track2/multi-auto-training)"\s*element=\{<(\w+)', app, flags=re.S)
    if not m and "/track2/multi-auto-training" in app:
        pages.append({"path": "/track2/multi-auto-training",
                      "name": labels.get("/track2/multi-auto-training", "Multi-Auto Training"),
                      "file": "routes/Track2ModelsPage.tsx"})
    return pages


def build() -> dict:
    files = sorted(p for p in list((WEB / "routes").glob("*.tsx")) + list((WEB / "components").glob("*.tsx")))
    graph = {f.relative_to(WEB).as_posix(): _imports(f) for f in files}
    pages = _pages()
    used_by: dict[str, set] = {}
    for page in pages:
        stack, seen = [page["file"]], set()
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            used_by.setdefault(current, set()).add(page["name"])
            stack.extend(graph.get(current, []))
    sections = []
    for f in files:
        rel = f.relative_to(WEB).as_posix()
        info = describe(f)
        if not info["text"]:
            continue
        sections.append({"id": rel, "file": rel, "pages": sorted(used_by.get(rel, [])),
                         "cards": info["cards"], "fields": info["fields"], "text": info["text"]})
    return {"source": "web/src (generated by ppe_pipeline_api.help_guide.build)",
            "pages": pages, "sections": sections}


def main() -> None:
    OUT.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
