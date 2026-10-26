"""Every `payload.<field>` a route reads must exist on its request model.

Nothing else catches this. Pydantic validates what comes IN; it says
nothing about a handler reading a field the model never declared. The
route imports, the app starts, the OpenAPI schema is correct, and the
failure waits for whoever presses the button:

    'BaseModelRegisterRequest' object has no attribute 'patience'

That one reached a person registering a checkpoint. It was a single
stray argument in a call, and every other signal said the code was fine.

Checked by reading the source rather than by calling the routes, because
calling them needs AWS and the bug is visible without it.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

ROUTERS = sorted((pathlib.Path(__file__).resolve().parents[1]
                  / "src" / "ppe_pipeline_api" / "routers").glob("*.py"))


def _model_fields(tree: ast.Module) -> dict:
    """{class name: set of field names}, following single inheritance
    between models declared in the same module."""
    bases: dict = {}
    fields: dict = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        own = {
            stmt.target.id
            for stmt in node.body
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
        }
        fields[node.name] = own
        bases[node.name] = [b.id for b in node.bases if isinstance(b, ast.Name)]

    def resolved(name: str, seen: frozenset = frozenset()) -> set:
        if name in seen:
            return set()
        out = set(fields.get(name, set()))
        for base in bases.get(name, []):
            out |= resolved(base, seen | {name})
        return out

    return {name: resolved(name) for name in fields}


def _violations(path: pathlib.Path) -> list:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    fields = _model_fields(tree)
    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        # The argument conventionally named `payload`, and the model it
        # is annotated with.
        model = None
        for arg in list(node.args.args) + list(node.args.kwonlyargs):
            if arg.arg == "payload" and isinstance(arg.annotation, ast.Name):
                model = arg.annotation.id
        if model is None or model not in fields:
            continue
        known = fields[model]
        for sub in ast.walk(node):
            if (isinstance(sub, ast.Attribute)
                    and isinstance(sub.value, ast.Name)
                    and sub.value.id == "payload"
                    and not sub.attr.startswith("_")
                    # pydantic's own API, not a declared field.
                    and sub.attr not in {"model_dump", "model_dump_json",
                                         "dict", "json", "copy",
                                         "model_copy", "model_fields"}
                    and sub.attr not in known):
                bad.append(
                    f"{path.name}:{sub.lineno} {node.name}() reads "
                    f"payload.{sub.attr}, which {model} does not declare "
                    f"(it has: {', '.join(sorted(known)) or 'nothing'})"
                )
    return bad


@pytest.mark.parametrize("path", ROUTERS, ids=lambda p: p.name)
def test_routes_only_read_fields_their_request_model_declares(path):
    bad = _violations(path)
    assert not bad, "\n".join(bad)


def test_the_check_would_have_caught_the_real_one(tmp_path):
    """The bug this exists for, reproduced exactly -- otherwise the test
    above passing proves only that it found nothing, not that it looks."""
    sample = tmp_path / "sample.py"
    sample.write_text(
        "from pydantic import BaseModel\n"
        "class BaseModelRegisterRequest(BaseModel):\n"
        "    key: str\n"
        "    detector_slug: str\n"
        "    architecture: str\n"
        "    notes: str = ''\n"
        "def base_model_register(payload: BaseModelRegisterRequest):\n"
        "    return register(key=payload.key, patience=payload.patience)\n",
        encoding="utf-8",
    )
    bad = _violations(sample)
    assert len(bad) == 1
    assert "payload.patience" in bad[0]
    assert "BaseModelRegisterRequest does not declare" in bad[0]


def test_inherited_fields_are_not_reported(tmp_path):
    """A model that extends another has its parent's fields too; flagging
    those would make the check noise and it would be switched off."""
    sample = tmp_path / "sample.py"
    sample.write_text(
        "from pydantic import BaseModel\n"
        "class Base(BaseModel):\n"
        "    shared: str\n"
        "class Child(Base):\n"
        "    own: int\n"
        "def route(payload: Child):\n"
        "    return payload.shared, payload.own\n",
        encoding="utf-8",
    )
    assert _violations(sample) == []


# --- the other half of the same bug ------------------------------------------
#
# `patience=payload.patience` was wrong twice over: the field did not
# exist AND `register_uploaded_base_model` does not take that argument.
# Either alone is a 500 for whoever pressed the button, so both are
# worth checking.

def _shared_call_violations(path: pathlib.Path) -> list:
    """Keyword arguments passed to a `t2.<name>(...)` the function does
    not accept."""
    import inspect
    from ppe_pipeline_api import track2_shared as t2

    tree = ast.parse(path.read_text(encoding="utf-8"))
    bad = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "t2"):
            continue
        target = getattr(t2, node.func.attr, None)
        if not inspect.isfunction(target):
            continue
        sig = inspect.signature(target)
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            continue
        for kw in node.keywords:
            if kw.arg and kw.arg not in sig.parameters:
                bad.append(
                    f"{path.name}:{node.lineno} calls t2.{node.func.attr}() "
                    f"with {kw.arg}=, which it does not accept "
                    f"(it takes: {', '.join(sig.parameters) or 'nothing'})"
                )
    return bad


@pytest.mark.parametrize("path", ROUTERS, ids=lambda p: p.name)
def test_routes_pass_only_arguments_the_function_accepts(path):
    bad = _shared_call_violations(path)
    assert not bad, chr(10).join(bad)
