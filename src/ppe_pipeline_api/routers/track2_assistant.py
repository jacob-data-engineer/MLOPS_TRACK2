"""The help assistant's routes (2026-11-20). Every one requires a session.

  GET    /track2/assistant/status        on or off, model, region, documents
  POST   /track2/assistant/chat          one question, answered from the docs
  GET    /track2/assistant/docs          uploaded documentation
  POST   /track2/assistant/docs          upload a .md / .txt
  DELETE /track2/assistant/docs/{name}   remove an uploaded document

Mounted under /track2/assistant, not /track2/help: /track2/help is a page,
and /track2/help/popout a browser route the SPA must keep answering.
"""
from __future__ import annotations

import hashlib

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from ..auth import require_session
from ..help_guide import chat, docs

router = APIRouter()


class ChatRequest(BaseModel):
    messages: list[dict] = Field(default_factory=list, max_length=40)
    context: dict = Field(default_factory=dict)


def _session_key(request: Request) -> str:
    """Who is asking, for the per-session limits -- a hash, never the token."""
    raw = request.headers.get("authorization", "") or request.cookies.get("session", "") or (
        request.client.host if request.client else "")
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


@router.get("/status")
def status(session=Depends(require_session)) -> dict:
    cfg = chat.settings()
    try:
        uploaded = len(docs.list_docs())
    except Exception:                                               # noqa: BLE001
        uploaded = None
    return {"enabled": cfg["enabled"], "model": cfg["model"], "region": cfg["region"],
            "guide_sections": len(docs.guide()["sections"]), "uploaded_documents": uploaded,
            "read_only": True}


@router.post("/chat")
def ask(payload: ChatRequest, request: Request, session=Depends(require_session)) -> dict:
    try:
        return chat.answer(payload.messages, payload.context, session_key=_session_key(request))
    except chat.HelpUnavailable as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc))


@router.get("/docs")
def list_docs(session=Depends(require_session)) -> dict:
    try:
        return {"documents": docs.list_docs(), "max_bytes": docs.MAX_DOC_BYTES,
                "allowed": list(docs.ALLOWED_SUFFIXES)}
    except Exception as exc:                                       # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Could not list the documentation: {exc}")


@router.post("/docs")
async def upload_doc(file: UploadFile, session=Depends(require_session)) -> dict:
    body = await file.read(docs.MAX_DOC_BYTES + 1)
    try:
        return docs.save_doc(file.filename or "", body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.delete("/docs/{name}")
def delete_doc(name: str, session=Depends(require_session)) -> dict:
    try:
        return docs.delete_doc(name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
