"""Pydantic payload models for the model-serving container's /invocations route."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class InvocationPayload(BaseModel):
    """JSON body accepted by POST /invocations."""

    image: str = Field(..., description="Raw image bytes, base64-encoded (no data-URI prefix required).")
    prompt: Optional[str] = Field(
        default=None,
        max_length=512,
        description="Natural-language restoration instruction, e.g. 'remove the noise and brighten the image'.",
    )


class InvocationResponse(BaseModel):
    image_base64: str
    model_type: str
    prompt: Optional[str] = None
    latency_ms: float
    output_width: int
    output_height: int


class ErrorResponse(BaseModel):
    detail: str
    error_code: str


class PingResponse(BaseModel):
    status: Literal["ok"] = "ok"
    model_type: str
    device: str
