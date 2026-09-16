"""
SageMaker-compliant inference server: GET /ping and POST /invocations on
port 8080. Shared across every model in this repo (selected via MODEL_TYPE).

Container contract
-------------------
The Dockerfile copies `models/<MODEL_TYPE>/inference.py` to
`/opt/program/model_inference.py`, which must expose a module-level
singleton named `handler` implementing:

    class ModelHandler:
        name: str
        is_ready: bool
        def load(self) -> None: ...
        def predict(self, image: PIL.Image.Image, *, prompt: str | None) -> PIL.Image.Image: ...

`load()` is called once from the FastAPI lifespan handler at container
startup (SageMaker's singleton-load contract); `predict()` runs inside a
threadpool worker per request so the event loop stays free to answer
/ping and accept new connections while a forward pass is in flight.
"""

from __future__ import annotations

import importlib
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from common.schemas import ErrorResponse, InvocationPayload, InvocationResponse
from common.utils import (
    InferenceOOMError,
    InvalidImageError,
    PayloadTooLargeError,
    bytes_to_image,
    decode_base64_image,
    image_to_base64,
    image_to_bytes,
    validate_payload_size,
)

logging.basicConfig(level=os.environ.get("RESTORE_LOG_LEVEL", "INFO").upper())
logger = logging.getLogger("server")

MODEL_TYPE = os.environ.get("MODEL_TYPE", "").strip()
if not MODEL_TYPE:
    raise RuntimeError("MODEL_TYPE environment variable must be set to a models/<name> package.")

# Inside a built container, model_inference.py lives at /opt/program/model_inference.py
# (top-level module, on PYTHONPATH). For local, non-Docker development, fall back to
# importing models.<MODEL_TYPE>.inference directly.
try:
    import model_inference as _model_inference  # type: ignore[import-not-found]
except ModuleNotFoundError:
    _model_inference = importlib.import_module(f"models.{MODEL_TYPE}.inference")

handler = _model_inference.handler


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        handler.load()
        logger.info("Model loaded: %s", handler.name)
    except Exception:
        # Do not crash the process: /ping reports 503 so SageMaker's
        # container health check fails cleanly (and CreateEndpoint surfaces
        # the failure) instead of the process crash-looping.
        logger.exception("Model load failed; /ping will report unhealthy.")
    yield


app = FastAPI(title=f"{MODEL_TYPE}-inference", lifespan=lifespan)


@app.get("/ping")
async def ping() -> Response:
    if getattr(handler, "is_ready", False):
        return Response(status_code=200)
    return Response(status_code=503)


@app.post("/invocations")
async def invocations(request: Request):
    raw = await request.body()
    try:
        validate_payload_size(raw)
    except PayloadTooLargeError as exc:
        return JSONResponse(
            status_code=413,
            content=ErrorResponse(detail=str(exc), error_code="payload_too_large").model_dump(),
        )

    content_type = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
    prompt: Optional[str] = None

    try:
        if content_type == "application/json":
            payload = InvocationPayload.model_validate_json(raw)
            image = decode_base64_image(payload.image)
            prompt = payload.prompt
        elif content_type.startswith("image/"):
            image = bytes_to_image(raw)
            prompt = request.headers.get("x-restoration-prompt")
        else:
            return JSONResponse(
                status_code=415,
                content=ErrorResponse(
                    detail=f"Unsupported Content-Type: {content_type or '<missing>'}",
                    error_code="unsupported_media_type",
                ).model_dump(),
            )
    except InvalidImageError as exc:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(detail=str(exc), error_code="invalid_image").model_dump(),
        )
    except ValueError as exc:  # pydantic validation errors on malformed JSON
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(detail=str(exc), error_code="invalid_payload").model_dump(),
        )

    if not getattr(handler, "is_ready", False):
        return JSONResponse(
            status_code=503,
            content=ErrorResponse(detail="Model is not loaded yet.", error_code="model_not_ready").model_dump(),
        )

    start = time.perf_counter()
    try:
        result_image = await run_in_threadpool(handler.predict, image, prompt=prompt)
    except InferenceOOMError as exc:
        return JSONResponse(
            status_code=503,
            content=ErrorResponse(detail=str(exc), error_code="out_of_memory").model_dump(),
        )
    except Exception:
        logger.exception("Inference failed")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(detail="Inference failed.", error_code="inference_error").model_dump(),
        )
    latency_ms = (time.perf_counter() - start) * 1000

    headers = {"X-Model-Type": handler.name, "X-Inference-Latency-Ms": f"{latency_ms:.2f}"}
    accept = (request.headers.get("accept") or "").lower()

    if "image/" in accept and "application/json" not in accept:
        media_type = "image/png"
        return Response(content=image_to_bytes(result_image, fmt="PNG"), media_type=media_type, headers=headers)

    width, height = result_image.size
    response = InvocationResponse(
        image_base64=image_to_base64(result_image, fmt="PNG"),
        model_type=handler.name,
        prompt=prompt,
        latency_ms=round(latency_ms, 2),
        output_width=width,
        output_height=height,
    )
    return JSONResponse(content=response.model_dump(), headers=headers)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(detail="Internal server error.", error_code="internal_error").model_dump(),
    )
