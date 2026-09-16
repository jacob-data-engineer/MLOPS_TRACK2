"""
Shared, model-agnostic helpers: device management, tensor <-> image
conversion, base64 codec, and OOM-safe execution. Reused unchanged across
every model's serving container (see server.py).
"""

from __future__ import annotations

import base64
import binascii
import io
import os

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

# SageMaker real-time endpoints hard-cap request/response payloads at 6 MB.
SAGEMAKER_MAX_PAYLOAD_BYTES = 6 * 1024 * 1024


class PayloadTooLargeError(ValueError):
    pass


class InvalidImageError(ValueError):
    pass


class InferenceOOMError(RuntimeError):
    """Raised when a CUDA OOM occurs during a forward pass, after cache cleanup."""


def resolve_device() -> torch.device:
    requested = os.environ.get("RESTORE_DEVICE", "auto").lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("RESTORE_DEVICE=cuda was requested but CUDA is not available.")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_dtype(device: torch.device) -> torch.dtype:
    use_fp16 = os.environ.get("RESTORE_USE_FP16", "true").lower() == "true"
    return torch.float16 if (use_fp16 and device.type == "cuda") else torch.float32


def validate_payload_size(raw: bytes, max_bytes: int = SAGEMAKER_MAX_PAYLOAD_BYTES) -> None:
    if len(raw) > max_bytes:
        raise PayloadTooLargeError(
            f"Payload is {len(raw)} bytes, exceeding the {max_bytes}-byte SageMaker real-time limit."
        )


def decode_base64_image(b64_str: str) -> Image.Image:
    try:
        raw = base64.b64decode(b64_str, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidImageError("`image` is not valid base64.") from exc
    return bytes_to_image(raw)


def bytes_to_image(raw: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError("Could not decode image bytes.") from exc
    return image.convert("RGB")


def image_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    return base64.b64encode(image_to_bytes(image, fmt)).decode("ascii")


def image_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()


def pad_to_multiple(tensor: torch.Tensor, multiple: int) -> tuple[torch.Tensor, int, int]:
    """Reflect-pad H/W up to the next multiple so windowed attention / conv
    stages don't choke on arbitrary input resolutions."""
    _, _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h or pad_w:
        tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return tensor, h, w


def preprocess(
    image: Image.Image, device: torch.device, dtype: torch.dtype, window_size: int = 8
) -> tuple[torch.Tensor, int, int]:
    """PIL RGB image -> normalized, padded, batched float tensor on `device`."""
    array = np.asarray(image, dtype=np.float32) / 255.0  # HWC, [0,1]
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)  # 1xCxHxW
    tensor = tensor.to(device=device, dtype=dtype)
    tensor, orig_h, orig_w = pad_to_multiple(tensor, window_size)
    return tensor, orig_h, orig_w


def postprocess(tensor: torch.Tensor, orig_h: int, orig_w: int) -> Image.Image:
    """Model output tensor -> cropped-back, denormalized PIL image."""
    tensor = tensor[:, :, :orig_h, :orig_w]
    tensor = tensor.clamp(0.0, 1.0).float().cpu()
    array = (tensor.squeeze(0).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def run_with_oom_guard(fn, *args, **kwargs):
    """Call `fn`, translating a CUDA OOM into InferenceOOMError after freeing
    the cache, so callers get a clean, catchable exception instead of a
    poisoned CUDA context."""
    try:
        return fn(*args, **kwargs)
    except torch.cuda.OutOfMemoryError as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise InferenceOOMError(
            "GPU ran out of memory while processing this request. Try a smaller image."
        ) from exc


def ensure_local_model_dir(model_dir: str, s3_uri_env: str = "MODEL_S3_URI") -> str:
    """SageMaker's real-time endpoint contract already extracts model.tar.gz
    into `model_dir` (/opt/ml/model) before the container's health check is
    polled, so this is normally a no-op. It exists for local/manual testing
    where a model artifact is pulled straight from S3 (env var `s3_uri_env`)
    into `model_dir` instead of relying on the SageMaker runtime to do it."""
    os.makedirs(model_dir, exist_ok=True)
    if any(os.scandir(model_dir)):
        return model_dir

    s3_uri = os.environ.get(s3_uri_env)
    if not s3_uri:
        return model_dir  # let the caller's own checkpoint lookup raise a clear error

    import tarfile
    import tempfile

    import boto3

    bucket, _, key = s3_uri.removeprefix("s3://").partition("/")
    s3 = boto3.client("s3")

    with tempfile.NamedTemporaryFile(suffix=".tar.gz") as tmp:
        s3.download_file(bucket, key, tmp.name)
        with tarfile.open(tmp.name, "r:gz") as tar:
            tar.extractall(model_dir)  # noqa: S202 -- trusted artifact from our own model registry

    return model_dir
