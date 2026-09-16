"""Serving entrypoint for the SageMaker PyTorch Inference container
(SageMaker's model_fn/input_fn/predict_fn/output_fn contract) — separate
from train.py, since AWS's training and inference base images are
different (see build_and_push.sh, which builds and pushes both).

Output shape ({"detections": [{"confidence": ..., "class": ..., "box":
[...]}]}) is a fixed contract confidence_monitor.py depends on.
"""

import io
import json

from PIL import Image
from ultralytics import YOLO

_MODEL = None


def model_fn(model_dir):
    global _MODEL
    _MODEL = YOLO(f"{model_dir}/model.pt")
    return _MODEL


def input_fn(request_body, content_type):
    if content_type != "image/jpeg":
        raise ValueError(f"Unsupported content type: {content_type}")
    return Image.open(io.BytesIO(request_body)).convert("RGB")


def predict_fn(image, model):
    results = model.predict(image, verbose=False)[0]
    detections = []
    for box in results.boxes:
        detections.append(
            {
                "confidence": float(box.conf[0]),
                "class": int(box.cls[0]),
                "box": [float(v) for v in box.xyxyn[0]],
            }
        )
    return {"detections": detections}


def output_fn(prediction, accept):
    return json.dumps(prediction), "application/json"
