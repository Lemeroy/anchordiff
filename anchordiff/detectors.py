"""Detector adapters for AnchorDiff.

The core never imports a particular detector. This keeps the baseline usable with
hand-authored boxes, classical CV, ONNX Runtime, RKNN or an Ultralytics model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence
import json

import numpy as np

from .core import Detection
from .sanitize import load_json_payload, validate_detection_payload


class Detector(Protocol):
    def detect(self, image: np.ndarray, frame_name: str = "") -> Sequence[Detection]: ...


class JsonDetector:
    """Read detector output from a portable JSON fixture."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.payload = load_json_payload(self.path)
        validate_detection_payload(self.payload)

    def detect(self, image: np.ndarray | None, frame_name: str = "before") -> list[Detection]:
        items = self.payload.get(frame_name, self.payload.get("before", []))
        return [
            Detection(
                str(item["label"]),
                tuple(float(value) for value in item["bbox"]),
                float(item.get("confidence", 1.0)),
            )
            for item in items
        ]


class UltralyticsDetector:
    """Optional YOLO adapter; importing it is deferred until construction."""

    def __init__(self, weights: str, confidence: float = 0.35, *, allow_download: bool = False):
        weight_path = Path(weights)
        if not allow_download and not weight_path.is_file():
            raise FileNotFoundError(
                "local model weights are required by default; pass allow_download=True only in a trusted environment"
            )
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise RuntimeError("install anchordiff[ultralytics] to use this adapter") from exc
        self.model = YOLO(str(weight_path) if weight_path.is_file() else weights)
        self.confidence = confidence

    def detect(self, image: np.ndarray, frame_name: str = "") -> list[Detection]:
        result = self.model.predict(image, conf=self.confidence, verbose=False)[0]
        names = result.names
        detections: list[Detection] = []
        for box, confidence, class_id in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            x0, y0, x1, y1 = [float(value) for value in box.tolist()]
            detections.append(
                Detection(
                    str(names[int(class_id)]),
                    (x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)),
                    float(confidence),
                )
            )
        return detections
