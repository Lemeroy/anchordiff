"""Optional detector adapters with one small, model-agnostic contract.

The core only consumes :class:`~anchordiff.core.Detection` values. Adapters in
this module defer optional imports until construction, keep model files local by
default, and normalize boxes and full-frame masks so SCA remains deterministic.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image

from .core import Detection
from .sanitize import load_json_payload, validate_detection_payload


class Detector(Protocol):
    """Detector boundary shared by JSON, Ultralytics, ONNX and RKNN backends."""

    def detect(self, image: np.ndarray, frame_name: str = "") -> Sequence[Detection]: ...


def _coerce_mask(mask: Any, image_shape: tuple[int, int]) -> np.ndarray:
    """Convert a backend mask to a bounded full-frame boolean mask."""

    arr = np.asarray(mask)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("detector mask must be a non-empty 2D array")
    height, width = image_shape
    if arr.shape != (height, width):
        image = Image.fromarray((arr > 0).astype(np.uint8) * 255, mode="L")
        image = image.resize((width, height), Image.Resampling.NEAREST)
        arr = np.asarray(image) > 0
    return np.asarray(arr, dtype=bool)


def _normalize_detection(item: Detection | Mapping[str, Any], image_shape: tuple[int, int]) -> Detection:
    if isinstance(item, Detection):
        if item.mask is None:
            return item
        return Detection(item.label, item.bbox, item.confidence, _coerce_mask(item.mask, image_shape))
    label = str(item["label"])
    bbox = tuple(float(value) for value in item["bbox"])
    confidence = float(item.get("confidence", 1.0))
    mask = item.get("mask")
    return Detection(label, bbox, confidence, _coerce_mask(mask, image_shape) if mask is not None else None)


class JsonDetector:
    """Read detector output from a portable JSON fixture."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.payload = load_json_payload(self.path)
        validate_detection_payload(self.payload)

    def detect(self, image: np.ndarray | None, frame_name: str = "before") -> list[Detection]:
        items = self.payload.get(frame_name, self.payload.get("before", []))
        if image is None:
            if any(item.get("mask") is not None for item in items):
                raise ValueError("an image is required to decode detector masks")
            return [
                Detection(
                    str(item["label"]),
                    tuple(float(value) for value in item["bbox"]),
                    float(item.get("confidence", 1.0)),
                )
                for item in items
            ]
        shape = tuple(np.asarray(image).shape[:2])
        for item in items:
            if item.get("mask") is not None and np.asarray(item["mask"]).shape != shape:
                raise ValueError("JSON detector mask must match the image dimensions")
        return [_normalize_detection(item, shape) for item in items]


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
        mask_data = getattr(getattr(result, "masks", None), "data", None)
        if mask_data is not None and hasattr(mask_data, "cpu"):
            mask_data = mask_data.cpu().numpy()
        detections: list[Detection] = []
        for index, (box, confidence, class_id) in enumerate(
            zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls)
        ):
            x0, y0, x1, y1 = [float(value) for value in box.tolist()]
            mask = None
            if mask_data is not None and index < len(mask_data):
                mask = _coerce_mask(mask_data[index], tuple(np.asarray(image).shape[:2]))
            detections.append(
                Detection(
                    str(names[int(class_id)]),
                    (x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)),
                    float(confidence),
                    mask,
                )
            )
        return detections


def _default_decode(outputs: Sequence[Any], image_shape: tuple[int, int], class_names: Sequence[str], confidence: float) -> list[Detection]:
    """Decode common ``[x0,y0,x1,y1,score,class]`` output into detections."""

    if not outputs:
        return []
    rows = np.asarray(outputs[0])
    rows = np.squeeze(rows)
    if rows.size == 0:
        return []
    if rows.ndim == 1:
        rows = rows[None, :]
    if rows.ndim != 2 or rows.shape[1] < 6:
        raise ValueError("detector output needs rows with x0,y0,x1,y1,score,class")
    detections: list[Detection] = []
    height, width = image_shape
    for row in rows:
        x0, y0, x1, y1, score, class_id = [float(value) for value in row[:6]]
        if score < confidence:
            continue
        class_index = int(class_id)
        label = class_names[class_index] if 0 <= class_index < len(class_names) else str(class_index)
        if max(abs(x0), abs(y0), abs(x1), abs(y1)) <= 1.5:
            x0, x1 = x0 * width, x1 * width
            y0, y1 = y0 * height, y1 * height
        detections.append(Detection(label, (x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)), score))
    return detections


class ONNXDetector:
    """ONNX Runtime adapter with an injectable decoder for model-specific heads."""

    def __init__(
        self,
        weights: str | Path,
        class_names: Sequence[str] = (),
        confidence: float = 0.35,
        *,
        providers: Sequence[str] | None = None,
        input_size: tuple[int, int] = (640, 640),
        decoder: Callable[[Sequence[Any], tuple[int, int]], Sequence[Detection | Mapping[str, Any]]] | None = None,
        session: Any | None = None,
        allow_download: bool = False,
    ):
        path = Path(weights)
        if session is None and not allow_download and not path.is_file():
            raise FileNotFoundError("local ONNX model weights are required by default")
        if session is None:
            try:
                import onnxruntime as ort
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("install anchordiff[onnx] to use this adapter (onnxruntime)") from exc
            session = ort.InferenceSession(str(path), providers=list(providers) if providers else None)
        self.session = session
        self.class_names = tuple(class_names)
        self.confidence = float(confidence)
        self.input_size = input_size
        self.decoder = decoder

    def detect(self, image: np.ndarray, frame_name: str = "") -> list[Detection]:
        arr = np.asarray(image)
        original_shape = tuple(arr.shape[:2])
        resized = np.asarray(Image.fromarray(arr.astype(np.uint8)).resize(self.input_size[::-1], Image.Resampling.BILINEAR), dtype=np.float32)
        tensor = np.transpose(resized / 255.0, (2, 0, 1))[None, ...]
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: tensor})
        decoded = self.decoder(outputs, original_shape) if self.decoder else _default_decode(outputs, original_shape, self.class_names, self.confidence)
        return [_normalize_detection(item, original_shape) for item in decoded]


class RKNNDetector:
    """RKNN Lite adapter with the same output and decoder contract as ONNX."""

    def __init__(
        self,
        weights: str | Path,
        class_names: Sequence[str] = (),
        confidence: float = 0.35,
        *,
        input_size: tuple[int, int] = (640, 640),
        decoder: Callable[[Sequence[Any], tuple[int, int]], Sequence[Detection | Mapping[str, Any]]] | None = None,
        runtime: Any | None = None,
    ):
        path = Path(weights)
        if runtime is None and not path.is_file():
            raise FileNotFoundError("local RKNN model weights are required by default")
        if runtime is None:
            try:
                from rknnlite.api import RKNN
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("install anchordiff[rknn] to use this adapter (rknnlite)") from exc
            runtime = RKNN(verbose=False)
            if runtime.load_rknn(str(path)) != 0 or runtime.init_runtime() != 0:
                raise RuntimeError("RKNN model failed to load or initialize")
        self.runtime = runtime
        self.class_names = tuple(class_names)
        self.confidence = float(confidence)
        self.input_size = input_size
        self.decoder = decoder

    def detect(self, image: np.ndarray, frame_name: str = "") -> list[Detection]:
        arr = np.asarray(image)
        original_shape = tuple(arr.shape[:2])
        resized = np.asarray(Image.fromarray(arr.astype(np.uint8)).resize(self.input_size[::-1], Image.Resampling.BILINEAR), dtype=np.uint8)
        outputs = self.runtime.inference(inputs=[resized])
        decoded = self.decoder(outputs, original_shape) if self.decoder else _default_decode(outputs, original_shape, self.class_names, self.confidence)
        return [_normalize_detection(item, original_shape) for item in decoded]

    def close(self) -> None:
        release = getattr(self.runtime, "release", None)
        if callable(release):
            release()
