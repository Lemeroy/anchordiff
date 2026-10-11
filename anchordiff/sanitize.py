"""Input and publication-safety helpers.

AnchorDiff never needs camera EXIF, filenames, user identifiers or arbitrary
metadata. These helpers create plain RGB arrays and validate small, bounded
machine-readable inputs before they enter the analysis pipeline.
"""

from __future__ import annotations

import json
from math import isfinite
from pathlib import Path
import re
from typing import Any

import numpy as np
from PIL import Image, ImageOps


MAX_IMAGE_PIXELS = 25_000_000
MAX_JSON_BYTES = 5 * 1024 * 1024
MAX_DETECTIONS = 512
MAX_LABEL_LENGTH = 64
MAX_MASK_SIDE = 4096
_LABEL_PATTERN = re.compile(r"^[\w .:/+\-]+$", re.UNICODE)


def load_sanitized_rgb(path: str | Path) -> np.ndarray:
    """Decode an image and return a metadata-free RGB array.

    EXIF orientation is applied for visual correctness and then discarded. No
    source path, EXIF field, ICC profile or comment is copied to the result.
    """

    source = Path(path)
    with Image.open(source) as image:
        width, height = image.size
        if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
            raise ValueError(f"image exceeds maximum size of {MAX_IMAGE_PIXELS} pixels")
        image.load()
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        return np.asarray(rgb, dtype=np.uint8).copy()


def load_json_payload(path: str | Path) -> dict[str, Any]:
    """Load bounded JSON while rejecting non-standard NaN/Infinity constants."""

    source = Path(path)
    if source.stat().st_size > MAX_JSON_BYTES:
        raise ValueError(f"JSON input exceeds maximum size of {MAX_JSON_BYTES} bytes")
    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"non-finite JSON value: {value}")),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid UTF-8 JSON input") from exc
    if not isinstance(payload, dict):
        raise ValueError("JSON root must be an object")
    return payload


def _finite_number(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def validate_detection_payload(payload: dict[str, Any]) -> None:
    """Validate detector JSON without trusting labels, counts or coordinates."""

    for frame in ("before", "after"):
        items = payload.get(frame, [])
        if not isinstance(items, list) or len(items) > MAX_DETECTIONS:
            raise ValueError(f"{frame} detections exceed maximum or are not a list")
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                raise ValueError(f"{frame}[{index}] must be an object")
            label = item.get("label")
            if not isinstance(label, str) or len(label) > MAX_LABEL_LENGTH:
                raise ValueError(f"{frame}[{index}] label exceeds maximum length")
            if not label or not _LABEL_PATTERN.fullmatch(label):
                raise ValueError(f"{frame}[{index}] label is invalid")
            bbox = item.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError(f"{frame}[{index}] bbox must contain four values")
            x, y, width, height = (_finite_number(value, f"{frame}[{index}] bbox") for value in bbox)
            if x < 0 or y < 0 or width <= 0 or height <= 0:
                raise ValueError(f"{frame}[{index}] bbox must be non-negative with positive size")
            confidence = _finite_number(item.get("confidence", 1.0), f"{frame}[{index}] confidence")
            if not 0.0 <= confidence <= 1.0:
                raise ValueError(f"{frame}[{index}] confidence must be between 0 and 1")
            if "mask" in item:
                mask = item["mask"]
                if not isinstance(mask, list) or not mask or len(mask) > MAX_MASK_SIDE:
                    raise ValueError(f"{frame}[{index}] mask must be a bounded 2D list")
                width = None
                for row in mask:
                    if not isinstance(row, list) or not row or len(row) > MAX_MASK_SIDE:
                        raise ValueError(f"{frame}[{index}] mask must be a bounded rectangular list")
                    width = len(row) if width is None else width
                    if len(row) != width:
                        raise ValueError(f"{frame}[{index}] mask rows must have equal length")
                    for value in row:
                        if value not in (0, 1, False, True):
                            raise ValueError(f"{frame}[{index}] mask values must be 0 or 1")


def validate_anchor_payload(payload: dict[str, Any]) -> None:
    """Validate matching 2D anchor points and reject non-finite values."""

    for frame in ("before", "after"):
        points = payload.get(frame)
        if points is None:
            continue
        if not isinstance(points, list) or not 4 <= len(points) <= 32:
            raise ValueError(f"{frame} anchors must contain between four and 32 points")
        for point in points:
            if not isinstance(point, list) or len(point) != 2:
                raise ValueError(f"{frame} anchor must contain two values")
            if any(not isfinite(float(value)) for value in point):
                raise ValueError(f"{frame} anchors must be finite")
