"""Privacy-safe deterministic fixtures for demos and regression tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyntheticFixture:
    before: np.ndarray
    after: np.ndarray
    metadata: dict


def _rectangle(image: np.ndarray, x: int, y: int, w: int, h: int, color: tuple[int, int, int]) -> None:
    image[y : y + h, x : x + w] = np.asarray(color, dtype=np.uint8)
    image[y : y + 2, x : x + w] = 255
    image[y + h - 2 : y + h, x : x + w] = 255
    image[y : y + h, x : x + 2] = 255
    image[y : y + h, x + w - 2 : x + w] = 255


def make_fixture(
    *, seed: int = 0, size: tuple[int, int] = (256, 256), clutter: bool = True
) -> SyntheticFixture:
    """Create a small before/after workspace pair without real user images."""

    rng = np.random.default_rng(seed)
    height, width = size
    before = np.full((height, width, 3), 218, dtype=np.uint8)
    # A light texture prevents a perfectly flat background from hiding quality issues.
    noise = rng.normal(0, 2.0, size=(height, width, 1))
    before = np.clip(before.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    objects = [
        {"label": "device_a", "bbox": (18, 20, 62, 42), "color": (45, 70, 105)},
        {"label": "device_b", "bbox": (142, 22, 55, 35), "color": (90, 48, 38)},
        {"label": "device_c", "bbox": (32, 146, 55, 35), "color": (45, 92, 65)},
    ]
    for item in objects:
        _rectangle(before, *item["bbox"], item["color"])
    after = before.copy()
    # Simulate a tracked object with a small translation.
    x, y, w, h = objects[0]["bbox"]
    after[y : y + h, x : x + w] = 218
    _rectangle(after, x + 5, y + 3, w, h, objects[0]["color"])
    residual_regions = 0
    if clutter:
        residual_regions = 2
        for x0, y0, w0, h0 in ((190, 160, 18, 9), (210, 184, 24, 5)):
            after[y0 : y0 + h0, x0 : x0 + w0] = (35, 35, 35)
    metadata = {
        "seed": seed,
        "ground_truth": {
            "object_states": {"device_a": "moved", "device_b": "returned", "device_c": "returned"},
            "residual_regions": residual_regions,
        },
    }
    return SyntheticFixture(before, after, metadata)
