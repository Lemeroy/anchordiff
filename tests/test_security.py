from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from anchordiff.detectors import JsonDetector, UltralyticsDetector
from anchordiff.sanitize import load_sanitized_rgb, validate_detection_payload


def test_image_loader_drops_exif_metadata(tmp_path: Path):
    path = tmp_path / "capture.jpg"
    image = Image.new("RGB", (12, 12), "white")
    exif = Image.Exif()
    exif[305] = "private-camera-software"
    image.save(path, exif=exif)

    array = load_sanitized_rgb(path)
    assert array.shape == (12, 12, 3)
    assert array.dtype == np.uint8


def test_detection_payload_rejects_non_finite_or_oversized_input():
    with pytest.raises(ValueError, match="finite"):
        validate_detection_payload({"before": [{"label": "object_a", "bbox": [0, 0, float("nan"), 4]}]})
    with pytest.raises(ValueError, match="maximum"):
        validate_detection_payload({"before": [{"label": "x" * 100, "bbox": [0, 0, 4, 4]}]})


def test_json_detector_validates_untrusted_fixture(tmp_path: Path):
    path = tmp_path / "detections.json"
    path.write_text('{"before": [{"label": "object_a", "bbox": [0, 0, -2, 4]}]}', encoding="utf-8")

    with pytest.raises(ValueError, match="bbox"):
        JsonDetector(path)


def test_ultralytics_adapter_does_not_download_weights_by_default():
    with pytest.raises(FileNotFoundError, match="local model weights"):
        UltralyticsDetector("yolo11n.pt")
