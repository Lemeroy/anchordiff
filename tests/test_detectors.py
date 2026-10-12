import json

import numpy as np
import pytest

from anchordiff.core import Detection
from anchordiff.detectors import JsonDetector, ONNXDetector, RKNNDetector


def test_json_detector_is_model_agnostic(tmp_path):
    path = tmp_path / "detections.json"
    path.write_text(
        json.dumps({"before": [{"label": "object_a", "bbox": [1, 2, 10, 12], "confidence": 0.9}]}),
        encoding="utf-8",
    )

    detections = JsonDetector(path).detect(None, "before")

    assert detections[0].label == "object_a"
    assert detections[0].bbox == (1.0, 2.0, 10.0, 12.0)


def test_json_detector_returns_bounded_object_mask(tmp_path):
    path = tmp_path / "detections.json"
    path.write_text(
        json.dumps(
            {
                "before": [
                    {
                        "label": "object_a",
                        "bbox": [1, 2, 4, 3],
                        "confidence": 0.9,
                        "mask": [[0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    detection = JsonDetector(path).detect(np.zeros((3, 4, 3), dtype=np.uint8), "before")[0]

    assert isinstance(detection, Detection)
    assert detection.mask is not None
    assert detection.mask.shape == (3, 4)
    assert int(detection.mask.sum()) == 4
    assert detection.mask.dtype == bool


def test_json_detector_rejects_mask_larger_than_image(tmp_path):
    path = tmp_path / "detections.json"
    path.write_text(
        json.dumps(
            {
                "before": [
                    {
                        "label": "object_a",
                        "bbox": [1, 2, 4, 3],
                        "mask": [[1] * 20 for _ in range(20)],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="mask"):
        JsonDetector(path).detect(np.zeros((3, 4, 3), dtype=np.uint8), "before")


def test_optional_detector_adapters_require_local_weights(tmp_path):
    with pytest.raises(FileNotFoundError, match="local ONNX"):
        ONNXDetector(tmp_path / "missing.onnx")

    with pytest.raises(FileNotFoundError, match="local RKNN"):
        RKNNDetector(tmp_path / "missing.rknn")


def test_onnx_adapter_uses_injected_session_and_common_row_decoder():
    class FakeInput:
        name = "images"

    class FakeSession:
        def get_inputs(self):
            return [FakeInput()]

        def run(self, output_names, feeds):
            assert feeds["images"].shape == (1, 3, 16, 16)
            return [np.asarray([[1, 2, 11, 12, 0.9, 0]], dtype=np.float32)]

    image = np.zeros((8, 12, 3), dtype=np.uint8)
    detections = ONNXDetector(
        "unused.onnx", class_names=["object_a"], input_size=(16, 16), session=FakeSession()
    ).detect(image)

    assert detections[0].label == "object_a"
    assert detections[0].bbox == (1.0, 2.0, 10.0, 10.0)


def test_rknn_adapter_accepts_injected_runtime_and_releases_it():
    class FakeRuntime:
        def __init__(self):
            self.released = False

        def inference(self, inputs):
            assert inputs[0].shape == (16, 16, 3)
            return [np.asarray([[0, 0, 1, 1, 0.95, 0]], dtype=np.float32)]

        def release(self):
            self.released = True

    runtime = FakeRuntime()
    detections = RKNNDetector(
        "unused.rknn", class_names=["object_a"], input_size=(16, 16), runtime=runtime
    ).detect(np.zeros((8, 12, 3), dtype=np.uint8))

    assert detections[0].bbox == (0.0, 0.0, 12.0, 8.0)
    RKNNDetector("unused.rknn", runtime=runtime).close()
    assert runtime.released is True
