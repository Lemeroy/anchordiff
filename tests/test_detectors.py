import json

from anchordiff.detectors import JsonDetector


def test_json_detector_is_model_agnostic(tmp_path):
    path = tmp_path / "detections.json"
    path.write_text(
        json.dumps({"before": [{"label": "scope", "bbox": [1, 2, 10, 12], "confidence": 0.9}]}),
        encoding="utf-8",
    )

    detections = JsonDetector(path).detect(None, "before")

    assert detections[0].label == "scope"
    assert detections[0].bbox == (1.0, 2.0, 10.0, 12.0)
