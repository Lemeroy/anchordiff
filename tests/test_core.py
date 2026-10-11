from __future__ import annotations

import numpy as np

from anchordiff.core import (
    Detection,
    ImageQuality,
    analyze_pair,
    align_pair,
    classify_disorder,
    match_detections,
    residual_features,
)


def test_hungarian_matching_prefers_same_class_and_nearby_object():
    baseline = [
        Detection("scope", (20, 20, 40, 40), 0.95),
        Detection("soldering_station", (120, 20, 40, 40), 0.94),
    ]
    final = [
        Detection("soldering_station", (118, 22, 40, 40), 0.92),
        Detection("scope", (24, 21, 40, 40), 0.93),
    ]

    matches = match_detections(baseline, final)

    assert [(m.baseline.label, m.final.label) for m in matches] == [
        ("scope", "scope"),
        ("soldering_station", "soldering_station"),
    ]
    assert all(m.status == "returned" for m in matches)


def test_residual_features_exclude_explained_instrument_area():
    baseline = np.zeros((100, 100, 3), dtype=np.uint8)
    final = baseline.copy()
    final[20:40, 20:40] = 255  # explained object movement
    final[70:80, 70:90] = 255  # unexplained clutter

    features = residual_features(
        baseline,
        final,
        explained_mask=np.pad(np.ones((20, 20), dtype=np.uint8), ((20, 60), (20, 60))),
    )

    assert 0.015 < features.residual_area_ratio < 0.025
    assert features.connected_components >= 1
    assert features.explained_area_ratio > 0.03


def test_low_quality_abstains_before_disorder_label():
    features = {
        "residual_area_ratio": 0.0,
        "connected_components": 0,
        "largest_component_ratio": 0.0,
        "moved_instruments": 0,
        "missing_instruments": 0,
    }
    quality = ImageQuality(
        blur_score=2.0,
        brightness=0.5,
        clipped_ratio=0.0,
        alignment_confidence=0.95,
    )

    assessment = classify_disorder(features, quality)

    assert assessment.level == "needs_review"
    assert "blur" in assessment.reasons


def test_end_to_end_returns_explainable_evidence():
    baseline = np.zeros((80, 80, 3), dtype=np.uint8)
    final = baseline.copy()
    final[55:65, 55:70] = 230

    result = analyze_pair(baseline, final)

    assert result.level in {"clean", "minor", "major", "needs_review"}
    assert result.evidence["method"] == "semantic_change_accounting"
    assert "residual_area_ratio" in result.evidence["features"]


def test_anchor_alignment_reports_confidence_and_restores_translation():
    baseline = np.zeros((32, 32, 3), dtype=np.uint8)
    final = np.zeros_like(baseline)
    final[8:12, 8:12] = 255
    src = np.array([[4, 4], [28, 4], [28, 28], [4, 28]], dtype=float)
    dst = src + np.array([2, 1])

    base, aligned, confidence, homography = align_pair(baseline, final, dst, src)

    assert base.shape == aligned.shape
    assert confidence > 0.9
    assert homography is not None
