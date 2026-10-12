from __future__ import annotations

import numpy as np
import pytest

from anchordiff.core import (
    Detection,
    ImageQuality,
    analyze_pair,
    analyze_sequence,
    align_pair,
    classify_disorder,
    match_detections,
    residual_features,
    estimate_homography,
    homography_quality,
)


def test_hungarian_matching_prefers_same_class_and_nearby_object():
    baseline = [
        Detection("object_a", (20, 20, 40, 40), 0.95),
        Detection("object_b", (120, 20, 40, 40), 0.94),
    ]
    final = [
        Detection("object_b", (118, 22, 40, 40), 0.92),
        Detection("object_a", (23, 21, 40, 40), 0.93),
    ]

    matches = match_detections(baseline, final)

    assert [(m.baseline.label, m.final.label) for m in matches] == [
        ("object_a", "object_a"),
        ("object_b", "object_b"),
    ]
    assert all(m.status == "stable" for m in matches)


def test_position_tolerance_marks_demo_shift_as_moved():
    baseline = [Detection("object_a", (18, 20, 62, 42), 0.96)]
    final = [Detection("object_a", (23, 23, 62, 42), 0.95)]

    matches = match_detections(baseline, final)

    assert matches[0].displacement == pytest.approx(0.1143, abs=1e-4)
    assert matches[0].status == "moved"


def test_position_tolerance_can_be_overridden_for_a_scene():
    baseline = [Detection("object_a", (18, 20, 62, 42), 0.96)]
    final = [Detection("object_a", (23, 23, 62, 42), 0.95)]

    matches = match_detections(baseline, final, position_tolerance=0.12)

    assert matches[0].status == "stable"


def test_residual_features_exclude_explained_object_area():
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


def test_low_quality_abstains_before_disorder_score():
    features = {
        "residual_area_ratio": 0.0,
        "connected_components": 0,
        "largest_component_ratio": 0.0,
        "moved_objects": 0,
        "missing_objects": 0,
        "new_objects": 0,
    }
    quality = ImageQuality(
        blur_score=2.0,
        brightness=0.5,
        clipped_ratio=0.0,
        alignment_confidence=0.95,
    )

    assessment = classify_disorder(features, quality)

    assert assessment.needs_review is True
    assert assessment.score is None
    assert "blur" in assessment.reasons


def test_end_to_end_returns_explainable_evidence():
    baseline = np.zeros((80, 80, 3), dtype=np.uint8)
    final = baseline.copy()
    final[55:65, 55:70] = 230

    result = analyze_pair(baseline, final)

    assert result.disorder_score is None
    assert isinstance(result.needs_review, bool)
    assert result.evidence["method"] == "semantic_change_accounting"
    assert "residual_area_ratio" in result.evidence["features"]


def test_residual_regions_include_stable_geometry_evidence():
    baseline = np.zeros((100, 100, 3), dtype=np.uint8)
    final = baseline.copy()
    final[10:20, 15:35] = 230  # area 200, centroid (24.5, 14.5)
    final[60:65, 70:76] = 230  # area 30, centroid (72.5, 62.0)

    result = analyze_pair(baseline, final)

    regions = result.evidence["residual_regions"]
    assert regions == [
        {
            "id": 1,
            "bbox": [15, 10, 20, 10],
            "action": "unexplained_change",
            "explanation": "changed pixels not accounted for by matched object masks or boxes",
            "area_px": 200,
            "area_ratio": 0.02,
            "centroid": [24.5, 14.5],
        },
        {
            "id": 2,
            "bbox": [70, 60, 6, 5],
            "action": "unexplained_change",
            "explanation": "changed pixels not accounted for by matched object masks or boxes",
            "area_px": 30,
            "area_ratio": 0.003,
            "centroid": [72.5, 62.0],
        },
    ]


def test_sequence_evidence_exposes_quality_for_each_candidate():
    baseline = np.zeros((48, 48, 3), dtype=np.uint8)
    first = baseline.copy()
    first[10:20, 10:20] = 180
    second = baseline.copy()
    second[5:43, 5:43] = 180

    result = analyze_sequence(baseline, [first, second])

    sequence = result.evidence["sequence"]
    assert sequence["candidates"] == 2
    assert len(sequence["frames"]) == 2
    assert sequence["selected_index"] in {0, 1}
    assert sequence["frames"][sequence["selected_index"]]["selected"] is True
    assert all("blur_score" in frame and "selection_score" in frame for frame in sequence["frames"])


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


def test_homography_quality_is_normalized_and_reports_degenerate_geometry():
    src = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=float)
    dst = src + np.array([4, -3])
    H, quality = estimate_homography(src, dst)

    assert quality.degenerate is False
    assert quality.reprojection_rmse < 1e-8
    assert quality.normalized_rmse < 1e-8
    assert quality.confidence > 0.99
    np.testing.assert_allclose(homography_quality(src, dst, H).normalized_rmse, 0.0, atol=1e-8)


def test_homography_quality_rejects_collinear_anchor_geometry():
    points = np.array([[0, 0], [10, 0], [20, 0], [30, 0]], dtype=float)
    with np.testing.assert_raises(ValueError):
        estimate_homography(points, points)


def test_analyze_pair_emits_normalized_geometry_evidence():
    image = np.full((32, 32, 3), 160, dtype=np.uint8)
    anchors = [[4, 4], [28, 4], [28, 28], [4, 28]]

    result = analyze_pair(image, image.copy(), baseline_anchors=anchors, final_anchors=anchors)

    assert result.evidence["geometry"]["degenerate"] is False
    assert result.evidence["geometry"]["normalized_rmse"] < 1e-8


def test_low_detector_confidence_abstains():
    image = np.full((40, 40, 3), 180, dtype=np.uint8)
    result = analyze_pair(
        image,
        image.copy(),
        baseline_detections=[Detection("object_a", (5, 5, 12, 12), 0.42)],
        final_detections=[Detection("object_a", (5, 5, 12, 12), 0.45)],
    )

    assert result.needs_review is True
    assert "detection_confidence" in result.evidence["reasons"]
    assert result.disorder_score is None


def test_object_masks_are_used_for_semantic_change_accounting():
    baseline = np.zeros((30, 30, 3), dtype=np.uint8)
    final = baseline.copy()
    final[10:20, 10:20] = 255
    baseline_mask = np.zeros((30, 30), dtype=bool)
    baseline_mask[10:15, 10:15] = True
    final_mask = np.zeros((30, 30), dtype=bool)
    final_mask[10:15, 10:15] = True

    result = analyze_pair(
        baseline,
        final,
        baseline_detections=[Detection("object_a", (10, 10, 10, 10), 0.95, baseline_mask)],
        final_detections=[Detection("object_a", (10, 10, 10, 10), 0.95, final_mask)],
    )

    # The changed pixels outside the predicted object mask remain residual evidence.
    assert result.residual.residual_area_ratio > 0.02
    assert result.residual.explained_area_ratio < 0.02
    assert result.evidence["matches"][0]["mask_area_px"] == 25


def test_demo_detection_report_is_generic_and_explains_each_object():
    from anchordiff.synthetic import make_fixture

    fixture = make_fixture(seed=11, size=(256, 256), clutter=True)
    baseline = [
        Detection("object_a", (18, 20, 62, 42), 0.96),
        Detection("object_b", (142, 22, 55, 35), 0.95),
        Detection("object_c", (32, 146, 55, 35), 0.95),
    ]
    final = [
        Detection("object_a", (23, 23, 62, 42), 0.95),
        Detection("object_b", (142, 22, 55, 35), 0.95),
        Detection("object_c", (32, 146, 55, 35), 0.95),
    ]

    result = analyze_pair(
        fixture.before,
        fixture.after,
        baseline_detections=baseline,
        final_detections=final,
    )

    objects = result.evidence["matches"]
    assert objects[0]["status"] == "moved"
    assert objects[0]["action"] == "moved"
    assert "explanation" in objects[0]
    assert [item["label"] for item in objects] == ["object_a", "object_b", "object_c"]
    assert len(result.evidence["residual_regions"]) == 2
