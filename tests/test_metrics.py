from __future__ import annotations

import numpy as np
import pytest

from anchordiff.metrics import (
    brier_score,
    expected_calibration_error,
    fit_isotonic_calibrator,
    risk_coverage_curve,
)


def test_calibration_metrics_are_deterministic():
    confidence = [0.1, 0.4, 0.8, 0.9]
    correct = [False, True, True, False]

    assert brier_score(confidence, correct) == pytest.approx(0.305)
    assert expected_calibration_error(confidence, correct, bins=2) == pytest.approx(0.3)


def test_risk_coverage_accepts_high_confidence_first():
    curve = risk_coverage_curve([0.9, 0.4, 0.8], [True, False, True])

    assert curve == [
        {"coverage": 0.0, "risk": 0.0, "threshold": 1.0},
        {"coverage": pytest.approx(1 / 3), "risk": 0.0, "threshold": 0.9},
        {"coverage": pytest.approx(2 / 3), "risk": 0.0, "threshold": 0.8},
        {"coverage": 1.0, "risk": pytest.approx(1 / 3), "threshold": 0.4},
    ]


def test_isotonic_calibrator_is_monotonic_and_clamps_scores():
    calibrator = fit_isotonic_calibrator([0.1, 0.2, 0.8, 0.9], [False, True, False, True])
    transformed = calibrator.transform(np.array([0.0, 0.2, 0.5, 1.0]))

    assert np.all(np.diff(transformed) >= 0)
    assert np.all((transformed >= 0.0) & (transformed <= 1.0))


def test_metrics_reject_mismatched_or_empty_inputs():
    with pytest.raises(ValueError):
        brier_score([], [])
    with pytest.raises(ValueError):
        risk_coverage_curve([0.5], [True, False])
    with pytest.raises(ValueError):
        expected_calibration_error([1.2], [True])
