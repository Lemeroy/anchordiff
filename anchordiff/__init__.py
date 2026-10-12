"""AnchorDiff: CPU-first, explainable before/after analysis for fixed-camera workspaces."""

from .core import (
    AnalysisResult,
    Detection,
    DisorderAssessment,
    HomographyQuality,
    ImageQuality,
    Match,
    POSITION_TOLERANCE,
    ResidualFeatures,
    analyze_pair,
    analyze_sequence,
    classify_disorder,
    match_detections,
    residual_features,
)
from .geometry import (
    AnchorObservation,
    OpenCVAprilTagDetector,
    OpenCVArucoDetector,
    observations_to_points,
)
from .metrics import (
    IsotonicCalibrator,
    brier_score,
    expected_calibration_error,
    fit_isotonic_calibrator,
    risk_coverage_curve,
)

__all__ = [
    "AnalysisResult",
    "Detection",
    "DisorderAssessment",
    "HomographyQuality",
    "ImageQuality",
    "Match",
    "POSITION_TOLERANCE",
    "ResidualFeatures",
    "analyze_pair",
    "analyze_sequence",
    "classify_disorder",
    "match_detections",
    "residual_features",
    "AnchorObservation",
    "OpenCVAprilTagDetector",
    "OpenCVArucoDetector",
    "observations_to_points",
    "IsotonicCalibrator",
    "brier_score",
    "expected_calibration_error",
    "fit_isotonic_calibrator",
    "risk_coverage_curve",
]
