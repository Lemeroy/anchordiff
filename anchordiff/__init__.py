"""AnchorDiff: CPU-first, explainable before/after analysis for fixed-camera workspaces."""

from .core import (
    AnalysisResult,
    Detection,
    DisorderAssessment,
    ImageQuality,
    Match,
    ResidualFeatures,
    analyze_pair,
    analyze_sequence,
    classify_disorder,
    match_detections,
    residual_features,
)

__all__ = [
    "AnalysisResult",
    "Detection",
    "DisorderAssessment",
    "ImageQuality",
    "Match",
    "ResidualFeatures",
    "analyze_pair",
    "analyze_sequence",
    "classify_disorder",
    "match_detections",
    "residual_features",
]
