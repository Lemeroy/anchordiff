"""Optional marker adapters for fixed-camera geometric alignment.

The core accepts plain point arrays and does not import OpenCV.  This module is
an integration boundary: OpenCV is imported only when an adapter is created.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class AnchorObservation:
    """One detected square marker and its four image-space corners."""

    marker_id: int
    corners: tuple[tuple[float, float], ...]
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if len(self.corners) != 4:
            raise ValueError("anchor marker must contain four corners")
        values = np.asarray(self.corners, dtype=np.float64)
        if values.shape != (4, 2) or not np.isfinite(values).all():
            raise ValueError("anchor corners must be finite 2D points")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("anchor confidence must be between 0 and 1")

    @property
    def center(self) -> tuple[float, float]:
        values = np.asarray(self.corners, dtype=np.float64)
        center = values.mean(axis=0)
        return float(center[0]), float(center[1])


class AnchorDetector(Protocol):
    def detect(self, image: np.ndarray) -> tuple[AnchorObservation, ...]: ...


def observations_to_points(
    source: Sequence[AnchorObservation],
    target: Sequence[AnchorObservation],
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """Flatten corners for marker IDs visible in both images.

    Marker IDs are sorted for deterministic correspondence.  Keeping all four
    corners per marker gives the homography estimator more constraints than
    using marker centers alone.
    """

    source_by_id = {int(item.marker_id): item for item in source}
    target_by_id = {int(item.marker_id): item for item in target}
    marker_ids = tuple(sorted(set(source_by_id) & set(target_by_id)))
    if not marker_ids:
        raise ValueError("at least four corresponding anchor points are required")
    source_points = np.concatenate(
        [np.asarray(source_by_id[marker_id].corners, dtype=np.float64) for marker_id in marker_ids],
        axis=0,
    )
    target_points = np.concatenate(
        [np.asarray(target_by_id[marker_id].corners, dtype=np.float64) for marker_id in marker_ids],
        axis=0,
    )
    if len(source_points) < 4:
        raise ValueError("at least four corresponding anchor points are required")
    return source_points, target_points, marker_ids


def _load_opencv():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError(
            "OpenCV is optional; install anchordiff[opencv] to use marker adapters"
        ) from exc
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "OpenCV ArUco support is unavailable; install anchordiff[opencv]"
        )
    return cv2


class _OpenCVMarkerDetector:
    def __init__(self, dictionary: str):
        self.cv2 = _load_opencv()
        aruco = self.cv2.aruco
        try:
            dictionary_id = getattr(aruco, dictionary)
        except AttributeError as exc:
            raise ValueError(f"unknown OpenCV marker dictionary: {dictionary}") from exc
        self.dictionary_name = dictionary
        self.dictionary = aruco.getPredefinedDictionary(dictionary_id)
        self.parameters = aruco.DetectorParameters()
        if hasattr(self.parameters, "cornerRefinementMethod") and hasattr(aruco, "CORNER_REFINE_SUBPIX"):
            self.parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        self._detector = (
            aruco.ArucoDetector(self.dictionary, self.parameters)
            if hasattr(aruco, "ArucoDetector")
            else None
        )

    def detect(self, image: np.ndarray) -> tuple[AnchorObservation, ...]:
        arr = np.asarray(image)
        if arr.ndim == 3:
            gray = self.cv2.cvtColor(arr, self.cv2.COLOR_RGB2GRAY)
        elif arr.ndim == 2:
            gray = arr
        else:
            raise ValueError("image must have shape HxW or HxWx3")
        if self._detector is not None:
            corners, ids, _ = self._detector.detectMarkers(gray)
        else:  # OpenCV 4.6 compatibility
            corners, ids, _ = self.cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.parameters)
        if ids is None:
            return ()
        observations = []
        for marker_corners, marker_id in zip(corners, np.asarray(ids).reshape(-1)):
            values = np.asarray(marker_corners, dtype=np.float64).reshape(4, 2)
            observations.append(AnchorObservation(int(marker_id), tuple(map(tuple, values))))
        return tuple(sorted(observations, key=lambda item: item.marker_id))


class OpenCVArucoDetector(_OpenCVMarkerDetector):
    """Detect standard ArUco markers through optional OpenCV support."""

    def __init__(self, dictionary: str = "DICT_4X4_50"):
        super().__init__(dictionary)


class OpenCVAprilTagDetector(_OpenCVMarkerDetector):
    """Detect AprilTags using OpenCV's built-in AprilTag dictionaries."""

    def __init__(self, dictionary: str = "DICT_APRILTAG_36h11"):
        super().__init__(dictionary)

