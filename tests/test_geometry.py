from __future__ import annotations

import numpy as np
import pytest

from anchordiff.geometry import (
    AnchorObservation,
    OpenCVAprilTagDetector,
    OpenCVArucoDetector,
    observations_to_points,
)


def test_observations_to_points_pairs_common_marker_corners_in_id_order():
    before = [
        AnchorObservation(7, ((70, 70), (80, 70), (80, 80), (70, 80))),
        AnchorObservation(2, ((20, 20), (30, 20), (30, 30), (20, 30))),
    ]
    after = [
        AnchorObservation(2, ((21, 22), (31, 22), (31, 32), (21, 32))),
        AnchorObservation(7, ((71, 72), (81, 72), (81, 82), (71, 82))),
    ]

    source, target, marker_ids = observations_to_points(after, before)

    assert marker_ids == (2, 7)
    assert source.shape == target.shape == (8, 2)
    np.testing.assert_allclose(source[:4], np.asarray(after[0].corners))
    np.testing.assert_allclose(target[:4], np.asarray(before[1].corners))


def test_observations_to_points_requires_four_corresponding_points():
    one = AnchorObservation(1, ((0, 0), (1, 0), (1, 1), (0, 1)))
    with pytest.raises(ValueError, match="at least four"):
        observations_to_points([one], [], )


def test_optional_opencv_adapters_are_lazy_and_actionable():
    for factory in (OpenCVArucoDetector, OpenCVAprilTagDetector):
        try:
            detector = factory()
        except RuntimeError as exc:
            assert "opencv" in str(exc).lower()
        else:
            assert detector.dictionary_name.startswith("DICT_")
