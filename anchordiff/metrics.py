"""Dependency-light confidence calibration and selective-risk metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _validated_inputs(
    confidence: Sequence[float], correctness: Sequence[bool]
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(confidence, dtype=np.float64)
    labels = np.asarray(correctness, dtype=bool)
    if scores.ndim != 1 or labels.ndim != 1 or len(scores) != len(labels) or not len(scores):
        raise ValueError("confidence and correctness must be non-empty 1D arrays of equal length")
    if not np.all(np.isfinite(scores)) or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("confidence values must be finite and within [0, 1]")
    return scores, labels


def brier_score(confidence: Sequence[float], correctness: Sequence[bool]) -> float:
    """Return the mean squared error between confidence and binary correctness."""

    scores, labels = _validated_inputs(confidence, correctness)
    return float(np.mean((scores - labels.astype(np.float64)) ** 2))


def expected_calibration_error(
    confidence: Sequence[float], correctness: Sequence[bool], *, bins: int = 10
) -> float:
    """Return equal-width expected calibration error for confidence scores."""

    if not isinstance(bins, int) or bins <= 0:
        raise ValueError("bins must be a positive integer")
    scores, labels = _validated_inputs(confidence, correctness)
    total = float(len(scores))
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        selected = (scores >= lower) & ((scores < upper) if index < bins - 1 else (scores <= upper))
        if np.any(selected):
            accuracy = float(np.mean(labels[selected]))
            mean_confidence = float(np.mean(scores[selected]))
            error += float(np.sum(selected)) / total * abs(accuracy - mean_confidence)
    return float(error)


def risk_coverage_curve(
    confidence: Sequence[float], correctness: Sequence[bool]
) -> list[dict[str, float]]:
    """Return selective risk as progressively lower-confidence samples are accepted."""

    scores, labels = _validated_inputs(confidence, correctness)
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    sorted_scores = scores[order]
    curve: list[dict[str, float]] = [{"coverage": 0.0, "risk": 0.0, "threshold": 1.0}]
    correct_count = 0
    total = len(sorted_scores)
    for count, (score, label) in enumerate(zip(sorted_scores, sorted_labels), start=1):
        correct_count += int(label)
        curve.append(
            {
                "coverage": float(count / total),
                "risk": float(1.0 - correct_count / count),
                "threshold": float(score),
            }
        )
    return curve


@dataclass(frozen=True)
class IsotonicCalibrator:
    """Monotone piecewise-constant mapping learned with PAVA."""

    thresholds: tuple[float, ...]
    values: tuple[float, ...]

    def transform(self, confidence: Sequence[float] | np.ndarray) -> np.ndarray:
        scores = np.asarray(confidence, dtype=np.float64)
        if not np.all(np.isfinite(scores)) or np.any((scores < 0.0) | (scores > 1.0)):
            raise ValueError("confidence values must be finite and within [0, 1]")
        return np.interp(
            scores,
            np.asarray(self.thresholds, dtype=np.float64),
            np.asarray(self.values, dtype=np.float64),
            left=self.values[0],
            right=self.values[-1],
        )


def fit_isotonic_calibrator(
    confidence: Sequence[float], correctness: Sequence[bool]
) -> IsotonicCalibrator:
    """Fit a deterministic increasing calibrator using the PAVA algorithm."""

    scores, labels = _validated_inputs(confidence, correctness)
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order].astype(np.float64)
    blocks: list[dict[str, float]] = []
    for score, label in zip(sorted_scores, sorted_labels):
        if blocks and score == blocks[-1]["score"]:
            blocks[-1]["weight"] += 1.0
            blocks[-1]["total"] += float(label)
            blocks[-1]["value"] = blocks[-1]["total"] / blocks[-1]["weight"]
        else:
            blocks.append({"score": float(score), "weight": 1.0, "total": float(label), "value": float(label)})
        while len(blocks) >= 2 and blocks[-2]["value"] > blocks[-1]["value"]:
            right = blocks.pop()
            left = blocks.pop()
            weight = left["weight"] + right["weight"]
            total = left["total"] + right["total"]
            blocks.append(
                {
                    "score": right["score"],
                    "weight": weight,
                    "total": total,
                    "value": total / weight,
                }
            )
    return IsotonicCalibrator(
        tuple(block["score"] for block in blocks),
        tuple(block["value"] for block in blocks),
    )
