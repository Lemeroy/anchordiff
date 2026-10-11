"""Reference-conditioned state analysis.

The implementation intentionally has no mandatory deep-learning dependency. A detector
adapter can provide :class:`Detection` objects, while this module handles matching,
semantic residual accounting, uncertainty and a deterministic evidence report.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Iterable, Sequence

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class Detection:
    """An object observation in ``(x, y, width, height)`` pixel coordinates."""

    label: str
    bbox: tuple[float, float, float, float]
    confidence: float = 1.0

    @property
    def center(self) -> tuple[float, float]:
        x, y, w, h = self.bbox
        return x + w / 2.0, y + h / 2.0

    @property
    def area(self) -> float:
        _, _, w, h = self.bbox
        return max(0.0, w) * max(0.0, h)


@dataclass(frozen=True)
class ImageQuality:
    blur_score: float
    brightness: float
    clipped_ratio: float
    alignment_confidence: float = 1.0


@dataclass(frozen=True)
class Match:
    baseline: Detection
    final: Detection | None
    status: str
    displacement: float
    cost: float


@dataclass(frozen=True)
class ResidualFeatures:
    residual_area_ratio: float
    explained_area_ratio: float
    changed_area_ratio: float
    connected_components: int
    largest_component_ratio: float
    spatial_entropy: float


@dataclass(frozen=True)
class DisorderAssessment:
    level: str
    score: float
    reasons: tuple[str, ...]
    rule_version: str = "sca-0.1"


@dataclass(frozen=True)
class AnalysisResult:
    level: str
    score: float
    quality: ImageQuality
    matches: tuple[Match, ...]
    residual: ResidualFeatures
    evidence: dict
    residual_mask: Array


def _gray(image: Array) -> Array:
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError("image must have shape HxW or HxWx3")
    return (
        0.299 * arr[..., 0].astype(np.float32)
        + 0.587 * arr[..., 1].astype(np.float32)
        + 0.114 * arr[..., 2].astype(np.float32)
    )


def image_quality(image: Array, alignment_confidence: float = 1.0) -> ImageQuality:
    """Compute cheap quality signals used by the abstention gate."""

    gray = _gray(image)
    if min(gray.shape) < 2:
        blur = 0.0
    else:
        gy, gx = np.gradient(gray)
        blur = float(np.var(gx) + np.var(gy))
    brightness = float(np.mean(gray) / 255.0)
    clipped = float(np.mean((gray <= 3.0) | (gray >= 252.0)))
    return ImageQuality(blur, brightness, clipped, alignment_confidence)


def _homography(src: Array, dst: Array) -> tuple[Array, float]:
    """Estimate a planar homography with a dependency-free normalized DLT."""

    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2 or len(src) < 4:
        raise ValueError("src and dst need at least four matching 2D points")
    def normalize(points: Array) -> tuple[Array, Array]:
        center = np.mean(points, axis=0)
        distances = np.linalg.norm(points - center, axis=1)
        scale = math.sqrt(2.0) / max(float(np.mean(distances)), 1e-12)
        T = np.array(
            [[scale, 0.0, -scale * center[0]], [0.0, scale, -scale * center[1]], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        normalized = _project_points(points, T)
        return normalized, T

    src_normalized, src_transform = normalize(src)
    dst_normalized, dst_transform = normalize(dst)
    rows = []
    for (x, y), (u, v) in zip(src_normalized, dst_normalized):
        rows.append([-x, -y, -1.0, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1.0, v * x, v * y, v])
    # Four point correspondences produce an 8x9 system; retain the ninth
    # right-singular vector so the null-space solution is available.
    _, _, vh = np.linalg.svd(np.asarray(rows), full_matrices=True)
    h = vh[-1]
    H_normalized = h.reshape(3, 3)
    H = np.linalg.inv(dst_transform) @ H_normalized @ src_transform
    if abs(H[2, 2]) < 1e-12:
        raise ValueError("degenerate anchor geometry")
    H /= H[2, 2]
    projected = _project_points(src, H)
    error = float(np.mean(np.linalg.norm(projected - dst, axis=1)))
    confidence = float(math.exp(-error / 4.0))
    return H, confidence


def _project_points(points: Array, H: Array) -> Array:
    points = np.asarray(points, dtype=np.float64)
    homogeneous = np.c_[points, np.ones(len(points))]
    projected = homogeneous @ H.T
    return projected[:, :2] / np.maximum(projected[:, 2:3], 1e-12)


def warp_perspective(image: Array, H: Array, output_shape: tuple[int, int] | None = None) -> Array:
    """Nearest-neighbor perspective warp implemented with NumPy for CPU portability."""

    arr = np.asarray(image)
    height, width = output_shape or arr.shape[:2]
    inv = np.linalg.inv(np.asarray(H, dtype=np.float64))
    yy, xx = np.indices((height, width), dtype=np.float64)
    target = np.c_[xx.ravel(), yy.ravel(), np.ones(height * width)]
    source = target @ inv.T
    source = source[:, :2] / np.maximum(source[:, 2:3], 1e-12)
    sx = np.rint(source[:, 0]).astype(int)
    sy = np.rint(source[:, 1]).astype(int)
    valid = (sx >= 0) & (sx < arr.shape[1]) & (sy >= 0) & (sy < arr.shape[0])
    result = np.zeros((height, width) + arr.shape[2:], dtype=arr.dtype)
    flat = result.reshape((height * width,) + arr.shape[2:])
    flat[valid] = arr[sy[valid], sx[valid]]
    return result


def align_pair(
    baseline: Array,
    final: Array,
    baseline_anchors: Sequence[Sequence[float]] | None = None,
    final_anchors: Sequence[Sequence[float]] | None = None,
) -> tuple[Array, Array, float, Array | None]:
    """Align ``final`` to ``baseline`` and return confidence and homography."""

    base = np.asarray(baseline)
    end = np.asarray(final)
    if base.shape != end.shape:
        raise ValueError("baseline and final must have the same shape before alignment")
    if baseline_anchors is None or final_anchors is None:
        return base, end, 1.0, None
    H, confidence = _homography(np.asarray(final_anchors), np.asarray(baseline_anchors))
    return base, warp_perspective(end, H, base.shape[:2]), confidence, H


def _iou(a: Detection, b: Detection) -> float:
    ax, ay, aw, ah = a.bbox
    bx, by, bw, bh = b.bbox
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, right - left) * max(0.0, bottom - top)
    union = a.area + b.area - inter
    return inter / union if union else 0.0


def _pair_cost(a: Detection, b: Detection) -> float:
    if a.label != b.label:
        return 10.0
    ax, ay = a.center
    bx, by = b.center
    distance = math.hypot(ax - bx, ay - by)
    size_delta = abs(math.log((a.area + 1.0) / (b.area + 1.0)))
    return 0.50 * (1.0 - _iou(a, b)) + 0.35 * min(distance / 200.0, 1.0) + 0.15 * min(size_delta, 1.0)


def _hungarian(cost: Sequence[Sequence[float]]) -> list[tuple[int, int]]:
    """Minimum-cost rectangular assignment (Kuhn-Munkres), pure Python."""

    if not cost or not cost[0]:
        return []
    n, m = len(cost), len(cost[0])
    transposed = False
    matrix = [list(map(float, row)) for row in cost]
    if n > m:
        matrix = [list(row) for row in zip(*matrix)]
        n, m = m, n
        transposed = True
    u, v = [0.0] * (n + 1), [0.0] * (m + 1)
    p, way = [0] * (m + 1), [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv, used = [float("inf")] * (m + 1), [False] * (m + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], float("inf"), 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = matrix[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    pairs = [(p[j] - 1, j - 1) for j in range(1, m + 1) if p[j]]
    if transposed:
        return [(j, i) for i, j in pairs]
    return pairs


def match_detections(
    baseline: Sequence[Detection],
    final: Sequence[Detection],
    return_threshold: float = 0.35,
) -> tuple[Match, ...]:
    """Match objects by class and geometry, retaining missing/new observations."""

    matches: list[Match] = []
    used_final: set[int] = set()
    labels = sorted({d.label for d in baseline} | {d.label for d in final})
    for label in labels:
        bi = [i for i, d in enumerate(baseline) if d.label == label]
        fi = [i for i, d in enumerate(final) if d.label == label]
        pairs = _hungarian([[_pair_cost(baseline[i], final[j]) for j in fi] for i in bi])
        for row, col in pairs:
            i, j = bi[row], fi[col]
            used_final.add(j)
            displacement = math.hypot(
                baseline[i].center[0] - final[j].center[0],
                baseline[i].center[1] - final[j].center[1],
            ) / max(math.sqrt(baseline[i].area), math.sqrt(final[j].area), 1.0)
            status = "returned" if displacement <= return_threshold else "moved"
            matches.append(Match(baseline[i], final[j], status, displacement, _pair_cost(baseline[i], final[j])))
        matched_base = {bi[row] for row, _ in pairs}
        for i in bi:
            if i not in matched_base:
                matches.append(Match(baseline[i], None, "missing", 1.0, 1.0))
    for j, detection in enumerate(final):
        if j not in used_final:
            placeholder = Detection(detection.label, detection.bbox, detection.confidence)
            matches.append(Match(placeholder, detection, "new", 1.0, 1.0))
    order = {id(d): i for i, d in enumerate(baseline)}
    return tuple(sorted(matches, key=lambda m: order.get(id(m.baseline), len(order))))


def _connected_components(mask: Array) -> list[Array]:
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros(mask.shape, dtype=bool)
    components: list[Array] = []
    height, width = mask.shape
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        stack = [(int(y), int(x))]
        seen[y, x] = True
        pixels: list[tuple[int, int]] = []
        while stack:
            cy, cx = stack.pop()
            pixels.append((cy, cx))
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        components.append(np.asarray(pixels, dtype=np.int32))
    return components


def _residual_region_evidence(mask: Array) -> list[dict]:
    """Return stable geometry records for each connected residual region."""

    mask = np.asarray(mask, dtype=bool)
    total = max(int(mask.size), 1)
    regions: list[dict] = []
    for component in _connected_components(mask):
        ys = component[:, 0]
        xs = component[:, 1]
        x0, x1 = int(np.min(xs)), int(np.max(xs))
        y0, y1 = int(np.min(ys)), int(np.max(ys))
        area = int(len(component))
        regions.append(
            {
                "bbox": [x0, y0, x1 - x0 + 1, y1 - y0 + 1],
                "area_px": area,
                "area_ratio": round(area / total, 6),
                "centroid": [round(float(np.mean(xs)), 4), round(float(np.mean(ys)), 4)],
            }
        )
    regions.sort(key=lambda item: (-item["area_px"], item["bbox"][1], item["bbox"][0]))
    for index, region in enumerate(regions, start=1):
        region["id"] = index
    return regions


def _robust_difference(baseline: Array, final: Array) -> Array:
    raw = np.abs(np.asarray(final, dtype=np.float32) - np.asarray(baseline, dtype=np.float32))
    diff = np.mean(raw, axis=-1) if raw.ndim == 3 else raw
    median = float(np.median(diff))
    mad = float(np.median(np.abs(diff - median)))
    threshold = max(12.0, median + 4.0 * max(mad, 1.0))
    return diff > threshold


def residual_features(
    baseline: Array,
    final: Array,
    explained_mask: Array | None = None,
) -> ResidualFeatures:
    """Perform semantic change accounting after subtracting explained regions."""

    changed = _robust_difference(baseline, final)
    explained = np.asarray(explained_mask, dtype=bool) if explained_mask is not None else np.zeros_like(changed)
    if explained.shape != changed.shape:
        raise ValueError("explained_mask must have the image height and width")
    residual = changed & ~explained
    total = float(changed.size)
    residual_ratio = float(np.mean(residual))
    explained_ratio = float(np.mean(changed & explained))
    changed_ratio = float(np.mean(changed))
    components = _connected_components(residual)
    sizes = sorted((len(c) for c in components), reverse=True)
    largest = float(sizes[0] / total) if sizes else 0.0
    if sizes:
        grid = np.zeros(16, dtype=np.float64)
        for component in components:
            cy, cx = np.mean(component, axis=0)
            cell = min(3, int(cy / max(residual.shape[0], 1) * 4)) * 4 + min(3, int(cx / max(residual.shape[1], 1) * 4))
            grid[cell] += len(component)
        probabilities = grid[grid > 0] / max(float(np.sum(grid)), 1.0)
        entropy = float(-np.sum(probabilities * np.log(probabilities))) / math.log(16)
    else:
        entropy = 0.0
    return ResidualFeatures(residual_ratio, explained_ratio, changed_ratio, len(components), largest, entropy)


def classify_disorder(features: ResidualFeatures | dict, quality: ImageQuality, *, rule_version: str = "sca-0.1") -> DisorderAssessment:
    """Apply monotonic, inspectable rules and abstain when evidence is unreliable."""

    if isinstance(features, dict):
        values = features
        residual_ratio = float(values.get("residual_area_ratio", 0.0))
        components = int(values.get("connected_components", 0))
        largest = float(values.get("largest_component_ratio", 0.0))
        moved = int(values.get("moved_instruments", 0))
        missing = int(values.get("missing_instruments", 0))
        entropy = 0.0
    else:
        residual_ratio = features.residual_area_ratio
        components = features.connected_components
        largest = features.largest_component_ratio
        moved = missing = 0
        entropy = features.spatial_entropy
    reasons: list[str] = []
    if quality.blur_score < 5.0:
        reasons.append("blur")
    if quality.brightness < 0.05 or quality.brightness > 0.95:
        reasons.append("exposure")
    if quality.clipped_ratio > 0.25:
        reasons.append("clipping")
    if quality.alignment_confidence < 0.80:
        reasons.append("alignment")
    if reasons:
        return DisorderAssessment("needs_review", 1.0, tuple(reasons), rule_version)
    score = min(
        1.0,
        0.55 * min(residual_ratio / 0.20, 1.0)
        + 0.15 * min(components / 12.0, 1.0)
        + 0.15 * min(largest / 0.12, 1.0)
        + 0.10 * min((moved + missing) / 3.0, 1.0)
        + 0.05 * entropy,
    )
    if missing or moved:
        reasons.append("instrument_state_change")
    if residual_ratio > 0.015:
        reasons.append("unexplained_residual")
    if score < 0.15:
        level = "clean"
    elif score < 0.35:
        level = "minor"
    else:
        level = "major"
    return DisorderAssessment(level, score, tuple(dict.fromkeys(reasons)), rule_version)


def _serialize_match(match: Match) -> dict:
    return {
        "class": match.baseline.label,
        "status": match.status,
        "displacement": round(match.displacement, 4),
        "cost": round(match.cost, 4),
        "baseline_bbox": list(match.baseline.bbox),
        "final_bbox": list(match.final.bbox) if match.final else None,
        "confidence": round(min(match.baseline.confidence, match.final.confidence if match.final else match.baseline.confidence), 4),
    }


def analyze_pair(
    baseline: Array,
    final: Array,
    *,
    baseline_detections: Sequence[Detection] = (),
    final_detections: Sequence[Detection] = (),
    baseline_anchors: Sequence[Sequence[float]] | None = None,
    final_anchors: Sequence[Sequence[float]] | None = None,
) -> AnalysisResult:
    """Run AnchorDiff's core pipeline and return evidence plus a residual mask."""

    base, aligned_final, alignment_confidence, _ = align_pair(
        baseline, final, baseline_anchors, final_anchors
    )
    quality = image_quality(aligned_final, alignment_confidence)
    matches = match_detections(baseline_detections, final_detections)
    explained = np.zeros(base.shape[:2], dtype=bool)
    for match in matches:
        if match.status in {"moved", "missing", "new"}:
            detection = match.final or match.baseline
            x, y, w, h = [int(round(v)) for v in detection.bbox]
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(base.shape[1], x + max(1, w)), min(base.shape[0], y + max(1, h))
            if x1 > x0 and y1 > y0:
                explained[y0:y1, x0:x1] = True
    residual = residual_features(base, aligned_final, explained)
    moved = sum(m.status == "moved" for m in matches)
    missing = sum(m.status == "missing" for m in matches)
    assessment = classify_disorder(
        {
            "residual_area_ratio": residual.residual_area_ratio,
            "connected_components": residual.connected_components,
            "largest_component_ratio": residual.largest_component_ratio,
            "moved_instruments": moved,
            "missing_instruments": missing,
        },
        quality,
    )
    if any(
        detection.confidence < 0.60
        for detection in (*baseline_detections, *final_detections)
    ):
        assessment = DisorderAssessment(
            "needs_review",
            max(assessment.score, 0.90),
            tuple(dict.fromkeys((*assessment.reasons, "detection_confidence"))),
            assessment.rule_version,
        )
    changed_mask = _robust_difference(base, aligned_final)
    residual_mask = changed_mask & ~explained
    evidence = {
        "method": "semantic_change_accounting",
        "method_version": "sca-0.1",
        "features": asdict(residual),
        "residual_regions": _residual_region_evidence(residual_mask),
        "quality": asdict(quality),
        "matches": [_serialize_match(match) for match in matches],
        "reasons": list(assessment.reasons),
    }
    return AnalysisResult(assessment.level, assessment.score, quality, matches, residual, evidence, residual_mask)


def analyze_sequence(
    baseline: Array,
    finals: Sequence[Array],
    **kwargs,
) -> AnalysisResult:
    """Analyze a short final-capture sequence by selecting its best-quality frame."""

    if not finals:
        raise ValueError("finals must contain at least one image")
    candidates = [
        (image_quality(frame).blur_score - image_quality(frame).clipped_ratio * 100.0, index, frame)
        for index, frame in enumerate(finals)
    ]
    _, selected_index, selected = max(candidates, key=lambda item: item[0])
    result = analyze_pair(baseline, selected, **kwargs)
    evidence = dict(result.evidence)
    evidence["sequence"] = {"candidates": len(finals), "selected_index": selected_index}
    return replace(result, evidence=evidence)
