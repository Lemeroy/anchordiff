# AnchorDiff

[![Release](https://img.shields.io/github/v/release/Lemeroy/anchordiff?sort=semver)](https://github.com/Lemeroy/anchordiff/releases)
[![Tests](https://github.com/Lemeroy/anchordiff/actions/workflows/test.yml/badge.svg)](https://github.com/Lemeroy/anchordiff/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**Explainable before/after state analysis for fixed-camera workspaces.**

AnchorDiff compares a reference capture with one or more later captures. It
reports which tracked objects were returned, moved, missing or newly visible,
which visual changes remain unexplained, and why the result was accepted or
marked `needs_review`.

The project is deliberately general. A workspace can be a storage area,
inspection surface, robot cell or any other scene where the camera is fixed or
mostly fixed and the expected state is represented by a reference image.

> AnchorDiff reports visible change and uncertainty. It does not infer intent,
> compliance or human performance, and it does not replace operational checks.

## What the v0.5.0 release contains

- CPU-first two-image and short-sequence analysis with JSON evidence and visual
  overlays.
- ArUco and AprilTag adapters with marker-ID pairing and normalized homography
  quality evidence.
- JSON, Ultralytics, ONNX Runtime and RKNN Lite detector adapters behind one
  `Detection` contract.
- Optional full-frame object masks for more precise semantic change accounting.
- Brier score, expected calibration error, PAVA isotonic calibration and
  risk-coverage utilities.
- Local-model ONNX and RKNN edge deployment examples.
- A deterministic synthetic smoke benchmark that contains no camera frames or
  model weights.

See the [v0.5.0 release](https://github.com/Lemeroy/anchordiff/releases/tag/v0.5.0)
for the release notes.

## How the analysis works

AnchorDiff uses **Semantic Change Accounting (SCA)** to keep one visual change
from being counted twice:

```text
reference + final capture(s)
  → marker or point-based alignment
  → geometry and image-quality gate
  → optional object detection
  → class-aware Hungarian matching
  → object state transitions
  → explained masks or boxes
  → robust pixel difference
  → residual connected components
  → rule-based score and confidence gate
  → JSON evidence, residual mask and numbered overlay
```

The accounting invariant is:

```text
changed pixels = explained object change + residual change + capture/unknown
```

If blur, clipping, occlusion, missing markers, poor detection confidence or
strong camera motion makes the evidence unreliable, the result is
`needs_review` instead of an overconfident label.

## Install

The CPU baseline supports Python 3.10 or newer:

```bash
python -m pip install -e ".[dev]"
```

Optional integrations are isolated from the core:

| Extra | Provides | Install |
|---|---|---|
| `opencv` | ArUco and AprilTag marker detection | `python -m pip install -e ".[opencv]"` |
| `ultralytics` | YOLO detection or segmentation | `python -m pip install -e ".[ultralytics]"` |
| `onnx` | ONNX Runtime inference | `python -m pip install -e ".[onnx]"` |
| `rknn` | RKNN Lite inference on supported Rockchip images | `python -m pip install -e ".[rknn]"` |

Model weights are local inputs. Adapters do not download weights by default.

## Quick start

Generate privacy-safe demo images and compare them:

```bash
python -m examples.make_demo
python -m anchordiff.cli analyze \
  --before examples/before.png \
  --after examples/after.png \
  --output out
```

The command creates:

```text
out/result.json        # state transitions, quality and evidence
out/residual_mask.png  # unexplained changed pixels
out/overlay.png        # final capture with residual regions numbered
```

Pass more than one `--after` image to select the best-quality frame from a
short capture sequence:

```bash
python -m anchordiff.cli analyze \
  --before examples/before.png \
  --after capture-01.png capture-02.png capture-03.png \
  --output out
```

The JSON evidence records the quality and selection score for every candidate.

## Detector input

For a portable detector fixture, provide before/after boxes in JSON:

```json
{
  "before": [
    {"label": "device_a", "bbox": [20, 20, 80, 55], "confidence": 0.94}
  ],
  "after": [
    {"label": "device_a", "bbox": [25, 22, 80, 55], "confidence": 0.93}
  ]
}
```

```bash
python -m anchordiff.cli analyze \
  --before examples/before.png \
  --after examples/after.png \
  --detections examples/detections.json \
  --output out
```

The detector boundary is:

```python
detector.detect(image)  # → Sequence[Detection]
```

Each `Detection` has a label, an `xywh` bounding box, a confidence and an
optional full-frame boolean mask. A mask is used for SCA when available; a
bounded box remains the fallback for box-only detectors.

## Alignment and markers

Hand-authored corresponding points can be supplied as JSON:

```json
{
  "before": [[20, 20], [236, 20], [236, 236], [20, 236]],
  "after":  [[22, 21], [238, 21], [238, 237], [22, 237]]
}
```

```bash
python -m anchordiff.cli analyze \
  --before examples/before.png \
  --after examples/after.png \
  --anchors examples/anchors.json \
  --output out
```

With the optional OpenCV extra, `OpenCVArucoDetector` and
`OpenCVAprilTagDetector` return marker IDs and four corners. Common marker IDs
are paired deterministically and passed to the dependency-free homography
estimator:

```python
from anchordiff import OpenCVArucoDetector, analyze_pair, observations_to_points
from anchordiff.sanitize import load_sanitized_rgb

before = load_sanitized_rgb("before.png")
after = load_sanitized_rgb("after.png")
detector = OpenCVArucoDetector()
before_markers = detector.detect(before)
after_markers = detector.detect(after)
after_points, before_points, _ = observations_to_points(after_markers, before_markers)

result = analyze_pair(
    before,
    after,
    baseline_anchors=before_points,
    final_anchors=after_points,
)
```

The evidence includes pixel RMSE, normalized RMSE, maximum reprojection error,
conditioning and a degenerate-geometry flag.

## Output contract

`result.json` contains a top-level `objects` list, residual region records and
the full evidence object:

```json
{
  "level": "minor",
  "score": 0.23,
  "objects": [
    {"class": "device_a", "status": "moved", "displacement": 0.18}
  ],
  "residual_regions": [
    {
      "id": 1,
      "bbox": [120, 80, 24, 18],
      "area_px": 286,
      "area_ratio": 0.0044,
      "centroid": [131.5, 88.2]
    }
  ],
  "evidence": {
    "method": "semantic_change_accounting",
    "method_version": "sca-0.2",
    "reasons": ["unexplained_residual"]
  }
}
```

Residual regions are connected components sorted by pixel area. Their bounding
boxes use `[x, y, width, height]` coordinates, and the same IDs are drawn in
`overlay.png` so a UI can link a JSON record to a visible region.

## Calibration and edge deployment

The dependency-light `anchordiff.metrics` module provides:

- `brier_score`;
- `expected_calibration_error`;
- `fit_isotonic_calibrator` using PAVA;
- `risk_coverage_curve` for selective acceptance analysis.

Read the [metrics contract](docs/metrics.md) and the [edge deployment guide](docs/deployment.md)
for examples. The deployment example accepts local ONNX or RKNN weights and
the common output row format `[x0, y0, x1, y1, score, class_id]`.

## Benchmark and verification

Run the deterministic synthetic smoke benchmark:

```bash
python -m examples.run_benchmark
```

The report declares `privacy: synthetic-only` and `backend: cpu`. It is a
regression fixture, not a production accuracy claim.

Run the full project checks:

```bash
pytest -q
python scripts/security_scan.py
```

The v0.5.0 release was published with 34 passing tests and a successful
credential/EXIF scan.

## Release sequence

| Version | Delivered |
|---|---|
| v0.1 | CPU baseline, CLI, evidence JSON and residual overlay |
| v0.2 | ArUco/AprilTag adapters and normalized homography quality |
| v0.3 | Detector adapters, object masks and benchmark fixtures |
| v0.4 | Calibration, risk-coverage metrics and short-sequence evidence |
| v0.5 | ONNX/RKNN deployment examples and privacy-safe benchmark |

## Scope and privacy

- The comparison assumes a fixed or mostly fixed camera and a reference image
  that represents the expected state.
- Small untracked items are represented as residual visual change rather than
  individually identified objects in the CPU baseline.
- Inputs are converted to plain RGB arrays; EXIF metadata is not copied to
  outputs. Do not commit images containing personal or identifying data.
- Lighting, occlusion, missing anchors and strong camera motion can produce
  `needs_review`.
- Target-board latency and peak-memory measurements, plus a separately licensed
  paired dataset, are deployment validation tasks for future releases.

AnchorDiff reports evidence for inspection and feedback. It does not replace
operational or safety checks required by a deployment context.

## Documentation

- [Algorithm note](docs/algorithm.md)
- [Benchmark contract](docs/benchmark.md)
- [Calibration metrics](docs/metrics.md)
- [Edge deployment](docs/deployment.md)
- [Release checklist](docs/release-checklist.md)
- [Security policy](SECURITY.md)

## License

Apache-2.0. See [LICENSE](LICENSE).
