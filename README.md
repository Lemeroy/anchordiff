# AnchorDiff

**Reference-conditioned state difference analysis for fixed-camera workspaces.**

AnchorDiff takes a baseline image and one final image (or a short final-capture
sequence) and returns:

- object state transitions such as `returned`, `moved`, `missing`, and `new`;
- a residual change mask after explained object changes are removed;
- a calibrated disorder level and score;
- machine-readable evidence explaining the geometry, quality and rule triggers;
- `needs_review` when the image does not support a reliable claim.

The abstraction is deliberately general: any workspace viewed by a mostly fixed
camera where a set of tracked objects has an expected reference state.

> AnchorDiff is an auditable engineering baseline, not a black-box semantic
> classifier. It reports visible change and uncertainty; it does not infer
> intent, compliance or human performance.

## Why this project exists

Object detectors answer “what is visible now?” and SSIM answers “how similar are
these images?”. Neither alone answers which object changed, which pixels remain
unexplained, or whether the camera moved enough to invalidate the comparison.

AnchorDiff contributes a small but useful abstraction: **Semantic Change
Accounting (SCA)**. Every changed region is assigned to one accounting bucket:

```text
tracked-object state change → explained residual → capture/geometry issue → unknown
```

An explained object movement is removed before residual change is scored, so one
event is not counted twice. The confidence gate can abstain instead of turning
bad alignment or an occluded capture into a confident label.

## Quick start

```bash
cd anchordiff
python -m pip install -e ".[dev]"
python -m anchordiff.cli analyze \
  --before examples/before.png \
  --after examples/after.png \
  --output out
```

The command writes:

```text
out/result.json        # level, score, object states and evidence
out/residual_mask.png  # unexplained changed pixels
out/overlay.png        # final image with residual pixels highlighted
```

Create privacy-safe demo images without downloading a dataset:

```bash
python -m examples.make_demo
```

For detector-backed analysis, pass a portable JSON file:

```json
{
  "before": [{"label": "device_a", "bbox": [20, 20, 80, 55], "confidence": 0.94}],
  "after":  [{"label": "device_a", "bbox": [25, 22, 80, 55], "confidence": 0.93}]
}
```

```bash
python -m anchordiff.cli analyze --before before.png --after after.png \
  --detections detections.json --output out
```

For a fixed-camera setup with four visible anchors, pass matching points in
baseline and final pixel coordinates:

```json
{"before": [[20, 20], [236, 20], [236, 236], [20, 236]],
 "after":  [[22, 21], [238, 21], [238, 237], [22, 237]]}
```

```bash
python -m anchordiff.cli analyze --before before.png --after after.png \
  --anchors anchors.json --output out
```

## Core pipeline

```text
two images / short sequence
  → anchor-aware alignment and geometry confidence
  → quality gate (blur, exposure, clipping)
  → detector adapter (optional YOLO/ONNX/RKNN or JSON boxes)
  → class-aware Hungarian object matching
  → object state transitions
  → robust pixel residual and connected components
  → subtract explained object regions (SCA)
  → monotonic rules + abstention
  → JSON evidence, mask and overlay
```

The core package depends only on NumPy and Pillow. OpenCV is an optional
accelerator/integration dependency. A detector is deliberately an adapter so a
CPU baseline, Ultralytics YOLO, ONNX Runtime or RKNN can share the same state and
evidence contract.

## Output example

```json
{
  "level": "minor",
  "score": 0.23,
  "instruments": [
    {"class": "device_a", "status": "moved", "displacement": 0.18}
  ],
  "evidence": {
    "method": "semantic_change_accounting",
    "features": {
      "residual_area_ratio": 0.021,
      "connected_components": 2,
      "largest_component_ratio": 0.006
    },
    "reasons": ["unexplained_residual"]
  }
}
```

## What makes it different

| Existing building block | What it provides | AnchorDiff adds |
|---|---|---|
| object detection | current objects and boxes | before/after object state transitions |
| SSIM / pixel difference | image-level change | residual change after object explanation |
| image anomaly detection | anomaly score or heatmap | explicit accounting buckets and abstention |
| industrial CV workflow tools | deployable model pipelines | a small, local, model-agnostic evidence contract |

The project does not claim to invent YOLO, SSIM or homography. The original
engineering contribution is the reusable **reference → state transition →
residual accounting → uncertainty** contract, together with deterministic
fixtures and evaluation hooks.

## Development

```bash
pytest -q
```

The intended release sequence is:

1. `v0.1`: two-image CPU baseline, CLI, evidence JSON and residual overlay;
2. `v0.2`: ArUco/AprilTag adapters and normalized homography quality;
3. `v0.3`: detector adapters, object masks and benchmark fixtures;
4. `v0.4`: calibration, risk-coverage metrics and short-sequence selection;
5. `v0.5`: ONNX/RKNN deployment examples and a privacy-safe benchmark.

## Scope and limits

- A fixed or mostly fixed camera is part of the problem definition.
- The baseline image defines the expected workspace state for that comparison.
- Small untracked items are represented as residual visual change, not
  individually identified objects in the CPU baseline.
- Lighting, occlusion, missing anchors and strong camera motion should produce
  `needs_review` rather than an overconfident label.
- Outputs are for inspection and feedback. They do not replace any operational
  or safety checks required by the deployment context.

## License

Apache-2.0. See [LICENSE](LICENSE).
