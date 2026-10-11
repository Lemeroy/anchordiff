# Release checklist

## v0.1 — CPU baseline

- [x] Pure CPU core with no mandatory model download
- [x] Two-image CLI and short-sequence selection
- [x] JSON evidence, residual mask and overlay
- [x] Deterministic privacy-safe synthetic fixture generator
- [x] Optional detector adapter boundary
- [x] Apache-2.0 license, scope limits and security scan

## v0.2 — geometric alignment

- [x] ArUco and AprilTag adapters behind an optional OpenCV extra
- [x] Stable marker-ID pairing and four-corner correspondences
- [x] Normalized homography RMSE, maximum error and conditioning evidence
- [x] Degenerate geometry checks that preserve `needs_review`

## v0.3 — detector and mask contract

- [x] JSON, Ultralytics, ONNX Runtime and RKNN Lite detector adapters
- [x] Common `Detector.detect(image) -> Detection[]` contract
- [x] Optional full-frame object masks with bounded validation
- [x] Mask-aware semantic change accounting with bounding-box fallback
- [x] Deterministic synthetic benchmark fixtures

## v0.4 — calibration and selective risk

- [x] Brier score and expected calibration error
- [x] PAVA isotonic confidence calibration
- [x] Risk-coverage curve for selective acceptance
- [x] Short-sequence frame quality evidence and selected-frame marker
- [x] Per-region residual evidence for report and UI integration

## v0.5 — edge deployment

- [x] Local-model ONNX Runtime deployment example
- [x] Local-model RKNN Lite deployment example
- [x] Privacy-safe synthetic smoke benchmark
- [x] Deployment notes covering backend, preprocessing, decoder and latency
- [ ] Measure latency and peak memory on each target RK3588 board
- [ ] Publish a licensed paired dataset with held-out evaluation metrics

The last two items require target hardware or a separately licensed dataset;
they are deployment validation tasks rather than dependencies of the portable
reference implementation.
