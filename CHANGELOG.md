# Changelog

## 0.6.0 - 2026-10-12

- Replaced categorical `level` output with numeric `disorder_score` and an
  explicit `needs_review` quality gate.
- Generalized object labels and examples so the public project carries no
  domain-specific equipment vocabulary.
- Added per-object actions, bbox explanations and configurable position
  tolerance, defaulting to `0.1` normalized center displacement.
- Added residual-region actions and explanations for every unexplained bbox.

## 0.5.0 - 2026-10-11

- Added optional ArUco and AprilTag marker adapters with normalized homography
  quality evidence and degenerate-geometry checks.
- Added ONNX Runtime and RKNN Lite detector adapters, object-mask accounting and
  local-model edge deployment examples.
- Added Brier/ECE metrics, PAVA confidence calibration, risk-coverage curves and
  short-sequence quality evidence.
- Added structured residual-region evidence and a privacy-safe synthetic smoke
  benchmark report.

## 0.1.0 - 2026-10-11

- Added the dependency-light AnchorDiff core pipeline.
- Added Hungarian object matching, robust residual accounting and abstention.
- Added a short-sequence selector, JSON/Ultralytics detector adapters and CLI.
- Added deterministic privacy-safe synthetic fixtures and regression tests.
