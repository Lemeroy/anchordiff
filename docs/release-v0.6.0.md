# AnchorDiff v0.6.0

AnchorDiff v0.6.0 reports a normalized disorder score and explains each tracked
object transition and unexplained residual region with bbox-level evidence.

## Changes

- Replaced categorical `level` results with `disorder_score` in `[0, 1]` and
  the explicit `needs_review` reliability gate. When evidence is unreliable,
  `disorder_score` is `null`.
- Set normalized center-displacement tolerance to `0.1` by default. Configure
  it through the Python API or CLI `--position-threshold` option.
- Added object `action`, `explanation`, baseline/final bboxes and the applied
  tolerance to detector evidence.
- Added `unexplained_change` actions and explanations to residual-region bboxes.
- Generalized fixture labels and documentation to use domain-neutral objects.

## Compatibility

Consumers of `result.json` should read `disorder_score`, `needs_review`,
`objects[].action` and `objects[].explanation`. The categorical `level` field
and CLI `score` field have been removed. `AnalysisResult.score` remains as a
Python compatibility alias for `disorder_score`.

The score is a versioned rule-based comparison signal, not a calibrated
probability or universal cleanliness scale. A real deployment should calibrate
its thresholds against a representative, authorized dataset.

## Verification

- 37 tests passed.
- The 10-sample synthetic CPU smoke benchmark completed.
- Credential-signature and EXIF security scan passed.
