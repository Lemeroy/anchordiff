# AnchorDiff v0.6.1

AnchorDiff v0.6.1 is the current documentation and packaging refresh for the
generic fixed-camera object-analysis workflow.

## Included

- Normalized `disorder_score` with an explicit `needs_review` quality gate.
- Generic object labels and actions: stable, moved, missing and new.
- Per-object `action`, `explanation`, baseline/final bbox and applied position
  tolerance.
- Residual-region `unexplained_change` actions and explanations.
- Default normalized position tolerance of `0.1`, configurable through the API
  or `--position-threshold`.
- Updated README, changelog, benchmark metadata and release documentation.

## Verification

- 37 tests passed.
- The 10-sample synthetic CPU smoke benchmark completed.
- Python compilation checks passed.
- Credential-signature and EXIF security scan passed.
