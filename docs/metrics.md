# Calibration and selective-risk metrics

AnchorDiff keeps confidence and correctness evaluation separate from the runtime
rules. The dependency-light `anchordiff.metrics` module provides deterministic
metrics for a labeled evaluation set:

- `brier_score` measures squared confidence error;
- `expected_calibration_error` (ECE) compares confidence and empirical accuracy
  in equal-width bins;
- `fit_isotonic_calibrator` learns a monotone post-hoc mapping with PAVA;
- `risk_coverage_curve` reports error risk as lower-confidence samples are
  progressively accepted.

```python
from anchordiff.metrics import (
    expected_calibration_error,
    fit_isotonic_calibrator,
    risk_coverage_curve,
)

confidence = [0.92, 0.81, 0.44, 0.31]
correct = [True, True, False, True]

calibrator = fit_isotonic_calibrator(confidence, correct)
calibrated = calibrator.transform(confidence)
curve = risk_coverage_curve(calibrated, correct)
print(expected_calibration_error(calibrated, correct))
```

The curve is intended for model and rule comparisons. It does not change the
runtime `needs_review` gate until a deployment chooses and validates a policy.
