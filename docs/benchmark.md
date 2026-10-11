# Benchmark contract

The repository ships a privacy-safe smoke benchmark, not a claim of production
accuracy. Run it from the project root:

```bash
python -m examples.run_benchmark
```

It creates `benchmark-report.json` from ten deterministic synthetic pairs. The
fixture generator applies a small tracked-object translation and residual regions,
so regression tests can verify that the pipeline remains deterministic.

For a real release, add a licensed paired dataset with the following labels:

- object state: returned, moved, missing, new, uncertain;
- residual region polygons or masks;
- capture quality and anchor reprojection error;
- ordinal disorder label and annotator agreement.

Report object-state macro F1, residual IoU and area error, rank correlation with
human ordering, risk-coverage for `needs_review`, and CPU/RK3588 latency. Keep
training, validation and test splits separated by session or workspace so the
same physical scene does not leak across splits.
