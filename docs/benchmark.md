# Benchmark contract

The repository ships a privacy-safe smoke benchmark, not a claim of production
accuracy. Run it from the project root:

```bash
python -m examples.run_benchmark
```

It creates `benchmark-report.json` from ten deterministic synthetic pairs. The
fixture generator applies a small tracked-object translation and residual regions,
so regression tests can verify that the pipeline remains deterministic. The report
declares `privacy: synthetic-only` and `backend: cpu`; no camera frames or model
weights are needed.

For a real release, add a licensed paired dataset with the following labels:

- object state: returned, moved, missing, new, uncertain;
- residual region polygons or masks;
- capture quality and anchor reprojection error;
- ordinal disorder label and annotator agreement.

Report object-state macro F1, residual IoU and area error, rank correlation with
human ordering, risk-coverage for `needs_review`, and CPU/RK3588 latency. Keep
training, validation and test splits separated by session or workspace so the
same physical scene does not leak across splits.

## Detector adapter fixtures (v0.3)

Detector integrations share the small `Detector.detect(image) -> Detection[]`
contract. `Detection` carries a label, `xywh` box, confidence, and an optional
full-frame boolean `mask`. The JSON adapter is useful for deterministic fixtures;
Ultralytics, ONNX Runtime, and RKNN Lite are optional adapters and are imported
only when constructed. ONNX and RKNN accept a decoder callback for model-specific
output heads, while the default decoder consumes rows of
`[x0, y0, x1, y1, score, class_id]`.

Object masks are part of the SCA evidence path. For a moved object, the union of
its before and after masks is marked as explained; changed pixels outside those
masks remain residual regions. A detector that emits only boxes continues to work
through the bounded-box fallback. JSON mask fixtures must be rectangular, binary,
and match the analyzed image dimensions; oversized or malformed masks are
rejected before analysis.

Optional dependencies are intentionally excluded from the CPU baseline:

```bash
pip install -e ".[onnx]"       # ONNX Runtime
pip install -e ".[rknn]"        # RKNN Lite (on a supported Rockchip image)
pip install -e ".[ultralytics]" # Ultralytics segmentation/detection
```

For benchmark records, include the detector backend, model identifier (without
private paths), confidence threshold, mask availability, and post-processing
decoder version. Do not commit model weights, camera frames, EXIF, or user
identifiers to the public fixture set.
