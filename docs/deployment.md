# Edge deployment examples

The CPU baseline remains the reference implementation. `examples.run_edge`
shows how to replace only the detector adapter while keeping matching, mask
accounting, residual evidence and abstention unchanged.

Install one optional backend and provide a local model file:

```bash
python -m pip install -e ".[onnx]"
python -m examples.run_edge \
  --backend onnx \
  --model models/detector.onnx \
  --classes device_a,device_b \
  --before examples/before.png \
  --after examples/after.png \
  --output out
```

On a supported Rockchip image:

```bash
python -m pip install -e ".[rknn]"
python -m examples.run_edge \
  --backend rknn \
  --model models/detector.rknn \
  --classes device_a,device_b \
  --before examples/before.png \
  --after examples/after.png \
  --output out
```

The adapters accept the common row format
`[x0, y0, x1, y1, score, class_id]`. Models with another output head can pass
a decoder callback to `ONNXDetector` or `RKNNDetector`. Model weights stay
outside the repository and are never downloaded by the example.

For a deployment report, record backend, model identifier, input size,
confidence threshold, preprocessing version, decoder version, latency and peak
memory. Keep camera frames and model paths out of public benchmark artifacts.
