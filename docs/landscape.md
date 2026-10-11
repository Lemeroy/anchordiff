# Similar solutions and the AnchorDiff boundary

This is a lightweight landscape check, not a claim that every commercial
product has been exhaustively tested.

| Solution | Confirmed scope | Boundary with AnchorDiff |
|---|---|---|
| [Roboflow Workflows](https://docs.roboflow.com/workflows) | Visual pipelines assembled from model and processing blocks | A deployment platform; AnchorDiff defines a small reference-to-evidence contract and CPU baseline |
| [OpenVINO Anomalib](https://github.com/open-edge-platform/anomalib) | Open-source visual anomaly detection toolbox | Anomaly scores/heatmaps are useful building blocks, but do not by themselves produce object state transitions or residual accounting |
| [OpenCV image quality examples](https://docs.opencv.org/4.x/d5/dc4/tutorial_video_input_psnr_ssim.html) | Pixel, PSNR and SSIM comparison primitives | Low-level comparison; AnchorDiff adds object-aware subtraction and abstention |
| [AWS Lookout for Vision](https://aws.amazon.com/lookout-for-vision/) | Managed industrial image anomaly detection | Cloud/model-centric inspection; AnchorDiff targets local, fixed-camera before/after state evidence |
| LandingAI / LandingLens | Industrial visual inspection and custom model workflows | Productized inspection; AnchorDiff is an inspectable open baseline and benchmark interface |

The differentiator is the combination rather than any individual primitive:

```text
reference image → object state transition → explained-region subtraction
→ residual disorder evidence → calibrated abstention
```

AnchorDiff should avoid claiming a new detector. Its originality is the
problem abstraction, the accounting invariant, the evidence schema, the
privacy-safe synthetic benchmark and the pluggable CPU/ONNX/RKNN interfaces.
