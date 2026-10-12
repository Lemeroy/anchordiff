"""Run AnchorDiff with a local ONNX Runtime or RKNN Lite detector."""

from __future__ import annotations

import argparse
from pathlib import Path

from anchordiff.cli import _write_outputs
from anchordiff.core import POSITION_TOLERANCE, analyze_pair
from anchordiff.detectors import ONNXDetector, RKNNDetector
from anchordiff.sanitize import load_sanitized_rgb


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run AnchorDiff with an optional edge detector")
    parser.add_argument("--backend", choices=("onnx", "rknn"), required=True)
    parser.add_argument("--model", required=True, help="local ONNX or RKNN model path")
    parser.add_argument("--before", required=True, help="baseline image")
    parser.add_argument("--after", required=True, help="final image")
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--classes", default="", help="comma-separated class names")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--position-threshold", type=float, default=POSITION_TOLERANCE)
    parser.add_argument("--input-size", type=int, nargs=2, default=(640, 640), metavar=("WIDTH", "HEIGHT"))
    return parser


def _make_detector(args: argparse.Namespace):
    class_names = tuple(name.strip() for name in args.classes.split(",") if name.strip())
    options = {
        "weights": args.model,
        "class_names": class_names,
        "confidence": args.confidence,
        "input_size": tuple(args.input_size),
    }
    if args.backend == "onnx":
        return ONNXDetector(**options)
    return RKNNDetector(**options)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    baseline = load_sanitized_rgb(args.before)
    final = load_sanitized_rgb(args.after)
    detector = _make_detector(args)
    try:
        before_detections = detector.detect(baseline, "before")
        final_detections = detector.detect(final, "after")
        result = analyze_pair(
            baseline,
            final,
            baseline_detections=before_detections,
            final_detections=final_detections,
            position_tolerance=args.position_threshold,
        )
        _write_outputs(result, final, Path(args.output))
    finally:
        close = getattr(detector, "close", None)
        if callable(close):
            close()
    score = "null" if result.disorder_score is None else f"{result.disorder_score:.4f}"
    print(f"disorder_score={score} needs_review={result.needs_review}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
