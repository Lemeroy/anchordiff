"""Command-line entry point for the dependency-light AnchorDiff baseline."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .core import Detection, analyze_pair, analyze_sequence
from .sanitize import (
    load_json_payload,
    load_sanitized_rgb,
    validate_anchor_payload,
    validate_detection_payload,
)


def _load_image(path: str) -> np.ndarray:
    return load_sanitized_rgb(path)


def _load_detections(path: str | None) -> tuple[list[Detection], list[Detection]]:
    if not path:
        return [], []
    payload = load_json_payload(path)
    validate_detection_payload(payload)

    def parse(items: list[dict[str, Any]]) -> list[Detection]:
        return [
            Detection(
                str(item["label"]),
                tuple(float(value) for value in item["bbox"]),
                float(item.get("confidence", 1.0)),
            )
            for item in items
        ]

    return parse(payload.get("before", [])), parse(payload.get("after", []))


def _load_anchors(path: str | None) -> tuple[list[list[float]] | None, list[list[float]] | None]:
    if not path:
        return None, None
    payload = load_json_payload(path)
    validate_anchor_payload(payload)
    return payload.get("before"), payload.get("after")


def _jsonable(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_outputs(result, selected: np.ndarray, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "level": result.level,
        "score": result.score,
        "quality": asdict(result.quality),
        "instruments": result.evidence["matches"],
        "evidence": result.evidence,
    }
    (output / "result.json").write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    mask = (result.residual_mask.astype(np.uint8) * 255)
    Image.fromarray(mask, mode="L").save(output / "residual_mask.png")
    overlay = selected.copy()
    overlay[result.residual_mask] = (
        0.35 * overlay[result.residual_mask] + np.array([255, 40, 40]) * 0.65
    ).astype(np.uint8)
    Image.fromarray(overlay, mode="RGB").save(output / "overlay.png")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="anchordiff", description="Explainable fixed-camera before/after analysis")
    commands = parser.add_subparsers(dest="command", required=True)
    analyze = commands.add_parser("analyze", help="compare a baseline with one or more final captures")
    analyze.add_argument("--before", required=True, help="baseline image")
    analyze.add_argument("--after", required=True, nargs="+", help="one final image or a short capture sequence")
    analyze.add_argument("--output", required=True, help="directory for JSON and visual evidence")
    analyze.add_argument("--detections", help="optional JSON with before/after detector outputs")
    analyze.add_argument("--anchors", help="optional JSON with before/after 2D anchor points")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "analyze":
        return 2
    baseline = _load_image(args.before)
    finals = [_load_image(path) for path in args.after]
    before_detections, after_detections = _load_detections(args.detections)
    before_anchors, after_anchors = _load_anchors(args.anchors)
    kwargs = {
        "baseline_detections": before_detections,
        "final_detections": after_detections,
        "baseline_anchors": before_anchors,
        "final_anchors": after_anchors,
    }
    if len(finals) == 1:
        result = analyze_pair(baseline, finals[0], **kwargs)
    else:
        result = analyze_sequence(baseline, finals, **kwargs)
    _write_outputs(result, finals[result.evidence.get("sequence", {}).get("selected_index", 0)], Path(args.output))
    print(json.dumps({"level": result.level, "score": round(result.score, 4)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
