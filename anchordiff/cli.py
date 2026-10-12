"""Command-line entry point for the dependency-light AnchorDiff baseline."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .core import POSITION_TOLERANCE, Detection, analyze_pair, analyze_sequence
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
        "disorder_score": result.disorder_score,
        "needs_review": result.needs_review,
        "quality": asdict(result.quality),
        "objects": result.evidence["matches"],
        "residual_regions": result.evidence["residual_regions"],
        "evidence": result.evidence,
    }
    (output / "result.json").write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    mask = (result.residual_mask.astype(np.uint8) * 255)
    Image.fromarray(mask, mode="L").save(output / "residual_mask.png")
    overlay = selected.copy()
    overlay[result.residual_mask] = (
        0.35 * overlay[result.residual_mask] + np.array([255, 40, 40]) * 0.65
    ).astype(np.uint8)
    overlay_image = Image.fromarray(overlay, mode="RGB")
    draw = ImageDraw.Draw(overlay_image)
    for region in result.evidence["residual_regions"]:
        x, y, width, height = region["bbox"]
        draw.rectangle((x, y, x + width - 1, y + height - 1), outline=(255, 255, 0), width=1)
        draw.text((x, y), str(region["id"]), fill=(255, 255, 0))
    overlay_image.save(output / "overlay.png")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="anchordiff", description="Explainable fixed-camera before/after analysis")
    commands = parser.add_subparsers(dest="command", required=True)
    analyze = commands.add_parser("analyze", help="compare a baseline with one or more final captures")
    analyze.add_argument("--before", required=True, help="baseline image")
    analyze.add_argument("--after", required=True, nargs="+", help="one final image or a short capture sequence")
    analyze.add_argument("--output", required=True, help="directory for JSON and visual evidence")
    analyze.add_argument("--detections", help="optional JSON with before/after detector outputs")
    analyze.add_argument("--anchors", help="optional JSON with before/after 2D anchor points")
    analyze.add_argument(
        "--position-threshold",
        type=float,
        default=POSITION_TOLERANCE,
        help="normalized center displacement accepted as approximately unchanged (default: 0.1)",
    )
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
        "position_tolerance": args.position_threshold,
    }
    if len(finals) == 1:
        result = analyze_pair(baseline, finals[0], **kwargs)
    else:
        result = analyze_sequence(baseline, finals, **kwargs)
    _write_outputs(result, finals[result.evidence.get("sequence", {}).get("selected_index", 0)], Path(args.output))
    print(
        json.dumps(
            {
                "disorder_score": round(result.disorder_score, 4) if result.disorder_score is not None else None,
                "needs_review": result.needs_review,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
