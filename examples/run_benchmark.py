"""Run the privacy-safe smoke benchmark used in the release README."""

from __future__ import annotations

import json
from pathlib import Path

from anchordiff.core import analyze_pair
from anchordiff.synthetic import make_fixture


def main() -> None:
    rows = []
    for seed in range(10):
        fixture = make_fixture(seed=seed, size=(128, 128), clutter=True)
        result = analyze_pair(fixture.before, fixture.after)
        rows.append(
            {
                "seed": seed,
                "level": result.level,
                "score": round(result.score, 6),
                "residual_area_ratio": round(result.residual.residual_area_ratio, 6),
                "components": result.residual.connected_components,
            }
        )
    report = {"name": "anchordiff-synthetic-smoke", "version": "0.1.0", "rows": rows}
    output = Path("benchmark-report.json")
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"samples": len(rows), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
