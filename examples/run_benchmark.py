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
                "disorder_score": round(result.disorder_score, 6),
                "needs_review": result.needs_review,
                "residual_area_ratio": round(result.residual.residual_area_ratio, 6),
                "components": result.residual.connected_components,
            }
        )
    report = {
        "name": "anchordiff-synthetic-smoke",
        "version": "0.6.1",
        "privacy": "synthetic-only",
        "backend": "cpu",
        "rows": rows,
    }
    output = Path("benchmark-report.json")
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"samples": len(rows), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
