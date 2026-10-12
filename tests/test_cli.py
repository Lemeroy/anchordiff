from pathlib import Path
import json

from PIL import Image

from anchordiff.cli import main


def test_cli_analyze_writes_machine_readable_evidence(tmp_path: Path):
    before = Image.new("RGB", (64, 64), (120, 120, 120))
    after = before.copy()
    pixels = after.load()
    for x in range(40, 55):
        for y in range(40, 50):
            pixels[x, y] = (20, 20, 20)
    before_path, after_path = tmp_path / "before.png", tmp_path / "after.png"
    before.save(before_path)
    after.save(after_path)
    output = tmp_path / "out"

    exit_code = main(["analyze", "--before", str(before_path), "--after", str(after_path), "--output", str(output)])

    assert exit_code == 0
    assert (output / "result.json").exists()
    assert (output / "residual_mask.png").exists()
    report = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert "residual_regions" in report["evidence"]
    assert "objects" in report
    assert "instruments" not in report
    assert "level" not in report
    assert 0.0 <= report["disorder_score"] <= 1.0
    assert "needs_review" in report
