from pathlib import Path

from PIL import Image

from anchordiff.cli import main


def test_cli_analyze_writes_machine_readable_evidence(tmp_path: Path):
    before = Image.new("RGB", (64, 64), "white")
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
