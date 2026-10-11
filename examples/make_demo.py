from pathlib import Path
import json

from anchordiff.synthetic import make_fixture


def main() -> None:
    root = Path(__file__).resolve().parent
    fixture = make_fixture(seed=11, size=(256, 256), clutter=True)
    from PIL import Image

    Image.fromarray(fixture.before).save(root / "before.png")
    Image.fromarray(fixture.after).save(root / "after.png")
    (root / "fixture.json").write_text(json.dumps(fixture.metadata, indent=2), encoding="utf-8")
    print(f"wrote {root / 'before.png'} and {root / 'after.png'}")


if __name__ == "__main__":
    main()
