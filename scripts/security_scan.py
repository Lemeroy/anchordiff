"""Small repository hygiene scan for accidental credentials and image metadata."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys

from PIL import Image


SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{20,}\b"),
    re.compile(r"(?i)\b(?:aws_secret_access_key|api_key|access_token)\s*[:=]\s*['\"][^'\"]{12,}['\"]"),
)
TEXT_SUFFIXES = {".py", ".toml", ".md", ".json", ".yaml", ".yml", ".cff", ".txt", ".ini", ".cfg", ".sh"}


def tracked_files(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True
    )
    return [root / item for item in result.stdout.decode("utf-8").split("\0") if item]


def scan(root: Path) -> list[str]:
    findings: list[str] = []
    for path in tracked_files(root):
        if not path.is_file():
            continue
        if path.suffix.lower() in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="replace")
            for pattern in SECRET_PATTERNS:
                if pattern.search(text):
                    findings.append(f"credential-like pattern in {path.relative_to(root)}")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}:
            try:
                with Image.open(path) as image:
                    if image.getexif():
                        findings.append(f"EXIF metadata in {path.relative_to(root)}")
            except Exception as exc:
                findings.append(f"unreadable image {path.relative_to(root)}: {exc}")
    return findings


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    findings = scan(root)
    if findings:
        print("security scan failed:")
        print("\n".join(f"- {finding}" for finding in findings))
        return 1
    print("security scan passed: no credential signatures or EXIF metadata found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
