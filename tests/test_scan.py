from pathlib import Path

from scripts.security_scan import scan


def test_repository_has_no_credential_signatures_or_image_exif():
    findings = scan(Path(__file__).resolve().parents[1])
    assert findings == []
