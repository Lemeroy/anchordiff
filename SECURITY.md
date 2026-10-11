# Security policy

AnchorDiff is intended for local or controlled deployments. Do not commit
images containing faces, student identifiers, employee identifiers or other
personal data. The project does not provide authentication, authorization or
secure image deletion; applications embedding it must add those controls.

The CLI converts inputs to plain RGB arrays and does not copy EXIF metadata to
outputs. Detector JSON is bounded and rejects non-finite coordinates, invalid
labels and oversized input. The optional Ultralytics adapter requires local
weights by default; it does not silently download a model. Run
`python scripts/security_scan.py` before a release to check tracked files for
common credential signatures and image EXIF.

Please report security issues privately to the repository maintainers rather
than opening a public issue with sensitive details.
