# Contributing to AnchorDiff

AnchorDiff is a small, evidence-first project. Contributions should preserve
the core contract: a fixed-camera comparison must expose its state transitions,
residual evidence and uncertainty.

Before opening a pull request:

1. Add or update a deterministic test for behavior changes.
2. Run `pytest -q` from this directory.
3. Keep optional detector integrations out of the dependency-light core.
4. Include a short note about runtime, memory and any new data license.
5. Do not add real student, employee or customer images. Use synthetic or
   explicitly licensed privacy-safe fixtures.

New detector adapters should return `Detection` objects and must not change the
meaning of `needs_review`.
