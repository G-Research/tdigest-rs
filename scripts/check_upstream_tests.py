"""Verify the complete, unmodified upstream Python test/benchmark snapshot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ORIGINAL_TEST_CASES = 91
UPSTREAM_COMMIT = "8d9cf1fe82c5c9254e5dc9593f01717e317f4333"


def verify_tests(root: Path) -> None:
    snapshot = root / "tests/upstream_legacy"
    manifest = json.loads((snapshot / "manifest.json").read_text())
    if manifest["commit"] != UPSTREAM_COMMIT or manifest["test_cases"] != ORIGINAL_TEST_CASES:
        raise RuntimeError("the pinned original-suite revision or test count changed")
    if manifest["adaptations"]:
        raise RuntimeError("the original suite must have no source adaptations")
    expected = {entry["path"]: entry["sha256"] for entry in manifest["files"]}
    actual = {
        path.relative_to(snapshot).as_posix()
        for path in (snapshot / "bindings/python").rglob("*.py")
        if "__pycache__" not in path.parts
    }
    if len(expected) != 7 or len(manifest["files"]) != 7 or actual != set(expected):
        raise RuntimeError("upstream test/benchmark file inventory changed")
    for relative, checksum in expected.items():
        path = snapshot / relative
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
            raise RuntimeError(f"original upstream test/support file changed: {relative}")
    print(
        "All seven original upstream test/support and benchmark files match their pinned bytes.",
        flush=True,
    )


if __name__ == "__main__":
    verify_tests(Path(__file__).resolve().parents[1])
