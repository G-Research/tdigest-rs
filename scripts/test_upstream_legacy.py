"""Run the untouched original suite separately and reject missing/skipped cases."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from check_upstream_tests import ORIGINAL_TEST_CASES
from test_legacy_wheel import verify_upstream_snapshot
from test_python_support import check_results


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    verify_upstream_snapshot(root)
    report = root / "build/ci/upstream-legacy.xml"
    report.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTEST_ADDOPTS", None)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    with tempfile.TemporaryDirectory(prefix="tdigest-original-suite-") as directory:
        work = Path(directory).resolve()
        tests = work / "tests"
        shutil.copytree(
            root / "tests/upstream_legacy/bindings/python/tests",
            tests,
            ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"),
        )
        config = work / "pytest.ini"
        config.write_text("[pytest]\n")
        subprocess.run(
            [
                sys.executable,
                "-I",
                "-m",
                "pytest",
                "-q",
                "-c",
                str(config),
                "--rootdir",
                str(work),
                "-p",
                "no:cacheprovider",
                "-o",
                "xfail_strict=true",
                "--junitxml",
                str(report),
                str(tests),
            ],
            cwd=work,
            env=env,
            check=True,
        )
    check_results(report, ORIGINAL_TEST_CASES)
    print(f"PASS: all {ORIGINAL_TEST_CASES} untouched upstream cases; no failures, skips or xfails.")


if __name__ == "__main__":
    main()
