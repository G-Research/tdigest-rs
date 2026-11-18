"""Run installed Python interfaces outside the checkout on each CI platform.

The interpreter must already contain the built project and locked development
dependencies. This script does not build, install, or download anything.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from test_legacy_wheel import stage_compatibility_tests, verify_upstream_snapshot
from check_upstream_tests import ORIGINAL_TEST_CASES
from test_python_support import (
    COMPATIBILITY_CASES,
    MODERN_CASES,
    MODERN_TESTS,
    check_results,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    verify_upstream_snapshot(root)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTEST_ADDOPTS", None)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    with tempfile.TemporaryDirectory(prefix="tdigest-platform-tests-") as directory:
        work = Path(directory).resolve()
        config = work / "pytest.ini"
        config.write_text("[pytest]\n", encoding="utf-8")
        tests, parity = stage_compatibility_tests(root, work)
        modern = work / "modern"
        modern.mkdir()
        for relative in MODERN_TESTS:
            shutil.copyfile(root / relative, modern / Path(relative).name)
        subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                "from importlib.metadata import version; import tdigest_rs; "
                "assert tdigest_rs.TDigest.__module__ == 'tdigest_rs.lib'; "
                "assert tdigest_rs.__version__ == version('tdigest-rs')",
            ],
            cwd=work,
            env=env,
            check=True,
        )
        for name, paths, expected in (
            ("upstream_legacy", [tests], ORIGINAL_TEST_CASES),
            ("compatibility", [parity], COMPATIBILITY_CASES - ORIGINAL_TEST_CASES),
            ("modern", [modern], MODERN_CASES),
        ):
            xml = work / f"{name}.xml"
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
                    str(xml),
                    *(str(path) for path in paths),
                ],
                cwd=work,
                env=env,
                check=True,
            )
            check_results(xml, expected)


if __name__ == "__main__":
    main()
