"""Check the original API and published pickle migration in a release Linux wheel."""

from __future__ import annotations

import argparse
from email.parser import Parser
from collections.abc import Callable
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile
from zipfile import ZipFile

from check_python_sdist import verify_sdist_contents
from check_upstream_tests import verify_tests


def verify_wheel_contents(wheel: Path) -> None:
    """Check the published description, license and package contents."""
    root = Path(__file__).resolve().parents[1]
    with ZipFile(wheel) as archive:
        (metadata,) = (name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
        description = Parser().parsestr(archive.read(metadata).decode("utf-8"))
        if description.get_payload().strip() != (root / "README.md").read_text(encoding="utf-8").strip():
            raise RuntimeError("wheel description differs from the root README")
        licenses = [name for name in archive.namelist() if name.endswith(".dist-info/licenses/LICENSE")]
        if len(licenses) != 1 or archive.read(licenses[0]) != (root / "LICENSE").read_bytes():
            raise RuntimeError("wheel must include the project license")
        required = {"tdigest_rs/_polars.py", "tdigest_rs/_shared.py"}
        if any(name.startswith("gr_tdigest/") for name in archive.namelist()):
            raise RuntimeError("wheel contains the retired gr_tdigest namespace")
        missing = required.difference(archive.namelist())
        if missing:
            raise RuntimeError("wheel omits Python adapter modules: " + ", ".join(sorted(missing)))
        unexpected = sorted(
            name
            for name in archive.namelist()
            if "__pycache__" in PurePosixPath(name).parts or PurePosixPath(name).suffix in {".pyc", ".pyo"}
        )
    if unexpected:
        raise RuntimeError("wheel contains Python cache/bytecode files: " + ", ".join(unexpected))
    print("Release wheel contains no Python cache/bytecode files.", flush=True)


def verify_upstream_snapshot(root: Path) -> None:
    verify_tests(root)


def stage_compatibility_tests(root: Path, temporary: Path) -> tuple[Path, Path]:
    tests = temporary / "tests"
    shutil.copytree(
        root / "tests/upstream_legacy/bindings/python/tests",
        tests,
        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"),
    )
    # Run exact outputs, API regressions, and genuine published pickles from the wheel.
    parity = temporary / "parity"
    (parity / "data").mkdir(parents=True)
    for name in (
        "test_original_numerics.py",
        "test_delta_wheel_parity.py",
        "test_legacy_compatibility.py",
        "test_optional_polars_import.py",
        "test_published_pickle_migration.py",
    ):
        shutil.copyfile(root / "bindings/python/tests" / name, parity / name)
    for name in (
        "upstream_numerical_golden.json",
        "delta_wheel_parity.json",
        "published_0_1_0_pickle_golden.json",
    ):
        shutil.copyfile(root / "bindings/python/tests/data" / name, parity / "data" / name)
    return tests, parity


def build_and_check_sdist(root: Path, output: Path, run: Callable[..., object]) -> Path:
    """Create the real source archive and check that Maturin preserves Cargo.lock."""
    lockfile = root / "Cargo.lock"
    before = lockfile.read_bytes()
    run(
        "uv",
        "run",
        "--project",
        str(root / "bindings/python"),
        "--no-sync",
        "maturin",
        "sdist",
        "-o",
        str(output),
        cwd=root / "bindings/python",
    )
    if lockfile.read_bytes() != before:
        raise RuntimeError("maturin sdist changed Cargo.lock; review the resolution before building")
    (sdist,) = output.glob("*.tar.gz")
    verify_sdist_contents(sdist, root)
    return sdist


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, help="reuse this release wheel without rebuilding")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    verify_upstream_snapshot(root)
    project = root / "bindings/python"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTEST_ADDOPTS", None)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"

    def run(*args: str, cwd: Path = root) -> None:
        subprocess.run(args, cwd=cwd, env=env, check=True)

    with tempfile.TemporaryDirectory(prefix="tdigest-wheel-test-") as directory:
        temporary = Path(directory)
        if args.wheel:
            wheel = args.wheel.resolve(strict=True)
        else:
            build_and_check_sdist(root, temporary / "sdist", run)
            wheels = temporary / "wheels"
            run(
                "uv",
                "run",
                "--no-sync",
                "maturin",
                "build",
                "--release",
                "--locked",
                "-F",
                "python",
                "--compatibility",
                "linux",
                "-o",
                str(wheels),
                cwd=project,
            )
            (wheel,) = wheels.glob("*.whl")
        verify_wheel_contents(wheel)
        requirements = temporary / "requirements.txt"
        run(
            "uv",
            "export",
            "--project",
            str(project),
            "--frozen",
            "--all-groups",
            "--no-emit-project",
            "--no-hashes",
            "--output-file",
            str(requirements),
        )
        venv = temporary / "venv"
        run("uv", "venv", "--python", sys.executable, str(venv))
        python = str(venv / "bin/python")
        run(
            "uv",
            "pip",
            "install",
            "--python",
            python,
            "-r",
            str(requirements),
            str(wheel),
        )
        tests, parity = stage_compatibility_tests(root, temporary)
        # Source checkout must not satisfy these imports.
        run(
            python,
            "-I",
            "-c",
            "import pathlib, sys, tdigest_rs, tdigest_rs._native as native; "
            "prefix = pathlib.Path(sys.prefix).resolve(); "
            "assert pathlib.Path(tdigest_rs.__file__).resolve().is_relative_to(prefix); "
            "assert pathlib.Path(native.__file__).resolve().is_relative_to(prefix); "
            "print('Installed legacy package:', tdigest_rs.__file__); "
            "print('Installed Rust extension:', native.__file__)",
            cwd=temporary,
        )
        run(python, "-I", "-m", "pytest", "-q", str(tests), str(parity), cwd=temporary)


if __name__ == "__main__":
    main()
