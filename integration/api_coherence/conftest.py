from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest
from helpers_cli import (
    assert_close,
    cli_build_args,
    run_cli,
)
from helpers_java import compile_run_java, java_builder_chain

# The CDF counts half the mass at an observed value.
DATA = [0.0, 1.0, 2.0, 3.0]
P = 0.5
X = 2.0
EXPECT_Q50 = 1.5
EXPECT_CDF_AT_2 = 0.625

BASE_CFG = {
    "max_size": 10,
    "precision_cli": "f64",
    "precision_py": "f64",
    "precision_pl": "f64",
    "precision_java": "F64",
    "scale_cli": "k2",
    "scale_py": "k2",
    "scale_pl": "k2",
    "scale_java": "K2",
    "singleton_cli": "use",
    "singleton_py": "use",
    "singleton_pl": "use",
    "singleton_java": "USE",
    "pin_per_side": 2,
}

@dataclass(frozen=True)
class Paths:
    root: Path
    cli_bin: Path
    classes_dir: Path
    native_dirs: list[Path]
    classpath_sep: str



@pytest.fixture(scope="session")
def paths() -> Paths:
    root = Path(__file__).resolve().parents[2]
    profile = os.environ.get("PROFILE", "dev")
    cargo_dir = "release" if profile == "release" else "debug"

    # Match Cargo and the Java resource packager when builds use an external cache.
    target_dir = root / os.environ.get("CARGO_TARGET_DIR", "target")
    cli_bin = target_dir / cargo_dir / ("tdigest.exe" if os.name == "nt" else "tdigest")

    java_src_dir = root / "bindings" / "java"
    gradlew = java_src_dir / ("gradlew.bat" if os.name == "nt" else "gradlew")
    if not gradlew.exists() or not os.access(gradlew, os.X_OK):
        raise AssertionError(f"Gradle wrapper not found or not executable: {gradlew}")

    classes_dir = java_src_dir / "build" / "classes" / "java" / "main"

    force_rebuild = os.environ.get("CI_CLEAN_JAVA", "").lower() in ("1", "true", "yes")
    if force_rebuild or not classes_dir.exists():
        try:
            subprocess.run(
                [str(gradlew), "--no-daemon", "--console=plain", "classes"],
                cwd=java_src_dir,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            raise AssertionError(
                "Gradle 'classes' build failed\n"
                f"cmd: {' '.join(e.cmd)}\n"
                f"stdout:\n{e.stdout}\n"
                f"stderr:\n{e.stderr}\n"
            ) from e

    system = platform.system().lower()
    machine = platform.machine().lower()
    os_tag = "macos" if system == "darwin" else system
    arch = {"amd64": "x86_64", "arm64": "aarch64"}.get(machine, machine)
    sys_tag = f"{os_tag}-{arch}"
    native_dirs = [
        java_src_dir / "build" / "resources" / "main" / "META-INF" / "native" / sys_tag,
        java_src_dir / "build" / "generated-resources" / "META-INF" / "native" / sys_tag,
    ]

    return Paths(
        root=root,
        cli_bin=cli_bin,
        classes_dir=classes_dir,
        native_dirs=native_dirs,
        classpath_sep=os.pathsep,
    )



@pytest.fixture(scope="session")
def expect() -> dict[str, float]:
    eps = 1e-6
    return {"P": P, "X": X, "Q50": EXPECT_Q50, "CDF2": EXPECT_CDF_AT_2, "EPS": eps}


@pytest.fixture(scope="session")
def dataset() -> dict[str, object]:
    return {"DATA": DATA}


@pytest.fixture(scope="session")
def cfg() -> dict[str, object]:
    return dict(BASE_CFG)


@pytest.fixture(scope="session")
def run_cli_fn():
    return run_cli

@pytest.fixture(scope="session")
def assert_close_fn():
    return assert_close

@pytest.fixture(scope="session")
def cli_build_args_fn():
    return cli_build_args

@pytest.fixture(scope="session")
def compile_run_java_fn(paths: Paths):
    return lambda tmp_path, class_name, java_src: compile_run_java(paths, tmp_path, class_name, java_src)


@pytest.fixture(scope="session")
def java_builder_chain_fn():
    return java_builder_chain
