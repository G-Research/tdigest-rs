from __future__ import annotations

import subprocess
import importlib.util
import os
from pathlib import Path
from typing import Any


def java_builder_chain(
    cfg: dict[str, object],
    *,
    precision: str | None = None,
) -> str:
    """
    Return Java TDigest builder chain lines for the test configuration.
    """
    prec = str(precision or cfg["precision_java"])

    lines = [
        f".maxSize({cfg['max_size']})",
        f".scale(Scale.{cfg['scale_java']})",
        f".singletonPolicy(SingletonPolicy.{cfg['singleton_java']})",
        f".precision(Precision.{prec})",
    ]
    return "\n                        ".join(lines)


def compile_run_java(paths: Any, tmp_path: Path, class_name: str, java_src: str) -> str:
    """
    Compile an in-memory Java class against built bindings and run it.
    Returns stdout stripped.
    """
    src = tmp_path / f"{class_name}.java"
    src.write_text(java_src)

    # These subprocesses do not go through Gradle. Honor the exact same JDK
    # selection, including hosts with no Java installation on PATH.
    spec = importlib.util.spec_from_file_location("tdigest_jvm_tools", paths.root / "bindings/java/bootstrap.py")
    assert spec is not None and spec.loader is not None, "JVM bootstrap module missing"
    bootstrap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bootstrap)
    env = bootstrap.java_environment(os.environ.copy())
    home = Path(env["JAVA_HOME"]) / "bin" if env.get("JAVA_HOME") else None
    javac = str(home / ("javac.exe" if os.name == "nt" else "javac")) if home else "javac"
    java = str(home / ("java.exe" if os.name == "nt" else "java")) if home else "java"

    subprocess.run(
        [javac, "-cp", str(paths.classes_dir), str(src)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    native_dir = next((p for p in paths.native_dirs if p.exists()), None)
    assert native_dir is not None, "gradle native dir missing"

    classpath = f".{paths.classpath_sep}{paths.classes_dir}"
    return subprocess.check_output(
        [java, f"-Djava.library.path={native_dir}", "-cp", classpath, class_name],
        cwd=tmp_path,
        text=True,
        env=env,
    ).strip()
