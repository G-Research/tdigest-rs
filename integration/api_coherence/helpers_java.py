from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


def java_builder_chain(
    cfg: dict[str, object],
    *,
    precision: str | None = None,
    max_size: int | None = None,
) -> str:
    """
    Return Java TDigest builder chain lines for the test configuration.
    """
    prec = str(precision or cfg["precision_java"])
    ms = int(max_size if max_size is not None else cfg["max_size"])

    lines = [
        f".maxSize({ms})",
        f".scale(Scale.{cfg['scale_java']})",
        f".singletonPolicy(SingletonPolicy.{cfg['singleton_java']})",
        f".precision(Precision.{prec})",
    ]
    if (
        str(cfg.get("singleton_java")) == "USE_WITH_PROTECTED_EDGES"
        and cfg.get("pin_per_side") is not None
    ):
        lines.append(f".edgesPerSide({int(cfg['pin_per_side'])})")
    return "\n                        ".join(lines)


def compile_run_java(paths: Any, tmp_path: Path, class_name: str, java_src: str) -> str:
    """
    Compile an in-memory Java class against built bindings and run it.
    Returns stdout stripped.
    """
    src = tmp_path / f"{class_name}.java"
    src.write_text(java_src)

    subprocess.run(
        ["javac", "-cp", str(paths.classes_dir), str(src)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    native_dir = next((p for p in paths.native_dirs if p.exists()), None)
    assert native_dir is not None, "gradle native dir missing"

    classpath = f".{paths.classpath_sep}{paths.classes_dir}"
    return subprocess.check_output(
        ["java", f"-Djava.library.path={native_dir}", "-cp", classpath, class_name],
        cwd=tmp_path,
        text=True,
    ).strip()
