from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable


def assert_close(a: float, b: float, eps: float) -> None:
    assert abs(a - b) <= eps, f"{a} != {b} (eps={eps})"


def run_cli(cli_bin: Path, args: list[str], data: Iterable[float]) -> str:
    return (
        subprocess.check_output(
            [str(cli_bin), *args],
            input=(" ".join(str(v) for v in data)).encode(),
        )
        .decode()
        .strip()
    )


def cli_build_args(cfg: dict) -> list[str]:
    """Common CLI args (add command + stdin/probe flags in tests)."""
    return [
        "--no-header",
        "--output",
        "csv",
        "--max-size",
        str(cfg["max_size"]),
        "--scale",
        cfg["scale_cli"],
        "--singleton-policy",
        cfg["singleton_cli"],
        "--precision",
        cfg["precision_cli"],
    ]
