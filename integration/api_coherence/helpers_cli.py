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
    args = [
        "--no-header",
        "--output",
        "csv",
        "--max-size",
        str(cfg["max_size"]),
        "--scale",
        cfg["scale_cli"],
        "--singleton-policy",
        cfg["singleton_cli"],  # "off"|"use"|"edges"
        "--precision",
        cfg["precision_cli"],  # "f64"|"f32"|("auto"?)
    ]
    if (
        str(cfg.get("singleton_cli", "")).lower() == "edges"
        and cfg.get("pin_per_side") is not None
    ):
        args += ["--pin-per-side", str(int(cfg["pin_per_side"]))]
    return args


def cli_supports_precision_auto(cli_bin: Path) -> bool:
    """Heuristic: check command help for 'auto' in precision options."""
    try:
        helps = [
            subprocess.check_output([str(cli_bin), "--help"], text=True).lower(),
            subprocess.check_output(
                [str(cli_bin), "quantile", "--help"], text=True
            ).lower(),
            subprocess.check_output([str(cli_bin), "cdf", "--help"], text=True).lower(),
            subprocess.check_output(
                [str(cli_bin), "median", "--help"], text=True
            ).lower(),
            subprocess.check_output(
                [str(cli_bin), "build", "--help"], text=True
            ).lower(),
        ]
        return any("precision" in out and "auto" in out for out in helps)
    except Exception:
        return False
