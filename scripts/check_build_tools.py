"""Fail early on a mismatched Rust/Cargo selection without downloading tools."""

import os
from pathlib import Path
import re
import shutil
import subprocess


def main():
    root = Path(__file__).resolve().parents[1]
    expected = re.search(r'channel\s*=\s*"([^"]+)"', (root / "rust-toolchain.toml").read_text())[1]
    if shutil.which("rustup"):
        installed = subprocess.check_output(["rustup", "toolchain", "list", "--quiet"], text=True)
        if not any(line.startswith(expected + "-") or line.split()[0] == expected for line in installed.splitlines()):
            raise SystemExit(
                f"Rust {expected} is not staged. Install it from your approved RUSTUP_DIST_SERVER before building; this check will not download it."
            )
    for command in ("rustc", "cargo"):
        output = subprocess.check_output([command, "--version"], cwd=root, text=True).strip()
        version = re.match(r"\w+ ([0-9]+\.[0-9]+\.[0-9]+)", output)
        if not version or version[1] != expected:
            raise SystemExit(
                f"Expected pinned {command} {expected}, got {output}. Check PATH and RUSTUP_TOOLCHAIN; Cargo must also read lockfile format 4."
            )
        print(output)
    offline = os.environ.get("TDIGEST_OFFLINE", "0").lower()
    if offline not in ("0", "1", "true", "false", "yes", "no"):
        raise SystemExit("TDIGEST_OFFLINE must be 0/1, true/false or yes/no")


if __name__ == "__main__":
    main()
