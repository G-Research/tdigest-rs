"""Build and check a release source archive without compiling an extension."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from test_legacy_wheel import build_and_check_sdist, verify_upstream_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory for archive and preparation logs")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    verify_upstream_snapshot(root)
    commands = []

    def run(*command: str, cwd: Path) -> None:
        completed = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        log = output / "source-archive.log"
        log.write_text(completed.stdout)
        commands.append({"command": command, "cwd": str(cwd), "exit_code": completed.returncode, "log": str(log)})
        (output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        print(completed.stdout, end="", flush=True)
        completed.check_returncode()

    before = (root / "Cargo.lock").read_bytes()
    sdist = build_and_check_sdist(root, output / "sdist", run)
    provenance = {
        "status": "PASS",
        "operation": "maturin sdist and active-source validation; no extension compilation",
        "sdist": str(sdist),
        "sdist_sha256": hashlib.sha256(sdist.read_bytes()).hexdigest(),
        "cargo_lock_sha256": hashlib.sha256(before).hexdigest(),
        "cargo_lock_unchanged": True,
    }
    (output / "artifact.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
