"""Explicitly run the prepared committed-lock oracle; exits nonzero on differences."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import math
import os
import struct
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def compare(actual, expected):
    failures = []
    numeric = {"exact_bits": 0, "nan_classification": 0}
    def equal(left, right, location):
        if left != right:
            failures.append({"path": location, "actual": left, "expected": right})
    def number(left, right, location):
        a = struct.unpack(">d", bytes.fromhex(left))[0]
        b = struct.unpack(">d", bytes.fromhex(right))[0]
        if math.isnan(b):
            numeric["nan_classification"] += 1
            if not math.isnan(a):
                failures.append({"path": location, "actual": left, "expected_nan": right})
        else:
            numeric["exact_bits"] += 1
            equal(left, right, location)
    equal(actual.get("upstream_commit"), expected["upstream_commit"], "upstream_commit")
    equal(len(actual["cases"]), 70, "case_count")
    actual_names = [c["name"] for c in actual["cases"]]
    expected_names = [c["name"] for c in expected["cases"]]
    equal(actual_names, expected_names, "case_names_and_order")
    for left, right in zip(actual["cases"], expected["cases"]):
        label = right["name"]
        equal(sorted(left), sorted(right), label + ".fields")
        for key in ("name", "dtype", "input", "weights"):
            equal(left[key], right[key], label + "." + key)
        equal(len(left["means"]), len(right["means"]), label + ".means.length")
        for i, (a, b) in enumerate(zip(left["means"], right["means"])):
            number(a, b, f"{label}.means[{i}]")
        number(left["median"], right["median"], label + ".median")
        for key, prefix in (("quantiles", 1), ("trimmed_means", 2)):
            equal(len(left[key]), len(right[key]), label + "." + key + ".length")
            for i, (a, b) in enumerate(zip(left[key], right[key])):
                equal(a[:prefix], b[:prefix], f"{label}.{key}[{i}].input")
                number(a[prefix], b[prefix], f"{label}.{key}[{i}].output")
    return {"case_count": len(actual["cases"]), "numeric_comparisons": numeric, "failures": failures}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="Explicitly compile the prepared source snapshot")
    parser.add_argument("--profiles", nargs="+", choices=("release", "debug"), default=["release"])
    parser.add_argument("--offline", action="store_true", help="Require dependencies already present in Cargo cache")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "source-manifest.json").read_text())
    plan = json.loads((ROOT / "plan.json").read_text())
    if not args.run:
        print(json.dumps({"status": "prepared; pass --run to run the comparison", "plan": plan}, indent=2))
        return 0
    def unchanged():
        for name, target in plan.get("original_symlinks", {}).items():
            assert (SOURCE / name).is_symlink() and os.readlink(SOURCE / name) == target, "Original symlink changed: " + name
        for name, expected_hash in manifest.items():
            assert sha(SOURCE / name) == expected_hash, "Original file changed: " + name
        assert sha(ROOT / "golden.json") == plan["golden_sha256"], "Golden fixture changed"
        assert sha(SOURCE / "examples" / "committed_lock_oracle.rs") == plan["oracle_source_sha256"]
    unchanged()
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(ROOT / "target")
    env["CARGO_BUILD_JOBS"] = "2"
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    result = {"started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "plan": plan, "runs": [], "toolchain": {}}
    for name, command in (("rustc", [shutil.which("rustc") or "rustc", "--version", "--verbose"]), ("cargo", [shutil.which("cargo") or "cargo", "--version"])):
        result["toolchain"][name] = subprocess.check_output(command, text=True)
    for profile in args.profiles:
        command = [shutil.which("cargo") or "cargo", "run", "--locked", "--package", "tdigest_core", "--example", "committed_lock_oracle"]
        if profile == "release":
            command.append("--release")
        if args.offline:
            command.append("--offline")
        start = time.monotonic()
        output = ROOT / (profile + "-output.json")
        error_log = logs / (profile + "-cargo.log")
        with output.open("w") as out, error_log.open("w") as err:
            completed = subprocess.run(command, cwd=SOURCE, env=env, stdout=out, stderr=err)
        run = {"profile": profile, "command": command, "cwd": str(SOURCE), "elapsed_seconds": time.monotonic() - start, "exit_code": completed.returncode, "stdout_sha256": sha(output), "stderr_sha256": sha(error_log)}
        unchanged()
        if completed.returncode == 0:
            run["comparison"] = compare(json.loads(output.read_text()), json.loads((ROOT / "golden.json").read_text()))
            binary = ROOT / "target" / ("release" if profile == "release" else "debug") / "examples" / "committed_lock_oracle"
            run["binary_sha256"] = sha(binary)
        result["runs"].append(run)
        (ROOT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    result["original_source_and_lock_unchanged"] = True
    result["success"] = all(run["exit_code"] == 0 and not run["comparison"]["failures"] for run in result["runs"])
    result["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (ROOT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"success": result["success"], "profiles": [{"profile": run["profile"], "exit_code": run["exit_code"], "comparison": run.get("comparison")} for run in result["runs"]]}, indent=2))
    return 0 if result["success"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
