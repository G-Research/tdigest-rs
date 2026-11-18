"""Compare installed baseline/legacy/modern wheels in alternating isolated processes.

Linux workstation benchmark, not a performance gate. Each environment must have
the same Python/NumPy versions and its respective release wheel installed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time


def worker(engine: str) -> dict:
    import copy
    import numpy as np
    import tdigest_rs as api

    assert Path(api.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    rows = []

    def values(n, dtype, seed=4242):
        rng = np.random.default_rng(seed)
        data = rng.normal(size=n)
        data[::10] += 6
        data[::31] = -2
        return data.astype(dtype)

    def build(data):
        if engine == "modern":
            return api.TDigest.from_array(data, max_size=1000, precision="f32" if data.dtype == np.float32 else "f64")
        return api.TDigest.from_array(data, delta=300)

    def describe(digest, data):
        means, weights = digest.means, digest.weights
        mean_bytes = data.dtype.itemsize
        width = mean_bytes + (4 if engine == "published" else 9)
        state = hashlib.sha256(means.tobytes() + weights.tobytes()).hexdigest()
        errors = []
        ordered = np.sort(data)
        for q in (0.01, 0.5, 0.99):
            if not len(data):
                continue
            answer = digest.quantile(q)
            left = np.searchsorted(ordered, answer, side="left") / len(data)
            right = np.searchsorted(ordered, answer, side="right") / len(data)
            errors.append(max(left - q, q - right, 0.0))
        return {
            "centroids": len(digest),
            "logical_centroid_bytes": len(digest) * width,
            "max_rank_error_at_01_50_99": max(errors, default=None),
            "state_sha256": state,
        }

    def measure(case, call, details):
        call()
        start = time.perf_counter_ns()
        result = call()
        del result
        elapsed = time.perf_counter_ns() - start
        iterations = max(1, min(20000, int(25_000_000 / max(1, elapsed))))
        was_enabled = gc.isenabled()
        gc.disable()
        try:
            start = time.perf_counter_ns()
            for _ in range(iterations):
                result = call()
                del result
            duration = time.perf_counter_ns() - start
        finally:
            if was_enabled:
                gc.enable()
        rows.append({"case": case, "iterations": iterations, "ns_per_operation": duration / iterations, **details})

    for dtype in (np.float32, np.float64):
        precision = "f32" if dtype == np.float32 else "f64"
        for n in (0, 1, 32, 1000, 10000, 100000, 1000000):
            data = values(n, dtype)
            if n == 0 and engine != "modern":

                def reject_empty():
                    try:
                        build(data)
                    except ValueError:
                        return
                    raise AssertionError("legacy empty construction unexpectedly succeeded")

                measure(f"{precision}/reject_empty/{n}", reject_empty, {"state_sha256": None})
                continue
            digest = build(data)
            details = describe(digest, data)
            measure(f"{precision}/build/{n}", lambda: build(data), details)
            if n in (1000, 1000000):
                measure(f"{precision}/median/{n}", digest.median, details)
                measure(f"{precision}/quantile_99/{n}", lambda: digest.quantile(0.99), details)
                if engine == "modern":
                    measure(f"{precision}/cdf_zero/{n}", lambda: digest.cdf(0.0), details)
        for child_size in (32, 1000, 10000):
            chunks = [values(child_size, dtype, 4242 + i) for i in range(40)]
            children = [build(chunk) for chunk in chunks]
            data = np.concatenate(chunks)

            def sequential():
                result = copy.copy(children[0])
                for other in children[1:]:
                    if engine == "modern":
                        result.merge(other)
                    else:
                        result = result.merge(other)
                return result

            measure(f"{precision}/merge_sequential_40/{child_size}", sequential, describe(sequential(), data))
            if engine == "modern":

                def merge_all():
                    return api.TDigest.merge_all(children)

                measure(f"{precision}/merge_all_40/{child_size}", merge_all, describe(merge_all(), data))
    return {
        "python": sys.version,
        "numpy": np.__version__,
        "distribution": metadata.version("tdigest-rs"),
        "module": api.__file__,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=("published", "legacy", "modern"))
    parser.add_argument("--baseline-python", type=Path)
    parser.add_argument("--candidate-python", type=Path)
    parser.add_argument("--baseline-wheel", type=Path)
    parser.add_argument("--candidate-wheel", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--rounds", type=int, default=7)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.worker)))
        return
    if not all(
        (args.baseline_python, args.candidate_python, args.baseline_wheel, args.candidate_wheel, args.output_dir)
    ):
        parser.error("both installed interpreters, wheel paths and a new output directory are required")
    if args.rounds < 1:
        parser.error("--rounds must be at least one")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.update(RAYON_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    affinity = sorted(os.sched_getaffinity(0))
    cpu = 8 if 8 in affinity else affinity[0]
    os.sched_setaffinity(0, {cpu})
    labels = ("published_legacy", "candidate_legacy", "candidate_modern")
    records = []
    for round_index in range(args.rounds):
        order = labels[round_index % 3 :] + labels[: round_index % 3]
        if round_index % 2:
            order = tuple(reversed(order))
        for label in order:
            python = args.baseline_python if label == "published_legacy" else args.candidate_python
            engine = {"published_legacy": "published", "candidate_legacy": "legacy", "candidate_modern": "modern"}[
                label
            ]
            command = [str(python.absolute()), "-I", str(Path(__file__).resolve()), "--worker", engine]
            completed = subprocess.run(command, cwd=output, env=env, text=True, capture_output=True, check=True)
            record = {"round": round_index + 1, "label": label, **json.loads(completed.stdout)}
            records.append(record)
            (output / f"round-{round_index + 1}-{label}.json").write_text(json.dumps(record, indent=2) + "\n")
            (output / f"round-{round_index + 1}-{label}.stderr").write_text(completed.stderr)
            print(f"round {round_index + 1}/{args.rounds}: {label} complete", flush=True)
    assert len({(r["python"], r["numpy"]) for r in records}) == 1, "baseline and candidate runtime mismatch"
    summary = {}
    for label in labels:
        grouped = {}
        for record in records:
            if record["label"] == label:
                for row in record["rows"]:
                    grouped.setdefault(row["case"], []).append(row)
        summary[label] = {}
        for case, rows in grouped.items():
            samples = [row["ns_per_operation"] for row in rows]
            summary[label][case] = {
                "median_ns": statistics.median(samples),
                "min_ns": min(samples),
                "max_ns": max(samples),
                "samples_ns": samples,
                **{k: v for k, v in rows[0].items() if k not in ("case", "iterations", "ns_per_operation")},
            }
    same = all(
        summary["published_legacy"][case]["state_sha256"] == row["state_sha256"]
        for case, row in summary["candidate_legacy"].items()
    )
    report = {
        "platform": platform.platform(),
        "cpu_affinity": [cpu],
        "rounds": args.rounds,
        "python": records[0]["python"],
        "numpy": records[0]["numpy"],
        "legacy_states_match": same,
        "summary": summary,
        "artifacts": {
            label: {"filename": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for label, path in (("baseline", args.baseline_wheel), ("candidate", args.candidate_wheel))
        },
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
