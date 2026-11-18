"""Bounded end-to-end workloads; comparisons, not release performance gates.

Each scenario runs in a fresh process so peak RSS includes its complete workflow.
Run installed Python with -I, fixed pool settings and no concurrent heavy builds.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import tdigest_rs as gr
import numpy as np
import polars as pl


def timed(operation, repeats=3):
    operation()  # independent warmup, excluded from samples
    samples = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = operation()
        samples.append(time.perf_counter() - start)
    return result, {
        "seconds": samples,
        "median_seconds": float(np.median(samples)),
        "p95_seconds": float(np.quantile(samples, 0.95)),
        "p99_seconds": float(np.quantile(samples, 0.99)),
        "samples": len(samples),
    }


def frame(groups, observations, columns, shift=0):
    index = np.arange(groups * observations, dtype=np.int64)
    return pl.DataFrame(
        {
            "g": index // observations,
            **{f"d{i}": ((index * 17 + i * 31 + shift) % 1009).astype(np.float64) for i in range(columns)},
        }
    )


def worker(scenario):
    if scenario.startswith("columns"):
        columns, groups, observations = int(scenario.removeprefix("columns")), 2048, 32
        source = frame(groups, observations, columns)
        names = [f"d{i}" for i in range(columns)]
        digests, training = timed(
            lambda: source.group_by("g", maintain_order=True).agg(
                *[gr.tdigest(name, max_size=200).alias(name) for name in names]
            )
        )
        digests = digests.with_columns(pl.lit(503.0).alias("probe"))
        _, cdf = timed(lambda: digests.select(*[gr.cdf(name, pl.col("probe")).alias(name) for name in names]), 12)
        _, median = timed(lambda: digests.select(*[gr.median(name).over("g").alias(name) for name in names]), 12)
        fragmented = pl.concat([digests.slice(i * 128, 128) for i in range(16)], rechunk=False)
        _, fragmented_cdf = timed(
            lambda: fragmented.select(*[gr.cdf(name, pl.col("probe")).alias(name) for name in names]), 12
        )
        return {
            "groups": groups,
            "columns": columns,
            "observations_per_group": observations,
            "capacity": 200,
            "training": training,
            "cdf": cdf,
            "median": median,
            "fragmented_cdf": fragmented_cdf,
            "fragments": 16,
            "training_values_per_second": groups * columns * observations / training["median_seconds"],
            "cdf_probabilities_per_second": groups * columns / cdf["median_seconds"],
        }
    if scenario == "fanin40":
        groups, columns, fanin, observations = 1024, 16, 40, 8
        names = [f"d{i}" for i in range(columns)]
        fragments = [
            frame(groups, observations, columns, shift=shard * 47)
            .group_by("g")
            .agg(*[gr.tdigest(name, max_size=200).alias(name) for name in names])
            for shard in range(fanin)
        ]
        inputs = pl.concat(fragments, rechunk=False)
        result, metrics = timed(
            lambda: inputs.group_by("g").agg(*[gr.merge_tdigests(name).alias(name) for name in names])
        )
        assert result.select(pl.col("d0").struct.field("count")).to_series().to_list() == [fanin * observations] * groups
        return {
            "groups": groups,
            "columns": columns,
            "fanin": fanin,
            "observations_per_shard": observations,
            "merge": metrics,
            "input_digests_per_second": groups * columns * fanin / metrics["median_seconds"],
        }
    if scenario == "stream":
        size, chunk = 1048576, 16384

        def construct():
            d = gr.TDigest.from_array([], max_size=1000)
            for start in range(0, size, chunk):
                d.add((np.arange(start, start + chunk, dtype=np.float64) * 17) % 1009)
            assert sum(d.weights) == size
            return d

        _, metrics = timed(construct)
        return {
            "values": size,
            "chunk_values": chunk,
            "capacity": 1000,
            "construction": metrics,
            "values_per_second": size / metrics["median_seconds"],
        }
    if scenario == "bridge":
        digests = [gr.TDigest.from_array(np.arange(32, dtype=np.float64) + i, max_size=200) for i in range(2048)]

        def roundtrip():
            for d in digests:
                restored = gr.TDigest.from_bytes(d.to_bytes())
                restored.cdf(np.array([0.0, 20.0, 40.0]))
                restored.median()

        _, metrics = timed(roundtrip, 12)
        return {
            "digests": len(digests),
            "includes": "bytes encode/decode, native allocation/copy, NumPy CDF output and median bridge",
            "roundtrip": metrics,
            "digests_per_second": len(digests) / metrics["median_seconds"],
        }
    if scenario == "concurrency":
        values = (np.arange(32768, dtype=np.float64) * 17) % 1009
        rows = []
        for workers in [1, 2, 4, 8, 16]:
            elapsed, latency = [], []
            for _ in range(3):
                with ThreadPoolExecutor(max_workers=workers) as pool:

                    def construct(submitted):
                        result = gr.TDigest.from_array(values, max_size=1000)
                        return time.perf_counter() - submitted, result.median()

                    start = time.perf_counter()
                    futures = [pool.submit(construct, time.perf_counter()) for _ in range(64)]
                    latency.extend(future.result()[0] for future in futures)
                    elapsed.append(time.perf_counter() - start)
            rows.append(
                {
                    "workers": workers,
                    "tasks_per_batch": 64,
                    "batch_seconds": elapsed,
                    "values_per_second": 64 * len(values) / float(np.median(elapsed)),
                    "submission_to_completion_p95_seconds": float(np.quantile(latency, 0.95)),
                    "submission_to_completion_p99_seconds": float(np.quantile(latency, 0.99)),
                    "latency_samples": len(latency),
                }
            )
        return {"independent_digests": True, "rows": rows}
    raise ValueError(scenario)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--worker")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.worker:
        result = worker(args.worker)
        native = Path(importlib.import_module("tdigest_rs._native").__file__)
        result.update(
            scenario=args.worker,
            source_sha=args.source_sha,
            version=gr.__version__,
            native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
            python=sys.version,
            numpy=np.__version__,
            polars=pl.__version__,
            peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            available_logical_cpus=sorted(os.sched_getaffinity(0)),
            pools={
                name: os.environ.get(name)
                for name in ["RAYON_NUM_THREADS", "POLARS_MAX_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"]
            },
        )
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return
    for scenario in ["columns16", "columns64", "columns128", "fanin40", "stream", "bridge", "concurrency"]:
        output = args.output.with_name(args.output.stem + "-" + scenario + ".json")
        subprocess.run(
            [
                sys.executable,
                "-I",
                str(Path(__file__).resolve()),
                "--source-sha",
                args.source_sha,
                "--output",
                str(output),
                "--worker",
                scenario,
            ],
            check=True,
        )
        print(scenario, output, flush=True)


if __name__ == "__main__":
    main()
