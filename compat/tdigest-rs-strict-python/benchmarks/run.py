from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from tdigest_rs import TDigest


def tdigest_rs_callback(arr: np.ndarray, delta: float) -> float:
    digest = TDigest.from_array(arr, delta=delta)
    digest = digest.merge(digest, delta=delta)
    return float(digest.quantile(0.1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manual legacy tdigest-rs Python benchmark smoke."
    )
    parser.add_argument("--n", type=int, default=16_000)
    parser.add_argument("--n-arrays", type=int, default=5_000)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--delta", type=float, default=10_000.0)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    arrays = [rng.standard_normal(args.n) for _ in range(args.n_arrays)]

    t0 = time.perf_counter()
    total = 0.0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for _ in range(args.iterations):
            for value in pool.map(
                lambda arr: tdigest_rs_callback(arr, args.delta), arrays
            ):
                if not np.isfinite(value):
                    raise RuntimeError(f"non-finite benchmark result: {value!r}")
                total += value

    elapsed = time.perf_counter() - t0
    print(
        "tdigest-rs strict legacy benchmark passed | "
        f"n={args.n} n_arrays={args.n_arrays} iterations={args.iterations} "
        f"delta={args.delta:g} workers={args.workers} elapsed={elapsed:.3f}s checksum={total:.6g}"
    )


if __name__ == "__main__":
    main()
