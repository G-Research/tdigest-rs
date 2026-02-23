import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from tdigest_rs import TDigest


def tdigest_rs_callback(arr: np.ndarray, delta: float) -> float:
    digest = TDigest.from_array(arr, delta=delta)
    q = digest.quantile(0.1)
    c = digest.cdf(0.0)
    return q + c


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Python benchmark for tdigest-rs legacy-delta construction path")
    p.add_argument("--n", type=int, default=16_000, help="values per input array")
    p.add_argument("--n-arrays", type=int, default=5_000, help="number of arrays per iteration")
    p.add_argument("--iterations", type=int, default=10, help="benchmark iterations")
    p.add_argument("--delta", type=float, default=10_000.0, help="legacy delta parameter")
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1), help="thread pool workers")
    p.add_argument("--seed", type=int, default=42, help="rng seed")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    arrays = [rng.standard_normal(args.n, dtype=np.float64) for _ in range(args.n_arrays)]

    # Warm up to avoid timing first-call effects.
    _ = tdigest_rs_callback(arrays[0], args.delta)

    t0 = time.perf_counter()
    total_tasks = 0
    for _ in range(args.iterations):
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(lambda arr: tdigest_rs_callback(arr, args.delta), arrays))
        total_tasks += len(results)
        assert all(np.isfinite(v) for v in results)

    dt = time.perf_counter() - t0
    print(
        "tdigest-rs python benchmark passed | "
        f"arrays={args.n_arrays} n={args.n} iterations={args.iterations} "
        f"delta={args.delta:g} workers={args.workers} "
        f"time_sec={dt:.3f} tasks_per_sec={total_tasks / dt:.2f}"
    )


if __name__ == "__main__":
    main()
