import time

import numpy as np
from joblib import Parallel, delayed

from tdigest_rs import TDigest

quantile = 0.1
n = 16_000
n_arrays = 5_000
delta = 10_000


def tdigest_rs_callback(arr):
    digest = TDigest.from_array(arr, delta=delta)
    digest = digest.merge(digest)


arrays = [np.random.randn(n) for _ in range(n_arrays)]

t0 = time.time()
for _ in range(10):
    tdigests = Parallel(backend="threading", verbose=3, n_jobs=-1)(
        delayed(tdigest_rs_callback)(arr=arr) for arr in arrays
    )
print(f"Total running time tdigest_rs: {time.time() - t0}")


# --- Individual operation benchmarks ---

def bench_quantile():
    """Benchmark quantile queries at various percentiles."""
    data = np.random.randn(100_000)
    digest = TDigest.from_array(data, delta=100)
    centroid_count = len(digest.means)
    percentiles = [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99]
    n_queries = 10_000

    print(f"\n--- quantile benchmark (input=100k, centroids={centroid_count}) ---")
    for p in percentiles:
        t0 = time.perf_counter()
        for _ in range(n_queries):
            digest.quantile(p)
        elapsed = time.perf_counter() - t0
        us_per_query = (elapsed / n_queries) * 1e6
        print(f"  p{int(p*100):02d}: {us_per_query:.3f} us/query")


def bench_median():
    """Benchmark median (quantile(0.5)) queries."""
    data = np.random.randn(100_000)
    digest = TDigest.from_array(data, delta=100)
    centroid_count = len(digest.means)
    n_queries = 10_000

    print(f"\n--- median benchmark (input=100k, centroids={centroid_count}) ---")
    t0 = time.perf_counter()
    for _ in range(n_queries):
        digest.median()
    elapsed = time.perf_counter() - t0
    us_per_query = (elapsed / n_queries) * 1e6
    print(f"  median: {us_per_query:.3f} us/query")


def bench_trimmed_mean():
    """Benchmark trimmed_mean queries."""
    data = np.random.randn(100_000)
    digest = TDigest.from_array(data, delta=100)
    centroid_count = len(digest.means)
    n_queries = 10_000

    print(f"\n--- trimmed_mean benchmark (input=100k, centroids={centroid_count}) ---")
    t0 = time.perf_counter()
    for _ in range(n_queries):
        digest.trimmed_mean(0.05, 0.95)
    elapsed = time.perf_counter() - t0
    us_per_query = (elapsed / n_queries) * 1e6
    print(f"  trim(0.05, 0.95): {us_per_query:.3f} us/query")


if __name__ == "__main__":
    bench_quantile()
    bench_median()
    bench_trimmed_mean()
