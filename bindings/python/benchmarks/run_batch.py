import time
import numpy as np
from joblib import Parallel, delayed
from tdigest_rs import TDigest

# Configuration
quantile = 0.1
n = 16_000
n_arrays = 5_000
delta = 10_000
batch_size = 100  # Process this many arrays per batch call


def tdigest_batch_callback(arrays_batch):
    digests = TDigest.batch_from_arrays(arrays_batch, delta=delta)
    merged = [d.merge(d, delta=delta) for d in digests]
    return merged


arrays = [np.random.randn(n).astype(np.float32) for _ in range(n_arrays)]
batches = [arrays[i : i+batch_size] for i in range(0, len(arrays), batch_size)]
t0 = time.time()

for iteration in range(10):
    batch_results = Parallel(backend="threading", verbose=5, n_jobs=-1)(
        delayed(tdigest_batch_callback)(batch) for batch in batches
    )
    tdigests = [digest for batch_result in batch_results for digest in batch_result]

total_time = time.time() - t0
avg_time = total_time / 10

print(f"\nResults:")
print(f"  Total running time: {total_time:.3f}s")
print(f"  Average per iteration: {avg_time:.3f}s")
print(f"  Throughput: {n_arrays * n / avg_time:,.0f} elements/second")
print(f"  Created {len(tdigests)} digests")

