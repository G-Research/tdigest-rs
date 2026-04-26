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
