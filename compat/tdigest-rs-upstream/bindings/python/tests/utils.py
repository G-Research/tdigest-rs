import math
from typing import Iterator

import numpy as np

from tdigest_rs import TDigest


def fit_batched_digest(arr: np.ndarray, batch_size: int, delta: float):
    digest = None
    for batch in batched_numpy_loader(arr, batch_size=batch_size):
        _digest = TDigest.from_array(batch, delta=delta)
        if digest is None:
            digest = _digest
        else:
            digest = digest.merge(_digest, delta=delta)

    return digest


def batched_numpy_loader(arr: np.ndarray, batch_size: int) -> Iterator[np.ndarray]:
    num_batches = math.ceil(len(arr) / batch_size)
    for i in range(num_batches):
        yield arr[i * batch_size : (i + 1) * batch_size]
