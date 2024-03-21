import math

import numpy as np
import pytest

from tests.constants import TEST_QUANTILES
from tests.utils import fit_batched_digest


@pytest.mark.parametrize("size", [1_000, 10_000])
@pytest.mark.parametrize("loc", [0.0, -0.1, 0.1, 10.0, -10.0])
@pytest.mark.parametrize("scale", [1.0, 0.5, 0.1])
def test_gaussian(loc: float, scale: float, size: int) -> None:
    arr = np.random.normal(loc=loc, scale=scale, size=size).astype(np.float32)
    tdigest = fit_batched_digest(arr, batch_size=100, delta=100.0)

    assert math.isclose(tdigest.median(), loc, rel_tol=0.1, abs_tol=0.02)
    assert math.isclose(tdigest.trimmed_mean(lower=0.05, upper=0.95), loc, rel_tol=0.1, abs_tol=0.02)

    for q in TEST_QUANTILES:
        assert math.isclose(np.quantile(arr, q), tdigest.quantile(q), rel_tol=0.1, abs_tol=0.02)


@pytest.mark.parametrize("size", [1_000, 10_000])
@pytest.mark.parametrize("loc", [0.0, -0.1, 0.1, 10.0, -10.0])
@pytest.mark.parametrize("scale", [1.0, 0.5, 0.1])
def test_gaussian_small_batches(loc: float, scale: float, size: int) -> None:
    arr = np.random.normal(loc=loc, scale=scale, size=size).astype(np.float32)
    tdigest = fit_batched_digest(arr, batch_size=10, delta=100.0)

    assert math.isclose(tdigest.median(), loc, rel_tol=0.05, abs_tol=0.05)
    assert math.isclose(tdigest.trimmed_mean(lower=0.05, upper=0.95), loc, rel_tol=0.1, abs_tol=0.02)

    for q in TEST_QUANTILES:
        assert math.isclose(np.quantile(arr, q), tdigest.quantile(q), rel_tol=0.1, abs_tol=0.05)


@pytest.mark.parametrize("size", [1_000, 10_000])
@pytest.mark.parametrize("low", [0.0, -1.0, -2.0])
@pytest.mark.parametrize("high", [1.0, 2.0])
def test_uniform(low: float, high: float, size: int) -> None:
    arr = np.random.uniform(low=low, high=high, size=size).astype(np.float32)
    tdigest = fit_batched_digest(arr, batch_size=100, delta=100.0)

    median = (high + low) / 2
    assert math.isclose(tdigest.median(), median, rel_tol=0.1, abs_tol=0.05)

    for q in TEST_QUANTILES:
        assert math.isclose(np.quantile(arr, q), tdigest.quantile(q), rel_tol=0.1, abs_tol=0.05)
