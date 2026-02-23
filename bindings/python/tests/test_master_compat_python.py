import math
import pickle
from copy import deepcopy
from typing import Iterator

import numpy as np
import pytest

from tdigest_rs import TDigest

TEST_QUANTILES = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]


def _batched_numpy_loader(arr: np.ndarray, batch_size: int) -> Iterator[np.ndarray]:
    num_batches = math.ceil(len(arr) / batch_size)
    for i in range(num_batches):
        yield arr[i * batch_size : (i + 1) * batch_size]


def _fit_batched_digest(arr: np.ndarray, batch_size: int, delta: float) -> TDigest:
    digest: TDigest | None = None
    for batch in _batched_numpy_loader(arr, batch_size=batch_size):
        d = TDigest.from_array(batch, delta=delta)
        digest = d if digest is None else digest.merge(d)
    assert digest is not None
    return digest


def _numpy_trimmed_mean(values: np.ndarray, lower: float, upper: float) -> float:
    xs = np.sort(values.astype(np.float64, copy=False))
    n = len(xs)
    lo = int(math.floor(lower * n))
    hi = int(math.ceil(upper * n))
    hi = min(max(hi, lo + 1), n)
    return float(np.mean(xs[lo:hi]))


@pytest.mark.parametrize("n", range(1, 4))
def test_short_arrays(n: int) -> None:
    vals = np.arange(n).astype(np.float64)
    weights = np.ones(n).astype(np.float64)
    tdigest = TDigest.from_means_weights(arr=vals, weights=weights)

    np.testing.assert_array_equal(tdigest.means, vals)
    np.testing.assert_array_equal(tdigest.weights, weights)


def test_quantile_single_value_data() -> None:
    vals = np.ones(10, dtype=np.float64)
    tdigest = TDigest.from_array(vals, delta=5.0)

    for x in [0.0, 1e-7, 0.6, 1.0]:
        assert tdigest.quantile(x) == 1.0


@pytest.mark.parametrize("n", [1000, 5000, 10000])
def test_quantile_randn_data(n: int) -> None:
    rng = np.random.default_rng(1234 + n)
    arr = rng.standard_normal(n, dtype=np.float64)
    tdigest = TDigest.from_array(arr)

    for q in TEST_QUANTILES:
        assert math.isclose(np.quantile(arr, q), tdigest.quantile(q), rel_tol=0.1, abs_tol=1e-2)


def test_median_random_data() -> None:
    rng = np.random.default_rng(1337)
    arr = rng.standard_normal(5000, dtype=np.float64)
    tdigest = TDigest.from_array(arr)

    assert math.isclose(np.median(arr), tdigest.median(), rel_tol=0.1, abs_tol=1e-2)
    assert math.isclose(tdigest.quantile(0.5), tdigest.median(), rel_tol=0.1, abs_tol=1e-2)


def test_trimmed_mean() -> None:
    rng = np.random.default_rng(2026)
    vals = rng.standard_normal(1000, dtype=np.float64)
    weights = np.ones(1000).astype(np.float64)
    tdigest = TDigest.from_means_weights(arr=vals, weights=weights)

    expected = _numpy_trimmed_mean(vals, 0.05, 0.95)
    assert math.isclose(tdigest.trimmed_mean(0.05, 0.95), expected, abs_tol=0.2)


def test_len() -> None:
    length = 100
    rng = np.random.default_rng(99)
    vals = rng.standard_normal(length, dtype=np.float64)
    weights = np.ones(length).astype(np.float64)
    tdigest = TDigest.from_means_weights(arr=vals, weights=weights)

    assert len(tdigest) == length


def test_only_nan_values() -> None:
    vals = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
    with pytest.raises(ValueError):
        TDigest.from_array(vals)


def test_single_nan_value() -> None:
    vals = np.array([10.0, 20.0, np.nan], dtype=np.float64)
    with pytest.raises(ValueError):
        TDigest.from_array(vals)


def test_only_inf_values() -> None:
    vals = np.array([-np.inf, np.inf, np.inf], dtype=np.float64)
    with pytest.raises(ValueError):
        TDigest.from_array(vals)


def test_non_inf_extremas() -> None:
    vals = np.array([-np.inf, -20.0, 8.0, 10.0, np.inf], dtype=np.float64)
    with pytest.raises(ValueError):
        TDigest.from_array(vals)


def test_merge_with_infs() -> None:
    vals1 = np.array([-np.inf, 1.0, 2.0, np.inf, np.inf], dtype=np.float64)
    vals2 = np.array([-np.inf, -np.inf, -1.0, 30.0, np.inf, np.inf], dtype=np.float64)
    with pytest.raises(ValueError):
        TDigest.from_array(vals1, delta=200)
    with pytest.raises(ValueError):
        TDigest.from_array(vals2, delta=200)


def test_pickle_unpickle() -> None:
    values = np.array([1.0, 2.0], dtype=np.float64)
    digest = TDigest.from_array(values)
    p = pickle.dumps(digest)
    loaded_digest = pickle.loads(p)

    np.testing.assert_allclose(digest.means, loaded_digest.means)
    np.testing.assert_allclose(digest.weights, loaded_digest.weights)


def test_deepcopy() -> None:
    values = np.array([1.0, 2.0], dtype=np.float64)
    digest = TDigest.from_array(values)
    copied_digest = deepcopy(digest)

    np.testing.assert_allclose(digest.means, copied_digest.means)
    np.testing.assert_allclose(digest.weights, copied_digest.weights)


def test_raises_invalid_input_type() -> None:
    values = np.array(["1.0", "2.0"], dtype=object)
    with pytest.raises(TypeError):
        TDigest.from_array(values)


def test_raises_merge_different_types() -> None:
    values = np.array([1.0, 2.0], dtype=np.float64)
    digest64 = TDigest.from_array(values, precision="f64")
    digest32 = TDigest.from_array(values.astype(np.float32), precision="f32")

    with pytest.raises(ValueError):
        digest32.merge(digest64)


@pytest.mark.parametrize("size", [1_000, 10_000])
@pytest.mark.parametrize("loc", [0.0, -0.1, 0.1, 10.0, -10.0])
@pytest.mark.parametrize("scale", [1.0, 0.5, 0.1])
def test_gaussian(loc: float, scale: float, size: int) -> None:
    rng = np.random.default_rng(17 + size)
    arr = rng.normal(loc=loc, scale=scale, size=size).astype(np.float64)
    tdigest = _fit_batched_digest(arr, batch_size=100, delta=100.0)

    assert math.isclose(tdigest.median(), float(np.median(arr)), rel_tol=0.1, abs_tol=0.08)
    assert math.isclose(
        tdigest.trimmed_mean(lower=0.05, upper=0.95),
        _numpy_trimmed_mean(arr, 0.05, 0.95),
        rel_tol=0.1,
        abs_tol=0.05,
    )

    for q in TEST_QUANTILES:
        assert math.isclose(np.quantile(arr, q), tdigest.quantile(q), rel_tol=0.1, abs_tol=0.02)


@pytest.mark.parametrize("size", [1_000, 10_000])
@pytest.mark.parametrize("loc", [0.0, -0.1, 0.1, 10.0, -10.0])
@pytest.mark.parametrize("scale", [1.0, 0.5, 0.1])
def test_gaussian_small_batches(loc: float, scale: float, size: int) -> None:
    rng = np.random.default_rng(31 + size)
    arr = rng.normal(loc=loc, scale=scale, size=size).astype(np.float64)
    tdigest = _fit_batched_digest(arr, batch_size=10, delta=100.0)

    assert math.isclose(tdigest.median(), float(np.median(arr)), rel_tol=0.1, abs_tol=0.08)
    assert math.isclose(
        tdigest.trimmed_mean(lower=0.05, upper=0.95),
        _numpy_trimmed_mean(arr, 0.05, 0.95),
        rel_tol=0.1,
        abs_tol=0.05,
    )

    for q in TEST_QUANTILES:
        assert math.isclose(np.quantile(arr, q), tdigest.quantile(q), rel_tol=0.1, abs_tol=0.05)


@pytest.mark.parametrize("size", [1_000, 10_000])
@pytest.mark.parametrize("low", [0.0, -1.0, -2.0])
@pytest.mark.parametrize("high", [1.0, 2.0])
def test_uniform(low: float, high: float, size: int) -> None:
    rng = np.random.default_rng(211 + size)
    arr = rng.uniform(low=low, high=high, size=size).astype(np.float64)
    tdigest = _fit_batched_digest(arr, batch_size=100, delta=100.0)

    assert math.isclose(tdigest.median(), float(np.median(arr)), rel_tol=0.1, abs_tol=0.08)

    for q in TEST_QUANTILES:
        assert math.isclose(np.quantile(arr, q), tdigest.quantile(q), rel_tol=0.1, abs_tol=0.05)
