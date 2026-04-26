import math
import pickle
from copy import deepcopy

import numpy as np
import pytest
from scipy.stats import trim_mean

from tdigest_rs import TDigest
from tests.constants import TEST_QUANTILES


@pytest.mark.parametrize("n", range(1, 4))
def test_short_arrays(n) -> None:
    vals = np.arange(n).astype(np.float32)
    weights = np.ones(n).astype(np.uint32)
    tdigest = TDigest.from_means_weights(arr=vals, weights=weights)

    np.testing.assert_array_equal(tdigest.means, vals)
    np.testing.assert_array_equal(tdigest.weights, weights)


def test_quantile_single_value_data() -> None:
    n = 10
    vals = np.ones(n).astype(np.float32)
    tdigest = TDigest.from_array(arr=vals, delta=5.0)

    for x in [0, 1e-7, 0.6, 1]:
        assert tdigest.quantile(x) == 1.0


@pytest.mark.parametrize("n", [1000, 5000, 10000])
def test_quantile_randn_data(n: int) -> None:
    arr = np.random.randn(n).astype(np.float32)
    tdigest = TDigest.from_array(arr=arr)

    for q in TEST_QUANTILES:
        assert math.isclose(np.quantile(arr, q), tdigest.quantile(q), rel_tol=0.1, abs_tol=1e-2)


def test_median_random_data() -> None:
    arr = np.random.randn(5000).astype(np.float32)
    tdigest = TDigest.from_array(arr=arr)

    assert math.isclose(np.median(arr), tdigest.median(), rel_tol=0.1, abs_tol=1e-2)
    assert tdigest.quantile(0.5) == tdigest.median()


def test_trimmed_mean() -> None:
    vals = np.random.randn(1000).astype(np.float32)
    weights = np.ones(1000).astype(np.uint32)
    tdigest = TDigest.from_means_weights(arr=vals, weights=weights)

    assert math.isclose(tdigest.trimmed_mean(0.05, 0.95), trim_mean(vals, 0.05), abs_tol=0.2)


def test_len() -> None:
    length = 100
    vals = np.random.randn(length).astype(np.float32)
    weights = np.ones(length).astype(np.uint32)
    tdigest = TDigest.from_means_weights(arr=vals, weights=weights)

    assert len(tdigest) == length


def test_only_nan_values() -> None:
    vals = np.array([np.nan, np.nan, np.nan]).astype(np.float32)
    tdigest = TDigest.from_array(arr=vals)

    assert len(tdigest) == 0
    assert len(tdigest.means) == 0
    assert len(tdigest.weights) == 0


def test_single_nan_value() -> None:
    vals = np.array([10.0, 20.0, np.nan]).astype(np.float32)
    tdigest = TDigest.from_array(arr=vals)

    assert len(tdigest) > 0
    assert np.all(np.isfinite(tdigest.means))


def test_only_inf_values() -> None:
    vals = np.array([-np.inf, np.inf, np.inf]).astype(np.float32)
    tdigest = TDigest.from_array(arr=vals)

    assert len(tdigest) == 2
    assert np.isneginf(tdigest.means[0])
    assert np.isposinf(tdigest.means[1])


def test_non_inf_extremas() -> None:
    vals = np.array([-np.inf, -20, 8, 10, np.inf]).astype(np.float32)
    tdigest = TDigest.from_array(vals)

    np.testing.assert_array_equal(tdigest.means, vals)


def test_merge_with_infs() -> None:
    vals1 = np.array([-np.inf, 1, 2, np.inf, np.inf]).astype(np.float32)
    tdigest1 = TDigest.from_array(vals1, delta=200)
    vals2 = np.array([-np.inf, -np.inf, -1, 30, np.inf, np.inf]).astype(np.float32)
    tdigest2 = TDigest.from_array(vals2, delta=200)
    tdigest = tdigest1.merge(tdigest2)

    assert len(tdigest) == 6


def test_pickle_unpickle() -> None:
    values = np.array([1, 2]).astype(np.float32)
    digest = TDigest.from_array(values)
    p = pickle.dumps(digest)
    loaded_digest = pickle.loads(p)

    assert np.all(digest.means == loaded_digest.means)
    assert np.all(digest.weights == loaded_digest.weights)


def test_deepcopy() -> None:
    values = np.array([1, 2]).astype(np.float32)
    digest = TDigest.from_array(values)
    copied_digest = deepcopy(digest)

    assert np.all(digest.means == copied_digest.means)
    assert np.all(digest.weights == copied_digest.weights)


def test_raises_invalid_input_type() -> None:
    values = np.array([1.0, 2.0]).astype(np.float16)
    with pytest.raises(TypeError, match="TDigest is not implemented for arr with type"):
        TDigest.from_array(values)


def test_raises_merge_different_types() -> None:
    values = np.array([1.0, 2.0]).astype(np.float32)
    digest32 = TDigest.from_array(values)

    values = np.array([1.0, 2.0])
    digest64 = TDigest.from_array(values)

    with pytest.raises(TypeError, match="has a different type"):
        digest32.merge(digest64)
