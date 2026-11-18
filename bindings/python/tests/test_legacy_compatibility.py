"""Contract regressions for the shipped legacy Python interface."""

import copy
import pickle

import numpy as np
import pytest

from tdigest_rs import TDigest


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_legacy_merge_and_update_return_new_digests_and_honor_delta(dtype):
    values = np.linspace(-10, 10, 2000, dtype=dtype)
    left = TDigest.from_array(values[:1000], delta=200)
    right = TDigest.from_array(values[1000:], delta=200)
    before = left.means.copy(), left.weights.copy()
    fine = left.merge(right, delta=300)
    coarse = left.merge(right, delta=10)
    updated = left.update(values[1000:], delta=200, merge_delta=300)
    assert fine is not left and fine is not right
    observations = np.concatenate((left.means, right.means))
    mass = np.concatenate((left.weights, right.weights))
    expected = TDigest.from_means_weights(observations, mass, delta=300)
    np.testing.assert_array_equal(fine.means, expected.means)
    np.testing.assert_array_equal(fine.weights, expected.weights)
    assert len(fine) > len(coarse)
    assert fine.weights.sum() == len(values)
    np.testing.assert_array_equal(left.means, before[0])
    np.testing.assert_array_equal(left.weights, before[1])
    np.testing.assert_array_equal(updated.means, fine.means)
    assert fine.means.dtype == dtype and fine.weights.dtype == np.uint32


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("restore", [copy.copy, copy.deepcopy, lambda d: pickle.loads(pickle.dumps(d))])
def test_original_copy_preserves_dtype_and_reconstructs_with_default_delta(dtype, restore):
    values = np.array([-np.inf, -np.inf, 0, 0, 10, np.inf, np.nan], dtype=dtype)
    original = TDigest.from_array(values, delta=17)
    restored = restore(original)
    assert restored is not original
    assert restored.means.dtype == dtype
    expected = TDigest.from_means_weights(original.means, original.weights)
    np.testing.assert_array_equal(restored.means, expected.means)
    np.testing.assert_array_equal(restored.weights, expected.weights)
    merged = restored.merge(original, delta=17)
    assert merged.weights.sum() == 12
    assert original.weights.sum() == 6


def test_old_dictionary_state_and_defaults_remain_supported():
    state = {"means": np.array([0, 10], dtype=np.float32), "weights": np.array([90, 10], dtype=np.uint32)}
    digest = TDigest.__new__(TDigest)
    digest.__setstate__(state)
    assert digest.median() == digest.quantile(0.5) == 1
    assert set(digest.to_dict()) == {"means", "weights"}


def test_original_float32_weights_are_never_rounded_to_float32():
    digest = TDigest.from_means_weights(np.array([1.0], dtype=np.float32), np.array([16_777_217], dtype=np.uint32))
    assert digest.weights.tolist() == [16_777_217]


def test_delta_uses_published_lower_tail_interpolation():
    digest = TDigest.from_array(np.array([1.0, 2.0]))
    assert digest.quantile(0.1) == float.fromhex("0x1.6666666666667p-1")


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_original_array_and_weight_argument_contract(dtype):
    with pytest.raises(ValueError, match="Array must be non-empty!"):
        TDigest.from_array(np.array([], dtype=dtype))
    with pytest.raises(TypeError):
        TDigest.from_means_weights(np.array([1.0], dtype=dtype), np.array([1], dtype=np.int64))
    digest = TDigest.from_array(np.array([1.0, 2.0], dtype=dtype))
    assert digest.means is digest.means  # Original cached_property behavior.
    assert set(digest.__getstate__()) == {"means", "weights"}


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_empty_delta_round_trips_and_mismatched_arrays(dtype):
    empty = TDigest.from_array(np.array([np.nan], dtype=dtype))
    for restored in (
        copy.copy(empty),
        copy.deepcopy(empty),
        pickle.loads(pickle.dumps(empty)),
        TDigest.from_dict(empty.to_dict()),
    ):
        assert len(restored) == 0
        assert restored.means.dtype == dtype
        assert restored.median() == 0.0
    with pytest.raises(ValueError, match="non-empty"):
        TDigest.from_means_weights(empty.means, empty.weights)
    for n, m in ((0, 1), (2, 0), (2, 1), (1, 2)):
        with pytest.raises(ValueError, match="length"):
            TDigest.from_means_weights(np.zeros(n, dtype=dtype), np.ones(m, dtype=np.uint32))
