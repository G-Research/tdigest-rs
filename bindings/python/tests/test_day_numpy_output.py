"""Direct NumPy output allocation retains values and ordinary ndarray ownership."""

import gc

import numpy as np
import pytest

from tdigest_rs import TDigest


@pytest.mark.parametrize("f32_mode", [False, True])
@pytest.mark.parametrize("query", ["quantile", "cdf"])
def test_numpy_query_outputs_match_list_bits_and_preserve_container_types(f32_mode, query):
    digest = TDigest.from_array([-10.0, -1.0, 0.0, 0.0, 2.0, 20.0], max_size=64, f32_mode=f32_mode)
    probes = (
        [0.0, 0.125, 0.5, 0.875, 1.0]
        if query == "quantile"
        else [-float("inf"), -10.0, 0.0, 1.0, 20.0, float("inf"), float("nan")]
    )
    method = getattr(digest, query)
    expected = np.asarray(method(probes), dtype=np.float64)
    output = method(np.asarray(probes, dtype=np.float64))
    assert type(output) is np.ndarray
    assert output.dtype == np.dtype(np.float64)
    assert output.shape == expected.shape
    np.testing.assert_array_equal(output.view(np.uint64), expected.view(np.uint64))
    assert isinstance(method(probes), list)
    assert isinstance(method(tuple(probes)), tuple)
    assert isinstance(method(probes[0]), float)
    empty = method(np.array([], dtype=np.float64))
    assert empty.shape == (0,)
    assert empty.dtype == np.dtype(np.float64)


@pytest.mark.parametrize("f32_mode", [False, True])
@pytest.mark.parametrize("output_kind", ["means", "weights", "quantile", "cdf", "empty_quantile", "empty_cdf"])
def test_numpy_output_is_owned_writable_resizable_and_does_not_alias_digest(f32_mode, output_kind):
    digest = TDigest.from_array([0.0, 1.0, 2.0, 3.0], max_size=64, f32_mode=f32_mode)
    before = digest.to_bytes()
    if output_kind in ("means", "weights"):
        output = getattr(digest, output_kind)
    elif output_kind.startswith("empty_"):
        output = getattr(digest, output_kind[6:])(np.array([], dtype=np.float64))
    else:
        output = getattr(digest, output_kind)(np.array([0.0, 0.5, 1.0], dtype=np.float64))
    assert output.dtype == np.dtype(np.float64)
    assert output.flags.owndata
    assert output.flags.writeable
    assert output.flags.c_contiguous
    assert output.base is None
    if output.size:
        output[0] = 1234.0
    assert digest.to_bytes() == before
    old_size = output.size
    output.resize(old_size + 2, refcheck=False)
    assert output.shape == (old_size + 2,)
    np.testing.assert_array_equal(output[-2:], [0.0, 0.0])
    expected = output.copy()
    del digest
    gc.collect()
    np.testing.assert_array_equal(output, expected)


@pytest.mark.parametrize("f32_mode", [False, True])
def test_numpy_centroid_outputs_are_separate_exact_snapshots(f32_mode):
    digest = TDigest.from_means_weights([0.0, 2.0, 5.0], [0.125, 0.5, 1.25], max_size=64, f32_mode=f32_mode)
    means, weights = digest.means, digest.weights
    np.testing.assert_array_equal(means, [0.0, 2.0, 5.0])
    np.testing.assert_array_equal(weights, [0.125, 0.5, 1.25])
    assert not np.shares_memory(means, digest.means)
    assert not np.shares_memory(weights, digest.weights)
