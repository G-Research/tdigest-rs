"""Training conversion must preserve the established generic-input behavior."""

import numpy as np
import pytest

from tdigest_rs import TDigest


class ArraySubclass(np.ndarray):
    pass


def arranged(values, dtype, layout):
    array = np.asarray(values, dtype=dtype)
    if layout == "contiguous":
        return array
    if layout == "readonly":
        array.setflags(write=False)
        return array
    if layout == "strided":
        storage = np.zeros(array.size * 2, dtype=dtype)
        storage[::2] = array
        return storage[::2]
    if layout == "reverse":
        return array[::-1]
    if layout == "unaligned":
        storage = np.zeros(array.nbytes + 1, dtype=np.uint8)
        out = np.ndarray(array.shape, dtype=dtype, buffer=storage, offset=1)
        out[:] = array
        assert not out.flags.aligned
        return out
    if layout == "byteswapped":
        return array.astype(array.dtype.newbyteorder("S"))
    if layout == "subclass":
        return array.view(ArraySubclass)
    raise AssertionError(layout)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("f32_mode", [False, True])
@pytest.mark.parametrize(
    "layout", ["contiguous", "readonly", "strided", "reverse", "unaligned", "byteswapped", "subclass"]
)
def test_numpy_training_layouts_match_generic_values_exactly(dtype, f32_mode, layout):
    array = arranged([-10.0, -0.0, 0.0, 0.125, 0.125, 2.0, 25.0], dtype, layout)
    before = array.tobytes()
    expected = TDigest.from_array(array.tolist(), max_size=64, f32_mode=f32_mode)
    actual = TDigest.from_array(array, max_size=64, f32_mode=f32_mode)
    assert actual.to_bytes() == expected.to_bytes()
    assert array.tobytes() == before
    stored = actual.to_bytes()
    if array.flags.writeable:
        array[0] = 99.0
        assert actual.to_bytes() == stored


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("f32_mode", [False, True])
@pytest.mark.parametrize("layout", ["contiguous", "strided", "unaligned", "byteswapped"])
def test_numpy_add_and_fractional_weighted_training_match_generic_exactly(dtype, f32_mode, layout):
    values = arranged([0.0, 2.0, 4.0, 6.0], dtype, layout)
    weights = arranged([0.125, 0.25, 0.5, 0.125], dtype, layout)
    actual = TDigest.from_means_weights(values, weights, max_size=64, f32_mode=f32_mode)
    expected = TDigest.from_means_weights(values.tolist(), weights.tolist(), max_size=64, f32_mode=f32_mode)
    assert actual.to_bytes() == expected.to_bytes()
    actual.add(values)
    expected.add(values.tolist())
    assert actual.to_bytes() == expected.to_bytes()
    actual.add_weighted(values, weights)
    expected.add_weighted(values.tolist(), weights.tolist())
    assert actual.to_bytes() == expected.to_bytes()


@pytest.mark.parametrize(
    "data",
    [
        np.array([], dtype=np.float64),
        np.array(2.0),
        np.array([1, 2, 3], dtype=np.int64),
        np.array([0.25, 2.0], dtype=object),
    ],
)
def test_numpy_generic_training_fallbacks_keep_existing_values(data):
    actual = TDigest.from_array(data, max_size=64)
    expected = TDigest.from_array(data.tolist(), max_size=64)
    assert actual.to_bytes() == expected.to_bytes()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_contiguous_numpy_nonfinite_rejection_is_transactional(dtype, bad):
    values = np.array([0.0, bad, 2.0], dtype=dtype)
    with pytest.raises(ValueError):
        TDigest.from_array(values, max_size=64)
    digest = TDigest.from_array([0.0, 1.0], max_size=64)
    before = digest.to_bytes()
    with pytest.raises(ValueError):
        digest.add(values)
    assert digest.to_bytes() == before
