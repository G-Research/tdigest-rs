"""Independent modern constructors preserve exact state and error isolation."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from tdigest_rs._modern import TDigest


def container(values, kind):
    if kind == "list":
        return values.tolist()
    if kind == "tuple":
        return tuple(values)
    if kind == "strided":
        storage = np.repeat(values, 2)
        values = storage[::2]
    values.setflags(write=False)
    return values


@pytest.mark.parametrize("f32_mode", [False, True], ids=["f64", "f32"])
@pytest.mark.parametrize("weighted", [False, True], ids=["raw", "weighted"])
@pytest.mark.parametrize("kind", ["array", "list", "tuple", "strided"])
def test_independent_constructors_match_serial_wire_and_keep_inputs(
    f32_mode, weighted, kind
):
    dtype = np.float32 if f32_mode else np.float64
    values = np.random.default_rng(20261020).normal(size=8192).astype(dtype)
    weights = (np.arange(values.size) % 7 + 1).astype(np.float64) / 8.0
    inputs = container(values.copy(), kind)
    input_weights = container(weights.copy(), kind)

    def build(_):
        options = {"delta": 300.0, "f32_mode": f32_mode}
        if weighted:
            return TDigest.from_means_weights(inputs, input_weights, **options)
        return TDigest.from_array(inputs, **options)

    expected = build(0).to_bytes()
    with ThreadPoolExecutor(max_workers=4) as pool:
        digests = list(pool.map(build, range(12)))
    assert len({id(digest) for digest in digests}) == 12
    assert all(digest.to_bytes() == expected for digest in digests)
    np.testing.assert_array_equal(np.asarray(inputs), values)
    np.testing.assert_array_equal(np.asarray(input_weights), weights)


@pytest.mark.parametrize("f32_mode", [False, True], ids=["f64", "f32"])
@pytest.mark.parametrize("weighted", [False, True], ids=["raw", "weighted"])
def test_concurrent_invalid_constructor_errors_stay_local(f32_mode, weighted):
    def construct(invalid):
        options = {"delta": 300.0, "f32_mode": f32_mode}
        try:
            if weighted:
                weights = [1.0, -1.0 if invalid else 0.5, 2.0]
                digest = TDigest.from_means_weights([0.0, 1.0, 2.0], weights, **options)
            else:
                values = [0.0, float("nan") if invalid else 1.0, 2.0]
                digest = TDigest.from_array(values, **options)
            return ("ok", digest.to_bytes())
        except ValueError as error:
            return ("error", str(error))

    expected = {invalid: construct(invalid) for invalid in [False, True]}
    assert expected[False][0] == "ok"
    assert expected[True][0] == "error"
    flags = [False, True] * 8
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(construct, flags))
    assert results == [expected[invalid] for invalid in flags]
