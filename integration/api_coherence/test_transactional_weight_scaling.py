"""Finite scale factors must not turn centroid weights into invalid state."""

import math

import numpy as np
import polars as pl
import pytest

from tdigest_rs import _modern as td


CASES = [
    pytest.param([math.ulp(0.0)], 0.5, id="subnormal-underflow"),
    pytest.param([1e308], 2.0, id="single-overflow"),
    pytest.param([1e-100, 1e-300], 1e-100, id="later-underflow"),
    pytest.param([5e307, 5e307], 2.0, id="total-overflow"),
]


@pytest.mark.parametrize("precision", ["f32", "f64"])
@pytest.mark.parametrize("surface", ["python", "polars"])
@pytest.mark.parametrize("weights,factor", CASES)
def test_invalid_scaled_weights_are_normal_errors_and_leave_state_unchanged(
    precision, surface, weights, factor
):
    digest = td.TDigest.from_means_weights(
        list(range(len(weights))), weights, precision=precision
    )
    before = digest.to_bytes()
    if surface == "python":
        with pytest.raises(ValueError, match="scaled (centroid|total) weight"):
            digest.scale_weights(factor)
    else:
        frame = pl.DataFrame({"blob": [before]}).select(d=td.from_bytes("blob", precision=precision))
        with pytest.raises(
            pl.exceptions.ComputeError, match="scaled (centroid|total) weight"
        ):
            frame.select(td.scale_weights("d", factor))
        assert frame.select(td.to_bytes("d")).item() == before
    assert digest.to_bytes() == before


@pytest.mark.parametrize("precision", ["f32", "f64"])
@pytest.mark.parametrize(
    "weights,factor", [([1.0, 3.0], math.ulp(0.0)), ([1e308], 0.5)]
)
def test_valid_extreme_scaling_retains_cross_surface_state_and_query_shape(
    precision, weights, factor
):
    values = list(range(len(weights)))
    digest = td.TDigest.from_means_weights(values, weights, precision=precision)
    before = digest.to_bytes()
    expected_cdf = digest.cdf(values)
    expected_quantiles = digest.quantile([0.0, 0.25, 0.5, 0.75, 1.0])
    frame = pl.DataFrame({"blob": [before]}).select(d=td.from_bytes("blob", precision=precision))
    result = frame.select(d=td.scale_weights("d", factor))
    digest.scale_weights(factor)
    assert result.select(td.to_bytes("d")).item() == digest.to_bytes()
    assert all(math.isfinite(w) and w > 0 for w in digest.weights)
    np.testing.assert_array_equal(digest.cdf(values), expected_cdf)
    np.testing.assert_array_equal(
        digest.quantile([0.0, 0.25, 0.5, 0.75, 1.0]), expected_quantiles
    )
