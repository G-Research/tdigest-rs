"""Tiny stored mass keeps query shape across Python and Polars."""

import math

import numpy as np
import polars as pl
import pytest

from tdigest_rs import _modern as td


@pytest.mark.parametrize("precision", ["f32", "f64"])
@pytest.mark.parametrize("shape", ["single_atomic", "two_atomic", "mixed"])
def test_subnormal_total_query_shape_and_state_across_surfaces(precision, shape):
    dtype = np.float32 if precision == "f32" else np.float64
    if shape == "single_atomic":
        reference = td.TDigest.from_means_weights([2.0], [1.0], precision=precision)
    elif shape == "two_atomic":
        reference = td.TDigest.from_means_weights(
            [-2.0, 6.0], [1.0, 1.0], precision=precision
        )
    else:
        reference = td.TDigest.from_array(
            np.arange(100, dtype=dtype), delta=1, precision=precision
        )
        assert len(reference.means) < 100
    probes = [-3.0, -2.0, 0.0, 2.0, 6.0, 25.0, 49.5, 50.0, 98.0, 99.0, 100.0]
    qs = [0.0, 0.1, 0.25, 0.5, 0.6, 0.75, 0.9, 1.0]
    expected_cdf = reference.cdf(probes)
    expected_quantiles = reference.quantile(qs)
    tiny = td.TDigest.from_bytes(reference.to_bytes())
    # Construct first at ordinary mass, then rescale integer centroid weights
    # exactly. This isolates queries from underflow during compression itself.
    tiny.scale_weights(math.ulp(0.0))
    before = tiny.to_bytes()
    np.testing.assert_allclose(tiny.cdf(probes), expected_cdf, rtol=0, atol=2e-14)
    np.testing.assert_allclose(
        tiny.quantile(qs), expected_quantiles, rtol=0, atol=2e-14
    )

    frame = pl.DataFrame({"blob": [before]}).select(d=td.from_bytes("blob", precision=precision))
    polars_cdf = frame.select(
        *[
            td.cdf("d", pl.lit(value, dtype=pl.Float64).alias(f"probe_{i}"))
            for i, value in enumerate(probes)
        ]
    ).row(0)
    np.testing.assert_allclose(polars_cdf, expected_cdf, rtol=0, atol=2e-14)
    for q, expected in zip(qs, expected_quantiles):
        # Polars quantile uses the mean dtype; Python returns a double.
        assert frame.select(td.quantile("d", q)).item() == pytest.approx(
            float(dtype(expected)), rel=0, abs=2e-14
        )
    assert frame.select(td.to_bytes("d")).item() == before
    assert tiny.to_bytes() == before
