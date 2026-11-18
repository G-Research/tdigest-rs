"""High delta must not turn exact piles into a synthetic continuous distribution."""

import numpy as np
import polars as pl
import pytest

from tdigest_rs import _modern as td


def assert_two_piles(digest):
    np.testing.assert_array_equal(digest.means, [0.0, 100.0])
    np.testing.assert_array_equal(digest.weights, [2.0, 1.0])
    np.testing.assert_array_equal(digest.quantile([0.0, 0.1, 0.5, 0.9, 1.0]), [0.0, 0.0, 0.0, 100.0, 100.0])
    np.testing.assert_array_equal(digest.cdf([0.0, 25.0, 100.0]), [1 / 3, 2 / 3, 5 / 6])


@pytest.mark.parametrize("precision", ["f32", "f64"])
@pytest.mark.parametrize("delta", [300.0, 600.0, 800.0, 810.0, 1200.0, 2400.0])
def test_high_delta_direct_update_and_merge_preserve_exact_piles(precision, delta):
    def make(values):
        return td.TDigest.from_array(values, delta=delta, precision=precision)

    direct = make([0.0, 0.0, 100.0])
    updated = make([0.0])
    updated.add([0.0, 100.0])
    merged = make([0.0, 0.0])
    merged.merge(make([100.0]))
    kway = td.TDigest.merge_all([make([0.0]), make([100.0]), make([0.0])])
    for digest in [direct, updated, merged, kway]:
        assert_two_piles(digest)


@pytest.mark.parametrize("precision", ["f32", "f64"])
@pytest.mark.parametrize("route", ["update", "merge"])
def test_polars_keeps_high_delta_when_mutating_serialized_state(precision, route):
    def make(values):
        return td.TDigest.from_array(values, delta=1200, precision=precision)

    direct = make([0.0, 0.0, 100.0])
    if route == "update":
        frame = pl.DataFrame({"blob": [make([0.0]).to_bytes()]}).select(d=td.from_bytes("blob", precision=precision))
        result = frame.select(d=td.add_values("d", pl.lit([0.0, 100.0])))
    else:
        frame = pl.DataFrame({"blob": [make([0.0, 0.0]).to_bytes(), make([100.0]).to_bytes()]}).select(
            d=td.from_bytes("blob", precision=precision)
        )
        result = frame.select(d=td.merge_tdigests("d"))
    blob = result.select(td.to_bytes("d")).item()
    assert blob == direct.to_bytes()
    assert_two_piles(td.TDigest.from_bytes(blob))
    assert result.select(td.median("d")).item() == 0.0


@pytest.mark.parametrize("precision", ["f32", "f64"])
@pytest.mark.parametrize("policy", ["use", "edges"])
def test_polars_explicit_k2norm_capacity_preserves_small_interiors(precision, policy):
    dtype = pl.Float32 if precision == "f32" else pl.Float64
    values = [-100.0, 0.0, 100.0, 200.0]
    options = {"max_size": 1200, "scale": "k2norm", "singleton_policy": policy, "precision": precision}
    if policy == "edges":
        options["pin_per_side"] = 1
    result = pl.DataFrame({"x": pl.Series(values, dtype=dtype)}).select(d=td.tdigest("x", **options))
    blob = result.select(td.to_bytes("d")).item()
    expected = td.TDigest.from_array(values, **options)
    assert blob == expected.to_bytes()
    np.testing.assert_array_equal(expected.means, values)
    np.testing.assert_array_equal(expected.weights, [1.0, 1.0, 1.0, 1.0])
