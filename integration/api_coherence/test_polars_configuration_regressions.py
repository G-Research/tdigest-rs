"""Configuration affects compression and survives identity operations and planning."""

import numpy as np
import polars as pl
import pytest

from tdigest_rs import _modern as td


def as_frame(digest):
    return pl.DataFrame({"blob": [digest.to_bytes()]}).select(d=td.from_bytes("blob", precision=digest.inner_kind()))


def legacy_schema(frame):
    return frame.select(
        d=pl.struct(
            pl.col("d")
            .struct.field("centroids")
            .list.eval(
                pl.struct(
                    pl.element().struct.field("mean"),
                    pl.element().struct.field("weight"),
                )
            ),
            *[
                pl.col("d").struct.field(k)
                for k in ("sum", "min", "max", "count", "max_size")
            ],
        )
    )


@pytest.mark.parametrize("dtype,precision", [(np.float32, "f32"), (np.float64, "f64")])
@pytest.mark.parametrize("scale,policy,pin", [("k1", "off", None), ("k3", "edges", 2)])
def test_polars_compresses_with_requested_configuration(
    dtype, precision, scale, policy, pin
):
    values = np.random.default_rng(123).lognormal(size=10_000).astype(dtype)
    kwargs = dict(max_size=10, scale=scale, singleton_policy=policy)
    if pin is not None:
        kwargs["pin_per_side"] = pin
    direct = td.TDigest.from_array(values, precision=precision, **kwargs)
    frame = pl.DataFrame({"x": values}).select(d=td.tdigest("x", **kwargs))
    restored = td.TDigest.from_bytes(frame.select(td.to_bytes("d")).item())
    assert restored.inner_kind() == precision
    assert restored.to_bytes() == direct.to_bytes()


def identity_expressions(precision):
    dtype = pl.Float32 if precision == "f32" else pl.Float64
    empty = pl.lit([], dtype=pl.List(dtype))
    return [
        td.merge_tdigests("d"),
        td.add_values("d", empty),
        td.add_weighted_values("d", empty, empty),
        td.scale_weights("d", 1.0),
        td.scale_values("d", 1.0),
    ]


@pytest.mark.parametrize("precision", ["f32", "f64"])
@pytest.mark.parametrize("empty", [False, True])
def test_polars_identity_mutations_preserve_delta_and_extrema(precision, empty):
    values = [] if empty else list(range(100))
    original = td.TDigest.from_array(values, delta=1, precision=precision)
    frame = as_frame(original)
    for expression in identity_expressions(precision):
        updated = frame.select(d=expression)
        assert updated.select(td.to_bytes("d")).item() == original.to_bytes()


@pytest.mark.parametrize("precision", ["f32", "f64"])
def test_legacy_struct_mutations_report_the_schema_they_emit(precision):
    old = legacy_schema(as_frame(td.TDigest.from_array([0, 1, 2], precision=precision)))
    for expression in identity_expressions(precision):
        lazy = old.lazy().select(d=expression)
        actual = lazy.collect()
        assert lazy.collect_schema() == actual.schema
        # This requires the planner to know the newly emitted configuration fields.
        assert (
            lazy.select(pl.col("d").struct.field("scale")).collect().item()
            == actual.select(pl.col("d").struct.field("scale")).item()
        )
