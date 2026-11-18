"""Capacity constructors and Polars expressions as used through the public package."""

import copy
import inspect
import math
import pickle
import struct

import numpy as np
import polars as pl
import pytest

import tdigest_rs as gr


def digest(values, **kwargs):
    return gr.TDigest.from_array(values, max_size=200, **kwargs)


@pytest.mark.parametrize("precision", ["f32", "f64"])
def test_weighted_constructor_and_independent_full_state_copy(precision):
    d = digest([0.0, 10.0], weights=[9.0, 1.0], precision=precision)
    assert d.median() == d.quantile(0.5) == 0.0
    assert d.cdf([0.0, 10.0]) == [0.45, 0.95]
    before = d.to_bytes()
    for restored in (copy.copy(d), copy.deepcopy(d), pickle.loads(pickle.dumps(d))):
        assert restored.to_bytes() == before
        restored.add(20.0)
        assert d.to_bytes() == before


@pytest.mark.parametrize("bad", [None, "invalid", gr.TDigest.from_array([1.0], max_size=1000)])
def test_multi_merge_is_failure_atomic_even_after_valid_prefix(bad):
    d = digest([0.0])
    other = digest([10.0])
    before = d.to_bytes()
    with pytest.raises((TypeError, ValueError)):
        d.merge([other, bad])
    assert d.to_bytes() == before
    assert other.cdf(10.0) == 0.5


@pytest.mark.parametrize("method", ["cdf", "quantile"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
def test_one_element_query_keeps_array_container(method, dtype):
    d = digest([0.0, 1.0])
    probe = np.array([0], dtype=dtype)
    result = getattr(d, method)(probe)
    assert isinstance(result, np.ndarray)
    assert result.shape == (1,) and result.dtype == np.float64
    assert result.flags.owndata
    result[0] = -123.0
    assert getattr(d, method)(probe)[0] != -123.0


def test_modern_query_shape_migration_is_explicit():
    d = digest([0.0, 1.0])
    assert isinstance(d.cdf(np.array(0.0)), float)
    assert d.cdf((0.0,)) == (0.25,)
    assert d.cdf([]) == []
    with pytest.raises(ValueError, match="one-dimensional"):
        d.cdf(np.ones((2, 2)))


@pytest.mark.parametrize(
    "operation",
    [
        lambda d: d.add([1.0, float("nan")]),
        lambda d: d.add_weighted([1.0, 2.0], [1.0, -1.0]),
        lambda d: d.add_weighted([1e308], [2.0]),
        lambda d: d.add_weighted([float.fromhex("0x0.0000000000001p-1022")], [0.1]),
        lambda d: d.scale_weights(float("inf")),
        lambda d: d.scale_values(1e308),
    ],
)
def test_failed_add_or_transform_preserves_receiver(operation):
    d = digest([1.0, 2.0])
    before = d.to_bytes()
    with pytest.raises(ValueError):
        operation(d)
    assert d.to_bytes() == before


def test_polars_aligned_empty_stays_empty_instead_of_broadcasting_neighbor():
    healthy = pl.DataFrame({"x": [0.0, 10.0]}).select(gr.tdigest("x", max_size=200).alias("d"))
    empty = pl.DataFrame({"x": pl.Series([], dtype=pl.Float64)}).select(gr.tdigest("x", max_size=200).alias("d"))
    frame = pl.concat([healthy, empty]).with_columns(x=pl.Series([5.0, 5.0]))
    result = frame.select(gr.cdf("d", pl.col("x")).alias("p"))["p"].to_list()
    assert result[0] == 0.5
    assert math.isnan(result[1])


def test_polars_missing_digest_is_rejected_for_queries_and_merges():
    healthy = pl.DataFrame({"x": [0.0, 10.0]}).select(gr.tdigest("x", max_size=200).alias("d"))
    missing = healthy.with_columns(pl.lit(None, dtype=healthy.schema["d"]).alias("d"))
    frame = pl.concat([healthy, missing])
    for expression in (gr.cdf("d", 5.0), gr.merge_tdigests("d")):
        with pytest.raises(pl.exceptions.ComputeError, match="missing digest|null in scalar field"):
            frame.select(expression)


@pytest.mark.parametrize("field,value", [("count", -1.0), ("count", 123.0), ("sum", float("nan")), ("max_size", -1)])
def test_polars_malformed_metadata_never_becomes_empty(field, value):
    frame = pl.DataFrame({"x": [0.0, 10.0]}).select(gr.tdigest("x", max_size=200).alias("d"))
    bad = frame.with_columns(pl.col("d").struct.with_fields(pl.lit(value).alias(field)))
    with pytest.raises(pl.exceptions.ComputeError):
        bad.select(gr.cdf("d", 5.0))


def test_grouped_dynamic_cdf_join_and_lazy_composition_preserves_rows():
    frame = pl.DataFrame({"row": [0, 1, 2, 3], "g": ["b", "a", "b", "a"], "x": [0.0, 100.0, 10.0, 200.0]})
    groups = frame.lazy().group_by("g").agg(gr.tdigest("x", max_size=200).alias("d"))
    out = (
        frame.lazy()
        .join(groups, on="g")
        .with_columns(gr.cdf("d", pl.col("x")).cast(pl.Float32).clip(0.0, 1.0).alias("p"))
        .sort("row")
        .collect()
    )
    assert out.height == frame.height
    assert out["p"].to_list() == [0.25, 0.25, 0.75, 0.75]
    assert out.schema["p"] == pl.Float32


@pytest.mark.parametrize("rows", [0, 1, 2])
def test_list_cdf_planner_schema_matches_runtime_for_every_cardinality(rows):
    single = pl.DataFrame({"x": [0.0, 10.0]}).select(gr.tdigest("x", max_size=200).alias("d"))
    states = single.clear() if rows == 0 else pl.concat([single] * rows)
    states = states.with_columns(pl.Series("probes", [[0.0, 5.0]] * rows, dtype=pl.List(pl.Float32)))
    lazy = states.lazy().select(gr.cdf("d", pl.col("probes")).alias("p"))
    assert lazy.collect_schema()["p"] == pl.List(pl.Float32)
    result = lazy.collect()
    assert result.height == rows and result.schema["p"] == pl.List(pl.Float32)
    assert result["p"].to_list() == [[0.25, 0.5]] * rows


@pytest.mark.parametrize("precision", ["f32", "f64"])
@pytest.mark.parametrize("rows", [0, 1, 2])
def test_binary_reduction_planner_matches_execution(precision, rows):
    blob = digest([0.0, 10.0], precision=precision).to_bytes()
    frame = pl.DataFrame({"blob": pl.Series([blob] * rows, dtype=pl.Binary)})
    lazy = frame.lazy().select(gr.from_bytes("blob", precision=precision).alias("d"))
    actual = lazy.collect()
    assert actual.schema == lazy.collect_schema()
    assert actual.height == 1
    # Use expressions: Polars 1.44.2 Series namespace dispatch has empty
    # methods on CPython 3.15, even for plain structs without this plugin.
    assert actual.select(pl.col("d").struct.field("count")).item() == 2 * rows
    assert actual.select(pl.col("d").struct.field("min")).to_series().dtype == (
        pl.Float32 if precision == "f32" else pl.Float64
    )
    with pytest.raises(ValueError, match="explicit precision"):
        gr.from_bytes("blob")


@pytest.mark.parametrize("corruption", ["mean", "count", "sum", "capacity"])
def test_invalid_binary_state_is_rejected_without_checksum_masking(corruption):
    payload = bytearray(digest([0.0, 10.0]).to_bytes())
    payload[5] &= ~1  # CRC is optional in v3; exercise semantic validation itself.
    payload[60:64] = bytes(4)
    if corruption == "mean":
        header = payload[6]
        struct.pack_into("<d", payload, header, float("nan"))
    elif corruption == "count":
        struct.pack_into("<d", payload, 20, 123.0)
    elif corruption == "sum":
        struct.pack_into("<d", payload, 52, float("inf"))
    else:
        struct.pack_into("<Q", payload, 12, 0)
    with pytest.raises(ValueError):
        gr.TDigest.from_bytes(bytes(payload))


def test_exact_tiny_point_preserves_queries_and_recovers_transformed_sum():
    tiny = float.fromhex("0x0.0000000000001p-1022")
    d = digest([tiny], weights=[0.1])
    assert d.median() == tiny
    assert d.cdf([0.0, tiny, tiny * 2]) == [0.0, 0.5, 1.0]
    before = d.to_bytes()
    restored = pickle.loads(pickle.dumps(d))
    assert restored.to_bytes() == before
    restored.scale_weights(10.0)
    assert restored.sum == tiny
    assert d.to_bytes() == before


@pytest.mark.parametrize("value", [float.fromhex("0x0.0000000000001p-1022"), 1e308])
def test_f32_training_never_silently_loses_nonzero_values(value):
    with pytest.raises(ValueError, match="precision"):
        digest([value], precision="f32")
    d = digest([1.0], precision="f32")
    before = d.to_bytes()
    for update in (lambda: d.add([2.0, value]), lambda: d.add_weighted([2.0, value], [1.0, 1.0])):
        with pytest.raises(ValueError, match="precision"):
            update()
        assert d.to_bytes() == before
    frame = pl.DataFrame({"x": pl.Series([1.0], dtype=pl.Float32)}).select(gr.tdigest("x").alias("d"))
    for expression in (gr.add_values("d", value), gr.add_weighted_values("d", value, 1.0)):
        with pytest.raises(pl.exceptions.ComputeError, match="precision"):
            frame.select(expression)


@pytest.mark.parametrize("precision,dtype", [("f32", pl.Float32), ("f64", pl.Float64)])
@pytest.mark.parametrize("engine", ["auto", "streaming"])
def test_polars_grouped_wire_roundtrip_across_execution_engines(precision, dtype, engine):
    # Preserve chunks to exercise plugin inputs beyond one contiguous Series.
    frame = pl.concat(
        [
            pl.DataFrame({"g": ["a", "b"], "x": pl.Series([0.0, 100.0], dtype=dtype)}),
            pl.DataFrame({"g": ["a", "b"], "x": pl.Series([10.0, 200.0], dtype=dtype)}),
        ],
        rechunk=False,
    )
    assert frame["x"].n_chunks() == 2
    grouped = frame.lazy().group_by("g").agg(gr.tdigest("x", max_size=200).alias("d"))
    encoded = grouped.with_columns(gr.to_bytes("d").over("g").alias("blob")).select("g", "blob")
    restored = encoded.group_by("g").agg(gr.from_bytes("blob", precision=precision).alias("d"))
    result = restored.select(
        "g",
        gr.median("d").over("g").alias("median"),
        gr.cdf("d", pl.lit(5.0, dtype=dtype)).alias("p"),
    ).sort("g")
    # The host Polars version owns the collect API; early 1.x uses streaming=.
    collect_options = (
        {"streaming": engine == "streaming"}
        if "streaming" in inspect.signature(pl.LazyFrame.collect).parameters
        else {"engine": engine}
    )
    actual = result.collect(**collect_options)
    assert actual.schema == result.collect_schema()
    assert actual["g"].to_list() == ["a", "b"]
    assert actual["median"].to_list() == [5.0, 150.0]
    assert actual["p"].to_list() == [0.5, 0.0]
    assert actual.schema["p"] == dtype


@pytest.mark.parametrize("invalid", ["null", "malformed", "wrong_precision", "mixed_precision"])
def test_polars_binary_decode_rejects_invalid_states(invalid):
    blob = digest([0.0, 10.0], precision="f64").to_bytes()
    other = digest([0.0, 10.0], precision="f32").to_bytes()
    blobs = {
        "null": [blob, None],
        "malformed": [blob, b"not a TDIG state"],
        "wrong_precision": [other],
        "mixed_precision": [blob, other],
    }[invalid]
    frame = pl.DataFrame({"blob": pl.Series(blobs, dtype=pl.Binary)})
    with pytest.raises(pl.exceptions.ComputeError):
        frame.lazy().select(gr.from_bytes("blob", precision="f64")).collect()
