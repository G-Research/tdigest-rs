"""Compressed weighted averages stay ordered across Python and Polars."""

import numpy as np
import polars as pl
import pytest

import tdigest_rs as td


SIGNED_OFFSETS = [
    (-1, 4),
    (-1, 7),
    (1, 8),
    (-1, 10),
    (1, 0),
    (-1, 2),
    (1, 3),
    (-1, 5),
    (1, 6),
    (1, 7),
    (-1, 8),
    (1, 9),
    (1, 10),
    (-1, 0),
    (1, 1),
    (1, 2),
    (-1, 3),
    (1, 4),
    (1, 5),
    (-1, 6),
]
WEIGHT_UNITS = [7, 1, 2, 4, 5, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4]


@pytest.mark.parametrize("precision", ["f32", "f64"])
@pytest.mark.parametrize("reverse", [False, True])
def test_adjacent_compressed_clusters_remain_ordered_across_surfaces(
    precision, reverse
):
    dtype = np.float32 if precision == "f32" else np.float64
    pl_dtype = pl.Float32 if precision == "f32" else pl.Float64
    step = float(np.spacing(dtype(1.0)))
    values = np.array(
        [sign * (1.0 + offset * step) for sign, offset in SIGNED_OFFSETS], dtype=dtype
    )
    weights = np.array(WEIGHT_UNITS, dtype=np.float64) * 1e100
    if reverse:
        values, weights = values[::-1], weights[::-1]
    digest = td.TDigest.from_means_weights(
        values, weights, max_size=19, precision=precision
    )

    # Exercise Polars compression independently, starting from the same empty
    # configuration and adding the weighted batch through its list-column API.
    empty = td.TDigest.from_array([], max_size=19, precision=precision)
    frame = pl.DataFrame(
        {
            "blob": [empty.to_bytes()],
            "values": pl.Series([values.tolist()], dtype=pl.List(pl_dtype)),
            "weights": pl.Series([weights.tolist()], dtype=pl.List(pl.Float64)),
        }
    ).with_columns(d=td.from_bytes("blob", precision=precision))
    result = frame.select(d=td.add_weighted_values("d", "values", "weights"))
    blob = result.select(td.to_bytes("d")).item()
    restored = td.TDigest.from_bytes(blob)
    assert blob == digest.to_bytes()
    for item in (digest, restored):
        assert np.all(np.diff(item.means) > 0.0)
        assert sum(item.weights) == pytest.approx(sum(weights), rel=1e-14)
        assert item.quantile(0.0) == float(values.min())
        assert item.quantile(1.0) == float(values.max())
        queries = item.quantile(np.linspace(0.0, 1.0, 129))
        assert np.all(np.isfinite(queries))
        assert np.all(np.diff(queries) >= 0.0)
    for q in [0.0, 0.25, 0.5, 0.75, 1.0]:
        # Polars returns the digest mean dtype; Python returns a double.
        assert result.select(td.quantile("d", q)).item() == float(
            dtype(digest.quantile(q))
        )
