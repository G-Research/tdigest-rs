"""Float32 means must not make large or fractional observation weights disappear."""

import base64
import subprocess

import numpy as np
import polars as pl
import pytest

from tdigest_rs import _modern as td


PILE = 2**24


def two_piles():
    return td.TDigest.from_means_weights([0.0, 10.0], [PILE, PILE], precision="f32")


def as_frame(digest):
    return pl.DataFrame({"blob": [digest.to_bytes()]}).select(d=td.from_bytes("blob", precision=digest.inner_kind()))


def old_f32_weight_struct(frame):
    # The prior schema already carried kinds/configuration, but narrowed each weight.
    centroids = (
        pl.col("d")
        .struct.field("centroids")
        .list.eval(
            pl.struct(
                pl.element().struct.field("mean"),
                pl.element().struct.field("weight").cast(pl.Float32),
                pl.element().struct.field("atomic"),
            )
        )
    )
    return frame.select(d=pl.col("d").struct.with_fields(centroids))


def test_large_f32_pile_keeps_each_new_observation_and_correct_median():
    digest = two_piles()
    assert digest.median() == 5.0
    for _ in range(1000):
        digest.add(0.0)
    np.testing.assert_array_equal(digest.weights, [PILE + 1000, PILE])
    assert digest.weights.dtype == np.float64
    assert digest.median() == 0.0
    assert digest.cdf([5.0])[0] == (PILE + 1000) / (2 * PILE + 1000)
    blob = digest.to_bytes()
    assert td.TDigest.from_bytes(blob).to_bytes() == blob
    assert digest.cast_precision("f64").cast_precision("f32").to_bytes() == blob


@pytest.mark.parametrize("old_schema", [False, True])
@pytest.mark.parametrize("operation", ["add", "weighted", "merge"])
def test_polars_promotes_old_weights_and_reports_canonical_update_schema(
    old_schema, operation
):
    digest = two_piles()
    frame = as_frame(digest)
    if old_schema:
        frame = old_f32_weight_struct(frame)
    if operation == "add":
        expression = td.add_values("d", pl.lit(0.0, dtype=pl.Float32))
    elif operation == "weighted":
        expression = td.add_weighted_values(
            "d", pl.lit(0.0, dtype=pl.Float32), pl.lit(1.0)
        )
    else:
        extra = as_frame(
            td.TDigest.from_array(np.array([0.0], dtype=np.float32), precision="f32")
        )
        if old_schema:
            extra = old_f32_weight_struct(extra)
        frame = pl.concat([frame, extra])
        expression = td.merge_tdigests("d")
    lazy = frame.lazy().select(d=expression)
    updated = lazy.collect()
    assert lazy.collect_schema() == updated.schema
    inner = (
        updated.select(pl.col("d").struct.field("centroids")).to_series().dtype.inner
    )
    assert inner == pl.Struct(
        {"mean": pl.Float32, "weight": pl.Float64, "atomic": pl.Boolean}
    )
    restored = td.TDigest.from_bytes(updated.select(td.to_bytes("d")).item())
    digest.add(0.0)
    assert restored.to_bytes() == digest.to_bytes()
    assert restored.median() == 0.0
    assert updated.select(td.median("d")).item() == 0.0


@pytest.mark.parametrize("weight", [1e-46, 0.1, float(PILE + 1), 1e39])
def test_f32_weights_retain_f64_mass_across_wire_and_polars(weight):
    digest = td.TDigest.from_means_weights([1.0], [weight], precision="f32")
    assert digest.weights[0] == weight
    blob = digest.to_bytes()
    assert blob[5] & 0x08  # old readers must reject unsupported weight precision
    restored = td.TDigest.from_bytes(blob)
    assert restored.weights[0] == weight
    frame = as_frame(digest)
    assert frame.select(td.to_bytes("d")).item() == blob
    assert frame.select(td.median("d")).item() == 1.0
    # Exporting older formats requires explicit mean promotion to an f64 digest.
    with pytest.raises(ValueError, match="TDIG v3"):
        digest.to_bytes(version=2)
    with pytest.raises(pl.exceptions.ComputeError, match="TDIG v3"):
        frame.select(td.to_bytes("d", version=2))
    widened = digest.cast_precision("f64")
    assert td.TDigest.from_bytes(widened.to_bytes(version=2)).weights[0] == weight


def test_f32_large_weights_survive_java_update_and_cli_query(
    tmp_path, paths, compile_run_java_fn
):
    digest = two_piles()
    encoded = base64.b64encode(digest.to_bytes()).decode("ascii")
    source = f"""
        import com.gresearch.tdigest.TDigest;
        import java.util.Base64;
        public class F32WeightMass {{
          public static void main(String[] args) {{
            try (TDigest d = TDigest.fromBytes(Base64.getDecoder().decode("{encoded}"))) {{
              d.add(0.0f);
              if (d.median() != 0.0) throw new AssertionError("median reversed");
              System.out.println(Base64.getEncoder().encodeToString(d.toBytes()));
            }}
          }}
        }}
    """
    digest.add(0.0)
    assert (
        base64.b64decode(compile_run_java_fn(tmp_path, "F32WeightMass", source))
        == digest.to_bytes()
    )
    blob_path = tmp_path / "large-f32-piles.tdig"
    blob_path.write_bytes(digest.to_bytes())
    output = subprocess.check_output(
        [
            str(paths.cli_bin),
            "quantile",
            "--from-digest",
            str(blob_path),
            "--p",
            "0.5",
            "--no-header",
            "--output",
            "csv",
        ],
        text=True,
    )
    assert float(output.strip().split(",")[-1]) == 0.0


@pytest.mark.parametrize(
    "field,value",
    [
        ("weight", 0.0),
        ("weight", -1.0),
        ("weight", float("inf")),
        ("weight", float("nan")),
        ("mean", float("nan")),
    ],
)
def test_malformed_polars_mass_or_mean_returns_error(field, value):
    frame = as_frame(two_piles())
    dtype = pl.Float32 if field == "mean" else pl.Float64
    replacement = (
        pl.col("d")
        .struct.field("centroids")
        .list.eval(
            pl.element().struct.with_fields(pl.lit(value, dtype=dtype).alias(field))
        )
    )
    broken = frame.select(d=pl.col("d").struct.with_fields(replacement))
    with pytest.raises(pl.exceptions.ComputeError, match=f"invalid field '{field}'"):
        broken.select(td.median("d"))
