"""Overflow errors and finite extreme queries across the public language surfaces."""

import base64
import subprocess

from tdigest_rs import _modern as td
import numpy as np
import polars as pl
import pytest
from tdigest_rs import TDigest as OriginalDigest


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_original_uint32_limits_and_published_float_overflow(dtype):
    maximum = np.iinfo(np.uint32).max
    with pytest.raises(ValueError, match="overflow"):
        OriginalDigest.from_means_weights(
            np.array([1, 2], dtype=dtype), np.array([maximum, 1], dtype=np.uint32)
        )
    full = OriginalDigest.from_means_weights(
        np.array([1], dtype=dtype), np.array([maximum], dtype=np.uint32)
    )
    before = full.means.copy(), full.weights.copy()
    one = OriginalDigest.from_array(np.array([2], dtype=dtype))
    with pytest.raises(ValueError, match="overflow"):
        full.merge(one)
    np.testing.assert_array_equal(full.means, before[0])
    np.testing.assert_array_equal(full.weights, before[1])
    assert full.quantile(0.5) == 1.0
    huge = np.finfo(dtype).max
    means = np.array([huge / 2, huge], dtype=dtype)
    weights = np.array([10, 10], dtype=np.uint32)
    extreme = OriginalDigest.from_means_weights(means, weights)
    # The published wheel performs both query products in the input dtype.
    assert extreme.median() == np.inf
    assert extreme.trimmed_mean(0.0, 1.0) == np.inf


@pytest.mark.parametrize("precision", ["f32", "f64"])
def test_modern_mass_overflow_is_atomic_in_python_and_rejected_by_polars(precision):
    maximum = np.finfo(np.float64).max
    d = td.TDigest.from_means_weights([0.0], [maximum], precision=precision)
    before = d.to_bytes()
    for operation in [
        lambda: d.add_weighted([0.0], [maximum]),
        lambda: d.merge(d),
        lambda: d.scale_weights(2.0),
    ]:
        with pytest.raises(ValueError):
            operation()
        assert d.to_bytes() == before
    single = pl.DataFrame({"blob": [before]}).select(d=td.from_bytes("blob", precision=precision))
    frame = pl.concat([single, single])
    with pytest.raises(pl.exceptions.ComputeError):
        frame.select(td.merge_tdigests("d"))


@pytest.mark.parametrize("precision,dtype", [("f32", np.float32), ("f64", np.float64)])
def test_extreme_queries_are_finite_in_python_and_polars(precision, dtype):
    huge = float(np.finfo(dtype).max)
    d = td.TDigest.from_array([-huge, huge], precision=precision)
    before = d.to_bytes()
    assert d.quantile(0.5) == 0.0
    assert d.cdf([0.0]) == [0.5]
    frame = pl.DataFrame({"blob": [before]}).select(d=td.from_bytes("blob", precision=precision))
    assert frame.select(td.quantile("d", 0.5)).item() == 0.0
    assert frame.select(td.cdf("d", 0.0)).item() == 0.5
    assert d.to_bytes() == before


def test_java_overflow_errors_preserve_state_and_extreme_queries(
    tmp_path, compile_run_java_fn
):
    full = td.TDigest.from_means_weights([0.0], [np.finfo(np.float64).max]).to_bytes()
    encoded = base64.b64encode(full).decode("ascii")
    source = f"""
        import com.gresearch.tdigest.TDigest;
        import java.util.Arrays;
        import java.util.Base64;
        public class OverflowBoundaries {{
          public static void main(String[] args) {{
            try (TDigest d = TDigest.fromBytes(Base64.getDecoder().decode("{encoded}"))) {{
              byte[] before = d.toBytes();
              for (Runnable operation : new Runnable[] {{
                  () -> d.addWeighted(0.0, Double.MAX_VALUE),
                  () -> d.merge(d), () -> d.scaleWeights(2.0) }}) {{
                try {{ operation.run(); throw new AssertionError("overflow accepted"); }}
                catch (IllegalArgumentException expected) {{}}
                if (!Arrays.equals(before, d.toBytes())) throw new AssertionError("partial update");
              }}
            }}
            try (TDigest d = TDigest.builder().build(new double[] {{-Double.MAX_VALUE, Double.MAX_VALUE}})) {{
              if (d.quantile(0.5) != 0.0 || d.cdf(new double[] {{0.0}})[0] != 0.5)
                throw new AssertionError("invalid extreme query");
            }}
            System.out.println("ok");
          }}
        }}
    """
    assert compile_run_java_fn(tmp_path, "OverflowBoundaries", source).strip() == "ok"


def test_cli_creation_rejects_overflow_and_extreme_queries_remain_finite(
    tmp_path, paths
):
    samples = tmp_path / "overflow.txt"
    samples.write_text("1.7976931348623157e308\n1.7976931348623157e308\n")
    result = subprocess.run(
        [
            str(paths.cli_bin),
            "build",
            "--input",
            str(samples),
            "--to-digest",
            str(tmp_path / "bad.tdig"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "overflow" in result.stderr
    huge = np.finfo(np.float64).max
    blob = tmp_path / "extreme.tdig"
    blob.write_bytes(td.TDigest.from_array([-huge, huge]).to_bytes())
    for command, probe, value, expected in [
        ("quantile", "--p", "0.5", 0.0),
        ("cdf", "--x", "0.0", 0.5),
    ]:
        result = subprocess.run(
            [
                str(paths.cli_bin),
                command,
                "--from-digest",
                str(blob),
                probe,
                value,
                "--no-header",
                "--output",
                "csv",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert float(result.stdout.strip().split(",")[-1]) == expected
