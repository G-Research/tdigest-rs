"""Weighted mixed tails keep valid probabilities across native surfaces."""

import base64
import subprocess

import numpy as np
import polars as pl
import pytest

from tdigest_rs import _modern as td


PROBES = [-1.0, 0.0, 0.01, 1.0, 10.0, 30.0, 48.0, 49.0, 75.0, 99.0, 100.0]


def compressed_tail(precision="f64"):
    return td.TDigest.from_array(
        np.arange(100, dtype=float), delta=1, precision=precision
    )


@pytest.mark.parametrize("precision", ["f32", "f64"])
def test_fractional_cdf_tail_preserves_shape_in_python_and_polars(precision):
    digest = compressed_tail(precision)
    expected = digest.cdf(PROBES)
    digest.scale_weights(0.001)
    actual = digest.cdf(PROBES)
    tolerance = 1e-7 if precision == "f32" else 1e-12
    assert actual == pytest.approx(expected, abs=tolerance)
    assert all(0.0 <= value <= 1.0 for value in actual)
    assert actual == sorted(actual)
    assert actual[1] == 0.0
    frame = pl.DataFrame({"blob": [digest.to_bytes()]}).select(d=td.from_bytes("blob", precision=precision))
    polars_cdf = frame.select(
        *[
            td.cdf("d", pl.lit(value).alias(f"probe_{i}"))
            for i, value in enumerate(PROBES)
        ]
    ).row(0)
    assert polars_cdf == pytest.approx(actual, abs=tolerance)
    assert frame.select(td.to_bytes("d")).item() == digest.to_bytes()


@pytest.mark.parametrize("precision", ["f32", "f64"])
def test_compressed_extrema_survive_python_and_polars_add(precision):
    original = compressed_tail(precision)
    frame = pl.DataFrame({"blob": [original.to_bytes()]}).select(
        d=td.from_bytes("blob", precision=precision)
    )
    original.add(50.0)
    updated = frame.select(d=td.add_values("d", pl.lit(50.0)))
    restored = td.TDigest.from_bytes(updated.select(td.to_bytes("d")).item())
    for digest in (original, restored):
        assert digest.quantile(0.0) == 0.0
        assert digest.quantile(1.0) == 99.0
    assert restored.to_bytes() == original.to_bytes()


def test_fractional_cdf_and_retained_extrema_in_java_and_cli(
    tmp_path, paths, compile_run_java_fn
):
    digest = compressed_tail()
    digest.scale_weights(0.001)
    expected_cdf = digest.cdf(PROBES)
    encoded = base64.b64encode(digest.to_bytes()).decode("ascii")
    java_probes = ", ".join(str(value) for value in PROBES)
    source = f"""
        import com.gresearch.tdigest.TDigest;
        import java.util.Base64;
        public class WeightedEdgeCoherence {{
          public static void main(String[] args) {{
            try (TDigest d = TDigest.fromBytes(Base64.getDecoder().decode("{encoded}"))) {{
              for (double probability : d.cdf(new double[] {{{java_probes}}})) {{
                System.out.println(probability);
              }}
              d.add(50.0);
              System.out.println(d.quantile(0.0));
              System.out.println(d.quantile(1.0));
            }}
          }}
        }}
    """
    lines = compile_run_java_fn(tmp_path, "WeightedEdgeCoherence", source).splitlines()
    assert [float(value) for value in lines[:-2]] == pytest.approx(
        expected_cdf, abs=1e-12
    )
    assert [float(value) for value in lines[-2:]] == [0.0, 99.0]

    digest.add(50.0)
    blob_path = tmp_path / "weighted-tail.tdig"
    blob_path.write_bytes(digest.to_bytes())
    for q, expected in [(0.0, 0.0), (1.0, 99.0)]:
        output = subprocess.check_output(
            [
                str(paths.cli_bin),
                "quantile",
                "--from-digest",
                str(blob_path),
                "--p",
                str(q),
                "--no-header",
                "--output",
                "csv",
            ],
            text=True,
        )
        assert float(output.strip().split(",")[-1]) == expected


@pytest.mark.parametrize("precision", ["f32", "f64"])
def test_fractional_point_mass_stays_at_the_original_mean(precision):
    digest = td.TDigest.from_means_weights(
        [-4.0, -3.0, -2.0], [1.0, 0.1, 1.0], precision=precision, max_size=1000
    )
    np.testing.assert_array_equal(digest.means, [-4.0, -3.0, -2.0])
    assert digest.cdf([-3.0])[0] == pytest.approx(0.5, abs=1e-14)
    frame = pl.DataFrame({"blob": [digest.to_bytes()]}).select(d=td.from_bytes("blob", precision=precision))
    assert frame.select(td.cdf("d", pl.lit(-3.0))).item() == pytest.approx(
        0.5, abs=1e-14
    )
    assert frame.select(td.to_bytes("d")).item() == digest.to_bytes()
