"""Shared-core regressions must survive surface and wire conversions."""

import base64
import subprocess

import polars as pl
import pytest

from tdigest_rs import _modern as td


@pytest.mark.parametrize("precision", ["f32", "f64"])
def test_delta_and_atomic_weights_survive_polars(precision):
    original = td.TDigest.from_means_weights(
        [0.0, 10.0], [90.0, 10.0], delta=17, precision=precision
    )
    frame = pl.DataFrame({"blob": [original.to_bytes()]}).select(
        d=td.from_bytes("blob", precision=precision)
    )
    assert frame.select(td.median("d")).item() == 0.0
    restored = td.TDigest.from_bytes(frame.select(td.to_bytes("d")).item())
    assert restored.to_bytes() == original.to_bytes()
    restored.merge(original)
    assert restored.weights.sum() == 200
    with pytest.raises(pl.exceptions.ComputeError, match="delta mode requires TDIG v3"):
        frame.select(td.to_bytes("d", version=2))


@pytest.mark.parametrize("scale,policy,pin", [("k1", "off", None), ("k3", "edges", 3)])
def test_polars_roundtrip_preserves_modern_configuration(scale, policy, pin):
    kwargs = {"max_size": 100, "scale": scale, "singleton_policy": policy}
    if pin is not None:
        kwargs["pin_per_side"] = pin
    original = td.TDigest.from_array(list(range(200)), **kwargs)
    frame = pl.DataFrame({"blob": [original.to_bytes()]}).select(
        d=td.from_bytes("blob", precision="f64")
    )
    restored = td.TDigest.from_bytes(frame.select(td.to_bytes("d")).item())
    assert restored.to_bytes() == original.to_bytes()
    restored.merge(original)


@pytest.mark.parametrize("precision", ["f32", "f64"])
@pytest.mark.parametrize("bad_config", ["scale", "use", "edges"])
def test_polars_rejects_incompatible_delta_struct_before_query_or_export(precision, bad_config):
    original = td.TDigest.from_array([0.0, 1.0, 2.0], delta=17, precision=precision)
    frame = pl.DataFrame({"blob": [original.to_bytes()]}).select(d=td.from_bytes("blob", precision=precision))
    replacements = {
        "scale": [pl.lit(2, dtype=pl.Int64).alias("scale")],
        "use": [pl.lit(1, dtype=pl.Int64).alias("policy")],
        "edges": [
            pl.lit(2, dtype=pl.Int64).alias("policy"),
            pl.lit(3, dtype=pl.Int64).alias("pin_per_side"),
        ],
    }
    broken = frame.select(d=pl.col("d").struct.with_fields(*replacements[bad_config]))
    for expression in [td.median("d"), td.to_bytes("d"), td.to_bytes("d", version=3)]:
        with pytest.raises(
            pl.exceptions.ComputeError,
            match="delta mode requires K2Norm scale and singleton policy Off",
        ):
            broken.select(expression)


@pytest.mark.parametrize("precision", ["f32", "f64"])
def test_old_polars_structs_remain_readable(precision):
    original = td.TDigest.from_array([0.0, 1.0, 2.0], precision=precision)
    frame = pl.DataFrame({"blob": [original.to_bytes()]}).select(
        d=td.from_bytes("blob", precision=precision)
    )
    # Reconstruct the previous six-field schema, without configuration or kind.
    old = frame.select(
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
                pl.col("d").struct.field(field)
                for field in ("sum", "min", "max", "count", "max_size")
            ],
        )
    )
    assert old.select(td.median("d")).item() == 1.0
    restored = td.TDigest.from_bytes(old.select(td.to_bytes("d")).item())
    assert restored.inner_kind() == precision
    assert restored.weights.sum() == 3
    assert restored.quantile(0.5) == 1.0


def test_weighted_median_and_delta_wire_through_java_and_cli(
    tmp_path, paths, compile_run_java_fn
):
    original = td.TDigest.from_means_weights([0.0, 10.0], [90.0, 10.0], delta=17)
    blob = original.to_bytes()
    encoded = base64.b64encode(blob).decode("ascii")
    source = f"""
        import com.gresearch.tdigest.TDigest;
        import java.util.Base64;
        public class LegacyWeightedWire {{
          public static void main(String[] args) {{
            try (TDigest d = TDigest.fromBytes(Base64.getDecoder().decode("{encoded}"))) {{
              System.out.println(d.median());
              System.out.println(Base64.getEncoder().encodeToString(d.toBytes()));
              try {{ d.toBytes(2); throw new AssertionError("lost delta"); }}
              catch (IllegalArgumentException expected) {{ }}
            }}
          }}
        }}
    """
    lines = compile_run_java_fn(tmp_path, "LegacyWeightedWire", source).splitlines()
    assert float(lines[0]) == 0.0
    assert base64.b64decode(lines[1]) == blob
    source_path = tmp_path / "delta.tdig"
    source_path.write_bytes(blob)
    output = subprocess.check_output(
        [
            str(paths.cli_bin),
            "median",
            "--from-digest",
            str(source_path),
            "--no-header",
            "--output",
            "csv",
        ],
        text=True,
    )
    assert float(output.strip().split(",")[-1]) == 0.0
