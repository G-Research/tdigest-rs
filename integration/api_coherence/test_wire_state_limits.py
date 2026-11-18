"""Wire formats must reject unrepresentable state instead of silently losing it."""

import base64
import subprocess

import polars as pl
import pytest

from tdigest_rs import _modern as td


@pytest.mark.parametrize("precision", ["f32", "f64"])
@pytest.mark.parametrize("weights", [[0.1, 0.9], [1.2, 1.8]])
def test_fractional_v1_export_errors_across_python_and_polars(precision, weights):
    digest = td.TDigest.from_means_weights([0.0, 10.0], weights, precision=precision)
    with pytest.raises(ValueError, match="TDIG v1 requires positive integer"):
        digest.to_bytes(version=1)
    blob = digest.to_bytes()
    assert td.TDigest.from_bytes(blob).to_bytes() == blob
    frame = pl.DataFrame({"blob": [blob]}).select(d=td.from_bytes("blob", precision=precision))
    with pytest.raises(
        pl.exceptions.ComputeError, match="TDIG v1 requires positive integer"
    ):
        frame.select(td.to_bytes("d", version=1))
    assert frame.select(td.to_bytes("d")).item() == blob


@pytest.mark.parametrize("precision", ["f32", "f64"])
def test_full_width_edge_pins_remain_mergeable_across_python_and_polars(precision):
    digest = td.TDigest.from_array(
        list(range(400)),
        max_size=1000,
        singleton_policy="edges",
        pin_per_side=300,
        precision=precision,
    )
    blob = digest.to_bytes()
    assert digest.weights.sum() == 400
    restored = td.TDigest.from_bytes(blob)
    assert restored.to_bytes() == blob
    restored.merge(digest)
    assert restored.weights.sum() == 800
    frame = pl.DataFrame({"blob": [blob]}).select(d=td.from_bytes("blob", precision=precision))
    assert frame.select(pl.col("d").struct.field("pin_per_side")).item() == 300
    assert frame.select(td.to_bytes("d")).item() == blob
    for version in [1, 2]:
        with pytest.raises(ValueError, match="pin_per_side > 255 requires TDIG v3"):
            digest.to_bytes(version=version)
        with pytest.raises(
            pl.exceptions.ComputeError, match="pin_per_side > 255 requires TDIG v3"
        ):
            frame.select(td.to_bytes("d", version=version))


def test_invalid_v1_zero_weight_is_value_error():
    digest = td.TDigest.from_array([1.0], precision="f64")
    blob = bytearray(digest.to_bytes(version=1))
    blob[64:72] = bytes(8)
    with pytest.raises(ValueError, match="centroid weight"):
        td.TDigest.from_bytes(bytes(blob))


def test_wire_limits_through_java_and_cli(tmp_path, paths, compile_run_java_fn):
    fractional = td.TDigest.from_means_weights([0.0, 10.0], [0.1, 0.9])
    large_pins = td.TDigest.from_array(
        [0.0, 1.0, 2.0], max_size=1000, singleton_policy="edges", pin_per_side=300
    )
    fractional_encoded = base64.b64encode(fractional.to_bytes()).decode("ascii")
    pins_encoded = base64.b64encode(large_pins.to_bytes()).decode("ascii")
    source = f"""
        import com.gresearch.tdigest.TDigest;
        import java.util.Base64;
        public class WireStateLimits {{
          public static void main(String[] args) {{
            try (TDigest fractional = TDigest.fromBytes(Base64.getDecoder().decode("{fractional_encoded}"));
                 TDigest pins = TDigest.fromBytes(Base64.getDecoder().decode("{pins_encoded}"))) {{
              try {{ fractional.toBytes(1); throw new AssertionError("lost fractional weight"); }}
              catch (IllegalArgumentException expected) {{ }}
              for (int version : new int[] {{1, 2}}) {{
                try {{ pins.toBytes(version); throw new AssertionError("lost pin config"); }}
                catch (IllegalArgumentException expected) {{ }}
              }}
              System.out.println(Base64.getEncoder().encodeToString(pins.toBytes()));
            }}
          }}
        }}
    """
    output = compile_run_java_fn(tmp_path, "WireStateLimits", source)
    assert base64.b64decode(output) == large_pins.to_bytes()
    # CLI accepts an existing digest when building/exporting another wire version.
    for name, digest, version, expected in [
        ("fractional", fractional, 1, "positive integer"),
        ("pins", large_pins, 2, "pin_per_side > 255"),
    ]:
        source_path = tmp_path / f"{name}.tdig"
        target_path = tmp_path / f"{name}-old.tdig"
        source_path.write_bytes(digest.to_bytes())
        result = subprocess.run(
            [
                str(paths.cli_bin),
                "build",
                "--from-digest",
                str(source_path),
                "--to-digest",
                str(target_path),
                "--wire-version",
                f"v{version}",
            ],
            text=True,
            capture_output=True,
        )
        assert result.returncode != 0
        assert expected in result.stderr
        assert not target_path.exists()
