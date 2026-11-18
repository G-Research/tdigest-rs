"""Malformed state must fail at native and struct reconstruction boundaries."""

import base64
import struct

from tdigest_rs import _modern as td
import polars as pl
import pytest


@pytest.mark.parametrize("precision", ["f32", "f64"])
def test_precision_roundtrip_keeps_raw_sum_and_remains_updatable(precision):
    original = td.TDigest.from_array([0.1, 0.2, 0.3], precision="f64")
    converted = original.cast_precision(precision).cast_precision("f64")
    converted.add([0.4])
    restored = td.TDigest.from_bytes(converted.to_bytes())
    assert restored.quantile(0.5) == converted.quantile(0.5)


@pytest.mark.parametrize("precision", ["f32", "f64"])
def test_legacy_export_rejects_unrepresentable_mixed_unit_bounds(precision):
    dtype = pl.Float32 if precision == "f32" else pl.Float64
    template = pl.DataFrame({"x": pl.Series([0.0, 10.0], dtype=dtype)}).select(d=td.tdigest("x"))
    state = template["d"][0]
    # A valid compressed summary of two half-weight observations.
    state.update(centroids=[{"mean": 5.0, "weight": 1.0, "atomic": False}], sum=5.0, count=1.0)
    frame = pl.DataFrame({"d": pl.Series("d", [state], dtype=template.schema["d"])})
    digest = td.TDigest.from_bytes(frame.select(td.to_bytes("d")).item())
    with pytest.raises(ValueError, match="unit-weight heuristic"):
        digest.to_bytes(version=1)
    with pytest.raises(pl.exceptions.ComputeError, match="unit-weight heuristic"):
        frame.select(td.to_bytes("d", version=1))
    for version in [2, 3]:
        restored = td.TDigest.from_bytes(digest.to_bytes(version=version))
        assert restored.quantile(0.0) == 0.0
        assert restored.quantile(1.0) == 10.0


@pytest.mark.parametrize("precision", ["f32", "f64"])
@pytest.mark.parametrize("version", [1, 2, 3])
def test_python_and_polars_reject_contradictory_wire_sum(precision, version):
    original = td.TDigest.from_array([0.0, 10.0], precision=precision)
    blob = bytearray(original.to_bytes(version=version))
    if version == 3:
        blob[5] &= ~1  # Test state validation independently of the checksum.
    struct.pack_into("<d", blob, 52 if version == 3 else 48, 2.0)
    with pytest.raises(ValueError):
        td.TDigest.from_bytes(bytes(blob))
    frame = pl.DataFrame({"blob": [bytes(blob)]})
    with pytest.raises(pl.exceptions.ComputeError):
        frame.select(td.from_bytes("blob", precision=precision))


@pytest.mark.parametrize("precision", ["f32", "f64"])
@pytest.mark.parametrize("missing", ["scale", "policy", "pin_per_side", "delta", "centroids"])
def test_polars_rejects_partial_configuration_and_missing_centroids(precision, missing):
    dtype = pl.Float32 if precision == "f32" else pl.Float64
    frame = pl.DataFrame({"x": pl.Series([], dtype=dtype)}).select(d=td.tdigest("x"))
    fields = frame.schema["d"].fields
    if missing == "centroids":
        broken = frame.select(
            d=pl.col("d").struct.with_fields(
                pl.lit(None, dtype=next(f.dtype for f in fields if f.name == missing)).alias(missing)
            )
        )
    else:
        broken = frame.select(d=pl.struct(*[pl.col("d").struct.field(f.name) for f in fields if f.name != missing]))
    for expression in [td.median("d"), td.to_bytes("d")]:
        with pytest.raises(pl.exceptions.ComputeError):
            broken.select(expression)


def test_java_and_cli_reject_contradictory_wire_sum(tmp_path, paths, compile_run_java_fn):
    blob = bytearray(td.TDigest.from_array([0.0, 10.0]).to_bytes())
    blob[5] &= ~1
    struct.pack_into("<d", blob, 52, 2.0)
    encoded = base64.b64encode(blob).decode("ascii")
    source = f"""
        import com.gresearch.tdigest.TDigest;
        import java.util.Base64;
        public class CheckedStateWire {{
          public static void main(String[] args) {{
            try (TDigest d = TDigest.fromBytes(Base64.getDecoder().decode("{encoded}"))) {{
              throw new AssertionError("accepted contradictory state");
            }} catch (IllegalArgumentException expected) {{
              System.out.println("rejected");
            }}
          }}
        }}
    """
    assert compile_run_java_fn(tmp_path, "CheckedStateWire", source).strip() == "rejected"
    import subprocess

    source_path = tmp_path / "invalid.tdig"
    source_path.write_bytes(blob)
    result = subprocess.run(
        [str(paths.cli_bin), "median", "--from-digest", str(source_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "invalid digest state" in result.stderr
