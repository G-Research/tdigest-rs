from __future__ import annotations

import base64
import textwrap
from pathlib import Path

import pytest


# =====================================================================
# Helpers (use fixtures defined in conftest)
# =====================================================================

# NOTE: We intentionally do NOT define a local _cli_build_args here to
# avoid drift. Always use the fixtures `dataset`, `expect`, `cfg`,
# `assert_close_fn`, `add_pin_kw_fn` defined in conftest.


def _run_java_emit_blob(
    *,
    class_name: str,
    compile_run_java_fn,
    java_builder_chain_fn,
    tmp_path: Path,
    cfg,
    dataset,
    expect,
    precision_override: str | None = None,
    version_override: int | None = None,
) -> tuple[float, float, bytes]:
    """
    Build a Java TDigest, emit p50,cdf(x),base64(blob), and return parsed values.
    """
    data_lit = ", ".join(str(float(v)) for v in dataset["DATA"])
    p_lit = str(expect["P"])
    x_lit = str(expect["X"])
    builder = java_builder_chain_fn(cfg, precision=precision_override)
    to_bytes_expr = (
        f"d.toBytes({int(version_override)})"
        if version_override is not None
        else "d.toBytes()"
    )

    java_src = textwrap.dedent(
        f"""
        import com.gresearch.tdigest.TDigest;
        import com.gresearch.tdigest.TDigest.Scale;
        import com.gresearch.tdigest.TDigest.SingletonPolicy;
        import com.gresearch.tdigest.TDigest.Precision;
        import java.util.Base64;

        public class {class_name} {{
          public static void main(String[] args) {{
            double[] data = new double[] {{{data_lit}}};
            try (TDigest d = TDigest.builder()
                    {builder}
                    .build(data)) {{
              double p50 = d.quantile({p_lit});
              double[] ps = d.cdf(new double[] {{{x_lit}}});
              double cdf2 = ps[0];

              byte[] buf = {to_bytes_expr};
              String b64 = Base64.getEncoder().encodeToString(buf);

              System.out.println(p50 + "," + cdf2 + "," + b64);
            }}
          }}
        }}
        """
    ).strip()

    out = compile_run_java_fn(tmp_path, class_name, java_src)
    p50_s, cdf_s, b64 = out.split(",", 2)
    return float(p50_s), float(cdf_s), base64.b64decode(b64.encode("ascii"))


# =====================================================================
# Category 1: JAVA → PYTHON — TDigest.toBytes() → TDigest.from_bytes()
#   Spec:
#     - Java builds digest from canonical data.
#     - Java emits base64(toBytes()).
#     - Python decodes/rebuilds and matches quantile/cdf.
# =====================================================================
class TestJavaToPythonSerde:
    @pytest.mark.parametrize(
        ("precision_override", "expected_kind", "class_name"),
        [
            (None, None, "TDJavaToPythonSerde"),
            ("F32", "f32", "TDJavaToPythonSerdeF32"),
        ],
        ids=["cfg_precision", "f32"],
    )
    def test_java_to_python(
        self,
        dataset,
        expect,
        cfg,
        tmp_path: Path,
        assert_close_fn,
        compile_run_java_fn,
        java_builder_chain_fn,
        precision_override,
        expected_kind,
        class_name,
    ):
        """
        Java -> Python bytes interop.
        Runs once with cfg precision and once forced to F32.
        """
        import tdigest_rs as td

        p50_java, cdf_java, blob = _run_java_emit_blob(
            class_name=class_name,
            compile_run_java_fn=compile_run_java_fn,
            java_builder_chain_fn=java_builder_chain_fn,
            tmp_path=tmp_path,
            cfg=cfg,
            dataset=dataset,
            expect=expect,
            precision_override=precision_override,
        )

        d_py = td.TDigest.from_bytes(blob)
        if expected_kind is not None:
            assert d_py.inner_kind() == expected_kind

        q_py = d_py.quantile(expect["P"])
        c_py = d_py.cdf(expect["X"])

        assert_close_fn(q_py, p50_java, expect["EPS"])
        assert_close_fn(c_py, cdf_java, expect["EPS"])
        assert_close_fn(q_py, expect["Q50"], expect["EPS"])
        assert_close_fn(c_py, expect["CDF2"], expect["EPS"])

        blob2 = d_py.to_bytes()
        d_py2 = td.TDigest.from_bytes(blob2)
        if expected_kind is not None:
            assert d_py2.inner_kind() == expected_kind

        q_py2 = d_py2.quantile(expect["P"])
        c_py2 = d_py2.cdf(expect["X"])
        assert_close_fn(q_py2, q_py, expect["EPS"])
        assert_close_fn(c_py2, c_py, expect["EPS"])

    def test_java_to_python_explicit_wire_versions(
        self,
        dataset,
        expect,
        cfg,
        tmp_path: Path,
        assert_close_fn,
        compile_run_java_fn,
        java_builder_chain_fn,
    ):
        """
        Java TDigest.toBytes(version=1|2|3) must emit decodable blobs in Python.
        """
        import tdigest_rs as td

        for version in (1, 2, 3):
            p50_java, cdf_java, blob = _run_java_emit_blob(
                class_name=f"TDJavaToPythonSerdeV{version}",
                compile_run_java_fn=compile_run_java_fn,
                java_builder_chain_fn=java_builder_chain_fn,
                tmp_path=tmp_path,
                cfg=cfg,
                dataset=dataset,
                expect=expect,
                precision_override="F64",
                version_override=version,
            )

            d_py = td.TDigest.from_bytes(blob)
            q_py = d_py.quantile(expect["P"])
            c_py = d_py.cdf(expect["X"])
            assert_close_fn(q_py, p50_java, expect["EPS"])
            assert_close_fn(c_py, cdf_java, expect["EPS"])


# =====================================================================
# Category 2: JAVA → POLARS — TDigest.toBytes() → td.from_bytes(...)
#   Spec:
#     - Java builds digest and emits base64(toBytes()).
#     - Python decodes base64 to raw bytes.
#     - Polars td.from_bytes on a Binary column reconstructs digest.
#     - quantile/cdf in Polars must match Java / canonical expectations.
#     - Extra:
#         * F64 full roundtrip inside Polars.
#         * Explicit F32 path with compact schema.
#         * "Strict" mismatch (treat F32 wire as F64) must error; F32 hint works.
# =====================================================================


class TestJavaToPolarsSerde:
    def test_java_to_polars_f64_roundtrip(
        self,
        dataset,
        expect,
        cfg,
        tmp_path: Path,
        assert_close_fn,
        compile_run_java_fn,
        java_builder_chain_fn,
    ):
        """
        Java (Precision from cfg) → Polars (F64 only):

        - Only runs when cfg["precision_java"] == "F64" (skip otherwise).
        - Java builds TDigest and prints: p50,cdf(x),base64(blob).
        - Python:
            * decodes bytes to blob
            * feeds blob into Polars td.from_bytes(..., precision="f64")
            * checks quantile/cdf vs Java + canonical expectations.
        - Extra: Polars td_col → td.to_bytes() → td.from_bytes(...)
          roundtrip is stable.
        """
        import pytest
        import polars as pl
        import tdigest_rs as td

        # Only test the F64 case here to keep semantics tight.
        if cfg["precision_java"] != "F64":
            pytest.skip(
                "test_java_to_polars_f64_roundtrip only runs for Java Precision.F64"
            )
        p50_java, cdf_java, blob = _run_java_emit_blob(
            class_name="TDJavaToPolarsSerdeF64",
            compile_run_java_fn=compile_run_java_fn,
            java_builder_chain_fn=java_builder_chain_fn,
            tmp_path=tmp_path,
            cfg=cfg,
            dataset=dataset,
            expect=expect,
            precision_override=None,
        )

        # Push bytes into Polars and reconstruct via td.from_bytes, explicit F64.
        df = pl.DataFrame({"blob": [blob]})
        df_td = df.with_columns(td_col=td.from_bytes("blob", precision="f64"))
        out_pl = df_td.select(
            q=td.quantile("td_col", float(expect["P"])),
            c=td.cdf("td_col", pl.lit(float(expect["X"]), dtype=pl.Float64)),
        )

        q_pl = float(out_pl["q"][0])
        c_pl = float(out_pl["c"][0])

        # Coherence with Java and canonical expectations
        assert_close_fn(q_pl, p50_java, expect["EPS"])
        assert_close_fn(c_pl, cdf_java, expect["EPS"])
        assert_close_fn(q_pl, expect["Q50"], expect["EPS"])
        assert_close_fn(c_pl, expect["CDF2"], expect["EPS"])

        # Extra: Polars → bytes → Polars roundtrip should be stable.
        df_rt = df_td.with_columns(blob_rt=td.to_bytes("td_col"))
        blob_rt = df_rt.select("blob_rt").row(0)[0]
        assert isinstance(blob_rt, (bytes, bytearray))
        # Wire version may evolve (e.g. v2 -> v3 re-encode), so assert
        # semantic roundtrip, not exact byte identity.
        d_src = td.TDigest.from_bytes(blob)
        d_rt = td.TDigest.from_bytes(bytes(blob_rt))
        assert_close_fn(
            d_rt.quantile(float(expect["P"])),
            d_src.quantile(float(expect["P"])),
            expect["EPS"],
        )
        assert_close_fn(
            d_rt.cdf(float(expect["X"])), d_src.cdf(float(expect["X"])), expect["EPS"]
        )

        df_td_rt = pl.DataFrame({"blob": [blob_rt]}).with_columns(
            td_col=td.from_bytes("blob", precision="f64")
        )
        out_pl_rt = df_td_rt.select(
            q=td.quantile("td_col", float(expect["P"])),
            c=td.cdf("td_col", pl.lit(float(expect["X"]), dtype=pl.Float64)),
        )

        q_pl_rt = float(out_pl_rt["q"][0])
        c_pl_rt = float(out_pl_rt["c"][0])

        assert_close_fn(q_pl_rt, q_pl, expect["EPS"])
        assert_close_fn(c_pl_rt, c_pl, expect["EPS"])

    def test_java_to_polars_f32(
        self,
        dataset,
        expect,
        cfg,
        tmp_path: Path,
        assert_close_fn,
        compile_run_java_fn,
        java_builder_chain_fn,
    ):
        """
        Java with Precision.F32 → Polars:

        - Force Java to use Precision.F32.
        - Python decodes base64 to blob.
        - Polars td.from_bytes(..., precision="f32") reconstructs a compact
          f32-backed struct (min/max are Float32).
        - quantile/cdf in Polars must match Java and canonical expectations.
        """
        import polars as pl
        import tdigest_rs as td

        p50_java, cdf_java, blob = _run_java_emit_blob(
            class_name="TDJavaToPolarsSerdeF32",
            compile_run_java_fn=compile_run_java_fn,
            java_builder_chain_fn=java_builder_chain_fn,
            tmp_path=tmp_path,
            cfg=cfg,
            dataset=dataset,
            expect=expect,
            precision_override="F32",
        )

        # Push bytes into Polars and reconstruct via td.from_bytes, explicit F32.
        df = pl.DataFrame({"blob": [blob]})
        df_td = df.with_columns(td_col=td.from_bytes("blob", precision="f32"))

        # Schema: compact struct (min/max Float32).
        td_dtype = df_td.schema["td_col"]
        fields = getattr(td_dtype, "fields", None)
        assert fields is not None
        min_field = next(f for f in fields if f.name == "min")
        max_field = next(f for f in fields if f.name == "max")
        assert min_field.dtype == pl.Float32
        assert max_field.dtype == pl.Float32

        out_pl = df_td.select(
            q=td.quantile("td_col", float(expect["P"])),
            c=td.cdf("td_col", pl.lit(float(expect["X"]), dtype=pl.Float64)),
        )

        q_pl = float(out_pl["q"][0])
        c_pl = float(out_pl["c"][0])

        # Coherence with Java and canonical expectations
        assert_close_fn(q_pl, p50_java, expect["EPS"])
        assert_close_fn(c_pl, cdf_java, expect["EPS"])
        assert_close_fn(q_pl, expect["Q50"], expect["EPS"])
        assert_close_fn(c_pl, expect["CDF2"], expect["EPS"])

    def test_java_f32_polars_strict_vs_precision_f32(
        self,
        dataset,
        expect,
        cfg,
        tmp_path: Path,
        assert_close_fn,
        compile_run_java_fn,
        java_builder_chain_fn,
    ):
        """
        Java F32 blobs + Polars plugin, strict vs hint:

        - Java builds Precision.F32 digest and emits base64(toBytes()).
        - In Polars:
            * Treating the column as F64 (precision="f64"/default) is a
              strict mismatch and must error.
            * Using precision="f32" succeeds and yields a compact struct.
        - Quantile/cdf from the Polars struct must match Java and canonical
          expectations.
        """
        import pytest
        import polars as pl
        import tdigest_rs as td

        p50_java, cdf_java, blob = _run_java_emit_blob(
            class_name="TDJavaF32ForPolarsStrict",
            compile_run_java_fn=compile_run_java_fn,
            java_builder_chain_fn=java_builder_chain_fn,
            tmp_path=tmp_path,
            cfg=cfg,
            dataset=dataset,
            expect=expect,
            precision_override="F32",
        )

        # Build a Polars DataFrame with the Java F32 blob.
        df = pl.DataFrame({"blob": [blob]})

        # "Strict" mismatch: treating F32 wire as F64 (precision="f64"/auto)
        # must error with a plugin ComputeError.
        with pytest.raises(
            pl.exceptions.ComputeError, match="mixed f32/f64 blobs in column"
        ):
            df.with_columns(td_col=td.from_bytes("blob", precision="f64")).collect()

        # Correct precision hint: precision="f32" → must succeed and be compact.
        df_td = df.with_columns(td_col=td.from_bytes("blob", precision="f32"))
        td_dtype = df_td.schema["td_col"]

        fields = getattr(td_dtype, "fields", None)
        assert fields is not None
        min_field = next(f for f in fields if f.name == "min")
        max_field = next(f for f in fields if f.name == "max")
        assert min_field.dtype == pl.Float32
        assert max_field.dtype == pl.Float32

        out_pl = df_td.select(
            q=td.quantile("td_col", float(expect["P"])),
            c=td.cdf("td_col", pl.lit(float(expect["X"]), dtype=pl.Float64)),
        )

        q_pl = float(out_pl["q"][0])
        c_pl = float(out_pl["c"][0])

        # Coherence with Java and canonical expectations
        assert_close_fn(q_pl, p50_java, expect["EPS"])
        assert_close_fn(c_pl, cdf_java, expect["EPS"])
        assert_close_fn(q_pl, expect["Q50"], expect["EPS"])
        assert_close_fn(c_pl, expect["CDF2"], expect["EPS"])
