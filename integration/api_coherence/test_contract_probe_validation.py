from __future__ import annotations

import math
import subprocess
from pathlib import Path

import pytest


# =====================================================================
# Helpers (use fixtures defined in conftest)
# =====================================================================

# NOTE: We intentionally do NOT define a local _cli_build_args here to
# avoid drift. Always use the fixture `cli_build_args_fn(cfg)`.


# =====================================================================
# Category 1: PROBE VALIDATION — NaN and out-of-range p
#   Spec:
#     - cdf(NaN)      → NaN
#     - quantile(NaN) → ValueError (strict)
#     - quantile(p) with finite p∉[0,1] → ValueError (strict)
# =====================================================================


class TestProbeValidation:
    def test_python(self, cfg):
        import tdigest_rs as td

        d = td.TDigest.from_array(
            [0.0, 1.0, 2.0, 3.0],
            max_size=cfg["max_size"],
            scale=cfg["scale_py"],
            singleton_policy=cfg["singleton_py"],
            precision=cfg["precision_py"],
        )
        assert math.isnan(d.cdf(float("nan")))
        with pytest.raises(ValueError):
            _ = d.quantile(float("nan"))
        for bad in (-0.1, 1.1):
            with pytest.raises(ValueError):
                _ = d.quantile(bad)

    def test_java(
        self, cfg, tmp_path: Path, compile_run_java_fn, java_builder_chain_fn
    ):
        import textwrap

        builder = java_builder_chain_fn(cfg)
        class_name = "TDProbeValidation"
        java_src = textwrap.dedent(f"""
            import com.gresearch.tdigest.TDigest;
            import com.gresearch.tdigest.TDigest.Scale;
            import com.gresearch.tdigest.TDigest.SingletonPolicy;
            import com.gresearch.tdigest.TDigest.Precision;

            public class {class_name} {{
              public static void main(String[] args) {{
                try (TDigest d = TDigest.builder()
                        {builder}
                        .build(new double[] {{0.0, 1.0, 2.0, 3.0}})) {{
                  boolean c_is_nan = Double.isNaN(d.cdf(new double[] {{ Double.NaN }})[0]);
                  boolean threw_nan = false, threw_lo = false, threw_hi = false;
                  try {{ d.quantile(Double.NaN); }} catch (Throwable t) {{ threw_nan = true; }}
                  try {{ d.quantile(-0.1); }} catch (Throwable t) {{ threw_lo = true; }}
                  try {{ d.quantile(1.1); }} catch (Throwable t) {{ threw_hi = true; }}
                  System.out.println(c_is_nan + "," + (threw_nan && threw_lo && threw_hi));
                }}
              }}
            }}
        """).strip()
        out = compile_run_java_fn(tmp_path, class_name, java_src)
        assert out == "true,true", f"unexpected output: {out}"

    def test_cli_quantile_nan_and_out_of_range(self, paths, cfg, cli_build_args_fn):
        # NaN should raise (strict)
        args = ["quantile", "--stdin", "--p", "NaN", *cli_build_args_fn(cfg)]
        with pytest.raises(subprocess.CalledProcessError) as e:
            subprocess.check_output(
                [str(paths.cli_bin), *args],
                input="0 1 2 3",
                text=True,
                stderr=subprocess.STDOUT,
            )
        assert "p must be" in e.value.output.lower()

        # p out of range should raise
        for bad in ("-0.1", "1.1"):
            args = ["quantile", "--stdin", "--p", bad, *cli_build_args_fn(cfg)]
            with pytest.raises(subprocess.CalledProcessError) as e:
                subprocess.check_output(
                    [str(paths.cli_bin), *args],
                    input="0 1 2 3",
                    text=True,
                    stderr=subprocess.STDOUT,
                )
            assert "p must be in [0,1]" in e.value.output.lower()

    def test_polars_cdf_nan_and_null_probe(self, cfg):
        import polars as pl
        import tdigest_rs as td
        from polars.exceptions import ComputeError

        df = pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]}).with_columns(
            pl.col("x").cast(pl.Float64)
        )
        df_nan = df.with_columns(
            probe=pl.Series([float("nan"), 0.0, 1.0, 2.0], dtype=pl.Float64)
        )
        out = df_nan.with_columns(
            td_col=td.tdigest(
                "x",
                max_size=cfg["max_size"],
                scale=cfg["scale_pl"],
                singleton_policy=cfg["singleton_pl"],
                precision=cfg["precision_pl"],
            )
        ).select(c=td.cdf("td_col", pl.col("probe")))
        s = out["c"]
        assert math.isnan(float(s[0]))

        # Null probe should ERROR (policy: invalid probe)
        df_null = df.with_columns(
            probe=pl.Series([None, 2.0, None, 1.0], dtype=pl.Float64)
        )
        with pytest.raises(ComputeError):
            (
                df_null.with_columns(
                    td_col=td.tdigest(
                        "x",
                        max_size=cfg["max_size"],
                        scale=cfg["scale_pl"],
                        singleton_policy=cfg["singleton_pl"],
                        precision=cfg["precision_pl"],
                    )
                ).select(c=td.cdf("td_col", pl.col("probe")))
            )


# =====================================================================
# Category 2: ±∞ PROBE behavior (tails) — Python/Java/Polars
#   Spec (non-empty digest):
#     - cdf(-∞) → 0.0 ; cdf(+∞) → 1.0
#     - quantile(±∞) → ValueError (strict)
#   Note: CLI covered via ProbeValidation (quantile strict).
# =====================================================================


class TestInfinityProbeBehavior:
    def test_python(self, cfg):
        import tdigest_rs as td

        d = td.TDigest.from_array(
            [0.0, 1.0, 2.0, 3.0],
            max_size=cfg["max_size"],
            scale=cfg["scale_py"],
            singleton_policy=cfg["singleton_py"],
            precision=cfg["precision_py"],
        )
        assert d.cdf(float("-inf")) == 0.0
        assert d.cdf(float("+inf")) == 1.0
        for bad in (float("-inf"), float("+inf")):
            with pytest.raises(ValueError):
                _ = d.quantile(bad)

    def test_java(
        self, cfg, tmp_path: Path, compile_run_java_fn, java_builder_chain_fn
    ):
        import textwrap

        builder = java_builder_chain_fn(cfg)
        class_name = "TDProbeInf"
        java_src = textwrap.dedent(f"""
            import com.gresearch.tdigest.TDigest;
            import com.gresearch.tdigest.TDigest.Scale;
            import com.gresearch.tdigest.TDigest.SingletonPolicy;
            import com.gresearch.tdigest.TDigest.Precision;

            public class {class_name} {{
              public static void main(String[] args) {{
                try (TDigest d = TDigest.builder()
                        {builder}
                        .build(new double[] {{0.0, 1.0, 2.0, 3.0}})) {{
                  double[] cs = d.cdf(new double[] {{
                      Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY
                  }});
                  boolean c_ok = (cs[0] == 0.0) && (cs[1] == 1.0);

                  boolean threw_neg = false, threw_pos = false;
                  try {{ d.quantile(Double.NEGATIVE_INFINITY); }} catch (Throwable t) {{ threw_neg = true; }}
                  try {{ d.quantile(Double.POSITIVE_INFINITY); }} catch (Throwable t) {{ threw_pos = true; }}

                  System.out.println(c_ok + "," + (threw_neg && threw_pos));
                }}
              }}
            }}
        """).strip()
        out = compile_run_java_fn(tmp_path, class_name, java_src)
        assert out == "true,true", f"unexpected output: {out}"

    def test_polars(self, cfg):
        import polars as pl
        import tdigest_rs as td

        df = pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]}).with_columns(
            pl.col("x").cast(pl.Float64)
        )
        df = df.with_columns(
            probe=pl.Series(
                [float("-inf"), float("+inf"), float("-inf"), float("+inf")],
                dtype=pl.Float64,
            )
        )

        out = df.with_columns(
            td_col=td.tdigest(
                "x",
                max_size=cfg["max_size"],
                scale=cfg["scale_pl"],
                singleton_policy=cfg["singleton_pl"],
                precision=cfg["precision_pl"],
            )
        ).select(c=td.cdf("td_col", pl.col("probe")))
        s = out["c"]
        assert float(s[0]) == 0.0
        assert float(s[1]) == 1.0
        assert float(s[2]) == 0.0
        assert float(s[3]) == 1.0
