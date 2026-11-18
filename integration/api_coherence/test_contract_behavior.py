from __future__ import annotations

import math
import subprocess
from pathlib import Path

import pytest


class TestInvalidTrainingData:
    def test_cli_rejects_nan(self, paths, cfg, cli_build_args_fn):
        args = ["quantile", "--stdin", "--p", "0.5", *cli_build_args_fn(cfg)]
        with pytest.raises(subprocess.CalledProcessError) as e:
            subprocess.check_output(
                [str(paths.cli_bin), *args],
                input="0 NaN 1",
                text=True,
                stderr=subprocess.STDOUT,
            )
        assert "nan" in e.value.output.lower()

    def test_cli_rejects_infinities(self, paths, cfg, cli_build_args_fn):
        args = ["quantile", "--stdin", "--p", "0.5", *cli_build_args_fn(cfg)]
        for tok in ("inf", "-inf", "+inf"):
            with pytest.raises(subprocess.CalledProcessError) as e:
                subprocess.check_output(
                    [str(paths.cli_bin), *args],
                    input=f"0 {tok} 1",
                    text=True,
                    stderr=subprocess.STDOUT,
                )
            assert "inf" in e.value.output.lower()

    @pytest.mark.parametrize(
        "bad",
        ["Double.NaN", "Double.POSITIVE_INFINITY", "Double.NEGATIVE_INFINITY"],
        ids=["nan", "pos_inf", "neg_inf"],
    )
    def test_java_rejects_nan_and_inf(
        self,
        bad: str,
        cfg,
        tmp_path: Path,
        compile_run_java_fn,
        java_builder_chain_fn,
    ):
        import textwrap

        builder = java_builder_chain_fn(cfg)
        class_name = "TDRejectBadTrain"
        java_src = textwrap.dedent(f"""
            import com.gresearch.tdigest.TDigest;
            import com.gresearch.tdigest.TDigest.Scale;
            import com.gresearch.tdigest.TDigest.SingletonPolicy;
            import com.gresearch.tdigest.TDigest.Precision;

            public class {class_name} {{
              public static void main(String[] args) {{
                double[] data = new double[] {{0.0, {bad}, 1.0}};
                try {{
                  try (TDigest d = TDigest.builder()
                        {builder}
                        .build(data)) {{
                    System.out.println("BUILT"); // should not happen
                  }}
                }} catch (IllegalArgumentException t) {{
                  System.out.println("CAUGHT");
                  return;
                }}
                System.out.println("NO-ERROR");
              }}
            }}
        """).strip()

        out = compile_run_java_fn(tmp_path, class_name, java_src)
        assert "CAUGHT" in out and "NO-ERROR" not in out and "BUILT" not in out


class TestEmptyDigestBehavior:
    def test_python(self, cfg):
        import tdigest_rs as td
        d = td.TDigest.from_array(
            [],
            max_size=cfg["max_size"],
            scale=cfg["scale_py"],
            singleton_policy=cfg["singleton_py"],
            precision=cfg["precision_py"],
        )
        assert math.isnan(d.cdf(2.0))
        assert math.isnan(d.cdf(float("-inf")))
        assert math.isnan(d.cdf(float("+inf")))
        assert math.isnan(d.quantile(0.25))
        for bad in (float("nan"), float("-inf"), float("+inf")):
            with pytest.raises(ValueError):
                _ = d.quantile(bad)

    def test_java(self, cfg, tmp_path: Path, compile_run_java_fn, java_builder_chain_fn):
        import textwrap

        builder = java_builder_chain_fn(cfg)
        class_name = "TDEmpty"
        java_src = textwrap.dedent(f"""
            import com.gresearch.tdigest.TDigest;
            import com.gresearch.tdigest.TDigest.Scale;
            import com.gresearch.tdigest.TDigest.SingletonPolicy;
            import com.gresearch.tdigest.TDigest.Precision;

            public class {class_name} {{
              public static void main(String[] args) {{
                try (TDigest d = TDigest.builder()
                        {builder}
                        .build(new double[] {{}})) {{
                  double[] cs = d.cdf(new double[] {{ 2.0, Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY }});
                  boolean all_nan = Double.isNaN(cs[0]) && Double.isNaN(cs[1]) && Double.isNaN(cs[2]);
                  boolean q_is_nan = Double.isNaN(d.quantile(0.75));
                  System.out.println(all_nan + "," + q_is_nan);
                }}
              }}
            }}
        """).strip()
        out = compile_run_java_fn(tmp_path, class_name, java_src)
        assert out == "true,true", f"unexpected output: {out}"

    def test_cli_quantile_empty(self, paths, cfg, cli_build_args_fn):
        # CLI: no stdin → empty digest; quantile should print "p,NaN" (finite p in-range)
        args = ["quantile", "--stdin", "--p", "0.5", *cli_build_args_fn(cfg)]
        out = subprocess.check_output([str(paths.cli_bin), *args], input="", text=True).strip()
        p_str, val_str = out.split(",", 1)
        assert p_str == "0.5"
        assert val_str.lower() == "nan"

    def test_polars_groupby_regular_and_empty(self, cfg):
        import polars as pl
        import tdigest_rs as td

        data   = pl.DataFrame({"g": ["A","A","A"], "x": [0.0, 1.0, 2.0]}).with_columns(pl.col("x").cast(pl.Float64))
        groups = pl.DataFrame({"g": ["A", "B"]})

        td_by_g = data.group_by("g").agg(
            td.tdigest(
                "x",
                max_size=cfg["max_size"],
                scale=cfg["scale_pl"],
                singleton_policy=cfg["singleton_pl"],
                precision=cfg["precision_pl"],
            ).alias("td")
        )

        joined = groups.join(td_by_g, on="g", how="left")
        # Missing is rejected under the explicitly selected modern contract.
        with pytest.raises(pl.exceptions.ComputeError, match="missing digest"):
            joined.with_columns(td2=td.merge_tdigests("td").over("g"))
        # The application explicitly supplies a genuine empty digest for B.
        empty = data.head(0).select(td.tdigest(
            "x", max_size=cfg["max_size"], scale=cfg["scale_pl"],
            singleton_policy=cfg["singleton_pl"], precision=cfg["precision_pl"],
        ).alias("td")).item()
        joined = joined.with_columns(pl.when(pl.col("td").is_null())
            .then(pl.lit(empty, dtype=td_by_g.schema["td"]))
            .otherwise(pl.col("td")).alias("td"))
        out = (
            joined
            .with_columns(td2 = td.merge_tdigests(pl.col("td")).over("g"))       # B→empty digest (n=0)
            .with_columns(
                probe_f = pl.repeat(pl.lit(2.0, dtype=pl.Float64), pl.len()),
                probe_lo = pl.repeat(pl.lit(float("-inf"), dtype=pl.Float64), pl.len()),
                probe_hi = pl.repeat(pl.lit(float("+inf"), dtype=pl.Float64), pl.len()),
            )
            # ⬇️ per-group to prevent leakage
            .with_columns(
                q = td.quantile(pl.col("td2"), 0.5).over("g"),                   # empty → None
                c_f = td.cdf(pl.col("td2"), pl.col("probe_f")).over("g"),        # empty → NaN
                c_lo = td.cdf(pl.col("td2"), pl.col("probe_lo")).over("g"),      # empty → NaN
                c_hi = td.cdf(pl.col("td2"), pl.col("probe_hi")).over("g"),      # empty → NaN
            )
            .select("g", "q", "c_f", "c_lo", "c_hi")
            .sort("g")
        )

        row_a, row_b = out.to_dicts()

        # A: finite
        assert row_a["g"] == "A"
        assert row_a["q"] is not None and not math.isnan(float(row_a["q"]))
        for k in ("c_f", "c_lo", "c_hi"):
            assert not math.isnan(float(row_a[k]))

        # B: empty digest → q=None, c=NaN (finite & ±∞)
        assert row_b["g"] == "B"
        assert row_b["q"] is None
        for k in ("c_f", "c_lo", "c_hi"):
            assert math.isnan(float(row_b[k]))


class TestSmokeCoherence:
    def test_cli(self, paths, dataset, expect, cfg, run_cli_fn, assert_close_fn, cli_build_args_fn):
        assert paths.cli_bin.exists() and paths.cli_bin.stat().st_mode & 0o111, (
            f"missing CLI: {paths.cli_bin}"
        )
        data = dataset["DATA"]

        q_args = ["quantile", "--stdin", "--p", str(expect["P"]), *cli_build_args_fn(cfg)]
        out = run_cli_fn(paths.cli_bin, q_args, data)
        p_str, v_str = out.split(",", 1)
        assert_close_fn(float(p_str), expect["P"], expect["EPS"])
        assert_close_fn(float(v_str), expect["Q50"], expect["EPS"])

        c_args = ["cdf", "--stdin", *cli_build_args_fn(cfg)]
        out = run_cli_fn(paths.cli_bin, c_args, data)
        p_at_x = None
        for line in out.splitlines():
            xs, ps = line.split(",", 1)
            if abs(float(xs) - expect["X"]) <= 1e-12:
                p_at_x = float(ps)
                break
        assert p_at_x is not None, f"x={expect['X']} not found in CLI CDF output:\n{out}"
        assert_close_fn(p_at_x, expect["CDF2"], expect["EPS"])

    def test_polars(self, dataset, expect, cfg, assert_close_fn):
        import polars as pl
        import tdigest_rs as td

        df = pl.DataFrame({"x": dataset["DATA"]}).with_columns(pl.col("x").cast(pl.Float64))

        td_kwargs = dict(
            max_size=cfg["max_size"],
            scale=cfg["scale_pl"],
            singleton_policy=cfg["singleton_pl"],
            precision=cfg["precision_pl"],
        )

        df2 = df.with_columns(td_col=td.tdigest("x", **td_kwargs))
        out_df = df2.select(
            p50=td.quantile("td_col", expect["P"]),
            cdf2=td.cdf("td_col", "x"),
        )
        p50_val = float(out_df["p50"][0])
        idx_of_x = dataset["DATA"].index(expect["X"])
        cdf2_val = float(out_df["cdf2"][idx_of_x])
        assert_close_fn(p50_val, expect["Q50"], expect["EPS"])
        assert_close_fn(cdf2_val, expect["CDF2"], expect["EPS"])


class TestPrecisionAuto:
    def test_polars(self):
        import polars as pl
        import tdigest_rs as td

        DATA = [0.0, 1.0, 2.0, 3.0]
        base = pl.DataFrame({"x": DATA})

        df32 = base.with_columns(x=pl.col("x").cast(pl.Float32))
        df64 = base.with_columns(x=pl.col("x").cast(pl.Float64))

        out32 = (
            df32.with_columns(td_col=td.tdigest("x", precision="auto"))
                .select(q=td.quantile("td_col", 0.5), c=td.cdf("td_col", "x"))
        )
        out64 = (
            df64.with_columns(td_col=td.tdigest("x", precision="auto"))
                .select(q=td.quantile("td_col", 0.5), c=td.cdf("td_col", "x"))
        )

        assert str(out32["q"].dtype) == "Float32"
        assert str(out64["q"].dtype) == "Float64"


class TestArgumentMatrix:
    DATA = [0.0, 1.0, 2.0, 3.0]
    P, X = 0.5, 2.0

    SIMPLE_SCALES = ("k1", "k2")
    SIMPLE_POLICIES = ("off", "use", "edges")
    PRECS = ("f32", "f64", "auto")

    def test_cli_matrix(self, paths, cfg):

        for scale in self.SIMPLE_SCALES:
            for pol in self.SIMPLE_POLICIES:
                for prec in self.PRECS:

                    args = [
                        "quantile", "--stdin", "--p", str(self.P),
                        "--no-header", "--output", "csv",
                        "--max-size", str(cfg["max_size"]),
                        "--scale", scale,
                        "--singleton-policy", pol,
                        "--precision", prec,
                    ]
                    if pol == "edges":
                        args += ["--pin-per-side", str(int(cfg.get("pin_per_side", 2)))]
                    out = subprocess.check_output(
                        [str(paths.cli_bin), *args],
                        input=" ".join(str(v) for v in self.DATA),
                        text=True,
                    ).strip()
                    probability, value = out.split(",", 1)
                    assert probability == str(self.P) and math.isfinite(float(value))

                    # CDF
                    args_c = [
                        "cdf", "--stdin",
                        "--no-header", "--output", "csv",
                        "--max-size", str(cfg["max_size"]),
                        "--scale", scale,
                        "--singleton-policy", pol,
                        "--precision", prec,
                    ]
                    if pol == "edges":
                        args_c += ["--pin-per-side", str(int(cfg.get("pin_per_side", 2)))]
                    out_c = subprocess.check_output(
                        [str(paths.cli_bin), *args_c],
                        input=" ".join(str(v) for v in self.DATA),
                        text=True,
                    ).strip()
                    found = None
                    for line in out_c.splitlines():
                        xs, ps = line.split(",", 1)
                        if abs(float(xs) - self.X) <= 1e-12:
                            found = float(ps)
                            break
                    assert found is not None and math.isfinite(found)

    def test_python_matrix(self, cfg):
        import tdigest_rs as td
        for scale in self.SIMPLE_SCALES:
            for pol in self.SIMPLE_POLICIES:
                for prec in self.PRECS:
                    kwargs = dict(
                        max_size=cfg["max_size"],
                        scale=scale,
                        singleton_policy=pol,
                        precision=prec,
                    )
                    if pol == "edges":
                        kwargs["pin_per_side"] = int(cfg.get("pin_per_side", 2))
                    d = td.TDigest.from_array(self.DATA, **kwargs)
                    assert math.isfinite(d.quantile(self.P))
                    assert math.isfinite(d.cdf(self.X))

    def test_polars_matrix(self, cfg):
        import polars as pl
        import tdigest_rs as td

        base = pl.DataFrame({"x": self.DATA}).with_columns(pl.col("x").cast(pl.Float64))

        for scale in self.SIMPLE_SCALES:
            for pol in self.SIMPLE_POLICIES:
                for prec in self.PRECS:
                    # Align input dtype with precision policy so this matrix stays “valid-only”
                    if prec == "f32":
                        df = base.with_columns(pl.col("x").cast(pl.Float32))
                    else:
                        df = base  # f64 and auto

                    td_kwargs = dict(
                        max_size=cfg["max_size"],
                        scale=scale,
                        singleton_policy=pol,
                        precision=prec,
                    )
                    if pol == "edges":
                        td_kwargs["pin_per_side"] = int(cfg.get("pin_per_side", 2))

                    out = (
                        df.with_columns(td_col=td.tdigest("x", **td_kwargs))
                          .select(
                              q=td.quantile("td_col", float(self.P)),
                              c=td.cdf("td_col", pl.lit(float(self.X))),
                          )
                    )
                    assert math.isfinite(float(out["q"][0]))
                    assert math.isfinite(float(out["c"][0]))
