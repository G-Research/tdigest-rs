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
# Category 1: INVALID TRAINING — all 4 surfaces
#   Spec:
#     - Training data must contain ONLY finite numbers.
#     - Any NaN, ±inf, or nulls in training → error.
# =====================================================================


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

    def test_python_rejects_nan_and_inf(self, cfg):
        import tdigest_rs as td

        cases = [
            [0.0, float("nan"), 1.0],
            [0.0, float("+inf"), 1.0],
            [0.0, float("-inf"), 1.0],
        ]
        for bad in cases:
            with pytest.raises(Exception) as e:
                td.TDigest.from_array(
                    bad,
                    max_size=cfg["max_size"],
                    scale=cfg["scale_py"],
                    singleton_policy=cfg["singleton_py"],
                    precision=cfg["precision_py"],
                )
            s = str(e.value).lower()
            assert "nan" in s or "inf" in s

    def test_polars_rejects_nan_inf_null(self, cfg):
        import polars as pl
        import tdigest_rs as td
        from polars.exceptions import ComputeError

        bad_frames = [
            pl.DataFrame({"x": [0.0, float("nan"), 1.0]}),
            pl.DataFrame({"x": [0.0, float("+inf"), 1.0]}),
            pl.DataFrame({"x": [0.0, float("-inf"), 1.0]}),
            pl.DataFrame({"x": [None, 1.0]}),
        ]
        for df in bad_frames:
            df = df.with_columns(pl.col("x").cast(pl.Float64))
            with pytest.raises(ComputeError):
                df.select(
                    td.tdigest(
                        "x",
                        max_size=cfg["max_size"],
                        scale=cfg["scale_pl"],
                        singleton_policy=cfg["singleton_pl"],
                        precision=cfg["precision_pl"],
                    )
                )

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
                }} catch (Throwable t) {{
                  System.out.println("CAUGHT");
                  return;
                }}
                System.out.println("NO-ERROR");
              }}
            }}
        """).strip()

        out = compile_run_java_fn(tmp_path, class_name, java_src)
        assert "CAUGHT" in out and "NO-ERROR" not in out and "BUILT" not in out


# =====================================================================
# Category 2: EMPTY DIGEST BEHAVIOR — all 4 surfaces
#   Spec:
#     - CDF(any finite x)    → NaN
#     - CDF(±∞)              → NaN
#     - quantile(any finite) → NaN
#     - quantile(NaN/±inf)   → ValueError (strict)
#   Note: CLI CDF has no explicit probe arg; we cover empty CDF on other surfaces.
# =====================================================================


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

    def test_java(
        self, cfg, tmp_path: Path, compile_run_java_fn, java_builder_chain_fn
    ):
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
        out = subprocess.check_output(
            [str(paths.cli_bin), *args], input="", text=True
        ).strip()
        p_str, val_str = out.split(",", 1)
        assert p_str == "0.5"
        assert val_str.lower() == "nan"

    def test_polars_groupby_regular_and_empty(self, cfg):
        """
        Two groups:
        A: trained digest → finite q, finite c
        B: no rows → merged (empty) digest → q=None, c=NaN (including ±∞ probes → NaN)
        """
        import polars as pl
        import tdigest_rs as td

        data = pl.DataFrame({"g": ["A", "A", "A"], "x": [0.0, 1.0, 2.0]}).with_columns(
            pl.col("x").cast(pl.Float64)
        )
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

        out = (
            groups.join(td_by_g, on="g", how="left")  # A→td; B→td=null
            .with_columns(
                td2=td.merge_tdigests(pl.col("td")).over("g")
            )  # B→empty digest (n=0)
            .with_columns(
                probe_f=pl.repeat(pl.lit(2.0, dtype=pl.Float64), pl.len()),
                probe_lo=pl.repeat(pl.lit(float("-inf"), dtype=pl.Float64), pl.len()),
                probe_hi=pl.repeat(pl.lit(float("+inf"), dtype=pl.Float64), pl.len()),
            )
            # ⬇️ per-group to prevent leakage
            .with_columns(
                q=td.quantile(pl.col("td2"), 0.5).over("g"),  # empty → None
                c_f=td.cdf(pl.col("td2"), pl.col("probe_f")).over("g"),  # empty → NaN
                c_lo=td.cdf(pl.col("td2"), pl.col("probe_lo")).over("g"),  # empty → NaN
                c_hi=td.cdf(pl.col("td2"), pl.col("probe_hi")).over("g"),  # empty → NaN
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


# =====================================================================
# Category 3: SMOKE / COHERENCE — quantile & cdf agree on CLI/Python/Polars
#   Spec:
#     - With canonical tiny data:
#         P=0.5 → Q50=1.5 ; CDF(2.0)=0.625
# =====================================================================


class TestSmokeCoherence:
    def test_cli(
        self,
        paths,
        dataset,
        expect,
        cfg,
        run_cli_fn,
        assert_close_fn,
        cli_build_args_fn,
    ):
        assert paths.cli_bin.exists() and paths.cli_bin.stat().st_mode & 0o111, (
            f"missing CLI: {paths.cli_bin}"
        )
        data = dataset["DATA"]

        q_args = [
            "quantile",
            "--stdin",
            "--p",
            str(expect["P"]),
            *cli_build_args_fn(cfg),
        ]
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
        assert p_at_x is not None, (
            f"x={expect['X']} not found in CLI CDF output:\n{out}"
        )
        assert_close_fn(p_at_x, expect["CDF2"], expect["EPS"])

    def test_python(self, dataset, expect, cfg, add_pin_kw_fn, assert_close_fn):
        import tdigest_rs as td

        kwargs = dict(
            max_size=cfg["max_size"],
            scale=cfg["scale_py"],
            singleton_policy=cfg["singleton_py"],
            precision=cfg["precision_py"],
        )
        if cfg.get("singleton_py") == "edges" and cfg.get("pin_per_side") is not None:
            kwargs = add_pin_kw_fn(kwargs)

        d = td.TDigest.from_array(dataset["DATA"], **kwargs)
        assert_close_fn(d.quantile(expect["P"]), expect["Q50"], expect["EPS"])
        assert_close_fn(d.cdf(expect["X"]), expect["CDF2"], expect["EPS"])

    def test_polars(self, dataset, expect, cfg, add_pin_kw_fn, assert_close_fn):
        import polars as pl
        import tdigest_rs as td

        df = pl.DataFrame({"x": dataset["DATA"]}).with_columns(
            pl.col("x").cast(pl.Float64)
        )

        td_kwargs = dict(
            max_size=cfg["max_size"],
            scale=cfg["scale_pl"],
            singleton_policy=cfg["singleton_pl"],
            precision=cfg["precision_pl"],
        )
        if cfg.get("singleton_pl") == "edges" and cfg.get("pin_per_side") is not None:
            td_kwargs = add_pin_kw_fn(td_kwargs)

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


# =====================================================================
# Category 4: PRECISION='auto' — CLI/Python/Polars
#   Spec:
#     - 'auto' selects precision based on surface rules:
#         * Python: TDigest API accepts precision='auto'
#         * Polars: tracks training column dtype
#         * CLI: optional; test is skipped if CLI lacks 'auto'
# =====================================================================


class TestPrecisionAuto:
    def test_python(self, dataset, expect):
        import tdigest_rs as td

        d = td.TDigest.from_array(
            dataset["DATA"],
            max_size=50,
            scale="k2",
            singleton_policy="use",
            precision="auto",
        )
        assert math.isfinite(d.quantile(expect["P"]))
        assert math.isfinite(d.cdf(expect["X"]))

    def test_polars(self):
        import polars as pl
        import tdigest_rs as td

        DATA = [0.0, 1.0, 2.0, 3.0]
        base = pl.DataFrame({"x": DATA})

        df32 = base.with_columns(x=pl.col("x").cast(pl.Float32))
        df64 = base.with_columns(x=pl.col("x").cast(pl.Float64))

        out32 = df32.with_columns(td_col=td.tdigest("x", precision="auto")).select(
            q=td.quantile("td_col", 0.5), c=td.cdf("td_col", "x")
        )
        out64 = df64.with_columns(td_col=td.tdigest("x", precision="auto")).select(
            q=td.quantile("td_col", 0.5), c=td.cdf("td_col", "x")
        )

        assert str(out32["q"].dtype) == "Float32"
        assert str(out64["q"].dtype) == "Float64"

    def test_cli(self, paths, dataset, cfg, cli_supports_auto_fn):
        if not cli_supports_auto_fn():
            pytest.skip("CLI lacks precision=auto")
        data = " ".join(str(float(x)) for x in dataset["DATA"])
        args = [
            "quantile",
            "--stdin",
            "--p",
            "0.5",
            "--no-header",
            "--output",
            "csv",
            "--max-size",
            str(cfg["max_size"]),
            "--scale",
            "k2",
            "--singleton-policy",
            "use",
            "--precision",
            "auto",
        ]
        out = subprocess.check_output(
            [str(paths.cli_bin), *args], input=data, text=True
        ).strip()
        p_str, v_str = out.split(",", 1)
        assert p_str == "0.5" and v_str.lower() != "nan"


# =====================================================================
# Category 5: ARGUMENT MATRIX — CLI/Python/Polars
#   Spec:
#     - Exercise a representative matrix:
#         * scale ∈ {k1, k2}
#         * singleton_policy ∈ {off, use, edges}
#         * precision ∈ {f32, f64, auto}
#       Expect finite quantile(0.5) and finite cdf(2.0) on canonical data.
#     - When policy='edges', pass pin_per_side=2 (or cfg['pin_per_side']).
#     - If a surface lacks 'auto', skip those cases gracefully.
# =====================================================================


class TestArgumentMatrix:
    DATA = [0.0, 1.0, 2.0, 3.0]
    P, X = 0.5, 2.0

    SIMPLE_SCALES = ("k1", "k2")
    SIMPLE_POLICIES = ("off", "use", "edges")
    PRECS = ("f32", "f64", "auto")

    # ---------- CLI ----------
    def test_cli_matrix(self, paths, cfg, cli_supports_auto_fn):
        cli_has_auto = cli_supports_auto_fn()

        for scale in self.SIMPLE_SCALES:
            for pol in self.SIMPLE_POLICIES:
                for prec in self.PRECS:
                    if prec == "auto" and not cli_has_auto:
                        continue

                    args = [
                        "quantile",
                        "--stdin",
                        "--p",
                        str(self.P),
                        "--no-header",
                        "--output",
                        "csv",
                        "--max-size",
                        str(cfg["max_size"]),
                        "--scale",
                        scale,
                        "--singleton-policy",
                        pol,
                        "--precision",
                        prec,
                    ]
                    if pol == "edges":
                        args += ["--pin-per-side", str(int(cfg.get("pin_per_side", 2)))]
                    out = subprocess.check_output(
                        [str(paths.cli_bin), *args],
                        input=" ".join(str(v) for v in self.DATA),
                        text=True,
                    ).strip()
                    _, v = out.split(",", 1)
                    assert v.lower() != "nan"

                    # CDF
                    args_c = [
                        "cdf",
                        "--stdin",
                        "--no-header",
                        "--output",
                        "csv",
                        "--max-size",
                        str(cfg["max_size"]),
                        "--scale",
                        scale,
                        "--singleton-policy",
                        pol,
                        "--precision",
                        prec,
                    ]
                    if pol == "edges":
                        args_c += [
                            "--pin-per-side",
                            str(int(cfg.get("pin_per_side", 2))),
                        ]
                    out_c = subprocess.check_output(
                        [str(paths.cli_bin), *args_c],
                        input=" ".join(str(v) for v in self.DATA),
                        text=True,
                    ).strip()
                    # find x==2.0
                    found = None
                    for line in out_c.splitlines():
                        xs, ps = line.split(",", 1)
                        if abs(float(xs) - self.X) <= 1e-12:
                            found = float(ps)
                            break
                    assert found is not None and math.isfinite(found)

    # ---------- Python ----------
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

    # ---------- Polars ----------
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

                    out = df.with_columns(td_col=td.tdigest("x", **td_kwargs)).select(
                        q=td.quantile("td_col", float(self.P)),
                        c=td.cdf("td_col", pl.lit(float(self.X))),
                    )
                    assert math.isfinite(float(out["q"][0]))
                    assert math.isfinite(float(out["c"][0]))
