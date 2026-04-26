from __future__ import annotations


# =====================================================================
# Category 1: PYTHON ROUND-TRIP — TDigest.to_bytes() / from_bytes()
# =====================================================================


class TestPythonRoundTripSerde:
    def test_python_round_trip(
        self, dataset, expect, cfg, add_pin_kw_fn, assert_close_fn
    ):
        import tdigest_rs as td

        kwargs = dict(
            max_size=cfg["max_size"],
            scale=cfg["scale_py"],
            singleton_policy=cfg["singleton_py"],
            precision=cfg["precision_py"],
        )
        if cfg.get("singleton_py") == "edges" and cfg.get("pin_per_side") is not None:
            kwargs = add_pin_kw_fn(kwargs)

        d1 = td.TDigest.from_array(dataset["DATA"], **kwargs)

        # Serialize → bytes
        blob = d1.to_bytes()
        assert isinstance(blob, (bytes, bytearray))

        # Deserialize → new digest
        d2 = td.TDigest.from_bytes(bytes(blob))

        # Coherence on key probes
        q1 = d1.quantile(expect["P"])
        q2 = d2.quantile(expect["P"])
        c1 = d1.cdf(expect["X"])
        c2 = d2.cdf(expect["X"])

        assert_close_fn(q2, q1, expect["EPS"])
        assert_close_fn(c2, c1, expect["EPS"])
        # Also match canonical expectations
        assert_close_fn(q2, expect["Q50"], expect["EPS"])
        assert_close_fn(c2, expect["CDF2"], expect["EPS"])


# =====================================================================
# Category 2: PYTHON ↔ POLARS wire interop
# =====================================================================


class TestPythonPolarsSerdeInterop:
    # ------------------------------------------------------------------
    # Small helpers (not collected by pytest)
    # ------------------------------------------------------------------
    @staticmethod
    def _build_python_digest(dataset, cfg, add_pin_kw_fn, precision: str):
        """
        Build a Python TDigest with explicit precision ("f32" or "f64").
        """
        import tdigest_rs as td

        py_kwargs = dict(
            max_size=cfg["max_size"],
            scale=cfg["scale_py"],
            singleton_policy=cfg["singleton_py"],
            precision=precision,
        )
        if cfg.get("singleton_py") == "edges" and cfg.get("pin_per_side") is not None:
            py_kwargs = add_pin_kw_fn(py_kwargs)

        d = td.TDigest.from_array(dataset["DATA"], **py_kwargs)
        # Sanity: inner_kind matches requested precision
        assert d.inner_kind() == precision
        return d

    @staticmethod
    def _build_polars_digest(dataset, cfg, add_pin_kw_fn, precision: str, float_dtype):
        """
        Build a Polars TDigest struct column over a single Float32/Float64 column.
        """
        import tdigest_rs as td
        import polars as pl

        df = pl.DataFrame({"x": dataset["DATA"]}).with_columns(
            pl.col("x").cast(float_dtype)
        )
        assert df.schema["x"] == float_dtype

        td_kwargs = dict(
            max_size=cfg["max_size"],
            scale=cfg["scale_pl"],
            singleton_policy=cfg["singleton_pl"],
            precision=precision,
        )
        if cfg.get("singleton_pl") == "edges" and cfg.get("pin_per_side") is not None:
            td_kwargs = add_pin_kw_fn(td_kwargs)

        df_td = df.with_columns(td_col=td.tdigest("x", **td_kwargs))
        assert "td_col" in df_td.schema
        return df_td

    @staticmethod
    def _assert_polars_digest_dtype(td_dtype, pl, expect_f32: bool):
        """
        Check struct schema:
        - compact (f32) → min/max are Float32
        - full   (f64) → min/max are Float64
        """
        # Polars dtypes don't have `.is_struct()`; we just rely on `.fields`.
        fields = getattr(td_dtype, "fields", None)
        assert fields is not None, f"expected Struct dtype, got {td_dtype!r}"

        min_field = next(f for f in fields if f.name == "min")
        max_field = next(f for f in fields if f.name == "max")

        if expect_f32:
            assert min_field.dtype == pl.Float32
            assert max_field.dtype == pl.Float32
        else:
            assert min_field.dtype == pl.Float64
            assert max_field.dtype == pl.Float64

    # ================================================================
    # 1. Python f32 → Polars f32
    # ================================================================
    def test_python_f32_to_polars_f32(
        self, dataset, cfg, add_pin_kw_fn, assert_close_fn
    ):
        """
        Python TDigest(f32) → to_bytes() → Polars td.from_bytes(...)
        - Polars struct column must be compact (Float32 min/max).
        - quantile/cdf must match Python digest.
        """
        import tdigest_rs as td
        import polars as pl

        d_py = self._build_python_digest(dataset, cfg, add_pin_kw_fn, precision="f32")
        blob = d_py.to_bytes()
        assert isinstance(blob, (bytes, bytearray, memoryview))

        df = pl.DataFrame({"blob": [blob]})
        df2 = df.with_columns(td_col=td.from_bytes("blob"))
        assert df2.height == 1

        # Schema: compact (f32) on the Polars side.
        td_dtype = df2.schema["td_col"]
        self._assert_polars_digest_dtype(td_dtype, pl, expect_f32=True)

        # Quantile / CDF coherence.
        x0 = float(dataset["DATA"][0])
        out = df2.select(
            q=td.quantile("td_col", 0.5),
            c=td.cdf("td_col", pl.lit(x0, dtype=pl.Float64)),
        )
        q_pl = float(out["q"][0])
        c_pl = float(out["c"][0])

        assert_close_fn(q_pl, d_py.quantile(0.5), 1e-9)
        assert_close_fn(c_pl, d_py.cdf(x0), 1e-9)

    # ================================================================
    # 2. Python f64 → Polars f64
    # ================================================================
    def test_python_f64_to_polars_f64(
        self, dataset, cfg, add_pin_kw_fn, assert_close_fn
    ):
        """
        Python TDigest(f64) → to_bytes() → Polars td.from_bytes(...)
        - Polars struct column must be full-precision (Float64 min/max).
        - quantile/cdf must match Python digest.
        """
        import tdigest_rs as td
        import polars as pl

        d_py = self._build_python_digest(dataset, cfg, add_pin_kw_fn, precision="f64")
        blob = d_py.to_bytes()

        df = pl.DataFrame({"blob": [blob]})
        df2 = df.with_columns(td_col=td.from_bytes("blob"))

        td_dtype = df2.schema["td_col"]
        self._assert_polars_digest_dtype(td_dtype, pl, expect_f32=False)

        x0 = float(dataset["DATA"][0])
        out = df2.select(
            q=td.quantile("td_col", 0.5),
            c=td.cdf("td_col", pl.lit(x0, dtype=pl.Float64)),
        )
        q_pl = float(out["q"][0])
        c_pl = float(out["c"][0])

        assert_close_fn(q_pl, d_py.quantile(0.5), 1e-9)
        assert_close_fn(c_pl, d_py.cdf(x0), 1e-9)

    # ================================================================
    # 3. Polars f32 → Python f32
    # ================================================================
    def test_polars_f32_to_python_f32(
        self, dataset, cfg, add_pin_kw_fn, assert_close_fn
    ):
        """
        Polars td.tdigest(..., precision="f32") on Float32 input:
        - Struct column is compact (Float32 min/max).
        - td.to_bytes(...) → Python TDigest.from_bytes(...) yields inner_kind() == "f32".
        - quantile/cdf coherent across the boundary.
        """
        import tdigest_rs as td
        import polars as pl

        df_td = self._build_polars_digest(
            dataset, cfg, add_pin_kw_fn, precision="f32", float_dtype=pl.Float32
        )

        # Schema check on Polars side.
        td_dtype = df_td.schema["td_col"]
        self._assert_polars_digest_dtype(td_dtype, pl, expect_f32=True)

        x0 = float(dataset["DATA"][0])
        out_pl = df_td.select(
            q=td.quantile("td_col", 0.5),
            c=td.cdf("td_col", pl.lit(x0, dtype=pl.Float64)),
        )
        q_pl = float(out_pl["q"][0])
        c_pl = float(out_pl["c"][0])

        # Serialize in Polars and rebuild in Python.
        blob = df_td.with_columns(blob=td.to_bytes("td_col")).select("blob").row(0)[0]

        d_py = td.TDigest.from_bytes(blob)
        assert isinstance(d_py, td.TDigest)
        assert d_py.inner_kind() == "f32"

        q_py = d_py.quantile(0.5)
        c_py = d_py.cdf(x0)

        assert_close_fn(q_py, q_pl, 1e-9)
        assert_close_fn(c_py, c_pl, 1e-9)

    # ================================================================
    # 4. Polars f64 → Python f64
    # ================================================================
    def test_polars_f64_to_python_f64(
        self, dataset, cfg, add_pin_kw_fn, assert_close_fn
    ):
        """
        Polars td.tdigest(..., precision="f64") on Float64 input:
        - Struct column is full-precision (Float64 min/max).
        - td.to_bytes(...) → Python TDigest.from_bytes(...) yields inner_kind() == "f64".
        - quantile/cdf coherent across the boundary.
        """
        import tdigest_rs as td
        import polars as pl

        df_td = self._build_polars_digest(
            dataset, cfg, add_pin_kw_fn, precision="f64", float_dtype=pl.Float64
        )

        td_dtype = df_td.schema["td_col"]
        self._assert_polars_digest_dtype(td_dtype, pl, expect_f32=False)

        x0 = float(dataset["DATA"][0])
        out_pl = df_td.select(
            q=td.quantile("td_col", 0.5),
            c=td.cdf("td_col", pl.lit(x0, dtype=pl.Float64)),
        )
        q_pl = float(out_pl["q"][0])
        c_pl = float(out_pl["c"][0])

        blob = df_td.with_columns(blob=td.to_bytes("td_col")).select("blob").row(0)[0]

        d_py = td.TDigest.from_bytes(blob)
        assert isinstance(d_py, td.TDigest)
        assert d_py.inner_kind() == "f64"

        q_py = d_py.quantile(0.5)
        c_py = d_py.cdf(x0)

        assert_close_fn(q_py, q_pl, 1e-9)
        assert_close_fn(c_py, c_pl, 1e-9)

    # ================================================================
    # 5. Python f32 → Polars → Python f32
    # ================================================================
    def test_python_f32_to_python_f32_via_polars(
        self, dataset, cfg, add_pin_kw_fn, assert_close_fn
    ):
        """
        Python TDigest(f32) → bytes → Polars from_bytes → td.to_bytes(...)
        → Python TDigest.from_bytes():
        - Polars struct is compact (Float32 min/max).
        - Final Python digest has inner_kind() == "f32".
        - quantile/cdf preserved end-to-end.
        """
        import tdigest_rs as td
        import polars as pl

        d_py = self._build_python_digest(dataset, cfg, add_pin_kw_fn, precision="f32")
        blob = d_py.to_bytes()

        df = pl.DataFrame({"blob": [blob]})
        df2 = df.with_columns(td_col=td.from_bytes("blob"))

        td_dtype = df2.schema["td_col"]
        self._assert_polars_digest_dtype(td_dtype, pl, expect_f32=True)

        # Serialize again from Polars and round-trip back to Python.
        blob2 = df2.with_columns(blob=td.to_bytes("td_col")).select("blob").row(0)[0]
        d_py2 = td.TDigest.from_bytes(blob2)

        assert isinstance(d_py2, td.TDigest)
        assert d_py2.inner_kind() == "f32"

        x0 = float(dataset["DATA"][0])
        assert_close_fn(d_py2.quantile(0.5), d_py.quantile(0.5), 1e-9)
        assert_close_fn(d_py2.cdf(x0), d_py.cdf(x0), 1e-9)

    # ================================================================
    # 6. Python f64 → Polars → Python f64
    # ================================================================
    def test_python_f64_to_python_f64_via_polars(
        self, dataset, cfg, add_pin_kw_fn, assert_close_fn
    ):
        """
        Python TDigest(f64) → bytes → Polars from_bytes → td.to_bytes(...)
        → Python TDigest.from_bytes():
        - Polars struct is full-precision (Float64 min/max).
        - Final Python digest has inner_kind() == "f64".
        - quantile/cdf preserved end-to-end.
        """
        import tdigest_rs as td
        import polars as pl

        d_py = self._build_python_digest(dataset, cfg, add_pin_kw_fn, precision="f64")
        blob = d_py.to_bytes()

        df = pl.DataFrame({"blob": [blob]})
        df2 = df.with_columns(td_col=td.from_bytes("blob"))

        td_dtype = df2.schema["td_col"]
        self._assert_polars_digest_dtype(td_dtype, pl, expect_f32=False)

        blob2 = df2.with_columns(blob=td.to_bytes("td_col")).select("blob").row(0)[0]
        d_py2 = td.TDigest.from_bytes(blob2)

        assert isinstance(d_py2, td.TDigest)
        assert d_py2.inner_kind() == "f64"

        x0 = float(dataset["DATA"][0])
        assert_close_fn(d_py2.quantile(0.5), d_py.quantile(0.5), 1e-9)
        assert_close_fn(d_py2.cdf(x0), d_py.cdf(x0), 1e-9)

    # ================================================================
    # 7. Polars f32 → Python → Polars f32
    # ================================================================
    def test_polars_f32_to_polars_f32_via_python(
        self, dataset, cfg, add_pin_kw_fn, assert_close_fn
    ):
        """
        Polars TDigest(f32) → td.to_bytes() → Python TDigest.from_bytes()
        → to_bytes() → Polars td.from_bytes():
        - Both Polars struct columns are compact (Float32 min/max).
        - Python intermediate digest has inner_kind() == "f32".
        - quantile/cdf preserved end-to-end.
        """
        import tdigest_rs as td
        import polars as pl

        df_td = self._build_polars_digest(
            dataset, cfg, add_pin_kw_fn, precision="f32", float_dtype=pl.Float32
        )

        # Original Polars side.
        td_dtype_orig = df_td.schema["td_col"]
        self._assert_polars_digest_dtype(td_dtype_orig, pl, expect_f32=True)

        x0 = float(dataset["DATA"][0])
        out_orig = df_td.select(
            q=td.quantile("td_col", 0.5),
            c=td.cdf("td_col", pl.lit(x0, dtype=pl.Float64)),
        )
        q_orig = float(out_orig["q"][0])
        c_orig = float(out_orig["c"][0])

        # Polars → Python
        blob = df_td.with_columns(blob=td.to_bytes("td_col")).select("blob").row(0)[0]
        d_py = td.TDigest.from_bytes(blob)
        assert d_py.inner_kind() == "f32"

        # Python → Polars
        blob2 = d_py.to_bytes()
        df2 = pl.DataFrame({"blob": [blob2]}).with_columns(td_rt=td.from_bytes("blob"))

        td_dtype_rt = df2.schema["td_rt"]
        self._assert_polars_digest_dtype(td_dtype_rt, pl, expect_f32=True)

        out_rt = df2.select(
            q=td.quantile("td_rt", 0.5),
            c=td.cdf("td_rt", pl.lit(x0, dtype=pl.Float64)),
        )
        q_rt = float(out_rt["q"][0])
        c_rt = float(out_rt["c"][0])

        assert_close_fn(q_rt, q_orig, 1e-9)
        assert_close_fn(c_rt, c_orig, 1e-9)

    # ================================================================
    # 8. Polars f64 → Python → Polars f64
    # ================================================================
    def test_polars_f64_to_polars_f64_via_python(
        self, dataset, cfg, add_pin_kw_fn, assert_close_fn
    ):
        """
        Polars TDigest(f64) → td.to_bytes() → Python TDigest.from_bytes()
        → to_bytes() → Polars td.from_bytes():
        - Both Polars struct columns are full-precision (Float64 min/max).
        - Python intermediate digest has inner_kind() == "f64".
        - quantile/cdf preserved end-to-end.
        """
        import tdigest_rs as td
        import polars as pl

        df_td = self._build_polars_digest(
            dataset, cfg, add_pin_kw_fn, precision="f64", float_dtype=pl.Float64
        )

        td_dtype_orig = df_td.schema["td_col"]
        self._assert_polars_digest_dtype(td_dtype_orig, pl, expect_f32=False)

        x0 = float(dataset["DATA"][0])
        out_orig = df_td.select(
            q=td.quantile("td_col", 0.5),
            c=td.cdf("td_col", pl.lit(x0, dtype=pl.Float64)),
        )
        q_orig = float(out_orig["q"][0])
        c_orig = float(out_orig["c"][0])

        blob = df_td.with_columns(blob=td.to_bytes("td_col")).select("blob").row(0)[0]
        d_py = td.TDigest.from_bytes(blob)
        assert d_py.inner_kind() == "f64"

        blob2 = d_py.to_bytes()
        df2 = pl.DataFrame({"blob": [blob2]}).with_columns(td_rt=td.from_bytes("blob"))

        td_dtype_rt = df2.schema["td_rt"]
        self._assert_polars_digest_dtype(td_dtype_rt, pl, expect_f32=False)

        out_rt = df2.select(
            q=td.quantile("td_rt", 0.5),
            c=td.cdf("td_rt", pl.lit(x0, dtype=pl.Float64)),
        )
        q_rt = float(out_rt["q"][0])
        c_rt = float(out_rt["c"][0])

        assert_close_fn(q_rt, q_orig, 1e-9)
        assert_close_fn(c_rt, c_orig, 1e-9)

    # ================================================================
    # 9. Mixed-precision blob column → infer_column_precision(strict=True) blows up
    # ================================================================
    def test_infer_column_precision_mixed_raises(self, dataset, cfg, add_pin_kw_fn):
        """
        Column contains both f32-encoded and f64-encoded TDIG blobs.
        - infer_column_precision(..., strict=True) must raise.
        - strict=False returns a fallback precision (currently "f64").
        """
        import tdigest_rs as td
        import polars as pl
        import pytest

        # Build one f32 and one f64 Python digest over the same data.
        d32 = self._build_python_digest(dataset, cfg, add_pin_kw_fn, precision="f32")
        d64 = self._build_python_digest(dataset, cfg, add_pin_kw_fn, precision="f64")

        blob32 = d32.to_bytes()
        blob64 = d64.to_bytes()

        df = pl.DataFrame({"blob": [blob32, blob64]})

        # Strict mode: must blow up on mixed precisions.
        with pytest.raises(ValueError, match="Mixed TDIG wire precisions"):
            td.infer_column_precision(df, "blob", strict=True)

        # Non-strict mode: returns a fallback (heavier) precision, currently "f64".
        prec = td.infer_column_precision(df, "blob", strict=False)
        assert prec in {"f64", "f32"}
        assert prec == "f64"

    # ================================================================
    # 10–13. Extra precision tests (homogeneous + nulls)
    # ================================================================
    def test_infer_column_precision_all_f32(self, dataset, cfg, add_pin_kw_fn):
        """
        All blobs are f32-encoded → infer_column_precision(..., strict=True/False) == "f32".
        """
        import tdigest_rs as td
        import polars as pl

        d32 = self._build_python_digest(dataset, cfg, add_pin_kw_fn, precision="f32")
        blob32 = d32.to_bytes()

        df = pl.DataFrame({"blob": [blob32, blob32]})

        prec_strict = td.infer_column_precision(df, "blob", strict=True)
        prec_relaxed = td.infer_column_precision(df, "blob", strict=False)

        assert prec_strict == "f32"
        assert prec_relaxed == "f32"

    def test_infer_column_precision_all_f64(self, dataset, cfg, add_pin_kw_fn):
        """
        All blobs are f64-encoded → infer_column_precision(..., strict=True/False) == "f64".
        """
        import tdigest_rs as td
        import polars as pl

        d64 = self._build_python_digest(dataset, cfg, add_pin_kw_fn, precision="f64")
        blob64 = d64.to_bytes()

        df = pl.DataFrame({"blob": [blob64, blob64]})

        prec_strict = td.infer_column_precision(df, "blob", strict=True)
        prec_relaxed = td.infer_column_precision(df, "blob", strict=False)

        assert prec_strict == "f64"
        assert prec_relaxed == "f64"

    def test_infer_column_precision_all_nulls_defaults_f64(self):
        """
        All-null blob column → default fallback is "f64".
        """
        import tdigest_rs as td
        import polars as pl

        df = pl.DataFrame({"blob": [None, None]}).with_columns(
            pl.col("blob").cast(pl.Binary)
        )
        prec = td.infer_column_precision(df, "blob", strict=True)
        assert prec == "f64"

    # ================================================================
    # 14–16. from_bytes + precision hints behaviour
    # ================================================================
    def test_from_bytes_auto_mixed_column_raises(self, dataset, cfg, add_pin_kw_fn):
        """
        td.from_bytes("blob") with precision="auto" must fail on mixed f32/f64 blobs.
        """
        import tdigest_rs as td
        import polars as pl
        import pytest

        d32 = self._build_python_digest(dataset, cfg, add_pin_kw_fn, precision="f32")
        d64 = self._build_python_digest(dataset, cfg, add_pin_kw_fn, precision="f64")

        blob32 = d32.to_bytes()
        blob64 = d64.to_bytes()

        df = pl.DataFrame({"blob": [blob32, blob64]})

        with pytest.raises(
            pl.exceptions.ComputeError, match="mixed f32/f64 blobs in column"
        ):
            df.with_columns(td_col=td.from_bytes("blob")).collect()

    def test_from_bytes_with_precision_f32_on_f64_blobs_raises(
        self, dataset, cfg, add_pin_kw_fn
    ):
        """
        td.from_bytes(..., precision="f32") must fail if blobs are f64-encoded.
        """
        import tdigest_rs as td
        import polars as pl
        import pytest

        d64 = self._build_python_digest(dataset, cfg, add_pin_kw_fn, precision="f64")
        blob64 = d64.to_bytes()
        df = pl.DataFrame({"blob": [blob64]})

        with pytest.raises(
            pl.exceptions.ComputeError, match="mixed f32/f64 blobs in column"
        ):
            df.with_columns(td_col=td.from_bytes("blob", precision="f32")).collect()

    def test_from_bytes_with_precision_f64_on_f32_blobs_raises(
        self, dataset, cfg, add_pin_kw_fn
    ):
        """
        td.from_bytes(..., precision="f64") must fail if blobs are f32-encoded.
        """
        import tdigest_rs as td
        import polars as pl
        import pytest

        d32 = self._build_python_digest(dataset, cfg, add_pin_kw_fn, precision="f32")
        blob32 = d32.to_bytes()
        df = pl.DataFrame({"blob": [blob32]})

        with pytest.raises(
            pl.exceptions.ComputeError, match="mixed f32/f64 blobs in column"
        ):
            df.with_columns(td_col=td.from_bytes("blob", precision="f64")).collect()

    def test_from_bytes_with_any_null_blob_row_raises(
        self, dataset, cfg, add_pin_kw_fn
    ):
        """
        Null blob rows are invalid input for td.from_bytes(...): strict error.
        """
        import tdigest_rs as td
        import polars as pl
        import pytest

        d64 = self._build_python_digest(dataset, cfg, add_pin_kw_fn, precision="f64")
        blob64 = d64.to_bytes()
        df = pl.DataFrame({"blob": [blob64, None]}).with_columns(
            pl.col("blob").cast(pl.Binary)
        )

        with pytest.raises(pl.exceptions.ComputeError, match="(?i)null"):
            df.with_columns(td_col=td.from_bytes("blob", precision="auto"))

    def test_from_bytes_all_null_blob_column_raises(self):
        """
        All-null blob columns are invalid for td.from_bytes(...): strict error.
        """
        import tdigest_rs as td
        import polars as pl
        import pytest

        df = pl.DataFrame({"blob": [None, None]}).with_columns(
            pl.col("blob").cast(pl.Binary)
        )

        with pytest.raises(pl.exceptions.ComputeError, match="(?i)null"):
            df.with_columns(td_col=td.from_bytes("blob", precision="auto"))

    def test_from_bytes_empty_blob_bytes_raises_decode_error(self):
        """
        Empty bytes (b"") are not a valid TDIG payload and must fail decode.
        """
        import tdigest_rs as td
        import polars as pl
        import pytest

        df = pl.DataFrame({"blob": [b""]})

        with pytest.raises(
            pl.exceptions.ComputeError, match="(?i)(tdig|decode|header|buffer)"
        ):
            df.with_columns(td_col=td.from_bytes("blob", precision="auto"))

    def test_from_bytes_with_precision_f32_homogeneous_column(
        self, dataset, cfg, add_pin_kw_fn, assert_close_fn
    ):
        """
        td.from_bytes(..., precision="f32") on homogeneous f32 blobs:
        - schema is compact (Float32 min/max)
        - quantile/cdf coherent with original Python digest.
        """
        import tdigest_rs as td
        import polars as pl

        d32 = self._build_python_digest(dataset, cfg, add_pin_kw_fn, precision="f32")
        blob32 = d32.to_bytes()

        df = pl.DataFrame({"blob": [blob32]})
        df2 = df.with_columns(td_col=td.from_bytes("blob", precision="auto"))

        td_dtype = df2.schema["td_col"]
        self._assert_polars_digest_dtype(td_dtype, pl, expect_f32=True)

        x0 = float(dataset["DATA"][0])
        out = df2.select(
            q=td.quantile("td_col", 0.5),
            c=td.cdf("td_col", pl.lit(x0, dtype=pl.Float64)),
        )
        q_pl = float(out["q"][0])
        c_pl = float(out["c"][0])

        assert_close_fn(q_pl, d32.quantile(0.5), 1e-9)
        assert_close_fn(c_pl, d32.cdf(x0), 1e-9)

    def test_from_bytes_precision_auto_homogeneous_f32(
        self, dataset, cfg, add_pin_kw_fn, assert_close_fn
    ):
        """
        td.from_bytes(..., precision="auto") on homogeneous f32 blobs:
        - schema is compact (Float32 min/max),
        - quantile/cdf coherent with original Python TDigest(f32),
        - Python roundtrip keeps inner_kind() == "f32".
        """
        import tdigest_rs as td
        import polars as pl

        # Build a Python f32 digest and serialize to wire bytes.
        d32 = self._build_python_digest(dataset, cfg, add_pin_kw_fn, precision="f32")
        blob32 = d32.to_bytes()

        # Single-row Binary column with only f32-encoded blobs.
        df = pl.DataFrame({"blob": [blob32]})

        # Use precision="auto" when reading — this should end up as a compact f32 struct.
        df2 = df.with_columns(td_col=td.from_bytes("blob", precision="auto"))

        # 1) Schema: compact (Float32-backed struct).
        td_dtype = df2.schema["td_col"]
        self._assert_polars_digest_dtype(td_dtype, pl, expect_f32=True)

        # 2) Quantile / CDF coherence vs original Python digest.
        x0 = float(dataset["DATA"][0])
        out_pl = df2.select(
            q=td.quantile("td_col", 0.5),
            c=td.cdf("td_col", pl.lit(x0, dtype=pl.Float64)),
        )
        q_pl = float(out_pl["q"][0])
        c_pl = float(out_pl["c"][0])

        assert_close_fn(q_pl, d32.quantile(0.5), 1e-9)
        assert_close_fn(c_pl, d32.cdf(x0), 1e-9)

        # 3) Roundtrip back to Python: still an f32-backed digest, and values preserved.
        blob_rt = df2.with_columns(blob=td.to_bytes("td_col")).select("blob").row(0)[0]
        d_rt = td.TDigest.from_bytes(blob_rt)
        assert d_rt.inner_kind() == "f32"

        q_rt = d_rt.quantile(0.5)
        c_rt = d_rt.cdf(x0)

        assert_close_fn(q_rt, d32.quantile(0.5), 1e-9)
        assert_close_fn(c_rt, d32.cdf(x0), 1e-9)

    def test_python_to_bytes_explicit_wire_versions_roundtrip(
        self, dataset, cfg, add_pin_kw_fn, assert_close_fn
    ):
        """
        Python to_bytes(version=1|2|3) must produce decodable TDIG blobs
        with coherent quantile/cdf behavior.
        """
        import tdigest_rs as td

        d = self._build_python_digest(dataset, cfg, add_pin_kw_fn, precision="f64")
        x0 = float(dataset["DATA"][0])

        for version in (1, 2, 3):
            blob = d.to_bytes(version=version)
            d_rt = td.TDigest.from_bytes(blob)
            assert d_rt.inner_kind() == "f64"
            assert_close_fn(d_rt.quantile(0.5), d.quantile(0.5), 1e-6)
            assert_close_fn(d_rt.cdf(x0), d.cdf(x0), 1e-6)

    def test_from_bytes_auto_accepts_mixed_v1_v2_same_precision(
        self, dataset, cfg, add_pin_kw_fn, assert_close_fn
    ):
        """
        Migration behavior: a Binary column mixing TDIG v1 and v2 blobs with
        the same precision (f64) must decode under precision=\"auto\".
        """
        import tdigest_rs as td
        import polars as pl

        d = self._build_python_digest(dataset, cfg, add_pin_kw_fn, precision="f64")
        blob_v1 = d.to_bytes(version=1)
        blob_v2 = d.to_bytes(version=2)

        df = pl.DataFrame({"blob": [blob_v1, blob_v2]})
        df2 = df.with_columns(td_col=td.from_bytes("blob", precision="auto"))

        out = df2.select(
            q=td.quantile("td_col", 0.5),
            c=td.cdf("td_col", pl.lit(float(dataset["DATA"][0]), dtype=pl.Float64)),
        )
        q = float(out["q"][0])
        c = float(out["c"][0])

        assert_close_fn(q, d.quantile(0.5), 1e-6)
        assert_close_fn(c, d.cdf(float(dataset["DATA"][0])), 1e-6)


# =====================================================================
# Category 3: JAVA → PYTHON — TDigest.toBytes() → TDigest.from_bytes()
#   Spec:
#     - Java builds digest from canonical data.
#     - Java calls toBytes() and prints base64(blob).
#     - Python decodes base64, TDigest.from_bytes(...).
#     - quantile/cdf must match Java expectations and canonical values.
#     - Round-trip: once Python has the blob, TDigest.from_bytes ⟷ to_bytes
#       must be stable.
# =====================================================================
