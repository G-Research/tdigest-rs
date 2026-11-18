from __future__ import annotations

import pytest


class TestPythonRoundTripSerde:
    def test_python_round_trip(self, dataset, expect, cfg, assert_close_fn):
        import tdigest_rs as td

        kwargs = dict(
            max_size=cfg["max_size"],
            scale=cfg["scale_py"],
            singleton_policy=cfg["singleton_py"],
            precision=cfg["precision_py"],
        )

        d1 = td.TDigest.from_array(dataset["DATA"], **kwargs)

        blob = d1.to_bytes()
        assert isinstance(blob, (bytes, bytearray))

        d2 = td.TDigest.from_bytes(bytes(blob))

        q1 = d1.quantile(expect["P"])
        q2 = d2.quantile(expect["P"])
        c1 = d1.cdf(expect["X"])
        c2 = d2.cdf(expect["X"])

        assert_close_fn(q1, expect["Q50"], expect["EPS"])
        assert_close_fn(c1, expect["CDF2"], expect["EPS"])
        assert_close_fn(q2, q1, expect["EPS"])
        assert_close_fn(c2, c1, expect["EPS"])
        assert_close_fn(q2, expect["Q50"], expect["EPS"])
        assert_close_fn(c2, expect["CDF2"], expect["EPS"])


class TestPythonPolarsSerdeInterop:
    @staticmethod
    def _build_python_digest(dataset, cfg, precision: str):
        import tdigest_rs as td

        py_kwargs = dict(
            max_size=cfg["max_size"],
            scale=cfg["scale_py"],
            singleton_policy=cfg["singleton_py"],
            precision=precision,
        )

        d = td.TDigest.from_array(dataset["DATA"], **py_kwargs)
        assert d.inner_kind() == precision
        return d

    @staticmethod
    def _build_polars_digest(dataset, cfg, precision: str, float_dtype):
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

    @pytest.mark.parametrize("precision", ["f32", "f64"])
    def test_python_to_polars_and_back(self, dataset, cfg, precision):
        import polars as pl
        import tdigest_rs as td

        original = self._build_python_digest(dataset, cfg, precision)
        blob = original.to_bytes()
        assert isinstance(blob, (bytes, bytearray, memoryview))
        decoded = pl.DataFrame({"blob": [blob]}).with_columns(td_col=td.from_bytes("blob", precision=precision))
        assert decoded.height == 1
        self._assert_polars_digest_dtype(decoded.schema["td_col"], pl, precision == "f32")

        x = float(dataset["DATA"][0])
        expected = (original.quantile(0.5), original.cdf(x))
        queries = decoded.select(q=td.quantile("td_col", 0.5), c=td.cdf("td_col", pl.lit(x, dtype=pl.Float64)))
        assert queries.row(0) == pytest.approx(expected, rel=0, abs=1e-9)

        restored = td.TDigest.from_bytes(decoded.select(td.to_bytes("td_col")).item())
        assert isinstance(restored, td.TDigest)
        assert restored.inner_kind() == precision
        assert (restored.quantile(0.5), restored.cdf(x)) == pytest.approx(expected, rel=0, abs=1e-9)

    @pytest.mark.parametrize("precision", ["f32", "f64"])
    def test_polars_to_python_and_back(self, dataset, cfg, precision):
        import polars as pl
        import tdigest_rs as td

        dtype = pl.Float32 if precision == "f32" else pl.Float64
        original = self._build_polars_digest(dataset, cfg, precision, dtype)
        self._assert_polars_digest_dtype(original.schema["td_col"], pl, precision == "f32")
        x = float(dataset["DATA"][0])
        queries = [td.quantile("td_col", 0.5).alias("q"), td.cdf("td_col", pl.lit(x, dtype=pl.Float64)).alias("c")]
        expected = original.select(queries).row(0)

        decoded = td.TDigest.from_bytes(original.select(td.to_bytes("td_col")).item())
        assert isinstance(decoded, td.TDigest)
        assert decoded.inner_kind() == precision
        assert (decoded.quantile(0.5), decoded.cdf(x)) == pytest.approx(expected, rel=0, abs=1e-9)

        restored = pl.DataFrame({"blob": [decoded.to_bytes()]}).select(
            td.from_bytes("blob", precision=precision).alias("td_col")
        )
        self._assert_polars_digest_dtype(restored.schema["td_col"], pl, precision == "f32")
        assert restored.select(queries).row(0) == pytest.approx(expected, rel=0, abs=1e-9)

    def test_infer_column_precision_mixed_raises(self, dataset, cfg):
        import tdigest_rs as td
        import polars as pl

        d32 = self._build_python_digest(dataset, cfg, precision="f32")
        d64 = self._build_python_digest(dataset, cfg, precision="f64")

        blob32 = d32.to_bytes()
        blob64 = d64.to_bytes()

        df = pl.DataFrame({"blob": [blob32, blob64]})

        with pytest.raises(ValueError, match="Mixed TDIG wire precisions"):
            td.infer_column_precision(df, "blob", strict=True)

        prec = td.infer_column_precision(df, "blob", strict=False)
        assert prec == "f64"

    @pytest.mark.parametrize("precision", ["f32", "f64"])
    def test_infer_homogeneous_precision(self, dataset, cfg, precision):
        import polars as pl
        import tdigest_rs as td

        blob = self._build_python_digest(dataset, cfg, precision).to_bytes()
        frame = pl.DataFrame({"blob": [blob, blob]})
        assert td.infer_column_precision(frame, "blob", strict=True) == precision
        assert td.infer_column_precision(frame, "blob", strict=False) == precision

    def test_infer_column_precision_all_nulls_defaults_f64(self):
        import tdigest_rs as td
        import polars as pl

        df = pl.DataFrame({"blob": [None, None]}).with_columns(
            pl.col("blob").cast(pl.Binary)
        )
        prec = td.infer_column_precision(df, "blob", strict=True)
        assert prec == "f64"

    def test_from_bytes_explicit_mixed_column_raises(self, dataset, cfg):
        import tdigest_rs as td
        import polars as pl

        d32 = self._build_python_digest(dataset, cfg, precision="f32")
        d64 = self._build_python_digest(dataset, cfg, precision="f64")

        blob32 = d32.to_bytes()
        blob64 = d64.to_bytes()

        df = pl.DataFrame({"blob": [blob32, blob64]})

        with pytest.raises(
            pl.exceptions.ComputeError, match="mixed f32/f64 blobs in column"
        ):
            df.with_columns(td_col=td.from_bytes("blob", precision="f64"))

    @pytest.mark.parametrize(("precision", "hint"), [("f32", "f64"), ("f64", "f32")])
    def test_from_bytes_rejects_wrong_precision(self, dataset, cfg, precision, hint):
        import polars as pl
        import tdigest_rs as td

        blob = self._build_python_digest(dataset, cfg, precision).to_bytes()
        frame = pl.DataFrame({"blob": [blob]})
        with pytest.raises(pl.exceptions.ComputeError, match="mixed f32/f64 blobs in column"):
            frame.with_columns(td_col=td.from_bytes("blob", precision=hint))

    def test_from_bytes_with_any_null_blob_row_raises(
        self, dataset, cfg
    ):
        import tdigest_rs as td
        import polars as pl

        d64 = self._build_python_digest(dataset, cfg, precision="f64")
        blob64 = d64.to_bytes()
        df = pl.DataFrame({"blob": [blob64, None]}).with_columns(
            pl.col("blob").cast(pl.Binary)
        )

        with pytest.raises(pl.exceptions.ComputeError, match="(?i)null"):
            df.with_columns(td_col=td.from_bytes("blob", precision="f64"))

    def test_from_bytes_all_null_blob_column_raises(self):
        import tdigest_rs as td
        import polars as pl

        df = pl.DataFrame({"blob": [None, None]}).with_columns(
            pl.col("blob").cast(pl.Binary)
        )

        with pytest.raises(pl.exceptions.ComputeError, match="(?i)null"):
            df.with_columns(td_col=td.from_bytes("blob", precision="f64"))

    def test_from_bytes_empty_blob_bytes_raises_decode_error(self):
        import tdigest_rs as td
        import polars as pl

        df = pl.DataFrame({"blob": [b""]})

        with pytest.raises(
            pl.exceptions.ComputeError, match="(?i)(tdig|decode|header|buffer)"
        ):
            df.with_columns(td_col=td.from_bytes("blob", precision="f64"))

    def test_inferred_f32_hint_preserves_wire_precision(
        self, dataset, cfg, assert_close_fn
    ):
        import tdigest_rs as td
        import polars as pl

        d32 = self._build_python_digest(dataset, cfg, precision="f32")
        blob32 = d32.to_bytes()

        df = pl.DataFrame({"blob": [blob32]})

        df2 = df.with_columns(td_col=td.from_bytes("blob", precision=td.infer_column_precision(df, "blob")))

        td_dtype = df2.schema["td_col"]
        self._assert_polars_digest_dtype(td_dtype, pl, expect_f32=True)

        x0 = float(dataset["DATA"][0])
        out_pl = df2.select(
            q=td.quantile("td_col", 0.5),
            c=td.cdf("td_col", pl.lit(x0, dtype=pl.Float64)),
        )
        q_pl = float(out_pl["q"][0])
        c_pl = float(out_pl["c"][0])

        assert_close_fn(q_pl, d32.quantile(0.5), 1e-9)
        assert_close_fn(c_pl, d32.cdf(x0), 1e-9)

        blob_rt = df2.with_columns(blob=td.to_bytes("td_col")).select("blob").row(0)[0]
        d_rt = td.TDigest.from_bytes(blob_rt)
        assert d_rt.inner_kind() == "f32"

        q_rt = d_rt.quantile(0.5)
        c_rt = d_rt.cdf(x0)

        assert_close_fn(q_rt, d32.quantile(0.5), 1e-9)
        assert_close_fn(c_rt, d32.cdf(x0), 1e-9)

    def test_python_to_bytes_explicit_wire_versions_roundtrip(
        self, dataset, cfg, assert_close_fn
    ):
        import tdigest_rs as td

        d = self._build_python_digest(dataset, cfg, precision="f64")
        x0 = float(dataset["DATA"][0])

        for version in (1, 2, 3):
            blob = d.to_bytes(version=version)
            d_rt = td.TDigest.from_bytes(blob)
            assert d_rt.inner_kind() == "f64"
            assert_close_fn(d_rt.quantile(0.5), d.quantile(0.5), 1e-6)
            assert_close_fn(d_rt.cdf(x0), d.cdf(x0), 1e-6)

    def test_from_bytes_explicit_accepts_mixed_v1_v2_same_precision(
        self, dataset, cfg, assert_close_fn
    ):
        import tdigest_rs as td
        import polars as pl

        d = self._build_python_digest(dataset, cfg, precision="f64")
        blob_v1 = d.to_bytes(version=1)
        blob_v2 = d.to_bytes(version=2)

        df = pl.DataFrame({"blob": [blob_v1, blob_v2]})
        df2 = df.with_columns(td_col=td.from_bytes("blob", precision="f64"))

        out = df2.select(
            q=td.quantile("td_col", 0.5),
            c=td.cdf("td_col", pl.lit(float(dataset["DATA"][0]), dtype=pl.Float64)),
        )
        q = float(out["q"][0])
        c = float(out["c"][0])

        assert_close_fn(q, d.quantile(0.5), 1e-6)
        assert_close_fn(c, d.cdf(float(dataset["DATA"][0])), 1e-6)
