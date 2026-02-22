import math

import numpy as np
import polars as pl
import pytest

from tdigest_rs import (
    ScaleFamily,
    add_weighted_values,
    add_values,
    cast_precision,
    cdf,
    from_bytes,
    median,
    merge_tdigests,
    quantile,
    scale_values,
    scale_weights,
    tdigest,
    to_bytes,
)


def _extract_vector(df: pl.DataFrame, col: str) -> list[float]:
    if df.height == 1:
        return df.item()
    return df.get_column(col).to_list()


class TestPolarsPluginSmoke:
    def test_quantile_cdf_median_smoke(self):
        df = pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
        d = df.select(tdigest("x", max_size=100, scale="k2").alias("td"))

        q = d.select(quantile("td", 0.5)).item()
        m = d.select(median("td")).item()
        c = d.select(cdf("td", 1.5)).item()

        assert q == pytest.approx(1.5, abs=1e-9)
        assert m == pytest.approx(1.5, abs=1e-9)
        assert c == pytest.approx(0.5, abs=1e-9)

    def test_add_values_and_merge_tdigests_smoke(self):
        df_a = pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
        df_b = pl.DataFrame({"x": [10.0, 11.0, 12.0, 13.0]})

        a = df_a.select(tdigest("x", max_size=100, scale="k2").alias("td"))
        b = df_b.select(tdigest("x", max_size=100, scale="k2").alias("td"))

        a2 = a.select(td2=add_values("td", 4.0)).select(td3=add_values("td2", [5.0, 6.0]))
        merged = pl.concat([a2.select("td3").rename({"td3": "td"}), b.select("td")], how="vertical")

        out = merged.select(td=merge_tdigests("td")).select(
            q=quantile("td", 0.5),
            m=median("td"),
            c=cdf("td", 3.0),
        )
        assert math.isfinite(float(out["q"][0]))
        assert math.isfinite(float(out["m"][0]))
        assert math.isfinite(float(out["c"][0]))

    def test_cdf_input_shapes(self):
        d = pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]}).select(
            tdigest("x", max_size=64, scale=ScaleFamily.K2).alias("td")
        )

        ys_list = _extract_vector(d.select(cdf("td", [0.0, 1.5, 3.0]).alias("cdf")), "cdf")
        assert ys_list == pytest.approx([0.125, 0.5, 0.875], abs=1e-9)

        y_scalar = d.select(cdf(pl.col("td"), 1.5)).item()
        assert y_scalar == pytest.approx(0.5, abs=1e-9)

        ys_np = _extract_vector(
            d.select(cdf("td", np.array([0.0, 1.5, 3.0], dtype=float)).alias("cdf")),
            "cdf",
        )
        assert ys_np == pytest.approx([0.125, 0.5, 0.875], abs=1e-9)

    def test_expression_list_in_select(self):
        df = pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "y": [3.0, 3.0, 1.0, 0.0]})
        df2 = df.with_columns(td_x=tdigest("x"), td_y=tdigest("y"))

        out = df2.select(
            [
                cdf(pl.col("td_x"), "x").alias("cx"),
                quantile("td_y", 0.5).alias("qy"),
            ]
        )

        assert out.columns == ["cx", "qy"]
        assert out["cx"].is_sorted()
        assert float(out["qy"][0]) == pytest.approx(1.6666666666666667, abs=1e-9)

    def test_scale_weights_and_values_smoke(self):
        d = pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]}).select(
            tdigest("x", max_size=64, scale=ScaleFamily.K2).alias("td")
        )

        q0 = float(d.select(quantile("td", 0.5)).item())
        c0 = float(d.select(cdf("td", 1.5)).item())

        scaled_w = d.select(td2=scale_weights("td", 2.0))
        assert float(scaled_w.select(quantile("td2", 0.5)).item()) == pytest.approx(q0, abs=1e-9)
        assert float(scaled_w.select(cdf("td2", 1.5)).item()) == pytest.approx(c0, abs=1e-9)

        scaled_v = d.select(td2=scale_values("td", 3.0))
        assert float(scaled_v.select(quantile("td2", 0.5)).item()) == pytest.approx(q0 * 3.0, abs=1e-9)
        assert float(scaled_v.select(median("td2")).item()) == pytest.approx(q0 * 3.0, abs=1e-9)
        assert float(scaled_v.select(cdf("td2", 1.5 * 3.0)).item()) == pytest.approx(c0, abs=1e-9)

    def test_add_weighted_values_and_cast_precision_smoke(self):
        d = pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]}).select(
            tdigest("x", max_size=128, scale=ScaleFamily.K2).alias("td")
        )

        d_w = d.select(td2=add_weighted_values("td", [10.0, 20.0], [2.0, 3.0]))
        q = float(d_w.select(quantile("td2", 0.5)).item())
        assert math.isfinite(q)

        d32 = d_w.select(td3=cast_precision("td2", precision="f32"))
        fields32 = d32.schema["td3"].fields
        assert next(f for f in fields32 if f.name == "min").dtype == pl.Float32

        d64 = d32.select(td4=cast_precision("td3", precision="f64"))
        fields64 = d64.schema["td4"].fields
        assert next(f for f in fields64 if f.name == "min").dtype == pl.Float64

    def test_to_bytes_supports_explicit_versions(self):
        d = pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]}).select(tdigest("x").alias("td"))

        for version in (1, 2, 3):
            blob = d.select(b=to_bytes("td", version=version)).item()
            assert isinstance(blob, (bytes, bytearray))
            rt = pl.DataFrame({"blob": [blob]}).with_columns(td2=from_bytes("blob", precision="f64"))
            assert math.isfinite(float(rt.select(quantile("td2", 0.5)).item()))


class TestPolarsPluginValidation:
    def test_tdigest_rejects_nan_inf_null_training(self):
        for df in [
            pl.DataFrame({"x": [0.0, float("nan"), 1.0]}),
            pl.DataFrame({"x": [0.0, float("+inf"), 1.0]}),
            pl.DataFrame({"x": [0.0, float("-inf"), 1.0]}),
            pl.DataFrame({"x": [None, 1.0]}),
        ]:
            df = df.with_columns(pl.col("x").cast(pl.Float64))
            with pytest.raises(pl.exceptions.ComputeError):
                df.select(tdigest("x"))

    def test_groupby_empty_digest_behavior(self):
        df = pl.DataFrame({"id": [0, 0, 1, 1, 1], "data": [-3.0, -2.0, 15.0, 25.0, 35.0]})
        agg = (
            df.group_by("id")
            .agg(tdigest(pl.col("data").filter(pl.col("data") > 0), max_size=64).alias("td"))
            .sort("id")
        )

        probe = (pl.col("id") * 0 + 25.0).cast(pl.Float64)
        out = agg.with_columns(
            q=quantile("td", 0.5).over("id"),
            c=cdf("td", probe).over("id"),
        ).sort("id")

        r0, r1 = out.row(0), out.row(1)
        assert r0[2] is None
        assert math.isnan(float(r0[3]))
        assert r1[2] == pytest.approx(25.0, abs=1e-9)
        assert r1[3] == pytest.approx(0.5, abs=1e-9)

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_scaling_rejects_invalid_factor(self, bad):
        d = pl.DataFrame({"x": [0.0, 1.0, 2.0]}).select(tdigest("x").alias("td"))
        with pytest.raises(pl.exceptions.ComputeError):
            d.select(scale_weights("td", bad))
        with pytest.raises(pl.exceptions.ComputeError):
            d.select(scale_values("td", bad))

    def test_add_weighted_values_rejects_invalid_inputs(self):
        d = pl.DataFrame({"x": [0.0, 1.0]}).select(tdigest("x").alias("td"))
        with pytest.raises(pl.exceptions.ComputeError):
            d.select(add_weighted_values("td", [1.0, 2.0], [1.0]))
        with pytest.raises(pl.exceptions.ComputeError):
            d.select(add_weighted_values("td", [1.0], [0.0]))
