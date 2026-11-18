import math
import pickle
from copy import deepcopy

import numpy as np
import pytest

from tdigest_rs import ScaleFamily, SingletonPolicy, TDigest


class TestPythonApiSmoke:
    def test_build_and_query_smoke(self):
        d = TDigest.from_array([0.0, 1.0, 2.0, 3.0], max_size=64, scale=ScaleFamily.K2)
        assert d.quantile(0.5) == pytest.approx(1.5, abs=1e-9)
        assert d.median() == pytest.approx(1.5, abs=1e-9)
        assert d.cdf(1.5) == pytest.approx(0.5, abs=1e-9)

    @pytest.mark.parametrize("f32_mode", [False, True])
    def test_add_scalar_and_vector(self, f32_mode):
        d = TDigest.from_array([0.0, 1.0, 2.0, 3.0], max_size=64, scale="k2", f32_mode=f32_mode)
        out = d.add(4.0).add([5.0, 6.0])
        assert out is d
        assert d.quantile(0.5) == pytest.approx(3.0, abs=1e-9)

    def test_cdf_scalar_and_vector_shapes(self):
        d = TDigest.from_array([0.0, 1.0, 2.0, 3.0], max_size=64, scale="k2")

        scalar = d.cdf(1.5)
        assert isinstance(scalar, float)
        assert scalar == pytest.approx(0.5, abs=1e-9)

        as_list = d.cdf([0.0, 1.5, 3.0])
        assert isinstance(as_list, list)
        assert as_list == pytest.approx([0.125, 0.5, 0.875], abs=1e-9)

        as_np = d.cdf(np.array([0.0, 1.5, 3.0], dtype=float))
        assert isinstance(as_np, np.ndarray)
        assert as_np.shape == (3,)
        assert list(as_np) == pytest.approx([0.125, 0.5, 0.875], abs=1e-9)

    def test_merge_and_merge_all_smoke(self):
        a = TDigest.from_array([0.0, 1.0, 2.0, 3.0], max_size=64, scale="k2")
        b = TDigest.from_array([10.0, 11.0, 12.0, 13.0], max_size=64, scale="k2")

        before = a.quantile(0.5)
        out = a.merge(b)
        assert out is a
        assert a.quantile(0.5) > before

        merged = TDigest.merge_all([a, b])
        assert math.isfinite(merged.quantile(0.5))
        assert math.isfinite(merged.cdf(3.0))
        assert math.isfinite(merged.median())

    def test_bytes_roundtrip_and_inner_kind(self):
        d = TDigest.from_array([10.0, 20.0, 30.0], max_size=32, scale="k2", precision="f32")
        blob = d.to_bytes()
        d2 = TDigest.from_bytes(blob)

        assert d.inner_kind() == "f32"
        assert d2.inner_kind() == "f32"
        assert d2.median() == pytest.approx(d.median(), abs=1e-9)

    def test_scale_weights_and_values_smoke(self):
        d = TDigest.from_array([0.0, 1.0, 2.0, 3.0], max_size=64, scale="k2")
        q0 = d.quantile(0.5)
        c0 = d.cdf(1.5)

        out_w = d.scale_weights(2.0)
        assert out_w is d
        assert d.quantile(0.5) == pytest.approx(q0, abs=1e-9)
        assert d.cdf(1.5) == pytest.approx(c0, abs=1e-9)

        out_v = d.scale_values(3.0)
        assert out_v is d
        assert d.quantile(0.5) == pytest.approx(q0 * 3.0, abs=1e-9)
        assert d.median() == pytest.approx(q0 * 3.0, abs=1e-9)
        assert d.cdf(1.5 * 3.0) == pytest.approx(c0, abs=1e-9)

    def test_weighted_add_cast_precision_and_versioned_bytes(self):
        d = TDigest.from_array([0.0, 1.0, 2.0], max_size=128, scale="k2", precision="f64")
        out = d.add_weighted([10.0, 20.0], [2.0, 3.0])
        assert out is d
        assert math.isfinite(d.quantile(0.5))

        d32 = d.cast_precision("f32")
        d64 = d32.cast_precision("f64")
        assert d32.inner_kind() == "f32"
        assert d64.inner_kind() == "f64"
        assert d64.quantile(0.5) == pytest.approx(d.quantile(0.5), abs=1e-4)

        for version in (1, 2, 3):
            blob = d.to_bytes(version=version)
            rt = TDigest.from_bytes(blob)
            assert math.isfinite(rt.quantile(0.5))

    def test_delta_mode_uses_legacy_k2_cluster_rule(self):
        d = TDigest.from_means_weights(
            np.arange(5, dtype=np.float64),
            np.array([100, 1, 1, 1, 1], dtype=np.uint32),
            delta=20.0,
        )

        assert len(d) == 4

    def test_pickle_and_deepcopy_roundtrip_via_wire_bytes(self):
        d = TDigest.from_array([0.0, 1.0, 2.0, 3.0], max_size=64, scale="k2", precision="f32")

        loaded = pickle.loads(pickle.dumps(d))
        copied = deepcopy(d)

        assert loaded.inner_kind() == "f32"
        assert copied.inner_kind() == "f32"
        assert loaded.median() == pytest.approx(d.median(), abs=1e-9)
        assert copied.quantile(0.5) == pytest.approx(d.quantile(0.5), abs=1e-9)


class TestPythonApiValidation:
    @pytest.mark.parametrize("bad", [float("nan"), float("+inf"), float("-inf")])
    def test_rejects_non_finite_training_and_add(self, bad):
        with pytest.raises(ValueError, match="(?i)nan|inf"):
            TDigest.from_array([0.0, bad, 1.0], max_size=64, scale="k2")

        d = TDigest.from_array([0.0, 1.0], max_size=64, scale="k2")
        with pytest.raises(ValueError):
            d.add([bad])

    @pytest.mark.parametrize("bad", [float("nan"), float("+inf"), float("-inf"), -0.1, 1.1])
    def test_quantile_probe_validation(self, bad):
        d = TDigest.from_array([0.0, 1.0, 2.0, 3.0], max_size=64, scale="k2")
        with pytest.raises(ValueError):
            d.quantile(bad)

    def test_merge_precision_mismatch_raises_clear_error(self):
        d64 = TDigest.from_array([0.0, 1.0, 2.0], max_size=64, scale="k2", precision="f64")
        d32 = TDigest.from_array([0.0, 1.0, 2.0], max_size=64, scale="k2", precision="f32")

        with pytest.raises(ValueError) as exc:
            d64.merge(d32)
        msg = str(exc.value).lower()
        assert "precision" in msg
        assert "cast" in msg

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_scale_rejects_invalid_factor(self, bad):
        d = TDigest.from_array([0.0, 1.0, 2.0], max_size=64, scale="k2")
        with pytest.raises(ValueError):
            d.scale_weights(bad)
        with pytest.raises(ValueError):
            d.scale_values(bad)

    def test_edges_policy_rules(self):
        with pytest.raises(ValueError):
            TDigest.from_array(
                [0.0, 1.0],
                max_size=64,
                scale="k2",
                singleton_policy=SingletonPolicy.EDGES,
            )

        with pytest.raises(ValueError):
            TDigest.from_array(
                [0.0, 1.0],
                max_size=64,
                scale="k2",
                singleton_policy=SingletonPolicy.USE,
                pin_per_side=2,
            )

    def test_weighted_add_rejects_invalid_inputs(self):
        d = TDigest.from_array([0.0, 1.0], max_size=64, scale="k2")
        with pytest.raises(ValueError):
            d.add_weighted([1.0, 2.0], [1.0])
        with pytest.raises(ValueError):
            d.add_weighted([1.0], [0.0])

    def test_delta_rejects_max_size_and_incompatible_options(self):
        with pytest.raises(ValueError, match="either max_size or delta"):
            TDigest.from_array([0.0, 1.0], max_size=64, delta=20.0)

        with pytest.raises(TypeError, match="specify max_size"):
            TDigest.from_array([0.0, 1.0], delta=20.0, scale="k3")

        with pytest.raises(TypeError, match="specify max_size"):
            TDigest.from_array([0.0, 1.0], delta=20.0, singleton_policy=SingletonPolicy.USE)
