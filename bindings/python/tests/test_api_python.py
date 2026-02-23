import math

import numpy as np
import pytest

from tdigest_rs import ScaleFamily, SingletonPolicy, TDigest


def _old_log_scale(q: float, delta: float, n: int) -> float:
    if q <= 0.0:
        return float("-inf")
    if q >= 1.0:
        return float("inf")
    factor = delta / (4.0 * math.log(n / delta) + 24.0)
    return factor * math.log(q / (1.0 - q))


def _old_inverse_log_scale(k: float, delta: float, n: int) -> float:
    if math.isinf(k):
        return 0.0 if k < 0.0 else 1.0
    factor = (4.0 * math.log(n / delta) + 24.0) / delta
    return 1.0 / (1.0 + math.exp(-k * factor))


def _old_log_q_limit(q0: float, delta: float, n: int) -> float:
    return _old_inverse_log_scale(_old_log_scale(q0, delta, n) + 1.0, delta, n)


def _old_cluster_count(values: np.ndarray, delta: float) -> int:
    values = np.sort(values.astype(np.float64, copy=False))
    n = len(values)
    if n == 0:
        return 0

    weights = np.ones(n, dtype=np.float64)
    return _old_cluster_count_weighted(values, weights, delta)


def _old_cluster_count_weighted(means: np.ndarray, weights: np.ndarray, delta: float) -> int:
    means = np.asarray(means, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    order = np.argsort(means, kind="mergesort")
    means = means[order]
    weights = weights[order]

    n = len(means)
    if n == 0:
        return 0

    total_weight = float(np.sum(weights))
    cumulative_weight = 0.0
    sigma_weight = float(weights[0])
    q_limit = _old_log_q_limit(0.0, delta, n)
    count = 0

    for w in weights[1:]:
        w = float(w)
        q = (cumulative_weight + sigma_weight + w) / total_weight
        if q <= q_limit:
            sigma_weight += w
        else:
            count += 1
            cumulative_weight += sigma_weight
            q_limit = _old_log_q_limit(cumulative_weight / total_weight, delta, n)
            sigma_weight = w

    count += 1
    return count


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
        if isinstance(as_list, np.ndarray):
            as_list = as_list.tolist()
        assert isinstance(as_list, list)
        assert as_list == pytest.approx([0.125, 0.5, 0.875], abs=1e-9)

        as_np = d.cdf(np.array([0.0, 1.5, 3.0], dtype=float))
        if not isinstance(as_np, np.ndarray):
            as_np = np.asarray(as_np)
        assert as_np.shape == (3,)
        assert list(as_np) == pytest.approx([0.125, 0.5, 0.875], abs=1e-9)

    def test_merge_and_merge_all_smoke(self):
        a = TDigest.from_array([0.0, 1.0, 2.0, 3.0], max_size=64, scale="k2")
        b = TDigest.from_array([10.0, 11.0, 12.0, 13.0], max_size=64, scale="k2")

        before_a = a.quantile(0.5)
        out = a.merge(b)
        assert out is not a
        assert a.quantile(0.5) == pytest.approx(before_a, abs=1e-9)
        assert out.quantile(0.5) > before_a

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

    def test_from_means_weights_update_trimmed_mean_and_dict_compat(self):
        d = TDigest.from_means_weights(
            arr=np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
            weights=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64),
            max_size=64,
            scale="k2",
        )
        assert len(d) > 0
        assert isinstance(d.means, np.ndarray)
        assert isinstance(d.weights, np.ndarray)
        assert d.weights.dtype == np.float64

        tm = d.trimmed_mean(0.1, 0.9)
        assert math.isfinite(tm)

        d2 = d.update([4.0, 5.0])
        assert d2 is not d
        assert d2.quantile(0.5) >= d.quantile(0.5)

        payload = d2.to_dict()
        d3 = TDigest.from_dict(payload)
        assert d3.quantile(0.5) == pytest.approx(d2.quantile(0.5), abs=1e-9)

        legacy = {"means": [0.0, 1.0, 2.0], "weights": [1.0, 2.0, 3.0]}
        d4 = TDigest.from_dict(legacy)
        assert math.isfinite(d4.quantile(0.5))


class TestPythonApiValidation:
    @pytest.mark.parametrize("bad", [float("nan"), float("+inf"), float("-inf")])
    def test_rejects_non_finite_training_and_add(self, bad):
        with pytest.raises(ValueError):
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

    def test_trimmed_mean_probe_validation(self):
        d = TDigest.from_array([0.0, 1.0, 2.0, 3.0], max_size=64, scale="k2")
        with pytest.raises(ValueError):
            d.trimmed_mean(float("nan"), 0.9)
        with pytest.raises(ValueError):
            d.trimmed_mean(0.8, 0.2)

    def test_delta_constructor_matches_old_k2_cluster_count(self):
        rng = np.random.default_rng(7)
        data = rng.normal(0.0, 1.0, size=5000).astype(np.float64)

        for delta in [100.0, 171.0, 300.0]:
            d = TDigest.from_array(data, delta=delta)
            assert len(d) == _old_cluster_count(data, delta)

    def test_delta_constructor_positional_argument_is_supported(self):
        data = np.linspace(-3.0, 3.0, 5000, dtype=np.float64)
        d_pos = TDigest.from_array(data, 171.0)
        d_kw = TDigest.from_array(data, delta=171.0)
        assert len(d_pos) == len(d_kw)
        assert d_pos.quantile(0.5) == pytest.approx(d_kw.quantile(0.5), abs=1e-12)

    def test_delta_from_means_weights_matches_old_k2_cluster_count(self):
        means = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        weights = np.array([100.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)

        for delta in [20.0, 100.0, 171.0]:
            d = TDigest.from_means_weights(means, weights, delta=delta)
            assert len(d) == _old_cluster_count_weighted(means, weights, delta)

    def test_delta_from_means_weights_positional_argument_is_supported(self):
        means = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
        weights = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        d_pos = TDigest.from_means_weights(means, weights, 171.0)
        d_kw = TDigest.from_means_weights(means, weights, delta=171.0)
        assert len(d_pos) == len(d_kw)
        assert d_pos.quantile(0.5) == pytest.approx(d_kw.quantile(0.5), abs=1e-12)

    def test_max_size_and_delta_are_mutually_exclusive(self):
        with pytest.raises(ValueError, match="either max_size or delta"):
            TDigest.from_array([0.0, 1.0], max_size=64, delta=100.0)
        with pytest.raises(ValueError, match="either max_size or delta"):
            TDigest.from_means_weights([0.0], [1.0], max_size=64, delta=100.0)

    def test_delta_mode_rejects_non_k2_scale(self):
        with pytest.raises(ValueError, match="scale='k2'"):
            TDigest.from_array([0.0, 1.0], delta=100.0, scale="k3")

    def test_delta_mode_rejects_non_off_singleton_policy(self):
        with pytest.raises(ValueError, match="singleton_policy='off'"):
            TDigest.from_array([0.0, 1.0], delta=100.0, singleton_policy="use")

    def test_delta_mode_defaults_singleton_policy_to_off(self):
        data = np.array([0.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float64)
        d_default = TDigest.from_array(data, delta=100.0)
        d_off = TDigest.from_array(data, delta=100.0, singleton_policy="off")
        assert len(d_default) == len(d_off)
        assert d_default.quantile(0.5) == pytest.approx(d_off.quantile(0.5), abs=1e-12)

    def test_default_constructor_uses_max_size_100(self):
        data = np.arange(1000, dtype=np.float64)
        d_default = TDigest.from_array(data, scale="k2")
        d_100 = TDigest.from_array(data, max_size=100, scale="k2")
        assert len(d_default) == len(d_100)
        assert d_default.quantile(0.5) == pytest.approx(d_100.quantile(0.5), abs=1e-12)

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_delta_validation(self, bad):
        with pytest.raises(ValueError):
            TDigest.from_array([0.0, 1.0], delta=bad)

    def test_update_delta_arguments_are_rejected(self):
        d = TDigest.from_array([0.0, 1.0], max_size=64, scale="k2")
        with pytest.raises(ValueError):
            d.update([2.0], delta=10.0)
