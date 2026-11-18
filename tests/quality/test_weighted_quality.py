"""Fast weighted probability controls; the reference uses exact rational mass."""

from fractions import Fraction
import math
import numpy as np
import pytest
from tdigest_rs import TDigest


@pytest.mark.parametrize("precision", ["f32", "f64"])
@pytest.mark.parametrize("policy", ["off", "use", "edges"])
@pytest.mark.parametrize("exponent", [-1070, -1020, 0, 500])
@pytest.mark.parametrize("pattern", ["binary_fraction", "decimal_fraction", "large_pile"])
def test_exact_weighted_cdf_and_state_across_mass_scales(precision, policy, exponent, pattern):
    values = np.array([-3.0, -1.0, 0.0, 0.0, 2.0, 7.0], dtype=np.float32 if precision == "f32" else np.float64)
    base = {
        "binary_fraction": [0.125, 0.25, 0.125, 0.125, 0.125, 0.25],
        "decimal_fraction": [0.1, 0.2, 0.3, 0.1, 0.2, 0.1],
        "large_pile": [1.0, 1.0, 1e6, 1e6, 1.0, 1.0],
    }[pattern]
    weights = [math.ldexp(x, exponent) for x in base]
    assert all(math.isfinite(w) and w > 0 for w in weights)
    digest = TDigest.from_means_weights(
        values,
        weights,
        max_size=128,
        precision=precision,
        singleton_policy=policy,
        **({"pin_per_side": 2} if policy == "edges" else {}),
    )
    rational = list(map(Fraction.from_float, weights))
    mass = sum(rational)
    assert Fraction.from_float(math.fsum(map(float, digest.weights))) == Fraction.from_float(float(mass))
    assert digest.inner_kind() == precision
    assert all(a < b for a, b in zip(digest.means, digest.means[1:]))
    probes = [-4.0, -3.0, -2.0, -1.0, -0.5, 0.0, 1.0, 2.0, 5.0, 7.0, 8.0]
    expected = []
    for x in probes:
        less = sum(w for v, w in zip(values, rational) if v < x)
        equal = sum(w for v, w in zip(values, rational) if v == x)
        expected.append(float((less + equal / 2) / mass))
    np.testing.assert_allclose(digest.cdf(probes), expected, rtol=2e-14, atol=2e-14)
    restored = TDigest.from_bytes(digest.to_bytes())
    assert restored.to_bytes() == digest.to_bytes()
    np.testing.assert_array_equal(restored.cdf(probes), digest.cdf(probes))
    qs = digest.quantile(np.linspace(0, 1, 65))
    assert np.all(np.isfinite(qs))
    assert np.all(np.diff(qs) >= 0)
    assert qs[0] == values.min() and qs[-1] == values.max()


@pytest.mark.parametrize("precision", ["f32", "f64"])
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_equal_fractional_weights_keep_symmetric_median(precision, sign):
    values = sign * np.arange(6, dtype=np.float32 if precision == "f32" else np.float64)
    digest = TDigest.from_means_weights(values, [0.1] * 6, max_size=128, precision=precision)
    assert digest.median() == pytest.approx(sign * 2.5, abs=1e-14)
