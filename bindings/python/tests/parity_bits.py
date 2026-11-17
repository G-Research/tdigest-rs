"""Compare a floating-point result to bytes recorded by the numerical oracle."""

import numpy as np


def assert_float_bits(actual, expected_hex, dtype=">f8"):
    expected = np.frombuffer(bytes.fromhex(expected_hex), dtype=dtype)
    actual = np.asarray(actual, dtype=dtype).reshape(-1)
    # Arithmetic NaNs can have different signs and payloads on different CPUs.
    nan = np.isnan(expected)
    np.testing.assert_array_equal(np.isnan(actual), nan)
    bits = expected.dtype.str.replace("f", "u")
    np.testing.assert_array_equal(actual.view(bits)[~nan], expected.view(bits)[~nan])
