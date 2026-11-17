"""Compare public delta operations with the original Rust numerical fixture."""

import copy
import json
import struct
from pathlib import Path
import pickle

import numpy as np
import pytest

from tdigest_rs import TDigest

REFERENCE = json.loads((Path(__file__).parent / "data/upstream_numerical_golden.json").read_text())


def dataset(name, dtype):
    tiny = np.finfo(dtype).tiny
    fixed = {
        "singleton": [3.0],
        "two": [1.0, 2.0],
        "ties": [3.0, 3.0, 1.0, 0.0, 0.0, -0.0],
        "nonfinite": [-np.inf, -2.0, 0.0, np.inf, np.nan],
        "allnan": [np.nan, np.nan],
        "tiny": [-tiny, -0.0, 0.0, tiny],
        "weighted": [0.0, 10.0],
        "large-weight": [1.0],
        "uint32-max": [1.0],
        "uneven": [-10.0, 0.0, 1.0, 3.0, 100.0],
    }
    if name == "linear":
        return np.arange(1000, dtype=dtype) / dtype(10)
    if name == "compressed":
        return ((np.arange(1000) * 37 % 1000).astype(dtype) - dtype(500)) / dtype(10)
    return np.array(fixed[name], dtype=dtype)


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_upstream_numerics_bit_for_bit(case):
    dtype = np.dtype(case["dtype"]).type
    params = case["input"]
    values = dataset(params["dataset"], dtype)
    op = params["op"]
    if op == "from_means_weights":
        digest = TDigest.from_means_weights(values, np.array(params["weights"], dtype=np.uint32), params["delta"])
    elif op in ("merge", "update"):
        split = params["split"]
        left = TDigest.from_array(values[:split], params["delta"])
        before = left.means.copy(), left.weights.copy()
        if op == "merge":
            right = TDigest.from_array(values[split:], params["other_delta"])
            digest = left.merge(right, params["merge_delta"])
        else:
            digest = left.update(values[split:], params["other_delta"], params["merge_delta"])
        np.testing.assert_array_equal(left.means, before[0])
        np.testing.assert_array_equal(left.weights, before[1])
    else:
        digest = TDigest.from_array(values, params["delta"])
        if op == "copy":
            digest = copy.copy(digest)
        elif op == "deepcopy":
            digest = copy.deepcopy(digest)
        elif op == "pickle":
            digest = pickle.loads(pickle.dumps(digest))
        elif op == "dict":
            digest = TDigest.from_dict(digest.to_dict())
        else:
            assert op == "from_array"
    assert digest.means.dtype == dtype
    assert digest.weights.dtype == np.uint32
    assert [struct.pack(">d", m).hex() for m in digest.means] == case["means"]
    assert digest.weights.tolist() == case["weights"]
    assert struct.pack(">d", digest.median()).hex() == case["median"]
    for q_bits, expected in case["quantiles"]:
        (q,) = struct.unpack(">d", bytes.fromhex(q_bits))
        assert struct.pack(">d", digest.quantile(q)).hex() == expected
    for lo, hi, expected in case["trimmed_means"]:
        assert struct.pack(">d", digest.trimmed_mean(lo, hi)).hex() == expected
