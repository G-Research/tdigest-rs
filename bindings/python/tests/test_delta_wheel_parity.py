"""Compare each public delta operation to the installed 0.1.0 wheel's recorded bits."""

import copy
import json
from pathlib import Path
import pickle
import struct

import numpy as np
import pytest

from tdigest_rs import TDigest

FIXTURE = Path(__file__).parent / "data" / "delta_wheel_parity.json"
DATASETS = ("tied", "zero_heavy", "signed_zero", "nonfinite", "tiny", "unsorted_weighted")
DELTAS = (7, 30, 300)
PROBES = (0, 0.001, 0.1, 0.25, 0.5, 0.75, 0.9, 0.999, 1)
TRIMS = ((0, 1), (0.1, 0.9), (0.49, 0.51), (0.5, 0.5))


def snapshot(d):
    return {
        "means": d.means.astype(d.means.dtype.newbyteorder("<"), copy=False).tobytes().hex(),
        "weights": d.weights.astype("<u4", copy=False).tobytes().hex(),
        "median": struct.pack(">d", d.median()).hex(),
        "quantiles": [struct.pack(">d", d.quantile(q)).hex() for q in PROBES],
        "trimmed_means": [struct.pack(">d", d.trimmed_mean(lo, hi)).hex() for lo, hi in TRIMS],
    }


def stages(dataset, dtype, delta):
    if dataset == "tied":
        values = np.arange(161) * 29 % 9 - 4
    elif dataset == "zero_heavy":
        values = [0] * 64 + [-1, 2, -3, 4, 1, 0, 2] * 5
    elif dataset == "signed_zero":
        values = [0.0, -0.0, -0.0, 1.0, -1.0, 0.0] * 7
    elif dataset == "nonfinite":
        values = [np.inf, np.nan, -np.inf, 2, -np.nan, 1, 0, np.inf, -3] * 3
    elif dataset == "tiny":
        t = np.finfo(dtype)
        values = [-t.tiny, t.smallest_subnormal, -0.0, -t.smallest_subnormal, 0.0, t.tiny] * 5
    else:
        values = [2, 0, 2, -0.0, -1, 0.0, 2, -2] * 4

    arr = np.array(values, dtype=dtype)
    weights = np.resize(np.array([1, 2, 0, 7, 4, 3, 1, 0, 2], dtype=np.uint32), arr.size)

    def build(part, mass):
        if dataset == "unsorted_weighted":
            return TDigest.from_means_weights(part, mass, delta)
        return TDigest.from_array(part, delta)

    d = build(arr, weights)
    yield "build", d
    shards = [build(arr[i::3], weights[i::3]) for i in range(3)]
    merged = shards[0].merge(shards[1], delta)
    yield "merge", merged
    merged = merged.merge(shards[2], 17)
    yield "merge_again", merged
    yield "reload", TDigest.from_means_weights(merged.means, merged.weights, delta)
    yield "pickle", pickle.loads(pickle.dumps(merged))
    yield "dict", TDigest.from_dict(merged.to_dict())
    yield "copy", copy.copy(merged)
    yield "deepcopy", copy.deepcopy(merged)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("delta", DELTAS)
@pytest.mark.parametrize("dataset", DATASETS)
def test_delta_matches_published_wheel_at_each_stage(dataset, delta, dtype):
    expected = json.loads(FIXTURE.read_text())["cases"][f"{dataset}-{dtype.__name__}-{delta}"]
    actual = dict(stages(dataset, dtype, delta))
    assert actual.keys() == expected.keys()
    for stage, digest in actual.items():
        assert digest.means.dtype == dtype
        assert digest.weights.dtype == np.uint32
        state = expected[stage]
        assert snapshot(digest) == (expected[state] if isinstance(state, str) else state), stage


def test_equal_merge_heads_continue_the_current_run():
    # Published 0.1.0 emits the right run's (1, 3) first because it supplied
    # the preceding zero. Its next weight, 4, then yields to the left's 3.
    left = TDigest.from_means_weights(np.array([1, 1, 2], dtype=np.float32), np.array([3, 1, 4], dtype=np.uint32))
    right = TDigest.from_means_weights(
        np.array([0, 1, 1, 2], dtype=np.float32), np.array([2, 3, 4, 5], dtype=np.uint32)
    )
    merged = left.merge(right)
    assert merged.means.tolist() == [0, 1, 1, 1, 1, 2, 2]
    assert merged.weights.tolist() == [2, 3, 3, 1, 4, 4, 5]
