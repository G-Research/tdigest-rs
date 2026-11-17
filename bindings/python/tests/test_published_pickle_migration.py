"""Read real published tdigest-rs 0.1.0 pickles and compare with their published reload results."""

import base64
import hashlib
import json
from pathlib import Path
import pickle
import struct


import pytest


FIXTURE = json.loads((Path(__file__).parent / "data" / "published_0_1_0_pickle_golden.json").read_text())


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda case: case["id"])
def test_published_0_1_0_pickle_matches_published_reload(case):
    # Only unpickle the checked-in, hash-verified synthetic release fixtures.
    payload = base64.b64decode(case["pickle_base64"], validate=True)
    assert hashlib.sha256(payload).hexdigest() == case["pickle_sha256"]
    digest = pickle.loads(payload)
    state = case["expected_old_restored"]
    assert type(digest).__module__ == state["class_module"]
    assert type(digest).__qualname__ == state["class_name"]
    assert digest.means.dtype.str == state["means_dtype"]
    assert digest.weights.dtype.str == state["weights_dtype"]
    assert list(digest.means.shape) == state["means_shape"]
    assert len(digest) == state["len"]
    assert digest.means.tobytes().hex() == state["means_hex"]
    assert digest.weights.tobytes().hex() == state["weights_hex"]
    assert struct.pack(">d", digest.median()).hex() == state["median_f64_bits"]
    assert [struct.pack(">d", digest.quantile(q)).hex() for q in FIXTURE["quantile_probes"]] == state[
        "quantiles_f64_bits"
    ]
    assert [struct.pack(">d", digest.trimmed_mean(lo, hi)).hex() for lo, hi in FIXTURE["trim_ranges"]] == state[
        "trimmed_mean_f64_bits"
    ]
