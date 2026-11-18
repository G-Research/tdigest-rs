"""The upstream distribution owns version reporting for both public APIs."""

from importlib.metadata import version
import subprocess
import sys

from packaging.version import Version


def test_public_versions_match_distribution_and_native():
    import tdigest_rs
    import tdigest_rs._native as native

    from importlib.util import find_spec

    assert find_spec("gr_tdigest") is None
    expected = version("tdigest-rs")
    assert tdigest_rs.__version__ == expected
    assert Version(native.__version__) == Version(expected)


def test_unrelated_private_distribution_cannot_supply_public_version():
    code = """
import importlib.metadata as metadata
original = metadata.version
expected = original('tdigest-rs')
def lookup(name):
    if name == 'gr-tdigest':
        return '999.0.0'
    return original(name)
metadata.version = lookup
import tdigest_rs
assert tdigest_rs.__version__ == expected
"""
    subprocess.run([sys.executable, "-I", "-c", code], check=True)


def test_only_public_delta_builders_warn():
    code = """
import copy, pickle, warnings
import numpy as np
from tdigest_rs import TDigest, _native, tdigest_rs
assert tdigest_rs._TDigestInternal32 is _native._TDigestInternal32
assert tdigest_rs._TDigestInternal64 is _native._TDigestInternal64
assert TDigest.__module__ == 'tdigest_rs.lib'
expected = ("Building a TDigest with `delta`, or with neither `delta` nor `max_size`, is deprecated "
            "and will be removed in a future release. It runs the tdigest-rs 0.1.0 algorithm, which "
            "is less accurate than the `max_size` digest and is kept only temporarily so that users "
            "can migrate. Pass `max_size` instead, for example `TDigest.from_array(values, max_size=200)`. "
            "Quantiles from `max_size` are not identical to those from `delta`.")
def checked(build, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        d = build(*args, **kwargs)
    assert [(w.category, str(w.message), w.filename) for w in caught] == [(DeprecationWarning, expected, '<string>')]
    return d
values = np.array([0., 1., 2., 3.])
weights = np.ones(4, dtype=np.uint32)
d = checked(TDigest.from_array, values)
checked(TDigest.from_means_weights, values, weights)
checked(TDigest.from_means_weights, values, weights, 100)
copy.copy(d)
copy.deepcopy(d)
pickle.loads(pickle.dumps(d))
TDigest.from_means_weights(values, weights, max_size=64)
d.merge(d).median()
d.update(np.array([4., 5.])).quantile(0.5)
TDigest.from_dict(d.to_dict()).median()
delta32 = checked(TDigest.from_array, np.array([1., 2.], dtype=np.float32), delta=300)
assert delta32.means.dtype == np.float32
assert delta32.cast_precision("f64").merge(d).means.dtype == np.float64
capacity = TDigest.from_array(np.array([1, 2], dtype=np.float16), max_size=64)
assert isinstance(capacity, TDigest)
assert TDigest.from_dict(capacity.to_dict()).to_bytes() == capacity.to_bytes()
assert TDigest.from_bytes(capacity.to_bytes()).inner_kind() == "f64"
assert capacity.merge(capacity) is capacity
assert capacity.count == 4
empty = checked(TDigest.from_array, np.array([np.nan], dtype=np.float32))
import pickle
assert len(pickle.loads(pickle.dumps(empty))) == 0
"""
    subprocess.run([sys.executable, "-I", "-W", "error::DeprecationWarning", "-c", code], check=True)
