"""Generate real-release pickle fixtures using an isolated tdigest-rs 0.1.0 install.

Usage: python -I generate.py --wheel ORIGINAL.whl --output FIXTURE.json
Use Python 3.12.12 and NumPy 1.26.4 for the recorded fixture. No source build runs.
"""

import argparse
import base64
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle
import platform
import struct
import sys
import zipfile

import numpy as np
import tdigest_rs
import tdigest_rs.tdigest_rs as native
from tdigest_rs import TDigest

WHEEL_NAME = "tdigest_rs-0.1.0-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
WHEEL_SHA256 = "62ee8d5a01226d020a2ab11a956292c38beab3f34d2d9fa7a8fca22c3e5d4bbf"
WHEEL_URL = (
    "https://files.pythonhosted.org/packages/89/3b/faabc31e51eeae07efe1ad55d1e92daa495a6dddea99371a6896b644e93a/"
    + WHEEL_NAME
)
PROBES = [0.0, 0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 1.0]
TRIMS = [(0.0, 1.0), (0.1, 0.9), (0.25, 0.75)]


def digest_snapshot(digest):
    def bits(value):
        return struct.pack(">d", float(value)).hex()

    return {
        "class_module": type(digest).__module__,
        "class_name": type(digest).__qualname__,
        "means_dtype": digest.means.dtype.str,
        "means_shape": list(digest.means.shape),
        "means_hex": digest.means.tobytes().hex(),
        "weights_dtype": digest.weights.dtype.str,
        "weights_hex": digest.weights.tobytes().hex(),
        "len": len(digest),
        "median_f64_bits": bits(digest.median()),
        "quantiles_f64_bits": [bits(digest.quantile(q)) for q in PROBES],
        "trimmed_mean_f64_bits": [bits(digest.trimmed_mean(a, b)) for a, b in TRIMS],
    }


def digest_cases():
    for dtype in [np.float32, np.float64]:
        precision = np.dtype(dtype).name
        yield (
            precision + "_tiny_default",
            {"operation": "from_array", "delta": 300},
            TDigest.from_array(np.array([0.0, 1.0], dtype=dtype)),
        )
        yield (
            precision + "_weighted_piles",
            {"operation": "from_means_weights", "delta": 300},
            TDigest.from_means_weights(
                np.array([0.0, 10.0], dtype=dtype), np.array([90, 10], dtype=np.uint32)
            ),
        )
        yield (
            precision + "_large_u32_weights",
            {"operation": "from_means_weights", "delta": 300},
            TDigest.from_means_weights(
                np.array([-1.0, 0.0, 1.0], dtype=dtype),
                np.array([2**24 + 1, 17, 2**24 + 3], dtype=np.uint32),
            ),
        )
        values = (((np.arange(2048, dtype=np.int64) * 7919) % 4093) - 2046).astype(
            dtype
        ) / dtype(16)
        for delta in [8.0, 30.0, 300.0, 5000.0]:
            yield (
                precision + f"_compressed_delta_{int(delta)}",
                {"operation": "from_array", "delta": delta, "input_count": len(values)},
                TDigest.from_array(values, delta=delta),
            )
        shards = [TDigest.from_array(values[i::3], delta=17.0) for i in range(3)]
        merged = shards[0].merge(shards[1], delta=11.0).merge(shards[2], delta=23.0)
        yield (
            precision + "_merge_distinct_deltas",
            {
                "operation": "merge",
                "shard_delta": 17,
                "merge_deltas": [11, 23],
                "input_count": len(values),
            },
            merged,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    def sha(data):
        return hashlib.sha256(data).hexdigest()

    assert sys.flags.isolated, "Run with python -I to avoid checkout imports."
    assert importlib.metadata.version("tdigest-rs") == "0.1.0"
    assert np.__version__ == "1.26.4"
    assert sha(args.wheel.read_bytes()) == WHEEL_SHA256
    assert (
        Path(tdigest_rs.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    )
    assert Path(native.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    with zipfile.ZipFile(args.wheel) as archive:
        for member, installed in [
            ("tdigest_rs/lib.py", Path(tdigest_rs.__file__).with_name("lib.py")),
            ("tdigest_rs/tdigest_rs.abi3.so", Path(native.__file__)),
        ]:
            assert archive.read(member) == installed.read_bytes(), (
                f"Installed package differs from verified wheel: {member}"
            )
    cases = []
    for name, recipe, digest in digest_cases():
        original = digest_snapshot(digest)
        for protocol in [4, 5]:
            payload = pickle.dumps(digest, protocol=protocol)
            expected = digest_snapshot(pickle.loads(payload))
            cases.append(
                {
                    "id": name + f"_protocol{protocol}",
                    "recipe": recipe,
                    "protocol": protocol,
                    "pickle_sha256": sha(payload),
                    "pickle_base64": base64.b64encode(payload).decode("ascii"),
                    "expected_old_restored": expected,
                    "old_pickle_changed_fields": [
                        key for key in original if original[key] != expected[key]
                    ],
                    "original_summary": {
                        "centroids": original["len"],
                        "median_f64_bits": original["median_f64_bits"],
                        "quantiles_f64_bits": original["quantiles_f64_bits"],
                    },
                }
            )
    fixture = {
        "schema_version": 1,
        "provenance": {
            "distribution": "tdigest-rs",
            "version": "0.1.0",
            "wheel_filename": WHEEL_NAME,
            "wheel_sha256": WHEEL_SHA256,
            "wheel_url": WHEEL_URL,
            "metadata_url": "https://pypi.org/pypi/tdigest-rs/0.1.0/json",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "public_class": "tdigest_rs.lib.TDigest",
            "native_module": "tdigest_rs.tdigest_rs",
            "native_wheel_member": "tdigest_rs/tdigest_rs.abi3.so",
            "native_sha256": sha(Path(native.__file__).read_bytes()),
            "wrapper_wheel_member": "tdigest_rs/lib.py",
            "wrapper_sha256": sha(
                Path(tdigest_rs.__file__).with_name("lib.py").read_bytes()
            ),
            "generator_sha256": sha(Path(__file__).read_bytes()),
            "source_pin": "8d9cf1fe82c5c9254e5dc9593f01717e317f4333",
            "source_pin_version": "0.2.0",
            "scope": "The genuine published 0.1.0 wheel is distinct from the pinned, unpublished 0.2.0 source.",
        },
        "quantile_probes": PROBES,
        "trim_ranges": TRIMS,
        "cases": cases,
        "expected_behavior": "Compare with old-package pickle.loads results; original pickle reconstruction uses default delta=300 and can itself recompress stored centroids.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2) + "\n")
    print(
        json.dumps(
            {
                "cases": len(cases),
                "output": str(args.output),
                "sha256": sha(args.output.read_bytes()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
