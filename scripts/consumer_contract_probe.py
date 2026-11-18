"""Check the public consumer APIs and record each probe result."""

import argparse
import copy
import hashlib
import importlib
import json
from pathlib import Path
import pickle
import sys

import tdigest_rs as gr
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--require-clean", action="store_true")
    args = parser.parse_args()
    rows = []

    def check(name, operation):
        try:
            operation()
            rows.append({"id": name, "status": "PASS"})
        except Exception as error:
            rows.append({"id": name, "status": "FAIL", "error": f"{type(error).__name__}: {error}"})

    def equal(actual, expected):
        assert actual == expected, (actual, expected)

    def reject(operation):
        try:
            operation()
        except (TypeError, ValueError):
            return
        raise AssertionError("invalid input was accepted")

    def digest(values=(0.0, 10.0), **kwargs):
        return gr.TDigest.from_array(values, max_size=200, **kwargs)

    weighted = gr.TDigest.from_means_weights([0.0, 10.0], [9.0, 1.0], max_size=200)
    check("weighted_90_10_median", lambda: equal(weighted.median(), 0.0))
    check("median_is_quantile", lambda: equal(weighted.median(), weighted.quantile(0.5)))
    check("weighted_from_array_argument", lambda: equal(digest(weights=[9.0, 1.0]).median(), 0.0))
    check("misspelled_argument_rejected", lambda: reject(lambda: digest(weigths=[9.0, 1.0])))
    check("single_mass_midpoint", lambda: equal(digest([3.0]).cdf([2.0, 3.0, 4.0]), [0.0, 0.5, 1.0]))
    check("quantile_endpoints", lambda: equal(weighted.quantile([0.0, 1.0]), [0.0, 10.0]))
    check("nan_training_rejected", lambda: reject(lambda: digest([np.nan])))
    check("inf_training_rejected", lambda: reject(lambda: digest([np.inf])))
    check("nan_quantile_rejected", lambda: reject(lambda: weighted.quantile(np.nan)))
    check("nan_cdf", lambda: equal(np.isnan(weighted.cdf(np.nan)), True))
    check("infinite_cdf", lambda: equal(weighted.cdf([-np.inf, np.inf]), [0.0, 1.0]))
    check("integer_cdf_floating", lambda: equal(digest().cdf(np.array([5], dtype=np.int64)).tolist(), [0.5]))
    check("tuple_query", lambda: equal(digest().cdf((5.0,)), (0.5,)))
    check("modern_copy", lambda: equal(copy.copy(weighted).to_bytes(), weighted.to_bytes()))
    check("modern_pickle", lambda: equal(pickle.loads(pickle.dumps(weighted)).to_bytes(), weighted.to_bytes()))
    check("wire_roundtrip", lambda: equal(gr.TDigest.from_bytes(weighted.to_bytes()).to_bytes(), weighted.to_bytes()))
    check(
        "fractional_v1_rejected",
        lambda: reject(lambda: gr.TDigest.from_means_weights([1.0], [0.5], max_size=1000).to_bytes(version=1)),
    )
    check("missing_merge_rejected", lambda: reject(lambda: gr.TDigest.merge_all([weighted, None])))
    check(
        "mixed_capacity_rejected",
        lambda: reject(lambda: gr.TDigest.merge_all([weighted, gr.TDigest.from_array([1.0], max_size=1000)])),
    )

    def multi_merge_atomic():
        receiver = digest([0.0])
        before = receiver.to_bytes()
        reject(lambda: receiver.merge([digest([10.0]), "invalid"]))
        equal(receiver.to_bytes(), before)

    check("multi_merge_failure_atomic", multi_merge_atomic)

    def scale_atomic():
        receiver = copy.copy(weighted)
        before = receiver.to_bytes()
        reject(lambda: receiver.scale_weights(float("inf")))
        equal(receiver.to_bytes(), before)

    check("scale_failure_atomic", scale_atomic)

    def f32_mass():
        d = gr.TDigest.from_means_weights([1.0], [16777216.0], max_size=1000, precision="f32")
        d.add_weighted([1.0], [1.0])
        equal(sum(d.weights), 16777217.0)

    check("f32_means_f64_mass", f32_mass)
    native = Path(importlib.import_module("tdigest_rs._native").__file__)
    result = {
        "source_sha": args.source_sha,
        "python": sys.version,
        "numpy": np.__version__,
        "distribution_version": gr.__version__,
        "engine": "modern tdigest_rs",
        "module": gr.__file__,
        "native_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
        "scope": "bounded Python probes only; no Polars/JVM/ABI/cluster qualification",
        "rows": rows,
        "passed": sum(row["status"] == "PASS" for row in rows),
        "failed": sum(row["status"] == "FAIL" for row in rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("source_sha", "passed", "failed")}))
    if args.require_clean and result["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
