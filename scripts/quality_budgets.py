"""Diagnostic capacity/scale sweeps, separate from frozen acceptance references.

An error-free measurement may still FAIL numerical accuracy. New shapes are
additional regression probes, not independent trials or a universal guarantee.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from statistics import NormalDist
import sys

import numpy as np
import tdigest_rs._native as native

# Load only the explicit reference file, so -I can keep package imports isolated
# to the installed wheel without adding the checkout to sys.path.
reference_spec = importlib.util.spec_from_file_location(
    "_tdigest_quality_reference", Path(__file__).resolve().parents[1] / "tests/quality/measurements.py"
)
q = importlib.util.module_from_spec(reference_spec)
sys.modules[reference_spec.name] = q
reference_spec.loader.exec_module(q)


MATRICES = {
    "capacity": {"k2": [16, 32, 64, 128, 200, 512, 1000], "quad": [32, 200, 1000], "k1": [32, 200, 1000]},
    "additional": {"k2": [32, 200, 1000], "quad": [32, 200, 1000]},
}


def additional_shapes(n=8192):
    u = [(i + 0.5) / n for i in range(n)]
    z = [NormalDist().inv_cdf(x) for x in u]
    shapes = {f"separated_{gap}": [a + (-gap if i % 2 else gap) for i, a in enumerate(z)] for gap in [3, 12, 40]}
    shapes.update(
        uneven_three_clusters=[
            -20 + u[i] if i < int(n * 0.8) else u[i] if i < int(n * 0.95) else 20 + u[i] for i in range(n)
        ],
        rare_far_tail=[z[i] if i < int(n * 0.999) else 1e9 for i in range(n)],
        wide_signed_axis=[(-1 if i % 2 else 1) * 10 ** (60 * u[i] - 30) for i in range(n)],
        near_equal=[1 + 1e-12 * x for x in u],
        small_signed_values=[(2 * x - 1) * 1e-30 for x in u],
        sparse_repeated_levels=[float((i * 17) % 37) ** 3 for i in range(n)],
        alternating_drift=[float((i // 205) ** 2) + (-100 if i % 2 else 100) for i in range(n)],
        extreme_point_pile=[0.0 if i < int(n * 0.985) else 10 ** (20 * u[i]) for i in range(n)],
        folded_bell=[abs(a) for a in z],
    )
    # Fixed coprime permutation changes shard composition; no RNG-quality claim.
    return {name: [values[(i * 4093) % n] for i in range(n)] for name, values in shapes.items()}


def cases(root, cohort):
    if cohort == "capacity":
        yield from q.datasets(root / "tests/quality/data", "extended")
    else:
        for name, values in additional_shapes().items():
            yield name, "additional", values


def summarize(rows):
    result = {}
    for cohort, scale, cap in sorted({(r["cohort"], r["scale"], r["max_size"]) for r in rows}):
        selected = [r for r in rows if (r["cohort"], r["scale"], r["max_size"]) == (cohort, scale, cap)]
        scored = [r for r in selected if "modern" in r]
        result[f"{cohort}/{scale}/{cap}"] = {
            "profiles": len(selected),
            "errors": sum("error" in r for r in selected),
            "diagnostic_failed_profiles": sum(bool(r.get("diagnostic_failures")) for r in selected),
            "worst_errors": {metric: max((r["modern"][metric] for r in scored), default=None) for metric in q.GATED},
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="new report file; existing measurements are never overwritten"
    )
    parser.add_argument("--cohort", choices=["capacity", "additional", "all"], default="all")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; select a new output path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    module = Path(native.__file__)
    receipt = {
        "source_inputs_sha256": q.source_hashes(),
        "native_sha256": q.sha(module),
        "runner_sha256": q.sha(Path(__file__)),
        "reference_runner_sha256": q.sha(Path(q.__file__)),
        "data_manifest_sha256": q.sha(root / "tests/quality/data/manifest.json"),
        "python": sys.version,
        "numpy": np.__version__,
        "protocol": {
            "matrices": MATRICES,
            "cohort": args.cohort,
            "diagnostic_criteria": "rank_max and cdf_max thresholds: 0.1 at capacity <= 32, otherwise 0.05",
            "reference": "unchanged exact observed-data scoring/invariants from tests/quality/measurements.py",
            "counts": "profiles share data and are not independent statistical trials",
        },
        "input_status": "NOT_VERIFIED",
        "rows": [],
    }
    cohorts = MATRICES if args.cohort == "all" else [args.cohort]
    for cohort in cohorts:
        for name, kind, raw in cases(root, cohort):
            for precision in ["f32", "f64"]:
                values = np.array(raw, dtype=np.float32 if precision == "f32" else np.float64)
                exact = q.Exact(values.tolist())
                for scale, capacities in MATRICES[cohort].items():
                    q.PROTOCOL["modern"]["scale"] = scale
                    for cap in capacities:
                        for order in ["natural", "reverse"]:
                            data = values if order == "natural" else values[::-1].copy()
                            for route in ["bulk", "left40", "balanced40"]:
                                row = dict(
                                    id=f"{cohort}/{name}/{precision}/{scale}/{cap}/{order}/{route}",
                                    cohort=cohort,
                                    dataset=name,
                                    kind=kind,
                                    precision=precision,
                                    scale=scale,
                                    max_size=cap,
                                    order=order,
                                    path=route,
                                    n=len(data),
                                    input_sha256=hashlib.sha256(data.tobytes()).hexdigest(),
                                )
                                try:
                                    digest = q.build(data, precision, cap, route)
                                    row["modern"] = q.score(digest, exact)
                                    row["modern_state"] = dict(
                                        centroids=len(digest),
                                        field_bytes=len(digest) * (data.dtype.itemsize + 9),
                                        wire_bytes=len(digest.to_bytes()),
                                        **q.invariants(digest, data, precision),
                                    )
                                    bound = 0.1 if cap <= 32 else 0.05
                                    row["diagnostic_failures"] = [
                                        m for m in ["rank_max", "cdf_max"] if row["modern"][m] > bound
                                    ]
                                except Exception as error:
                                    row["error"] = f"{type(error).__name__}: {error}"
                                receipt["rows"].append(row)
            q.atomic_json(args.output, receipt)
            print(cohort, name, "completed", flush=True)
    try:
        q.verify_inputs_unchanged(receipt, module)
    except RuntimeError:
        receipt["input_status"] = "INVALID_INPUTS_CHANGED"
        q.atomic_json(args.output, receipt)
        raise
    receipt["input_status"] = "UNCHANGED"
    receipt["summary"] = summarize(receipt["rows"])
    failed = any("error" in r or r.get("diagnostic_failures") for r in receipt["rows"])
    receipt["accuracy_status"] = "FAIL" if failed else "PASS"
    receipt["measurement_status"] = "COMPLETED"
    q.atomic_json(args.output, receipt)
    print(json.dumps({"accuracy_status": receipt["accuracy_status"], "profiles": len(receipt["rows"])}))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
