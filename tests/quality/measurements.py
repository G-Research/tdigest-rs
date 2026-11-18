"""Independent references and measurements shared by the quality tests.

Logical field bytes are a comparison budget, not an estimate of heap/RSS use.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
import csv
from dataclasses import dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
from tdigest_rs import TDigest

PROTOCOL = {
    "version": 1,
    "modern": {"scale": "k2", "singleton_policy": "use"},
    "budgets": {"fast": [64], "extended": [32, 128, 512]},
    "precisions": ["f32", "f64"],
    "paths": {"fast": ["bulk"], "extended": ["bulk", "left40", "balanced40"]},
    "orders": {"fast": ["natural"], "extended": ["natural", "reverse"]},
    "synthetic_rows": {"fast": 512, "extended": 8192},
    "real_rows": {"fast": "512 systematic rows over the complete timeline", "extended": "every source row"},
    "reference": "sorted observed input after precision casting; type7 quantile, ECDF rank interval, midpoint CDF",
    "budget": "modern centroid logical fields: mean4/8 + weight8 + kind1; original mean4/8 + weight4. No allocator, capacity or metadata claim.",
    "original_selection": "largest observed count within modern field-byte budget; count-only doubling/bisection, no accuracy selection",
    "comparison_eligible": "original occupies at least90% and at most100% of modern logical field bytes",
    "baseline_gate": "no relative deterioration allowance; rank metrics tolerate one observation, other metrics 2e-6(f32)/1e-12(f64); every metric separately",
    "probability_grid": "135 fixed probabilities: i/128 plus .0001,.001,.01,.99,.999,.9999",
}
Q = sorted(set([i / 128 for i in range(129)] + [0.0001, 0.001, 0.01, 0.99, 0.999, 0.9999]))
METRICS = ("rank_mean", "rank_max", "median_rank", "tail_rank_max", "value_iqr_mean", "value_iqr_max")
GATED = (*METRICS, "cdf_max", "cdf_mean")


def regression_limit(metric: str, frozen: float, n: int, precision: str) -> float:
    tolerance = 1 / n + 1e-12 if metric in METRICS[:4] else 2e-6 if precision == "f32" else 1e-12
    return frozen + tolerance


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    payload = path.read_bytes()
    return json.loads(gzip.decompress(payload) if path.suffix == ".gz" else payload)


def source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    paths = [
        root / "Cargo.toml",
        root / "Cargo.lock",
        root / "rust-toolchain.toml",
        root / "bindings/python/pyproject.toml",
        *sorted((root / "src").rglob("*.rs")),
        *sorted((root / "bindings/python/tdigest_rs").glob("*.py")),
    ]
    return {str(p.relative_to(root)): sha(p) for p in paths}


def verify_inputs_unchanged(receipt, native):
    if source_hashes() != receipt["source_inputs_sha256"] or sha(native) != receipt["native_sha256"]:
        raise RuntimeError("Source or installed native binary changed during quality measurements")


@dataclass
class Exact:
    values: list[float]

    def __post_init__(self):
        self.values = sorted(map(float, self.values))
        if not self.values or not all(map(math.isfinite, self.values)):
            raise ValueError("Reference requires finite nonempty data")

    def quantile(self, q: float) -> float:
        if not 0 <= q <= 1:
            raise ValueError("q outside [0,1]")
        position = q * (len(self.values) - 1)
        left = int(position)
        right = min(left + 1, len(self.values) - 1)
        fraction = position - left
        return self.values[left] * (1 - fraction) + self.values[right] * fraction

    def rank_interval(self, x: float) -> tuple[float, float]:
        n = len(self.values)
        return bisect_left(self.values, x) / n, bisect_right(self.values, x) / n

    def rank_error(self, q: float, x: float) -> float:
        lo, hi = self.rank_interval(x)
        return max(lo - q, q - hi, 0.0)

    def cdf(self, x: float) -> float:
        lo, hi = self.rank_interval(x)
        return (lo + hi) / 2

    @property
    def value_scale(self) -> float:
        iqr = self.quantile(0.75) - self.quantile(0.25)
        return iqr or (self.values[-1] - self.values[0]) or 1.0


def score(digest, exact: Exact) -> dict:
    predicted = [float(digest.quantile(q)) for q in Q]
    if not all(map(math.isfinite, predicted)):
        raise AssertionError("nonfinite quantile")
    tolerance = max(1.0, abs(exact.values[0]), abs(exact.values[-1])) * (
        2e-6 if digest.inner_kind() == "f32" else 2e-12
    )
    bounds = sum(x < exact.values[0] - tolerance or x > exact.values[-1] + tolerance for x in predicted)
    inversions = sum(a > b + tolerance for a, b in zip(predicted, predicted[1:]))
    if bounds or inversions:
        raise AssertionError("modern quantile bounds/monotonicity failure")
    ranks = [exact.rank_error(q, x) for q, x in zip(Q, predicted)]
    errors = [abs(x - exact.quantile(q)) for q, x in zip(Q, predicted)]
    tails = [(q, e) for q, e in zip(Q, ranks) if q <= 0.01 or q >= 0.99]
    result = dict(
        rank_mean=math.fsum(ranks) / len(ranks),
        rank_max=max(ranks),
        median_rank=ranks[Q.index(0.5)],
        tail_rank_max=max(e for _, e in tails),
        tail_relative_rank_max=max(e / max(min(q, 1 - q), 1 / len(exact.values)) for q, e in tails),
        value_abs_mean=math.fsum(errors) / len(errors),
        value_abs_max=max(errors),
        value_iqr_mean=math.fsum(errors) / len(errors) / exact.value_scale,
        value_iqr_max=max(errors) / exact.value_scale,
    )
    result.update(bounds_violations=bounds, monotonicity_violations=inversions)
    probes = sorted(set(exact.quantile(q) for q in Q))
    cdf = [float(digest.cdf(x)) for x in probes]
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in cdf):
        raise AssertionError("invalid CDF")
    if any(a > b + 1e-12 for a, b in zip(cdf, cdf[1:])):
        raise AssertionError("nonmonotone CDF")
    cdf_errors = [abs(v - exact.cdf(x)) for x, v in zip(probes, cdf)]
    result.update(
        cdf_mean=math.fsum(cdf_errors) / len(cdf_errors), cdf_max=max(cdf_errors), cdf_probe_count=len(probes)
    )
    return result


def datasets(directory: Path, mode: str):
    manifest = json.loads((directory / "manifest.json").read_text())
    for source in manifest:
        file = directory / source["derived_file"]
        if sha(file) != source["derived_sha256"]:
            raise RuntimeError("Pinned dataset checksum mismatch: " + file.name)
        with gzip.open(file, "rt", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != source["rows"]:
            raise RuntimeError("Pinned dataset row count mismatch")
        for column in source["columns"]:
            values = [float(row[column]) for row in rows]
            if mode == "fast":
                values = [values[i * (len(values) - 1) // 511] for i in range(512)]
            yield source["id"] + "/" + column, "real", values
    n = PROTOCOL["synthetic_rows"][mode]
    u = [(i + 0.5) / n for i in range(n)]
    normal = [NormalDist().inv_cdf(p) for p in u]
    cases = {
        "constant": [7.0] * n,
        "uniform_grid": list(map(float, range(n))),
        "gaussian_stratified": normal,
        "lognormal_stratified": [math.exp(x) for x in normal],
        "exponential_stratified": [-math.log1p(-p) for p in u],
        "pareto_stratified": [(1 - p) ** (-2 / 3) for p in u],
        "bimodal": [x + (-5 if i % 2 else 5) for i, x in enumerate(normal)],
        "rounded_ties": [round(x, 1) for x in normal],
        "zero_inflated": [0.0 if i < n * 0.98 else float(i * i) for i in range(n)],
        "skewed_two_point": [0.0 if i < n * 0.9 else 1.0 for i in range(n)],
        "drift": [float((i // max(1, n // 40)) ** 2) + (i % 7) / 7 for i in range(n)],
        "signed_scale": [(-1 if i % 2 else 1) * math.exp(x * 4) for i, x in enumerate(normal)],
    }
    for name, values in cases.items():
        yield "synthetic/" + name, "synthetic", values


def build(values, precision: str, budget: float, path: str):
    def raw(a):
        return TDigest.from_array(a, max_size=int(budget), precision=precision, **PROTOCOL["modern"])

    def merge(a, b):
        return TDigest.merge_all([a, b])

    if path == "bulk":
        return raw(values)
    parts = [raw(part) for part in np.array_split(values, min(40, len(values))) if len(part)]
    if path == "left40":
        result = parts[0]
        for part in parts[1:]:
            result = merge(result, part)
        return result
    while len(parts) > 1:
        parts = [merge(parts[i], parts[i + 1]) if i + 1 < len(parts) else parts[i] for i in range(0, len(parts), 2)]
    return parts[0]


def invariants(digest, values, precision: str):
    means = list(map(float, digest.means))
    weights = list(map(float, digest.weights))
    if not means or len(means) != len(weights) or not all(math.isfinite(x) for x in means):
        raise AssertionError("invalid centroids")
    inversions = sum(a > b for a, b in zip(means, means[1:]))
    if any(a >= b for a, b in zip(means, means[1:])):
        raise AssertionError("modern centroid means not strictly increasing")
    if not all(math.isfinite(w) and w > 0 for w in weights):
        raise AssertionError("invalid centroid weight")
    mass = math.fsum(weights)
    if abs(mass - len(values)) > len(values) * 1e-12:
        raise AssertionError(f"input mass lost: {mass} != {len(values)}")
    moment = math.fsum(m * w for m, w in zip(means, weights))
    expected = math.fsum(map(float, values))
    scale = max(1.0, math.fsum(abs(float(x)) for x in values))
    moment_relative_abs = abs(moment - expected) / scale
    moment_limit = 2e-6 if precision == "f32" else 2e-12
    if moment_relative_abs > moment_limit:
        raise AssertionError(f"weighted contribution drift: {moment_relative_abs}")
    if digest.inner_kind() != precision:
        raise AssertionError("wrong actual storage precision")
    restored = TDigest.from_bytes(digest.to_bytes())
    if restored.to_bytes() != digest.to_bytes():
        raise AssertionError("state roundtrip changed")
    return {
        "mass": mass,
        "moment_abs_error": abs(moment - expected),
        "moment_scaled_abs_error": moment_relative_abs,
        "moment_limit_exceeded": moment_relative_abs > moment_limit,
        "centroid_order_inversions": inversions,
    }


def atomic_json(path: Path, content):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def summarize(rows):
    groups = {}
    for kind in ("all", "real", "synthetic"):
        selected = [r for r in rows if kind == "all" or r["kind"] == kind]
        groups[kind] = {
            "profiles": len(selected),
            "errors": sum("error" in row for row in selected),
            "accuracy_limit_failures": sum(bool(row.get("accuracy_limits")) for row in selected),
        }
    return groups
