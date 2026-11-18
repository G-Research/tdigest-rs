"""Score an installed wheel on the nine recorded columns used in the quality suite."""

import argparse
import csv
from pathlib import Path
import statistics
import sys

import numpy as np
from tdigest_rs import TDigest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/quality"))
from measurements import Q, Exact, build, datasets  # noqa: E402


def delta_digest(values, route, delta=300):
    if route == "bulk":
        return TDigest.from_array(values, delta=delta)
    parts = [TDigest.from_array(chunk, delta=delta) for chunk in np.array_split(values, 40)]
    if route == "left40":
        result = parts[0]
        for other in parts[1:]:
            result = result.merge(other, delta=delta)
        return result
    while len(parts) > 1:
        parts = [
            parts[i].merge(parts[i + 1], delta=delta) if i + 1 < len(parts) else parts[i]
            for i in range(0, len(parts), 2)
        ]
    return parts[0]


def match_delta(values, route, target):
    # Select by final centroid count before looking at any query results.
    low, high = 0.0, 300.0
    best = delta_digest(values, route, high)
    best_delta = high
    if len(best) < target:
        raise ValueError("The initial delta must bracket the target count")
    for _ in range(24):
        if len(best) == target:
            return best, best_delta
        delta = (low + high) / 2
        candidate = delta_digest(values, route, delta)
        count = len(candidate)
        if (abs(count - target), -count) < (abs(len(best) - target), -len(best)):
            best, best_delta = candidate, delta
        if count < target:
            low = delta
        else:
            high = delta
    # Repeated merges can make counts skip or reverse near a boundary.
    if len(best) == target:
        return best, best_delta
    for step in range(1, 129):
        for delta in (best_delta - step / 8, best_delta + step / 8):
            if delta > 0:
                candidate = delta_digest(values, route, delta)
                if len(candidate) == target:
                    return candidate, delta
    raise ValueError(f"No exact match for {target} centroids on {route}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("delta", "capacity"))
    parser.add_argument(
        "--max-size",
        type=int,
        help="Centroid budget for capacity mode (default: 1000).",
    )
    parser.add_argument(
        "--match-counts",
        type=Path,
        help="Choose delta to match the final counts in a capacity run.",
    )
    states = parser.add_mutually_exclusive_group()
    states.add_argument(
        "--save-centroids",
        type=Path,
        help="Save each delta digest's means and weights for replay.",
    )
    states.add_argument(
        "--query-centroids",
        type=Path,
        help="Query saved centroids with the installed delta API.",
    )
    args = parser.parse_args()
    if (args.save_centroids or args.query_centroids) and args.mode != "delta":
        parser.error("Centroid replay compares delta queries.")
    if args.match_counts and (args.mode != "delta" or args.save_centroids or args.query_centroids):
        parser.error("Use --match-counts with a fresh delta build.")
    if args.max_size is not None and args.mode != "capacity":
        parser.error("--max-size requires capacity mode.")
    report_size = args.max_size is not None or args.match_counts is not None
    budgets = {}
    if args.match_counts:
        with args.match_counts.open(newline="") as source:
            for row in csv.DictReader(source):
                key = (row["column"], row["precision"], row["order"], row["route"])
                budgets[key] = int(row["centroids"])
    if args.save_centroids:
        args.save_centroids.mkdir(parents=True, exist_ok=True)
    writer = csv.writer(sys.stdout)
    writer.writerow(
        [
            "column",
            "precision",
            "order",
            "route",
            "centroids",
            "mean_rank_error",
            "tail_rank_error",
            "mean_value_error",
        ]
        + (["sizing_argument"] if report_size else [])
    )
    for column, kind, observations in datasets(ROOT / "tests/quality/data", "extended"):
        if kind != "real":
            continue
        for precision, dtype in (("f32", np.float32), ("f64", np.float64)):
            values = np.array(observations, dtype=dtype)
            exact = Exact(values.tolist())
            for order, data in (("forward", values), ("reverse", values[::-1].copy())):
                for route in ("bulk", "left40", "balanced40"):
                    name = f"{column.replace('/', '_')}-{precision}-{order}-{route}.npz"
                    size = 300
                    if args.query_centroids:
                        with np.load(args.query_centroids / name) as saved:
                            means, weights = saved["means"], saved["weights"]
                        # Keep the supplied centroids separate, then verify the reconstructed state.
                        size = 400 * len(means)
                        digest = TDigest.from_means_weights(means, weights, delta=size)
                        if digest.means.tobytes() != means.tobytes() or digest.weights.tobytes() != weights.tobytes():
                            raise ValueError(f"Reconstruction changed the saved centroids in {name}")
                    elif args.mode == "capacity":
                        size = args.max_size if args.max_size is not None else 1000
                        digest = build(data, precision, size, route)
                    elif args.match_counts:
                        digest, size = match_delta(data, route, budgets[column, precision, order, route])
                    else:
                        digest = delta_digest(data, route)
                    if args.save_centroids:
                        np.savez(
                            args.save_centroids / name,
                            means=digest.means,
                            weights=digest.weights,
                        )
                    estimates = [digest.quantile(q) for q in Q]
                    errors = [exact.rank_error(q, x) for q, x in zip(Q, estimates)]
                    tail = max(e for q, e in zip(Q, errors) if q <= 0.01 or q >= 0.99)
                    value_error = statistics.mean(abs(exact.quantile(q) - x) for q, x in zip(Q, estimates))
                    writer.writerow(
                        [
                            column,
                            precision,
                            order,
                            route,
                            len(digest),
                            statistics.mean(errors),
                            tail,
                            value_error,
                        ]
                        + ([size] if report_size else [])
                    )


if __name__ == "__main__":
    main()
