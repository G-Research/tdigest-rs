"""Time public Python calls on a fixed input. Run each wheel in its own interpreter."""

import argparse
import csv
from importlib.metadata import version
from statistics import median
import sys
from timeit import Timer

import numpy as np
from tdigest_rs import TDigest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("delta", "capacity"))
    mode = parser.parse_args().mode
    rng = np.random.default_rng(42)
    values = rng.permutation(np.arange(1_000_000, dtype=np.float64))
    options = {"delta": 300} if mode == "delta" else {"max_size": 146, "precision": "f64"}
    digest = TDigest.from_array(values, **options)
    assert len(digest) == 146
    assert digest.median() == 499_999.5 and digest.quantile(0.99) == 989_999.5
    context = {
        "TDigest": TDigest,
        "values": values,
        "options": options,
        "digest": digest,
    }
    cases = [
        ("build", "TDigest.from_array(values, **options)", 3),
        ("median", "digest.median()", 100_000),
        ("p99", "digest.quantile(0.99)", 100_000),
    ]
    if mode == "capacity":
        probes = rng.uniform(0, len(values), 65_536)
        ranks = digest.cdf(probes)
        assert ranks.shape == probes.shape and np.all((0 <= ranks) & (ranks <= 1))
        context["probes"] = probes
        cases.append(("cdf_65536", "digest.cdf(probes)", 10))
    else:
        other = TDigest.from_array(values + len(values), **options)
        assert len(other) == 146 and digest.merge(other).weights.sum() == 2 * len(values)
        context["other"] = other
        cases.append(("merge_pair", "digest.merge(other)", 5_000))

    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(
        [
            "version",
            "mode",
            "operation",
            "centroids",
            "median_ns",
            *[f"run_{i}_ns" for i in range(1, 8)],
        ]
    )
    for name, statement, iterations in cases:
        timer = Timer(statement, globals=context)
        timer.timeit(max(1, iterations // 10))
        samples = [round(seconds * 1e9 / iterations, 2) for seconds in timer.repeat(7, iterations)]
        writer.writerow([version("tdigest-rs"), mode, name, len(digest), median(samples), *samples])


if __name__ == "__main__":
    main()
