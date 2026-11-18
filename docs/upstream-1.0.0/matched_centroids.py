"""Compare installed tdigest-rs wheels at equal centroid counts or with identical centroids."""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import statistics
import time

from joblib import Parallel, delayed
import numpy as np
from tdigest_rs import TDigest


def build(row):
    return TDigest.from_array(row, delta=10_000)


def state(digest):
    return digest.means.tobytes(), digest.weights.tobytes()


def prepare(directory, seed, count):
    # The published wheel supplies the input centroid means and weights to both wheels.
    if version("tdigest-rs") != "0.1.0":
        raise ValueError("Use the published 0.1.0 wheel to prepare the centroid fixtures.")
    ref = build(np.random.RandomState(seed).standard_normal(16_000))
    means = np.lib.format.open_memmap(directory / "means.npy", mode="w+", dtype="float64", shape=(count, len(ref)))
    weights = ref.weights
    np.save(directory / "weights.npy", weights)
    rng = np.random.RandomState(seed)
    for i in range(count):
        digest = build(rng.standard_normal(16_000))
        assert np.array_equal(digest.weights, weights)
        means[i] = digest.means
    for offset in (0, 16_000):
        values = np.arange(offset, offset + 16_000, dtype=np.float64)
        d = build(values)
        np.savez(directory / f"range-{offset}.npz", means=d.means, weights=d.weights)
    d = build(np.arange(16_000, dtype=np.float64))
    other = build(np.arange(16_000, 32_000, dtype=np.float64))
    result = d.merge(other)
    np.savez(directory / "range-merged.npz", means=result.means, weights=result.weights)
    million = TDigest.from_array(np.arange(1_000_000, dtype=np.float64), delta=300)
    np.savez(directory / "range-million.npz", means=million.means, weights=million.weights)
    (directory / "run.json").write_text(json.dumps({"seed": seed, "arrays": count}))


def reference(directory, name):
    with np.load(directory / f"{name}.npz") as saved:
        return saved["means"].tobytes(), saved["weights"].tobytes()


def measure(operation, inputs, workers, repeats):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        Parallel(backend="threading", n_jobs=workers)(delayed(operation)(item) for item in inputs)
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def time_million(directory, seed):
    values = np.random.RandomState(seed).permutation(np.arange(1_000_000, dtype=np.float64))
    expected = reference(directory, "range-million")
    assert state(TDigest.from_array(values, delta=300)) == expected
    times = []
    for _ in range(7):
        start = time.perf_counter()
        digest = TDigest.from_array(values, delta=300)
        times.append(time.perf_counter() - start)
        assert state(digest) == expected
    print(json.dumps({"version": version("tdigest-rs"), "seed": seed, "centroids": len(digest), "seconds": times}))


def run(directory, seed, count, workers):
    if json.loads((directory / "run.json").read_text()) != {"seed": seed, "arrays": count}:
        raise ValueError("Prepare this directory with the same seed and number of arrays.")
    means = np.load(directory / "means.npy", mmap_mode="r")
    weights = np.load(directory / "weights.npy")
    assert len(means) == count and len(weights) == 4135
    digests = []
    output_state = hashlib.sha256()
    for row in means:
        d = TDigest.from_means_weights(row, weights, delta=10_000)
        assert state(d) == (row.tobytes(), weights.tobytes())
        merged = d.merge(d, delta=304)
        assert len(merged) == 161 and merged.weights.sum() == 32_000
        for part in state(merged):
            output_state.update(part)
        digests.append(d)

    def self_merge(digest):
        digest.merge(digest, delta=304)

    repeats = 3 if workers > 1 else 1
    normal = measure(self_merge, digests, workers, repeats)
    del digests

    rng = np.random.RandomState(seed)
    ranges = [np.arange(i, i + 16_000, dtype=np.float64) for i in (0, 16_000)]
    # Each row is a different unsorted permutation. Neighbouring rows have disjoint ranges.
    rows = [rng.permutation(ranges[i % 2]) for i in range(count)]
    built = [build(row) for row in rows]
    expected = [reference(directory, f"range-{i}") for i in (0, 16_000)]
    expected_merge = reference(directory, "range-merged")
    assert len(expected_merge[0]) == 160 * 8
    for i, d in enumerate(built):
        assert len(d) == 4135 and state(d) == expected[i % 2]
    pairs = list(zip(built[::2], built[1::2]))
    for left, right in pairs:
        assert state(left.merge(right)) == expected_merge

    def discard_build(row):
        build(row)

    def merge(pair):
        pair[0].merge(pair[1])

    def build_and_merge(pair):
        build(pair[0]).merge(build(pair[1]))

    single_build = measure(discard_build, rows, workers, repeats)
    pair_merge = measure(merge, pairs, workers, repeats)
    combined = measure(build_and_merge, list(zip(rows[::2], rows[1::2])), workers, repeats)
    print(
        json.dumps(
            dict(
                version=version("tdigest-rs"),
                seed=seed,
                workers=workers,
                arrays=count,
                normal_self_merge_s=normal,
                normal_output_sha256=output_state.hexdigest(),
                exact_build_s=single_build,
                exact_merge_s=pair_merge,
                exact_build_and_merge_s=combined,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "time", "million"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--arrays", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    if args.arrays <= 0 or args.arrays % 2:
        parser.error("--arrays must be positive and even")
    args.directory.mkdir(parents=True, exist_ok=True)
    if args.action == "prepare":
        prepare(args.directory, args.seed, args.arrays)
    elif args.action == "million":
        time_million(args.directory, args.seed)
    else:
        run(args.directory, args.seed, args.arrays, args.workers)
