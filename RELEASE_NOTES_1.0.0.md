# Reviewing tdigest-rs 1.0

tdigest-rs 1.0 adds native Polars aggregation and batch CDF queries backed by the same Rust core used by Python and Java. Python calls that use `delta=` or omit both sizing arguments retain the published 0.1.0 numerical results. The two public constructors emit a `DeprecationWarning` for these calls. To migrate, pass `max_size=`, for example `TDigest.from_array(values, max_size=200)`. Quantile estimates can change when switching modes, especially near the tails or when one value carries most of the weight.

## What changes for existing Python code?

All 91 original Python test cases pass on the 1.0 release wheel. Both releases return the same results for this Float32 merge. Each input keeps its original weight.

```python
import numpy as np
from tdigest_rs import TDigest

first = TDigest.from_array(np.array([10, 20, 30], dtype=np.float32), delta=300)
second = TDigest.from_array(np.array([40, 50], dtype=np.float32), delta=300)
combined = first.merge(second)
print(int(first.weights.sum()), int(combined.weights.sum()))
# 3 5
print(combined.median(), combined.quantile(0.95))
# 30.0 50.0
```

<details>
<summary>Input handling with delta and max_size</summary>

| Input or operation | Existing calls (`delta=` or default) | With `max_size=` |
| --- | --- | --- |
| Three NaNs | Skips all three and returns an empty digest. | Rejects NaN during construction. |
| `[10, 20, NaN]` | Keeps 10 and 20. | Rejects the array because it contains a NaN. |
| `[-∞, +∞, +∞]` | Stores two endpoints, with weight two at +∞. | Rejects infinite training values. |
| `[-∞, -20, 8, 10, +∞]` | Keeps all five means, including the endpoints. | Rejects the array because it contains infinities. |
| Merge `[-∞, 1, 2, +∞, +∞]` with `[-∞, -∞, -1, 30, +∞, +∞]` | Combines repeated endpoints and returns six centroids. | Construction rejects each input before the merge. |
| A Float16 NumPy array | Raises `TypeError`. The original API accepts Float32 and Float64 arrays. | Accepts Float16 input. Default mean storage is Float64. |
| Merge digests built from Float32 and Float64 arrays | The arrays select different stored precisions, so a direct merge raises `TypeError`. | Both constructors default to Float64, so the merge succeeds with Float64 means. If you choose different stored precisions, cast explicitly before merging. |

</details>

In delta mode, loading a 0.1.0 pickle, copying a digest or loading its dictionary rebuilds the stored means and weights at the original default `delta=300`. The result matches the published wheel's reload, which can recluster centroids built at another delta. Direct `from_means_weights` calls consume pairs in the supplied order. The compatibility tests compare the exact bits of both arrays, quantiles and trimmed means, including 32 pickles from the published wheel.

Three edge cases have explicit handling. Empty digests can be copied and pickled. A total weight above `u32::MAX` and inputs with different lengths each raise `ValueError`. Rust callers using `tdigest_core` should move to the root `tdigest-rs` crate and handle checked operations.

## Quantiles when migrating to capacity mode

| Python example | 0.1.0 and 1.0 with `delta=300` | 1.0 with `max_size=200` |
| --- | ---: | ---: |
| P10 of two observations, 1 and 2 | 0.7 | 1.0 |
| Median of values 0 and 10 supplied with weights 90 and 10 | 1.0 | 0.0 |

The original algorithm interpolates between neighboring centroid means, including across repeated observations. Near the lower tail it can extrapolate below the smallest observation. Capacity mode returns the observed value inside a point mass and interpolates inside a centroid that combines distinct values.

**What can you choose?** With `max_size=`, `singleton_policy="edges"` protects up to `pin_per_side` distinct values at each tail during compression. With `"off"`, the compressor can combine those values. The query rule is the same in both cases. For the integers 0–99, the 1.75th percentile lands in an interpolated range with protection off and inside the observed value 1 with the first three tail values protected.

```python
import numpy as np
from tdigest_rs import TDigest

values = np.arange(100, dtype=np.float64)
pooled = TDigest.from_array(values, max_size=10, singleton_policy="off")
pinned = TDigest.from_array(values, max_size=10, singleton_policy="edges", pin_per_side=3)
print(f"{pooled.quantile(0.0175):.3f}", f"{pinned.quantile(0.0175):.3f}")
# 1.125 1.000
```

## Accuracy on recorded data

We compared the releases at **equal final centroid counts**. For each case, 1.0 used `max_size=128`. We selected the 0.1.0 `delta` using only the resulting count. Here is the actual setup for one case. Run it from the repository root in each wheel’s environment.

```python
import csv
import gzip
from importlib.metadata import version

import numpy as np
from tdigest_rs import TDigest

with gzip.open("tests/quality/data/bike.csv.gz", "rt") as source:
    rides = np.array([float(row["cnt"]) for row in csv.DictReader(source)], dtype=np.float32)

release = version("tdigest-rs")
if release == "0.1.0":
    digest = TDigest.from_array(rides, delta=276.5625)
else:
    digest = TDigest.from_array(rides, max_size=128, precision="f32",
                                scale="k2", singleton_policy="use")
print(f"{release}: {len(rides)} rows, {len(digest)} centroids, P99 {digest.quantile(0.99):.2f}")
# 0.1.0: 17379 rows, 128 centroids, P99 781.84
# 1.0.0: 17379 rows, 128 centroids, P99 782.24
```

The 108 cases cover nine bike-rental and household-energy columns, two precisions, two input orders, and three build patterns (one bulk build, 40 sequential merges, or 40 parts merged in pairs). All 108 pairs matched in size, from 8 to 128 centroids. We checked 135 quantiles per digest against the sorted observations.

| Error across 108 matched pairs | Lower in 1.0 | Equal | Higher in 1.0 |
| --- | ---: | ---: | ---: |
| Average error in percentile rank | 102 | 0 | 6 |
| Worst rank error in the outer 1% of either tail | 104 | 3 | 1 |
| Average error in the column’s original units | 104 | 0 | 4 |

All six increases in average rank error occurred on outdoor temperature. The largest increase was **0.011 percentage points** after merging 40 Float32 parts in pairs. [Per-case settings, counts, and errors](docs/upstream-1.0.0/matched-accuracy.csv) · [Known accuracy limits](tests/quality/known-limits.json)

## Initialization, quantile, and CDF timings

Python 3.12.15 and NumPy 1.26.4 on one logical CPU of a Ryzen 9 9950X3D. Each build received the same million shuffled Float64 integers. Every digest held **146 centroids** and returned the same median (499,999.5) and P99 (989,999.5). Query timings exclude construction. Each cell is the median of seven timed rounds.

| Operation | 0.1.0, `delta=300` | 1.0, `delta=300` | 1.0, `max_size=146` |
| --- | ---: | ---: | ---: |
| Initialize from the array | 108.0 ms | 20.1 ms | 29.9 ms |
| Query the median | 109 ns | 90 ns | 56 ns |
| Query P99 | 145 ns | 114 ns | 77 ns |
| Query the CDF for 65,536 values in one call | Unavailable | Unavailable | 2.56 ms |

Merging two prebuilt 146-centroid digests at `delta=300` took 3.22 µs on the published wheel and 2.81 µs on the 1.0 release build. [Timing samples](docs/upstream-1.0.0/release-timings.csv)

## CDF in a Polars group

This 1.0 query ranks each request against its own service. The two 48 ms requests show how the same latency can be high for one service and low for another. With two observations per service, the lower value receives rank 0.25 and the higher one receives 0.75.

```python
import polars as pl
import tdigest_rs as td

requests = pl.LazyFrame({
    "service": ["search", "search", "checkout", "checkout"],
    "latency_ms": [12.0, 48.0, 48.0, 90.0],
})
digests = requests.group_by("service").agg(
    td.tdigest("latency_ms", max_size=200).alias("digest")
)
ranked = requests.join(digests, on="service").select(
    "service", "latency_ms", td.cdf("digest", pl.col("latency_ms")).alias("cdf")
)
for row in ranked.sort("service", "latency_ms").collect().iter_rows():
    print(*row)
# checkout 48.0 0.25
# checkout 90.0 0.75
# search 12.0 0.25
# search 48.0 0.75
```

<details>
<summary>Reproduce the comparisons</summary>

From the repository root, set `OLD_PY` and `PR_PY` to absolute paths to Python executables in two isolated environments with NumPy 1.26.4. Install the published 0.1.0 wheel in the first and the 1.0 release wheel built from this checkout in the second. The accuracy runner uses the checked-in datasets and selects the baseline compression parameter before it scores any quantile.

```bash
mkdir -p build/review
compare=docs/upstream-1.0.0/compare_recorded.py
"$PR_PY" -I "$compare" capacity --max-size 128 > build/review/1.0.csv
"$OLD_PY" -I "$compare" delta --match-counts build/review/1.0.csv > build/review/0.1.0.csv
wc -l < build/review/1.0.csv
# 109
wc -l < build/review/0.1.0.csv
# 109
```

For timings, choose one available logical CPU (CPU 8 in the reported run). The driver uses seed 42, warms up each operation, and writes the per-call median and all seven samples in nanoseconds. Times vary by machine.

```bash
export OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1
timing=docs/upstream-1.0.0/time_operations.py
taskset -c 8 "$OLD_PY" -I "$timing" delta > build/review/old-timing.csv
taskset -c 8 "$PR_PY" -I "$timing" delta > build/review/delta-timing.csv
taskset -c 8 "$PR_PY" -I "$timing" capacity > build/review/capacity-timing.csv
cut -d, -f3 build/review/capacity-timing.csv
# operation
# build
# median
# p99
# cdf_65536
```

The original suite is pinned to G-Research commit `8d9cf1fe82c5c9254e5dc9593f01717e317f4333`. The numerical and pickle fixtures record their source and generation methods separately. [Original suite manifest](tests/upstream_legacy/manifest.json) · [Numerical generator](compat/numerical-parity/generate.rs) · [Pickle provenance](compat/published-pickle/PROVENANCE.json) · [Dataset credits](tests/quality/data/README.md)

</details>
