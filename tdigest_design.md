# tdigest_design.md

This document is the **current-state implementation design** for the Rust TDigest core.

- Scope: internals in `src/tdigest/*` and how frontends use them.
- It describes what the code does today.
- For external implementation/paper comparisons and tradeoff positioning, see `comparison_design.md`.

For target cross-surface API contracts, see `api_design.md`.

## 1. Document boundary

This file covers:
- Core data model and module responsibilities.
- Ingest/merge/compression/query/wire internals.
- Current frontend-service layering.

Out of scope for this file:
- Target-state product/API policy (see `api_design.md`).
- External library or paper comparisons.

## 2. Core architecture

### 2.1 Main modules

- `src/tdigest/tdigest.rs`: digest type, builders, add/merge entry points, bytes entry points.
- `src/tdigest/merges.rs`: stream-merging helpers + stage-1 normalization.
  - Includes a heap-based streaming k-way centroid merge used by digest-to-digest merge.
- `src/tdigest/compressor.rs`: staged compressor (`compress_into`) and policy logic.
- `src/tdigest/centroids.rs`: centroid representation and `Atomic` vs `Mixed` behavior.
- `src/tdigest/quantile.rs`: quantile and median internals.
- `src/tdigest/cdf.rs`: CDF internals.
- `src/tdigest/wire.rs`: TDIG codec.
- `src/tdigest/frontends.rs`: shared frontend validation/config/merge/decode helpers.

### 2.2 Digest and centroid model

`TDigest<F>` where `F` is `f32` or `f64`:
- Centroid means/weights are stored in `F`.
- Digest stats `sum` and `count` are `f64`.
- Digest min/max are stored in `F` and surfaced as `f64`.

Centroid kind is explicit:
- `Atomic`: exact identical-value mass (unit or pile).
- `Mixed`: merged/interpolated mass.

`is_singleton()` is kept as compatibility naming and maps to `is_atomic()`.

## 3. How new values are merged

### 3.1 Add path (`add` / `add_many`)

`add_many(values)` is a thin wrapper:
1. Empty input: no-op.
2. Otherwise call `merge_unsorted(values)`.
3. Replace `self` with the merged result.

`add(value)` delegates to `add_many([value])`.

Weighted add path:
- `add_weighted(value, weight)` and `add_weighted_many(values, weights)` are supported in Rust core.
- Inputs must satisfy:
  - same length for values/weights
  - finite values
  - finite, strictly positive weights
- Weighted values are converted to atomic centroids, then merged through the same digest merge/compression path.
- Frontend surfaces route weighted add through `FrontendDigest::add_weighted_f64(...)`, so Python/JNI/Polars share the same validation and merge semantics.

### 3.2 Raw-value ingest (`merge_unsorted` / `merge_sorted`)

`merge_unsorted`:
1. Reject non-finite values.
2. Sort ascending.
3. Call `merge_sorted`.

`merge_sorted`:
1. Reject non-finite values.
2. Empty input: return clone.
3. Build a result shell (`new_result_for_values`) with updated count/min/max placeholders.
4. Build a sorted stream with `MergeByMean::from_centroids_and_values`:
- Existing centroids are reused as-is.
- Each new scalar value becomes an atomic unit centroid (`weight=1`).
- This is a streaming two-way merge iterator (no prebuilt merged buffer).
5. Run the full compressor pipeline with `compress_into`.

Important: compressor stage 1 recomputes digest metadata (`count`, `sum`, `min`, `max`) from the normalized stream and writes those back to the result.

Weighted constructor/merge helpers:
- `from_weighted_unsorted(values, weights, max_size)` builds from weighted input directly.
- `merge_weighted_unsorted(values, weights)` merges weighted input into an existing digest.

### 3.3 Digest-to-digest merge (`merge_digests`)

`TDigest::merge_digests(digests)`:
1. Scan digests and keep only non-empty runs (`count>0` and non-empty centroid vector).
2. Pick the first non-empty digest as the config source (`max_size`, `scale`, `policy`).
3. Perform a streaming k-way merge of sorted centroid runs via `KWayCentroidMerge::from_runs`.
  - Implementation uses a min-heap over run heads (`O(total_len * log(k))`).
  - It avoids materializing one full concat+sort buffer before compression.
  - Historical concat+sort run merge is retained only as a `#[cfg(test)]` baseline helper.
4. Recompress with `compress_into` using the chosen config.
5. If all digests are empty, return `TDigest::default()`.

Note: core merge is permissive by design. Strict precision/config checks are enforced in `FrontendDigest::merge_in_place`.

## 4. Compression pipeline internals

`compress_into` runs stages 1->6.

### 4.1 Stage 1: normalize

`normalize_stream` does three jobs:
1. Validate non-decreasing mean order (panic on decreasing mean).
2. Coalesce adjacent equal-mean items into one atomic run (`new_singleton_f64(mean, combined_weight)`).
3. Accumulate:
- `total_w`
- `total_mw = sum(mean * weight)`
- `min`/`max` mean

Output at this stage is strictly increasing by mean.

### 4.2 Stage 2: policy slice

Policy is derived from `SingletonPolicy` and current `max_size`.

`Off`:
- `left=[]`, `right=[]`, `interior=all`.
- `core_cap = max_size`.
- Fast paths:
  - if `len <= max_size`, passthrough.
  - if `max_size == 1`, collapse all to one mixed centroid.
  - if `max_size == 2`, bucketize directly into 2.

`Use`:
- Preserve up to one centroid on each side (first and last when present).
- `core_cap = max_size - left_len - right_len` (saturating).
- Interior excludes the preserved endpoints.

`UseWithProtectedEdges(k)`:
- Protect up to `k` consecutive atomic centroids from the left and right edges.
- Edge scanning stops at the first mixed centroid.
- Protected edges are outside interior cap.
- `core_cap = max_size` (applies only to interior).

### 4.3 Stage 3: k-limit merge

`klimit_merge(items, d=max_size, family)` greedily builds interior clusters.

Definitions:
- `total_w = sum(item.weight)`.
- `q_to_k(q, d, family)` maps quantile position to k-space.
- Keep cluster growth while `delta_k <= 1 + KLIMIT_TOL` (`KLIMIT_TOL = 1e-12`).

Loop sketch:
1. Track completed cluster mass `c_acc`.
2. For candidate item with weight `w_next`:
- `q_r = (c_acc + acc_cluster_weight + w_next) / total_w`
- `k_right = q_to_k(q_r, d, family)`
- Compare vs current `k_left`.
3. If within limit, absorb item.
4. Else flush current cluster and start a new one.

Cluster kind emission:
- Single-item cluster stays atomic only if its head item was atomic.
- Multi-item cluster is always emitted as mixed.

### 4.4 Stage 4: cap

If interior count exceeds `core_cap`, run `bucketize_equal_weight`:
1. Compute `target = total_weight / buckets`.
2. Sweep centroids in order, accumulating weighted stats.
3. Emit a mixed centroid when accumulated weight reaches target.
4. Emit trailing remainder if non-empty and bucket budget remains.

This is order-preserving and weight-preserving (up to floating roundoff).

### 4.5 Stage 5: assemble

Concatenate without mutation:
- `left + core_capped + right`.

### 4.6 Stage 6: post-policy finalize

`Use` policy only:
- If assembled length still exceeds `max_size`, bucketize whole assembled vector to enforce global cap.

`Off` and `UseWithProtectedEdges(k)`:
- No additional post cap.

### 4.7 How `max_size` actually applies

`max_size` is policy-dependent in effect:
- `Off`: hard cap on total centroid count.
- `Use`: tries to preserve endpoints but still enforces final total `<= max_size`.
- `UseWithProtectedEdges(k)`: hard cap on interior only; total can exceed `max_size` by protected edges.

`max_size` also sets `d` for `q_to_k`, so it influences cluster admissibility in stage 3 even before stage-4/6 capping.

## 5. Query internals

### 5.1 CDF (`src/tdigest/cdf.rs`)

`TDigest::cdf(vals)`:
1. Empty digest -> vector of `NaN`.
2. Build one prefix-weight array (`prefix[i] = cumulative weight before centroid i`).
3. For each probe:
- `NaN` probe -> `NaN`.
- Else evaluate via `cdf_at_val_fast` (binary search by centroid mean).

`cdf_at_val_fast` cases:

Exact centroid hit (`idx`):
- Return midpoint mass: `(prefix[idx] + 0.5 * w_idx) / total_w`.

Left of first centroid:
- `< min` -> `0.0`.
- Between `min` and first mean: guarded linear ramp using first centroid half-weight.

Right of last centroid:
- `> max` -> `1.0`.
- Between last mean and `max`: symmetric guarded ramp.

Between centroids `li` and `ri`:
- `gap = mean_r - mean_l`.
- Exclusions:
  - `left_excl = 0.5 * w_l` if left centroid is atomic else `0`.
  - `right_excl = 0.5 * w_r` if right centroid is atomic else `0`.
- Span mass:
  - `dw_center = 0.5 * (w_l + w_r)`
  - `dw_span = dw_center - left_excl - right_excl`
- Base mass:
  - `base = prefix[li] + 0.5 * w_l + left_excl`
- Output:
  - `(base + dw_span * frac) / total_w`, `frac=(x-mean_l)/gap`.

If both neighbors are atomic, `dw_span = 0`, so this becomes a strict step.

Performance:
- Rayon path is used when probe count `>= PAR_MIN` where `PAR_MIN = 32768`.

### 5.2 Quantile and median (`src/tdigest/quantile.rs`)

`TDigest::quantile(q)`:
1. `q=NaN` -> `NaN`.
2. Empty digest -> `NaN`.
3. Finite `q` is clamped to `[0,1]` in core.
4. Convert to target weight index: `index = q * total_weight`.
5. Edge clamps:
- `index < 1` -> `min`
- `index > total_weight - 1` -> `max`
6. Find bracketing centroids using half-weight center-to-center spans.
7. Interpolate with singleton-aware rules.

Singleton-aware interpolation rules:

Atomic pile snap (`weight > 1`):
- If `index` is strictly inside pile half-width, return pile mean exactly.

Unit singleton snap (`weight == 1`):
- If probe is within `0.5` of centroid center, snap to centroid mean.

Otherwise interpolate with dead zones:
- `dead_left = 0.5` when left is unit singleton else `0`.
- `dead_right = 0.5` when right is unit singleton else `0`.
- Interpolate in remaining span.

This keeps quantile behavior symmetric with CDF atomic exclusions.

`median()`:
- Empty -> `NaN`.
- Odd total count -> `quantile(0.5)`.
- Even total count -> average means of bracketing centroids (dedicated even-count branch).

### 5.3 Strictness split for quantile probes

Current split is intentional:
- Core `quantile` clamps finite probes.
- Frontend layer (`FrontendDigest::quantile_strict`) rejects non-finite and out-of-range probes.

### 5.4 Under-capacity exactness (current behavior)

When total training cardinality `N` is below `max_size` (and no later operation forces an over-capacity recompression), exactness is pinned down as follows:
- `cdf(x)` at training values is exact under midpoint ECDF semantics over ties: `(#<x + 0.5*#=x) / N`.
- `quantile(q)` is exact at mid-ranks `q = (i + 0.5) / N`, returning the exact order statistic at index `i`.

Non-goal:
- Exactness for arbitrary quantile probes is not guaranteed; the exactness contract is mid-rank-specific.

These semantics are asserted by focused tests in:
- `src/tdigest/cdf.rs`
- `src/tdigest/quantile.rs`

## 6. TDIG wire format

Implemented in `src/tdigest/wire.rs`.

Versioning:
- Encoder default writes TDIG **v3**.
- Decoder supports **v1**, **v2**, and **v3**.
- Explicit versioned encoding is available via:
  - Rust core: `to_bytes_with_version(WireVersion::V1|V2|V3)`
  - Frontend service: `to_bytes_with_version(...)`
  - Python/Java/Polars surfaces: `to_bytes(version=...)` / `toBytes(version)`

Payload layouts:
- v1:
  - `f32 mean + u64 weight` per centroid, or
  - `f64 mean + u64 weight` per centroid.
- v2:
  - `f32 mean + f64 weight + kind(u8)` per centroid, or
  - `f64 mean + f64 weight + kind(u8)` per centroid.
- v3:
  - Header adds `flags`, `header_len`, and explicit `payload_precision` code.
  - Header includes optional 4-byte checksum (CRC32) controlled by flags.
  - Payload layout is the same as v2 (`mean + f64 weight + kind`), but width is explicit in header.

Decode behavior:
- v1/v2: payload length determines wire precision.
- v3: payload precision is explicit in header and payload length is validated against it.
- v1 keeps legacy heuristic (`w==1` => atomic unit, otherwise mixed).
- v2 decodes centroid kind explicitly from payload and preserves atomic-vs-mixed identity.
- v2/v3 preserve fractional centroid weights on wire.
- v3 checksum is verified when present.

## 7. Frontend service role

`src/tdigest/frontends.rs` centralizes behavior used by Python/JNI/Polars/CLI:
- Parse/normalize scale, policy, precision hints.
- Validate training values.
- Enforce strict quantile probes (`finite` + `[0,1]`).
- Enforce strict merge compatibility (precision + config).
- Provide explicit precision casting (`cast_precision`).
- Provide precision-aware decode checks (`from_bytes_with_expected`).

Adapters (`src/py.rs`, `src/jni.rs`, `src/polars_expr.rs`) should remain thin around this layer.

## 8. Known current-state gaps

1. Legacy wire compatibility mode.
- Decoder supports v1 and v2.
- Re-encoding a v1 blob produces v2 output (format changes intentionally).

2. Strict merge location.
- Strict compatibility checks are frontend-layer behavior, not core merge behavior.

3. Doc split discipline.
- Keep target behavior only in `api_design.md` and current internals only here.
