# Compressor Pipeline

`src/tdigest/compressor.rs` is the main algorithmic choke point. All raw-value
ingest and digest-to-digest merge paths feed it sorted centroid streams, and it
returns the centroid vector that becomes the new digest body.

## Inputs And Ownership

The compressor receives a `TDigest<F>` result shell, a configured `max_size`,
and an iterator of `Centroid<F>`. It assumes the stream is non-decreasing by
mean. Raw scalar input is converted to atomic unit centroids by
`MergeByMean`; digest merge input is produced by `KWayCentroidMerge`.

The compressor writes digest-level metadata into the result shell:

- `count`: total centroid weight after normalization.
- `sum`: sum of `mean * weight`.
- `min` / `max`: observed support endpoints.

## Stage 1: Normalize

`normalize_stream` lives in `src/tdigest/merges.rs` and is deliberately small.
It validates ordering, coalesces adjacent equal-mean items into one atomic
centroid, and computes total weight/sum/min/max.

Equal-mean coalescing is the only place where same-mean runs are collapsed.
After this stage, centroid means are strictly increasing, which makes query
interpolation and binary search unambiguous.

## Stage 2: Slice

Policy slicing decides which centroids are protected from the interior merge:

- `Off`: no protected edges; `core_cap = max_size`.
- `Use`: preserve at most the first and last centroid; `core_cap` subtracts
  those edges and Stage 6 enforces total `<= max_size`.
- `UseWithProtectedEdges(k)`: preserve up to `k` consecutive atomic centroids
  on each side, stopping at the first mixed centroid; `core_cap = max_size`
  applies only to the interior.

This is where the semantic difference between "total cap" and "interior cap"
is introduced.

## Stage 3: K-Limit Merge

`klimit_merge` greedily scans the interior and grows a cluster while the
scale-family delta in k-space is within `1 + KLIMIT_TOL`.

Important rules:

- `q_to_k(q, d, family)` comes from `scale.rs`.
- `d` is the configured `max_size` as `f64`.
- Single-item clusters remain atomic only if their input head was atomic.
- Multi-item clusters are always mixed.
- Order and total weight are preserved.

This keeps more resolution where the selected scale family wants it, usually
near tails for the non-uniform families.

## Stage 4: Cap

Stage 4 only runs if Stage 3 produced more interior centroids than `core_cap`.
The current strategy is a second k-limit merge with a smaller searched `d'`.

The search picks the largest `d'` in `(0, d]` that yields `<= core_cap`, so the
cap pass keeps the same scale geometry while removing only as much detail as
needed. If plateau or rare non-monotone length behavior prevents convergence,
the code tightens `d'` a few times, then falls back to order-preserving
equal-weight bucketization as a safety net.

That fallback should be treated as a guardrail, not the normal path. If a
change makes it common, update quality docs and benchmarks before merging.

## Stage 5: Assemble

Assembly concatenates `left + core_capped + right` without mutating centroids.
Because Stage 1 normalized means and Stage 2 only took slices, the assembled
vector remains ordered when Stage 3 and Stage 4 preserve order.

## Stage 6: Post

Only `Use` applies a post-policy total cap. If the assembled vector still
exceeds `max_size`, it bucketizes the full assembled vector.

`Off` and `UseWithProtectedEdges(k)` skip this step:

- `Off` already treats the entire digest as core.
- `UseWithProtectedEdges(k)` intentionally allows protected edges outside the
  interior budget.

## Change Checklist

When changing compressor behavior, run at least:

- `cargo test -- --quiet`
- the pinned quality tests in `src/quality/*`
- `make build`
- `make test`

For accuracy-sensitive changes, also run the ignored story matrices and record
why any baseline update is being blessed.
