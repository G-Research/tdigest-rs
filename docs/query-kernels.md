# Query Kernels

Queries live in `src/tdigest/cdf.rs` and `src/tdigest/quantile.rs`. They share
the same centroid model but use different search strategies today.

## CDF

`TDigest::cdf(&[x])` is optimized for repeated probes against one digest.

Per call it:

1. Returns `NaN` for every probe if the digest has no centroids.
2. Builds one prefix-weight vector where `prefix[i]` is mass before centroid
   `i`.
3. Evaluates each probe with `cdf_at_val_fast`.

`cdf_at_val_fast` uses binary search over centroid means:

- exact hit: midpoint mass `(prefix[i] + 0.5 * weight[i]) / total`.
- left of first centroid: clamp below `min`, otherwise use a guarded ramp.
- right of last centroid: clamp above `max`, otherwise use a guarded ramp.
- between centroids: interpolate center-to-center while excluding half-mass
  from atomic neighbors.

When both adjacent centroids are atomic, the interpolation span becomes zero
and the CDF is a step. This is the key rule that prevents discrete mass from
being smeared across gaps.

## CDF Speed Path

CDF has two performance levers:

- The prefix vector is built once per call and reused for every probe.
- Probe batches with length `>= 32768` use Rayon parallel iteration.

Small batches stay scalar because Rayon startup overhead is not free. Large
batches get parallel probe evaluation while sharing the same immutable centroid
and prefix slices.

## Quantile

`TDigest::quantile(q)` currently uses a half-weight bracketing scan, not binary
search.

The flow is:

1. Propagate `NaN` probe as `NaN`.
2. Return `NaN` for empty digest.
3. Clamp finite core `q` into `[0, 1]`; frontend surfaces reject invalid probes
   before reaching this core path.
4. Convert `q` to a target weight index.
5. Clamp near edges to `min` / `max`.
6. Scan adjacent centroid spans until the target lies between two centroid
   centers.
7. Interpolate with singleton/atomic pile rules.

The scan is simple and branch-local, which is fine for typical digest sizes,
but it is not the same as the CDF binary search path.

## Quantile Atomic Rules

Atomic piles and unit singletons get special handling:

- A target strictly inside an atomic pile returns that pile mean exactly.
- A target close enough to a unit singleton center snaps to that singleton.
- Otherwise interpolation removes unit-singleton dead zones from the span.

These rules are paired with CDF midpoint/step behavior and are why
under-capacity exactness works at training mid-ranks.

## Median

`median()` returns `NaN` for empty digests. For odd total count it delegates to
`quantile(0.5)`. For even count it averages the bracketing centroid means so
the common two-middle-values case does not over-interpolate.

## Possible Future Optimization

Quantile could be changed to a prefix-span binary search: precompute centroid
center positions, binary-search the target index, then apply the same
interpolation rules. That would reduce query work for large centroid counts,
but it must preserve all current atomic pile, unit singleton, edge, and
under-capacity exactness tests.

Do not describe quantile as binary-search based until that implementation is
actually in place.
