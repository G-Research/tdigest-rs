# Wire Format And Precision

`src/tdigest/wire.rs` implements the canonical TDIG byte format. It is the
cross-surface serialization contract for Python, Polars, Java, CLI, and Rust.

## Versions

TDIG currently supports decode for v1, v2, and v3. The default encoder writes
v3.

Version summary:

- v1: header plus `mean + u64 weight`; kind is inferred with the legacy
  `weight == 1` heuristic.
- v2: header plus `mean + f64 weight + kind`; preserves fractional weights
  and atomic-vs-mixed kind.
- v3: explicit payload precision in the header, header length, flags, and
  optional CRC32 checksum; payload layout matches v2.

Versioned encode controls are exposed across surfaces as `to_bytes(version=...)`
or `toBytes(version)`.

## Precision

Digest storage precision is controlled by `TDigest<F>`:

- `F = f32`: centroid means and weights are stored compactly.
- `F = f64`: centroid means and weights use full precision.
- `sum` and `count` remain `f64` in both cases.

Wire precision follows the digest storage type. v1/v2 infer precision from
payload length; v3 stores it explicitly in the header.

Strict frontend merge rejects mixed precision. Users must cast explicitly with
the public cast APIs before merging.

## Polars Struct Codec

`src/tdigest/codecs.rs` maps digests to a Polars struct column. The struct is
compact when centroid `mean`, centroid `weight`, `min`, and `max` are `Float32`;
it is full precision when those fields are `Float64`.

Decode is strict:

- The incoming struct dtype must match the target `TDigest<F>`.
- Mixed centroid mean/weight dtypes are rejected.
- Null scalar fields and null centroid entries are rejected.

`src/polars_expr.rs` decides output dtype at planning time. Float32 training
input produces the compact digest schema; Float64 and other numeric inputs
produce the f64 schema unless explicitly cast.

## `from_bytes` In Polars

Polars `from_bytes` has strict blob behavior:

- Null blobs are errors.
- Empty byte blobs are invalid TDIG and error.
- Mixed f32/f64 blobs in one logical deserialize operation are rejected unless
  the user normalizes explicitly first.

When no precision hint is supplied, the plugin sniffs the first blob and then
requires the rest of the column to match.

## Checksums

v3 writes a CRC32 checksum when the checksum flag is set. The checksum covers
the header with the checksum slot zeroed plus the payload. Decode verifies it
when present, so payload corruption fails before a malformed digest reaches
query code.

## Change Guidance

Any wire change is high impact. Update the wire docs, Python/Polars/Java
interop tests, and coherence tests together. Do not silently promote or demote
precision during strict decode paths.
