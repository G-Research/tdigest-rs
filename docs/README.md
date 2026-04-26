# tdigest-rs Internals Docs

This directory is the maintainer map for the implementation. It complements the
root design documents instead of replacing them.

Canonical root docs:

- `../api_design.md`: target public API contract across Rust, Python, Polars,
  Java, and CLI.
- `../tdigest_design.md`: current Rust core implementation design.
- `../comparison_design.md`: external comparison and positioning.

Read these docs when changing internals:

- `compressor-pipeline.md`: how `src/tdigest/compressor.rs` turns sorted
  centroid streams into a bounded digest.
- `query-kernels.md`: CDF, quantile, median, binary search, prefix weights,
  Rayon, and exactness contracts.
- `quality-and-benchmarks.md`: pinned quality checks, story matrices, and
  benchmark intent.
- `wire-and-precision.md`: TDIG versions, precision, Polars struct codec, and
  cross-surface serialization.
- `review-compat-suite.md`: copied upstream `tdigest-rs` tests and benchmark
  used for release review.

The short version: Rust owns semantics, `frontends.rs` owns strict public
validation, adapters stay thin, and `integration/api_coherence/` proves that
surface behavior stays aligned.
