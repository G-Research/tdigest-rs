# Quality Checks And Benchmarks

The quality modules under `src/quality/` are test-only tools for detecting
accuracy drift. They are separate from API coherence tests: coherence says
"all surfaces agree"; quality says "the algorithm still behaves like the
blessed implementation."

## Shared Quality Base

`quality_base.rs` provides:

- deterministic dataset generation via `crates/testdata`;
- `QualityReport { n, ks, mae, score }`;
- shared exact quantile and midpoint-ECDF helpers;
- `build_digest_sorted`, which applies the configured precision profile and
  builds a digest with a selected scale.

`score` is a heuristic scalar where higher is better. It combines MAE and
KS-like max error so runs can be compared quickly, but the raw `ks` and `mae`
numbers matter more when reviewing changes.

## Pinned Regression Tests

`cdf_quality.rs` and `quantile_quality.rs` each contain a pinned regression
case for a representative mixture distribution.

These tests intentionally fail on meaningful drift, even if the score improves.
That keeps algorithm changes explicit: update the baseline only when the new
behavior is understood and blessed.

When blessing a change:

1. Run the test and capture the printed report.
2. Explain why the new result is expected.
3. Update the baseline constants and changelog entry together.
4. Run `make build` and `make test`.

## Story Matrices

Ignored tests named `*_story_matrix` print larger sweeps across distributions,
`max_size`, scale families, and precision modes. They are not default gates
because they are slower and diagnostic, but they are the right tool when a
compressor, scale, query, or precision change might shift accuracy.

Run them directly when needed, for example:

```bash
cargo test cdf_story_matrix -- --ignored --nocapture
cargo test quantile_story_matrix -- --ignored --nocapture
```

## Criterion Benchmarks

The repo has Rust benchmarks under `benches/`:

- `tdigest_bench.rs`: build/merge behavior.
- `cdf_quantile_bench.rs`: query kernels.
- `codecs_bench.rs`: serialization and codec paths.

Use them to confirm performance-sensitive changes, especially when touching:

- `compressor.rs`;
- `merges.rs`;
- `cdf.rs`;
- `quantile.rs`;
- `wire.rs`;
- Polars codec logic.

## Multithreading

The main query-level multithreading is in CDF: large probe batches use Rayon
after a conservative threshold. Digest construction and merge paths emphasize
streaming and lower peak allocation rather than broad parallelism inside one
digest operation.

Python and Java bindings should keep expensive native work outside host-layer
loops where possible. If a binding grows a new batch operation, prefer routing
through shared Rust operations instead of threading in the adapter.

## Review Compatibility Benchmark

`compat/tdigest-rs-upstream/bindings/python/benchmarks/run.py` is copied from
the old upstream `tdigest-rs` project. It is intentionally kept separate from
normal benchmarks and from default CI because it is heavy and aimed at review
parity.

Use the dedicated Make target documented in `review-compat-suite.md`.
