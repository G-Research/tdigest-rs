# comparison_design.md

This document compares `tdigest-rs` with relevant papers and production implementations, and records the design tradeoffs behind this repository.

- `api_design.md` defines target cross-surface behavior.
- `tdigest_design.md` defines current implementation internals.
- This file defines external comparison and product-positioning decisions.

## 1. Scope and evaluation criteria

Compared targets:
- tdunning reference implementation and papers.
- Appian `polars-tdigest` plugin.
- Elasticsearch t-digest fork.
- Apache DataSketches quantile family (KLL/REQ/t-digest).
- Representative ecosystem ports (PostgreSQL extension, Python ports).

Evaluation criteria:
- Tail accuracy behavior and interpolation quality.
- Formal error guarantees vs empirical behavior.
- Mergeability in distributed pipelines.
- Multi-language interoperability and wire stability.
- Validation strictness and predictable error model.
- API coverage (`quantile`, `cdf`, `median`, merge, serialization).
- Operational concerns (memory control, production integration).

## 2. What this project optimizes for

`tdigest-rs` is intentionally optimized for:
- One Rust source of truth for behavior across Rust/Python/Polars/Java.
- Strict cross-surface contracts (probe/training validation, merge compatibility, null rules).
- Explicit precision handling (`f32`/`f64`) and strict mixed-precision rejection in strict frontends.
- Canonical TDIG wire interoperability across surfaces.
- Tail-aware behavior with explicit singleton policy controls.

It is intentionally *not* optimized first for:
- The strongest known formal rank-error guarantees.
- Single-runtime-only minimalism.
- Highly specialized off-heap allocators or engine-specific resource accounting.

## 3. Comparison summary (high-level)

### 3.1 `tdigest-rs` vs tdunning reference (`tdunning/t-digest`)

Where we align:
- Same family of t-digest goals: compact, mergeable quantile sketch with strong tail behavior.
- Scale-function-centric compression behavior.
- Heavy emphasis on singleton handling and interpolation quality.

Where we differ by design:
- `tdigest-rs` makes cross-surface behavior itself a first-class contract.
- `tdigest-rs` enforces strict frontend merge compatibility checks (precision + config) before merge.
- `tdigest-rs` bakes explicit precision semantics and strict decode behavior into shared frontend helpers.
- `tdigest-rs` has explicit API-level singleton policy controls (`Off`, `Use`, `UseWithProtectedEdges(k)`) as surfaced product behavior across faces.

Practical consequence:
- tdunning is still the canonical reference line for algorithm evolution.
- `tdigest-rs` is more opinionated for predictable multi-language product behavior.

### 3.2 `tdigest-rs` vs Appian `polars-tdigest`

Appian package positioning (from published package metadata):
- Polars plugin wrapper around a Rust t-digest implementation.
- Emphasizes estimated quantile plus digest creation/merge for distributed workflows.

Where `tdigest-rs` is broader:
- Multi-surface support (Rust/Python/Polars/Java), not Polars-only.
- Explicit `cdf`, merge-policy strictness, wire-interop contract, and precision-aware decode rules.
- Coherence/integration tests that enforce behavior parity across faces.

Important documentation caveat:
- Latest PyPI release metadata (`0.1.11`, October 2, 2025) shows no project description, so public feature claims are less explicit than earlier `0.1.10` metadata.

### 3.3 `tdigest-rs` vs Elasticsearch t-digest fork

Elasticsearch fork strengths:
- Deep production integration for Elasticsearch percentile workloads.
- Fork explicitly calls out fixes around AVL/merging behavior and substantial performance gains.
- Includes runtime-specific implementation options (`MergingDigest`, `HybridDigest`, `SortingDigest`) and resource-accounting integration.

`tdigest-rs` strengths relative to Elastic fork:
- Cleaner standalone multi-language API contract outside one host engine.
- Explicit cross-language wire-format coherence tests as core project contract.
- User-facing strict error semantics aligned across Rust/Python/Polars/Java.

Tradeoff:
- Elastic is stronger for Elasticsearch-native operational integration.
- `tdigest-rs` is stronger as a standalone cross-language library contract.

### 3.4 `tdigest-rs` vs Apache DataSketches (KLL/REQ/t-digest)

DataSketches strengths:
- Broader quantile-sketch portfolio:
  - KLL with strong compactness and proven rank-error behavior.
  - REQ with relative-rank accuracy focus at chosen tail.
  - t-digest as empirical option.
- Strong theoretical framing and error-bound tooling for KLL/REQ.

`tdigest-rs` strengths relative to DataSketches stack:
- Focused t-digest specialization with explicit cross-surface behavioral contracts.
- Multi-surface API coherence and strict validation semantics in one Rust-first codebase.

Key algorithmic difference in philosophy:
- DataSketches emphasizes guaranteed-rank-error families (KLL/REQ) when guarantees are required.
- `tdigest-rs` emphasizes t-digest tail quality + practical interoperability contracts.

## 4. Which is best? (scenario-based)

There is no single globally best sketch/library. Best choice is workload-driven:

1. Best for strict, provable rank-error contracts:
- DataSketches KLL/REQ.

2. Best for canonical t-digest reference behavior and ongoing algorithm lineage:
- tdunning/t-digest.

3. Best for Elasticsearch-native production percentile integration:
- Elasticsearch fork (`org.elasticsearch.tdigest` / `TDigestState`).

4. Best for Polars-only plugin adoption with minimal surface area:
- Appian `polars-tdigest`.

5. Best for cross-language API coherence with strict validation and wire interoperability in one project:
- `tdigest-rs`.

## 5. Feature-gap assessment for `tdigest-rs`

The following are meaningful potential gaps or expansion points compared with the broader ecosystem.

1. Formal error guarantees and bound APIs.
- Current approach is empirical quality + contract tests.
- Gap vs KLL/REQ-style explicit probabilistic rank-error guarantees.

2. Published benchmark/accuracy dashboards.
- Internals and tests are strong, but there is no public benchmark matrix equivalent to large cross-sketch studies.

3. Richer deployment-specific implementations.
- No direct equivalents of Elastic `SortingDigest`/`HybridDigest` modes for workload-adaptive behavior.

4. Fractional-weight wire fidelity.
- Core wire now defaults to TDIG v3 and still supports v1/v2 decode.
- Versioned encode controls (`to_bytes(version=...)`) are exposed across Python/Java/Polars for mixed data-lake migration.
- Remaining gap is mostly operational guidance/benchmarking for large mixed-version estates.

5. Explicit cast API completion across all surfaces.
- Explicit cast APIs are now exposed across Rust/Python/Java/Polars surfaces.
- Remaining gap is mostly ergonomic polish and additional migration docs/examples.

6. Resource-accounting and off-heap integrations.
- No engine-specific memory-accounting model comparable to Elastic BigArrays integration.

7. More explicit stability/performance SLOs by configuration.
- Could document practical operating envelopes by `max_size`, scale family, singleton policy, and data-shape regimes.

## 6. Design choices this comparison validates

This comparison supports keeping the following choices in `tdigest-rs`:
- Rust-first single semantic layer (`frontends.rs`) with thin adapters.
- Strict input/probe/merge/decode behavior as product contract, not ad-hoc binding behavior.
- Explicit precision policy and strict mixed-precision handling.
- Contract-level coherence tests across all exposed surfaces.

And it highlights where optional roadmap items should be considered:
- Optional guarantee-oriented companion sketch(s) or bound reporting.
- Stronger benchmark publication and reproducible perf/accuracy studies.
- Optional alternate digest modes for workloads prioritizing exactness-over-memory or vice versa.

## 7. Source index

Primary references used for this comparison:

1. tdunning reference library
- https://github.com/tdunning/t-digest

2. Original t-digest paper (arXiv)
- https://arxiv.org/abs/1902.04023

3. Software Impacts paper (2021)
- https://doi.org/10.1016/j.simpa.2020.100049
- https://www.sciencedirect.com/science/article/pii/S2665963820300403

4. Appian / polars-tdigest package pages
- https://pypi.org/project/polars-tdigest/
- https://github.com/appian/polars-tdigest

5. Elasticsearch t-digest package overview
- https://artifacts.elastic.co/javadoc/org/elasticsearch/elasticsearch-tdigest/8.17.9/org.elasticsearch.tdigest/org/elasticsearch/tdigest/package-summary.html

6. Apache DataSketches quantiles overview
- https://datasketches.apache.org/docs/QuantilesAll/QuantilesOverview.html

7. Apache DataSketches KLL docs
- https://datasketches.apache.org/docs/KLL/KLLSketch.html

8. Apache DataSketches t-digest overview
- https://datasketches.apache.org/docs/tdigest/tdigest.html

9. KLL theory paper
- https://arxiv.org/abs/1603.05346

10. Apache DataSketches REQ docs
- https://datasketches.apache.org/docs/REQ/ReqSketch.html

11. Representative ecosystem implementations
- https://github.com/tvondra/tdigest
- https://github.com/CamDavidsonPilon/tdigest
- https://github.com/protivinsky/pytdigest
- https://github.com/RedisBloom/t-digest-c
