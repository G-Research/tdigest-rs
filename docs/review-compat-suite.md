# Upstream Compatibility Suite

This repository includes a separate copied review suite from the old upstream
`tdigest-rs` project. It is not mixed into the normal tests because the suite
checks legacy Python API behavior, including behavior that intentionally differs
from the stricter current `tdigest_rs` public API.

## Location

Copied tests:

```text
compat/tdigest-rs-upstream/bindings/python/tests/
```

Copied benchmark:

```text
compat/tdigest-rs-upstream/bindings/python/benchmarks/run.py
```

Provenance and checksums are recorded in
`compat/tdigest-rs-upstream/PROVENANCE.md`.

## Compatibility Module

The copied suite imports `tdigest_rs`. The dedicated runner places an isolated
compatibility module at `compat/tdigest-rs-upstream/python/tdigest_rs/` ahead of
the real package for that review path. It preserves legacy-facing Python API
behavior such as:

- `TDigest.from_array(arr, delta=...)`;
- `TDigest.from_means_weights(...)`;
- `means` / `weights` arrays;
- immutable `merge`;
- `trimmed_mean`;
- `update`;
- `to_dict` / `from_dict`;
- pickle and deepcopy support;
- dtype-based f32/f64 merge compatibility errors.

This compatibility layer is intentionally separate from the production package;
it must not relax the strict validation rules documented in `api_design.md`.

## Running

Run the exact copied test suite:

```bash
make compat-test
```

Run the copied upstream benchmark manually:

```bash
make compat-bench
```

The benchmark is intentionally not part of default `make test` because it is
large and depends on `joblib`.

## Review Rule

Do not edit files under the copied `tests/` or copied `benchmarks/run.py` unless
the source snapshot is intentionally refreshed. Add wrappers, fixtures, or
documentation outside the copied tree.
