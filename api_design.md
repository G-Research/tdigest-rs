# api_design.md

This is the **target-state API contract** for cross-surface behavior (Rust, Python, Polars, Java).

- It defines what behavior we want after refactor completion.
- It may differ from current implementation.
- If current behavior and this target contract diverge, this document wins.
- For external implementation/paper comparisons and tradeoff positioning, see `comparison_design.md`.

## 1. Scope boundary

This file defines:
- Public capabilities and semantic contracts.
- Validation, merge, serialization, and error behavior across surfaces.

This file does not define:
- TDigest algorithm internals or compression mechanics.
- External implementation comparisons.

For current implementation internals, see `tdigest_design.md`.

## 2. Locked decisions

1. Rust-first behavior ownership.
- Validation, merge policy, query rules, and decode policy live in Rust.
- Bindings stay thin (argument conversion + error mapping).

2. Strict null policy on Polars surfaces.
- Null probes and null add-values are errors.
- Null blobs in `from_bytes` are errors.

3. Explicit cast only.
- No implicit precision/config coercion in merge or decode.
- Precision cast helpers are explicit API calls.

## 3. Capability parity contract

All surfaces must support:
- Build/train from numeric values.
- Add one value.
- Add multiple values.
- Add weighted value(s) (`values[i]` + `weights[i]`).
- Merge one digest.
- Merge multiple digests.
- `quantile`
- `cdf`
- `median`
- Explicit precision cast (`f32`/`f64`).
- `to_bytes` / `from_bytes`

Notes:
- Polars naming can stay surface-specific (`add_values`, `merge_tdigests`).
- New APIs should prefer common cross-surface names unless a surface constraint requires otherwise.
- CLI maps query capabilities to subcommands: `tdigest quantile`, `tdigest cdf`, `tdigest median`.
- CLI build/serialization flow is explicit: `tdigest build --to-digest ...`, then query via `--from-digest` (with optional additional training/merge input).

## 4. Normative semantics (target)

### 4.1 Training and add

- Empty training input is valid and returns an empty digest.
- Any non-finite value (`NaN`, `+inf`, `-inf`) is an error.
- Any null value (where representable) is an error.
- Adding an empty vector/list is a no-op success.

### 4.2 Queries

Quantile:
- Probe `q` must be finite and in `[0,1]`.
- `q=NaN`, `q=+/-inf`, `q<0`, `q>1` are errors.
- Empty digest + valid `q`:
  - Rust/Python/Java: `NaN`
  - Polars: `None`

CDF:
- Empty digest + any probe returns `NaN`.
- Non-empty digest:
  - finite probe -> finite value in `[0,1]`
  - `NaN` probe -> `NaN`
  - `-inf` probe -> `0.0`
  - `+inf` probe -> `1.0`
- Null probe(s) in Polars are errors.

Median:
- Empty digest:
  - Rust/Python/Java: `NaN`
  - Polars: `None`
- Non-empty digest returns finite value.

Shape behavior:
- Scalar probe in -> scalar out.
- Vector/list probe in -> vector/list out (surface-appropriate shape preservation).

### 4.3 Merge

- Precision mismatch merge is an error (`f32` vs `f64`).
- Config mismatch merge is an error:
  - `max_size`
  - `scale`
  - `singleton_policy`
- No implicit coercion in merge.

Canonical empty for `merge_all([])`:
- precision: `f64`
- `max_size=1000`
- `scale=K2`
- `singleton_policy=Use`

### 4.4 Serialization / `from_bytes`

- TDIG written by one surface must decode on all surfaces.
- `to_bytes` supports explicit version selection (`1|2|3`) where surfaced.
- Wire precision (`f32`/`f64`) must round-trip.
- Empty digest to/from bytes is valid.
- Empty blob bytes (`b""`) are invalid TDIG and must error.
- Null blobs (Polars) are errors.
- Mixed `f32`/`f64` blobs in one logical deserialize operation are errors unless the user explicitly normalizes first.

## 5. Error model (target)

Rust defines canonical behavior/errors. Bindings map to native exception classes.

Canonical classes:
- `InvalidTrainingData`
- `InvalidProbe`
- `InvalidScale`
- `IncompatibleMerge`
- `DecodeError`
- `ClosedHandle` (Java/JNI lifecycle)
- `InvariantViolation`

Surface mapping:
- Rust: `Result<T, TdError>`
- Python: `ValueError` / `TypeError`
- Java: `IllegalArgumentException` / `IllegalStateException`
- Polars: `ComputeError`

## 6. Architecture target

- Keep `src/tdigest/frontends.rs` as the shared semantic layer.
- Route all public operation paths through shared Rust behavior.
- Continue removing duplicated validation/business logic from:
  - `src/py.rs`
  - `src/jni.rs`
  - `src/polars_expr.rs`
  - `bindings/python/tdigest_rs/__init__.py`
- Expose explicit precision-cast APIs across surfaces.

## 7. Acceptance criteria

- `make build` passes.
- `make test` passes.
- Capability/coherence/serialization suites pass.
- No documented cross-surface contract drift.
- Binding layers are thinner with less duplicated validation logic.
