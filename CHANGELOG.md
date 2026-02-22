# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Changed
- Snapshot overwrite of implementation to the unified Rust core + Python/Polars/CLI/Java/JNI stack.
- Project identity normalized to `tdigest-rs` / `tdigest_rs` across code, docs, and packaging metadata.
- Python API removed legacy `delta` configuration in favor of `max_size` + `scale` and keeps strict non-finite validation.
- Java namespace migrated to `com.gresearch.tdigest` with aligned JNI symbol names.
- CI matrix and dependency setup were adjusted for portability:
  - removed unsupported `macos-13` jobs from Rust/Python workflows,
  - fixed Python CI mypy config discovery path,
  - removed `patchelf` extras from default Python build/dev dependencies to avoid Windows sync failures.
- Python CI test jobs now prebuild integration-test artifacts (`target/debug/tdigest` and Gradle `classes` with bundled natives) before running pytest.
- Moved `jemalloc-ctl` to Linux GNU-only dev dependencies to prevent Windows Rust test failures.
- Coherence test path detection was hardened for Windows (`tdigest.exe`, `gradlew.bat`, normalized native arch tags).

### Added
- Python compatibility methods/properties:
  - `from_means_weights(...)`
  - `update(...)` returning a new digest
  - `merge(...)` returning a new digest
  - `means`, `weights`, `__len__`
  - `to_dict()` / `from_dict()` with new schema and legacy schema support
- Rust core + Python binding `trimmed_mean(lower, upper)`.

## [0.2.4] - 2026-02-16

### Changed
- Release/version transition from `0.2.3` to `0.2.4`.
- **Breaking (CLI)**: Rust CLI moved from `--cmd ...` to subcommands: `tdigest build`, `tdigest quantile`, `tdigest cdf`, `tdigest median`.
- Rust CLI now has explicit build/serialization flow (`build --to-digest`, query with `--from-digest`) and keeps optional `--to-digest` on query commands.
- Rust CLI structured ingest paths are standardized across training/probes (`text|csv|json|ndjson` + optional column selectors).
- API coherence CLI tests were updated to the new subcommand interface.
- `README.md` was reorganized for a cleaner user flow (examples directly under features; community/license moved to bottom).
- Publish/release setup details were moved from `README.md` into dedicated `PUBLISH.md`.
- `README.md` features section now includes the full detailed cross-surface capability list again.
- `README.md` future-improvements bullets were restored to weight-scaling and scale-suggestion items.
- Added a hard panic guard for centroid weight overflow when converting to storage precision (temporary fail-fast behavior).
- `AGENTS.md` now explicitly incorporates local production-workload context guidance for tuning decisions.
- `typical_usecase.md` is now ignored via `.gitignore` for local-only workload notes.
- Added focused Rust tests and design docs clarifying under-capacity exactness: CDF at training values follows exact midpoint ECDF, and quantile is exact at mid-ranks.

## [0.2.3] - 2026-02-14

### Changed
- Release/version transition from `0.2.2` to `0.2.3`.
- `bindings/java/build.gradle` now declares `sourcesJar` depends on `copyNative`, fixing Gradle task-graph validation failure during Maven preflight publish.
- `Cargo.lock` is synchronized for the release version so `cargo publish --locked` can run successfully in CI.
- `README.md` release tag examples now use `v0.2.3`.

## [0.2.2] - 2026-02-14

### Added
- Version-controlled GitHub ruleset specs:
  - `.github/rulesets/master.json`
  - `.github/rulesets/tags_vstar.json`
- `scripts/apply_github_rulesets.sh` to apply/update repository merge settings and rulesets via `gh api`.

### Changed
- Release/version transition from `0.2.1` to `0.2.2`.
- CI (`.github/workflows/ci.yml`) now runs `Lint` and `Build and test` as parallel jobs for faster feedback.
- CI now caches uv and Gradle artifacts in addition to Rust caches to reduce rebuild time.
- Release workflows now verify that release tags point to commits reachable from `origin/master`.
- Cargo release now runs `cargo publish --dry-run --locked --package tdigest-rs` before publish.
- Maven release now runs a preflight `publishToMavenLocal` before repository publish.
- PyPI release now runs `twine check` on built distributions before publish.
- `README.md` and `CONTRIBUTING.md` now reference repository protection setup guidance.
- `Makefile` `setup-hooks` now installs pre-commit via `.venv/bin/python -m pre_commit` and auto-recovers missing `pre_commit` in `.venv` by syncing `bindings/python` dependencies.
- `.github/REPO_SETTINGS.md` now documents a configuration-as-code workflow for GitHub rulesets.
- `master` protection policy is now no-force-push/no-deletion with direct pushes allowed; PR merges remain squash-only via repository merge settings.
- Cargo release now validates `CARGO_REGISTRY_TOKEN` before running `cargo publish` so missing credentials fail fast.
- Maven release now validates required publish credentials before running preflight publish tasks.

### Fixed
- Track `bindings/java/gradle/wrapper/gradle-wrapper.jar` so GitHub Actions can run Gradle wrapper (`org.gradle.wrapper.GradleWrapperMain` available in CI).
- Python packaging metadata now uses a local `bindings/python/README.md`, fixing PyPI release sdist failures caused by `../../README.md` archive paths.

## [0.2.1] - 2026-02-14

### Added
- Open-source governance and community docs:
  - `CONTRIBUTING.md`
  - `CODE_OF_CONDUCT.md`
  - `SECURITY.md`
- Minimal GitHub bug-reporting templates:
  - `.github/ISSUE_TEMPLATE/bug_report.yml`
  - `.github/ISSUE_TEMPLATE/config.yml`
- Minimal GitHub Actions CI workflow:
  - `.github/workflows/ci.yml`
  - push trigger covers `master`
- Tag-triggered release workflows:
  - `.github/workflows/release_pypi.yml`
  - `.github/workflows/release_cargo.yml`
  - `.github/workflows/release_maven.yml`

### Changed
- Release/version transition from `0.2.0` to `0.2.1`.
- `CODE_OF_CONDUCT.md` simplified to a minimal, practical policy.
- Security reporting flow moved to GitHub private vulnerability reporting (`security/advisories/new`).
- `README.md` now includes links to community and support docs.
- Removed old mixed CI/release workflow (`.github/workflows/publish_to_pypi.yml`) in favor of a clean CI-only start.
- `.github/workflows/ci.yml` now syncs Python dependencies directly instead of running hook-installing setup steps.
- `Makefile` `py-test` now invokes `uv run --project bindings/python python -m pytest ...` for robust local/CI execution.
- Added local `make publish` orchestration with per-registry publish targets (`publish-pypi`, `publish-cargo`, `publish-maven`) and dry-run mode via `PUBLISH_DRY_RUN=1`.
- `Cargo.toml` now includes crate publish metadata (description/license/repository/homepage/documentation/readme/keywords/categories) for crates.io readiness.
- `bindings/java/build.gradle` now supports repository-configured Maven publishing via environment variables, optional in-memory PGP signing, and publishes sources/javadoc jars with SCM/license metadata.
- `README.md` now documents GitHub release workflow setup for PyPI, Cargo, and Maven publishing.
- `AGENTS.md` now requires moving `## [Unreleased]` notes into a versioned section before pushing release tags.

## [0.2.0] - 2026-02-14

### Added
- Rust weighted ingestion APIs:
  - `TDigest::add_weighted(value, weight)`
  - `TDigest::add_weighted_many(values, weights)`
  - `TDigest::merge_weighted_unsorted(values, weights)`
  - `TDigest::from_weighted_unsorted(values, weights, max_size)`
- Frontend weighted-ingest APIs:
  - Python: `TDigest.add_weighted(values, weights)`
  - Java/JNI: `TDigest.addWeighted(...)`
  - Polars plugin: `add_weighted_values(...)`
- Explicit precision casting in core/frontend paths:
  - `TDigest::cast_precision::<f32|f64>()`
  - `FrontendDigest::cast_precision(...)`
- Public cast-precision APIs on all user surfaces:
  - Python `TDigest.cast_precision(...)`
  - Java `TDigest.castPrecision(...)`
  - Polars plugin `cast_precision(...)`
- Explicit wire-version encode controls:
  - Python `to_bytes(version=...)`
  - Java `toBytes(version)`
  - Polars plugin `to_bytes(..., version=...)`
- TDIG v3 wire format support in codec:
  - default encoder now writes v3
  - decoder supports v1/v2/v3
  - v3 adds flags + header length + explicit payload precision code + optional 4-byte checksum
  - v2/v3 preserve fractional centroid weights and centroid kind

### Changed
- Release/version transition from `0.1.13` to `0.2.0`.
- Design docs split and deepened (`api_design.md`, `tdigest_design.md`, `comparison_design.md`) with stricter contract/current/comparison separation.
- `comparison_design.md` gap analysis updated to reflect cast API completion and TDIG v3/version-controls progress.
- Repository process/tooling now enforces changelog updates via pre-commit (`scripts/check_changelog_update.sh` + `.pre-commit-config.yaml`).
- `make setup` now installs pre-commit hooks with repo-local caches (`.pre-commit-cache`, `.uv-cache`) to avoid host cache permission issues.
- Agent workflow guidance tightened (`AGENTS.md`) to require changelog updates, design-doc review/updates, and README maintenance.
- `Makefile` now defaults `UV_CACHE_DIR` to repo-local `.uv-cache` for more reliable sandboxed/dev runs.
- Core digest merge switched from concat+sort run materialization to heap-stream k-way merge in `KWayCentroidMerge::from_runs`, reducing peak merge allocations and improving heavy merge throughput.
- Raw ingest merge (`MergeByMean::from_centroids_and_values`) switched to a streaming two-way iterator, removing prebuilt merged-buffer materialization in `merge_sorted`.

### Testing / Contracts
- Cross-surface contract/coherence test suite expanded and consolidated.
- Added wire migration tests for mixed v1/v2 blobs and explicit versioned encoding on Python/Java paths.
- Added merge strategy proof tests:
  - randomized correctness parity (`heap` vs historical `concat+sort`)
  - ignored heavy benchmark/proof test for CPU + memory + precision comparison (current profile: `n=10M`, `shards=40`, `max_size=1000`)
  - historical concat+sort comparator retained as test-only baseline (`#[cfg(test)]`)
- Full project gates continue to pass (`make build`, `make test`).

## [0.1.13] - 2026-02-09

### Changed
- Version bump from `0.1.12` to `0.1.13` (no functional delta in bump commit).

## [0.1.12] - 2026-02-09

### Added
- Java API support for digest merge and serialization interop.
- Java value-merge APIs and Gradle/Make wiring for Java-side tests.

### Changed
- Release/version transition from `0.1.11` to `0.1.12`.

## [0.1.11] - 2025-12-06

### Added
- Python support for `merge` and `merge_all` digest operations.

## [0.1.10] - 2025-11-24

### Added
- Digest encoding/decoding support (cross-surface byte serialization path introduced).

## [0.1.9] - 2025-11-07

### Added
- Polars behavior support for empty group-by scenarios.

## [0.1.8] - 2025-11-04

### Changed
- Introduced shared `frontends.rs` layer and dried up duplicated binding logic.
- Improved CDF handling around atomic-centroid semantics.

### Docs
- README improvements.

## [0.1.7] - 2025-11-03

### Testing / Contracts
- Introduced unified API coherence test direction across surfaces.

## [0.1.6] - 2025-10-31

### Changed
- Tightened non-finite handling and edge-policy behavior.

## [0.1.5] - 2025-10-30

### Added
- End-to-end float precision behavior (`f32`/`f64`) across surfaces.

## [0.1.4] - 2025-10-30

### Added
- List-oriented CDF/quantile expression support in Polars workflows.

## [0.1.3] - 2025-10-30

### Changed
- Large refactor pass across Gradle/bindings/docs/build setup.

## [0.1.2] - 2024-06-14

### Changed
- Library naming and packaging adjustments (`polars_tdigest` era alignment).

## [0.1.1] - 2024-06-13

### Added
- First `0.1.x` release line established.
- Initial packaging/release plumbing for early distribution.
