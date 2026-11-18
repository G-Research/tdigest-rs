# Changelog

## Unreleased

* Upgrade the Rust toolchain to 1.97, update ethnum to its compatible 1.5.3 patch, and pin cargo-edit 0.13.14 with locked dependencies for release version updates.

* Compare legacy arithmetic NaNs portably while checking all other bits against 0.1.0. Cover two-singleton and zero-heavy shard merges in the published-wheel fixture.

* Deprecate Python delta and default constructors and restore the published 0.1.0 arithmetic, merge order and two-array pickle semantics. Empty digests can be copied and pickled. Weight overflow and unequal input lengths raise `ValueError`. Use `max_size=` for the capacity digest.

* Preserve the original test package as an exact rename in the release diff.

* Add grouped Polars digests and CDF queries with shared Rust, Python and Java implementations.
* Preserve the original Python calling convention and support portable TDIG state in capacity mode.
* Rebuild changed delta merges in Rust and return the updated Python digest directly from native mutation methods.
* Introduce the library with a concise feature list covering Float16 input, parallel CDF queries, and TDIG serialization across languages. Include runnable Python, Polars, Rust, CLI, and Java examples and document the legacy Python API separately.
* Document observed example outputs, quantile interpolation and edge protection, and release-wheel comparisons at matched centroid counts.

See the [1.0 release notes](RELEASE_NOTES_1.0.0.md) for compatibility cases, measurements and reproduction commands. [Earlier development history](https://github.com/ingolfured/tdigest-rs/blob/980f0be3f795ae703b86bdf3da8c77f2f7ed752b/CHANGELOG.md)
