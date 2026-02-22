#![allow(clippy::module_inception)]
//! t-digest core types and public API.
//!
//! - [`TDigest`] — main data structure and builder.
//! - **Compression** — internal compressor enforces a k-limit ([`ScaleFamily`]) while
//!   preserving extremes and discrete *singleton piles*.
//! - **Scales** — see [`ScaleFamily`] for available k-mappings.
//! - **CDF** — see [`TDigest::cdf`] for exact semantics and guarantees.

pub mod cdf;
pub mod centroids;
pub mod codecs;
pub mod compressor;
pub mod frontends;
pub mod merges;
pub mod precision;
pub mod quantile;
pub mod scale;
pub mod singleton_policy;
pub mod tdigest;
pub mod wire;

// Re-exports for a clean public surface
pub use self::precision::Precision;
pub use self::scale::ScaleFamily;
pub use self::tdigest::{DigestStats, TDigest, TDigestBuilder};
pub use self::wire::{WirePrecision, WireVersion};

#[cfg(test)]
mod test_helpers;
