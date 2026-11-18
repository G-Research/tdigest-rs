#![allow(clippy::module_inception)]
//! t-digest core types and public API.
//!
//! The compressor uses a [`ScaleFamily`] to group centroids and a
//! singleton policy to preserve selected values at the tails. See
//! [`TDigest::cdf`] for the rank assigned to ties.

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
pub(crate) mod summation;
pub mod tdigest;
pub mod wire;

pub use self::precision::Precision;
pub use self::scale::ScaleFamily;
pub use self::tdigest::{DigestStats, TDigest, TDigestBuilder};
pub use self::wire::{WirePrecision, WireVersion};

#[cfg(test)]
mod test_helpers;
