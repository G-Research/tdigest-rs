use serde::{Deserialize, Serialize};

/// Controls which atomic centroids the compressor reserves at the tails.
/// An atomic centroid contains observations at one value, including repeated
/// observations. A centroid that combines distinct values is mixed.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SingletonPolicy {
    /// Compress all centroids within the configured capacity.
    Off,
    /// Reserve each endpoint within the configured capacity when there is room for an interior.
    #[default]
    Use,
    /// Preserve up to `k` singletons/piles per edge outside the main capacity.
    UseWithProtectedEdges(usize),
}
