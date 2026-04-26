use serde::{Deserialize, Serialize};

/// Policy controlling how the compressor handles **data-true singletons**
/// (points with `weight == 1`) and **piles** (multiple identical values collapsed to one mean).
///
/// # Overview
/// This setting determines whether and how the compressor preserves singleton or edge values
/// during compression. It affects how capacity (`max_size`) is allocated across the digest.
///
/// # Variants
/// - [`SingletonPolicy::Off`]: Treat all centroids uniformly.
///   No special handling of tails or singletons; the full digest is compressed within
///   the global capacity.
///
/// - [`SingletonPolicy::Use`]: Respect true singletons and piles during compression,
///   but still apply the global capacity to the **entire** digest.
///   In the final stage (Stage 6), the digest may still be bucketized back to `max_size`.
///
/// - [`SingletonPolicy::UseWithProtectedEdges`]: Preserve up to `k` singletons or piles
///   at each edge (tails) outside the main compression region,
///   effectively granting the tails extra capacity.
///
/// # Notes
/// A *singleton* is a data fact: either a point with `weight == 1` or a pile (identical values)
/// with `weight > 1` that shares the same mean.
/// Protecting them can improve tail accuracy at the cost of additional memory.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SingletonPolicy {
    /// Treat all centroids uniformly — no special handling of singletons or edges.
    Off,
    /// Respect singletons and piles, but apply uniform capacity limits.
    #[default]
    Use,
    /// Preserve up to `k` singletons/piles per edge outside the main capacity.
    UseWithProtectedEdges(usize),
}
