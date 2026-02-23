//! Centroid representation for TDigest.
//!
//! This module defines a compact centroid type that stores:
//! - `mean`   in generic `F` (`f32` or `f64`)
//! - `weight` in generic `F`
//! - a `Kind` flag: `Atomic` (all identical inputs at this mean) or `Mixed`.
//!
//! ## Semantics
//! - **Atomic** means the centroid represents *identical* samples at exactly the same
//!   value (the centroid mean). It can be:
//!   - an **atomic unit** (weight == 1), or
//!   - an **atomic pile** (weight > 1), both represented by `Kind::Atomic`.
//! - **Mixed** means the centroid arises from merging across different means or
//!   from synthetic aggregation (e.g., bucketization).
//!
//! The codebase should rely on `is_atomic()` + `weight_f64()` to drive behavior.
//! For convenience and clarity, helpers `is_atomic_unit()` / `is_atomic_pile()`
//! are provided. A small **compatibility shim** keeps
//! `new_singleton_f64(..)` and `is_singleton()` working, mapping to the new model.

use ordered_float::FloatCore;

use crate::tdigest::precision::FloatLike;

/// Centroid kind flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// All composing items are identical (same value as the centroid mean).
    /// Weight may be 1 (atomic unit) or >1 (atomic pile).
    Atomic,
    /// Composed from different values or synthetic aggregation.
    Mixed,
}

/// A centroid stores the mean and total weight of a cluster (in `F`), and a kind flag.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Centroid<F: FloatLike + FloatCore> {
    mean: F,
    weight: F,
    kind: Kind,
}

impl<F: FloatLike + FloatCore> Centroid<F> {
    #[inline]
    fn checked_weight(weight: f64) -> F {
        assert!(
            weight.is_finite() && weight > 0.0,
            "centroid weight must be finite and > 0"
        );
        let w = F::from_f64(weight);
        let w64 = w.to_f64();
        assert!(
            w64.is_finite() && w64 > 0.0,
            "centroid weight overflow while converting to storage precision"
        );
        w
    }

    /* ---------------------------------------------------------------------
     * Constructors
     * ------------------------------------------------------------------ */

    /// Create an **atomic unit** centroid (exact single sample).
    #[inline]
    pub fn new_atomic_unit_f64(mean: f64) -> Self {
        Self {
            mean: F::from_f64(mean),
            weight: F::from_f64(1.0),
            kind: Kind::Atomic,
        }
    }

    /// Create an **atomic** centroid with arbitrary positive weight (>0).
    ///
    /// Use this for both units (w=1) and piles (w>1) that remain atomic.
    #[inline]
    pub fn new_atomic_f64(mean: f64, weight: f64) -> Self {
        Self {
            mean: F::from_f64(mean),
            weight: Self::checked_weight(weight),
            kind: Kind::Atomic,
        }
    }

    /// Create a **mixed** centroid with positive weight (>0).
    #[inline]
    pub fn new_mixed_f64(mean: f64, weight: f64) -> Self {
        Self {
            mean: F::from_f64(mean),
            weight: Self::checked_weight(weight),
            kind: Kind::Mixed,
        }
    }

    /// Compatibility shim: historically called a "singleton".
    ///
    /// - If `weight <= 1`, we emit an **atomic unit** (weight coerced to 1).
    /// - If `weight > 1`, we emit an **atomic (pile)**.
    ///
    /// Prefer `new_atomic_unit_f64` / `new_atomic_f64` in new code.
    #[inline]
    pub fn new_singleton_f64(mean: f64, weight: f64) -> Self {
        if weight <= 1.0 {
            Self::new_atomic_unit_f64(mean)
        } else {
            Self::new_atomic_f64(mean, weight)
        }
    }

    /* ---------------------------------------------------------------------
     * Accessors
     * ------------------------------------------------------------------ */

    #[inline]
    pub fn mean(&self) -> F {
        self.mean
    }

    #[inline]
    pub fn weight(&self) -> F {
        self.weight
    }

    #[inline]
    pub fn mean_f64(&self) -> f64 {
        self.mean.to_f64()
    }

    #[inline]
    pub fn weight_f64(&self) -> f64 {
        self.weight.to_f64()
    }

    #[inline]
    pub fn kind(&self) -> Kind {
        self.kind
    }

    /* ---------------------------------------------------------------------
     * Kind helpers (preferred in new code)
     * ------------------------------------------------------------------ */

    /// Returns `true` for both atomic unit (w=1) and atomic pile (w>1).
    #[inline]
    pub fn is_atomic(&self) -> bool {
        matches!(self.kind, Kind::Atomic)
    }

    /// `true` iff atomic and weight == 1.
    #[inline]
    pub fn is_atomic_unit(&self) -> bool {
        self.is_atomic() && self.weight_f64() == 1.0
    }

    /// `true` iff atomic and weight > 1.
    #[inline]
    pub fn is_atomic_pile(&self) -> bool {
        self.is_atomic() && self.weight_f64() > 1.0
    }

    /// `true` iff centroid is mixed.
    #[inline]
    pub fn is_mixed(&self) -> bool {
        matches!(self.kind, Kind::Mixed)
    }

    /* ---------------------------------------------------------------------
     * Back-compat helpers (avoid in new code)
     * ------------------------------------------------------------------ */

    /// Historical predicate retained for minimal churn.
    ///
    /// This maps to **atomic** in the new model.
    #[inline]
    pub fn is_singleton(&self) -> bool {
        self.is_atomic()
    }
}

#[cfg(test)]
mod tests {
    use super::Centroid;

    #[test]
    #[should_panic(expected = "centroid weight overflow while converting to storage precision")]
    fn f32_centroid_weight_overflow_panics() {
        let _ = Centroid::<f32>::new_atomic_f64(0.0, f64::MAX);
    }
}

/* -------------------------------------------------------------------------
 * Utilities
 * ---------------------------------------------------------------------- */

/// Verify strict increasing order by centroid mean.
#[inline]
pub fn is_sorted_strict_by_mean<F: FloatLike + FloatCore>(cs: &[Centroid<F>]) -> bool {
    // We require strictly increasing means to avoid ambiguous interpolation.
    for w in cs.windows(2) {
        match w[0].mean_f64().partial_cmp(&w[1].mean_f64()) {
            Some(std::cmp::Ordering::Less) => {}
            _ => return false,
        }
    }
    true
}
