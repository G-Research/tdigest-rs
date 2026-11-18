use serde::{Deserialize, Serialize};

/// Map rank to a scale coordinate that limits how much mass a centroid can hold.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum ScaleFamily {
    /// Piecewise quadratic scale.
    Quad,
    /// Arcsine scale.
    K1,
    /// Logistic scale with fixed normalization `d / (4·ln 2) · ln(q/(1−q))`.
    ///
    /// Matches Dunning's Java `K_2_NO_NORM`. The normalizer is independent of sample size.
    #[default]
    K2,
    /// Normalized logistic scale `δ / (4·ln(n/δ) + 24) · ln(q/(1−q))`.
    ///
    /// Matches Dunning's Java `K_2`. Delta mode uses the number of centroids
    /// entering the current merge pass for `n`.
    K2Norm,
    /// Double logarithmic scale.
    K3,
}

#[inline]
pub(crate) fn clamp(v: f64, lo: f64, hi: f64) -> f64 {
    v.max(lo).min(hi)
}

#[cfg(test)]
thread_local! {
    pub(super) static K2_LOG_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

pub(super) const K2_DENOMINATOR: f64 = 4.0 * std::f64::consts::LN_2;
const RANK_EPS: f64 = 1e-15;

#[inline]
pub(super) fn k2_log_odds(q: f64) -> f64 {
    #[cfg(test)]
    K2_LOG_CALLS.with(|calls| calls.set(calls.get() + 1));
    let q = clamp(q, RANK_EPS, 1.0 - RANK_EPS);
    (q / (1.0 - q)).ln()
}

/// Only K2Norm depends on the number of centroids entering the merge pass.
#[inline]
pub(crate) fn q_to_k(q: f64, d: f64, family: ScaleFamily, n: f64) -> f64 {
    use std::f64::consts::PI;
    let eps = RANK_EPS;
    let qq = clamp(q, eps, 1.0 - eps);
    match family {
        ScaleFamily::Quad => {
            // Inverse of: r=k/d; q = 2r^2 (r<0.5), else 1-2(1-r)^2
            let r = if qq < 0.5 {
                (qq * 0.5).sqrt()
            } else {
                1.0 - ((1.0 - qq) * 0.5).sqrt()
            };
            d * r
        }
        ScaleFamily::K1 => {
            let s = (2.0 * qq - 1.0).clamp(-1.0, 1.0).asin();
            (d / (2.0 * PI)) * s
        }
        ScaleFamily::K2 => (d / K2_DENOMINATOR) * k2_log_odds(q),
        // Paper eq (8): k(q) = δ / (4·ln(n/δ) + 24) · ln(q/(1-q)).
        // Normalizer constants and size-bound rationale: Dunning,
        // "The Size of a t-Digest", section 5, https://arxiv.org/abs/1903.09921.
        ScaleFamily::K2Norm => {
            let factor = d / ((n / d).ln().mul_add(4.0, 24.0));
            factor * (qq / (1.0 - qq)).ln()
        }
        ScaleFamily::K3 => {
            let a = (1.0 / (1.0 - qq)).ln(); // ln(1/(1-q))
            let b = (1.0 / qq).ln(); // ln(1/q)
            let ratio = (a / b).max(eps);
            (d / 4.0) * ratio.ln()
        }
    }
}
