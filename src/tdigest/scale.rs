use serde::{Deserialize, Serialize};

/// Scale families define the q→k mapping that controls compression density.
///
/// **Used in Stage 3 (k-limit merge).**
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")] // accept "quad","k1","k2","k3"
#[derive(Default)]
pub enum ScaleFamily {
    /// Piecewise-quadratic tail-friendly scale.
    Quad,
    /// k1: arcsine scale.
    K1,
    /// k2: logistic scale (DEFAULT).
    #[default]
    K2,
    /// k3: double-log scale.
    K3,
}

#[inline]
pub(crate) fn clamp(v: f64, lo: f64, hi: f64) -> f64 {
    v.max(lo).min(hi)
}

/// Family-aware `q → k` mapping. `d` is the scale denominator (≈ `max_size`).
///
/// This mapping shapes cluster budget along the distribution:
/// more resolution near tails for some scales, more uniform for others.
/// Consumed by Stage **3** to evaluate the Δk ≤ 1 condition.
#[inline]
pub(crate) fn q_to_k(q: f64, d: f64, family: ScaleFamily) -> f64 {
    use std::f64::consts::{LN_2, PI};
    let eps = 1e-15;
    let qq = clamp(q, eps, 1.0 - eps);
    match family {
        // Piecewise-quadratic
        ScaleFamily::Quad => {
            // Inverse of: r=k/d; q = 2r^2 (r<0.5), else 1-2(1-r)^2
            let r = if qq < 0.5 {
                (qq * 0.5).sqrt()
            } else {
                1.0 - ((1.0 - qq) * 0.5).sqrt()
            };
            d * r
        }
        // k1: arcsine scale
        ScaleFamily::K1 => {
            let s = (2.0 * qq - 1.0).clamp(-1.0, 1.0).asin();
            (d / (2.0 * PI)) * s
        }
        // k2: logistic scale
        ScaleFamily::K2 => {
            let s = (qq / (1.0 - qq)).ln();
            (d / (4.0 * LN_2)) * s
        }
        // k3: double-log
        ScaleFamily::K3 => {
            let a = (1.0 / (1.0 - qq)).ln(); // ln(1/(1-q))
            let b = (1.0 / qq).ln(); // ln(1/q)
            let ratio = (a / b).max(eps);
            (d / 4.0) * ratio.ln()
        }
    }
}
