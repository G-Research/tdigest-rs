use crate::traits::FloatConst;
use anyhow::Result;
use num::Float;

/// Context struct to cache expensive logarithmic calculations for scale functions.
/// Precomputes factors that depend only on delta and n, eliminating repeated
/// logarithm evaluations in hot loops.
pub struct ScaleContext<T> {
    scale_factor: T,       // delta / base_factor (for log_scale)
    inverse_factor: T,     // base_factor / delta (for inverse_log_scale)
}

impl<T: Float + FloatConst> ScaleContext<T> {
    /// Create a new ScaleContext with precomputed factors.
    pub fn new(delta: T, n: usize) -> Self {
        let base_factor = (T::from(n).unwrap() / delta)
            .log(T::E)
            .mul_add(T::FOUR, T::TWENTYFOUR);

        Self {
            scale_factor: delta / base_factor,
            inverse_factor: base_factor / delta,
        }
    }

    /// Compute log_q_limit using cached factors (optimized version).
    #[inline]
    pub fn log_q_limit(&self, q0: T) -> T {
        // Equivalent to: inverse_log_scale(log_scale(q0, delta, n) + 1, delta, n)
        // but with precomputed factors
        let k = self.scale_factor * (q0 / (T::ONE - q0)).log(T::E) + T::ONE;
        (T::ONE + (-k * self.inverse_factor).exp()).recip()
    }
}

// Keep original functions for backward compatibility and non-hot-path uses
#[allow(dead_code)]
pub fn log_q_limit<T>(q0: T, delta: T, n: usize) -> Result<T>
where
    T: Float + FloatConst,
{
    inverse_log_scale(log_scale(q0, delta, n)? + T::ONE, delta, n)
}

#[allow(dead_code)]
pub fn inverse_log_scale<T>(k: T, delta: T, n: usize) -> Result<T>
where
    T: Float + FloatConst,
{
    let factor = (T::from(n).unwrap() / delta)
        .log(T::E)
        .mul_add(T::FOUR, T::TWENTYFOUR)
        / delta;
    Ok((T::ONE + (-k * factor).exp()).recip())
}

#[allow(dead_code)]
pub fn log_scale<T>(q: T, delta: T, n: usize) -> Result<T>
where
    T: Float + FloatConst,
{
    let factor = delta
        / (T::from(n).unwrap() / delta)
            .log(T::E)
            .mul_add(T::FOUR, T::TWENTYFOUR);
    Ok(factor * (q / (T::ONE - q)).log(T::E))
}
