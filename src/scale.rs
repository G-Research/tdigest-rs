use crate::traits::FloatConst;
use anyhow::Result;
use num::Float;

pub fn log_q_limit<T>(q0: T, delta: T, n: usize) -> Result<T>
where
    T: Float + FloatConst,
{
    inverse_log_scale(log_scale(q0, delta, n)? + T::ONE, delta, n)
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scale_function_properties() {
        let delta = 100.0_f64;
        let n = 10000;

        for q in [0.1, 0.25, 0.5, 0.75, 0.9] {
            let k = log_scale(q, delta, n).unwrap();
            let q_recovered = inverse_log_scale(k, delta, n).unwrap();

            let error = (q - q_recovered).abs() / q;
            assert!(
                error < 1e-10,
                "Scale functions not inverse at q={}: recovered {}, error {:.2e}",
                q,
                q_recovered,
                error
            );
        }
    }

    #[test]
    fn test_scale_function_monotonicity() {
        let delta = 50.0_f64;
        let n = 5000;

        let quantiles = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95];
        let mut prev_k = f64::NEG_INFINITY;

        for &q in &quantiles {
            let k = log_scale(q, delta, n).unwrap();
            assert!(
                k > prev_k,
                "log_scale not monotonic: at q={}, k={:.4} <= prev_k={:.4}",
                q,
                k,
                prev_k
            );
            prev_k = k;
        }
    }

    #[test]
    fn test_log_q_limit_properties() {
        let delta = 100.0_f64;
        let n = 10000;

        for q0 in [0.1, 0.3, 0.5, 0.7, 0.9] {
            let q_limit = log_q_limit(q0, delta, n).unwrap();

            assert!(
                q_limit > q0,
                "log_q_limit({}) = {} should be > {}",
                q0,
                q_limit,
                q0
            );

            let ratio = q_limit / q0;
            assert!(
                ratio < 2.0,
                "log_q_limit ratio too large: q_limit/q0 = {:.3}",
                ratio
            );
        }
    }

    #[test]
    fn test_scale_function_boundary_behavior() {
        let delta = 100.0_f64;
        let n = 10000;

        let q_small = 0.001;
        let k_small = log_scale(q_small, delta, n).unwrap();
        assert!(
            k_small.is_finite() && k_small < 0.0,
            "log_scale should be finite and negative near q=0, got {}",
            k_small
        );

        let q_large = 0.999;
        let k_large = log_scale(q_large, delta, n).unwrap();
        assert!(
            k_large.is_finite() && k_large > 0.0,
            "log_scale should be finite and positive near q=1, got {}",
            k_large
        );

        let q_small_recovered = inverse_log_scale(k_small, delta, n).unwrap();
        let q_large_recovered = inverse_log_scale(k_large, delta, n).unwrap();

        assert!((q_small - q_small_recovered).abs() < 1e-6);
        assert!((q_large - q_large_recovered).abs() < 1e-6);
    }

    #[test]
    fn test_delta_effect_on_scale() {
        let n = 10000;
        let q = 0.5;

        let small_delta = 20.0_f64;
        let large_delta = 200.0_f64;

        let _k_precise = log_scale(q, small_delta, n).unwrap();
        let _k_coarse = log_scale(q, large_delta, n).unwrap();

        let q_limit_precise = log_q_limit(q, small_delta, n).unwrap();
        let q_limit_coarse = log_q_limit(q, large_delta, n).unwrap();

        let step_precise = q_limit_precise - q;
        let step_coarse = q_limit_coarse - q;

        assert!(step_coarse < step_precise,
            "Larger delta should create smaller quantile steps (more permissive clustering): {:.6} vs {:.6}",
            step_coarse, step_precise);
    }
}
