//! Cumulative probabilities from ordered centroids. Each batch shares one array of
//! cumulative weights. Each probe uses a binary search over the centroid means.

use crate::tdigest::centroids::Centroid;
use crate::tdigest::precision::FloatLike;
use crate::tdigest::TDigest;
use ordered_float::FloatCore;
use rayon::iter::IndexedParallelIterator;
use rayon::prelude::*;

/// Crossover for parallel evaluation with Rayon.
///
/// Keep this conservative: Rayon setup has a fixed cost. Below this size a
/// scalar loop is typically faster; above it, parallelism wins.
const PAR_MIN: usize = 32_768;

impl<F: FloatLike + FloatCore> TDigest<F> {
    /// Estimate the rank of each probe. An exact centroid hit gets its midpoint rank.
    /// Between centroids, interpolate only the mass held by mixed centroids. In the
    /// tails, interpolate to the observed extrema. An empty digest or a NaN probe
    /// yields NaN. Probes strictly outside the observed range give zero or one.
    pub fn cdf(&self, vals: &[f64]) -> Vec<f64> {
        let cents = self.centroids();
        let n = cents.len();

        if n == 0 {
            return vec![f64::NAN; vals.len()];
        }
        if vals.is_empty() {
            return Vec::new();
        }

        let mut prefix: Vec<f64> = Vec::with_capacity(n);
        let mut run: f64 = 0.0;
        for c in cents {
            prefix.push(run);
            run += c.weight_f64();
        }
        let count_f64 = run;
        if count_f64 > 0.0 && count_f64 < f64::MIN_POSITIVE {
            // Half-weight arithmetic can underflow even when every stored
            // weight is positive. Only this rare path allocates a query copy.
            return self
                .normalized_subnormal_mass_for_queries(count_f64)
                .cdf(vals);
        }

        let min_v = self.min();
        let max_v = self.max();

        let rank = |&value: &f64| {
            if value.is_nan() {
                f64::NAN
            } else {
                cdf_at_val_fast(value, cents, &prefix, count_f64, min_v, max_v)
            }
        };
        if vals.len() >= PAR_MIN {
            vals.par_iter().with_min_len(4096).map(rank).collect()
        } else {
            vals.iter().map(rank).collect()
        }
    }
}

// A difference between finite opposite-sign endpoints can overflow. Halving
// first preserves the fraction without introducing an infinite denominator.
fn fraction_between(value: f64, left: f64, right: f64) -> f64 {
    let span = right - left;
    if span.is_infinite() {
        (value * 0.5 - left * 0.5) / (right * 0.5 - left * 0.5)
    } else {
        (value - left) / span
    }
}

/// Optimized evaluation kernel (reference semantics).
///
/// Exact hit → (prefix[idx] + 0.5*w)/N.
/// Left/right tails → guarded ramps using min/max and edge centroid half-weights.
/// Between centroids → center-to-center interpolation with **atomic** exclusion.
#[inline(always)]
fn cdf_at_val_fast<F: FloatLike + FloatCore>(
    val: f64,
    cents: &[Centroid<F>],
    prefix: &[f64],
    count_f64: f64,
    min_v: f64,
    max_v: f64,
) -> f64 {
    let n = cents.len();

    match cents.binary_search_by(|c| c.mean_f64().partial_cmp(&val).unwrap()) {
        // Exact centroid hit: midpoint semantics (half-weight).
        Ok(idx) => (prefix[idx] + 0.5 * cents[idx].weight_f64()) / count_f64,

        Err(idx) => {
            // Left of first centroid mean
            if idx == 0 {
                if val < min_v {
                    return 0.0;
                }
                let m0 = cents[0].mean_f64();
                let w0 = cents[0].weight_f64();
                let gap = m0 - min_v;
                if gap > 0.0 {
                    // A mixed edge carries half its mass between the data minimum
                    // and its mean. No unit sample can be assumed for weighted data:
                    // fixed masses here break monotonicity after scaling weights.
                    return fraction_between(val, min_v, m0) * (0.5 * (w0 / count_f64));
                } else {
                    // Degenerate (all equal); idx==0 implies val<=min.
                    return 0.0;
                }
            }

            // Right of last centroid mean
            if idx == n {
                if val > max_v {
                    return 1.0;
                }
                let mn = cents[n - 1].mean_f64();
                let wn = cents[n - 1].weight_f64();
                let gap = max_v - mn;
                if gap > 0.0 {
                    // Symmetric half-mass ramp, independent of the weight units.
                    let dq = (1.0 - fraction_between(val, mn, max_v)) * (0.5 * (wn / count_f64));
                    return 1.0 - dq;
                } else {
                    return 1.0;
                }
            }

            // Between centroids idx-1 and idx
            let li = idx - 1;
            let ri = idx;
            let ml = cents[li].mean_f64();
            let mr = cents[ri].mean_f64();
            let wl = cents[li].weight_f64();
            let wr = cents[ri].weight_f64();

            let gap = mr - ml;
            if gap <= 0.0 {
                // Pathological/too-close: fall back to midpoint mass to preserve monotonicity.
                let dw = 0.5 * (wl + wr);
                return (prefix[li] + dw) / count_f64;
            }

            // Atomic-aware exclusion: subtract w/2 for ANY atomic neighbor.
            let left_excl = if cents[li].is_atomic() { wl * 0.5 } else { 0.0 };
            let right_excl = if cents[ri].is_atomic() { wr * 0.5 } else { 0.0 };

            // Two atomics ⇒ dw_span == 0 ⇒ pure step.
            let dw_center = 0.5 * (wl + wr);
            let dw_span = dw_center - left_excl - right_excl;

            // Base mass at left center plus any left exclusion.
            let base = prefix[li] + wl * 0.5 + left_excl;
            let frac = fraction_between(val, ml, mr);

            (base + dw_span * frac) / count_f64
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::tdigest::{tdigest::TDigestBuilder, ScaleFamily};

    fn exact_ecdf_for_sorted(sorted: &[f64]) -> Vec<f64> {
        let n = sorted.len();
        if n == 0 {
            return Vec::new();
        }

        let nf = n as f64;
        let mut out = Vec::with_capacity(n);
        let mut i = 0usize;
        while i < n {
            let mut j = i + 1;
            while j < n && sorted[j] == sorted[i] {
                j += 1;
            }
            let mid = (i + j) as f64 * 0.5;
            let val = mid / nf;
            out.extend(std::iter::repeat_n(val, j - i));
            i = j;
        }
        out
    }

    #[test]
    fn cdf_small_max10_medn_100_simple() {
        let mut values: Vec<f64> = (-30..=69).map(|x| x as f64).collect();
        values[0] = -1e9; // extreme left tail
        values[99] = 1e9; // extreme right tail
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let td = TDigestBuilder::new()
            .max_size(10)
            .scale(ScaleFamily::Quad)
            .build()
            .unwrap()
            .merge_sorted(values.clone())
            .expect("no NaNs");
        let approx = td.cdf(&values);
        assert_eq!(approx.len(), values.len());

        // 1) bounds + 2) monotone
        for i in 0..approx.len() {
            let p = approx[i];
            assert!((0.0..=1.0).contains(&p), "cdf[{}]={} out of [0,1]", i, p);
            if i > 0 {
                assert!(approx[i] + 1e-12 >= approx[i - 1], "non-monotone at {}", i);
            }
        }

        // 3) tails hit ~1%/~99% (allow tiny slack for compression)
        let eps = 1e-6;
        assert!(
            approx[0] <= 0.01 + eps,
            "left tail too large: {}",
            approx[0]
        );
        assert!(
            approx[99] >= 0.99 - eps,
            "right tail too small: {}",
            approx[99]
        );

        // 4) middle rank (~median) should be roughly around 0.5
        let mid = 50;
        assert!(
            (0.49..=0.51).contains(&approx[mid]),
            "median sanity: {}",
            approx[mid]
        );
    }

    #[test]
    fn cdf_exact_with_enough_capacity() {
        use crate::tdigest::test_helpers::assert_exact;
        use rand::{rngs::StdRng, Rng, SeedableRng};

        const N: usize = 9_999;
        let mut rng = StdRng::seed_from_u64(42);
        let mut vals: Vec<f64> = (0..N)
            .map(|_| rng.random_range(0..N as u64) as f64)
            .collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let exact = exact_ecdf_for_sorted(&vals);
        let approx = TDigestBuilder::new()
            .max_size(N + 1)
            .build()
            .unwrap()
            .merge_sorted(vals.clone())
            .expect("no NaNs")
            .cdf(&vals);

        for (i, (&e, &a)) in exact.iter().zip(&approx).enumerate() {
            assert_exact(&format!("CDF[{i}]"), e, a);
        }
    }

    #[test]
    fn cdf_midpoint_ecdf_is_exact_at_training_values_under_capacity() {
        use crate::tdigest::test_helpers::assert_exact;

        // Sorted training data with ties; N is intentionally far below max_size.
        let vals = vec![-2.0, -2.0, -1.0, 0.0, 0.0, 0.0, 3.0, 7.0, 7.0];
        let n = vals.len();
        let td = TDigestBuilder::new()
            .max_size(1000)
            .build()
            .unwrap()
            .merge_sorted(vals.clone())
            .expect("no NaNs");

        assert!(n < td.max_size());

        let expected = exact_ecdf_for_sorted(&vals); // midpoint over ties
        let got = td.cdf(&vals);
        for (i, (&e, &g)) in expected.iter().zip(&got).enumerate() {
            assert_exact(&format!("CDF(midpoint exact @ training value) [{i}]"), e, g);
        }
    }

    #[test]
    fn cdf_between_two_atomic_centroids_is_flat_step() {
        use crate::tdigest::ScaleFamily;

        // Build two atomic centroids:
        // - left:  five identical zeros → atomic with w=5 at mean=0.0
        // - right: seven identical tens → atomic with w=7 at mean=10.0
        // N = 12 total
        let mut values: Vec<f64> = Vec::new();
        values.extend([0.0; 5]);
        values.extend([10.0; 7]);
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let td = TDigestBuilder::new()
            .max_size(64)
            .scale(ScaleFamily::K2)
            .build()
            .unwrap()
            .merge_sorted(values)
            .expect("no NaNs");

        assert_eq!(td.centroids().len(), 2, "expected two centroids (0 and 10)");
        assert!((td.centroids()[0].mean_f64() - 0.0).abs() < 1e-12);
        assert!((td.centroids()[1].mean_f64() - 10.0).abs() < 1e-12);
        assert!((td.centroids()[0].weight_f64() - 5.0).abs() < 1e-12);
        assert!((td.centroids()[1].weight_f64() - 7.0).abs() < 1e-12);

        // Probes strictly between the two means.
        let probes = [1.0, 5.0, 9.0];

        // Expected: with correct atomic handling, there is **no** interpolative mass
        // contributed by either atomic neighbor (left_excl = wl/2, right_excl = wr/2),
        // so the between-mean CDF is a **flat step** at prefix_left + wl (all of left mass).
        // Here: prefix_left = 0, wl = 5 ⇒ expected = 5/12.
        let expected_between = 5.0 / 12.0;

        let out = td.cdf(&probes);
        assert_eq!(out.len(), probes.len());

        for (i, &p) in out.iter().enumerate() {
            assert!(
                (p - expected_between).abs() <= 1e-9,
                "cdf({}) = {}, expected flat step {} between atomic centroids",
                probes[i],
                p,
                expected_between
            );
        }

        // A probe exactly at zero counts half the five observations tied there.
        let exact = td.cdf(&[0.0])[0];
        assert!(
            (exact - (2.5 / 12.0)).abs() <= 1e-12,
            "exact hit at left mean should be midpoint mass"
        );
    }
}
