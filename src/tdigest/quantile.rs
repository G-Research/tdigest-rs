//! Quantile evaluation for `TDigest`.
//!
//! This module implements `TDigest::quantile(q)` using the same
//! **half-weight bracketing** semantics as the reference MergingDigest:
//!
//! - **Index mapping**: map `q ∈ [0,1]` to a target cumulative weight index
//!   `i = q·N`, where `N = ∑w` (total weight).
//! - **Center-to-center spans**: treat each centroid's *center* as located at
//!   half its weight from its left boundary. Between adjacent centroids, the
//!   span weight is `(w_left + w_right)/2`.
//! - **Edges**: probabilities 0 and 1 return extrema; beyond the outer centroid
//!   centers, retain atomic mass or interpolate toward the data extrema.
//! - **Atomic mass**: strictly inside a point mass, return its mean. This applies
//!   to all positive weights, including fractional weights.
//! - **Interpolation**: interpolate linearly in weight between centers after
//!   excluding each atomic neighbor's half-weight, mirroring the CDF.
//!
//! # Guarantees
//! - The result is **monotone** in `q`.
//! - `quantile(0.0) == min()` and `quantile(1.0) == max()`.
//! - With sufficient capacity (no compression), results match **exact order
//!   statistics** at mid-ranks `((i+0.5)/N)`.
//!
//! # Edge cases (explicit semantics)
//! - **Empty digest** → **`NaN`**.
//! - **`q` is NaN** → **`NaN`**.
//!
//! See also: [`TDigest::cdf`] for CDF semantics, which pair with this
//! quantile implementation via the same center/half-weight rules.

use crate::tdigest::precision::FloatLike;
use crate::tdigest::TDigest;
use ordered_float::FloatCore;

// Convex interpolation without overflowing a finite opposite-sign span.
fn interpolate_value(left: f64, right: f64, fraction: f64) -> f64 {
    if fraction <= 0.0 {
        return left;
    }
    if fraction >= 1.0 {
        return right;
    }
    if left.is_sign_negative() == right.is_sign_negative() {
        left + (right - left) * fraction
    } else {
        left * (1.0 - fraction) + right * fraction
    }
}

impl<F: FloatLike + FloatCore> TDigest<F> {
    /// Estimate the value at quantile `q` (inclusive) using half-weight
    /// bracketing and singleton-aware interpolation.
    ///
    /// - `q` is clamped to `[0, 1]` (when finite).
    /// - **Empty digest** → **`NaN`**.
    /// - **NaN `q`** → **`NaN`**.
    /// - A single atomic centroid is constant; a mixed centroid retains its tails.
    pub fn quantile(&self, q: f64) -> f64 {
        if q.is_nan() {
            return f64::NAN;
        }
        if self.centroids().is_empty() {
            return f64::NAN;
        }
        let q = q.clamp(0.0, 1.0);
        let (target_index, total_weight) = self.quantile_to_weight_index(q);

        if q == 0.0 {
            return self.min();
        }
        if q == 1.0 {
            return self.max();
        }
        let first = &self.centroids()[0];
        if self.centroids().len() == 1 {
            let mean = first.mean_f64();
            if first.is_atomic() {
                return mean;
            }
            // One mixed cluster still represents a range, with its center at
            // q=0.5. Use probabilities directly so rounded cumulative mass cannot
            // create a gap between the two half-weight branches.
            return if q <= 0.5 {
                interpolate_value(self.min(), mean, 2.0 * q)
            } else {
                interpolate_value(mean, self.max(), 2.0 * q - 1.0)
            };
        }
        // Keep the ordinary path free of an extra centroid scan. A malformed
        // normal header hiding subnormal stored mass is outside this fallback.
        if total_weight > 0.0 && total_weight < f64::MIN_POSITIVE {
            let stored_mass: f64 = self.centroids().iter().map(|c| c.weight_f64()).sum();
            if stored_mass > 0.0 && stored_mass < f64::MIN_POSITIVE {
                // q*total and centroid half-weights would lose rank resolution.
                // Use the same actual mass as CDF, without altering the digest.
                return self
                    .normalized_subnormal_mass_for_queries(stored_mass)
                    .quantile(q);
            }
        }
        let last = self.centroids().last().unwrap();
        let first_center = first.weight_f64() / 2.0;
        let last_half = last.weight_f64() / 2.0;
        if target_index <= first_center {
            return if first.is_atomic() {
                first.mean_f64()
            } else {
                interpolate_value(self.min(), first.mean_f64(), target_index / first_center)
            };
        }
        if target_index >= total_weight - last_half {
            return if last.is_atomic() {
                last.mean_f64()
            } else {
                interpolate_value(
                    last.mean_f64(),
                    self.max(),
                    1.0 - (total_weight - target_index) / last_half,
                )
            };
        }

        let (left_idx, right_idx, left_center_cum_w, center_span_w) =
            self.find_bracketing_centroids(target_index);

        self.interpolate_between_centroids(
            left_idx,
            right_idx,
            left_center_cum_w,
            center_span_w,
            target_index,
        )
    }

    /// Map `q ∈ [0,1]` to a cumulative weight index in `[0, total_weight]`.
    #[inline]
    fn quantile_to_weight_index(&self, q: f64) -> (f64, f64) {
        let total_weight = self.count();
        (q * total_weight, total_weight)
    }

    /// Half-weight bracketing: find adjacent centroids whose center-to-center
    /// span contains `index`.
    ///
    /// Returns `(left_idx, right_idx, cumulative_weight_at_left_center, center_span_weight)`.
    fn find_bracketing_centroids(&self, index: f64) -> (usize, usize, f64, f64) {
        let first_w = self.centroids()[0].weight_f64();
        let mut cum_w_at_left_center = first_w / 2.0;

        for left_idx in 0..(self.centroids().len() - 1) {
            let w_left = self.centroids()[left_idx].weight_f64();
            let w_right = self.centroids()[left_idx + 1].weight_f64();
            let center_span_weight = (w_left + w_right) / 2.0;

            if cum_w_at_left_center + center_span_weight > index {
                return (
                    left_idx,
                    left_idx + 1,
                    cum_w_at_left_center,
                    center_span_weight,
                );
            }
            cum_w_at_left_center += center_span_weight;
        }

        // Fallback: last pair (shouldn't happen in normal flow).
        let m = self.centroids().len();
        let w_last = self.centroids()[m - 1].weight_f64();
        let span_w = (self.centroids()[m - 2].weight_f64() + w_last) / 2.0;
        (m - 2, m - 1, self.count() - w_last / 2.0 - span_w, span_w)
    }

    /// Interpolate between bracketing centroids with symmetric singleton/pile logic.
    ///
    /// Snap inside atomic mass; otherwise interpolate after excluding atomic
    /// half-weights. Mixed centroids retain their interpolated mass.
    fn interpolate_between_centroids(
        &self,
        left_idx: usize,
        right_idx: usize,
        cum_w_at_left_center: f64,
        center_span_weight: f64,
        target_index: f64,
    ) -> f64 {
        let left = &self.centroids()[left_idx];
        let right = &self.centroids()[right_idx];

        let (w_left, w_right) = (left.weight_f64(), right.weight_f64());
        let (m_left, m_right) = (left.mean_f64(), right.mean_f64());
        let right_center_cum_w = cum_w_at_left_center + center_span_weight;

        if left.is_atomic() && right.is_atomic() {
            // Atomic neighbors share one boundary, with no interpolated mass.
            // Deriving that boundary independently from both centers can leave
            // a rounding-sized positive span (even for equal weights like 0.1),
            // incorrectly moving the boundary quantile to one of the means.
            let boundary = crate::tdigest::summation::sum(
                self.centroids()[..=left_idx].iter().map(|c| c.weight_f64()),
            );
            return if target_index < boundary {
                m_left
            } else if target_index > boundary {
                m_right
            } else {
                interpolate_value(m_left, m_right, 0.5)
            };
        }

        // Every rank strictly inside an atomic centroid has its exact mean.
        if left.is_singleton()
            && Self::inside_pile_strict(target_index, cum_w_at_left_center, w_left)
        {
            return m_left;
        }
        if right.is_singleton()
            && Self::inside_pile_strict(target_index, right_center_cum_w, w_right)
        {
            return m_right;
        }

        // Atomic mass has no interpolated interior, including fractional piles.
        let dead_left = if left.is_atomic() { w_left / 2.0 } else { 0.0 };
        let dead_right = if right.is_atomic() {
            w_right / 2.0
        } else {
            0.0
        };
        let weight_toward_right = target_index - cum_w_at_left_center - dead_left;
        let weight_toward_left = right_center_cum_w - target_index - dead_right;
        let denom = weight_toward_right + weight_toward_left;

        if denom <= 0.0 {
            interpolate_value(m_left, m_right, 0.5)
        } else {
            interpolate_value(m_left, m_right, weight_toward_right / denom)
        }
    }

    #[inline]
    fn inside_pile_strict(target_index: f64, center_cum_w: f64, pile_weight: f64) -> bool {
        // Atomic point masses may have any positive, including fractional, weight.
        let half = pile_weight / 2.0;
        (center_cum_w - half) < target_index && target_index < (center_cum_w + half)
    }

    /// Estimate the median using the same weighted interpolation as `quantile(0.5)`.
    /// Empty digests return `NaN`; centroid weights may be fractional.
    pub fn median(&self) -> f64 {
        self.quantile(0.5)
    }

    /// Approximate trimmed mean over quantile bounds `[lower, upper]`.
    ///
    /// Invalid bounds and empty digests return `NaN`.
    pub fn trimmed_mean(&self, lower: f64, upper: f64) -> f64 {
        if crate::tdigest::frontends::validate_trimmed_mean_bounds(lower, upper).is_err() {
            return f64::NAN;
        }

        let total = self.count();
        if total <= 0.0 || self.centroids().is_empty() {
            return f64::NAN;
        }

        let min_w = lower * total;
        let max_w = upper * total;
        if max_w <= min_w {
            return f64::NAN;
        }

        let mut cur = 0.0_f64;
        let mut acc_sum = 0.0_f64;
        let mut acc_w = 0.0_f64;
        let magnitude = self.min().abs().max(self.max().abs());
        let mut normalized_sum = crate::tdigest::summation::Sum::default();
        let mut normalized_mass = crate::tdigest::summation::Sum::default();

        for c in self.centroids() {
            let w = c.weight_f64();
            if w <= 0.0 {
                continue;
            }
            let next = cur + w;
            if next <= min_w {
                cur = next;
                continue;
            }
            if cur >= max_w {
                break;
            }

            let take_start = min_w.max(cur);
            let take_end = max_w.min(next);
            let take = (take_end - take_start).max(0.0);
            if take > 0.0 {
                acc_sum += c.mean_f64() * take;
                acc_w += take;
                if magnitude > 0.0 {
                    let relative_mass = take / total;
                    normalized_sum.add((c.mean_f64() / magnitude) * relative_mass);
                    normalized_mass.add(relative_mass);
                }
            }
            cur = next;
        }

        if acc_w > 0.0 {
            if acc_sum.is_finite() {
                acc_sum / acc_w
            } else {
                // A selected tail's moment can overflow although its average
                // lies between finite extrema. Divide before multiplying.
                (normalized_sum.total() / normalized_mass.total()).clamp(-1.0, 1.0) * magnitude
            }
        } else {
            f64::NAN
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::tdigest::test_helpers::*;
    use crate::tdigest::TDigestBuilder;

    #[test]
    fn centroid_addition_regression_pr_1() {
        let vals = vec![1.0, 1.0, 1.0, 2.0, 1.0, 1.0];
        let mut t = TDigestBuilder::new().max_size(10).build().unwrap();
        for v in vals {
            t = t.merge_unsorted(vec![v]).expect("no NaNs");
        }
        assert_rel_close("median", 1.0, t.quantile(0.5), 0.01);
        assert_rel_close("q=0.95", 2.0, t.quantile(0.95), 0.01);

        let means: Vec<f64> = t.centroids().iter().map(|c| c.mean_f64()).collect();
        assert_eq!(
            means.len(),
            2,
            "expected exactly two centroids, got {:?}",
            means
        );
        assert!(
            means.contains(&1.0) && means.contains(&2.0),
            "expected centroid means [1.0, 2.0], got {:?}",
            means
        );
    }

    /// n=10, max_size=10 — edge clamps, median bracket, monotone grid.
    #[test]
    fn quantiles_small_max10_smalln() {
        let mut values = vec![-10.0, -1.0, 0.0, 0.0, 2e-10, 1.0, 2.0, 10.0, 1e9, -1e9];
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let t = TDigestBuilder::new()
            .max_size(10)
            .build()
            .unwrap()
            .merge_sorted(values.clone())
            .expect("no NaNs");

        assert_exact("Q(0)", *values.first().unwrap(), t.quantile(0.0));
        assert_exact("Q(1)", *values.last().unwrap(), t.quantile(1.0));

        let (lo_m, hi_m, _, _) = bracket(&values, 0.5);
        let med = t.quantile(0.5);
        assert_in_bracket("median", med, lo_m, hi_m, 4, 5);

        let grid = [
            t.quantile(0.01),
            t.quantile(0.10),
            t.quantile(0.25),
            med,
            t.quantile(0.75),
            t.quantile(0.90),
            t.quantile(0.99),
        ];
        assert_monotone_chain("quantiles grid", &grid);
    }

    /// n=100, max_size=10 — quantile lies between bracketing order stats.
    #[test]
    fn quantiles_small_max10_medn_100() {
        let mut values: Vec<f64> = (-30..=69).map(|x| x as f64).collect(); // 100 values
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let t = TDigestBuilder::new()
            .max_size(10)
            .build()
            .unwrap()
            .merge_sorted(values.clone())
            .expect("no NaNs");

        for &(q, label) in &[
            (0.01_f64, "Q(0.01)"),
            (0.10_f64, "Q(0.10)"),
            (0.25_f64, "Q(0.25)"),
            (0.50_f64, "Q(0.50)"),
            (0.75_f64, "Q(0.75)"),
            (0.90_f64, "Q(0.90)"),
            (0.99_f64, "Q(0.99)"),
        ] {
            let (lo, hi, i_lo, i_hi) = bracket(&values, q);
            let x = t.quantile(q);
            assert_in_bracket(label, x, lo, hi, i_lo, i_hi);
        }
    }

    #[test]
    fn median_between_centroids_even_count() {
        // With symmetric bracketing, Q(0.5) equals the average of the two middle piles (→ 0.0).
        for num in [1, 2, 3, 10, 20] {
            let mut t = TDigestBuilder::new().max_size(100).build().unwrap();
            for _ in 0..num {
                t = t.merge_sorted(vec![-1.0]).expect("no NaNs");
            }
            for _ in 0..num {
                t = t.merge_sorted(vec![1.0]).expect("no NaNs");
            }

            assert_exact("Q(0.5)", 0.0, t.quantile(0.5));
            assert_exact("median()", 0.0, t.median());
        }
    }

    #[test]
    fn median_respects_unequal_and_fractional_weights() {
        for weights in [
            [90.0, 10.0],
            [10.0, 90.0],
            [9.5, 0.5],
            [0.1, 0.9],
            [0.9, 0.1],
        ] {
            let td = TDigestBuilder::<f64>::new()
                .max_size(1000)
                .build()
                .unwrap()
                .merge_weighted_unsorted(&[0.0, 10.0], &weights)
                .unwrap();
            let expected = if weights[0] > weights[1] { 0.0 } else { 10.0 };
            assert_eq!(td.median(), expected);
            assert_eq!(td.median(), td.quantile(0.5));
        }
        assert!(TDigestBuilder::<f64>::new()
            .build()
            .unwrap()
            .median()
            .is_nan());
    }

    #[test]
    fn quantile_exact_with_enough_capacity() {
        use crate::tdigest::test_helpers::assert_exact;
        use rand::{rngs::StdRng, Rng, SeedableRng};

        const N: usize = 9_999;
        let mut v: Vec<f64> = {
            let mut r = StdRng::seed_from_u64(42);
            (0..N).map(|_| r.random_range(0..N as u64) as f64).collect()
        };
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let td = TDigestBuilder::new()
            .max_size(N + 1)
            .build()
            .unwrap()
            .merge_sorted(v.clone())
            .expect("no NaNs");

        assert_exact("Q(0)", v[0], td.quantile(0.0));
        assert_exact("Q(1)", v[N - 1], td.quantile(1.0));
        for (i, &x) in v.iter().enumerate() {
            let q = (i as f64 + 0.5) / N as f64; // mid-rank → exact order stat
            assert_exact("Q(mid)", x, td.quantile(q));
        }
    }

    #[test]
    fn quantile_midrank_is_exact_under_capacity() {
        use crate::tdigest::test_helpers::assert_exact;

        // Sorted training data with ties; N is intentionally far below max_size.
        let v = vec![-2.0, -2.0, -1.0, 0.0, 0.0, 0.0, 3.0, 7.0, 7.0];
        let n = v.len();
        let td = TDigestBuilder::new()
            .max_size(1000)
            .build()
            .unwrap()
            .merge_sorted(v.clone())
            .expect("no NaNs");

        assert!(n < td.max_size());
        assert_exact("Q(0)", v[0], td.quantile(0.0));
        assert_exact("Q(1)", v[n - 1], td.quantile(1.0));
        for (i, &x) in v.iter().enumerate() {
            let q = (i as f64 + 0.5) / n as f64; // exactness contract point
            assert_exact(&format!("Q(mid-rank exact) [{i}]"), x, td.quantile(q));
        }
    }
}
