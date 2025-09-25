mod core;
mod scale;
mod simd;
mod traits;

use crate::{
    core::{compute, compute_quantile, compute_trimmed_mean, create_clusters, merge_clusters},
    traits::FloatConst,
};
use anyhow::Result;
use num::Float;
use std::num::NonZeroU32;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Delta<T>(T);

impl<T> Delta<T>
where
    T: Float + FloatConst + PartialOrd + std::fmt::Debug,
{
    pub fn new(value: T) -> Result<Self> {
        if value <= T::ZERO {
            return Err(anyhow::anyhow!("Delta must be positive, got {:?}", value));
        }
        if value > T::from(10000).unwrap() {
            return Err(anyhow::anyhow!("Delta too large (> 10000), got {:?}", value));
        }
        Ok(Delta(value))
    }

    pub fn value(&self) -> T {
        self.0
    }

}

impl<T> Default for Delta<T>
where
    T: Float + FloatConst + PartialOrd + std::fmt::Debug,
{
    fn default() -> Self {
        Delta(T::from(300).unwrap())
    }
}

pub struct TDigest<T, const MAX_CLUSTERS: usize = 1000> {
    pub means: Vec<T>,
    weights: Vec<u32>,
    delta: Delta<T>,
}

impl<T, const MAX_CLUSTERS: usize> TDigest<T, MAX_CLUSTERS>
where
    T: Float + FloatConst + std::fmt::Debug,
{
    pub fn means(&self) -> &[T] {
        &self.means
    }

    pub fn weights(&self) -> impl Iterator<Item = NonZeroU32> + '_ {
        self.weights.iter().map(|&w| NonZeroU32::new(w).unwrap_or(NonZeroU32::new(1).unwrap()))
    }

    pub fn len(&self) -> usize {
        self.means.len()
    }

    pub fn is_empty(&self) -> bool {
        self.means.is_empty()
    }

    pub fn delta(&self) -> &Delta<T> {
        &self.delta
    }
    pub fn from_array(arr: &[T], delta: Delta<T>) -> Result<Self> {
        let weights: Vec<u32> = vec![1; arr.len()];
        let (means, weights, _) = create_clusters(arr, &weights, delta.value())?;

        if weights.len() > MAX_CLUSTERS {
            return Err(anyhow::anyhow!(
                "Resulting clusters ({}) exceed maximum ({})",
                weights.len(),
                MAX_CLUSTERS
            ));
        }

        Ok(TDigest { means, weights, delta })
    }

    pub fn from_means_weights(arr: &[T], weights: &[NonZeroU32], delta: Delta<T>) -> Result<Self> {
        let weights_u32: Vec<u32> = weights.iter().map(|w| w.get()).collect();
        let mask = vec![true; arr.len()];
        let (means, weights, _) = compute(arr, &weights_u32, &mask, delta.value())?;

        if weights.len() > MAX_CLUSTERS {
            return Err(anyhow::anyhow!(
                "Resulting clusters ({}) exceed maximum ({})",
                weights.len(),
                MAX_CLUSTERS
            ));
        }

        Ok(TDigest { means, weights, delta })
    }

    pub fn quantile(&self, x: T) -> Result<T> {
        compute_quantile(&self.means, &self.weights, x)
    }

    pub fn median(&self) -> Result<T> {
        self.quantile(T::from(0.5).unwrap())
    }

    pub fn trimmed_mean(&self, lower: T, upper: T) -> Result<T> {
        compute_trimmed_mean(&self.means, &self.weights, lower, upper)
    }

    pub fn merge(&self, other: &Self, delta: Delta<T>) -> Result<Self> {
        let (means, weights, _) = merge_clusters(
            &self.means,
            &self.weights,
            &other.means,
            &other.weights,
            delta.value(),
        )?;
        Ok(Self { means, weights, delta })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroU32;

    #[test]
    fn test_delta_creation_valid() {
        let delta = Delta::new(100.0_f64).unwrap();
        assert_eq!(delta.value(), 100.0);

        let delta = Delta::new(1.0_f32).unwrap();
        assert_eq!(delta.value(), 1.0);
    }

    #[test]
    fn test_delta_creation_invalid_negative() {
        let result = Delta::new(-10.0_f64);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("positive"));
    }

    #[test]
    fn test_delta_creation_invalid_zero() {
        let result = Delta::new(0.0_f64);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("positive"));
    }

    #[test]
    fn test_delta_creation_too_large() {
        let result = Delta::new(20000.0_f64);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("too large"));
    }

    #[test]
    fn test_delta_default() {
        let delta = Delta::<f64>::default();
        assert_eq!(delta.value(), 300.0);
    }

    #[test]
    fn test_delta_equality() {
        let delta1 = Delta::new(100.0_f64).unwrap();
        let delta2 = Delta::new(100.0_f64).unwrap();
        let delta3 = Delta::new(200.0_f64).unwrap();

        assert_eq!(delta1, delta2);
        assert_ne!(delta1, delta3);
    }

    #[test]
    fn test_tdigest_from_array_basic() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let delta = Delta::new(100.0).unwrap();
        let digest: TDigest<f64> = TDigest::from_array(&data, delta).unwrap();

        assert!(!digest.means.is_empty());
        assert_eq!(digest.means.len(), digest.len());
    }

    #[test]
    fn test_tdigest_from_array_single_value() {
        let data = vec![42.0_f64];
        let delta = Delta::new(100.0).unwrap();
        let digest: TDigest<f64> = TDigest::from_array(&data, delta).unwrap();

        assert_eq!(digest.means.len(), 1);
        assert_eq!(digest.means[0], 42.0);
        let weight_iter: Vec<NonZeroU32> = digest.weights().collect();
        assert_eq!(weight_iter[0].get(), 1);
    }

    #[test]
    fn test_tdigest_from_array_empty() {
        let data: Vec<f64> = vec![];
        let delta = Delta::new(100.0).unwrap();
        let digest: TDigest<f64> = TDigest::from_array(&data, delta).unwrap();

        assert!(digest.means.is_empty());
        assert!(digest.is_empty());
    }

    #[test]
    fn test_tdigest_from_means_weights() {
        let means = vec![1.0_f64, 2.0, 3.0];
        let weights = vec![
            NonZeroU32::new(1).unwrap(),
            NonZeroU32::new(2).unwrap(),
            NonZeroU32::new(1).unwrap(),
        ];
        let delta = Delta::new(100.0).unwrap();
        let digest: TDigest<f64> = TDigest::from_means_weights(&means, &weights, delta).unwrap();

        assert!(!digest.means.is_empty());
        assert_eq!(digest.means.len(), digest.len());
    }

    #[test]
    fn test_tdigest_max_clusters_limit() {
        let data: Vec<f64> = (0..2000).map(|i| i as f64).collect();
        let delta = Delta::new(1.0).unwrap(); // Very small delta to force many clusters

        let result = TDigest::<_, 100>::from_array(&data, delta);
        match result {
            Ok(digest) => {
                assert!(digest.means.len() <= 100);
            }
            Err(e) => {
                assert!(e.to_string().contains("exceed maximum"));
            }
        }
    }

    #[test]
    fn test_tdigest_const_generic_default() {
        let data = vec![1.0_f64, 2.0, 3.0];
        let delta = Delta::new(100.0).unwrap();
        let digest: TDigest<f64> = TDigest::from_array(&data, delta).unwrap();

        assert!(!digest.means.is_empty());
    }

    #[test]
    fn test_quantile_basic() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let delta = Delta::new(100.0).unwrap();
        let digest: TDigest<f64> = TDigest::from_array(&data, delta).unwrap();

        let q25 = digest.quantile(0.25).unwrap();
        let q50 = digest.quantile(0.5).unwrap();
        let q75 = digest.quantile(0.75).unwrap();

        assert!(q25 >= 2.0 && q25 <= 4.0);
        assert!(q50 >= 4.0 && q50 <= 7.0);
        assert!(q75 >= 7.0 && q75 <= 9.0);
    }

    #[test]
    fn test_quantile_extremes() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let delta = Delta::new(100.0).unwrap();
        let digest: TDigest<f64> = TDigest::from_array(&data, delta).unwrap();

        let q0 = digest.quantile(0.0).unwrap();
        let q1 = digest.quantile(1.0).unwrap();

        assert!(q0 <= 2.0);
        assert!(q1 >= 4.0);
    }

    #[test]
    fn test_median() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let delta = Delta::new(100.0).unwrap();
        let digest: TDigest<f64> = TDigest::from_array(&data, delta).unwrap();

        let median = digest.median().unwrap();
        assert!(median >= 2.0 && median <= 4.0);
    }

    #[test]
    fn test_trimmed_mean() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let delta = Delta::new(100.0).unwrap();
        let digest: TDigest<f64> = TDigest::from_array(&data, delta).unwrap();

        let trimmed = digest.trimmed_mean(0.1, 0.9).unwrap();

        assert!(trimmed >= 3.0 && trimmed <= 8.0);
    }

    #[test]
    fn test_trimmed_mean_no_trim() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let delta = Delta::new(100.0).unwrap();
        let digest: TDigest<f64> = TDigest::from_array(&data, delta).unwrap();

        let mean = digest.trimmed_mean(0.0, 1.0).unwrap();

        assert!(mean >= 2.5 && mean <= 3.5);
    }

    #[test]
    fn test_quantile_single_value() {
        let data = vec![42.0_f64];
        let delta = Delta::new(100.0).unwrap();
        let digest: TDigest<f64> = TDigest::from_array(&data, delta).unwrap();

        assert_eq!(digest.quantile(0.0).unwrap(), 42.0);
        assert_eq!(digest.quantile(0.5).unwrap(), 42.0);
        assert_eq!(digest.quantile(1.0).unwrap(), 42.0);
        assert_eq!(digest.median().unwrap(), 42.0);
    }

    #[test]
    fn test_merge_basic() {
        let data1 = vec![1.0_f64, 2.0, 3.0];
        let data2 = vec![4.0_f64, 5.0, 6.0];
        let delta = Delta::new(100.0).unwrap();

        let digest1: TDigest<f64> = TDigest::from_array(&data1, delta).unwrap();
        let digest2: TDigest<f64> = TDigest::from_array(&data2, delta).unwrap();

        let merged = digest1.merge(&digest2, delta).unwrap();

        assert!(!merged.means.is_empty());
        assert_eq!(merged.means.len(), merged.weights.len());

        let median = merged.median().unwrap();
        assert!(median >= 2.0 && median <= 5.0);
    }

    #[test]
    fn test_merge_same_digest() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let delta = Delta::new(100.0).unwrap();
        let digest: TDigest<f64> = TDigest::from_array(&data, delta).unwrap();

        let merged = digest.merge(&digest, delta).unwrap();

        let original_median = digest.median().unwrap();
        let merged_median = merged.median().unwrap();

        assert!((original_median - merged_median).abs() < 0.5);
    }

    #[test]
    fn test_merge_empty_digests() {
        let empty_data: Vec<f64> = vec![];
        let delta = Delta::new(100.0).unwrap();

        let digest1: TDigest<f64> = TDigest::from_array(&empty_data, delta).unwrap();
        let digest2: TDigest<f64> = TDigest::from_array(&empty_data, delta).unwrap();

        let merged = digest1.merge(&digest2, delta).unwrap();
        assert!(merged.means.is_empty());
        assert!(merged.is_empty());
    }

    #[test]
    fn test_merge_one_empty() {
        let data = vec![1.0_f64, 2.0, 3.0];
        let empty_data: Vec<f64> = vec![];
        let delta = Delta::new(100.0).unwrap();

        let digest_full: TDigest<f64> = TDigest::from_array(&data, delta).unwrap();
        let digest_empty: TDigest<f64> = TDigest::from_array(&empty_data, delta).unwrap();

        let merged1 = digest_full.merge(&digest_empty, delta).unwrap();
        let merged2 = digest_empty.merge(&digest_full, delta).unwrap();

        assert!(!merged1.means.is_empty());
        assert!(!merged2.means.is_empty());
    }

    #[test]
    fn test_merge_different_deltas() {
        let data1 = vec![1.0_f64, 2.0, 3.0];
        let data2 = vec![4.0_f64, 5.0, 6.0];

        let delta1 = Delta::new(50.0).unwrap();
        let delta2 = Delta::new(200.0).unwrap();

        let digest1: TDigest<f64> = TDigest::from_array(&data1, delta1).unwrap();
        let digest2: TDigest<f64> = TDigest::from_array(&data2, delta1).unwrap();

        let merged = digest1.merge(&digest2, delta2).unwrap();

        assert!(!merged.means.is_empty());
        assert_eq!(merged.means.len(), merged.weights.len());
    }

    #[test]
    fn test_merge_large_datasets() {
        let data1: Vec<f64> = (1..=1000).map(|i| i as f64).collect();
        let data2: Vec<f64> = (1001..=2000).map(|i| i as f64).collect();
        let delta = Delta::new(100.0).unwrap();

        let digest1: TDigest<f64> = TDigest::from_array(&data1, delta).unwrap();
        let digest2: TDigest<f64> = TDigest::from_array(&data2, delta).unwrap();

        let merged = digest1.merge(&digest2, delta).unwrap();

        let q25 = merged.quantile(0.25).unwrap();
        let q50 = merged.quantile(0.5).unwrap();
        let q75 = merged.quantile(0.75).unwrap();

        assert!(q25 >= 400.0 && q25 <= 600.0);
        assert!(q50 >= 900.0 && q50 <= 1100.0);
        assert!(q75 >= 1400.0 && q75 <= 1600.0);
    }

    #[test]
    fn test_special_values_nan() {
        let data = vec![1.0_f64, f64::NAN, 3.0, 4.0, 5.0];
        let delta = Delta::new(100.0).unwrap();
        let digest: TDigest<f64> = TDigest::from_array(&data, delta).unwrap();

        let median = digest.median().unwrap();
        assert!(!median.is_nan());
    }

    #[test]
    fn test_special_values_infinity() {
        let data = vec![1.0_f64, 2.0, f64::INFINITY, f64::NEG_INFINITY, 5.0];
        let delta = Delta::new(100.0).unwrap();
        let digest: TDigest<f64> = TDigest::from_array(&data, delta).unwrap();

        assert!(!digest.means.is_empty());

        let median = digest.median().unwrap();
        assert!(median.is_finite());
    }

    #[test]
    fn test_quantile_edge_cases() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let delta = Delta::new(100.0).unwrap();
        let digest: TDigest<f64> = TDigest::from_array(&data, delta).unwrap();

        let q_min = digest.quantile(0.0).unwrap();
        let q_max = digest.quantile(1.0).unwrap();

        assert!(q_min.is_finite());
        assert!(q_max.is_finite());
        assert!(q_min <= q_max);
    }

    #[test]
    fn test_trimmed_mean_edge_cases() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let delta = Delta::new(100.0).unwrap();
        let digest: TDigest<f64> = TDigest::from_array(&data, delta).unwrap();

        let heavily_trimmed = digest.trimmed_mean(0.4, 0.6).unwrap();
        assert!(heavily_trimmed.is_finite());

        let full_mean = digest.trimmed_mean(0.0, 1.0).unwrap();
        assert!(full_mean >= 2.0 && full_mean <= 4.0);
    }

    #[test]
    fn test_very_small_dataset() {
        let data = vec![42.0_f64];
        let delta = Delta::new(100.0).unwrap();
        let digest: TDigest<f64> = TDigest::from_array(&data, delta).unwrap();

        let q25 = digest.quantile(0.25).unwrap();
        let median = digest.median().unwrap();
        let q75 = digest.quantile(0.75).unwrap();
        let trimmed = digest.trimmed_mean(0.1, 0.9).unwrap();

        assert!((q25 - 42.0).abs() < 1e-10);
        assert!((median - 42.0).abs() < 1e-10);
        assert!((q75 - 42.0).abs() < 1e-10);
        assert!((trimmed - 42.0).abs() < 1e-10);
    }

    #[test]
    fn test_duplicate_values() {
        let data = vec![5.0_f64; 100]; // 100 identical values
        let delta = Delta::new(100.0).unwrap();
        let digest: TDigest<f64> = TDigest::from_array(&data, delta).unwrap();

        assert_eq!(digest.quantile(0.0).unwrap(), 5.0);
        assert_eq!(digest.median().unwrap(), 5.0);
        assert_eq!(digest.quantile(1.0).unwrap(), 5.0);
        assert_eq!(digest.trimmed_mean(0.1, 0.9).unwrap(), 5.0);
    }

    #[test]
    fn test_large_range_values() {
        let data = vec![1e-10_f64, 1.0, 1e10];
        let delta = Delta::new(100.0).unwrap();
        let digest: TDigest<f64> = TDigest::from_array(&data, delta).unwrap();

        let median = digest.median().unwrap();
        assert!(median.is_finite());
        assert!(median > 0.0);
    }

    #[test]
    fn test_f32_precision() {
        let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let delta = Delta::new(100.0_f32).unwrap();
        let digest: TDigest<f32> = TDigest::from_array(&data, delta).unwrap();

        let median = digest.median().unwrap();
        assert!(median >= 2.0 && median <= 4.0);
    }


    #[test]
    fn test_quantile_accuracy_uniform_distribution() {
        let data: Vec<f64> = (1..=10000).map(|i| i as f64).collect();
        let delta = Delta::new(100.0).unwrap();
        let digest: TDigest<f64> = TDigest::from_array(&data, delta).unwrap();

        let q01 = digest.quantile(0.01).unwrap();
        let q99 = digest.quantile(0.99).unwrap();

        let expected_q01 = 100.0;
        let expected_q99 = 9900.0;

        let error_q01 = (q01 - expected_q01).abs() / expected_q01;
        let error_q99 = (q99 - expected_q99).abs() / expected_q99;

        assert!(error_q01 < 0.01, "q01 error too large: {:.4} (got {:.2}, expected {:.2})", error_q01, q01, expected_q01);
        assert!(error_q99 < 0.01, "q99 error too large: {:.4} (got {:.2}, expected {:.2})", error_q99, q99, expected_q99);
    }

    #[test]
    fn test_quantile_accuracy_normal_distribution() {
        let mut data = Vec::new();

        let mut seed1 = 12345u64;
        let mut seed2 = 67890u64;

        for _ in 0..5000 {  // Generate 5000 pairs = 10000 values
            seed1 = (seed1.wrapping_mul(1103515245).wrapping_add(12345)) & 0x7FFFFFFF;
            seed2 = (seed2.wrapping_mul(16807).wrapping_add(0)) & 0x7FFFFFFF;

            let u1 = (seed1 as f64) / (0x7FFFFFFF as f64);
            let u2 = (seed2 as f64) / (0x7FFFFFFF as f64);

            let u1 = u1.max(1e-10).min(1.0 - 1e-10);

            let magnitude = (-2.0 * u1.ln()).sqrt();
            let angle = 2.0 * std::f64::consts::PI * u2;

            let z1 = magnitude * angle.cos();
            let z2 = magnitude * angle.sin();

            data.push(z1);
            data.push(z2);
        }

        let delta = Delta::new(100.0).unwrap();
        let digest: TDigest<f64> = TDigest::from_array(&data, delta).unwrap();

        let median = digest.median().unwrap();
        let q25 = digest.quantile(0.25).unwrap();
        let q75 = digest.quantile(0.75).unwrap();
        let iqr = q75 - q25;

        assert!(median.abs() < 0.1, "Normal distribution median too far from 0: {:.3}", median);
        assert!(iqr > 1.2 && iqr < 1.5, "IQR not in expected range for normal distribution: {:.3}", iqr);
    }

    #[test]
    fn test_compression_effectiveness() {

        let large_data: Vec<f64> = (1..=100000).map(|i| (i as f64).sin() * 1000.0).collect();
        let delta = Delta::new(100.0).unwrap();
        let digest: TDigest<f64> = TDigest::from_array(&large_data, delta).unwrap();

        let num_centroids = digest.means.len();
        assert!(num_centroids < 500, "Too many centroids: {} (should be < 500 for δ=100)", num_centroids);
        assert!(num_centroids > 10, "Too few centroids: {} (should be > 10 for varied data)", num_centroids);

        let compression_ratio = large_data.len() as f64 / num_centroids as f64;
        assert!(compression_ratio > 100.0, "Compression ratio too low: {:.1}", compression_ratio);
    }

    #[test]
    fn test_merge_associativity() {

        let data_a: Vec<f64> = (1..=1000).map(|i| i as f64).collect();
        let data_b: Vec<f64> = (1001..=2000).map(|i| i as f64).collect();
        let data_c: Vec<f64> = (2001..=3000).map(|i| i as f64).collect();
        let delta = Delta::new(50.0).unwrap();

        let digest_a: TDigest<f64> = TDigest::from_array(&data_a, delta).unwrap();
        let digest_b: TDigest<f64> = TDigest::from_array(&data_b, delta).unwrap();
        let digest_c: TDigest<f64> = TDigest::from_array(&data_c, delta).unwrap();

        let ab = digest_a.merge(&digest_b, delta).unwrap();
        let abc_left = ab.merge(&digest_c, delta).unwrap();

        let bc = digest_b.merge(&digest_c, delta).unwrap();
        let abc_right = digest_a.merge(&bc, delta).unwrap();

        for q in [0.1, 0.25, 0.5, 0.75, 0.9] {
            let left_quantile = abc_left.quantile(q).unwrap();
            let right_quantile = abc_right.quantile(q).unwrap();
            let relative_error = (left_quantile - right_quantile).abs() / left_quantile;

            assert!(relative_error < 0.05,
                "Merge not associative at q={}: left={:.2}, right={:.2}, error={:.4}",
                q, left_quantile, right_quantile, relative_error);
        }
    }

    #[test]
    fn test_delta_effect_on_accuracy() {

        let data: Vec<f64> = (1..=10000).map(|i| (i as f64 * 0.1).sin() + i as f64).collect();

        let small_delta = Delta::new(20.0).unwrap();
        let large_delta = Delta::new(200.0).unwrap();

        let digest_precise: TDigest<f64> = TDigest::from_array(&data, small_delta).unwrap();
        let digest_compressed: TDigest<f64> = TDigest::from_array(&data, large_delta).unwrap();

        assert!(digest_precise.means.len() < digest_compressed.means.len(),
            "Smaller delta should create fewer centroids (stricter compression): {} vs {}",
            digest_precise.means.len(), digest_compressed.means.len());

        let true_median = data.len() as f64 / 2.0 + 0.5; // Approximate for our data
        let precise_median = digest_precise.median().unwrap();
        let compressed_median = digest_compressed.median().unwrap();

        let precise_error = (precise_median - true_median).abs();
        let compressed_error = (compressed_median - true_median).abs();

        assert!(precise_error <= compressed_error * 2.0,
            "Precise digest should be at least competitive: {:.2} vs {:.2}",
            precise_error, compressed_error);
    }

    #[test]
    fn test_monotonicity_property() {
        let data: Vec<f64> = (1..=1000).map(|i| (i as f64).powf(1.5)).collect();
        let delta = Delta::new(100.0).unwrap();
        let digest: TDigest<f64> = TDigest::from_array(&data, delta).unwrap();

        let quantiles = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95];
        let mut prev_value = f64::NEG_INFINITY;

        for &q in &quantiles {
            let current_value = digest.quantile(q).unwrap();
            assert!(current_value >= prev_value,
                "Quantile function not monotonic: Q({}) = {:.2} < Q(prev) = {:.2}",
                q, current_value, prev_value);
            prev_value = current_value;
        }
    }


    #[test]
    fn test_performance_regression_large_dataset() {
        let data: Vec<f64> = (0..100_000).map(|i| (i as f64 * 0.001).sin() + i as f64).collect();

        let start = std::time::Instant::now();
        let digest: TDigest<f64> = TDigest::from_array(&data, Delta::new(100.0).unwrap()).unwrap();
        let construction_time = start.elapsed();

        let start = std::time::Instant::now();
        let _q50 = digest.median().unwrap();
        let query_time = start.elapsed();

        let start = std::time::Instant::now();
        let _q25 = digest.quantile(0.25).unwrap();
        let _q75 = digest.quantile(0.75).unwrap();
        let multi_query_time = start.elapsed();

        println!("Performance baseline: construction={:?}, query={:?}, multi_query={:?}",
                construction_time, query_time, multi_query_time);

        assert!(construction_time.as_millis() < 1000, "Construction too slow: {:?}", construction_time);
        assert!(query_time.as_micros() < 1000, "Query too slow: {:?}", query_time);
        assert!(multi_query_time.as_micros() < 2000, "Multi-query too slow: {:?}", multi_query_time);
    }

    #[test]
    fn test_memory_allocation_bounds() {
        let data: Vec<f64> = (0..50_000).map(|i| (i as f64 * 0.01).sin() * (i % 1000) as f64).collect();
        let digest: TDigest<f64> = TDigest::from_array(&data, Delta::new(100.0).unwrap()).unwrap();

        println!("Compression: {} input points → {} centroids (ratio: {:.1}x)",
                data.len(), digest.means.len(),
                data.len() as f64 / digest.means.len() as f64);

        assert!(digest.means.len() < 500, "Too many centroids: {}", digest.means.len());
        assert!(digest.means.len() > 20, "Too few centroids: {}", digest.means.len());

        let compact_digest: TDigest<f64> = TDigest::from_array(&data, Delta::new(50.0).unwrap()).unwrap();
        let loose_digest: TDigest<f64> = TDigest::from_array(&data, Delta::new(200.0).unwrap()).unwrap();

        assert!(compact_digest.means.len() < loose_digest.means.len(),
            "Delta compression relationship: compact({}) vs loose({})",
            compact_digest.means.len(), loose_digest.means.len());
    }

    #[test]
    fn test_numerical_stability_edge_cases() {
        let mut data = vec![1e-100_f64, 1.0, 1e100];
        data.extend((1..1000).map(|i| i as f64));

        let digest: TDigest<f64> = TDigest::from_array(&data, Delta::new(100.0).unwrap()).unwrap();

        for q in [0.01, 0.1, 0.5, 0.9, 0.99] {
            let quantile = digest.quantile(q).unwrap();
            assert!(quantile.is_finite(), "Quantile not finite at q={}: {}", q, quantile);
            assert!(!quantile.is_nan(), "Quantile is NaN at q={}: {}", q, quantile);
        }

        let extreme_data = vec![-1e308_f64, 0.0, 1e308_f64];
        let extreme_digest: TDigest<f64> = TDigest::from_array(&extreme_data, Delta::new(100.0).unwrap()).unwrap();

        let median = extreme_digest.median().unwrap();
        assert!(median.is_finite(), "Extreme value median not finite: {}", median);
    }

    #[test]
    fn test_merge_performance_scaling() {
        let sizes = [100, 1000, 10000];

        for &size in &sizes {
            let data1: Vec<f64> = (0..size).map(|i| i as f64).collect();
            let data2: Vec<f64> = (size..size*2).map(|i| i as f64).collect();

            let digest1: TDigest<f64> = TDigest::from_array(&data1, Delta::new(100.0).unwrap()).unwrap();
            let digest2: TDigest<f64> = TDigest::from_array(&data2, Delta::new(100.0).unwrap()).unwrap();

            let start = std::time::Instant::now();
            let _merged = digest1.merge(&digest2, Delta::new(100.0).unwrap()).unwrap();
            let merge_time = start.elapsed();

            println!("Merge performance for size {}: {:?}", size, merge_time);

            assert!(merge_time.as_millis() < 100, "Merge too slow for size {}: {:?}", size, merge_time);
        }
    }

    #[test]
    fn test_internal_weight_consistency() {
        let means = vec![1.0_f64, 2.0, 3.0];
        let weights = vec![
            NonZeroU32::new(1).unwrap(),
            NonZeroU32::new(10).unwrap(),
            NonZeroU32::new(5).unwrap(),
        ];
        let digest: TDigest<f64> = TDigest::from_means_weights(&means, &weights, Delta::new(100.0).unwrap()).unwrap();

        let q50 = digest.quantile(0.5).unwrap();
        assert!(q50.is_finite(), "Quantile should be finite");

        let q50_again = digest.quantile(0.5).unwrap();
        assert!((q50 - q50_again).abs() < 1e-10, "Quantile should be deterministic");

        let trimmed = digest.trimmed_mean(0.1, 0.9).unwrap();
        assert!(trimmed.is_finite(), "Trimmed mean should be finite");

        println!("Weight consistency test: q50={}, trimmed_mean={}", q50, trimmed);
    }

    #[test]
    fn test_vector_capacity_optimization() {
        let data: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        let digest: TDigest<f64> = TDigest::from_array(&data, Delta::new(100.0).unwrap()).unwrap();

        assert!(digest.means.len() < 200, "Too many centroids after clustering: {}", digest.means.len());
        assert!(digest.means.len() > 5, "Too few centroids: {}", digest.means.len());

        let data2: Vec<f64> = (1000..2000).map(|i| i as f64).collect();
        let digest2: TDigest<f64> = TDigest::from_array(&data2, Delta::new(100.0).unwrap()).unwrap();

        let merged = digest.merge(&digest2, Delta::new(100.0).unwrap()).unwrap();

        assert!(merged.means.len() < 400, "Merged digest has too many centroids: {}", merged.means.len());
        assert!(merged.means.len() > 10, "Merged digest has too few centroids: {}", merged.means.len());
    }

    #[test]
    fn test_large_merge_memory_efficiency() {
        let sizes = [1000, 5000, 10000];

        for &size in &sizes {
            let data1: Vec<f64> = (0..size).map(|i| (i as f64).sin() * 100.0).collect();
            let data2: Vec<f64> = (size..size*2).map(|i| (i as f64).cos() * 100.0).collect();

            let digest1: TDigest<f64> = TDigest::from_array(&data1, Delta::new(100.0).unwrap()).unwrap();
            let digest2: TDigest<f64> = TDigest::from_array(&data2, Delta::new(100.0).unwrap()).unwrap();

            let merged = digest1.merge(&digest2, Delta::new(100.0).unwrap()).unwrap();

            let compression_ratio = (data1.len() + data2.len()) as f64 / merged.means.len() as f64;
            assert!(compression_ratio > 10.0, "Poor compression ratio for size {}: {:.1}", size, compression_ratio);
            assert!(merged.means.len() < 500, "Too many centroids for size {}: {}", size, merged.means.len());

            println!("Size {}: {} centroids, compression ratio: {:.1}x",
                    size * 2, merged.means.len(), compression_ratio);
        }
    }

    #[test]
    fn test_empty_and_small_allocation_edge_cases() {
        let empty_data: Vec<f64> = vec![];
        let empty_digest: TDigest<f64> = TDigest::from_array(&empty_data, Delta::new(100.0).unwrap()).unwrap();

        assert!(empty_digest.is_empty());
        assert_eq!(empty_digest.len(), 0);

        let single_data = vec![42.0_f64];
        let single_digest: TDigest<f64> = TDigest::from_array(&single_data, Delta::new(100.0).unwrap()).unwrap();

        assert_eq!(single_digest.len(), 1);
        assert_eq!(single_digest.means[0], 42.0);

        let merged = empty_digest.merge(&single_digest, Delta::new(100.0).unwrap()).unwrap();
        assert_eq!(merged.len(), 1);
        assert_eq!(merged.means[0], 42.0);

        let small_data1 = vec![1.0_f64, 2.0];
        let small_data2 = vec![3.0_f64, 4.0];
        let digest1: TDigest<f64> = TDigest::from_array(&small_data1, Delta::new(100.0).unwrap()).unwrap();
        let digest2: TDigest<f64> = TDigest::from_array(&small_data2, Delta::new(100.0).unwrap()).unwrap();

        let small_merged = digest1.merge(&digest2, Delta::new(100.0).unwrap()).unwrap();
        assert!(small_merged.len() <= 4); // Should not exceed input size for small data
        assert!(small_merged.len() >= 1); // Should have at least one centroid
    }

    #[test]
    fn test_allocation_bounds_with_const_generics() {
        let large_data: Vec<f64> = (0..10000).map(|i| i as f64).collect();

        type SmallDigest = TDigest<f64, 50>;
        let small_result = SmallDigest::from_array(&large_data, Delta::new(10.0).unwrap());

        match small_result {
            Ok(digest) => {
                assert!(digest.len() <= 50, "Digest exceeded MAX_CLUSTERS: {}", digest.len());
                let median = digest.median().unwrap();
                assert!(median >= 4000.0 && median <= 6000.0, "Median out of expected range: {}", median);
            }
            Err(e) => {
                assert!(e.to_string().contains("exceed maximum"));
            }
        }

        type LargeDigest = TDigest<f64, 500>;
        let large_digest = LargeDigest::from_array(&large_data, Delta::new(100.0).unwrap()).unwrap();
        assert!(large_digest.len() <= 500);
        assert!(large_digest.len() > 10); // Should use reasonable number of clusters
    }
}
