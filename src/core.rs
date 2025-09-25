use anyhow::Result;
use itertools::izip;
use num::Float;

use crate::{
    scale::log_q_limit,
    simd::{sum_weights_optimized, merge_sorted_optimized},
    traits::FloatConst,
};

pub fn argsort<T>(arr: &[T]) -> Result<Vec<usize>>
where
    T: Float,
{
    let mut indices = (0..arr.len()).collect::<Vec<usize>>();
    indices.sort_by(|&i, &j| {
        arr[i].partial_cmp(&arr[j]).unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(indices)
}
pub fn sort_by_indices<T: Copy>(arr: &[T], indices: &[usize]) -> Result<Vec<T>> {
    Ok(indices.iter().map(|&i| arr[i]).collect())
}

pub fn create_clusters<T>(
    means: &[T],
    weights: &[u32],
    delta: T,
) -> Result<(Vec<T>, Vec<u32>, Vec<bool>)>
where
    T: Float + FloatConst,
{
    let indices = argsort(means)?;
    let means = sort_by_indices(means, &indices)?;
    let weights = sort_by_indices(weights, &indices)?;
    let mask = vec![true; means.len()];
    compute(&means, &weights, &mask, delta)
}

pub fn merge_clusters<T>(
    means1: &[T],
    weights1: &[u32],
    means2: &[T],
    weights2: &[u32],
    delta: T,
) -> Result<(Vec<T>, Vec<u32>, Vec<bool>)>
where
    T: Float + FloatConst + Copy,
{
    let total_capacity = means1.len() + means2.len();
    let mut means = Vec::with_capacity(total_capacity);
    let mut weights = Vec::with_capacity(total_capacity);

    merge_sorted_optimized(means1, weights1, means2, weights2, &mut means, &mut weights);

    let mask = vec![true; means.len()];
    compute(&means, &weights, &mask, delta)
}

pub fn compute<T>(
    means: &[T],
    weights: &[u32],
    mask: &[bool],
    delta: T,
) -> Result<(Vec<T>, Vec<u32>, Vec<bool>)>
where
    T: Float + FloatConst,
{
    let mut n = means.len();
    let mut new_means = Vec::with_capacity(n);
    let mut new_weights = Vec::with_capacity(n);
    let mut new_mask = Vec::with_capacity(n);

    let mut start = 0;
    let mut end = n;
    while start < n && means[start] == T::NEG_INFINITY {
        start += 1;
    }
    while end > start && means[end - 1] == T::INFINITY {
        end -= 1;
    }

    if start > 0 {
        new_means.push(T::NEG_INFINITY);
        let neg_inf_weight = sum_weights_optimized(&weights[..start]);
        new_weights.push(neg_inf_weight.try_into()
            .map_err(|_| anyhow::anyhow!("Weight sum too large for u32"))?);
        new_mask.push(true);
    }
    let inf_exists = end < n;
    let mut inf_weight = 0u64;
    if inf_exists {
        inf_weight = sum_weights_optimized(&weights[end..]);
    }
    let mean_slice = &means[start..end];
    let weight_slice = &weights[start..end];
    let mask_slice = &mask[start..end];
    n = mean_slice.len();

    if n > 0 {
        let total_weight_u32 = sum_weights_optimized(weight_slice) as u32;
        let total_weight = T::from(total_weight_u32).unwrap();
        let mut cumulative_weight_u32 = 0u32;
        let mut sigma_mean = mean_slice[0];
        let mut sigma_weight = weight_slice[0];
        let mut sigma_mask = mask_slice[0];
        let mut q_limit = log_q_limit(T::ZERO, delta, n)?;

        for (&mu, &wght, &msk) in izip!(
            mean_slice.iter().skip(1),
            weight_slice.iter().skip(1),
            mask_slice.iter().skip(1)
        ) {
            if mu.is_nan() {
                continue;
            }

            let candidate_weight_u32 = cumulative_weight_u32 + sigma_weight + wght;
            let q = T::from(candidate_weight_u32).unwrap() / total_weight;

            if q <= q_limit {
                let sigma_weight_t = T::from(sigma_weight).unwrap();
                let wght_t = T::from(wght).unwrap();
                let new_weight_t = T::from(sigma_weight + wght).unwrap();

                sigma_mean = ((sigma_mean * sigma_weight_t) + mu * wght_t) / new_weight_t;
                sigma_weight += wght;
                sigma_mask = false;
            } else {
                new_means.push(sigma_mean);
                new_weights.push(sigma_weight);
                new_mask.push(sigma_mask);

                cumulative_weight_u32 += sigma_weight;
                let cumulative_weight_t = T::from(cumulative_weight_u32).unwrap();
                q_limit = log_q_limit(cumulative_weight_t / total_weight, delta, n)?;
                sigma_mean = mu;
                sigma_weight = wght;
                sigma_mask = msk;
            }
        }
        if !sigma_mean.is_nan() {
            new_means.push(sigma_mean);
            new_weights.push(sigma_weight);
            new_mask.push(sigma_mask);
        }
    }

    if inf_exists {
        new_means.push(T::INFINITY);
        new_weights.push(inf_weight.try_into()
            .map_err(|_| anyhow::anyhow!("Positive infinity weight sum too large for u32"))?);
        new_mask.push(true);
    }

    Ok((new_means, new_weights, new_mask))
}

pub fn compute_quantile<T>(means: &[T], weights: &[u32], x: T) -> Result<T>
where
    T: Float + FloatConst,
{
    let total_weight = T::from(sum_weights_optimized(weights) as u32).unwrap();

    if total_weight == T::ZERO {
        return Ok(total_weight);
    }
    if x == T::ZERO {
        return Ok(means[0]);
    }
    let search_position = x * total_weight;
    let mut m_previous = means[0];
    let mut w_previous = T::from(weights[0]).unwrap();
    let mut previous_position = w_previous / T::TWO;

    for (&m, &w) in means.iter().skip(1).zip(weights.iter().skip(1)) {
        let w = T::from(w).unwrap();
        let diff = (w + w_previous) / T::TWO;
        let next_position = previous_position + diff;
        if search_position <= next_position {
            let z1 = search_position - previous_position;
            let z2 = next_position - search_position;
            return Ok((m_previous * z2 + m * z1) / diff);
        }
        m_previous = m;
        w_previous = w;
        previous_position = previous_position + diff;
    }
    Ok(means[means.len() - 1])
}

pub fn compute_trimmed_mean<T>(means: &[T], weights: &[u32], lower: T, upper: T) -> Result<T>
where
    T: Float + FloatConst,
{
    let n = T::from(sum_weights_optimized(weights) as u32).unwrap();
    let min_count = lower * n;
    let max_count = upper * n;

    let mut trimmed_sum = T::ZERO;
    let mut trimmed_count = T::ZERO;
    let mut curr_count = T::ZERO;

    for (&m, &w) in means.iter().zip(weights.iter()) {
        let mut d_count = T::from(w).unwrap();
        let next_count = curr_count + d_count;
        if next_count < min_count {
            curr_count = next_count;
            continue;
        }
        if curr_count < min_count {
            d_count = next_count - min_count;
        } else if next_count > max_count {
            d_count = d_count - (next_count - max_count);
        }

        trimmed_sum = trimmed_sum + (d_count * m);
        trimmed_count = trimmed_count + d_count;

        if next_count >= max_count {
            break;
        }
        curr_count = next_count;
    }
    Ok(trimmed_sum / trimmed_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_argsort_basic() {
        let data = vec![3.0_f64, 1.0, 4.0, 1.5, 2.0];
        let indices = argsort(&data).unwrap();

        let sorted_values: Vec<f64> = indices.iter().map(|&i| data[i]).collect();
        let expected = [1.0, 1.5, 2.0, 3.0, 4.0];

        for (actual, expected) in sorted_values.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_argsort_empty() {
        let data: Vec<f64> = vec![];
        let indices = argsort(&data).unwrap();
        assert!(indices.is_empty());
    }

    #[test]
    fn test_argsort_single() {
        let data = vec![42.0_f64];
        let indices = argsort(&data).unwrap();
        assert_eq!(indices, vec![0]);
    }

    #[test]
    fn test_argsort_with_nan() {
        let data = vec![3.0_f64, f64::NAN, 1.0, 2.0];
        let indices = argsort(&data).unwrap();

        assert_eq!(indices.len(), 4);
    }

    #[test]
    fn test_sort_by_indices() {
        let data = vec![10, 20, 30, 40, 50];
        let indices = vec![4, 0, 2, 1, 3];
        let sorted = sort_by_indices(&data, &indices).unwrap();

        assert_eq!(sorted, vec![50, 10, 30, 20, 40]);
    }

    #[test]
    fn test_overflow_protection_large_weights() {
        let means = vec![f64::NEG_INFINITY; 10];
        let weights = vec![u32::MAX; 10];
        let mask = vec![true; 10];
        let delta = 100.0;

        let result = compute(&means, &weights, &mask, delta);

        match result {
            Ok((new_means, _new_weights, _)) => {
                assert!(!new_means.is_empty());
                assert_eq!(new_means.len(), _new_weights.len());
            }
            Err(e) => {
                let error_msg = e.to_string();
                assert!(
                    error_msg.contains("overflow") || error_msg.contains("too large"),
                    "Expected overflow-related error, got: {}",
                    error_msg
                );
            }
        }
    }

    #[test]
    fn test_overflow_protection_positive_infinity() {
        let means = vec![f64::INFINITY; 5];
        let weights = vec![u32::MAX; 5];
        let mask = vec![true; 5];
        let delta = 100.0;

        let result = compute(&means, &weights, &mask, delta);

        match result {
            Ok((new_means, _new_weights, _)) => {
                assert!(!new_means.is_empty());
            }
            Err(e) => {
                assert!(e.to_string().contains("overflow") || e.to_string().contains("too large"));
            }
        }
    }

    #[test]
    fn test_compute_quantile_empty() {
        let means: Vec<f64> = vec![];
        let weights: Vec<u32> = vec![];

        let result = compute_quantile(&means, &weights, 0.5);

        if let Ok(q) = result {
            assert_eq!(q, 0.0);
        }
    }

    #[test]
    fn test_compute_quantile_single_value() {
        let means = vec![42.0_f64];
        let weights = vec![1u32];

        let q0 = compute_quantile(&means, &weights, 0.0).unwrap();
        let q50 = compute_quantile(&means, &weights, 0.5).unwrap();
        let q100 = compute_quantile(&means, &weights, 1.0).unwrap();

        assert_eq!(q0, 42.0);
        assert_eq!(q50, 42.0);
        assert_eq!(q100, 42.0);
    }

    #[test]
    fn test_compute_trimmed_mean_basic() {
        let means = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let weights = vec![1u32; 5];

        let trimmed = compute_trimmed_mean(&means, &weights, 0.2, 0.8).unwrap();

        assert!((2.0..=4.0).contains(&trimmed));
    }

    #[test]
    fn test_compute_trimmed_mean_no_trim() {
        let means = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let weights = vec![2u32, 1, 1, 1, 2];

        let mean = compute_trimmed_mean(&means, &weights, 0.0, 1.0).unwrap();
        let expected = (1.0 * 2.0 + 2.0 * 1.0 + 3.0 * 1.0 + 4.0 * 1.0 + 5.0 * 2.0) / 7.0;
        assert!((mean - expected).abs() < 1e-10);
    }

    #[test]
    fn test_create_clusters_basic() {
        let means = vec![3.0_f64, 1.0, 4.0, 2.0];
        let weights = vec![1u32, 1, 1, 1];
        let delta = 100.0;

        let (new_means, new_weights, new_mask) = create_clusters(&means, &weights, delta).unwrap();

        assert_eq!(new_means.len(), new_weights.len());
        assert_eq!(new_means.len(), new_mask.len());
        assert!(!new_means.is_empty());
    }

    #[test]
    fn test_merge_clusters_basic() {
        let means1 = vec![1.0_f64, 2.0];
        let weights1 = vec![1u32, 1];
        let means2 = vec![3.0_f64, 4.0];
        let weights2 = vec![1u32, 1];
        let delta = 100.0;

        let (merged_means, merged_weights, merged_mask) =
            merge_clusters(&means1, &weights1, &means2, &weights2, delta).unwrap();

        assert_eq!(merged_means.len(), merged_weights.len());
        assert_eq!(merged_means.len(), merged_mask.len());
        assert!(!merged_means.is_empty());
    }
}
