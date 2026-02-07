use anyhow::Result;
use itertools::{izip, Itertools};
use num::Float;

use crate::{
    scale::{log_q_limit, ScaleParams},
    traits::{FloatConst, TotalOrd},
};

pub fn create_clusters<T>(
    means: &[T],
    weights: &[u32],
    delta: T,
) -> Result<(Vec<T>, Vec<u32>)>
where
    T: Float + FloatConst + TotalOrd<T>,
{
    let mut pairs: Vec<(T, u32)> = means.iter().copied().zip(weights.iter().copied()).collect();
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (sorted_means, sorted_weights): (Vec<T>, Vec<u32>) = pairs.into_iter().unzip();
    compute(&sorted_means, &sorted_weights, delta)
}

pub fn merge_clusters<T>(
    means1: &Vec<T>,
    weights1: &Vec<u32>,
    means2: &Vec<T>,
    weights2: &Vec<u32>,
    delta: T,
) -> Result<(Vec<T>, Vec<u32>)>
where
    T: Float + FloatConst,
{
    let (means, weights): (Vec<T>, Vec<u32>) =
        vec![izip!(means1, weights1), izip!(means2, weights2)]
            .into_iter()
            .kmerge()
            .unzip();
    compute(&means, &weights, delta)
}

pub fn compute<T>(
    means: &[T],
    weights: &[u32],
    delta: T,
) -> Result<(Vec<T>, Vec<u32>)>
where
    T: Float + FloatConst,
{
    let mut n = means.len();
    let mut new_means = Vec::with_capacity(n);
    let mut new_weights = Vec::with_capacity(n);

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
        new_weights.push(weights[..start].iter().sum::<u32>());
    }
    let inf_exists = end < n;
    let mut inf_weight = 0;
    if inf_exists {
        inf_weight = weights[end..].iter().sum();
    }
    let mean_slice = &means[start..end];
    let weight_slice = &weights[start..end];
    n = mean_slice.len();

    if n > 0 {
        let total_weight: T = T::from(weight_slice.iter().sum::<u32>()).unwrap();
        let params = ScaleParams::new(delta, n);

        let mut cumulative_weight_f = T::ZERO;
        let mut sigma_mean = mean_slice[0];
        let mut sigma_weight = weight_slice[0];
        let mut sigma_weight_f = T::from(sigma_weight).unwrap();
        let mut q_limit = log_q_limit(T::ZERO, &params);

        for (&mu, &wght) in mean_slice.iter().skip(1).zip(weight_slice.iter().skip(1)) {
            if mu.is_nan() {
                continue;
            }

            let wght_f = T::from(wght).unwrap();
            let q = (cumulative_weight_f + sigma_weight_f + wght_f) / total_weight;
            if q <= q_limit {
                let new_weight_f = sigma_weight_f + wght_f;
                sigma_mean = (sigma_mean * sigma_weight_f + mu * wght_f) / new_weight_f;
                sigma_weight += wght;
                sigma_weight_f = new_weight_f;
            } else {
                new_means.push(sigma_mean);
                new_weights.push(sigma_weight);

                cumulative_weight_f = cumulative_weight_f + sigma_weight_f;
                q_limit = log_q_limit(cumulative_weight_f / total_weight, &params);
                sigma_mean = mu;
                sigma_weight = wght;
                sigma_weight_f = wght_f;
            }
        }
        if !sigma_mean.is_nan() {
            new_means.push(sigma_mean);
            new_weights.push(sigma_weight);
        }
    }

    if inf_exists {
        new_means.push(T::INFINITY);
        new_weights.push(inf_weight);
    }

    Ok((new_means, new_weights))
}

pub fn compute_quantile<T>(means: &[T], weights: &[u32], x: T) -> Result<T>
where
    T: Float + FloatConst,
{
    let total_weight = T::from(weights.iter().sum::<u32>()).unwrap();

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

    for (m, w) in means.iter().skip(1).zip(weights.iter().skip(1)) {
        let m = T::from(*m).unwrap();
        let w = T::from(*w).unwrap();
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
    let n = T::from(weights.iter().sum::<u32>()).unwrap();
    let min_count = lower * n;
    let max_count = upper * n;

    let mut trimmed_sum = T::ZERO;
    let mut trimmed_count = T::ZERO;
    let mut curr_count = T::ZERO;

    for (m, w) in means.iter().zip(weights.iter()) {
        let mut d_count = T::from(*w).unwrap();
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

        trimmed_sum = trimmed_sum + (d_count * (*m));
        trimmed_count = trimmed_count + d_count;

        if next_count >= max_count {
            break;
        }
        curr_count = next_count;
    }
    Ok(trimmed_sum / trimmed_count)
}
