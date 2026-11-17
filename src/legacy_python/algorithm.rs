//! Private port of v0.1.0's core.rs, scale.rs and lib.rs. Operation order and
//! precision here are observable in historical Python pickles and queries.

use ordered_float::Float;
use std::cmp::Ordering;

pub(super) trait LegacyFloat: Float {
    const E: Self;
    fn total_order(a: &Self, b: &Self) -> Ordering;
}

impl LegacyFloat for f32 {
    const E: Self = std::f32::consts::E;
    fn total_order(a: &Self, b: &Self) -> Ordering {
        a.total_cmp(b)
    }
}
impl LegacyFloat for f64 {
    const E: Self = std::f64::consts::E;
    fn total_order(a: &Self, b: &Self) -> Ordering {
        a.total_cmp(b)
    }
}

fn log_q_limit<T: LegacyFloat>(q: T, delta: T, n: usize) -> T {
    let normalizer = (T::from(n).unwrap() / delta)
        .log(T::E)
        .mul_add(T::from(4).unwrap(), T::from(24).unwrap());
    let k = (delta / normalizer) * (q / (T::one() - q)).log(T::E) + T::one();
    (T::one() + (-k * (normalizer / delta)).exp()).recip()
}

pub(super) struct Digest<T> {
    pub means: Vec<T>,
    pub weights: Vec<u32>,
}

impl<T: LegacyFloat> Digest<T> {
    pub fn from_array(mut values: Vec<T>, delta: T) -> Result<Self, &'static str> {
        if values.is_empty() {
            return Err("Array must be non-empty!");
        }
        u32::try_from(values.len()).map_err(|_| "uint32 total weight overflow")?;
        values.sort_by(T::total_order);
        let weights = vec![1; values.len()];
        Ok(Self::cluster(&values, &weights, delta))
    }

    pub fn from_means_weights(
        means: &[T],
        weights: &[u32],
        delta: T,
    ) -> Result<Self, &'static str> {
        if means.len() != weights.len() {
            return Err("means and weights must have the same length");
        }
        weights
            .iter()
            .try_fold(0u32, |n, &w| n.checked_add(w))
            .ok_or("uint32 total weight overflow")?;
        // These are ordered runs in the original API. Preserve even an
        // unsorted caller's order exactly as v0.1.0 did.
        Ok(Self::cluster(means, weights, delta))
    }

    fn cluster(means: &[T], weights: &[u32], delta: T) -> Self {
        let mut out = Self {
            means: Vec::with_capacity(means.len()),
            weights: Vec::with_capacity(means.len()),
        };
        let start = means
            .iter()
            .take_while(|&&m| m == T::neg_infinity())
            .count();
        let end = means.len()
            - means[start..]
                .iter()
                .rev()
                .take_while(|&&m| m == T::infinity())
                .count();
        if start > 0 {
            out.push(T::neg_infinity(), weights[..start].iter().sum());
        }
        let run = &means[start..end];
        let mass = &weights[start..end];
        if let Some(&first) = run.first() {
            let total = T::from(mass.iter().sum::<u32>()).unwrap();
            let mut cumulative = 0;
            let mut mean = first;
            let mut weight = mass[0];
            let mut limit = log_q_limit(T::zero(), delta, run.len());
            for (&value, &w) in run[1..].iter().zip(&mass[1..]) {
                if value.is_nan() {
                    continue;
                }
                let q = T::from(cumulative + weight + w).unwrap() / total;
                if q <= limit {
                    mean = (mean * T::from(weight).unwrap() + value * T::from(w).unwrap())
                        / T::from(weight + w).unwrap();
                    weight += w;
                } else {
                    out.push(mean, weight);
                    cumulative += weight;
                    limit = log_q_limit(T::from(cumulative).unwrap() / total, delta, run.len());
                    mean = value;
                    weight = w;
                }
            }
            if !mean.is_nan() {
                out.push(mean, weight);
            }
        }
        if end < means.len() {
            out.push(T::infinity(), weights[end..].iter().sum());
        }
        out
    }

    fn push(&mut self, mean: T, weight: u32) {
        self.means.push(mean);
        self.weights.push(weight);
    }

    pub fn merge(&self, other: &Self, delta: T) -> Result<Self, &'static str> {
        // Two-run itertools::kmerge compares (mean, weight) lexicographically.
        // Equal heads leave the current run at the root, including after a swap.
        let mut active = self.means.iter().zip(&self.weights).peekable();
        let mut next = other.means.iter().zip(&other.weights).peekable();
        let mut means = Vec::with_capacity(active.len() + next.len());
        let mut weights = Vec::with_capacity(means.capacity());
        while let (Some(a), Some(b)) = (active.peek(), next.peek()) {
            if b < a {
                std::mem::swap(&mut active, &mut next);
            }
            let (&m, &w) = active.next().unwrap();
            means.push(m);
            weights.push(w);
        }
        for (&m, &w) in active.chain(next) {
            means.push(m);
            weights.push(w);
        }
        Self::from_means_weights(&means, &weights, delta)
    }

    pub fn quantile(&self, x: T) -> T {
        let total = T::from(self.weights.iter().sum::<u32>()).unwrap();
        if total == T::zero() {
            return total;
        }
        if x == T::zero() {
            return self.means[0];
        }
        let search = x * total;
        let two = T::from(2).unwrap();
        let mut previous_mean = self.means[0];
        let mut previous_weight = T::from(self.weights[0]).unwrap();
        let mut previous_position = previous_weight / two;
        for (&m, &w) in self.means[1..].iter().zip(&self.weights[1..]) {
            let w = T::from(w).unwrap();
            let diff = (w + previous_weight) / two;
            let next_position = previous_position + diff;
            if search <= next_position {
                let z1 = search - previous_position;
                let z2 = next_position - search;
                return (previous_mean * z2 + m * z1) / diff;
            }
            previous_weight = w;
            previous_mean = m;
            previous_position = next_position;
        }
        *self.means.last().unwrap()
    }

    pub fn trimmed_mean(&self, lower: T, upper: T) -> T {
        let total = T::from(self.weights.iter().sum::<u32>()).unwrap();
        let min_count = lower * total;
        let max_count = upper * total;
        let mut sum = T::zero();
        let mut count = T::zero();
        let mut current = T::zero();
        for (&mean, &weight) in self.means.iter().zip(&self.weights) {
            let mut portion = T::from(weight).unwrap();
            let next = current + portion;
            if next < min_count {
                current = next;
                continue;
            }
            if current < min_count {
                portion = next - min_count;
            } else if next > max_count {
                portion = portion - (next - max_count);
            }
            sum = sum + portion * mean;
            count = count + portion;
            if next >= max_count {
                break;
            }
            current = next;
        }
        sum / count
    }
}
