mod core;
mod scale;
mod traits;

use crate::{
    core::{compute, compute_quantile, compute_trimmed_mean, create_clusters, merge_clusters},
    traits::{FloatConst, TotalOrd},
};
use anyhow::Result;
use num::Float;

pub struct TDigest<T> {
    pub means: Vec<T>,
    pub weights: Vec<u32>,
}

impl<T> TDigest<T>
where
    T: Float + FloatConst + TotalOrd<T>,
{
    pub fn from_array(arr: &[T], delta: T) -> Result<Self> {
        let weights: Vec<u32> = vec![1; arr.len()];
        let (means, weights, _) = create_clusters(arr, &weights, delta)?;
        Ok(TDigest { means, weights })
    }

    pub fn from_means_weights(arr: &[T], weights: &[u32], delta: T) -> Result<Self> {
        let mask = vec![true; arr.len()];
        let (means, weights, _) = compute(arr, weights, &mask, delta)?;
        Ok(TDigest { means, weights })
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

    pub fn merge(&self, other: &Self, delta: T) -> Result<Self> {
        let (means, weights, _) = merge_clusters(
            &self.means,
            &self.weights,
            &other.means,
            &other.weights,
            delta,
        )?;
        Ok(Self { means, weights })
    }

    pub fn n_zero_weights(&self) -> Result<usize> {
        Ok(self.weights.iter().filter(|&w| *w == 0).count())
    }
}
