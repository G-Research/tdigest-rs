//! Merge helpers for TDigest centroids.
//!
//! This module provides two utilities used by the TDigest builder/merge path:
//! - `MergeByMean<F>`: merges a slice of centroids with a sorted slice of values
//!   into a single centroid stream ordered by mean.
//! - `KWayCentroidMerge<F>`: k-way merge of multiple centroid runs, ordered by mean.
//! - `normalize_stream`: tiny normalization pass used by the compressor (kept here
//!   for minimal churn), coalescing equal-mean runs into singleton piles and
//!   validating sort order.
//!
//! The API uses `mean_f64()` / `weight_f64()` to keep math in `f64`,
//! and constructs new centroids via `Centroid::<F>::new_singleton_f64(...)`
//! when materializing scalars.

use ordered_float::FloatCore;
use ordered_float::OrderedFloat;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::tdigest::centroids::Centroid;
use crate::tdigest::precision::FloatLike;

/* =============================================================================
 * MergeByMean
 * ============================================================================= */

pub struct MergeByMean<'a, F: FloatLike + FloatCore> {
    centroids: &'a [Centroid<F>],
    values_sorted: &'a [F],
    i: usize,
    j: usize,
    remaining: usize,
}

impl<'a, F: FloatLike + FloatCore> MergeByMean<'a, F> {
    /// Merge two sorted sources by mean:
    /// - `centroids`: already-sorted by mean (TDigest invariant)
    /// - `values_sorted`: raw scalar values sorted ascending
    pub fn from_centroids_and_values(centroids: &'a [Centroid<F>], values_sorted: &'a [F]) -> Self {
        Self {
            centroids,
            values_sorted,
            i: 0,
            j: 0,
            remaining: centroids.len() + values_sorted.len(),
        }
    }
}

impl<'a, F: FloatLike + FloatCore> Iterator for MergeByMean<'a, F> {
    type Item = Centroid<F>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.i >= self.centroids.len() && self.j >= self.values_sorted.len() {
            return None;
        }

        let out = if self.i >= self.centroids.len() {
            let v = self.values_sorted[self.j].to_f64();
            self.j += 1;
            Centroid::<F>::new_singleton_f64(v, 1.0)
        } else if self.j >= self.values_sorted.len() {
            let c = self.centroids[self.i];
            self.i += 1;
            c
        } else {
            let cm = self.centroids[self.i].mean_f64();
            let vm = self.values_sorted[self.j].to_f64();
            if cm <= vm {
                let c = self.centroids[self.i];
                self.i += 1;
                c
            } else {
                self.j += 1;
                Centroid::<F>::new_singleton_f64(vm, 1.0)
            }
        };

        self.remaining = self.remaining.saturating_sub(1);
        Some(out)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<'a, F: FloatLike + FloatCore> ExactSizeIterator for MergeByMean<'a, F> {}

/* =============================================================================
 * K-way merge of centroid runs
 * ============================================================================= */

pub struct KWayCentroidMerge<'a, F: FloatLike + FloatCore> {
    runs: Vec<&'a [Centroid<F>]>,
    heads: BinaryHeap<RunHead>,
    remaining: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RunHead {
    mean: OrderedFloat<f64>,
    run_idx: usize,
    elem_idx: usize,
}

impl RunHead {
    #[inline]
    fn new(mean: f64, run_idx: usize, elem_idx: usize) -> Self {
        Self {
            mean: OrderedFloat(mean),
            run_idx,
            elem_idx,
        }
    }
}

impl Ord for RunHead {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is max-first. Reverse ordering to pop the smallest mean.
        other
            .mean
            .cmp(&self.mean)
            .then_with(|| other.run_idx.cmp(&self.run_idx))
            .then_with(|| other.elem_idx.cmp(&self.elem_idx))
    }
}

impl PartialOrd for RunHead {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[inline]
fn is_non_decreasing_by_mean<F: FloatLike + FloatCore>(run: &[Centroid<F>]) -> bool {
    run.windows(2)
        .all(|w| w[0].mean_f64().total_cmp(w[1].mean_f64()) != Ordering::Greater)
}

impl<'a, F: FloatLike + FloatCore> KWayCentroidMerge<'a, F> {
    /// Build a streaming k-way merge over sorted centroid runs.
    ///
    /// The iterator yields centroids in non-decreasing mean order using a
    /// min-heap over run heads (`O(total_len * log(k))`).
    pub fn from_runs(runs: &'a [&'a [Centroid<F>]]) -> Self {
        let mut owned_runs = Vec::with_capacity(runs.len());
        let mut heads = BinaryHeap::with_capacity(runs.len());
        let mut remaining = 0usize;

        for (run_idx, run) in runs.iter().copied().enumerate() {
            debug_assert!(
                is_non_decreasing_by_mean(run),
                "k-way merge requires each run sorted by mean"
            );
            remaining += run.len();
            if let Some(first) = run.first().copied() {
                heads.push(RunHead::new(first.mean_f64(), run_idx, 0));
            }
            owned_runs.push(run);
        }

        Self {
            runs: owned_runs,
            heads,
            remaining,
        }
    }
}

impl<'a, F: FloatLike + FloatCore> Iterator for KWayCentroidMerge<'a, F> {
    type Item = Centroid<F>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let head = self.heads.pop()?;
        let run = &self.runs[head.run_idx];
        let out = run[head.elem_idx];

        let next_idx = head.elem_idx + 1;
        if let Some(next) = run.get(next_idx) {
            self.heads
                .push(RunHead::new(next.mean_f64(), head.run_idx, next_idx));
        }

        self.remaining = self.remaining.saturating_sub(1);
        Some(out)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<'a, F: FloatLike + FloatCore> ExactSizeIterator for KWayCentroidMerge<'a, F> {}

#[cfg(test)]
pub(crate) fn merge_runs_concat_sort<F: FloatLike + FloatCore>(
    runs: &[&[Centroid<F>]],
) -> Vec<Centroid<F>> {
    let total_len = runs.iter().map(|r| r.len()).sum();
    let mut all: Vec<Centroid<F>> = Vec::with_capacity(total_len);
    for run in runs {
        all.extend_from_slice(run);
    }
    all.sort_by(|a, b| a.mean_f64().total_cmp(b.mean_f64()));
    all
}

/* =============================================================================
 * Normalization shim used by compressor (Stage 1)
 * ============================================================================= */

/// Output of `normalize_stream`.
pub struct Normalized<F: FloatLike + FloatCore> {
    pub out: Vec<Centroid<F>>,
    pub total_w: f64,
    pub total_mw: f64,
    pub min: f64,
    pub max: f64,
}

/// Normalize an input stream of centroids:
/// - **validate** non-decreasing means (panic on violation to match tests)
/// - **coalesce** adjacent equal-mean centroids into a **singleton pile**
/// - **accumulate** total weight/sum and min/max
pub fn normalize_stream<F, I>(items: I) -> Normalized<F>
where
    F: FloatLike + FloatCore,
    I: IntoIterator<Item = Centroid<F>>,
{
    let mut it = items.into_iter();
    let mut out: Vec<Centroid<F>> = Vec::new();

    let mut total_w = 0.0_f64;
    let mut total_mw = 0.0_f64;

    let first = match it.next() {
        Some(c) => c,
        None => {
            return Normalized {
                out,
                total_w,
                total_mw,
                min: 0.0,
                max: 0.0,
            }
        }
    };

    let mut cur_mean = first.mean_f64();
    let mut cur_w = first.weight_f64();

    let min_mean = cur_mean;
    let mut max_mean = cur_mean;

    // Fold contiguous equal-means; validate ordering.
    for c in it {
        let m = c.mean_f64();
        let w = c.weight_f64();

        if m < cur_mean {
            panic!("compress_into requires non-decreasing means");
        }

        if m == cur_mean {
            cur_w += w;
        } else {
            // Flush previous run as a **singleton** pile.
            out.push(Centroid::<F>::new_singleton_f64(cur_mean, cur_w));
            total_w += cur_w;
            total_mw += cur_mean * cur_w;

            // Start new run
            cur_mean = m;
            cur_w = w;
        }

        if m > max_mean {
            max_mean = m;
        }
    }

    // Flush last run
    out.push(Centroid::<F>::new_singleton_f64(cur_mean, cur_w));
    total_w += cur_w;
    total_mw += cur_mean * cur_w;

    Normalized {
        out,
        total_w,
        total_mw,
        min: min_mean,
        max: max_mean,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    fn merge_by_mean_baseline(
        centroids: &[Centroid<f64>],
        values_sorted: &[f64],
    ) -> Vec<Centroid<f64>> {
        let mut out = Vec::with_capacity(centroids.len() + values_sorted.len());
        let mut i = 0usize;
        let mut j = 0usize;

        while i < centroids.len() && j < values_sorted.len() {
            if centroids[i].mean_f64() <= values_sorted[j] {
                out.push(centroids[i]);
                i += 1;
            } else {
                out.push(Centroid::<f64>::new_singleton_f64(values_sorted[j], 1.0));
                j += 1;
            }
        }
        while i < centroids.len() {
            out.push(centroids[i]);
            i += 1;
        }
        while j < values_sorted.len() {
            out.push(Centroid::<f64>::new_singleton_f64(values_sorted[j], 1.0));
            j += 1;
        }
        out
    }

    fn random_run(rng: &mut StdRng, len: usize) -> Vec<Centroid<f64>> {
        let mut out = Vec::with_capacity(len);
        let mut mean = rng.random_range(-1_000.0..1_000.0);
        for _ in 0..len {
            // Keep each run sorted; allow repeats to exercise equal-mean handling.
            mean += rng.random_range(0.0..4.0);
            let w = rng.random_range(0.5..8.0);
            let c = if rng.random_bool(0.35) {
                Centroid::new_atomic_f64(mean, w)
            } else {
                Centroid::new_mixed_f64(mean, w)
            };
            out.push(c);
        }
        out
    }

    #[test]
    fn merge_by_mean_stream_matches_baseline() {
        let mut rng = StdRng::seed_from_u64(0xBEEF_5678_9ABC);

        for trial in 0..300usize {
            let clen = rng.random_range(0..=300usize);
            let vlen = rng.random_range(0..=400usize);

            let mut centroids = Vec::with_capacity(clen);
            let mut mean = rng.random_range(-500.0..500.0);
            for _ in 0..clen {
                mean += rng.random_range(0.01..5.0);
                let w = rng.random_range(0.5..6.0);
                centroids.push(if rng.random_bool(0.4) {
                    Centroid::<f64>::new_atomic_f64(mean, w)
                } else {
                    Centroid::<f64>::new_mixed_f64(mean, w)
                });
            }

            let mut values = Vec::with_capacity(vlen);
            for _ in 0..vlen {
                values.push(rng.random_range(-500.0..500.0));
            }
            values.sort_by(|a, b| a.total_cmp(*b));

            let expected = merge_by_mean_baseline(&centroids, &values);
            let got: Vec<_> = MergeByMean::from_centroids_and_values(&centroids, &values).collect();
            assert_eq!(
                got, expected,
                "streaming MergeByMean diverged from baseline at trial={trial}"
            );
        }
    }

    #[test]
    fn kway_heap_merge_matches_concat_sort_baseline() {
        let mut rng = StdRng::seed_from_u64(0x5EED_1234_ABCD);

        for trial in 0..256usize {
            let run_count = rng.random_range(1..=48usize);
            let mut runs: Vec<Vec<Centroid<f64>>> = Vec::with_capacity(run_count);
            for _ in 0..run_count {
                let len = rng.random_range(0..=220usize);
                runs.push(random_run(&mut rng, len));
            }

            let refs: Vec<&[Centroid<f64>]> = runs.iter().map(|r| r.as_slice()).collect();
            let expected = merge_runs_concat_sort(&refs);
            let actual: Vec<Centroid<f64>> = KWayCentroidMerge::from_runs(&refs).collect();

            assert_eq!(
                actual, expected,
                "heap k-way output diverged from concat+sort baseline at trial={trial}"
            );
        }
    }
}
