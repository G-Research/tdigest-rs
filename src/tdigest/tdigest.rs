use crate::{TdError, TdResult};
use ordered_float::{FloatCore, OrderedFloat};

use crate::tdigest::centroids::{is_sorted_strict_by_mean, Centroid};
use crate::tdigest::compressor::compress_into;
use crate::tdigest::merges::{KWayCentroidMerge, MergeByMean};
use crate::tdigest::precision::{FloatLike, Precision};
use crate::tdigest::scale::ScaleFamily;
use crate::tdigest::singleton_policy::SingletonPolicy;
use crate::tdigest::wire::{self, WireDecodedDigest, WireError, WireVersion};

/// TDigest orchestration + public API, generic over centroid float storage `F` (`f32` or `f64`).
///
/// - Centroids (means/weights) and `min/max` are stored in `F` to control **in-memory size**.
/// - `sum` and `count` remain `f64` for **stable accumulation** and to match the on-wire schema.
/// - The compressor/merges do their internal math in `f64` and convert at the boundaries.
#[derive(Debug, PartialEq, Clone)]
pub struct TDigest<F: FloatLike + FloatCore> {
    centroids: Vec<Centroid<F>>,
    max_size: usize,
    sum: f64,   // ∑x over raw data (f64 for stability / wire)
    count: f64, // ∑w over raw data (f64 for stability / wire)
    max: OrderedFloat<F>,
    min: OrderedFloat<F>,
    scale: ScaleFamily,
    policy: SingletonPolicy, // interpreted as atomic/edge policy
    legacy_delta: Option<f64>,
}

pub type TDigestF64 = TDigest<f64>;
pub type TDigestF32 = TDigest<f32>;

impl<F: FloatLike + FloatCore> Default for TDigest<F> {
    fn default() -> Self {
        Self {
            centroids: Vec::new(),
            max_size: 1000,
            sum: 0.0,
            count: 0.0,
            max: OrderedFloat::from(F::from_f64(f64::NAN)),
            min: OrderedFloat::from(F::from_f64(f64::NAN)),
            scale: ScaleFamily::K2,
            policy: SingletonPolicy::Use,
            legacy_delta: None,
        }
    }
}

/* =============================================================================
 * Input adapters
 * ============================================================================= */

/// Accepts containers already typed as `F` and yields a `Vec<F>`.
pub trait IntoVecF<F: FloatLike> {
    fn into_vec_f(self) -> Vec<F>;
}
impl<F: FloatLike> IntoVecF<F> for Vec<F> {
    #[inline]
    fn into_vec_f(self) -> Vec<F> {
        self
    }
}
impl<F: FloatLike> IntoVecF<F> for &[F] {
    #[inline]
    fn into_vec_f(self) -> Vec<F> {
        self.to_vec()
    }
}

fn ensure_no_non_finite_values<F: FloatLike + FloatCore>(values: &[F]) -> TdResult<()> {
    if values.iter().any(|v| !v.is_finite()) {
        return Err(TdError::NonFiniteInput {
            context: "sample value (NaN or ±inf)",
        });
    }
    Ok(())
}

#[inline]
fn ensure_no_invalid_weights(weights: &[f64]) -> TdResult<()> {
    if weights.iter().any(|w| !w.is_finite()) {
        return Err(TdError::NonFiniteInput {
            context: "sample weight (NaN or ±inf)",
        });
    }
    if weights.iter().any(|w| *w <= 0.0) {
        return Err(TdError::InvalidWeight {
            context: "sample weight must be > 0",
        });
    }
    Ok(())
}

#[inline]
fn ensure_weighted_input_lengths(values_len: usize, weights_len: usize) -> TdResult<()> {
    if values_len != weights_len {
        return Err(TdError::MismatchedInputLength {
            context: "weighted values/weights",
            values_len,
            weights_len,
        });
    }
    Ok(())
}

#[inline]
fn ensure_positive_finite_scale_factor(factor: f64, context: &'static str) -> TdResult<()> {
    if !factor.is_finite() || factor <= 0.0 {
        return Err(TdError::InvalidScaleFactor { context });
    }
    Ok(())
}

#[inline]
fn rescaled_centroid<F: FloatLike + FloatCore>(
    c: &Centroid<F>,
    mean_scale: f64,
    weight_scale: f64,
) -> Centroid<F> {
    let mean = c.mean_f64() * mean_scale;
    let weight = c.weight_f64() * weight_scale;

    if c.is_atomic_unit() {
        if (weight - 1.0).abs() <= f64::EPSILON {
            Centroid::<F>::new_atomic_unit_f64(mean)
        } else {
            Centroid::<F>::new_atomic_f64(mean, weight)
        }
    } else if c.is_atomic() {
        Centroid::<F>::new_atomic_f64(mean, weight)
    } else {
        Centroid::<F>::new_mixed_f64(mean, weight)
    }
}

#[inline]
fn cast_centroid<From, To>(c: &Centroid<From>) -> Centroid<To>
where
    From: FloatLike + FloatCore,
    To: FloatLike + FloatCore,
{
    let mean = c.mean_f64();
    let weight = c.weight_f64();
    if c.is_atomic_unit() {
        Centroid::<To>::new_atomic_unit_f64(mean)
    } else if c.is_atomic() {
        Centroid::<To>::new_atomic_f64(mean, weight)
    } else {
        Centroid::<To>::new_mixed_f64(mean, weight)
    }
}

/* =============================================================================
 * Options / Builder
 * ============================================================================= */

#[derive(Debug, Clone, Copy)]
pub struct DigestOptions<F: FloatLike> {
    pub max_size: usize,
    pub scale: ScaleFamily,
    /// Atomic/edge policy (kept as `SingletonPolicy` for API stability).
    pub singleton_policy: SingletonPolicy,
    _marker: std::marker::PhantomData<F>,
}
impl<F: FloatLike> Default for DigestOptions<F> {
    fn default() -> Self {
        Self {
            max_size: 1000,
            scale: ScaleFamily::K2,
            singleton_policy: SingletonPolicy::Use,
            _marker: std::marker::PhantomData,
        }
    }
}

/// Builder for [`TDigest<F>`].
///
/// Use the builder when you want to:
/// - construct an empty digest with chosen parameters, or
/// - seed a digest with *existing centroids* and *data-level stats*.
#[derive(Debug, Clone)]
pub struct TDigestBuilder<F: FloatLike + FloatCore> {
    max_size: usize,
    scale: ScaleFamily,
    policy: SingletonPolicy,
    legacy_delta: Option<f64>,
    init_centroids: Option<Vec<Centroid<F>>>,
    init_stats: Option<DigestStats>,
    override_max_size: Option<usize>,
}
impl<F: FloatLike + FloatCore> Default for TDigestBuilder<F> {
    fn default() -> Self {
        Self {
            max_size: 1000,
            scale: ScaleFamily::K2,
            policy: SingletonPolicy::Use,
            legacy_delta: None,
            init_centroids: None,
            init_stats: None,
            override_max_size: None,
        }
    }
}
impl<F: FloatLike + FloatCore> TDigestBuilder<F> {
    /// Create a new builder with defaults.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the compression parameter (`max_size`).
    #[inline]
    pub fn max_size(mut self, n: usize) -> Self {
        self.max_size = n;
        self
    }

    /// Choose the scale family used by the k-limit (Stage **3**).
    #[inline]
    pub fn scale(mut self, s: ScaleFamily) -> Self {
        self.scale = s;
        self
    }

    /// Set the atomic/edge policy influencing Stages **2** and **6**.
    #[inline]
    pub fn singleton_policy(mut self, p: SingletonPolicy) -> Self {
        self.policy = p;
        self
    }

    /// Enable legacy tdigest-rs `delta` mode.
    #[inline]
    pub fn legacy_delta(mut self, delta: f64) -> Self {
        assert!(
            delta.is_finite() && delta > 0.0,
            "legacy_delta must be finite and > 0"
        );
        self.legacy_delta = Some(delta);
        self
    }

    /// Seed with centroids + data-level stats.
    pub fn with_centroids_and_stats(
        mut self,
        centroids: Vec<Centroid<F>>,
        stats: DigestStats,
    ) -> Self {
        self.init_centroids = Some(centroids);
        self.init_stats = Some(stats);
        self
    }

    /// Seed with centroids and raw stats (convenience mirror of [`TDigest::new`]).
    pub fn with_centroids(
        mut self,
        centroids: Vec<Centroid<F>>,
        sum: f64,
        count: f64,
        max: F,
        min: F,
        max_size_override: usize,
    ) -> Self {
        let stats = DigestStats {
            data_sum: sum,
            total_weight: count,
            data_min: min.to_f64(),
            data_max: max.to_f64(),
        };
        self.init_centroids = Some(centroids);
        self.init_stats = Some(stats);
        self.override_max_size = Some(max_size_override);
        self
    }

    /// Build the digest, seeding if seeds were provided.
    pub fn build(self) -> TDigest<F> {
        if let (Some(cents), Some(st)) = (self.init_centroids, self.init_stats) {
            let max_size = self.override_max_size.unwrap_or(self.max_size);
            TDigest {
                centroids: cents,
                max_size,
                sum: st.data_sum,
                count: st.total_weight,
                min: OrderedFloat::from(F::from_f64(st.data_min)),
                max: OrderedFloat::from(F::from_f64(st.data_max)),
                scale: self.scale,
                policy: self.policy,
                legacy_delta: self.legacy_delta,
            }
        } else {
            TDigest {
                centroids: Vec::new(),
                max_size: self.max_size,
                sum: 0.0,
                count: 0.0,
                max: OrderedFloat::from(F::from_f64(f64::NAN)),
                min: OrderedFloat::from(F::from_f64(f64::NAN)),
                scale: self.scale,
                policy: self.policy,
                legacy_delta: self.legacy_delta,
            }
        }
    }
}

/* =============================================================================
 * Digest
 * ============================================================================= */

/// Data-level stats for seeding a digest (∑x/∑w/min/max of the **raw data**, not centroids).
#[derive(Debug, Clone, Copy)]
pub struct DigestStats {
    /// Sum of raw data values (∑x), not sum of centroid means.
    pub data_sum: f64,
    /// Total weight (∑w). For raw samples this equals the sample count.
    pub total_weight: f64,
    /// Minimum observed raw value (in f64; converted to F in builder).
    pub data_min: f64,
    /// Maximum observed raw value (in f64; converted to F in builder).
    pub data_max: f64,
}

impl<F: FloatLike + FloatCore> TDigest<F> {
    #[inline]
    pub fn max(&self) -> f64 {
        self.max.into_inner().to_f64()
    }
    #[inline]
    pub fn min(&self) -> f64 {
        self.min.into_inner().to_f64()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.centroids.is_empty()
    }

    /// Empty-for-queries semantics used by front-ends:
    /// treat either zero count OR no centroids as empty.
    #[inline]
    pub fn is_effectively_empty(&self) -> bool {
        self.count == 0.0 || self.centroids.is_empty()
    }

    /// CDF wrapper that yields NaN for all probes when the digest is effectively empty.
    /// This centralizes the “empty → NaN” rule.
    #[inline]
    pub fn cdf_or_nan(&self, xs: &[f64]) -> Vec<f64> {
        if self.is_effectively_empty() {
            let mut v = Vec::with_capacity(xs.len());
            v.resize(xs.len(), f64::NAN);
            v
        } else {
            self.cdf(xs)
        }
    }

    #[inline]
    pub fn centroids(&self) -> &[Centroid<F>] {
        &self.centroids
    }

    /// Entry point for fluent construction.
    #[inline]
    pub fn builder() -> TDigestBuilder<F> {
        TDigestBuilder::default()
    }

    /// Report the precision implied by the type parameter.
    #[inline]
    pub fn precision(&self) -> Precision {
        Precision::of_type::<F>()
    }

    /// Cast this digest to a different centroid-storage precision.
    ///
    /// This preserves configuration and digest-level stats while converting
    /// centroid means/weights to the target float type.
    pub fn cast_precision<T>(&self) -> TDigest<T>
    where
        T: FloatLike + FloatCore,
    {
        let cents: Vec<Centroid<T>> = self
            .centroids()
            .iter()
            .map(|c| cast_centroid::<F, T>(c))
            .collect();
        let stats = DigestStats {
            data_sum: self.sum(),
            total_weight: self.count(),
            data_min: self.min(),
            data_max: self.max(),
        };
        let mut builder = TDigest::<T>::builder()
            .max_size(self.max_size())
            .scale(self.scale())
            .singleton_policy(self.singleton_policy());
        if let Some(delta) = self.legacy_delta() {
            builder = builder.legacy_delta(delta);
        }
        builder.with_centroids_and_stats(cents, stats).build()
    }

    #[inline]
    pub(crate) fn set_sum(&mut self, s: f64) {
        self.sum = s;
    }
    #[inline]
    pub(crate) fn set_count(&mut self, c: f64) {
        self.count = c;
    }
    #[inline]
    pub(crate) fn set_min(&mut self, v: f64) {
        self.min = OrderedFloat::from(F::from_f64(v));
    }
    #[inline]
    pub(crate) fn set_max(&mut self, v: f64) {
        self.max = OrderedFloat::from(F::from_f64(v));
    }

    /// Build from **unsorted** values (convenience over [`TDigest::merge_unsorted`]).
    pub fn from_unsorted(values: &[F], max_size: usize) -> TdResult<TDigest<F>> {
        ensure_no_non_finite_values(values)?;
        let base = Self::builder().max_size(max_size).build();
        base.merge_unsorted(values.to_vec())
    }

    /// Build from weighted values in any order.
    ///
    /// Equivalent to constructing an empty digest with `max_size` and calling
    /// [`TDigest::merge_weighted_unsorted`].
    pub fn from_weighted_unsorted(
        values: &[F],
        weights: &[f64],
        max_size: usize,
    ) -> TdResult<TDigest<F>> {
        let base = Self::builder().max_size(max_size).build();
        base.merge_weighted_unsorted(values, weights)
    }

    /// The configured scale family used by the compressor’s k-limit.
    #[inline]
    pub fn scale(&self) -> ScaleFamily {
        self.scale
    }

    /// The configured atomic/edge policy (`SingletonPolicy`).
    #[inline]
    pub fn singleton_policy(&self) -> SingletonPolicy {
        self.policy
    }

    #[inline]
    pub fn legacy_delta(&self) -> Option<f64> {
        self.legacy_delta
    }

    /// Ingest **unsorted** values; behavior matches [`TDigest::merge_sorted`] after sorting.
    pub fn merge_unsorted(&self, mut unsorted_values: Vec<F>) -> TdResult<TDigest<F>> {
        ensure_no_non_finite_values(&unsorted_values)?;
        unsorted_values.sort_by(|a, b| a.total_cmp(*b));
        self.merge_sorted(unsorted_values)
    }

    /// Ingest **sorted** values by interleaving with existing centroids and running the pipeline.
    pub fn merge_sorted(&self, sorted_values: Vec<F>) -> TdResult<TDigest<F>> {
        ensure_no_non_finite_values(&sorted_values)?;
        if sorted_values.is_empty() {
            return Ok(self.clone());
        }
        let mut result = self.new_result_for_values(&sorted_values);

        let stream = MergeByMean::from_centroids_and_values(&self.centroids, &sorted_values);

        let compressed: Vec<Centroid<F>> = compress_into(&mut result, self.max_size, stream);
        result.centroids = compressed;

        debug_assert!(
            is_sorted_strict_by_mean(&result.centroids),
            "duplicate centroid means after merge"
        );
        Ok(result)
    }

    /// Ingest weighted values (`values[i]` with `weights[i]`) in any order.
    ///
    /// Rules:
    /// - `values.len()` must equal `weights.len()`.
    /// - values must be finite.
    /// - weights must be finite and strictly positive.
    ///
    /// Each pair is treated as exact atomic mass at the given value, then merged
    /// through the same digest merge/compression path.
    pub fn merge_weighted_unsorted(&self, values: &[F], weights: &[f64]) -> TdResult<TDigest<F>> {
        ensure_weighted_input_lengths(values.len(), weights.len())?;
        ensure_no_non_finite_values(values)?;
        ensure_no_invalid_weights(weights)?;
        if values.is_empty() {
            return Ok(self.clone());
        }

        let mut pairs: Vec<(F, f64)> = values
            .iter()
            .copied()
            .zip(weights.iter().copied())
            .collect();
        pairs.sort_by(|a, b| a.0.total_cmp(b.0));

        let mut cents: Vec<Centroid<F>> = Vec::with_capacity(pairs.len());
        let mut data_sum = 0.0_f64;
        let mut total_weight = 0.0_f64;
        for (v, w) in &pairs {
            let mean = <F as FloatLike>::to_f64(*v);
            if (*w - 1.0).abs() <= f64::EPSILON {
                cents.push(Centroid::<F>::new_atomic_unit_f64(mean));
            } else {
                cents.push(Centroid::<F>::new_atomic_f64(mean, *w));
            }
            data_sum += mean * *w;
            total_weight += *w;
        }

        let stats = DigestStats {
            data_sum,
            total_weight,
            data_min: <F as FloatLike>::to_f64(pairs[0].0),
            data_max: <F as FloatLike>::to_f64(pairs[pairs.len() - 1].0),
        };
        let mut weighted_builder = TDigest::<F>::builder()
            .max_size(self.max_size)
            .scale(self.scale)
            .singleton_policy(self.policy);
        if let Some(delta) = self.legacy_delta {
            weighted_builder = weighted_builder.legacy_delta(delta);
        }
        let weighted = weighted_builder
            .with_centroids_and_stats(cents, stats)
            .build();

        Ok(TDigest::merge_digests(vec![self.clone(), weighted]))
    }

    /// Merge multiple digests by k-way merging their centroid runs and sending the result through
    /// the same pipeline used for raw values.
    pub fn merge_digests(digests: Vec<TDigest<F>>) -> TDigest<F> {
        let mut chosen: Option<(usize, ScaleFamily, SingletonPolicy, Option<f64>)> = None;
        let mut runs: Vec<&[Centroid<F>]> = Vec::with_capacity(digests.len());
        let mut total_count = 0.0_f64;
        let mut min = OrderedFloat::from(F::from_f64(f64::INFINITY));
        let mut max = OrderedFloat::from(F::from_f64(f64::NEG_INFINITY));

        for d in &digests {
            let n = d.count();
            if n > 0.0 && !d.centroids.is_empty() {
                if chosen.is_none() {
                    chosen = Some((d.max_size, d.scale, d.policy, d.legacy_delta));
                }
                total_count += n;
                min = std::cmp::min(min, d.min);
                max = std::cmp::max(max, d.max);
                runs.push(&d.centroids);
            }
        }
        if total_count == 0.0 {
            return TDigest::default();
        }
        let (chosen_max_size, chosen_scale, chosen_policy, chosen_legacy_delta) = chosen.unwrap();

        let mut result = TDigest::<F> {
            centroids: Vec::new(),
            max_size: chosen_max_size,
            sum: 0.0,
            count: total_count,
            max,
            min,
            scale: chosen_scale,
            policy: chosen_policy,
            legacy_delta: chosen_legacy_delta,
        };

        let merged_stream = KWayCentroidMerge::from_runs(&runs);

        let compressed: Vec<Centroid<F>> =
            compress_into(&mut result, chosen_max_size, merged_stream);
        result.centroids = compressed;

        debug_assert!(
            is_sorted_strict_by_mean(&result.centroids),
            "duplicate centroid means after merge_digests"
        );
        result
    }

    /// In-place merge with another digest of the same precision.
    ///
    /// This is a convenience wrapper over [`TDigest::merge_digests`].
    pub fn merge(&mut self, other: &TDigest<F>) -> &mut Self {
        *self = TDigest::merge_digests(vec![self.clone(), other.clone()]);
        self
    }

    /// In-place merge with multiple digests of the same precision.
    ///
    /// This is a convenience wrapper over [`TDigest::merge_digests`].
    pub fn merge_many(&mut self, others: &[TDigest<F>]) -> &mut Self {
        if others.is_empty() {
            return self;
        }
        let mut all = Vec::with_capacity(others.len() + 1);
        all.push(self.clone());
        all.extend_from_slice(others);
        *self = TDigest::merge_digests(all);
        self
    }

    /// Add a single value in-place.
    pub fn add(&mut self, value: F) -> TdResult<&mut Self> {
        self.add_many([value].as_slice())
    }

    /// Add one or more values in-place.
    pub fn add_many<A: IntoVecF<F>>(&mut self, values: A) -> TdResult<&mut Self> {
        let vals = values.into_vec_f();
        if vals.is_empty() {
            return Ok(self);
        }
        let merged = self.merge_unsorted(vals)?;
        *self = merged;
        Ok(self)
    }

    /// Add a single weighted value in-place.
    pub fn add_weighted(&mut self, value: F, weight: f64) -> TdResult<&mut Self> {
        self.add_weighted_many(&[value], &[weight])
    }

    /// Add weighted values in-place (`values[i]` with `weights[i]`).
    pub fn add_weighted_many(&mut self, values: &[F], weights: &[f64]) -> TdResult<&mut Self> {
        let merged = self.merge_weighted_unsorted(values, weights)?;
        *self = merged;
        Ok(self)
    }

    /// Scale all centroid weights by `factor`.
    ///
    /// Effects:
    /// - centroid weights are multiplied by `factor`
    /// - digest `count` and `sum` are multiplied by `factor`
    /// - `min`/`max` are unchanged
    ///
    /// This preserves quantile/cdf/median shape while re-weighting the digest.
    pub fn scale_weights(&mut self, factor: f64) -> TdResult<&mut Self> {
        ensure_positive_finite_scale_factor(factor, "weight scale factor must be finite and > 0")?;
        if self.centroids.is_empty() || self.count == 0.0 {
            return Ok(self);
        }

        self.centroids = self
            .centroids
            .iter()
            .map(|c| rescaled_centroid(c, 1.0, factor))
            .collect();
        self.count *= factor;
        self.sum *= factor;
        Ok(self)
    }

    /// Scale centroid means by `factor` (value-axis scaling).
    ///
    /// Effects:
    /// - centroid means are multiplied by `factor`
    /// - digest `min`/`max` and `sum` are multiplied by `factor`
    /// - centroid weights and digest `count` are unchanged
    ///
    /// Note: this requires a strictly positive factor to preserve monotone ordering.
    pub fn scale_values(&mut self, factor: f64) -> TdResult<&mut Self> {
        ensure_positive_finite_scale_factor(factor, "value scale factor must be finite and > 0")?;
        if self.centroids.is_empty() || self.count == 0.0 {
            return Ok(self);
        }

        self.centroids = self
            .centroids
            .iter()
            .map(|c| rescaled_centroid(c, factor, 1.0))
            .collect();

        self.set_min(self.min() * factor);
        self.set_max(self.max() * factor);
        self.sum *= factor;
        Ok(self)
    }

    /* ===========================
     * Small utilities
     * =========================== */

    /// Construct a digest from raw centroids + stats.
    ///
    /// If `centroids.len() > max_size`, this creates a temporary large digest (via the builder)
    /// and then merges down through the pipeline to respect capacity. This mirrors the case where
    /// a producer generates more than `max_size` centroids and requires compression.
    pub fn new(
        centroids: Vec<Centroid<F>>,
        sum: f64,
        count: f64,
        max: F,
        min: F,
        max_size: usize,
    ) -> Self {
        if centroids.len() <= max_size {
            TDigest {
                centroids,
                max_size,
                sum,
                count,
                max: OrderedFloat::from(max),
                min: OrderedFloat::from(min),
                scale: ScaleFamily::K2,
                policy: SingletonPolicy::Use,
                legacy_delta: None,
            }
        } else {
            let sz = centroids.len();
            let digests = vec![
                TDigest::<F>::builder().max_size(100).build(),
                TDigest::<F>::builder()
                    .with_centroids(centroids, sum, count, max, min, sz)
                    .build(),
            ];
            Self::merge_digests(digests)
        }
    }

    /// Mean of the represented distribution (`∑x / ∑w`) in **f64** for stability.
    #[inline]
    pub fn mean(&self) -> f64 {
        let n = self.count;
        if n > 0.0 {
            self.sum / n
        } else {
            0.0
        }
    }

    /// Sum of raw values (∑x).
    #[inline]
    pub fn sum(&self) -> f64 {
        self.sum
    }

    /// Total weight (∑w). For raw samples, this equals the sample count.
    #[inline]
    pub fn count(&self) -> f64 {
        self.count
    }

    /// The configured compression parameter.
    #[inline]
    pub fn max_size(&self) -> usize {
        self.max_size
    }

    /// Borrow the internal centroids.
    #[inline]
    pub fn centroids_ref(&self) -> &[Centroid<F>] {
        &self.centroids
    }

    /// Prepare a result shell when ingesting a batch of **sorted** raw values.
    ///
    /// Sets `count` to existing `∑w + values.len()`, resets `sum` to be filled by Stage **1**,
    /// and updates min/max by comparing current bounds with the new batch.
    fn new_result_for_values(&self, values: &[F]) -> TDigest<F> {
        let mut r = TDigest::<F> {
            centroids: Vec::new(),
            max_size: self.max_size(),
            sum: 0.0,
            count: self.count() + (values.len() as f64),
            max: OrderedFloat::from(F::from_f64(f64::NAN)),
            min: OrderedFloat::from(F::from_f64(f64::NAN)),
            scale: self.scale,
            policy: self.policy,
            legacy_delta: self.legacy_delta,
        };

        let vmin = OrderedFloat::from(values[0]);
        let vmax = OrderedFloat::from(values[values.len() - 1]);
        if self.count() > 0.0 {
            r.min = std::cmp::min(self.min, vmin);
            r.max = std::cmp::max(self.max, vmax);
        } else {
            r.min = vmin;
            r.max = vmax;
        }
        r
    }

    /* ===========================
     * Convenience constructors
     * =========================== */

    /// Convenience: build from an arbitrary numeric container already typed as `F`, using options.
    pub fn from_array_with<A: IntoVecF<F>>(arr: A, opts: DigestOptions<F>) -> TdResult<TDigest<F>> {
        let vals = arr.into_vec_f();
        ensure_no_non_finite_values(&vals)?;
        let base = TDigest::<F>::builder()
            .max_size(opts.max_size)
            .scale(opts.scale)
            .singleton_policy(opts.singleton_policy)
            .build();
        base.merge_unsorted(vals)
    }

    /// Convenience: defaults (max_size=1000, scale=K2, policy=Use).
    pub fn from_array<A: IntoVecF<F>>(arr: A) -> TdResult<TDigest<F>> {
        Self::from_array_with(arr, DigestOptions::<F>::default())
    }
}

impl<F> TDigest<F>
where
    F: FloatLike + FloatCore,
{
    /// Encode this digest into the canonical TDIG binary format.
    /// Wire precision (centroid mean) follows `F` (f32 → f32, f64 → f64).
    #[inline]
    pub fn to_bytes(&self) -> Vec<u8> {
        wire::encode_digest(self)
    }

    /// Encode this digest into TDIG with an explicit wire version.
    #[inline]
    pub fn to_bytes_with_version(&self, version: WireVersion) -> Vec<u8> {
        wire::encode_digest_with_version(self, version)
    }
}

impl TDigest<f64> {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, WireError> {
        match wire::decode_digest(bytes)? {
            WireDecodedDigest::F64(td) => Ok(td),
            WireDecodedDigest::F32(td32) => Ok(td32.cast_precision::<f64>()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hint::black_box;
    use std::time::Instant;
    use testdata::{gen_dataset, DistKind};

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() <= eps * (1.0 + a.abs() + b.abs())
    }

    #[derive(Clone, Copy, Debug)]
    enum MergeStrategy {
        ConcatSort,
        HeapStream,
    }

    #[derive(Clone)]
    struct MergeScan<'a> {
        chosen_max_size: usize,
        chosen_scale: ScaleFamily,
        chosen_policy: SingletonPolicy,
        chosen_legacy_delta: Option<f64>,
        total_count: f64,
        min: OrderedFloat<f64>,
        max: OrderedFloat<f64>,
        runs: Vec<&'a [Centroid<f64>]>,
    }

    #[derive(Debug, Clone, Copy)]
    struct AccuracyMetrics {
        q_mae: f64,
        q_max: f64,
        cdf_mae: f64,
        cdf_max: f64,
    }

    #[derive(Clone, Copy)]
    struct PerfCase {
        label: &'static str,
        dist: DistKind,
        n: usize,
        shards: usize,
        max_size: usize,
    }

    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    fn allocated_now_bytes() -> usize {
        use jemalloc_ctl::{epoch, stats};
        epoch::advance().expect("jemalloc epoch advance");
        stats::allocated::read().expect("jemalloc allocated")
    }

    #[cfg(not(all(target_os = "linux", target_env = "gnu")))]
    fn allocated_now_bytes() -> usize {
        0
    }

    fn scan_non_empty_runs<'a>(digests: &'a [TDigest<f64>]) -> Option<MergeScan<'a>> {
        let mut chosen: Option<(usize, ScaleFamily, SingletonPolicy, Option<f64>)> = None;
        let mut runs: Vec<&[Centroid<f64>]> = Vec::with_capacity(digests.len());
        let mut total_count = 0.0_f64;
        let mut min = OrderedFloat::from(f64::INFINITY);
        let mut max = OrderedFloat::from(f64::NEG_INFINITY);

        for d in digests {
            let n = d.count();
            if n > 0.0 && !d.centroids.is_empty() {
                if chosen.is_none() {
                    chosen = Some((d.max_size, d.scale, d.policy, d.legacy_delta));
                }
                total_count += n;
                min = std::cmp::min(min, d.min);
                max = std::cmp::max(max, d.max);
                runs.push(&d.centroids);
            }
        }

        let (chosen_max_size, chosen_scale, chosen_policy, chosen_legacy_delta) = chosen?;

        Some(MergeScan {
            chosen_max_size,
            chosen_scale,
            chosen_policy,
            chosen_legacy_delta,
            total_count,
            min,
            max,
            runs,
        })
    }

    fn merge_result_shell(scan: &MergeScan<'_>) -> TDigest<f64> {
        TDigest::<f64> {
            centroids: Vec::new(),
            max_size: scan.chosen_max_size,
            sum: 0.0,
            count: scan.total_count,
            max: scan.max,
            min: scan.min,
            scale: scan.chosen_scale,
            policy: scan.chosen_policy,
            legacy_delta: scan.chosen_legacy_delta,
        }
    }

    fn merge_with_strategy(digests: &[TDigest<f64>], strategy: MergeStrategy) -> TDigest<f64> {
        let Some(scan) = scan_non_empty_runs(digests) else {
            return TDigest::<f64>::default();
        };
        let mut result = merge_result_shell(&scan);
        let compressed: Vec<Centroid<f64>> = match strategy {
            MergeStrategy::ConcatSort => {
                let merged = crate::tdigest::merges::merge_runs_concat_sort(&scan.runs);
                compress_into(&mut result, scan.chosen_max_size, merged)
            }
            MergeStrategy::HeapStream => {
                let merged = KWayCentroidMerge::from_runs(&scan.runs);
                compress_into(&mut result, scan.chosen_max_size, merged)
            }
        };
        result.centroids = compressed;
        result
    }

    fn merge_with_peak_alloc(
        digests: &[TDigest<f64>],
        strategy: MergeStrategy,
    ) -> (TDigest<f64>, usize) {
        let before = allocated_now_bytes();
        let Some(scan) = scan_non_empty_runs(digests) else {
            return (TDigest::<f64>::default(), 0);
        };
        let mut result = merge_result_shell(&scan);

        let (compressed, peak_snapshot): (Vec<Centroid<f64>>, usize) = match strategy {
            MergeStrategy::ConcatSort => {
                let merged = crate::tdigest::merges::merge_runs_concat_sort(&scan.runs);
                let after_stream = allocated_now_bytes();
                let compressed = compress_into(&mut result, scan.chosen_max_size, merged);
                let after_compress = allocated_now_bytes();
                (compressed, after_stream.max(after_compress))
            }
            MergeStrategy::HeapStream => {
                let merged = KWayCentroidMerge::from_runs(&scan.runs);
                let after_stream = allocated_now_bytes();
                let compressed = compress_into(&mut result, scan.chosen_max_size, merged);
                let after_compress = allocated_now_bytes();
                (compressed, after_stream.max(after_compress))
            }
        };

        result.centroids = compressed;
        let peak_extra_bytes = peak_snapshot.saturating_sub(before);
        (result, peak_extra_bytes)
    }

    fn median_full_merge_time_ns(
        digests: &[TDigest<f64>],
        strategy: MergeStrategy,
        reps: usize,
    ) -> u128 {
        let mut samples = Vec::with_capacity(reps);
        for _ in 0..reps {
            let t0 = Instant::now();
            let td = merge_with_strategy(digests, strategy);
            black_box(td.centroids.len());
            samples.push(t0.elapsed().as_nanos());
        }
        samples.sort_unstable();
        samples[samples.len() / 2]
    }

    fn expected_quantile(sorted: &[f64], q: f64) -> f64 {
        let n = sorted.len();
        if n == 0 {
            return f64::NAN;
        }
        if q <= 0.0 {
            return sorted[0];
        }
        if q >= 1.0 {
            return sorted[n - 1];
        }
        let t = q * (n as f64 - 1.0);
        let lo = t.floor() as usize;
        let hi = t.ceil() as usize;
        if lo == hi {
            sorted[lo]
        } else {
            let alpha = t - lo as f64;
            (1.0 - alpha) * sorted[lo] + alpha * sorted[hi]
        }
    }

    fn exact_cdf_leq(sorted: &[f64], x: f64) -> f64 {
        if sorted.is_empty() {
            return f64::NAN;
        }
        let idx = sorted.partition_point(|v| *v <= x);
        (idx as f64) / (sorted.len() as f64)
    }

    fn accuracy_metrics(td: &TDigest<f64>, sorted: &[f64]) -> AccuracyMetrics {
        if sorted.is_empty() {
            return AccuracyMetrics {
                q_mae: f64::NAN,
                q_max: f64::NAN,
                cdf_mae: f64::NAN,
                cdf_max: f64::NAN,
            };
        }

        let quant_steps = 199usize;
        let mut q_mae = 0.0;
        let mut q_max = 0.0;
        for i in 1..=quant_steps {
            let q = (i as f64) / ((quant_steps + 1) as f64);
            let got = td.quantile(q);
            let exp = expected_quantile(sorted, q);
            let e = (got - exp).abs();
            q_mae += e;
            if e > q_max {
                q_max = e;
            }
        }
        q_mae /= quant_steps as f64;

        let cdf_steps = 256usize;
        let lo = sorted[0];
        let hi = sorted[sorted.len() - 1];
        let probes: Vec<f64> = if lo == hi {
            vec![lo; cdf_steps + 1]
        } else {
            (0..=cdf_steps)
                .map(|i| lo + (i as f64) * (hi - lo) / (cdf_steps as f64))
                .collect()
        };
        let est = td.cdf(&probes);
        let mut cdf_mae = 0.0;
        let mut cdf_max = 0.0;
        for (x, got) in probes.iter().zip(est.iter()) {
            let exp = exact_cdf_leq(sorted, *x);
            let e = (*got - exp).abs();
            cdf_mae += e;
            if e > cdf_max {
                cdf_max = e;
            }
        }
        cdf_mae /= probes.len() as f64;

        AccuracyMetrics {
            q_mae,
            q_max,
            cdf_mae,
            cdf_max,
        }
    }

    fn build_shard_digests(
        data: &[f64],
        shards: usize,
        max_size: usize,
        scale: ScaleFamily,
        policy: SingletonPolicy,
    ) -> Vec<TDigest<f64>> {
        if data.is_empty() {
            return Vec::new();
        }
        let shard_count = shards.max(1);
        let chunk_size = data.len().div_ceil(shard_count);
        let mut out: Vec<TDigest<f64>> = Vec::with_capacity(shard_count);
        for chunk in data.chunks(chunk_size.max(1)) {
            if chunk.is_empty() {
                continue;
            }
            let td = TDigest::<f64>::builder()
                .max_size(max_size)
                .scale(scale)
                .singleton_policy(policy)
                .build()
                .merge_unsorted(chunk.to_vec())
                .expect("chunk digest");
            out.push(td);
        }
        out
    }

    #[test]
    fn weighted_add_matches_expanded_input_for_integer_weights() {
        let values = vec![3.0, -1.0, 5.0, 3.0];
        let weights = vec![4.0, 2.0, 3.0, 1.0];

        let mut weighted = TDigest::<f64>::builder().max_size(128).build();
        weighted
            .add_weighted_many(&values, &weights)
            .expect("weighted add");

        let mut expanded_values = Vec::new();
        for (&v, &w) in values.iter().zip(weights.iter()) {
            expanded_values.extend(std::iter::repeat_n(v, w as usize));
        }

        let mut expanded = TDigest::<f64>::builder().max_size(128).build();
        expanded.add_many(expanded_values).expect("expanded add");

        assert!(approx(weighted.count(), expanded.count(), 1e-12));
        assert!(approx(weighted.sum(), expanded.sum(), 1e-12));
        assert!(approx(weighted.min(), expanded.min(), 1e-12));
        assert!(approx(weighted.max(), expanded.max(), 1e-12));

        for q in [0.0, 0.1, 0.5, 0.9, 1.0] {
            assert!(
                approx(weighted.quantile(q), expanded.quantile(q), 1e-9),
                "quantile mismatch at q={q}: weighted={} expanded={}",
                weighted.quantile(q),
                expanded.quantile(q)
            );
        }
        for x in [-1.0, 0.0, 3.0, 4.0, 5.0] {
            let cw = weighted.cdf(&[x])[0];
            let ce = expanded.cdf(&[x])[0];
            assert!(
                approx(cw, ce, 1e-9),
                "cdf mismatch at x={x}: weighted={cw} expanded={ce}"
            );
        }
    }

    #[test]
    fn weighted_add_rejects_bad_inputs() {
        let mut td = TDigest::<f64>::builder().build();

        let err = td
            .add_weighted_many(&[1.0, 2.0], &[1.0])
            .expect_err("length mismatch should fail");
        assert!(matches!(err, TdError::MismatchedInputLength { .. }));

        let err = td
            .add_weighted_many(&[1.0], &[f64::NAN])
            .expect_err("non-finite weight should fail");
        assert!(matches!(err, TdError::NonFiniteInput { .. }));

        let err = td
            .add_weighted_many(&[1.0], &[0.0])
            .expect_err("zero weight should fail");
        assert!(matches!(err, TdError::InvalidWeight { .. }));

        let err = td
            .add_weighted_many(&[f64::INFINITY], &[1.0])
            .expect_err("non-finite value should fail");
        assert!(matches!(err, TdError::NonFiniteInput { .. }));
    }

    #[test]
    fn from_weighted_unsorted_builds_digest() {
        let td = TDigest::<f64>::from_weighted_unsorted(&[10.0, 0.0], &[2.0, 3.0], 64)
            .expect("weighted build");
        assert!(approx(td.count(), 5.0, 1e-12));
        assert!(approx(td.sum(), 20.0, 1e-12));
        assert!(approx(td.min(), 0.0, 1e-12));
        assert!(approx(td.max(), 10.0, 1e-12));
    }

    #[test]
    fn add_weighted_single_updates_digest_stats() {
        let mut td = TDigest::<f64>::builder().build();
        td.add_weighted(2.5, 4.0).expect("add weighted");
        assert!(approx(td.count(), 4.0, 1e-12));
        assert!(approx(td.sum(), 10.0, 1e-12));
        assert!(approx(td.min(), 2.5, 1e-12));
        assert!(approx(td.max(), 2.5, 1e-12));
    }

    #[test]
    fn cast_precision_roundtrip_preserves_shape() {
        let mut td = TDigest::<f64>::builder()
            .max_size(256)
            .scale(ScaleFamily::K3)
            .singleton_policy(SingletonPolicy::UseWithProtectedEdges(2))
            .build();
        td.add_many(vec![-3.0, -1.0, 0.0, 0.5, 10.0, 12.0])
            .expect("seed");
        td.add_weighted_many(&[5.0, 6.0], &[3.0, 2.0])
            .expect("weighted");

        let td32 = td.cast_precision::<f32>();
        let td64 = td32.cast_precision::<f64>();

        assert_eq!(td32.precision(), Precision::F32);
        assert_eq!(td64.precision(), Precision::F64);
        assert_eq!(td.scale(), td64.scale());
        assert_eq!(td.singleton_policy(), td64.singleton_policy());
        assert_eq!(td.max_size(), td64.max_size());
        assert!(approx(td.count(), td64.count(), 1e-6));
        assert!(approx(td.sum(), td64.sum(), 1e-4));

        for q in [0.0, 0.1, 0.5, 0.9, 1.0] {
            assert!(
                approx(td.quantile(q), td64.quantile(q), 1e-4),
                "quantile mismatch at q={q}: {} vs {}",
                td.quantile(q),
                td64.quantile(q)
            );
        }
    }

    #[test]
    #[ignore = "slow proof run; prints CPU/memory/precision summary for heap merge"]
    fn heap_kway_merge_proof_cpu_memory_precision() {
        let cases = [PerfCase {
            label: "mix-10m-k40-max1000",
            dist: DistKind::Mixture,
            n: 10_000_000,
            shards: 40,
            max_size: 1_000,
        }];

        let trials_per_case = 1usize;
        let reps = 3usize;
        let memory_supported = cfg!(all(target_os = "linux", target_env = "gnu"));

        let mut total_trials = 0usize;
        let mut cpu_wins = 0usize;
        let mut mem_wins = 0usize;
        let mut precision_non_worse = 0usize;
        let mut sort_cpu_total_ns = 0u128;
        let mut heap_cpu_total_ns = 0u128;

        println!();
        println!(
            "=== heap k-way merge proof: concat+sort baseline vs heap stream (trials={}, reps={}) ===",
            cases.len() * trials_per_case,
            reps
        );
        println!(
            "memory metric = jemalloc allocated-byte peak delta (linux-gnu only): {}",
            if memory_supported {
                "enabled"
            } else {
                "disabled"
            }
        );

        for (case_idx, case) in cases.iter().enumerate() {
            for trial in 0..trials_per_case {
                total_trials += 1;
                let seed = 42_000 + (case_idx as u64) * 100 + trial as u64;
                let data = gen_dataset(case.dist, case.n, seed);
                let mut sorted = data.clone();
                sorted.sort_by(|a, b| a.total_cmp(b));

                let digests = build_shard_digests(
                    &data,
                    case.shards,
                    case.max_size,
                    ScaleFamily::K2,
                    SingletonPolicy::Use,
                );

                let _ = merge_with_strategy(&digests, MergeStrategy::ConcatSort);
                let _ = merge_with_strategy(&digests, MergeStrategy::HeapStream);

                let sort_ns = median_full_merge_time_ns(&digests, MergeStrategy::ConcatSort, reps);
                let heap_ns = median_full_merge_time_ns(&digests, MergeStrategy::HeapStream, reps);
                let cpu_ratio = (heap_ns as f64) / (sort_ns.max(1) as f64);
                let cpu_win = heap_ns < sort_ns;
                sort_cpu_total_ns += sort_ns;
                heap_cpu_total_ns += heap_ns;
                if cpu_win {
                    cpu_wins += 1;
                }

                let (sort_td, sort_mem_bytes) =
                    merge_with_peak_alloc(&digests, MergeStrategy::ConcatSort);
                let (heap_td, heap_mem_bytes) =
                    merge_with_peak_alloc(&digests, MergeStrategy::HeapStream);
                let mem_ratio = (heap_mem_bytes as f64) / (sort_mem_bytes.max(1) as f64);
                let mem_win = heap_mem_bytes < sort_mem_bytes;
                if memory_supported && mem_win {
                    mem_wins += 1;
                }

                let sort_acc = accuracy_metrics(&sort_td, &sorted);
                let heap_acc = accuracy_metrics(&heap_td, &sorted);
                let precision_ok = heap_acc.q_mae <= sort_acc.q_mae + 1e-12
                    && heap_acc.q_max <= sort_acc.q_max + 1e-12
                    && heap_acc.cdf_mae <= sort_acc.cdf_mae + 1e-12
                    && heap_acc.cdf_max <= sort_acc.cdf_max + 1e-12;
                if precision_ok {
                    precision_non_worse += 1;
                }

                println!(
                    "[{} trial {}/{} seed={}] cpu(ns): sort={} heap={} ratio={:.3} | mem(bytes): sort={} heap={} ratio={:.3} | q_mae: sort={:.3e} heap={:.3e} | cdf_mae: sort={:.3e} heap={:.3e}",
                    case.label,
                    trial + 1,
                    trials_per_case,
                    seed,
                    sort_ns,
                    heap_ns,
                    cpu_ratio,
                    sort_mem_bytes,
                    heap_mem_bytes,
                    mem_ratio,
                    sort_acc.q_mae,
                    heap_acc.q_mae,
                    sort_acc.cdf_mae,
                    heap_acc.cdf_mae,
                );
            }
        }

        println!(
            "summary: cpu_wins={}/{} ({:.1}%), cpu_total_ratio={:.3}, mem_wins={}/{} ({:.1}%{}), precision_non_worse={}/{}",
            cpu_wins,
            total_trials,
            100.0 * (cpu_wins as f64) / (total_trials as f64),
            (heap_cpu_total_ns as f64) / (sort_cpu_total_ns.max(1) as f64),
            mem_wins,
            total_trials,
            100.0 * (mem_wins as f64) / (total_trials as f64),
            if memory_supported { "" } else { ", metric unavailable" },
            precision_non_worse,
            total_trials
        );

        assert!(
            heap_cpu_total_ns * 100 <= sort_cpu_total_ns * 130,
            "heap total CPU regressed by >30% on this profile (sort_total={} heap_total={})",
            sort_cpu_total_ns,
            heap_cpu_total_ns
        );

        if memory_supported {
            assert!(
                mem_wins * 100 >= total_trials * 90,
                "heap should use less extra allocated memory in at least 90% of trials (got {mem_wins}/{total_trials})"
            );
        }

        assert_eq!(
            precision_non_worse, total_trials,
            "heap merge changed precision metrics in at least one trial"
        );
    }
}
