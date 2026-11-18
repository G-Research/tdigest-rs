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
/// - Centroid means and `min/max` are stored in `F` to control **in-memory size**.
/// - Centroid weights stay in `f64` so f32 means do not discard small mass updates.
/// - `sum` and `count` remain `f64` for **stable accumulation** and to match the on-wire schema.
/// - The compressor/merges do their internal math in `f64` and convert at the boundaries.
#[derive(Debug, PartialEq, Clone)]
pub struct TDigest<F: FloatLike + FloatCore> {
    centroids: Vec<Centroid<F>>,
    max_size: usize,
    sum: f64,   // Sum of value times weight.
    count: f64, // ∑w over raw data (f64 for stability / wire)
    max: OrderedFloat<F>,
    min: OrderedFloat<F>,
    scale: ScaleFamily,
    policy: SingletonPolicy, // interpreted as atomic/edge policy
    delta: Option<f64>,
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
            delta: None,
        }
    }
}

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
        if weight == 1.0 {
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
    delta: Option<f64>,
    scale_explicit: bool,
    policy_explicit: bool,
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
            delta: None,
            scale_explicit: false,
            policy_explicit: false,
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

    /// Choose the scale used to limit centroid mass.
    #[inline]
    pub fn scale(mut self, s: ScaleFamily) -> Self {
        self.scale = s;
        self.scale_explicit = true;
        self
    }

    /// Set the policy for protecting centroids at the edges.
    #[inline]
    pub fn singleton_policy(mut self, p: SingletonPolicy) -> Self {
        self.policy = p;
        self.policy_explicit = true;
        self
    }

    /// Select K2Norm compression with singleton protection off.
    #[inline]
    pub fn delta(mut self, delta: f64) -> Self {
        self.delta = Some(delta);
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

    /// Build valid modern state or return an error without exposing a partial digest.
    /// Delta mode defaults to K2Norm/Off; explicitly incompatible settings error.
    pub fn build(mut self) -> TdResult<TDigest<F>> {
        if self.delta.is_some() {
            if !self.scale_explicit {
                self.scale = ScaleFamily::K2Norm;
            }
            if !self.policy_explicit {
                self.policy = SingletonPolicy::Off;
            }
        }
        if self
            .init_centroids
            .as_ref()
            .is_some_and(|centroids| !centroids.is_empty())
        {
            if let Some(stats) = self.init_stats {
                F::try_from_f64(stats.data_min)?;
                F::try_from_f64(stats.data_max)?;
            }
        }
        let digest = self.build_unchecked();
        digest.validate_state()?;
        Ok(digest)
    }

    // Only internal transient streams may bypass state validation. They are
    // normalized/coalesced before a checked public result is returned.
    fn build_unchecked(self) -> TDigest<F> {
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
                delta: self.delta,
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
                delta: self.delta,
            }
        }
    }
}

/// Observed extrema, total mass and weighted sum used to restore a digest.
#[derive(Debug, Clone, Copy)]
pub struct DigestStats {
    /// Sum of observation values multiplied by their weights.
    pub data_sum: f64,
    /// Total weight (∑w). For raw samples this equals the sample count.
    pub total_weight: f64,
    /// Minimum observed raw value (in f64; converted to F in builder).
    pub data_min: f64,
    /// Maximum observed raw value (in f64; converted to F in builder).
    pub data_max: f64,
}

impl<F: FloatLike + FloatCore> TDigest<F> {
    /// Temporary query view for positive subnormal mass. Two exact power-of-two
    /// scales make every weight normal and place the actual total in [1, 2).
    /// Keeping the factors separate avoids an overflowing combined reciprocal.
    /// The caller supplies the actual centroid total; stored state is untouched.
    /// This cannot restore information already lost during compression.
    pub(super) fn normalized_subnormal_mass_for_queries(&self, total_mass: f64) -> Self {
        debug_assert!(total_mass > 0.0 && total_mass < f64::MIN_POSITIVE);
        const FIRST_SCALE: f64 = 1.0 / f64::MIN_POSITIVE;
        let first_total = total_mass * FIRST_SCALE;
        // first_total is normal in [2^-52, 1), so the second exponent is 1..52.
        let exponent = ((first_total.to_bits() >> 52) & 0x7ff) as i32 - 1023;
        let second_scale = f64::from_bits(((1023 - exponent) as u64) << 52);
        let mut normalized = self.clone();
        for centroid in &mut normalized.centroids {
            let first = rescaled_centroid(centroid, 1.0, FIRST_SCALE);
            *centroid = rescaled_centroid(&first, 1.0, second_scale);
        }
        normalized.count = first_total * second_scale;
        normalized.sum = (normalized.sum * FIRST_SCALE) * second_scale;
        debug_assert!(normalized.count >= 1.0 && normalized.count < 2.0);
        normalized
    }

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
    #[inline]
    pub fn cdf_or_nan(&self, xs: &[f64]) -> Vec<f64> {
        if self.is_effectively_empty() {
            vec![f64::NAN; xs.len()]
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

    /// Cast this digest to a different mean-storage precision.
    ///
    /// This preserves configuration, f64 centroid weights and digest-level stats
    /// while converting centroid means and extrema to the target float type.
    fn cast_precision_unchecked<T>(&self) -> TDigest<T>
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
        if let Some(delta) = self.delta() {
            builder = builder.delta(delta);
        }
        builder
            .with_centroids_and_stats(cents, stats)
            .build_unchecked()
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
        let base = Self::builder().max_size(max_size).build()?;
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
        let base = Self::builder().max_size(max_size).build()?;
        base.merge_weighted_unsorted(values, weights)
    }

    /// The configured scale family used by the compressor's k-limit.
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
    pub fn delta(&self) -> Option<f64> {
        self.delta
    }

    pub(crate) fn set_delta(&mut self, delta: Option<f64>) {
        self.delta = delta;
    }

    /// Ingest **unsorted** values; behavior matches [`TDigest::merge_sorted`] after sorting.
    pub fn merge_unsorted(&self, mut unsorted_values: Vec<F>) -> TdResult<TDigest<F>> {
        ensure_no_non_finite_values(&unsorted_values)?;
        unsorted_values.sort_by(|a, b| a.total_cmp(*b));
        self.merge_sorted(unsorted_values)
    }

    /// Ingest **sorted** values by interleaving with existing centroids and running the pipeline.
    pub fn merge_sorted(&self, sorted_values: Vec<F>) -> TdResult<TDigest<F>> {
        self.validate_state()?;
        ensure_no_non_finite_values(&sorted_values)?;
        if sorted_values.windows(2).any(|pair| pair[0] > pair[1]) {
            return Err(TdError::Invariant {
                what: "merge_sorted requires nondecreasing values",
            });
        }
        if sorted_values.is_empty() {
            return Ok(self.clone());
        }
        let mut result = self.new_result_for_values(&sorted_values);

        let stream = MergeByMean::from_centroids_and_values(&self.centroids, &sorted_values);

        let data_sum = crate::tdigest::summation::sum(
            std::iter::once(self.sum)
                .chain(sorted_values.iter().map(|v| <F as FloatLike>::to_f64(*v))),
        );
        if !data_sum.is_finite() || !result.count.is_finite() {
            return Err(TdError::NonFiniteInput {
                context: "aggregate sum or mass overflow",
            });
        }
        let count = result.count;
        let compressed: Vec<Centroid<F>> = compress_into(&mut result, self.max_size, stream)?;
        result.centroids = compressed;
        result.sum = data_sum;
        result.count = count;
        result.validate_state()?;

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
        self.validate_state()?;
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

        // An exact point retains its full distribution even when its rounded
        // Float64 weighted moment is zero. Mixed observations cannot use that
        // recovery rule: their individual underflows remain explicit errors.
        let same_point = pairs
            .first()
            .map(|p| p.0)
            .filter(|first| pairs.iter().all(|p| p.0 == *first));
        let combined_point = same_point.is_some_and(|mean| {
            self.centroids.is_empty()
                || (self.is_exact_point()
                    && self.centroids[0].mean_f64() == <F as FloatLike>::to_f64(mean))
        });
        let mut point_product_underflow = false;
        let mut cents: Vec<Centroid<F>> = Vec::with_capacity(pairs.len());
        let mut moment = crate::tdigest::summation::Sum::default();
        let mut mass = crate::tdigest::summation::Sum::default();
        for (v, w) in &pairs {
            let mean = <F as FloatLike>::to_f64(*v);
            if *w == 1.0 {
                cents.push(Centroid::<F>::new_atomic_unit_f64(mean));
            } else {
                cents.push(Centroid::<F>::new_atomic_f64(mean, *w));
            }
            let product = mean * *w;
            point_product_underflow |= mean != 0.0 && product == 0.0;
            if !product.is_finite() || (point_product_underflow && !combined_point) {
                return Err(TdError::NonFiniteInput {
                    context: "weighted value product overflow or underflow",
                });
            }
            moment.add(product);
            mass.add(*w);
        }

        let total_weight = mass.total();
        let data_sum = if point_product_underflow {
            <F as FloatLike>::to_f64(same_point.unwrap()) * total_weight
        } else {
            moment.total()
        };
        if !data_sum.is_finite() || !total_weight.is_finite() {
            return Err(TdError::NonFiniteInput {
                context: "weighted aggregate overflow",
            });
        }

        if !(self.count + total_weight).is_finite() || !(self.sum + data_sum).is_finite() {
            return Err(TdError::NonFiniteInput {
                context: "merged weighted aggregate overflow",
            });
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
        if let Some(delta) = self.delta {
            weighted_builder = weighted_builder.delta(delta);
        }
        let weighted = weighted_builder
            .with_centroids_and_stats(cents, stats)
            .build_unchecked();

        // The private weighted stream deliberately contains equal-mean runs;
        // normalize it before requiring the external strict-state invariant.
        let recover_point_moment = point_product_underflow || self.has_subnormal_point_moment();
        let mut merged = TDigest::merge_digests_unchecked([self, &weighted].into_iter())?;
        if recover_point_moment && merged.is_exact_point() {
            merged.sum = merged.centroids[0].mean_f64() * merged.count;
        }
        merged.validate_state()?;
        Ok(merged)
    }

    /// Merge multiple digests by k-way merging their centroid runs and sending the result through
    /// the same pipeline used for raw values.
    fn merge_digests_unchecked<'a>(
        digests: impl ExactSizeIterator<Item = &'a Self>,
    ) -> TdResult<Self> {
        // Decide defaults by first non-empty digest to keep semantics stable.
        let mut chosen: Option<(usize, ScaleFamily, SingletonPolicy, Option<f64>)> = None;
        let mut runs: Vec<&[Centroid<F>]> = Vec::with_capacity(digests.len());
        let mut moment = crate::tdigest::summation::Sum::default();
        let mut mass = crate::tdigest::summation::Sum::default();
        let mut min = OrderedFloat::from(F::from_f64(f64::INFINITY));
        let mut max = OrderedFloat::from(F::from_f64(f64::NEG_INFINITY));

        for d in digests {
            let n = d.count();
            if n > 0.0 && !d.centroids.is_empty() {
                if chosen.is_none() {
                    chosen = Some((d.max_size, d.scale, d.policy, d.delta));
                }
                mass.add(n);
                moment.add(d.sum());
                min = std::cmp::min(min, d.min);
                max = std::cmp::max(max, d.max);
                runs.push(&d.centroids);
            }
        }
        let total_count = mass.total();
        if total_count == 0.0 {
            return Ok(TDigest::default());
        }
        let (chosen_max_size, chosen_scale, chosen_policy, chosen_delta) = chosen.unwrap();

        let mut result = TDigest::<F> {
            centroids: Vec::new(),
            max_size: chosen_max_size,
            sum: 0.0,
            count: total_count,
            max,
            min,
            scale: chosen_scale,
            policy: chosen_policy,
            delta: chosen_delta,
        };

        let merged_stream = KWayCentroidMerge::from_runs(&runs);

        let compressed: Vec<Centroid<F>> =
            compress_into(&mut result, chosen_max_size, merged_stream)?;
        result.centroids = compressed;
        // Full incoming aggregate statistics remain authoritative. Centroid
        // means have already been rounded/compressed and cannot restore them.
        result.sum = moment.total();
        result.count = total_count;

        debug_assert!(
            is_sorted_strict_by_mean(&result.centroids),
            "duplicate centroid means after merge_digests"
        );
        Ok(result)
    }

    /// Checked modern merge. All inputs, including empties, must have matching
    /// configuration. The raw recompression helper is private.
    pub fn try_merge_digests(digests: Vec<TDigest<F>>) -> TdResult<TDigest<F>> {
        for digest in &digests {
            digest.validate_state()?;
        }
        if let Some(first) = digests.first() {
            for digest in &digests[1..] {
                if first.max_size != digest.max_size
                    || first.scale != digest.scale
                    || first.policy != digest.policy
                    || first.delta != digest.delta
                {
                    return Err(TdError::Invariant {
                        what: "incompatible merge configuration",
                    });
                }
            }
        }
        let count = crate::tdigest::summation::sum(digests.iter().map(|d| d.count));
        let sum = crate::tdigest::summation::sum(digests.iter().map(|d| d.sum));
        if !count.is_finite() || !sum.is_finite() {
            return Err(TdError::NonFiniteInput {
                context: "merge aggregate overflow",
            });
        }
        if !digests.is_empty() && count == 0.0 {
            return Ok(digests[0].clone());
        }
        Self::merge_validated(digests.iter())
    }

    // The caller has checked every input, the common configuration and aggregate bounds.
    pub(super) fn merge_validated<'a>(
        digests: impl ExactSizeIterator<Item = &'a Self> + Clone,
    ) -> TdResult<Self> {
        let recover_point_moment = digests.clone().any(Self::has_subnormal_point_moment);
        let mut result = Self::merge_digests_unchecked(digests)?;
        if recover_point_moment && result.is_exact_point() {
            result.sum = result.centroids[0].mean_f64() * result.count;
        }
        result.validate_state()?;
        Ok(result)
    }

    /// Checked explicit precision conversion; never returns malformed state.
    pub fn try_cast_precision<T: FloatLike + FloatCore>(&self) -> TdResult<TDigest<T>> {
        self.validate_state()?;
        if !self.centroids.is_empty() {
            T::try_from_f64(self.min())?;
            T::try_from_f64(self.max())?;
        }
        let mut converted = self.cast_precision_unchecked::<T>();
        for (before, after) in self.centroids.iter().zip(converted.centroids.iter()) {
            if before.mean_f64() != 0.0 && after.mean_f64() == 0.0 {
                return Err(TdError::NonFiniteInput {
                    context: "precision conversion underflow",
                });
            }
        }
        let mut coalesced: Vec<Centroid<T>> = Vec::with_capacity(converted.centroids.len());
        for centroid in converted.centroids {
            if let Some(previous) = coalesced.last_mut() {
                if previous.mean_f64() == centroid.mean_f64() {
                    *previous = Centroid::<T>::new_mixed_f64(
                        previous.mean_f64(),
                        previous.weight_f64() + centroid.weight_f64(),
                    );
                    continue;
                }
            }
            coalesced.push(centroid);
        }
        converted.centroids = coalesced;
        converted.validate_state()?;
        Ok(converted)
    }

    /// Validate and merge independent modern state, rejecting incompatible configuration.
    pub fn merge_digests(digests: Vec<TDigest<F>>) -> TdResult<TDigest<F>> {
        Self::try_merge_digests(digests)
    }

    /// Explicit checked precision conversion, preserving Float64 mass.
    pub fn cast_precision<T: FloatLike + FloatCore>(&self) -> TdResult<TDigest<T>> {
        self.try_cast_precision::<T>()
    }

    /// In-place merge with another digest of the same precision.
    ///
    /// This is a convenience wrapper over [`TDigest::merge_digests`].
    pub fn merge(&mut self, other: &TDigest<F>) -> TdResult<&mut Self> {
        *self = TDigest::try_merge_digests(vec![self.clone(), other.clone()])?;
        Ok(self)
    }

    /// In-place merge with multiple digests of the same precision.
    ///
    /// This is a convenience wrapper over [`TDigest::merge_digests`].
    pub fn merge_many(&mut self, others: &[TDigest<F>]) -> TdResult<&mut Self> {
        if others.is_empty() {
            self.validate_state()?;
            return Ok(self);
        }
        let mut all = Vec::with_capacity(others.len() + 1);
        all.push(self.clone());
        all.extend_from_slice(others);
        *self = TDigest::try_merge_digests(all)?;
        Ok(self)
    }

    /// Add a single value in-place.
    pub fn add(&mut self, value: F) -> TdResult<&mut Self> {
        self.add_many([value].as_slice())
    }

    /// Add one or more values in-place.
    pub fn add_many<A: IntoVecF<F>>(&mut self, values: A) -> TdResult<&mut Self> {
        let vals = values.into_vec_f();
        if vals.is_empty() {
            self.validate_state()?;
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
    /// This preserves quantile/cdf/median shape while re-weighting the digest,
    /// subject to floating-point rounding. If a scaled centroid weight becomes
    /// zero/nonfinite or either mass total becomes nonfinite, return an error without
    /// changing the digest. Weighted-sum overflow/underflow is checked too.
    pub fn scale_weights(&mut self, factor: f64) -> TdResult<&mut Self> {
        ensure_positive_finite_scale_factor(factor, "weight scale factor must be finite and > 0")?;
        if self.centroids.is_empty() {
            return Ok(self);
        }

        let mut scaled_centroids = Vec::with_capacity(self.centroids.len());
        let mut scaled_mass = 0.0;
        for centroid in &self.centroids {
            let weight = centroid.weight_f64() * factor;
            if !weight.is_finite() || weight <= 0.0 {
                return Err(TdError::InvalidWeight {
                    context: "scaled centroid weight must be finite and > 0",
                });
            }
            scaled_mass += weight;
            if !scaled_mass.is_finite() {
                return Err(TdError::InvalidWeight {
                    context: "scaled total weight must be finite and > 0",
                });
            }
            // The product is validated before a checked centroid constructor.
            scaled_centroids.push(rescaled_centroid(centroid, 1.0, factor));
        }
        let scaled_count = self.count * factor;
        if !scaled_count.is_finite() || scaled_count <= 0.0 {
            return Err(TdError::InvalidWeight {
                context: "scaled total weight must be finite and > 0",
            });
        }

        let scaled_sum = if self.has_subnormal_point_moment() {
            self.centroids[0].mean_f64() * scaled_count
        } else {
            self.sum * factor
        };
        if !scaled_sum.is_finite()
            || (self.sum != 0.0 && scaled_sum == 0.0 && !self.is_exact_point())
        {
            return Err(TdError::NonFiniteInput {
                context: "scaled sum overflow or underflow",
            });
        }
        // Commit only after all products and both total representations pass.
        let mut candidate = self.clone();
        candidate.centroids = scaled_centroids;
        candidate.count = scaled_count;
        candidate.sum = scaled_sum;
        candidate.validate_state()?;
        *self = candidate;
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

        let scaled = |value: f64| -> TdResult<f64> {
            let result = <F as FloatLike>::to_f64(F::from_f64(value * factor));
            if !result.is_finite() || (value != 0.0 && result == 0.0) {
                return Err(TdError::NonFiniteInput {
                    context: "value scale overflows or underflows selected precision",
                });
            }
            Ok(result)
        };
        for centroid in &self.centroids {
            scaled(centroid.mean_f64())?;
        }
        let minimum = scaled(self.min())?;
        let maximum = scaled(self.max())?;
        let sum = if self.has_subnormal_point_moment() {
            minimum * self.count
        } else {
            self.sum * factor
        };
        if !sum.is_finite() || (self.sum != 0.0 && sum == 0.0 && !self.is_exact_point()) {
            return Err(TdError::NonFiniteInput {
                context: "scaled sum overflow or underflow",
            });
        }
        let mut candidate = self.clone();
        candidate.centroids = self
            .centroids
            .iter()
            .map(|c| rescaled_centroid(c, factor, 1.0))
            .collect();

        candidate.set_min(minimum);
        candidate.set_max(maximum);
        candidate.sum = sum;
        candidate.validate_state()?;
        *self = candidate;
        Ok(self)
    }

    /// Construct a digest from raw centroids + stats.
    ///
    /// Recompress an oversized seed at the requested capacity.
    pub fn new(
        centroids: Vec<Centroid<F>>,
        sum: f64,
        count: f64,
        max: F,
        min: F,
        max_size: usize,
    ) -> TdResult<Self> {
        let seeded = Self::builder()
            .max_size(max_size)
            .with_centroids(centroids, sum, count, max, min, max_size)
            .build()?;
        if seeded.centroids.len() <= max_size {
            return Ok(seeded);
        }
        let result = Self::merge_digests_unchecked(std::iter::once(&seeded))?;
        result.validate_state()?;
        Ok(result)
    }

    fn is_exact_point(&self) -> bool {
        self.centroids.len() == 1
            && self.centroids[0].is_atomic()
            && self.min() == self.max()
            && self.min() == self.centroids[0].mean_f64()
    }

    fn has_subnormal_point_moment(&self) -> bool {
        self.is_exact_point()
            && self.sum.abs() < f64::MIN_POSITIVE
            && self.centroids[0].mean_f64() != 0.0
            && self.centroids[0].mean_f64() * self.count == self.sum
    }

    /// Validate an externally reconstructed modern state without recompressing.
    /// Aggregate sum is data-level information, not a sum recomputed from lossy
    /// centroid means. Count must agree with represented mass within summation
    /// roundoff. Empty states have zero aggregates and undefined NaN extrema.
    pub fn validate_state(&self) -> TdResult<()> {
        let invalid = |what| TdError::Invariant { what };
        if self.max_size == 0 {
            return Err(invalid("capacity must be positive"));
        }
        if self.policy == SingletonPolicy::UseWithProtectedEdges(0) {
            return Err(invalid("edges policy requires pin_per_side >= 1"));
        }
        if let Some(delta) = self.delta {
            if !delta.is_finite() || delta <= 0.0 {
                return Err(invalid("delta must be finite and positive"));
            }
            if self.scale != ScaleFamily::K2Norm || self.policy != SingletonPolicy::Off {
                return Err(invalid(
                    "delta mode requires K2Norm scale and singleton policy Off",
                ));
            }
        }
        if self.centroids.is_empty() {
            return if self.count == 0.0
                && self.sum == 0.0
                && self.min().is_nan()
                && self.max().is_nan()
            {
                Ok(())
            } else {
                Err(invalid(
                    "empty state must have zero mass/sum and NaN extrema",
                ))
            };
        }
        if !self.count.is_finite()
            || self.count <= 0.0
            || !self.sum.is_finite()
            || !self.min().is_finite()
            || !self.max().is_finite()
            || self.min() > self.max()
        {
            return Err(invalid(
                "nonempty state requires finite aggregates and ordered extrema",
            ));
        }
        let aggregate_mean = self.sum / self.count;
        // A digest may have been upcast from f32 while retaining its original
        // f64 sum. Current storage width does not record that rounding history.
        let mean_epsilon = f32::EPSILON as f64;
        let magnitude = self.min().abs().max(self.max().abs());
        let mean_quantum = if magnitude >= f32::from_bits(1) as f64 {
            f32::from_bits(1) as f64
        } else {
            f64::from_bits(1)
        };
        let tolerance = 4.0 * mean_epsilon * magnitude + mean_quantum;
        if (!aggregate_mean.is_finite()
            || aggregate_mean < self.min() - tolerance
            || aggregate_mean > self.max() + tolerance)
            && !self.has_subnormal_point_moment()
        {
            return Err(invalid(
                "weighted sum is inconsistent with observed extrema",
            ));
        }
        let mut mass = 0.0;
        let mut previous = f64::NEG_INFINITY;
        if magnitude == 0.0 && self.sum != 0.0 {
            return Err(invalid("zero-valued state must have zero sum"));
        }
        let mut normalized_moment = crate::tdigest::summation::Sum::default();
        let mut all_atomic = true;
        for centroid in &self.centroids {
            let mean = centroid.mean_f64();
            let weight = centroid.weight_f64();
            if !mean.is_finite()
                || mean <= previous
                || mean < self.min()
                || mean > self.max()
                || !weight.is_finite()
                || weight <= 0.0
            {
                return Err(invalid("invalid or unordered centroid"));
            }
            previous = mean;
            mass += weight;
            all_atomic &= centroid.is_atomic();
            if magnitude > 0.0 {
                // Divide before multiplying: mean * weight can overflow even
                // when the aggregate is representable after cancellation.
                normalized_moment.add((mean / magnitude) * (weight / self.count));
            }
        }
        let roundoff = f64::EPSILON * (self.centroids.len() as f64 + 2.0) * mass.max(self.count);
        if !mass.is_finite() || (mass - self.count).abs() > roundoff {
            return Err(invalid("count differs from represented centroid mass"));
        }
        if all_atomic && (self.min() != self.centroids[0].mean_f64() || self.max() != previous) {
            return Err(invalid("atomic state extrema differ from observed points"));
        }
        if magnitude > 0.0 && !self.has_subnormal_point_moment() {
            // Raw sums remain authoritative: rounded/compressed centroid means
            // cannot reproduce them exactly. Check agreement within a numerical
            // envelope, including absolute quantization at subnormal scales.
            let terms = self.centroids.len() as f64 + 2.0;
            let tolerance = 32.0 * mean_epsilon
                + 8.0 * f64::EPSILON * terms
                + mean_quantum / magnitude
                + (f64::from_bits(1) / self.count) / magnitude * terms;
            if (normalized_moment.total() - aggregate_mean / magnitude).abs() > tolerance {
                return Err(invalid(
                    "weighted sum differs from represented centroid moment",
                ));
            }
        }
        Ok(())
    }

    /// Mean of the represented distribution (`sum / count`) in **f64** for stability.
    #[inline]
    pub fn mean(&self) -> f64 {
        if self.has_subnormal_point_moment() {
            return self.centroids[0].mean_f64();
        }
        let n = self.count;
        if n > 0.0 {
            self.sum / n
        } else {
            0.0
        }
    }

    /// Sum of observation values multiplied by their weights.
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

    /// Carry the observed extrema forward when combining a sorted batch with this digest.
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
            delta: self.delta,
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

    /// Convenience: build from an arbitrary numeric container already typed as `F`, using options.
    pub fn from_array_with<A: IntoVecF<F>>(arr: A, opts: DigestOptions<F>) -> TdResult<TDigest<F>> {
        let vals = arr.into_vec_f();
        ensure_no_non_finite_values(&vals)?;
        let base = TDigest::<F>::builder()
            .max_size(opts.max_size)
            .scale(opts.scale)
            .singleton_policy(opts.singleton_policy)
            .build()?;
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

            // Upcast f32-backed digest into f64-backed digest.
            WireDecodedDigest::F32(td32) => td32
                .cast_precision::<f64>()
                .map_err(|_| WireError::InvalidPayload("invalid precision conversion")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quality::quality_base::{empirical_cdf, expected_quantile};
    use std::hint::black_box;
    use std::time::Instant;
    use testdata::{gen_dataset, DistKind};

    #[test]
    fn damaged_state_cannot_bypass_validation_through_noops_or_export() {
        use crate::tdigest::frontends::FrontendDigest;
        let mut damaged = TDigest::<f64>::from_unsorted(&[0.0, 10.0], 100).unwrap();
        damaged.set_sum(2.0);
        let before = damaged.clone();
        assert!(damaged.add_many(Vec::<f64>::new()).is_err());
        assert!(damaged.add_weighted_many(&[], &[]).is_err());
        assert!(damaged.merge_many(&[]).is_err());
        assert!(TDigest::merge_digests(vec![damaged.clone()]).is_err());
        assert!(FrontendDigest::merge_all(vec![FrontendDigest::F64(damaged.clone())]).is_err());
        assert!(WireVersion::V3.validate_digest(&damaged).is_err());
        assert!(damaged.to_series("td").is_err());
        assert_eq!(damaged, before);
    }

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
        chosen_delta: Option<f64>,
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
                    chosen = Some((d.max_size, d.scale, d.policy, d.delta));
                }
                total_count += n;
                min = std::cmp::min(min, d.min);
                max = std::cmp::max(max, d.max);
                runs.push(&d.centroids);
            }
        }

        let (chosen_max_size, chosen_scale, chosen_policy, chosen_delta) = chosen?;

        Some(MergeScan {
            chosen_max_size,
            chosen_scale,
            chosen_policy,
            chosen_delta,
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
            delta: scan.chosen_delta,
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
                compress_into(&mut result, scan.chosen_max_size, merged).unwrap()
            }
            MergeStrategy::HeapStream => {
                let merged = KWayCentroidMerge::from_runs(&scan.runs);
                compress_into(&mut result, scan.chosen_max_size, merged).unwrap()
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
                let compressed = compress_into(&mut result, scan.chosen_max_size, merged).unwrap();
                let after_compress = allocated_now_bytes();
                (compressed, after_stream.max(after_compress))
            }
            MergeStrategy::HeapStream => {
                let merged = KWayCentroidMerge::from_runs(&scan.runs);
                let after_stream = allocated_now_bytes();
                let compressed = compress_into(&mut result, scan.chosen_max_size, merged).unwrap();
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
            let exp = empirical_cdf(sorted, *x);
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
                .unwrap()
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

        let mut weighted = TDigest::<f64>::builder().max_size(128).build().unwrap();
        weighted
            .add_weighted_many(&values, &weights)
            .expect("weighted add");

        let mut expanded_values = Vec::new();
        for (&v, &w) in values.iter().zip(weights.iter()) {
            expanded_values.extend(std::iter::repeat_n(v, w as usize));
        }

        let mut expanded = TDigest::<f64>::builder().max_size(128).build().unwrap();
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
        let mut td = TDigest::<f64>::builder().build().unwrap();

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
        let mut td = TDigest::<f64>::builder().build().unwrap();
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
            .build()
            .unwrap();
        td.add_many(vec![-3.0, -1.0, 0.0, 0.5, 10.0, 12.0])
            .expect("seed");
        td.add_weighted_many(&[5.0, 6.0], &[3.0, 2.0])
            .expect("weighted");

        let td32 = td.cast_precision::<f32>().unwrap();
        let td64 = td32.cast_precision::<f64>().unwrap();

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
    #[ignore = "measures two merge strategies on ten million observations"]
    fn heap_merge_time_allocation_and_accuracy() {
        let cases = [PerfCase {
            label: "mix-10m-k40-max1000",
            dist: DistKind::Mixture,
            n: 10_000_000,
            shards: 40,
            max_size: 1_000,
        }];

        // Keep these conservative because n is intentionally very large.
        let trials_per_case = 1usize;
        let reps = 3usize;
        let memory_supported = cfg!(all(
            target_os = "linux",
            target_env = "gnu",
            not(feature = "java")
        ));

        let mut total_trials = 0usize;
        let mut cpu_wins = 0usize;
        let mut mem_wins = 0usize;
        let mut precision_non_worse = 0usize;
        let mut sort_cpu_total_ns = 0u128;
        let mut heap_cpu_total_ns = 0u128;

        println!();
        println!(
            "=== Sorted concatenation and heap merge (trials={}, reps={}) ===",
            cases.len() * trials_per_case,
            reps
        );
        println!(
            "memory metric = largest sampled change in jemalloc allocated bytes (GNU Linux): {}",
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

                // Warm-up both strategies before timed reps.
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
                    "[{} trial {}/{} seed={}] elapsed(ns): sort={} heap={} ratio={:.3} | mem(bytes): sort={} heap={} ratio={:.3} | q_mae: sort={:.3e} heap={:.3e} | cdf_mae: sort={:.3e} heap={:.3e}",
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

        // With max_size=1000 and moderate shard counts, CPU can be mixed:
        // heap usually wins memory strongly and preserves precision, but may
        // not always beat concat+sort on wall time. Keep a guard against
        // pathological regressions while letting this profile stay realistic.
        assert!(
            heap_cpu_total_ns * 100 <= sort_cpu_total_ns * 130,
            "heap elapsed time increased by >30% on this profile (sort_total={} heap_total={})",
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
