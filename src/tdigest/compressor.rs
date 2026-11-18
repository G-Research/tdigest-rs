//! Compress ordered centroids while preserving their order and total mass.
//! Equal means coalesce before the singleton policy reserves the edges.
//! The interior is merged under the selected scale and then reduced to its capacity.

use ordered_float::FloatCore;

use crate::tdigest::centroids::Centroid;
use crate::tdigest::merges::normalize_stream;
use crate::tdigest::precision::FloatLike;
use crate::tdigest::scale::{k2_log_odds, q_to_k, ScaleFamily, K2_DENOMINATOR};
use crate::tdigest::singleton_policy::SingletonPolicy;
use crate::tdigest::TDigest;

const KLIMIT_TOL: f64 = 1e-12;
const CAP_SEARCH_ITERS: usize = 28;
const CAP_FALLBACK_ITERS: usize = 12;
const CAP_SEARCH_LO: f64 = 1e-6;

#[derive(Clone, Copy, Debug, Default)]
struct WeightedStats {
    w_sum: f64,
    mw_sum: f64,
    moment: crate::tdigest::summation::Sum,
    first_mean: f64,
    last_mean: f64,
    first_atomic: bool,
    items: usize,
}

impl WeightedStats {
    #[inline]
    fn add<F: FloatLike + FloatCore>(&mut self, centroid: &Centroid<F>) {
        if self.items == 0 {
            self.first_mean = centroid.mean_f64();
            self.first_atomic = centroid.is_atomic();
        }
        self.last_mean = centroid.mean_f64();
        self.items += 1;
        self.w_sum += centroid.weight_f64();
        self.mw_sum += centroid.weight_f64() * centroid.mean_f64();
        self.moment.add(centroid.weight_f64() * centroid.mean_f64());
    }
    #[inline]
    fn mean(&self) -> f64 {
        if self.items == 1 {
            // A pass-through centroid has not been averaged. Recomputing m*w/w
            // can move its mean by an ulp, changing point-mass CDFs and making
            // later merges produce duplicate or out-of-order means.
            self.first_mean
        } else if self.w_sum > 0.0 {
            // Preserve the established same-sign finite averaging order. A
            // different last bit can change later cluster membership. Use
            // compensation when cancellation or intermediate overflow needs it.
            let moment =
                if (self.first_mean < 0.0 && self.last_mean > 0.0) || !self.mw_sum.is_finite() {
                    self.moment.total()
                } else {
                    self.mw_sum
                };
            let mean = moment / self.w_sum;
            // Weighted averaging can round beyond both input means by an ulp.
            // Disjoint sorted clusters must retain disjoint mean intervals.
            // Do not hide aggregate/weighted-sum overflow behind this bound.
            if self.w_sum.is_finite() && self.mw_sum.is_finite() && mean.is_finite() {
                mean.clamp(self.first_mean, self.last_mean)
            } else {
                mean
            }
        } else {
            0.0
        }
    }
    #[inline]
    fn centroid<F: FloatLike + FloatCore>(&self) -> Centroid<F> {
        if self.items == 1 && self.first_atomic {
            Centroid::<F>::new_atomic_f64(self.mean(), self.w_sum)
        } else {
            Centroid::<F>::new_mixed_f64(self.mean(), self.w_sum)
        }
    }
    #[inline]
    fn is_empty(&self) -> bool {
        self.w_sum <= 0.0
    }
    #[inline]
    fn clear(&mut self) {
        self.w_sum = 0.0;
        self.mw_sum = 0.0;
        self.moment = crate::tdigest::summation::Sum::default();
        self.items = 0;
    }
}

/// Compress a sorted stream and update the digest statistics.
pub(crate) fn compress_into<F, I>(
    result: &mut TDigest<F>,
    max_size: usize,
    items: I,
) -> crate::TdResult<Vec<Centroid<F>>>
where
    F: FloatLike + FloatCore,
    I: IntoIterator<Item = Centroid<F>>,
{
    let norm = normalize_stream::<F, _>(items)?;
    if norm.out.is_empty() {
        return Ok(norm.out);
    }

    validate_compression_stream(&norm.out)?;

    // Centroid mass and weighted sums can be recomputed, but centroid means
    // do not retain the true extrema of mixed clusters. Keep the data bounds
    // carried by the add/merge result shell; use the stream for an unset bound.
    result.set_count(norm.total_w);
    result.set_sum(norm.total_mw);
    result.set_min(result.min().min(norm.min));
    result.set_max(result.max().max(norm.max));

    let policy = policy_from(result.singleton_policy(), max_size);
    let delta = result.delta();
    if delta.is_none() {
        if let Some(v) = policy.fast_path(&norm.out) {
            validate_compression_stream(&v)?;
            debug_assert!(crate::tdigest::centroids::is_sorted_strict_by_mean(&v));
            return Ok(v);
        }
    }
    let slices = policy.slice(&norm.out);

    let family: ScaleFamily = result.scale();
    let d: f64 = result.max_size() as f64;
    let core = if let Some(delta) = delta {
        // delta-mode is contract-validated upstream (scale = K2Norm, policy = Off);
        // Use the strict delta-mode k-limit; this is not a byte-identity guarantee.
        debug_assert_eq!(family, ScaleFamily::K2Norm);
        klimit_merge::<F>(slices.interior, delta, family, 0.0)
    } else {
        klimit_merge::<F>(slices.interior, d, family, KLIMIT_TOL)
    };

    validate_compression_stream(&core)?;

    // Second k-limit cap pass if the interior exceeds its budget.
    // Fallback to equal-weight bucketization remains as a safety net in cap_core.
    let core_capped = if delta.is_some() {
        core
    } else {
        cap_core::<F>(core, slices.core_cap, d, family)
    };

    let assembled = assemble_with_edges::<F>(slices.left, core_capped, slices.right);

    validate_compression_stream(&assembled)?;

    let compressed = if delta.is_some() {
        assembled
    } else {
        policy.post(assembled)
    };

    validate_compression_stream(&compressed)?;
    debug_assert!(crate::tdigest::centroids::is_sorted_strict_by_mean(
        &compressed
    ));
    Ok(compressed)
}

// Check between stages before rounded mass can reach a checked constructor or
// an overflowed mean can reach an ordering assertion. A finite positive naive
// total also bounds every contiguous accumulator in the following stage.
fn validate_compression_stream<F: FloatLike + FloatCore>(
    cs: &[Centroid<F>],
) -> crate::TdResult<()> {
    let mut mass = 0.0;
    for c in cs {
        mass += c.weight_f64();
        if !mass.is_finite() || !c.mean_f64().is_finite() {
            return Err(crate::TdError::NonFiniteInput {
                context: "compression mass or mean overflow",
            });
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
enum Policy {
    Off { max: usize },
    Use { max: usize },
    UseProt { max: usize, k: usize },
}

fn policy_from(sp: SingletonPolicy, max: usize) -> Policy {
    match sp {
        SingletonPolicy::Off => Policy::Off { max },
        SingletonPolicy::Use => Policy::Use { max },
        SingletonPolicy::UseWithProtectedEdges(k) => Policy::UseProt { max, k },
    }
}

impl Policy {
    fn fast_path<F: FloatLike + FloatCore>(&self, out: &[Centroid<F>]) -> Option<Vec<Centroid<F>>> {
        match *self {
            Policy::Off { max } => {
                if out.len() <= max {
                    return Some(out.to_vec());
                }
                if max == 1 {
                    return Some(vec![weighted_collapse(out)]);
                }
                if max == 2 {
                    return Some(bucketize_equal_weight(out, 2));
                }
                None
            }
            Policy::Use { max } if max <= 2 && max > 0 => Some(bucketize_equal_weight(out, max)),
            Policy::Use { .. } | Policy::UseProt { .. } => None,
        }
    }

    fn slice<'a, F: FloatLike + FloatCore>(&self, out: &'a [Centroid<F>]) -> EdgeSlices<'a, F> {
        match *self {
            Policy::Off { max } => EdgeSlices {
                left: &[],
                interior: out,
                right: &[],
                core_cap: max,
            },
            Policy::Use { max } => {
                let n = out.len();
                let l = usize::from(n >= 1);
                let r = usize::from(n >= 2);

                let left = &out[..l];
                let right = if n > l { &out[n - r..] } else { &[] };
                let interior = if n > (l + r) { &out[l..n - r] } else { &[] };
                let core_cap = max.saturating_sub(l + r);

                EdgeSlices {
                    left,
                    interior,
                    right,
                    core_cap,
                }
            }
            Policy::UseProt { max, k } => {
                let n = out.len();
                let l_prot = edge_run_len(out, k, true);
                let r_prot = edge_run_len(out, k, false);

                let left_end = l_prot.min(n);
                // Protected ranges may touch or overlap on small digests. Keep
                // their union exactly once instead of discarding the right tail.
                let right_start = n.saturating_sub(r_prot).max(left_end);

                let left = &out[..left_end];
                let right = &out[right_start..];
                let interior = &out[left_end..right_start];

                EdgeSlices {
                    left,
                    interior,
                    right,
                    core_cap: max,
                }
            }
        }
    }

    fn post<F: FloatLike + FloatCore>(&self, assembled: Vec<Centroid<F>>) -> Vec<Centroid<F>> {
        match *self {
            Policy::Use { max } => {
                if assembled.len() <= max {
                    assembled
                } else if max == 0 {
                    Vec::new()
                } else {
                    bucketize_equal_weight(&assembled, max)
                }
            }
            Policy::Off { .. } | Policy::UseProt { .. } => assembled,
        }
    }
}

/// Merge adjacent centroids while their combined rank span satisfies the scale limit.
/// Delta compression uses zero tolerance. A cluster with multiple inputs is mixed.
fn klimit_merge<F: FloatLike + FloatCore>(
    items: &[Centroid<F>],
    d: f64,
    family: ScaleFamily,
    tol: f64,
) -> Vec<Centroid<F>> {
    if items.is_empty() {
        return Vec::new();
    }
    let total_w: f64 = items.iter().map(|c| c.weight_f64()).sum();
    // This implementation uses the current merge pass's centroid count as n
    // in the Dunning K2 normalizer; non-K2Norm families ignore this argument.
    let n = items.len() as f64;

    // Z = 4*ln(n/d) + 24 is Dunning's published K2 normalizer, not a
    // threshold fitted for this guard. The factor 4 comes from the centroid
    // count bound; 24 is the chosen offset for that bound over practical n/d.
    // See "The Size of a t-Digest", section 5:
    // https://arxiv.org/abs/1903.09921
    // K2Norm increases with q only when its normalizer is positive. At a
    // sufficiently large delta relative to the normalized input count, a
    // nonpositive normalizer would reverse the scale and accept every merge.
    // There is ample capacity in this regime: retain the existing means,
    // weights and Atomic/Mixed kinds rather than discard information.
    // Positive infinity is deliberately NOT caught: a tiny positive delta
    // can overflow n/d and keeps its existing aggressive-compression path.
    if family == ScaleFamily::K2Norm && (n / d).ln().mul_add(4.0, 24.0) <= 0.0 {
        return items.to_vec();
    }

    // C = mass of completed clusters already emitted.
    let mut c_acc = 0.0_f64; // running capacity accumulator

    let mut clusters: Vec<Centroid<F>> = Vec::with_capacity(items.len());
    let mut acc = WeightedStats::default();

    let mut q_l = c_acc / total_w;
    let mut k_left = q_to_k(q_l, d, family, n);

    for c in items {
        let w_next = c.weight_f64();
        let q_r = (c_acc + acc.w_sum + w_next) / total_w;
        let k_right = q_to_k(q_r, d, family, n);

        if (k_right - k_left) <= 1.0 + tol {
            acc.add(c);
        } else {
            flush_cluster::<F>(&mut clusters, &mut acc);
            if let Some(last) = clusters.last() {
                c_acc += last.weight_f64();
            }
            q_l = c_acc / total_w;
            k_left = q_to_k(q_l, d, family, n);

            acc.add(c);
        }
    }
    flush_cluster::<F>(&mut clusters, &mut acc);

    clusters
}

/// Count a trial k-limit pass without constructing discarded centroid means.
/// Decisions depend only on weights; preserve the full pass's addition order.
fn klimit_count<F: FloatLike + FloatCore>(
    items: &[Centroid<F>],
    d: f64,
    family: ScaleFamily,
    tol: f64,
) -> usize {
    if items.is_empty() {
        return 0;
    }
    let total_w: f64 = items.iter().map(|c| c.weight_f64()).sum();
    let n = items.len() as f64;
    if family == ScaleFamily::K2Norm && (n / d).ln().mul_add(4.0, 24.0) <= 0.0 {
        return items.len();
    }
    let mut c_acc = 0.0_f64;
    let mut weight = 0.0_f64;
    let mut clusters = 0;
    let mut k_left = q_to_k(c_acc / total_w, d, family, n);
    for centroid in items {
        let next = centroid.weight_f64();
        let q_r = (c_acc + weight + next) / total_w;
        if (q_to_k(q_r, d, family, n) - k_left) <= 1.0 + tol {
            weight += next;
        } else {
            if weight > 0.0 {
                clusters += 1;
                c_acc += weight;
                weight = 0.0;
            }
            k_left = q_to_k(c_acc / total_w, d, family, n);
            weight += next;
        }
    }
    clusters + usize::from(weight > 0.0)
}

// With integral weights totaling less than 2^53, every regrouping has the
// same exact prefix masses. Cache the K2 log-odds at those boundaries once.
// Fractional and large masses must keep each trial's original addition order.
fn integer_k2_boundaries<F: FloatLike + FloatCore>(
    core: &[Centroid<F>],
    family: ScaleFamily,
) -> Option<Vec<f64>> {
    if family != ScaleFamily::K2 {
        return None;
    }
    let mut total = 0.0;
    for c in core {
        let w = c.weight_f64();
        if w.fract() != 0.0 {
            return None;
        }
        total += w;
    }
    if total >= (1u64 << 53) as f64 {
        return None;
    }

    let mut boundaries = Vec::with_capacity(core.len() + 1);
    boundaries.push(k2_log_odds(0.0));
    let mut prefix = 0.0;
    for c in core {
        prefix += c.weight_f64();
        boundaries.push(k2_log_odds(prefix / total));
    }
    Some(boundaries)
}

fn count_integer_k2_clusters(boundaries: &[f64], d: f64) -> usize {
    let factor = d / K2_DENOMINATOR;
    let mut left = factor * boundaries[0];
    let mut previous = factor * boundaries[1];
    let mut clusters = 1;
    for &boundary in &boundaries[2..] {
        let right = factor * boundary;
        // Keep the two multiplications before subtraction, as in klimit_merge.
        if right - left > 1.0 + KLIMIT_TOL {
            clusters += 1;
            left = previous;
        }
        previous = right;
    }
    clusters
}

/// Search for the largest scale parameter whose clusters fit the interior capacity.
/// When the count stays above the limit, group the centroids by weight.
fn cap_core<F: FloatLike + FloatCore>(
    core: Vec<Centroid<F>>,
    core_cap: usize,
    d: f64,
    family: ScaleFamily,
) -> Vec<Centroid<F>> {
    if core.len() <= core_cap {
        return core;
    }
    if core_cap == 0 {
        return Vec::new();
    }
    cap_with_second_klimit(&core, core_cap, d, family)
}

fn cap_with_second_klimit<F: FloatLike + FloatCore>(
    core: &[Centroid<F>],
    core_cap: usize,
    d: f64,
    family: ScaleFamily,
) -> Vec<Centroid<F>> {
    debug_assert!(core_cap > 0);
    if core.len() <= core_cap {
        return core.to_vec();
    }

    // Find the largest d' that satisfies the cap to preserve as much detail as possible.
    let mut lo = CAP_SEARCH_LO;
    let mut hi = d.max(CAP_SEARCH_LO);
    let mut best = lo;

    let boundaries = integer_k2_boundaries(core, family);
    for _ in 0..CAP_SEARCH_ITERS {
        let mid = 0.5 * (lo + hi);
        let len_mid = match &boundaries {
            Some(ranks) => count_integer_k2_clusters(ranks, mid),
            None => klimit_count(core, mid, family, KLIMIT_TOL),
        };
        if len_mid <= core_cap {
            best = mid;
            lo = mid;
        } else {
            hi = mid;
        }
    }

    let mut out = klimit_merge(core, best, family, KLIMIT_TOL);

    // Guard against rare non-monotone/plateau effects.
    let mut guard = 0usize;
    let mut tuned = best;
    while out.len() > core_cap && guard < CAP_FALLBACK_ITERS {
        tuned *= 0.95;
        out = klimit_merge(core, tuned, family, KLIMIT_TOL);
        guard += 1;
    }

    if out.len() > core_cap {
        return bucketize_equal_weight(core, core_cap);
    }

    out
}

/// Group in order and put residual mass in the final bucket. A bucket
/// with one centroid keeps its kind. Combining multiple centroids makes it mixed.
fn bucketize_equal_weight<F: FloatLike + FloatCore>(
    cs: &[Centroid<F>],
    buckets: usize,
) -> Vec<Centroid<F>> {
    debug_assert!(buckets > 0);
    if cs.is_empty() {
        return Vec::new();
    }
    if buckets == 1 {
        return vec![weighted_collapse(cs)];
    }

    let total_w: f64 = cs.iter().map(|c| c.weight_f64()).sum();
    if total_w <= 0.0 {
        return cs.iter().take(buckets).copied().collect();
    }

    let target = total_w / buckets as f64;
    let mut out = Vec::with_capacity(buckets);
    let mut acc = WeightedStats::default();

    for c in cs {
        acc.add(c);
        // Reserve the final bucket for all remaining positive contributions,
        // even when rounded total/target boundaries were reached early.
        if acc.w_sum >= target && acc.w_sum > 0.0 && out.len() < buckets - 1 {
            out.push(acc.centroid::<F>());
            acc.clear();
        }
    }
    if !acc.is_empty() && out.len() < buckets {
        out.push(acc.centroid::<F>());
    }
    out
}

fn assemble_with_edges<F: FloatLike + FloatCore>(
    left: &[Centroid<F>],
    core: Vec<Centroid<F>>,
    right: &[Centroid<F>],
) -> Vec<Centroid<F>> {
    let mut v = Vec::with_capacity(left.len() + core.len() + right.len());
    v.extend_from_slice(left);
    v.extend(core);
    v.extend_from_slice(right);
    v
}

#[derive(Debug)]
struct EdgeSlices<'a, F: FloatLike + FloatCore> {
    left: &'a [Centroid<F>],
    interior: &'a [Centroid<F>],
    right: &'a [Centroid<F>],
    core_cap: usize,
}

/// Only consecutive atomic centroids at an edge qualify for protection.
fn edge_run_len<F: FloatLike + FloatCore>(cs: &[Centroid<F>], k: usize, from_left: bool) -> usize {
    if k == 0 || cs.is_empty() {
        return 0;
    }
    let mut cnt = 0usize;
    if from_left {
        for c in cs {
            if c.is_singleton() && cnt < k {
                cnt += 1;
            } else {
                break;
            }
        }
    } else {
        for c in cs.iter().rev() {
            if c.is_singleton() && cnt < k {
                cnt += 1;
            } else {
                break;
            }
        }
    }
    cnt
}

fn flush_cluster<F: FloatLike + FloatCore>(
    clusters: &mut Vec<Centroid<F>>,
    acc: &mut WeightedStats,
) {
    if acc.is_empty() {
        return;
    }
    clusters.push(acc.centroid::<F>());
    acc.clear();
}

/// Collapse multiple centroids into one mixed cluster; preserve a lone input.
fn weighted_collapse<F: FloatLike + FloatCore>(slice: &[Centroid<F>]) -> Centroid<F> {
    let mut acc = WeightedStats::default();
    for c in slice {
        acc.add(c);
    }
    acc.centroid::<F>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tdigest::{singleton_policy::SingletonPolicy, ScaleFamily, TDigest};

    #[test]
    fn nonpositive_k2norm_domain_retains_existing_centroid_kinds_and_bits() {
        macro_rules! assert_retained {
            ($float:ty) => {{
                let items = [
                    Centroid::<$float>::new_mixed_f64(0.125, 0.25),
                    Centroid::<$float>::new_atomic_f64(100.0, 0.5),
                ];
                assert_eq!(
                    klimit_merge(&items, 1200.0, ScaleFamily::K2Norm, 0.0),
                    items.to_vec()
                );
            }};
        }
        assert_retained!(f32);
        assert_retained!(f64);
    }

    #[test]
    fn finite_cluster_average_stays_inside_its_input_mean_interval() {
        let left = 1.0000000000000009_f64;
        let right = 1.000000000000001_f64;
        let weights = [2e100, 3.0000000000000002e100];
        let raw_mean = (left * weights[0] + right * weights[1]) / (weights[0] + weights[1]);
        assert!(
            raw_mean > right,
            "fixture must exercise averaging overshoot"
        );
        for sign in [1.0, -1.0] {
            let mut inputs = [
                Centroid::<f64>::new_atomic_f64(left * sign, weights[0]),
                Centroid::<f64>::new_atomic_f64(right * sign, weights[1]),
            ];
            inputs.sort_by(|a, b| f64::total_cmp(&a.mean_f64(), &b.mean_f64()));
            let mut stats = WeightedStats::default();
            for centroid in &inputs {
                stats.add(centroid);
            }
            assert!(stats.mean() >= inputs[0].mean_f64());
            assert!(stats.mean() <= inputs[1].mean_f64());
            assert_eq!(
                stats.centroid::<f64>().weight_f64(),
                weights.iter().sum::<f64>()
            );
            assert!(!stats.centroid::<f64>().is_atomic());
        }
    }

    #[test]
    fn cluster_interval_bound_does_not_hide_nonfinite_accumulators() {
        for (mean, weight) in [(2.0, 1e308), (0.0, 1e308)] {
            let mut stats = WeightedStats::default();
            for _ in 0..2 {
                stats.add(&Centroid::<f64>::new_atomic_f64(mean, weight));
            }
            assert!(!stats.w_sum.is_finite());
            assert_eq!(stats.mean().is_nan(), (stats.mw_sum / stats.w_sum).is_nan());
            if !stats.mean().is_nan() {
                assert_eq!(stats.mean(), stats.mw_sum / stats.w_sum);
            }
        }
        let mut weighted_sum_overflow = WeightedStats::default();
        for _ in 0..2 {
            weighted_sum_overflow.add(&Centroid::<f64>::new_atomic_f64(1e308, 2.0));
        }
        assert!(weighted_sum_overflow.w_sum.is_finite());
        assert!(weighted_sum_overflow.mean().is_infinite());
    }

    #[test]
    fn all_compressor_paths_preserve_single_input_mean_bits_and_kind() {
        for centroid in [
            Centroid::<f64>::new_atomic_f64(-3.0, 0.1),
            Centroid::<f64>::new_mixed_f64(-3.0, 0.1),
            Centroid::<f64>::new_atomic_f64(-0.0, 0.1),
            Centroid::<f64>::new_mixed_f64(-0.0, 0.1),
        ] {
            let input = [centroid];
            let outputs = [
                klimit_merge(&input, 1000.0, ScaleFamily::K2, KLIMIT_TOL),
                bucketize_equal_weight(&input, 1),
                bucketize_equal_weight(&input, 2),
                vec![weighted_collapse(&input)],
                cap_core(input.to_vec(), 1, 1000.0, ScaleFamily::K2),
            ];
            for output in outputs {
                assert_eq!(output.len(), 1);
                assert_eq!(
                    output[0].mean_f64().to_bits(),
                    centroid.mean_f64().to_bits()
                );
                assert_eq!(
                    output[0].weight_f64().to_bits(),
                    centroid.weight_f64().to_bits()
                );
                assert_eq!(output[0].kind(), centroid.kind());
            }
        }
    }

    fn c<F: FloatLike + FloatCore>(m: f64, w: f64, singleton: bool) -> Centroid<F> {
        if singleton {
            Centroid::<F>::new_singleton_f64(m, w)
        } else {
            Centroid::<F>::new_mixed_f64(m, w)
        }
    }
    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() <= eps * (1.0 + a.abs() + b.abs())
    }

    type Fp = f64; // run these tests in f64 storage; f32 is covered elsewhere

    #[test]
    fn coalescing_equal_means_preserves_mixed_provenance() {
        let mut td = TDigest::<Fp>::builder()
            .max_size(100)
            .scale(ScaleFamily::K2)
            .singleton_policy(SingletonPolicy::Off)
            .build()
            .unwrap();

        let input = vec![
            c::<Fp>(1.0, 1.0, true),
            c::<Fp>(1.0, 2.5, false),
            c::<Fp>(2.0, 3.0, true),
        ];

        let out = super::compress_into(&mut td, 100, input).unwrap();
        assert_eq!(out.len(), 2);
        assert!(
            !out[0].is_singleton(),
            "a mixed centroid cannot become an exact point mass"
        );
        assert!(approx(out[0].mean_f64(), 1.0, 1e-12));
        assert!(approx(out[0].weight_f64(), 3.5, 1e-12));
        assert!(out[1].is_singleton());
        assert!(approx(out[1].mean_f64(), 2.0, 1e-12));
        assert!(approx(out[1].weight_f64(), 3.0, 1e-12));
    }

    #[test]
    #[should_panic(expected = "compress_into requires non-decreasing means")]
    fn compress_into_panics_on_unsorted_means() {
        let mut td = TDigest::<Fp>::builder()
            .max_size(10)
            .singleton_policy(SingletonPolicy::Off)
            .build()
            .unwrap();
        // `normalize_stream` should panic if means are decreasing.
        let input = vec![c::<Fp>(1.0, 1.0, true), c::<Fp>(0.9, 1.0, true)];
        let _ = super::compress_into(&mut td, 10, input).unwrap();
    }

    #[test]
    fn bucketize_equal_weight_preserves_total_weight_and_order() {
        let cs = vec![
            c::<Fp>(0.0, 1.0, false),
            c::<Fp>(2.0, 1.0, false),
            c::<Fp>(4.0, 1.0, false),
        ];
        let out = super::bucketize_equal_weight(&cs, 2);
        assert_eq!(out.len(), 2);

        let w_in: f64 = cs.iter().map(|x| x.weight_f64()).sum();
        let w_out: f64 = out.iter().map(|x| x.weight_f64()).sum();
        assert!(approx(w_in, w_out, 1e-12), "weight preserved");

        assert!(
            out[0].mean_f64() <= out[1].mean_f64(),
            "means remain ordered"
        );
    }

    #[test]
    fn bucketize_equal_weight_single_bucket_collapses_to_mixed() {
        let cs = vec![c::<Fp>(1.0, 2.0, false), c::<Fp>(3.0, 2.0, false)];
        let out = super::bucketize_equal_weight(&cs, 1);
        assert_eq!(out.len(), 1);
        assert!(approx(
            out[0].mean_f64(),
            (1.0 * 2.0 + 3.0 * 2.0) / 4.0,
            1e-12
        ));
        assert!(approx(out[0].weight_f64(), 4.0, 1e-12));
        assert!(!out[0].is_singleton(), "collapse yields mixed centroid");
    }

    #[test]
    fn cap_core_second_klimit_respects_cap_and_preserves_weight_order() {
        let core: Vec<_> = (0..200).map(|i| c::<Fp>(i as f64, 1.0, true)).collect();
        let out = super::cap_core(core.clone(), 32, 1000.0, ScaleFamily::K2);

        assert!(out.len() <= 32, "cap must be respected");

        let w_in: f64 = core.iter().map(|x| x.weight_f64()).sum();
        let w_out: f64 = out.iter().map(|x| x.weight_f64()).sum();
        assert!(approx(w_in, w_out, 1e-12), "weight preserved");

        for w in out.windows(2) {
            assert!(w[0].mean_f64() <= w[1].mean_f64(), "means remain ordered");
        }
    }

    #[test]
    fn klimit_merge_preserves_weight_order_and_flags() {
        let items = vec![
            c::<Fp>(0.0, 1.0, true),
            c::<Fp>(1.0, 1.0, true),
            c::<Fp>(1.1, 1.0, true),
            c::<Fp>(3.0, 2.0, false),
        ];
        let out = super::klimit_merge(&items, 10.0, ScaleFamily::K2, super::KLIMIT_TOL);
        assert!(!out.is_empty());

        let w_in: f64 = items.iter().map(|x| x.weight_f64()).sum();
        let w_out: f64 = out.iter().map(|x| x.weight_f64()).sum();
        assert!(approx(w_in, w_out, 1e-12), "weight preserved");

        for w in out.windows(2) {
            assert!(w[0].mean_f64() <= w[1].mean_f64(), "means non-decreasing");
        }

        if out.len() == 2 {
            assert!(out[0].is_singleton(), "single-item cluster remains atomic");
            assert!(!out[1].is_singleton(), "multi-item cluster must be mixed");
        }
    }

    #[test]
    fn delta_merge_matches_old_k2_rule_for_weighted_input() {
        // The 0.1.0 K2 rule uses the number of centroids as n and cumulative weight for q.
        let items = vec![
            c::<Fp>(0.0, 100.0, true),
            c::<Fp>(1.0, 1.0, true),
            c::<Fp>(2.0, 1.0, true),
            c::<Fp>(3.0, 1.0, true),
            c::<Fp>(4.0, 1.0, true),
        ];
        let delta = 20.0;
        let out = super::klimit_merge(&items, delta, ScaleFamily::K2Norm, 0.0);

        // Independently count clusters by solving for the upper rank of each cluster.
        let total_w: f64 = items.iter().map(|x| x.weight_f64()).sum();
        let n = items.len() as f64;
        let q_limit_at = |q0: f64| -> f64 {
            // q_limit = inv(k(q0) + 1). We compute it numerically by solving
            // q_to_k(q, delta, K2Norm, n) = q_to_k(q0, ..) + 1 via bisection.
            let target = crate::tdigest::scale::q_to_k(q0, delta, ScaleFamily::K2Norm, n) + 1.0;
            let mut lo = q0;
            let mut hi = 1.0 - 1e-15;
            for _ in 0..200 {
                let mid = 0.5 * (lo + hi);
                let k = crate::tdigest::scale::q_to_k(mid, delta, ScaleFamily::K2Norm, n);
                if k < target {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            0.5 * (lo + hi)
        };

        let mut count = 0usize;
        let mut cumulative = 0.0_f64;
        let mut sigma = items[0].weight_f64();
        let mut q_limit = q_limit_at(0.0);
        for c in items.iter().skip(1) {
            let w = c.weight_f64();
            let q = (cumulative + sigma + w) / total_w;
            if q <= q_limit {
                sigma += w;
            } else {
                count += 1;
                cumulative += sigma;
                q_limit = q_limit_at(cumulative / total_w);
                sigma = w;
            }
        }
        count += 1;

        assert_eq!(out.len(), count);
        let w_out: f64 = out.iter().map(|x| x.weight_f64()).sum();
        assert!(approx(total_w, w_out, 1e-12), "weight preserved");
    }

    #[test]
    fn delta_mode_bypasses_off_fast_path_and_runs_merge() {
        let mut td = TDigest::<Fp>::builder()
            .max_size(100)
            .scale(ScaleFamily::K2Norm)
            .singleton_policy(SingletonPolicy::Off)
            .delta(20.0)
            .build()
            .unwrap();

        let input = vec![
            c::<Fp>(0.0, 100.0, true),
            c::<Fp>(1.0, 1.0, true),
            c::<Fp>(2.0, 1.0, true),
            c::<Fp>(3.0, 1.0, true),
            c::<Fp>(4.0, 1.0, true),
        ];
        let out = super::compress_into(&mut td, 100, input).unwrap();
        assert_eq!(
            out.len(),
            4,
            "legacy mode must still merge when len <= max_size"
        );
    }

    #[test]
    fn edge_run_len_counts_atomic_only_on_requested_side() {
        let cs = vec![
            c::<Fp>(0.0, 1.0, true),
            c::<Fp>(0.1, 1.0, true),
            c::<Fp>(0.2, 1.0, false),
            c::<Fp>(0.3, 1.0, true),
            c::<Fp>(0.4, 1.0, true),
        ];
        assert_eq!(
            super::edge_run_len(&cs, 3, true),
            2,
            "left run stops at first mixed"
        );
        assert_eq!(super::edge_run_len(&cs, 1, true), 1);
        assert_eq!(
            super::edge_run_len(&cs, 2, false),
            2,
            "right run counts trailing atomic"
        );
        assert_eq!(super::edge_run_len(&cs, 10, false), 2);
    }

    #[test]
    fn policy_off_passthrough_under_capacity() {
        let mut td = TDigest::<Fp>::builder()
            .max_size(10)
            .singleton_policy(SingletonPolicy::Off)
            .build()
            .unwrap();

        let input = vec![
            c::<Fp>(0.0, 1.0, true),
            c::<Fp>(1.0, 1.0, true),
            c::<Fp>(2.0, 1.0, true),
        ];
        let out = super::compress_into(&mut td, 10, input.clone()).unwrap();
        assert_eq!(out.len(), 3);
        for (i, o) in out.iter().enumerate() {
            assert!(o.is_singleton());
            assert!(approx(o.mean_f64(), i as f64, 1e-12));
        }
    }

    #[test]
    fn policy_off_respects_max_size_via_bucketize() {
        let mut td = TDigest::<Fp>::builder()
            .max_size(3)
            .singleton_policy(SingletonPolicy::Off)
            .build()
            .unwrap();

        let input: Vec<_> = (0..10).map(|i| c::<Fp>(i as f64, 1.0, true)).collect();
        let out = super::compress_into(&mut td, 3, input).unwrap();
        assert!(
            out.len() <= 3,
            "must not exceed max_size (got {})",
            out.len()
        );
        let total_w: f64 = out.iter().map(|x| x.weight_f64()).sum();
        assert!(approx(total_w, 10.0, 1e-12), "weight preserved");
    }

    #[test]
    fn policy_use_protects_one_edge_each_and_caps_total() {
        let mut td = TDigest::<Fp>::builder()
            .max_size(3) // total cap in this policy (edges included)
            .singleton_policy(SingletonPolicy::Use)
            .build()
            .unwrap();

        let input: Vec<_> = (0..8).map(|i| c::<Fp>(i as f64, 1.0, true)).collect();
        let out = super::compress_into(&mut td, 3, input).unwrap();

        assert!(out.len() <= 3, "Use policy keeps total <= max_size");
        assert!(
            approx(out.first().unwrap().mean_f64(), 0.0, 1e-12),
            "leftmost preserved"
        );
        assert!(
            approx(out.last().unwrap().mean_f64(), 7.0, 1e-12),
            "rightmost preserved"
        );
        if out.len() == 3 {
            assert!(
                (out[1].mean_f64() > 0.0) && (out[1].mean_f64() < 7.0),
                "interior stays interior"
            );
        }
    }

    #[test]
    fn policy_use_with_protected_edges_keeps_k_atomic_edges_extra() {
        // Protect k=2 atomic at each side; core cap = max_size independently of edges.
        let mut td = TDigest::<Fp>::builder()
            .max_size(2) // core capacity only
            .singleton_policy(SingletonPolicy::UseWithProtectedEdges(2))
            .build()
            .unwrap();

        let mut input: Vec<_> = (0..12).map(|i| c::<Fp>(i as f64, 1.0, true)).collect();
        input[5] = c::<Fp>(5.0, 2.0, false);

        let out = super::compress_into(&mut td, 2, input).unwrap();

        assert!(out.len() >= 4 && out.len() <= 6);
        assert!(out[0].is_singleton() && approx(out[0].mean_f64(), 0.0, 1e-12));
        assert!(out[1].is_singleton() && approx(out[1].mean_f64(), 1.0, 1e-12));
        assert!(
            out[out.len() - 1].is_singleton() && approx(out[out.len() - 1].mean_f64(), 11.0, 1e-12)
        );
        assert!(
            out[out.len() - 2].is_singleton() && approx(out[out.len() - 2].mean_f64(), 10.0, 1e-12)
        );

        let core_count = out.len() - 4;
        assert!(core_count <= 2, "core must respect max_size");
    }

    #[test]
    fn cluster_mean_recovers_after_intermediate_moment_overflow() {
        let mut stats = super::WeightedStats::default();
        // Each product is finite; only the partial sum overflows.
        for mean in [-f64::MAX, -f64::MAX, 1.0, f64::MAX, f64::MAX] {
            stats.add(&Centroid::<f64>::new_atomic_f64(mean, 1.0));
        }
        assert_eq!(stats.mean(), 0.2);
        assert_eq!(stats.centroid::<f64>().weight(), 5.0);
        // Reusing the accumulator must discard its old compensation and scale.
        stats.clear();
        stats.add(&Centroid::<f64>::new_atomic_f64(6.0, 2.0));
        stats.add(&Centroid::<f64>::new_atomic_f64(12.0, 1.0));
        assert_eq!(stats.mean(), 8.0);
    }

    #[test]
    fn result_metadata_count_sum_min_max_are_set() {
        let mut td = TDigest::<Fp>::builder()
            .max_size(10)
            .singleton_policy(SingletonPolicy::Off)
            .build()
            .unwrap();

        let input = vec![
            c::<Fp>(-3.0, 1.0, true),
            c::<Fp>(-1.0, 2.0, true),
            c::<Fp>(4.0, 3.0, true),
        ];
        let w_sum: f64 = input.iter().map(|x| x.weight_f64()).sum();
        let mw_sum: f64 = input.iter().map(|x| x.weight_f64() * x.mean_f64()).sum();

        let out = super::compress_into(&mut td, 10, input).unwrap();
        assert!(!out.is_empty());
        assert!(approx(td.count(), w_sum, 1e-12));
        assert!(approx(td.sum(), mw_sum, 1e-12));
        assert!(approx(td.min(), -3.0, 1e-12));
        assert!(approx(td.max(), 4.0, 1e-12));
    }
}

#[cfg(test)]
mod count_probe_tests {
    use super::*;

    #[test]
    fn integer_k2_capacity_search_bounds_logarithm_calls() {
        let items: Vec<_> = (0..1500)
            .map(|i| Centroid::<f64>::new_atomic_f64(i as f64, (1 + i % 3) as f64))
            .collect();
        super::super::scale::K2_LOG_CALLS.with(|calls| calls.set(0));
        let out = cap_with_second_klimit(&items, 998, 1000.0, ScaleFamily::K2);
        assert!(out.len() <= 998);
        let calls = super::super::scale::K2_LOG_CALLS.with(|calls| calls.get());
        assert!(
            calls <= 4 * items.len() + 2,
            "{calls} K2 logarithms for {} centroids",
            items.len()
        );
    }

    #[test]
    fn cached_k2_trials_match_original_cluster_boundaries() {
        let items: Vec<_> = (0..300)
            .map(|i| Centroid::<f64>::new_atomic_f64(i as f64, [1., 17., 16777217.][i % 3]))
            .collect();
        let ranks = integer_k2_boundaries(&items, ScaleFamily::K2).unwrap();
        for trial in 1..1000 {
            let d = trial as f64 / 7.0;
            assert_eq!(
                count_integer_k2_clusters(&ranks, d),
                klimit_merge(&items, d, ScaleFamily::K2, KLIMIT_TOL).len()
            );
        }
        for weight in [0.5, (1u64 << 53) as f64, f64::MAX] {
            let c = Centroid::<f64>::new_atomic_f64(0., weight);
            assert!(integer_k2_boundaries(&[c], ScaleFamily::K2).is_none());
        }
    }

    fn compare_counts<F: FloatLike + FloatCore>() {
        for size in [0, 1, 2, 3, 16, 67, 200, 1003] {
            for weighted in [false, true] {
                let weights = [1.0, 0.5, 0.25, 16777217.0, 1e-100, 1e100];
                let items: Vec<Centroid<F>> = (0..size)
                    .map(|i| {
                        let mean = (i as f64 - size as f64 / 2.0) * 1e25;
                        let weight = if weighted {
                            weights[i % weights.len()]
                        } else {
                            1.0
                        };
                        if i % 2 == 0 {
                            Centroid::new_atomic_f64(mean, weight)
                        } else {
                            Centroid::new_mixed_f64(mean, weight)
                        }
                    })
                    .collect();
                for family in [
                    ScaleFamily::Quad,
                    ScaleFamily::K1,
                    ScaleFamily::K2,
                    ScaleFamily::K2Norm,
                    ScaleFamily::K3,
                ] {
                    for delta in [
                        f64::MIN_POSITIVE,
                        1e-6,
                        0.5,
                        2.0,
                        16.0,
                        32.0,
                        200.0,
                        1000.0,
                        1e100,
                    ] {
                        for tolerance in [0.0, KLIMIT_TOL] {
                            assert_eq!(klimit_count(&items, delta, family, tolerance), klimit_merge(&items, delta, family, tolerance).len(), "size={size} weighted={weighted} family={family:?} delta={delta} tolerance={tolerance}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn count_probe_matches_full_pass_f32() {
        compare_counts::<f32>();
    }

    #[test]
    fn count_probe_matches_full_pass_f64() {
        compare_counts::<f64>();
    }
}
