//! Compressor: runs the TDigest pipeline stages **1→6** on an input stream of centroids.
//!
//! ### Pipeline (referenced by numbers throughout)
//! 1. **Normalize**: validate non-decreasing means; coalesce adjacent equal-means into a single
//!    **atomic** centroid at that mean (weight adds up); compute (∑w, ∑w·mean, min, max).
//! 2. **Slice**: partition `(left edges, interior, right edges)` and compute interior capacity
//!    from the active [`SingletonPolicy`].
//! 3. **Merge (k-limit)**: greedily merge interior under Δk ≤ 1 for the selected [`ScaleFamily`].
//! 4. **Cap**: if interior exceeds its budget, apply order-preserving equal-weight bucketization.
//! 5. **Assemble**: concatenate `left + core + right`.
//! 6. **Post**: optional policy finalization (e.g., total-cap for `Use`).
//!
//! #### Invariants
//! - Output centroids are strictly increasing by mean.
//! - Total weight is preserved (up to floating rounding).
//! - Stage ownership is explicit: same-mean runs are coalesced only in **1**; edge rules only in **2/6**.

use ordered_float::FloatCore;

use crate::tdigest::centroids::Centroid;
use crate::tdigest::merges::normalize_stream;
use crate::tdigest::precision::FloatLike;
use crate::tdigest::scale::{q_to_k, ScaleFamily};
use crate::tdigest::singleton_policy::SingletonPolicy;
use crate::tdigest::TDigest;

const KLIMIT_TOL: f64 = 1e-12;

/* =============================================================================
 * Internal utilities
 * ============================================================================= */

#[derive(Clone, Copy, Debug, Default)]
struct WeightedStats {
    w_sum: f64,
    mw_sum: f64,
}

impl WeightedStats {
    #[inline]
    fn add(&mut self, mean_f64: f64, weight_f64: f64) {
        self.w_sum += weight_f64;
        self.mw_sum += weight_f64 * mean_f64;
    }
    #[inline]
    fn mean(&self) -> f64 {
        if self.w_sum > 0.0 {
            self.mw_sum / self.w_sum
        } else {
            0.0
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
    }
}

/// Build a centroid from an accumulated cluster (used in **3**).
///
/// - Single-item clusters remain **atomic** only if the head was data-true atomic.
/// - Multi-item clusters are emitted as **mixed**.
#[inline]
fn build_centroid<F: FloatLike + FloatCore>(
    mean_f64: f64,
    weight_f64: f64,
    singleton_head: bool,
    cluster_len: usize,
) -> Centroid<F> {
    if singleton_head && cluster_len == 1 {
        Centroid::<F>::new_singleton_f64(mean_f64, weight_f64)
    } else {
        Centroid::<F>::new_mixed_f64(mean_f64, weight_f64)
    }
}

/// Run the full pipeline (**1→6**) on `items` and write digest-level metadata to `result`.
///
/// Stages:
/// 1) Normalize → `normalize_stream`
/// 2) Slice     → policy-dependent edge/interior partition
/// 3) Merge     → `klimit_merge`
/// 4) Cap       → `cap_core` / `bucketize_equal_weight`
/// 5) Assemble  → `assemble_with_edges`
/// 6) Post      → policy-specific finalization
pub(crate) fn compress_into<F, I>(
    result: &mut TDigest<F>,
    max_size: usize,
    items: I,
) -> Vec<Centroid<F>>
where
    F: FloatLike + FloatCore,
    I: IntoIterator<Item = Centroid<F>>,
{
    // -- 1) Normalize ---------------------------------------------------------
    let norm = normalize_stream::<F, _>(items);
    if norm.out.is_empty() {
        return norm.out;
    }

    // Update digest-level metadata derived from normalized stream.
    result.set_count(norm.total_w);
    result.set_sum(norm.total_mw);
    result.set_min(norm.min);
    result.set_max(norm.max);

    // -- 2) Slice -------------------------------------------------------------
    let policy = policy_from(result.singleton_policy(), max_size);
    if let Some(v) = policy.fast_path(&norm.out) {
        debug_assert!(crate::tdigest::centroids::is_sorted_strict_by_mean(&v));
        return v;
    }
    let slices = policy.slice(&norm.out);

    // -- 3) Merge (k-limit) ---------------------------------------------------
    let family: ScaleFamily = result.scale();
    let d: f64 = result.max_size() as f64;
    let core = klimit_merge::<F>(slices.interior, d, family);

    // -- 4) Cap ---------------------------------------------------------------
    // Equal-weight bucketization if the interior exceeds its budget.
    // We may emit fewer than N to avoid tiny last buckets, which improves tail stability.
    let core_capped = cap_core::<F>(core, slices.caps.core_cap);

    // -- 5) Assemble ----------------------------------------------------------
    let assembled = assemble_with_edges::<F>(slices.left, core_capped, slices.right);

    // -- 6) Post --------------------------------------------------------------
    let compressed = policy.post(assembled);

    debug_assert!(crate::tdigest::centroids::is_sorted_strict_by_mean(
        &compressed
    ));
    compressed
}

/* =============================================================================
 * Policy (Stages 2 & 6)
 * ============================================================================= */

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
    /// Fast-path for trivial/exact cases (part of **2**). Returns `Some` when no further work is needed.
    fn fast_path<F: FloatLike + FloatCore>(&self, out: &[Centroid<F>]) -> Option<Vec<Centroid<F>>> {
        match *self {
            Policy::Off { max } => {
                if out.len() <= max {
                    return Some(out.to_vec());
                }
                if max == 1 {
                    return Some(if out.len() == 1 {
                        out.to_vec()
                    } else {
                        vec![weighted_collapse_mixed(out)]
                    });
                }
                if max == 2 {
                    return Some(bucketize_equal_weight(out, 2));
                }
                None
            }
            Policy::Use { .. } | Policy::UseProt { .. } => None,
        }
    }

    /// **(2) Slice** — compute `(left, interior, right)` and interior capacity.
    fn slice<'a, F: FloatLike + FloatCore>(&self, out: &'a [Centroid<F>]) -> EdgeSlices<'a, F> {
        match *self {
            // No edge protection; core capacity equals `max`.
            Policy::Off { max } => EdgeSlices {
                left: &[],
                interior: out,
                right: &[],
                caps: EdgeCaps { core_cap: max },
            },
            // Preserve endpoints; core capacity excludes the two edges.
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
                    caps: EdgeCaps { core_cap },
                }
            }
            // Protect up to `k` **atomic** centroids per edge; interior has independent cap.
            Policy::UseProt { max, k } => {
                let n = out.len();
                let l_prot = edge_run_len(out, k, true);
                let r_prot = edge_run_len(out, k, false);

                let left_end = l_prot.min(n);
                let right_start = n.saturating_sub(r_prot);

                let left = &out[..left_end];
                let right = if right_start > left_end {
                    &out[right_start..]
                } else {
                    &[]
                };
                let interior = if right_start > left_end {
                    &out[left_end..right_start]
                } else {
                    &[]
                };

                EdgeSlices {
                    left,
                    interior,
                    right,
                    caps: EdgeCaps { core_cap: max },
                }
            }
        }
    }

    /// **(6) Post** — finalize per policy (e.g., total-cap enforcement for `Use`).
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

/* =============================================================================
 * Core operations (Stages 3, 4, 5)
 * ============================================================================= */

/// **(3) Merge (k-limit)** — greedy interior merge under Δk ≤ 1.
///
/// Guarantees:
/// - Order and total weight are preserved.
/// - Single-item clusters remain **atomic** only when the head was data-true atomic.
/// - Multi-item clusters are **mixed**.
fn klimit_merge<F: FloatLike + FloatCore>(
    items: &[Centroid<F>],
    d: f64,
    family: ScaleFamily,
) -> Vec<Centroid<F>> {
    if items.is_empty() {
        return Vec::new();
    }
    let total_w: f64 = items.iter().map(|c| c.weight_f64()).sum();

    // C = mass of completed clusters already emitted.
    let mut c_acc = 0.0_f64; // running capacity accumulator

    let mut clusters: Vec<Centroid<F>> = Vec::with_capacity(items.len());
    let mut acc = WeightedStats::default();
    let mut cluster_len: usize = 0;
    let mut singleton_head = true;

    let mut q_l = c_acc / total_w;
    let mut k_left = q_to_k(q_l, d, family);

    for c in items {
        let w_next = c.weight_f64();
        let q_r = (c_acc + acc.w_sum + w_next) / total_w;
        let k_right = q_to_k(q_r, d, family);

        if (k_right - k_left) <= 1.0 + KLIMIT_TOL {
            acc.add(c.mean_f64(), w_next);
            if cluster_len == 0 {
                singleton_head = c.is_singleton();
            } else {
                singleton_head = false;
            }
            cluster_len += 1;
        } else {
            flush_cluster::<F>(
                &mut clusters,
                &mut acc,
                &mut cluster_len,
                &mut singleton_head,
            );
            if let Some(last) = clusters.last() {
                c_acc += last.weight_f64();
            }
            q_l = c_acc / total_w;
            k_left = q_to_k(q_l, d, family);

            acc.add(c.mean_f64(), w_next);
            cluster_len = 1;
            singleton_head = c.is_singleton();
        }
    }
    flush_cluster::<F>(
        &mut clusters,
        &mut acc,
        &mut cluster_len,
        &mut singleton_head,
    );

    clusters
}

/// **(4) Cap** — reduce to at most `core_cap` by equal-weight grouping.
///
/// The last group may be omitted if it would be too small; this avoids micro-weights that
/// degrade numerical stability near the tails.
fn cap_core<F: FloatLike + FloatCore>(core: Vec<Centroid<F>>, core_cap: usize) -> Vec<Centroid<F>> {
    if core.len() <= core_cap {
        return core;
    }
    if core_cap == 0 {
        return Vec::new();
    }
    bucketize_equal_weight(&core, core_cap)
}

/// **(4) Cap** — order-preserving equal-weight bucketization into `buckets` groups.
///
/// - Emits up to `buckets` outputs; a tiny trailing remainder may be skipped (see [`cap_core`]).
/// - Outputs are **mixed** centroids.
fn bucketize_equal_weight<F: FloatLike + FloatCore>(
    cs: &[Centroid<F>],
    buckets: usize,
) -> Vec<Centroid<F>> {
    debug_assert!(buckets > 0);
    if cs.is_empty() {
        return Vec::new();
    }
    if buckets == 1 {
        return vec![weighted_collapse_mixed(cs)];
    }

    let total_w: f64 = cs.iter().map(|c| c.weight_f64()).sum();
    if total_w <= 0.0 {
        return cs
            .iter()
            .take(buckets)
            .map(|c| Centroid::<F>::new_mixed_f64(c.mean_f64(), c.weight_f64()))
            .collect();
    }

    let target = total_w / buckets as f64;
    let mut out = Vec::with_capacity(buckets);
    let mut acc = WeightedStats::default();

    for c in cs {
        acc.add(c.mean_f64(), c.weight_f64());
        if acc.w_sum >= target && acc.w_sum > 0.0 {
            out.push(Centroid::<F>::new_mixed_f64(acc.mean(), acc.w_sum));
            acc.clear();
            if out.len() == buckets {
                break;
            }
        }
    }
    if !acc.is_empty() && out.len() < buckets {
        out.push(Centroid::<F>::new_mixed_f64(acc.mean(), acc.w_sum));
    }
    out
}

/// **(5) Assemble** — concatenate `left`, `core`, `right` without element mutation.
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

/* =============================================================================
 * Edge helpers (used by Stage 2)
 * ============================================================================= */

#[derive(Debug, Clone, Copy)]
struct EdgeCaps {
    core_cap: usize,
}

#[derive(Debug)]
struct EdgeSlices<'a, F: FloatLike + FloatCore> {
    left: &'a [Centroid<F>],
    interior: &'a [Centroid<F>],
    right: &'a [Centroid<F>],
    caps: EdgeCaps,
}

/// Count up to `k` consecutive **atomic** centroids on one edge.
///
/// Stops at the first **mixed** centroid. When `from_left=false`, scans from the right.
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

/* =============================================================================
 * Small helpers
 * ============================================================================= */

/// Emit the current cluster (if any) and reset accumulation (used by **3**).
fn flush_cluster<F: FloatLike + FloatCore>(
    clusters: &mut Vec<Centroid<F>>,
    acc: &mut WeightedStats,
    cluster_len: &mut usize,
    singleton_head: &mut bool,
) {
    if acc.is_empty() {
        return;
    }
    let mean = acc.mean();
    let out = build_centroid::<F>(mean, acc.w_sum, *singleton_head, *cluster_len);
    clusters.push(out);

    acc.clear();
    *cluster_len = 0;
    *singleton_head = true;
}

/// Collapse a slice into a single **mixed** centroid (helper for **4**).
fn weighted_collapse_mixed<F: FloatLike + FloatCore>(slice: &[Centroid<F>]) -> Centroid<F> {
    let mut acc = WeightedStats::default();
    for c in slice {
        acc.add(c.mean_f64(), c.weight_f64());
    }
    Centroid::<F>::new_mixed_f64(acc.mean(), acc.w_sum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tdigest::{singleton_policy::SingletonPolicy, ScaleFamily, TDigest};

    // ---- tiny helpers ----
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

    // ---------- streaming/coalescing pass ----------

    #[test]
    fn coalesces_equal_means_into_atomic_centroid() {
        let mut td = TDigest::<Fp>::builder()
            .max_size(100)
            .scale(ScaleFamily::K2)
            .singleton_policy(SingletonPolicy::Off)
            .build();

        let input = vec![
            c::<Fp>(1.0, 1.0, true),
            c::<Fp>(1.0, 2.5, false),
            c::<Fp>(2.0, 3.0, true),
        ];

        let out = super::compress_into(&mut td, 100, input);
        assert_eq!(out.len(), 2);
        assert!(out[0].is_singleton(), "equal-mean run collapses to atomic");
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
            .build();
        // `normalize_stream` should panic if means are decreasing.
        let input = vec![c::<Fp>(1.0, 1.0, true), c::<Fp>(0.9, 1.0, true)];
        let _ = super::compress_into(&mut td, 10, input);
    }

    // ---------- bucketizer ----------

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

    // ---------- k-limit merge core ----------

    #[test]
    fn klimit_merge_preserves_weight_order_and_flags() {
        let items = vec![
            c::<Fp>(0.0, 1.0, true),
            c::<Fp>(1.0, 1.0, true),
            c::<Fp>(1.1, 1.0, true),
            c::<Fp>(3.0, 2.0, false),
        ];
        let out = super::klimit_merge(&items, 10.0, ScaleFamily::K2);
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

    // ---------- edge run length ----------

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

    // ---------- compress_into policies ----------

    #[test]
    fn policy_off_passthrough_under_capacity() {
        let mut td = TDigest::<Fp>::builder()
            .max_size(10)
            .singleton_policy(SingletonPolicy::Off)
            .build();

        let input = vec![
            c::<Fp>(0.0, 1.0, true),
            c::<Fp>(1.0, 1.0, true),
            c::<Fp>(2.0, 1.0, true),
        ];
        let out = super::compress_into(&mut td, 10, input.clone());
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
            .build();

        let input: Vec<_> = (0..10).map(|i| c::<Fp>(i as f64, 1.0, true)).collect();
        let out = super::compress_into(&mut td, 3, input);
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
            .build();

        let input: Vec<_> = (0..8).map(|i| c::<Fp>(i as f64, 1.0, true)).collect();
        let out = super::compress_into(&mut td, 3, input);

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
            .build();

        let mut input: Vec<_> = (0..12).map(|i| c::<Fp>(i as f64, 1.0, true)).collect();
        input[5] = c::<Fp>(5.0, 2.0, false);

        let out = super::compress_into(&mut td, 2, input);

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

    // ---------- metadata on result ----------

    #[test]
    fn result_metadata_count_sum_min_max_are_set() {
        let mut td = TDigest::<Fp>::builder()
            .max_size(10)
            .singleton_policy(SingletonPolicy::Off)
            .build();

        let input = vec![
            c::<Fp>(-3.0, 1.0, true),
            c::<Fp>(-1.0, 2.0, true),
            c::<Fp>(4.0, 3.0, true),
        ];
        let w_sum: f64 = input.iter().map(|x| x.weight_f64()).sum();
        let mw_sum: f64 = input.iter().map(|x| x.weight_f64() * x.mean_f64()).sum();

        let out = super::compress_into(&mut td, 10, input);
        assert!(!out.is_empty());
        assert!(approx(td.count(), w_sum, 1e-12));
        assert!(approx(td.sum(), mw_sum, 1e-12));
        assert!(approx(td.min(), -3.0, 1e-12));
        assert!(approx(td.max(), 4.0, 1e-12));
    }
}
