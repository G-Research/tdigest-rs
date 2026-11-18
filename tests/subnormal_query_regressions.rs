//! Query-only normalization of positive subnormal mass.

use ordered_float::FloatCore;
use tdigest_rs::tdigest::centroids::Centroid;
use tdigest_rs::tdigest::precision::FloatLike;
use tdigest_rs::tdigest::{DigestStats, TDigest};

fn seeded<F: FloatLike + FloatCore>(
    weights: &[f64],
    atomic: &[bool],
    min: f64,
    max: f64,
) -> TDigest<F> {
    let means: Vec<f64> = if weights.len() == 1 {
        vec![0.0]
    } else {
        vec![-2.0, 6.0]
    };
    let cents = means
        .iter()
        .zip(weights)
        .zip(atomic)
        .map(|((&mean, &weight), &is_atomic)| {
            if is_atomic {
                Centroid::<F>::new_atomic_f64(mean, weight)
            } else {
                Centroid::<F>::new_mixed_f64(mean, weight)
            }
        })
        .collect();
    TDigest::<F>::builder()
        .with_centroids_and_stats(
            cents,
            DigestStats {
                data_sum: means
                    .iter()
                    .zip(weights)
                    .map(|(mean, weight)| mean * weight)
                    .sum(),
                total_weight: weights.iter().sum(),
                data_min: min,
                data_max: max,
            },
        )
        .build()
        .unwrap()
}

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 2e-14,
        "actual={actual}, expected={expected}"
    );
}

fn single_pile<F: FloatLike + FloatCore>() {
    let td = seeded::<F>(&[f64::from_bits(1)], &[true], 0.0, 0.0);
    let before = td.to_bytes();
    assert_eq!(td.cdf(&[-1.0, 0.0, 1.0]), vec![0.0, 0.5, 1.0]);
    for q in [0.0, 0.1, 0.5, 0.9, 1.0] {
        assert_eq!(td.quantile(q), 0.0);
    }
    assert_eq!(td.to_bytes(), before);
}

#[test]
fn minimum_subnormal_atomic_pile_retains_midpoint_cdf_in_both_precisions() {
    single_pile::<f32>();
    single_pile::<f64>();
}

fn two_piles<F: FloatLike + FloatCore>() {
    let td = seeded::<F>(&[f64::from_bits(1); 2], &[true, true], -2.0, 6.0);
    let before = td.to_bytes();
    assert_eq!(
        td.cdf(&[f64::NEG_INFINITY, -2.0, 2.0, 6.0, f64::INFINITY]),
        vec![0.0, 0.25, 0.5, 0.75, 1.0]
    );
    for (q, expected) in [
        (0.0, -2.0),
        (0.1, -2.0),
        (0.25, -2.0),
        (0.5, 2.0),
        (0.6, 6.0),
        (0.75, 6.0),
        (1.0, 6.0),
    ] {
        assert_eq!(td.quantile(q), expected);
    }
    assert!(td.quantile(f64::NAN).is_nan());
    assert!(td.cdf(&[f64::NAN])[0].is_nan());
    assert!(td.cdf(&[]).is_empty());
    assert_eq!(td.to_bytes(), before);
}

#[test]
fn minimum_subnormal_two_piles_keep_rank_resolution_and_endpoints() {
    two_piles::<f32>();
    two_piles::<f64>();
}

fn single_mixed<F: FloatLike + FloatCore>() {
    let td = seeded::<F>(&[f64::from_bits(1)], &[false], -4.0, 6.0);
    let before = td.to_bytes();
    assert_eq!(
        td.cdf(&[-4.0, -2.0, 0.0, 3.0, 6.0]),
        vec![0.0, 0.25, 0.5, 0.75, 1.0]
    );
    for q in [0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0] {
        close(td.cdf(&[td.quantile(q)])[0], q);
    }
    assert_eq!(td.quantile(0.0), -4.0);
    assert_eq!(td.quantile(1.0), 6.0);
    assert_eq!(td.to_bytes(), before);
}

#[test]
fn minimum_subnormal_single_mixed_centroid_retains_both_tails() {
    single_mixed::<f32>();
    single_mixed::<f64>();
}

fn invariant_shapes<F: FloatLike + FloatCore>() {
    for kinds in [[true, true], [true, false], [false, true], [false, false]] {
        let min = if kinds[0] { -2.0 } else { -6.0 };
        let max = if kinds[1] { 6.0 } else { 10.0 };
        let reference = seeded::<F>(&[1.0, 3.0], &kinds, min, max);
        for factor in [
            f64::from_bits(1),
            f64::from_bits(17),
            f64::MIN_POSITIVE / 16.0,
            f64::MIN_POSITIVE / 4.0,
            1.0,
        ] {
            let td = seeded::<F>(&[factor, factor * 3.0], &kinds, min, max);
            let before = td.to_bytes();
            let probes = [-7.0, -6.0, -4.0, -2.0, 0.0, 2.0, 6.0, 8.0, 10.0, 11.0];
            let actual = td.cdf(&probes);
            for (got, want) in actual.iter().zip(reference.cdf(&probes)) {
                close(*got, want);
            }
            assert!(actual.windows(2).all(|pair| pair[0] <= pair[1]));
            for q in [0.0, 0.01, 0.125, 0.25, 0.5, 0.625, 0.75, 0.99, 1.0] {
                close(td.quantile(q), reference.quantile(q));
            }
            assert_eq!(td.to_bytes(), before);
        }
    }
}

#[test]
fn atomic_and_mixed_shapes_are_invariant_to_subnormal_mass_units() {
    invariant_shapes::<f32>();
    invariant_shapes::<f64>();
}

fn scaled_ingestion<F: FloatLike + FloatCore>() {
    let reference = TDigest::<F>::from_weighted_unsorted(
        &[F::from_f64(-2.0), F::from_f64(6.0)],
        &[1.0, 3.0],
        1000,
    )
    .unwrap();
    let mut tiny = reference.clone();
    tiny.scale_weights(f64::from_bits(1)).unwrap();
    let before = tiny.to_bytes();
    assert_eq!(
        tiny.cdf(&[-2.0, 2.0, 6.0]),
        reference.cdf(&[-2.0, 2.0, 6.0])
    );
    for q in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0] {
        assert_eq!(tiny.quantile(q), reference.quantile(q));
    }
    assert_eq!(tiny.to_bytes(), before);
}

#[test]
fn public_weight_scaling_keeps_subnormal_query_shape() {
    scaled_ingestion::<f32>();
    scaled_ingestion::<f64>();
}

#[test]
fn empty_and_normal_mass_behavior_is_unchanged() {
    let empty = TDigest::<f64>::default();
    assert!(empty.cdf(&[0.0])[0].is_nan());
    assert!(empty.quantile(0.5).is_nan());
    let normal = seeded::<f64>(&[1.0, 3.0], &[false, false], -6.0, 10.0);
    assert_eq!(
        normal.cdf(&[-6.0, -2.0, 2.0, 6.0, 10.0]),
        vec![0.0, 0.125, 0.375, 0.625, 1.0]
    );
    assert_eq!(normal.quantile(0.0), -6.0);
    assert_eq!(normal.quantile(0.125), -2.0);
    assert_eq!(normal.quantile(0.625), 6.0);
    assert_eq!(normal.quantile(1.0), 10.0);
}

fn with_count<F: FloatLike + FloatCore>(
    td: &TDigest<F>,
    count: f64,
) -> tdigest_rs::TdResult<TDigest<F>> {
    TDigest::<F>::builder()
        .with_centroids_and_stats(
            td.centroids().to_vec(),
            DigestStats {
                data_sum: td.sum(),
                total_weight: count,
                data_min: td.min(),
                data_max: td.max(),
            },
        )
        .build()
}

fn different_subnormal_header<F: FloatLike + FloatCore>() {
    let tiny = f64::from_bits(1);
    let correct = seeded::<F>(&[tiny, tiny], &[true, true], -2.0, 6.0);
    assert!(with_count(&correct, tiny * 3.0).is_err());
}

#[test]
fn inconsistent_subnormal_header_is_rejected_at_construction() {
    different_subnormal_header::<f32>();
    different_subnormal_header::<f64>();
}

fn inconsistent_normal_boundary<F: FloatLike + FloatCore>() {
    let tiny = f64::from_bits(1);
    let normal = seeded::<F>(&[1.0, 1.0], &[true, true], -2.0, 6.0);
    let subnormal = seeded::<F>(&[tiny, tiny], &[true, true], -2.0, 6.0);
    assert!(with_count(&normal, tiny).is_err());
    assert!(with_count(&subnormal, 1.0).is_err());
}

#[test]
fn inconsistent_normal_and_subnormal_headers_are_rejected() {
    inconsistent_normal_boundary::<f32>();
    inconsistent_normal_boundary::<f64>();
}

fn normal_total_tiny_component<F: FloatLike + FloatCore>() {
    let td = seeded::<F>(&[f64::from_bits(1), 1.0], &[true, true], -2.0, 6.0);
    let before = td.to_bytes();
    assert_eq!(td.count(), 1.0);
    // A subnormal component with normal total deliberately keeps the ordinary
    // path. This checks ordinary-scale queries, not the minimum-subnormal rank
    // boundary: resolving that component's half-weight is a separate issue.
    assert_eq!(td.cdf(&[-3.0, 6.0, 7.0]), vec![0.0, 0.5, 1.0]);
    assert_eq!(td.quantile(0.0), -2.0);
    for q in [0.25, 0.5, 0.75, 1.0] {
        assert_eq!(td.quantile(q), 6.0);
    }
    assert_eq!(td.to_bytes(), before);
}

#[test]
fn normal_total_with_subnormal_component_keeps_ordinary_query_behavior() {
    normal_total_tiny_component::<f32>();
    normal_total_tiny_component::<f64>();
}

#[test]
fn tiny_positive_quantiles_retain_large_finite_mixed_tail_values() {
    let build = |weight: f64| {
        TDigest::<f64>::builder()
            .with_centroids_and_stats(
                vec![
                    Centroid::new_mixed_f64(1e308, weight),
                    Centroid::new_mixed_f64(1.2e308, weight),
                ],
                DigestStats {
                    data_sum: 1e308 * weight + 1.2e308 * weight,
                    total_weight: 2.0 * weight,
                    data_min: 0.0,
                    data_max: 1.4e308,
                },
            )
            .build()
            .unwrap()
    };
    let tiny = build(f64::from_bits(1));
    let normal = build(0.5);
    let before = tiny.to_bytes();
    for q in [f64::from_bits(1), 1e-310, 1e-300, 0.25, 0.5, 0.75, 1.0] {
        let expected = normal.quantile(q);
        let actual = tiny.quantile(q);
        assert!(expected > 0.0 && expected.is_finite());
        assert_eq!(actual, expected, "q={q}");
        let roundtrip = tiny.cdf(&[actual])[0];
        assert_eq!(roundtrip, normal.cdf(&[expected])[0]);
        // Absolute epsilons would accept zero for the tiny probabilities.
        assert!(roundtrip > 0.0);
        assert!(
            ((roundtrip - q) / q).abs() <= 1e-12,
            "q={q}, roundtrip={roundtrip}"
        );
    }
    assert_eq!(tiny.to_bytes(), before);
}
