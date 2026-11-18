//! Regressions for weighted tails and retained extrema in compressed digests.

use ordered_float::FloatCore;
use tdigest_rs::tdigest::centroids::Centroid;
use tdigest_rs::tdigest::precision::FloatLike;
use tdigest_rs::tdigest::singleton_policy::SingletonPolicy;
use tdigest_rs::tdigest::{DigestStats, ScaleFamily, TDigest};

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1e-12,
        "actual={actual}, expected={expected}"
    );
}

fn mixed_edges() -> TDigest<f64> {
    TDigest::builder()
        .with_centroids_and_stats(
            vec![
                Centroid::new_mixed_f64(10.0, 0.2),
                Centroid::new_mixed_f64(90.0, 0.3),
            ],
            DigestStats {
                data_sum: 29.0,
                total_weight: 0.5,
                data_min: 0.0,
                data_max: 100.0,
            },
        )
        .build()
        .unwrap()
}

#[test]
fn fractional_mixed_tails_are_bounded_monotone_and_inverse_to_quantile() {
    let td = mixed_edges();
    let probes = [-1.0, 0.0, 1.0, 5.0, 10.0, 50.0, 90.0, 95.0, 100.0, 101.0];
    let expected = [0.0, 0.0, 0.02, 0.1, 0.2, 0.45, 0.7, 0.85, 1.0, 1.0];
    let actual = td.cdf(&probes);
    for (&got, &want) in actual.iter().zip(&expected) {
        assert!((0.0..=1.0).contains(&got));
        close(got, want);
    }
    assert!(actual.windows(2).all(|pair| pair[0] <= pair[1]));
    for q in [0.0, 0.01, 0.1, 0.2, 0.5, 0.7, 0.9, 0.99, 1.0] {
        close(td.cdf(&[td.quantile(q)])[0], q);
    }
}

#[test]
fn rescaling_weight_units_preserves_mixed_tail_probabilities() {
    let td = mixed_edges();
    let probes = [-1.0, 0.0, 1.0, 5.0, 10.0, 50.0, 90.0, 95.0, 100.0, 101.0];
    let expected = td.cdf(&probes);
    for factor in [1e-6, 0.001, 17.0, 1e6] {
        let mut scaled = td.clone();
        scaled.scale_weights(factor).unwrap();
        for (got, &want) in scaled.cdf(&probes).into_iter().zip(&expected) {
            close(got, want);
        }
        for q in [0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0] {
            close(scaled.quantile(q), td.quantile(q));
        }
    }
}

fn compressed_extrema_survive_updates<F: FloatLike + FloatCore>() {
    let values: Vec<F> = (0..100).map(|x| F::from_f64(x as f64)).collect();
    let base = TDigest::<F>::builder()
        .scale(ScaleFamily::K2Norm)
        .singleton_policy(SingletonPolicy::Off)
        .delta(1.0)
        .build()
        .unwrap()
        .merge_unsorted(values)
        .unwrap();
    assert!(base.centroids()[0].mean_f64() > base.min());

    let mut added = base.clone();
    added.add(F::from_f64(50.0)).unwrap();
    let mut weighted = base.clone();
    weighted.add_weighted(F::from_f64(50.0), 0.25).unwrap();
    let merged = TDigest::merge_digests(vec![base.clone(), base.clone()]).unwrap();
    let single = TDigest::merge_digests(vec![base.clone()]).unwrap();
    for td in [&base, &added, &weighted, &merged, &single] {
        assert_eq!(td.min(), 0.0);
        assert_eq!(td.max(), 99.0);
        assert_eq!(td.quantile(0.0), 0.0);
        assert_eq!(td.quantile(1.0), 99.0);
    }

    let mut extended = added;
    extended
        .add_many(vec![F::from_f64(-10.0), F::from_f64(120.0)])
        .unwrap();
    extended.add(F::from_f64(50.0)).unwrap();
    assert_eq!(extended.min(), -10.0);
    assert_eq!(extended.max(), 120.0);

    // Exercise both mixed edge clusters, including a non-atomic upper edge.
    let mixed = TDigest::<F>::builder()
        .singleton_policy(SingletonPolicy::Off)
        .with_centroids_and_stats(
            vec![
                Centroid::new_mixed_f64(10.0, 50.0),
                Centroid::new_mixed_f64(90.0, 50.0),
            ],
            DigestStats {
                data_sum: 5000.0,
                total_weight: 100.0,
                data_min: 0.0,
                data_max: 100.0,
            },
        )
        .build()
        .unwrap();
    let mut updated = mixed.clone();
    updated.add(F::from_f64(50.0)).unwrap();
    let merged = TDigest::merge_digests(vec![mixed.clone(), mixed]).unwrap();
    for td in [&updated, &merged] {
        assert_eq!(td.min(), 0.0);
        assert_eq!(td.max(), 100.0);
        assert_eq!(td.quantile(0.0), 0.0);
        assert_eq!(td.quantile(1.0), 100.0);
    }
}

#[test]
fn f32_compressed_extrema_survive_add_and_merge() {
    compressed_extrema_survive_updates::<f32>();
}

#[test]
fn f64_compressed_extrema_survive_add_and_merge() {
    compressed_extrema_survive_updates::<f64>();
}

#[test]
fn one_mixed_centroid_retains_both_tails() {
    let base = TDigest::<f64>::builder()
        .with_centroids_and_stats(
            vec![Centroid::new_mixed_f64(40.0, 0.25)],
            DigestStats {
                data_sum: 10.0,
                total_weight: 0.25,
                data_min: 0.0,
                data_max: 100.0,
            },
        )
        .build()
        .unwrap();
    for factor in [0.001, 1.0, 1000.0] {
        let mut td = base.clone();
        td.scale_weights(factor).unwrap();
        for (q, expected) in [
            (0.0, 0.0),
            (0.25, 20.0),
            (0.5, 40.0),
            (0.75, 70.0),
            (1.0, 100.0),
        ] {
            close(td.quantile(q), expected);
            close(td.cdf(&[td.quantile(q)])[0], q);
        }
    }
    let atomic = TDigest::<f64>::from_weighted_unsorted(&[40.0], &[0.25], 10).unwrap();
    for q in [0.0, 0.25, 0.5, 0.75, 1.0] {
        assert_eq!(atomic.quantile(q), 40.0);
    }
}

#[test]
fn finite_extreme_centroid_ranges_do_not_overflow_interpolation() {
    let td = TDigest::<f64>::builder()
        .with_centroids_and_stats(
            vec![Centroid::new_mixed_f64(1e308, 0.25)],
            DigestStats {
                data_sum: 2.5e307,
                total_weight: 0.25,
                data_min: -1e308,
                data_max: 1.1e308,
            },
        )
        .build()
        .unwrap();
    assert_eq!(td.quantile(0.0), -1e308);
    assert_eq!(td.quantile(0.25), 0.0);
    assert_eq!(td.quantile(0.5), 1e308);
    assert_eq!(td.quantile(1.0), 1.1e308);
    for q in [0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0] {
        let value = td.quantile(q);
        assert!(value.is_finite());
        close(td.cdf(&[value])[0], q);
    }

    // These finite observations have an unrepresentable f64 data sum. The
    // constructor rejects their aggregate overflow.
    assert!(TDigest::<f64>::from_unsorted(&[1e308, 1.1e308], 10).is_err());
    let middle =
        TDigest::<f64>::from_weighted_unsorted(&[1e308, 1.1e308], &[0.5, 0.5], 10).unwrap();
    assert!(middle.median().is_finite());
    assert!((1e308..=1.1e308).contains(&middle.median()));
}

#[test]
fn touching_or_overlapping_protected_edges_keep_all_mass() {
    for n in 1..=20 {
        for pin in 1..=10 {
            let values: Vec<f64> = (0..n).map(|i| i as f64).collect();
            let td = TDigest::builder()
                .max_size(1000)
                .singleton_policy(SingletonPolicy::UseWithProtectedEdges(pin))
                .build()
                .unwrap()
                .merge_unsorted(values.clone())
                .unwrap();
            assert_eq!(td.count(), n as f64);
            assert_eq!(
                td.centroids().iter().map(|c| c.weight_f64()).sum::<f64>(),
                n as f64
            );
            assert_eq!(td.centroids().len(), n);
            for (i, &value) in values.iter().enumerate() {
                assert_eq!(td.quantile((i as f64 + 0.5) / n as f64), value);
            }
        }
    }
}
