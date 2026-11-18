use tdigest_rs::tdigest::centroids::Centroid;
use tdigest_rs::tdigest::scale::ScaleFamily;
use tdigest_rs::tdigest::singleton_policy::SingletonPolicy;
use tdigest_rs::tdigest::tdigest::DigestStats;
use tdigest_rs::tdigest::TDigest;

#[test]
fn public_builder_never_exposes_invalid_configuration() {
    assert!(TDigest::<f64>::builder().max_size(0).build().is_err());
    assert!(TDigest::<f64>::from_unsorted(&[], 0).is_err());
    for delta in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(TDigest::<f64>::builder().delta(delta).build().is_err());
    }
    let valid = TDigest::<f64>::builder().delta(50.0).build().unwrap();
    assert_eq!(valid.scale(), ScaleFamily::K2Norm);
    assert_eq!(valid.singleton_policy(), SingletonPolicy::Off);
    assert!(TDigest::<f64>::builder()
        .scale(ScaleFamily::K2)
        .delta(50.0)
        .build()
        .is_err());
    assert!(TDigest::<f64>::builder()
        .delta(50.0)
        .scale(ScaleFamily::K2)
        .build()
        .is_err());
    assert!(TDigest::<f64>::builder()
        .singleton_policy(SingletonPolicy::Use)
        .delta(50.0)
        .build()
        .is_err());
}

#[test]
fn public_seed_rejects_corrupt_mass_sum_order_and_mean() {
    let stats = DigestStats {
        data_sum: 2.0,
        total_weight: 1.0,
        data_min: 2.0,
        data_max: 2.0,
    };
    for bad in [
        DigestStats {
            total_weight: 10.0,
            ..stats
        },
        DigestStats {
            data_sum: 1e100,
            ..stats
        },
        DigestStats {
            data_min: f64::NAN,
            ..stats
        },
    ] {
        assert!(TDigest::<f64>::builder()
            .with_centroids_and_stats(vec![Centroid::new_atomic_unit_f64(2.0)], bad)
            .build()
            .is_err());
    }
    assert!(TDigest::<f64>::builder()
        .with_centroids_and_stats(vec![Centroid::new_atomic_unit_f64(f64::NAN)], stats)
        .build()
        .is_err());
    assert!(TDigest::<f64>::builder()
        .with_centroids_and_stats(
            vec![
                Centroid::new_atomic_unit_f64(2.0),
                Centroid::new_atomic_unit_f64(2.0)
            ],
            DigestStats {
                total_weight: 2.0,
                data_sum: 4.0,
                ..stats
            }
        )
        .build()
        .is_err());
}

#[test]
fn complete_seed_preserves_real_bounds_and_owned_state() {
    let d = TDigest::<f64>::builder()
        .max_size(200)
        .with_centroids_and_stats(
            vec![Centroid::new_mixed_f64(2.0, 3.0)],
            DigestStats {
                data_sum: 6.0,
                total_weight: 3.0,
                data_min: 0.0,
                data_max: 5.0,
            },
        )
        .build()
        .unwrap();
    assert_eq!((d.count(), d.sum(), d.min(), d.max()), (3.0, 6.0, 0.0, 5.0));
    assert_eq!(
        TDigest::<f64>::from_bytes(&d.to_bytes())
            .unwrap()
            .to_bytes(),
        d.to_bytes()
    );
}

#[test]
fn oversize_seed_compression_uses_requested_capacity() {
    let d = TDigest::<f64>::new(
        vec![0., 1., 2., 3.]
            .into_iter()
            .map(Centroid::new_atomic_unit_f64)
            .collect(),
        6.0,
        4.0,
        3.0,
        0.0,
        2,
    )
    .unwrap();
    assert_eq!(d.max_size(), 2);
    assert!(d.centroids().len() <= 2);
    assert_eq!((d.count(), d.sum(), d.min(), d.max()), (4.0, 6.0, 0.0, 3.0));
}

#[test]
fn seed_and_conversion_cannot_silently_underflow_observed_extrema() {
    let stats = DigestStats {
        data_sum: 1.0,
        total_weight: 1.0,
        data_min: f64::from_bits(1),
        data_max: 2.0,
    };
    assert!(TDigest::<f32>::builder()
        .with_centroids_and_stats(vec![Centroid::new_mixed_f64(1.0, 1.0)], stats)
        .build()
        .is_err());
    let full = TDigest::<f64>::builder()
        .with_centroids_and_stats(vec![Centroid::new_mixed_f64(1.0, 1.0)], stats)
        .build()
        .unwrap();
    let before = full.to_bytes();
    assert!(full.cast_precision::<f32>().is_err());
    assert_eq!(full.to_bytes(), before);
}
