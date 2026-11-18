use ordered_float::FloatCore;
use tdigest_rs::tdigest::{precision::FloatLike, TDigest};

#[test]
fn rounded_coalescing_at_maximum_mass_does_not_panic() {
    // The exact rounded sum is f64::MAX; naive accumulation overflows on term 11.
    let weight = f64::MAX / 11.0;
    let d = TDigest::<f64>::from_weighted_unsorted(&[0.0; 11], &[weight; 11], 1000).unwrap();
    assert_eq!(d.count(), f64::MAX);
    assert_eq!(d.cdf(&[0.0]), vec![0.5]);
    assert_eq!(d.quantile(0.5), 0.0);
    let compact = TDigest::<f32>::from_weighted_unsorted(&[0.0; 11], &[weight; 11], 1000).unwrap();
    assert_eq!(compact.count(), f64::MAX);
    assert_eq!(compact.cdf(&[0.0]), vec![0.5]);
}

#[test]
fn trimmed_mean_avoids_overflow_of_representable_average() {
    let d =
        TDigest::<f64>::from_unsorted(&[-f64::MAX, -f64::MAX, f64::MAX, f64::MAX], 1000).unwrap();
    assert_eq!(d.trimmed_mean(0.5, 1.0), f64::MAX);
    assert_eq!(d.trimmed_mean(0.0, 0.5), -f64::MAX);
    assert_eq!(d.trimmed_mean(0.0, 1.0), 0.0);
}

#[test]
fn creation_merge_and_scaling_fail_without_changing_valid_state() {
    assert!(TDigest::<f64>::from_unsorted(&[f64::MAX, f64::MAX], 1000).is_err());
    assert!(TDigest::<f64>::from_weighted_unsorted(&[f64::MAX], &[2.0], 1000).is_err());
    let mut d = TDigest::<f64>::from_unsorted(&[f64::MAX], 1000).unwrap();
    let before = d.to_bytes();
    assert!(d.add(f64::MAX).is_err());
    assert!(d.merge(&d.clone()).is_err());
    assert!(d.scale_weights(2.0).is_err());
    assert!(d.scale_values(2.0).is_err());
    assert_eq!(d.to_bytes(), before);
}

#[test]
fn extreme_value_queries_are_finite_bounded_and_read_only() {
    let d = TDigest::<f64>::from_unsorted(&[-f64::MAX, f64::MAX], 1000).unwrap();
    let before = d.to_bytes();
    for q in [0.0, 0.1, 0.5, 0.9, 1.0] {
        let value = d.quantile(q);
        assert!(value.is_finite());
        assert!((-f64::MAX..=f64::MAX).contains(&value));
    }
    let cdf = d.cdf(&[-f64::MAX, -1.0, 0.0, 1.0, f64::MAX]);
    assert!(cdf.iter().all(|p| p.is_finite() && (0.0..=1.0).contains(p)));
    assert!(cdf.windows(2).all(|p| p[0] <= p[1]));
    assert_eq!(d.to_bytes(), before);
}

fn extreme_mass_grid<F: FloatLike + FloatCore>() {
    let masses = [f64::MIN_POSITIVE, 0.1, 1.0, 1e100, f64::MAX / 2.0, f64::MAX];
    for left in masses {
        for right in masses {
            for size in [1, 2, 1000] {
                let result = TDigest::<F>::from_weighted_unsorted(
                    &[F::from_f64(-0.5), F::from_f64(0.5)],
                    &[left, right],
                    size,
                );
                if let Ok(d) = result {
                    let before = d.to_bytes();
                    let probabilities = d.cdf(&[-0.5, -0.25, 0.0, 0.25, 0.5]);
                    assert!(probabilities
                        .iter()
                        .all(|p| p.is_finite() && (0.0..=1.0).contains(p)));
                    assert!(probabilities.windows(2).all(|p| p[0] <= p[1]));
                    for q in [0.0, f64::MIN_POSITIVE, 0.25, 0.5, 0.75, 1.0] {
                        let value = d.quantile(q);
                        assert!(value.is_finite() && (-0.5..=0.5).contains(&value));
                    }
                    assert_eq!(d.to_bytes(), before);
                    let mut receiver = d.clone();
                    let merged = receiver.merge(&d);
                    if merged.is_err() {
                        assert_eq!(receiver.to_bytes(), before);
                    } else {
                        receiver.validate_state().unwrap();
                    }
                }
            }
        }
    }
}

#[test]
fn extreme_mass_queries_and_merges_in_both_precisions() {
    extreme_mass_grid::<f32>();
    extreme_mass_grid::<f64>();
}
