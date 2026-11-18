//! Regression controls for mass, first moments and atomic quantile boundaries.
use tdigest_rs::tdigest::{singleton_policy::SingletonPolicy, TDigest};

macro_rules! precision_controls {
    ($module:ident, $float:ty) => {
        mod $module {
            use super::*;

            #[test]
            fn tiny_use_capacity_keeps_mass_and_tail_contribution() {
                for cap in [1, 2] {
                    let d = TDigest::<$float>::builder()
                        .max_size(cap)
                        .singleton_policy(SingletonPolicy::Use)
                        .build()
                        .unwrap()
                        .merge_unsorted(vec![0., 1., 2., 100.])
                        .unwrap();
                    assert!(d.centroids_ref().len() <= cap);
                    assert_eq!(d.count(), 4.);
                    assert_eq!(
                        d.centroids_ref().iter().map(|c| c.weight()).sum::<f64>(),
                        4.
                    );
                    let moment: f64 = d
                        .centroids_ref()
                        .iter()
                        .map(|c| c.mean_f64() * c.weight())
                        .sum();
                    assert!((moment - 103.).abs() < 1e-5, "cap={cap}: {moment}");
                    assert_eq!(d.quantile(0.), 0.);
                    assert_eq!(d.quantile(1.), 100.);
                }
            }

            #[test]
            fn rounded_bucket_boundary_keeps_tiny_high_value_tail() {
                let d = TDigest::<$float>::builder()
                    .max_size(2)
                    .singleton_policy(SingletonPolicy::Off)
                    .build()
                    .unwrap()
                    .merge_weighted_unsorted(
                        &[0., 1., 2f64.powi(60) as $float],
                        &[0.5, 0.5, 2f64.powi(-60)],
                    )
                    .unwrap();
                // The tail's weight rounds away in total mass, but its moment is 1.
                let moment: f64 = d
                    .centroids_ref()
                    .iter()
                    .map(|c| c.mean_f64() * c.weight())
                    .sum();
                assert_eq!(moment, 1.5);
                assert!(d.centroids_ref().len() <= 2);
            }

            #[test]
            fn cancellation_preserves_digest_and_compressed_centroid_moments() {
                for cap in [1, 128] {
                    let d = TDigest::<$float>::builder()
                        .max_size(cap)
                        .singleton_policy(SingletonPolicy::Off)
                        .build()
                        .unwrap()
                        .merge_unsorted(vec![-1e16 as $float, 1., 1e16 as $float])
                        .unwrap();
                    assert_eq!(d.sum(), 1.);
                    assert_eq!(d.mean(), 1. / 3.);
                    if cap == 1 {
                        let c = &d.centroids_ref()[0];
                        assert!((c.mean_f64() - 1. / 3.).abs() < 1e-7);
                        assert_eq!(c.weight(), 3.);
                    }
                }
            }

            #[test]
            fn fractional_symmetric_atomic_boundary_uses_complete_prefix() {
                for weights in [
                    [0.1, 0.2, 0.3, 0.3, 0.2, 0.1],
                    [0.3, 0.2, 0.1, 0.1, 0.2, 0.3],
                    [0.1; 6],
                ] {
                    let d = TDigest::<$float>::builder()
                        .max_size(128)
                        .singleton_policy(SingletonPolicy::Off)
                        .build()
                        .unwrap()
                        .merge_weighted_unsorted(&[0., 1., 2., 3., 4., 5.], &weights)
                        .unwrap();
                    assert_eq!(d.median(), 2.5, "weights={weights:?}");
                    // Probe well beyond rounding of q * total weight on either side.
                    assert_eq!(d.quantile(0.5 - 1e-12), 2.);
                    assert_eq!(d.quantile(0.5 + 1e-12), 3.);
                }
            }
        }
    };
}
precision_controls!(f32_controls, f32);
precision_controls!(f64_controls, f64);
