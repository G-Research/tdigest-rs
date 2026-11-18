//! Finite compressed cluster means must remain in their input intervals.

use ordered_float::FloatCore;
use tdigest_rs::tdigest::precision::FloatLike;
use tdigest_rs::tdigest::TDigest;

fn compressed_adjacent_means<F: FloatLike + FloatCore>() {
    let points: [(f64, f64); 20] = [
        (-1.0000000000000009, 7e+100),
        (-1.0000000000000016, 1e+100),
        (1.0000000000000018, 2e+100),
        (-1.0000000000000022, 4e+100),
        (1.0, 5e+100),
        (-1.0000000000000004, 7e+100),
        (1.0000000000000007, 8e+100),
        (-1.000000000000001, 1e+100),
        (1.0000000000000013, 2e+100),
        (1.0000000000000016, 3.0000000000000002e+100),
        (-1.0000000000000018, 4e+100),
        (1.000000000000002, 5e+100),
        (1.0000000000000022, 6.0000000000000005e+100),
        (-1.0, 7e+100),
        (1.0000000000000002, 8e+100),
        (1.0000000000000004, 9e+100),
        (-1.0000000000000007, 1e+100),
        (1.0000000000000009, 2e+100),
        (1.000000000000001, 3.0000000000000002e+100),
        (-1.0000000000000013, 4e+100),
    ];
    for scale in [1e-100, 1e-50, 0.1, 1.0, 10.0] {
        let step = if F::from_f64(1.0 + f64::EPSILON).to_f64() == 1.0 {
            f32::EPSILON as f64
        } else {
            f64::EPSILON
        };
        let values: Vec<F> = points
            .iter()
            .map(|(x, _)| {
                let offset = ((x.abs() - 1.0) / f64::EPSILON).round();
                F::from_f64(x.signum() * (1.0 + offset * step))
            })
            .collect();
        let weights: Vec<f64> = points.iter().map(|(_, w)| w * scale).collect();
        let td = TDigest::<F>::from_weighted_unsorted(&values, &weights, 19).unwrap();
        let cents = td.centroids();
        assert!(cents
            .windows(2)
            .all(|pair| pair[0].mean_f64() < pair[1].mean_f64()));
        let total: f64 = weights.iter().sum();
        let actual: f64 = cents.iter().map(|c| c.weight_f64()).sum();
        assert!((actual - total).abs() <= total * 1e-14);
        let min = values
            .iter()
            .map(|x| FloatLike::to_f64(*x))
            .fold(f64::INFINITY, f64::min);
        let max = values
            .iter()
            .map(|x| FloatLike::to_f64(*x))
            .fold(f64::NEG_INFINITY, f64::max);
        assert_eq!(td.min(), min);
        assert_eq!(td.max(), max);
        let mut probes: Vec<f64> = cents.iter().map(|c| c.mean_f64()).collect();
        probes.extend([min, max]);
        probes.sort_by(f64::total_cmp);
        let cdf = td.cdf(&probes);
        assert!(cdf.iter().all(|x| x.is_finite() && *x >= 0.0 && *x <= 1.0));
        assert!(cdf.windows(2).all(|pair| pair[0] <= pair[1]));
        let quantiles: Vec<f64> = (0..=256).map(|i| td.quantile(i as f64 / 256.0)).collect();
        assert!(quantiles
            .iter()
            .all(|x| x.is_finite() && *x >= min && *x <= max));
        assert!(quantiles.windows(2).all(|pair| pair[0] <= pair[1]));
    }
}

#[test]
fn compressed_cluster_rounding_cannot_collide_with_next_mean() {
    compressed_adjacent_means::<f32>();
    compressed_adjacent_means::<f64>();
}
