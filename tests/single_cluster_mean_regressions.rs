//! One-centroid compressor clusters must be exact pass-throughs.

use tdigest_rs::tdigest::TDigest;

#[test]
fn fractional_atomic_mean_and_cdf_stay_at_the_observed_value() {
    for scale in [1e-300, 1e-46, 1.0, 1e39, 1e300] {
        let td = TDigest::<f64>::from_weighted_unsorted(
            &[-4.0, -3.0, -2.0],
            &[scale, 0.1 * scale, scale],
            1000,
        )
        .unwrap();
        assert_eq!(td.centroids()[1].mean_f64().to_bits(), (-3.0_f64).to_bits());
        assert!(td.centroids()[1].is_atomic());
        assert!((td.cdf(&[-3.0])[0] - 0.5).abs() < 1e-14);
    }
}

#[test]
fn later_duplicate_sample_coalesces_after_fractional_atomic_merge() {
    let pairs = [
        (-3.0, 0.1),
        (-4.0, 0.30000000000000004),
        (-1.0, 0.5),
        (-3.0, 0.1),
    ];
    let shards: Vec<_> = pairs
        .iter()
        .map(|&(value, weight)| {
            TDigest::<f64>::from_weighted_unsorted(&[value], &[weight], 1000).unwrap()
        })
        .collect();
    let mut sequential = shards[0].clone();
    for shard in &shards[1..] {
        sequential.merge(shard).unwrap();
    }
    let kway = TDigest::merge_digests(shards).unwrap();
    for td in [sequential, kway] {
        assert_eq!(
            td.centroids()
                .iter()
                .map(|c| c.mean_f64())
                .collect::<Vec<_>>(),
            vec![-4.0, -3.0, -1.0]
        );
        assert_eq!(
            td.centroids()
                .iter()
                .map(|c| c.weight_f64())
                .collect::<Vec<_>>(),
            vec![0.30000000000000004, 0.2, 0.5]
        );
        assert!(td.centroids().iter().all(|c| c.is_atomic()));
        assert!((td.cdf(&[-3.0])[0] - 0.4).abs() < 1e-14);
    }
}
