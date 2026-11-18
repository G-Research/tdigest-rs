//! An oversized delta must retain information when K2Norm's normalizer is nonpositive.

use tdigest_rs::tdigest::singleton_policy::SingletonPolicy;
use tdigest_rs::tdigest::wire::{decode_digest, WireDecodedDigest};
use tdigest_rs::tdigest::{ScaleFamily, TDigest};

fn computed_domain_boundaries() -> [f64; 3] {
    // libm rounding can put several adjacent deltas on the zero plateau.
    // Locate all three computed domains rather than assume exp/ln invert
    // exactly or that one next_up/next_down step crosses the boundary.
    let normalizer = |delta: f64| (2.0 / delta).ln().mul_add(4.0, 24.0);
    assert!(normalizer(800.0) > 0.0);
    assert!(normalizer(810.0) < 0.0);
    let mut lower = 2.0 * 6.0_f64.exp();
    let mut upper = lower;
    let mut found = [None; 3];
    for _ in 0..=64 {
        for delta in [lower, upper] {
            let value = normalizer(delta);
            let slot = if value > 0.0 {
                0
            } else if value == 0.0 {
                1
            } else {
                2
            };
            found[slot].get_or_insert(delta);
        }
        if found.iter().all(Option::is_some) {
            break;
        }
        lower = lower.next_down();
        upper = upper.next_up();
    }
    let result = found.map(|value| value.expect("computed K2Norm domain missing near 2*exp(6)"));
    assert!(normalizer(result[0]) > 0.0);
    assert_eq!(normalizer(result[1]), 0.0);
    assert!(normalizer(result[2]) < 0.0);
    result
}

macro_rules! domain_regression {
    ($name:ident, $float:ty, $variant:ident) => {
        #[test]
        fn $name() {
            let [positive, zero, negative] = computed_domain_boundaries();
            for delta in [
                300.0,
                600.0,
                800.0,
                positive,
                zero.next_down(),
                zero,
                zero.next_up(),
                negative,
                810.0,
                1200.0,
                2400.0,
                1e6,
                f64::MAX,
            ] {
                let empty = || {
                    TDigest::<$float>::builder()
                        .delta(delta)
                        .scale(ScaleFamily::K2Norm)
                        .singleton_policy(SingletonPolicy::Off)
                        .build()
                        .unwrap()
                };
                let check = |td: &TDigest<$float>| {
                    assert_eq!(td.centroids().len(), 2, "delta={delta}");
                    let points: Vec<_> = td
                        .centroids()
                        .iter()
                        .map(|c| (c.mean_f64(), c.weight_f64(), c.is_atomic()))
                        .collect();
                    assert_eq!(points, vec![(0.0, 2.0, true), (100.0, 1.0, true)]);
                    assert_eq!(td.count(), 3.0);
                    assert_eq!(td.sum(), 100.0);
                    assert_eq!((td.min(), td.max()), (0.0, 100.0));
                    assert_eq!(td.quantile(0.1), 0.0);
                    assert_eq!(td.quantile(0.5), 0.0);
                    assert_eq!(td.quantile(0.9), 100.0);
                    assert_eq!(
                        td.cdf(&[0.0, 25.0, 100.0]),
                        vec![1.0 / 3.0, 2.0 / 3.0, 5.0 / 6.0]
                    );
                };
                let direct = empty().merge_unsorted(vec![0.0, 0.0, 100.0]).unwrap();
                check(&direct);
                let weighted = empty()
                    .merge_weighted_unsorted(&[0.0, 100.0], &[2.0, 1.0])
                    .unwrap();
                check(&weighted);
                let mut updated = empty();
                for value in [0.0, 100.0, 0.0] {
                    updated.add(value).unwrap();
                }
                check(&updated);
                let mut merged = empty().merge_unsorted(vec![0.0, 0.0]).unwrap();
                merged
                    .merge(&empty().merge_unsorted(vec![100.0]).unwrap())
                    .unwrap();
                check(&merged);
                let kway = TDigest::merge_digests(vec![
                    empty().merge_unsorted(vec![0.0]).unwrap(),
                    empty().merge_unsorted(vec![100.0]).unwrap(),
                    empty().merge_unsorted(vec![0.0]).unwrap(),
                ])
                .unwrap();
                check(&kway);
                let WireDecodedDigest::$variant(restored) =
                    decode_digest(&direct.to_bytes()).unwrap()
                else {
                    panic!("wire precision changed");
                };
                check(&restored);

                // Three distinct means have a positive normalizer at 1200,
                // then cross into the guarded regime at 2400 and above.
                let three = empty().merge_unsorted(vec![0.0, 50.0, 100.0]).unwrap();
                assert_eq!(three.centroids().len(), 3);
                assert!(three.centroids().iter().all(|c| c.is_atomic()));
                assert_eq!(three.median(), 50.0);
            }
        }
    };
}

domain_regression!(f32_large_delta_preserves_piles_across_mutations, f32, F32);
domain_regression!(f64_large_delta_preserves_piles_across_mutations, f64, F64);

#[test]
fn max_size_k2norm_retains_tiny_interiors_and_keeps_normal_caps() {
    for policy in [
        SingletonPolicy::Use,
        SingletonPolicy::UseWithProtectedEdges(1),
    ] {
        let td = TDigest::<f64>::builder()
            .max_size(1200)
            .scale(ScaleFamily::K2Norm)
            .singleton_policy(policy)
            .build()
            .unwrap()
            .merge_unsorted(vec![-100.0, 0.0, 100.0, 200.0])
            .unwrap();
        assert_eq!(td.centroids().len(), 4);
        assert!(td.centroids().iter().all(|c| c.is_atomic()));
        assert_eq!(td.count(), 4.0);
        assert_eq!(
            td.cdf(&[-100.0, 0.0, 100.0, 200.0]),
            vec![0.125, 0.375, 0.625, 0.875]
        );
    }
    let capped = TDigest::<f64>::builder()
        .max_size(10)
        .scale(ScaleFamily::K2Norm)
        .singleton_policy(SingletonPolicy::Use)
        .build()
        .unwrap()
        .merge_unsorted((0..30).map(|x| x as f64).collect())
        .unwrap();
    assert!(capped.centroids().len() <= 10);
    assert_eq!(capped.count(), 30.0);
    assert_eq!(
        capped
            .centroids()
            .iter()
            .map(|c| c.weight_f64())
            .sum::<f64>(),
        30.0
    );
    assert_eq!((capped.min(), capped.max()), (0.0, 29.0));
}

#[test]
fn tiny_positive_delta_keeps_its_existing_aggressive_compression() {
    let delta = f64::from_bits(1);
    assert!((2.0 / delta).is_infinite());
    let td = TDigest::<f64>::builder()
        .delta(delta)
        .scale(ScaleFamily::K2Norm)
        .singleton_policy(SingletonPolicy::Off)
        .build()
        .unwrap()
        .merge_unsorted(vec![0.0, 100.0])
        .unwrap();
    assert_eq!(td.centroids().len(), 1);
    assert_eq!(td.centroids()[0].mean_f64(), 50.0);
    assert_eq!(td.centroids()[0].weight_f64(), 2.0);
    assert!(!td.centroids()[0].is_atomic());
}
