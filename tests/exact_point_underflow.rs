//! A known exact point needs no division of an unrepresentably small moment.
use tdigest_rs::tdigest::frontends::{DigestConfig, DigestPrecision, FrontendDigest};
use tdigest_rs::tdigest::scale::ScaleFamily;
use tdigest_rs::tdigest::singleton_policy::SingletonPolicy;
use tdigest_rs::tdigest::wire::{decode_digest, WireDecodedDigest};
use tdigest_rs::tdigest::TDigest;

fn config() -> DigestConfig {
    DigestConfig {
        max_size: 1000,
        scale: ScaleFamily::K2,
        policy: SingletonPolicy::Use,
        delta: None,
    }
}

#[test]
fn smallest_point_retains_mass_queries_and_full_state() {
    let tiny = f64::from_bits(1);
    for weight in [0.1, 0.6, 1.0] {
        let d = TDigest::<f64>::from_weighted_unsorted(&[tiny], &[weight], 1000).unwrap();
        assert_eq!(d.count(), weight);
        assert_eq!(d.sum(), tiny * weight);
        assert_eq!(d.mean(), tiny);
        assert_eq!(d.median(), tiny);
        assert_eq!(d.cdf(&[0.0, tiny, 2.0 * tiny]), vec![0.0, 0.5, 1.0]);
        let restored = TDigest::<f64>::from_bytes(&d.to_bytes()).unwrap();
        assert_eq!(restored.to_bytes(), d.to_bytes());
        assert_eq!(restored.mean(), tiny);
        assert!(restored.centroids()[0].is_atomic());
    }
}

#[test]
fn exact_point_transforms_and_merges_recover_representable_moments() {
    let tiny = f64::from_bits(1);
    let d = TDigest::<f64>::from_weighted_unsorted(&[tiny], &[0.1], 1000).unwrap();
    let before = d.to_bytes();
    let mut reweighted = d.clone();
    reweighted.scale_weights(10.0).unwrap();
    assert_eq!(reweighted.sum(), tiny);
    assert_eq!(reweighted.count(), 1.0);
    let mut rescaled = d.clone();
    rescaled.scale_values(1e300).unwrap();
    assert_eq!(rescaled.sum(), rescaled.min() * rescaled.count());
    assert_eq!(rescaled.median(), tiny * 1e300);
    let merged = TDigest::merge_digests(vec![d.clone(); 10]).unwrap();
    assert_eq!(merged.count(), 1.0);
    assert_eq!(merged.sum(), tiny);
    assert_eq!(merged.median(), tiny);
    let mut added = d.clone();
    added.add_weighted_many(&[tiny, tiny], &[0.1, 0.8]).unwrap();
    assert_eq!(added.sum(), tiny);
    assert_eq!(added.count(), 1.0);
    assert_eq!(d.to_bytes(), before);
}

#[test]
fn mixed_observation_product_underflow_stays_an_atomic_error() {
    let tiny = f64::from_bits(1);
    assert!(TDigest::<f64>::from_weighted_unsorted(&[tiny, 1.0], &[0.1, 1.0], 1000).is_err());
    let mut d = TDigest::<f64>::from_unsorted(&[1.0], 1000).unwrap();
    let before = d.to_bytes();
    assert!(d.add_weighted(tiny, 0.1).is_err());
    assert_eq!(d.to_bytes(), before);
}

#[test]
fn f32_rounding_preserves_f64_statistics_and_rejects_lost_values() {
    let source = TDigest::<f64>::from_unsorted(&[0.1], 1000).unwrap();
    let mut compact = source.cast_precision::<f32>().unwrap();
    assert_eq!(compact.sum(), source.sum());
    compact.scale_values(0.1).unwrap();
    let WireDecodedDigest::F32(restored) = decode_digest(&compact.to_bytes()).unwrap() else {
        panic!("wrong precision");
    };
    assert_eq!(restored.sum(), compact.sum());
    for value in [f64::from_bits(1), f64::MAX, f64::NAN] {
        assert!(FrontendDigest::from_values(vec![value], config(), DigestPrecision::F32).is_err());
        let mut d = FrontendDigest::from_values(vec![1.0], config(), DigestPrecision::F32).unwrap();
        let before = d.to_bytes();
        assert!(d.add_values_f64(vec![2.0, value]).is_err());
        assert_eq!(d.to_bytes(), before);
        assert!(d
            .add_weighted_f64(vec![2.0, value], vec![1.0, 1.0])
            .is_err());
        assert_eq!(d.to_bytes(), before);
    }
}

#[test]
fn exact_point_transforms_may_round_the_moment_to_zero_without_losing_mass() {
    let mut weighted = TDigest::<f64>::from_unsorted(&[1e-300], 1000).unwrap();
    weighted.scale_weights(1e-30).unwrap();
    assert_eq!(weighted.count(), 1e-30);
    assert_eq!(weighted.sum(), 0.0);
    assert_eq!(weighted.mean(), 1e-300);
    assert_eq!(weighted.median(), 1e-300);
    let mut valued = TDigest::<f64>::from_weighted_unsorted(&[1.0], &[1e-300], 1000).unwrap();
    valued.scale_values(1e-30).unwrap();
    assert_eq!(valued.count(), 1e-300);
    assert_eq!(valued.sum(), 0.0);
    assert_eq!(valued.mean(), 1e-30);
    assert_eq!(valued.cdf(&[1e-30]), vec![0.5]);
    let restored = TDigest::<f64>::from_bytes(&valued.to_bytes()).unwrap();
    assert_eq!(restored.mean(), 1e-30);
}
