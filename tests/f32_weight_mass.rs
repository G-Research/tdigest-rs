//! Mean precision must not silently discard distribution mass.

use tdigest_rs::tdigest::centroids::Centroid;
use tdigest_rs::tdigest::singleton_policy::SingletonPolicy;
use tdigest_rs::tdigest::wire::{
    decode_digest, encode_digest_with_version, WireDecodedDigest, WireVersion,
};
use tdigest_rs::tdigest::TDigest;

fn weights(td: &TDigest<f32>) -> Vec<f64> {
    td.centroids().iter().map(|c| c.weight_f64()).collect()
}

fn restored_f32(bytes: &[u8]) -> TDigest<f32> {
    let WireDecodedDigest::F32(td) = decode_digest(bytes).unwrap() else {
        panic!("mean precision changed");
    };
    td
}

#[test]
fn one_left_observation_moves_balanced_large_pile_median_left() {
    let mass = 16_777_216.0;
    let mut td = TDigest::<f32>::from_weighted_unsorted(&[0.0, 10.0], &[mass, mass], 100).unwrap();
    assert_eq!(td.median(), 5.0);
    td.add(0.0).unwrap();
    assert_eq!(weights(&td), vec![mass + 1.0, mass]);
    assert_eq!(td.count(), 2.0 * mass + 1.0);
    assert_eq!(td.median(), 0.0);
    assert_eq!(td.cdf(&[5.0])[0], (mass + 1.0) / (2.0 * mass + 1.0));
    td.add(10.0).unwrap();
    assert_eq!(td.median(), 5.0);
}

#[test]
fn repeated_unit_add_weighted_add_and_merge_retain_mass() {
    let initial = 1_000_000_000.0;
    let base = TDigest::<f32>::from_weighted_unsorted(&[0.0], &[initial], 100).unwrap();
    let unit = TDigest::<f32>::from_unsorted(&[0.0], 100).unwrap();
    let mut added = base.clone();
    let mut weighted = base.clone();
    let mut merged = base.clone();
    for _ in 0..1001 {
        added.add(0.0).unwrap();
        weighted.add_weighted(0.0, 1.0).unwrap();
        merged.merge(&unit).unwrap();
    }
    for td in [&added, &weighted, &merged] {
        assert_eq!(weights(td), vec![initial + 1001.0]);
        assert_eq!(td.count(), initial + 1001.0);
    }
}

#[test]
fn identical_global_statistics_do_not_erase_where_new_mass_arrived() {
    let initial = 1_000_000_000.0;
    let base =
        TDigest::<f32>::from_weighted_unsorted(&[-1.0, 0.0, 1.0], &[initial; 3], 100).unwrap();
    let mut outer = base.clone();
    let mut middle = base;
    for _ in 0..500 {
        outer.add(-1.0).unwrap();
        outer.add(1.0).unwrap();
        middle.add(0.0).unwrap();
        middle.add(0.0).unwrap();
    }
    outer.add(0.0).unwrap();
    middle.add(0.0).unwrap();
    assert_eq!(outer.count(), 3.0 * initial + 1001.0);
    assert_eq!(outer.count(), middle.count());
    assert_eq!(outer.sum(), middle.sum());
    assert_eq!(
        weights(&outer),
        vec![initial + 500.0, initial + 1.0, initial + 500.0]
    );
    assert_eq!(weights(&middle), vec![initial, initial + 1001.0, initial]);
    assert!(outer.cdf(&[-0.5])[0] > middle.cdf(&[-0.5])[0]);
    assert_ne!(outer.to_bytes(), middle.to_bytes());
}

#[test]
fn compressed_large_mass_and_modest_fan_in_stay_conservative() {
    let values: Vec<f32> = (0..40).map(|x| x as f32).collect();
    let mut td = TDigest::<f32>::builder()
        .max_size(2)
        .singleton_policy(SingletonPolicy::Off)
        .build()
        .unwrap()
        .merge_weighted_unsorted(&values, &[1_000_000_000.0; 40])
        .unwrap();
    for _ in 0..100 {
        td.add(10.0).unwrap();
    }
    assert_eq!(td.count(), 40_000_000_100.0);
    assert_eq!(weights(&td).iter().sum::<f64>(), td.count());
    assert_eq!(td.quantile(0.0), 0.0);
    assert_eq!(td.quantile(1.0), 39.0);
    let shards: Vec<_> = (0..40).map(|_| td.clone()).collect();
    let merged = TDigest::merge_digests(shards).unwrap();
    assert_eq!(merged.count(), 40.0 * td.count());
    assert_eq!(weights(&merged).iter().sum::<f64>(), merged.count());
    assert!(merged
        .cdf(&[-1.0, 0.0, 10.0, 39.0, 40.0])
        .iter()
        .all(|p| (0.0..=1.0).contains(p)));
}

#[test]
fn mean_precision_casts_preserve_weight_bits_and_scaling() {
    let source =
        TDigest::<f64>::from_weighted_unsorted(&[0.0, 1.0, 2.0], &[16_777_217.0, 0.1, 0.25], 100)
            .unwrap();
    let compact = source.cast_precision::<f32>().unwrap();
    for (actual, expected) in compact.centroids().iter().zip(source.centroids()) {
        assert_eq!(
            actual.weight_f64().to_bits(),
            expected.weight_f64().to_bits()
        );
    }
    assert_eq!(compact.count(), source.count());
    let full = compact.cast_precision::<f64>().unwrap();
    assert_eq!(full.centroids(), source.centroids());
    let mut scaled = compact.clone();
    scaled.scale_weights(0.25).unwrap();
    for (actual, expected) in scaled.centroids().iter().zip(compact.centroids()) {
        assert_eq!(actual.weight_f64(), expected.weight_f64() * 0.25);
    }
    assert_eq!(scaled.median(), compact.median());
    assert_eq!(
        scaled.cdf(&[0.0, 0.5, 1.0, 2.0]),
        compact.cdf(&[0.0, 0.5, 1.0, 2.0])
    );
}

#[test]
fn tiny_and_large_finite_mass_no_longer_depend_on_mean_precision() {
    for mass in [1e-46, 0.1, 16_777_217.0, 1e39] {
        let td = TDigest::<f32>::from_weighted_unsorted(&[0.0, 1.0], &[mass, mass], 100).unwrap();
        assert_eq!(weights(&td), vec![mass, mass]);
        assert_eq!(td.count(), 2.0 * mass);
        assert_eq!(td.median(), 0.5);
        assert_eq!(td.cdf(&[0.5]), vec![0.5]);
        let mut restored = restored_f32(&td.to_bytes());
        assert_eq!(restored.centroids(), td.centroids());
        restored.add_weighted(0.0, mass).unwrap();
        assert_eq!(weights(&restored), vec![2.0 * mass, mass]);
        assert_eq!(restored.median(), 0.0);
    }
}

#[test]
fn wide_f32_weights_are_flagged_and_old_exports_fail_explicitly() {
    let mut td =
        TDigest::<f32>::from_weighted_unsorted(&[0.0, 10.0], &[16_777_217.0, 16_777_216.0], 100)
            .unwrap();
    let bytes = td.to_bytes();
    assert_eq!(bytes[5] & 8, 8);
    assert_eq!(bytes[6], 64); // Capability flag adds no header or payload bytes.
    assert_eq!(bytes.len(), 64 + 2 * 13);
    let mut restored = restored_f32(&bytes);
    assert_eq!(restored.to_bytes(), bytes);
    restored.add(10.0).unwrap();
    td.add(10.0).unwrap();
    assert_eq!(restored.centroids(), td.centroids());
    assert_eq!(restored.median(), 5.0);
    let wide = restored_f32(&bytes);
    for version in [WireVersion::V1, WireVersion::V2] {
        let error = version.validate_digest(&wide).unwrap_err().to_string();
        assert!(error.contains("weights requiring f64 precision"));
        assert!(version
            .validate_digest(&wide.cast_precision::<f64>().unwrap())
            .is_ok());
    }
    let mut corrupt = bytes.clone();
    corrupt[5] &= !8; // Capability flag is covered by the CRC.
    assert!(decode_digest(&corrupt).is_err());
    let mut invalid = wide.cast_precision::<f64>().unwrap().to_bytes();
    invalid[5] = (invalid[5] & !1) | 8;
    assert!(decode_digest(&invalid)
        .unwrap_err()
        .to_string()
        .contains("wide-weight flag requires f32"));
}

#[test]
fn old_v2_f32_payloads_promote_weights_without_new_byte_layout() {
    let td = TDigest::<f32>::from_unsorted(&[0.0], 100).unwrap();
    assert_eq!(td.to_bytes()[5] & 8, 0);
    let v2 = encode_digest_with_version(&td, WireVersion::V2);
    for mass in [1e-46_f64, 0.1, 16_777_217.0, 1e39] {
        let mut bytes = v2.clone();
        bytes[16..24].copy_from_slice(&mass.to_le_bytes());
        bytes[60..68].copy_from_slice(&mass.to_le_bytes());
        let restored = restored_f32(&bytes);
        assert_eq!(weights(&restored), vec![mass]);
        assert_eq!(restored.count(), mass);
        assert_eq!(restored.to_bytes()[5] & 8, 8);
    }
}

#[test]
fn exactly_f32_representable_weights_keep_old_format_support() {
    let td =
        TDigest::<f32>::from_weighted_unsorted(&[0.0, 10.0], &[16_777_216.0, 0.25], 100).unwrap();
    assert_eq!(td.to_bytes()[5] & 8, 0);
    let restored = restored_f32(&encode_digest_with_version(&td, WireVersion::V2));
    assert_eq!(restored.centroids(), td.centroids());
    assert_eq!(restored.to_bytes(), td.to_bytes());
}

#[test]
#[cfg(target_pointer_width = "64")]
fn centroid_layout_retains_compact_means_with_explicit_mass_cost() {
    assert_eq!(std::mem::size_of::<Centroid<f32>>(), 16);
    assert_eq!(std::mem::size_of::<Centroid<f64>>(), 24);
}

#[test]
fn wide_weight_capability_composes_with_delta_and_pin_extensions() {
    use tdigest_rs::tdigest::ScaleFamily;
    let delta = TDigest::<f32>::builder()
        .scale(ScaleFamily::K2Norm)
        .singleton_policy(SingletonPolicy::Off)
        .delta(17.0)
        .build()
        .unwrap()
        .merge_weighted_unsorted(&[0.0, 10.0], &[16_777_217.0, 16_777_216.0])
        .unwrap();
    let pinned = TDigest::<f32>::builder()
        .scale(ScaleFamily::K3)
        .singleton_policy(SingletonPolicy::UseWithProtectedEdges(300))
        .build()
        .unwrap()
        .merge_weighted_unsorted(&[0.0, 10.0], &[16_777_217.0, 16_777_216.0])
        .unwrap();
    for (td, extension_flag) in [(delta, 2), (pinned, 4)] {
        let bytes = td.to_bytes();
        assert_eq!(bytes[5], 1 | 8 | extension_flag);
        assert_eq!(bytes[6], 72);
        let restored = restored_f32(&bytes);
        assert_eq!(restored.centroids(), td.centroids());
        assert_eq!(restored.delta(), td.delta());
        assert_eq!(restored.singleton_policy(), td.singleton_policy());
        assert_eq!(restored.to_bytes(), bytes);
    }
}

#[test]
fn adjacent_to_unit_weights_are_never_snapped_to_one() {
    for mass in [
        f64::from_bits(1.0_f64.to_bits() - 1),
        f64::from_bits(1.0_f64.to_bits() + 1),
    ] {
        let td32 = TDigest::<f32>::from_weighted_unsorted(&[0.0], &[mass], 100).unwrap();
        let td64 = TDigest::<f64>::from_weighted_unsorted(&[0.0], &[mass], 100).unwrap();
        assert_eq!(td32.centroids()[0].weight_f64().to_bits(), mass.to_bits());
        assert_eq!(td64.centroids()[0].weight_f64().to_bits(), mass.to_bits());
        assert_eq!(
            restored_f32(&td32.to_bytes()).centroids()[0]
                .weight_f64()
                .to_bits(),
            mass.to_bits()
        );
        for version in [WireVersion::V2, WireVersion::V3] {
            let WireDecodedDigest::F64(restored) =
                decode_digest(&encode_digest_with_version(&td64, version)).unwrap()
            else {
                panic!("changed mean precision");
            };
            assert_eq!(
                restored.centroids()[0].weight_f64().to_bits(),
                mass.to_bits()
            );
        }
        let mut old_v2 = TDigest::<f32>::from_unsorted(&[0.0], 100)
            .unwrap()
            .to_bytes_with_version(WireVersion::V2);
        old_v2[16..24].copy_from_slice(&mass.to_le_bytes());
        old_v2[60..68].copy_from_slice(&mass.to_le_bytes());
        assert_eq!(
            restored_f32(&old_v2).centroids()[0].weight_f64().to_bits(),
            mass.to_bits()
        );
        let mut unit = TDigest::<f32>::from_unsorted(&[0.0], 100).unwrap();
        unit.scale_weights(mass).unwrap();
        assert_eq!(unit.centroids()[0].weight_f64().to_bits(), mass.to_bits());
    }
}

#[test]
fn fractional_atomic_boundary_has_no_phantom_interpolation_span() {
    for mass in [1e-46, 0.1, 1e39] {
        let td = TDigest::<f64>::from_weighted_unsorted(&[0.0, 10.0], &[mass, mass], 100).unwrap();
        assert_eq!(td.quantile(0.5), 5.0);
        assert_eq!(td.quantile(f64::from_bits(0.5_f64.to_bits() - 1)), 0.0);
        assert_eq!(td.quantile(f64::from_bits(0.5_f64.to_bits() + 1)), 10.0);
        assert_eq!(td.cdf(&[5.0]), vec![0.5]);
    }
}

#[test]
fn public_weight_getter_does_not_narrow_f32_mean_mass() {
    let centroid =
        tdigest_rs::tdigest::centroids::Centroid::<f32>::new_atomic_f64(1.0, 16_777_217.0);
    let mass: f64 = centroid.weight();
    assert_eq!(mass, 16_777_217.0);
    assert_eq!(mass, centroid.weight_f64());
}
