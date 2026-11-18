//! Checks on independently supplied state, including the serialization boundaries.
use ordered_float::FloatCore;
use polars::prelude::*;
use tdigest_rs::tdigest::{
    centroids::Centroid,
    codecs::WireOf,
    precision::FloatLike,
    singleton_policy::SingletonPolicy,
    wire::{decode_digest, WireVersion},
    DigestStats, TDigest,
};

fn reject_contradictory_stats<F: FloatLike + FloatCore>() {
    for atomic in [true, false] {
        let cents = [0.0, 10.0]
            .map(|mean| {
                if atomic {
                    Centroid::<F>::new_atomic_unit_f64(mean)
                } else {
                    Centroid::<F>::new_mixed_f64(mean, 1.0)
                }
            })
            .to_vec();
        assert!(
            TDigest::<F>::builder()
                .with_centroids_and_stats(
                    cents,
                    DigestStats {
                        data_sum: 2.0,
                        total_weight: 2.0,
                        data_min: 0.0,
                        data_max: 10.0,
                    }
                )
                .build()
                .is_err(),
            "contradictory sum, atomic={atomic}"
        );
    }
    assert!(
        TDigest::<F>::builder()
            .with_centroids_and_stats(
                vec![Centroid::<F>::new_atomic_unit_f64(2.0)],
                DigestStats {
                    data_sum: 2.0,
                    total_weight: 1.0,
                    data_min: 0.0,
                    data_max: 5.0
                },
            )
            .build()
            .is_err(),
        "atomic state cannot have unobserved extrema"
    );
    assert!(
        TDigest::<F>::builder()
            .singleton_policy(SingletonPolicy::UseWithProtectedEdges(0))
            .build()
            .is_err(),
        "zero edge pins cannot round-trip"
    );
}

#[test]
fn constructor_rejects_inconsistent_moments_bounds_and_policy() {
    reject_contradictory_stats::<f32>();
    reject_contradictory_stats::<f64>();
}

fn wire_rejects_contradictory_stats<F: FloatLike + FloatCore>() {
    let td = TDigest::<F>::from_unsorted(&[F::from_f64(0.0), F::from_f64(10.0)], 100).unwrap();
    for version in [WireVersion::V1, WireVersion::V2, WireVersion::V3] {
        let mut bytes = td.to_bytes_with_version(version);
        let sum_offset = if version == WireVersion::V3 {
            bytes[5] &= !1;
            52
        } else {
            48
        };
        bytes[sum_offset..sum_offset + 8].copy_from_slice(&2.0_f64.to_le_bytes());
        assert!(
            decode_digest(&bytes).is_err(),
            "contradictory sum in {version:?}"
        );
        let original = td.to_bytes_with_version(version);
        let shift = if version == WireVersion::V3 { 4 } else { 0 };
        for (offset, value) in [(24, f64::NAN), (24, -10.0), (32, -1.0), (48, f64::INFINITY)] {
            let mut corrupt = original.clone();
            if version == WireVersion::V3 {
                corrupt[5] &= !1;
            }
            corrupt[offset + shift..offset + shift + 8].copy_from_slice(&value.to_le_bytes());
            assert!(
                decode_digest(&corrupt).is_err(),
                "bad field at {offset} in {version:?}"
            );
        }
        let mut zero_capacity = original.clone();
        if version == WireVersion::V3 {
            zero_capacity[5] &= !1;
        }
        zero_capacity[8 + shift..16 + shift].copy_from_slice(&0_u64.to_le_bytes());
        assert!(decode_digest(&zero_capacity).is_err());
        let mut duplicate_mean = original;
        let header_len = if version == WireVersion::V3 {
            duplicate_mean[5] &= !1;
            64
        } else {
            56
        };
        let width = std::mem::size_of::<F>();
        let stride = width + 8 + if version == WireVersion::V1 { 0 } else { 1 };
        duplicate_mean[header_len + stride..header_len + stride + width].fill(0);
        assert!(
            decode_digest(&duplicate_mean).is_err(),
            "duplicate means in {version:?}"
        );
    }
    for version in [WireVersion::V1, WireVersion::V2] {
        let mut bytes = TDigest::<F>::default().to_bytes_with_version(version);
        bytes.push(0);
        assert!(
            decode_digest(&bytes).is_err(),
            "trailing empty payload in {version:?}"
        );
    }
}

#[test]
fn every_binary_version_checks_state_and_empty_payload_length() {
    wire_rejects_contradictory_stats::<f32>();
    wire_rejects_contradictory_stats::<f64>();
}

#[test]
fn legacy_export_rejects_unit_weight_kind_loss_that_would_invalidate_bounds() {
    let td = TDigest::<f64>::builder()
        .max_size(1)
        .singleton_policy(SingletonPolicy::Off)
        .build()
        .unwrap()
        .merge_weighted_unsorted(&[0.0, 10.0], &[0.5, 0.5])
        .unwrap();
    assert!(td.centroids()[0].is_mixed());
    assert_eq!(td.centroids()[0].weight_f64(), 1.0);
    assert!(WireVersion::V1.validate_digest(&td).is_err());
    for version in [WireVersion::V2, WireVersion::V3] {
        assert!(decode_digest(&td.to_bytes_with_version(version)).is_ok());
    }
}

fn replace_field(s: &Series, name: &str, replacement: Option<Series>) -> Series {
    let mut fields = s.struct_().unwrap().fields_as_series();
    fields.retain(|f| f.name().as_str() != name);
    if let Some(field) = replacement {
        fields.push(field);
    }
    StructChunked::from_series(s.name().clone(), s.len(), fields.iter())
        .unwrap()
        .into_series()
}

fn codec_rejects_missing_state<F: FloatLike + FloatCore + WireOf>() {
    let td = TDigest::<F>::from_unsorted(&[F::from_f64(0.0), F::from_f64(10.0)], 100).unwrap();
    let series = td.to_series("td").unwrap();
    let bad_sum = replace_field(&series, "sum", Some(Series::new("sum".into(), [2.0])));
    assert!(TDigest::<F>::from_series(&bad_sum).is_err());
    for name in ["scale", "policy", "pin_per_side", "delta"] {
        assert!(
            TDigest::<F>::from_series(&replace_field(&series, name, None)).is_err(),
            "missing {name}"
        );
    }
    let empty = TDigest::<F>::default().to_series("td").unwrap();
    let dtype = empty
        .struct_()
        .unwrap()
        .field_by_name("centroids")
        .unwrap()
        .dtype()
        .clone();
    let missing = replace_field(
        &empty,
        "centroids",
        Some(Series::full_null("centroids".into(), 1, &dtype)),
    );
    assert!(
        TDigest::<F>::from_series(&missing).is_err(),
        "missing centroid list is not empty state"
    );
    // The original schema deliberately omitted all configuration fields.
    let mut legacy = series;
    for name in ["scale", "policy", "pin_per_side", "delta"] {
        legacy = replace_field(&legacy, name, None);
    }
    assert!(TDigest::<F>::from_series(&legacy).is_ok());
}

#[test]
fn struct_codec_checks_metadata_and_distinguishes_missing_from_empty() {
    codec_rejects_missing_state::<f32>();
    codec_rejects_missing_state::<f64>();
}

#[test]
#[cfg(target_pointer_width = "64")]
fn struct_export_rejects_configuration_outside_int64() {
    let too_large = i64::MAX as usize + 1;
    let capacity = TDigest::<f64>::builder()
        .max_size(too_large)
        .build()
        .unwrap();
    assert!(capacity.to_series("td").is_err());
    let pins = TDigest::<f64>::builder()
        .singleton_policy(SingletonPolicy::UseWithProtectedEdges(too_large))
        .build()
        .unwrap();
    assert!(pins.to_series("td").is_err());
    // The wider binary format still represents this configuration faithfully.
    assert!(decode_digest(&capacity.to_bytes()).is_ok());
    assert!(decode_digest(&pins.to_bytes()).is_ok());
}

fn valid_extreme_moments<F: FloatLike + FloatCore>() {
    // Normalizing moments must not overflow or discard a meaningful subnormal product.
    for (values, weights) in [
        (vec![-1e30, 1.0, 1e30], vec![1.0; 3]),
        (vec![0.0, 1.0], vec![f64::from_bits(1); 2]),
        (vec![1.0, 2.0], vec![1e300; 2]),
    ] {
        let values: Vec<F> = values.into_iter().map(F::from_f64).collect();
        let td = TDigest::<F>::from_weighted_unsorted(&values, &weights, 100).unwrap();
        assert!(td.validate_state().is_ok());
        assert!(decode_digest(&td.to_bytes()).is_ok());
    }
}

#[test]
fn valid_cancellation_subnormal_mass_and_large_moments_remain_supported() {
    valid_extreme_moments::<f32>();
    valid_extreme_moments::<f64>();
    let td = TDigest::<f64>::from_weighted_unsorted(&[1e308], &[f64::from_bits(1)], 100).unwrap();
    assert!(td.validate_state().is_ok());
}

#[test]
fn precision_roundtrips_preserve_authoritative_sums_after_rounding() {
    for values in [vec![0.1], vec![0.1, 0.2, 0.3], vec![1e-45]] {
        let source = TDigest::<f64>::from_unsorted(&values, 100).unwrap();
        let compact = source.cast_precision::<f32>().unwrap();
        let mut full = compact.cast_precision::<f64>().unwrap();
        assert_eq!(full.sum(), source.sum());
        full.add(0.4).unwrap();
        assert!(full.validate_state().is_ok());
        assert!(decode_digest(&full.to_bytes()).is_ok());
    }
}

#[test]
fn supported_updates_merges_and_casts_keep_valid_state() {
    use tdigest_rs::tdigest::ScaleFamily;
    for scale in [
        ScaleFamily::Quad,
        ScaleFamily::K1,
        ScaleFamily::K2,
        ScaleFamily::K2Norm,
        ScaleFamily::K3,
    ] {
        for policy in [
            SingletonPolicy::Off,
            SingletonPolicy::Use,
            SingletonPolicy::UseWithProtectedEdges(2),
        ] {
            for capacity in [1, 10, 100] {
                let mut digest = TDigest::<f64>::builder()
                    .max_size(capacity)
                    .scale(scale)
                    .singleton_policy(policy)
                    .build()
                    .unwrap();
                for shard in 0..40 {
                    let values: Vec<f64> = (0..30)
                        .map(|i| ((i * 71 + shard * 13) % 101) as f64 / 7.0 - 5.0)
                        .collect();
                    let weights: Vec<f64> = (0..30).map(|i| (i + 1) as f64 / 11.0).collect();
                    let mut other = TDigest::<f64>::builder()
                        .max_size(capacity)
                        .scale(scale)
                        .singleton_policy(policy)
                        .build()
                        .unwrap();
                    other.add_weighted_many(&values, &weights).unwrap();
                    digest.merge(&other).unwrap();
                    let compact = digest.cast_precision::<f32>().unwrap();
                    digest = compact.cast_precision::<f64>().unwrap();
                    assert!(digest.validate_state().is_ok());
                    assert!(decode_digest(&digest.to_bytes()).is_ok());
                }
            }
        }
    }
}
