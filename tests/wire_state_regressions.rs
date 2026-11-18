//! Format-specific export limits and configuration preservation.

use ordered_float::FloatCore;
use tdigest_rs::tdigest::precision::FloatLike;
use tdigest_rs::tdigest::singleton_policy::SingletonPolicy;
use tdigest_rs::tdigest::wire::{
    decode_digest, encode_digest_with_version, WireDecodedDigest, WireVersion,
};
use tdigest_rs::tdigest::{ScaleFamily, TDigest};

#[test]
fn v1_requires_representable_integer_weights_and_total() {
    for weights in [
        [0.1, 0.9],
        [1.2, 1.8],
        [18446744073709551616.0, 1.0],
        [9223372036854775808.0; 2],
    ] {
        let td = TDigest::<f64>::from_weighted_unsorted(&[0.0, 10.0], &weights, 1000).unwrap();
        assert!(WireVersion::V1.validate_digest(&td).is_err());
        for version in [WireVersion::V2, WireVersion::V3] {
            let bytes = encode_digest_with_version(&td, version);
            let WireDecodedDigest::F64(restored) = decode_digest(&bytes).unwrap() else {
                panic!("changed precision");
            };
            assert_eq!(restored.centroids(), td.centroids());
        }
    }
    let td = TDigest::<f64>::from_weighted_unsorted(&[0.0, 10.0], &[90.0, 10.0], 1000).unwrap();
    assert!(WireVersion::V1.validate_digest(&td).is_ok());
    let WireDecodedDigest::F64(restored) =
        decode_digest(&encode_digest_with_version(&td, WireVersion::V1)).unwrap()
    else {
        panic!("changed precision");
    };
    assert!(restored.centroids().iter().all(|c| c.is_mixed()));
}

fn invalid_v1_weight_is_error<F: FloatLike + FloatCore>() {
    let td = TDigest::<F>::from_unsorted(&[F::from_f64(1.0)], 10).unwrap();
    let mut bytes = encode_digest_with_version(&td, WireVersion::V1);
    let weight_offset = 56 + std::mem::size_of::<F>();
    bytes[weight_offset..weight_offset + 8].copy_from_slice(&0_u64.to_le_bytes());
    assert!(decode_digest(&bytes).is_err());
}

#[test]
fn zero_v1_centroid_weights_return_errors_in_both_precisions() {
    invalid_v1_weight_is_error::<f32>();
    invalid_v1_weight_is_error::<f64>();
}

#[test]
fn invalid_payload_weights_return_errors() {
    let td = TDigest::<f32>::from_unsorted(&[1.0], 10).unwrap();
    for weight in [0.0_f64, -1.0, f64::NAN, f64::INFINITY] {
        let mut bytes = encode_digest_with_version(&td, WireVersion::V2);
        bytes[60..68].copy_from_slice(&weight.to_le_bytes());
        assert!(decode_digest(&bytes).is_err());
    }
}

#[test]
fn full_width_edge_pins_roundtrip_and_extensions_are_checked() {
    let td = TDigest::<f64>::builder()
        .max_size(1000)
        .scale(ScaleFamily::K3)
        .singleton_policy(SingletonPolicy::UseWithProtectedEdges(300))
        .build()
        .unwrap()
        .merge_unsorted((0..400).map(|i| i as f64).collect())
        .unwrap();
    assert_eq!(td.centroids().len(), 400);
    assert!(WireVersion::V1.validate_digest(&td).is_err());
    assert!(WireVersion::V2.validate_digest(&td).is_err());
    let bytes = td.to_bytes();
    assert_eq!(bytes[5] & 4, 4);
    assert_eq!(bytes[6], 72);
    assert_eq!(u64::from_le_bytes(bytes[64..72].try_into().unwrap()), 300);
    let WireDecodedDigest::F64(restored) = decode_digest(&bytes).unwrap() else {
        panic!("changed precision");
    };
    assert_eq!(
        restored.singleton_policy(),
        SingletonPolicy::UseWithProtectedEdges(300)
    );
    assert_eq!(restored.to_bytes(), bytes);

    let mut corrupt = bytes.clone();
    corrupt[64] ^= 1;
    assert!(decode_digest(&corrupt).is_err());
    let mut truncated = bytes.clone();
    truncated[6] = 64;
    assert!(decode_digest(&truncated).is_err());
    let mut zero = bytes.clone();
    zero[64..72].copy_from_slice(&0_u64.to_le_bytes());
    assert!(decode_digest(&zero).is_err());
    let mut wrong_policy = bytes;
    wrong_policy[9] = 0;
    assert!(decode_digest(&wrong_policy).is_err());
}

#[test]
fn delta_headers_reject_incompatible_scale_and_policy_before_use() {
    let td = TDigest::<f64>::builder()
        .scale(ScaleFamily::K2Norm)
        .singleton_policy(SingletonPolicy::Off)
        .delta(17.0)
        .build()
        .unwrap();
    let valid = td.to_bytes();
    assert!(decode_digest(&valid).is_ok());

    // Disable checksums so these fail for their configuration, not a stale CRC.
    let mut wrong_scale = valid.clone();
    wrong_scale[5] &= !1;
    wrong_scale[8] = 2;
    let error = decode_digest(&wrong_scale).unwrap_err().to_string();
    assert!(error.contains("delta mode requires K2Norm"));

    let mut wrong_policy = valid.clone();
    wrong_policy[5] &= !1;
    wrong_policy[9] = 1;
    let error = decode_digest(&wrong_policy).unwrap_err().to_string();
    assert!(error.contains("singleton policy Off"));

    let mut wide_pin = valid;
    wide_pin[5] = 2 | 4;
    wide_pin[6] = 80;
    wide_pin[9] = 2;
    wide_pin[10] = 255;
    wide_pin.splice(72..72, 300_u64.to_le_bytes());
    let error = decode_digest(&wide_pin).unwrap_err().to_string();
    assert!(error.contains("singleton policy Off"));
}

#[test]
fn rust_builder_defaults_delta_and_rejects_explicit_incompatible_configuration() {
    let default_config = TDigest::<f64>::builder().delta(17.0).build().unwrap();
    assert_eq!(default_config.scale(), ScaleFamily::K2Norm);
    assert_eq!(default_config.singleton_policy(), SingletonPolicy::Off);
    assert!(WireVersion::V3.validate_digest(&default_config).is_ok());
    let wrong_policy = TDigest::<f64>::builder()
        .scale(ScaleFamily::K2Norm)
        .singleton_policy(SingletonPolicy::UseWithProtectedEdges(300))
        .delta(17.0)
        .build();
    assert!(wrong_policy.is_err());
    let valid = TDigest::<f64>::builder()
        .scale(ScaleFamily::K2Norm)
        .singleton_policy(SingletonPolicy::Off)
        .delta(17.0)
        .build()
        .unwrap();
    assert!(WireVersion::V3.validate_digest(&valid).is_ok());
    assert!(decode_digest(&valid.to_bytes()).is_ok());
}
