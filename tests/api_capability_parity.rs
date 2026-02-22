use tdigest_rs::tdigest::wire::{decode_digest, wire_precision, WireDecodedDigest, WirePrecision};
use tdigest_rs::tdigest::{ScaleFamily, TDigest};

#[test]
fn rust_f64_supports_add_merge_quantile_cdf_median_and_wire_roundtrip() {
    let mut a = TDigest::<f64>::builder()
        .max_size(128)
        .scale(ScaleFamily::K2)
        .build();
    a.add_many(vec![0.0, 1.0, 2.0, 3.0]).expect("add base");
    a.add(4.0).expect("add scalar");
    a.add_many(vec![5.0, 6.0]).expect("add batch");

    let mut b = TDigest::<f64>::builder()
        .max_size(128)
        .scale(ScaleFamily::K2)
        .build();
    b.add_many(vec![10.0, 11.0, 12.0, 13.0]).expect("add rhs");

    let mut merged_in_place = a.clone();
    merged_in_place.merge(&b);
    let merged_static = TDigest::<f64>::merge_digests(vec![a.clone(), b.clone()]);

    let q = merged_in_place.quantile(0.5);
    let c = merged_in_place.cdf_or_nan(&[3.0])[0];
    let m = merged_in_place.median();
    assert!(q.is_finite());
    assert!(c.is_finite());
    assert!(m.is_finite());

    assert!((merged_in_place.quantile(0.5) - merged_static.quantile(0.5)).abs() <= 1e-9);
    assert!((merged_in_place.median() - merged_static.median()).abs() <= 1e-9);

    let bytes = merged_in_place.to_bytes();
    assert_eq!(
        wire_precision(&bytes).expect("wire precision"),
        WirePrecision::F64
    );

    let decoded = TDigest::<f64>::from_bytes(&bytes).expect("decode f64");
    assert!((decoded.quantile(0.5) - merged_in_place.quantile(0.5)).abs() <= 1e-9);
    assert!((decoded.median() - merged_in_place.median()).abs() <= 1e-9);
}

#[test]
fn rust_f32_supports_add_merge_and_f32_wire_precision() {
    let mut a = TDigest::<f32>::builder()
        .max_size(128)
        .scale(ScaleFamily::K2)
        .build();
    a.add_many(vec![0.0_f32, 1.0, 2.0, 3.0]).expect("add base");
    a.add(4.0_f32).expect("add scalar");
    a.add_many(vec![5.0_f32, 6.0]).expect("add batch");

    let mut b = TDigest::<f32>::builder()
        .max_size(128)
        .scale(ScaleFamily::K2)
        .build();
    b.add_many(vec![10.0_f32, 11.0, 12.0, 13.0])
        .expect("add rhs");

    let mut merged = a.clone();
    merged.merge_many(&[b.clone()]);

    let q = merged.quantile(0.5);
    let c = merged.cdf_or_nan(&[3.0])[0];
    let m = merged.median();
    assert!(q.is_finite());
    assert!(c.is_finite());
    assert!(m.is_finite());

    let bytes = merged.to_bytes();
    assert_eq!(
        wire_precision(&bytes).expect("wire precision"),
        WirePrecision::F32
    );

    match decode_digest(&bytes).expect("decode digest") {
        WireDecodedDigest::F32(td32) => {
            assert!((td32.quantile(0.5) - merged.quantile(0.5)).abs() <= 1e-6);
            assert!((td32.median() - merged.median()).abs() <= 1e-6);
        }
        WireDecodedDigest::F64(_) => panic!("expected f32 wire decode"),
    }
}

#[test]
fn rust_core_rejects_infinities_in_training_and_add_paths() {
    let from_array_err = TDigest::<f64>::from_array(vec![0.0_f64, f64::INFINITY]);
    assert!(from_array_err.is_err(), "from_array must reject +inf");

    let mut d = TDigest::<f64>::builder()
        .max_size(64)
        .scale(ScaleFamily::K2)
        .build();
    assert!(d.add(f64::NEG_INFINITY).is_err(), "add must reject -inf");
    assert!(
        d.add_many(vec![1.0_f64, f64::INFINITY]).is_err(),
        "add_many must reject +inf in batch"
    );
}

#[test]
fn rust_scaling_weights_and_values_behaves_as_expected() {
    let mut d = TDigest::<f64>::builder()
        .max_size(128)
        .scale(ScaleFamily::K2)
        .build();
    d.add_many(vec![0.0, 1.0, 2.0, 3.0]).expect("add");

    let q0 = d.quantile(0.5);
    let c0 = d.cdf_or_nan(&[1.5])[0];
    let count0 = d.count();
    let sum0 = d.sum();
    let min0 = d.min();
    let max0 = d.max();

    d.scale_weights(3.0).expect("scale weights");
    assert!((d.quantile(0.5) - q0).abs() <= 1e-9);
    assert!((d.cdf_or_nan(&[1.5])[0] - c0).abs() <= 1e-9);
    assert!((d.count() - count0 * 3.0).abs() <= 1e-9);
    assert!((d.sum() - sum0 * 3.0).abs() <= 1e-9);
    assert!((d.min() - min0).abs() <= 1e-9);
    assert!((d.max() - max0).abs() <= 1e-9);

    d.scale_values(2.0).expect("scale values");
    assert!((d.quantile(0.5) - q0 * 2.0).abs() <= 1e-9);
    assert!((d.median() - q0 * 2.0).abs() <= 1e-9);
    assert!((d.cdf_or_nan(&[1.5 * 2.0])[0] - c0).abs() <= 1e-9);
    assert!((d.sum() - sum0 * 3.0 * 2.0).abs() <= 1e-9);
    assert!((d.count() - count0 * 3.0).abs() <= 1e-9);
    assert!((d.min() - min0 * 2.0).abs() <= 1e-9);
    assert!((d.max() - max0 * 2.0).abs() <= 1e-9);
}

#[test]
fn rust_scaling_rejects_invalid_factors() {
    let mut d = TDigest::<f64>::builder()
        .max_size(64)
        .scale(ScaleFamily::K2)
        .build();
    d.add_many(vec![0.0, 1.0, 2.0]).expect("add");

    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(
            d.scale_weights(bad).is_err(),
            "scale_weights({bad}) must fail"
        );
        assert!(
            d.scale_values(bad).is_err(),
            "scale_values({bad}) must fail"
        );
    }
}
