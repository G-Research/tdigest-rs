//! Small, shared parsing & normalization helpers for all front-ends (Python, Polars, JNI, CLI).
//! Keep this dependency-light and free of PyO3/JNI/Polars types.

use std::fmt::{Display, Formatter};

use crate::tdigest::wire::{
    decode_digest, encode_digest, encode_digest_with_version, wire_precision, WireDecodedDigest,
    WirePrecision, WireVersion,
};
use crate::tdigest::TDigest;
use crate::tdigest::{scale::ScaleFamily, singleton_policy::SingletonPolicy};
use crate::{TdError, TdResult};

#[derive(Debug, Clone)]
pub enum ParseError {
    InvalidScale(String),
    InvalidPolicy(String),
    MissingEdges,        // 'edges' requires pin_per_side
    InvalidEdges(usize), // pin_per_side < 1
    InvalidPrecision(String),
}

impl Display for ParseError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::InvalidScale(s) => write!(
                f,
                "invalid scale: {s} (expected 'quad', 'k1', 'k2', or 'k3')"
            ),
            ParseError::InvalidPolicy(s) => write!(
                f,
                "invalid singleton policy: {s} (expected 'off', 'use', or 'edges')"
            ),
            ParseError::MissingEdges => {
                write!(f, "singleton_policy='edges' requires pin_per_side (>= 1)")
            }
            ParseError::InvalidEdges(k) => write!(
                f,
                "pin_per_side must be >= 1 when singleton_policy='edges' (got {k})"
            ),
            ParseError::InvalidPrecision(s) => write!(
                f,
                "unknown precision={s:?}; expected 'auto', 'f32' or 'f64'"
            ),
        }
    }
}
impl std::error::Error for ParseError {}

/// Lowercase/normalize a free-form string by removing `_` and spaces.
#[inline]
fn norm(s: &str) -> String {
    s.trim().to_ascii_lowercase().replace(['_', ' '], "")
}

pub fn parse_scale_str(raw: Option<&str>) -> Result<ScaleFamily, ParseError> {
    match raw.map(|x| x.trim().to_ascii_lowercase()) {
        None => Ok(ScaleFamily::K2),
        Some(ref v) if v == "quad" => Ok(ScaleFamily::Quad),
        Some(ref v) if v == "k1" => Ok(ScaleFamily::K1),
        Some(ref v) if v == "k2" => Ok(ScaleFamily::K2),
        Some(ref v) if v == "k3" => Ok(ScaleFamily::K3),
        Some(v) => Err(ParseError::InvalidScale(v)),
    }
}

pub fn scale_to_str(s: ScaleFamily) -> &'static str {
    match s {
        ScaleFamily::Quad => "quad",
        ScaleFamily::K1 => "k1",
        ScaleFamily::K2 => "k2",
        ScaleFamily::K3 => "k3",
    }
}

/// Accepts: off | use | edges | (legacy) usewithprotectededges
pub fn parse_singleton_policy_str(
    kind: Option<&str>,
    pin_per_side: Option<usize>,
) -> Result<SingletonPolicy, ParseError> {
    match kind.map(norm) {
        None => Ok(SingletonPolicy::Use),
        Some(ref v) if v == "off" => Ok(SingletonPolicy::Off),
        Some(ref v) if v == "use" => Ok(SingletonPolicy::Use),
        Some(ref v) if v == "edges" || v == "usewithprotectededges" => {
            let k = pin_per_side.ok_or(ParseError::MissingEdges)?;
            if k < 1 {
                return Err(ParseError::InvalidEdges(k));
            }
            Ok(SingletonPolicy::UseWithProtectedEdges(k))
        }
        Some(v) => Err(ParseError::InvalidPolicy(v)),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrecisionHint {
    Auto,
    F32,
    F64,
}

pub fn parse_precision_hint(s: &str) -> Result<PrecisionHint, ParseError> {
    match norm(s).as_str() {
        "auto" => Ok(PrecisionHint::Auto),
        "f32" => Ok(PrecisionHint::F32),
        "f64" => Ok(PrecisionHint::F64),
        other => Err(ParseError::InvalidPrecision(other.to_string())),
    }
}

pub fn policy_from_code_edges(code: i32, edges: i32) -> Result<SingletonPolicy, ParseError> {
    match code {
        0 => Ok(SingletonPolicy::Off),
        1 => Ok(SingletonPolicy::Use),
        2 => {
            let k = edges.max(0) as usize;
            if k < 1 {
                Err(ParseError::InvalidEdges(k))
            } else {
                Ok(SingletonPolicy::UseWithProtectedEdges(k))
            }
        }
        other => Err(ParseError::InvalidPolicy(other.to_string())),
    }
}

/// Shared finite-data guard for training/addition paths across front-ends.
#[inline]
pub fn ensure_finite_training_values(values: &[f64]) -> TdResult<()> {
    if values.iter().any(|v| !v.is_finite()) {
        return Err(TdError::NonFiniteInput {
            context: "sample value (NaN or ±inf)",
        });
    }
    Ok(())
}

/// Strict quantile probe validation shared by JNI/Python wrappers.
#[inline]
pub fn validate_quantile_probe(q: f64) -> Result<(), &'static str> {
    if !q.is_finite() {
        return Err("q must be a finite number in [0,1]");
    }
    if !(0.0..=1.0).contains(&q) {
        return Err("q must be in [0,1]");
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DigestConfig {
    pub max_size: usize,
    pub scale: ScaleFamily,
    pub policy: SingletonPolicy,
    pub legacy_delta: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DigestPrecision {
    F32,
    F64,
}

impl DigestPrecision {
    #[inline]
    fn as_str(self) -> &'static str {
        match self {
            DigestPrecision::F32 => "f32",
            DigestPrecision::F64 => "f64",
        }
    }
}

impl From<WirePrecision> for DigestPrecision {
    fn from(value: WirePrecision) -> Self {
        match value {
            WirePrecision::F32 => DigestPrecision::F32,
            WirePrecision::F64 => DigestPrecision::F64,
        }
    }
}

#[derive(Debug, Clone)]
pub enum FrontendError {
    InvalidTrainingData(String),
    InvalidProbe(String),
    InvalidScale(String),
    IncompatibleMerge(String),
    DecodeError(String),
}

impl Display for FrontendError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            FrontendError::InvalidTrainingData(msg)
            | FrontendError::InvalidProbe(msg)
            | FrontendError::InvalidScale(msg)
            | FrontendError::IncompatibleMerge(msg)
            | FrontendError::DecodeError(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for FrontendError {}

impl From<TdError> for FrontendError {
    fn from(value: TdError) -> Self {
        FrontendError::InvalidTrainingData(value.to_string())
    }
}

#[derive(Debug, Clone)]
pub enum FrontendDigest {
    F32(TDigest<f32>),
    F64(TDigest<f64>),
}

impl FrontendDigest {
    #[inline]
    pub fn canonical_empty() -> Self {
        FrontendDigest::F64(
            TDigest::<f64>::builder()
                .max_size(1000)
                .scale(ScaleFamily::K2)
                .singleton_policy(SingletonPolicy::Use)
                .build(),
        )
    }

    pub fn from_values(
        values: Vec<f64>,
        config: DigestConfig,
        precision: DigestPrecision,
    ) -> Result<Self, FrontendError> {
        ensure_finite_training_values(&values).map_err(FrontendError::from)?;
        if let Some(delta) = config.legacy_delta {
            if !(delta.is_finite() && delta > 0.0) {
                return Err(FrontendError::InvalidScale(
                    "legacy_delta must be finite and > 0".to_string(),
                ));
            }
            if config.scale != ScaleFamily::K2 {
                return Err(FrontendError::InvalidScale(
                    "legacy_delta mode only supports scale='k2'".to_string(),
                ));
            }
            if config.policy != SingletonPolicy::Off {
                return Err(FrontendError::InvalidScale(
                    "legacy_delta mode only supports singleton_policy='off'".to_string(),
                ));
            }
        }

        match precision {
            DigestPrecision::F32 => {
                let xs: Vec<f32> = values.into_iter().map(|v| v as f32).collect();
                let mut builder = TDigest::<f32>::builder()
                    .max_size(config.max_size)
                    .scale(config.scale)
                    .singleton_policy(config.policy);
                if let Some(delta) = config.legacy_delta {
                    builder = builder.legacy_delta(delta);
                }
                let td = builder
                    .build()
                    .merge_unsorted(xs)
                    .map_err(FrontendError::from)?;
                Ok(FrontendDigest::F32(td))
            }
            DigestPrecision::F64 => {
                let mut builder = TDigest::<f64>::builder()
                    .max_size(config.max_size)
                    .scale(config.scale)
                    .singleton_policy(config.policy);
                if let Some(delta) = config.legacy_delta {
                    builder = builder.legacy_delta(delta);
                }
                let td = builder
                    .build()
                    .merge_unsorted(values)
                    .map_err(FrontendError::from)?;
                Ok(FrontendDigest::F64(td))
            }
        }
    }

    #[inline]
    pub fn precision(&self) -> DigestPrecision {
        match self {
            FrontendDigest::F32(_) => DigestPrecision::F32,
            FrontendDigest::F64(_) => DigestPrecision::F64,
        }
    }

    #[inline]
    pub fn config(&self) -> DigestConfig {
        match self {
            FrontendDigest::F32(td) => DigestConfig {
                max_size: td.max_size(),
                scale: td.scale(),
                policy: td.singleton_policy(),
                legacy_delta: td.legacy_delta(),
            },
            FrontendDigest::F64(td) => DigestConfig {
                max_size: td.max_size(),
                scale: td.scale(),
                policy: td.singleton_policy(),
                legacy_delta: td.legacy_delta(),
            },
        }
    }

    #[inline]
    pub fn cast_precision(&self, target: DigestPrecision) -> Self {
        match (self, target) {
            (FrontendDigest::F32(td), DigestPrecision::F32) => FrontendDigest::F32(td.clone()),
            (FrontendDigest::F64(td), DigestPrecision::F64) => FrontendDigest::F64(td.clone()),
            (FrontendDigest::F32(td), DigestPrecision::F64) => {
                FrontendDigest::F64(td.cast_precision::<f64>())
            }
            (FrontendDigest::F64(td), DigestPrecision::F32) => {
                FrontendDigest::F32(td.cast_precision::<f32>())
            }
        }
    }

    #[inline]
    pub fn is_effectively_empty(&self) -> bool {
        match self {
            FrontendDigest::F32(td) => td.is_effectively_empty(),
            FrontendDigest::F64(td) => td.is_effectively_empty(),
        }
    }

    pub fn add_values_f64(&mut self, values: Vec<f64>) -> Result<(), FrontendError> {
        ensure_finite_training_values(&values).map_err(FrontendError::from)?;

        match self {
            FrontendDigest::F32(td) => {
                let xs: Vec<f32> = values.into_iter().map(|v| v as f32).collect();
                td.add_many(xs).map_err(FrontendError::from)?;
            }
            FrontendDigest::F64(td) => {
                td.add_many(values).map_err(FrontendError::from)?;
            }
        }
        Ok(())
    }

    pub fn add_weighted_f64(
        &mut self,
        values: Vec<f64>,
        weights: Vec<f64>,
    ) -> Result<(), FrontendError> {
        match self {
            FrontendDigest::F32(td) => {
                let xs: Vec<f32> = values.into_iter().map(|v| v as f32).collect();
                td.add_weighted_many(&xs, &weights)
                    .map_err(FrontendError::from)?;
            }
            FrontendDigest::F64(td) => {
                td.add_weighted_many(&values, &weights)
                    .map_err(FrontendError::from)?;
            }
        }
        Ok(())
    }

    pub fn scale_weights(&mut self, factor: f64) -> Result<(), FrontendError> {
        match self {
            FrontendDigest::F32(td) => {
                td.scale_weights(factor)
                    .map_err(|e| FrontendError::InvalidScale(e.to_string()))?;
            }
            FrontendDigest::F64(td) => {
                td.scale_weights(factor)
                    .map_err(|e| FrontendError::InvalidScale(e.to_string()))?;
            }
        }
        Ok(())
    }

    pub fn scale_values(&mut self, factor: f64) -> Result<(), FrontendError> {
        match self {
            FrontendDigest::F32(td) => {
                td.scale_values(factor)
                    .map_err(|e| FrontendError::InvalidScale(e.to_string()))?;
            }
            FrontendDigest::F64(td) => {
                td.scale_values(factor)
                    .map_err(|e| FrontendError::InvalidScale(e.to_string()))?;
            }
        }
        Ok(())
    }

    pub fn merge_in_place(&mut self, other: &Self) -> Result<(), FrontendError> {
        if self.precision() != other.precision() {
            return Err(FrontendError::IncompatibleMerge(format!(
                "tdigest merge: incompatible digests (precision {} vs {}). \
Cast explicitly before merge (e.g. cast_precision('f64')).",
                self.precision().as_str(),
                other.precision().as_str()
            )));
        }

        if self.is_effectively_empty() && !other.is_effectively_empty() {
            *self = other.clone();
            return Ok(());
        }
        if other.is_effectively_empty() {
            return Ok(());
        }

        let lhs_cfg = self.config();
        let rhs_cfg = other.config();
        if lhs_cfg != rhs_cfg {
            return Err(FrontendError::IncompatibleMerge(format!(
                "tdigest merge: incompatible configs (max_size {} vs {}, scale {:?} vs {:?}, singleton_policy {:?} vs {:?}, legacy_delta {:?} vs {:?}). \
Rebuild or cast to a shared configuration before merge.",
                lhs_cfg.max_size,
                rhs_cfg.max_size,
                lhs_cfg.scale,
                rhs_cfg.scale,
                lhs_cfg.policy,
                rhs_cfg.policy,
                lhs_cfg.legacy_delta,
                rhs_cfg.legacy_delta
            )));
        }

        match (self, other) {
            (FrontendDigest::F32(a), FrontendDigest::F32(b)) => {
                *a = TDigest::<f32>::merge_digests(vec![a.clone(), b.clone()]);
            }
            (FrontendDigest::F64(a), FrontendDigest::F64(b)) => {
                *a = TDigest::<f64>::merge_digests(vec![a.clone(), b.clone()]);
            }
            _ => unreachable!("precision mismatch checked above"),
        }
        Ok(())
    }

    pub fn merge_all(digests: Vec<Self>) -> Result<Self, FrontendError> {
        if digests.is_empty() {
            return Ok(Self::canonical_empty());
        }

        let first = digests[0].clone();
        let mut acc = first.clone();
        for d in digests.iter().skip(1) {
            acc.merge_in_place(d)?;
        }
        Ok(acc)
    }

    pub fn quantile_strict(&self, q: f64) -> Result<f64, FrontendError> {
        validate_quantile_probe(q).map_err(|msg| FrontendError::InvalidProbe(msg.to_string()))?;
        Ok(match self {
            FrontendDigest::F32(td) => td.quantile(q),
            FrontendDigest::F64(td) => td.quantile(q),
        })
    }

    #[inline]
    pub fn median(&self) -> f64 {
        match self {
            FrontendDigest::F32(td) => td.median(),
            FrontendDigest::F64(td) => td.median(),
        }
    }

    #[inline]
    pub fn cdf(&self, xs: &[f64]) -> Vec<f64> {
        match self {
            FrontendDigest::F32(td) => td.cdf_or_nan(xs),
            FrontendDigest::F64(td) => td.cdf_or_nan(xs),
        }
    }

    pub fn trimmed_mean_strict(&self, lower: f64, upper: f64) -> Result<f64, FrontendError> {
        if !lower.is_finite() || !upper.is_finite() {
            return Err(FrontendError::InvalidProbe(
                "trimmed_mean bounds must be finite values in [0,1]".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&lower) || !(0.0..=1.0).contains(&upper) || lower > upper {
            return Err(FrontendError::InvalidProbe(
                "trimmed_mean bounds must satisfy 0 <= lower <= upper <= 1".to_string(),
            ));
        }

        Ok(match self {
            FrontendDigest::F32(td) => td.trimmed_mean(lower, upper),
            FrontendDigest::F64(td) => td.trimmed_mean(lower, upper),
        })
    }

    #[inline]
    pub fn centroid_means_f64(&self) -> Vec<f64> {
        match self {
            FrontendDigest::F32(td) => td.centroids_ref().iter().map(|c| c.mean_f64()).collect(),
            FrontendDigest::F64(td) => td.centroids_ref().iter().map(|c| c.mean_f64()).collect(),
        }
    }

    #[inline]
    pub fn centroid_weights_f64(&self) -> Vec<f64> {
        match self {
            FrontendDigest::F32(td) => td.centroids_ref().iter().map(|c| c.weight_f64()).collect(),
            FrontendDigest::F64(td) => td.centroids_ref().iter().map(|c| c.weight_f64()).collect(),
        }
    }

    #[inline]
    pub fn centroid_len(&self) -> usize {
        match self {
            FrontendDigest::F32(td) => td.centroids_ref().len(),
            FrontendDigest::F64(td) => td.centroids_ref().len(),
        }
    }

    #[inline]
    pub fn to_bytes(&self) -> Vec<u8> {
        match self {
            FrontendDigest::F32(td) => encode_digest(td),
            FrontendDigest::F64(td) => encode_digest(td),
        }
    }

    #[inline]
    pub fn to_bytes_with_version(&self, version: WireVersion) -> Vec<u8> {
        match self {
            FrontendDigest::F32(td) => encode_digest_with_version(td, version),
            FrontendDigest::F64(td) => encode_digest_with_version(td, version),
        }
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, FrontendError> {
        match decode_digest(bytes) {
            Ok(WireDecodedDigest::F32(td32)) => Ok(FrontendDigest::F32(td32)),
            Ok(WireDecodedDigest::F64(td64)) => Ok(FrontendDigest::F64(td64)),
            Err(e) => Err(FrontendError::DecodeError(format!(
                "tdigest decode error: {e}"
            ))),
        }
    }

    pub fn from_bytes_with_expected(
        bytes: &[u8],
        expected: WirePrecision,
    ) -> Result<Self, FrontendError> {
        let actual = wire_precision(bytes)
            .map_err(|e| FrontendError::DecodeError(format!("tdigest.from_bytes: {e}")))?;

        if actual != expected {
            return Err(FrontendError::DecodeError(format!(
                "tdigest.from_bytes: mixed f32/f64 blobs in column (expected {})",
                match expected {
                    WirePrecision::F32 => "f32",
                    WirePrecision::F64 => "f64",
                }
            )));
        }
        Self::from_bytes(bytes)
    }

    #[inline]
    pub fn inner_kind(&self) -> &'static str {
        match self {
            FrontendDigest::F32(_) => "f32",
            FrontendDigest::F64(_) => "f64",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_k2_use(max_size: usize) -> DigestConfig {
        DigestConfig {
            max_size,
            scale: ScaleFamily::K2,
            policy: SingletonPolicy::Use,
            legacy_delta: None,
        }
    }

    #[test]
    fn parse_helpers_accept_expected_forms_and_reject_invalid() {
        assert_eq!(
            parse_scale_str(Some("K2")).expect("parse scale"),
            ScaleFamily::K2
        );
        assert!(parse_scale_str(Some("wat")).is_err());

        assert_eq!(
            parse_singleton_policy_str(Some("use"), None).expect("parse policy"),
            SingletonPolicy::Use
        );
        assert_eq!(
            parse_singleton_policy_str(Some("edges"), Some(2)).expect("parse edges"),
            SingletonPolicy::UseWithProtectedEdges(2)
        );
        assert!(parse_singleton_policy_str(Some("edges"), None).is_err());

        assert_eq!(
            parse_precision_hint("auto").expect("parse precision"),
            PrecisionHint::Auto
        );
        assert!(parse_precision_hint("weird").is_err());
    }

    #[test]
    fn finite_and_quantile_probe_validation_is_strict() {
        assert!(ensure_finite_training_values(&[0.0, 1.0, 2.0]).is_ok());
        assert!(ensure_finite_training_values(&[0.0, f64::NAN, 1.0]).is_err());
        assert!(ensure_finite_training_values(&[0.0, f64::INFINITY, 1.0]).is_err());
        assert!(ensure_finite_training_values(&[0.0, f64::NEG_INFINITY, 1.0]).is_err());

        assert!(validate_quantile_probe(0.0).is_ok());
        assert!(validate_quantile_probe(1.0).is_ok());
        assert!(validate_quantile_probe(-0.1).is_err());
        assert!(validate_quantile_probe(1.1).is_err());
        assert!(validate_quantile_probe(f64::NAN).is_err());
        assert!(validate_quantile_probe(f64::INFINITY).is_err());
    }

    #[test]
    fn frontend_digest_supports_add_merge_quantile_cdf_median_and_roundtrip() {
        let mut a = FrontendDigest::from_values(
            vec![0.0, 1.0, 2.0, 3.0],
            cfg_k2_use(128),
            DigestPrecision::F64,
        )
        .expect("build a");
        let b = FrontendDigest::from_values(
            vec![10.0, 11.0, 12.0, 13.0],
            cfg_k2_use(128),
            DigestPrecision::F64,
        )
        .expect("build b");

        a.add_values_f64(vec![4.0]).expect("add scalar");
        a.add_values_f64(vec![5.0, 6.0]).expect("add batch");
        a.merge_in_place(&b).expect("merge");

        let q = a.quantile_strict(0.5).expect("q");
        let c = a.cdf(&[3.0])[0];
        let m = a.median();
        assert!(q.is_finite());
        assert!(c.is_finite());
        assert!(m.is_finite());

        let bytes = a.to_bytes();
        let rt = FrontendDigest::from_bytes(&bytes).expect("decode");
        let q_rt = rt.quantile_strict(0.5).expect("q rt");
        assert!((q_rt - q).abs() <= 1e-9);
    }

    #[test]
    fn frontend_merge_all_empty_is_canonical_f64_empty() {
        let empty = FrontendDigest::merge_all(vec![]).expect("merge empty");
        assert_eq!(empty.inner_kind(), "f64");
        assert!(empty.quantile_strict(0.5).expect("q empty").is_nan());
    }

    #[test]
    fn frontend_add_and_merge_reject_invalid_inputs() {
        let mut f64d =
            FrontendDigest::from_values(vec![0.0, 1.0, 2.0], cfg_k2_use(64), DigestPrecision::F64)
                .expect("build f64");
        let f32d =
            FrontendDigest::from_values(vec![0.0, 1.0, 2.0], cfg_k2_use(64), DigestPrecision::F32)
                .expect("build f32");

        let add_err = f64d
            .add_values_f64(vec![0.0, f64::NAN])
            .expect_err("add must reject nan");
        assert!(add_err.to_string().to_lowercase().contains("nan"));

        let merge_err = f64d
            .merge_in_place(&f32d)
            .expect_err("merge precision mismatch");
        let msg = merge_err.to_string().to_lowercase();
        assert!(msg.contains("precision"));
        assert!(msg.contains("cast explicitly"));
    }

    #[test]
    fn frontend_cast_precision_roundtrip_preserves_behavior() {
        let d = FrontendDigest::from_values(
            vec![-2.0, -1.0, 0.0, 1.0, 10.0, 20.0],
            DigestConfig {
                max_size: 96,
                scale: ScaleFamily::K3,
                policy: SingletonPolicy::UseWithProtectedEdges(2),
                legacy_delta: None,
            },
            DigestPrecision::F64,
        )
        .expect("build");

        let d32 = d.cast_precision(DigestPrecision::F32);
        let d64 = d32.cast_precision(DigestPrecision::F64);

        assert_eq!(d32.inner_kind(), "f32");
        assert_eq!(d64.inner_kind(), "f64");
        assert_eq!(d.config(), d64.config());
        assert!(
            (d.quantile_strict(0.5).expect("q0") - d64.quantile_strict(0.5).expect("q1")).abs()
                <= 1e-4
        );
        assert!((d.cdf(&[1.5])[0] - d64.cdf(&[1.5])[0]).abs() <= 1e-4);
    }

    #[test]
    fn frontend_scaling_supports_weights_and_values() {
        let mut d = FrontendDigest::from_values(
            vec![0.0, 1.0, 2.0, 3.0],
            cfg_k2_use(128),
            DigestPrecision::F64,
        )
        .expect("build");

        let q0 = d.quantile_strict(0.5).expect("q0");
        let c0 = d.cdf(&[1.5])[0];

        d.scale_weights(2.0).expect("scale weights");
        assert!((d.quantile_strict(0.5).expect("q1") - q0).abs() <= 1e-9);
        assert!((d.cdf(&[1.5])[0] - c0).abs() <= 1e-9);

        d.scale_values(3.0).expect("scale values");
        assert!((d.quantile_strict(0.5).expect("q2") - q0 * 3.0).abs() <= 1e-9);
        assert!((d.median() - q0 * 3.0).abs() <= 1e-9);
        assert!((d.cdf(&[1.5 * 3.0])[0] - c0).abs() <= 1e-9);

        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(d.scale_weights(bad).is_err());
            assert!(d.scale_values(bad).is_err());
        }
    }

    #[test]
    fn frontend_merge_rejects_config_mismatch_with_details() {
        let mut a = FrontendDigest::from_values(
            vec![0.0, 1.0, 2.0],
            DigestConfig {
                max_size: 64,
                scale: ScaleFamily::K2,
                policy: SingletonPolicy::Use,
                legacy_delta: None,
            },
            DigestPrecision::F64,
        )
        .expect("build a");
        let b = FrontendDigest::from_values(
            vec![10.0, 11.0, 12.0],
            DigestConfig {
                max_size: 128,
                scale: ScaleFamily::K3,
                policy: SingletonPolicy::Use,
                legacy_delta: None,
            },
            DigestPrecision::F64,
        )
        .expect("build b");

        let err = a.merge_in_place(&b).expect_err("config mismatch");
        let msg = err.to_string().to_lowercase();
        assert!(msg.contains("max_size"));
        assert!(msg.contains("scale"));
        assert!(msg.contains("k2"));
        assert!(msg.contains("k3"));
    }

    #[test]
    fn frontend_quantile_is_strict_on_probe_and_cdf_propagates_nan_probe() {
        let d = FrontendDigest::from_values(
            vec![0.0, 1.0, 2.0, 3.0],
            cfg_k2_use(64),
            DigestPrecision::F64,
        )
        .expect("build");

        assert!(d.quantile_strict(f64::NAN).is_err());
        assert!(d.quantile_strict(-0.1).is_err());
        assert!(d.quantile_strict(1.1).is_err());

        let out = d.cdf(&[f64::NAN, f64::NEG_INFINITY, f64::INFINITY]);
        assert!(out[0].is_nan());
        assert_eq!(out[1], 0.0);
        assert_eq!(out[2], 1.0);
    }

    #[test]
    fn frontend_trimmed_mean_is_available_and_strict() {
        let d = FrontendDigest::from_values(
            vec![0.0, 1.0, 2.0, 3.0, 100.0],
            cfg_k2_use(64),
            DigestPrecision::F64,
        )
        .expect("build");

        let tm = d.trimmed_mean_strict(0.1, 0.9).expect("trimmed mean");
        assert!(tm.is_finite());

        assert!(d.trimmed_mean_strict(f64::NAN, 0.9).is_err());
        assert!(d.trimmed_mean_strict(0.1, f64::INFINITY).is_err());
        assert!(d.trimmed_mean_strict(-0.1, 0.9).is_err());
        assert!(d.trimmed_mean_strict(0.9, 0.1).is_err());
    }

    #[test]
    fn frontend_weighted_add_supports_f32_and_f64() {
        let mut d32 = FrontendDigest::from_values(vec![0.0], cfg_k2_use(64), DigestPrecision::F32)
            .expect("build f32");
        let mut d64 = FrontendDigest::from_values(vec![0.0], cfg_k2_use(64), DigestPrecision::F64)
            .expect("build f64");

        d32.add_weighted_f64(vec![10.0, 20.0], vec![2.0, 3.0])
            .expect("weighted f32");
        d64.add_weighted_f64(vec![10.0, 20.0], vec![2.0, 3.0])
            .expect("weighted f64");

        assert!(d32.quantile_strict(0.5).expect("q32").is_finite());
        assert!(d64.quantile_strict(0.5).expect("q64").is_finite());
    }

    #[test]
    fn frontend_to_bytes_with_version_supports_v1_v2_v3() {
        let d = FrontendDigest::from_values(
            vec![-1.0, 0.0, 1.0, 2.0, 3.0],
            cfg_k2_use(96),
            DigestPrecision::F64,
        )
        .expect("build");

        for v in [WireVersion::V1, WireVersion::V2, WireVersion::V3] {
            let blob = d.to_bytes_with_version(v);
            let rt = FrontendDigest::from_bytes(&blob).expect("decode");
            assert!(rt.quantile_strict(0.5).expect("q").is_finite());
        }
    }
}
