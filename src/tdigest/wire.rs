// Canonical TDigest binary wire codec ("TDIG" format).
//
// v1/v2 header (56 bytes, little-endian):
//   0..4   : magic = b"TDIG"
//   4      : version (1|2)
//   5      : scale_code
//   6      : policy_code
//   7      : pin_per_side
//   8..16  : max_size (u64)
//   16..24 : total_weight (v1=u64, v2=f64)
//   24..32 : min (f64)
//   32..40 : max (f64)
//   40..48 : centroid_count (u64)
//   48..56 : data_sum (f64)
//
// v1 payload:
//   - f32: mean(f32) + weight(u64)
//   - f64: mean(f64) + weight(u64)
//
// v2 payload:
//   - f32: mean(f32) + weight(f64) + kind(u8)
//   - f64: mean(f64) + weight(f64) + kind(u8)
//
// v3 header (64 bytes minimum, little-endian):
//   0..4   : magic = b"TDIG"
//   4      : version (3)
//   5      : flags (bit0: checksum, bit1: delta extension, bit2: full-width pin extension, bit3: f64 mass required for f32 means)
//   6      : header_len (u8, >= 64)
//   7      : payload_precision (1=f32, 2=f64)
//   8      : scale_code
//   9      : policy_code
//   10     : pin_per_side
//   11     : reserved
//   12..20 : max_size (u64)
//   20..28 : total_weight (f64)
//   28..36 : min (f64)
//   36..44 : max (f64)
//   44..52 : centroid_count (u64)
//   52..60 : data_sum (f64)
//   60..64 : checksum (u32, CRC32 when flags bit0 set)
//   64..   : optional extensions, in flag order:
//            bit1: delta (f64, 8 bytes)
//            bit2: pin_per_side (u64, 8 bytes); follows delta when both are present
//            full-width pins require the edges policy; the fixed pin byte is 255
//            header_len includes every extension; checksums cover all extension bytes
//
// v3 payload:
//   - f32: mean(f32) + weight(f64) + kind(u8)
//   - f64: mean(f64) + weight(f64) + kind(u8)

use std::fmt;

use crc32fast::Hasher;
use ordered_float::FloatCore;

use crate::tdigest::centroids::Centroid;
use crate::tdigest::precision::{FloatLike, Precision};
use crate::tdigest::scale::ScaleFamily;
use crate::tdigest::singleton_policy::SingletonPolicy;
use crate::tdigest::{DigestStats, TDigest};

const MAGIC: &[u8; 4] = b"TDIG";
const VERSION_V1: u8 = 1;
const VERSION_V2: u8 = 2;
const VERSION_V3: u8 = 3;

const HEADER_LEN_V12: usize = 56;
const HEADER_LEN_V3_MIN: usize = 64;
const V3_CHECKSUM_OFFSET: usize = 60;

const V3_FLAG_CHECKSUM: u8 = 0x01;
const V3_FLAG_DELTA: u8 = 0x02;
const V3_FLAG_WIDE_PIN: u8 = 0x04;
// No extra bytes: v2/v3 payload weights are already f64. This flag prevents
// older readers from silently narrowing weights to the f32 mean precision.
const V3_FLAG_WIDE_WEIGHT: u8 = 0x08;

const PRECISION_CODE_F32: u8 = 1;
const PRECISION_CODE_F64: u8 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WireVersion {
    V1,
    V2,
    V3,
}

impl WireVersion {
    pub const LATEST: Self = Self::V3;

    /// Older wire versions have no field for the compression delta.
    pub fn validate_delta(self, delta: Option<f64>) -> WireResult<()> {
        if delta.is_some() && self != Self::V3 {
            return Err(WireError::InvalidHeader("delta mode requires TDIG v3"));
        }
        Ok(())
    }

    /// Check whether this version can represent the digest's weights and config.
    /// V1 still reconstructs centroid kind using its historical weight heuristic.
    pub fn validate_digest<F: FloatLike + FloatCore>(self, td: &TDigest<F>) -> WireResult<()> {
        td.validate_state()
            .map_err(|_| WireError::InvalidPayload("invalid digest state"))?;
        self.validate_delta(td.delta())?;
        if td.delta().is_some()
            && (td.scale() != ScaleFamily::K2Norm || td.singleton_policy() != SingletonPolicy::Off)
        {
            return Err(WireError::InvalidHeader(
                "delta mode requires K2Norm scale and singleton policy Off",
            ));
        }
        if self != Self::V3 {
            if let SingletonPolicy::UseWithProtectedEdges(pin) = td.singleton_policy() {
                if pin > u8::MAX as usize {
                    return Err(WireError::InvalidHeader(
                        "pin_per_side > 255 requires TDIG v3",
                    ));
                }
            }
        }
        if self == Self::V1 {
            let mut total = 0_u64;
            for centroid in td.centroids() {
                let weight = centroid.weight_f64();
                // The cast of u64::MAX rounds UP to 2^64; this is an exclusive bound.
                if !weight.is_finite()
                    || weight <= 0.0
                    || weight.fract() != 0.0
                    || weight >= u64::MAX as f64
                {
                    return Err(WireError::InvalidPayload(
                        "TDIG v1 requires positive integer centroid weights within u64 range",
                    ));
                }
                total = total
                    .checked_add(weight as u64)
                    .ok_or(WireError::InvalidHeader(
                        "TDIG v1 total weight exceeds u64 range",
                    ))?;
            }
            // V1 decodes every unit weight as atomic. Do not export a mixed
            // summary whose bounds would contradict that reconstructed kind.
            if !td.is_empty()
                && td.centroids().iter().all(|c| c.weight_f64() == 1.0)
                && (td.min() != td.centroids()[0].mean_f64()
                    || td.max() != td.centroids()[td.centroids().len() - 1].mean_f64())
            {
                return Err(WireError::InvalidPayload(
                    "TDIG v1 unit-weight heuristic cannot represent mixed extrema; use v2 or v3",
                ));
            }
        }
        if self != Self::V3 && requires_wide_weights(td) {
            return Err(WireError::InvalidPayload(
                "f32 means with weights requiring f64 precision need TDIG v3; cast to f64 before older-format export",
            ));
        }
        Ok(())
    }

    #[inline]
    pub fn as_u8(self) -> u8 {
        match self {
            WireVersion::V1 => VERSION_V1,
            WireVersion::V2 => VERSION_V2,
            WireVersion::V3 => VERSION_V3,
        }
    }

    #[inline]
    pub fn from_u8(v: u8) -> WireResult<Self> {
        match v {
            VERSION_V1 => Ok(WireVersion::V1),
            VERSION_V2 => Ok(WireVersion::V2),
            VERSION_V3 => Ok(WireVersion::V3),
            other => Err(WireError::UnsupportedVersion(other)),
        }
    }
}

fn requires_wide_weights<F: FloatLike + FloatCore>(td: &TDigest<F>) -> bool {
    Precision::of_type::<F>() == Precision::F32
        && td
            .centroids()
            .iter()
            .any(|centroid| centroid.weight_f64() as f32 as f64 != centroid.weight_f64())
}

#[derive(Debug)]
pub enum WireDecodedDigest {
    F32(TDigest<f32>),
    F64(TDigest<f64>),
}

#[derive(Debug)]
pub enum WireError {
    InvalidMagic,
    UnsupportedVersion(u8),
    InvalidScale(u8),
    InvalidPolicy(u8),
    InvalidPrecisionCode(u8),
    InvalidFlags(u8),
    InvalidHeader(&'static str),
    InvalidPayload(&'static str),
    InvalidChecksum,
}

impl fmt::Display for WireError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use WireError::*;
        match self {
            InvalidMagic => write!(f, "invalid TDIG magic header"),
            UnsupportedVersion(v) => write!(f, "unsupported TDIG version: {v}"),
            InvalidScale(c) => write!(f, "invalid TDIG scale code: {c}"),
            InvalidPolicy(c) => write!(f, "invalid TDIG policy code: {c}"),
            InvalidPrecisionCode(c) => write!(f, "invalid TDIG precision code: {c}"),
            InvalidFlags(bits) => write!(f, "invalid TDIG flags: 0x{bits:02x}"),
            InvalidHeader(msg) => write!(f, "invalid TDIG header: {msg}"),
            InvalidPayload(msg) => write!(f, "invalid TDIG payload: {msg}"),
            InvalidChecksum => write!(f, "invalid TDIG checksum"),
        }
    }
}

impl std::error::Error for WireError {}

pub type WireResult<T> = Result<T, WireError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WireFloatWidth {
    F32,
    F64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WirePrecision {
    F32,
    F64,
}

impl WireFloatWidth {
    #[inline]
    fn to_precision(self) -> WirePrecision {
        match self {
            WireFloatWidth::F32 => WirePrecision::F32,
            WireFloatWidth::F64 => WirePrecision::F64,
        }
    }

    #[inline]
    fn to_precision_code(self) -> u8 {
        match self {
            WireFloatWidth::F32 => PRECISION_CODE_F32,
            WireFloatWidth::F64 => PRECISION_CODE_F64,
        }
    }
}

#[inline]
fn width_from_precision_code(code: u8) -> WireResult<WireFloatWidth> {
    match code {
        PRECISION_CODE_F32 => Ok(WireFloatWidth::F32),
        PRECISION_CODE_F64 => Ok(WireFloatWidth::F64),
        other => Err(WireError::InvalidPrecisionCode(other)),
    }
}

#[inline]
fn centroid_stride(version: WireVersion, width: WireFloatWidth) -> usize {
    match version {
        WireVersion::V1 => match width {
            WireFloatWidth::F32 => 4 + 8,
            WireFloatWidth::F64 => 8 + 8,
        },
        WireVersion::V2 | WireVersion::V3 => match width {
            WireFloatWidth::F32 => 4 + 8 + 1,
            WireFloatWidth::F64 => 8 + 8 + 1,
        },
    }
}

#[inline]
fn expected_payload_len(
    version: WireVersion,
    width: WireFloatWidth,
    centroid_count: usize,
) -> WireResult<usize> {
    centroid_count
        .checked_mul(centroid_stride(version, width))
        .ok_or(WireError::InvalidPayload(
            "overflow computing payload length",
        ))
}

#[derive(Debug, Clone, Copy)]
struct DecodedV3Header {
    header_len: usize,
    width: WireFloatWidth,
    scale: ScaleFamily,
    policy: SingletonPolicy,
    max_size: usize,
    total_weight: f64,
    min: f64,
    max: f64,
    centroid_count: usize,
    data_sum: f64,
    delta: Option<f64>,
}

pub fn wire_precision(bytes: &[u8]) -> WireResult<WirePrecision> {
    if bytes.len() < 5 {
        return Err(WireError::InvalidHeader("buffer too small"));
    }
    if &bytes[0..4] != MAGIC {
        return Err(WireError::InvalidMagic);
    }

    let version = WireVersion::from_u8(bytes[4])?;
    match version {
        WireVersion::V1 | WireVersion::V2 => wire_precision_v12(bytes, version),
        WireVersion::V3 => {
            let h = parse_v3_header(bytes)?;
            Ok(h.width.to_precision())
        }
    }
}

fn wire_precision_v12(bytes: &[u8], version: WireVersion) -> WireResult<WirePrecision> {
    if bytes.len() < HEADER_LEN_V12 {
        return Err(WireError::InvalidHeader("buffer too small"));
    }

    let mut offset = 40;
    let centroid_count_u64 = read_u64(bytes, &mut offset)?;
    let centroid_count = usize::try_from(centroid_count_u64)
        .map_err(|_| WireError::InvalidHeader("centroid_count exceeds platform size"))?;
    let payload_len = bytes.len() - HEADER_LEN_V12;
    if centroid_count == 0 {
        if payload_len != 0 {
            return Err(WireError::InvalidPayload(
                "empty digest has trailing payload",
            ));
        }
        return Ok(WirePrecision::F64);
    }
    let f32_len = expected_payload_len(version, WireFloatWidth::F32, centroid_count)?;
    let f64_len = expected_payload_len(version, WireFloatWidth::F64, centroid_count)?;

    if payload_len == f32_len {
        Ok(WirePrecision::F32)
    } else if payload_len == f64_len {
        Ok(WirePrecision::F64)
    } else {
        Err(WireError::InvalidPayload(
            "payload length does not match f32 or f64 layout",
        ))
    }
}

#[inline]
fn write_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

#[inline]
fn write_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

#[inline]
fn write_f64(buf: &mut Vec<u8>, v: f64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

#[inline]
fn write_f32(buf: &mut Vec<u8>, v: f32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

#[inline]
fn write_u8(buf: &mut Vec<u8>, v: u8) {
    buf.push(v);
}

#[inline]
fn read_u64(bytes: &[u8], offset: &mut usize) -> WireResult<u64> {
    if *offset + 8 > bytes.len() {
        return Err(WireError::InvalidHeader("truncated u64"));
    }
    let mut arr = [0u8; 8];
    arr.copy_from_slice(&bytes[*offset..*offset + 8]);
    *offset += 8;
    Ok(u64::from_le_bytes(arr))
}

#[inline]
fn read_u32(bytes: &[u8], offset: &mut usize) -> WireResult<u32> {
    if *offset + 4 > bytes.len() {
        return Err(WireError::InvalidHeader("truncated u32"));
    }
    let mut arr = [0u8; 4];
    arr.copy_from_slice(&bytes[*offset..*offset + 4]);
    *offset += 4;
    Ok(u32::from_le_bytes(arr))
}

#[inline]
fn read_f64(bytes: &[u8], offset: &mut usize) -> WireResult<f64> {
    if *offset + 8 > bytes.len() {
        return Err(WireError::InvalidHeader("truncated f64"));
    }
    let mut arr = [0u8; 8];
    arr.copy_from_slice(&bytes[*offset..*offset + 8]);
    *offset += 8;
    Ok(f64::from_le_bytes(arr))
}

#[inline]
fn read_u8(bytes: &[u8], offset: &mut usize) -> WireResult<u8> {
    if *offset + 1 > bytes.len() {
        return Err(WireError::InvalidPayload("truncated u8"));
    }
    let v = bytes[*offset];
    *offset += 1;
    Ok(v)
}

pub(crate) fn scale_to_code(s: ScaleFamily) -> u8 {
    match s {
        ScaleFamily::Quad => 0,
        ScaleFamily::K1 => 1,
        ScaleFamily::K2 => 2,
        ScaleFamily::K3 => 3,
        ScaleFamily::K2Norm => 4,
    }
}

pub(crate) fn code_to_scale(c: u8) -> WireResult<ScaleFamily> {
    match c {
        0 => Ok(ScaleFamily::Quad),
        1 => Ok(ScaleFamily::K1),
        2 => Ok(ScaleFamily::K2),
        3 => Ok(ScaleFamily::K3),
        4 => Ok(ScaleFamily::K2Norm),
        _ => Err(WireError::InvalidScale(c)),
    }
}

fn policy_to_code(p: &SingletonPolicy) -> (u8, u8) {
    match *p {
        SingletonPolicy::Off => (0, 0),
        SingletonPolicy::Use => (1, 0),
        SingletonPolicy::UseWithProtectedEdges(k) => {
            let k_u8 = if k > 255 { 255 } else { k as u8 };
            (2, k_u8)
        }
    }
}

fn code_to_policy(code: u8, pin_per_side: u8) -> WireResult<SingletonPolicy> {
    match code {
        0 => Ok(SingletonPolicy::Off),
        1 => Ok(SingletonPolicy::Use),
        2 => {
            if pin_per_side == 0 {
                return Err(WireError::InvalidHeader(
                    "edges policy requires pin_per_side >= 1",
                ));
            }
            Ok(SingletonPolicy::UseWithProtectedEdges(
                pin_per_side as usize,
            ))
        }
        _ => Err(WireError::InvalidPolicy(code)),
    }
}

#[inline]
fn kind_to_code<F: FloatLike + FloatCore>(c: &Centroid<F>) -> u8 {
    if c.is_atomic() {
        0
    } else {
        1
    }
}

#[inline]
fn code_is_atomic(code: u8) -> WireResult<bool> {
    match code {
        0 => Ok(true),
        1 => Ok(false),
        _ => Err(WireError::InvalidPayload("invalid centroid kind code")),
    }
}

#[inline]
fn checksum_v3(bytes: &[u8]) -> u32 {
    let mut hasher = Hasher::new();

    // The checksum covers the header, extensions and payload with the checksum field zeroed.
    hasher.update(&bytes[..V3_CHECKSUM_OFFSET]);
    hasher.update(&[0u8; 4]);
    hasher.update(&bytes[V3_CHECKSUM_OFFSET + 4..]);
    hasher.finalize()
}

pub fn encode_digest<F>(td: &TDigest<F>) -> Vec<u8>
where
    F: FloatLike + FloatCore,
{
    encode_digest_with_version(td, WireVersion::LATEST)
}

/// Encode a digest using the selected TDIG format.
///
/// # Panics
/// Panics when the selected version cannot represent the digest's state:
/// V1 requires positive integer weights and a u64 total; V1/V2 cannot store
/// delta, edge pins above 255, or f32-mean digests whose weights need f64 precision.
/// Delta always requires K2Norm and policy Off.
/// Call [`WireVersion::validate_digest`] for a
/// fallible check, or use [`encode_digest`] to select the latest format.
pub fn encode_digest_with_version<F>(td: &TDigest<F>, version: WireVersion) -> Vec<u8>
where
    F: FloatLike + FloatCore,
{
    version
        .validate_digest(td)
        .expect("selected TDIG version cannot represent digest state");
    match version {
        WireVersion::V1 | WireVersion::V2 => encode_digest_v12(td, version),
        WireVersion::V3 => encode_digest_v3(td),
    }
}

fn encode_digest_v12<F>(td: &TDigest<F>, version: WireVersion) -> Vec<u8>
where
    F: FloatLike + FloatCore,
{
    debug_assert!(matches!(version, WireVersion::V1 | WireVersion::V2));

    let n = td.centroids().len() as u64;
    let width = match Precision::of_type::<F>() {
        Precision::F32 => WireFloatWidth::F32,
        _ => WireFloatWidth::F64,
    };
    let per_centroid_len = centroid_stride(version, width);

    let mut buf = Vec::with_capacity(HEADER_LEN_V12 + per_centroid_len * (n as usize));

    // magic + version
    buf.extend_from_slice(MAGIC);
    buf.push(version.as_u8());

    // scale / policy / pin_per_side
    let scale_code = scale_to_code(td.scale());
    let (policy_code, pin_per_side) = policy_to_code(&td.singleton_policy());
    buf.push(scale_code);
    buf.push(policy_code);
    buf.push(pin_per_side);

    // max_size
    write_u64(&mut buf, td.max_size() as u64);

    let cents = td.centroids();
    if version == WireVersion::V1 {
        // validate_digest has checked exact integer conversion and total range.
        let total_weight_u64: u64 = cents.iter().map(|c| c.weight_f64() as u64).sum();
        write_u64(&mut buf, total_weight_u64);
    } else {
        // Version 2 stores exact f64 total_weight.
        write_f64(&mut buf, td.count());
    }

    // min / max / centroid_count / data_sum
    write_f64(&mut buf, td.min());
    write_f64(&mut buf, td.max());
    write_u64(&mut buf, n);
    write_f64(&mut buf, td.sum());

    debug_assert_eq!(buf.len(), HEADER_LEN_V12);

    // payload: centroids
    for c in cents {
        let mean_f64 = c.mean_f64();
        let w = c.weight_f64();

        match width {
            WireFloatWidth::F32 => {
                write_f32(&mut buf, mean_f64 as f32);
                if version == WireVersion::V1 {
                    write_u64(&mut buf, w as u64);
                } else {
                    write_f64(&mut buf, w);
                    write_u8(&mut buf, kind_to_code(c));
                }
            }
            WireFloatWidth::F64 => {
                write_f64(&mut buf, mean_f64);
                if version == WireVersion::V1 {
                    write_u64(&mut buf, w as u64);
                } else {
                    write_f64(&mut buf, w);
                    write_u8(&mut buf, kind_to_code(c));
                }
            }
        }
    }

    buf
}

fn encode_digest_v3<F>(td: &TDigest<F>) -> Vec<u8>
where
    F: FloatLike + FloatCore,
{
    let n = td.centroids().len() as u64;
    let width = match Precision::of_type::<F>() {
        Precision::F32 => WireFloatWidth::F32,
        _ => WireFloatWidth::F64,
    };

    let wide_pin = match td.singleton_policy() {
        SingletonPolicy::UseWithProtectedEdges(pin) if pin > u8::MAX as usize => Some(pin),
        _ => None,
    };
    let flags = V3_FLAG_CHECKSUM
        | if td.delta().is_some() {
            V3_FLAG_DELTA
        } else {
            0
        }
        | if wide_pin.is_some() {
            V3_FLAG_WIDE_PIN
        } else {
            0
        }
        | if requires_wide_weights(td) {
            V3_FLAG_WIDE_WEIGHT
        } else {
            0
        };
    let header_len = HEADER_LEN_V3_MIN
        + if td.delta().is_some() { 8 } else { 0 }
        + if wide_pin.is_some() { 8 } else { 0 };
    let per_centroid_len = centroid_stride(WireVersion::V3, width);

    let mut buf = Vec::with_capacity(header_len + per_centroid_len * (n as usize));

    // fixed v3 header
    buf.extend_from_slice(MAGIC); // 0..4
    buf.push(WireVersion::V3.as_u8()); // 4
    buf.push(flags); // 5
    buf.push(header_len as u8); // 6
    buf.push(width.to_precision_code()); // 7

    let scale_code = scale_to_code(td.scale());
    let (policy_code, pin_per_side) = policy_to_code(&td.singleton_policy());
    buf.push(scale_code); // 8
    buf.push(policy_code); // 9
    buf.push(pin_per_side); // 10
    buf.push(0); // 11 reserved

    write_u64(&mut buf, td.max_size() as u64); // 12..20
    write_f64(&mut buf, td.count()); // 20..28
    write_f64(&mut buf, td.min()); // 28..36
    write_f64(&mut buf, td.max()); // 36..44
    write_u64(&mut buf, n); // 44..52
    write_f64(&mut buf, td.sum()); // 52..60
    write_u32(&mut buf, 0); // 60..64 checksum placeholder
    if let Some(delta) = td.delta() {
        write_f64(&mut buf, delta); // first optional extension
    }
    if let Some(pin) = wide_pin {
        write_u64(&mut buf, pin as u64); // follows delta when present
    }

    debug_assert_eq!(buf.len(), header_len);

    // payload: same as v2 layout
    for c in td.centroids() {
        let mean_f64 = c.mean_f64();
        let w = c.weight_f64();

        match width {
            WireFloatWidth::F32 => {
                write_f32(&mut buf, mean_f64 as f32);
                write_f64(&mut buf, w);
                write_u8(&mut buf, kind_to_code(c));
            }
            WireFloatWidth::F64 => {
                write_f64(&mut buf, mean_f64);
                write_f64(&mut buf, w);
                write_u8(&mut buf, kind_to_code(c));
            }
        }
    }

    if (flags & V3_FLAG_CHECKSUM) != 0 {
        let checksum = checksum_v3(&buf);
        buf[V3_CHECKSUM_OFFSET..V3_CHECKSUM_OFFSET + 4].copy_from_slice(&checksum.to_le_bytes());
    }

    buf
}

pub fn decode_digest(bytes: &[u8]) -> WireResult<WireDecodedDigest> {
    if bytes.len() < 5 {
        return Err(WireError::InvalidHeader("buffer too small"));
    }

    if &bytes[0..4] != MAGIC {
        return Err(WireError::InvalidMagic);
    }

    let version = WireVersion::from_u8(bytes[4])?;
    match version {
        WireVersion::V1 | WireVersion::V2 => decode_digest_v12(bytes, version),
        WireVersion::V3 => decode_digest_v3(bytes),
    }
}

fn decode_digest_v12(bytes: &[u8], version: WireVersion) -> WireResult<WireDecodedDigest> {
    if bytes.len() < HEADER_LEN_V12 {
        return Err(WireError::InvalidHeader("buffer too small"));
    }

    let scale_code = bytes[5];
    let policy_code = bytes[6];
    let pin_per_side = bytes[7];

    let mut offset = 8;

    let max_size_u64 = read_u64(bytes, &mut offset)?;
    let total_weight = if version == WireVersion::V1 {
        read_u64(bytes, &mut offset)? as f64
    } else {
        let tw = read_f64(bytes, &mut offset)?;
        if !tw.is_finite() || tw < 0.0 {
            return Err(WireError::InvalidHeader("invalid total_weight"));
        }
        tw
    };
    let min = read_f64(bytes, &mut offset)?;
    let max = read_f64(bytes, &mut offset)?;
    let centroid_count_u64 = read_u64(bytes, &mut offset)?;
    let data_sum = read_f64(bytes, &mut offset)?;

    if offset != HEADER_LEN_V12 {
        return Err(WireError::InvalidHeader("header length mismatch"));
    }

    let max_size = usize::try_from(max_size_u64)
        .map_err(|_| WireError::InvalidHeader("max_size exceeds platform size"))?;
    let centroid_count = usize::try_from(centroid_count_u64)
        .map_err(|_| WireError::InvalidHeader("centroid_count exceeds platform size"))?;

    let scale = code_to_scale(scale_code)?;
    let policy = code_to_policy(policy_code, pin_per_side)?;

    let payload_len = bytes
        .len()
        .checked_sub(HEADER_LEN_V12)
        .ok_or(WireError::InvalidHeader("buffer too small"))?;

    let width = if centroid_count == 0 {
        if payload_len != 0 {
            return Err(WireError::InvalidPayload(
                "empty digest has trailing payload",
            ));
        }
        WireFloatWidth::F64
    } else {
        let f32_len = expected_payload_len(version, WireFloatWidth::F32, centroid_count)?;
        let f64_len = expected_payload_len(version, WireFloatWidth::F64, centroid_count)?;

        if payload_len == f32_len {
            WireFloatWidth::F32
        } else if payload_len == f64_len {
            WireFloatWidth::F64
        } else {
            return Err(WireError::InvalidPayload(
                "payload length does not match f32 or f64 layout",
            ));
        }
    };

    let stats = DigestStats {
        data_sum,
        total_weight,
        data_min: min,
        data_max: max,
    };

    decode_payload_into_digest(
        bytes,
        HEADER_LEN_V12,
        centroid_count,
        width,
        version,
        max_size,
        scale,
        policy,
        stats,
    )
}

fn parse_v3_header(bytes: &[u8]) -> WireResult<DecodedV3Header> {
    if bytes.len() < HEADER_LEN_V3_MIN {
        return Err(WireError::InvalidHeader("buffer too small"));
    }

    let flags = bytes[5];
    if (flags & !(V3_FLAG_CHECKSUM | V3_FLAG_DELTA | V3_FLAG_WIDE_PIN | V3_FLAG_WIDE_WEIGHT)) != 0 {
        return Err(WireError::InvalidFlags(flags));
    }

    let header_len = bytes[6] as usize;
    if header_len < HEADER_LEN_V3_MIN {
        return Err(WireError::InvalidHeader("v3 header_len must be >= 64"));
    }
    if bytes.len() < header_len {
        return Err(WireError::InvalidHeader(
            "buffer smaller than v3 header_len",
        ));
    }

    let width = width_from_precision_code(bytes[7])?;
    if flags & V3_FLAG_WIDE_WEIGHT != 0 && !matches!(width, WireFloatWidth::F32) {
        return Err(WireError::InvalidHeader(
            "wide-weight flag requires f32 mean precision",
        ));
    }

    let delta = if flags & V3_FLAG_DELTA != 0 {
        if header_len < HEADER_LEN_V3_MIN + 8 {
            return Err(WireError::InvalidHeader("missing delta extension"));
        }
        let mut delta_offset = HEADER_LEN_V3_MIN;
        let delta = read_f64(bytes, &mut delta_offset)?;
        if !delta.is_finite() || delta <= 0.0 {
            return Err(WireError::InvalidHeader("delta must be finite and > 0"));
        }
        Some(delta)
    } else {
        None
    };

    let scale = code_to_scale(bytes[8])?;
    let mut policy = code_to_policy(bytes[9], bytes[10])?;
    if flags & V3_FLAG_WIDE_PIN != 0 {
        let mut pin_offset = HEADER_LEN_V3_MIN + if delta.is_some() { 8 } else { 0 };
        if header_len < pin_offset + 8 {
            return Err(WireError::InvalidHeader("missing pin_per_side extension"));
        }
        if bytes[9] != 2 || bytes[10] != u8::MAX {
            return Err(WireError::InvalidHeader(
                "pin_per_side extension requires edges policy and pin byte 255",
            ));
        }
        let pin_u64 = read_u64(bytes, &mut pin_offset)?;
        let pin = usize::try_from(pin_u64)
            .map_err(|_| WireError::InvalidHeader("pin_per_side exceeds platform size"))?;
        if pin <= u8::MAX as usize {
            return Err(WireError::InvalidHeader(
                "full-width pin_per_side must exceed 255",
            ));
        }
        policy = SingletonPolicy::UseWithProtectedEdges(pin);
    }
    if delta.is_some() && (scale != ScaleFamily::K2Norm || policy != SingletonPolicy::Off) {
        return Err(WireError::InvalidHeader(
            "delta mode requires K2Norm scale and singleton policy Off",
        ));
    }

    let mut offset = 12;

    let max_size_u64 = read_u64(bytes, &mut offset)?;
    let total_weight = read_f64(bytes, &mut offset)?;
    if !total_weight.is_finite() || total_weight < 0.0 {
        return Err(WireError::InvalidHeader("invalid total_weight"));
    }

    let min = read_f64(bytes, &mut offset)?;
    let max = read_f64(bytes, &mut offset)?;
    let centroid_count_u64 = read_u64(bytes, &mut offset)?;
    let data_sum = read_f64(bytes, &mut offset)?;

    if offset != V3_CHECKSUM_OFFSET {
        return Err(WireError::InvalidHeader("v3 fixed header parse mismatch"));
    }

    let mut checksum_offset = V3_CHECKSUM_OFFSET;
    let expected_checksum = read_u32(bytes, &mut checksum_offset)?;

    if (flags & V3_FLAG_CHECKSUM) != 0 {
        let actual_checksum = checksum_v3(bytes);
        if actual_checksum != expected_checksum {
            return Err(WireError::InvalidChecksum);
        }
    }

    let centroid_count = usize::try_from(centroid_count_u64)
        .map_err(|_| WireError::InvalidHeader("centroid_count exceeds platform size"))?;
    let payload_len = bytes
        .len()
        .checked_sub(header_len)
        .ok_or(WireError::InvalidHeader(
            "buffer smaller than v3 header_len",
        ))?;
    let expected_payload_len = expected_payload_len(WireVersion::V3, width, centroid_count)?;
    if payload_len != expected_payload_len {
        return Err(WireError::InvalidPayload(
            "payload length does not match v3 precision layout",
        ));
    }

    Ok(DecodedV3Header {
        header_len,
        width,
        scale,
        policy,
        max_size: usize::try_from(max_size_u64)
            .map_err(|_| WireError::InvalidHeader("max_size exceeds platform size"))?,
        total_weight,
        min,
        max,
        centroid_count,
        data_sum,
        delta,
    })
}

fn decode_digest_v3(bytes: &[u8]) -> WireResult<WireDecodedDigest> {
    let h = parse_v3_header(bytes)?;

    let stats = DigestStats {
        data_sum: h.data_sum,
        total_weight: h.total_weight,
        data_min: h.min,
        data_max: h.max,
    };

    decode_payload_into_digest(
        bytes,
        h.header_len,
        h.centroid_count,
        h.width,
        WireVersion::V3,
        h.max_size,
        h.scale,
        h.policy,
        stats,
    )
    .map(|mut decoded| {
        match &mut decoded {
            WireDecodedDigest::F32(td) => td.set_delta(h.delta),
            WireDecodedDigest::F64(td) => td.set_delta(h.delta),
        }
        decoded
    })
}

fn validate_decoded_weight(weight: f64) -> WireResult<()> {
    if !weight.is_finite() || weight <= 0.0 {
        return Err(WireError::InvalidPayload(
            "centroid weight must be finite and positive",
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn decode_payload_into_digest(
    bytes: &[u8],
    payload_start: usize,
    centroid_count: usize,
    width: WireFloatWidth,
    version: WireVersion,
    max_size: usize,
    scale: ScaleFamily,
    policy: SingletonPolicy,
    stats: DigestStats,
) -> WireResult<WireDecodedDigest> {
    match width {
        WireFloatWidth::F32 => {
            let mut cents: Vec<Centroid<f32>> = Vec::with_capacity(centroid_count);
            let mut p = payload_start;

            for _ in 0..centroid_count {
                if p + 4 > bytes.len() {
                    return Err(WireError::InvalidPayload("truncated f32 mean"));
                }
                let mut arr_m = [0u8; 4];
                arr_m.copy_from_slice(&bytes[p..p + 4]);
                p += 4;
                let mean_f32 = f32::from_le_bytes(arr_m);
                let mean_f64 = mean_f32 as f64;

                let c = if version == WireVersion::V1 {
                    if p + 8 > bytes.len() {
                        return Err(WireError::InvalidPayload("truncated weight u64"));
                    }
                    let mut arr_w = [0u8; 8];
                    arr_w.copy_from_slice(&bytes[p..p + 8]);
                    p += 8;
                    let w_u64 = u64::from_le_bytes(arr_w);
                    let w_f64 = w_u64 as f64;
                    validate_decoded_weight(w_f64)?;
                    if w_u64 == 1 {
                        Centroid::<f32>::new_atomic_unit_f64(mean_f64)
                    } else {
                        Centroid::<f32>::new_mixed_f64(mean_f64, w_f64)
                    }
                } else {
                    let w_f64 = read_f64(bytes, &mut p)?;
                    validate_decoded_weight(w_f64)?;
                    let kind = read_u8(bytes, &mut p)?;
                    if code_is_atomic(kind)? {
                        if w_f64 == 1.0 {
                            Centroid::<f32>::new_atomic_unit_f64(mean_f64)
                        } else {
                            Centroid::<f32>::new_atomic_f64(mean_f64, w_f64)
                        }
                    } else {
                        Centroid::<f32>::new_mixed_f64(mean_f64, w_f64)
                    }
                };
                cents.push(c);
            }

            let td = TDigest::<f32>::builder()
                .max_size(max_size)
                .scale(scale)
                .singleton_policy(policy)
                .with_centroids_and_stats(cents, stats)
                .build()
                .map_err(|_| WireError::InvalidPayload("invalid digest state"))?;
            Ok(WireDecodedDigest::F32(td))
        }
        WireFloatWidth::F64 => {
            let mut cents: Vec<Centroid<f64>> = Vec::with_capacity(centroid_count);
            let mut p = payload_start;

            for _ in 0..centroid_count {
                if p + 8 > bytes.len() {
                    return Err(WireError::InvalidPayload("truncated f64 mean"));
                }
                let mut arr_m = [0u8; 8];
                arr_m.copy_from_slice(&bytes[p..p + 8]);
                p += 8;
                let mean_f64 = f64::from_le_bytes(arr_m);

                let c = if version == WireVersion::V1 {
                    if p + 8 > bytes.len() {
                        return Err(WireError::InvalidPayload("truncated weight u64"));
                    }
                    let mut arr_w = [0u8; 8];
                    arr_w.copy_from_slice(&bytes[p..p + 8]);
                    p += 8;
                    let w_u64 = u64::from_le_bytes(arr_w);
                    let w_f64 = w_u64 as f64;
                    validate_decoded_weight(w_f64)?;
                    if w_u64 == 1 {
                        Centroid::<f64>::new_atomic_unit_f64(mean_f64)
                    } else {
                        Centroid::<f64>::new_mixed_f64(mean_f64, w_f64)
                    }
                } else {
                    let w_f64 = read_f64(bytes, &mut p)?;
                    validate_decoded_weight(w_f64)?;
                    let kind = read_u8(bytes, &mut p)?;
                    if code_is_atomic(kind)? {
                        if w_f64 == 1.0 {
                            Centroid::<f64>::new_atomic_unit_f64(mean_f64)
                        } else {
                            Centroid::<f64>::new_atomic_f64(mean_f64, w_f64)
                        }
                    } else {
                        Centroid::<f64>::new_mixed_f64(mean_f64, w_f64)
                    }
                };
                cents.push(c);
            }

            let td = TDigest::<f64>::builder()
                .max_size(max_size)
                .scale(scale)
                .singleton_policy(policy)
                .with_centroids_and_stats(cents, stats)
                .build()
                .map_err(|_| WireError::InvalidPayload("invalid digest state"))?;
            Ok(WireDecodedDigest::F64(td))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delta_extension_roundtrips_and_rejects_corruption() {
        let td = TDigest::<f64>::builder()
            .scale(ScaleFamily::K2Norm)
            .singleton_policy(SingletonPolicy::Off)
            .delta(17.0)
            .build()
            .unwrap()
            .merge_unsorted(vec![0.0, 10.0])
            .unwrap();
        let blob = encode_digest(&td);
        let WireDecodedDigest::F64(restored) = decode_digest(&blob).unwrap() else {
            panic!("precision changed")
        };
        assert_eq!(restored.delta(), Some(17.0));
        assert_eq!(encode_digest(&restored), blob);
        let mut corrupted = blob.clone();
        corrupted[64] ^= 1;
        assert!(matches!(
            decode_digest(&corrupted),
            Err(WireError::InvalidChecksum)
        ));
        let mut truncated_header = blob.clone();
        truncated_header[6] = 64;
        assert!(matches!(
            decode_digest(&truncated_header),
            Err(WireError::InvalidHeader(_))
        ));
        for value in [f64::NAN, f64::INFINITY, 0.0, -1.0] {
            let mut invalid = blob.clone();
            invalid[64..72].copy_from_slice(&value.to_le_bytes());
            assert!(matches!(
                decode_digest(&invalid),
                Err(WireError::InvalidHeader(_))
            ));
        }
        assert!(WireVersion::V1.validate_delta(Some(17.0)).is_err());
        assert!(WireVersion::V2.validate_delta(Some(17.0)).is_err());
    }

    fn sample_digest() -> TDigest<f64> {
        let mut td = TDigest::<f64>::builder()
            .max_size(128)
            .scale(ScaleFamily::K3)
            .singleton_policy(SingletonPolicy::UseWithProtectedEdges(2))
            .build()
            .unwrap();
        td.add_many(vec![-3.0, -1.0, 0.0, 0.25, 10.0, 11.0])
            .expect("seed");
        td.add_weighted_many(&[1.5, 2.5], &[2.0, 3.0])
            .expect("weighted");
        td
    }

    #[test]
    fn v1_v2_v3_decode_roundtrip() {
        let td = sample_digest();

        for version in [WireVersion::V1, WireVersion::V2, WireVersion::V3] {
            let blob = encode_digest_with_version(&td, version);
            let decoded = decode_digest(&blob).expect("decode");
            match decoded {
                WireDecodedDigest::F64(d) => {
                    assert_eq!(d.max_size(), td.max_size());
                    assert_eq!(d.scale(), td.scale());
                    assert_eq!(d.singleton_policy(), td.singleton_policy());
                }
                WireDecodedDigest::F32(_) => panic!("expected f64 payload"),
            }
        }
    }

    #[test]
    fn v3_wire_precision_uses_header_code() {
        let td = sample_digest();
        let blob = encode_digest_with_version(&td, WireVersion::V3);
        assert_eq!(
            wire_precision(&blob).expect("precision"),
            WirePrecision::F64
        );
    }

    #[test]
    fn v3_checksum_detects_corruption() {
        let td = sample_digest();
        let mut blob = encode_digest_with_version(&td, WireVersion::V3);
        let payload_start = HEADER_LEN_V3_MIN;
        blob[payload_start] ^= 0x01;

        let err = decode_digest(&blob).expect_err("must fail checksum");
        assert!(matches!(err, WireError::InvalidChecksum));
    }

    #[test]
    fn v3_header_len_and_precision_code_are_present() {
        let td = sample_digest();
        let blob = encode_digest_with_version(&td, WireVersion::V3);

        assert_eq!(&blob[0..4], MAGIC);
        assert_eq!(blob[4], VERSION_V3);
        assert_eq!(blob[6] as usize, HEADER_LEN_V3_MIN);
        assert_eq!(blob[7], PRECISION_CODE_F64);
    }
}
