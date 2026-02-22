// src/tdigest/wire.rs
//
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
//   5      : flags (bit0: checksum present)
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
    flags: u8,
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
            let h = parse_v3_header(bytes, true)?;
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
    let centroid_count = centroid_count_u64 as usize;

    if centroid_count == 0 {
        return Ok(WirePrecision::F64);
    }

    let payload_len = bytes
        .len()
        .checked_sub(HEADER_LEN_V12)
        .ok_or(WireError::InvalidHeader("buffer too small"))?;

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

/* ============================
 * Small helpers
 * ============================ */

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

fn scale_to_code(s: ScaleFamily) -> u8 {
    match s {
        ScaleFamily::Quad => 0,
        ScaleFamily::K1 => 1,
        ScaleFamily::K2 => 2,
        ScaleFamily::K3 => 3,
    }
}

fn code_to_scale(c: u8) -> WireResult<ScaleFamily> {
    match c {
        0 => Ok(ScaleFamily::Quad),
        1 => Ok(ScaleFamily::K1),
        2 => Ok(ScaleFamily::K2),
        3 => Ok(ScaleFamily::K3),
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
fn checksum_v3(bytes: &[u8], header_len: usize) -> u32 {
    let mut hasher = Hasher::new();

    // bytes before checksum slot
    hasher.update(&bytes[..V3_CHECKSUM_OFFSET]);
    // checksum slot treated as zeros
    hasher.update(&[0u8; 4]);
    // any v3 header extension bytes
    if header_len > V3_CHECKSUM_OFFSET + 4 {
        hasher.update(&bytes[V3_CHECKSUM_OFFSET + 4..header_len]);
    }
    // payload
    hasher.update(&bytes[header_len..]);

    hasher.finalize()
}

/* ============================
 * Encode
 * ============================ */

pub fn encode_digest<F>(td: &TDigest<F>) -> Vec<u8>
where
    F: FloatLike + FloatCore,
{
    encode_digest_with_version(td, WireVersion::LATEST)
}

pub fn encode_digest_with_version<F>(td: &TDigest<F>, version: WireVersion) -> Vec<u8>
where
    F: FloatLike + FloatCore,
{
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
        // Backward-compatible integerized weight header.
        let mut total_weight_u64: u64 = 0;
        for c in cents {
            let w = c.weight_f64();
            let w_rounded = w.round();
            let w_u64 = if w_rounded <= 0.0 {
                0
            } else if w_rounded > u64::MAX as f64 {
                u64::MAX
            } else {
                w_rounded as u64
            };
            total_weight_u64 = total_weight_u64.saturating_add(w_u64);
        }
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
                    let w_rounded = w.round();
                    let w_u64 = if w_rounded <= 0.0 {
                        0
                    } else if w_rounded > u64::MAX as f64 {
                        u64::MAX
                    } else {
                        w_rounded as u64
                    };
                    write_u64(&mut buf, w_u64);
                } else {
                    write_f64(&mut buf, w);
                    write_u8(&mut buf, kind_to_code(c));
                }
            }
            WireFloatWidth::F64 => {
                write_f64(&mut buf, mean_f64);
                if version == WireVersion::V1 {
                    let w_rounded = w.round();
                    let w_u64 = if w_rounded <= 0.0 {
                        0
                    } else if w_rounded > u64::MAX as f64 {
                        u64::MAX
                    } else {
                        w_rounded as u64
                    };
                    write_u64(&mut buf, w_u64);
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

    let flags = V3_FLAG_CHECKSUM;
    let header_len = HEADER_LEN_V3_MIN;
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
        let checksum = checksum_v3(&buf, header_len);
        buf[V3_CHECKSUM_OFFSET..V3_CHECKSUM_OFFSET + 4].copy_from_slice(&checksum.to_le_bytes());
    }

    buf
}

/* ============================
 * Decode
 * ============================ */
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

    let max_size = max_size_u64 as usize;
    let centroid_count = centroid_count_u64 as usize;

    let scale = code_to_scale(scale_code)?;
    let policy = code_to_policy(policy_code, pin_per_side)?;

    let payload_len = bytes
        .len()
        .checked_sub(HEADER_LEN_V12)
        .ok_or(WireError::InvalidHeader("buffer too small"))?;

    let width = if centroid_count == 0 {
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

fn parse_v3_header(bytes: &[u8], verify_checksum: bool) -> WireResult<DecodedV3Header> {
    if bytes.len() < HEADER_LEN_V3_MIN {
        return Err(WireError::InvalidHeader("buffer too small"));
    }

    let flags = bytes[5];
    if (flags & !V3_FLAG_CHECKSUM) != 0 {
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

    let scale = code_to_scale(bytes[8])?;
    let policy = code_to_policy(bytes[9], bytes[10])?;

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

    if (flags & V3_FLAG_CHECKSUM) != 0 && verify_checksum {
        let actual_checksum = checksum_v3(bytes, header_len);
        if actual_checksum != expected_checksum {
            return Err(WireError::InvalidChecksum);
        }
    }

    let centroid_count = centroid_count_u64 as usize;
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
        flags,
        header_len,
        width,
        scale,
        policy,
        max_size: max_size_u64 as usize,
        total_weight,
        min,
        max,
        centroid_count,
        data_sum,
    })
}

fn decode_digest_v3(bytes: &[u8]) -> WireResult<WireDecodedDigest> {
    let h = parse_v3_header(bytes, true)?;

    let _ = h.flags; // reserved for future behavior branching

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
                    if w_u64 == 1 {
                        Centroid::<f32>::new_atomic_unit_f64(mean_f64)
                    } else {
                        Centroid::<f32>::new_mixed_f64(mean_f64, w_f64)
                    }
                } else {
                    let w_f64 = read_f64(bytes, &mut p)?;
                    if !w_f64.is_finite() || w_f64 <= 0.0 {
                        return Err(WireError::InvalidPayload("invalid centroid weight"));
                    }
                    let kind = read_u8(bytes, &mut p)?;
                    if code_is_atomic(kind)? {
                        if (w_f64 - 1.0).abs() <= f64::EPSILON {
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
                .build();

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
                    if w_u64 == 1 {
                        Centroid::<f64>::new_atomic_unit_f64(mean_f64)
                    } else {
                        Centroid::<f64>::new_mixed_f64(mean_f64, w_f64)
                    }
                } else {
                    let w_f64 = read_f64(bytes, &mut p)?;
                    if !w_f64.is_finite() || w_f64 <= 0.0 {
                        return Err(WireError::InvalidPayload("invalid centroid weight"));
                    }
                    let kind = read_u8(bytes, &mut p)?;
                    if code_is_atomic(kind)? {
                        if (w_f64 - 1.0).abs() <= f64::EPSILON {
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
                .build();

            Ok(WireDecodedDigest::F64(td))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_digest() -> TDigest<f64> {
        let mut td = TDigest::<f64>::builder()
            .max_size(128)
            .scale(ScaleFamily::K3)
            .singleton_policy(SingletonPolicy::UseWithProtectedEdges(2))
            .build();
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
