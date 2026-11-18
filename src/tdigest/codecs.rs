//! Encode digests as Polars struct columns and read them at their stored precision.
//!
//! # Design
//! - Means and extrema follow `F`; centroid weights, `sum`, and `count` are always `f64`.
//! - Reader requires the mean/extrema dtype to match `F` (no lossy auto-cast).
//! - Old f32-weight structs remain readable: their weights promote exactly to f64.
//!
//! # Public API on `TDigest<F>`
//! - `td.to_series(name)           -> polars::prelude::Series`
//! - `TDigest::<F>::from_series(series) -> Vec<TDigest<F>>`
//! - `TDigest::<F>::polars_dtype() -> polars::prelude::DataType` (exact struct dtype for `F`)
//!
//! # Errors & strictness
//! `from_series` returns an error if the wire dtype does not match the expected `F`
//! (e.g., trying to read an `f32` wire digest into `TDigest<f64>`). This prevents
//! silent precision changes.
//!
//! # Example
//! ```rust
//! use polars::prelude::*;
//! use tdigest_rs::tdigest::TDigest;
//!
//! let td: TDigest<f64> = TDigest::builder().max_size(256).build().unwrap();
//! let series: Series = td.to_series("td")?;
//! let round_tripped: Vec<TDigest<f64>> = TDigest::<f64>::from_series(&series)?;
//!
//! let dtype = TDigest::<f64>::polars_dtype();
//! assert!(matches!(dtype, DataType::Struct(_)));
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use std::fmt;

use ordered_float::FloatCore;
use polars::prelude::*;

use crate::tdigest::centroids::Centroid;
use crate::tdigest::precision::FloatLike;
use crate::tdigest::singleton_policy::SingletonPolicy;
use crate::tdigest::wire::{code_to_scale, scale_to_code};
use crate::tdigest::{DigestStats, ScaleFamily, TDigest};

pub const F_CENTROIDS: &str = "centroids";
pub const F_MEAN: &str = "mean";
pub const F_WEIGHT: &str = "weight";
pub const F_SUM: &str = "sum";
pub const F_MIN: &str = "min";
pub const F_MAX: &str = "max";
pub const F_COUNT: &str = "count";
pub const F_MAX_SIZE: &str = "max_size";
pub const F_ATOMIC: &str = "atomic";
pub const F_SCALE: &str = "scale";
pub const F_POLICY: &str = "policy";
pub const F_PIN: &str = "pin_per_side";
pub const F_DELTA: &str = "delta";

#[non_exhaustive]
#[derive(Debug)]
pub enum CodecError {
    NotAStruct {
        series_name: String,
    },
    MissingField(&'static str),
    InvalidField(&'static str),
    InvalidConfiguration(&'static str),
    WrongDtype {
        field: &'static str,
        found: Box<DataType>,
        expected: Box<DataType>,
    },
    BadCentroidLayout {
        found: Box<DataType>,
    },
    MixedCentroidDtypes, // unsupported mean/weight pair
    WirePrecisionMismatch {
        expected: DataType,
    },
    NullField {
        field: &'static str,
        row: usize,
    },
    NullCentroid {
        row: usize,
        index: usize,
    },
}

impl fmt::Display for CodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use CodecError::*;
        match self {
            NotAStruct { series_name } => {
                write!(f, "series '{series_name}' is not a struct column")
            }
            MissingField(field) => write!(f, "missing field '{field}'"),
            InvalidField(field) => write!(f, "invalid field '{field}'"),
            InvalidConfiguration(message) => write!(f, "{message}"),
            WrongDtype {
                field,
                found,
                expected,
            } => write!(
                f,
                "field '{field}' has dtype {found:?}, expected {expected:?}"
            ),
            BadCentroidLayout { found } => write!(
                f,
                "centroids must be List<Struct{{mean,weight}}>, found {found:?}"
            ),
            MixedCentroidDtypes => {
                write!(f, "centroid means must be f32/f64 with f64 weights (old f32/f32 is also accepted)")
            }
            WirePrecisionMismatch { expected } => write!(
                f,
                "wire precision mismatch (expected means/min/max as {expected:?})"
            ),
            NullField { field, row } => write!(f, "null in scalar field '{field}' at row {row}"),
            NullCentroid { row, index } => {
                write!(f, "null in centroids at row {row}, index {index}")
            }
        }
    }
}

impl std::error::Error for CodecError {}

/// Trait over Polars float families (Float32Type / Float64Type).
pub trait PlFloat: PolarsNumericType + 'static {
    const DTYPE: DataType;

    fn series_from_vec(name: &str, v: Vec<<Self as PolarsNumericType>::Native>) -> Series;

    fn as_chunked(s: &Series, field: &'static str) -> Result<ChunkedArray<Self>, CodecError>;

    fn from_f64(x: f64) -> <Self as PolarsNumericType>::Native;
    fn to_f64(x: <Self as PolarsNumericType>::Native) -> f64;
}

impl PlFloat for Float32Type {
    const DTYPE: DataType = DataType::Float32;

    fn series_from_vec(name: &str, v: Vec<f32>) -> Series {
        Series::new(name.into(), v)
    }

    fn as_chunked(s: &Series, field: &'static str) -> Result<ChunkedArray<Self>, CodecError> {
        s.f32().cloned().map_err(|_| CodecError::WrongDtype {
            field,
            found: Box::new(s.dtype().clone()),
            expected: Box::new(Self::DTYPE.clone()),
        })
    }

    #[inline]
    fn from_f64(x: f64) -> f32 {
        x as f32
    }
    #[inline]
    fn to_f64(x: f32) -> f64 {
        x as f64
    }
}

impl PlFloat for Float64Type {
    const DTYPE: DataType = DataType::Float64;

    fn series_from_vec(name: &str, v: Vec<f64>) -> Series {
        Series::new(name.into(), v)
    }

    fn as_chunked(s: &Series, field: &'static str) -> Result<ChunkedArray<Self>, CodecError> {
        s.f64().cloned().map_err(|_| CodecError::WrongDtype {
            field,
            found: Box::new(s.dtype().clone()),
            expected: Box::new(Self::DTYPE.clone()),
        })
    }

    #[inline]
    fn from_f64(x: f64) -> f64 {
        x
    }
    #[inline]
    fn to_f64(x: f64) -> f64 {
        x
    }
}

/// Map digest scalar type `F` → which Polars float family to use on the wire.
pub trait WireOf {
    type Pl: PlFloat;
}
impl WireOf for f32 {
    type Pl = Float32Type;
}
impl WireOf for f64 {
    type Pl = Float64Type;
}

impl<F> TDigest<F>
where
    F: FloatLike + FloatCore + WireOf,
{
    /// Encode; means/min/max follow `F`, while weights always retain f64 precision.
    pub fn to_series(&self, name: &str) -> PolarsResult<Series> {
        tdigest_to_series::<F, <F as WireOf>::Pl>(self, name)
    }

    /// Decode strictly: incoming wire must match `<F as WireOf>::Pl::DTYPE`.
    pub fn from_series(input: &Series) -> Result<Vec<TDigest<F>>, CodecError> {
        parse_tdigests::<F, <F as WireOf>::Pl>(input)
    }

    /// Exact struct dtype for `F` (useful for expr planning).
    pub fn polars_dtype() -> DataType {
        let num = <<F as WireOf>::Pl as PlFloat>::DTYPE.clone();
        DataType::Struct(vec![
            Field::new(
                F_CENTROIDS.into(),
                DataType::List(Box::new(DataType::Struct(vec![
                    Field::new(F_MEAN.into(), num.clone()),
                    Field::new(F_WEIGHT.into(), DataType::Float64),
                    Field::new(F_ATOMIC.into(), DataType::Boolean),
                ]))),
            ),
            Field::new(F_SUM.into(), DataType::Float64),
            Field::new(F_MIN.into(), num.clone()),
            Field::new(F_MAX.into(), num),
            Field::new(F_COUNT.into(), DataType::Float64),
            Field::new(F_MAX_SIZE.into(), DataType::Int64),
            Field::new(F_SCALE.into(), DataType::Int64),
            Field::new(F_POLICY.into(), DataType::Int64),
            Field::new(F_PIN.into(), DataType::Int64),
            Field::new(F_DELTA.into(), DataType::Float64),
        ])
    }
}

fn tdigest_to_series<F, N>(td: &TDigest<F>, name: &str) -> PolarsResult<Series>
where
    F: FloatLike + FloatCore,
    N: PlFloat,
{
    td.validate_state()
        .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
    let max_size = i64::try_from(td.max_size())
        .map_err(|_| PolarsError::ComputeError("capacity exceeds Polars Int64 range".into()))?;
    let means = td
        .centroids()
        .iter()
        .map(|c| N::from_f64(c.mean_f64()))
        .collect();
    let wgts_f64: Vec<f64> = td.centroids().iter().map(|c| c.weight_f64()).collect();
    let mean_s = N::series_from_vec(F_MEAN, means);
    let weight_s = Series::new(F_WEIGHT.into(), wgts_f64);
    let atomic_s = Series::new(
        F_ATOMIC.into(),
        td.centroids()
            .iter()
            .map(|c| c.is_atomic())
            .collect::<Vec<_>>(),
    );
    let inner = StructChunked::from_series(
        "centroid".into(),
        mean_s.len(),
        [&mean_s, &weight_s, &atomic_s].into_iter(),
    )?
    .into_series();
    let centroids_s = Series::new(F_CENTROIDS.into(), &[inner]);

    // Scalar fields
    let sum_s = Series::new(F_SUM.into(), [td.sum()]);
    let min_s = N::series_from_vec(F_MIN, vec![N::from_f64(td.min())]);
    let max_s = N::series_from_vec(F_MAX, vec![N::from_f64(td.max())]);
    let count_s = Series::new(F_COUNT.into(), [td.count()]);
    let max_size_s = Series::new(F_MAX_SIZE.into(), [max_size]);
    let scale_s = Series::new(F_SCALE.into(), [scale_to_code(td.scale()) as i64]);
    let (policy, pin) = match td.singleton_policy() {
        SingletonPolicy::Off => (0_i64, 0_i64),
        SingletonPolicy::Use => (1, 0),
        SingletonPolicy::UseWithProtectedEdges(k) => (
            2,
            i64::try_from(k).map_err(|_| {
                PolarsError::ComputeError("pin_per_side exceeds Polars Int64 range".into())
            })?,
        ),
    };
    let policy_s = Series::new(F_POLICY.into(), [policy]);
    let pin_s = Series::new(F_PIN.into(), [pin]);
    let delta_s = Series::new(F_DELTA.into(), [td.delta()]);

    // Outer struct (single row)
    StructChunked::from_series(
        name.into(),
        1,
        [
            &centroids_s,
            &sum_s,
            &min_s,
            &max_s,
            &count_s,
            &max_size_s,
            &scale_s,
            &policy_s,
            &pin_s,
            &delta_s,
        ]
        .into_iter(),
    )
    .map(|sc| sc.into_series())
}

fn parse_tdigests<F, N>(input: &Series) -> Result<Vec<TDigest<F>>, CodecError>
where
    F: FloatLike + FloatCore,
    N: PlFloat,
{
    if input.null_count() != 0 {
        return Err(CodecError::InvalidConfiguration(
            "missing digest rows are not allowed",
        ));
    }
    let s = input.struct_().map_err(|_| CodecError::NotAStruct {
        series_name: input.name().to_string(),
    })?;
    let config_fields = [F_SCALE, F_POLICY, F_PIN, F_DELTA];
    let has_config = config_fields
        .iter()
        .any(|name| s.field_by_name(name).is_ok());
    if has_config {
        for name in config_fields {
            let _ = field(s, name)?;
        }
    }

    // Required fields
    let centroids = field(s, F_CENTROIDS)?;
    let sum = field(s, F_SUM)?;
    let min = field(s, F_MIN)?;
    let max = field(s, F_MAX)?;
    let count = field(s, F_COUNT)?;
    let max_size = field(s, F_MAX_SIZE)?;

    // Fixed dtypes
    expect_dtype(&sum, DataType::Float64, F_SUM)?;
    expect_dtype(&count, DataType::Float64, F_COUNT)?;
    expect_dtype(&max_size, DataType::Int64, F_MAX_SIZE)?;

    // Verify centroid layout and dtype match N::DTYPE; enforce min/max to N::DTYPE.
    ensure_centroid_dtype::<N>(&centroids)?;
    expect_dtype(&min, N::DTYPE.clone(), F_MIN)?;
    expect_dtype(&max, N::DTYPE.clone(), F_MAX)?;

    let centroids_list = centroids
        .list()
        .map_err(|_| CodecError::BadCentroidLayout {
            found: Box::new(centroids.dtype().clone()),
        })?;
    let sum_col = sum.f64().unwrap();
    let count_col = count.f64().unwrap();
    let maxsz_col = max_size.i64().unwrap();

    let n = input.len();
    let mut out = Vec::with_capacity(n);

    for i in 0..n {
        let data_sum = sum_col.get(i).ok_or(CodecError::NullField {
            field: F_SUM,
            row: i,
        })?;
        let total_weight = count_col.get(i).ok_or(CodecError::NullField {
            field: F_COUNT,
            row: i,
        })?;
        let max_size_i64 = maxsz_col.get(i).ok_or(CodecError::NullField {
            field: F_MAX_SIZE,
            row: i,
        })?;
        let max_size = usize::try_from(max_size_i64)
            .ok()
            .filter(|value| *value > 0)
            .ok_or(CodecError::InvalidField(F_MAX_SIZE))?;

        let data_min = read_scalar::<N>(&min, F_MIN, i)?;
        let data_max = read_scalar::<N>(&max, F_MAX, i)?;
        let cents = unpack_centroids::<N, F>(centroids_list, i)?;

        let mut builder = TDigest::<F>::builder().max_size(max_size);
        // Older structs omitted configuration and kind; retain their historical defaults.
        if has_config {
            let integer = |name| -> Result<i64, CodecError> {
                let column = field(s, name)?;
                expect_dtype(&column, DataType::Int64, name)?;
                column.i64().unwrap().get(i).ok_or(CodecError::NullField {
                    field: name,
                    row: i,
                })
            };
            let scale_code =
                u8::try_from(integer(F_SCALE)?).map_err(|_| CodecError::InvalidField(F_SCALE))?;
            let scale = code_to_scale(scale_code).map_err(|_| CodecError::InvalidField(F_SCALE))?;
            let pin = integer(F_PIN)?;
            let policy = match integer(F_POLICY)? {
                0 if pin == 0 => SingletonPolicy::Off,
                1 if pin == 0 => SingletonPolicy::Use,
                2 if pin > 0 => SingletonPolicy::UseWithProtectedEdges(
                    usize::try_from(pin).map_err(|_| CodecError::InvalidField(F_PIN))?,
                ),
                _ => return Err(CodecError::InvalidField(F_POLICY)),
            };
            builder = builder.scale(scale).singleton_policy(policy);
            let delta_s = field(s, F_DELTA)?;
            expect_dtype(&delta_s, DataType::Float64, F_DELTA)?;
            if let Some(delta) = delta_s.f64().unwrap().get(i) {
                if !delta.is_finite() || delta <= 0.0 {
                    return Err(CodecError::InvalidField(F_DELTA));
                }
                if scale != ScaleFamily::K2Norm || policy != SingletonPolicy::Off {
                    return Err(CodecError::InvalidConfiguration(
                        "delta mode requires K2Norm scale and singleton policy Off",
                    ));
                }
                builder = builder.delta(delta);
            }
        }
        let digest = builder
            .with_centroids_and_stats(
                cents,
                DigestStats {
                    data_sum,
                    total_weight,
                    data_min,
                    data_max,
                },
            )
            .build()
            .map_err(|_| {
                CodecError::InvalidConfiguration(
                    "invalid digest state: mass, extrema or aggregate metadata",
                )
            })?;
        out.push(digest);
    }

    Ok(out)
}

#[inline]
fn field(s: &StructChunked, name: &'static str) -> Result<Series, CodecError> {
    s.field_by_name(name)
        .map_err(|_| CodecError::MissingField(name))
}

#[inline]
fn expect_dtype(s: &Series, expected: DataType, field: &'static str) -> Result<(), CodecError> {
    if s.dtype() != &expected {
        Err(CodecError::WrongDtype {
            field,
            found: Box::new(s.dtype().clone()),
            expected: Box::new(expected),
        })
    } else {
        Ok(())
    }
}

fn ensure_centroid_dtype<N: PlFloat>(centroids: &Series) -> Result<(), CodecError> {
    let DataType::List(inner) = centroids.dtype() else {
        return Err(CodecError::BadCentroidLayout {
            found: Box::new(centroids.dtype().clone()),
        });
    };
    let DataType::Struct(fields) = inner.as_ref() else {
        return Err(CodecError::BadCentroidLayout {
            found: Box::new(centroids.dtype().clone()),
        });
    };
    if !(fields.len() == 2
        || fields.len() == 3
            && fields[2].name() == F_ATOMIC
            && fields[2].dtype() == &DataType::Boolean)
        || fields[0].name() != F_MEAN
        || fields[1].name() != F_WEIGHT
    {
        return Err(CodecError::BadCentroidLayout {
            found: Box::new(centroids.dtype().clone()),
        });
    }
    let dt_m = fields[0].dtype();
    let dt_w = fields[1].dtype();

    let canonical =
        matches!(dt_m, DataType::Float32 | DataType::Float64) && dt_w == &DataType::Float64;
    let old_f32 = dt_m == &DataType::Float32 && dt_w == &DataType::Float32;

    if !(canonical || old_f32) {
        return Err(CodecError::MixedCentroidDtypes);
    }
    if *dt_m != N::DTYPE {
        return Err(CodecError::WirePrecisionMismatch {
            expected: N::DTYPE.clone(),
        });
    }
    Ok(())
}

fn read_scalar<N: PlFloat>(s: &Series, field: &'static str, row: usize) -> Result<f64, CodecError> {
    let c = N::as_chunked(s, field)?;
    c.get(row)
        .map(N::to_f64)
        .ok_or(CodecError::NullField { field, row })
}

fn unpack_centroids<N, F>(lc: &ListChunked, row: usize) -> Result<Vec<Centroid<F>>, CodecError>
where
    N: PlFloat,
    F: FloatLike + FloatCore,
{
    let row_ser = lc.get_as_series(row).ok_or(CodecError::NullField {
        field: F_CENTROIDS,
        row,
    })?;
    if row_ser.null_count() != 0 {
        return Err(CodecError::InvalidField(F_CENTROIDS));
    }
    let sc = row_ser
        .struct_()
        .map_err(|_| CodecError::BadCentroidLayout {
            found: Box::new(row_ser.dtype().clone()),
        })?;
    let m_ser = sc
        .field_by_name(F_MEAN)
        .map_err(|_| CodecError::MissingField(F_MEAN))?;
    let w_ser = sc
        .field_by_name(F_WEIGHT)
        .map_err(|_| CodecError::MissingField(F_WEIGHT))?;
    let m = N::as_chunked(&m_ser, F_MEAN)?;
    // Promotion of old f32 weights is exact. Canonical f64 weights are never narrowed.
    let w = match w_ser.dtype() {
        DataType::Float32 => w_ser
            .cast(&DataType::Float64)
            .map_err(|_| CodecError::InvalidField(F_WEIGHT))?,
        _ => w_ser,
    };
    let w = Float64Type::as_chunked(&w, F_WEIGHT)?;
    let atomic_ser = sc.field_by_name(F_ATOMIC).ok();

    let len = m.len().min(w.len());
    let mut out = Vec::with_capacity(len);
    for (idx, (mm, ww)) in m.iter().zip(w.iter()).enumerate() {
        let (Some(mv), Some(wv)) = (mm, ww) else {
            return Err(CodecError::NullCentroid { row, index: idx });
        };
        let mean_f64 = N::to_f64(mv);
        let w_f64 = wv;
        if !mean_f64.is_finite() {
            return Err(CodecError::InvalidField(F_MEAN));
        }
        if !w_f64.is_finite() || w_f64 <= 0.0 {
            return Err(CodecError::InvalidField(F_WEIGHT));
        }

        // Structs without a kind field identify only unit weights as atomic.
        let atomic = if let Some(column) = &atomic_ser {
            expect_dtype(column, DataType::Boolean, F_ATOMIC)?;
            column
                .bool()
                .unwrap()
                .get(idx)
                .ok_or(CodecError::NullCentroid { row, index: idx })?
        } else {
            w_f64 == 1.0
        };
        if atomic {
            out.push(Centroid::<F>::new_atomic_f64(mean_f64, w_f64));
        } else {
            out.push(Centroid::<F>::new_mixed_f64(mean_f64, w_f64));
        }
    }
    Ok(out)
}
