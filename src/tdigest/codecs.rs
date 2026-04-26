//! Polars ↔ TDigest codec — strict, DRY, fully generic.
//!
//! # Design
//! - Writer mirrors the digest’s in-memory precision: `F = f32` → f32 on the wire; `F = f64` → f64 on the wire.
//! - Reader is strict: the wire dtype must match `F`’s mapped wire dtype (no lossy auto-cast).
//! - `sum`/`count` are always `f64` on the wire for numerical stability.
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
//! // build a digest (details elided) and round-trip via a Series:
//! let td: TDigest<f64> = TDigest::builder().max_size(256).build();
//! let series: Series = td.to_series("td")?;
//! let round_tripped: Vec<TDigest<f64>> = TDigest::<f64>::from_series(&series)?;
//!
//! // dtype advertised to Polars matches the wire schema for F = f64:
//! let dtype = TDigest::<f64>::polars_dtype();
//! assert!(matches!(dtype, DataType::Struct(_)));
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use std::fmt;

use ordered_float::FloatCore;
use polars::prelude::*;

use crate::tdigest::centroids::Centroid;
use crate::tdigest::precision::FloatLike;
use crate::tdigest::{DigestStats, TDigest};

/* --------------------- field names (public) -------------------------------- */

pub const F_CENTROIDS: &str = "centroids";
pub const F_MEAN: &str = "mean";
pub const F_WEIGHT: &str = "weight";
pub const F_SUM: &str = "sum";
pub const F_MIN: &str = "min";
pub const F_MAX: &str = "max";
pub const F_COUNT: &str = "count";
pub const F_MAX_SIZE: &str = "max_size";

/* --------------------- errors (public) ------------------------------------- */

#[non_exhaustive]
#[derive(Debug)]
pub enum CodecError {
    NotAStruct {
        series_name: String,
    },
    MissingField(&'static str),
    WrongDtype {
        field: &'static str,
        found: Box<DataType>,
        expected: Box<DataType>,
    },
    BadCentroidLayout {
        found: Box<DataType>,
    },
    MixedCentroidDtypes, // mean/weight not both f32 or both f64
    WirePrecisionMismatch {
        expected: DataType,
    }, // incoming wire doesn't match F’s wire dtype
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
                write!(f, "centroid mean/weight must both be f32 or both be f64")
            }
            WirePrecisionMismatch { expected } => write!(
                f,
                "wire precision mismatch (expected centroids/min/max as {expected:?})"
            ),
            NullField { field, row } => write!(f, "null in scalar field '{field}' at row {row}"),
            NullCentroid { row, index } => {
                write!(f, "null in centroids at row {row}, index {index}")
            }
        }
    }
}

impl std::error::Error for CodecError {}

/* --------------------- Polars float abstraction (public) ------------------- */

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

/* --------------------- TDigest public API (thin shim) ---------------------- */

impl<F> TDigest<F>
where
    F: FloatLike + FloatCore + WireOf,
{
    /// Encode; wire precision (centroids/min/max) follows `F` (f32→f32, f64→f64).
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
                    Field::new(F_WEIGHT.into(), num.clone()),
                ]))),
            ),
            Field::new(F_SUM.into(), DataType::Float64),
            Field::new(F_MIN.into(), num.clone()),
            Field::new(F_MAX.into(), num),
            Field::new(F_COUNT.into(), DataType::Float64),
            Field::new(F_MAX_SIZE.into(), DataType::Int64),
        ])
    }
}

/* --------------------- writer (generic, branch-free) ----------------------- */

fn tdigest_to_series<F, N>(td: &TDigest<F>, name: &str) -> PolarsResult<Series>
where
    F: FloatLike + FloatCore,
    N: PlFloat,
{
    // Gather once in f64 from centroids
    let means_f64: Vec<f64> = td.centroids().iter().map(|c| c.mean_f64()).collect();
    let wgts_f64: Vec<f64> = td.centroids().iter().map(|c| c.weight_f64()).collect();

    // Cast into wire-native
    let means: Vec<<N as PolarsNumericType>::Native> =
        means_f64.iter().copied().map(N::from_f64).collect();
    let weights: Vec<<N as PolarsNumericType>::Native> =
        wgts_f64.iter().copied().map(N::from_f64).collect();

    // Build centroid inner struct and wrap as a single-element list
    let mean_s = N::series_from_vec(F_MEAN, means);
    let weight_s = N::series_from_vec(F_WEIGHT, weights);
    let inner = StructChunked::from_series(
        "centroid".into(),
        mean_s.len(),
        [&mean_s, &weight_s].into_iter(),
    )?
    .into_series();
    let centroids_s = Series::new(F_CENTROIDS.into(), &[inner]);

    // Scalar fields
    let sum_s = Series::new(F_SUM.into(), [td.sum()]);
    let min_s = N::series_from_vec(F_MIN, vec![N::from_f64(td.min())]);
    let max_s = N::series_from_vec(F_MAX, vec![N::from_f64(td.max())]);
    let count_s = Series::new(F_COUNT.into(), [td.count()]);
    let max_size_s = Series::new(F_MAX_SIZE.into(), [td.max_size() as i64]);

    // Outer struct (single row)
    StructChunked::from_series(
        name.into(),
        1,
        [&centroids_s, &sum_s, &min_s, &max_s, &count_s, &max_size_s].into_iter(),
    )
    .map(|sc| sc.into_series())
}

/* --------------------- reader (generic, strict) ---------------------------- */

fn parse_tdigests<F, N>(input: &Series) -> Result<Vec<TDigest<F>>, CodecError>
where
    F: FloatLike + FloatCore,
    N: PlFloat,
{
    let s = input.struct_().map_err(|_| CodecError::NotAStruct {
        series_name: input.name().to_string(),
    })?;

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
        let max_size = max_size_i64 as usize;

        let data_min = read_scalar::<N>(&min, F_MIN, i)?;
        let data_max = read_scalar::<N>(&max, F_MAX, i)?;
        let cents = unpack_centroids::<N, F>(centroids_list, i)?;

        out.push(
            TDigest::<F>::builder()
                .max_size(max_size)
                .with_centroids_and_stats(
                    cents,
                    DigestStats {
                        data_sum,
                        total_weight,
                        data_min,
                        data_max,
                    },
                )
                .build(),
        );
    }

    Ok(out)
}

/* --------------------- small helpers (private) ----------------------------- */

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
    if fields.len() != 2 || fields[0].name() != F_MEAN || fields[1].name() != F_WEIGHT {
        return Err(CodecError::BadCentroidLayout {
            found: Box::new(centroids.dtype().clone()),
        });
    }
    let dt_m = fields[0].dtype();
    let dt_w = fields[1].dtype();

    let both_f64 = dt_m == &DataType::Float64 && dt_w == &DataType::Float64;
    let both_f32 = dt_m == &DataType::Float32 && dt_w == &DataType::Float32;

    if !(both_f32 || both_f64) {
        return Err(CodecError::MixedCentroidDtypes);
    }
    if *dt_m != N::DTYPE || *dt_w != N::DTYPE {
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
    let Some(row_ser) = lc.get_as_series(row) else {
        return Ok(Vec::new());
    };
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
    let w = N::as_chunked(&w_ser, F_WEIGHT)?;

    let len = m.len().min(w.len());
    let mut out = Vec::with_capacity(len);
    for (idx, (mm, ww)) in m.into_iter().zip(w.into_iter()).enumerate() {
        let (Some(mv), Some(wv)) = (mm, ww) else {
            return Err(CodecError::NullCentroid { row, index: idx });
        };
        let mean_f64 = N::to_f64(mv);
        let w_f64 = N::to_f64(wv);

        // Heuristic reconstruction of kind:
        // - weight == 1 → atomic unit (flat CDF step behavior preserved)
        // - weight > 1  → mixed (cannot reliably infer “atomic pile” from wire)
        if w_f64 == 1.0 {
            out.push(Centroid::<F>::new_atomic_unit_f64(mean_f64));
        } else {
            out.push(Centroid::<F>::new_mixed_f64(mean_f64, w_f64));
        }
    }
    Ok(out)
}
