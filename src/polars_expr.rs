#![cfg(feature = "python")]
#![allow(clippy::unused_unit)]

use ordered_float::FloatCore;
use polars::prelude::ListChunked;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::tdigest::codecs::{PlFloat, WireOf};
use crate::tdigest::frontends::{
    parse_precision_hint, parse_singleton_policy_str, FrontendDigest, PrecisionHint,
};
use crate::tdigest::precision::FloatLike;
use crate::tdigest::singleton_policy::SingletonPolicy;
use crate::tdigest::wire::{encode_digest_with_version, WirePrecision, WireVersion};
use crate::tdigest::{ScaleFamily, TDigest};

fn cdf_output_dtype(inputs: &[Field]) -> PolarsResult<Field> {
    let vdt = inputs.get(1).map(|f| f.dtype()).unwrap_or(&DataType::Null);
    let out = match vdt {
        DataType::List(inner) => match inner.as_ref() {
            DataType::Float32 => DataType::List(Box::new(DataType::Float32)),
            _ => DataType::List(Box::new(DataType::Float64)),
        },
        DataType::Float32 => DataType::Float32,
        _ => DataType::Float64,
    };
    Ok(Field::new("".into(), out))
}

// The digest struct is compact if its "min" field is Float32.
fn is_compact_digest_dtype(dt: &DataType) -> bool {
    if let DataType::Struct(fields) = dt {
        for f in fields {
            if f.name() == "min" {
                return matches!(f.dtype(), DataType::Float32);
            }
        }
    }
    false
}

fn digest_field(compact: bool) -> Field {
    let dtype = if compact {
        TDigest::<f32>::polars_dtype()
    } else {
        TDigest::<f64>::polars_dtype()
    };
    Field::new("tdigest".into(), dtype)
}

// Construction takes precision from the values, casting from the hint column.
fn tdigest_output_dtype(inputs: &[Field]) -> PolarsResult<Field> {
    Ok(digest_field(
        inputs
            .first()
            .is_some_and(|f| f.dtype() == &DataType::Float32),
    ))
}

fn cast_precision_output_dtype(inputs: &[Field]) -> PolarsResult<Field> {
    Ok(digest_field(
        inputs
            .get(1)
            .is_some_and(|f| f.dtype() == &DataType::Float32),
    ))
}

// Mutations retain the mean precision but return the current struct schema.
fn merge_output_dtype(inputs: &[Field]) -> PolarsResult<Field> {
    Ok(digest_field(
        inputs
            .first()
            .is_some_and(|f| is_compact_digest_dtype(f.dtype())),
    ))
}

// quantile returns f32 if input digest column is compact; else f64
fn quantile_output_dtype(input_fields: &[Field]) -> PolarsResult<Field> {
    let dt = input_fields
        .first()
        .map(|f| f.dtype())
        .unwrap_or(&DataType::Null);
    let out = if is_compact_digest_dtype(dt) {
        DataType::Float32
    } else {
        DataType::Float64
    };
    Ok(Field::new("".into(), out))
}

#[polars_expr(output_type_func = cdf_output_dtype)]
fn cdf(inputs: &[Series]) -> PolarsResult<Series> {
    polars_ensure!(inputs.len() == 2, ComputeError: "`cdf` expects (digest, values)");
    let digest_s = &inputs[0];
    let values_s = &inputs[1];

    if is_compact_digest_dtype(digest_s.dtype()) {
        cdf_generic::<f32>(digest_s, values_s)
    } else {
        cdf_generic::<f64>(digest_s, values_s)
    }
}

#[polars_expr(output_type_func = quantile_output_dtype)]
fn quantile(inputs: &[Series], kwargs: QuantileKwargs) -> PolarsResult<Series> {
    quantile_impl(inputs, kwargs.q)
}

#[polars_expr(output_type = Float64)]
fn median(inputs: &[Series]) -> PolarsResult<Series> {
    let value = if is_compact_digest_dtype(inputs[0].dtype()) {
        let td = parse_and_merge::<f32>(inputs)?;
        if td.is_empty() {
            None
        } else {
            Some(td.median())
        }
    } else {
        let td = parse_and_merge::<f64>(inputs)?;
        if td.is_empty() {
            None
        } else {
            Some(td.median())
        }
    };
    Ok(Series::new("".into(), [value]))
}

#[polars_expr(output_type_func = merge_output_dtype)]
fn merge_tdigests(inputs: &[Series]) -> PolarsResult<Series> {
    if is_compact_digest_dtype(inputs[0].dtype()) {
        let td = parse_and_merge::<f32>(inputs)?;
        td.to_series(inputs[0].name())
    } else {
        let td = parse_and_merge::<f64>(inputs)?;
        td.to_series(inputs[0].name())
    }
}

#[polars_expr(output_type_func = merge_output_dtype)]
fn add_values(inputs: &[Series]) -> PolarsResult<Series> {
    polars_ensure!(
        inputs.len() == 2,
        ComputeError: "`add_values` expects (digest, values)"
    );
    let digest_s = &inputs[0];
    let values_s = &inputs[1];

    if is_compact_digest_dtype(digest_s.dtype()) {
        add_values_generic::<f32>(digest_s, values_s)
    } else {
        add_values_generic::<f64>(digest_s, values_s)
    }
}

#[polars_expr(output_type_func = merge_output_dtype)]
fn add_weighted_values(inputs: &[Series]) -> PolarsResult<Series> {
    polars_ensure!(
        inputs.len() == 3,
        ComputeError: "`add_weighted_values` expects (digest, values, weights)"
    );
    let digest_s = &inputs[0];
    let values_s = &inputs[1];
    let weights_s = &inputs[2];

    if is_compact_digest_dtype(digest_s.dtype()) {
        add_weighted_values_generic::<f32>(digest_s, values_s, weights_s)
    } else {
        add_weighted_values_generic::<f64>(digest_s, values_s, weights_s)
    }
}

#[polars_expr(output_type_func = merge_output_dtype)]
fn scale_weights(inputs: &[Series], kwargs: ScaleKwargs) -> PolarsResult<Series> {
    polars_ensure!(
        inputs.len() == 1,
        ComputeError: "`scale_weights` expects (digest)"
    );
    let digest_s = &inputs[0];

    if is_compact_digest_dtype(digest_s.dtype()) {
        scale_weights_generic::<f32>(digest_s, kwargs.factor)
    } else {
        scale_weights_generic::<f64>(digest_s, kwargs.factor)
    }
}

#[polars_expr(output_type_func = merge_output_dtype)]
fn scale_values(inputs: &[Series], kwargs: ScaleKwargs) -> PolarsResult<Series> {
    polars_ensure!(
        inputs.len() == 1,
        ComputeError: "`scale_values` expects (digest)"
    );
    let digest_s = &inputs[0];

    if is_compact_digest_dtype(digest_s.dtype()) {
        scale_values_generic::<f32>(digest_s, kwargs.factor)
    } else {
        scale_values_generic::<f64>(digest_s, kwargs.factor)
    }
}

#[polars_expr(output_type_func = cast_precision_output_dtype)]
fn cast_precision(inputs: &[Series]) -> PolarsResult<Series> {
    polars_ensure!(
        inputs.len() == 2,
        ComputeError: "`cast_precision` expects (digest, precision_hint)"
    );
    let digest_s = &inputs[0];
    let want_f32 = matches!(inputs[1].dtype(), DataType::Float32);

    if want_f32 {
        if is_compact_digest_dtype(digest_s.dtype()) {
            let td = parse_and_merge::<f32>(std::slice::from_ref(digest_s))?;
            td.to_series(digest_s.name())
        } else {
            let td = parse_and_merge::<f64>(std::slice::from_ref(digest_s))?;
            td.try_cast_precision::<f32>()
                .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?
                .to_series(digest_s.name())
        }
    } else if is_compact_digest_dtype(digest_s.dtype()) {
        let td = parse_and_merge::<f32>(std::slice::from_ref(digest_s))?;
        td.try_cast_precision::<f64>()
            .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?
            .to_series(digest_s.name())
    } else {
        let td = parse_and_merge::<f64>(std::slice::from_ref(digest_s))?;
        td.to_series(digest_s.name())
    }
}

#[polars_expr(output_type = Binary)]
fn to_bytes(inputs: &[Series]) -> PolarsResult<Series> {
    polars_ensure!(
        inputs.len() == 1,
        ComputeError: "`to_bytes` expects a single TDigest struct column"
    );
    let s = &inputs[0];

    let td_bytes: Vec<u8> = if is_compact_digest_dtype(s.dtype()) {
        let td = parse_and_merge::<f32>(inputs)?;
        encode_digest_with_version(&td, WireVersion::LATEST)
    } else {
        let td = parse_and_merge::<f64>(inputs)?;
        encode_digest_with_version(&td, WireVersion::LATEST)
    };

    let mut builder = BinaryChunkedBuilder::new("".into(), 1);
    builder.append_value(&td_bytes);
    Ok(builder.finish().into_series())
}

#[polars_expr(output_type = Binary)]
fn to_bytes_versioned(inputs: &[Series], kwargs: ToBytesKwargs) -> PolarsResult<Series> {
    polars_ensure!(
        inputs.len() == 1,
        ComputeError: "`to_bytes_versioned` expects a single TDigest struct column"
    );
    let s = &inputs[0];
    let version = kwargs.version.ok_or_else(|| {
        PolarsError::ComputeError(
            "tdigest.to_bytes_versioned: missing `version` (expected 1, 2, or 3)".into(),
        )
    })?;
    let version = WireVersion::from_u8(version).map_err(|e| {
        PolarsError::ComputeError(format!("tdigest.to_bytes_versioned: {e}").into())
    })?;

    let td_bytes: Vec<u8> = if is_compact_digest_dtype(s.dtype()) {
        let td = parse_and_merge::<f32>(inputs)?;
        version
            .validate_digest(&td)
            .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
        encode_digest_with_version(&td, version)
    } else {
        let td = parse_and_merge::<f64>(inputs)?;
        version
            .validate_digest(&td)
            .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
        encode_digest_with_version(&td, version)
    };

    let mut builder = BinaryChunkedBuilder::new("".into(), 1);
    builder.append_value(&td_bytes);
    Ok(builder.finish().into_series())
}

// Binary columns require an explicit precision hint at planning and execution.
fn from_bytes_output_dtype(inputs: &[Field]) -> PolarsResult<Field> {
    polars_ensure!(
        inputs.len() == 2,
        ComputeError: "`from_bytes` expects a binary column and explicit precision hint"
    );
    let hint_dt = inputs[1].dtype();

    let dt = if matches!(hint_dt, DataType::Float32) {
        TDigest::<f32>::polars_dtype()
    } else {
        TDigest::<f64>::polars_dtype()
    };

    Ok(Field::new("tdigest".into(), dt))
}

#[polars_expr(output_type_func = from_bytes_output_dtype)]
fn from_bytes(inputs: &[Series]) -> PolarsResult<Series> {
    polars_ensure!(
        inputs.len() == 2,
        ComputeError: "`from_bytes` expects a binary column and explicit precision hint"
    );
    let s = &inputs[0];

    if s.is_empty() {
        return if inputs[1].dtype() == &DataType::Float32 {
            TDigest::<f32>::builder()
                .build()
                .unwrap()
                .to_series("tdigest")
        } else {
            TDigest::<f64>::builder()
                .build()
                .unwrap()
                .to_series("tdigest")
        };
    }

    let bin = s.binary().map_err(|e| {
        PolarsError::ComputeError(format!("from_bytes: expected Binary column, got {e}").into())
    })?;
    if bin.null_count() > 0 {
        polars_bail!(
            ComputeError: "tdigest.from_bytes: blob column contains nulls; null blobs are not allowed"
        );
    }

    let expected_prec = if inputs[1].dtype() == &DataType::Float32 {
        WirePrecision::F32
    } else {
        WirePrecision::F64
    };
    let mut digests: Vec<FrontendDigest> = Vec::with_capacity(s.len());
    for bytes in bin.no_null_iter() {
        let d = FrontendDigest::from_bytes_with_expected(bytes, expected_prec)
            .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
        digests.push(d);
    }

    let merged = FrontendDigest::merge_all(digests)
        .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
    match merged {
        FrontendDigest::F32(td) => td.to_series("tdigest"),
        FrontendDigest::F64(td) => td.to_series("tdigest"),
    }
}

#[derive(Deserialize)]
struct QuantileKwargs {
    /// Scalar probability in [0,1]
    q: f64,
}

#[derive(Deserialize)]
struct ScaleKwargs {
    factor: f64,
}

#[derive(Deserialize)]
struct ToBytesKwargs {
    #[serde(default)]
    version: Option<u8>,
}

#[inline]
fn default_max_size() -> usize {
    1000
}
#[inline]
fn default_scale() -> ScaleFamily {
    ScaleFamily::K2
}
#[inline]
fn default_singleton_mode() -> String {
    "use".to_string()
}
#[inline]
fn default_precision() -> String {
    "auto".to_string()
}

#[derive(Deserialize)]
struct TDigestKwargs {
    #[serde(default = "default_max_size")]
    max_size: usize,
    #[serde(default = "default_scale")]
    scale: ScaleFamily,
    #[serde(default = "default_singleton_mode")]
    singleton_mode: String,
    #[serde(
        default,
        alias = "edges_to_preserve",
        alias = "edges_per_side",
        alias = "edges",
        alias = "pin_total"
    )]
    pin_per_side: Option<usize>,
    #[serde(default = "default_precision")]
    precision: String, // "auto" | "f32" | "f64"
}

#[polars_expr(output_type_func = tdigest_output_dtype)]
fn tdigest(inputs: &[Series], kwargs: TDigestKwargs) -> PolarsResult<Series> {
    let policy = parse_singleton_policy_str(Some(&kwargs.singleton_mode), kwargs.pin_per_side)
        .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;

    // Inference rule (matches planner): Float32 input → compact
    let input_is_f32 = matches!(inputs[0].dtype(), DataType::Float32);

    // If user provided `precision`, enforce it unless "auto".
    match parse_precision_hint(&kwargs.precision)
        .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?
    {
        PrecisionHint::Auto => {}
        PrecisionHint::F32 => {
            if !input_is_f32 {
                polars_bail!(ComputeError:
                    "precision=\"f32\" conflicts with input dtype {:?}. \
                     Schema is inferred from input values: Float32→compact(f32), otherwise f64.",
                    inputs[0].dtype());
            }
        }
        PrecisionHint::F64 => {
            if input_is_f32 {
                polars_bail!(ComputeError:
                    "precision=\"f64\" conflicts with input dtype {:?}. \
                     Schema is inferred from input values: Float32→compact(f32), otherwise f64.",
                    inputs[0].dtype());
            }
        }
    }

    if input_is_f32 {
        build_from_values::<f32>(&inputs[0], kwargs.max_size, kwargs.scale, policy)
            .and_then(|td| td.to_series(inputs[0].name()))
    } else {
        build_from_values::<f64>(&inputs[0], kwargs.max_size, kwargs.scale, policy)
            .and_then(|td| td.to_series(inputs[0].name()))
    }
}

// Training conversions share the same representability rule as Python/JNI.
fn checked_training_values<F: FloatLike>(
    values: impl IntoIterator<Item = f64>,
) -> PolarsResult<Vec<F>> {
    values
        .into_iter()
        .map(|value| {
            F::try_from_f64(value).map_err(|e| PolarsError::ComputeError(e.to_string().into()))
        })
        .collect()
}

fn build_from_values<F>(
    values: &Series,
    max_size: usize,
    scale: ScaleFamily,
    policy: SingletonPolicy,
) -> PolarsResult<TDigest<F>>
where
    F: FloatLike + FloatCore + WireOf,
{
    if values.null_count() > 0 {
        return Err(PolarsError::ComputeError(
            "tdigest: input contains nulls; nulls are not allowed in training data".into(),
        ));
    }

    let vf: Vec<F> = match values.dtype() {
        DataType::Float64 => {
            let ca = values.f64()?;
            if ca
                .iter()
                .any(|opt| opt.map(|v| !v.is_finite()).unwrap_or(false))
            {
                return Err(PolarsError::ComputeError(
                    "tdigest: input contains non-finite values (NaN or ±inf)".into(),
                ));
            }
            checked_training_values(ca.no_null_iter())?
        }
        DataType::Float32 => {
            let ca = values.f32()?;
            if ca
                .iter()
                .any(|opt| opt.map(|v| !(v as f64).is_finite()).unwrap_or(false))
            {
                return Err(PolarsError::ComputeError(
                    "tdigest: input contains non-finite values (NaN or ±inf)".into(),
                ));
            }
            checked_training_values(ca.no_null_iter().map(|x| x as f64))?
        }
        dt if dt.is_numeric() => {
            let casted = values.cast(&DataType::Float64)?;
            let ca = casted.f64()?;
            if ca
                .iter()
                .any(|opt| opt.map(|v| !v.is_finite()).unwrap_or(false))
            {
                return Err(PolarsError::ComputeError(
                    "tdigest: input contains non-finite values (NaN or ±inf)".into(),
                ));
            }
            checked_training_values(ca.no_null_iter())?
        }
        other => {
            return Err(PolarsError::ComputeError(
                format!("tdigest: unsupported dtype {other:?}; expected a numeric column").into(),
            ))
        }
    };

    // Apply the requested configuration before compression, including for empty input.
    TDigest::<F>::builder()
        .max_size(max_size)
        .scale(scale)
        .singleton_policy(policy)
        .build()
        .and_then(|digest| digest.merge_unsorted(vf))
        .map_err(|e| PolarsError::ComputeError(e.to_string().into()))
}

fn parse_and_merge<F>(inputs: &[Series]) -> PolarsResult<TDigest<F>>
where
    F: Copy + 'static + FloatCore + FloatLike + WireOf,
{
    let s = &inputs[0];
    let mut parsed = parse_digest_column_strict::<F>(s)?;
    if parsed.len() == 1 {
        return Ok(parsed.pop().unwrap());
    }
    TDigest::<F>::try_merge_digests(parsed)
        .map_err(|e| PolarsError::ComputeError(e.to_string().into()))
}

fn add_values_generic<F>(digest_s: &Series, values_s: &Series) -> PolarsResult<Series>
where
    F: Copy + 'static + FloatCore + FloatLike + WireOf,
{
    let mut td = parse_and_merge::<F>(std::slice::from_ref(digest_s))?;
    let vals = extract_add_values::<F>(values_s)?;
    if !vals.is_empty() {
        td.add_many(vals)
            .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
    }
    td.to_series(digest_s.name())
}

fn add_weighted_values_generic<F>(
    digest_s: &Series,
    values_s: &Series,
    weights_s: &Series,
) -> PolarsResult<Series>
where
    F: Copy + 'static + FloatCore + FloatLike + WireOf,
{
    let mut td = parse_and_merge::<F>(std::slice::from_ref(digest_s))?;
    let (vals, weights) = extract_weighted_add_values::<F>(values_s, weights_s)?;
    if !vals.is_empty() {
        td.add_weighted_many(&vals, &weights)
            .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
    }
    td.to_series(digest_s.name())
}

fn scale_weights_generic<F>(digest_s: &Series, factor: f64) -> PolarsResult<Series>
where
    F: Copy + 'static + FloatCore + FloatLike + WireOf,
{
    let mut td = parse_and_merge::<F>(std::slice::from_ref(digest_s))?;
    td.scale_weights(factor)
        .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
    td.to_series(digest_s.name())
}

fn scale_values_generic<F>(digest_s: &Series, factor: f64) -> PolarsResult<Series>
where
    F: Copy + 'static + FloatCore + FloatLike + WireOf,
{
    let mut td = parse_and_merge::<F>(std::slice::from_ref(digest_s))?;
    td.scale_values(factor)
        .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
    td.to_series(digest_s.name())
}

fn extract_add_values<F>(values: &Series) -> PolarsResult<Vec<F>>
where
    F: Copy + 'static + FloatCore + FloatLike,
{
    match values.dtype() {
        DataType::List(_) => {
            let lc = values.list()?;
            if lc.null_count() > 0 {
                polars_bail!(ComputeError:
                    "tdigest.add_values: values list contains nulls; nulls are not allowed");
            }

            let mut out: Vec<F> = Vec::new();
            for i in 0..lc.len() {
                let sub = lc
                    .get_as_series(i)
                    .unwrap_or_else(|| Series::new("".into(), Vec::<f64>::new()));
                if sub.null_count() > 0 {
                    polars_bail!(ComputeError:
                        "tdigest.add_values: values list contains nulls; nulls are not allowed");
                }
                let vals_f64 = series_values_f64(&sub)?;
                if vals_f64.iter().any(|v| !v.is_finite()) {
                    polars_bail!(ComputeError:
                        "tdigest.add_values: values contain non-finite values (NaN or ±inf)");
                }
                out.extend(checked_training_values::<F>(vals_f64)?);
            }
            Ok(out)
        }
        dt if dt.is_numeric() => {
            if values.null_count() > 0 {
                polars_bail!(ComputeError:
                    "tdigest.add_values: values contain nulls; nulls are not allowed");
            }
            let vals_f64 = series_values_f64(values)?;
            if vals_f64.iter().any(|v| !v.is_finite()) {
                polars_bail!(ComputeError:
                    "tdigest.add_values: values contain non-finite values (NaN or ±inf)");
            }
            checked_training_values(vals_f64)
        }
        other => polars_bail!(ComputeError:
            "tdigest.add_values: unsupported values dtype {other:?}; expected numeric or list"),
    }
}

fn extract_weighted_add_values<F>(
    values: &Series,
    weights: &Series,
) -> PolarsResult<(Vec<F>, Vec<f64>)>
where
    F: Copy + 'static + FloatCore + FloatLike,
{
    match (values.dtype(), weights.dtype()) {
        (DataType::List(_), DataType::List(_)) => {
            let lv = values.list()?;
            let lw = weights.list()?;

            if lv.null_count() > 0 || lw.null_count() > 0 {
                polars_bail!(ComputeError:
                    "tdigest.add_weighted_values: values/weights lists contain nulls; nulls are not allowed");
            }
            if lv.len() != lw.len() {
                polars_bail!(ComputeError:
                    "tdigest.add_weighted_values: list columns must have the same number of rows");
            }

            let mut out_v: Vec<F> = Vec::new();
            let mut out_w: Vec<f64> = Vec::new();

            for i in 0..lv.len() {
                let sub_v = lv
                    .get_as_series(i)
                    .unwrap_or_else(|| Series::new("".into(), Vec::<f64>::new()));
                let sub_w = lw
                    .get_as_series(i)
                    .unwrap_or_else(|| Series::new("".into(), Vec::<f64>::new()));

                if sub_v.null_count() > 0 || sub_w.null_count() > 0 {
                    polars_bail!(ComputeError:
                        "tdigest.add_weighted_values: values/weights lists contain nulls; nulls are not allowed");
                }

                let vals_f64 = series_values_f64(&sub_v)?;
                let weights_f64 = series_values_f64(&sub_w)?;
                if vals_f64.len() != weights_f64.len() {
                    polars_bail!(ComputeError:
                        "tdigest.add_weighted_values: each values list must match its weights list length");
                }
                if vals_f64.iter().any(|v| !v.is_finite()) {
                    polars_bail!(ComputeError:
                        "tdigest.add_weighted_values: values contain non-finite values (NaN or ±inf)");
                }
                if weights_f64.iter().any(|w| !w.is_finite() || *w <= 0.0) {
                    polars_bail!(ComputeError:
                        "tdigest.add_weighted_values: weights must be finite and > 0");
                }

                out_v.extend(checked_training_values::<F>(vals_f64)?);
                out_w.extend(weights_f64);
            }

            Ok((out_v, out_w))
        }
        (vdt, wdt) if vdt.is_numeric() && wdt.is_numeric() => {
            if values.null_count() > 0 || weights.null_count() > 0 {
                polars_bail!(ComputeError:
                    "tdigest.add_weighted_values: values/weights contain nulls; nulls are not allowed");
            }

            let vals_f64 = series_values_f64(values)?;

            let weights_f64 = series_values_f64(weights)?;

            if vals_f64.len() != weights_f64.len() {
                polars_bail!(ComputeError:
                    "tdigest.add_weighted_values: values and weights must have equal lengths");
            }
            if vals_f64.iter().any(|v| !v.is_finite()) {
                polars_bail!(ComputeError:
                    "tdigest.add_weighted_values: values contain non-finite values (NaN or ±inf)");
            }
            if weights_f64.iter().any(|w| !w.is_finite() || *w <= 0.0) {
                polars_bail!(ComputeError:
                    "tdigest.add_weighted_values: weights must be finite and > 0");
            }

            Ok((checked_training_values(vals_f64)?, weights_f64))
        }
        _ => polars_bail!(ComputeError:
            "tdigest.add_weighted_values: values and weights must both be numeric or both be list"),
    }
}

fn cdf_generic<F>(digest_s: &Series, values_s: &Series) -> PolarsResult<Series>
where
    F: Copy + 'static + FloatCore + FloatLike + WireOf,
{
    let digests: Vec<TDigest<F>> = parse_digest_column_strict::<F>(digest_s)?;
    let digest_len = digest_s.len();
    // A scalar/list literal probe may also broadcast over aligned digest rows.
    let repeated_values;
    let values_s = if digest_len != 1 && values_s.len() == 1 {
        repeated_values = values_s.new_from_index(0, digest_len);
        &repeated_values
    } else {
        values_s
    };

    // Cardinality, never digest contents, selects broadcasting. An aligned
    // empty digest must keep its own NaN result rather than borrow a neighbor.
    polars_ensure!(digest_len == 1 || digest_len == values_s.len(), ComputeError:
        "tdigest.cdf: expected one digest or one digest per probe row");
    match values_s.dtype() {
        DataType::List(inner) => {
            if matches!(inner.as_ref(), DataType::Float32) {
                cdf_on_list_out::<F, Float32Type>(&digests, values_s.list()?)
            } else {
                cdf_on_list_out::<F, Float64Type>(&digests, values_s.list()?)
            }
        }
        DataType::Float32 => cdf_on_scalar_out::<F, Float32Type>(&digests, values_s),
        _ => cdf_on_scalar_out::<F, Float64Type>(&digests, values_s),
    }
}

fn parse_digest_column_strict<F>(s: &Series) -> PolarsResult<Vec<TDigest<F>>>
where
    F: Copy + 'static + FloatCore + FloatLike + WireOf,
{
    if s.is_empty() {
        return Ok(Vec::new());
    }

    polars_ensure!(s.null_count() == 0, ComputeError: "tdigest: missing digest rows are not allowed");
    TDigest::<F>::from_series(s)
        .map_err(|e| PolarsError::ComputeError(format!("tdigest parse error: {e}").into()))
}

#[inline]
fn series_values_f64(values: &Series) -> PolarsResult<Vec<f64>> {
    match values.dtype() {
        DataType::Float32 => Ok(values.f32()?.no_null_iter().map(f64::from).collect()),
        DataType::Float64 => Ok(values.f64()?.no_null_iter().collect()),
        _ => Ok(values
            .cast(&DataType::Float64)?
            .f64()?
            .no_null_iter()
            .collect()),
    }
}

/// Scalar values column → output Series of the *Out* float family (f32/f64).
/// Strict policy: any NULL in the probe column → ERROR.
fn cdf_on_scalar_out<F, Out>(digests: &[TDigest<F>], values_s: &Series) -> PolarsResult<Series>
where
    F: Copy + 'static + FloatCore + FloatLike,
    Out: PlFloat,
{
    if values_s.null_count() > 0 {
        polars_bail!(ComputeError: "tdigest.cdf: probe column contains nulls; nulls are not allowed");
    }

    let vals_f64 = series_values_f64(values_s)?;

    let out = if let [digest] = digests {
        digest
            .cdf_or_nan(&vals_f64)
            .into_iter()
            .map(Out::from_f64)
            .collect()
    } else {
        digests
            .iter()
            .zip(vals_f64)
            .map(|(digest, x)| Out::from_f64(digest.cdf_or_nan(&[x])[0]))
            .collect()
    };
    Ok(Out::series_from_vec("", out))
}

/// List values column → output Series of lists with *Out* item dtype (f32/f64).
/// Strict policy: any NULL list cell or inner NULL → ERROR.
fn cdf_on_list_out<F, Out>(digests: &[TDigest<F>], lc: &ListChunked) -> PolarsResult<Series>
where
    F: Copy + 'static + FloatCore + FloatLike,
    Out: PlFloat,
{
    if lc.null_count() > 0 {
        polars_bail!(ComputeError: "tdigest.cdf: probe list contains null values; nulls are not allowed");
    }

    let mut rows: Vec<Series> = Vec::with_capacity(lc.len());
    for i in 0..lc.len() {
        let td = if digests.len() == 1 {
            &digests[0]
        } else {
            &digests[i]
        };
        let sub = lc
            .get_as_series(i)
            .unwrap_or_else(|| Series::new("".into(), Vec::<f64>::new()));

        if sub.null_count() > 0 {
            polars_bail!(ComputeError: "tdigest.cdf: probe list contains nulls; nulls are not allowed");
        }

        let vals = series_values_f64(&sub)?;
        let out_native: Vec<<Out as PolarsNumericType>::Native> = td
            .cdf_or_nan(&vals)
            .into_iter()
            .map(Out::from_f64)
            .collect();
        rows.push(Out::series_from_vec("", out_native));
    }

    let out_lc: ListChunked = rows.into_iter().collect();
    let item_dtype = Out::series_from_vec("", Vec::new()).dtype().clone();
    out_lc
        .into_series()
        .cast(&DataType::List(Box::new(item_dtype)))
}

fn quantile_impl(inputs: &[Series], q: f64) -> PolarsResult<Series> {
    if !q.is_finite() {
        polars_bail!(ComputeError: "q must be a finite number in [0, 1]");
    }
    if !(0.0..=1.0).contains(&q) {
        polars_bail!(ComputeError: "q must be in [0, 1]. Example: quantile(q=0.95) for the 95th percentile");
    }

    let want_f32 = is_compact_digest_dtype(inputs[0].dtype());

    if want_f32 {
        let td = parse_and_merge::<f32>(inputs)?;
        let is_empty = td.is_effectively_empty();
        if is_empty {
            return Ok(Series::new("".into(), [None::<f32>]));
        }
        let val = td.quantile(q);
        if val.is_nan() {
            Ok(Series::new("".into(), [None::<f32>]))
        } else {
            Ok(Series::new("".into(), [Some(val as f32)]))
        }
    } else {
        let td = parse_and_merge::<f64>(inputs)?;
        let is_empty = td.is_effectively_empty();
        if is_empty {
            return Ok(Series::new("".into(), [None::<f64>]));
        }
        let val = td.quantile(q);
        if val.is_nan() {
            Ok(Series::new("".into(), [None::<f64>]))
        } else {
            Ok(Series::new("".into(), [Some(val)]))
        }
    }
}
