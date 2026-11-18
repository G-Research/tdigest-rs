use numpy::{PyArray1, PyArrayDescrMethods, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyFloat, PyInt, PyIterator, PyList, PyTuple, PyType};

use crate::tdigest::frontends::{
    parse_precision_hint, parse_scale_str, parse_singleton_policy_str, scale_to_str, DigestConfig,
    DigestPrecision, FrontendDigest, FrontendError, PrecisionHint,
};
use crate::tdigest::singleton_policy::SingletonPolicy;
use crate::tdigest::wire::{wire_precision, WirePrecision, WireVersion};
use crate::tdigest::ScaleFamily;

fn parse_scale(s: Option<&str>) -> Result<ScaleFamily, PyErr> {
    parse_scale_str(s).map_err(|e| PyValueError::new_err(e.to_string()))
}

fn parse_policy(kind: Option<&str>, pin_per_side: Option<usize>) -> Result<SingletonPolicy, PyErr> {
    parse_singleton_policy_str(kind, pin_per_side).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Delta compression uses K2Norm without protected edges.
fn resolve_delta_mode_defaults(
    scale: Option<&str>,
    singleton_policy: Option<&str>,
    pin_per_side: Option<usize>,
    delta: Option<f64>,
) -> Result<(ScaleFamily, SingletonPolicy), PyErr> {
    if delta.is_none() {
        let sc = parse_scale(scale)?;
        let policy = parse_policy(singleton_policy, pin_per_side)?;
        return Ok((sc, policy));
    }
    let sc = match scale {
        None => ScaleFamily::K2Norm,
        Some(s) => {
            let parsed = parse_scale(Some(s))?;
            if parsed != ScaleFamily::K2Norm {
                return Err(PyValueError::new_err(
                    "delta-mode only supports scale='k2norm' (canonical Dunning K2); \
                     omit `scale` or pass scale='k2norm' to use delta",
                ));
            }
            parsed
        }
    };
    if pin_per_side.is_some() {
        return Err(PyValueError::new_err(
            "delta-mode only supports singleton_policy='off' with no pin_per_side",
        ));
    }
    let policy = match (singleton_policy, pin_per_side) {
        (None, None) => SingletonPolicy::Off,
        _ => {
            let parsed = parse_policy(singleton_policy, pin_per_side)?;
            if parsed != SingletonPolicy::Off {
                return Err(PyValueError::new_err(
                    "delta-mode only supports singleton_policy='off' with no pin_per_side; \
                     omit both, or drop `delta` if you need edge protection",
                ));
            }
            parsed
        }
    };
    Ok((sc, policy))
}

fn map_frontend_err(err: FrontendError) -> PyErr {
    match err {
        FrontendError::InvalidTrainingData(msg)
        | FrontendError::InvalidProbe(msg)
        | FrontendError::InvalidScale(msg)
        | FrontendError::IncompatibleMerge(msg)
        | FrontendError::DecodeError(msg) => PyValueError::new_err(msg),
    }
}

fn parse_cast_precision(raw: &str) -> PyResult<DigestPrecision> {
    match parse_precision_hint(raw).map_err(|e| PyValueError::new_err(e.to_string()))? {
        PrecisionHint::F32 => Ok(DigestPrecision::F32),
        PrecisionHint::F64 => Ok(DigestPrecision::F64),
        PrecisionHint::Auto => Err(PyValueError::new_err(
            "cast_precision expects 'f32' or 'f64' (not 'auto')",
        )),
    }
}

fn parse_wire_version(raw: u8) -> PyResult<WireVersion> {
    WireVersion::from_u8(raw).map_err(|e| PyValueError::new_err(e.to_string()))
}

#[derive(Clone, Copy)]
enum ContainerKind {
    List,
    Tuple,
    Other,
}

fn container_kind(py: Python<'_>, obj: &Py<PyAny>) -> ContainerKind {
    let bound = obj.bind(py);
    if bound.is_instance_of::<PyList>() {
        ContainerKind::List
    } else if bound.is_instance_of::<PyTuple>() {
        ContainerKind::Tuple
    } else {
        ContainerKind::Other
    }
}

fn array_values(py: Python<'_>, obj: Py<PyAny>) -> PyResult<(usize, Vec<f64>)> {
    let np = py.import("numpy")?;
    let arr = np.call_method1("asarray", (obj,))?;
    let ndim = arr
        .getattr("ndim")
        .ok()
        .and_then(|x| x.extract::<usize>().ok())
        .unwrap_or(1);

    if ndim == 0 {
        let v: f64 = arr.extract()?;
        Ok((0, vec![v]))
    } else if ndim == 1 {
        let values: Vec<f64> = arr.extract()?;
        Ok((ndim, values))
    } else {
        Err(PyValueError::new_err("expected a scalar or one-dimensional input; flatten multidimensional arrays explicitly"))
    }
}

// Training keeps ownership of its input vector. Copy compatible NumPy buffers
// directly, without constructing a Python scalar for each element. Unsupported
// layouts and dtypes retain the established generic conversion path.
fn training_array_values(py: Python<'_>, obj: Py<PyAny>) -> PyResult<(usize, Vec<f64>)> {
    let bound = obj.bind(py);
    if let Ok(array) = bound.cast::<PyArray1<f64>>() {
        if array.is_c_contiguous()
            && array.dtype().is_native_byteorder() == Some(true)
            && array.data().is_aligned()
        {
            if let Ok(view) = array.try_readonly() {
                if let Ok(values) = view.as_slice() {
                    return Ok((1, values.to_vec()));
                }
            }
        }
    }
    if let Ok(array) = bound.cast::<PyArray1<f32>>() {
        if array.is_c_contiguous()
            && array.dtype().is_native_byteorder() == Some(true)
            && array.data().is_aligned()
        {
            if let Ok(view) = array.try_readonly() {
                if let Ok(values) = view.as_slice() {
                    return Ok((1, values.iter().map(|&value| f64::from(value)).collect()));
                }
            }
        }
    }
    array_values(py, obj)
}

fn output_for_values(py: Python<'_>, kind: ContainerKind, values: Vec<f64>) -> PyResult<Py<PyAny>> {
    match kind {
        ContainerKind::List => Ok(PyList::new(py, &values)?.into_any().unbind()),
        ContainerKind::Tuple => Ok(PyTuple::new(py, &values)?.into_any().unbind()),
        ContainerKind::Other => {
            // Keep NumPy ownership (including resize support) while avoiding
            // the intermediate Python list and per-element Python floats.
            Ok(PyArray1::from_slice(py, &values).into_any().unbind())
        }
    }
}

#[pyclass(name = "TDigest", subclass)]
/// Distribution summary with Float32/Float64 means and Float64 mass.
///
/// Construct with from_array or from_means_weights. Updates mutate this object;
/// merge returns it for chaining. Queries do not mutate state. Use copy.copy or
/// to_bytes/from_bytes for independent full-state copies.
pub struct PyTDigest {
    inner: FrontendDigest,
}

#[pymethods]
impl PyTDigest {
    #[staticmethod]
    #[pyo3(signature = (values, max_size=1000, scale=None, f32_mode=false, singleton_policy=None, pin_per_side=None, delta=None))]
    /// Build from finite scalar or one-dimensional observations. An explicit delta selects K2Norm compression with singleton protection off.
    pub fn from_array(
        py: Python<'_>,
        values: Py<PyAny>,
        max_size: usize,
        scale: Option<&str>,
        f32_mode: bool,
        singleton_policy: Option<&str>,
        pin_per_side: Option<usize>,
        delta: Option<f64>,
    ) -> PyResult<Self> {
        if max_size == 0 {
            return Err(PyValueError::new_err("max_size must be > 0"));
        }

        let (sc, policy) =
            resolve_delta_mode_defaults(scale, singleton_policy, pin_per_side, delta)?;
        let (_ndim, values_f64) = training_array_values(py, values)?;

        let config = DigestConfig {
            max_size,
            scale: sc,
            policy,
            delta,
        };
        let precision = if f32_mode {
            DigestPrecision::F32
        } else {
            DigestPrecision::F64
        };

        // Python input conversion is complete; only owned Rust data crosses
        // the GIL boundary, before the new digest is exposed to Python.
        let inner = py
            .detach(|| FrontendDigest::from_values(values_f64, config, precision))
            .map_err(map_frontend_err)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (means, weights, max_size=1000, scale=None, f32_mode=false, singleton_policy=None, pin_per_side=None, delta=None))]
    /// Build from positive weighted observations. Means/weights alone do not restore compressed extrema, aggregate metadata or centroid kind.
    pub fn from_means_weights(
        py: Python<'_>,
        means: Py<PyAny>,
        weights: Py<PyAny>,
        max_size: usize,
        scale: Option<&str>,
        f32_mode: bool,
        singleton_policy: Option<&str>,
        pin_per_side: Option<usize>,
        delta: Option<f64>,
    ) -> PyResult<Self> {
        if max_size == 0 {
            return Err(PyValueError::new_err("max_size must be > 0"));
        }

        let (sc, policy) =
            resolve_delta_mode_defaults(scale, singleton_policy, pin_per_side, delta)?;
        let (_ndim_m, means_f64) = training_array_values(py, means)?;
        let (_ndim_w, weights_f64) = training_array_values(py, weights)?;
        if means_f64.len() != weights_f64.len() {
            return Err(PyValueError::new_err(format!(
                "from_means_weights: means and weights must have the same length (got {} vs {})",
                means_f64.len(),
                weights_f64.len()
            )));
        }

        let config = DigestConfig {
            max_size,
            scale: sc,
            policy,
            delta,
        };
        let precision = if f32_mode {
            DigestPrecision::F32
        } else {
            DigestPrecision::F64
        };

        let mut inner =
            FrontendDigest::from_values(Vec::new(), config, precision).map_err(map_frontend_err)?;
        // This digest is still local and all input vectors are owned.
        py.detach(|| inner.add_weighted_f64(means_f64, weights_f64))
            .map_err(map_frontend_err)?;
        Ok(Self { inner })
    }

    #[classmethod]
    /// Return independent merged state. Inputs are unchanged; missing/non-digest entries are rejected. An empty iterable returns Float64/1000/K2/Use state.
    pub fn merge_all(_cls: &Bound<'_, PyType>, digests: Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(single) = digests.extract::<PyRef<PyTDigest>>() {
            single.inner.validate_state().map_err(map_frontend_err)?;
            return Ok(PyTDigest {
                inner: single.inner.clone(),
            });
        }

        let iter = PyIterator::from_object(&digests).map_err(|_| {
            PyTypeError::new_err("tdigest merge_all: expected a TDigest or an iterable of TDigest")
        })?;

        let mut ds: Vec<FrontendDigest> = Vec::new();
        for item_res in iter {
            let item = item_res?;
            let d: PyRef<PyTDigest> = item.extract().map_err(|_| {
                PyTypeError::new_err(
                    "tdigest merge_all: iterable must contain only TDigest objects",
                )
            })?;
            ds.push(d.inner.clone());
        }

        let merged = FrontendDigest::merge_all(ds).map_err(map_frontend_err)?;
        Ok(PyTDigest { inner: merged })
    }

    /// Weighted 0.5 quantile; NaN for empty state. Does not mutate this digest.
    pub fn median(&self) -> PyResult<f64> {
        Ok(self.inner.median())
    }

    /// Mean between finite probability bounds 0 <= lower <= upper <= 1. Returns NaN for empty state.
    pub fn trimmed_mean(&self, lower: f64, upper: f64) -> PyResult<f64> {
        self.inner
            .trimmed_mean_strict(lower, upper)
            .map_err(map_frontend_err)
    }

    #[getter]
    /// Independent Float64 array of centroid means. Stored mean precision is reported by inner_kind.
    pub fn means(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(PyArray1::from_slice(py, &self.inner.centroid_means_f64())
            .into_any()
            .unbind())
    }

    #[getter]
    /// Independent Float64 array of statistical masses, never narrowed to Float32.
    pub fn weights(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(PyArray1::from_slice(py, &self.inner.centroid_weights_f64())
            .into_any()
            .unbind())
    }

    fn __len__(&self) -> usize {
        self.inner.centroid_len()
    }

    /// Total statistical mass (Float64), distinct from len(self) centroid count.
    #[getter]
    pub fn count(&self) -> f64 {
        self.inner.count()
    }

    /// Stored weighted data sum (Float64), preserved across recompression.
    #[getter]
    pub fn sum(&self) -> f64 {
        self.inner.sum()
    }

    /// Observed minimum; NaN for empty state.
    #[getter]
    pub fn min(&self) -> f64 {
        match &self.inner {
            FrontendDigest::F32(td) => td.min(),
            FrontendDigest::F64(td) => td.min(),
        }
    }

    /// Observed maximum; NaN for empty state.
    #[getter]
    pub fn max(&self) -> f64 {
        match &self.inner {
            FrontendDigest::F32(td) => td.max(),
            FrontendDigest::F64(td) => td.max(),
        }
    }

    /// Configured capacity, not the number of observations.
    #[getter]
    pub fn max_size(&self) -> usize {
        self.inner.config().max_size
    }

    /// Selected modern scale family, in lowercase.
    #[getter]
    pub fn scale(&self) -> &'static str {
        scale_to_str(self.inner.config().scale)
    }

    /// Modern delta setting, or None in max-size mode.
    #[getter]
    pub fn delta(&self) -> Option<f64> {
        self.inner.config().delta
    }

    /// Strict quantile(s): finite probabilities in [0,1]. Scalar/0-D returns float; list/tuple retains its container; 1-D NumPy input returns an independent Float64 array. Multidimensional input is rejected.
    pub fn quantile(&self, py: Python<'_>, q: Py<PyAny>) -> PyResult<Py<PyAny>> {
        if q.bind(py).is_instance_of::<PyFloat>() || q.bind(py).is_instance_of::<PyInt>() {
            let v = q.extract::<f64>(py)?;
            let out = self.inner.quantile_strict(v).map_err(map_frontend_err)?;
            return Ok(PyFloat::new(py, out).into_any().unbind());
        }

        let kind = container_kind(py, &q);
        let (ndim, qs) = array_values(py, q)?;
        if ndim == 0 {
            let out = self
                .inner
                .quantile_strict(qs[0])
                .map_err(map_frontend_err)?;
            return Ok(PyFloat::new(py, out).into_any().unbind());
        }
        let mut out = Vec::with_capacity(qs.len());
        for p in qs {
            out.push(self.inner.quantile_strict(p).map_err(map_frontend_err)?);
        }
        output_for_values(py, kind, out)
    }

    /// Add finite observations and return the updated digest.
    pub fn add(mut slf: PyRefMut<'_, Self>, values: Py<PyAny>) -> PyResult<PyRefMut<'_, Self>> {
        let (_, values) = training_array_values(slf.py(), values)?;
        slf.inner.add_values_f64(values).map_err(map_frontend_err)?;
        Ok(slf)
    }

    /// Add finite observations with positive weights and return the updated digest.
    pub fn add_weighted(
        mut slf: PyRefMut<'_, Self>,
        values: Py<PyAny>,
        weights: Py<PyAny>,
    ) -> PyResult<PyRefMut<'_, Self>> {
        let (_, values) = training_array_values(slf.py(), values)?;
        let (_, weights) = training_array_values(slf.py(), weights)?;
        slf.inner
            .add_weighted_f64(values, weights)
            .map_err(map_frontend_err)?;
        Ok(slf)
    }

    /// Multiply the mass and weighted sum by a positive finite factor.
    pub fn scale_weights<'py>(
        slf: &Bound<'py, Self>,
        factor: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        let factor = slf.py().get_type::<PyFloat>().call1((factor,))?.extract()?;
        slf.try_borrow_mut()?
            .inner
            .scale_weights(factor)
            .map_err(map_frontend_err)?;
        Ok(slf.clone())
    }

    /// Multiply the value axis and sum by a positive finite factor.
    pub fn scale_values<'py>(
        slf: &Bound<'py, Self>,
        factor: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        let factor = slf.py().get_type::<PyFloat>().call1((factor,))?.extract()?;
        slf.try_borrow_mut()?
            .inner
            .scale_values(factor)
            .map_err(map_frontend_err)?;
        Ok(slf.clone())
    }

    /// Midpoint CDF: mass below plus half tied mass, divided by total mass. NaN probes return NaN; infinities give 0/1 for nonempty state. Containers follow quantile; integer probes produce floating probabilities.
    pub fn cdf(&self, py: Python<'_>, x: Py<PyAny>) -> PyResult<Py<PyAny>> {
        if x.bind(py).is_instance_of::<PyFloat>() || x.bind(py).is_instance_of::<PyInt>() {
            let v = x.extract::<f64>(py)?;
            let out = self.inner.cdf(&[v])[0];
            return Ok(PyFloat::new(py, out).into_any().unbind());
        }

        let kind = container_kind(py, &x);
        let (ndim, xs) = array_values(py, x)?;
        let out = self.inner.cdf(&xs);
        if ndim == 0 {
            return Ok(PyFloat::new(py, out[0]).into_any().unbind());
        }
        output_for_values(py, kind, out)
    }

    #[pyo3(signature = (version=None))]
    /// Encode full state, default TDIG v3. Versions 1/2 reject unsupported mass/configuration rather than silently losing information.
    pub fn to_bytes<'py>(
        &self,
        py: Python<'py>,
        version: Option<u8>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = if let Some(v) = version {
            let parsed = parse_wire_version(v)?;
            self.inner
                .validate_wire_version(parsed)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            self.inner.to_bytes_with_version(parsed)
        } else {
            self.inner.to_bytes()
        };
        Ok(PyBytes::new(py, &bytes))
    }

    #[staticmethod]
    /// Restore independent full state from validated TDIG v1/v2/v3 bytes. Invalid state raises ValueError.
    pub fn from_bytes(b: Bound<'_, PyBytes>) -> PyResult<Self> {
        let inner = FrontendDigest::from_bytes(b.as_bytes()).map_err(map_frontend_err)?;
        Ok(Self { inner })
    }

    /// Stored mean precision: f32 or f64. Weight precision is always Float64.
    pub fn inner_kind(&self) -> &'static str {
        self.inner.inner_kind()
    }

    #[pyo3(signature = (precision))]
    /// Return independent state with explicit f32/f64 mean precision and Float64 mass. Reject unrepresentable or unordered converted state.
    pub fn cast_precision(&self, precision: &str) -> PyResult<Self> {
        let target = parse_cast_precision(precision)?;
        Ok(Self {
            inner: self
                .inner
                .try_cast_precision(target)
                .map_err(map_frontend_err)?,
        })
    }

    /// Merge one digest or an iterable in place. All inputs must match precision/configuration, including empties. A failure leaves this digest unchanged; other inputs are never modified.
    pub fn merge(&mut self, other: Bound<'_, PyAny>) -> PyResult<()> {
        if let Ok(one) = other.extract::<PyRef<PyTDigest>>() {
            return self
                .inner
                .merge_in_place(&one.inner)
                .map_err(map_frontend_err);
        }

        let iter = PyIterator::from_object(&other).map_err(|_| {
            PyTypeError::new_err("tdigest merge: expected a TDigest or an iterable of TDigest")
        })?;

        // Prepare independently: iteration, type errors and a late incompatible
        // digest must leave the receiver unchanged.
        let mut candidate = self.inner.clone();
        for item_res in iter {
            let item = item_res?;
            let d: PyRef<PyTDigest> = item.extract().map_err(|_| {
                PyTypeError::new_err("tdigest merge: iterable must contain only TDigest objects")
            })?;
            candidate
                .merge_in_place(&d.inner)
                .map_err(map_frontend_err)?;
        }
        self.inner = candidate;
        Ok(())
    }
}

#[pyfunction]
fn wire_precision_py(b: &Bound<PyAny>) -> PyResult<String> {
    let py_bytes: &Bound<PyBytes> = b.cast()?;
    let bytes = py_bytes.as_bytes();

    match wire_precision(bytes) {
        Ok(WirePrecision::F32) => Ok("f32".to_string()),
        Ok(WirePrecision::F64) => Ok("f64".to_string()),
        Err(e) => Err(PyValueError::new_err(format!("wire_precision: {e}"))),
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTDigest>()?;
    m.add_function(wrap_pyfunction!(wire_precision_py, m)?)?;
    Ok(())
}
