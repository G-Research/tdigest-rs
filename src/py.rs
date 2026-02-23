use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyFloat, PyIterator, PyList, PyTuple, PyType};

use crate::tdigest::frontends::{
    parse_precision_hint, parse_scale_str, parse_singleton_policy_str, DigestConfig,
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

fn container_kind(py: Python<'_>, obj: &PyObject) -> ContainerKind {
    let bound = obj.bind(py);
    if bound.is_instance_of::<PyList>() {
        ContainerKind::List
    } else if bound.is_instance_of::<PyTuple>() {
        ContainerKind::Tuple
    } else {
        ContainerKind::Other
    }
}

fn array_values(py: Python<'_>, obj: PyObject) -> PyResult<(usize, Vec<f64>)> {
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
    } else {
        let values: Vec<f64> = arr.extract()?;
        Ok((ndim, values))
    }
}

fn output_for_values(py: Python<'_>, kind: ContainerKind, values: Vec<f64>) -> PyResult<PyObject> {
    match kind {
        ContainerKind::List => Ok(PyList::new(py, &values)?.into_any().unbind()),
        ContainerKind::Tuple => Ok(PyTuple::new(py, &values)?.into_any().unbind()),
        ContainerKind::Other => {
            let np = py.import("numpy")?;
            let out = np.call_method1("asarray", (values,))?;
            Ok(out.unbind())
        }
    }
}

#[pyclass(name = "TDigest", subclass)]
pub struct PyTDigest {
    inner: FrontendDigest,
}

impl PyTDigest {
    fn merge_inner(&mut self, other: &PyTDigest) -> PyResult<()> {
        self.inner
            .merge_in_place(&other.inner)
            .map_err(map_frontend_err)
    }
}

#[pymethods]
impl PyTDigest {
    #[staticmethod]
    #[allow(clippy::too_many_arguments)] // PyO3 constructor signature mirrors public Python API.
    #[pyo3(signature = (values, max_size=1000, scale=None, f32_mode=false, singleton_policy=None, pin_per_side=None, legacy_delta=None))]
    pub fn from_array(
        py: Python<'_>,
        values: PyObject,
        max_size: usize,
        scale: Option<&str>,
        f32_mode: bool,
        singleton_policy: Option<&str>,
        pin_per_side: Option<usize>,
        legacy_delta: Option<f64>,
    ) -> PyResult<Self> {
        if max_size == 0 {
            return Err(PyValueError::new_err("max_size must be > 0"));
        }

        let sc = parse_scale(scale)?;
        let policy = parse_policy(singleton_policy, pin_per_side)?;
        if let Some(delta) = legacy_delta {
            if !delta.is_finite() || delta <= 0.0 {
                return Err(PyValueError::new_err("legacy_delta must be finite and > 0"));
            }
            if sc != ScaleFamily::K2 {
                return Err(PyValueError::new_err(
                    "legacy_delta mode only supports scale='k2'",
                ));
            }
            if policy != SingletonPolicy::Off {
                return Err(PyValueError::new_err(
                    "legacy_delta mode only supports singleton_policy='off'",
                ));
            }
        }
        let (_ndim, values_f64) = array_values(py, values)?;

        let config = DigestConfig {
            max_size,
            scale: sc,
            policy,
            legacy_delta,
        };
        let precision = if f32_mode {
            DigestPrecision::F32
        } else {
            DigestPrecision::F64
        };

        let inner =
            FrontendDigest::from_values(values_f64, config, precision).map_err(map_frontend_err)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)] // PyO3 constructor signature mirrors public Python API.
    #[pyo3(signature = (means, weights, max_size=1000, scale=None, f32_mode=false, singleton_policy=None, pin_per_side=None, legacy_delta=None))]
    pub fn from_means_weights(
        py: Python<'_>,
        means: PyObject,
        weights: PyObject,
        max_size: usize,
        scale: Option<&str>,
        f32_mode: bool,
        singleton_policy: Option<&str>,
        pin_per_side: Option<usize>,
        legacy_delta: Option<f64>,
    ) -> PyResult<Self> {
        if max_size == 0 {
            return Err(PyValueError::new_err("max_size must be > 0"));
        }

        let sc = parse_scale(scale)?;
        let policy = parse_policy(singleton_policy, pin_per_side)?;
        if let Some(delta) = legacy_delta {
            if !delta.is_finite() || delta <= 0.0 {
                return Err(PyValueError::new_err("legacy_delta must be finite and > 0"));
            }
            if sc != ScaleFamily::K2 {
                return Err(PyValueError::new_err(
                    "legacy_delta mode only supports scale='k2'",
                ));
            }
            if policy != SingletonPolicy::Off {
                return Err(PyValueError::new_err(
                    "legacy_delta mode only supports singleton_policy='off'",
                ));
            }
        }
        let (_ndim_m, means_f64) = array_values(py, means)?;
        let (_ndim_w, weights_f64) = array_values(py, weights)?;
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
            legacy_delta,
        };
        let precision = if f32_mode {
            DigestPrecision::F32
        } else {
            DigestPrecision::F64
        };

        let mut inner =
            FrontendDigest::from_values(Vec::new(), config, precision).map_err(map_frontend_err)?;
        inner
            .add_weighted_f64(means_f64, weights_f64)
            .map_err(map_frontend_err)?;
        Ok(Self { inner })
    }

    #[classmethod]
    pub fn merge_all(_cls: &Bound<'_, PyType>, digests: Bound<'_, PyAny>) -> PyResult<Self> {
        merge_all_impl(digests)
    }

    pub fn median(&self) -> PyResult<f64> {
        Ok(self.inner.median())
    }

    pub fn trimmed_mean(&self, lower: f64, upper: f64) -> PyResult<f64> {
        self.inner
            .trimmed_mean_strict(lower, upper)
            .map_err(map_frontend_err)
    }

    #[getter]
    pub fn means(&self, py: Python<'_>) -> PyResult<PyObject> {
        let np = py.import("numpy")?;
        let arr = np.call_method1("asarray", (self.inner.centroid_means_f64(),))?;
        Ok(arr.unbind())
    }

    #[getter]
    pub fn weights(&self, py: Python<'_>) -> PyResult<PyObject> {
        let np = py.import("numpy")?;
        let arr = np.call_method1("asarray", (self.inner.centroid_weights_f64(),))?;
        Ok(arr.unbind())
    }

    fn __len__(&self) -> usize {
        self.inner.centroid_len()
    }

    pub fn quantile(&self, py: Python<'_>, q: PyObject) -> PyResult<PyObject> {
        if let Ok(v) = q.extract::<f64>(py) {
            let out = self.inner.quantile_strict(v).map_err(map_frontend_err)?;
            return Ok(PyFloat::new(py, out).into_any().unbind());
        }

        let kind = container_kind(py, &q);
        let (_ndim, qs) = array_values(py, q)?;
        let mut out = Vec::with_capacity(qs.len());
        for p in qs {
            out.push(self.inner.quantile_strict(p).map_err(map_frontend_err)?);
        }
        output_for_values(py, kind, out)
    }

    pub fn add(&mut self, py: Python<'_>, values: PyObject) -> PyResult<()> {
        let (_ndim, values_f64) = array_values(py, values)?;
        self.inner
            .add_values_f64(values_f64)
            .map_err(map_frontend_err)
    }

    pub fn add_weighted(
        &mut self,
        py: Python<'_>,
        values: PyObject,
        weights: PyObject,
    ) -> PyResult<()> {
        let (_ndim_v, values_f64) = array_values(py, values)?;
        let (_ndim_w, weights_f64) = array_values(py, weights)?;
        self.inner
            .add_weighted_f64(values_f64, weights_f64)
            .map_err(map_frontend_err)
    }

    pub fn scale_weights(&mut self, factor: f64) -> PyResult<()> {
        self.inner.scale_weights(factor).map_err(map_frontend_err)
    }

    pub fn scale_values(&mut self, factor: f64) -> PyResult<()> {
        self.inner.scale_values(factor).map_err(map_frontend_err)
    }

    pub fn cdf(&self, py: Python<'_>, x: PyObject) -> PyResult<PyObject> {
        if let Ok(v) = x.extract::<f64>(py) {
            let out = self.inner.cdf(&[v])[0];
            return Ok(PyFloat::new(py, out).into_any().unbind());
        }

        let kind = container_kind(py, &x);
        let (_ndim, xs) = array_values(py, x)?;
        let out = self.inner.cdf(&xs);
        output_for_values(py, kind, out)
    }

    #[pyo3(signature = (version=None))]
    pub fn to_bytes<'py>(
        &self,
        py: Python<'py>,
        version: Option<u8>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = if let Some(v) = version {
            let parsed = parse_wire_version(v)?;
            self.inner.to_bytes_with_version(parsed)
        } else {
            self.inner.to_bytes()
        };
        Ok(PyBytes::new(py, &bytes))
    }

    #[staticmethod]
    pub fn from_bytes(b: Bound<'_, PyBytes>) -> PyResult<Self> {
        let inner = FrontendDigest::from_bytes(b.as_bytes()).map_err(map_frontend_err)?;
        Ok(Self { inner })
    }

    pub fn inner_kind(&self) -> &'static str {
        self.inner.inner_kind()
    }

    #[pyo3(signature = (precision))]
    pub fn cast_precision(&self, precision: &str) -> PyResult<Self> {
        let target = parse_cast_precision(precision)?;
        Ok(Self {
            inner: self.inner.cast_precision(target),
        })
    }

    pub fn merge(&mut self, other: Bound<'_, PyAny>) -> PyResult<()> {
        if let Ok(one) = other.extract::<PyRef<PyTDigest>>() {
            self.merge_inner(&one)?;
            return Ok(());
        }

        let iter = PyIterator::from_object(&other).map_err(|_| {
            PyTypeError::new_err("tdigest merge: expected a TDigest or an iterable of TDigest")
        })?;

        for item_res in iter {
            let item = item_res?;
            let d: PyRef<PyTDigest> = item.extract().map_err(|_| {
                PyTypeError::new_err("tdigest merge: iterable must contain only TDigest objects")
            })?;
            self.merge_inner(&d)?;
        }

        Ok(())
    }
}

fn merge_all_impl(digests: Bound<'_, PyAny>) -> PyResult<PyTDigest> {
    if let Ok(single) = digests.extract::<PyRef<PyTDigest>>() {
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
            PyTypeError::new_err("tdigest merge_all: iterable must contain only TDigest objects")
        })?;
        ds.push(d.inner.clone());
    }

    let merged = FrontendDigest::merge_all(ds).map_err(map_frontend_err)?;
    Ok(PyTDigest { inner: merged })
}

#[pyfunction]
fn wire_precision_py(b: &Bound<PyAny>) -> PyResult<String> {
    let py_bytes: &Bound<PyBytes> = b.downcast()?;
    let bytes = py_bytes.as_bytes();

    match wire_precision(bytes) {
        Ok(WirePrecision::F32) => Ok("f32".to_string()),
        Ok(WirePrecision::F64) => Ok("f64".to_string()),
        Err(e) => Err(PyValueError::new_err(format!("wire_precision: {e}"))),
    }
}

#[pyfunction]
fn merge_all(digests: Bound<'_, PyAny>) -> PyResult<PyTDigest> {
    merge_all_impl(digests)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTDigest>()?;
    m.add_function(wrap_pyfunction!(wire_precision_py, m)?)?;
    m.add_function(wrap_pyfunction!(merge_all, m)?)?;
    Ok(())
}
