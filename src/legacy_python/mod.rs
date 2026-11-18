//! Python's deprecated delta API. Its persisted arrays are reproduced by the
//! v0.1.0 algorithm, independently of the capacity and wire-format digest.
mod algorithm;

use algorithm::Digest;
use numpy::{Element, PyArray1, PyReadonlyArray1, PyUntypedArrayMethods};
use pyo3::{exceptions::PyValueError, prelude::*};

fn owned<T: Element + Copy>(arr: PyReadonlyArray1<'_, T>) -> PyResult<Vec<T>> {
    if arr.is_aligned() {
        return Ok(arr.as_array().to_vec());
    }
    // NumPy permits unaligned data and strides. Make an aligned copy before
    // forming Rust references. PyReadonlyArray enforces the input dtype/shape.
    let copy = arr.call_method0("copy")?;
    let copy = copy.extract::<PyReadonlyArray1<T>>()?;
    Ok(copy.as_slice()?.to_vec())
}

macro_rules! binding {
    ($class:ident, $float:ty) => {
        #[pyclass(module = "tdigest_rs.tdigest_rs", frozen)]
        struct $class(Digest<$float>);

        #[pymethods]
        impl $class {
            #[staticmethod]
            fn from_array(
                py: Python<'_>,
                arr: PyReadonlyArray1<$float>,
                delta: $float,
            ) -> PyResult<Self> {
                let arr = owned(arr)?;
                py.detach(|| Digest::from_array(arr, delta))
                    .map(Self)
                    .map_err(PyValueError::new_err)
            }

            #[staticmethod]
            fn empty() -> Self {
                Self(Digest {
                    means: Vec::new(),
                    weights: Vec::new(),
                })
            }

            #[staticmethod]
            fn from_means_weights(
                py: Python<'_>,
                arr: PyReadonlyArray1<$float>,
                weights: PyReadonlyArray1<u32>,
                delta: $float,
            ) -> PyResult<Self> {
                if arr.len() != weights.len() {
                    return Err(PyValueError::new_err(
                        "means and weights must have the same length",
                    ));
                }
                if arr.is_empty() {
                    return Err(PyValueError::new_err("Means must be non-empty!"));
                }
                let arr = owned(arr)?;
                let weights = owned(weights)?;
                py.detach(|| Digest::from_means_weights(&arr, &weights, delta))
                    .map(Self)
                    .map_err(PyValueError::new_err)
            }

            #[getter]
            fn means<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<$float>> {
                PyArray1::from_vec(py, self.0.means.clone())
            }

            #[getter]
            fn weights<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
                PyArray1::from_vec(py, self.0.weights.clone())
            }

            fn __len__(&self) -> usize {
                self.0.means.len()
            }
            fn n_zero_weights(&self) -> usize {
                self.0.weights.iter().filter(|&&w| w == 0).count()
            }
            fn quantile(&self, py: Python<'_>, x: $float) -> $float {
                py.detach(|| self.0.quantile(x))
            }
            fn median(&self, py: Python<'_>) -> $float {
                self.quantile(py, 0.5)
            }
            fn trimmed_mean(&self, py: Python<'_>, lower: $float, upper: $float) -> $float {
                py.detach(|| self.0.trimmed_mean(lower, upper))
            }
            fn merge(&self, py: Python<'_>, other: &Self, delta: $float) -> PyResult<Self> {
                py.detach(|| self.0.merge(&other.0, delta))
                    .map(Self)
                    .map_err(PyValueError::new_err)
            }
            fn cast_f32(&self) -> PyResult<_TDigestInternal32> {
                let means = self
                    .0
                    .means
                    .iter()
                    .map(|&m| {
                        let converted = m as f32;
                        if m.is_finite()
                            && (!converted.is_finite() || (m != 0.0 && converted == 0.0))
                        {
                            return Err(PyValueError::new_err(
                                "centroid mean overflows or underflows f32",
                            ));
                        }
                        Ok(converted)
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                Ok(_TDigestInternal32(Digest {
                    means,
                    weights: self.0.weights.clone(),
                }))
            }
            fn cast_f64(&self) -> _TDigestInternal64 {
                _TDigestInternal64(Digest {
                    means: self.0.means.iter().map(|&m| f64::from(m)).collect(),
                    weights: self.0.weights.clone(),
                })
            }
        }
    };
}
binding!(_TDigestInternal32, f32);
binding!(_TDigestInternal64, f64);

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<_TDigestInternal32>()?;
    m.add_class::<_TDigestInternal64>()
}
