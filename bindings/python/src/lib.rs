use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::*;

use std::num::NonZeroU32;
use tdigest_core::{Delta, TDigest};

macro_rules! generate {
    ($name: ident, $type: ident) => {
        #[pyclass]
        pub struct $name {
            pub inner: TDigest<$type>,
        }

        #[pymethods]
        impl $name {
            #[getter]
            fn means(&self, py: Python) -> PyResult<Py<PyArray1<$type>>> {
                let arr = PyArray1::from_vec(py, self.inner.means.clone());
                Ok(arr.unbind())
            }

            #[getter]
            fn weights(&self, py: Python) -> PyResult<Py<PyArray1<u32>>> {
                let weights: Vec<u32> = self.inner.weights().map(|w| w.get()).collect();
                let arr = PyArray1::from_vec(py, weights);
                Ok(arr.unbind())
            }

            fn __len__(&self) -> PyResult<usize> {
                Ok(self.inner.len())
            }

            #[classmethod]
            fn from_array(
                _cls: &Bound<'_, PyType>,
                arr: PyReadonlyArray1<$type>,
                delta: $type,
            ) -> PyResult<Self> {
                if arr.len()? == 0 {
                    return Err(PyValueError::new_err("Array must be non-empty!"));
                }
                let arr = arr.as_array().to_vec();
                let delta = Delta::new(delta)?;
                Ok(Self {
                    inner: TDigest::from_array(&arr, delta)?,
                })
            }

            #[classmethod]
            fn from_means_weights(
                _cls: &Bound<'_, PyType>,
                arr: PyReadonlyArray1<$type>,
                weights: PyReadonlyArray1<u32>,
                delta: $type,
            ) -> PyResult<Self> {
                if arr.len()? == 0 {
                    return Err(PyValueError::new_err("Means must be non-empty!"));
                }
                if weights.len()? == 0 {
                    return Err(PyValueError::new_err("Weights must be non-empty!"));
                }
                let arr = arr.as_array().to_vec();
                let weights_vec = weights.as_array().to_vec();
                let weights_nonzero: Result<Vec<NonZeroU32>, _> = weights_vec
                    .iter()
                    .map(|&w| {
                        NonZeroU32::new(w)
                            .ok_or_else(|| PyValueError::new_err("Weights cannot be zero"))
                    })
                    .collect();
                let weights_nonzero = weights_nonzero?;

                let delta = Delta::new(delta)?;
                Ok(Self {
                    inner: TDigest::from_means_weights(&arr, &weights_nonzero, delta)?,
                })
            }

            fn quantile(&self, x: $type) -> PyResult<$type> {
                Ok(self.inner.quantile(x)?)
            }

            fn median(&self) -> PyResult<$type> {
                Ok(self.inner.median()?)
            }

            fn trimmed_mean(&self, lower: $type, upper: $type) -> PyResult<$type> {
                Ok(self.inner.trimmed_mean(lower, upper)?)
            }

            fn merge(&self, other: &Self, delta: $type) -> PyResult<Self> {
                let delta = Delta::new(delta)?;
                Ok(Self {
                    inner: self.inner.merge(&other.inner, delta)?,
                })
            }
        }
    };
}

generate!(_TDigestInternal32, f32);
generate!(_TDigestInternal64, f64);

#[pymodule]
fn tdigest_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<_TDigestInternal32>()?;
    m.add_class::<_TDigestInternal64>()?;
    Ok(())
}
