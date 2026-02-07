use numpy::{PyArray1, PyReadonlyArray1, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::*;

use tdigest_core::TDigest;

macro_rules! generate {
    ($name: ident, $type: ident) => {
        #[pyclass]
        pub struct $name {
            pub inner: TDigest<$type>,
        }

        #[pymethods]
        impl $name {
            #[getter]
            fn means<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<$type>>> {
                Ok(PyArray1::from_slice(py, &self.inner.means))
            }

            #[getter]
            fn weights<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u32>>> {
                Ok(PyArray1::from_slice(py, &self.inner.weights))
            }

            fn __len__(&self) -> PyResult<usize> {
                Ok(self.inner.means.len())
            }

            #[classmethod]
            fn from_array(
                _cls: &Bound<'_, PyType>,
                py: Python,
                arr: PyReadonlyArray1<'_, $type>,
                delta: $type,
            ) -> PyResult<Self> {
                if arr.len() == 0 {
                    return Err(PyValueError::new_err("Array must be non-empty!"));
                }
                let arr = arr.as_slice().expect("non-contiguous array").to_vec();
                py.detach(|| {
                    Ok(Self {
                        inner: TDigest::from_array(&arr, delta)?,
                    })
                })
            }

            #[classmethod]
            fn from_means_weights(
                _cls: &Bound<'_, PyType>,
                py: Python,
                arr: PyReadonlyArray1<'_, $type>,
                weights: PyReadonlyArray1<'_, u32>,
                delta: $type,
            ) -> PyResult<Self> {
                if arr.len() == 0 {
                    return Err(PyValueError::new_err("Means must be non-empty!"));
                }
                if weights.len() == 0 {
                    return Err(PyValueError::new_err("Means must be non-empty!"));
                }
                let arr = arr.as_slice().expect("non-contiguous array").to_vec();
                let weights = weights.as_slice().expect("non-contiguous array").to_vec();

                py.detach(|| {
                    Ok(Self {
                        inner: TDigest::from_means_weights(&arr, &weights, delta)?,
                    })
                })
            }

            fn quantile(&self, py: Python, x: $type) -> PyResult<$type> {
                py.detach(|| Ok(self.inner.quantile(x)?))
            }

            fn quantiles<'py>(
                &self,
                py: Python<'py>,
                qs: PyReadonlyArray1<'_, $type>,
            ) -> PyResult<Bound<'py, PyArray1<$type>>> {
                let qs_slice = qs.as_slice().expect("non-contiguous array");
                let results = py.detach(|| self.inner.quantiles(qs_slice))?;
                Ok(PyArray1::from_vec(py, results))
            }

            fn median(&self, py: Python) -> PyResult<$type> {
                py.detach(|| Ok(self.inner.median()?))
            }

            fn trimmed_mean(&self, py: Python, lower: $type, upper: $type) -> PyResult<$type> {
                py.detach(|| Ok(self.inner.trimmed_mean(lower, upper)?))
            }

            fn merge(&self, py: Python, other: &Self, delta: $type) -> PyResult<Self> {
                py.detach(|| {
                    Ok(Self {
                        inner: self.inner.merge(&other.inner, delta)?,
                    })
                })
            }

            fn update(
                &self,
                py: Python,
                buffer: PyReadonlyArray1<'_, $type>,
                delta: $type,
                merge_delta: $type,
            ) -> PyResult<Self> {
                let buf = buffer.as_slice().expect("non-contiguous array").to_vec();
                py.detach(|| {
                    let buf_digest = TDigest::from_array(&buf, delta)?;
                    Ok(Self {
                        inner: self.inner.merge(&buf_digest, merge_delta)?,
                    })
                })
            }

            fn n_zero_weights(&self) -> PyResult<usize> {
                Ok(self.inner.n_zero_weights()?)
            }
        }
    };
}

generate!(_TDigestInternal32, f32);
generate!(_TDigestInternal64, f64);

#[pymodule]
fn tdigest_rs(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<_TDigestInternal32>()?;
    m.add_class::<_TDigestInternal64>()?;
    Ok(())
}
