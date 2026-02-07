use numpy::{PyArray1, PyReadonlyArray1};
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
            fn means<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray1<$type>> {
                Ok(PyArray1::from_vec(py, self.inner.means.clone()))
            }

            #[getter]
            fn weights<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray1<u32>> {
                Ok(PyArray1::from_vec(py, self.inner.weights.clone()))
            }

            fn __len__(&self) -> PyResult<usize> {
                Ok(self.inner.means.len())
            }

            #[classmethod]
            fn from_array(
                _cls: &PyType,
                py: Python,
                arr: PyReadonlyArray1<$type>,
                delta: $type,
            ) -> PyResult<Self> {
                if arr.len() == 0 {
                    return Err(PyValueError::new_err("Array must be non-empty!"));
                }
                let arr = arr.as_array().to_vec();
                py.allow_threads(|| {
                    Ok(Self {
                        inner: TDigest::from_array(&arr, delta)?,
                    })
                })
            }

            #[classmethod]
            fn from_means_weights(
                _cls: &PyType,
                py: Python,
                arr: PyReadonlyArray1<$type>,
                weights: PyReadonlyArray1<u32>,
                delta: $type,
            ) -> PyResult<Self> {
                if arr.len() == 0 {
                    return Err(PyValueError::new_err("Means must be non-empty!"));
                }
                if weights.len() == 0 {
                    return Err(PyValueError::new_err("Means must be non-empty!"));
                }
                let arr = arr.as_array().to_vec();
                let weights = weights.as_array().to_vec();

                py.allow_threads(|| {
                    Ok(Self {
                        inner: TDigest::from_means_weights(&arr, &weights, delta)?,
                    })
                })
            }

            fn quantile(&self, py: Python, x: $type) -> PyResult<$type> {
                py.allow_threads(|| Ok(self.inner.quantile(x)?))
            }

            fn median(&self, py: Python) -> PyResult<$type> {
                py.allow_threads(|| Ok(self.inner.median()?))
            }

            fn trimmed_mean(&self, py: Python, lower: $type, upper: $type) -> PyResult<$type> {
                py.allow_threads(|| Ok(self.inner.trimmed_mean(lower, upper)?))
            }

            fn merge(&self, py: Python, other: &Self, delta: $type) -> PyResult<Self> {
                py.allow_threads(|| {
                    Ok(Self {
                        inner: self.inner.merge(&other.inner, delta)?,
                    })
                })
            }

            fn update(
                &self,
                py: Python,
                buffer: PyReadonlyArray1<$type>,
                delta: $type,
                merge_delta: $type,
            ) -> PyResult<Self> {
                if buffer.len() == 0 {
                    return Err(PyValueError::new_err("Buffer must be non-empty!"));
                }

                let buffer_vec = buffer.as_array().to_vec();

                py.allow_threads(|| {
                    // Create a TDigest from the buffer and merge with self
                    // Use delta for buffer creation, merge_delta for merging
                    let buffer_digest = TDigest::from_array(&buffer_vec, delta)?;
                    Ok(Self {
                        inner: self.inner.merge(&buffer_digest, merge_delta)?,
                    })
                })
            }

            fn n_zero_weights(&self) -> PyResult<usize> {
                Ok(self.inner.n_zero_weights()?)
            }

            /// Batch update: process multiple buffers at once with parallel processing.
            /// Returns a vector of updated digests.
            #[pyo3(signature = (buffers, delta, merge_delta=None))]
            fn batch_update(
                &self,
                py: Python,
                buffers: Vec<PyReadonlyArray1<$type>>,
                delta: $type,
                merge_delta: Option<$type>,
            ) -> PyResult<Vec<Self>> {
                let merge_delta = merge_delta.unwrap_or(delta);

                if buffers.is_empty() {
                    return Ok(vec![]);
                }

                // Convert all arrays to vecs before releasing GIL
                let mut buffer_vecs = Vec::with_capacity(buffers.len());
                for buffer in buffers {
                    if buffer.len() == 0 {
                        return Err(PyValueError::new_err("Buffer must be non-empty!"));
                    }
                    buffer_vecs.push(buffer.as_array().to_vec());
                }

                py.allow_threads(|| {
                    // Use parallel iterator for batch processing
                    let buffer_refs: Vec<&[$type]> = buffer_vecs.iter().map(|v| v.as_slice()).collect();
                    let updated = self.inner.batch_update(&buffer_refs, merge_delta)?;

                    Ok(updated.into_iter().map(|inner| Self { inner }).collect())
                })
            }

            /// Batch create: create multiple digests from arrays at once with parallel processing.
            /// Returns a vector of digests.
            #[classmethod]
            fn batch_from_arrays(
                _cls: &PyType,
                py: Python,
                arrays: Vec<PyReadonlyArray1<$type>>,
                delta: $type,
            ) -> PyResult<Vec<Self>> {
                if arrays.is_empty() {
                    return Ok(vec![]);
                }

                // Convert all arrays to vecs before releasing GIL
                let mut vecs = Vec::with_capacity(arrays.len());
                for arr in arrays {
                    if arr.len() == 0 {
                        return Err(PyValueError::new_err("Array must be non-empty!"));
                    }
                    vecs.push(arr.as_array().to_vec());
                }

                py.allow_threads(|| {
                    // Use parallel iterator for batch processing
                    let vec_refs: Vec<&[$type]> = vecs.iter().map(|v| v.as_slice()).collect();
                    let digests = TDigest::batch_from_arrays(&vec_refs, delta)?;

                    Ok(digests.into_iter().map(|inner| Self { inner }).collect())
                })
            }
        }
    };
}

generate!(_TDigestInternal32, f32);
generate!(_TDigestInternal64, f64);

#[pymodule]
fn tdigest_rs(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<_TDigestInternal32>()?;
    m.add_class::<_TDigestInternal64>()?;
    Ok(())
}
