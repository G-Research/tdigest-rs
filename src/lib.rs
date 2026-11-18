//! TDigest for Rust with optional Python and Java bindings.
//!
//! The `python` feature builds the `tdigest_rs._native` CPython extension.
//! The `java` feature enables JNI. Linux GNU builds use jemalloc unless
//! `java` is enabled, in which case the JVM library uses the system allocator.

mod polars_expr;

pub mod tdigest;

mod error;
pub use error::{TdError, TdResult};

#[cfg(feature = "java")]
pub mod jni;

#[cfg(test)]
pub mod quality {
    pub mod cdf_quality;
    pub mod quality_base;
    pub mod quantile_quality;
}

#[cfg(all(target_os = "linux", target_env = "gnu", not(feature = "java")))]
use jemallocator::Jemalloc;

#[cfg(all(target_os = "linux", target_env = "gnu", not(feature = "java")))]
#[global_allocator]
static ALLOC: Jemalloc = Jemalloc;

#[cfg(feature = "python")]
mod legacy_python;
#[cfg(feature = "python")]
mod py;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyModuleMethods;

/// CPython module entry point.
#[cfg(feature = "python")]
#[pymodule]
#[pyo3(name = "_native")]
fn _native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    crate::py::register(m)?;
    legacy_python::register(m)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
