#![allow(non_snake_case)]

//! TDigest for Rust with optional Python and Java bindings.
//!
//! Features
//! - python: builds a CPython extension module called `tdigest_rs`.
//! - java: enables JNI bindings in the `jni` module.
//! - On Linux, the global allocator is set to jemalloc to reduce fragmentation
//!   and improve multi-threaded performance.
//!
//! Python (dev loop)
//!   uv run maturin develop -r -F python
//!   python -c "import tdigest_rs; print(tdigest_rs.__version__)"
//!
//! Jemalloc (Linux-only)
//! We set jemalloc as the global allocator on Linux targets. If you prefer the
//! system allocator, remove the `jemallocator` dependency and the
//! `#[global_allocator]` block below.

mod polars_expr;

pub mod tdigest;

mod error;
pub use error::{TdError, TdResult};

#[cfg(feature = "java")]
pub mod jni;

// ---- test-only quality modules ----------------------------------------------
#[cfg(test)]
pub mod quality {
    pub mod cdf_quality;
    pub mod quality_base;
    pub mod quantile_quality;
}

#[cfg(test)]
pub use crate::quality::quality_base::{print_banner, print_report, print_section};

// ---- jemalloc on linux-gnu (kept intentionally) -----------------------------
#[cfg(all(target_os = "linux", target_env = "gnu"))]
use jemallocator::Jemalloc;

#[cfg(all(target_os = "linux", target_env = "gnu"))]
#[global_allocator]
static ALLOC: Jemalloc = Jemalloc;

#[cfg(feature = "python")]
mod py;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyModuleMethods;

/// CPython module entry point.
#[cfg(feature = "python")]
#[pymodule]
#[pyo3(name = "_tdigest_rs")] // ← confirm desired import name
fn _tdigest_rs(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    crate::py::register(m)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
