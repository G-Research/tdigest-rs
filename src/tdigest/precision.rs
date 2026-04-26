// src/tdigest/precision.rs
use serde::{Deserialize, Serialize};
use std::any::TypeId;
use std::cmp::Ordering;
use std::ops::{Add, Div, Mul, Sub};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")] // "f32" | "f64" when serialized
pub enum Precision {
    F32,
    F64,
}

impl Precision {
    #[inline]
    pub fn is_f32(self) -> bool {
        matches!(self, Precision::F32)
    }
    #[inline]
    pub fn is_f64(self) -> bool {
        matches!(self, Precision::F64)
    }

    /// Map a float *type* to a Precision at compile-time.
    #[inline]
    pub fn of_type<F: 'static>() -> Self {
        if TypeId::of::<F>() == TypeId::of::<f32>() {
            Precision::F32
        } else {
            Precision::F64
        }
    }
}

/// Minimal trait that lets us write generic code over `f32` and `f64`
/// without pulling in a big numeric framework.
pub trait FloatLike:
    Copy
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Send
    + Sync
    + 'static
{
    const ZERO: Self;
    const ONE: Self;

    fn from_f64(x: f64) -> Self;
    fn to_f64(self) -> f64;

    /// Total ordering consistent with IEEE semantics.
    #[inline]
    fn total_cmp(self, other: Self) -> Ordering {
        // Use native total_cmp on f32/f64 (stable), but expose a unified API here.
        self.to_f64().total_cmp(other.to_f64())
    }
}

impl FloatLike for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    #[inline]
    fn from_f64(x: f64) -> Self {
        x as f32
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline]
    fn total_cmp(self, other: Self) -> Ordering {
        f32::total_cmp(&self, &other)
    }
}

impl FloatLike for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    #[inline]
    fn from_f64(x: f64) -> Self {
        x
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
    #[inline]
    fn total_cmp(self, other: Self) -> Ordering {
        f64::total_cmp(&self, &other)
    }
}
