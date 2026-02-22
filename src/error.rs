// src/error.rs
use core::fmt;

/// Library-wide error for tdigest-rs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TdError {
    /// User tried to insert NaN/±inf during digest construction.
    /// `context` pinpoints where it came from (e.g., "sample value", "sample weight", "wire mean").
    NonFiniteInput { context: &'static str },

    /// Back-compat alias (some code paths might still raise this).
    NaNInput { context: &'static str },

    /// Internal invariant violation (should never happen in release builds).
    Invariant { what: &'static str },

    /// Invalid scaling factor for digest-level scaling operations.
    InvalidScaleFactor { context: &'static str },

    /// Weight was non-positive in a weighted-ingest path.
    InvalidWeight { context: &'static str },

    /// Values/weights length mismatch in weighted-ingest path.
    MismatchedInputLength {
        context: &'static str,
        values_len: usize,
        weights_len: usize,
    },
}

impl fmt::Display for TdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TdError::NonFiniteInput { context } => write!(
                f,
                "tdigest: non-finite values are not allowed ({}). \
hint: clean your data or drop NaN/±inf before building the digest",
                context
            ),
            TdError::NaNInput { context } => write!(
                f,
                "tdigest: NaN values are not allowed (got {}). \
hint: clean your data or drop NaNs before building the digest",
                context
            ),
            TdError::Invariant { what } => {
                write!(f, "tdigest: internal invariant violation: {}", what)
            }
            TdError::InvalidScaleFactor { context } => write!(
                f,
                "tdigest: invalid scale factor ({}). hint: factor must be finite and > 0",
                context
            ),
            TdError::InvalidWeight { context } => write!(
                f,
                "tdigest: invalid weight ({}). hint: weights must be finite and > 0",
                context
            ),
            TdError::MismatchedInputLength {
                context,
                values_len,
                weights_len,
            } => write!(
                f,
                "tdigest: mismatched input lengths ({}): values_len={} weights_len={}",
                context, values_len, weights_len
            ),
        }
    }
}

impl std::error::Error for TdError {}

pub type TdResult<T> = Result<T, TdError>;
