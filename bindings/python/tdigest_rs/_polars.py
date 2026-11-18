"""Polars adapter: lazy imports, expression conversion and native plugin wiring.

Numerical behavior belongs to the shared Rust core. This module handles the
Polars Python API without importing Polars until an expression operation runs.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, cast

import numpy as np

from ._shared import (
    ScaleFamily,
    SingletonPolicy,
    _coerce_cast_precision,
    _coerce_precision,
    _coerce_scale_for_plugin,
    _norm_policy,
    _validate_max_size,
    _validate_pin_per_side,
    wire_precision,
)

if TYPE_CHECKING:
    import polars as pl
    from polars.type_aliases import IntoExpr

# The native plugin lives beside this adapter in both wheels and editable installs.
lib = Path(__file__).parent


def _require_polars() -> ModuleType:
    """Load Polars only for expression/DataFrame operations."""
    try:
        import polars
    except ModuleNotFoundError as exc:
        if exc.name != "polars":
            raise
        raise ModuleNotFoundError(
            "Polars is required for expression and DataFrame operations; install polars to use this API.",
            name="polars",
        ) from exc
    return polars


class _PolarsNamespace:
    """Resolve postponed Polars annotations without eager package imports."""

    def __getattr__(self, name: str) -> Any:
        return getattr(_require_polars(), name)


if not TYPE_CHECKING:
    pl = _PolarsNamespace()


def _register_plugin_function(**kwargs: Any) -> pl.Expr:
    from polars.plugins import register_plugin_function

    return register_plugin_function(**kwargs)


def _into_expr(x: "IntoExpr | Any") -> pl.Expr:
    pl = _require_polars()
    if isinstance(x, pl.Expr):
        return x
    if isinstance(x, str):
        return pl.col(x)
    if isinstance(x, np.ndarray):
        try:
            x = x.tolist()
        except Exception:
            pass
    return pl.lit(x)


def infer_column_precision(
    df: pl.DataFrame,
    col: str,
    *,
    sample: int = 64,
    strict: bool = True,
) -> Literal["f32", "f64"]:
    """
    Infer TDIG precision for a binary column by sampling up to `sample`
    non-null rows and inspecting their wire headers.

    - If all sampled blobs agree on \"f32\" → return \"f32\".
    - If all sampled blobs agree on \"f64\" → return \"f64\".
    - If mixed and strict=True → raise ValueError.
    - If mixed and strict=False → default to \"f64\".
    - If no non-null blobs → default to \"f64\".
    """
    pl = _require_polars()
    s = df[col]
    if s.null_count() == len(s):
        return "f64"

    bin_s = s.cast(pl.Binary)

    idxs = [i for i, v in enumerate(bin_s) if v is not None]

    if len(idxs) > sample:
        import random

        idxs = random.sample(idxs, sample)

    kinds = {wire_precision(bin_s[i]) for i in idxs}

    if not kinds:
        return "f64"

    if len(kinds) > 1:
        msg = f"Mixed TDIG wire precisions in column {col!r}: {sorted(kinds)}"
        if strict:
            raise ValueError(msg)
        # non-strict fallback: choose the \"heavier\" format
        return "f64"

    return kinds.pop()


def tdigest(
    values: "IntoExpr",
    max_size: int = 1000,
    *,
    scale: ScaleFamily | str = "k2",
    precision: str = "auto",
    singleton_policy: SingletonPolicy | str | None = "use",
    pin_per_side: Optional[int] = None,
) -> pl.Expr:
    v_expr = _into_expr(values)

    max_size = _validate_max_size(max_size)
    scale_norm = _coerce_scale_for_plugin(scale)
    prec_norm = _coerce_precision(precision)
    mode_norm = _norm_policy(singleton_policy)

    if mode_norm != "edges" and pin_per_side is not None:
        raise ValueError("pin_per_side is only allowed when singleton_policy='edges'")
    eps = _validate_pin_per_side(pin_per_side, max_size) if mode_norm == "edges" else None

    kwargs: Dict[str, Any] = {
        "max_size": int(max_size),
        "scale": scale_norm,
        "singleton_mode": mode_norm,
        "precision": prec_norm,
    }
    if eps is not None:
        kwargs["edges_per_side"] = int(eps)

    return _register_plugin_function(
        plugin_path=str(lib),
        function_name="tdigest",
        args=[v_expr],
        kwargs=kwargs,
        returns_scalar=True,
    )


def _output_name_or_raise(e: pl.Expr, ctx: str) -> str:
    name = e.meta.output_name()
    if not isinstance(name, str) or name == "":
        raise ValueError(f"{ctx}: 'values' must be a named column/expression.")
    return name


def cdf(digest: "IntoExpr", values: "IntoExpr") -> pl.Expr:
    d_expr = _into_expr(digest)
    pl = _require_polars()
    # Python sequences/Series are a probe vector. A List-valued expression is
    # instead a column of per-row lists and keeps List output even for one row.
    if isinstance(values, np.ndarray) and values.ndim > 1:
        raise ValueError("Polars CDF probe arrays must be scalar or one-dimensional")
    if isinstance(values, (list, tuple, pl.Series)) or isinstance(values, np.ndarray) and values.ndim == 1:
        probes = values if isinstance(values, pl.Series) else pl.Series("probe", values)
        v_expr = pl.lit(probes)
    else:
        v_expr = _into_expr(values)

    expr = _register_plugin_function(
        plugin_path=str(lib),
        function_name="cdf",
        args=[d_expr, v_expr],
        kwargs=None,
        returns_scalar=False,
    )

    vname = _output_name_or_raise(v_expr, "cdf(values)")
    return expr.alias(f"{vname}_cdf")


def quantile(digest: "IntoExpr", q: float) -> pl.Expr:
    d_expr = _into_expr(digest)

    try:
        q_val = float(cast(float, q))
    except Exception as exc:  # noqa: BLE001
        raise TypeError(f"q must be a float in [0, 1]; got {type(q).__name__}.") from exc

    if not (0.0 <= q_val <= 1.0):
        hint = (
            "If you meant a percent, divide by 100 (e.g., 95 → 0.95). "
            "If you meant an absolute data value (not a probability), use cdf() instead."
        )
        raise ValueError(f"q must be within [0, 1]; got {q_val}. {hint}")

    return _register_plugin_function(
        plugin_path=str(lib),
        function_name="quantile",
        args=[d_expr],
        kwargs={"q": q_val},
        returns_scalar=True,
    )


def median(digest: "IntoExpr") -> pl.Expr:
    d_expr = _into_expr(digest)
    return _register_plugin_function(
        plugin_path=str(lib),
        function_name="median",
        args=[d_expr],
        kwargs=None,
        returns_scalar=True,
    )


def merge_tdigests(digest: "IntoExpr") -> pl.Expr:
    """
    Merge TDigest structs.
    Use `.over("g")` to merge each group. Empty input produces an empty digest.
    A null digest row raises a Polars error.
    """
    d_expr = _into_expr(digest)
    return _register_plugin_function(
        plugin_path=str(lib),
        function_name="merge_tdigests",
        args=[d_expr],
        kwargs=None,
        returns_scalar=True,
    )


def add_values(digest: "IntoExpr", values: "IntoExpr") -> pl.Expr:
    """
    Add one or more values into an existing digest expression.
    `values` may be a scalar, numeric column, or list expression.
    """
    d_expr = _into_expr(digest)
    v_expr = _into_expr(values)
    return _register_plugin_function(
        plugin_path=str(lib),
        function_name="add_values",
        args=[d_expr, v_expr],
        kwargs=None,
        returns_scalar=True,
    )


def add_weighted_values(digest: "IntoExpr", values: "IntoExpr", weights: "IntoExpr") -> pl.Expr:
    """
    Add weighted values into an existing digest expression.
    `values` and `weights` must have compatible shapes and equal lengths.
    """
    d_expr = _into_expr(digest)
    v_expr = _into_expr(values)
    w_expr = _into_expr(weights)
    return _register_plugin_function(
        plugin_path=str(lib),
        function_name="add_weighted_values",
        args=[d_expr, v_expr, w_expr],
        kwargs=None,
        returns_scalar=True,
    )


def cast_precision(digest: "IntoExpr", *, precision: str) -> pl.Expr:
    """
    Cast digest centroid precision to explicit backend mode ("f32" or "f64").
    """
    d_expr = _into_expr(digest)
    prec = _coerce_cast_precision(precision)
    pl = _require_polars()
    hint = pl.lit(0.0, dtype=pl.Float32) if prec == "f32" else pl.lit(0.0, dtype=pl.Float64)
    return _register_plugin_function(
        plugin_path=str(lib),
        function_name="cast_precision",
        args=[d_expr, hint],
        kwargs=None,
        returns_scalar=True,
    )


def scale_weights(digest: "IntoExpr", factor: float) -> pl.Expr:
    d_expr = _into_expr(digest)
    try:
        f = float(factor)
    except Exception as exc:  # noqa: BLE001
        raise TypeError(f"factor must be a float; got {type(factor).__name__}.") from exc

    return _register_plugin_function(
        plugin_path=str(lib),
        function_name="scale_weights",
        args=[d_expr],
        kwargs={"factor": f},
        returns_scalar=True,
    )


def scale_values(digest: "IntoExpr", factor: float) -> pl.Expr:
    d_expr = _into_expr(digest)
    try:
        f = float(factor)
    except Exception as exc:  # noqa: BLE001
        raise TypeError(f"factor must be a float; got {type(factor).__name__}.") from exc

    return _register_plugin_function(
        plugin_path=str(lib),
        function_name="scale_values",
        args=[d_expr],
        kwargs={"factor": f},
        returns_scalar=True,
    )


def to_bytes(digest: "IntoExpr", *, version: int | None = None) -> pl.Expr:
    """
    Serialize a TDigest struct column to a single binary blob (scalar expr).

    Typical use:
        df = df.with_columns(td_col=td.tdigest("x", ...))
        df = df.with_columns(blob=td.to_bytes("td_col"))
    """
    d_expr = _into_expr(digest)
    if version is not None:
        try:
            v = int(version)
        except Exception as exc:  # noqa: BLE001
            raise TypeError(f"version must be an int in {{1,2,3}}; got {type(version).__name__}.") from exc
        if v not in {1, 2, 3}:
            raise ValueError(f"version must be one of 1, 2, or 3; got {v}.")
        return _register_plugin_function(
            plugin_path=str(lib),
            function_name="to_bytes_versioned",
            args=[d_expr],
            kwargs={"version": v},
            returns_scalar=True,
        )

    return _register_plugin_function(
        plugin_path=str(lib),
        function_name="to_bytes",
        args=[d_expr],
        kwargs=None,
        returns_scalar=True,
    )


def from_bytes(blob: "IntoExpr", *, precision: str = "auto") -> pl.Expr:
    """Decode and merge one logical group of binary states.

    precision='f32' or 'f64' is required for a deterministic planned schema.
    The hint checks the encoded precision; it never casts state. Inspect eager
    bytes with wire_precision/infer_column_precision before constructing a lazy
    expression when their precision is not already known. Empty input reduces
    to one empty digest in the specified precision. Null/malformed blobs error.
    """
    pl = _require_polars()
    b_expr = _into_expr(blob)
    prec_norm = _coerce_precision(precision)

    if prec_norm == "auto":
        raise ValueError(
            "Polars from_bytes requires explicit precision='f32' or 'f64' so the planned schema matches runtime; inspect eager bytes with wire_precision or infer_column_precision first"
        )
    hint = pl.lit(0.0, dtype=pl.Float32 if prec_norm == "f32" else pl.Float64)
    args = [b_expr, hint]

    return _register_plugin_function(
        plugin_path=str(lib),
        function_name="from_bytes",
        args=args,
        kwargs=None,
        returns_scalar=True,
    )
