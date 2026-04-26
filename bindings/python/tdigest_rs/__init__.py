from __future__ import annotations

from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Literal, Optional, cast

import base64
import polars as pl
from polars.plugins import register_plugin_function

if TYPE_CHECKING:
    from polars.type_aliases import IntoExpr

# Path to the compiled plugin (shared library lives next to this file)
lib = Path(__file__).parent

# --- native import (fail loudly so tests don't silently pass with None) ---
try:
    from ._tdigest_rs import TDigest as _NativeTDigest, __version__ as __native_version__
    from ._tdigest_rs import wire_precision_py as _wire_precision_native

    __version__ = __native_version__
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ImportError(
        "Failed to import the compiled extension '_tdigest_rs'. Build it with: `uv run maturin develop -r -F python`."
    ) from exc

# Fallback for editable/dev installs where importlib.metadata might not see the wheel
try:  # pragma: no cover
    __version__ = version("tdigest-rs")
except PackageNotFoundError:  # pragma: no cover
    pass


# --- enums --------------------------------------------------------------------
class ScaleFamily(str, Enum):
    QUAD = "QUAD"
    K1 = "K1"
    K2 = "K2"
    K3 = "K3"


class SingletonPolicy(str, Enum):
    OFF = "off"
    USE = "use"
    EDGES = "edges"


# --- helpers ------------------------------------------------------------------
def _coerce_scale_for_class(scale: ScaleFamily | str) -> str:
    s = scale.value if isinstance(scale, ScaleFamily) else str(scale).strip().upper()
    if s in {"QUAD", "K1", "K2", "K3"}:
        return s
    raise ValueError(f"Unknown scale family: {scale!r}. Use one of: 'QUAD'|'K1'|'K2'|'K3' (case-insensitive).")


def _coerce_scale_for_plugin(scale: ScaleFamily | str) -> str:
    s = scale.value.lower() if isinstance(scale, ScaleFamily) else str(scale).strip().lower()
    if s in {"quad", "k1", "k2", "k3"}:
        return s
    raise ValueError(f"Unknown scale family: {scale!r}. Use one of: 'quad'|'k1'|'k2'|'k3' (case-insensitive).")


def _coerce_precision(precision: str | None) -> str:
    if precision is None:
        return "auto"
    s = str(precision).strip().lower()
    if s in {"auto", "f64", "f32"}:
        return s
    raise ValueError(f"Unknown precision: {precision!r}. Use 'auto' (default), 'f64', or 'f32'.")


def _coerce_cast_precision(precision: str) -> str:
    s = _coerce_precision(precision)
    if s == "auto":
        raise ValueError("cast_precision requires explicit precision: 'f32' or 'f64'.")
    return s


def _norm_policy(mode: SingletonPolicy | str | None) -> str:
    if mode is None:
        return "use"
    if isinstance(mode, SingletonPolicy):
        return mode.value
    s = str(mode).strip().lower()
    if s in {"off", "use", "edges"}:
        return s
    raise ValueError("singleton_policy must be 'off'|'use'|'edges' or SingletonPolicy enum.")


def _into_expr(x: "IntoExpr | Any") -> pl.Expr:
    if isinstance(x, pl.Expr):
        return x
    if isinstance(x, str):
        return pl.col(x)
    try:
        import numpy as _np

        if isinstance(x, _np.ndarray):
            x = x.tolist()
    except Exception:
        pass
    return pl.lit(x)


def _validate_max_size(max_size: int) -> int:
    try:
        m = int(max_size)
    except Exception as exc:  # noqa: BLE001
        raise TypeError(f"max_size must be an integer; got {type(max_size).__name__}.") from exc
    if m < 10:
        raise ValueError("max_size must be >= 10.")
    if m > 20000:
        raise ValueError("max_size too large (>20_000).")
    return m


def _validate_pin_per_side(pin_per_side: Optional[int], max_size: int) -> Optional[int]:
    if pin_per_side is None:
        return None
    try:
        p = int(pin_per_side)
    except Exception as exc:  # noqa: BLE001
        raise TypeError(f"pin_per_side must be an integer; got {type(pin_per_side).__name__}.") from exc
    if p < 1:
        raise ValueError("pin_per_side must be >= 1.")
    hard_cap = max_size // 2
    if p > hard_cap:
        raise ValueError(f"pin_per_side={p} exceeds limit for max_size={max_size} (<= {hard_cap}).")
    return p


def wire_precision(blob: bytes) -> Literal["f32", "f64"]:
    """
    Inspect a TDIG wire blob and return \"f32\" or \"f64\" depending on
    the encoded backend precision. Raises ValueError on invalid blob.
    """
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise TypeError(f"wire_precision expects a bytes-like object; got {type(blob).__name__}")
    return _wire_precision_native(bytes(blob))


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
    s = df[col]
    if s.null_count() == len(s):
        return "f64"

    bin_s = s.cast(pl.Binary)

    # Indices of non-null rows
    idxs = [i for i, v in enumerate(bin_s) if v is not None]
    if not idxs:
        return "f64"

    if len(idxs) > sample:
        import random

        idxs = random.sample(idxs, sample)

    kinds: set[str] = set()
    for i in idxs:
        b = bin_s[i]
        if b is None:
            continue
        kinds.add(wire_precision(b))

    if not kinds:
        return "f64"

    if len(kinds) > 1:
        msg = f"Mixed TDIG wire precisions in column {col!r}: {sorted(kinds)}"
        if strict:
            raise ValueError(msg)
        # non-strict fallback: choose the \"heavier\" format
        return "f64"

    return cast(Literal["f32", "f64"], kinds.pop())


# --- Python-side shim: TDigest.from_array() argument normalization ------------
_native_from_array_raw: Any = getattr(_NativeTDigest, "from_array", None)
if _native_from_array_raw is None:  # pragma: no cover
    raise AttributeError("Native TDigest is missing 'from_array'")
_native_from_array = cast(Callable[..., Any], _native_from_array_raw)


def _from_array_cls(
    cls: type[_NativeTDigest],
    data: Any,
    *,
    max_size: int = 200,
    scale: ScaleFamily | str = "k2",
    singleton_policy: SingletonPolicy | str | None = "use",
    pin_per_side: Optional[int] = None,
    **kwargs: Any,
) -> _NativeTDigest:
    """
    Python-facing TDigest.from_array shim.

    Supports both:
      - new API: precision="auto" | "f32" | "f64"
      - legacy API: f32_mode=True/False
    """
    s = _coerce_scale_for_class(scale)
    m = _norm_policy(singleton_policy)
    max_size = _validate_max_size(max_size)

    if m != "edges" and pin_per_side is not None:
        raise ValueError("pin_per_side is only allowed when singleton_policy='edges'")
    eps = _validate_pin_per_side(pin_per_side, max_size) if m == "edges" else None

    policy_str = {"off": "off", "use": "use", "edges": "edges"}[m]

    # --- precision / f32_mode reconciliation ---------------------------------
    prec_raw = kwargs.pop("precision", None)
    f32_mode_raw = kwargs.pop("f32_mode", None)
    delta_raw = kwargs.pop("delta", None)

    if delta_raw is not None:
        raise ValueError("delta is no longer supported. Use max_size and scale to configure compression.")

    prec_norm = _coerce_precision(prec_raw)  # "auto" | "f64" | "f32"

    if f32_mode_raw is not None:
        f32_flag = bool(f32_mode_raw)
        if prec_norm != "auto":
            expected_flag = prec_norm == "f32"
            if f32_flag != expected_flag:
                raise ValueError(f"Conflicting precision arguments: precision={prec_norm!r} and f32_mode={f32_flag!r}")
    else:
        # No explicit f32_mode provided; derive from precision.
        if prec_norm == "auto":
            f32_flag = False  # default to f64 backend when auto and no legacy flag
        else:
            f32_flag = prec_norm == "f32"

    call_kwargs: Dict[str, Any] = {
        "max_size": int(max_size),
        "scale": s,
        "f32_mode": f32_flag,
        "singleton_policy": policy_str,
    }
    if eps is not None:
        call_kwargs["pin_per_side"] = int(eps)

    if kwargs:
        extra = ", ".join(sorted(kwargs.keys()))
        raise TypeError(f"Unsupported argument(s): {extra}")

    out = _native_from_array(data, **call_kwargs)
    return out


setattr(_NativeTDigest, "from_array", classmethod(_from_array_cls))

_native_from_means_weights_raw: Any = getattr(_NativeTDigest, "from_means_weights", None)
if _native_from_means_weights_raw is None:  # pragma: no cover
    raise AttributeError("Native TDigest is missing 'from_means_weights'")
_native_from_means_weights = cast(Callable[..., Any], _native_from_means_weights_raw)


def _from_means_weights_cls(
    cls: type[_NativeTDigest],
    arr: Any,
    weights: Any,
    *,
    max_size: int = 200,
    scale: ScaleFamily | str = "k2",
    singleton_policy: SingletonPolicy | str | None = "use",
    pin_per_side: Optional[int] = None,
    **kwargs: Any,
) -> _NativeTDigest:
    s = _coerce_scale_for_class(scale)
    m = _norm_policy(singleton_policy)
    max_size = _validate_max_size(max_size)

    if m != "edges" and pin_per_side is not None:
        raise ValueError("pin_per_side is only allowed when singleton_policy='edges'")
    eps = _validate_pin_per_side(pin_per_side, max_size) if m == "edges" else None

    policy_str = {"off": "off", "use": "use", "edges": "edges"}[m]

    prec_raw = kwargs.pop("precision", None)
    f32_mode_raw = kwargs.pop("f32_mode", None)
    delta_raw = kwargs.pop("delta", None)

    if delta_raw is not None:
        raise ValueError("delta is no longer supported. Use max_size and scale to configure compression.")

    prec_norm = _coerce_precision(prec_raw)

    if f32_mode_raw is not None:
        f32_flag = bool(f32_mode_raw)
        if prec_norm != "auto":
            expected_flag = prec_norm == "f32"
            if f32_flag != expected_flag:
                raise ValueError(f"Conflicting precision arguments: precision={prec_norm!r} and f32_mode={f32_flag!r}")
    else:
        f32_flag = prec_norm == "f32"

    call_kwargs: Dict[str, Any] = {
        "max_size": int(max_size),
        "scale": s,
        "f32_mode": f32_flag,
        "singleton_policy": policy_str,
    }
    if eps is not None:
        call_kwargs["pin_per_side"] = int(eps)

    if kwargs:
        extra = ", ".join(sorted(kwargs.keys()))
        raise TypeError(f"Unsupported argument(s): {extra}")

    out = _native_from_means_weights(arr, weights, **call_kwargs)
    return out


setattr(_NativeTDigest, "from_means_weights", classmethod(_from_means_weights_cls))


TDigest = _NativeTDigest
_native_merge_raw: Any = getattr(_NativeTDigest, "merge", None)
if _native_merge_raw is None:  # pragma: no cover
    raise AttributeError("Native TDigest is missing 'merge'")
_native_merge = cast(Callable[..., Any], _native_merge_raw)
_native_add_raw: Any = getattr(_NativeTDigest, "add", None)
if _native_add_raw is None:  # pragma: no cover
    raise AttributeError("Native TDigest is missing 'add'")
_native_add = cast(Callable[..., Any], _native_add_raw)
_native_add_weighted_raw: Any = getattr(_NativeTDigest, "add_weighted", None)
if _native_add_weighted_raw is None:  # pragma: no cover
    raise AttributeError("Native TDigest is missing 'add_weighted'")
_native_add_weighted = cast(Callable[..., Any], _native_add_weighted_raw)
_native_scale_weights_raw: Any = getattr(_NativeTDigest, "scale_weights", None)
if _native_scale_weights_raw is None:  # pragma: no cover
    raise AttributeError("Native TDigest is missing 'scale_weights'")
_native_scale_weights = cast(Callable[..., Any], _native_scale_weights_raw)
_native_scale_values_raw: Any = getattr(_NativeTDigest, "scale_values", None)
if _native_scale_values_raw is None:  # pragma: no cover
    raise AttributeError("Native TDigest is missing 'scale_values'")
_native_scale_values = cast(Callable[..., Any], _native_scale_values_raw)


def _merge_patched(self: _NativeTDigest, other: Any) -> _NativeTDigest:
    out = TDigest.from_bytes(self.to_bytes())
    _native_merge(out, other)
    return out


setattr(TDigest, "merge", _merge_patched)


def _update_patched(self: _NativeTDigest, buffer: Any, **kwargs: Any) -> _NativeTDigest:
    if kwargs.pop("delta", None) is not None or kwargs.pop("merge_delta", None) is not None:
        raise ValueError("delta/merge_delta are no longer supported. Use update(buffer) or add(buffer).")
    if kwargs:
        extra = ", ".join(sorted(kwargs.keys()))
        raise TypeError(f"Unsupported argument(s): {extra}")
    out = TDigest.from_bytes(self.to_bytes())
    _native_add(out, buffer)
    return out


setattr(TDigest, "update", _update_patched)


def _add_patched(self: _NativeTDigest, values: Any) -> _NativeTDigest:
    # Native mutates in-place; return self for fluent parity with Java.
    _native_add(self, values)
    return self


setattr(TDigest, "add", _add_patched)


def _add_weighted_patched(self: _NativeTDigest, values: Any, weights: Any) -> _NativeTDigest:
    _native_add_weighted(self, values, weights)
    return self


setattr(TDigest, "add_weighted", _add_weighted_patched)


def _scale_weights_patched(self: _NativeTDigest, factor: float) -> _NativeTDigest:
    _native_scale_weights(self, float(factor))
    return self


setattr(TDigest, "scale_weights", _scale_weights_patched)


def _scale_values_patched(self: _NativeTDigest, factor: float) -> _NativeTDigest:
    _native_scale_values(self, float(factor))
    return self


setattr(TDigest, "scale_values", _scale_values_patched)


def _to_dict_patched(self: _NativeTDigest) -> Dict[str, Any]:
    return {
        "schema_version": 2,
        "precision": self.inner_kind(),
        "means": self.means.tolist(),
        "weights": self.weights.tolist(),
        "wire_b64_v3": base64.b64encode(self.to_bytes(version=3)).decode("ascii"),
    }


def _from_dict_cls(cls: type[_NativeTDigest], d: Dict[str, Any]) -> _NativeTDigest:
    if not isinstance(d, dict):
        raise TypeError(f"from_dict expects a dict; got {type(d).__name__}")

    blob_b64 = d.get("wire_b64_v3")
    if isinstance(blob_b64, str) and blob_b64:
        return cls.from_bytes(base64.b64decode(blob_b64))

    if "means" not in d or "weights" not in d:
        raise ValueError("from_dict expects either 'wire_b64_v3' or both 'means' and 'weights'.")

    kwargs: Dict[str, Any] = {}
    if "precision" in d:
        kwargs["precision"] = d["precision"]
    if "max_size" in d:
        kwargs["max_size"] = d["max_size"]
    if "scale" in d:
        kwargs["scale"] = d["scale"]
    if "singleton_policy" in d:
        kwargs["singleton_policy"] = d["singleton_policy"]
    if "pin_per_side" in d:
        kwargs["pin_per_side"] = d["pin_per_side"]

    return cls.from_means_weights(d["means"], d["weights"], **kwargs)


setattr(TDigest, "to_dict", _to_dict_patched)
setattr(TDigest, "from_dict", classmethod(_from_dict_cls))


# --- Polars plugin wrappers ---------------------------------------------------


def tdigest(
    values: "IntoExpr",
    max_size: int = 200,
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
        "singleton_mode": mode_norm,  # expected by Rust
        "precision": prec_norm,
    }
    if eps is not None:
        kwargs["edges_per_side"] = int(eps)

    return register_plugin_function(
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
    v_expr = _into_expr(values)

    expr = register_plugin_function(
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

    return register_plugin_function(
        plugin_path=str(lib),
        function_name="quantile",
        args=[d_expr],
        kwargs={"q": q_val},
        returns_scalar=True,
    )


def median(digest: "IntoExpr") -> pl.Expr:
    d_expr = _into_expr(digest)
    return register_plugin_function(
        plugin_path=str(lib),
        function_name="median",
        args=[d_expr],
        kwargs=None,
        returns_scalar=True,
    )


def merge_tdigests(digest: "IntoExpr") -> pl.Expr:
    """
    Merge TDigest structs.
    - Use with `.over("g")` to do a per-group merge.
    - If a group's input is null/empty, this yields a *real empty digest* (n=0) for that group.
    """
    d_expr = _into_expr(digest)
    return register_plugin_function(
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
    return register_plugin_function(
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
    return register_plugin_function(
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
    hint = pl.lit(0.0, dtype=pl.Float32) if prec == "f32" else pl.lit(0.0, dtype=pl.Float64)
    return register_plugin_function(
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

    return register_plugin_function(
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

    return register_plugin_function(
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
        return register_plugin_function(
            plugin_path=str(lib),
            function_name="to_bytes_versioned",
            args=[d_expr],
            kwargs={"version": v},
            returns_scalar=True,
        )

    return register_plugin_function(
        plugin_path=str(lib),
        function_name="to_bytes",
        args=[d_expr],
        kwargs=None,
        returns_scalar=True,
    )


def from_bytes(blob: "IntoExpr", *, precision: str = "auto") -> pl.Expr:
    """
    Deserialize a TDigest from a binary blob produced by td.to_bytes().

    precision:
      - "auto": let the Rust plugin sniff the wire; schema defaults to compact f32
        when no hint is provided.
      - "f64": Polars schema uses Float64-backed struct
      - "f32": Polars schema uses the compact Float32-backed struct
    """
    b_expr = _into_expr(blob)
    prec_norm = _coerce_precision(precision)  # "auto" | "f64" | "f32"

    if prec_norm == "auto":
        # No hint: Rust side infers from wire; planner defaults to compact f32.
        args = [b_expr]
    else:
        if prec_norm == "f32":
            hint = pl.lit(0.0, dtype=pl.Float32)
        else:  # "f64"
            hint = pl.lit(0.0, dtype=pl.Float64)
        args = [b_expr, hint]

    return register_plugin_function(
        plugin_path=str(lib),
        function_name="from_bytes",
        args=args,
        kwargs=None,
        returns_scalar=True,
    )


__all__ = [
    "TDigest",
    "__version__",
    "ScaleFamily",
    "SingletonPolicy",
    "tdigest",
    "cdf",
    "quantile",
    "median",
    "merge_tdigests",
    "add_values",
    "add_weighted_values",
    "cast_precision",
    "scale_weights",
    "scale_values",
    "to_bytes",
    "from_bytes",
    "wire_precision",
    "infer_column_precision",
]
