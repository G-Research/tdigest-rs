"""Shared Python argument normalization and TDIG header inspection."""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Literal, Optional

from ._native import wire_precision_py as _wire_precision_native


class ScaleFamily(str, Enum):
    QUAD = "QUAD"
    K1 = "K1"
    K2 = "K2"
    K2NORM = "K2NORM"
    K3 = "K3"


class SingletonPolicy(str, Enum):
    OFF = "off"
    USE = "use"
    EDGES = "edges"


DEFAULT_MAX_SIZE: int = 1000
DEFAULT_DELTA: float = 300.0

_UNSET = object()


def _coerce_scale_for_class(scale: ScaleFamily | str) -> str:
    s = scale.value if isinstance(scale, ScaleFamily) else str(scale).strip().upper()
    if s in {"QUAD", "K1", "K2", "K2NORM", "K3"}:
        return s
    raise ValueError(f"Unknown scale family: {scale!r}. Use one of: 'QUAD'|'K1'|'K2'|'K2NORM'|'K3' (case-insensitive).")


def _coerce_scale_for_plugin(scale: ScaleFamily | str) -> str:
    s = scale.value.lower() if isinstance(scale, ScaleFamily) else str(scale).strip().lower()
    if s in {"quad", "k1", "k2", "k2norm", "k3"}:
        return s
    raise ValueError(f"Unknown scale family: {scale!r}. Use one of: 'quad'|'k1'|'k2'|'k2norm'|'k3' (case-insensitive).")


def _resolve_scale_for_mode(scale_raw: Any, is_delta: bool) -> str:
    """Resolve the scale string honoring the delta-mode contract.

    Mirrors native ``resolve_delta_mode_defaults`` (src/py.rs): when ``delta``
    is set, scale defaults to ``K2NORM`` (canonical Dunning K2) and anything
    else is rejected; otherwise the historical ``K2`` default applies.
    """
    if scale_raw is _UNSET:
        return "K2NORM" if is_delta else "K2"
    s = _coerce_scale_for_class(scale_raw)
    if is_delta and s != "K2NORM":
        raise ValueError(
            "delta mode only supports scale='k2norm' (canonical Dunning K2); "
            "omit scale or pass scale='k2norm' to use delta."
        )
    return s


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


def _validate_delta(delta: Any) -> float:
    try:
        d = float(delta)
    except Exception as exc:  # noqa: BLE001
        raise TypeError(f"delta must be a finite number; got {type(delta).__name__}.") from exc
    if not math.isfinite(d):
        raise ValueError("delta must be finite and > 0.")
    if d <= 0.0:
        raise ValueError("delta must be > 0.")
    return d


def _resolve_size_mode(max_size_raw: Any, delta_raw: Any) -> tuple[int, Optional[float]]:
    has_max_size = max_size_raw is not _UNSET and max_size_raw is not None
    has_delta = delta_raw is not _UNSET and delta_raw is not None

    if has_max_size and has_delta:
        raise ValueError(
            f"Specify either max_size or delta (or neither for default max_size={DEFAULT_MAX_SIZE}), not both."
        )

    if has_max_size:
        return _validate_max_size(max_size_raw), None

    if has_delta:
        return DEFAULT_MAX_SIZE, _validate_delta(delta_raw)

    return DEFAULT_MAX_SIZE, None


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
