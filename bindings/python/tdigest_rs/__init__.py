from __future__ import annotations

from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Callable, Dict, Literal, Optional, cast

import base64
import math

try:
    from ._tdigest_rs import TDigest as _NativeTDigest, __version__ as __native_version__
    from ._tdigest_rs import wire_precision_py as _wire_precision_native

    __version__ = __native_version__
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ImportError(
        "Failed to import the compiled extension '_tdigest_rs'. Build it with: `uv run maturin develop -r -F python`."
    ) from exc

try:  # pragma: no cover
    __version__ = version("tdigest-rs")
except PackageNotFoundError:  # pragma: no cover
    pass


class ScaleFamily(str, Enum):
    QUAD = "QUAD"
    K1 = "K1"
    K2 = "K2"
    K3 = "K3"


class SingletonPolicy(str, Enum):
    OFF = "off"
    USE = "use"
    EDGES = "edges"


_DEFAULT_MAX_SIZE = 100
_UNSET = object()


def _coerce_scale_for_class(scale: ScaleFamily | str) -> str:
    s = scale.value if isinstance(scale, ScaleFamily) else str(scale).strip().upper()
    if s in {"QUAD", "K1", "K2", "K3"}:
        return s
    raise ValueError(f"Unknown scale family: {scale!r}. Use one of: 'QUAD'|'K1'|'K2'|'K3' (case-insensitive).")


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
        raise ValueError("Specify either max_size or delta (or neither for default max_size=100), not both.")

    if has_max_size:
        return _validate_max_size(max_size_raw), None

    if has_delta:
        return _DEFAULT_MAX_SIZE, _validate_delta(delta_raw)

    return _DEFAULT_MAX_SIZE, None


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


_native_from_array_raw: Any = getattr(_NativeTDigest, "from_array", None)
if _native_from_array_raw is None:  # pragma: no cover
    raise AttributeError("Native TDigest is missing 'from_array'")
_native_from_array = cast(Callable[..., Any], _native_from_array_raw)


def _from_array_cls(
    cls: type[_NativeTDigest],
    data: Any,
    delta: Any = _UNSET,
    *,
    max_size: Any = _UNSET,
    scale: ScaleFamily | str = "k2",
    singleton_policy: SingletonPolicy | str | None = None,
    pin_per_side: Optional[int] = None,
    **kwargs: Any,
) -> _NativeTDigest:
    """
    Python-facing TDigest.from_array shim.

    Supports both:
      - new API: precision="auto" | "f32" | "f64"
      - legacy API: f32_mode=True/False
    """
    max_size, legacy_delta = _resolve_size_mode(max_size, delta)
    s = _coerce_scale_for_class(scale)
    m = _norm_policy(singleton_policy)

    if legacy_delta is not None:
        if s != "K2":
            raise ValueError("delta legacy mode only supports scale='k2'.")
        if singleton_policy is not None and m != "off":
            raise ValueError("delta legacy mode only supports singleton_policy='off'.")
        if pin_per_side is not None:
            raise ValueError("pin_per_side is not supported when using delta legacy mode.")
        m = "off"

    if m != "edges" and pin_per_side is not None:
        raise ValueError("pin_per_side is only allowed when singleton_policy='edges'")
    eps = _validate_pin_per_side(pin_per_side, max_size) if m == "edges" else None

    policy_str = {"off": "off", "use": "use", "edges": "edges"}[m]

    prec_raw = kwargs.pop("precision", None)
    f32_mode_raw = kwargs.pop("f32_mode", None)

    prec_norm = _coerce_precision(prec_raw)  # "auto" | "f64" | "f32"

    if f32_mode_raw is not None:
        f32_flag = bool(f32_mode_raw)
        if prec_norm != "auto":
            expected_flag = prec_norm == "f32"
            if f32_flag != expected_flag:
                raise ValueError(f"Conflicting precision arguments: precision={prec_norm!r} and f32_mode={f32_flag!r}")
    else:
        if prec_norm == "auto":
            f32_flag = False
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
    if legacy_delta is not None:
        call_kwargs["delta"] = legacy_delta

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
    delta: Any = _UNSET,
    *,
    max_size: Any = _UNSET,
    scale: ScaleFamily | str = "k2",
    singleton_policy: SingletonPolicy | str | None = None,
    pin_per_side: Optional[int] = None,
    **kwargs: Any,
) -> _NativeTDigest:
    max_size, legacy_delta = _resolve_size_mode(max_size, delta)
    s = _coerce_scale_for_class(scale)
    m = _norm_policy(singleton_policy)

    if legacy_delta is not None:
        if s != "K2":
            raise ValueError("delta legacy mode only supports scale='k2'.")
        if singleton_policy is not None and m != "off":
            raise ValueError("delta legacy mode only supports singleton_policy='off'.")
        if pin_per_side is not None:
            raise ValueError("pin_per_side is not supported when using delta legacy mode.")
        m = "off"

    if m != "edges" and pin_per_side is not None:
        raise ValueError("pin_per_side is only allowed when singleton_policy='edges'")
    eps = _validate_pin_per_side(pin_per_side, max_size) if m == "edges" else None

    policy_str = {"off": "off", "use": "use", "edges": "edges"}[m]

    prec_raw = kwargs.pop("precision", None)
    f32_mode_raw = kwargs.pop("f32_mode", None)

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
    if legacy_delta is not None:
        call_kwargs["delta"] = legacy_delta

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

__all__ = [
    "TDigest",
    "__version__",
    "ScaleFamily",
    "SingletonPolicy",
    "wire_precision",
]
