from __future__ import annotations

import base64
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Dict, Optional

try:
    from ._native import TDigest as _NativeTDigest, __version__ as __native_version__

    __version__ = __native_version__
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ImportError(
        "Failed to import the compiled extension '_native'. Build it with: `uv run maturin develop -r -F python`."
    ) from exc

# Fallback for editable/dev installs where importlib.metadata might not see the wheel
try:  # pragma: no cover
    __version__ = version("tdigest-rs")
except PackageNotFoundError:  # pragma: no cover
    pass


from ._polars import (
    add_values,
    add_weighted_values,
    cast_precision,
    cdf,
    from_bytes,
    infer_column_precision,
    median,
    merge_tdigests,
    quantile,
    scale_values,
    scale_weights,
    tdigest,
    to_bytes,
)
from ._shared import (
    DEFAULT_DELTA as DEFAULT_DELTA,
    DEFAULT_MAX_SIZE as DEFAULT_MAX_SIZE,
    ScaleFamily,
    SingletonPolicy,
    _UNSET,
    _coerce_precision,
    _norm_policy,
    _resolve_scale_for_mode,
    _resolve_size_mode,
    _validate_pin_per_side,
    wire_precision,
)

# Keep the public enum identity stable for introspection and existing pickles.
ScaleFamily.__module__ = __package__
SingletonPolicy.__module__ = __package__


_native_from_array = _NativeTDigest.from_array


def _constructor_options(
    max_size: Any,
    delta: Any,
    scale: Any,
    singleton_policy: SingletonPolicy | str | None,
    pin_per_side: int | None,
    precision: str | None,
    f32_mode: bool | None,
) -> Dict[str, Any]:
    max_size, legacy_delta = _resolve_size_mode(max_size, delta)
    s = _resolve_scale_for_mode(scale, legacy_delta is not None)
    m = _norm_policy(singleton_policy)

    if legacy_delta is not None:
        if singleton_policy is not None and m != "off":
            raise ValueError("delta mode only supports singleton_policy='off'.")
        if pin_per_side is not None:
            raise ValueError("pin_per_side is not supported when using delta mode.")
        m = "off"

    if m != "edges" and pin_per_side is not None:
        raise ValueError("pin_per_side is only allowed when singleton_policy='edges'")
    eps = _validate_pin_per_side(pin_per_side, max_size) if m == "edges" else None

    prec_norm = _coerce_precision(precision)
    f32_flag = prec_norm == "f32" if f32_mode is None else bool(f32_mode)
    if f32_mode is not None and prec_norm != "auto" and f32_flag != (prec_norm == "f32"):
        raise ValueError(f"Conflicting precision arguments: precision={prec_norm!r} and f32_mode={f32_flag!r}")

    call_kwargs: Dict[str, Any] = {
        "max_size": int(max_size),
        "scale": s,
        "f32_mode": f32_flag,
        "singleton_policy": m,
    }
    if eps is not None:
        call_kwargs["pin_per_side"] = int(eps)
    if legacy_delta is not None:
        call_kwargs["delta"] = legacy_delta

    return call_kwargs


def _from_array_cls(
    cls: type[_NativeTDigest],
    data: Any = _UNSET,
    delta: Any = _UNSET,
    *,
    max_size: Any = _UNSET,
    scale: Any = _UNSET,
    singleton_policy: SingletonPolicy | str | None = None,
    pin_per_side: Optional[int] = None,
    weights: Any = _UNSET,
    precision: str | None = None,
    f32_mode: bool | None = None,
    **kwargs: Any,
) -> _NativeTDigest:
    """
    Build a modern digest from finite observations, optionally with positive weights.

    The default capacity is 1,000 and automatic object precision is Float64.
    Scalar or one-dimensional training inputs are copied. Fractional weights
    use Float64 independently of mean precision. Unknown arguments raise
    TypeError; invalid data/configuration or unrepresentable products raise
    ValueError. This constructs observations, not restored compressed state.

    Supports both:
      - new API: precision="auto" | "f32" | "f64"
      - legacy API: f32_mode=True/False
    """
    if data is _UNSET:
        data = kwargs.pop("arr", _UNSET)
    if data is _UNSET:
        raise TypeError("TDigest.from_array() missing required argument: 'data'")

    call_kwargs = _constructor_options(max_size, delta, scale, singleton_policy, pin_per_side, precision, f32_mode)

    if kwargs:
        extra = ", ".join(sorted(kwargs.keys()))
        raise TypeError(f"Unsupported argument(s): {extra}")

    if weights is not _UNSET:
        # Weighted observations, not compressed-state reconstruction.
        return _native_from_means_weights(data, weights, **call_kwargs)
    return _native_from_array(data, **call_kwargs)


setattr(_NativeTDigest, "from_array", classmethod(_from_array_cls))

_native_from_means_weights = _NativeTDigest.from_means_weights


def _from_means_weights_cls(
    cls: type[_NativeTDigest],
    arr: Any,
    weights: Any,
    delta: Any = _UNSET,
    *,
    max_size: Any = _UNSET,
    scale: Any = _UNSET,
    singleton_policy: SingletonPolicy | str | None = None,
    pin_per_side: Optional[int] = None,
    **kwargs: Any,
) -> _NativeTDigest:
    call_kwargs = _constructor_options(
        max_size,
        delta,
        scale,
        singleton_policy,
        pin_per_side,
        kwargs.pop("precision", None),
        kwargs.pop("f32_mode", None),
    )

    if kwargs:
        extra = ", ".join(sorted(kwargs.keys()))
        raise TypeError(f"Unsupported argument(s): {extra}")

    return _native_from_means_weights(arr, weights, **call_kwargs)


setattr(_NativeTDigest, "from_means_weights", classmethod(_from_means_weights_cls))


TDigest = _NativeTDigest
_native_merge = _NativeTDigest.merge


def _merge_patched(self: _NativeTDigest, other: Any, delta: Any = _UNSET, **kwargs: Any) -> _NativeTDigest:
    if delta is not _UNSET and delta is not None:
        raise ValueError("modern merge cannot override delta; construct compatible digests before merging")
    if kwargs:
        extra = ", ".join(sorted(kwargs.keys()))
        raise TypeError(f"Unsupported argument(s): {extra}")
    if other is self:
        other = TDigest.merge_all([self])
    _native_merge(self, other)
    return self


setattr(TDigest, "merge", _merge_patched)


def _restore_from_bytes(blob: bytes) -> _NativeTDigest:
    return TDigest.from_bytes(blob)


def _reduce_patched(self: _NativeTDigest) -> tuple[Any, tuple[bytes]]:
    return (_restore_from_bytes, (self.to_bytes(),))


setattr(TDigest, "__reduce__", _reduce_patched)


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

    options = ("precision", "max_size", "scale", "singleton_policy", "pin_per_side")
    kwargs = {key: d[key] for key in options if key in d}
    return cls.from_means_weights(d["means"], d["weights"], **kwargs)


setattr(TDigest, "to_dict", _to_dict_patched)
setattr(TDigest, "from_dict", classmethod(_from_dict_cls))


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
