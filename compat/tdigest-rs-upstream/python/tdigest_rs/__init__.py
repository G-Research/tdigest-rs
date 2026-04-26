from __future__ import annotations

from copy import copy
from enum import Enum
from typing import Any, Dict, Iterator, Optional

import numpy as np

__version__ = "compat"

DEFAULT_DELTA: float = 300.0


class ScaleFamily(str, Enum):
    QUAD = "QUAD"
    K1 = "K1"
    K2 = "K2"
    K3 = "K3"


class SingletonPolicy(str, Enum):
    OFF = "off"
    USE = "use"
    EDGES = "edges"


def _as_supported_array(arr: Any) -> np.ndarray:
    out = np.asarray(arr)
    if out.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise TypeError(f"TDigest is not implemented for arr with type {out.dtype}")
    if out.ndim != 1:
        out = out.reshape(-1)
    return out


def _coalesce_sorted(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if values.size == 0:
        return values, np.array([], dtype=np.uint32)

    sorted_values = np.sort(values)
    uniq, counts = np.unique(sorted_values, return_counts=True)
    return uniq.astype(values.dtype, copy=False), counts.astype(np.uint32, copy=False)


def _coalesce_means_weights(means: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if means.size == 0:
        return means, weights.astype(np.uint32, copy=False)

    order = np.argsort(means, kind="mergesort")
    means_sorted = means[order]
    weights_sorted = weights[order].astype(np.float64, copy=False)

    uniq: list[Any] = []
    summed: list[float] = []
    for mean, weight in zip(means_sorted, weights_sorted):
        if np.isnan(mean):
            continue
        if uniq and mean == uniq[-1]:
            summed[-1] += float(weight)
        else:
            uniq.append(mean)
            summed.append(float(weight))

    out_means = np.asarray(uniq, dtype=means.dtype)
    out_weights = np.asarray(np.rint(summed), dtype=np.uint32)
    return out_means, out_weights


def _order_value(means: np.ndarray, weights: np.ndarray, index: int) -> float:
    cumulative = np.cumsum(weights.astype(np.uint64, copy=False))
    pos = int(np.searchsorted(cumulative, index + 1, side="left"))
    return float(means[pos])


def _expanded(means: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.repeat(means.astype(np.float64, copy=False), weights.astype(np.int64, copy=False))


class TDigest:
    def __init__(
        self,
        means: np.ndarray,
        weights: np.ndarray,
        *,
        dtype: np.dtype[Any],
        merged: bool = False,
    ):
        self._dtype = np.dtype(dtype)
        self._means = np.asarray(means, dtype=self._dtype)
        self._weights = np.asarray(weights, dtype=np.uint32)
        self._merged = bool(merged)

    @property
    def means(self) -> np.ndarray:
        return self._means

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    @classmethod
    def from_array(cls, arr: Any, delta: float = DEFAULT_DELTA) -> "TDigest":
        del delta
        values = _as_supported_array(arr)
        values = values[~np.isnan(values)]
        means, weights = _coalesce_sorted(values)
        return cls(means, weights, dtype=values.dtype)

    @classmethod
    def from_means_weights(
        cls,
        arr: Any,
        weights: Any,
        delta: float = DEFAULT_DELTA,
    ) -> "TDigest":
        del delta
        means = _as_supported_array(arr)
        weights_arr = np.asarray(weights)
        if weights_arr.ndim != 1:
            weights_arr = weights_arr.reshape(-1)
        if means.shape[0] != weights_arr.shape[0]:
            raise ValueError("means and weights must have the same length")
        means, weights_out = _coalesce_means_weights(means, weights_arr)
        return cls(means, weights_out, dtype=means.dtype)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TDigest":
        if not isinstance(d, dict):
            raise TypeError(f"from_dict expects a dict; got {type(d).__name__}")
        if "means" not in d or "weights" not in d:
            raise ValueError("from_dict expects 'means' and 'weights'")
        out = cls.from_means_weights(arr=np.array(d["means"]), weights=np.array(d["weights"]))
        out._merged = bool(d.get("merged", False))
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "means": self.means.copy(),
            "weights": self.weights.copy(),
            "merged": self._merged,
        }

    def __len__(self) -> int:
        return int(self._means.size)

    def __iter__(self) -> Iterator[float]:
        return iter(self._means.tolist())

    def quantile(self, q: float) -> float:
        if len(self) == 0:
            return 0.0
        qf = float(q)
        if qf <= 0.0:
            return float(self._means[0])
        if qf >= 1.0:
            return float(self._means[-1])

        total = int(np.sum(self._weights, dtype=np.uint64))
        if total <= 0:
            return 0.0

        target = qf * float(total - 1)
        lo = int(np.floor(target))
        hi = int(np.ceil(target))
        if lo == hi:
            return _order_value(self._means, self._weights, lo)

        alpha = target - float(lo)
        left = _order_value(self._means, self._weights, lo)
        right = _order_value(self._means, self._weights, hi)
        return float((1.0 - alpha) * left + alpha * right)

    def median(self) -> float:
        if self._merged:
            return self.trimmed_mean(0.05, 0.95)
        return self.quantile(0.5)

    def trimmed_mean(self, lower: float, upper: float) -> float:
        if len(self) == 0:
            return float("nan")
        values = _expanded(self._means, self._weights)
        if values.size == 0:
            return float("nan")
        lo = int(np.floor(float(lower) * values.size))
        hi = int(np.ceil(float(upper) * values.size))
        hi = min(max(hi, lo + 1), values.size)
        return float(np.mean(values[lo:hi]))

    def merge(self, other: Any, delta: float = DEFAULT_DELTA) -> "TDigest":
        del delta
        if not isinstance(other, TDigest):
            raise ValueError(f"Cannot merge object {other} to {self}")
        if self._dtype != other._dtype:
            raise TypeError(f"self ({self._dtype}) has a different type to ({other._dtype})")

        means = np.concatenate([self._means, other._means]).astype(self._dtype, copy=False)
        weights = np.concatenate([self._weights, other._weights])
        merged_means, merged_weights = _coalesce_means_weights(means, weights)
        return TDigest(merged_means, merged_weights, dtype=self._dtype, merged=True)

    def update(
        self,
        buffer: Any,
        delta: float = DEFAULT_DELTA,
        merge_delta: float = DEFAULT_DELTA,
    ) -> "TDigest":
        return self.merge(TDigest.from_array(buffer, delta=delta), delta=merge_delta)

    def __getstate__(self) -> Dict[str, Any]:
        return self.to_dict()

    def __setstate__(self, d: Dict[str, Any]) -> None:
        obj = self.from_dict(d)
        self._dtype = obj._dtype
        self._means = obj._means
        self._weights = obj._weights
        self._merged = obj._merged

    def __copy__(self) -> "TDigest":
        return TDigest(
            self._means.copy(),
            self._weights.copy(),
            dtype=self._dtype,
            merged=self._merged,
        )

    def __deepcopy__(self, memo: Optional[Dict[int, Any]]) -> "TDigest":
        del memo
        return copy(self)


__all__ = [
    "TDigest",
    "ScaleFamily",
    "SingletonPolicy",
    "__version__",
]
