from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict, Type, Union

import numpy as np
import numpy.typing as npt

from tdigest_rs.tdigest_rs import _TDigestInternal32, _TDigestInternal64  # type: ignore

DEFAULT_DELTA: float = 300.0


NpFloatArr = npt.NDArray[Union[np.float32, np.float64]]
_TDigestInternal = Union[_TDigestInternal32, _TDigestInternal64]


@dataclass
class TDigest:
    _digest: _TDigestInternal

    def merge(self, other: Any, delta: float = DEFAULT_DELTA) -> "TDigest":
        if not isinstance(other, TDigest):
            raise ValueError(f"Cannot merge object {other} to {self}")

        if (sc := self._digest.__class__) != (oc := other._digest.__class__):
            raise TypeError(f"self ({sc}) has a different type to ({oc})")

        digest = self._digest.merge(other._digest, delta=delta)
        return self.__class__(digest)

    @cached_property
    def means(self) -> npt.NDArray[np.float32]:
        return self._digest.means

    @cached_property
    def weights(self) -> npt.NDArray[np.uint32]:
        return self._digest.weights

    def quantile(self, q: float) -> float:
        return self._digest.quantile(q)

    def median(self) -> float:
        return self._digest.median()

    def trimmed_mean(self, lower: float, upper: float) -> float:
        return self._digest.trimmed_mean(lower, upper)

    @classmethod
    def from_array(cls, arr: npt.NDArray[np.float32], delta: float = DEFAULT_DELTA) -> "TDigest":
        _cls = cls._get_internal_cls(arr)
        digest = _cls.from_array(arr, delta)
        return cls(digest)

    @classmethod
    def from_means_weights(
        cls, arr: npt.NDArray[np.float32], weights: npt.NDArray[np.uint32], delta: float = DEFAULT_DELTA
    ) -> "TDigest":
        _cls = cls._get_internal_cls(arr)
        _digest = _cls.from_means_weights(arr, weights, delta=delta)
        return cls(_digest)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TDigest":
        return cls.from_means_weights(arr=np.array(d["means"]), weights=np.array(d["weights"]))

    def to_dict(self) -> Dict[str, Any]:
        return {"means": self.means, "weights": self.weights}

    def __len__(self) -> int:
        return len(self._digest)

    def __getstate__(self) -> object:
        return self.to_dict()

    def __setstate__(self, d: Dict[str, Any]) -> None:
        obj = self.__class__.from_dict(d)
        self._digest = obj._digest

    def __copy__(self) -> "TDigest":
        means = self.means.copy()
        weights = self.weights.copy()
        return self.__class__.from_means_weights(means, weights)

    def __deepcopy__(self, memo: Any) -> "TDigest":
        return self.__copy__()

    @staticmethod
    def _get_internal_cls(arr: NpFloatArr) -> Type[_TDigestInternal]:
        match arr.dtype:
            case np.float32:
                return _TDigestInternal32
            case np.float64:
                return _TDigestInternal64
            case _:
                raise TypeError(f"TDigest is not implemented for arr with type {arr.dtype}")
