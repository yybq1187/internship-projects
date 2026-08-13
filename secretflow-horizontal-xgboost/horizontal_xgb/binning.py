"""????????????????????"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _validate_features(features: ArrayLike) -> NDArray[np.float64]:
    """???????? float64?????????????"""
    array = np.asarray(features, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("features ??????????")
    if not np.all(np.isfinite(array)):
        raise ValueError("features ????? NaN ?????")
    return array


@dataclass(frozen=True)
class BinningModel:
    """?????????????"""

    feature_names: tuple[str, ...]
    thresholds: tuple[tuple[float, ...], ...]
    max_bin: int

    def __post_init__(self) -> None:
        """????????????"""
        if len(self.feature_names) == 0:
            raise ValueError("???????????????")
        if len(self.feature_names) != len(self.thresholds):
            raise ValueError("feature_names ? thresholds ??????")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("????????")
        if self.max_bin < 2:
            raise ValueError("max_bin ??? 2?")
        for feature_thresholds in self.thresholds:
            values = np.asarray(feature_thresholds, dtype=np.float64)
            if values.size > self.max_bin - 1:
                raise ValueError("??????????? max_bin - 1?")
            if not np.all(np.isfinite(values)):
                raise ValueError("????????????")
            if values.size > 1 and not np.all(np.diff(values) > 0.0):
                raise ValueError("???????????")

    def to_dict(self) -> dict[str, Any]:
        """??? JSON ???????"""
        return {
            "feature_names": list(self.feature_names),
            "thresholds": [list(values) for values in self.thresholds],
            "max_bin": self.max_bin,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BinningModel":
        """??????????"""
        required = {"feature_names", "thresholds", "max_bin"}
        if not required.issubset(payload):
            raise ValueError("???????????")
        return cls(
            feature_names=tuple(str(name) for name in payload["feature_names"]),
            thresholds=tuple(
                tuple(float(value) for value in values)
                for values in payload["thresholds"]
            ),
            max_bin=int(payload["max_bin"]),
        )


def local_feature_extrema(features: ArrayLike) -> NDArray[np.float64]:
    """??????????????????"""
    array = _validate_features(features)
    return np.stack((np.min(array, axis=0), np.max(array, axis=0)), axis=1)


def fit_equal_width_binning(
    local_extrema: Sequence[ArrayLike],
    feature_names: Sequence[str],
    max_bin: int,
) -> BinningModel:
    """????????????????????????"""
    if max_bin < 2:
        raise ValueError("max_bin ??? 2?")
    names = tuple(str(name) for name in feature_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("feature_names ??????????")
    if len(local_extrema) < 1:
        raise ValueError("???????????????")

    extrema_arrays = [np.asarray(values, dtype=np.float64) for values in local_extrema]
    expected_shape = (len(names), 2)
    if any(values.shape != expected_shape for values in extrema_arrays):
        raise ValueError(f"????????????? {expected_shape}?")
    if any(not np.all(np.isfinite(values)) for values in extrema_arrays):
        raise ValueError("??????????????")
    if any(np.any(values[:, 0] > values[:, 1]) for values in extrema_arrays):
        raise ValueError("???????????????")

    stacked = np.stack(extrema_arrays, axis=0)
    global_min = np.min(stacked[:, :, 0], axis=0)
    global_max = np.max(stacked[:, :, 1], axis=0)
    thresholds: list[tuple[float, ...]] = []
    for minimum, maximum in zip(global_min, global_max):
        if np.isclose(minimum, maximum, rtol=0.0, atol=0.0):
            thresholds.append(())
            continue
        edges = np.linspace(minimum, maximum, max_bin + 1, dtype=np.float64)
        thresholds.append(tuple(float(value) for value in edges[1:-1]))
    return BinningModel(names, tuple(thresholds), max_bin)


def transform_with_binning(
    features: ArrayLike,
    model: BinningModel,
    feature_names: Sequence[str] | None = None,
) -> NDArray[np.int64]:
    """?????????????????????"""
    array = _validate_features(features)
    if array.shape[1] != len(model.feature_names):
        raise ValueError("??????????????")
    if feature_names is not None and tuple(feature_names) != model.feature_names:
        raise ValueError("?????????????????")
    binned = np.empty(array.shape, dtype=np.int64)
    for feature_index, thresholds in enumerate(model.thresholds):
        binned[:, feature_index] = np.searchsorted(
            np.asarray(thresholds, dtype=np.float64),
            array[:, feature_index],
            side="right",
        )
    if np.any(binned < 0) or np.any(binned >= model.max_bin):
        raise RuntimeError("???????????")
    return binned

