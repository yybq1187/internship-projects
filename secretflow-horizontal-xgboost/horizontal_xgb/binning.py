"""实现教学型全局等宽分桶及其可序列化模型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _validate_features(features: ArrayLike) -> NDArray[np.float64]:
    """将特征转换为二维 float64，并拒绝空值和非有限数值。"""
    array = np.asarray(features, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("features 必须是非空二维数组。")
    if not np.all(np.isfinite(array)):
        raise ValueError("features 不允许包含 NaN 或无穷值。")
    return array


@dataclass(frozen=True)
class BinningModel:
    """保存各特征的等宽分桶阈值。"""

    feature_names: tuple[str, ...]
    thresholds: tuple[tuple[float, ...], ...]
    max_bin: int

    def __post_init__(self) -> None:
        """验证特征和阈值结构一致。"""
        if len(self.feature_names) == 0:
            raise ValueError("分桶模型必须至少包含一个特征。")
        if len(self.feature_names) != len(self.thresholds):
            raise ValueError("feature_names 与 thresholds 数量不一致。")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("特征名不能重复。")
        if self.max_bin < 2:
            raise ValueError("max_bin 至少为 2。")
        for feature_thresholds in self.thresholds:
            values = np.asarray(feature_thresholds, dtype=np.float64)
            if values.size > self.max_bin - 1:
                raise ValueError("单个特征的阈值数量超过 max_bin - 1。")
            if not np.all(np.isfinite(values)):
                raise ValueError("分桶阈值必须为有限数值。")
            if values.size > 1 and not np.all(np.diff(values) > 0.0):
                raise ValueError("分桶阈值必须严格递增。")

    def to_dict(self) -> dict[str, Any]:
        """转换为 JSON 可序列化字典。"""
        return {
            "feature_names": list(self.feature_names),
            "thresholds": [list(values) for values in self.thresholds],
            "max_bin": self.max_bin,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BinningModel":
        """从字典恢复分桶模型。"""
        required = {"feature_names", "thresholds", "max_bin"}
        if not required.issubset(payload):
            raise ValueError("分桶模型缺少必要字段。")
        return cls(
            feature_names=tuple(str(name) for name in payload["feature_names"]),
            thresholds=tuple(
                tuple(float(value) for value in values)
                for values in payload["thresholds"]
            ),
            max_bin=int(payload["max_bin"]),
        )


def local_feature_extrema(features: ArrayLike) -> NDArray[np.float64]:
    """计算客户端逐特征局部最小值和最大值。"""
    array = _validate_features(features)
    return np.stack((np.min(array, axis=0), np.max(array, axis=0)), axis=1)


def fit_equal_width_binning(
    local_extrema: Sequence[ArrayLike],
    feature_names: Sequence[str],
    max_bin: int,
) -> BinningModel:
    """由多个客户端的局部极值生成共享的全局等宽桶边界。"""
    if max_bin < 2:
        raise ValueError("max_bin 至少为 2。")
    names = tuple(str(name) for name in feature_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("feature_names 必须非空且不能重复。")
    if len(local_extrema) < 1:
        raise ValueError("至少需要一个客户端的局部极值。")

    extrema_arrays = [np.asarray(values, dtype=np.float64) for values in local_extrema]
    expected_shape = (len(names), 2)
    if any(values.shape != expected_shape for values in extrema_arrays):
        raise ValueError(f"每份局部极值都必须具有形状 {expected_shape}。")
    if any(not np.all(np.isfinite(values)) for values in extrema_arrays):
        raise ValueError("局部极值必须全部为有限数值。")
    if any(np.any(values[:, 0] > values[:, 1]) for values in extrema_arrays):
        raise ValueError("局部最小值不能大于局部最大值。")

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
    """使用共享阈值把数值特征映射为确定的桶编号。"""
    array = _validate_features(features)
    if array.shape[1] != len(model.feature_names):
        raise ValueError("特征列数量与分桶模型不一致。")
    if feature_names is not None and tuple(feature_names) != model.feature_names:
        raise ValueError("预测特征名或顺序与分桶模型不一致。")
    binned = np.empty(array.shape, dtype=np.int64)
    for feature_index, thresholds in enumerate(model.thresholds):
        binned[:, feature_index] = np.searchsorted(
            np.asarray(thresholds, dtype=np.float64),
            array[:, feature_index],
            side="right",
        )
    if np.any(binned < 0) or np.any(binned >= model.max_bin):
        raise RuntimeError("分桶结果超出合法范围。")
    return binned

