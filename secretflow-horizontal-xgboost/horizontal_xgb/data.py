"""定义水平联邦样本分区及严格的数据契约。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _normalize_sample_ids(sample_ids: ArrayLike) -> NDArray[np.str_]:
    """把样本编号规范化为字符串，便于跨来源执行泄漏检查。"""
    ids = np.asarray(sample_ids).astype(str)
    if ids.ndim != 1 or ids.size == 0:
        raise ValueError("sample_id 必须是非空一维数组。")
    if np.any(np.char.str_len(ids) == 0):
        raise ValueError("sample_id 不能为空字符串。")
    if len(set(ids.tolist())) != ids.size:
        raise ValueError("同一数据分区内存在重复 sample_id。")
    return ids


def _normalize_labels(labels: ArrayLike, sample_count: int) -> NDArray[np.float64]:
    """把标签规范化为 float64，并限制为二分类集合 {0, 1}。"""
    values = np.asarray(labels, dtype=np.float64)
    if values.shape != (sample_count,):
        raise ValueError("标签必须是一维数组且与样本数一致。")
    if not np.all(np.isfinite(values)) or not np.all(np.isin(values, (0.0, 1.0))):
        raise ValueError("二分类标签只能包含 0 和 1。")
    return values


@dataclass(frozen=True)
class HorizontalPartition:
    """保存一个客户端独占的同构特征、标签和样本编号。"""

    sample_ids: NDArray[np.str_]
    features: NDArray[np.float64]
    labels: NDArray[np.float64]
    feature_names: tuple[str, ...]
    feature_dtypes: tuple[str, ...]

    @classmethod
    def create(
        cls,
        sample_ids: ArrayLike,
        features: ArrayLike,
        labels: ArrayLike,
        feature_names: Sequence[str],
        feature_dtypes: Sequence[str] | None = None,
    ) -> "HorizontalPartition":
        """从数组创建分区，并立即执行完整本地校验。"""
        ids = _normalize_sample_ids(sample_ids)
        raw_features = np.asarray(features)
        if raw_features.ndim != 2 or raw_features.shape[0] != ids.size:
            raise ValueError("features 必须是与 sample_id 数量一致的二维数组。")
        if raw_features.shape[1] == 0 or not np.issubdtype(raw_features.dtype, np.number):
            raise ValueError("features 必须包含至少一个数值特征。")
        values = raw_features.astype(np.float64, copy=True)
        if not np.all(np.isfinite(values)):
            raise ValueError("features 不允许包含缺失值、NaN 或无穷值。")
        names = tuple(str(name) for name in feature_names)
        if len(names) != values.shape[1] or len(set(names)) != len(names):
            raise ValueError("特征名称数量必须匹配列数且不能重复。")
        if feature_dtypes is None:
            # ndarray 只有统一 dtype；显式保存逐列模式以供跨方契约比较。
            dtypes = tuple(str(raw_features[:, index].dtype) for index in range(values.shape[1]))
        else:
            dtypes = tuple(str(dtype) for dtype in feature_dtypes)
            if len(dtypes) != values.shape[1]:
                raise ValueError("feature_dtypes 数量必须与特征列数一致。")
        return cls(ids, values, _normalize_labels(labels, ids.size), names, dtypes)

    @property
    def sample_count(self) -> int:
        """返回该客户端分区的样本数。"""
        return int(self.features.shape[0])

    def validate_schema(self, other: "HorizontalPartition") -> None:
        """验证两个客户端的特征名称、顺序和 dtype 完全一致。"""
        if self.feature_names != other.feature_names:
            raise ValueError("Alice 与 Bob 的特征列名或顺序不一致。")
        if self.feature_dtypes != other.feature_dtypes:
            raise ValueError("Alice 与 Bob 的特征 dtype 模式不一致。")


@dataclass(frozen=True)
class HorizontalDataset:
    """保存 Alice/Bob 的训练、测试分区及统一元数据。"""

    alice_train: HorizontalPartition
    bob_train: HorizontalPartition
    alice_test: HorizontalPartition
    bob_test: HorizontalPartition

    def __post_init__(self) -> None:
        """拒绝跨客户端重叠、训练测试泄漏和单类别训练分区。"""
        partitions = (self.alice_train, self.bob_train, self.alice_test, self.bob_test)
        reference = partitions[0]
        for partition in partitions[1:]:
            reference.validate_schema(partition)

        sets = [set(partition.sample_ids.tolist()) for partition in partitions]
        names = ("alice_train", "bob_train", "alice_test", "bob_test")
        for left in range(len(sets)):
            for right in range(left + 1, len(sets)):
                overlap = sets[left].intersection(sets[right])
                if overlap:
                    raise ValueError(
                        f"{names[left]} 与 {names[right]} 存在 sample_id 重叠："
                        f"{sorted(overlap)[:3]}"
                    )
        for party_name, partition in (
            ("Alice", self.alice_train),
            ("Bob", self.bob_train),
        ):
            if np.unique(partition.labels).size != 2:
                raise ValueError(f"{party_name} 的训练分区必须同时包含两个类别。")

    @property
    def feature_names(self) -> tuple[str, ...]:
        """返回全体分区共同使用的特征模式。"""
        return self.alice_train.feature_names

    def combined_train(self) -> HorizontalPartition:
        """构造仅用于集中式参考的训练集，并按 sample_id 稳定排序。"""
        return combine_partitions((self.alice_train, self.bob_train))

    def combined_test(self) -> HorizontalPartition:
        """构造集中评估数据，并按 sample_id 排序防止标签预测错位。"""
        return combine_partitions((self.alice_test, self.bob_test))


def combine_partitions(partitions: Sequence[HorizontalPartition]) -> HorizontalPartition:
    """拼接同构分区并按 sample_id 进行稳定排序。"""
    if not partitions:
        raise ValueError("至少需要一个分区才能合并。")
    reference = partitions[0]
    for partition in partitions[1:]:
        reference.validate_schema(partition)
    ids = np.concatenate([partition.sample_ids for partition in partitions])
    if len(set(ids.tolist())) != ids.size:
        raise ValueError("待合并分区之间存在重复 sample_id。")
    features = np.concatenate([partition.features for partition in partitions], axis=0)
    labels = np.concatenate([partition.labels for partition in partitions])
    order = np.argsort(ids, kind="stable")
    return HorizontalPartition.create(
        ids[order],
        features[order],
        labels[order],
        reference.feature_names,
        reference.feature_dtypes,
    )
