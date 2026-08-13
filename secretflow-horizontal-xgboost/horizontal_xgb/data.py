"""???????????????????"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _normalize_sample_ids(sample_ids: ArrayLike) -> NDArray[np.str_]:
    """?????????????????????????"""
    ids = np.asarray(sample_ids).astype(str)
    if ids.ndim != 1 or ids.size == 0:
        raise ValueError("sample_id ??????????")
    if np.any(np.char.str_len(ids) == 0):
        raise ValueError("sample_id ????????")
    if len(set(ids.tolist())) != ids.size:
        raise ValueError("??????????? sample_id?")
    return ids


def _normalize_labels(labels: ArrayLike, sample_count: int) -> NDArray[np.float64]:
    """??????? float64?????????? {0, 1}?"""
    values = np.asarray(labels, dtype=np.float64)
    if values.shape != (sample_count,):
        raise ValueError("?????????????????")
    if not np.all(np.isfinite(values)) or not np.all(np.isin(values, (0.0, 1.0))):
        raise ValueError("????????? 0 ? 1?")
    return values


@dataclass(frozen=True)
class HorizontalPartition:
    """???????????????????????"""

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
        """????????????????????"""
        ids = _normalize_sample_ids(sample_ids)
        raw_features = np.asarray(features)
        if raw_features.ndim != 2 or raw_features.shape[0] != ids.size:
            raise ValueError("features ???? sample_id ??????????")
        if raw_features.shape[1] == 0 or not np.issubdtype(raw_features.dtype, np.number):
            raise ValueError("features ?????????????")
        values = raw_features.astype(np.float64, copy=True)
        if not np.all(np.isfinite(values)):
            raise ValueError("features ?????????NaN ?????")
        names = tuple(str(name) for name in feature_names)
        if len(names) != values.shape[1] or len(set(names)) != len(names):
            raise ValueError("??????????????????")
        if feature_dtypes is None:
            # ndarray ???? dtype??????????????????
            dtypes = tuple(str(raw_features[:, index].dtype) for index in range(values.shape[1]))
        else:
            dtypes = tuple(str(dtype) for dtype in feature_dtypes)
            if len(dtypes) != values.shape[1]:
                raise ValueError("feature_dtypes ????????????")
        return cls(ids, values, _normalize_labels(labels, ids.size), names, dtypes)

    @property
    def sample_count(self) -> int:
        """?????????????"""
        return int(self.features.shape[0])

    def validate_schema(self, other: "HorizontalPartition") -> None:
        """???????????????? dtype ?????"""
        if self.feature_names != other.feature_names:
            raise ValueError("Alice ? Bob ????????????")
        if self.feature_dtypes != other.feature_dtypes:
            raise ValueError("Alice ? Bob ??? dtype ??????")


@dataclass(frozen=True)
class HorizontalDataset:
    """?? Alice/Bob ???????????????"""

    alice_train: HorizontalPartition
    bob_train: HorizontalPartition
    alice_test: HorizontalPartition
    bob_test: HorizontalPartition

    def __post_init__(self) -> None:
        """????????????????????????"""
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
                        f"{names[left]} ? {names[right]} ?? sample_id ???"
                        f"{sorted(overlap)[:3]}"
                    )
        for party_name, partition in (
            ("Alice", self.alice_train),
            ("Bob", self.bob_train),
        ):
            if np.unique(partition.labels).size != 2:
                raise ValueError(f"{party_name} ????????????????")

    @property
    def feature_names(self) -> tuple[str, ...]:
        """????????????????"""
        return self.alice_train.feature_names

    def combined_train(self) -> HorizontalPartition:
        """????????????????? sample_id ?????"""
        return combine_partitions((self.alice_train, self.bob_train))

    def combined_test(self) -> HorizontalPartition:
        """??????????? sample_id ???????????"""
        return combine_partitions((self.alice_test, self.bob_test))


def combine_partitions(partitions: Sequence[HorizontalPartition]) -> HorizontalPartition:
    """???????? sample_id ???????"""
    if not partitions:
        raise ValueError("?????????????")
    reference = partitions[0]
    for partition in partitions[1:]:
        reference.validate_schema(partition)
    ids = np.concatenate([partition.sample_ids for partition in partitions])
    if len(set(ids.tolist())) != ids.size:
        raise ValueError("??????????? sample_id?")
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
