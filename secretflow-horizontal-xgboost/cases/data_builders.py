"""构造两个确定性水平联邦数据集。"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from horizontal_xgb.data import HorizontalDataset, HorizontalPartition


TOY_FEATURE_NAMES = ("age", "income", "purchase_count", "credit_score")


def build_toy_horizontal_dataset() -> HorizontalDataset:
    """构造 16 条确定性样本，并按预设所有权划分训练集和测试集。"""
    # 每方 1--6/9--14 用于训练，7--8/15--16 用于测试；各分区均含两类。
    rows = np.asarray(
        [
            [22, 28, 1, 530, 0],
            [25, 32, 2, 560, 0],
            [29, 38, 2, 590, 0],
            [36, 68, 6, 710, 1],
            [41, 76, 7, 745, 1],
            [45, 85, 9, 780, 1],
            [31, 42, 3, 610, 0],
            [39, 72, 7, 730, 1],
            [23, 30, 1, 540, 0],
            [27, 35, 2, 575, 0],
            [32, 44, 3, 620, 0],
            [38, 70, 6, 720, 1],
            [43, 80, 8, 760, 1],
            [48, 92, 10, 800, 1],
            [30, 40, 2, 600, 0],
            [46, 88, 9, 790, 1],
        ],
        dtype=np.float64,
    )
    ids = np.asarray([f"toy_{index:02d}" for index in range(1, 17)])

    def partition(indices: list[int]) -> HorizontalPartition:
        return HorizontalPartition.create(
            ids[indices], rows[indices, :-1], rows[indices, -1], TOY_FEATURE_NAMES
        )

    return HorizontalDataset(
        alice_train=partition(list(range(0, 6))),
        bob_train=partition(list(range(8, 14))),
        alice_test=partition([6, 7]),
        bob_test=partition([14, 15]),
    )


def build_breast_cancer_horizontal_dataset(seed: int = 42) -> HorizontalDataset:
    """按统一分层切分、再按标签分层均分给 Alice 和 Bob。"""
    source = load_breast_cancer()
    features = np.asarray(source.data, dtype=np.float64)
    labels = np.asarray(source.target, dtype=np.float64)
    feature_names = tuple(str(name) for name in source.feature_names)
    original_indices = np.arange(features.shape[0])

    train_indices, test_indices = train_test_split(
        original_indices,
        test_size=0.2,
        random_state=seed,
        stratify=labels,
    )
    alice_train_indices, bob_train_indices = train_test_split(
        train_indices,
        test_size=0.5,
        random_state=seed,
        stratify=labels[train_indices],
    )
    alice_test_indices, bob_test_indices = train_test_split(
        test_indices,
        test_size=0.5,
        random_state=seed,
        stratify=labels[test_indices],
    )

    def partition(indices: np.ndarray) -> HorizontalPartition:
        stable_ids = np.asarray([f"breast_cancer_{index:03d}" for index in indices])
        return HorizontalPartition.create(
            stable_ids,
            features[indices],
            labels[indices],
            feature_names,
        )

    return HorizontalDataset(
        alice_train=partition(alice_train_indices),
        bob_train=partition(bob_train_indices),
        alice_test=partition(alice_test_indices),
        bob_test=partition(bob_test_indices),
    )
