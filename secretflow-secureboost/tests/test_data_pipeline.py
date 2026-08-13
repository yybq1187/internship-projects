"""验证纵向数据对齐、隔离约束和联邦数据对象构造。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from secretflow.data import PartitionWay

from secureboost_demo.data_pipeline import (
    build_federated_dataset,
    prepare_vertical_dataset,
)
from secureboost_demo.devices import create_devices, shutdown_devices


@pytest.fixture
def source_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """提供 Bob 行顺序被刻意打乱的两方测试数据。"""
    alice_table = pd.DataFrame(
        {
            "sample_id": [1, 2, 3, 4, 5, 6, 7, 8],
            "alice_feature": [11, 12, 13, 14, 15, 16, 17, 18],
            "label": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    bob_table = pd.DataFrame(
        {
            "sample_id": [8, 3, 1, 7, 2, 6, 4, 5],
            "bob_feature": [80, 30, 10, 70, 20, 60, 40, 50],
        }
    )
    return alice_table, bob_table


def _prepare_dataset(
    source_tables: tuple[pd.DataFrame, pd.DataFrame],
):
    """使用统一参数构造便于各测试复用的本地纵向数据。"""
    alice_table, bob_table = source_tables
    return prepare_vertical_dataset(
        alice_table,
        bob_table,
        alice_feature_columns=["alice_feature"],
        bob_feature_columns=["bob_feature"],
        label_column="label",
        test_size=0.25,
        random_state=42,
    )


def test_prepare_vertical_dataset_aligns_rows_and_isolates_labels(
    source_tables,
) -> None:
    """Bob 的乱序样本应对齐到 Alice，且标签仅位于 Alice 标签数据中。"""
    dataset = _prepare_dataset(source_tables)

    for partition in (dataset.train, dataset.test):
        sample_ids = partition.sample_ids.to_numpy()
        np.testing.assert_array_equal(
            partition.bob_features["bob_feature"].to_numpy(), sample_ids * 10
        )
        assert "label" not in partition.alice_features.columns
        assert "label" not in partition.bob_features.columns
        assert partition.labels.columns.tolist() == ["label"]

    assert not (set(dataset.train.sample_ids) & set(dataset.test.sample_ids))
    assert dataset.train.sample_count + dataset.test.sample_count == 8


def test_prepare_vertical_dataset_rejects_duplicate_sample_ids(source_tables) -> None:
    """任一参与方出现重复 sample_id 时必须在上传数据前失败。"""
    alice_table, bob_table = source_tables
    bob_table.loc[1, "sample_id"] = bob_table.loc[0, "sample_id"]

    with pytest.raises(ValueError, match="sample_id"):
        prepare_vertical_dataset(
            alice_table,
            bob_table,
            alice_feature_columns=["alice_feature"],
            bob_feature_columns=["bob_feature"],
            label_column="label",
        )


def test_prepare_vertical_dataset_rejects_bob_label_column(source_tables) -> None:
    """Bob 表一旦包含标签列，必须被视为数据泄漏并拒绝处理。"""
    alice_table, bob_table = source_tables
    bob_table["label"] = alice_table["label"]

    with pytest.raises(ValueError, match="Bob.*标签"):
        prepare_vertical_dataset(
            alice_table,
            bob_table,
            alice_feature_columns=["alice_feature"],
            bob_feature_columns=["bob_feature"],
            label_column="label",
        )


def test_prepare_vertical_dataset_rejects_overlapping_feature_columns(
    source_tables,
) -> None:
    """同名特征不能同时交给 Alice 和 Bob，避免纵向分区语义混乱。"""
    alice_table, bob_table = source_tables
    alice_table["shared_feature"] = np.arange(len(alice_table))
    bob_table["shared_feature"] = np.arange(len(bob_table))

    with pytest.raises(ValueError, match="特征列不能重叠"):
        prepare_vertical_dataset(
            alice_table,
            bob_table,
            alice_feature_columns=["shared_feature"],
            bob_feature_columns=["shared_feature"],
            label_column="label",
        )


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf])
def test_prepare_vertical_dataset_rejects_nan_and_infinity(
    source_tables, invalid_value
) -> None:
    """模型输入含 NaN 或无穷值时必须在训练前被拦截。"""
    alice_table, bob_table = source_tables
    alice_table.loc[0, "alice_feature"] = invalid_value

    with pytest.raises(ValueError, match="NaN 或无穷值"):
        prepare_vertical_dataset(
            alice_table,
            bob_table,
            alice_feature_columns=["alice_feature"],
            bob_feature_columns=["bob_feature"],
            label_column="label",
        )


def test_build_federated_dataset_uses_vertical_partitions(source_tables) -> None:
    """特征必须分布在 Alice/Bob，标签分区必须只属于 Alice。"""
    local_dataset = _prepare_dataset(source_tables)
    devices = create_devices()
    try:
        federated_dataset = build_federated_dataset(local_dataset, devices)

        assert federated_dataset.train_features.partition_way == PartitionWay.VERTICAL
        assert federated_dataset.test_features.partition_way == PartitionWay.VERTICAL
        assert set(federated_dataset.train_features.partitions) == {
            devices.alice,
            devices.bob,
        }
        assert set(federated_dataset.train_labels.partitions) == {devices.alice}
        assert set(federated_dataset.test_labels.partitions) == {devices.alice}
        assert federated_dataset.train_sample_count == local_dataset.train.sample_count
        assert federated_dataset.test_sample_count == local_dataset.test.sample_count
    finally:
        shutdown_devices()
