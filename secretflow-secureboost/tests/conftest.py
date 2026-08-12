"""供 SecureBoost 训练测试共享的本地仿真夹具。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from cases.toy_case import build_toy_source_tables
from configs.training_config import SecureBoostTrainingConfig
from secureboost_demo.data_pipeline import (
    build_federated_dataset,
    prepare_vertical_dataset,
)
from secureboost_demo.devices import (
    SecureBoostDevices,
    create_devices,
    shutdown_devices,
)
from secureboost_demo.trainer import TrainingResult, train_secureboost


@dataclass(frozen=True)
class TrainedToyContext:
    """训练测试共享的设备、联邦数据和已训练模型。"""

    devices: SecureBoostDevices
    dataset: Any
    training_result: TrainingResult


@pytest.fixture(scope="session")
def trained_toy_context() -> TrainedToyContext:
    """只训练一次小型玩具模型，供烟雾与重载测试共同使用。"""
    alice_table, bob_table = build_toy_source_tables()
    local_dataset = prepare_vertical_dataset(
        alice_table,
        bob_table,
        alice_feature_columns=["age", "income"],
        bob_feature_columns=["purchase_count", "credit_score"],
        label_column="label",
        test_size=0.33,
        random_state=42,
    )
    devices = create_devices()
    try:
        federated_dataset = build_federated_dataset(local_dataset, devices)
        # 两轮树足以验证 HEU/Sgb 训练链路，同时控制自动化测试耗时。
        training_result = train_secureboost(
            federated_dataset,
            devices,
            SecureBoostTrainingConfig(num_boost_round=2, max_depth=2),
        )
        yield TrainedToyContext(
            devices=devices,
            dataset=federated_dataset,
            training_result=training_result,
        )
    finally:
        shutdown_devices()
