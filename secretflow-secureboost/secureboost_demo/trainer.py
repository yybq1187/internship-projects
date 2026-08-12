"""SecureBoost 训练入口与分布式模型保存、加载封装。"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from secretflow.ml.boost.sgb_v import Sgb
from secretflow.ml.boost.sgb_v.model import load_model

from configs.training_config import SecureBoostTrainingConfig
from secureboost_demo.data_pipeline import FederatedVerticalDataset
from secureboost_demo.devices import SecureBoostDevices


@dataclass(frozen=True)
class TrainingResult:
    """保存训练产物及公开可记录的训练耗时。"""

    model: Any
    training_seconds: float
    parameters: dict


@dataclass(frozen=True)
class DistributedModelPaths:
    """记录 Alice 与 Bob 各自保存的分布式模型文件前缀。"""

    alice_path: Path
    bob_path: Path


def train_secureboost(
    dataset: FederatedVerticalDataset,
    devices: SecureBoostDevices,
    config: SecureBoostTrainingConfig = SecureBoostTrainingConfig(),
) -> TrainingResult:
    """调用官方 ``Sgb`` 完成一次纵向联邦 SecureBoost 训练。

    本函数不实现树构造、梯度计算或同态加密运算，只负责参数准备、
    训练调用和耗时记录。原始特征、标签、梯度和中间统计量均不会被
    ``sf.reveal`` 到协调端。
    """
    _validate_training_dataset(dataset, devices)
    parameters = config.to_secretflow_params()
    trainer = Sgb(devices.heu)

    start_time = perf_counter()
    model = trainer.train(
        params=parameters,
        dtrain=dataset.train_features,
        label=dataset.train_labels,
    )
    training_seconds = perf_counter() - start_time
    return TrainingResult(
        model=model,
        training_seconds=training_seconds,
        parameters=parameters,
    )


def save_distributed_model(
    model: Any,
    devices: SecureBoostDevices,
    output_directory: str | Path,
) -> DistributedModelPaths:
    """将模型分片保存至输出目录下的 Alice 与 Bob 子目录。

    ``SgbModel.save_model`` 会为每个参与方写入它应持有的树分片。
    此处使用文件前缀而不是单一文件名，以兼容 SecretFlow 生成的
    ``common.json``、``leaf_weight.json`` 和 ``split_tree.json`` 文件。
    """
    output_path = Path(output_directory).resolve()
    alice_path = output_path / "alice" / "secureboost_model"
    bob_path = output_path / "bob" / "secureboost_model"
    model.save_model(
        {
            devices.alice: str(alice_path),
            devices.bob: str(bob_path),
        }
    )
    return DistributedModelPaths(alice_path=alice_path, bob_path=bob_path)


def load_distributed_model(
    paths: DistributedModelPaths,
    devices: SecureBoostDevices,
) -> Any:
    """从 Alice/Bob 的模型分片重新构造分布式 SecureBoost 模型。"""
    return load_model(
        {
            devices.alice: str(paths.alice_path),
            devices.bob: str(paths.bob_path),
        },
        label_holder=devices.alice,
    )


def save_training_parameters(parameters: dict, output_path: str | Path) -> Path:
    """将本次实际传给 ``Sgb.train`` 的公开参数保存为 UTF-8 JSON 快照。"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(parameters, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def _validate_training_dataset(
    dataset: FederatedVerticalDataset,
    devices: SecureBoostDevices,
) -> None:
    """确认训练数据仍保持两方特征和 Alice 独占标签的分区结构。"""
    if dataset.train_sample_count <= 0:
        raise ValueError("训练样本数必须大于零。")
    if set(dataset.train_features.partitions) != {devices.alice, devices.bob}:
        raise ValueError("训练特征必须分别位于 Alice 与 Bob 的 PYU。")
    if set(dataset.train_labels.partitions) != {devices.alice}:
        raise ValueError("训练标签必须仅位于 Alice 的 PYU。")
